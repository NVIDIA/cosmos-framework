# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NVFP4 linear layers for the cosmos3 interactive causal model.

``NVFP4Linear`` is a drop-in ``nn.Linear`` replacement backed by the FP4 ops from the
``cosmos-nvfp4`` wheel (``import cosmos_nvfp4``; activation-quantize + FP4 GEMM, registered as
``torch.ops.cosmos3_fp4.*`` with ``register_fake`` so they are ``torch.compile(fullgraph=True)``
+ CUDA-graph traceable; they also work in plain eager).

The model converts its decoder linears to NVFP4 *after* the checkpoint loads — see
``OmniMoTCausalModel.maybe_convert_linears_to_nvfp4`` (gated by ``config.nvfp4_linears``). The
conversion is post-load on purpose: DCP loads the bf16 ``weight`` into plain ``nn.Linear`` first,
and pre-quantized buffers (no ``weight`` param, no lazy-quant) keep the layer compile-clean.
"""

from __future__ import annotations

import os
import re

import torch
from torch import nn

from cosmos_framework.utils import log

# Decoder sublayers kept in high precision (router gates, norms, heads, adaLN /
# timestep modulation, embeddings, latent<->hidden projections). The conversion
# only walks the decoder layers, so embeds/heads outside are already excluded;
# these patterns are belt-and-suspenders and cover future model variants.
_SKIP_PATTERNS = (
    r".*\.gate$",
    r".*norm.*",
    r".*lm_head.*",
    r".*\.head($|\..*)",
    r".*modulation.*",
    r".*ada_?ln.*",
    r".*scale_shift.*",
    r".*embed.*",
    r".*time_.*",
    r".*(vae2llm|llm2vae).*",
)

# Vendored FP4 op handles + the static_6 dequant denominator, bound lazily on first use.
_FP4_READY = False
_SCALE_RULE_ID: int | None = None
_DEQUANT_DENOM: float | None = None
_QUANTIZE = None
_GEMM = None


def resolve_legacy_nvfp4_mode(config_mode: str | None) -> str | None:
    """Return the effective legacy NVFP4 mode."""
    if config_mode is not None:
        return config_mode if config_mode else None
    if os.environ.get("COSMOS3_W4A16_TORCHAO", "").lower() in ("1", "true", "yes"):
        return "w4a16"
    if os.environ.get("COSMOS3_NVFP4", "").lower() in ("1", "true", "yes"):
        return "w4a4"
    return None


def _ensure_fp4_ops() -> None:
    """Import the vendored FP4 ops and bind the quantize / FP4-GEMM handles (idempotent).

    Importing ``fp4_ops`` loads the ``_C`` extension (registering
    ``torch.ops.cosmos3_fp4.*``) and its ``register_fake`` impls, which is what
    makes the ops traceable under ``torch.compile`` + CUDA graphs.
    """
    global _FP4_READY, _SCALE_RULE_ID, _DEQUANT_DENOM, _QUANTIZE, _GEMM
    if _FP4_READY:
        return
    from cosmos_nvfp4 import fp4_ops

    # static_6: fixed E2M1 max = 6, no per-block 4/6 search. The 4/6 search is 30-48% of
    # the activation-quant cost at multi-token shapes (activation quant is the FP4 path's
    # bottleneck on GB200, not the GEMM), so static_6 is the TE-style choice that keeps
    # W4A4 competitive with bf16. Weights use the same rule so the GEMM dequant scale is
    # a single (e2m1*e4m3)^2.
    _SCALE_RULE_ID = fp4_ops.SCALE_RULE_STATIC_6
    _DEQUANT_DENOM = fp4_ops.DEQUANT_DENOM
    _QUANTIZE = fp4_ops.quantize_to_fp4
    _GEMM = fp4_ops.gemm_nvfp4nvfp4_bf16
    _FP4_READY = True


def _prequantize_weight(weight: torch.Tensor):
    """Offline weight quantization -> (values uint8, blocked scale_factors, amax)."""
    from cosmos_nvfp4 import fp4_ops

    return fp4_ops.prequantize_nvfp4_weight(weight)


class NVFP4Linear(nn.Module):
    """Drop-in ``nn.Linear`` replacement using the vendored NVFP4 W4A4 ops.

    Weights are quantized offline; activations are quantized per forward. Built
    from compile/CUDA-graph-safe ops so it can ride the optional captured-graph path.
    """

    def __init__(self, module: nn.Linear, module_name: str) -> None:
        super().__init__()
        _ensure_fp4_ops()
        if module.weight.device.type != "cuda":
            raise ValueError(f"NVFP4Linear requires a CUDA module; {module_name} is on {module.weight.device}")
        self.module_name = module_name
        self.in_features = module.in_features
        self.out_features = module.out_features
        with torch.no_grad():
            w_values, w_scale, w_amax = _prequantize_weight(module.weight.detach().to(torch.bfloat16))
        # Non-persistent buffers: ride with the module's device and stay stable for CUDA graphs.
        self.register_buffer("w_values", w_values, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)
        self.register_buffer("w_amax", w_amax, persistent=False)
        self.register_buffer(
            "bias_param",
            module.bias.detach().to(torch.bfloat16) if module.bias is not None else None,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, cin = x.shape
        m = 1
        for s in batch:
            m *= s
        # Empty token dim (an inactive und/gen pathway on some frames): the FP4 GEMM's
        # TMA descriptor can't address 0 rows, so return empty like nn.Linear would.
        if m == 0:
            return x.new_empty((*batch, self.out_features))
        x2 = x.reshape(m, cin).contiguous()
        pad = (-m) % 128  # FP4 GEMM requires the row dim padded to a multiple of 128
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        x_values, x_scale, x_amax = _QUANTIZE(x2, True, True, False, False, False, _SCALE_RULE_ID, 0)
        alpha = (x_amax * self.w_amax / _DEQUANT_DENOM).to(torch.float32)
        out = _GEMM(x_values, self.w_values, x_scale, self.w_scale, alpha)[:m, : self.out_features]
        if self.bias_param is not None:
            out = out + self.bias_param
        return out.reshape(*batch, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_param is not None}"


def _is_skipped(name: str) -> bool:
    return any(re.fullmatch(p, name) for p in _SKIP_PATTERNS)


def _convert_decoder_linears_w4a4(model: nn.Module) -> list[str]:
    """Replace every eligible decoder ``nn.Linear`` with the W4A4 ``NVFP4Linear``."""
    layers = getattr(model, "net", model).language_model.model.layers
    replaced: list[str] = []
    for li, layer in enumerate(layers):
        targets = [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear) and not _is_skipped(n)]
        for name, sub in targets:
            layer.set_submodule(name, NVFP4Linear(sub, f"layers.{li}.{name}"))
            replaced.append(f"layers.{li}.{name}")
    log.info(f"[nvfp4] converted {len(replaced)} nn.Linear -> NVFP4 W4A4 across {len(layers)} decoder layers")
    return replaced


def _apply_w4a16_torchao(model: nn.Module) -> None:
    """W4A16 weight-only via torchao ``NVFP4WeightOnlyConfig`` (packed 4-bit weights, bf16
    activations) — real packed quant, not fake-quant (~3.5x weight-memory saving). Pair with
    ``torch.compile`` so inductor fuses the dequant into the GEMM and it runs ~bf16-fast."""
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    from torchao.quantization import quantize_

    layers = getattr(model, "net", model).language_model.model.layers
    n_lin = sum(isinstance(m, nn.Linear) for m in layers.modules())
    quantize_(
        layers,
        NVFP4WeightOnlyConfig(),
        filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and not _is_skipped(fqn),
    )
    log.info(f"[nvfp4] torchao NVFP4 W4A16 (real/packed) across decoder layers ({n_lin} nn.Linear seen)")


def convert_decoder_linears_to_nvfp4(model: nn.Module, mode: str) -> nn.Module:
    """Convert the MoT decoder linears to NVFP4 in place. Called post-checkpoint-load.

    ``mode='w4a4'``  -> weights + activations in NVFP4 (vendored FP4 GEMM, ``NVFP4Linear``).
    ``mode='w4a16'`` -> weight-only via torchao (packed 4-bit weights, bf16 activations).
    ``torch.compile`` / CUDA graphs are optional run-flag add-ons; we raise the dynamo
    recompile limit here so the AR warmup (several distinct static shapes) does not trip it.
    """
    # AR warmup hits several distinct static shapes (context fill + active/empty und-gen
    # paths), each needing its own compiled graph; the default dynamo limit (8) trips before
    # steady state under fullgraph=True. Raise it: warmup recompiles a few times, then replays.
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 512
    if mode == "w4a4":
        _convert_decoder_linears_w4a4(model)
    elif mode == "w4a16":
        _apply_w4a16_torchao(model)
    else:
        raise ValueError(f"unknown nvfp4 linear mode: {mode!r} (expected 'w4a4' or 'w4a16')")
    return model
