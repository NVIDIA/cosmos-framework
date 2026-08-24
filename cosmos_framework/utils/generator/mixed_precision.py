# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mixed-precision diffusion steps for ModelOpt static-FP8 checkpoints.

The first/last N denoising steps run generation-path FP8 linears with 16-bit
activations (W8A16: the checkpoint's E4M3 weight dequantized to the compute
dtype + dense GEMM); the middle steps keep the TorchAO FP8-activation path
(W8A8). Weights are always the checkpoint's FP8 tensors — only the activation
treatment changes per step. Port of vllm-omni's Cosmos3 mixed-precision
feature (branch ``mixed-precision-diffusion-steps``) onto the framework's
TorchAO FP8 path.
"""

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig

PRECISION_PATHS = ("reasoner", "generation")
W8A16_CACHE_MODES = ("none", "generation", "all", "cpu_block", "gpu_block")


def use_w8a16_step(first_steps: int, last_steps: int, step_index: int, num_steps: int) -> bool:
    """Return whether one denoising step runs the W8A16 path.

    ``num_steps == 1`` always selects W8A8: FSDP collective alignment pads
    slow ranks with dummy single-step samples, and those must stay on the
    cheap path (a genuine 1-step request also gets W8A8 — same open question
    as the vllm-omni reference).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if step_index < 0 or step_index >= num_steps:
        raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
    if num_steps == 1:
        return False
    return step_index < first_steps or step_index >= num_steps - last_steps


def _unwrap_float8(weight: torch.Tensor) -> torch.Tensor:
    """Return the local PrototypeFloat8Tensor behind an (optionally DTensor) weight."""
    return weight.to_local() if isinstance(weight, DTensor) else weight


def _dequantize_w8a16_weight(weight: torch.Tensor) -> torch.Tensor:
    """Dequantize a full (gathered) PrototypeFloat8Tensor to a dense (N, K) matrix."""
    return weight.qdata.to(weight.dtype) * weight.scale.to(weight.dtype)


def _validate_fp8_linear(module: nn.Module, fqn: str) -> None:
    """Bind-time validation, parity with vllm-omni's Fp8W8A8W8A16Strategy._validate_layer."""
    if hasattr(module, "pre_quant_scale"):
        raise ValueError(
            f"{fqn} carries pre_quant_scale; SmoothQuant checkpoints are not supported "
            "by mixed-precision diffusion steps."
        )
    local = _unwrap_float8(module.weight)
    qdata = local.qdata
    if qdata.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{fqn} must hold a canonical float8_e4m3fn weight; got {qdata.dtype}")
    if qdata.ndim != 2:
        raise ValueError(f"{fqn} must hold a rank-2 FP8 weight; got shape {tuple(qdata.shape)}")
    scale = local.scale
    if scale.numel() != 1:
        raise ValueError(f"{fqn} requires one per-tensor FP8 weight scale, got {scale.numel()}")
    scale_value = scale.detach().float()
    if not torch.isfinite(scale_value).all() or not (scale_value > 0).all():
        raise ValueError(f"{fqn} has a non-finite or non-positive FP8 weight scale")


class MixedPrecisionRuntime:
    """Owns per-step precision state and the tagged FP8 linear inventory."""

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config
        self.installed_counts: dict[str, int] = {path: 0 for path in PRECISION_PATHS}
        self.last_trace: tuple[str, ...] = ()
        self.is_sharded = False
        self._modules_by_path: dict[str, list[nn.Module]] = {path: [] for path in PRECISION_PATHS}
        self._generation_high_precision = False
        self._trace: list[str] = []
        self._block_provider = None  # set by Task 4
        self.cached_bytes = 0  # set by Task 3

    def install(self, net: nn.Module) -> None:
        """Discover, validate, and tag every ModelOpt FP8 linear under ``net``."""
        # Import here: quantization.py imports are lazy-torchao friendly and
        # mixed_precision must not become a hard torchao dependency at import.
        from cosmos_framework.utils.generator.quantization import _ModelOptFloat8Linear

        for fqn, module in net.named_modules():
            if not isinstance(module, _ModelOptFloat8Linear):
                continue
            _validate_fp8_linear(module, fqn)
            path = "generation" if "_moe_gen" in fqn else "reasoner"
            module._mixed_precision_path = path
            module._mixed_precision_runtime = self
            self._modules_by_path[path].append(module)
            self.installed_counts[path] += 1
            self.is_sharded = self.is_sharded or isinstance(module.weight, DTensor)

        missing = [path for path, count in self.installed_counts.items() if count == 0]
        if missing:
            raise ValueError(
                f"Mixed precision found no ModelOpt FP8 linears on path(s) {missing}; "
                f"discovered counts={self.installed_counts}"
            )
        if self.is_sharded and self.config.mixed_precision_w8a16_cache != "none":
            raise ValueError(
                "FSDP-sharded FP8 weights support only mixed_precision_w8a16_cache='none'; "
                f"got {self.config.mixed_precision_w8a16_cache!r}"
            )
        net._mixed_precision_runtime = self
        cache_mode = self.config.mixed_precision_w8a16_cache
        if cache_mode in ("generation", "all"):
            cached_paths = PRECISION_PATHS if cache_mode == "all" else ("generation",)
            for path in cached_paths:
                for module in self._modules_by_path[path]:
                    cache = _dequantize_w8a16_weight(_unwrap_float8(module.weight)).contiguous()
                    module.register_buffer("_w8a16_weight_cache", cache, persistent=False)
                    self.cached_bytes += cache.numel() * cache.element_size()
        reasoner_label = "W8A16" if self.config.mixed_precision_reasoner_policy == "high_precision" else "W8A8"
        log.info(
            f"Mixed precision installed: reasoner={reasoner_label}, "
            f"generation=first {self.config.mixed_precision_first_steps} + "
            f"last {self.config.mixed_precision_last_steps} W8A16 / middle W8A8, "
            f"cache={self.config.mixed_precision_w8a16_cache}, linears={self.installed_counts}"
        )
        log.info(
            f"Mixed precision W8A16 cache ready: mode={cache_mode}, "
            f"resident_bytes={self.cached_bytes}"
        )

    def use_high_precision(self, path: str) -> bool:
        """Resolve the static reasoner policy or the current generation step selection."""
        if path == "reasoner":
            return self.config.mixed_precision_reasoner_policy == "high_precision"
        return self._generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> None:
        """Select one precision at the sampler-step boundary and record the trace."""
        self._generation_high_precision = use_w8a16_step(
            self.config.mixed_precision_first_steps,
            self.config.mixed_precision_last_steps,
            step_index,
            num_steps,
        )
        if self._block_provider is not None and self._generation_high_precision:
            self._block_provider.preload_first()
        self._trace.append("W8A16" if self._generation_high_precision else "W8A8")

    def set_base_precision(self) -> None:
        """Force W8A8 (used before FSDP-alignment dummy padding calls)."""
        self._generation_high_precision = False

    def resolve_w8a16_weight(self, module: nn.Module) -> torch.Tensor:
        """Resolve a dense (N, K) weight: staged slot -> full cache -> on-the-fly."""
        staged = module._mixed_precision_staged_weight
        if staged is not None:
            return staged
        cached = getattr(module, "_w8a16_weight_cache", None)
        if cached is not None:
            return cached
        return _dequantize_w8a16_weight(_unwrap_float8(module.weight))

    def reset(self) -> None:
        """Finish the request: log the trace and return to idle W8A8 state."""
        if self._trace:
            self.last_trace = tuple(self._trace)
            log.info(f"MIXED_PRECISION_TRACE steps={','.join(self.last_trace)}")
        self._trace.clear()
        self._generation_high_precision = False
        if self._block_provider is not None:
            self._block_provider.reset()


def install_mixed_precision_runtime(
    net: nn.Module, quantization_config: QuantizationConfig
) -> MixedPrecisionRuntime | None:
    """Install mixed precision on ``net`` when enabled; return the runtime or None."""
    if not quantization_config.mixed_precision_enabled:
        return None
    runtime = MixedPrecisionRuntime(quantization_config)
    runtime.install(net)
    return runtime
