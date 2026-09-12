# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Quantize-dequantize (Q/DQ) simulation of the *non-GEMM* edges of the generation tower.

``int8_sim`` / ``fp8_sim`` (see ``quantization.py``) fake-quantize the weights and
inputs of the selected linears. This module extends the simulation to the tensors
that travel *between* operators on the ``_moe_gen`` pathway, so the accuracy cost
of an end-to-end 8-bit pipeline can be attributed edge by edge. Every granularity
is one a real kernel can consume, i.e. a scale never varies along the reduction
dimension of the matmul that consumes the tensor:

===========  ==========================================================================
edge         tensors fake-quantized (generation tower only) and their scale layout
===========  ==========================================================================
gemm_out     outputs of the quantized linears; one scale per ``group_size`` along the
             output features (what an epilogue tile can compute)
residual     the residual stream after each residual add; group_size along hidden
attn_qkv     inside attention: Q and K (after RoPE) with one scale per (token, head)
             -- constant along head_dim, the reduction dim of Q·Kᵀ. K is smoothed
             first (K − mean over keys, per head/channel), which leaves softmax
             unchanged (SageAttention).
attn_pv      inside attention, via a dense fp32 reference: P (softmax) as unsigned
             8-bit with one scale per (query row, block of keys); V with one scale
             per (head, channel) over all keys -- or per (key block, head, channel)
             -- constant along the key dim, the reduction dim of P·V. V is smoothed
             (V − mean over keys) and the mean added back to O since Σ_j P_ij = 1
             (SageAttention2).
und_kv       the cached text K/V, quantized once when written: K per (token, head),
             V per (head, channel) over the cached keys
===========  ==========================================================================

The runtime state below is process-global and configured by
``apply_quantization_inplace``; with no edge enabled every hook is an identity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

SIM_EDGES: tuple[str, ...] = ("gemm_out", "residual", "attn_qkv", "attn_pv", "und_kv")
_UINT8_QMAX = 255.0


@dataclass
class QdqSimEdgeRuntime:
    """Process-global switchboard for the edge simulation."""

    edges: frozenset[str] = field(default_factory=frozenset)
    method: str = "int8_sim"
    group_size: int = 0
    # ``residual`` edge overrides: 0 = same group size as the rest; bits 8 (default) or 16.
    residual_group_size: int = 0
    residual_bits: int = 8
    # attention: V scale per (head, channel) over all keys (0) or per block of this many keys
    attn_v_block_size: int = 0
    # attention: subtract the per-channel key mean from K (softmax-invariant) and from V
    # (added back to O after P·V) before quantizing them
    attn_smoothing: bool = True
    # attention: P scale per (query row, this many keys)
    attn_p_block_size: int = 64
    # attention operand formats: V "int8" (signed, per-channel scale) / "fp8" (E4M3, per-channel
    # scale) / "none" (keep bf16); P "uint8" (per row/key-block scale) / "fp8" (E4M3, fixed scale
    # since P ∈ [0,1]) / "none"
    attn_v_format: str = "int8"
    attn_p_format: str = "uint8"
    # P·V accumulator: "fp32" (default) or "fp16" -- SageAttention v1 style: P and V in fp16, the
    # running output accumulator rounded to fp16 after every ``attn_accum_block`` keys (HMMA k=16)
    attn_pv_accum: str = "fp32"
    attn_accum_block: int = 16
    # ``attn_qkv`` scope: "all" = Q and every key (text + gen); "gen" = Q and the gen keys only,
    # the cached/joint text keys stay bf16; "und" = only the text keys (Q and gen keys stay bf16).
    # Anything but "all" needs the caller to pass ``und_key_mask`` to ``sim_attention``.
    attn_k_scope: str = "all"
    # ---- experimental attention options (also settable through QDQ_SIM_ATTN_OPTS, see _parse_attn_opts)
    # fraction of gen keys kept in bf16 (K and V) -- the most outlying keys by K/V peak ratio
    attn_outlier_frac: float = 0.0
    # subtract the per-channel query mean before quantizing Q; the q̄·k term is added back exactly
    attn_q_smoothing: bool = False
    # Q/K scale per block of this many tokens (0 = per token), SageAttention-style
    attn_qk_block_tokens: int = 0
    # gating (eager runs only): attention stays bf16 for the first N sampler steps / in these layers
    attn_bf16_first_steps: int = 0
    attn_bf16_layers: tuple[int, ...] = ()
    attn_qk_bf16_layers: tuple[int, ...] = ()  # layers whose Q/K stay bf16 (P/V still quantized)
    attn_pv_bf16_layers: tuple[int, ...] = ()  # layers whose P/V stay bf16 (Q/K still quantized)
    attn_layers_per_forward: int = 36
    # in-situ error log (eager runs only): CSV path; each attention call also runs the bf16 kernel
    attn_err_log: str = ""
    # Q/K number format override ("" = follow ``method``; "int8" / "fp8"); V keeps ``attn_v_format``
    attn_qk_format: str = ""
    # exact (softmax-preserving) Q/K pre-transforms applied before the INT8 rounding:
    #   balance: per-(kv-head, channel) rescale q·s, k/s with s = (amax_k/amax_q)^alpha (SmoothAttention style)
    #   hadamard: rotate q and k along head_dim by a normalized Walsh-Hadamard matrix (QuaRot style)
    attn_qk_balance: bool = False
    attn_qk_balance_alpha: float = 0.5
    attn_qk_hadamard: bool = False

    def enabled(self, edge: str) -> bool:
        return edge in self.edges


_runtime = QdqSimEdgeRuntime()


def configure_sim_edges(
    edges: list[str] | tuple[str, ...] | frozenset[str],
    *,
    method: str,
    group_size: int,
    residual_group_size: int = 0,
    residual_bits: int = 8,
    attn_v_block_size: int = 0,
    attn_smoothing: bool = True,
    attn_v_format: str = "int8",
    attn_p_format: str = "uint8",
    attn_pv_accum: str = "fp32",
    attn_k_scope: str = "all",
) -> None:
    """Enable the given edges for the current process (empty = all hooks are identities)."""
    unknown = sorted(set(edges) - set(SIM_EDGES))
    if unknown:
        raise ValueError(f"Unknown sim edges {unknown}; valid: {list(SIM_EDGES)}")
    if method not in ("int8_sim", "fp8_sim"):
        raise ValueError(f"Edge simulation supports int8_sim / fp8_sim, got {method!r}")
    if edges and group_size <= 0:
        raise ValueError("Edge simulation requires qdq_group_size > 0 (group-wise scales along the last dim)")
    if residual_bits not in (8, 16):
        raise ValueError(f"residual_bits must be 8 or 16, got {residual_bits}")
    if attn_v_block_size < 0:
        raise ValueError("attn_v_block_size must be >= 0")
    if attn_v_format not in ("int8", "fp8", "none"):
        raise ValueError(f"attn_v_format must be int8 / fp8 / none, got {attn_v_format!r}")
    if attn_p_format not in ("uint8", "fp8", "none"):
        raise ValueError(f"attn_p_format must be uint8 / fp8 / none, got {attn_p_format!r}")
    if attn_pv_accum not in ("fp32", "fp16"):
        raise ValueError(f"attn_pv_accum must be fp32 / fp16, got {attn_pv_accum!r}")
    if attn_k_scope not in ("all", "gen", "und"):
        raise ValueError(f"attn_k_scope must be all / gen / und, got {attn_k_scope!r}")
    _runtime.__dict__.update(QdqSimEdgeRuntime().__dict__)  # restore every default first (experimental options too)
    _runtime.edges = frozenset(edges)
    _runtime.method = method
    _runtime.group_size = group_size
    _runtime.residual_group_size = residual_group_size or group_size
    _runtime.residual_bits = residual_bits
    _runtime.attn_v_block_size = attn_v_block_size
    _runtime.attn_smoothing = attn_smoothing
    _runtime.attn_v_format = attn_v_format
    _runtime.attn_p_format = attn_p_format
    _runtime.attn_pv_accum = attn_pv_accum
    _runtime.attn_k_scope = attn_k_scope
    _apply_attn_opts_from_env()
    _state.reset()


def reset_sim_edges() -> None:
    configure_sim_edges((), method="int8_sim", group_size=0)


def sim_edges_enabled() -> frozenset[str]:
    return _runtime.edges


def edge_enabled(edge: str) -> bool:
    return _runtime.enabled(edge)


# ----------------------------------------------------------------------------- quantizers
def _rows(values: torch.Tensor) -> torch.Tensor:
    """Signed Q/DQ with one scale per row of the last dimension (whole row = one group)."""
    from cosmos_framework.utils.generator.quantization import _fake_quant  # local import: avoid cycle

    return _fake_quant(values, _runtime.method, per_row=True, group_size=0)


def _groups(values: torch.Tensor, group_size: int) -> torch.Tensor:
    from cosmos_framework.utils.generator.quantization import _fake_quant  # local import: avoid cycle

    return _fake_quant(values, _runtime.method, per_row=True, group_size=group_size)


def fake_quant_uint8(values: torch.Tensor, *, group_size: int) -> torch.Tensor:
    """Unsigned 8-bit Q/DQ for non-negative tensors: one scale per ``group_size`` along the last dim.

    A trailing partial group (last dim not divisible by ``group_size``) gets its own scale.
    """
    values32 = values.float()
    n = values32.shape[-1]
    full = (n // group_size) * group_size

    def _q(x: torch.Tensor, g: int) -> torch.Tensor:
        work = x.reshape(*x.shape[:-1], x.shape[-1] // g, g)
        scale = (work.amax(dim=-1, keepdim=True) / _UINT8_QMAX).clamp_(min=torch.finfo(torch.float32).tiny)
        return (torch.round(work / scale).clamp_(0.0, _UINT8_QMAX) * scale).reshape(x.shape)

    parts = []
    if full:
        parts.append(_q(values32[..., :full], group_size))
    if n - full:
        parts.append(_q(values32[..., full:], n - full))
    return torch.cat(parts, dim=-1).to(values.dtype)


def fake_quant_int16(values: torch.Tensor, *, group_size: int) -> torch.Tensor:
    """Symmetric 16-bit integer Q/DQ (qmax 32767), one scale per ``group_size`` along the last dim."""
    from cosmos_framework.utils.generator.quantization import _grouped_view  # local import: avoid cycle

    values32 = values.float()
    work = _grouped_view(values32, group_size)
    scale = (work.abs().amax(dim=-1, keepdim=True) / 32767.0).clamp_(min=torch.finfo(torch.float32).tiny)
    quantized = torch.round(work / scale).clamp_(-32767.0, 32767.0)
    return (quantized * scale).reshape(values.shape).to(values.dtype)


def fake_quant_edge(values: torch.Tensor, edge: str) -> torch.Tensor:
    """Group-wise Q/DQ along the last dim for the ``gemm_out`` / ``residual`` edges (identity if disabled)."""
    if not _runtime.enabled(edge) or values.numel() == 0:
        return values
    if edge == "residual":
        if _runtime.residual_bits == 16:
            return fake_quant_int16(values, group_size=_runtime.residual_group_size)
        return _groups(values, _runtime.residual_group_size)
    return _groups(values, _runtime.group_size)


# ----------------------------------------------------------------------------- attention operands
def _qk_method() -> str:
    return {"int8": "int8_sim", "fp8": "fp8_sim"}.get(_runtime.attn_qk_format, _runtime.method)


def quant_k_rows(key: torch.Tensor) -> torch.Tensor:
    """Q or K with one scale per (token, head): constant along head_dim, the reduction dim of Q·Kᵀ.

    ``key``: [..., S, H, D] (or [S, H, D]); the last dim must be head_dim. Format: ``attn_qk_format``
    when set, else the edge method.
    """
    from cosmos_framework.utils.generator.quantization import _fake_quant  # local import: avoid cycle

    return _fake_quant(key, _qk_method(), per_row=True, group_size=0)


def quant_v_channels(value: torch.Tensor, *, seq_dim: int) -> torch.Tensor:
    """V with one scale per (head, channel) over the keys (optionally per block of keys).

    The scale is constant along the key dim, the reduction dim of P·V, so an 8-bit
    P·V MMA can apply it after accumulation. ``seq_dim`` is the key/token dim.
    """
    if _runtime.attn_v_format == "none":
        return value
    from cosmos_framework.utils.generator.quantization import fake_quant_fp8  # local import: avoid cycle

    rows = (lambda t: fake_quant_fp8(t, per_row=True)) if _runtime.attn_v_format == "fp8" else _rows
    block = _runtime.attn_v_block_size
    moved = value.movedim(seq_dim, -1)  # [..., H, D, S]
    if block and moved.shape[-1] > block:
        chunks = [rows(c) for c in moved.split(block, dim=-1)]  # per (block, h, d)
        out = torch.cat(chunks, dim=-1)
    else:
        out = rows(moved)  # per (h, d) over all keys
    return out.movedim(-1, seq_dim)


def quant_probs(probs: torch.Tensor) -> torch.Tensor:
    """P (softmax output, in [0,1]) in the configured format.

    ``uint8``: one scale per (query row, ``attn_p_block_size`` keys). ``fp8``: E4M3 with a fixed
    scale (448 = FP8 max maps 1.0), the SageAttention2 choice -- the exponent keeps small
    probabilities at relative precision, no per-block statistics needed.
    """
    fmt = _runtime.attn_p_format
    if fmt == "none":
        return probs
    if fmt == "fp8":
        p32 = probs.float() * 448.0
        return (p32.to(torch.float8_e4m3fn).float() / 448.0).to(probs.dtype)
    return fake_quant_uint8(probs, group_size=_runtime.attn_p_block_size)


def quantize_cached_kv(key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``und_kv`` edge: cached text K per (token, head), cached text V per (head, channel). K/V: [B, S, H, D]."""
    if not _runtime.enabled("und_kv") or key.numel() == 0:
        return key, value
    return quant_k_rows(key), quant_v_channels(value, seq_dim=1)


def _smooth_k(key: torch.Tensor) -> torch.Tensor:
    """K − mean_over_keys(K) per (head, channel). Softmax over keys is invariant to it."""
    return key - key.float().mean(dim=1, keepdim=True).to(key.dtype)


# ----------------------------------------------------------------------------- experimental options / state
def _parse_attn_opts(text: str) -> dict:
    """``QDQ_SIM_ATTN_OPTS="outlier_frac=0.01,q_smoothing=1,qk_block=128,bf16_first_steps=6,bf16_layers=0-1-35,err_log=/p.csv"``."""
    opts: dict = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, _, val = item.partition("=")
        key = key.strip()
        if key == "outlier_frac":
            opts["attn_outlier_frac"] = float(val)
        elif key == "q_smoothing":
            opts["attn_q_smoothing"] = val.strip().lower() in ("1", "true", "yes")
        elif key == "qk_block":
            opts["attn_qk_block_tokens"] = int(val)
        elif key == "bf16_first_steps":
            opts["attn_bf16_first_steps"] = int(val)
        elif key == "bf16_layers":
            opts["attn_bf16_layers"] = tuple(int(x) for x in val.split("-") if x != "")
        elif key == "qk_bf16_layers":
            opts["attn_qk_bf16_layers"] = tuple(int(x) for x in val.split("-") if x != "")
        elif key == "pv_bf16_layers":
            opts["attn_pv_bf16_layers"] = tuple(int(x) for x in val.split("-") if x != "")
        elif key == "layers_per_forward":
            opts["attn_layers_per_forward"] = int(val)
        elif key == "err_log":
            opts["attn_err_log"] = val.strip()
        elif key == "qk_format":
            if val.strip() not in ("int8", "fp8"):
                raise ValueError("qk_format must be int8 or fp8")
            opts["attn_qk_format"] = val.strip()
        elif key == "qk_balance":
            opts["attn_qk_balance"] = val.strip().lower() in ("1", "true", "yes")
        elif key == "qk_balance_alpha":
            opts["attn_qk_balance_alpha"] = float(val)
        elif key == "qk_hadamard":
            opts["attn_qk_hadamard"] = val.strip().lower() in ("1", "true", "yes")
        else:
            raise ValueError(f"unknown QDQ_SIM_ATTN_OPTS key {key!r}")
    return opts


def _apply_attn_opts_from_env() -> None:
    text = os.environ.get("QDQ_SIM_ATTN_OPTS", "")
    opts = _parse_attn_opts(text) if text else {}
    for name, value in opts.items():
        setattr(_runtime, name, value)
    if opts:
        import logging

        logging.getLogger(__name__).info(f"qdq-sim attention options from QDQ_SIM_ATTN_OPTS: {opts}")


class _SimState:
    """Sampler step / attention call counters (maintained in eager runs; unused under torch.compile)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.call_idx = 0
        self.layer = -1  # set by the decoder-layer loop (outside compile); -1 = unknown -> derived from call_idx
        self.step_t: torch.Tensor | None = None  # device scalar mirror of ``step`` (compile-safe gating)
        self.layer_t: torch.Tensor | None = None  # device scalar mirror of ``layer`` (compile-safe gating)
        self._log_fh = None
        self._log_rows = 0

    def log(self, row: str) -> None:
        if self._log_fh is None:
            os.makedirs(os.path.dirname(os.path.abspath(_runtime.attn_err_log)), exist_ok=True)
            self._log_fh = open(_runtime.attn_err_log, "a")
            if os.path.getsize(_runtime.attn_err_log) == 0:
                self._log_fh.write("step,call,layer,branch,s_q,s_kv,n_bf16_keys,gated,rel_out,rel_k,rel_v,rel_q\n")
        self._log_fh.write(row + "\n")
        self._log_rows += 1
        if self._log_rows % 36 == 0:
            self._log_fh.flush()


_state = _SimState()


def set_sampler_step(step: int) -> None:
    """Called once per sampler step (by the velocity_fn wrapper, outside compile); resets the per-step
    attention call counter and refreshes the device-side step scalar used for compile-safe step gating."""
    _state.step = int(step)
    _state.call_idx = 0
    if torch.cuda.is_available():
        if _state.step_t is None:
            _state.step_t = torch.zeros((), dtype=torch.int32, device="cuda")
        _state.step_t.fill_(int(step))


def set_layer(layer: int) -> None:
    """Called by the decoder-layer loop (outside the per-layer compiled forward) so the attention
    simulator knows its layer. Kept both as a Python int (eager gating / logging) and as a device
    scalar (compiled gating: a graph input, so no per-layer recompiles)."""
    _state.layer = int(layer)
    if torch.cuda.is_available():
        if _state.layer_t is None:
            _state.layer_t = torch.zeros((), dtype=torch.int32, device="cuda")
        _state.layer_t.fill_(int(layer))


# ----------------------------------------------------------------------------- attention helpers
def _smooth_q(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Q − mean_over_queries(Q) per (head, channel); the mean is added back after Q/DQ (exact: the
    kernel would add the per-key bias q̄·kⱼ to every logit)."""
    q_mean = query.float().mean(dim=1, keepdim=True).to(query.dtype)
    return query - q_mean, q_mean


def _rows_block(values: torch.Tensor, block: int) -> torch.Tensor:
    """Q/DQ with one scale per (block of ``block`` tokens, head): values [B, S, H, D]; a trailing
    partial block gets its own scale."""
    from cosmos_framework.utils.generator.quantization import _fake_quant  # local import: avoid cycle

    parts = []
    for lo in range(0, values.shape[1], block):
        chunk = values[:, lo : lo + block]  # [B, b, H, D]
        flat = chunk.permute(0, 2, 1, 3).reshape(chunk.shape[0], chunk.shape[2], -1)  # [B, H, b*D]
        q = _fake_quant(flat, _qk_method(), per_row=True, group_size=0)
        parts.append(q.reshape(chunk.shape[0], chunk.shape[2], chunk.shape[1], chunk.shape[3]).permute(0, 2, 1, 3))
    return torch.cat(parts, dim=1)


def _quant_rows_or_blocks(values: torch.Tensor) -> torch.Tensor:
    block = _runtime.attn_qk_block_tokens
    return _rows_block(values, block) if block > 0 else quant_k_rows(values)


def _outlier_keys(k_s: torch.Tensor, v_s: torch.Tensor, und: torch.Tensor, n_und: int, frac: float) -> torch.Tensor:
    """Top ``frac`` of the gen keys by peak ratio (max over heads of ‖row‖∞ relative to the median gen
    key, for smoothed K and smoothed V). Static top-k so it traces under torch.compile. Bool [S_kv]."""
    s_kv = k_s.shape[1]
    k_top = max(0, int(round(frac * (s_kv - n_und))))
    mask = torch.zeros(s_kv, dtype=torch.bool, device=k_s.device)
    if k_top == 0:
        return mask
    k_peak = k_s.float().abs().amax(dim=-1)[0].amax(dim=-1)  # [S]
    v_peak = v_s.float().abs().amax(dim=-1)[0].amax(dim=-1)  # [S]
    nan = torch.full_like(k_peak, float("nan"))
    k_med = torch.where(und, nan, k_peak).nanmedian().clamp_min(1e-12)
    v_med = torch.where(und, nan, v_peak).nanmedian().clamp_min(1e-12)
    score = torch.maximum(k_peak / k_med, v_peak / v_med)
    score = torch.where(und, torch.full_like(score, float("-inf")), score)
    top = torch.topk(score, k_top).indices
    return mask.scatter(0, top, True)


def _bf16_key_mask(
    k_s: torch.Tensor, v_s: torch.Tensor, und_key_mask: torch.Tensor | None, n_und: int | None = None
) -> torch.Tensor:
    """Keys that stay bf16 in K and V: the text keys when ``attn_k_scope == "gen"``, plus the top
    ``attn_outlier_frac`` outlying gen keys. ``n_und``: number of text keys (Python int) for static top-k."""
    s_kv = k_s.shape[1]
    und = (
        und_key_mask.to(k_s.device)
        if und_key_mask is not None
        else torch.zeros(s_kv, dtype=torch.bool, device=k_s.device)
    )
    if n_und is None:
        n_und = int(und.sum().item()) if und_key_mask is not None else 0
    if _MASK_OVERRIDE == "zeros":
        und = torch.zeros_like(und)
        n_und = 0
    elif _MASK_OVERRIDE == "ones":
        und = torch.ones_like(und)
        n_und = s_kv
    keep = und.clone() if _runtime.attn_k_scope == "gen" else torch.zeros_like(und)
    if _runtime.attn_outlier_frac > 0:
        keep |= _outlier_keys(k_s, v_s, und, n_und, _runtime.attn_outlier_frac)
    return keep


_HADAMARD_CACHE: dict = {}
_LAST_QT: dict = {}  # error-log helper: transformed unrounded Q of the last _quant_qk call


def _hadamard(n: int, device, dtype) -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix (Sylvester construction), n a power of two. H Hᵀ = I."""
    key = (n, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        if n & (n - 1):
            raise ValueError(f"Hadamard transform needs a power-of-two head_dim, got {n}")
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        _HADAMARD_CACHE[key] = (h / (n**0.5)).to(device=device, dtype=dtype)
    return _HADAMARD_CACHE[key]


def _qk_transforms(q_c: torch.Tensor, k_c: torch.Tensor):
    """Exact Q/K pre-transforms (S = QKᵀ unchanged): channel balancing between Q and K per kv-head, then an
    optional Hadamard rotation along head_dim. Returns (T_q, T_k) callables applied to [B,S,H,D] tensors."""
    steps_q, steps_k = [], []
    if _runtime.attn_qk_balance:
        heads, kv_heads = q_c.shape[2], k_c.shape[2]
        rep = heads // kv_heads
        aq = (
            q_c.float().abs().reshape(q_c.shape[0], q_c.shape[1], kv_heads, rep, q_c.shape[3]).amax(dim=(0, 1, 3))
        )  # [H_kv, D]
        ak = k_c.float().abs().amax(dim=(0, 1))  # [H_kv, D]
        s_bal = ((ak.clamp_min(1e-6) / aq.clamp_min(1e-6)) ** _runtime.attn_qk_balance_alpha).clamp(
            1e-2, 1e2
        )  # [H_kv, D]
        s_q = s_bal.repeat_interleave(rep, dim=0).view(1, 1, heads, -1)
        s_k = s_bal.view(1, 1, kv_heads, -1)
        steps_q.append(lambda t: (t.float() * s_q).to(t.dtype))
        steps_k.append(lambda t: (t.float() / s_k).to(t.dtype))
    if _runtime.attn_qk_hadamard:
        h = _hadamard(q_c.shape[-1], q_c.device, torch.float32)
        steps_q.append(lambda t: (t.float() @ h).to(t.dtype))
        steps_k.append(lambda t: (t.float() @ h).to(t.dtype))

    def apply(steps):
        def f(t):
            for st in steps:
                t = st(t)
            return t

        return f

    return apply(steps_q), apply(steps_k)


def _quant_qk(query: torch.Tensor, key: torch.Tensor, bf16_keys: torch.Tensor, und_key_mask: torch.Tensor | None):
    """``attn_qkv`` on Q and K. K keeps the smoothing shift on every key (softmax-invariant), only the
    keys outside ``bf16_keys`` (and inside the scope) are rounded. Optional exact pre-transforms
    (channel balancing, Hadamard) are applied to both Q and K before rounding; the kernel sees the
    transformed operands (Q·Kᵀ is unchanged), so bf16 keys carry the transformed-but-unrounded K."""
    scope = _runtime.attn_k_scope
    k_c = _smooth_k(key) if _runtime.attn_smoothing else key
    if scope == "und":
        if und_key_mask is None:
            raise ValueError("attn_k_scope='und' needs und_key_mask")
        quantize_key = und_key_mask.to(key.device)
        k_q = _quant_rows_or_blocks(k_c)
        return query, torch.where(quantize_key.view(1, -1, 1, 1), k_q, k_c), k_c
    q_c, q_mean = _smooth_q(query) if _runtime.attn_q_smoothing else (query, None)
    t_q, t_k = _qk_transforms(q_c, k_c)
    q_t, k_t = t_q(q_c), t_k(k_c)
    k_q = _quant_rows_or_blocks(k_t)
    quantize_key = ~bf16_keys
    key_out = torch.where(quantize_key.view(1, -1, 1, 1), k_q, k_t)
    query_out = _quant_rows_or_blocks(q_t)
    if q_mean is not None:
        query_out = query_out + t_q(q_mean)  # exact add-back: T_q(q̄)·T_k(k) = q̄·k
        q_t = q_t + t_q(q_mean)
    _LAST_QT["q_t"] = q_t  # for the error log only
    return query_out, key_out, k_t


def _quant_qk_scoped(query: torch.Tensor, key: torch.Tensor, und_key_mask: torch.Tensor | None):
    """Backwards-compatible entry (tests): Q/K quantization with the scope rule, no V-based outliers."""
    scope = _runtime.attn_k_scope
    if scope != "all" and und_key_mask is None:
        raise ValueError(f"attn_k_scope={scope!r} needs und_key_mask (which keys are text)")
    if und_key_mask is not None and (und_key_mask.dtype != torch.bool or und_key_mask.shape != (key.shape[1],)):
        raise ValueError(
            f"und_key_mask must be bool [S_kv={key.shape[1]}], got {und_key_mask.dtype} {tuple(und_key_mask.shape)}"
        )
    k_s = _smooth_k(key) if _runtime.attn_smoothing else key
    bf16_keys = _bf16_key_mask(k_s, key, und_key_mask)
    q_out, k_out, _ = _quant_qk(query, key, bf16_keys, und_key_mask)
    return q_out, k_out


def _quant_v_masked(v_src: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    """V per-(head, channel) Q/DQ over the keys with keep == False; kept keys pass through unchanged.
    Formats: int8 (qmax 127) or fp8 (E4M3, scale amax/448). Shapes: v_src [B, S, H, D], keep [S]."""
    fmt = _runtime.attn_v_format
    v32 = v_src.float()
    mag = v32.abs()
    if keep is not None:
        mag = torch.where(keep.to(v_src.device).view(1, -1, 1, 1), torch.zeros_like(mag), mag)
    amax = mag.amax(dim=1, keepdim=True)  # [B,1,H,D] over the quantized keys
    if fmt == "fp8":
        scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
        q = (v32 / scale).to(torch.float8_e4m3fn).float() * scale
    else:
        scale = (amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
        q = torch.round(v32 / scale).clamp_(-127.0, 127.0) * scale
    q = q.to(v_src.dtype)
    if keep is None:
        return q
    return torch.where(keep.to(v_src.device).view(1, -1, 1, 1), v_src, q)


def _quant_v(value: torch.Tensor, bf16_keys: torch.Tensor | None):
    """V smoothing + per-(head, channel) Q/DQ over the quantized keys; keys in ``bf16_keys`` keep their
    (shifted) bf16 values. Returns (v_used = quantized(V − m), m) with m = None when smoothing is off."""
    if _runtime.attn_v_format == "none":
        return value, None
    v_mean = None
    v_src = value
    if _runtime.attn_smoothing:
        v_mean = value.float().mean(dim=1, keepdim=True)  # [B,1,H_kv,D]
        v_src = (value.float() - v_mean).to(value.dtype)
    if _runtime.attn_v_block_size and bf16_keys is None:
        return quant_v_channels(v_src, seq_dim=1), v_mean  # legacy per-key-block option
    return _quant_v_masked(v_src, bf16_keys), v_mean


def reference_attention_qdq(
    query: torch.Tensor,  # [B,S_q,H,D]
    key: torch.Tensor,  # [B,S_kv,H_kv,D]
    value: torch.Tensor,  # [B,S_kv,H_kv,D]
    *,
    scale: float | None = None,
    quantize_pv: bool = True,
    bf16_keys: torch.Tensor | None = None,
) -> torch.Tensor:  # [B,S_q,H,D]
    """Dense fp32 reference attention with the ``attn_pv`` Q/DQ on P and V.

    P: unsigned 8-bit, one scale per (query row, ``attn_p_block_size`` keys) or FP8 E4M3.
    V: 8-bit, one scale per (head, channel) over the quantized keys; when smoothing is on,
    V − mean_keys(V) is quantized and the mean is added back to O (exact because Σ_j P_ij = 1).
    Keys in ``bf16_keys`` keep bf16 V. Q/K are taken as given (quantized by ``attn_qkv`` when
    enabled).
    """
    batch, s_q, heads, head_dim = query.shape
    kv_heads = key.shape[2]
    if heads % kv_heads:
        raise ValueError(f"query heads {heads} not divisible by kv heads {kv_heads}")
    rep = heads // kv_heads
    if scale is None:
        scale = head_dim**-0.5
    v_mean = None
    v_use = value
    if quantize_pv:
        v_use, v_mean = _quant_v(value, bf16_keys)
    q = query.float().permute(0, 2, 1, 3)  # [B,H,S_q,D]
    k = key.float().permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)  # [B,H,S_kv,D]
    v = v_use.float().permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)  # [B,H,S_kv,D]
    out = torch.empty_like(q)
    chunk = max(1, min(s_q, (2048 * 1024 * 1024) // max(1, k.shape[2] * heads * 4)))
    for start in range(0, s_q, chunk):
        stop = min(s_q, start + chunk)
        probs = torch.softmax(torch.matmul(q[:, :, start:stop], k.transpose(-1, -2)) * scale, dim=-1)
        if quantize_pv:
            probs = quant_probs(probs)
        if _runtime.attn_pv_accum == "fp16":
            out[:, :, start:stop] = _pv_fp16_accumulate(probs, v, _runtime.attn_accum_block)
        else:
            out[:, :, start:stop] = torch.matmul(probs, v)
    out = out.permute(0, 2, 1, 3)  # [B,S_q,H,D]
    if v_mean is not None:
        out = out + v_mean.repeat_interleave(rep, dim=2)  # Σ_j P_ij = 1 → add the removed mean back
    return out.to(query.dtype)


def _pv_fp16_accumulate(probs: torch.Tensor, v: torch.Tensor, block: int) -> torch.Tensor:
    """P·V with fp16 operands and an fp16 accumulator rounded after every ``block`` keys.

    Mirrors an HMMA with .f16 accumulator (k = 16 per instruction): each block's partial
    dot products are formed at higher precision, then added into the fp16 running sum.
    ``probs``: [B,H,c,S_kv] fp32, ``v``: [B,H,S_kv,D] fp32 -> returns fp32 [B,H,c,D].
    """
    p16 = probs.to(torch.float16)
    v16 = v.to(torch.float16)
    acc = torch.zeros(*probs.shape[:-1], v.shape[-1], dtype=torch.float16, device=probs.device)
    for lo in range(0, probs.shape[-1], block):
        hi = min(lo + block, probs.shape[-1])
        partial = torch.matmul(p16[..., lo:hi].float(), v16[..., lo:hi, :].float())
        acc = (acc.float() + partial).to(torch.float16)
    return acc.float()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)).item()


def _attention_variant(query, key, value, *, qkv: bool, pv: bool, und_key_mask, n_und, kwargs):
    """One evaluation of the simulated attention with the given edge switches (no gating, no logging).
    Returns (out, info) where info carries the operands for the error log."""
    from cosmos_framework.model.attention import attention

    k_s = _smooth_k(key) if _runtime.attn_smoothing else key
    bf16_keys = _bf16_key_mask(k_s, value, und_key_mask, n_und)
    q_used, k_used, v_used, v_mean = query, key, value, None
    if qkv:
        q_used, k_used, k_s = _quant_qk(query, key, bf16_keys, und_key_mask)
    # dense fp32 reference when P is quantized, the accumulator is not fp32, or no fused kernel exists (CPU tests)
    dense = pv and (_runtime.attn_p_format != "none" or _runtime.attn_pv_accum != "fp32" or not query.is_cuda)
    if pv and not dense:
        v_used, v_mean = _quant_v(value, bf16_keys)
    if dense:
        out = reference_attention_qdq(
            q_used, k_used, value, scale=kwargs.get("scale"), quantize_pv=True, bf16_keys=bf16_keys
        )
    else:
        out = attention(q_used, k_used, v_used, **kwargs)
        if v_mean is not None:
            rep = query.shape[2] // key.shape[2]
            out = out + v_mean.to(out.dtype).repeat_interleave(rep, dim=2)
    info = dict(
        k_s=k_s,
        k_used=k_used,
        q_used=q_used,
        q_t=_LAST_QT.get("q_t", query) if qkv else query,
        bf16_keys=bf16_keys,
        v_used=v_used,
        v_mean=v_mean,
        dense=dense,
    )
    return out, info


def _isin_const(layer_t: torch.Tensor, layers: tuple[int, ...]) -> torch.Tensor:
    """0-dim bool: layer_t in layers (tensor op, so it traces without value guards)."""
    if not layers:
        return torch.zeros((), dtype=torch.bool, device=layer_t.device)
    table = torch.tensor(layers, dtype=layer_t.dtype, device=layer_t.device)
    return (table == layer_t).any()


def sim_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    und_key_mask: torch.Tensor | None = None,
    n_und: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Attention with the ``attn_qkv`` / ``attn_pv`` edges applied on the full (und + gen) K/V.

    ``attn_qkv``: Q (optionally smoothed) and K (smoothed) per (token, head) or per token block;
    ``attn_pv``: 8-bit P and per-channel V. Keys in the bf16 set (text keys under scope "gen",
    plus ``attn_outlier_frac`` outlying gen keys) keep bf16 K and V. When P is not quantized and
    the accumulator is fp32 the fused kernel is used on the dequantized operands (mathematically
    identical to the dense reference); otherwise the dense fp32 reference runs.
    Step / layer gating: Python branches in eager mode; under torch.compile the gates are device
    scalars (``_state.step_t`` / ``_state.layer_t``, refreshed outside the compiled frames) and the
    gated variants are blended with ``torch.where`` (every needed variant is computed).
    Neither edge enabled → the normal kernel. Layouts: [B, S, H, D]. ``und_key_mask``: bool
    [S_kv], True on the text (und) keys; ``n_und``: their count (Python int).
    """
    from cosmos_framework.model.attention import attention

    qkv_on, pv_on = _runtime.enabled("attn_qkv"), _runtime.enabled("attn_pv")
    if not (qkv_on or pv_on):
        return attention(query, key, value, **kwargs)
    eager = not torch.compiler.is_compiling()
    call_idx = _state.call_idx
    if eager:
        _state.call_idx += 1
    layers_per_fwd = max(1, _runtime.attn_layers_per_forward)
    branch = call_idx // layers_per_fwd
    all_l, qk_l, pv_l = _runtime.attn_bf16_layers, _runtime.attn_qk_bf16_layers, _runtime.attn_pv_bf16_layers
    first_steps = _runtime.attn_bf16_first_steps
    layer_gating = bool(all_l or qk_l or pv_l)

    if eager:
        layer = _state.layer if _state.layer >= 0 else call_idx % layers_per_fwd
        if layer in all_l or _state.step < first_steps:
            qkv_on = pv_on = False
        qkv_on = qkv_on and layer not in qk_l
        pv_on = pv_on and layer not in pv_l
        if not (qkv_on or pv_on):
            out = attention(query, key, value, **kwargs)
            if _runtime.attn_err_log:
                _state.log(f"{_state.step},{call_idx},{layer},{branch},{query.shape[1]},{key.shape[1]},0,1,0,0,0,0")
            return out
        out, info = _attention_variant(
            query, key, value, qkv=qkv_on, pv=pv_on, und_key_mask=und_key_mask, n_und=n_und, kwargs=kwargs
        )
        if _runtime.attn_err_log:
            ref = attention(query, key, value, **kwargs)
            rel_k = _rel(info["k_used"], info["k_s"]) if qkv_on else 0.0
            rel_q = _rel(info["q_used"], info.get("q_t", query)) if qkv_on else 0.0  # vs the transformed, unrounded Q
            if pv_on:
                v_cmp, m = (info["v_used"], info["v_mean"]) if not info["dense"] else _quant_v(value, info["bf16_keys"])
                v_ref = (value.float() - m).to(value.dtype) if m is not None else value
                rel_v = _rel(v_cmp, v_ref)
            else:
                rel_v = 0.0
            _state.log(
                f"{_state.step},{call_idx},{layer},{branch},{query.shape[1]},{key.shape[1]},{int(info['bf16_keys'].sum())},0,"
                f"{_rel(out, ref):.6g},{rel_k:.6g},{rel_v:.6g},{rel_q:.6g}"
            )
        return out

    # ---- compiled path: tensor gates + where-blend (no Python reads of step / layer)
    out_full, _ = _attention_variant(
        query, key, value, qkv=qkv_on, pv=pv_on, und_key_mask=und_key_mask, n_und=n_und, kwargs=kwargs
    )
    if not layer_gating and first_steps <= 0:
        return out_full
    out = out_full
    if layer_gating:
        if _state.layer_t is None:
            raise RuntimeError("attention layer gating under torch.compile needs set_layer() from the decoder loop")
        lt = _state.layer_t
        g_all = _isin_const(lt, all_l)
        g_qk = g_all | _isin_const(lt, qk_l)  # Q/K stays bf16 in this layer
        g_pv = g_all | _isin_const(lt, pv_l)  # P/V stays bf16 in this layer
        if qkv_on and pv_on:
            out_pv_only, _ = (
                _attention_variant(
                    query, key, value, qkv=False, pv=True, und_key_mask=und_key_mask, n_und=n_und, kwargs=kwargs
                )
                if qk_l or all_l
                else (out_full, None)
            )
            out_qk_only, _ = (
                _attention_variant(
                    query, key, value, qkv=True, pv=False, und_key_mask=und_key_mask, n_und=n_und, kwargs=kwargs
                )
                if pv_l or all_l
                else (out_full, None)
            )
            out_bf16 = attention(query, key, value, **kwargs)
            out = torch.where(
                g_qk & g_pv, out_bf16, torch.where(g_qk, out_pv_only, torch.where(g_pv, out_qk_only, out_full))
            )
        else:
            gate = g_qk if qkv_on else g_pv
            out = torch.where(gate, attention(query, key, value, **kwargs), out_full)
    if first_steps > 0:
        if _state.step_t is None:
            raise RuntimeError("attention step gating under torch.compile needs set_sampler_step()")
        out = torch.where(_state.step_t < first_steps, attention(query, key, value, **kwargs), out)
    return out


_DEBUG = os.environ.get("QDQ_SIM_DEBUG", "") == "1"
_MASK_OVERRIDE = os.environ.get("QDQ_SIM_MASK_OVERRIDE", "")  # "" / "zeros" / "ones" (debugging aid)


# backwards-compatible name used by earlier call sites
sdpa_or_reference = sim_attention

__all__ = [
    "SIM_EDGES",
    "configure_sim_edges",
    "edge_enabled",
    "fake_quant_edge",
    "fake_quant_int16",
    "fake_quant_uint8",
    "quant_k_rows",
    "quant_probs",
    "quant_v_channels",
    "quantize_cached_kv",
    "reference_attention_qdq",
    "reset_sim_edges",
    "sdpa_or_reference",
    "sim_attention",
    "sim_edges_enabled",
    "set_sampler_step",
    "set_layer",
]
