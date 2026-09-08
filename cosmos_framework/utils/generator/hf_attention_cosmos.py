# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HF ``ALL_ATTENTION_FUNCTIONS`` adapter delegating to ``cosmos_framework.model.attention``.

Registered as the ``"cosmos"`` entry in HF's attention dispatch.
``cosmos_framework.model.attention`` owns backend selection (cuDNN / NATTEN / flash2 /
flash3); to fall back to HF's own flash_attention_2 set
``policy.attn_implementation=flash_attention_2``.

Layout: HF passes Q/K/V as BHSD ``[B, num_heads, N, head_dim]`` and expects
BSHD output. ``cosmos_framework.model.attention`` is BSHD throughout, so we transpose on
entry; output layout already matches HF's expected return.

Varlen: HF's flash-attention protocol passes ``cu_seq_lens_{q,k}`` / ``max_length_{q,k}``
for sequence-packed batches, which map onto ``cosmos_framework.model.attention``'s
``cumulative_seqlen_{Q,KV}`` / ``max_seqlen_{Q,KV}``. Backend selection accounts for it —
cuDNN has no varlen integration and its checker rejects the call, so the dispatcher lands
on flash2/flash3 — meaning callers get varlen by passing the kwargs and nothing else.

Strict guards (raise rather than silently break loss parity):
- ``dropout > 0`` — ``cosmos_framework.model.attention`` has no dropout parameter.
  Qwen3-VL has ``attention_dropout=0`` so this never triggers in practice.
- ``attention_mask is not None`` — adapter expects causal mask via
  ``is_causal=True`` (and no padding, i.e. Qwen3-VL VLM training with
  ``max_batch_size=1``). A 4-D additive mask would need explicit handling.
- ``max_length_{q,k}`` given as tensors — see :func:`hf_attention_cosmos`.

True sequence packing: when the batch is a single
``B=1`` concatenated row, the standard ``cu_seq_lens_{q,k}`` / ``max_length_{q,k}``
metadata are threaded down as kwargs
(``VLMModel.training_step`` -> ``Qwen3VLForConditionalGeneration.forward(**kwargs)`` ->
... -> this adapter). When present, we run **block-diagonal varlen causal** attention
via ``cosmos_framework.model.attention``'s varlen path (one segment per packed sample plus the FP8
pad tail) so each sample attends causally only within itself — semantically identical to
the per-row causal mask of padded batching. ``attention_mask`` stays ``None`` in this mode
(it is popped in ``training_step`` and ``create_causal_mask`` early-exits under cosmos).

Backend: with ``is_causal=True`` + ``CausalType.TopLeft`` + varlen, the only compatible
``cosmos_framework.model.attention`` backend is **NATTEN** (``blackwell-fmha`` on GB200/sm100,
``hopper-fmha`` on sm90, ``cutlass-fmha`` fallback) — NOT Flash: flash2 varlen is banned
and flash2/flash3 only accept ``BottomRight``/``DontCare`` causal, while cuDNN varlen is
not integrated (see ``cosmos_framework/attention/*/checks.py``).

Narrow causal-varlen contract: true-packed text uses block-diagonal causal
self-attention with shared Q/KV boundaries (``B=1``, ``q_len == kv_len``). Those
extra guards apply only when ``module.is_causal`` is true; the existing non-causal
Qwen vision-tower varlen path remains supported.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from cosmos_framework.model.attention import attention as imag_attention
from cosmos_framework.model.attention.masks import CausalType


def hf_attention_cosmos(
    module: torch.nn.Module,
    query: torch.Tensor,  # [B, num_heads, N, head_dim] (BHSD)
    key: torch.Tensor,  # [B, num_kv_heads, N_kv, head_dim]
    value: torch.Tensor,  # [B, num_kv_heads, N_kv, head_dim]
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,  # [num_sequences+1]
    cu_seq_lens_k: torch.Tensor | None = None,  # [num_sequences+1]
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """HF-protocol attention callable that delegates to cosmos_framework.model.attention.

    Returns ``(attn_output, attn_weights)``. ``attn_weights`` is always ``None``
    — the in-house attention does not materialize the attention matrix, and
    Qwen3-VL does not consume ``attn_weights``.

    Passing ``cu_seq_lens_{q,k}`` and ``max_length_{q,k}`` selects varlen attention over a
    sequence-packed batch (batch dim 1, sequences concatenated without padding), which is
    how the Qwen3-VL vision tower packs the images and frames of a step.

    ``max_length_{q,k}`` must be Python ints. HF's own flash-attention call sites pass the
    tensor ``(cu_seqlens[1:] - cu_seqlens[:-1]).max()`` instead, and accepting that would
    mean calling ``.item()`` here — a device-to-host copy, per attention module, per
    forward, that stalls the CPU until the queued GPU work drains. The caller is expected
    to compute the value once for the whole tower instead (see
    ``monkey_patch.patch_qwen3_vl_vision_varlen_attention``), so a tensor here is a bug
    worth surfacing rather than silently paying for.
    """
    if dropout != 0.0:
        raise NotImplementedError(
            f"cosmos adapter does not support dropout > 0 (got {dropout}); "
            "cosmos_framework.model.attention has no dropout parameter. "
            "Qwen3-VL config has attention_dropout=0 so this should never trigger."
        )

    if attention_mask is not None:
        raise NotImplementedError(
            "cosmos adapter does not support explicit attention_mask. "
            "Qwen3-VL VLM training with max_batch_size=1 should pass None here "
            "(causal mask is handled via is_causal=True). If you hit this assert, "
            "either the batch contains padding — pack it and pass cu_seq_lens_{q,k} / "
            "max_length_{q,k} instead — or a 4-D additive mask was supplied (needs "
            "explicit handling). Silently ignoring it would break loss parity with "
            "the HF FA2 baseline."
        )
    if sliding_window is not None:
        raise NotImplementedError(
            "cosmos adapter does not support sliding_window. Qwen3-VL VLM training should pass None here."
        )

    legacy_varlen_keys = {"cu_seqlens", "max_seqlen"} & kwargs.keys()
    if legacy_varlen_keys:
        raise ValueError(
            f"legacy varlen aliases {sorted(legacy_varlen_keys)} are unsafe because they can silently "
            "select dense attention; use cu_seq_lens_q/cu_seq_lens_k and max_length_q/max_length_k"
        )

    is_causal = bool(getattr(module, "is_causal", False))
    is_varlen = cu_seq_lens_q is not None or cu_seq_lens_k is not None
    if is_varlen:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            raise ValueError(
                f"varlen attention needs both cu_seq_lens_q and cu_seq_lens_k, got {cu_seq_lens_q=}, {cu_seq_lens_k=}."
            )
        if isinstance(max_length_q, torch.Tensor) or isinstance(max_length_k, torch.Tensor):
            raise TypeError(
                "max_length_q/max_length_k must be Python ints, not tensors: reading a tensor here "
                "costs a device-to-host sync per attention call. Compute the maximum once per "
                "forward and pass the int (see monkey_patch.patch_qwen3_vl_vision_varlen_attention)."
            )
        if max_length_q is None or max_length_k is None:
            raise ValueError(
                f"varlen attention needs both max_length_q and max_length_k, got {max_length_q=}, {max_length_k=}."
            )
        if is_causal and cu_seq_lens_q is not cu_seq_lens_k:
            raise ValueError("causal true-packed self-attention requires one shared Q/KV boundary tensor")
        # cosmos_framework.model.attention requires int32; HF's vision tower already builds cu_seqlens as
        # int32 (except under torch.jit tracing), so this is normally a no-op and never a sync.
        cu_seq_lens_q = cu_seq_lens_q.to(torch.int32)
        cu_seq_lens_k = cu_seq_lens_k.to(torch.int32)
        if is_causal:
            if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
                raise NotImplementedError(
                    "causal varlen attention is reserved for a single B=1 true-packed row; "
                    f"got q={query.shape[0]}, k={key.shape[0]}, v={value.shape[0]}"
                )
            if not (query.shape[-2] == key.shape[-2] == value.shape[-2]):
                raise NotImplementedError(
                    "causal varlen attention requires self-attention with q_len == k_len == v_len; "
                    f"got q={query.shape[-2]}, k={key.shape[-2]}, v={value.shape[-2]}"
                )
            if cu_seq_lens_q.shape != cu_seq_lens_k.shape or max_length_q != max_length_k:
                raise ValueError(
                    "causal varlen attention requires identical Q/KV segmentation; "
                    f"got shapes {cu_seq_lens_q.shape}/{cu_seq_lens_k.shape} and "
                    f"max lengths {max_length_q}/{max_length_k}"
                )

    # BHSD -> BSHD
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # Cast fp32 -> bf16 if needed.
    # cosmos_framework's flash2/flash3/cuDNN backends only accept fp16/bf16; NATTEN
    # also accepts fp32 but routing fp32 attention loses Tensor Core
    # acceleration (10-20x slower). HF's flash_attention_2 internally casts
    # fp32->bf16 and we replicate that so this adapter is a drop-in replacement
    # and performance-equivalent regardless of which backend gets selected.
    # In practice FSDP2's mp_policy almost always hands us bf16 already, so
    # this branch is taken rarely.
    orig_dtype = q.dtype
    needs_cast = orig_dtype not in (torch.float16, torch.bfloat16)
    if needs_cast:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    causal_type = CausalType.TopLeft if is_causal else None

    out = imag_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        causal_type=causal_type,
        scale=scaling,
        cumulative_seqlen_Q=cu_seq_lens_q,
        cumulative_seqlen_KV=cu_seq_lens_k,
        max_seqlen_Q=max_length_q,
        max_seqlen_KV=max_length_k,
    )
    # imag_attention returns a bare Tensor unless return_lse=True (which we don't pass);
    # cast narrows the Tensor | tuple[Tensor, Tensor] return type.
    out = cast(torch.Tensor, out)

    if needs_cast:
        out = out.to(orig_dtype)

    # out is BSHD [B, N, num_heads, head_dim_v] — matches HF's expected return shape.
    return out, None
