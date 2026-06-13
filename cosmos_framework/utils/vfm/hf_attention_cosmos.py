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

Strict guards (raise rather than silently break loss parity):
- ``dropout > 0`` — ``cosmos_framework.model.attention`` has no dropout parameter.
  Qwen3-VL has ``attention_dropout=0`` so this never triggers in practice.
- ``attention_mask is not None`` — adapter expects causal mask via
  ``is_causal=True`` (and no padding, i.e. Qwen3-VL VLM training with
  ``max_batch_size=1``). A 4-D additive mask would need explicit handling.
"""

from __future__ import annotations

from typing import Any

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
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """HF-protocol attention callable that delegates to cosmos_framework.model.attention.

    Returns ``(attn_output, attn_weights)``. ``attn_weights`` is always ``None``
    — the in-house attention does not materialize the attention matrix, and
    Qwen3-VL does not consume ``attn_weights``.
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
            "either the batch contains padding (need varlen routing) or a 4-D "
            "additive mask was supplied (need explicit handling). Silently "
            "ignoring it would break loss parity with the HF FA2 baseline."
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

    is_causal = bool(getattr(module, "is_causal", False))
    causal_type = CausalType.TopLeft if is_causal else None

    out = imag_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        causal_type=causal_type,
        scale=scaling,
    )

    if needs_cast:
        out = out.to(orig_dtype)

    # out is BSHD [B, N, num_heads, head_dim_v] — matches HF's expected return shape.
    return out, None
