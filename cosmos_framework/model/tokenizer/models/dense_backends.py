# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Backend selection helpers for the dense tokenizer runtime."""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.model.tokenizer.models.modules.attention.full_attn import (
    tensor_dense_scaled_dot_product_attention,
)
from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import (
    SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_FULL_LAYER,
    SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_MLP_ONLY,
    SPARSE_TRANSFORMER_CHECKPOINT_SCOPES,
)

DenseRuntimeBackend = Literal["varlen", "batched", "batched_with_padding", "auto"]
DenseResolvedBackend = Literal["varlen", "batched", "batched_with_padding"]


def _validate_checkpoint_group_size(checkpoint_group_size: int) -> None:
    """Validate one dense-stack grouped-checkpointing request."""
    if (
        isinstance(checkpoint_group_size, bool)
        or not isinstance(checkpoint_group_size, int)
        or checkpoint_group_size < 1
    ):
        raise ValueError(f"checkpoint_group_size must be a positive integer, got {checkpoint_group_size!r}.")


def _can_group_checkpoint_blocks(blocks: nn.ModuleList, checkpoint_group_size: int) -> bool:
    """Return whether every block can move under one checkpoint per group."""
    if checkpoint_group_size <= 1 or len(blocks) == 0:
        return False
    return all(
        block.training
        and getattr(block, "use_checkpoint", False)
        and _block_gradient_checkpoint_scope(block) == SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_FULL_LAYER
        for block in blocks
    )


def _block_gradient_checkpoint_scope(block: nn.Module) -> str:
    """Resolve and validate one block checkpoint scope."""
    scope = getattr(block, "gradient_checkpoint_scope", SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_FULL_LAYER)
    if not isinstance(scope, str) or scope not in SPARSE_TRANSFORMER_CHECKPOINT_SCOPES:
        raise ValueError(
            f"Unsupported block gradient_checkpoint_scope={scope!r}; "
            f"expected one of {sorted(SPARSE_TRANSFORMER_CHECKPOINT_SCOPES)}."
        )
    return scope


def _run_varlen_checkpoint_group(
    feats: torch.Tensor,
    *,
    blocks: tuple[nn.Module, ...],
    q_seqlen: list[int],
    cu_seqlens_q: torch.Tensor,
    max_q_seqlen: int,
    q_freqs_cis: torch.Tensor | None,
) -> torch.Tensor:
    """Run consecutive varlen blocks with their inner checkpoints disabled."""
    output = feats  # [T,D]
    for block in blocks:
        output = block.forward_tensor_no_cache(  # [T,D]
            output,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            q_freqs_cis=q_freqs_cis,
            checkpoint_override=False,
        )
    return output  # [T,D]


def _run_batched_checkpoint_group(
    feats: torch.Tensor,
    *,
    blocks: tuple[nn.Module, ...],
    cu_seqlens_q: torch.Tensor | None,
    max_q_seqlen: int | None,
    q_freqs_cis: torch.Tensor | None,
) -> torch.Tensor:
    """Run consecutive batched blocks inside one outer checkpoint."""
    output = feats  # [B,S,D]
    for block in blocks:
        output = run_batched_block(
            block,
            output,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            q_freqs_cis=q_freqs_cis,
        )  # [B,S,D]
    return output  # [B,S,D]


def resolve_dense_backend(backend: DenseRuntimeBackend, use_compile: bool) -> DenseResolvedBackend:
    """Resolve the dense-runtime backend for the current execution mode.

    Args:
        backend: Requested backend mode.
        use_compile: Whether the caller intends to run under ``torch.compile``.

    Returns:
        Concrete backend name.

    Raises:
        ValueError: If ``backend`` is not one of the supported values.
    """
    if backend == "auto":
        return "batched" if use_compile else "varlen"
    if backend in ("varlen", "batched", "batched_with_padding"):
        return backend
    raise ValueError(f"Unsupported dense runtime backend: {backend}")


def run_varlen_block_stack(
    blocks: nn.ModuleList,
    feats: torch.Tensor,
    q_seqlen: list[int],
    cu_seqlens_q: torch.Tensor,
    max_q_seqlen: int,
    q_freqs_cis: torch.Tensor | None = None,
    checkpoint_group_size: int = 1,
) -> torch.Tensor:
    """Run the tensor no-cache block path while preserving 2D or 3D input shape."""
    _validate_checkpoint_group_size(checkpoint_group_size)
    if feats.ndim not in (2, 3):
        raise ValueError(f"Varlen dense backend expects [T, D] or [B, S, D] features, got shape {tuple(feats.shape)}.")

    if len(blocks) == 0:
        return feats

    input_shape = feats.shape
    flat_feats = feats if feats.ndim == 2 else feats.reshape(-1, feats.shape[-1])  # [T,D]
    if _can_group_checkpoint_blocks(blocks, checkpoint_group_size):
        for group_start in range(0, len(blocks), checkpoint_group_size):
            checkpoint_blocks = tuple(blocks[group_start : group_start + checkpoint_group_size])
            flat_feats = torch.utils.checkpoint.checkpoint(  # [T,D]
                partial(
                    _run_varlen_checkpoint_group,
                    blocks=checkpoint_blocks,
                    q_seqlen=q_seqlen,
                    cu_seqlens_q=cu_seqlens_q,
                    max_q_seqlen=max_q_seqlen,
                    q_freqs_cis=q_freqs_cis,
                ),
                flat_feats,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        return flat_feats if feats.ndim == 2 else flat_feats.reshape(input_shape)  # [T,D] or [B,S,D]

    for block in blocks:
        flat_feats = block.forward_tensor_no_cache(
            flat_feats,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            q_freqs_cis=q_freqs_cis,
        )
    return flat_feats if feats.ndim == 2 else flat_feats.reshape(input_shape)  # [T,D] or [B,S,D]


def run_batched_block_stack(
    blocks: nn.ModuleList,
    feats: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    max_q_seqlen: int | None = None,
    q_freqs_cis: torch.Tensor | None = None,
    checkpoint_group_size: int = 1,
) -> torch.Tensor:
    """Run the dense batched block path over uniform `[B, S, D]` chunks."""
    _validate_checkpoint_group_size(checkpoint_group_size)
    if feats.ndim != 3:
        raise ValueError(f"Batched dense backend expects [B, S, D] features, got shape {tuple(feats.shape)}.")

    output = feats
    if _can_group_checkpoint_blocks(blocks, checkpoint_group_size):
        for group_start in range(0, len(blocks), checkpoint_group_size):
            checkpoint_blocks = tuple(blocks[group_start : group_start + checkpoint_group_size])
            output = torch.utils.checkpoint.checkpoint(  # [B,S,D]
                partial(
                    _run_batched_checkpoint_group,
                    blocks=checkpoint_blocks,
                    cu_seqlens_q=cu_seqlens_q,
                    max_q_seqlen=max_q_seqlen,
                    q_freqs_cis=q_freqs_cis,
                ),
                output,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        return output

    for block in blocks:
        checkpoint_scope = _block_gradient_checkpoint_scope(block)
        if (
            block.training
            and getattr(block, "use_checkpoint", False)
            and checkpoint_scope == SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_FULL_LAYER
        ):
            output = torch.utils.checkpoint.checkpoint(
                partial(
                    run_batched_block,
                    block,
                    cu_seqlens_q=cu_seqlens_q,
                    max_q_seqlen=max_q_seqlen,
                    q_freqs_cis=q_freqs_cis,
                ),
                output,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        else:
            output = run_batched_block(
                block, output, cu_seqlens_q=cu_seqlens_q, max_q_seqlen=max_q_seqlen, q_freqs_cis=q_freqs_cis
            )
    return output


def run_batched_block(
    block: nn.Module,
    feats: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    max_q_seqlen: int | None = None,
    q_freqs_cis: torch.Tensor | None = None,
    checkpoint_override: bool | None = None,
) -> torch.Tensor:
    """Run one batched block, optionally checkpointing only its MLP residual."""
    if getattr(block, "multiscale", None) is not None:
        raise NotImplementedError("Dense runtime batched backend does not support multiscale blocks.")
    if getattr(block.attn, "_type", None) != "self":
        raise NotImplementedError("Dense runtime batched backend only supports self-attention blocks.")

    checkpoint_scope = _block_gradient_checkpoint_scope(block)
    use_checkpoint = getattr(block, "use_checkpoint", False) if checkpoint_override is None else checkpoint_override
    feats = _run_batched_attention_residual(  # [B,S,D]
        block,
        feats,
        cu_seqlens_q=cu_seqlens_q,
        max_q_seqlen=max_q_seqlen,
        q_freqs_cis=q_freqs_cis,
    )
    if block.training and use_checkpoint and checkpoint_scope == SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_MLP_ONLY:
        return torch.utils.checkpoint.checkpoint(  # [B,S,D]
            partial(_run_batched_mlp_residual, block),
            feats,
            preserve_rng_state=True,
            use_reentrant=False,
        )
    return _run_batched_mlp_residual(block, feats)  # [B,S,D]


def _run_batched_attention_residual(
    block: nn.Module,
    feats: torch.Tensor,  # [B,S,D]
    *,
    cu_seqlens_q: torch.Tensor | None = None,  # [B+1]
    max_q_seqlen: int | None = None,
    q_freqs_cis: torch.Tensor | None = None,  # [B*S,D_rope]
) -> torch.Tensor:
    """Apply one batched attention residual without the MLP residual."""
    residual = feats  # [B,S,D]
    h = block.norm1(feats)  # [B,S,D]
    h = run_batched_attention(  # [B,S,D]
        block.attn, h, cu_seqlens_q=cu_seqlens_q, max_q_seqlen=max_q_seqlen, q_freqs_cis=q_freqs_cis
    )
    return residual + h  # [B,S,D]


def _run_batched_mlp_residual(block: nn.Module, feats: torch.Tensor) -> torch.Tensor:  # [B,S,D]
    """Apply norm2 and one batched MLP residual."""
    residual = feats  # [B,S,D]
    h = block.norm2(feats)  # [B,S,D]
    h = block.mlp.forward_tensor(h)  # [B,S,D]
    return residual + h  # [B,S,D]


def run_batched_attention(
    attention: nn.Module,
    feats: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    max_q_seqlen: int | None = None,
    q_freqs_cis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one dense self-attention layer via the cosmos_framework attention frontend."""
    if not hasattr(attention, "to_qkv"):
        raise ValueError("Dense runtime batched backend requires fused to_qkv linear projections.")
    if not hasattr(attention, "to_out"):
        raise ValueError("Dense runtime batched backend requires an output projection linear layer.")

    # feats: [B, S_padded, hidden]  (S_padded = pad_to tokens per batch item, padded for CUDA graph)
    batch_size, seq_len, hidden_size = feats.shape
    # qkv: [B, S_padded, 3, H, D]
    qkv = F.linear(feats, attention.to_qkv.weight, attention.to_qkv.bias).reshape(
        batch_size,
        seq_len,
        3,
        attention.num_heads,
        -1,
    )
    # q, k, v: [B, S_padded, H, D]
    q, k, v = qkv.unbind(dim=2)

    if getattr(attention, "qk_rms_norm", False):
        # flatten to [B*S_padded, H, D] for per-token RMSNorm, then restore
        flat_q = q.reshape(batch_size * seq_len, attention.num_heads, -1)
        flat_k = k.reshape(batch_size * seq_len, attention.num_heads, -1)
        q = attention.q_rms_norm(flat_q).reshape(batch_size, seq_len, attention.num_heads, -1)
        k = attention.k_rms_norm(flat_k).reshape(batch_size, seq_len, attention.num_heads, -1)

    if getattr(attention, "use_rope", False):
        if q_freqs_cis is None:
            raise ValueError("Dense runtime batched backend requires precomputed q_freqs_cis when RoPE is enabled.")
        # flatten to [B*S_padded, H, D] for RoPE application, then restore to [B, S_padded, H, D]
        flat_q = q.reshape(batch_size * seq_len, attention.num_heads, -1)
        flat_k = k.reshape(batch_size * seq_len, attention.num_heads, -1)
        flat_q, flat_k = attention.rope.apply_rotary_emb(
            flat_q,
            flat_k,
            freqs_cis=q_freqs_cis,
            xk_freqs_cis=q_freqs_cis,
        )
        q = flat_q.reshape(batch_size, seq_len, attention.num_heads, -1)
        k = flat_k.reshape(batch_size, seq_len, attention.num_heads, -1)

    # q, k, v: [B, S_padded, H, D] → attention → h: [B, S_padded, H, D]
    h = tensor_dense_scaled_dot_product_attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_q,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_q_seqlen,
    )
    # h: [B, S_padded, hidden]
    h = h.reshape(batch_size, seq_len, hidden_size)
    return F.linear(h, attention.to_out.weight, attention.to_out.bias)
