# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Full scaled dot-product attention for sparse tensors.

This module provides full attention implementations supporting:
    - Self-attention with packed QKV
    - Cross-attention with separate Q and KV
    - Separate Q, K, V tensors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cosmos_framework.model.attention.frontend import attention as i4_attention
from cosmos_framework.model.attention.varlen import generate_varlen_parameters
from cosmos_framework.model.tokenizer.utils.tensors import cat_with_bounded_inputs

if TYPE_CHECKING:
    from cosmos_framework.model.tokenizer.models.modules.sparse_tensor import SparseTensor


__all__ = [
    "sparse_scaled_dot_product_attention",
    "tensor_dense_scaled_dot_product_attention",
    "tensor_varlen_scaled_dot_product_attention",
]


def _generate_varlen_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: list[int],
    kv_seqlen: list[int],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build varlen metadata once via cosmos_framework.model.attention utilities.

    This path is intended for the generic sparse-attention codepaths that do
    not already receive precomputed varlen metadata from upstream. Tensor fast
    paths should continue to pass cumulative seqlens and max lengths directly.
    """
    q_seqlens_tensor = torch.tensor(q_seqlen, dtype=torch.int32, device=q.device)
    kv_seqlens_tensor = torch.tensor(kv_seqlen, dtype=torch.int32, device=q.device)
    cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen = generate_varlen_parameters(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        q_seqlens_tensor,
        kv_seqlens_tensor,
    )
    assert cu_seqlens_q is not None
    assert cu_seqlens_kv is not None
    return cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen


def tensor_varlen_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_q_seqlen: int,
    max_kv_seqlen: int,
) -> torch.Tensor:
    """Apply tokenizer packed varlen attention through cosmos_framework.model.attention."""
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], v.shape[-1]))

    out = i4_attention(
        query=q.unsqueeze(0).contiguous(),
        key=k.unsqueeze(0).contiguous(),
        value=v.unsqueeze(0).contiguous(),
        cumulative_seqlen_Q=cu_seqlens_q,
        cumulative_seqlen_KV=cu_seqlens_kv,
        max_seqlen_Q=max_q_seqlen,
        max_seqlen_KV=max_kv_seqlen,
    )
    return out.squeeze(0)


def tensor_dense_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
    max_q_seqlen: int | None = None,
    max_kv_seqlen: int | None = None,
) -> torch.Tensor:
    """Apply dense batched attention via the cosmos_framework attention frontend."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "Dense tensor attention expects [B, S, H, D]-style tensors, "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], q.shape[2], v.shape[-1]))

    return i4_attention(
        query=q.contiguous(),
        key=k.contiguous(),
        value=v.contiguous(),
        cumulative_seqlen_Q=cu_seqlens_q,
        cumulative_seqlen_KV=cu_seqlens_kv,
        max_seqlen_Q=max_q_seqlen,
        max_seqlen_KV=max_kv_seqlen,
    )


def _cacheable_tensor_version(tensor: torch.Tensor) -> int:
    """Return a tensor version suitable for a coordinate-derived cache key."""
    try:
        return tensor._version
    except RuntimeError:
        # Inference-mode tensors do not expose version counters. Sparse coordinates
        # are treated as immutable, matching the other coordinate caches.
        return -1


TemporalCausalAttentionGroup = tuple[int, int, int, int]


@dataclass(frozen=True)
class SparseTemporalCausalAttentionPlan:
    """Linear-size execution plan for temporal block-causal self-attention."""

    groups: tuple[TemporalCausalAttentionGroup, ...]
    token_order: torch.Tensor | None
    inverse_token_order: torch.Tensor | None


def _sparse_coordinates_are_aligned(lhs: "SparseTensor", rhs: "SparseTensor") -> bool:
    """Return whether two sparse tensors use the same coordinate rows and layout."""
    if lhs.coords.shape != rhs.coords.shape or lhs.layout != rhs.layout or lhs.coords.device != rhs.coords.device:
        return False
    if lhs.coords.data_ptr() == rhs.coords.data_ptr():
        return True
    return torch.equal(lhs.coords, rhs.coords)


def _build_sparse_temporal_causal_plan(q: "SparseTensor") -> SparseTemporalCausalAttentionPlan:
    """Build timestep query slices and their inclusive K/V prefixes.

    The normal decoder layout is already batch/time ordered and therefore needs
    only ``O(batch * timesteps)`` Python slice metadata. A non-canonical but
    batch-contiguous input is stable-sorted once with cached ``O(tokens)``
    permutations; it never falls back to a token-square mask.
    """
    groups: list[TemporalCausalAttentionGroup] = []
    batch_orders: list[torch.Tensor | None] = []
    requires_reordering = False

    for batch_idx, batch_slice in enumerate(q.layout):
        if batch_slice.start == batch_slice.stop:
            batch_orders.append(None)
            continue

        batch_times = q.coords[batch_slice, 1].contiguous()  # [Tb]
        canonical_times = torch.where(batch_times < 0, -1, batch_times)  # [Tb]
        if canonical_times.numel() > 1:
            nondecreasing = torch.all(canonical_times[1:] >= canonical_times[:-1])  # []
            batch_is_sorted = bool(nondecreasing.item())
        else:
            batch_is_sorted = True

        if batch_is_sorted:
            sorted_times = canonical_times  # [Tb]
            batch_orders.append(None)
        else:
            relative_order = torch.argsort(canonical_times, stable=True)  # [Tb]
            sorted_times = canonical_times.index_select(0, relative_order)  # [Tb]
            absolute_order = relative_order + batch_slice.start  # [Tb]
            batch_orders.append(absolute_order)
            requires_reordering = True

        run_times_tensor, run_counts_tensor = torch.unique_consecutive(  # run_times_tensor: [G], run_counts_tensor: [G]
            sorted_times,
            return_counts=True,
        )
        run_times = [int(value) for value in run_times_tensor.tolist()]
        run_counts = [int(value) for value in run_counts_tensor.tolist()]
        if any(current_time <= previous_time for previous_time, current_time in zip(run_times, run_times[1:])):
            raise ValueError(f"Temporal coordinates could not be ordered for causal attention batch {batch_idx}.")

        run_start = batch_slice.start
        special_end = batch_slice.start
        regular_started = False
        for run_time, run_count in zip(run_times, run_counts, strict=True):
            run_end = run_start + run_count
            if run_time < 0:
                if regular_started:
                    raise ValueError(f"Special tokens must precede regular timesteps in batch {batch_idx}.")
                special_end = run_end
            else:
                if not regular_started and special_end > batch_slice.start:
                    groups.append((batch_slice.start, special_end, batch_slice.start, special_end))
                regular_started = True
                groups.append((run_start, run_end, batch_slice.start, run_end))
            run_start = run_end

        if not regular_started and special_end > batch_slice.start:
            groups.append((batch_slice.start, special_end, batch_slice.start, special_end))
        if run_start != batch_slice.stop:
            raise RuntimeError(
                f"Temporal causal plan did not cover batch {batch_idx}: end={run_start}, expected={batch_slice.stop}."
            )

    token_order: torch.Tensor | None = None
    inverse_token_order: torch.Tensor | None = None
    if requires_reordering:
        order_parts: list[torch.Tensor] = []
        for batch_slice, batch_order in zip(q.layout, batch_orders, strict=True):
            if batch_order is None:
                batch_order = torch.arange(batch_slice.start, batch_slice.stop, device=q.device)  # [Tb]
            order_parts.append(batch_order)
        token_order = cat_with_bounded_inputs(order_parts, dim=0)  # [Tq]
        canonical_indices = torch.arange(token_order.shape[0], device=q.device)  # [Tq]
        inverse_token_order = torch.empty_like(token_order)  # [Tq]
        inverse_token_order[token_order] = canonical_indices  # [Tq]

    return SparseTemporalCausalAttentionPlan(
        groups=tuple(groups),
        token_order=token_order,
        inverse_token_order=inverse_token_order,
    )


def _get_sparse_temporal_causal_plan(q: "SparseTensor") -> SparseTemporalCausalAttentionPlan:
    """Return a cached linear-size temporal-causal execution plan."""
    cache_key = (
        "temporal_causal_attention_plan",
        q.coords.data_ptr(),
        _cacheable_tensor_version(q.coords),
        tuple(q.coords.shape),
    )
    cached_plan = q.get_spatial_cache(cache_key)
    if cached_plan is None:
        cached_plan = _build_sparse_temporal_causal_plan(q)
        q.register_spatial_cache(cache_key, cached_plan)
    if not isinstance(cached_plan, SparseTemporalCausalAttentionPlan):
        raise TypeError(f"Temporal causal plan cache has unexpected type {type(cached_plan).__name__}.")
    return cached_plan


def _unmasked_group_scaled_dot_product_attention(
    q: torch.Tensor,  # [Tq,H,D]
    k: torch.Tensor,  # [Tkv,H,D]
    v: torch.Tensor,  # [Tkv,H,Dv]
) -> torch.Tensor:  # returns [Tq,H,Dv]
    """Run one unmasked group through native fused SDPA when available."""
    q_heads_first = q.permute(1, 0, 2).unsqueeze(0)  # [1,H,Tq,D]
    k_heads_first = k.permute(1, 0, 2).unsqueeze(0)  # [1,H,Tkv,D]
    v_heads_first = v.permute(1, 0, 2).unsqueeze(0)  # [1,H,Tkv,Dv]
    if q.is_cuda:
        # A math fallback would retain token-square attention state for backward.
        # Require a fused CUDA implementation so this path stays linear-memory.
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            output_heads_first = F.scaled_dot_product_attention(
                q_heads_first,
                k_heads_first,
                v_heads_first,
                dropout_p=0.0,
                is_causal=False,
            )  # [1,H,Tq,Dv]
    else:
        output_heads_first = F.scaled_dot_product_attention(
            q_heads_first,
            k_heads_first,
            v_heads_first,
            dropout_p=0.0,
            is_causal=False,
        )  # [1,H,Tq,Dv]
    return output_heads_first.squeeze(0).permute(1, 0, 2).contiguous()  # [Tq,H,Dv]


def _sparse_temporal_causal_scaled_dot_product_attention(
    q: "SparseTensor",
    k: "SparseTensor",
    v: "SparseTensor",
) -> "SparseTensor":
    """Apply linear-memory temporal-causal attention one timestep group at a time."""
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            f"Batch size mismatch for temporal causal attention: q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
        )
    if q.feats.ndim != 3 or k.feats.ndim != 3 or v.feats.ndim != 3:
        raise ValueError(
            "Temporal causal attention expects [tokens, heads, channels] features, "
            f"got q={tuple(q.feats.shape)}, k={tuple(k.feats.shape)}, v={tuple(v.feats.shape)}."
        )
    if k.feats.shape[0] != v.feats.shape[0]:
        raise ValueError(
            "Temporal causal attention requires matching K/V token counts, "
            f"got k={k.feats.shape[0]} and v={v.feats.shape[0]}."
        )
    if not _sparse_coordinates_are_aligned(q, k) or not _sparse_coordinates_are_aligned(q, v):
        raise ValueError("Temporal causal attention requires aligned self-attention Q/K/V coordinates and layouts.")
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"Temporal causal attention requires Q/K/V on one device, got {q.device}, {k.device}, and {v.device}."
        )

    if q.feats.shape[0] == 0:
        return q.replace(q.feats.clone())  # [0,H,D]

    plan = _get_sparse_temporal_causal_plan(q)
    if plan.token_order is None:
        q_feats = q.feats.contiguous()  # [Tq,H,D]
        k_feats = k.feats.contiguous()  # [Tkv,H,D]
        v_feats = v.feats.contiguous()  # [Tkv,H,Dv]
    else:
        q_feats = q.feats.index_select(0, plan.token_order)  # [Tq,H,D]
        k_feats = k.feats.index_select(0, plan.token_order)  # [Tkv,H,D]
        v_feats = v.feats.index_select(0, plan.token_order)  # [Tkv,H,Dv]

    output_groups: list[torch.Tensor] = []
    for q_start, q_end, kv_start, kv_end in plan.groups:
        q_group = q_feats[q_start:q_end]  # [Tq_group,H,D]
        k_prefix = k_feats[kv_start:kv_end]  # [Tkv_prefix,H,D]
        v_prefix = v_feats[kv_start:kv_end]  # [Tkv_prefix,H,Dv]
        group_output = _unmasked_group_scaled_dot_product_attention(q_group, k_prefix, v_prefix)  # [Tq_group,H,Dv]
        output_groups.append(group_output)

    if not output_groups:
        raise RuntimeError("Temporal causal plan produced no attention groups for a non-empty query tensor.")
    canonical_out_feats = cat_with_bounded_inputs(output_groups, dim=0)  # [Tq,H,Dv]
    if canonical_out_feats.shape[0] != q.feats.shape[0]:
        raise RuntimeError(
            f"Temporal causal plan produced {canonical_out_feats.shape[0]} outputs for {q.feats.shape[0]} queries."
        )
    if plan.inverse_token_order is None:
        out_feats = canonical_out_feats  # [Tq,H,Dv]
    else:
        out_feats = canonical_out_feats.index_select(0, plan.inverse_token_order)  # [Tq,H,Dv]
    return q.replace(out_feats)


@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: "SparseTensor", v: "SparseTensor") -> torch.Tensor:
    """Apply scaled dot product attention to a sparse tensor.

    Args:
        q: A [N, L, H, Ci] dense tensor containing Qs.
        k: A [N, *, H, Ci] sparse tensor containing Ks.
        v: A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...


def sparse_scaled_dot_product_attention(*args: Any, **kwargs: Any) -> "SparseTensor" | torch.Tensor:
    """Flexible scaled dot-product attention for sparse tensors.

    Supports three calling conventions:
        1. Single packed QKV tensor: qkv of shape [N, *, 3, H, C]
        2. Separate Q and packed KV: q of shape [N, *, H, C], kv of shape [N, *, 2, H, C]
        3. Separate Q, K, V tensors: q, k, v each of shape [N, *, H, C]

    Args:
        *args: Positional arguments (qkv, or q+kv, or q+k+v).
        **kwargs: Keyword arguments for the above.

    Returns:
        Attention output with same structure as query input.
    """
    from cosmos_framework.model.tokenizer.models.modules.sparse_tensor import SparseTensor

    temporal_causal_mask = kwargs.pop("temporal_causal_mask", False)
    kv_seqlen_override = kwargs.pop("kv_seqlen", None)
    cu_seqlens_kv_override = kwargs.pop("cu_seqlens_kv", None)
    max_kv_seqlen_override = kwargs.pop("max_kv_seqlen", None)
    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args) :]:
        assert key in kwargs, f"Missing argument {key}"

    if temporal_causal_mask:
        q_arg = args[0] if len(args) > 0 else kwargs["q"]
        k_arg = args[1] if len(args) > 1 else kwargs["k"]
        v_arg = args[2] if len(args) > 2 else kwargs["v"]
        if num_all_args != 3:
            raise ValueError("temporal_causal_mask only supports separate q, k, v inputs.")
        if not (
            isinstance(q_arg, SparseTensor) and isinstance(k_arg, SparseTensor) and isinstance(v_arg, SparseTensor)
        ):
            raise ValueError("temporal_causal_mask requires sparse q, k, v inputs.")
        return _sparse_temporal_causal_scaled_dot_product_attention(q_arg, k_arg, v_arg)

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, (
            f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        )
        device = qkv.device

        s = qkv
        q_seqlen = qkv.get_batch_seq_lens()
        kv_seqlen = q_seqlen
        qkv = qkv.feats  # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(kv, (SparseTensor, torch.Tensor))
            or isinstance(q, torch.Tensor)
            and isinstance(kv, SparseTensor)
        ), f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = q.get_batch_seq_lens()
            q = q.feats  # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)  # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, (
                f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            )
            kv_seqlen = kv.get_batch_seq_lens()
            kv = kv.feats  # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)  # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(k, (SparseTensor, torch.Tensor))
            and type(k) == type(v)
            or isinstance(q, torch.Tensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor)
        ), f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        packed_flat_kv = (
            isinstance(q, SparseTensor)
            and isinstance(k, torch.Tensor)
            and isinstance(v, torch.Tensor)
            and len(k.shape) == 3
            and len(v.shape) == 3
        )
        if packed_flat_kv:
            assert kv_seqlen_override is not None or cu_seqlens_kv_override is not None, (
                "Packed flat KV tensors require kv_seqlen or cu_seqlens_kv overrides."
            )
            if kv_seqlen_override is not None:
                assert q.shape[0] == len(kv_seqlen_override), (
                    f"Batch size mismatch, got {q.shape[0]} query batches and {len(kv_seqlen_override)} KV segments."
                )
        else:
            assert q.shape[0] == k.shape[0] == v.shape[0], (
                f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
            )
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = q.get_batch_seq_lens()
            q = q.feats  # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = k.get_batch_seq_lens()
            k = k.feats  # [T_KV, H, Ci]
            v = v.feats  # [T_KV, H, Co]
        else:
            if len(k.shape) == 3 and len(v.shape) == 3:
                if kv_seqlen_override is None:
                    assert cu_seqlens_kv_override is not None
                    kv_seqlen_override = (
                        (cu_seqlens_kv_override[1:] - cu_seqlens_kv_override[:-1]).to(dtype=torch.int64).tolist()
                    )
                kv_seqlen = kv_seqlen_override
                if max_kv_seqlen_override is None:
                    max_kv_seqlen_override = max(kv_seqlen) if kv_seqlen else 0
            else:
                assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
                assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
                N, L, H, CI, CO = *k.shape, v.shape[-1]
                kv_seqlen = [L] * N
                k = k.reshape(N * L, H, CI)  # [T_KV, H, Ci]
                v = v.reshape(N * L, H, CO)  # [T_KV, H, Co]

    if num_all_args == 1:
        q, k, v = qkv.unbind(dim=1)
    elif num_all_args == 2:
        k, v = kv.unbind(dim=1)

    if num_all_args in [1, 2, 3]:
        cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen = _generate_varlen_metadata(
            q=q,
            k=k,
            v=v,
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )
        if cu_seqlens_kv_override is not None:
            cu_seqlens_kv = cu_seqlens_kv_override.to(device=device, dtype=torch.int32)
            max_kv_seqlen = max_kv_seqlen_override if max_kv_seqlen_override is not None else max_kv_seqlen

    out = tensor_varlen_scaled_dot_product_attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_kv_seqlen,
    )

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
