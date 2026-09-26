# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal replay using the upstream maskless sensor/caption passes and LSE merge.

Replay roles determine each query group's keys. Grouping happens once before the
decoder, over metadata runs rather than a quadratic token mask. Queries sharing
the same keys share one varlen group, including across frames in a replay chunk.
The two sensor passes deliberately overlap, as in bidirectional maskless attention.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import accumulate

import attrs
import torch

from cosmos_framework.model.attention import attention
from cosmos_framework.model.attention.backends import choose_backend
from cosmos_framework.model.attention.natten import (
    NATTEN_BLACKWELL_DETERMINISTIC_VERSION,
    NATTEN_SUPPORTED,
    natten_version_satisfies,
)
from cosmos_framework.model.attention.natten.checks import choose_natten_backend
from cosmos_framework.model.attention.utils.environment import is_torch_compiling
from cosmos_framework.model.generator.mot.flex_attention import SensorMaskItem
from cosmos_framework.model.generator.mot.merge_bridge import MergeAttentionsBridge
from cosmos_framework.model.generator.mot.multiview_maskless_attention import (
    _gather_from_packed,
    _scatter_to_packed,
)
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    _ROLE_PADDING,
    _ROLE_UND,
    TeacherForcingFlexMetadata,
    _key_stream_fields,
    _query_stream_fields,
    _stream_metadata_groups,
    _StreamFields,
    _teacher_forcing_pair_predicate,
)
from cosmos_framework.model.generator.mot.merge_attention import merge_attentions_ac_safe

if NATTEN_SUPPORTED and natten_version_satisfies(NATTEN_BLACKWELL_DETERMINISTIC_VERSION):
    import natten.backends.blackwell_fmha as _blackwell_fmha
else:
    _blackwell_fmha = None

_REPLAY_KV_BATCH_TOKENS = 1 << 22


@dataclass(frozen=True)
class ReplayMasklessPass:
    """Nonempty varlen groups for one upstream maskless partition."""

    name: str
    q_gather: torch.Tensor  # [N_Q_pass]
    kv_gather: torch.Tensor  # [N_KV_pass]
    q_offsets: torch.Tensor  # [G+1]
    kv_offsets: torch.Tensor  # [G+1]
    q_max_len: int
    kv_max_len: int


@dataclass(frozen=True)
class ReplayMasklessPlan:
    """Layer-independent plan in global (post-Ulysses) Q/KV coordinates."""

    passes: tuple[ReplayMasklessPass, ...]
    q_len: int
    kv_len: int
    real_queries: torch.Tensor  # [Q]
    batches: tuple[ReplayMasklessPass, ...] = ()
    compiled_indices: tuple[torch.Tensor, ...] = ()  # Gather indices and packed batch offsets
    batch_layout: torch.Tensor | None = None  # [B,9] on CPU


def _batch_pass(partition: ReplayMasklessPass) -> tuple[ReplayMasklessPass, ...]:
    """Bound repeated KV gathers without splitting an independent attention group."""
    q_offsets = partition.q_offsets.cpu().tolist()  # list[G+1]
    kv_offsets = partition.kv_offsets.cpu().tolist()  # list[G+1]
    batches: list[ReplayMasklessPass] = []
    first = 0
    while first < len(q_offsets) - 1:
        last = first + 1
        while last < len(q_offsets) - 1 and kv_offsets[last + 1] - kv_offsets[first] <= _REPLAY_KV_BATCH_TOKENS:
            last += 1
        batches.append(
            ReplayMasklessPass(
                name=partition.name,
                q_gather=partition.q_gather[q_offsets[first] : q_offsets[last]],  # [N_Q_batch]
                kv_gather=partition.kv_gather[kv_offsets[first] : kv_offsets[last]],  # [N_KV_batch]
                q_offsets=partition.q_offsets[first : last + 1] - q_offsets[first],  # [G_batch+1]
                kv_offsets=partition.kv_offsets[first : last + 1] - kv_offsets[first],  # [G_batch+1]
                q_max_len=max(q_offsets[i + 1] - q_offsets[i] for i in range(first, last)),
                kv_max_len=max(kv_offsets[i + 1] - kv_offsets[i] for i in range(first, last)),
            )
        )
        first = last
    return tuple(batches)


def _attend_batch(
    q: torch.Tensor,  # [1,N_Q_batch,H,D]
    k: torch.Tensor,  # [1,N_KV_batch,H_KV,D]
    v: torch.Tensor,  # [1,N_KV_batch,H_KV,D]
    batch: ReplayMasklessPass,
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,N_Q_batch,H,D], [1,N_Q_batch,H]
    return attention(
        q,
        k,
        v,
        cumulative_seqlen_Q=batch.q_offsets,
        cumulative_seqlen_KV=batch.kv_offsets,
        max_seqlen_Q=batch.q_max_len,
        max_seqlen_KV=batch.kv_max_len,
        return_lse=True,
    )  # [1,N_Q_batch,H,D], [1,N_Q_batch,H]


def _blackwell_backward_config(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
) -> tuple[int, int] | None:
    """Reuse the selected native backward without changing backend preferences."""
    if _blackwell_fmha is None or q.device.type != "cuda":
        return None
    deterministic = torch.are_deterministic_algorithms_enabled()
    backend = choose_backend(
        query_shape=q.shape,
        key_shape=k.shape,
        value_shape=v.shape,
        dtype=q.dtype,
        device=q.device,
        requires_grad=True,
        is_causal=False,
        causal_type=None,
        is_varlen=True,
        deterministic=deterministic,
        return_lse=True,
        is_compiling=is_torch_compiling(),
        raise_error=False,
    )
    if (
        backend != "natten"
        or choose_natten_backend(
            query_shape=q.shape,
            key_shape=k.shape,
            value_shape=v.shape,
            dtype=q.dtype,
            device=q.device,
            requires_grad=True,
            is_causal=False,
            is_varlen=True,
            deterministic=deterministic,
        )
        != "blackwell-fmha"
    ):
        return None
    return _blackwell_fmha.check_cutlass_blackwell_fmha_backward_config(input_tensor=q)


def _merge_batch(
    output: torch.Tensor,  # [1,Q,H,D], FP32 accumulation
    merged_lse: torch.Tensor,  # [1,Q,H], FP32
    indices: torch.Tensor,  # [N_Q_batch], unique within this batch
    out: torch.Tensor,  # [1,N_Q_batch,H,D]
    lse: torch.Tensor,  # [1,N_Q_batch,H]
) -> None:
    previous_lse = merged_lse[:, indices]  # [1,N_Q_batch,H]
    combined_lse = torch.logaddexp(previous_lse, lse)  # [1,N_Q_batch,H]
    combined_out = output[:, indices] * torch.exp(previous_lse - combined_lse).unsqueeze(-1) + out.float() * torch.exp(
        lse - combined_lse
    ).unsqueeze(-1)  # [1,N_Q_batch,H,D]
    output.index_copy_(1, indices, combined_out)  # [1,Q,H,D]
    merged_lse.index_copy_(1, indices, combined_lse)  # [1,Q,H]


# Fuse merge intermediates independently of transformer compilation.
_compiled_merge_batch = torch.compile(_merge_batch, fullgraph=True, dynamic=True)


def _batched_forward(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    batches: tuple[ReplayMasklessPass, ...],
    *,
    compiled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,Q,H,D], [1,Q,H]
    output = torch.zeros_like(q, dtype=torch.float32)  # [1,Q,H,D]
    merged_lse = torch.full(q.shape[:-1], -torch.inf, dtype=torch.float32, device=q.device)  # [1,Q,H]
    merge = _compiled_merge_batch if compiled else _merge_batch
    for batch in batches:
        out, lse = _attend_batch(
            q[:, batch.q_gather],  # [1,N_Q_batch,H,D]
            k[:, batch.kv_gather],  # [1,N_KV_batch,H_KV,D]
            v[:, batch.kv_gather],  # [1,N_KV_batch,H_KV,D]
            batch,
        )  # [1,N_Q_batch,H,D], [1,N_Q_batch,H]
        lse = lse.squeeze(-1) if lse.ndim == 4 else lse  # [1,N_Q_batch,H]
        merge(output, merged_lse, batch.q_gather, out, lse)  # [1,Q,H,D], [1,Q,H], updated in place
        # The merged buffers own this group's result. Release its temporaries
        # before gathering the next group or casting the final output.
        del out, lse
    output = output.to(q.dtype)  # [1,Q,H,D]
    return output, merged_lse  # [1,Q,H,D], [1,Q,H]


def _batched_backward(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    output: torch.Tensor,  # [1,Q,H,D]
    merged_lse: torch.Tensor,  # [1,Q,H]
    grad_output: torch.Tensor,  # [1,Q,H,D]
    batches: tuple[ReplayMasklessPass, ...],
    backward_config: tuple[int, int] | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Q: [1,Q,H,D], KV: [1,KV,H_KV,D]
    gradients = [torch.zeros_like(tensor, dtype=torch.float32) for tensor in (q, k, v)]  # each input shape
    for batch in batches:
        indices = (batch.q_gather, batch.kv_gather, batch.kv_gather)  # [N_Q_batch], 2*[N_KV_batch]
        if backward_config is not None:
            assert _blackwell_fmha is not None
            inputs = [tensor[:, index] for tensor, index in zip((q, k, v), indices)]  # each gathered input shape
            # The native backward already accepts Q/K/V and final O/LSE.
            # Rebuilding its autograd graph would run a redundant forward.
            grads = _blackwell_fmha.blackwell_fmha_backward(
                *inputs,
                output[:, batch.q_gather],  # [1,N_Q_batch,H,D]
                grad_output[:, batch.q_gather],  # [1,N_Q_batch,H,D]
                merged_lse[:, batch.q_gather],  # [1,N_Q_batch,H]
                False,
                q.shape[-1] ** -0.5,
                *backward_config,
                batch.q_offsets,
                batch.kv_offsets,
                batch.q_max_len,
                batch.kv_max_len,
                deterministic,
            )  # each gathered input shape
        else:
            with torch.enable_grad():
                inputs = [
                    tensor[:, index].detach().requires_grad_() for tensor, index in zip((q, k, v), indices)
                ]  # [1,N_Q_batch,H,D], 2*[1,N_KV_batch,H_KV,D]
                out, lse = _attend_batch(*inputs, batch)  # [1,N_Q_batch,H,D], [1,N_Q_batch,H]
                # Native backward must read the final merged O/LSE, just as it
                # does through MergeAttentionsBridge. Recomputing unpatched
                # component outputs would silently lose cross-pass gradients.
                out.data.copy_(output[:, batch.q_gather])  # [1,N_Q_batch,H,D]
                patched_lse = merged_lse[:, batch.q_gather]  # [1,N_Q_batch,H]
                lse.data.copy_(patched_lse.unsqueeze(-1) if lse.ndim == 4 else patched_lse)  # native LSE shape
                grads = torch.autograd.grad(out, inputs, grad_output[:, batch.q_gather])  # each input shape
                del out, lse, patched_lse
        for total, index, gradient in zip(gradients, indices, grads):
            total.index_add_(1, index, gradient.float())  # full input shape
        # Do not overlap a completed group's gathered inputs/gradients with
        # the next native backward. Loop variables also retain their last tensors.
        del inputs, grads, gradient, total
    # Retire each FP32 accumulation buffer as its returned gradient is cast;
    # retaining the list through all three casts overlaps both full copies.
    dq = gradients.pop(0).to(q.dtype)  # [1,Q,H,D]
    dk = gradients.pop(0).to(k.dtype)  # [1,KV,H_KV,D]
    dv = gradients.pop(0).to(v.dtype)  # [1,KV,H_KV,D]
    return dq, dk, dv  # input shapes


def _decode_batches(
    indices: list[torch.Tensor],  # 2*[N_pass] per partition, then 2*[sum(G_batch+1)]
    layout: torch.Tensor,  # [B,9] on CPU
) -> tuple[ReplayMasklessPass, ...]:
    """Read runtime batch boundaries without specializing the compiled transformer."""
    batches = []
    for partition, qs, qe, ks, ke, offset_start, offset_end, q_max, kv_max in layout.tolist():
        batches.append(
            ReplayMasklessPass(
                name="compiled",
                q_gather=indices[2 * partition][qs:qe],  # [N_Q_batch]
                kv_gather=indices[2 * partition + 1][ks:ke],  # [N_KV_batch]
                q_offsets=indices[-2][offset_start:offset_end],  # [G_batch+1]
                kv_offsets=indices[-1][offset_start:offset_end],  # [G_batch+1]
                q_max_len=q_max,
                kv_max_len=kv_max,
            )
        )
    return tuple(batches)


@torch.library.custom_op("cosmos3::bounded_replay_forward", mutates_args=())
def _opaque_replay_forward(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    indices: list[torch.Tensor],  # Gather indices and packed batch offsets
    layout: torch.Tensor,  # [B,9] on CPU
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,Q,H,D], [1,Q,H]
    return _batched_forward(q, k, v, _decode_batches(indices, layout), compiled=True)  # [1,Q,H,D], [1,Q,H]


@_opaque_replay_forward.register_fake
def _opaque_replay_forward_fake(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    indices: list[torch.Tensor],  # Gather indices and packed batch offsets
    layout: torch.Tensor,  # [B,9] on CPU
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,Q,H,D], [1,Q,H]
    return torch.empty_like(q), q.new_empty(q.shape[:-1], dtype=torch.float32)  # [1,Q,H,D], [1,Q,H]


@torch.library.custom_op("cosmos3::bounded_replay_backward", mutates_args=())
def _opaque_replay_backward(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    output: torch.Tensor,  # [1,Q,H,D]
    merged_lse: torch.Tensor,  # [1,Q,H]
    grad_output: torch.Tensor,  # [1,Q,H,D]
    indices: list[torch.Tensor],  # Gather indices and packed batch offsets
    layout: torch.Tensor,  # [B,9] on CPU
    backward_config: list[int],
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Q: [1,Q,H,D], KV: [1,KV,H_KV,D]
    config = (backward_config[0], backward_config[1]) if backward_config else None
    return _batched_backward(
        q, k, v, output, merged_lse, grad_output, _decode_batches(indices, layout), config, deterministic
    )  # Q: [1,Q,H,D], KV: [1,KV,H_KV,D]


@_opaque_replay_backward.register_fake
def _opaque_replay_backward_fake(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    output: torch.Tensor,  # [1,Q,H,D]
    merged_lse: torch.Tensor,  # [1,Q,H]
    grad_output: torch.Tensor,  # [1,Q,H,D]
    indices: list[torch.Tensor],  # Gather indices and packed batch offsets
    layout: torch.Tensor,  # [B,9] on CPU
    backward_config: list[int],
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Q: [1,Q,H,D], KV: [1,KV,H_KV,D]
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)  # input shapes


class _BatchedReplayAttention(torch.autograd.Function):
    """Gather one KV batch at a time, retaining the native merged-O/LSE gradient contract."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,  # [1,Q,H,D]
        k: torch.Tensor,  # [1,KV,H_KV,D]
        v: torch.Tensor,  # [1,KV,H_KV,D]
        batches: tuple[ReplayMasklessPass, ...],
        indices: tuple[torch.Tensor, ...] = (),  # Gather indices and packed batch offsets
        layout: torch.Tensor | None = None,  # [B,9] on CPU
    ) -> torch.Tensor:  # [1,Q,H,D]
        ctx.backward_config = _blackwell_backward_config(q, k, v)
        ctx.deterministic = torch.are_deterministic_algorithms_enabled()
        ctx.opaque = layout is not None
        if layout is not None:
            output, merged_lse = _opaque_replay_forward(q, k, v, list(indices), layout)  # [1,Q,H,D], [1,Q,H]
            ctx.save_for_backward(q, k, v, output, merged_lse, *indices, layout)
        else:
            # CUDA replay can fuse this merge while the surrounding model stays eager.
            output, merged_lse = _batched_forward(q, k, v, batches, compiled=q.is_cuda)  # [1,Q,H,D], [1,Q,H]
            ctx.save_for_backward(q, k, v, output, merged_lse)
        ctx.batches = batches
        return output  # [1,Q,H,D]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,  # [1,Q,H,D]
    ) -> tuple[torch.Tensor | None, ...]:
        q, k, v, output, merged_lse, *metadata = ctx.saved_tensors  # Q: [1,Q,H,D], KV: [1,KV,H_KV,D]
        if ctx.opaque:
            gradients = _opaque_replay_backward(
                q,
                k,
                v,
                output,
                merged_lse,
                grad_output,
                metadata[:-1],
                metadata[-1],
                list(ctx.backward_config or ()),
                ctx.deterministic,
            )  # Q: [1,Q,H,D], KV: [1,KV,H_KV,D]
        else:
            gradients = _batched_backward(
                q, k, v, output, merged_lse, grad_output, ctx.batches, ctx.backward_config, ctx.deterministic
            )  # Q: [1,Q,H,D], KV: [1,KV,H_KV,D]
        return (*gradients, *((None,) * (len(ctx.needs_input_grad) - 3)))  # input shapes


class _ConcatReplayKV(torch.autograd.Function):
    """Keep a queued clean-cache gradient from retaining the full KV gradient."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        *parts: torch.Tensor,  # each [S_part,H,D], last part is clean memory
    ) -> torch.Tensor:  # [S_total,H,D]
        ctx.lengths = tuple(part.shape[0] for part in parts)
        return torch.cat(parts)  # [S_total,H,D]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        gradient: torch.Tensor,  # [S_total,H,D]
    ) -> tuple[torch.Tensor, ...]:  # each [S_part,H,D]
        parts = gradient.split(ctx.lengths)  # tuple[[S_part,H,D]]
        # Clean replay backpropagates after the noisy pass. Ordinary CatBackward
        # returns views, keeping caption/current-token storage alive at every
        # layer until then. Only the clean-memory slice needs independent storage.
        return (*parts[:-1], parts[-1].clone())  # tuple[[S_part,H,D]]


def cat_replay_kv(parts: list[torch.Tensor]) -> torch.Tensor:  # each [S_part,H,D]; returns [S_total,H,D]
    """Concatenate replay KV with a compact gradient for the final clean-memory part."""
    return _ConcatReplayKV.apply(*parts)  # [S_total,H,D]


def _run_fields(fields_: _StreamFields) -> tuple[_StreamFields, list[int]]:
    """Copy one representative per metadata run to the CPU for grouping."""
    _, representatives = _stream_metadata_groups(fields_, fields_.sample_id.device)  # [S], [R]
    compact = _StreamFields(
        **{field.name: getattr(fields_, field.name)[representatives].cpu() for field in fields(fields_)}  # each [R]
    )
    return compact, representatives.tolist() + [fields_.sample_id.numel()]


def _instant_ids(
    fields_: _StreamFields, sensor_items: Sequence[Sequence[SensorMaskItem]]
) -> tuple[torch.Tensor, torch.Tensor]:  # returns [R], [R]
    """Use upstream camera-anchored midpoint buckets, including cached frame ids."""
    instants = torch.full_like(fields_.frame_id, -1)  # [R]
    has_cross_view = torch.zeros_like(fields_.is_control)  # [R]
    for sample, items in enumerate(sensor_items):
        camera = next((item for item in items if item.caption_access == "camera"), None)
        if camera is None:
            raise ValueError("Maskless replay requires a camera anchor in every sample.")
        view_count = len(
            {view for item in items for view in range(item.view_offset, item.view_offset + item.num_views)}
        )
        has_cross_view |= (fields_.sample_id == sample) & (view_count > 1)  # [R]
        for item in items:
            belongs = (
                (fields_.sample_id == sample)
                & (fields_.view_id >= item.view_offset)
                & (fields_.view_id < item.view_offset + item.num_views)
            )  # [R]
            bucket = torch.floor(
                (fields_.frame_id.double() + 0.5) * item.seconds_per_frame / camera.seconds_per_frame
            ).long()  # [R]
            instants = torch.where(belongs, bucket, instants)  # [R]
    return instants, has_cross_view


def _gather_ranges(ranges: list[tuple[int, int]], device: torch.device) -> torch.Tensor:  # returns [N]
    """Expand CPU token intervals with one device allocation per index tensor."""
    lengths = [stop - start for start, stop in ranges]
    offsets = list(accumulate(lengths, initial=0))
    starts = torch.tensor([start - offset for (start, _), offset in zip(ranges, offsets)], device=device)  # [R]
    repeats = torch.tensor(lengths, device=device)  # [R]
    return torch.repeat_interleave(starts, repeats, output_size=offsets[-1]) + torch.arange(
        offsets[-1], device=device
    )  # [N]


def _build_pass(
    name: str,
    visibility: torch.Tensor,  # [R_Q,R_KV] on CPU
    q_starts: list[int],
    kv_starts: list[int],
    device: torch.device,
) -> ReplayMasklessPass | None:
    """Coalesce equal key sets; omit queries with no keys instead of empty kernels."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for query_run, row in enumerate(visibility):  # row: [R_KV]
        key_runs = tuple(row.nonzero().flatten().tolist())
        if key_runs:
            groups.setdefault(key_runs, []).append(query_run)
    if not groups:
        return None
    q_ranges: list[tuple[int, int]] = []
    kv_ranges: list[tuple[int, int]] = []
    q_lengths: list[int] = []
    kv_lengths: list[int] = []
    for keys, queries in groups.items():
        q_ranges.extend((q_starts[index], q_starts[index + 1]) for index in queries)
        kv_ranges.extend((kv_starts[index], kv_starts[index + 1]) for index in keys)
        q_lengths.append(sum(q_starts[index + 1] - q_starts[index] for index in queries))
        kv_lengths.append(sum(kv_starts[index + 1] - kv_starts[index] for index in keys))
    return ReplayMasklessPass(
        name=name,
        q_gather=_gather_ranges(q_ranges, device),  # [N_Q_pass]
        kv_gather=_gather_ranges(kv_ranges, device),  # [N_KV_pass]
        q_offsets=torch.tensor(list(accumulate(q_lengths, initial=0)), dtype=torch.int32, device=device),  # [G+1]
        kv_offsets=torch.tensor(list(accumulate(kv_lengths, initial=0)), dtype=torch.int32, device=device),  # [G+1]
        q_max_len=max(q_lengths),
        kv_max_len=max(kv_lengths),
    )


def build_replay_maskless_plan(
    metadata: TeacherForcingFlexMetadata,
    sensor_items: Sequence[Sequence[SensorMaskItem]],
) -> ReplayMasklessPlan:
    """Intersect replay roles/causality with the upstream maskless partitions."""
    policy = metadata.teacher_forcing_replay_policy
    if policy.multiview_attention_scope not in ("same_view", "decomposed"):
        raise ValueError("Maskless replay supports same_view or decomposed scope.")
    if policy.decomposed_temporal_window_seconds is not None:
        raise ValueError(
            "Maskless replay requires decomposed_temporal_window_seconds=None for same-instant cross-view attention."
        )
    q, q_starts = _run_fields(_query_stream_fields(metadata))
    kv, kv_starts = _run_fields(_key_stream_fields(metadata))
    # Spatial scope is applied by the partitions below. Keep every existing role,
    # sample, condition, caption and causal rule in the shared replay predicate.
    role_policy = attrs.evolve(policy, multiview_attention_scope="all_views")
    q_ids = torch.arange(q.sample_id.numel())[:, None]  # [R_Q,1]
    kv_ids = torch.arange(kv.sample_id.numel())[None, :]  # [1,R_KV]
    unused = torch.tensor(0)  # []
    allowed = _teacher_forcing_pair_predicate(q, kv, role_policy)(unused, unused, q_ids, kv_ids)  # [R_Q,R_KV]
    caption = kv.token_role_id[None, :] == _ROLE_UND  # [1,R_KV]
    padding = (q.token_role_id[:, None] == _ROLE_PADDING) & (kv.token_role_id[None, :] == _ROLE_PADDING)  # [R_Q,R_KV]
    same_view = q.view_id[:, None] == kv.view_id[None, :]  # [R_Q,R_KV]
    partitions = [("same_view", allowed & ~caption & (same_view | padding))]  # list[[R_Q,R_KV]]
    if policy.multiview_attention_scope == "decomposed":
        q_instants, cross_enabled = _instant_ids(q, sensor_items)  # [R_Q], [R_Q]
        kv_instants, _ = _instant_ids(kv, sensor_items)  # [R_KV], [R_KV]
        # Match the bidirectional maskless model's camera-anchored instants.
        # All spatial tokens at that instant remain visible, subject to replay roles.
        cross_time = q_instants[:, None] == kv_instants[None, :]  # [R_Q,R_KV]
        cross = (
            allowed
            & ~caption
            & ~padding
            & ~q.is_control[:, None]
            & ~kv.is_control[None, :]
            & cross_enabled[:, None]
            & cross_time
        )  # [R_Q,R_KV]
        # Preserve production maskless weighting: a same-view key present in
        # both partitions contributes twice to the merged softmax denominator.
        partitions.append(("cross_instant", cross))
    partitions.append(("caption", allowed & caption))  # [R_Q,R_KV]
    device = metadata.sample_id.device
    passes = tuple(
        plan
        for name, visibility in partitions
        if (plan := _build_pass(name, visibility, q_starts, kv_starts, device)) is not None
    )
    if not passes:
        raise ValueError("Maskless replay has no nonempty attention groups.")
    batches: list[ReplayMasklessPass] = []
    layout: list[tuple[int, ...]] = []
    offset = 0
    if any(partition.kv_gather.numel() > _REPLAY_KV_BATCH_TOKENS for partition in passes):
        for index, partition in enumerate(passes):
            for batch in _batch_pass(partition):
                qs = batch.q_gather.storage_offset() - partition.q_gather.storage_offset()
                ks = batch.kv_gather.storage_offset() - partition.kv_gather.storage_offset()
                end = offset + batch.q_offsets.numel()
                layout.append(
                    (
                        index,
                        qs,
                        qs + batch.q_gather.numel(),
                        ks,
                        ks + batch.kv_gather.numel(),
                        offset,
                        end,
                        batch.q_max_len,
                        batch.kv_max_len,
                    )
                )
                batches.append(batch)
                offset = end
    # Keep the large gather indices as views of the existing partitions. Only
    # small offsets and CPU batch boundaries are packed for the opaque operator.
    compiled_indices = (
        tuple(tensor for partition in passes for tensor in (partition.q_gather, partition.kv_gather))
        + (torch.cat([batch.q_offsets for batch in batches]), torch.cat([batch.kv_offsets for batch in batches]))
        if batches
        else ()
    )  # 2*[N_pass] per partition, then 2*[sum(G_batch+1)]
    return ReplayMasklessPlan(
        passes=passes,
        q_len=metadata.q_len,
        kv_len=metadata.seq_len,
        real_queries=metadata.query.token_role_id != _ROLE_PADDING,  # [Q]
        batches=tuple(batches),
        compiled_indices=compiled_indices,
        batch_layout=torch.tensor(layout, dtype=torch.int64, device="cpu") if batches else None,  # [B,9] on CPU
    )


def replay_maskless_attention(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    plan: ReplayMasklessPlan,
) -> torch.Tensor:  # [1,Q,H,D]
    """Run native varlen passes with upstream bridges preserving cached-K/V gradients."""
    if is_torch_compiling():
        # State the plan invariants without branching on unbacked sequence lengths.
        torch._check(q.shape[1] == plan.q_len)
        torch._check(k.shape[1] == plan.kv_len)
        torch._check(v.shape[1] == plan.kv_len)
    elif q.shape[1] != plan.q_len or k.shape[1] != plan.kv_len or v.shape[1] != plan.kv_len:
        raise ValueError("Maskless replay plan does not match the global Q/KV streams.")
    if (
        is_torch_compiling()
        and plan.batch_layout is not None
        and torch.is_grad_enabled()
        and _blackwell_backward_config(q, k, v) is not None
    ):
        # Batch count varies per pack. Do not unroll that Python loop into each
        # compiled transformer or retain a separate kernel set for every count.
        output = _BatchedReplayAttention.apply(q, k, v, (), plan.compiled_indices, plan.batch_layout)  # [1,Q,H,D]
        return torch.where(plan.real_queries[None, :, None, None], output, 0)  # [1,Q,H,D]
    if plan.batches:
        output = _BatchedReplayAttention.apply(q, k, v, plan.batches)  # [1,Q,H,D]
        return torch.where(plan.real_queries[None, :, None, None], output, 0)  # [1,Q,H,D]
    outputs: list[torch.Tensor] = []  # list[[1,Q,H,D]]
    lses: list[torch.Tensor] = []  # list[[1,Q,H]]
    for partition in plan.passes:
        out, lse = attention(
            q[:, partition.q_gather],  # [1,N_Q_pass,H,D]
            k[:, partition.kv_gather],  # [1,N_KV_pass,H_KV,D]
            v[:, partition.kv_gather],  # [1,N_KV_pass,H_KV,D]
            cumulative_seqlen_Q=partition.q_offsets,
            cumulative_seqlen_KV=partition.kv_offsets,
            max_seqlen_Q=partition.q_max_len,
            max_seqlen_KV=partition.kv_max_len,
            return_lse=True,
        )  # [1,N_Q_pass,H,D], [1,N_Q_pass,H]
        out, lse = MergeAttentionsBridge.apply(
            out,
            lse,
            _scatter_to_packed(partition.q_gather, plan.q_len),
            _gather_from_packed(partition.q_gather),
        )  # [1,Q,H,D], [1,Q,H]
        outputs.append(out)
        lses.append(lse)
    output, _ = (
        (outputs[0], lses[0]) if len(outputs) == 1 else merge_attentions_ac_safe(outputs=outputs, lse_tensors=lses)
    )  # [1,Q,H,D], [1,Q,H]
    return torch.where(plan.real_queries[None, :, None, None], output, 0)  # [1,Q,H,D]
