# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared Ulysses communication and sequence ownership for native VLMs."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class ContextParallelBatchPolicy:
    """Preserve the native position layout and input-sharding stage.

    ``position_axes=None`` denotes [B,T]; an integer denotes [A,B,T].
    Packed positions are prepared by the model's native position routine.
    ``keep_full_inputs`` also retains text-only tokens until decoder entry.
    Otherwise only enabled multimodal batches defer sharding until fusion.
    """

    position_axes: int | None = None
    allow_multimodal: bool = False
    keep_full_inputs: bool = False
    requires_packed: bool = False


@dataclass(frozen=True)
class ContextParallelModelSpec:
    """Explicit native-model bindings for ``parallelize_vlm.apply_cp``.

    Module references are metadata, not additional registered parameters.
    Integrations validate native variants before returning this spec. Merely
    exposing q/k/v projections does not establish CP support. Position
    generation, attention kernels and loss precision stay with the model.
    """

    decoder: nn.Module
    attention_modules: tuple[nn.Module, ...]
    text_config: Any
    fused_ce_enabled: bool
    position_axes: int | None
    keep_full_inputs: bool = False
    requires_packed: bool = False
    require_default_rope: bool = False
    decoder_packing_key: str | None = None
    fusion_module: nn.Module | None = None


def _all_to_all(tensor: torch.Tensor, group: dist.ProcessGroup, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    size = dist.get_world_size(group)
    if tensor.shape[scatter_dim] % size:
        raise ValueError("Ulysses scatter dimension must be divisible by CP size")
    chunks = tensor.chunk(size, dim=scatter_dim)
    shape = chunks[0].shape
    packed = torch.cat([chunk.contiguous().view(-1) for chunk in chunks])
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, group=group)
    return torch.cat([chunk.view(shape) for chunk in received.chunk(size)], dim=gather_dim).contiguous()


class _SequenceAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, group: dist.ProcessGroup, scatter_dim: int, gather_dim: int
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all(tensor, group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        return _all_to_all(gradient, ctx.group, ctx.gather_dim, ctx.scatter_dim), None, None, None


def sequence_all_to_all(
    tensor: torch.Tensor, group: dist.ProcessGroup, *, scatter_dim: int, gather_dim: int
) -> torch.Tensor:
    """Swap sequence shards for head shards; the backward performs the inverse."""
    return _SequenceAllToAll.apply(tensor, group, scatter_dim, gather_dim)


def shard_text_batch(
    data: dict[str, Any], group: dist.ProcessGroup, *, position_axes: int | None = None
) -> dict[str, Any]:
    """Slice replicated text inputs, global positions, and pre-shifted next-token labels.

    The loader must replicate each logical sample across adjacent CP ranks. Labels
    are shifted BEFORE slicing, preserving the target at every shard boundary.
    Every CP shard keeps the full model; FSDP still averages gradients over WORLD.
    """
    size = dist.get_world_size(group)
    if data.get("context_parallel_size") != size:
        raise ValueError("CP requires a loader that replicates each sample within its CP group")
    packed = data.get("cu_seq_lens_q") is not None
    if data.get("true_packing", False):
        raise ValueError("Prepare global packed positions and boundaries before CP slicing")
    if any(
        data.get(key) is not None
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "inputs_embeds",
            "past_key_values",
            "cache_position",
            "audio_values",
            "input_features",
        )
    ):
        raise NotImplementedError("Text CP requires token inputs without media or KV cache")
    if not packed and data.get("position_ids") is not None:
        raise ValueError("Unpacked text CP computes its own global positions")
    if packed and data.get("cu_seq_lens_q") is not data.get("cu_seq_lens_k"):
        raise ValueError("Packed CP requires shared global Q/KV boundaries")
    ids, labels, mask = data["input_ids"], data["labels"], data["attention_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or labels.shape != ids.shape or mask.shape != ids.shape:
        raise ValueError("Text CP requires aligned [1,T] input_ids, labels and attention_mask")
    if ids.shape[1] % size:
        raise ValueError("Global sequence length must be divisible by CP size")
    if bool(((mask != 0) & (mask != 1)).any()) or bool((mask[:, 1:] > mask[:, :-1]).any()):
        raise ValueError("Text CP accepts right padding only")
    if bool((labels[mask == 0] != -100).any()):
        raise ValueError("Padding labels must be ignored")
    local_length = ids.shape[1] // size
    start = dist.get_rank(group) * local_length
    section = slice(start, start + local_length)
    targets = torch.cat([labels[:, 1:], torch.full_like(labels[:, :1], -100)], dim=1)
    positions = data.get("position_ids") if packed else (mask.long().cumsum(-1) - 1).clamp(min=0)
    if not packed and position_axes is not None:
        positions = positions.unsqueeze(0).expand(position_axes, -1, -1)
    expected_shape = (1, ids.shape[1]) if position_axes is None else (position_axes, 1, ids.shape[1])
    if not isinstance(positions, torch.Tensor) or positions.shape != expected_shape:
        raise ValueError("Packed CP requires complete native per-conversation positions")
    output = dict(data)
    output.update(
        input_ids=ids[:, section].contiguous(),
        labels=targets[:, section].contiguous(),
        attention_mask=mask[:, section].contiguous(),
        position_ids=positions[..., section].contiguous(),
    )
    return output


def context_parallel_router_mean(
    local_mean: torch.Tensor, local_tokens: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Scale a shard's router mean so WORLD averaging recovers each full sample mean."""
    sequence_tokens = local_tokens.detach().clone()
    dist.all_reduce(sequence_tokens, group=group)
    return local_mean * (local_tokens * dist.get_world_size(group) / sequence_tokens.clamp_min(1))


def prepare_context_parallel_batch(
    data: dict[str, Any], group: dist.ProcessGroup, *, policy: ContextParallelBatchPolicy
) -> dict[str, Any]:
    """Keep media and full tokens until fusion, but shift/shard labels before CE.

    The caller must supply identical preprocessed media within each CP group.
    This first version does not distribute raw-media decoding or vision encoding.
    """
    if policy.requires_packed and data.get("cu_seq_lens_q") is None:
        raise ValueError("This native CP integration requires true-packed inputs")
    media_keys = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
    has_media = any(data.get(key) is not None for key in media_keys)
    if not policy.keep_full_inputs and (not has_media or not policy.allow_multimodal):
        return shard_text_batch(data, group, position_axes=policy.position_axes)
    if data.get("cache_position") is not None:
        raise NotImplementedError("Multimodal CP requires a complete training sequence without cache")
    # Reuse shape, padding, replication, unsupported-input and boundary-target checks.
    local = shard_text_batch(
        {key: value for key, value in data.items() if key not in media_keys}, group, position_axes=policy.position_axes
    )
    output = dict(data)
    output["labels"] = local["labels"]
    output["context_parallel_token_mask"] = local["attention_mask"]
    return output


def shard_decoder_inputs(
    inputs: dict[str, Any], group: dist.ProcessGroup, *, position_axes: int | None, clone_embeddings: bool = False
) -> dict[str, Any]:
    """Shard after complete media fusion and native global position calculation.

    Visual rows in DeepStack tensors are compressed (text rows are absent), so
    select by global visual-token positions rather than the decoder row slice.
    Empty selections retain their autograd connection to every vision module.
    FSDP WORLD averaging and the CP-scaled fused objective perform gradient
    aggregation; an additional vision-gradient all-reduce would double count.

    ``clone_embeddings`` preserves integrations whose decoder modifies the
    fused embedding buffer in place. Other integrations retain contiguous views.
    Global packed boundaries and all nonsequence arguments pass through intact.
    """
    size = dist.get_world_size(group)
    hidden, positions = inputs.get("inputs_embeds"), inputs.get("position_ids")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("CP requires complete fused embeddings shaped [1,T,H]")
    length = hidden.shape[1]
    expected_shape = (1, length) if position_axes is None else (position_axes, 1, length)
    if length % size or not isinstance(positions, torch.Tensor) or positions.shape != expected_shape:
        raise ValueError("CP requires divisible tokens and complete native positions")
    local_length = length // size
    start = dist.get_rank(group) * local_length
    section = slice(start, start + local_length)
    output = dict(inputs)
    output["inputs_embeds"] = hidden[:, section].clone() if clone_embeddings else hidden[:, section].contiguous()
    output["position_ids"] = positions[..., section].contiguous()
    for key in ("attention_mask", "padding_mask", "visual_pos_masks"):
        value = output.get(key)
        if value is not None:
            if not isinstance(value, torch.Tensor) or value.shape != (1, length):
                raise ValueError(f"CP requires a full [1,T] {key}")
            output[key] = value[:, section].contiguous()
    features = inputs.get("deepstack_visual_embeds")
    if features is not None:
        visual = inputs.get("visual_pos_masks")
        if not isinstance(visual, torch.Tensor):
            raise ValueError("Compressed visual features require the full visual token mask")
        visual_positions = visual[0].nonzero(as_tuple=True)[0]
        selected = (visual_positions >= start) & (visual_positions < start + local_length)
        if any(value.shape[0] != visual_positions.numel() for value in features):
            raise ValueError("DeepStack rows must match all global visual tokens")
        output["deepstack_visual_embeds"] = [value[selected].contiguous() for value in features]
    return output
