# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Turning a step's raw vision items into latents, optionally balanced across ranks.

:class:`VisionEncoder` owns the whole "collect -> balance -> use" path: it splits vision
items into the individual VAE encode calls they cost, optionally redistributes those calls
across the ``lb`` group so every rank does a comparable share of the work (see
``models/mot/vae_load_balance.py``), and regroups the resulting latents back into one per
item. Callers get identical latents either way; only where each encode ran differs.

It is deliberately a small object over injected collaborators rather than a mixin on the
model. Its whole dependency surface is the four constructor arguments below, which is what
lets it be built and exercised without a real model, a config, or a process group.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch

from cosmos_framework.model.generator.mot.vae_load_balance import offload_encode


@dataclasses.dataclass(frozen=True)
class VisionEncodeUnit:
    """One encode call: the tensor to encode, and the item its latent belongs to.

    A plain vision item is a single unit. A multiview item is one unit PER CAMERA VIEW,
    because that is how many encode calls it actually costs. Splitting items into units is
    what lets the load balancer move a single camera view between ranks, and what keeps the
    count of planned work equal to the count of encode calls for every kind of batch --
    including the multi-item-per-sample batches (e.g. image editing) that arrive here
    already flattened one item per entry.
    """

    item_index: int
    tensor: torch.Tensor


def get_vae_pixel_shapes(
    raw_state_vision: Sequence[torch.Tensor | None] | None,
) -> list[tuple[int, int, int]]:
    """Extract pixel-space ``(T,H,W)`` metadata, used for cost prediction and FLOPs accounting."""
    shapes: list[tuple[int, int, int]] = []
    if raw_state_vision is None:
        return shapes
    for vision_item in raw_state_vision:
        if vision_item is None:
            continue
        if vision_item.dim() not in (4, 5):
            raise ValueError(f"VAE inputs must have shape [C,T,H,W] or [B,C,T,H,W], got {tuple(vision_item.shape)}.")
        t_h_w = (
            (int(vision_item.shape[2]), int(vision_item.shape[3]), int(vision_item.shape[4]))
            if vision_item.dim() == 5
            else (int(vision_item.shape[1]), int(vision_item.shape[2]), int(vision_item.shape[3]))
        )
        shapes.append(t_h_w)
    return shapes


def normalize_uint8_item(state: torch.Tensor, fp32_kwargs: dict[str, Any]) -> torch.Tensor:
    """Convert one GPU-resident uint8 vision item to fp32 and normalize to ``[-1,1]``.

    A module function rather than a method because both encode paths need it: the local
    per-item encode on ``OmniMoTModel``, and the unit building here.
    """
    if state.dtype != torch.uint8:
        raise ValueError(f"Per-camera VAE encoding requires uint8 pixels, got {state.dtype}.")
    normalized_state = state.to(**fp32_kwargs)  # [...,C,T,H,W]
    normalized_state.div_(127.5).sub_(1.0)  # [...,C,T,H,W]
    return normalized_state


def validate_multiview_length(state: torch.Tensor, *, num_views: int, frames_per_view: int) -> None:
    """Check a camera-major multiview item packs exactly ``num_views`` clips along its T axis."""
    actual_frames = int(state.shape[state.ndim - 3])
    expected_frames = num_views * frames_per_view
    if actual_frames != expected_frames:
        raise ValueError(
            "Multiview vision length must equal num_views * frames_per_view: "
            f"got T={actual_frames}, num_views={num_views}, frames_per_view={frames_per_view}."
        )


def regroup_vision_latents(
    units: list[VisionEncodeUnit],
    latents: list[torch.Tensor],
    raw_state_vision: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Reassemble per-unit latents into one latent per vision item, in item order.

    A multiview item's view latents are concatenated camera-major along the temporal axis.
    ``encode`` preserves tensor rank, so the item's own ``ndim`` locates that axis in its
    latents too.
    """
    per_item: list[list[torch.Tensor]] = [[] for _ in raw_state_vision]
    for unit, latent in zip(units, latents, strict=True):
        per_item[unit.item_index].append(latent)
    return [
        parts[0] if len(parts) == 1 else torch.cat(parts, dim=state.ndim - 3)
        for parts, state in zip(per_item, raw_state_vision, strict=True)
    ]


class VisionEncoder:
    """Encodes a step's vision items, optionally balancing the compute across the ``lb`` group."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        parallel_dims: Any,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        fp32_kwargs: dict[str, Any],
    ) -> None:
        """
        Args:
            tokenizer: The vision tokenizer. Read for ``encode_seconds_benchmarked`` and
                ``predicted_encode_seconds``; may be ``None`` or a tokenizer with neither,
                in which case balancing is simply unavailable.
            parallel_dims: Supplies the ``lb`` mesh (``lb_enabled``/``lb_mesh``/``lb_rank``/
                ``lb_size``). May be ``None`` when no parallelism is configured.
            encode_fn: Runs the real VAE encode on one prepared tensor. Passed as a callable
                rather than pulled off the tokenizer so it stays LATE-BOUND: ``IterSpeed``
                installs a ``MethodTimer`` over the model's ``encode`` attribute after this
                object may already exist, and capturing a bound method here would make the
                timer invisible and silently zero out the VAE timing metric.
            fp32_kwargs: ``{"device": ..., "dtype": torch.float32}`` used to normalize uint8
                multiview pixels, and to allocate receive buffers for offloaded units. Taken
                from the model rather than from a tensor because a rank can legitimately have
                zero vision items on a step and still has to join the group's collectives.
        """
        self._tokenizer = tokenizer
        self._parallel_dims = parallel_dims
        self._encode_fn = encode_fn
        self._fp32_kwargs = fp32_kwargs

    def balancing_available(self) -> bool:
        """Whether this step may balance VAE encode across the ``lb`` group.

        Every condition here is RANK-UNIFORM by construction: the mesh comes from config, and
        the benchmark table is installed by the ``CompileTokenizer`` callback on a fixed
        iteration. So every rank in the group reaches the same answer, and either all of them
        enter :func:`offload_encode`'s collectives or none do.

        Nothing here may depend on this rank's own batch. Ranks can legitimately draw
        different modalities on the same step (see ``RandomJointDataLoader``, whose ranks
        sample independently), so a data-dependent answer would let one rank skip while its
        peers block in an ``all_gather_object`` -- a hang, not an error.
        """
        return (
            self._parallel_dims is not None
            and self._parallel_dims.lb_enabled
            and getattr(self._tokenizer, "encode_seconds_benchmarked", False)
        )

    def _build_units(
        self,
        raw_state_vision: list[torch.Tensor],
        num_views_per_vision_item: list[int],
        frames_per_vision_item: list[int | None],
    ) -> list[VisionEncodeUnit]:
        """Split vision items into the individual encode calls they require.

        Mirrors ``OmniMoTModel._encode_vision_item``'s own structure: a plain item
        contributes its tensor unchanged, and a multiview item contributes one normalized
        camera view each.

        Views are normalized here rather than one-at-a-time during encoding because a unit
        may be shipped to a peer rank to encode, so it has to hold a real tensor rather than
        a promise. That trades :meth:`encode_item`'s peak-memory behaviour (which materializes
        a single normalized view at a time) for the ability to balance multiview work at all;
        it applies only when balancing is on.
        """
        units: list[VisionEncodeUnit] = []
        for item_index, (state, num_views, frames_per_view) in enumerate(
            zip(raw_state_vision, num_views_per_vision_item, frames_per_vision_item, strict=True)
        ):
            if frames_per_view is None:
                if num_views != 1:
                    raise ValueError("frames_per_view is required when num_views is greater than one.")
                units.append(VisionEncodeUnit(item_index=item_index, tensor=state))
                continue

            temporal_dim = state.ndim - 3
            validate_multiview_length(state, num_views=num_views, frames_per_view=frames_per_view)
            for view_idx in range(num_views):
                view_state = state.narrow(temporal_dim, view_idx * frames_per_view, frames_per_view)
                units.append(
                    VisionEncodeUnit(item_index=item_index, tensor=normalize_uint8_item(view_state, self._fp32_kwargs))
                )
        return units

    def encode_balanced(
        self,
        raw_state_vision: list[torch.Tensor],
        num_views_per_vision_item: list[int],
        frames_per_vision_item: list[int | None],
    ) -> list[torch.Tensor]:
        """Encode every vision item with the work spread across the ``lb`` group.

        The items are split into the encode calls they actually cost (see
        :class:`VisionEncodeUnit`), those units are redistributed so every rank does a
        comparable share of the VAE work, and the returned latents are regrouped back into one
        per item. Produces exactly what encoding every item locally would; only where each
        encode ran differs.

        Callers must check :meth:`balancing_available` first -- this enters collectives
        unconditionally, which is what keeps every rank of the group in step.
        """
        units = self._build_units(raw_state_vision, num_views_per_vision_item, frames_per_vision_item)
        unit_tensors = [unit.tensor for unit in units]
        latents = offload_encode(
            local_tensors=unit_tensors,
            local_predicted_seconds=[
                self._tokenizer.predicted_encode_seconds(t, h, w) for t, h, w in get_vae_pixel_shapes(unit_tensors)
            ],
            encode_fn=self._encode_prepared,
            group=self._parallel_dims.lb_mesh.get_group(),
            group_rank=self._parallel_dims.lb_rank,
            group_size=self._parallel_dims.lb_size,
            # Not taken from a unit tensor: a rank can legitimately have zero units this step
            # (its batch carried no vision) and must still join the group's collectives.
            device=self._fp32_kwargs["device"],
        )
        return regroup_vision_latents(units, latents, raw_state_vision)

    def _encode_prepared(self, state: torch.Tensor) -> torch.Tensor:
        """Encode one tensor that is already in the layout the VAE expects."""
        return self._encode_fn(state).contiguous().float()
