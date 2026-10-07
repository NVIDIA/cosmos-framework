# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Turning a step's raw sensor items (camera clips, LiDAR range clips) into latents across ranks.

:class:`SensorEncoder` owns the whole "collect -> distribute -> use" path: it splits sensor
items into the individual VAE encode calls they cost, spreads those calls over the ``lb``
group, and regroups the resulting latents back into one per item. Callers get the same
latents as encoding every item locally; only where each encode ran differs.

It distributes in one of two ways, because the ``lb`` group means different things in the two
settings that build one:

* Training (:meth:`SensorEncoder.encode_balanced`): every rank holds DIFFERENT samples, so
  the busiest ranks offload raw camera views to idle ones (``models/mot/vae_load_balance.py``).
* Inference (:meth:`SensorEncoder.encode_replicated`): the group is one inference replica,
  every rank holds the SAME sample, so each camera view and LiDAR item is encoded once, on
  one rank, and the latents are broadcast (``models/mot/replica_partition.py``). Camera and
  LiDAR units share one round-robin plan: camera views in item order, then LiDAR items.

It is deliberately a small object over injected collaborators rather than a mixin on the
model. Its whole dependency surface is the constructor arguments below, which is what lets it
be built and exercised without a real model, a config, or a process group.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch

from cosmos_framework.model.generator.mot.replica_partition import ReplicaPartitioner, WorkUnit
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
    """Move one uint8 vision item to the requested device as fp32 and normalize it to ``[-1,1]``.

    A module function rather than a method because both encode paths need it: the local
    per-item encode on ``OmniMoTModel``, and the unit building here.
    """
    if state.dtype != torch.uint8:
        raise ValueError(f"Per-camera VAE encoding requires uint8 pixels, got {state.dtype}.")
    # Move the uint8 levels first and widen them on the device: converting a host-resident item on
    # the host would run on one CPU thread under torchrun and copy four times the bytes. The
    # uint8 -> fp32 conversion is exact either way.
    normalized_state = state.to(device=fp32_kwargs["device"]).to(dtype=fp32_kwargs["dtype"])  # [...,C,T,H,W]
    normalized_state.div_(127.5).sub_(1.0)  # [...,C,T,H,W]
    return normalized_state


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
    return _concat_views_per_item([unit.item_index for unit in units], latents, raw_state_vision)


def _concat_views_per_item(
    item_indexes: list[int],
    latents: list[torch.Tensor],
    raw_state_vision: list[torch.Tensor],
) -> list[torch.Tensor]:
    """:func:`regroup_vision_latents` over bare item indexes, for units that are not vision units."""
    per_item: list[list[torch.Tensor]] = [[] for _ in raw_state_vision]
    for item_index, latent in zip(item_indexes, latents, strict=True):
        per_item[item_index].append(latent)
    return [
        parts[0] if len(parts) == 1 else torch.cat(parts, dim=state.ndim - 3)
        for parts, state in zip(per_item, raw_state_vision, strict=True)
    ]


class SensorEncoder:
    """Encodes a step's sensor items, spreading the VAE calls across the ``lb`` group when it can."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        parallel_dims: Any,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        fp32_kwargs: dict[str, Any],
        lidar_encode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        partitioner: ReplicaPartitioner | None = None,
    ) -> None:
        """
        Args:
            tokenizer: The vision tokenizer. Read for ``encode_seconds_benchmarked``
                and ``predicted_encode_seconds``; may be ``None`` or a
                tokenizer without them, in which case training balancing is unavailable.
                Replicated inference units are assigned round-robin independently of timing.
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
            lidar_encode_fn: Runs the LiDAR VAE encode on one tokenizer-native range clip.
                Late-bound for the same reason as ``encode_fn``. Needed only to encode LiDAR.
            partitioner: The inference replica's work partitioner, or ``None`` outside a
                multi-rank inference replica. Required by :meth:`encode_replicated`.
        """
        self._tokenizer = tokenizer
        self._parallel_dims = parallel_dims
        self._encode_fn = encode_fn
        self._fp32_kwargs = fp32_kwargs
        self._lidar_encode_fn = lidar_encode_fn
        self._partitioner = partitioner

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
    ) -> list[VisionEncodeUnit]:
        """Split vision items into the individual encode calls they require.

        Mirrors ``OmniMoTModel._encode_vision_item``'s own structure: each item contributes
        one unit per camera view, uint8 pixels normalized to ``[-1,1]``. A single-view item
        contributes one unit whose tensor is a full-extent (no-op) narrow of the original.

        Views are normalized here rather than one-at-a-time during encoding because a unit
        may be shipped to a peer rank to encode, so it has to hold a real tensor rather than
        a promise. That trades :meth:`encode_item`'s peak-memory behaviour (which materializes
        a single normalized view at a time) for the ability to balance multiview work at all;
        it applies only when balancing is on.

        ``num_views_per_vision_item`` is trusted as-is: callers derive it from
        ``OmniMoTModel._validate_and_get_num_views``, which has already checked each entry is
        positive and divides its item's frame count evenly.
        """
        units: list[VisionEncodeUnit] = []
        for item_index, (state, num_views) in enumerate(zip(raw_state_vision, num_views_per_vision_item, strict=True)):
            temporal_dim = state.ndim - 3
            num_frames = int(state.shape[temporal_dim])
            frames_per_view = num_frames // num_views
            for view_idx in range(num_views):
                view_state = state.narrow(temporal_dim, view_idx * frames_per_view, frames_per_view)
                units.append(
                    VisionEncodeUnit(
                        item_index=item_index,
                        tensor=(
                            view_state
                            if torch.is_floating_point(view_state)
                            else normalize_uint8_item(view_state, self._fp32_kwargs)
                        ),
                    )
                )
        return units

    def encode_balanced(
        self,
        raw_state_vision: list[torch.Tensor],
        num_views_per_vision_item: list[int],
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
        units = self._build_units(raw_state_vision, num_views_per_vision_item)
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

    def encode_replicated(
        self,
        raw_state_vision: list[torch.Tensor],
        num_views_per_vision_item: list[int],
        raw_state_lidar: list[torch.Tensor] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Encode every camera view and LiDAR item once across the inference replica.

        Each camera view of each vision item and each LiDAR item is one unit. The partitioner
        plans all of them together, every rank encodes only the units it owns, and the latents
        are broadcast so every rank ends with the full set, as the sampler needs: it packs the
        whole sequence on every rank and CP only shards it inside the network.

        Each unit runs exactly the calls the local path runs for it
        (``OmniMoTModel._encode_vision_item`` per view, ``encode_lidar`` per item), so the
        latents are the owner's bits of the same computation. A camera view is normalized on
        its owner, one view at a time, so no rank materializes views it does not encode.

        Requires a ``partitioner``, and every rank of the replica must call this with the same
        items: it enters collectives unconditionally. Whether to call it has to be decided from
        rank-uniform inputs only, as ``OmniMoTModel._encode_sensor_x0_tokens`` does.

        Returns:
            The vision latents, one per item with views concatenated camera-major along T, and
            the LiDAR latents, one per item (``None`` when ``raw_state_lidar`` is ``None``).
        """
        assert self._partitioner is not None, "encode_replicated requires a replica partitioner"

        view_item_indexes: list[int] = []
        view_states: list[torch.Tensor] = []
        for item_index, (state, num_views) in enumerate(zip(raw_state_vision, num_views_per_vision_item, strict=True)):
            temporal_dim = state.ndim - 3
            frames_per_view = int(state.shape[temporal_dim]) // num_views
            for view_index in range(num_views):
                view_states.append(state.narrow(temporal_dim, view_index * frames_per_view, frames_per_view))
                view_item_indexes.append(item_index)
        lidar_states = list(raw_state_lidar or [])
        runs = [self._view_encoder(state) for state in view_states]
        runs += [self._lidar_encoder(state) for state in lidar_states]
        units = [WorkUnit(run=run) for run in runs]
        num_view_units = len(view_states)

        # Broadcast delivery: every entry is a tensor on every rank.
        latents = [latent for latent in self._partitioner.run(units, label="sensor encode") if latent is not None]
        assert len(latents) == len(units), "every rank must receive every encoded latent"
        vision_latents = _concat_views_per_item(view_item_indexes, latents[:num_view_units], raw_state_vision)
        if raw_state_lidar is None:
            return vision_latents, None
        return vision_latents, latents[num_view_units:]

    def _view_encoder(self, view_state: torch.Tensor) -> Callable[[], torch.Tensor]:
        """The encode call ``OmniMoTModel._encode_vision_item`` makes for one camera view."""

        def encode_view() -> torch.Tensor:
            encode_input = (
                view_state
                if torch.is_floating_point(view_state)
                else normalize_uint8_item(view_state, self._fp32_kwargs)
            )  # [...,C,T_v,H,W]
            return self._encode_prepared(encode_input)  # [...,C_latent,T_latent_v,H_latent,W_latent]

        return encode_view

    def _lidar_encoder(self, lidar_state: torch.Tensor) -> Callable[[], torch.Tensor]:
        """The encode call ``OmniMoTModel._encode_lidar_items`` makes for one LiDAR item."""
        lidar_encode_fn = self._lidar_encode_fn
        assert lidar_encode_fn is not None

        def encode_lidar() -> torch.Tensor:
            return lidar_encode_fn(lidar_state).contiguous().float()  # [1,C_latent,T_latent,H_latent,W_latent]

        return encode_lidar
