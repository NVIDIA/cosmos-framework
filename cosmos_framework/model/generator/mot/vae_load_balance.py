# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Equalize each rank's VAE-encode COMPUTE within a small group, before encoding.

Every rank in a Cosmos3 VFM job pulls its own local batch of raw video/image samples from
its own dataloader shard (see ``datasets/joint_dataloader.py`` -- there is no cross-rank
coordination of WHICH samples a rank sees, only which dataset/modality it reads from).
Resolution and clip length vary a lot within one modality, and the VAE encode's cost is
sample-shape-dependent (see ``tokenizers/wan2pt2_vae_4x16x16.py``'s AOT chunking), so two
ranks that each pulled a "full" batch can still end up with very different total VAE work
for the same step -- one rank stalls the others once collectives (FSDP all-gather, gradient
reduce-scatter) force them back into lockstep.

This module fixes that by OFFLOADING compute, not by migrating data: a sample's raw video
tensor may be encoded on a different rank than the one that owns it, but that rank keeps
owning the sample for everything else (caption, image_size, sequence-packing slot,
sound/action/lidar counterparts, loss). ``encode(video) -> latents`` is a pure function of
the video tensor alone, so the only things that ever cross the wire are the raw video going
out and the resulting latent coming back to the exact slot it left from -- nothing else in
the training pipeline (``OmniMoTModel._prepare_training_data`` and everything downstream of
it) needs to change, reorder, or even be aware that the encode happened elsewhere.

Two independent pieces, same split as before:

* :func:`plan_rebalance` -- pure Python, no torch, no distributed state. Takes each group
  member's local sample costs and returns a list of single-sample moves (now read as "who
  ENCODES this sample", not "who OWNS it") that equalizes per-rank compute totals while
  moving as few samples as possible. Fully unit-testable on CPU, unchanged from before.
* :func:`offload_encode` -- the distributed half. Gathers cost/shape metadata across the
  group (cheap), computes the SAME plan independently on every rank (a pure function of
  that gathered data, so no further round-trip is needed to agree on it), then runs a
  two-phase point-to-point exchange: (1) ship out the raw videos being offloaded and
  receive in whatever videos this rank is encoding on a peer's behalf; (2) after running
  the actual encode locally (both for this rank's own retained samples and for anything it
  received to encode for someone else), ship the newly-computed latents back and receive
  back the latents for videos this rank sent away. The result is returned in this rank's
  ORIGINAL local sample order, so the caller's own per-sample bookkeeping for every field
  besides the video/latent is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Move:
    """One sample's ENCODE-SITE reassignment, in terms of its ORIGINAL (owning-rank) position.

    The sample itself never leaves ``from_rank`` -- only where its VAE encode runs moves.

    Attributes:
        local_index: Index into ``from_rank``'s original local sample list.
        from_rank: Local rank (0..``lb_size``-1) within the load-balancing group that OWNS
            the sample (holds its caption, packing slot, etc.) both before and after.
        to_rank: Local rank within the group that will run the VAE encode for this sample
            and send the resulting latent back to ``from_rank``.
    """

    local_index: int
    from_rank: int
    to_rank: int


def plan_rebalance(local_predicted_seconds: dict[int, list[float]], max_iterations: int = 10_000) -> list[Move]:
    """Greedy diff-based rebalance: equalize per-rank ENCODE compute, offloading as few samples as possible.

    Repeatedly finds the currently most-overloaded and most-underloaded rank and offloads
    ONE sample's encode between them -- specifically, among the overloaded rank's
    still-available samples, the one whose cost brings the two totals closest together
    (``argmin |spread - 2*cost|``, where ``spread`` is the current gap). Stops once no
    available sample would reduce the spread, which is also the point at which every sample
    is offloaded at most once: a sample already reassigned to a new encode site is never
    reconsidered for a second reassignment, which is what "moving as few samples as
    possible" means in practice -- an unbounded chain of re-offloads would minimize the
    final spread further but at unbounded communication cost, which is exactly the tradeoff
    this is meant to avoid.

    This is NOT guaranteed to find the perfectly balanced assignment (that is a partition
    problem, NP-hard in general) -- it is a standard greedy heuristic (closely related to
    longest-processing-time list scheduling) that trades a small amount of residual
    imbalance for a plan that is cheap to compute and cheap to execute.

    Args:
        local_predicted_seconds: For each LOCAL rank (0..``lb_size``-1) in the group, the
            predicted VAE-encode seconds of every sample currently OWNED by that rank, in
            that rank's own local order. Ranks with no samples must still appear (with an
            empty list) so the target total accounts for them.
        max_iterations: Safety bound on the number of moves computed, so a pathological
            input cannot loop indefinitely. Real groups have at most a handful of samples
            per rank, so this is never the actual limit in practice.

    Returns:
        A list of :class:`Move`, in the order they were decided. Empty if the group is
        already balanced (or has fewer than two ranks with samples).
    """
    ranks = list(local_predicted_seconds.keys())
    if len(ranks) < 2:
        return []

    totals: dict[int, float] = {rank: sum(costs) for rank, costs in local_predicted_seconds.items()}
    # (local_index, cost) pairs not yet offloaded, per rank.
    available: dict[int, list[tuple[int, float]]] = {
        rank: list(enumerate(costs)) for rank, costs in local_predicted_seconds.items()
    }

    moves: list[Move] = []
    for _ in range(max_iterations):
        over_rank = max(totals, key=lambda r: totals[r])
        under_rank = min(totals, key=lambda r: totals[r])
        spread = totals[over_rank] - totals[under_rank]
        if spread <= 0:
            break

        candidates = available[over_rank]
        if not candidates:
            break
        local_index, cost = min(candidates, key=lambda pair: abs(spread - 2 * pair[1]))
        # Offloading this sample is only worth it if it actually shrinks the gap; a sample
        # bigger than the whole spread would overshoot past the target and land the
        # ranks further apart than leaving it alone.
        if abs(spread - 2 * cost) >= spread:
            break

        available[over_rank] = [pair for pair in candidates if pair[0] != local_index]
        totals[over_rank] -= cost
        totals[under_rank] += cost
        moves.append(Move(local_index=local_index, from_rank=over_rank, to_rank=under_rank))

    return moves


def offload_encode(
    local_tensors: list[torch.Tensor],
    local_predicted_seconds: list[float],
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    group: dist.ProcessGroup,
    group_rank: int,
    group_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """One VAE-load-balancing round for this rank's slice of an ``lb`` group.

    Gathers every group member's ``(predicted cost, video shape, video dtype)`` via one
    small ``all_gather_object`` -- cheap, since it moves scalars and shape tuples, never
    tensor data -- then calls :func:`plan_rebalance`, which every rank in the group computes
    IDENTICALLY from that same gathered data, so no further round-trip is needed for the
    group to agree on the plan. Execution is then a two-phase point-to-point exchange
    (mirroring ``OmniMoTModel._run_classifier_free_guidance``'s CFG exchange, generalized
    from a fixed 2-rank swap to an arbitrary set of offloads across ``lb_size`` ranks):

    1. Ship out the raw videos being offloaded away, and receive in whatever videos this
       rank has been assigned to encode on a peer's behalf. Batched into one round.
    2. Call ``encode_fn`` locally -- both on this rank's own retained samples, and on
       whatever it just received to encode for a peer -- then exchange a second small
       ``all_gather_object`` of the resulting LATENT shapes/dtypes (needed because a video's
       latent shape depends on VAE-specific compression this module has no knowledge of, so
       it cannot be predicted ahead of the real ``encode_fn`` call the way the video's own
       shape could be). Ship the newly-computed latents back, and receive back the latents
       for videos this rank sent away. Also batched into one round.

    Args:
        local_tensors: This rank's raw per-sample video tensors, in this rank's own
            ORIGINAL sample order (any shape/dtype, so long as every rank's video tensors
            share a dtype and device -- receive buffers are allocated from the sender's
            reported shape/dtype and this rank's own ``device``).
        local_predicted_seconds: Parallel to ``local_tensors``: each sample's predicted
            VAE-encode cost, from ``Wan2pt2VAEInterface.predicted_encode_seconds``.
        encode_fn: Runs the real VAE encode on one video tensor, returning its latent.
            Called on this rank for whichever videos this rank ends up encoding --
            its own retained ones, plus any received to encode on a peer's behalf.
        group: The process group backing this rank's ``lb`` mesh
            (``parallel_dims.lb_mesh.get_group()``).
        group_rank: This rank's LOCAL rank within ``group`` (``parallel_dims.lb_rank``),
            i.e. the same numbering :func:`plan_rebalance`'s ``Move.from_rank`` /
            ``Move.to_rank`` use -- NOT the global rank.
        group_size: Ranks in ``group`` (``parallel_dims.lb_size``).
        device: Device to allocate receive buffers on. Required rather than inferred from
            ``local_tensors[0]``, since a rank can legitimately start a round with zero
            local samples (e.g. its dataloader shard yielded none this step) and would
            then have nothing to infer from; the caller always knows its own device
            (e.g. the model's), so asking for it explicitly is free and removes a hidden
            CUDA-availability assumption from this otherwise device-agnostic function --
            which is also what makes it possible to exercise this function on CPU/gloo in
            a test.

    Returns:
        This rank's latents, ONE PER ENTRY OF ``local_tensors``, in the SAME order --
        i.e. ``result[i]`` is the latent for ``local_tensors[i]``, whether it was computed
        locally or offloaded to a peer and sent back. Every other per-sample field the
        caller is tracking for sample ``i`` needs no corresponding change.
    """
    if len(local_tensors) != len(local_predicted_seconds):
        raise ValueError(
            f"local_tensors ({len(local_tensors)}) and local_predicted_seconds "
            f"({len(local_predicted_seconds)}) must be the same length."
        )

    local_meta = [
        (float(cost), tuple(tensor.shape), tensor.dtype)
        for cost, tensor in zip(local_predicted_seconds, local_tensors, strict=True)
    ]
    # all_gather_object overwrites every entry of the pre-allocated list wholesale, so it
    # needs no seeding beyond its length.
    gathered: list[list[tuple[float, tuple[int, ...], torch.dtype]]] = [[] for _ in range(group_size)]
    dist.all_gather_object(gathered, local_meta, group=group)

    costs_by_rank = {rank: [cost for cost, _, _ in meta] for rank, meta in enumerate(gathered)}
    moves = plan_rebalance(costs_by_rank)
    if not moves:
        return [encode_fn(tensor) for tensor in local_tensors]

    # -- Phase 1: exchange the raw videos being offloaded. -----------------------------
    offloaded_away = {move.local_index for move in moves if move.from_rank == group_rank}
    incoming_video: dict[int, torch.Tensor] = {}  # move_id -> received video, to encode here

    video_ops: list[dist.P2POp] = []
    for move_id, move in enumerate(moves):
        if move.from_rank == group_rank:
            video_ops.append(
                dist.P2POp(
                    op=dist.isend,
                    tensor=local_tensors[move.local_index].contiguous(),
                    group_peer=move.to_rank,
                    group=group,
                )
            )
        if move.to_rank == group_rank:
            _, shape, dtype = gathered[move.from_rank][move.local_index]
            buf = torch.empty(shape, dtype=dtype, device=device)
            incoming_video[move_id] = buf
            video_ops.append(dist.P2POp(op=dist.irecv, tensor=buf, group_peer=move.from_rank, group=group))
    if video_ops:
        for req in dist.batch_isend_irecv(video_ops):
            req.wait()

    # -- Local compute: this rank's own retained samples, plus anything offloaded to it. --
    own_latents = {i: encode_fn(tensor) for i, tensor in enumerate(local_tensors) if i not in offloaded_away}
    computed_for_peer = {move_id: encode_fn(video) for move_id, video in incoming_video.items()}

    # -- Exchange latent shape/dtype for the return trip: VAE-specific compression means --
    # -- this module cannot predict a latent's shape from its video's shape on its own.  --
    my_latent_meta = {move_id: (tuple(t.shape), t.dtype) for move_id, t in computed_for_peer.items()}
    gathered_latent_meta: list[dict[int, tuple[tuple[int, ...], torch.dtype]]] = [{} for _ in range(group_size)]
    dist.all_gather_object(gathered_latent_meta, my_latent_meta, group=group)

    # -- Phase 2: send back the latents computed for peers, receive back our own. -------
    latent_ops: list[dist.P2POp] = []
    incoming_latent: dict[int, torch.Tensor] = {}  # move_id -> our latent, sent back by the encoding rank
    for move_id, move in enumerate(moves):
        if move.to_rank == group_rank:
            latent_ops.append(
                dist.P2POp(
                    op=dist.isend,
                    tensor=computed_for_peer[move_id].contiguous(),
                    group_peer=move.from_rank,
                    group=group,
                )
            )
        if move.from_rank == group_rank:
            shape, dtype = gathered_latent_meta[move.to_rank][move_id]
            buf = torch.empty(shape, dtype=dtype, device=device)
            incoming_latent[move_id] = buf
            latent_ops.append(dist.P2POp(op=dist.irecv, tensor=buf, group_peer=move.to_rank, group=group))
    if latent_ops:
        for req in dist.batch_isend_irecv(latent_ops):
            req.wait()

    # -- Assemble the result in this rank's ORIGINAL local sample order. ---------------
    result: list[torch.Tensor | None] = [None] * len(local_tensors)
    for i, latent in own_latents.items():
        result[i] = latent
    for move_id, move in enumerate(moves):
        if move.from_rank == group_rank:
            result[move.local_index] = incoming_latent[move_id]
    assert all(entry is not None for entry in result), "every local sample must end up with exactly one latent"
    return result  # type: ignore[return-value]
