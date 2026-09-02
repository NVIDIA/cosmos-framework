# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for VAE load balancing: the pure-Python planner, the distributed offload itself,
and the real end-to-end model path on GPU.

``TestPlanRebalance`` exercises :func:`plan_rebalance` only -- no torch, no process group,
just the greedy-balancing arithmetic. ``TestOffloadEncode`` exercises
:func:`offload_encode`'s full two-phase point-to-point protocol against a real (CPU/gloo)
process group, using the same ``mp.spawn`` + file-init pattern as
``utils/model_loader_test.py``'s distributed tests -- no GPU required, gloo runs entirely
on CPU. Both are marked ``CPU`` and run in the normal (non-GPU) CI lane.

``test_balanced_vae_encode_on_a_real_4gpu_node`` is marked ``GPU`` and CANNOT run without
one: it needs a real 4-GPU node under ``torchrun`` (see ``_GPU_ONLY_REASON`` and the command
below), because it drives ``VisionEncoder.encode_items`` through the actual Wan2.2
VAE encode. There is deliberately no CPU/gloo fallback for it; anywhere else it skips.

    torchrun --nproc_per_node=4 --master_port=12345 -m pytest \\
        cosmos_framework/model/generator/mot/vae_load_balance_test.py --GPU -s -v
"""

from __future__ import annotations

import os
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_framework.model.generator.mot.vae_load_balance import Move, offload_encode, plan_rebalance


def _totals_after(local_predicted_seconds: dict[int, list[float]], moves: list[Move]) -> dict[int, float]:
    """Replay ``moves`` against the starting costs and return each rank's resulting total."""
    totals = {rank: sum(costs) for rank, costs in local_predicted_seconds.items()}
    for move in moves:
        cost = local_predicted_seconds[move.from_rank][move.local_index]
        totals[move.from_rank] -= cost
        totals[move.to_rank] += cost
    return totals


@pytest.mark.L0
@pytest.mark.CPU
class TestPlanRebalance:
    def test_already_balanced_produces_no_moves(self) -> None:
        assert plan_rebalance({0: [1.0, 1.0], 1: [1.0, 1.0]}) == []

    def test_single_rank_produces_no_moves(self) -> None:
        """Nothing to balance against."""
        assert plan_rebalance({0: [5.0, 3.0, 1.0]}) == []

    def test_empty_input_produces_no_moves(self) -> None:
        assert plan_rebalance({}) == []

    def test_all_ranks_with_empty_sample_lists_produce_no_moves(self) -> None:
        assert plan_rebalance({0: [], 1: [], 2: []}) == []

    def test_two_ranks_one_overloaded_moves_the_closest_fitting_sample(self) -> None:
        """rank 0 has 10, rank 1 has 0. Moving the 4-cost sample lands at 6/4 (spread 2),
        better than the 3-cost (7/3, spread 4) or the 5-cost -- picking 4 is the closest
        fit to the halfway point.
        """
        local = {0: [3.0, 4.0, 3.0], 1: []}
        moves = plan_rebalance(local)

        assert len(moves) == 1
        assert moves[0].from_rank == 0
        assert moves[0].to_rank == 1
        assert local[0][moves[0].local_index] == pytest.approx(4.0)

    def test_a_sample_never_moves_twice(self) -> None:
        """Each Move.local_index refers to the ORIGINAL list, and no local_index from the
        same origin rank appears twice -- a sample that moved away is never reconsidered.
        """
        local = {0: [10.0, 1.0, 1.0, 1.0], 1: [0.5], 2: [0.5]}
        moves = plan_rebalance(local)

        moved_from_0 = [m.local_index for m in moves if m.from_rank == 0]
        assert len(moved_from_0) == len(set(moved_from_0))

    def test_totals_move_strictly_closer_to_balanced_or_stop(self) -> None:
        """Every recorded move must reduce (never grow) the spread between the two ranks
        it touches -- otherwise the planner would be making things worse.
        """
        local = {0: [8.0, 5.0, 2.0], 1: [1.0], 2: [0.0]}
        totals = {rank: sum(costs) for rank, costs in local.items()}
        moves = plan_rebalance(local)

        for move in moves:
            before = abs(totals[move.from_rank] - totals[move.to_rank])
            cost = local[move.from_rank][move.local_index]
            totals[move.from_rank] -= cost
            totals[move.to_rank] += cost
            after = abs(totals[move.from_rank] - totals[move.to_rank])
            assert after < before

    def test_result_is_at_least_as_balanced_as_the_start(self) -> None:
        """The final max-min spread across ranks must not exceed the starting spread."""
        local = {0: [12.0, 3.0], 1: [1.0, 1.0, 1.0], 2: [0.5]}
        starting_totals = {rank: sum(costs) for rank, costs in local.items()}
        starting_spread = max(starting_totals.values()) - min(starting_totals.values())

        moves = plan_rebalance(local)
        final_totals = _totals_after(local, moves)
        final_spread = max(final_totals.values()) - min(final_totals.values())

        assert final_spread <= starting_spread

    def test_three_ranks_converge_to_the_best_achievable_split(self) -> None:
        local = {0: [1.0] * 20, 1: [], 2: []}
        moves = plan_rebalance(local)
        totals = _totals_after(local, moves)

        # 20 unit-cost samples over 3 ranks cannot split more evenly than 7/7/6 -- every
        # total is an integer, so the max-min spread cannot go below 1.
        assert max(totals.values()) - min(totals.values()) <= 1.0

    def test_no_move_helps_when_every_sample_would_overshoot(self) -> None:
        """A single huge sample on the light rank and nothing small enough to trade away
        on the heavy rank: moving it would flip who is heavier by more than doing nothing,
        so the planner correctly declines to move anything.
        """
        local = {0: [1.0, 1.0], 1: [1.9]}
        assert plan_rebalance(local) == []

    def test_ranks_are_not_required_to_be_contiguous_or_zero_based(self) -> None:
        """plan_rebalance keys on whatever rank labels it is given -- callers are expected
        to pass local (0..lb_size-1) ranks, but the function itself doesn't assume it.
        """
        local = {5: [4.0, 4.0], 9: [0.0]}
        moves = plan_rebalance(local)

        assert moves
        assert {move.from_rank for move in moves} <= {5, 9}
        assert {move.to_rank for move in moves} <= {5, 9}

    def test_max_iterations_bounds_the_number_of_moves(self) -> None:
        local = {0: [1.0] * 100, 1: []}
        moves = plan_rebalance(local, max_iterations=3)
        assert len(moves) <= 3


# ---------------------------------------------------------------------------
# offload_encode: real distributed (gloo/CPU) exercise of the two-phase protocol
# ---------------------------------------------------------------------------


def _offload_worker(rank: int, world_size: int, init_file: str, shapes_by_rank: dict, result_queue) -> None:
    """Spawn target: build this rank's local videos, run offload_encode, report the result.

    ``encode_fn`` here is a fake VAE (deterministic, cheap, and checkable) -- the point of
    this test is the distributed exchange protocol, not any real tokenizer. It records
    every marker value it actually encoded, so the driver can verify not just that content
    round-trips correctly but WHERE the compute actually ran.
    """
    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
        group = dist.group.WORLD

        shapes = shapes_by_rank.get(rank, [])
        local_tensors = [
            torch.full(shape, fill_value=rank * 10 + i, dtype=torch.uint8) for i, shape in enumerate(shapes)
        ]
        local_costs = [float(t.numel()) for t in local_tensors]

        encoded_here: list[int] = []

        def encode_fn(video: torch.Tensor) -> torch.Tensor:
            encoded_here.append(int(video.flatten()[0].item()))
            # Deliberately a different shape/dtype than the video, like a real VAE latent,
            # to exercise the latent-shape metadata round-trip rather than assume identity.
            return video.flatten()[:1].float() / 2.0

        latents = offload_encode(
            local_tensors=local_tensors,
            local_predicted_seconds=local_costs,
            encode_fn=encode_fn,
            group=group,
            group_rank=rank,
            group_size=world_size,
            device=torch.device("cpu"),
        )
        markers = [rank * 10 + i for i in range(len(shapes))]
        report = [(marker, float(latent.flatten()[0].item())) for marker, latent in zip(markers, latents, strict=True)]
        result_queue.put((rank, "ok", report, encoded_here))
    except Exception:
        result_queue.put((rank, "error", traceback.format_exc(), []))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_offload(shapes_by_rank: dict[int, list[tuple[int, ...]]], world_size: int, timeout: float = 60.0) -> dict:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [
            ctx.Process(target=_offload_worker, args=(rank, world_size, init_file, shapes_by_rank, result_queue))
            for rank in range(world_size)
        ]
        for p in procs:
            p.start()
        results = {}
        for _ in range(world_size):
            rank, status, report, encoded_here = result_queue.get(timeout=timeout)
            results[rank] = (status, report, encoded_here)
        for p in procs:
            p.join(timeout=30)
    for rank, (status, report, _) in results.items():
        assert status == "ok", f"rank {rank} failed:\n{report}"
    return results


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
class TestOffloadEncode:
    def test_offloaded_samples_return_correct_latents_to_their_owner(self) -> None:
        """Rank 0 owns 3 samples heavy enough to force offloading; ranks 1/2 own none.
        Every latent that comes back to rank 0 must be correct and in its original slot,
        regardless of which peer actually computed it.
        """
        shapes = {0: [(3, 4, 8, 8), (3, 4, 8, 8), (3, 2, 8, 8)], 1: [], 2: []}
        results = _run_offload(shapes, world_size=3)

        markers = [0, 1, 2]
        for marker, latent_value in results[0][1]:
            assert latent_value == pytest.approx(marker / 2.0), f"sample {marker} got wrong latent {latent_value}"
        assert [m for m, _ in results[0][1]] == markers

    def test_owner_skips_encoding_samples_it_offloaded(self) -> None:
        """The two costliest samples (cost 768 each) get offloaded; the cheapest (384)
        doesn't clear plan_rebalance's overshoot guard and stays local -- see
        TestPlanRebalance for that arithmetic in isolation. Rank 0 must encode ONLY the
        one that stayed.
        """
        shapes = {0: [(3, 4, 8, 8), (3, 4, 8, 8), (3, 2, 8, 8)], 1: [], 2: []}
        results = _run_offload(shapes, world_size=3)

        assert results[0][2] == [2], f"rank 0 should only self-encode its leftover sample, got {results[0][2]}"

    def test_offloaded_compute_lands_on_peers_exactly_once_each(self) -> None:
        shapes = {0: [(3, 4, 8, 8), (3, 4, 8, 8), (3, 2, 8, 8)], 1: [], 2: []}
        results = _run_offload(shapes, world_size=3)

        encoded_elsewhere = sorted(results[1][2] + results[2][2])
        assert encoded_elsewhere == [0, 1], f"expected markers [0, 1] encoded across peers, got {encoded_elsewhere}"

    def test_ranks_with_no_samples_report_nothing_and_own_nothing(self) -> None:
        shapes = {0: [(3, 4, 8, 8), (3, 4, 8, 8), (3, 2, 8, 8)], 1: [], 2: []}
        results = _run_offload(shapes, world_size=3)

        assert results[1][1] == []
        assert results[2][1] == []

    def test_already_balanced_batch_encodes_everything_locally(self) -> None:
        """No moves means offload_encode should short-circuit to plain local encoding --
        every rank encodes exactly its own samples, nothing crosses the wire."""
        shapes = {0: [(3, 2, 4, 4)], 1: [(3, 2, 4, 4)]}
        results = _run_offload(shapes, world_size=2)

        assert results[0][2] == [0]
        assert results[1][2] == [10]
        assert results[0][1] == [(0, 0.0)]
        assert results[1][1] == [(10, 5.0)]

    def test_a_rank_can_both_send_and_receive_in_the_same_round(self) -> None:
        """4 ranks, costs chosen so rank 1 both offloads one of its own samples AND
        receives one from a peer within the same plan -- exercises the batched P2P ops
        not deadlocking or cross-wiring when a rank is on both sides of the exchange.
        """
        # Costs (via tensor numel): rank0=[154,90,59]->approx via shape, keep simple ints.
        shapes_by_rank = {
            0: [(1, 1, 14, 11), (1, 1, 9, 10), (1, 1, 6, 10)],  # numel: 154, 90, 60
            1: [(1, 1, 9, 10), (1, 1, 16, 10), (1, 1, 7, 10)],  # numel: 90, 160, 70
            2: [(1, 1, 8, 10), (1, 1, 18, 10), (1, 1, 5, 10)],  # numel: 80, 180, 50
            3: [(1, 1, 7, 9)],  # numel: 63
        }
        results = _run_offload(shapes_by_rank, world_size=4)

        # Every original sample's marker must reappear exactly once across all reports,
        # with the correct latent value, regardless of who encoded it.
        all_markers_before = sorted(
            rank * 10 + i for rank, shapes in shapes_by_rank.items() for i in range(len(shapes))
        )
        all_reported = sorted(marker for rank in results for marker, _ in results[rank][1])
        assert all_reported == all_markers_before

        for rank in results:
            for marker, latent_value in results[rank][1]:
                assert latent_value == pytest.approx(marker / 2.0)

        # Every marker was encoded by exactly one rank in total.
        all_encoded = sorted(marker for rank in results for marker in results[rank][2])
        assert all_encoded == all_markers_before


# ---------------------------------------------------------------------------
# GPU-ONLY: real single-node, 4-GPU exercise of VisionEncoder.encode_items
#
# Everything above runs the offload_encode/plan_rebalance primitives directly against a
# fake CPU encode_fn under gloo -- it never exercises VisionEncoder.encode_items
# itself, and in particular never catches whether the video handed to the REAL VAE encode
# is normalized the way self.encode expects (this is exactly the bug the GPU test was
# written to catch: the balanced path was feeding offload_encode raw uint8 [0,255]
# pixels instead of the [-1,1] float [B,C,T,H,W] tensors get_data_and_condition's own
# normalization produces).
#
# The test below therefore has NO CPU/gloo fallback -- it needs real GPUs and a real
# torchrun launch, and skips everywhere else (see _GPU_ONLY_REASON).
# ---------------------------------------------------------------------------

EXPERIMENT = "t2w_mot_dryrun_exp200_001_qwen3_vl_0p6b_480res_qwen3_captions_mrope_v2"

_GPU_ONLY_REASON = (
    "GPU-only test: requires a real 4-GPU node launched with "
    "`torchrun --nproc_per_node=4 --master_port=12345 -m pytest "
    "cosmos_framework/model/generator/mot/vae_load_balance_test.py --GPU -s -v`."
)

# Evaluated at import time, which is safe: torchrun sets WORLD_SIZE before pytest starts.
requires_four_gpus_under_torchrun = pytest.mark.skipif(
    not (dist.is_available() and torch.cuda.is_available() and os.environ.get("WORLD_SIZE") == "4"),
    reason=_GPU_ONLY_REASON,
)


def _owner_marker(rank: int, local_index: int) -> int:
    """Encode (owning rank, local sample index) as one uint8 pixel value, decodable after normalization."""
    return rank * 10 + local_index


def _decode_marker(normalized_pixel: float) -> int:
    """Invert ``_normalize_video_databatch_inplace``'s ``uint8/127.5 - 1.0`` exactly (uint8 has no rounding loss)."""
    return round((normalized_pixel + 1.0) * 127.5)


def _imbalanced_batch(rank: int, height: int, width: int) -> dict:
    """Rank 0: 3 video clips (81 frames each). Ranks 1-3: 1 still image each.

    Deliberately imbalanced: without load balancing, rank 0 would run several VAE encodes
    while ranks 1-3 sit on one apiece -- exactly the skew ``vae_load_balance_group_size=4``
    is meant to spread out.

    Each sample's very first pixel is stamped with ``_owner_marker(rank, i)`` so that
    whichever rank ends up actually running ``self.encode`` on it -- its owner, or a peer
    it was offloaded to -- can be identified downstream from the (post-normalization)
    tensor content alone, without threading any side-channel metadata through
    ``VisionEncoder.encode_balanced``/``offload_encode``.
    """
    # Imported lazily: this pulls in the whole cost_model/sequence_packing stack, which the
    # CPU tests above neither need nor should pay for at collection time.
    from cosmos_framework.data.generator.sequence_packing import SequencePlan

    if rank == 0:
        num_samples, frames, is_image = 3, 81, False
    else:
        num_samples, frames, is_image = 1, 1, True

    media_key = "images" if is_image else "video"
    pixel_shape = (3, height, width) if is_image else (3, frames, height, width)

    def _stamped(local_index: int) -> torch.Tensor:
        item = torch.randint(0, 256, pixel_shape, dtype=torch.uint8)
        item[(0, 0, 0) if is_image else (0, 0, 0, 0)] = _owner_marker(rank, local_index)
        return item.cuda()

    return {
        "ai_caption": ["VAE load-balance GPU test sample."] * num_samples,
        media_key: [[_stamped(i)] for i in range(num_samples)],
        "image_size": [
            torch.tensor([[height, width, height, width]], dtype=torch.float32).cuda() for _ in range(num_samples)
        ],
        "text_token_ids": [[torch.randint(0, 1000, (16,), dtype=torch.long).cuda()] for _ in range(num_samples)],
        "conditioning_fps": [torch.tensor([24.0]).cuda().to(torch.bfloat16) for _ in range(num_samples)],
        "sequence_plan": [
            SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]) for _ in range(num_samples)
        ],
    }


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.serial
@requires_four_gpus_under_torchrun
def test_balanced_vae_encode_on_a_real_4gpu_node() -> None:
    """Drive a real training step through ``VisionEncoder.encode_balanced`` on 4 GPUs.

    If the balanced path's gate, normalization, or the ``offload_encode`` exchange
    itself were wrong, this either crashes (shape/dtype mismatch reaching the real VAE),
    hangs (P2P op mismatch across ranks), or produces a non-finite loss (garbage pixels
    reaching the transformer) -- all of which this test catches, unlike a shape-only
    assertion.

    A passing loss/shape check alone would NOT prove any sample actually crossed ranks --
    if ``offload_encode`` silently no-op'd and every rank just encoded its own samples
    locally (as it would with load balancing disabled), the loss would still be finite and
    every sample would still get encoded exactly once. So each sample's marker (see
    ``_owner_marker``/``_decode_marker``) is traced back out of whatever tensor reaches
    ``self.encode``, on whichever rank that turns out to be. The final assertion requires at
    least one sample's encoding rank to differ from its owning rank -- the actual, positive
    evidence that rebalancing moved compute.
    """
    # See _imbalanced_batch: kept out of the module imports so the CPU tests above stay
    # importable and cheap without the cost_model stack.
    from cosmos_framework.utils.generator.cost_model.benchmark import build_model

    world_size = int(os.environ["WORLD_SIZE"])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model, optimizer, scheduler = build_model(
        EXPERIMENT,
        world_size=world_size,
        extra_overrides=["model.config.parallelism.vae_load_balance_group_size=4"],
    )
    assert model.parallel_dims.lb_enabled, "vae_load_balance_group_size=4 should enable the lb overlay mesh."
    assert model.parallel_dims.lb_size == 4

    # predicted_encode_seconds() (called from VisionEncoder.encode_balanced) reads a benchmarked
    # per-chunk-shape timing table that only exists after compile_encode() has run -- in
    # production this happens once via the compile_tokenizer callback, which this
    # trimmed-down harness (build_model + run_step, no ImaginaireTrainer/callbacks) never
    # runs on its own. "16,9" @ "480" is exactly the (832, 480) bucket _imbalanced_batch
    # uses, so this compiles only the chunk shape this test actually needs.
    model.tokenizer_vision_gen.compile_encode(
        warmup_resolutions=["480"],
        output_dir=os.path.join(tempfile.gettempdir(), "vae_load_balance_gpu_test_aot"),
        aspect_ratio="16,9",
    )

    batch = _imbalanced_batch(rank, height=480, width=832)
    own_markers = {_owner_marker(rank, i) for i in range(len(batch.get("video") or batch.get("images")))}

    encoded_here: list[int] = []
    original_encode = model.encode

    def _counting_encode(state: torch.Tensor) -> torch.Tensor:
        assert torch.is_floating_point(state), f"encode() received non-float input: {state.dtype}"
        assert torch.all((state >= -1.0001) & (state <= 1.0001)), (
            f"encode() received un-normalized input, range [{state.min():.2f}, {state.max():.2f}]"
        )
        encoded_here.append(_decode_marker(float(state[0, 0, 0, 0, 0].item())))
        return original_encode(state)

    model.encode = _counting_encode
    try:
        # Inlined rather than benchmark.run_step(): that helper only returns
        # output_batch (it consumes loss locally for backward), and this test needs
        # the actual loss tensor to check for non-finite values.
        _output_batch, loss = model.training_step(batch, iteration=0)
        loss.backward()
        model.on_after_backward()
        model.on_before_optimizer_step(optimizer, scheduler, iteration=0)
        optimizer.step()
        scheduler.step()
        model.on_before_zero_grad(optimizer, scheduler, iteration=0)
        optimizer.zero_grad(set_to_none=True)
    finally:
        model.encode = original_encode

    assert torch.isfinite(loss).all(), f"Non-finite loss after offloaded VAE encode: {loss}"

    # Gather every rank's (own_markers, encoded_here) so each rank can independently check
    # the whole group's behavior -- not just its own local count, which (as noted in the
    # docstring) can't distinguish "load balancing worked" from "it silently never engaged
    # and every rank just encoded its own samples, as it would with lb disabled".
    per_rank = [None] * world_size
    dist.all_gather_object(per_rank, (rank, sorted(own_markers), sorted(encoded_here)))

    all_owned = sorted(m for _, owned, _ in per_rank for m in owned)
    all_encoded = sorted(m for _, _, encoded in per_rank for m in encoded)
    assert all_owned == [0, 1, 2, 10, 20, 30], f"Unexpected ownership set: {all_owned}"
    assert all_encoded == all_owned, (
        f"Every owned sample must be encoded EXACTLY once across the group (no drops, no "
        f"duplicates): owned={all_owned}, encoded={all_encoded}"
    )

    # The actual load-balancing check: rank 0's 3 long clips vastly outweigh the other three
    # ranks' 1 short image each, so plan_rebalance (see vae_load_balance.py) should move at
    # least one of rank 0's samples to a lighter rank. If every rank's encoded_here equals
    # its own own_markers, nothing moved -- offloading silently no-op'd.
    moved_at_least_one = any(sorted(owned) != sorted(encoded) for _, owned, encoded in per_rank)
    assert moved_at_least_one, (
        f"No sample crossed ranks -- offload_encode did not rebalance the deliberately "
        f"skewed batch (rank 0: 3 long clips vs. ranks 1-3: 1 short image each). Per-rank "
        f"(owned, encoded): {per_rank}"
    )
    if rank == 0:
        moved_from = [m for m in own_markers if m not in encoded_here]
        print(f"VAE load balancing moved rank 0's samples {moved_from} off-rank. Full trace: {per_rank}")

    if dist.is_initialized():
        dist.barrier()
