# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Distributed dataloader state checkpoint/resume test.

Runs with torchrun --nproc_per_node=4. Tests the full path for both
shuffle=True and shuffle=False:

  1. Each rank trains N batches with DataPackerDataLoader.
  2. DataLoaderStateCallback collects per-worker (epoch, index) state —
     both epoch and position within the epoch are saved.
  3. Each rank saves its state to rank_{rank}.pkl via pickle
     (same format as DistributedCheckpointer._save_as_pkl).
  4. Rank 0 verifies all 4 pkl files are non-empty, contain both epoch
     and index, and that each rank will resume to a distinct item ID
     (confirming disjoint sharding).
  5. All ranks load their pkl back and call load_state_dict()
     -> sets DP_STATE_WORKER_*_EPOCH/INDEX env vars.
  6. Each rank creates a new DataPackerDataLoader and verifies the first
     item resumes from the correct (epoch, position).

shuffle=True:  per-epoch randperm — expected next id from perm[rank::world_size][saved_index+1]
shuffle=False: sequential        — expected next id == saved_index+1 (within this rank's stride)

Usage:
    torchrun --nproc_per_node=4 --master_port=50025 \
        cosmos_framework/data/vfm/test_dp_state_distributed.py
"""

import os
import pickle
import shutil
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

from cosmos_framework.data.vfm.data_packer import DataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader, JointDataPackerDataLoader
from cosmos_framework.callbacks.dataloader_state import DataLoaderStateCallback, JointDataLoaderStateCallback


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

class SimplePacker(DataPacker):
    def sft_process_sample(self, item):
        return item

    def compute_num_tokens(self, sample):
        return 1

    def sft_collate_fn(self, samples, max_len, ignore_label_id=-100):
        return {"ids": torch.tensor([s["id"] for s in samples])}


class SimpleDataset(torch.utils.data.Dataset):
    """Map-style dataset: items are {'id': i} for i in range(n)."""
    def __init__(self, n=10_000):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return {"id": i}


# ---------------------------------------------------------------------------
# Reusable test helper
# ---------------------------------------------------------------------------

def run_state_test(rank, world_size, shuffle, seed, tmp_dir, n_batches=5, dataset_size=10_000):
    """Run the full train → pkl-save → verify → resume cycle for one shuffle mode."""

    label = f"shuffle={'True' if shuffle else 'False'}"

    class FakeParallelDims:
        @property
        def dp_coord(self):
            return (rank, world_size)

    # ------------------------------------------------------------------
    # Phase 1: train n_batches, collect state via callback
    # ------------------------------------------------------------------
    cb = DataLoaderStateCallback(distributor_type="data_packer")
    loader = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=4,
        shuffle=shuffle,
        seed=seed,
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )

    for i, batch in enumerate(loader):
        cb._update_state_from_batch(batch)
        if i + 1 >= n_batches:
            break

    state = cb.state_dict()
    assert state, f"[rank {rank}][{label}] empty state after {n_batches} batches"
    saved_epoch = state[0]["epoch"]
    saved_index = state[0]["index"]
    print(f"[rank {rank}][{label}] phase1: epoch={saved_epoch}, index={saved_index}", flush=True)

    # Save pkl (one file per rank, matching DistributedCheckpointer._save_as_pkl)
    pkl_path = os.path.join(tmp_dir, f"rank_{rank}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(state, f)

    dist.barrier()

    # ------------------------------------------------------------------
    # Phase 2: rank 0 verifies all pkl files and disjoint sharding
    # ------------------------------------------------------------------
    if rank == 0:
        print(f"\n[rank 0][{label}] Verifying all rank pkl files...", flush=True)
        all_states = {}
        for r in range(world_size):
            path = os.path.join(tmp_dir, f"rank_{r}.pkl")
            assert os.path.exists(path), f"missing {path}"
            with open(path, "rb") as f:
                s = pickle.load(f)
            assert s, f"rank {r}: empty state"
            assert 0 in s, f"rank {r}: worker_id 0 missing"
            assert "epoch" in s[0] and "index" in s[0], \
                f"rank {r}: state missing epoch or index keys — got {s[0].keys()}"
            all_states[r] = s
            print(
                f"  rank_{r}.pkl: epoch={s[0]['epoch']}, index={s[0]['index']}",
                flush=True,
            )

        # Reconstruct ground-truth next id for each rank and verify disjoint
        if shuffle:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(dataset_size, generator=g).tolist()
        else:
            perm = list(range(dataset_size))

        first_ids = []
        for r in range(world_size):
            saved_idx = all_states[r][0]["index"]
            stream_slice = perm[r::world_size]  # num_workers=0 → stream_id=r
            first_ids.append(stream_slice[saved_idx + 1])
            print(f"  rank_{r}: index={saved_idx}, next_id={first_ids[-1]}", flush=True)

        assert len(set(first_ids)) == world_size, \
            f"ranks share next item ids — sharding broken: {first_ids}"
        print(
            f"  All {world_size} ranks will resume to distinct item IDs: {first_ids}  OK",
            flush=True,
        )

    dist.barrier()

    # ------------------------------------------------------------------
    # Phase 3: each rank loads its pkl and resumes
    # ------------------------------------------------------------------
    with open(pkl_path, "rb") as f:
        loaded_state = pickle.load(f)

    cb2 = DataLoaderStateCallback(distributor_type="data_packer")
    cb2.load_state_dict(loaded_state)

    assert os.environ.get("DP_STATE_WORKER_0_EPOCH") == str(saved_epoch), \
        f"[rank {rank}][{label}] env EPOCH mismatch"
    assert os.environ.get("DP_STATE_WORKER_0_INDEX") == str(saved_index), \
        f"[rank {rank}][{label}] env INDEX mismatch"

    loader2 = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=1,
        shuffle=shuffle,
        seed=seed,
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )
    first_batch = next(iter(loader2))
    actual_pos = first_batch["sample_index"][0].item()
    actual_id  = first_batch["ids"][0].item()

    # Ground truth: reconstruct this rank's stream slice
    if shuffle:
        g = torch.Generator().manual_seed(seed + saved_epoch)
        perm = torch.randperm(dataset_size, generator=g).tolist()
    else:
        perm = list(range(dataset_size))
    stream_slice = perm[rank::world_size]
    expected_pos = saved_index + 1
    expected_id  = stream_slice[expected_pos]

    assert actual_pos == expected_pos, \
        f"[rank {rank}][{label}] position mismatch: expected {expected_pos}, got {actual_pos}"
    assert actual_id == expected_id, \
        f"[rank {rank}][{label}] id mismatch: expected {expected_id}, got {actual_id}"

    print(
        f"[rank {rank}][{label}] resume: pos={actual_pos} (expected {expected_pos}), "
        f"id={actual_id} (expected {expected_id})  OK",
        flush=True,
    )

    dist.barrier()

    # Clean up pkl files for next test run
    os.remove(pkl_path)
    dist.barrier()


def run_state_test_multi_worker(
    rank, world_size, shuffle, seed, tmp_dir, n_batches=20, dataset_size=10_000, num_workers=2
):
    """State checkpoint/resume test with num_workers > 1.

    With num_workers workers per rank, DataLoaderStateCallback tracks state
    for each worker_id (0..num_workers-1) independently.  The saved pkl
    contains entries for all worker_ids; on resume each worker reads its own
    env var and fast-forwards to the correct position.

    Verification: after resume, every (worker_id, sample_index) pair seen in
    the first resumed batches must have sample_index >= saved_index_for_that_worker + 1.
    """
    label = f"shuffle={'True' if shuffle else 'False'}, num_workers={num_workers}"

    class FakeParallelDims:
        @property
        def dp_coord(self):
            return (rank, world_size)

    # ------------------------------------------------------------------
    # Phase 1: train n_batches, collect per-worker state
    # ------------------------------------------------------------------
    cb = DataLoaderStateCallback(distributor_type="data_packer")
    loader = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=4,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=2,
        parallel_dims=FakeParallelDims(),
    )

    for i, batch in enumerate(loader):
        cb._update_state_from_batch(batch)
        if i + 1 >= n_batches:
            break

    state = cb.state_dict()
    assert state, f"[rank {rank}][{label}] empty state after {n_batches} batches"
    assert len(state) == num_workers, \
        f"[rank {rank}][{label}] expected {num_workers} worker entries, got {len(state)}"
    for wid in range(num_workers):
        assert wid in state, f"[rank {rank}][{label}] worker_id {wid} missing from state"
        assert "epoch" in state[wid] and "index" in state[wid], \
            f"[rank {rank}][{label}] worker {wid} state missing epoch/index"
    print(
        f"[rank {rank}][{label}] phase1: "
        + ", ".join(f"w{wid}=(epoch={state[wid]['epoch']},idx={state[wid]['index']})"
                    for wid in sorted(state)),
        flush=True,
    )

    # Save pkl
    pkl_path = os.path.join(tmp_dir, f"rank_{rank}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(state, f)

    dist.barrier()

    # ------------------------------------------------------------------
    # Phase 2: rank 0 verifies env vars will be set for all worker_ids
    # ------------------------------------------------------------------
    if rank == 0:
        print(f"\n[rank 0][{label}] Verifying pkl files contain all worker_ids...", flush=True)
        for r in range(world_size):
            path = os.path.join(tmp_dir, f"rank_{r}.pkl")
            with open(path, "rb") as f:
                s = pickle.load(f)
            for wid in range(num_workers):
                assert wid in s, f"rank {r}: worker_id {wid} missing"
            print(f"  rank_{r}: workers {sorted(s.keys())} — OK", flush=True)

    dist.barrier()

    # ------------------------------------------------------------------
    # Phase 3: load pkl, verify env vars set for all workers, resume
    # ------------------------------------------------------------------
    with open(pkl_path, "rb") as f:
        loaded_state = pickle.load(f)

    cb2 = DataLoaderStateCallback(distributor_type="data_packer")
    cb2.load_state_dict(loaded_state)

    for wid in range(num_workers):
        saved_epoch = loaded_state[wid]["epoch"]
        saved_index = loaded_state[wid]["index"]
        assert os.environ.get(f"DP_STATE_WORKER_{wid}_EPOCH") == str(saved_epoch), \
            f"[rank {rank}][{label}] w{wid} env EPOCH mismatch"
        assert os.environ.get(f"DP_STATE_WORKER_{wid}_INDEX") == str(saved_index), \
            f"[rank {rank}][{label}] w{wid} env INDEX mismatch"

    # Resume: iterate until we have seen the first batch from each worker, then
    # verify exact (position, item_id) matches the deterministic permutation.
    # This confirms the ordering is identical to what would have been produced
    # without a checkpoint, not merely that positions are monotonically increasing.
    loader2 = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=1,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=2,
        parallel_dims=FakeParallelDims(),
    )

    # Collect the first batch produced by each worker after resume.
    first_per_worker: dict = {}
    for batch in loader2:
        wid = int(batch["sample_worker_id"][0].item())
        if wid not in first_per_worker:
            first_per_worker[wid] = (
                int(batch["sample_index"][0].item()),
                int(batch["ids"][0].item()),
            )
        if len(first_per_worker) == num_workers:
            break

    # Reconstruct the ground-truth permutation for this epoch.
    saved_epoch0 = loaded_state[0]["epoch"]  # all workers share the same epoch
    if shuffle:
        g = torch.Generator().manual_seed(seed + saved_epoch0)
        perm = torch.randperm(dataset_size, generator=g).tolist()
    else:
        perm = list(range(dataset_size))

    for wid in range(num_workers):
        saved_index = loaded_state[wid]["index"]
        # stream_id for this worker on this rank:
        #   stream_id = rank * num_workers + wid
        stream_id = rank * num_workers + wid
        stream_slice = perm[stream_id::(world_size * num_workers)]
        expected_pos = saved_index + 1
        expected_id  = stream_slice[expected_pos]

        actual_pos, actual_id = first_per_worker[wid]
        assert actual_pos == expected_pos, \
            f"[rank {rank}][{label}] w{wid} pos mismatch: expected {expected_pos}, got {actual_pos}"
        assert actual_id == expected_id, \
            f"[rank {rank}][{label}] w{wid} id mismatch: expected {expected_id}, got {actual_id}"
        print(
            f"[rank {rank}][{label}] w{wid}: resume pos={actual_pos} (expected {expected_pos}), "
            f"id={actual_id} (expected {expected_id})  OK",
            flush=True,
        )

    dist.barrier()

    os.remove(pkl_path)
    dist.barrier()


# ---------------------------------------------------------------------------
# JointDataPackerDataLoader tests
# ---------------------------------------------------------------------------

def run_joint_selection_test(rank, world_size, seed=42, n_batches=20, dataset_size=10_000):
    """Verify JointDataPackerDataLoader produces the expected deterministic selection sequence.

    Reconstructs the expected dataset_name sequence using the same
    np.random.RandomState(seed + global_id) formula and asserts it matches.
    Each rank runs independently (selection is identical across ranks since
    it depends only on seed + global_id, not on rank).
    """

    class FakeParallelDims:
        @property
        def dp_coord(self):
            return (rank, world_size)

    loader_a = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=4,
        name="ds_a",
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )
    loader_b = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=4,
        name="ds_b",
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )
    joint = JointDataPackerDataLoader(
        dataloaders={
            "ds_a": {"dataloader": loader_a, "ratio": 3},
            "ds_b": {"dataloader": loader_b, "ratio": 1},
        },
        seed=seed,
    )

    observed = []
    for i, batch in enumerate(joint):
        assert "dataset_name" in batch, f"[rank {rank}] dataset_name key missing from batch"
        observed.append(batch["dataset_name"])
        if i + 1 >= n_batches:
            break

    # Reconstruct expected sequence
    probs = np.array([3, 1], dtype=float) / 4.0
    expected = []
    for i in range(n_batches):
        rng = np.random.RandomState(seed + i)
        idx = int(rng.choice(2, p=probs))
        expected.append(["ds_a", "ds_b"][idx])

    assert observed == expected, (
        f"[rank {rank}] selection mismatch:\n  observed={observed}\n  expected={expected}"
    )
    print(f"[rank {rank}][TEST 5] deterministic selection OK: {observed}", flush=True)

    dist.barrier()


def run_joint_state_test(rank, world_size, shuffle, seed, tmp_dir, n_batches=10, dataset_size=10_000):
    """Full checkpoint/resume cycle for JointDataPackerDataLoader.

    Phase 1: train n_batches, collect state via JointDataLoaderStateCallback.
    Phase 2: save to pkl, reload, verify global_id.
    Phase 3: create fresh joint loader, call load_state_dict, verify:
      - first batch dataset_name matches expected selection at global_id step
      - first batch sample_index == saved_index + 1 for that dataset
    """
    label = f"shuffle={'True' if shuffle else 'False'}"

    class FakeParallelDims:
        @property
        def dp_coord(self):
            return (rank, world_size)

    # ------------------------------------------------------------------
    # Phase 1: train n_batches, collect state
    # ------------------------------------------------------------------
    loader_a = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=4,
        shuffle=shuffle,
        seed=seed,
        name="ds_a",
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )
    loader_b = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=4,
        shuffle=shuffle,
        seed=seed + 1,
        name="ds_b",
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )
    joint = JointDataPackerDataLoader(
        dataloaders={
            "ds_a": {"dataloader": loader_a, "ratio": 3},
            "ds_b": {"dataloader": loader_b, "ratio": 1},
        },
        seed=seed,
    )
    cb = JointDataLoaderStateCallback(outer_loader=joint, distributor_type="data_packer")

    for i, batch in enumerate(joint):
        cb._update_state_from_batch(batch)
        if i + 1 >= n_batches:
            break

    saved_state = cb.state_dict()
    saved_global_id = saved_state["global_id"]
    assert saved_global_id == n_batches, \
        f"[rank {rank}][{label}] expected global_id={n_batches}, got {saved_global_id}"

    print(
        f"[rank {rank}][{label}] phase1: global_id={saved_global_id}, "
        + ", ".join(
            f"{name}=w0(epoch={saved_state[name][0]['epoch']},idx={saved_state[name][0]['index']})"
            for name in ("ds_a", "ds_b")
            if name in saved_state and saved_state[name]
        ),
        flush=True,
    )

    pkl_path = os.path.join(tmp_dir, f"joint_rank_{rank}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(saved_state, f)

    dist.barrier()

    # ------------------------------------------------------------------
    # Phase 2: reload, verify global_id
    # ------------------------------------------------------------------
    with open(pkl_path, "rb") as f:
        loaded_state = pickle.load(f)

    assert loaded_state["global_id"] == saved_global_id, \
        f"[rank {rank}][{label}] global_id mismatch after reload"

    # ------------------------------------------------------------------
    # Phase 3: resume, verify first batch
    # ------------------------------------------------------------------
    loader2_a = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=1,
        shuffle=shuffle,
        seed=seed,
        name="ds_a",
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )
    loader2_b = DataPackerDataLoader(
        data_source=SimpleDataset(dataset_size),
        data_packer=SimplePacker(),
        max_tokens=50,
        max_batch_size=1,
        shuffle=shuffle,
        seed=seed + 1,
        name="ds_b",
        num_workers=0,
        parallel_dims=FakeParallelDims(),
    )
    joint2 = JointDataPackerDataLoader(
        dataloaders={
            "ds_a": {"dataloader": loader2_a, "ratio": 3},
            "ds_b": {"dataloader": loader2_b, "ratio": 1},
        },
        seed=seed,
    )
    cb2 = JointDataLoaderStateCallback(outer_loader=joint2, distributor_type="data_packer")
    cb2.load_state_dict(loaded_state)

    # Determine which dataset the first resumed batch should come from
    probs = np.array([3, 1], dtype=float) / 4.0
    rng = np.random.RandomState(seed + saved_global_id)
    expected_first_ds = ["ds_a", "ds_b"][int(rng.choice(2, p=probs))]

    first_batch = next(iter(joint2))
    actual_ds = first_batch["dataset_name"]
    assert actual_ds == expected_first_ds, \
        f"[rank {rank}][{label}] first dataset mismatch: expected={expected_first_ds}, got={actual_ds}"

    # Verify sample position within the selected dataset matches saved_index + 1
    ds_inner_state = loaded_state.get(actual_ds, {})
    if ds_inner_state:
        saved_index = ds_inner_state[0]["index"]
        saved_epoch  = ds_inner_state[0]["epoch"]
        actual_pos   = int(first_batch["sample_index"][0].item())
        expected_pos = saved_index + 1

        # Also verify exact item id matches the deterministic permutation
        seed_for_ds = seed if actual_ds == "ds_a" else seed + 1
        if shuffle:
            g = torch.Generator().manual_seed(seed_for_ds + saved_epoch)
            perm = torch.randperm(dataset_size, generator=g).tolist()
        else:
            perm = list(range(dataset_size))
        stream_slice = perm[rank::world_size]
        expected_id = stream_slice[expected_pos]
        actual_id   = int(first_batch["ids"][0].item())

        assert actual_pos == expected_pos, \
            f"[rank {rank}][{label}][{actual_ds}] pos mismatch: expected {expected_pos}, got {actual_pos}"
        assert actual_id == expected_id, \
            f"[rank {rank}][{label}][{actual_ds}] id mismatch: expected {expected_id}, got {actual_id}"

    print(
        f"[rank {rank}][{label}] resume OK: global_id={saved_global_id}, "
        f"first_ds={actual_ds} (expected {expected_first_ds}), "
        f"pos={first_batch['sample_index'][0].item()}, id={first_batch['ids'][0].item()}",
        flush=True,
    )

    dist.barrier()
    os.remove(pkl_path)
    dist.barrier()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # All ranks need to agree on the scratch dir, so pick a deterministic
    # location under the system tempdir rather than mkdtemp() (which would
    # return a different path on each rank).
    tmp_dir = os.path.join(tempfile.gettempdir(), "cosmos_dp_state_test_tmp")
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"[rank 0] Using tmp_dir: {tmp_dir}", flush=True)
    dist.barrier()

    # Test 1: shuffle=True, num_workers=0
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[rank 0] TEST 1: shuffle=True, num_workers=0", flush=True)
        print("=" * 60, flush=True)
    run_state_test(rank, world_size, shuffle=True, seed=99, tmp_dir=tmp_dir)

    # Test 2: shuffle=False, num_workers=0
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[rank 0] TEST 2: shuffle=False, num_workers=0", flush=True)
        print("=" * 60, flush=True)
    run_state_test(rank, world_size, shuffle=False, seed=0, tmp_dir=tmp_dir)

    # Test 3: shuffle=True, num_workers=2
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[rank 0] TEST 3: shuffle=True, num_workers=2", flush=True)
        print("=" * 60, flush=True)
    run_state_test_multi_worker(rank, world_size, shuffle=True, seed=77, tmp_dir=tmp_dir, num_workers=2)

    # Test 4: shuffle=False, num_workers=2
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[rank 0] TEST 4: shuffle=False, num_workers=2", flush=True)
        print("=" * 60, flush=True)
    run_state_test_multi_worker(rank, world_size, shuffle=False, seed=0, tmp_dir=tmp_dir, num_workers=2)

    # Test 5: JointDataPackerDataLoader — deterministic selection
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[rank 0] TEST 5: JointDataPackerDataLoader deterministic selection", flush=True)
        print("=" * 60, flush=True)
    run_joint_selection_test(rank, world_size, seed=42)

    # Test 6a: JointDataPackerDataLoader — stateful resume, shuffle=True
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[rank 0] TEST 6a: JointDataPackerDataLoader state resume, shuffle=True", flush=True)
        print("=" * 60, flush=True)
    run_joint_state_test(rank, world_size, shuffle=True, seed=99, tmp_dir=tmp_dir)

    # Test 6b: JointDataPackerDataLoader — stateful resume, shuffle=False
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("[rank 0] TEST 6b: JointDataPackerDataLoader state resume, shuffle=False", flush=True)
        print("=" * 60, flush=True)
    run_joint_state_test(rank, world_size, shuffle=False, seed=0, tmp_dir=tmp_dir)

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("\n=== ALL DISTRIBUTED STATE TESTS PASSED ===", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
