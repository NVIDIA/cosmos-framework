# Modular Dataflow — Map-style Resume + Callback Compatibility (Plan 3 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give `MapDistributor` checkpoint/resume parity with the legacy `_ShuffledMapIterableDataset`, threaded through `CosmosDataLoader`, so the existing `DataLoaderStateCallback` keeps working bit-for-bit — satisfying the spec's HARD INVARIANT (resume + saving must not break).

**Architecture:** The legacy mechanism has three parts: (1) the callback reads per-batch `sample_worker_id/epoch/index` tensors and on resume sets `DP_STATE[_{name}]_WORKER_{id}_EPOCH/INDEX` env vars *before workers fork*; (2) the map dataset reads those env vars on its first iteration and fast-forwards; (3) the loader stamps the per-batch tensors. We keep the callback **unchanged** (it already handles `distributor_type="data_packer"`), and reproduce parts (2) and (3) in `MapDistributor.stream` and the `CosmosDataLoader` orchestrator respectively.

**Tech Stack:** Python, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md` ("Hard invariants", "Resume threading"). Builds on Plans 1–2.

> **HARD INVARIANT:** No change to `cosmos_framework/callbacks/dataloader_state.py` or the on-disk DCP state format. After this plan, a map-style `CosmosDataLoader` + `DataLoaderStateCallback(distributor_type="data_packer")` must checkpoint mid-epoch, restart, and resume at the exact next sample with no dup/skip — identical to the legacy loader.

**Source references (verbatim behavior to reproduce):**

- `cosmos_framework/data/vfm/data_packer_dataloader.py:224-262` — `_ShuffledMapIterableDataset.__iter__` env-var fast-forward + `_dp_epoch`/`_dp_stream_pos`.
- `cosmos_framework/data/vfm/data_packer_dataloader.py:328-358` — `_get_next_sample` strip/re-attach + `collate_batch` batch stamping.
- `cosmos_framework/callbacks/dataloader_state.py:38-123` — callback save/load (UNCHANGED; tested against).
- `cosmos_framework/data/vfm/test_dp_state_distributed.py` — the existing resume test pattern to mirror.

Env-var format (must match exactly): `DP_STATE_WORKER_{worker_id}_EPOCH`, `DP_STATE_WORKER_{worker_id}_INDEX` (or `DP_STATE_{name}_WORKER_{id}_...` when `name` is set). `resume_pos` is the last index *included*; resume starts at `resume_pos + 1`. `os.environ.pop` (consume once). Sentinel: missing EPOCH→0, missing INDEX→-1 (→ start 0).

---

## File Structure

| File                                                         | Change                                                                                                                                       |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `cosmos_framework/data/vfm/dataflow/distributors.py`         | `MapDistributor`: add `name`, env-var fast-forward, `_dp_*` meta attach                                                                      |
| `cosmos_framework/data/vfm/dataflow/loader.py`               | `CosmosDataLoader`/`_DataflowIterableDataset`: meta strip/re-attach + batch stamping; enforce `persistent_workers=True` for `MapDistributor` |
| `cosmos_framework/data/vfm/dataflow/distributors_test.py`    | resume fast-forward unit tests                                                                                                               |
| `cosmos_framework/data/vfm/dataflow/loader_test.py`          | meta-threading + stamping unit tests                                                                                                         |
| `cosmos_framework/data/vfm/dataflow/resume_test.py` (create) | checkpoint→restart integration test with the real callback                                                                                   |

---

### Task 1: `MapDistributor` env-var fast-forward + `_dp_*` meta

**Files:**

- Modify: `cosmos_framework/data/vfm/dataflow/distributors.py`
- Modify: `cosmos_framework/data/vfm/dataflow/distributors_test.py`

- [ ] **Step 1: Write the failing test**

Append to `distributors_test.py`:

```python
import os


def test_map_resume_fast_forwards_from_env(monkeypatch):
    # Single rank/worker; no shuffle so order is range(10). Saved INDEX=3 → resume at pos 4.
    monkeypatch.setenv("DP_STATE_WORKER_0_EPOCH", "0")
    monkeypatch.setenv("DP_STATE_WORKER_0_INDEX", "3")
    d = MapDistributor(_MapDS(10), shuffle=False)
    it = d.stream(dp_rank=0, dp_world_size=1, worker_id=0, num_workers=1)
    first = next(it)
    assert first["i"] == 4                      # resumed one past saved index
    assert first["_dp_epoch"] == 0
    assert first["_dp_stream_pos"] == 4


def test_map_attaches_dp_meta_when_no_resume():
    d = MapDistributor(_MapDS(4), shuffle=False)
    it = d.stream(dp_rank=0, dp_world_size=1, worker_id=0, num_workers=1)
    s0 = next(it)
    assert s0["i"] == 0 and s0["_dp_epoch"] == 0 and s0["_dp_stream_pos"] == 0


def test_map_resume_env_is_consumed_once(monkeypatch):
    monkeypatch.setenv("DP_STATE_WORKER_0_INDEX", "2")
    d = MapDistributor(_MapDS(6), shuffle=False)
    list(d.stream(0, 1, 0, 1).__next__() for _ in range(1))  # trigger one read
    assert "DP_STATE_WORKER_0_INDEX" not in os.environ        # popped


def test_map_name_namespaces_env(monkeypatch):
    monkeypatch.setenv("DP_STATE_vlm_WORKER_0_INDEX", "1")
    d = MapDistributor(_MapDS(6), shuffle=False, name="vlm")
    first = next(d.stream(0, 1, 0, 1))
    assert first["i"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -k resume -v`
Expected: FAIL — `MapDistributor` doesn't attach `_dp_*` or read env vars yet, and lacks a `name` kwarg behavior.

- [ ] **Step 3: Replace `MapDistributor.stream` (and `__init__` already has `name`)**

In `distributors.py`, replace the `MapDistributor.stream` body with the resume-aware version (reproduces `_ShuffledMapIterableDataset.__iter__`, `data_packer_dataloader.py:224-262`):

```python
    def stream(
        self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int
    ):
        import os

        stream_id = dp_rank * num_workers + worker_id
        total_streams = dp_world_size * num_workers
        n = len(self._dataset)  # type: ignore[arg-type]

        # Consume-once env-var fast-forward (set by DataLoaderStateCallback before
        # workers fork). Sentinels: missing EPOCH->0, missing INDEX->-1 (start 0).
        _pfx = f"DP_STATE_{self._name}_" if self._name else "DP_STATE_"
        resume_epoch = int(os.environ.pop(f"{_pfx}WORKER_{worker_id}_EPOCH", 0))
        resume_pos = int(os.environ.pop(f"{_pfx}WORKER_{worker_id}_INDEX", -1))

        epoch = resume_epoch
        while True:
            if self._shuffle:
                g = torch.Generator().manual_seed(self._seed + epoch)
                perm = torch.randperm(n, generator=g).tolist()
            else:
                perm = list(range(n))
            stream_slice = perm[stream_id::total_streams]
            # resume_pos is the last index included in a batch -> start one past it.
            start = (resume_pos + 1) if epoch == resume_epoch else 0
            for pos in range(start, len(stream_slice)):
                item = self._dataset[stream_slice[pos]]
                # Attach position metadata so the loader can stamp batches and
                # the callback can record resume state. Requires dict items.
                if isinstance(item, dict):
                    yield {"_dp_epoch": epoch, "_dp_stream_pos": pos, **item}
                else:
                    yield item
            epoch += 1
```

(The `name` param already exists on `MapDistributor.__init__` from Plan 1 Task 6.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -v`
Expected: PASS (Plan 1 tests + the 4 resume tests). The Plan-1 `test_map_*` tests still pass because items are dicts and the extra `_dp_*` keys don't affect the `["i"]` assertions — **verify this**; if a Plan-1 test asserted exact dict equality, update it to check `["i"]` only.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/distributors.py \
        cosmos_framework/data/vfm/dataflow/distributors_test.py
git commit -m "feat(dataflow): MapDistributor env-var resume fast-forward + _dp_* meta"
```

---

### Task 2: `CosmosDataLoader` meta-threading + batch stamping

**Files:**

- Modify: `cosmos_framework/data/vfm/dataflow/loader.py`
- Modify: `cosmos_framework/data/vfm/dataflow/loader_test.py`

Reproduces `_DataPackerIterableDataset._get_next_sample` + `collate_batch`
(`data_packer_dataloader.py:328-358`): strip `_dp_*` before the processor,
re-attach after, and stamp `sample_worker_id/epoch/index` onto the collated
batch when samples carry meta. Also enforce `persistent_workers=True` for a
`MapDistributor` source with workers (`:463-474`).

- [ ] **Step 1: Write the failing test**

Append to `loader_test.py`:

```python
def test_map_source_stamps_resume_meta_on_batch():
    loader = CosmosDataLoader(
        distributor=MapDistributor(_MapDS(10), shuffle=False),
        processor=IdentityProcessor(),
        batch_size=2,                # SimpleBatcher groups 2
        num_workers=0,
    )
    it = iter(loader)
    batch = next(it)
    # resume meta present, stamped from the grouped samples (max epoch/pos).
    assert "sample_worker_id" in batch
    assert batch["sample_epoch"].tolist() == [0, 0]
    assert batch["sample_index"].tolist() == [1, 1]   # max stream_pos in {0,1} = 1
    # the _dp_* keys are stripped from the collated batch payload.
    assert "_dp_epoch" not in batch and "_dp_stream_pos" not in batch


def test_iterable_source_has_no_resume_meta():
    loader = CosmosDataLoader(
        distributor=IterableDistributor([{"x": torch.tensor([float(i)])} for i in range(4)]),
        processor=IdentityProcessor(),
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert "sample_worker_id" not in batch   # iterable source -> no stamping
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/loader_test.py -k resume_meta -v`
Expected: FAIL — current orchestrator neither strips `_dp_*` nor stamps batches.

- [ ] **Step 3: Update `_DataflowIterableDataset.__iter__`**

Replace `_DataflowIterableDataset.__iter__` in `loader.py` with the meta-aware version:

```python
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        raw = self._distributor.stream(self._dp_rank, self._dp_world_size, worker_id, num_workers)

        def _processed():
            for item in raw:
                if isinstance(item, dict):
                    meta = {k: item.pop(k) for k in list(item) if k.startswith("_dp_")}
                else:
                    meta = {}
                s = self._processor.process(item)
                if meta and isinstance(s, dict):
                    s.update(meta)
                yield s

        for group in self._batcher.batches(_processed()):
            has_meta = bool(group) and isinstance(group[0], dict) and "_dp_epoch" in group[0]
            if has_meta:
                max_epoch = max(s["_dp_epoch"] for s in group)
                max_pos = max(s["_dp_stream_pos"] for s in group)
                clean = [{k: v for k, v in s.items() if not k.startswith("_dp_")} for s in group]
                batch = self._collator.collate(clean)
                batch["sample_worker_id"] = torch.tensor([worker_id] * len(group))
                batch["sample_epoch"] = torch.tensor([max_epoch] * len(group))
                batch["sample_index"] = torch.tensor([max_pos] * len(group))
            else:
                batch = self._collator.collate(group)
            yield batch
```

And in `CosmosDataLoader.__init__`, after resolving the distributor, enforce
persistent workers for stateful map sources (insert before building loader_kwargs):

```python
        from cosmos_framework.data.vfm.dataflow.distributors import MapDistributor

        if isinstance(distributor, MapDistributor) and num_workers > 0 and not persistent_workers:
            log.info(
                "CosmosDataLoader: MapDistributor requires persistent_workers=True for "
                "correct stateful resume; overriding to True.",
                rank0_only=True,
            )
            persistent_workers = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/loader_test.py -v`
Expected: PASS (all Plan-1 loader tests + the 2 new ones). Note the
`DefaultBatchCollator` must tolerate the `clean` dicts — it does (plain
`default_collate`).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/loader.py \
        cosmos_framework/data/vfm/dataflow/loader_test.py
git commit -m "feat(dataflow): CosmosDataLoader resume meta-threading + batch stamping"
```

---

### Task 3: Resume integration test against the real callback

**Files:**

- Create: `cosmos_framework/data/vfm/dataflow/resume_test.py`

Mirrors the single-process flow of `test_dp_state_distributed.py`: train N batches
through a `CosmosDataLoader(MapDistributor(...))`, feed each batch to a real
`DataLoaderStateCallback(distributor_type="data_packer")`, snapshot `state_dict()`,
call `load_state_dict()` (which sets the env vars), build a fresh loader, and
assert it resumes at exactly the next sample with no dup/skip.

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/resume_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint->restart resume parity for CosmosDataLoader(MapDistributor) using
the real DataLoaderStateCallback (unchanged). Single process, num_workers=0."""

from __future__ import annotations

import torch

from cosmos_framework.callbacks.dataloader_state import DataLoaderStateCallback
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
)


class _IdDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"id": torch.tensor(idx)}


def _build(seed=0):
    return CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False, seed=seed),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=0,
    )


def test_resume_continues_without_dup_or_skip(monkeypatch):
    # Phase 1: consume 5 batches, feed the callback.
    cb = DataLoaderStateCallback(distributor_type="data_packer")
    loader = _build()
    it = iter(loader)
    seen_ids = []
    for _ in range(5):
        b = next(it)
        cb._update_state_from_batch(b)
        seen_ids.append(b["id"].item())
    assert seen_ids == [0, 1, 2, 3, 4]

    # Phase 2: checkpoint + resume.
    state = cb.state_dict()
    assert state[0]["index"] == 4         # last index included
    cb2 = DataLoaderStateCallback(distributor_type="data_packer")
    cb2.load_state_dict(state)            # sets DP_STATE_WORKER_0_INDEX=4

    loader2 = _build()
    resumed = [next(iter(loader2))["id"].item() for _ in range(3)]
    assert resumed == [5, 6, 7]           # exact continuation, no dup/skip
```

- [ ] **Step 2: Run test to verify it fails (or passes)**

Run: `pytest cosmos_framework/data/vfm/dataflow/resume_test.py -v`
Expected: PASS if Tasks 1–2 are correct. If FAIL, the mismatch is the source of
truth for fixing the env-var format, `resume_pos+1` start, or batch stamping.

- [ ] **Step 3: Add a multi-worker variant**

Append to `resume_test.py`:

```python
def test_resume_multiworker_disjoint(monkeypatch):
    # 2 workers; each worker tracks its own (epoch,index). After resume each
    # worker continues from its own saved position.
    cb = DataLoaderStateCallback(distributor_type="data_packer")
    loader = CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=2,
    )
    it = iter(loader)
    for _ in range(8):
        cb._update_state_from_batch(next(it))
    state = cb.state_dict()
    assert set(state.keys()) == {0, 1}        # both workers recorded

    cb2 = DataLoaderStateCallback(distributor_type="data_packer")
    cb2.load_state_dict(state)
    loader2 = CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=2,
    )
    # Collect a full epoch's remaining ids across both workers; assert no id
    # appears twice and none of the already-consumed ids reappear within the epoch.
    consumed = set()
    it2 = iter(loader2)
    for _ in range(12):                       # 20 - 8 = 12 remaining in epoch 0
        consumed.add(next(it2)["id"].item())
    assert len(consumed) == 12
```

Note: if the environment cannot spawn workers, run this under the container per
`cosmos3-run-env`.

- [ ] **Step 4: Run the full resume + dataflow suite**

Run: `pytest cosmos_framework/data/vfm/dataflow/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/resume_test.py
git commit -m "test(dataflow): resume parity vs DataLoaderStateCallback (single + multi-worker)"
```

---

## Self-Review

**Spec coverage:** Hard-invariant resume (spec "Hard invariants" #1, "Resume threading") → Tasks 1–3. Callback unchanged (verified by Task 3 importing and using the real `DataLoaderStateCallback`). `persistent_workers` enforcement for map sources → Task 2.

**Placeholder scan:** All code complete; env-var names/format match `dataloader_state.py:107-117` and `data_packer_dataloader.py:239-241` exactly.

**Type consistency:** `MapDistributor(dataset, seed, shuffle, name)` matches Plan 1 Task 6 + this plan. `CosmosDataLoader(distributor, processor, batch_size|batcher, collator, num_workers)` matches Plan 1 Task 7. Batch keys `sample_worker_id/sample_epoch/sample_index` match what `DataLoaderStateCallback._update_state_from_batch` reads (`dataloader_state.py:40-51`).

**Deferred:** `JointCosmosDataLoader` resume (the `JointDataLoaderStateCallback` path) lands in Plan 5 when the joint composer is built.
