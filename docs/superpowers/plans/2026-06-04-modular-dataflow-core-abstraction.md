# Modular Dataflow — Core Abstraction Implementation Plan (Plan 1 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four-role dataflow abstraction (`DataDistributor`, `RawItemProcessor`, `SampleBatcher`, `BatchCollator`) plus the slim `DataPackerDataLoader` orchestrator and the simple/default built-ins, as a new package that touches no existing recipe.

**Architecture:** A new `cosmos_framework/data/vfm/dataflow/` package. Four small ABCs, each with one responsibility and a different arity (item→sample, sample-stream→list, list→batch). A new `DataPackerDataLoader` wires them in fixed order (`distribute → process → batch → collate`) inside each DataLoader worker, resolving DP coordinates and passing them into `DataDistributor.stream(...)`. The existing `data_packer_dataloader.py` / `joint_dataloader.py` / `packing_iterable_dataset.py` stay UNTOUCHED — this plan only adds code.

**Tech Stack:** Python, PyTorch (`torch.utils.data.DataLoader`/`IterableDataset`), pytest (co-located `*_test.py`).

**Scope boundaries (explicit):**
- IN: the 4 ABCs; `IdentityProcessor`, `DefaultBatchCollator`, `SimpleBatcher`, `IterableDistributor`, `MapDistributor` (shuffle + sharding only); the new `DataPackerDataLoader` (DP-coord resolution, `batch_size` sugar, worker config, `__iter__` orchestration); unit + integration tests.
- OUT (later plans): `PoolPackingBatcher` / `SequentialPackingBatcher`, `RankPartitionedDistributor`, `MixtureDistributor`; resume/`state_dict` env-var fast-forward + `DataLoaderStateCallback` integration + `_dp_*` meta threading; all recipe migrations; `docs/dataflow.md`; deletion of old code.
- `state_dict`/`load_state_dict` are defined on the ABC as no-op defaults so the contract exists; their real bodies land in the resume plan.

**Spec:** `docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `cosmos_framework/data/vfm/dataflow/__init__.py` | Re-export the 4 ABCs + built-ins |
| `cosmos_framework/data/vfm/dataflow/base.py` | The 4 role ABCs |
| `cosmos_framework/data/vfm/dataflow/processors.py` | `IdentityProcessor` |
| `cosmos_framework/data/vfm/dataflow/collators.py` | `DefaultBatchCollator` |
| `cosmos_framework/data/vfm/dataflow/batchers.py` | `SimpleBatcher` |
| `cosmos_framework/data/vfm/dataflow/distributors.py` | `IterableDistributor`, `MapDistributor` |
| `cosmos_framework/data/vfm/dataflow/loader.py` | `DataPackerDataLoader` orchestrator |
| `cosmos_framework/data/vfm/dataflow/base_test.py` | ABC contract tests |
| `cosmos_framework/data/vfm/dataflow/processors_test.py` | `IdentityProcessor` test |
| `cosmos_framework/data/vfm/dataflow/collators_test.py` | `DefaultBatchCollator` test |
| `cosmos_framework/data/vfm/dataflow/batchers_test.py` | `SimpleBatcher` test |
| `cosmos_framework/data/vfm/dataflow/distributors_test.py` | distributor sharding/shuffle tests |
| `cosmos_framework/data/vfm/dataflow/loader_test.py` | end-to-end orchestrator tests |

The new `DataPackerDataLoader` lives in `dataflow/loader.py` (NOT the old `data_packer_dataloader.py`) so both coexist during migration; the cleanup PR later deletes the old module and makes `dataflow` canonical.

---

### Task 1: Scaffold package + the four role ABCs

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Create: `cosmos_framework/data/vfm/dataflow/base.py`
- Test: `cosmos_framework/data/vfm/dataflow/base_test.py`

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/base_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Contract tests for the four dataflow role ABCs."""

from __future__ import annotations

import inspect

import pytest

from cosmos_framework.data.vfm.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)


def test_abcs_cannot_be_instantiated():
    for cls in (DataDistributor, RawItemProcessor, SampleBatcher, BatchCollator):
        with pytest.raises(TypeError):
            cls()  # abstract


def test_distributor_state_dict_defaults_are_noops():
    class _D(DataDistributor):
        def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
            yield from ()

    d = _D()
    assert d.state_dict() == {}
    assert d.load_state_dict({"anything": 1}) is None  # no-op default


def test_role_method_signatures():
    assert list(inspect.signature(DataDistributor.stream).parameters) == [
        "self", "dp_rank", "dp_world_size", "worker_id", "num_workers",
    ]
    assert list(inspect.signature(RawItemProcessor.process).parameters) == ["self", "item"]
    assert list(inspect.signature(SampleBatcher.batches).parameters) == ["self", "samples"]
    assert list(inspect.signature(BatchCollator.collate).parameters) == ["self", "samples"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/base_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cosmos_framework.data.vfm.dataflow'`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/data/vfm/dataflow/base.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The four dataflow role ABCs.

A raw item flows through four independently-swappable roles in a fixed order
enforced by the loader:

    DataDistributor -> RawItemProcessor -> SampleBatcher -> BatchCollator

See docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class DataDistributor(ABC):
    """Owns the raw dataset, shards it disjointly across DP ranks x workers,
    shuffles, and (later) carries checkpoint/resume state."""

    @abstractmethod
    def stream(
        self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int
    ) -> Iterator[Any]:
        """Yield this (rank, worker)'s disjoint slice of raw items, indefinitely."""

    def state_dict(self) -> dict:
        """Resume state. No-op default; resumable distributors override."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore resume state. No-op default; resumable distributors override."""
        return None


class RawItemProcessor(ABC):
    """Transforms one raw dataset item into one training-ready sample dict."""

    @abstractmethod
    def process(self, item: Any) -> dict:
        ...


class SampleBatcher(ABC):
    """Consumes a stream of samples and yields groups (the selection strategy)."""

    @abstractmethod
    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        """Pull from ``samples``; yield one ``list[dict]`` per batch."""

    def sample_size(self, sample: dict) -> int:
        """Per-sample token cost for packing batchers. Non-packing batchers
        never call this; packing batchers override it (or inject a size_fn)."""
        raise NotImplementedError


class BatchCollator(ABC):
    """Collates one group of samples into one batch dict for ``model.forward()``."""

    @abstractmethod
    def collate(self, samples: list[dict]) -> dict:
        ...
```

`cosmos_framework/data/vfm/dataflow/__init__.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Modular training dataflow: DataDistributor -> RawItemProcessor ->
SampleBatcher -> BatchCollator, wired by DataPackerDataLoader."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)

__all__ = [
    "BatchCollator",
    "DataDistributor",
    "RawItemProcessor",
    "SampleBatcher",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/base_test.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/__init__.py \
        cosmos_framework/data/vfm/dataflow/base.py \
        cosmos_framework/data/vfm/dataflow/base_test.py
git commit -m "feat(dataflow): add four role ABCs (DataDistributor/RawItemProcessor/SampleBatcher/BatchCollator)"
```

---

### Task 2: `IdentityProcessor`

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/processors.py`
- Modify: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Test: `cosmos_framework/data/vfm/dataflow/processors_test.py`

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/processors_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for IdentityProcessor."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.processors import IdentityProcessor


def test_identity_returns_item_unchanged():
    item = {"input_ids": [1, 2, 3], "label": 7}
    out = IdentityProcessor().process(item)
    assert out is item  # no copy, no mutation


def test_identity_passes_non_dict_through():
    # Items are typically dicts, but IdentityProcessor must not assume so.
    obj = object()
    assert IdentityProcessor().process(obj) is obj
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/processors_test.py -v`
Expected: FAIL — `ModuleNotFoundError: ... dataflow.processors`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/data/vfm/dataflow/processors.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in RawItemProcessor implementations."""

from __future__ import annotations

from typing import Any

from cosmos_framework.data.vfm.dataflow.base import RawItemProcessor


class IdentityProcessor(RawItemProcessor):
    """No-op processor: the dataset already yields training-ready samples.

    Used by recipes that keep heavy per-sample processing inside the dataset
    (VFM SFTDataset, DROID) — see the spec's "Processor placement" section.
    """

    def process(self, item: Any) -> Any:
        return item
```

Add to `__init__.py` (alphabetical in the import block and `__all__`):
```python
from cosmos_framework.data.vfm.dataflow.processors import IdentityProcessor
```
and add `"IdentityProcessor",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/processors_test.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/processors.py \
        cosmos_framework/data/vfm/dataflow/processors_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add IdentityProcessor"
```

---

### Task 3: `DefaultBatchCollator`

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/collators.py`
- Modify: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Test: `cosmos_framework/data/vfm/dataflow/collators_test.py`

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/collators_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for DefaultBatchCollator."""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.dataflow.collators import DefaultBatchCollator


def test_default_collator_stacks_like_torch():
    samples = [
        {"x": torch.tensor([1.0, 2.0]), "y": 0},
        {"x": torch.tensor([3.0, 4.0]), "y": 1},
    ]
    batch = DefaultBatchCollator().collate(samples)
    assert batch["x"].shape == (2, 2)
    assert torch.equal(batch["x"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(batch["y"], torch.tensor([0, 1]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/collators_test.py -v`
Expected: FAIL — `ModuleNotFoundError: ... dataflow.collators`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/data/vfm/dataflow/collators.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in BatchCollator implementations."""

from __future__ import annotations

import torch.utils.data

from cosmos_framework.data.vfm.dataflow.base import BatchCollator


class DefaultBatchCollator(BatchCollator):
    """Stacks samples with torch's default_collate — stock DataLoader behavior."""

    def collate(self, samples: list[dict]) -> dict:
        return torch.utils.data.default_collate(samples)
```

Add to `__init__.py`:
```python
from cosmos_framework.data.vfm.dataflow.collators import DefaultBatchCollator
```
and `"DefaultBatchCollator",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/collators_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/collators.py \
        cosmos_framework/data/vfm/dataflow/collators_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add DefaultBatchCollator"
```

---

### Task 4: `SimpleBatcher`

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/batchers.py`
- Modify: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Test: `cosmos_framework/data/vfm/dataflow/batchers_test.py`

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/batchers_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for SimpleBatcher."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.batchers import SimpleBatcher


def _src(n):
    return iter([{"i": i} for i in range(n)])


def test_simple_batcher_groups_in_fixed_size():
    groups = list(SimpleBatcher(batch_size=3).batches(_src(7)))
    assert [[s["i"] for s in g] for g in groups] == [[0, 1, 2], [3, 4, 5], [6]]


def test_simple_batcher_drop_last_discards_partial():
    groups = list(SimpleBatcher(batch_size=3, drop_last=True).batches(_src(7)))
    assert [[s["i"] for s in g] for g in groups] == [[0, 1, 2], [3, 4, 5]]


def test_simple_batcher_empty_source_yields_nothing():
    assert list(SimpleBatcher(batch_size=3).batches(_src(0))) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/batchers_test.py -v`
Expected: FAIL — `ModuleNotFoundError: ... dataflow.batchers`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/data/vfm/dataflow/batchers.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in SampleBatcher implementations."""

from __future__ import annotations

from typing import Iterator

from cosmos_framework.data.vfm.dataflow.base import SampleBatcher


class SimpleBatcher(SampleBatcher):
    """Fixed-size batching — stock DataLoader behavior. Never needs sample_size."""

    def __init__(self, batch_size: int, drop_last: bool = False):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size
        self.drop_last = drop_last

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        buf: list[dict] = []
        for s in samples:
            buf.append(s)
            if len(buf) == self.batch_size:
                yield buf
                buf = []
        if buf and not self.drop_last:
            yield buf
```

Add to `__init__.py`:
```python
from cosmos_framework.data.vfm.dataflow.batchers import SimpleBatcher
```
and `"SimpleBatcher",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/batchers_test.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/batchers.py \
        cosmos_framework/data/vfm/dataflow/batchers_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add SimpleBatcher"
```

---

### Task 5: `IterableDistributor`

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/distributors.py`
- Modify: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Test: `cosmos_framework/data/vfm/dataflow/distributors_test.py`

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/distributors_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for IterableDistributor (and, later, MapDistributor)."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.distributors import IterableDistributor


def _take(it, n):
    out = []
    for _ in range(n):
        out.append(next(it))
    return out


def test_iterable_single_rank_single_worker_sees_everything():
    d = IterableDistributor(range(6))
    got = list(d.stream(dp_rank=0, dp_world_size=1, worker_id=0, num_workers=1))
    assert got == [0, 1, 2, 3, 4, 5]


def test_iterable_sharding_is_disjoint_and_covers_all():
    # 2 ranks x 2 workers = 4 disjoint streams over range(12).
    seen = []
    for r in range(2):
        for w in range(2):
            d = IterableDistributor(range(12))
            seen.append(set(d.stream(dp_rank=r, dp_world_size=2, worker_id=w, num_workers=2)))
    # disjoint
    for a in range(4):
        for b in range(a + 1, 4):
            assert seen[a].isdisjoint(seen[b]), (a, b, seen[a], seen[b])
    # full coverage
    assert set().union(*seen) == set(range(12))


def test_iterable_stream_indices_match_formula():
    # rank=1, world=2, worker=0, workers=2 -> total=4, mine=2 -> indices 2,6,10
    d = IterableDistributor(range(12))
    got = list(d.stream(dp_rank=1, dp_world_size=2, worker_id=0, num_workers=2))
    assert got == [2, 6, 10]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -v`
Expected: FAIL — `ModuleNotFoundError: ... dataflow.distributors`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/data/vfm/dataflow/distributors.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in DataDistributor implementations.

IterableDistributor wraps any Python iterable / IterableDataset with
round-robin DP x worker sharding (no resume). MapDistributor wraps a map-style
Dataset with per-epoch shuffle + slice sharding (resume lands in a later plan).
"""

from __future__ import annotations

from typing import Any, Iterator

from cosmos_framework.data.vfm.dataflow.base import DataDistributor


class IterableDistributor(DataDistributor):
    """Round-robin shard of an iterable: each (rank, worker) sees every
    ``dp_world_size * num_workers``-th item starting at
    ``dp_rank * num_workers + worker_id``. Generalizes the old _IterableWrapper.
    Not resumable (an arbitrary iterable cannot be random-accessed)."""

    def __init__(self, iterable: Any):
        self._iterable = iterable

    def stream(
        self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int
    ) -> Iterator[Any]:
        total_streams = dp_world_size * num_workers
        my_stream = dp_rank * num_workers + worker_id
        for i, item in enumerate(self._iterable):
            if i % total_streams == my_stream:
                yield item
```

Add to `__init__.py`:
```python
from cosmos_framework.data.vfm.dataflow.distributors import IterableDistributor
```
and `"IterableDistributor",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/distributors.py \
        cosmos_framework/data/vfm/dataflow/distributors_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add IterableDistributor with round-robin sharding"
```

---

### Task 6: `MapDistributor` (shuffle + sharding; no resume yet)

**Files:**
- Modify: `cosmos_framework/data/vfm/dataflow/distributors.py`
- Modify: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Modify: `cosmos_framework/data/vfm/dataflow/distributors_test.py`

- [ ] **Step 1: Write the failing test**

Append to `cosmos_framework/data/vfm/dataflow/distributors_test.py`:
```python
import torch

from cosmos_framework.data.vfm.dataflow.distributors import MapDistributor


class _MapDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"i": idx}


def test_map_no_shuffle_is_sequential_and_sharded():
    d = MapDistributor(_MapDS(12), shuffle=False)
    it = d.stream(dp_rank=1, dp_world_size=2, worker_id=0, num_workers=2)
    # perm = range(12); stream_id = 1*2+0 = 2; total = 4 -> 2,6,10 then epoch 2 -> repeats
    first = [next(it)["i"] for _ in range(3)]
    assert first == [2, 6, 10]
    # infinite: wraps to next epoch (same order since no shuffle)
    assert next(it)["i"] == 2


def test_map_shuffle_is_seeded_and_reproducible():
    a = MapDistributor(_MapDS(20), shuffle=True, seed=123)
    b = MapDistributor(_MapDS(20), shuffle=True, seed=123)
    sa = [next(a.stream(0, 1, 0, 1))["i"] for _ in range(1)]  # noqa: F841 (smoke)
    ia = a.stream(0, 1, 0, 1)
    ib = b.stream(0, 1, 0, 1)
    assert [next(ia)["i"] for _ in range(20)] == [next(ib)["i"] for _ in range(20)]


def test_map_shuffle_first_epoch_is_a_permutation():
    d = MapDistributor(_MapDS(20), shuffle=True, seed=7)
    it = d.stream(0, 1, 0, 1)
    first_epoch = [next(it)["i"] for _ in range(20)]
    assert sorted(first_epoch) == list(range(20))


def test_map_sharding_disjoint_and_covers_one_epoch():
    # 2 ranks x 2 workers, shuffle off; collect exactly one epoch (3 each) per stream.
    seen = []
    for r in range(2):
        for w in range(2):
            it = MapDistributor(_MapDS(12), shuffle=False).stream(r, 2, w, 2)
            seen.append({next(it)["i"] for _ in range(3)})
    for a in range(4):
        for b in range(a + 1, 4):
            assert seen[a].isdisjoint(seen[b])
    assert set().union(*seen) == set(range(12))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'MapDistributor'`.

- [ ] **Step 3: Write minimal implementation**

Append to `cosmos_framework/data/vfm/dataflow/distributors.py`:
```python
import torch


class MapDistributor(DataDistributor):
    """Per-epoch shuffle + slice sharding of a map-style Dataset. Generalizes the
    old _ShuffledMapIterableDataset (sharding/shuffle half only).

    - shuffle=True:  per-epoch ``torch.randperm(n)`` seeded ``seed + epoch``.
    - shuffle=False: sequential ``range(n)`` each epoch.
    - sharding:      ``stream_id = dp_rank * num_workers + worker_id``; each
                     stream yields ``perm[stream_id :: dp_world_size * num_workers]``.
    - infinite:      loops epochs forever (training pulls what it needs).

    Resume (state_dict/load_state_dict env-var fast-forward) is added in a later
    plan; for now the ABC's no-op defaults apply.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        seed: int = 0,
        shuffle: bool = True,
        name: str = "",
    ):
        self._dataset = dataset
        self._seed = seed
        self._shuffle = shuffle
        self._name = name

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def stream(
        self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int
    ):
        stream_id = dp_rank * num_workers + worker_id
        total_streams = dp_world_size * num_workers
        n = len(self._dataset)  # type: ignore[arg-type]
        epoch = 0
        while True:
            if self._shuffle:
                g = torch.Generator().manual_seed(self._seed + epoch)
                perm = torch.randperm(n, generator=g).tolist()
            else:
                perm = list(range(n))
            stream_slice = perm[stream_id::total_streams]
            for pos in range(len(stream_slice)):
                yield self._dataset[stream_slice[pos]]
            epoch += 1
```

Add to `__init__.py`:
```python
from cosmos_framework.data.vfm.dataflow.distributors import IterableDistributor, MapDistributor
```
(replace the existing single-name import line) and add `"MapDistributor",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -v`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/distributors.py \
        cosmos_framework/data/vfm/dataflow/distributors_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add MapDistributor (per-epoch shuffle + slice sharding)"
```

---

### Task 7: `DataPackerDataLoader` orchestrator

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/loader.py`
- Modify: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Test: `cosmos_framework/data/vfm/dataflow/loader_test.py`

This task has two internal classes: a private `_DataflowIterableDataset` (wires
the roles inside the worker) and the public `DataPackerDataLoader`. DP coords:
`parallel_dims.dp_coord` > `torch.distributed` > `(0, 1)` — mirrors
`data_packer_dataloader.py:476-496`.

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/loader_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""End-to-end tests for the DataPackerDataLoader orchestrator (Plan 1 scope:
explicit roles + batch_size sugar, single process, no resume)."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.data.vfm.dataflow import (
    DataPackerDataLoader,
    IdentityProcessor,
    IterableDistributor,
    MapDistributor,
    SimpleBatcher,
)


class _MapDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"x": torch.tensor([float(idx)]), "i": idx}


def test_explicit_roles_end_to_end():
    loader = DataPackerDataLoader(
        distributor=IterableDistributor([{"x": torch.tensor([float(i)])} for i in range(6)]),
        processor=IdentityProcessor(),
        batcher=SimpleBatcher(batch_size=2),
    )
    batches = []
    it = iter(loader)
    for _ in range(3):
        batches.append(next(it))
    assert [b["x"].shape[0] for b in batches] == [2, 2, 2]
    assert torch.equal(batches[0]["x"].flatten(), torch.tensor([0.0, 1.0]))


def test_batch_size_sugar_builds_simple_batcher_and_default_collator():
    loader = DataPackerDataLoader(
        distributor=MapDistributor(_MapDS(10), shuffle=False),
        processor=IdentityProcessor(),
        batch_size=4,
    )
    it = iter(loader)
    batch = next(it)
    assert batch["x"].shape == (4, 1)
    assert batch["i"].tolist() == [0, 1, 2, 3]


def test_batch_size_with_explicit_batcher_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        DataPackerDataLoader(
            distributor=IterableDistributor([]),
            processor=IdentityProcessor(),
            batch_size=4,
            batcher=SimpleBatcher(batch_size=2),
        )


def test_requires_batcher_or_batch_size():
    with pytest.raises(ValueError, match="batcher.*batch_size|batch_size.*batcher"):
        DataPackerDataLoader(
            distributor=IterableDistributor([]),
            processor=IdentityProcessor(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/loader_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'DataPackerDataLoader'`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/data/vfm/dataflow/loader.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DataPackerDataLoader — slim orchestrator that wires the four dataflow roles
(DataDistributor -> RawItemProcessor -> SampleBatcher -> BatchCollator) inside
each DataLoader worker.

Lives in dataflow/loader.py during the migration so it coexists with the legacy
cosmos_framework/data/vfm/data_packer_dataloader.py; the cleanup PR makes this
the canonical DataPackerDataLoader.
"""

from __future__ import annotations

import torch
import torch.utils.data

from cosmos_framework.utils import log
from cosmos_framework.data.vfm.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)
from cosmos_framework.data.vfm.dataflow.batchers import SimpleBatcher
from cosmos_framework.data.vfm.dataflow.collators import DefaultBatchCollator


class _DataflowIterableDataset(torch.utils.data.IterableDataset):
    """Wires distributor -> processor -> batcher -> collator inside a worker."""

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher,
        collator: BatchCollator,
        dp_rank: int,
        dp_world_size: int,
    ):
        super().__init__()
        self._distributor = distributor
        self._processor = processor
        self._batcher = batcher
        self._collator = collator
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        raw = self._distributor.stream(self._dp_rank, self._dp_world_size, worker_id, num_workers)
        samples = (self._processor.process(item) for item in raw)
        for group in self._batcher.batches(samples):
            yield self._collator.collate(group)


class DataPackerDataLoader(torch.utils.data.DataLoader):
    """Public entry point: bring any dataset into training via four roles.

    Either pass an explicit ``batcher`` (and optional ``collator``), or pass a
    bare ``batch_size=N`` for stock fixed-size batching — the loader then builds
    ``SimpleBatcher(N)`` + ``DefaultBatchCollator()``. Passing both is an error.

    DP coordinates: ``parallel_dims.dp_coord`` > ``torch.distributed`` > (0, 1).
    """

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher | None = None,
        collator: BatchCollator | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        parallel_dims=None,
    ):
        if batch_size is not None and batcher is not None:
            raise ValueError(
                "Pass either batch_size= (sugar) or an explicit batcher=, not both."
            )
        if batch_size is None and batcher is None:
            raise ValueError("Provide either a batcher= or a batch_size=.")
        if batch_size is not None:
            batcher = SimpleBatcher(batch_size=batch_size)
        if collator is None:
            collator = DefaultBatchCollator()

        # Resolve data-parallel rank/world-size.
        if parallel_dims is not None:
            dp_rank, dp_world_size = parallel_dims.dp_coord
        elif torch.distributed.is_initialized():
            dp_rank = torch.distributed.get_rank()
            dp_world_size = torch.distributed.get_world_size()
            if dp_world_size > 1:
                log.info(
                    "DataPackerDataLoader: using global rank for DP sharding. "
                    "For FSDP+TP/PP pass parallel_dims= for the correct DP rank.",
                    rank0_only=True,
                )
        else:
            dp_rank, dp_world_size = 0, 1

        dataset = _DataflowIterableDataset(
            distributor=distributor,
            processor=processor,
            batcher=batcher,
            collator=collator,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )

        loader_kwargs: dict = dict(
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=pin_memory,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        # batch_size=None: the roles already yield fully-collated batch dicts;
        # disable torch's automatic re-collation.
        super().__init__(dataset, batch_size=None, **loader_kwargs)
```

Add to `__init__.py`:
```python
from cosmos_framework.data.vfm.dataflow.loader import DataPackerDataLoader
```
and `"DataPackerDataLoader",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/loader_test.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/loader.py \
        cosmos_framework/data/vfm/dataflow/loader_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add DataPackerDataLoader orchestrator + batch_size sugar"
```

---

### Task 8: Multi-worker integration test (disjoint coverage through the real DataLoader)

Verifies the orchestrator + distributor sharding give disjoint, complete
coverage when run through an actual multi-worker `torch.utils.data.DataLoader`
(not just direct `stream()` calls).

**Files:**
- Modify: `cosmos_framework/data/vfm/dataflow/loader_test.py`

- [ ] **Step 1: Write the failing test**

Append to `cosmos_framework/data/vfm/dataflow/loader_test.py`:
```python
def test_multiworker_disjoint_and_complete_one_epoch():
    # 1 rank, 2 workers over a 12-item map dataset, batch_size=1, no shuffle.
    # Each worker streams its slice; collect one epoch (6 items/worker) and
    # assert union == all indices, with no duplicates.
    loader = DataPackerDataLoader(
        distributor=MapDistributor(_MapDS(12), shuffle=False),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=2,
    )
    it = iter(loader)
    seen = [next(it)["i"].item() for _ in range(12)]  # 12 = full epoch across 2 workers
    assert sorted(seen) == list(range(12))
    assert len(set(seen)) == 12  # no duplicates
```

- [ ] **Step 2: Run test to verify it fails (then passes — confirm no regression)**

Run: `pytest cosmos_framework/data/vfm/dataflow/loader_test.py::test_multiworker_disjoint_and_complete_one_epoch -v`
Expected: PASS (the orchestrator already supports this). If it FAILS with duplicates/gaps, the worker-sharding wiring in `_DataflowIterableDataset.__iter__` is wrong — fix there.

Note: this is a characterization test for already-built behavior; it has no
separate implementation step. If your environment cannot spawn DataLoader
workers, run with `num_workers=0` semantics by setting `num_workers=2` under the
container per `cosmos3-run-env`.

- [ ] **Step 3: Run the full dataflow suite**

Run: `pytest cosmos_framework/data/vfm/dataflow/ -v`
Expected: PASS (all tasks' tests green).

- [ ] **Step 4: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/loader_test.py
git commit -m "test(dataflow): multi-worker disjoint-coverage integration test"
```

---

## Self-Review

**Spec coverage (Plan 1 scope only):**
- Four role ABCs (spec "Role contracts") → Task 1. ✅
- `IdentityProcessor` (spec "Processor placement") → Task 2. ✅
- `DefaultBatchCollator` (spec built-ins) → Task 3. ✅
- `SimpleBatcher` (spec built-ins) → Task 4. ✅
- `IterableDistributor` / `MapDistributor` (spec built-ins, sharding/shuffle) → Tasks 5–6. ✅
- `DataPackerDataLoader` orchestration + DP-coord resolution + `batch_size` sugar (spec "Loader orchestration") → Task 7. ✅
- Disjoint-coverage sharding behavior (spec correctness claim) → Tasks 5, 6, 8. ✅
- Explicitly OUT of Plan 1 (tracked for later plans): `PoolPackingBatcher`/`SequentialPackingBatcher`, `RankPartitionedDistributor`/`MixtureDistributor`, resume (`state_dict` env-var fast-forward + `DataLoaderStateCallback` + `_dp_*` meta), recipe migrations, `docs/dataflow.md`, old-code deletion. These are named in the spec's Implementation Order phases 2–7.

**Placeholder scan:** No TBD/TODO; every code step contains complete, runnable code; every test step shows the assertions.

**Type consistency:** `DataDistributor.stream(dp_rank, dp_world_size, worker_id, num_workers)` is used identically in Tasks 5, 6, 7. `SampleBatcher.batches(samples)` and `BatchCollator.collate(samples)` match Task 1's ABCs in Tasks 3, 4, 7. `DataPackerDataLoader(distributor=, processor=, batcher=, collator=, batch_size=)` signature in Task 7 matches its usage in Tasks 7–8. `__init__.py` re-exports accumulate consistently (`base` → `processors` → `collators` → `batchers` → `distributors` → `loader`).

**Note for the resume plan (next):** `MapDistributor.stream` will need the `_dp_epoch`/`_dp_stream_pos` meta and env-var fast-forward (`data_packer_dataloader.py:239-260`), and the orchestrator will need to strip/re-attach that meta around `processor.process` and tag batches with `sample_worker_id`/`sample_epoch`/`sample_index` (`:328-358`) so `DataLoaderStateCallback` keeps working. Plan 1 intentionally omits this.
