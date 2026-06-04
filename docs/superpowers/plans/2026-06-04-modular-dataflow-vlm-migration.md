# Modular Dataflow — PoolPackingBatcher + VLM Migration (Plan 2 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the pool-based bin-packing engine into a `PoolPackingBatcher`, extract the VLM sample-processing and collation into `VLMProcessor` / `VLMCollator`, wire a behavior-preserving mirror experiment on the new `CosmosDataLoader`, and validate it by golden-batch equality + a loss-curve regression run vs. the untouched `llava_ov` baseline.

**Architecture:** Builds on Plan 1's `dataflow/` package. `PoolPackingBatcher` faithfully re-homes `PackingIterableDataset`'s greedy bin-packing (`_best_fit_batch` / `_find_best_candidate_*` / `_max_tokens` / `_get_modality`) but pulls from the upstream sample iterator instead of owning dataset iterators. `VLMProcessor` and `VLMCollator` are 1:1 extractions of `VLMDataPacker.sft_process_sample` / `sft_collate_fn`. A new mirror experiment `pre_exp012_llava_ov_datapacker_v2` differs from the original **only** in dataloader wiring. The original `VLMDataPacker` / `llava_ov` experiment and all legacy dataloaders stay UNTOUCHED (living baseline).

**Tech Stack:** Python, PyTorch, Hydra `LazyCall`, pytest, `torchrun` launch wrappers.

**Spec:** `docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md`

> **HARD INVARIANT (from spec):** This refactor must not break dataloader resume or checkpoint saving. VLM uses an iterable (streaming HF) source, which is non-resumable today; the mirror must preserve that exact behavior (placeholder `(epoch=0,index=0)` stamps, model/optim/scheduler resume unaffected). It must NOT touch `DataLoaderStateCallback` / `JointDataLoaderStateCallback` or the checkpoint format. Map-style resume is a separate later plan.

**Source references (read before porting):**
- Pool engine to port: `cosmos_framework/data/vfm/packing_iterable_dataset.py:30-277`.
- VLM roles to extract: `cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_experiment.py:128-358`.

---

## File Structure

| File | Responsibility |
|---|---|
| `cosmos_framework/data/vfm/dataflow/batchers.py` (modify) | add `PoolPackingBatcher` next to `SimpleBatcher` |
| `cosmos_framework/data/vfm/dataflow/batchers_test.py` (modify) | add pool-packing tests |
| `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py` (create) | `VLMProcessor`, `VLMCollator` (recipe-specific roles) |
| `cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py` (create) | unit tests for the two roles |
| `cosmos_framework/data/vfm/dataflow/golden_vlm_test.py` (create) | old-loader vs new-loader batch equality |
| `cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_experiment.py` (create) | mirror experiment wiring the new loader |
| `examples/toml/sft_config/llava_ov_datapacker_v2.toml` (create) | mirror recipe TOML (selects the v2 experiment) |
| `examples/launch_sft_llava_ov_datapacker.sh` (create) | launch wrapper for the mirror (TOML-based, same path as baseline) |
| `cosmos_framework/data/vfm/dataflow/__init__.py` (modify) | export `PoolPackingBatcher` |

---

### Task 1: `PoolPackingBatcher` — port the bin-packing engine

**Files:**
- Modify: `cosmos_framework/data/vfm/dataflow/batchers.py`
- Modify: `cosmos_framework/data/vfm/dataflow/__init__.py`
- Modify: `cosmos_framework/data/vfm/dataflow/batchers_test.py`

- [ ] **Step 1: Write the failing test**

Append to `cosmos_framework/data/vfm/dataflow/batchers_test.py`:
```python
import torch

from cosmos_framework.data.vfm.dataflow.batchers import PoolPackingBatcher


def _txt(n_tokens, tag=0):
    # Text-modality sample (no pixel_values/pixel_values_videos key).
    return {"input_ids": torch.zeros(n_tokens, dtype=torch.long), "tag": tag}


def test_pool_emits_oversized_sample_as_singleton():
    # A sample >= long_threshold is emitted alone.
    b = PoolPackingBatcher(max_tokens=1000, pool_size=4, max_batch_size=8, long_threshold=500)
    groups = list(b.batches(iter([_txt(600), _txt(10), _txt(10)])))
    assert [len(g) for g in groups][0] == 1
    assert sum(len(g) for g in groups) == 3


def test_pool_respects_max_batch_size():
    b = PoolPackingBatcher(max_tokens=10_000, pool_size=8, max_batch_size=1, long_threshold=6400)
    groups = list(b.batches(iter([_txt(10) for _ in range(5)])))
    assert all(len(g) == 1 for g in groups)
    assert len(groups) == 5


def test_pool_packs_multiple_within_budget():
    # max_batch_size large, small samples -> they pack together under budget.
    b = PoolPackingBatcher(max_tokens=100, pool_size=8, max_batch_size=8, long_threshold=6400)
    groups = list(b.batches(iter([_txt(10) for _ in range(8)])))
    # padded cost = cur_max * k; with cur_max=10 and budget 100, up to 10 fit; pool_size caps at 8.
    assert len(groups) == 1
    assert len(groups[0]) == 8


def test_pool_sample_size_default_is_len_input_ids():
    b = PoolPackingBatcher(max_tokens=100, pool_size=4, max_batch_size=4, long_threshold=6400)
    assert b.sample_size(_txt(7)) == 7


def test_pool_sample_size_fn_override():
    b = PoolPackingBatcher(
        max_tokens=100, pool_size=4, max_batch_size=4, long_threshold=6400,
        size_fn=lambda s: 3,
    )
    assert b.sample_size(_txt(7)) == 3


def test_pool_does_not_mix_modalities():
    # An image sample (has pixel_values) must not share a batch with a text sample.
    img = {"input_ids": torch.zeros(10, dtype=torch.long), "pixel_values": torch.zeros(4, 8)}
    txt = _txt(10)
    b = PoolPackingBatcher(max_tokens=10_000, pool_size=8, max_batch_size=8, long_threshold=6400)
    groups = list(b.batches(iter([img, txt])))
    # Two separate batches: the seed's modality gates candidates.
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/batchers_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'PoolPackingBatcher'`.

- [ ] **Step 3: Write minimal implementation**

Append to `cosmos_framework/data/vfm/dataflow/batchers.py` (ported from `packing_iterable_dataset.py:30-277`, pulling from the upstream `samples` iterator instead of owning dataset iterators):
```python
from collections import deque
from enum import Enum
from typing import Callable, Optional


class _Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class PoolPackingBatcher(SampleBatcher):
    """Pool-based greedy bin-packing batcher (re-homed from PackingIterableDataset).

    Buffers ``pool_size`` samples and assembles each batch by greedily selecting
    candidates that fit within the padded token budget, never mixing modalities
    within a batch. ``sample_size`` defaults to ``len(sample["input_ids"])``;
    pass ``size_fn`` to override, or subclass and override the method.

    Parameters mirror the legacy engine: ``max_tokens``, ``pool_size``,
    ``max_batch_size`` (0/None = no cap), ``long_threshold`` (samples this large
    are emitted as singletons), ``batching_strategy`` ("prefer_closest" |
    "prefer_first"), ``apply_long_sample_halving`` (halve budget when the largest
    sample >= 1000 tokens — memory safety).
    """

    def __init__(
        self,
        max_tokens: int,
        pool_size: int = 16,
        max_batch_size: int = 1,
        long_threshold: int = 6400,
        batching_strategy: str = "prefer_closest",
        apply_long_sample_halving: bool = True,
        size_fn: Optional[Callable[[dict], int]] = None,
    ):
        assert batching_strategy in ("prefer_first", "prefer_closest"), (
            f"batching_strategy must be 'prefer_first' or 'prefer_closest', got {batching_strategy!r}"
        )
        self.max_tokens = max_tokens
        self.pool_size = pool_size
        self.max_batch_size = max_batch_size
        self.long_threshold = long_threshold
        self.batching_strategy = batching_strategy
        self.apply_long_sample_halving = apply_long_sample_halving
        self._size_fn = size_fn

    # --- size ---------------------------------------------------------------
    def sample_size(self, sample: dict) -> int:
        if self._size_fn is not None:
            return self._size_fn(sample)
        return int(sample["input_ids"].shape[0])

    # --- public API ---------------------------------------------------------
    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        pool: deque[dict] = deque()
        src = iter(samples)
        exhausted = False
        while True:
            # Fill the pool.
            while not exhausted and len(pool) < self.pool_size:
                try:
                    pool.append(next(src))
                except StopIteration:
                    exhausted = True
            if not pool:
                return
            yield self._best_fit_batch(pool)

    # --- internals (ported verbatim, parameterized by `pool`) ---------------
    def _max_tokens(self, cur_max: int) -> int:
        if not self.apply_long_sample_halving:
            return self.max_tokens
        if cur_max < 1000:
            return self.max_tokens
        return self.max_tokens // 2

    @staticmethod
    def _get_modality(sample: dict) -> "_Modality":
        if "pixel_values" in sample:
            return _Modality.IMAGE
        elif "pixel_values_videos" in sample:
            return _Modality.VIDEO
        return _Modality.TEXT

    @staticmethod
    def _padded_cost(cur_max: int, k: int) -> int:
        return cur_max * k

    def _best_fit_batch(self, pool: deque) -> list[dict]:
        seed = pool.popleft()
        seed_modality = self._get_modality(seed)
        L0 = self.sample_size(seed)
        if L0 >= self.long_threshold or L0 >= self._max_tokens(L0):
            return [seed]
        chosen = [seed]
        cur_max = L0
        while pool:
            if self.max_batch_size and len(chosen) >= self.max_batch_size:
                break
            best_idx = self._find_best_candidate(pool, cur_max, len(chosen), seed_modality)
            if best_idx is None:
                break
            cand = self._remove_from_pool(pool, best_idx)
            chosen.append(cand)
            cur_max = max(cur_max, self.sample_size(cand))
        return chosen

    def _find_best_candidate(self, pool, cur_max, num_chosen, seed_modality):
        if self.batching_strategy == "prefer_first":
            return self._find_best_candidate_prefer_first(pool, cur_max, num_chosen, seed_modality)
        return self._find_best_candidate_prefer_closest(pool, cur_max, num_chosen, seed_modality)

    def _find_best_candidate_prefer_first(self, pool, cur_max, num_chosen, seed_modality):
        best_idx = None
        best_new_tokens = None
        for idx, cand in enumerate(pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.sample_size(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                if best_new_tokens is None or new_tokens < best_new_tokens:
                    best_new_tokens = new_tokens
                    best_idx = idx
        return best_idx

    def _find_best_candidate_prefer_closest(self, pool, cur_max, num_chosen, seed_modality):
        best_idx = None
        best_new_tokens = None
        smallest_length_diff = None
        for idx, cand in enumerate(pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.sample_size(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                length_diff = abs(L - cur_max)
                if (
                    best_new_tokens is None
                    or new_tokens < best_new_tokens
                    or (new_tokens == best_new_tokens and length_diff < smallest_length_diff)
                ):
                    best_new_tokens = new_tokens
                    best_idx = idx
                    smallest_length_diff = length_diff
        return best_idx

    @staticmethod
    def _remove_from_pool(pool: deque, idx: int) -> dict:
        if idx == 0:
            return pool.popleft()
        elif idx == len(pool) - 1:
            return pool.pop()
        else:
            pool.rotate(-idx)
            item = pool.popleft()
            pool.rotate(idx)
            return item
```

Add to `__init__.py`:
```python
from cosmos_framework.data.vfm.dataflow.batchers import PoolPackingBatcher, SimpleBatcher
```
(replace the existing `SimpleBatcher`-only import) and add `"PoolPackingBatcher",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/batchers_test.py -v`
Expected: PASS (SimpleBatcher tests + 6 PoolPacking tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/batchers.py \
        cosmos_framework/data/vfm/dataflow/batchers_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add PoolPackingBatcher (port of PackingIterableDataset engine)"
```

---

### Task 2: `VLMProcessor` + `VLMCollator` (extract from VLMDataPacker)

**Files:**
- Create: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py`
- Test: `cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py`

These are 1:1 extractions of `VLMDataPacker.sft_process_sample` / `sft_collate_fn`
(`llava_ov_datapacker_experiment.py:214-278`). `VLMProcessor` holds the processor
and the ShareGPT→OpenAI helpers; `VLMCollator` does the `max_batch_size=1`
collation including the resume-meta stamps (zeros for the streaming source).

- [ ] **Step 1: Write the failing test**

`cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for VLMProcessor / VLMCollator extracted from VLMDataPacker."""

from __future__ import annotations

import torch

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor


class _FakeProcessor:
    """Minimal stand-in for the Qwen3-VL processor used by VLMProcessor."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # 6 tokens; pretend tokens 3-5 are the assistant turn.
        return {
            "input_ids": torch.arange(6),
            "pixel_values": torch.zeros(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

    def add_assistant_tokens_mask(self, input_ids):
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[3:] = True
        return mask


def _item():
    return {
        "image": None,  # no image branch exercised here
        "conversations": [
            {"from": "human", "value": "hi <image>"},
            {"from": "gpt", "value": "hello"},
        ],
    }


def test_vlmprocessor_builds_input_ids_and_masked_labels():
    p = VLMProcessor(processor=_FakeProcessor(), ignore_index=-100)
    s = p.process(_item())
    assert s["input_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    # non-assistant tokens (0-2) masked to -100; assistant (3-5) kept.
    assert s["labels"].tolist() == [-100, -100, -100, 3, 4, 5]
    assert "pixel_values" in s and "image_grid_thw" in s


def test_vlmcollator_adds_batch_dim_and_resume_meta():
    p = VLMProcessor(processor=_FakeProcessor(), ignore_index=-100)
    s = p.process(_item())
    batch = VLMCollator().collate([s])
    assert batch["input_ids"].shape == (1, 6)
    assert batch["labels"].shape == (1, 6)
    # pixel_values / image_grid_thw stay flat.
    assert batch["pixel_values"].shape == (4, 8)
    assert batch["image_grid_thw"].shape == (1, 3)
    # resume meta present (zeros for streaming source).
    assert batch["sample_epoch"].tolist() == [0]
    assert batch["sample_index"].tolist() == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py -v`
Expected: FAIL — `ModuleNotFoundError: ... dataflow_roles`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VLM dataflow roles (RawItemProcessor + BatchCollator) extracted 1:1 from
VLMDataPacker (llava_ov_datapacker_experiment.py). Behavior-preserving."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.data.vfm.dataflow.base import BatchCollator, RawItemProcessor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD


class VLMProcessor(RawItemProcessor):
    """ShareGPT image+conversation record -> VLM training tensors."""

    def __init__(self, processor: Any, ignore_index: int = IGNORE_INDEX) -> None:
        # `processor` is a built Qwen3-VL processor (has apply_chat_template).
        self._processor = processor
        self._ignore_index = ignore_index

    @staticmethod
    def _decode_image(image: Any) -> Any:
        if isinstance(image, dict):
            import io

            from PIL import Image

            raw = image.get("bytes")
            if raw:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            path = image.get("path")
            if path:
                return Image.open(path).convert("RGB")
            return None
        return image

    def _sharegpt_to_openai(self, item: dict) -> list[dict]:
        conversations = item.get("conversations", [])
        image = self._decode_image(item.get("image"))
        messages: list[dict] = []
        image_inserted = False
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text = turn["value"].replace("<image>", "").strip()
            if role == "user" and not image_inserted and image is not None:
                content: Any = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ]
                image_inserted = True
            else:
                content = text
            messages.append({"role": role, "content": content})
        return messages

    def process(self, item: dict) -> dict:
        messages = self._sharegpt_to_openai(item)
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids)
        labels = input_ids.clone()
        labels[~token_mask] = self._ignore_index
        result: dict = {"input_ids": input_ids, "labels": labels}
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result


class VLMCollator(BatchCollator):
    """max_batch_size=1 collation: batch-dim the sequence tensors, keep vision
    tensors flat, stamp resume meta (zeros — streaming source has no position)."""

    def collate(self, samples: list[dict]) -> dict:
        assert len(samples) == 1, f"VLMCollator expects max_batch_size=1, got {len(samples)}"
        s = samples[0]
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        batch: dict = {
            "input_ids": s["input_ids"].unsqueeze(0),
            "labels": s["labels"].unsqueeze(0),
            "sample_worker_id": torch.tensor([worker_id]),
            "sample_epoch": torch.tensor([0]),
            "sample_index": torch.tensor([0]),
        }
        if "attention_mask" in s and s["attention_mask"] is not None:
            batch["attention_mask"] = s["attention_mask"].unsqueeze(0)
        for key in (
            "pixel_values", "pixel_values_videos", "image_grid_thw",
            "video_grid_thw", "second_per_grid_ts",
        ):
            if key in s and s[key] is not None:
                batch[key] = s[key]
        return batch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/configs/base/vlm/experiment/dataflow_roles.py \
        cosmos_framework/configs/base/vlm/experiment/dataflow_roles_test.py
git commit -m "feat(vlm): extract VLMProcessor/VLMCollator from VLMDataPacker"
```

---

### Task 3: Golden-batch equality test (old loader vs new loader)

**Files:**
- Create: `cosmos_framework/data/vfm/dataflow/golden_vlm_test.py`

Drives a fixed in-memory dataset through BOTH the legacy `DataPackerDataLoader`
(`data_packer_dataloader.DataPackerDataLoader` + `VLMDataPacker`) and the new
`dataflow.CosmosDataLoader` (`IterableDistributor` + `VLMProcessor` +
`PoolPackingBatcher` + `VLMCollator`), then asserts the first N batches are
tensor-identical. `num_workers=0`, fixed `random.seed`.

- [ ] **Step 1: Write the failing test**

`cosmos_framework/data/vfm/dataflow/golden_vlm_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Golden-batch equality: legacy DataPackerDataLoader+VLMDataPacker vs the new
four-role dataflow loader on the same fixed source must yield identical batches."""

from __future__ import annotations

import random

import torch

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.configs.base.vlm.experiment.llava_ov_datapacker_experiment import VLMDataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader as LegacyLoader
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader as NewLoader,
    IterableDistributor,
    PoolPackingBatcher,
)


class _FakeProcessor:
    """Deterministic processor: token length grows with the conversation length."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        n = 4 + sum(len(m["content"]) for m in messages if isinstance(m["content"], str))
        return {"input_ids": torch.arange(n)}

    def add_assistant_tokens_mask(self, input_ids):
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[len(input_ids) // 2 :] = True
        return mask


class _FixedIterable(torch.utils.data.IterableDataset):
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        yield from self._items


def _items(k):
    out = []
    for i in range(k):
        out.append({
            "image": None,
            "conversations": [
                {"from": "human", "value": "q" * (i % 5 + 1)},
                {"from": "gpt", "value": "a" * (i % 3 + 1)},
            ],
        })
    return out


def _drain(loader, n):
    it = iter(loader)
    return [next(it) for _ in range(n)]


def test_vlm_golden_batches_match():
    proc = _FakeProcessor()
    items = _items(40)

    random.seed(0)
    legacy = LegacyLoader(
        data_source=_FixedIterable(list(items)),
        data_packer=VLMDataPacker(tokenizer_config=proc, max_seq_len=200),
        max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400,
        num_workers=0,
    )
    random.seed(0)
    new = NewLoader(
        distributor=IterableDistributor(list(items)),
        processor=VLMProcessor(processor=proc),
        batcher=PoolPackingBatcher(max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400),
        collator=VLMCollator(),
        num_workers=0,
    )

    a = _drain(legacy, 10)
    b = _drain(new, 10)
    for ba, bb in zip(a, b):
        assert ba.keys() == bb.keys(), (ba.keys(), bb.keys())
        for k in ba:
            assert torch.equal(ba[k], bb[k]), f"mismatch at key {k}"
```

- [ ] **Step 2: Run test to verify it fails (or surfaces a real diff)**

Run: `pytest cosmos_framework/data/vfm/dataflow/golden_vlm_test.py -v`
Expected: initially may FAIL if any extraction diverged. Investigate any mismatch — the goal is to make legacy and new identical. Common causes: a missing key in `VLMCollator`, or `PoolPackingBatcher` candidate selection differing from the legacy engine.

- [ ] **Step 3: Fix divergences in the already-built code**

No new module here — if the test fails, fix `PoolPackingBatcher` (Task 1) or `VLMProcessor`/`VLMCollator` (Task 2) until batches match. The legacy engine is the source of truth.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/golden_vlm_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/golden_vlm_test.py
git commit -m "test(dataflow): golden-batch equality VLM legacy vs new loader"
```

---

### Task 4: Mirror experiment `pre_exp012_llava_ov_datapacker_v2`

**Files:**
- Create: `cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_experiment.py`

A copy of `pre_exp012_llava_ov_datapacker` (`llava_ov_datapacker_experiment.py:286-369`)
that swaps the legacy `data_packer=`/`DataPackerDataLoader` wiring for the
four-role `dataflow.CosmosDataLoader`. Everything else (model, optimizer, checkpoint,
defaults) is identical. Imports the existing `get_llava_ov_streaming` and
`build_processor` from the original module to avoid duplication.

- [ ] **Step 1: Write the failing test**

Add a registration smoke test at `cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_test.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The v2 mirror experiment must register in the Hydra ConfigStore."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore


def test_v2_experiment_is_registered():
    import cosmos_framework.configs.base.vlm.experiment.llava_ov_datapacker_v2_experiment  # noqa: F401

    repo = ConfigStore.instance().repo
    assert "experiment" in repo
    names = set(repo["experiment"].keys())
    assert "pre_exp012_llava_ov_datapacker_v2.yaml" in names, sorted(names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_test.py -v`
Expected: FAIL — `ModuleNotFoundError: ... llava_ov_datapacker_v2_experiment`.

- [ ] **Step 3: Write minimal implementation**

`cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_experiment.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mirror of pre_exp012_llava_ov_datapacker on the four-role dataflow loader.

Differs from the original ONLY in dataloader wiring (DataDistributor +
RawItemProcessor + SampleBatcher + BatchCollator via the new
cosmos_framework.data.vfm.dataflow.CosmosDataLoader). Used as the
loss-curve regression mirror — see the spec's Testing strategy.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    IterableDistributor,
    PoolPackingBatcher,
)
from cosmos_framework.data.vfm.processors import build_processor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX
from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.configs.base.vlm.experiment.llava_ov_datapacker_experiment import (
    get_llava_ov_streaming,
)

cs = ConfigStore.instance()


pre_exp012_llava_ov_datapacker_v2 = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "basic_log"]},
            "_self_",
        ],
        job=dict(
            name="pre_exp012_llava_ov_datapacker_v2_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            group="vlm_llava_ov_demo",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=10,
            logging_iter=1,
            run_validation=False,
        ),
        optimizer=dict(
            lr=1e-5,
            fused=True,
        ),
        model=dict(
            config=dict(
                freeze=dict(trainable_params=[".*"]),
                parallelism=dict(
                    data_parallel_shard_degree=4,
                    data_parallel_replicate_degree=-1,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=100000,
            load_from_object_store=dict(enabled=False, credentials="", bucket=""),
            save_to_object_store=dict(enabled=False, credentials="", bucket=""),
        ),
        dataloader_train=L(CosmosDataLoader)(
            distributor=L(IterableDistributor)(
                iterable=L(get_llava_ov_streaming)(subset="ai2d(gpt4v)", split="train"),
            ),
            processor=L(VLMProcessor)(
                processor=L(build_processor)(
                    tokenizer_type="${model.config.policy.backbone.model_name}",
                    config_variant="hf",
                ),
                ignore_index=IGNORE_INDEX,
            ),
            batcher=L(PoolPackingBatcher)(
                max_tokens=16000,
                pool_size=16,
                max_batch_size=1,
                long_threshold=6400,
            ),
            collator=L(VLMCollator)(),
            num_workers=2,
        ),
    )
)

cs.store(group="experiment", package="_global_", name="pre_exp012_llava_ov_datapacker_v2", node=pre_exp012_llava_ov_datapacker_v2)
```

Note: the original recipe builds `dataloader_val` from the parent defaults; this
mirror sets `run_validation=False`, so no val loader is required. If a later run
enables validation, add a `dataloader_val` mirror the same way.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_experiment.py \
        cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_v2_test.py
git commit -m "feat(vlm): add mirror experiment pre_exp012_llava_ov_datapacker_v2 on dataflow loader"
```

---

### Task 5: Mirror TOML + launch wrapper + loss-curve regression run

**Files:**
- Create: `examples/toml/sft_config/llava_ov_datapacker_v2.toml`
- Create: `examples/launch_sft_llava_ov_datapacker.sh`

The baseline `examples/launch_sft_llava_ov.sh` sources `_sft_launcher_common.sh`
and drives `cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/llava_ov_datapacker.toml`
(verified: `_sft_launcher_common.sh:86-90`). To keep the mirror launch path
**identical** to the baseline, the mirror is a TOML whose `[job].experiment`
points at the v2 ConfigStore node from Task 4; `load_experiment_from_toml` then
resolves it and overlays the TOML scalars. The launch wrapper just sets
`TOML_FILE` + `TAIL_OVERRIDES` and sources the same common launcher.

- [ ] **Step 1: Write the mirror TOML**

`examples/toml/sft_config/llava_ov_datapacker_v2.toml` (mirror of
`llava_ov_datapacker.toml`, differing only in `experiment`, `name`, and the
wandb/iters knobs for the regression run):
```toml
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Dataflow-loader mirror of llava_ov_datapacker for loss-curve regression.
# Selects the v2 experiment (new four-role CosmosDataLoader); everything
# else matches llava_ov_datapacker.toml.

[job]
task         = "vlm"
experiment   = "pre_exp012_llava_ov_datapacker_v2"
project      = "cosmos_oss_alignment"
group        = "vlm_llava_ov_demo"
name         = "pre_exp012_llava_ov_datapacker_v2"
wandb_mode   = "online"

[model]
attn_implementation = "cosmos"
precision           = "bfloat16"

[model.backbone]
model_name = "Qwen/Qwen3-VL-8B-Instruct"

[model.ema]
enabled         = false
rate            = 0.1
iteration_shift = 0

[model.parallelism]
data_parallel_shard_degree      = 8
data_parallel_replicate_degree  = -1
context_parallel_shard_degree   = 1
cfg_parallel_shard_degree       = 1

[model.compile]
enabled         = false
compile_dynamic = true

[model.activation_checkpointing]
mode               = "full"
save_ops_regex     = ["fmha"]
preserve_rng_state = true
determinism_check  = "default"

[optimizer]
betas        = [0.9, 0.95]
eps          = 1.0e-8
fused        = true
lr           = 1.0e-5
weight_decay = 0.1

[scheduler]
cycle_lengths      = [500]
f_max              = [1.0]
f_min              = [0.5]
f_start            = [0.05]
verbosity_interval = 0
warm_up_steps      = [1000]

[trainer]
distributed_parallelism = "fsdp"
grad_accum_iter         = 1
logging_iter            = 1
max_iter                = 500

[trainer.callbacks.compile_tokenizer]
compile_after_iterations = 3
enabled                  = false

[trainer.callbacks.grad_clip]
clip_norm    = 1.0
force_finite = false

[checkpoint]
keys_to_skip_loading = []
load_path            = "???"
save_iter            = 100

[dataloader_train]
max_samples_per_batch = 1
```

Note: `data_setting.max_tokens` is supplied at launch via `TAIL_OVERRIDES` exactly
as the baseline does (see the baseline launch wrapper's tail args).

- [ ] **Step 2: Write the launch wrapper**

`examples/launch_sft_llava_ov_datapacker.sh`:
```bash
#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Launch the dataflow-loader mirror of the VLM llava_ov recipe
# (pre_exp012_llava_ov_datapacker_v2) for loss-curve regression vs the baseline
# launched by launch_sft_llava_ov.sh. Same launch path (sft-toml + common launcher).
#
# Usage (inside the training container):
#   bash examples/launch_sft_llava_ov_datapacker.sh

TOML_FILE="examples/toml/sft_config/llava_ov_datapacker_v2.toml"

# Match the baseline's model + max_tokens tail; add a fresh run name + iters.
: "${RUN_NAME:=llava_ov_datapacker_v2_$(date +%Y%m%d_%H%M%S)}"
TAIL_OVERRIDES=(
    "model.config.policy.backbone.model_name=${BACKBONE:?Set BACKBONE to the Qwen3-VL backbone path/name}"
    "data_setting.max_tokens=16000"
    "trainer.logging_iter=1"
    "trainer.max_iter=500"
    "job.project=cosmos_oss_alignment"
    "job.wandb_mode=online"
    "job.name=${RUN_NAME}"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
```

- [ ] **Step 3: Stage env + run BOTH baseline and mirror**

Use the `cosmos3-run-env` skill for the `/tmp/run_*.sh` env-prep block (venv
activate, `LD_LIBRARY_PATH=`, `WANDB_API_KEY`) and the `slurm-node` skill for the
node. The baseline reuses the existing `launch_sft_llava_ov.sh` with the same
regression overrides via `TAIL_OVERRIDES`.

Baseline (`/tmp/run_llava_baseline.sh`):
```bash
set -uo pipefail
cd /lustre/fsw/portfolios/sw/projects/sw_aidot/users/maoshengl/nvda/cosmos-framework
source .venv/bin/activate && export LD_LIBRARY_PATH=
export WANDB_API_KEY=<from cosmos3-run-env skill>
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOML_FILE="examples/toml/sft_config/llava_ov_datapacker.toml"
export RUN_NAME="llava_ov_baseline_$(date +%Y%m%d_%H%M%S)"
export BACKBONE=<qwen3-vl path>
# Append regression overrides to the baseline launch by exporting TAIL_OVERRIDES
# before sourcing the common launcher (the baseline wrapper sets TOML_FILE; we
# drive the common launcher directly here to inject TAIL_OVERRIDES):
export TAIL_OVERRIDES=(
  "model.config.policy.backbone.model_name=${BACKBONE}"
  "data_setting.max_tokens=16000"
  "trainer.logging_iter=1" "trainer.max_iter=500"
  "job.project=cosmos_oss_alignment" "job.wandb_mode=online" "job.name=${RUN_NAME}"
)
source examples/_sft_launcher_common.sh
```

Mirror (`/tmp/run_llava_v2.sh`):
```bash
set -uo pipefail
cd /lustre/fsw/portfolios/sw/projects/sw_aidot/users/maoshengl/nvda/cosmos-framework
source .venv/bin/activate && export LD_LIBRARY_PATH=
export WANDB_API_KEY=<from cosmos3-run-env skill>
export CUDA_VISIBLE_DEVICES=4,5,6,7
export BACKBONE=<qwen3-vl path>
bash examples/launch_sft_llava_ov_datapacker.sh
```

Run via:
```bash
JOB_ID=$(squeue -u maoshengl --format="%i" -h | head -1)
srun --overlap --jobid "$JOB_ID" --container-name bob_echo_dev bash -s < /tmp/run_llava_baseline.sh &
srun --overlap --jobid "$JOB_ID" --container-name bob_echo_dev bash -s < /tmp/run_llava_v2.sh &
wait
```

For the exact-equality check, prepend `PYTHONHASHSEED=42` and add `--deterministic`
to the `torchrun ... train` line (set via an env knob or a one-off edit), accepting
slower runs.

- [ ] **Step 4: Verify equivalence**

Compare the two runs' `loss` series in wandb (`cosmos_oss_alignment`). Acceptance:
under `--deterministic` the per-iteration loss matches to within float noise
(ideally identical) for all 500 iters. Record the two run URLs in the PR.

- [ ] **Step 5: Commit**

```bash
git add examples/toml/sft_config/llava_ov_datapacker_v2.toml \
        examples/launch_sft_llava_ov_datapacker.sh
git commit -m "chore(vlm): mirror TOML + launch wrapper for llava_ov dataflow regression"
```

---

## Self-Review

**Spec coverage (Plan 2 scope):**
- `PoolPackingBatcher` (spec built-ins; "Goal 1 VLM mapping") → Task 1. ✅
- `VLMProcessor` / `VLMCollator` extraction (spec "Goal 1 VLM mapping", "Processor placement: VLM = real processor") → Task 2. ✅
- Golden-batch equality (spec Testing tier 2) → Task 3. ✅
- Mirror experiment, originals untouched (spec Implementation Order phase 2; "living baseline") → Task 4. ✅
- Loss-curve regression with the exact run config (spec Testing tier 3: launch scripts, `logging_iter=1`, `max_iter=500`, wandb `cosmos_oss_alignment`, fresh names, `cosmos3-run-env`) → Task 5. ✅
- HARD INVARIANT resume/saving: VLM source is iterable/non-resumable; mirror preserves placeholder zero-stamps and does not touch the state callbacks or checkpoint format (called out in preamble; Task 2 `VLMCollator` keeps the zero stamps). ✅
- OUT of Plan 2: `SequentialPackingBatcher`, `RankPartitioned`/`Mixture` distributors, map-style resume + callback integration, videophy2/VFM migrations, docs, deletion.

**Placeholder scan:** Run config has intentional `<from cosmos3-run-env skill>` / `<qwen3-vl path>` placeholders in Task 5 *operational* commands (secrets/paths resolved at run time via the named skills, not committed) — these are not code placeholders. All code steps contain complete code.

**Type consistency:** `PoolPackingBatcher(max_tokens, pool_size, max_batch_size, long_threshold, batching_strategy, apply_long_sample_halving, size_fn)` used identically in Tasks 1, 3, 4. `VLMProcessor(processor, ignore_index)` and `VLMCollator()` consistent across Tasks 2, 3, 4. New `CosmosDataLoader(distributor, processor, batcher, collator, num_workers)` matches Plan 1 Task 7's signature. `IterableDistributor(iterable=...)` keyword matches Plan 1 Task 5.

**Launch path (resolved):** The baseline `launch_sft_llava_ov.sh` drives `train.py --sft-toml=...` via `_sft_launcher_common.sh` (verified `:86-90`). Task 5 mirrors that exactly: a v2 TOML whose `[job].experiment` selects the Task-4 ConfigStore node, launched through the same common launcher with `TAIL_OVERRIDES`. No `--config`/`experiment=` path is used.

**Note on `[job]` TOML keys:** the v2 TOML sets `project`/`name`/`wandb_mode` under `[job]`; confirm these map through `SFTExperimentConfig` the same way the baseline `llava_ov_datapacker.toml` `[job]` block does (it does today). The regression overrides (`trainer.*`, `job.*`, `data_setting.max_tokens`) are applied as Hydra tail args, which win over the TOML.
