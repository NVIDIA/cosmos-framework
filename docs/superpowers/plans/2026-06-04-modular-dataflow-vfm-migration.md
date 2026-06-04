# Modular Dataflow — VFM Migration + Mixture/Joint (Plan 5 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the VFM `vision_sft_nano` recipe onto the four-role `CosmosDataLoader` (sequential packing, rank-partitioned distribution, list collation), add `MixtureDistributor` + `JointCosmosDataLoader`, and validate by golden-batch equality + loss-curve regression vs. the untouched baseline.

**Architecture:** VFM's legacy stack is `SFTDataset` (IterableDataset, self-shards via `shard_world_size/shard_rank/shard_id`) → `RankPartitionedDataLoader` (rank→dataset allocation) → `PackingDataLoader` (sequential pull-until-budget packing) → `custom_collate_fn` (media kept as lists). Migration: `RankPartitionedDistributor` (sets shard attrs + iterates the dataset) + `IdentityProcessor` (SFTDataset already processes — the spec's accepted hollow-Processor deviation) + `SequentialPackingBatcher` (sequential packing + VFM VAE token formula as `sample_size`) + `VFMListCollator` (= `custom_collate_fn`). Plus `MixtureDistributor` (homogeneous ratio mixing) and `JointCosmosDataLoader` (heterogeneous batch-interleave, replacing `JointDataPackerDataLoader`).

**Tech Stack:** Python, PyTorch, Hydra `LazyCall`, pytest, `--sft-toml` launch.

**Spec:** `docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md` ("Goal 2 — VFM", "Multi-dataset joining"). Builds on Plans 1–3.

> **HARD INVARIANT:** Resume + saving must not break. VFM resume goes through the `RankPartitionedDistributor` + the existing callbacks; the golden + a resume check guard it. The model side (VAE-encode-in-model, sequence packing, flow-matching loss) is downstream and unchanged.

**Source references (verbatim to port):**

- `cosmos_framework/data/vfm/joint_dataloader.py:16-110` — `custom_collate_fn` + `_aggregate_worker_timing` + `_BATCH_TIMING_KEYS` → `VFMListCollator`.
- `cosmos_framework/data/vfm/joint_dataloader.py:325-400` — `_compute_num_tokens_per_sample` (VFM VAE token formula) → `SequentialPackingBatcher.sample_size`.
- `cosmos_framework/data/vfm/joint_dataloader.py:819-876` — `PackingDataLoader.__iter__` (sequential packing + lookahead + oversized discard) → `SequentialPackingBatcher.batches`.
- `cosmos_framework/data/vfm/joint_dataloader.py:660-757` — `RankPartitionedDataLoader.__init__` (rank allocation + shard-setting) → `RankPartitionedDistributor`.
- `cosmos_framework/data/vfm/joint_dataloader.py:176-184` — `_get_next_sample` weighted `random.choices` → `MixtureDistributor`.
- `cosmos_framework/data/vfm/data_packer_dataloader.py:526-624` — `JointDataPackerDataLoader` → `JointCosmosDataLoader`.
- `cosmos_framework/data/vfm/local_datasets/sft_dataset.py:64,117-120,332-425` — `SFTDataset` shard attrs + `__iter__` yields processed samples.
- `cosmos_framework/configs/base/experiment/sft/vision_sft_nano.py:217-270` — dataloader wiring.
- `examples/toml/sft_config/vision_sft_nano.toml`, `examples/launch_sft_vision_nano.sh`.

---

## File Structure

| File                                                                          | Change                                                 |
| ----------------------------------------------------------------------------- | ------------------------------------------------------ |
| `cosmos_framework/data/vfm/dataflow/collators.py`                             | add `VFMListCollator`                                  |
| `cosmos_framework/data/vfm/dataflow/batchers.py`                              | add `SequentialPackingBatcher`                         |
| `cosmos_framework/data/vfm/dataflow/distributors.py`                          | add `RankPartitionedDistributor`, `MixtureDistributor` |
| `cosmos_framework/data/vfm/dataflow/loader.py`                                | add `JointCosmosDataLoader`                            |
| `cosmos_framework/data/vfm/dataflow/__init__.py`                              | export the four new symbols                            |
| `*_test.py` co-located                                                        | unit tests per symbol                                  |
| `cosmos_framework/data/vfm/dataflow/golden_vfm_test.py` (create)              | legacy-vs-new VFM batch equality                       |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_nano_v2.py` (create) | mirror experiment                                      |
| `examples/toml/sft_config/vision_sft_nano_v2.toml` (create)                   | mirror TOML                                            |
| `examples/launch_sft_vision_nano_datapacker.sh` (create)                      | launch wrapper                                         |

---

### Task 1: `VFMListCollator` (port `custom_collate_fn`)

**Files:**

- Modify: `cosmos_framework/data/vfm/dataflow/collators.py`, `__init__.py`
- Modify: `cosmos_framework/data/vfm/dataflow/collators_test.py`

- [ ] **Step 1: Write the failing test**

Append to `collators_test.py`:

```python
from cosmos_framework.data.vfm.dataflow.collators import VFMListCollator


def test_vfm_list_collator_keeps_media_as_lists_and_stacks_scalars():
    s1 = {"video": torch.zeros(3, 4, 8, 8), "text_token_ids": torch.arange(5), "domain_id": 0}
    s2 = {"video": torch.zeros(3, 2, 8, 8), "text_token_ids": torch.arange(7), "domain_id": 1}
    out = VFMListCollator().collate([s1, s2])
    assert isinstance(out["video"], list) and len(out["video"]) == 2
    assert isinstance(out["text_token_ids"], list) and len(out["text_token_ids"]) == 2
    # domain_id is in list_collate_keys -> stays a list
    assert out["domain_id"] == [0, 1]


def test_vfm_list_collator_preserves_sparse_sound_none():
    s1 = {"video": torch.zeros(3, 1, 8, 8), "sound": torch.zeros(2)}
    s2 = {"video": torch.zeros(3, 1, 8, 8), "sound": None}
    out = VFMListCollator().collate([s1, s2])
    assert out["sound"][1] is None and out["sound"][0] is not None   # 1:1 alignment kept


def test_vfm_list_collator_drops_optional_key_missing_in_some():
    s1 = {"video": torch.zeros(3, 1, 8, 8), "extra_meta": 5}
    s2 = {"video": torch.zeros(3, 1, 8, 8)}  # no extra_meta
    out = VFMListCollator().collate([s1, s2])
    assert "extra_meta" not in out   # optional non-sparse key dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/collators_test.py -k vfm -v`
Expected: FAIL — `VFMListCollator` not defined.

- [ ] **Step 3: Implement (port `custom_collate_fn`)**

Append to `collators.py`. **Copy verbatim** the module-level constants
(`_TIMING_KEYS`, `_BATCH_TIMING_KEYS`), the `custom_collate_fn` body, and
`_aggregate_worker_timing` from `joint_dataloader.py:16-110`, then wrap as a
`BatchCollator`:

```python
import torch
from torch.utils.data import default_collate

# --- COPY VERBATIM from joint_dataloader.py:16-110 ---
_TIMING_KEYS = {"_sample_time", "_aug_time", "_pre_aug_time", "_aug_step_times"}
_BATCH_TIMING_KEYS = {
    "_worker_batch_time", "_worker_aug_time", "_worker_io_time",
    "_worker_aug_step_times", "_worker_id",
}

def _vfm_collate(batch):
    ...  # paste custom_collate_fn body verbatim (list_collate_keys, sparse_data_keys,
         # union-of-keys, optional-key drop, _aggregate_worker_timing call)

def _aggregate_worker_timing(samples):
    ...  # paste verbatim
# --- end verbatim ---

class VFMListCollator(BatchCollator):
    """custom_collate_fn as a BatchCollator: media kept as lists, sparse `sound`
    None placeholders preserved 1:1 with sequence_plan, optional keys dropped,
    per-worker timing aggregated. Behavior-identical to the legacy collate_fn."""

    def collate(self, samples: list[dict]) -> dict:
        return _vfm_collate(samples)
```

Export `VFMListCollator` in `__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/collators_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/collators.py \
        cosmos_framework/data/vfm/dataflow/collators_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add VFMListCollator (port of custom_collate_fn)"
```

---

### Task 2: `SequentialPackingBatcher`

**Files:**

- Modify: `cosmos_framework/data/vfm/dataflow/batchers.py`, `__init__.py`, `batchers_test.py`

Port `PackingDataLoader.__iter__` (`joint_dataloader.py:819-876`): pull samples in
order, accumulate until `max_sequence_length`, discard oversized when the batch is
empty, otherwise carry skipped samples to the next batch (lookahead). `sample_size`
ports `_compute_num_tokens_per_sample` (`:325-400`, the VAE token formula).

- [ ] **Step 1: Write the failing test**

Append to `batchers_test.py`:

```python
from cosmos_framework.data.vfm.dataflow.batchers import SequentialPackingBatcher


def _vid(text_len, t=1, h=64, w=64):
    # SFTDataset-shaped single sample: text_token_ids + video [C,T,H,W].
    return {"text_token_ids": torch.arange(text_len), "video": torch.zeros(3, t, h, w)}


def test_sequential_size_uses_vae_formula():
    b = SequentialPackingBatcher(
        max_sequence_length=100000,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
    )
    # text(=5) + 1 (eos) + vision: latent_h=64//(16*2)=2, latent_w=2, latent_t=1 -> 2*2*1+2=6
    assert b.sample_size(_vid(5)) == 5 + 1 + 6


def test_sequential_packs_in_order_until_budget():
    b = SequentialPackingBatcher(
        max_sequence_length=40,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
    )
    # each ~12 tokens (text5+1+6); budget 40 -> 3 fit (36) then 4th would hit >=40.
    groups = list(b.batches(iter([_vid(5) for _ in range(7)])))
    assert [len(g) for g in groups][0] == 3


def test_sequential_discards_oversized_when_batch_empty():
    b = SequentialPackingBatcher(
        max_sequence_length=10,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
    )
    # a huge sample alone exceeds budget and is discarded (logged); a small one follows.
    big = _vid(50)
    small = _vid(1)
    groups = list(b.batches(iter([big, small])))
    # big discarded -> only small emitted (size 2+? <10)
    flat = [s for g in groups for s in g]
    assert all(s["text_token_ids"].shape[0] == 1 for s in flat)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/batchers_test.py -k sequential -v`
Expected: FAIL — `SequentialPackingBatcher` not defined.

- [ ] **Step 3: Implement**

Append to `batchers.py`:

```python
from collections import deque as _deque

from cosmos_framework.utils import log


class SequentialPackingBatcher(SampleBatcher):
    """Order-preserving pull-until-budget packing (port of PackingDataLoader).

    Accumulates samples in stream order until `max_sequence_length` (or
    `max_samples_per_batch`); a sample that would overflow a non-empty batch is
    carried to the next batch (bounded by `lookahead_limit`); a sample that alone
    exceeds the budget is discarded with a log. `sample_size` ports the VFM VAE
    token formula and needs the tokenizer compression factors + patch size.
    """

    def __init__(
        self,
        max_sequence_length: int,
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_samples_per_batch: int | None = None,
        lookahead_limit: int = 10,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
    ):
        self.max_sequence_length = max_sequence_length
        self.tokenizer_spatial_compression_factor = tokenizer_spatial_compression_factor
        self.tokenizer_temporal_compression_factor = tokenizer_temporal_compression_factor
        self.patch_spatial = patch_spatial
        self.max_samples_per_batch = max_samples_per_batch
        self.lookahead_limit = lookahead_limit
        self.sound_latent_fps = sound_latent_fps
        self.audio_sample_rate = audio_sample_rate

    def sample_size(self, sample: dict) -> int:
        # --- PORT VERBATIM from joint_dataloader.py:325-400 ---
        # _compute_num_tokens_per_sample(self, data_batch). Operates on a SINGLE
        # SFTDataset sample. IMPORTANT adaptation: text_token_ids on a raw sample
        # may be a 1-D tensor or a list[int]; handle both (the original received
        # a collated list). Validate the exact count against the legacy loader in
        # the golden test (Task 6) — that is the source of truth.
        ...

    def batches(self, samples):
        src = iter(samples)
        carry: _deque = _deque()   # samples skipped (lookahead) from prior round
        exhausted = False
        while True:
            current_len = 0
            num_samples = 0
            group: list[dict] = []
            skipped: _deque = _deque()
            lookahead = 0
            # drain carry first, then the source
            def _next():
                if carry:
                    return carry.popleft()
                return next(src)
            while True:
                if self.max_samples_per_batch is not None and num_samples >= self.max_samples_per_batch:
                    break
                if group and lookahead >= self.lookahead_limit:
                    break
                try:
                    s = _next()
                except StopIteration:
                    exhausted = True
                    break
                n = self.sample_size(s)
                if current_len + n >= self.max_sequence_length:
                    if not group:
                        log.error(
                            f"SequentialPackingBatcher: discarding oversized sample with {n} "
                            f"tokens (max_sequence_length={self.max_sequence_length})",
                            rank0_only=False,
                        )
                        continue
                    skipped.append(s)
                    lookahead += 1
                    continue
                current_len += n
                num_samples += 1
                group.append(s)
            # carry skipped samples (front) to the next batch, preserving order
            for s in reversed(skipped):
                carry.appendleft(s)
            if group:
                yield group
            if exhausted and not carry:
                return
```

Replace the `sample_size` `...` with the verbatim port of
`_compute_num_tokens_per_sample`. Export `SequentialPackingBatcher`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/batchers_test.py -v`
Expected: PASS. If the formula's `text_token_ids` handling trips on a 1-D tensor
vs list, fix `sample_size` (the golden test in Task 6 is the ultimate check).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/batchers.py \
        cosmos_framework/data/vfm/dataflow/batchers_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add SequentialPackingBatcher (port of PackingDataLoader packing)"
```

---

### Task 3: `RankPartitionedDistributor`

**Files:**

- Modify: `cosmos_framework/data/vfm/dataflow/distributors.py`, `__init__.py`, `distributors_test.py`

Ports `RankPartitionedDataLoader.__init__` rank-allocation (`:707-744`) and
shard-setting (`:751-753`). `stream(dp_rank, dp_world_size, worker_id, num_workers)`
allocates this rank to one dataset, instantiates it (cached), sets
`shard_world_size`/`shard_rank`/`shard_id`, and `yield from` it (the dataset
self-shards across workers via `worker_info`).

- [ ] **Step 1: Write the failing test**

Append to `distributors_test.py`:

```python
from cosmos_framework.data.vfm.dataflow.distributors import RankPartitionedDistributor


class _ShardAwareDS(torch.utils.data.IterableDataset):
    """Records the shard attrs the distributor sets, then yields a few items."""
    def __init__(self, tag):
        self.tag = tag
        self.shard_world_size = None
        self.shard_rank = None
        self.shard_id = None
    def __iter__(self):
        yield {"tag": self.tag, "sw": self.shard_world_size, "sr": self.shard_rank, "sid": self.shard_id}


def test_rank_partition_allocates_single_dataset_and_sets_shards():
    d = RankPartitionedDistributor({
        "video": {"dataset": _ShardAwareDS("video"), "ratio": 3},
        "image": {"dataset": _ShardAwareDS("image"), "ratio": 1},
    })
    # world=4, ratios 3:1 -> ranks 0-2 video (shard_world_size=3), rank 3 image.
    r0 = next(d.stream(dp_rank=0, dp_world_size=4, worker_id=0, num_workers=1))
    assert r0["tag"] == "video" and r0["sw"] == 3 and r0["sr"] == 0
    r3 = next(RankPartitionedDistributor({
        "video": {"dataset": _ShardAwareDS("video"), "ratio": 3},
        "image": {"dataset": _ShardAwareDS("image"), "ratio": 1},
    }).stream(dp_rank=3, dp_world_size=4, worker_id=0, num_workers=1))
    assert r3["tag"] == "image" and r3["sw"] == 1 and r3["sr"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -k rank_partition -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**

Append to `distributors.py`:

```python
from cosmos_framework.utils.lazy_config import instantiate


class RankPartitionedDistributor(DataDistributor):
    """Allocate whole DP ranks to datasets by ratio; the chosen dataset self-shards.
    Ports RankPartitionedDataLoader (joint_dataloader.py:660-757) minus the inner
    torch DataLoader (CosmosDataLoader owns workers/collation)."""

    def __init__(self, datasets: dict[str, dict]):
        self._datasets_cfg = datasets
        self._cached = None  # (dataset_instance,) for this rank, set on first stream

    def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
        if self._cached is None:
            ds = self._allocate_and_build(dp_rank, dp_world_size)
            self._cached = ds
        yield from iter(self._cached)

    def _allocate_and_build(self, rank, world_size):
        # --- PORT VERBATIM the allocation from joint_dataloader.py:680-744 ---
        # Build names/dataset_configs/ratios (skip ratio<=0); compute ideal,
        # floor allocations (max(1,int)), distribute remainder by largest
        # remainder, handle deficit; find this rank's dataset idx + shard_rank.
        names, dataset_configs, ratios = [], [], []
        for name, cfg in self._datasets_cfg.items():
            if cfg["ratio"] <= 0:
                continue
            names.append(name); dataset_configs.append(cfg["dataset"]); ratios.append(cfg["ratio"])
        assert world_size >= len(names)
        total = sum(ratios)
        ideal = [r / total * world_size for r in ratios]
        allocations = [max(1, int(q)) for q in ideal]
        remaining = world_size - sum(allocations)
        if remaining > 0:
            order = sorted(range(len(ratios)), key=lambda i: ideal[i] - allocations[i], reverse=True)
            for j in range(remaining):
                allocations[order[j]] += 1
        elif remaining < 0:
            deficit = -remaining
            while deficit > 0:
                best = max((i for i in range(len(allocations)) if allocations[i] > 1),
                           key=lambda i: (allocations[i] - ideal[i], allocations[i]))
                allocations[best] -= 1; deficit -= 1
        cumulative = 0; idx = -1
        for i, a in enumerate(allocations):
            if rank < cumulative + a:
                idx = i; break
            cumulative += a
        assert idx >= 0
        shard_rank = rank - cumulative
        shard_world_size = allocations[idx]
        # --- end port ---
        ds = instantiate(dataset_configs[idx]) if not hasattr(dataset_configs[idx], "__iter__") or _is_lazy(dataset_configs[idx]) else dataset_configs[idx]
        ds.shard_world_size = shard_world_size
        ds.shard_rank = shard_rank
        ds.shard_id = idx
        return ds
```

Add a small `_is_lazy` helper (or simply always `instantiate` when the cfg is a
LazyCall/dict and use as-is when it's already an IterableDataset — mirror
`PackingIterableDataset.__init__`'s isinstance check at
`packing_iterable_dataset.py:110-114`). Keep it consistent with how the codebase
detects "already-built" datasets. Export `RankPartitionedDistributor`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/distributors.py \
        cosmos_framework/data/vfm/dataflow/distributors_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add RankPartitionedDistributor (port of RankPartitionedDataLoader)"
```

---

### Task 4: `MixtureDistributor`

**Files:**

- Modify: `cosmos_framework/data/vfm/dataflow/distributors.py`, `__init__.py`, `distributors_test.py`

Ratio-weighted merge of multiple `DataDistributor`s into one stream (ports the
weighted `random.choices` selection of `PackingIterableDataset._get_next_sample`,
`:176-184`). Homogeneous joining: one pipeline downstream.

- [ ] **Step 1: Write the failing test**

Append to `distributors_test.py`:

```python
import random as _random

from cosmos_framework.data.vfm.dataflow.distributors import MixtureDistributor


def test_mixture_draws_from_both_by_ratio():
    a = IterableDistributor([{"src": "a", "i": i} for i in range(1000)])
    b = IterableDistributor([{"src": "b", "i": i} for i in range(1000)])
    m = MixtureDistributor({"a": (a, 3.0), "b": (b, 1.0)}, seed=0)
    it = m.stream(0, 1, 0, 1)
    draws = [next(it)["src"] for _ in range(400)]
    frac_a = draws.count("a") / len(draws)
    assert 0.6 < frac_a < 0.85   # ~0.75 by ratio, allow sampling noise


def test_mixture_is_seeded_reproducible():
    def build():
        a = IterableDistributor([{"i": i} for i in range(1000)])
        b = IterableDistributor([{"i": -i} for i in range(1000)])
        return MixtureDistributor({"a": (a, 1.0), "b": (b, 1.0)}, seed=42)
    s1 = [next(build().stream(0, 1, 0, 1))["i"] for _ in range(1)]
    it1 = build().stream(0, 1, 0, 1); it2 = build().stream(0, 1, 0, 1)
    assert [next(it1)["i"] for _ in range(50)] == [next(it2)["i"] for _ in range(50)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -k mixture -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**

Append to `distributors.py`:

```python
import random as _random_mod


class MixtureDistributor(DataDistributor):
    """Ratio-weighted merge of multiple distributors into one stream (homogeneous
    join). Generalizes PackingIterableDataset's weighted _get_next_sample."""

    def __init__(self, sources: dict[str, tuple], seed: int = 0):
        # sources: {name: (DataDistributor, ratio_float)}
        self._names = list(sources.keys())
        self._dists = [sources[n][0] for n in self._names]
        self._ratios = [float(sources[n][1]) for n in self._names]
        self._seed = seed

    def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
        rng = _random_mod.Random(self._seed + dp_rank * 100003 + worker_id)
        iters = [d.stream(dp_rank, dp_world_size, worker_id, num_workers) for d in self._dists]
        while True:
            idx = rng.choices(range(len(iters)), weights=self._ratios, k=1)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = self._dists[idx].stream(dp_rank, dp_world_size, worker_id, num_workers)
                yield next(iters[idx])
```

Export `MixtureDistributor`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/distributors_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/distributors.py \
        cosmos_framework/data/vfm/dataflow/distributors_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add MixtureDistributor (ratio-weighted homogeneous join)"
```

---

### Task 5: `JointCosmosDataLoader`

**Files:**

- Modify: `cosmos_framework/data/vfm/dataflow/loader.py`, `__init__.py`
- Create: `cosmos_framework/data/vfm/dataflow/joint_loader_test.py`

Port `JointDataPackerDataLoader` (`data_packer_dataloader.py:526-624`) renamed to
`JointCosmosDataLoader`, composing `CosmosDataLoader`s. Keep the SAME public
surface the existing `JointDataLoaderStateCallback` depends on (`_names`,
`_global_id`, `set_start_iteration`, the `"dataset_name"` batch tag) so resume
works unchanged.

- [ ] **Step 1: Write the failing test**

`joint_loader_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""JointCosmosDataLoader: ratio-based batch interleave + state-callback surface."""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    IterableDistributor,
    JointCosmosDataLoader,
    SimpleBatcher,
)


def _loader(tag):
    return CosmosDataLoader(
        distributor=IterableDistributor([{"x": torch.tensor([float(i)]), "tag": tag} for i in range(1000)]),
        processor=IdentityProcessor(),
        batcher=SimpleBatcher(batch_size=1),
        num_workers=0,
    )


def test_joint_tags_batches_and_interleaves_by_ratio():
    j = JointCosmosDataLoader(
        {"a": {"dataloader": _loader("a"), "ratio": 3}, "b": {"dataloader": _loader("b"), "ratio": 1}},
        seed=0,
    )
    it = iter(j)
    names = [next(it)["dataset_name"] for _ in range(400)]
    assert set(names) == {"a", "b"}
    assert 0.6 < names.count("a") / len(names) < 0.85


def test_joint_exposes_callback_surface():
    j = JointCosmosDataLoader({"a": {"dataloader": _loader("a"), "ratio": 1}}, seed=0)
    assert j._names == ["a"]
    assert hasattr(j, "set_start_iteration") and hasattr(j, "_global_id")
    j.set_start_iteration(5)
    assert j._global_id == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/data/vfm/dataflow/joint_loader_test.py -v`
Expected: FAIL — `JointCosmosDataLoader` not defined.

- [ ] **Step 3: Implement (port + rename)**

Append to `loader.py` the `JointDataPackerDataLoader` class body verbatim from
`data_packer_dataloader.py:526-624`, renamed to `JointCosmosDataLoader`, with the
type hints referring to `CosmosDataLoader`. Keep `_names`, `_loaders`, `_probs`,
`_global_id`, `set_start_iteration`, lazy `_iterators`, and the `"dataset_name"`
batch tag exactly. Export `JointCosmosDataLoader`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/data/vfm/dataflow/joint_loader_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/loader.py \
        cosmos_framework/data/vfm/dataflow/joint_loader_test.py \
        cosmos_framework/data/vfm/dataflow/__init__.py
git commit -m "feat(dataflow): add JointCosmosDataLoader (port of JointDataPackerDataLoader)"
```

---

### Task 6: VFM golden-batch + mirror experiment + regression

**Files:**

- Create: `cosmos_framework/data/vfm/dataflow/golden_vfm_test.py`
- Create: `cosmos_framework/configs/base/experiment/sft/vision_sft_nano_v2.py`
- Create: `examples/toml/sft_config/vision_sft_nano_v2.toml`
- Create: `examples/launch_sft_vision_nano_datapacker.sh`

- [ ] **Step 1: Golden-batch equality test (legacy stack vs new stack)**

`golden_vfm_test.py`: build a small fixed map-style stub dataset that yields
SFTDataset-shaped samples (`video`, `text_token_ids`, `sequence_plan`,
`image_size`, ...). Drive BOTH:

- legacy: `PackingDataLoader(dataloader=RankPartitionedDataLoader(datasets={"video": {dataset: stub, ratio: 1}}, batch_size=1), max_sequence_length=..., tokenizer_spatial_compression_factor=16, tokenizer_temporal_compression_factor=4, patch_spatial=2)` (run under `torch.distributed` single-process init, or stub `shard_world_size/shard_rank`).
- new: `CosmosDataLoader(distributor=RankPartitionedDistributor({"video": {dataset: stub, ratio: 1}}), processor=IdentityProcessor(), batcher=SequentialPackingBatcher(max_sequence_length=..., tokenizer_spatial_compression_factor=16, tokenizer_temporal_compression_factor=4, patch_spatial=2), collator=VFMListCollator())`.
Assert the first N packed batches are equal (compare per-key: list lengths +
tensor equality element-wise). The token formula correctness is enforced here.

```python
# Sketch — fill in the stub dataset to emit deterministic SFTDataset-shaped samples.
# Compare: same number of samples per packed batch, same per-sample tensors,
# same list keys (video/text_token_ids/sequence_plan/image_size).
```

Run: `pytest cosmos_framework/data/vfm/dataflow/golden_vfm_test.py -v` and fix
`SequentialPackingBatcher.sample_size` / `VFMListCollator` until identical.

- [ ] **Step 2: Mirror experiment**

`vision_sft_nano_v2.py`: copy `vision_sft_nano.py` (`:50-275`), replacing only the
`dataloader_train` block with the four-role wiring:

```python
dataloader_train=L(CosmosDataLoader)(
    distributor=L(RankPartitionedDistributor)(
        datasets=dict(video=dict(ratio=1, dataset=L(get_sft_dataset)( ... same args ... ))),
    ),
    processor=L(IdentityProcessor)(),
    batcher=L(SequentialPackingBatcher)(
        max_sequence_length=45056,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=None,
        sound_latent_fps=0,
        audio_sample_rate=48000,
    ),
    collator=L(VFMListCollator)(),
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
),
```

Keep the `get_sft_dataset(...)` args identical to the original (`:235-268`).
Register as `vision_sft_nano_v2` in the ConfigStore. Add a registration smoke test.

- [ ] **Step 3: Mirror TOML + launch wrapper**

`examples/toml/sft_config/vision_sft_nano_v2.toml`: copy `vision_sft_nano.toml`,
set `[job].experiment="vision_sft_nano_v2"`, `[job].name` likewise,
`[job].project="cosmos_oss_alignment"`, `[job].wandb_mode="online"`,
`[trainer].logging_iter=1`, `[trainer].max_iter=500`.

`examples/launch_sft_vision_nano_datapacker.sh`:

```bash
#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
TOML_FILE="examples/toml/sft_config/vision_sft_nano_v2.toml"
: "${DATASET_PATH:=examples/data/bridge-v2-subset-synthetic-captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'
: "${RUN_NAME:=vision_sft_nano_datapacker_v2_$(date +%Y%m%d_%H%M%S)}"
TAIL_OVERRIDES=(
    "trainer.logging_iter=1" "trainer.max_iter=500"
    "job.project=cosmos_oss_alignment" "job.wandb_mode=online" "job.name=${RUN_NAME}"
)
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
```

- [ ] **Step 4: Run baseline + mirror regression**

Baseline: `bash examples/launch_sft_vision_nano.sh` + same `TAIL_OVERRIDES` +
`RUN_NAME=vision_sft_nano_baseline_...`, GPUs 0-3. Mirror:
`bash examples/launch_sft_vision_nano_datapacker.sh`, GPUs 4-7. Env via
`cosmos3-run-env` (DATASET_PATH, WAN_VAE_PATH, BASE_CHECKPOINT_PATH=Cosmos3-Nano,
WANDB_API_KEY), node via `slurm-node`. Compare `loss` in `cosmos_oss_alignment`;
under `--deterministic` curves must match. Record run URLs.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/data/vfm/dataflow/golden_vfm_test.py \
        cosmos_framework/configs/base/experiment/sft/vision_sft_nano_v2.py \
        examples/toml/sft_config/vision_sft_nano_v2.toml \
        examples/launch_sft_vision_nano_datapacker.sh
git commit -m "feat(vfm): vision_sft_nano_v2 mirror + golden test + regression launch"
```

---

## Self-Review

**Spec coverage:** Goal 2 VFM (RankPartitioned + Sequential + IdentityProcessor + VFMListCollator) → Tasks 1–3, 6. `MixtureDistributor` + `JointCosmosDataLoader` (spec "Multi-dataset joining", Impl phase 4) → Tasks 4–5. Golden + regression (spec Testing) → Task 6. HARD INVARIANT: VFMListCollator preserves sparse-key/optional-key/timing behavior (Task 1 tests); resume via existing callbacks + JointCosmosDataLoader keeps `_names`/`_global_id`/`set_start_iteration` (Task 5 test).

**Placeholder scan:** Intentional cite-and-port markers (each with exact source lines + the required adaptation): `custom_collate_fn` (Task 1), `_compute_num_tokens_per_sample` (Task 2 `sample_size`), `RankPartitionedDataLoader` allocation (Task 3), `JointDataPackerDataLoader` body (Task 5). Task 6 Step 1 is a sketch requiring a deterministic SFTDataset-shaped stub — the golden equality is the correctness gate. These are not vague placeholders: each names the exact function/lines to copy and the one adaptation needed.

**Type consistency:** `SequentialPackingBatcher(max_sequence_length, tokenizer_spatial_compression_factor, tokenizer_temporal_compression_factor, patch_spatial, max_samples_per_batch, lookahead_limit, sound_latent_fps, audio_sample_rate)` consistent Tasks 2, 6. `RankPartitionedDistributor({name: {dataset, ratio}})` consistent Tasks 3, 6. `MixtureDistributor({name: (distributor, ratio)}, seed)` Task 4. `JointCosmosDataLoader({name: {dataloader, ratio}}, seed)` Task 5. `VFMListCollator()` Tasks 1, 6. `CosmosDataLoader(... persistent_workers, prefetch_factor ...)` matches Plan 1/3.

**Risk callouts for the executor:** (1) `SequentialPackingBatcher.sample_size` must handle SFTDataset's single-sample `text_token_ids` shape (1-D tensor or list) — golden test enforces. (2) The legacy golden comparison needs `torch.distributed` single-process init (RankPartitionedDataLoader reads global rank/world) — initialize a 1-process group or stub `shard_world_size/shard_rank` on the dataset. (3) `RankPartitionedDistributor` "already-built vs lazy" dataset detection should mirror `packing_iterable_dataset.py:110-114`.
