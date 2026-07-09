# Custom Datasets for Generator and Reasoner Training

Bring your own dataset into Cosmos training with **`CosmosDataLoader`** — the
OSS-facing data layer that works without any internal infrastructure (no
WebDataset, no object-store credentials).

`CosmosDataLoader` turns any dataset into training batches by composing four
small, swappable roles. Pick a built-in for each slot, or write your own — the
loader wires them together in a fixed, safe order.

```
DataDistributor   →   RawItemProcessor   →   SampleBatcher   →   BatchCollator
(raw items, sharded     (one raw item        (a stream of         (a group of
 across DP ranks ×       → one sample         samples → groups     samples → one
 workers, shuffled,      dict)                that form a batch)   batch dict for
 resumable)                                                        model.forward)
```

- **`DataDistributor`** owns the dataset and yields *this* rank/worker's disjoint
  slice of raw items (sharding, shuffle, checkpoint/resume).
- **`RawItemProcessor`** turns one raw item into one training-ready sample dict
  (decode, tokenize, etc.).
- **`SampleBatcher`** pulls from the sample stream and decides *which* samples go
  together in a batch (fixed size, token-budget packing, …).
- **`BatchCollator`** turns a chosen group of samples into one batch dict.

Everything lives in `cosmos_framework.data.generator.dataflow`. The loader is a
`torch.utils.data.DataLoader` subclass, so it drops into existing training loops.

---

## Contents

1. [Quickstart (60 seconds)](#1-quickstart-60-seconds)
2. [The four roles](#2-the-four-roles)
3. [Recipes by use-case](#3-recipes-by-use-case)
4. [Wiring into a training recipe (Hydra)](#4-wiring-into-a-training-recipe-hydra)
5. [Checkpoint / resume](#5-checkpoint--resume)
6. [Distributed & sharding](#6-distributed--sharding)
7. [Troubleshooting / FAQ](#7-troubleshooting--faq)
8. [End-to-end worked example](#8-end-to-end-worked-example-custom-dataset--training)
9. [Real-world examples](#9-real-world-examples)
10. [Checklist for a new dataset](#10-checklist-for-a-new-dataset)

---

## 1. Quickstart (60 seconds)

"I have a map-style dataset and just want normal, shuffled, resumable batches":

```python
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader, MapDistributor, IdentityProcessor,
)

loader = CosmosDataLoader(
    distributor=MapDistributor(my_dataset, shuffle=True, seed=0),  # any torch map Dataset
    processor=IdentityProcessor(),                                 # dataset already yields samples
    batch_size=32,                                                 # sugar → SimpleBatcher + DefaultBatchCollator
)

for batch in loader:
    out = model(**batch)
```

`batch_size=N` is convenience sugar: when you don't pass an explicit `batcher`,
the loader builds a `SimpleBatcher(N)` + `DefaultBatchCollator()` (stock
`torch.utils.data` stacking). Pass an explicit `batcher`/`collator` for anything
fancier. (Passing both `batch_size=` and `batcher=` is an error.)

---

## 2. The four roles

Each role is a tiny ABC in `cosmos_framework.data.generator.dataflow.base`. Implement
the one method (plus, for distributors, optional resume hooks).

```python
class DataDistributor(ABC):
    def stream(self, dp_rank, dp_world_size, worker_id, num_workers) -> Iterator[Any]:
        """Yield this (rank, worker)'s disjoint slice of raw items, indefinitely."""
    def state_dict(self) -> dict: ...          # optional, for resume
    def load_state_dict(self, state) -> None: ...

class RawItemProcessor(ABC):
    def process(self, item) -> dict: ...        # one raw item → one sample dict

class SampleBatcher(ABC):
    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]: ...
    def sample_size(self, sample) -> int: ...   # only packing batchers need this

class BatchCollator(ABC):
    def collate(self, samples: list[dict]) -> dict: ...   # group → batch dict
```

The loader passes the rank/worker coordinates *into* `stream()` — you never read
`get_worker_info()` yourself. The fixed order (`distribute → process → batch →
collate`) is enforced by the loader, so the stages can't be misordered.

### Built-ins

| Role        | Built-in                                                                                                                                                                | Use it when                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Distributor | `IterableDistributor(iterable)`                                                                                                                                         | streaming / `IterableDataset` source (round-robin shard; **not** resumable) |
|             | `MapDistributor(dataset, seed=0, shuffle=True, name="")`                                                                                                                | map-style `Dataset` (per-epoch shuffle, slice shard, **resumable**)         |
|             | `RankPartitionedDistributor({name: {"dataset":…, "ratio":…}})`                                                                                                          | assign whole DP ranks to different datasets by ratio                        |
|             | `MixtureDistributor({name: (distributor, ratio)}, seed=0)`                                                                                                              | mix several distributors into one stream at the sample level                |
| Processor   | `IdentityProcessor()`                                                                                                                                                   | the dataset already yields finished sample dicts                            |
|             | *(write your own)*                                                                                                                                                      | decode/tokenize/transform a raw record                                      |
| Batcher     | `SimpleBatcher(batch_size, drop_last=False)`                                                                                                                            | fixed-size batches                                                          |
|             | `PoolPackingBatcher(max_tokens, pool_size=16, max_batch_size=1, long_threshold=6400, batching_strategy="prefer_closest", apply_long_sample_halving=True, size_fn=None)` | token-budget bin-packing (reorders within a pool to minimize padding)       |
|             | `SequentialPackingBatcher(max_sequence_length, …, max_samples_per_batch=None)`                                                                                          | order-preserving pack-until-budget (no reordering)                          |
| Collator    | `DefaultBatchCollator()`                                                                                                                                                | stack with `torch.utils.data.default_collate`                               |
|             | `VFMListCollator()`                                                                                                                                                     | VFM packed batches (media kept as per-sample lists)                         |

Recipe-specific roles live next to their recipes, e.g. `VLMProcessor` /
`VLMCollator` (in `configs/base/vlm/experiment/dataflow_roles.py`) and
`VideoPhy2Processor`.

---

## 3. Recipes by use-case

### Bring your own map-style dataset (shuffle + resume)

```python
class MyImageCaptionDataset(torch.utils.data.Dataset):
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]   # a plain dict

loader = CosmosDataLoader(
    distributor=MapDistributor(MyImageCaptionDataset(...), shuffle=True, seed=42),
    processor=MyProcessor(),     # turns the record into {"input_ids": …, "pixel_values": …}
    batch_size=8,
    num_workers=4,
)
```

Map-style sources are **resumable** (see §5).

### Bring your own streaming / iterable dataset

```python
hf_stream = load_dataset("some/dataset", split="train", streaming=True)
loader = CosmosDataLoader(
    distributor=IterableDistributor(hf_stream),   # round-robin shard across rank×worker
    processor=MyProcessor(),
    batch_size=8,
)
```

Iterable sources are **not** resumable (you can't random-access to fast-forward).

### Token-budget packing for variable-length sequences

```python
loader = CosmosDataLoader(
    distributor=IterableDistributor(stream),
    processor=MyProcessor(),                       # yields {"input_ids": Tensor[L], …}
    batcher=PoolPackingBatcher(max_tokens=16000, pool_size=16, max_batch_size=1),
    collator=MyCollator(),
)
```

`PoolPackingBatcher.sample_size` defaults to `len(sample["input_ids"])`; pass
`size_fn=lambda s: …` (or subclass and override `sample_size`) for a custom cost.

### Order-preserving sequence packing

```python
batcher=SequentialPackingBatcher(
    max_sequence_length=45056,
    tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4,
    patch_spatial=2,
)
```

Packs samples in stream order until the token budget is hit (no reordering).
Exactly one of `max_sequence_length` (token-budget mode) or
`max_samples_per_batch` (count-only mode) must be set.

### Mix multiple datasets by ratio (one pipeline)

```python
distributor=MixtureDistributor(
    {"webvid": (IterableDistributor(webvid), 3.0),
     "internal": (IterableDistributor(internal), 1.0)},
    seed=0,
)
```

Ratio-weighted merge into a single stream — use when the datasets share one
processor/batcher/collator (homogeneous join).

### Interleave heterogeneous pipelines (different processors/collators)

```python
from cosmos_framework.data.generator.dataflow import JointCosmosDataLoader

joint = JointCosmosDataLoader(
    dataloaders={
        "vlm": {"dataloader": vlm_loader, "ratio": 1},
        "vfm": {"dataloader": vfm_loader, "ratio": 3},
    },
    seed=42,
)
```

Each output batch comes from one selected inner `CosmosDataLoader` (ratio-weighted,
seeded). Use when the joined datasets need *different* processing — each inner
loader is a full four-role pipeline. Every yielded batch is tagged with
`"dataset_name"`. (`"global_id"` is reserved by the checkpoint state and cannot be
used as a dataset name.)

---

## 4. Wiring into a training recipe (Hydra)

Recipes build the loader with `LazyCall` so CLI overrides work:

```python
from cosmos_framework.utils.lazy_config import LazyCall as L

dataloader_train = L(CosmosDataLoader)(
    distributor=L(MapDistributor)(dataset=L(my_dataset_factory)(...), shuffle=True),
    processor=L(MyProcessor)(...),
    batcher=L(PoolPackingBatcher)(max_tokens="${data_setting.max_tokens}", max_batch_size=1),
    collator=L(MyCollator)(),
    num_workers=2,
)
```

Override from the CLI like any Hydra node, e.g.
`dataloader_train.batcher.max_tokens=8000`. See the live recipes for full examples:
`pre_exp012_llava_ov` (VLM), `videophy2_sft_nano` (videophy2),
`pre_exp012_llava_ov_mapstyle_dataloader` (map-style resumable VLM), and
`vision_sft_nano_mapstyle_dataloader` (VFM, alongside the legacy `vision_sft_nano`).

> **Structured-TOML launches.** When you launch a VLM recipe via `--sft-toml`,
> the flat `[dataloader_train]` knobs `max_samples_per_batch` and
> `max_sequence_length` are routed onto the loader's nested batcher
> (`dataloader_train.batcher.max_batch_size` and `…batcher.max_tokens`) by
> `PATH_REMAPS["vlm"]` in `configs/toml_config/toml_config_helper.py`. This only
> works for experiments whose batcher actually has those fields (e.g.
> `PoolPackingBatcher`).

---

## 5. Checkpoint / resume

Resume is handled by `CosmosDataLoaderStateCallback`:

```python
from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
cb = CosmosDataLoaderStateCallback()
```

- Use a **`MapDistributor`** source. On save, the callback records each worker's
  `(epoch, index)` from the per-batch `sample_worker_id`/`sample_epoch`/`sample_index`
  tensors the loader stamps. On load, it sets `COSMOS_DL_STATE_*` env vars *before*
  workers fork; `MapDistributor.stream` reads them and fast-forwards to the exact
  next sample — no duplicated or skipped samples.
- **Iterable** sources are not resumable (no position to fast-forward to); the
  stream restarts from the beginning.
- For multiple loaders sharing a process (e.g. inside `JointCosmosDataLoader`),
  give each a distinct `name=` so resume env vars are namespaced
  (`COSMOS_DL_STATE_{name}_WORKER_{id}_{EPOCH,INDEX}`), and use a single
  `JointCosmosDataLoaderStateCallback(outer_loader=joint_loader)`
  instead of one `CosmosDataLoaderStateCallback` per inner loader.
- Use `ckpt_type=dcp` (the default) — not `ckpt_type=dummy`, which disables all
  checkpointing. The on-disk checkpoint format is unchanged.

> **Validated:** a live save→stop→resume on `pre_exp012_llava_ov_mapstyle_dataloader`
> (8 dp ranks, `save_iter=100`) reproduces the original run's per-rank
> `input_ids` shapes exactly across the resume boundary — no duplicated or
> skipped samples on any rank.

---

## 6. Distributed & sharding

- The loader resolves the data-parallel coordinates as
  `parallel_dims.dp_coord` > `torch.distributed` > `(0, 1)`. For FSDP+TP/PP, pass
  `parallel_dims=` so sharding uses the correct DP rank (not the global rank).
- `IterableDistributor`/`MapDistributor` give each `(dp_rank, worker_id)` pair a
  disjoint, complete slice: stream `i` is taken iff
  `i % (dp_world_size × num_workers) == dp_rank × num_workers + worker_id`.
- `RankPartitionedDistributor` instead assigns *whole ranks* to datasets by ratio
  and sets `shard_world_size`/`shard_rank` on the chosen dataset (which then
  self-shards across the ranks sharing it). If your dataset already shards
  internally, disable that (`dataset.shard_world_size = 1; dataset.shard_rank = 0`)
  before handing it to a per-`(rank,worker)`-sharding distributor.
- A `MapDistributor` source with `num_workers > 0` is automatically promoted to
  `persistent_workers=True` (required for correct stateful resume). The `fork`
  start method (the Linux/CUDA default) is required; `spawn` is not supported.

---

## 7. Troubleshooting / FAQ

- **OOM on large packed batches.** `PoolPackingBatcher(apply_long_sample_halving=True)`
  (default) halves the token budget for any batch whose largest sample ≥ 1000
  tokens. Set `False` only after validating memory headroom at the literal budget.
- **`ValueError: Provide either a batcher= or a batch_size=`.** You passed neither;
  give one. Passing both is also an error.
- **`ValueError: Map-style resume cannot safely stamp a multi-sample batch …`.**
  A reordering batcher (pool packing) with `batch_size > 1` on a resumable
  `MapDistributor` can't record a gap-free resume position. Use `max_batch_size=1`
  with pool packing, a sequential (order-preserving) batcher, or an iterable
  (non-resumable) source.
- **`'int' object is not iterable` / wrong tensor shapes in the model.** Your
  `BatchCollator` is producing a different batch structure than the model expects.
  Match the structure the model consumes (for VFM, that's `VFMListCollator`, which
  keeps media as per-sample lists).
- **Oversized samples silently dropped (sequential packing).** A single sample
  larger than `max_sequence_length` is discarded with a logged error — increase
  the budget or filter upstream.
- **`num_workers` / `persistent_workers`.** `persistent_workers=True` is ignored
  (with a log) when `num_workers=0`.

---

## 8. End-to-end worked example (custom dataset → training)

A local image-caption folder, fully custom processor, normal batching:

```python
import torch
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader, MapDistributor, RawItemProcessor, DefaultBatchCollator, SimpleBatcher,
)

class ImageCaptionFolder(torch.utils.data.Dataset):
    def __init__(self, records): self.records = records          # [{"image_path":…, "caption":…}, …]
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]

class ImageCaptionProcessor(RawItemProcessor):
    def __init__(self, tokenizer, image_loader):
        self.tokenizer, self.image_loader = tokenizer, image_loader
    def process(self, item):
        return {
            "pixel_values": self.image_loader(item["image_path"]),   # Tensor[C,H,W]
            "input_ids": self.tokenizer(item["caption"]),            # Tensor[L]
        }

loader = CosmosDataLoader(
    distributor=MapDistributor(ImageCaptionFolder(records), shuffle=True, seed=0),
    processor=ImageCaptionProcessor(tokenizer, image_loader),
    batcher=SimpleBatcher(batch_size=16),
    collator=DefaultBatchCollator(),
    num_workers=4,
)

for batch in loader:                  # batch = {"pixel_values": [16,C,H,W], "input_ids": [16,L]}
    loss = model(**batch)
    loss.backward()
```

To pack variable-length captions by token budget instead of fixed size, swap the
batcher for `PoolPackingBatcher(max_tokens=…, max_batch_size=1)` and provide a
collator that pads/stacks accordingly — nothing else changes.

---

## 9. Real-world examples

### Reasoner (VLM) — HuggingFace image-text dataset, streaming

**File**: `cosmos_framework/configs/base/reasoner/experiment/llava_ov_vlm.py`
(`pre_exp012_llava_ov`)

```
distributor: IterableDistributor(get_llava_ov_streaming(...))   # lmms-lab/LLaVA-OneVision-Data
processor:   VLMProcessor   (ShareGPT → OpenAI messages → Qwen3-VL processor)
batcher:     PoolPackingBatcher(max_tokens≈16000, max_batch_size=1)
collator:    VLMCollator
```

Streaming source → **not** resumable. For a resumable variant of the same recipe,
see `llava_ov_mapstyle_dataloader_experiment.py` (`pre_exp012_llava_ov_mapstyle_dataloader`): it loads
the subset as a real map-style `Dataset` (`load_dataset(..., streaming=False)`) and
wraps it in a `MapDistributor`, so checkpoint/resume works (see §5).

### Reasoner (VLM) — local video dialog dataset

**File**: `cosmos_framework/configs/base/reasoner/experiment/videophy2_sft_nano.py`
(`videophy2_sft_nano`)

```
distributor: IterableDistributor(build_videophy2_local_dataset(...))
processor:   VideoPhy2Processor
batcher:     PoolPackingBatcher(max_tokens≈16000, max_batch_size=1)
collator:    VLMCollator
```

### Generator (VFM) — Cosmos video SFT

**File**: the `vision_sft_nano_mapstyle_dataloader` experiment (the new-loader VFM variant,
alongside the legacy `vision_sft_nano`).

```
distributor: IterableDistributor over the Cosmos video dataset
processor:   recipe processor (decode + tokenize)
batcher:     SequentialPackingBatcher(max_sequence_length=…)   # order-preserving token packing
collator:    VFMListCollator                                   # media kept as per-sample lists
```

---

## 10. Checklist for a new dataset

### Single dataset (`CosmosDataLoader`)

- [ ] Pick a **distributor**: `MapDistributor` (map-style `Dataset`, shuffle +
      **resume**) or `IterableDistributor` (streaming, not resumable).
- [ ] Write a **`RawItemProcessor`** (or use `IdentityProcessor` if your dataset
      already yields finished sample dicts).
- [ ] Pick a **batcher**: `batch_size=N` sugar, `SimpleBatcher`,
      `PoolPackingBatcher` (token-budget, reorders), or `SequentialPackingBatcher`
      (token-budget, order-preserving).
- [ ] Pick a **collator**: `DefaultBatchCollator`, `VFMListCollator`, or your own
      (must match the structure the model consumes).
- [ ] For real resume: use a `MapDistributor`, add
      `CosmosDataLoaderStateCallback()`, and
      `ckpt_type=dcp` (not `dummy`).
- [ ] For FSDP+TP/PP, pass `parallel_dims=` so the correct DP rank is used.
- [ ] Register the experiment in the Hydra ConfigStore
      (`cs.store(group="experiment", …)`).
- [ ] Smoke-test with `--dryrun` (config build) then `trainer.max_iter=10` before a
      full run.

### Multiple datasets (`JointCosmosDataLoader`)

- [ ] Build each inner pipeline as its own `CosmosDataLoader`; give each a unique
      `name=` matching its key in `dataloaders` (namespaces resume env vars).
- [ ] Set each dataset's `ratio` (controls how often it is visited, per batch).
- [ ] Use a single
      `JointCosmosDataLoaderStateCallback(outer_loader=joint_loader)`
      — do **not** also register a standalone `CosmosDataLoaderStateCallback` per inner
      loader.
- [ ] Avoid `"global_id"` as a dataset name (reserved by the checkpoint state).
- [ ] Use `ckpt_type=dcp` for real checkpoint/resume.

---

## Reference: where things live

- ABCs + built-ins: `cosmos_framework/data/generator/dataflow/` (`base.py`,
  `distributors.py`, `batchers.py`, `collators.py`, `processors.py`, `loader.py`).
- Public symbols are re-exported from `cosmos_framework.data.generator.dataflow`.
- Live recipes using the loader: `pre_exp012_llava_ov`,
  `pre_exp012_llava_ov_mapstyle_dataloader`, `videophy2_sft_nano`, and `vision_sft_nano_mapstyle_dataloader`.
