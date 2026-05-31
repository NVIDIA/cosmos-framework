# Custom Datasets for Generator and Reasoner Training

This guide explains how to bring your own dataset into Cosmos training using
`DataPackerDataLoader` and `JointDataPackerDataLoader` — the OSS-facing data
layer that works without any internal infrastructure (no WebDataset, no
object-store credentials).

---

## Contents

1. [Overview](#overview)
2. [DataPackerDataLoader](#datapackerdataloader)
   - [Step 1 — Prepare your data source](#step-1--prepare-your-data-source)
   - [Step 2 — Write your DataPacker](#step-2--write-your-datapacker)
   - [Step 3 — Wire into an experiment config](#step-3--wire-everything-into-an-experiment-config)
   - [Key parameters](#key-parameters)
   - [Shuffle and stateful checkpoint/resume](#shuffle-and-stateful-checkpointresume)
   - [Data-parallel sharding](#data-parallel-sharding)
3. [JointDataPackerDataLoader](#jointdatapackerdataloader)
   - [When to use it](#when-to-use-it)
   - [How to wire it up](#how-to-wire-it-up)
   - [Stateful checkpoint/resume](#stateful-checkpointresume)
4. [Real-world examples](#real-world-examples)
5. [Checklist](#checklist-for-a-new-dataset)

---

## Overview

The data pipeline has two parts you control:

```
Your dataset (Dataset or IterableDataset)
        │
        ▼
┌────────────────────────────────────────────────┐
│           DataPackerDataLoader                 │
│                                                │
│  map-style Dataset (any shuffle setting):      │
│  ┌──────────────────────────────────────────┐  │
│  │    _ShuffledMapIterableDataset           │  │
│  │  • per-epoch randperm (shuffle=True)     │  │
│  │    or sequential (shuffle=False)         │  │
│  │  • DP × worker sharding                  │  │
│  │  • position metadata for stateful resume │  │
│  └──────────────────┬───────────────────────┘  │
│                     │                          │
│  IterableDataset:                              │
│  ┌──────────────────────────────────────────┐  │
│  │    _IterableWrapper                      │  │
│  │  • DP × worker sharding only             │  │
│  │  • no stateful resume                    │  │
│  └──────────────────┬───────────────────────┘  │
│                     │ raw item                 │
│  ┌──────────────────▼───────────────────────┐  │
│  │    _DataPackerIterableDataset            │  │
│  │    (subclass of PackingIterableDataset)  │  │
│  │                                          │  │
│  │  • fill pool (pool_size samples)         │  │
│  │  • greedy bin-pack within max_tokens     │  │
│  │  • cap at max_batch_size                 │  │
│  │                                          │  │
│  │  → DataPacker.sft_process_sample  ← you  │  │
│  │  → DataPacker.compute_num_tokens  ← you  │  │
│  │  → DataPacker.sft_collate_fn      ← you  │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
        │ fully-collated batch dict
        ▼
     Trainer / model.forward()
```

Key point: **all map-style datasets** (whether `shuffle=True` or `shuffle=False`)
are routed through `_ShuffledMapIterableDataset`, which attaches position
metadata to every sample. This means stateful checkpoint/resume works regardless
of whether shuffle is enabled.

---

## DataPackerDataLoader

### Step 1 — Prepare your data source

`DataPackerDataLoader` accepts either a **map-style** `torch.utils.data.Dataset`
or an **iterable-style** `torch.utils.data.IterableDataset`. Plain lists and
generators are rejected with a `TypeError`.

| Type                                      | Notes                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------- |
| `torch.utils.data.Dataset` (map-style)    | Pass directly. Supports `shuffle=True/False` and stateful checkpoint/resume.          |
| `torch.utils.data.IterableDataset`        | Pass directly. No shuffle, no stateful resume — shuffle externally if needed.         |
| HuggingFace `Dataset`                     | Is a `torch.utils.data.Dataset` subclass — pass directly, `shuffle=True` works.       |
| HuggingFace `IterableDataset` (streaming) | Is a `torch.utils.data.IterableDataset` — pass directly, use `.shuffle()` externally. |

#### Loading from HuggingFace (simplest)

```python
from cosmos_framework.data.vfm.data_packer_dataloader import load_data_source

# HuggingFace Hub dataset (downloaded, map-style)
data_source = load_data_source("liuhaotian/LLaVA-Instruct-150K", split="train")

# Dataset saved with dataset.save_to_disk()
data_source = load_data_source("/path/to/my_saved_dataset", split="train")

# Then pass with shuffle for per-epoch shuffling + stateful resume
DataPackerDataLoader(data_source=data_source, ..., shuffle=True, seed=42)
```

#### Streaming from HuggingFace (no disk space)

```python
from datasets import load_dataset

data_source = load_dataset(
    "lmms-lab/LLaVA-OneVision-Data", name="si", split="train", streaming=True
)
# shuffle before passing — IterableDataset does not support internal shuffle
data_source = data_source.shuffle(seed=42, buffer_size=10_000)
```

#### Custom map-style dataset

```python
class MyMapDataset(torch.utils.data.Dataset):
    def __len__(self): return 10_000
    def __getitem__(self, idx): return {"video": ..., "text": ...}

# Pass directly — DataPackerDataLoader handles sharding and shuffle internally
DataPackerDataLoader(data_source=MyMapDataset(), ..., shuffle=True, seed=42)
```

---

### Step 2 — Write your DataPacker

`DataPacker` is an abstract base class. Implement all three methods, then place
the class in the same experiment config file that uses it.

```python
from cosmos_framework.data.vfm.data_packer import DataPacker

class MyDataPacker(DataPacker):

    def sft_process_sample(self, item: dict) -> dict:
        """
        Convert one raw item from data_source into a training-ready sample.
        Called inside DataLoader workers — tokenization, decoding, transforms go here.
        """
        ...
        return {"input_ids": ..., "labels": ..., ...}

    def compute_num_tokens(self, sample: dict) -> int:
        """
        Return the token cost of one processed sample.
        Used by the packing engine to decide how many samples fit in a batch.
        """
        return int(sample["input_ids"].shape[0])

    def sft_collate_fn(self, samples: list[dict], max_len: int,
                       ignore_label_id: int = -100) -> dict:
        """
        Collate a list of processed samples into one batch dict.
        max_len is the longest token sequence in this batch (for padding).
        """
        ...
        return {"input_ids": ..., "labels": ..., ...}
```

> **Note on extra batch keys**: For map-style datasets, `DataPackerDataLoader`
> automatically appends `sample_worker_id`, `sample_epoch`, and `sample_index` to
> every batch dict. These are used by `DataLoaderStateCallback` for stateful
> checkpoint/resume and are transparent to the model as long as `training_step`
> accesses the batch by key (not `**kwargs` unpack).

#### Token counting for Generator models

```python
import math

def compute_num_tokens(self, sample: dict) -> int:
    tokens = 1 + len(sample.get("text_token_ids", []))
    v = sample.get("video")   # shape [C, T, H, W]
    if v is not None:
        _, T, H, W = v.shape
        latent_h = math.ceil(H / (self.spatial_compression * self.patch_spatial))
        latent_w = math.ceil(W / (self.spatial_compression * self.patch_spatial))
        latent_t = 1 + (T - 1) // self.temporal_compression
        tokens += latent_h * latent_w * latent_t + 2
    return tokens
```

Typical values: `spatial_compression=16`, `temporal_compression=4`, `patch_spatial=2`.

---

### Step 3 — Wire everything into an experiment config

```python
from cosmos_framework.utils.lazy_config import LazyCall as L, LazyDict
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader, load_data_source
from cosmos_framework.callbacks.dataloader_state import DataLoaderStateCallback
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

my_experiment = LazyDict(dict(
    defaults=[...],   # inherit model, optimizer, scheduler from a base

    trainer=dict(
        callbacks=dict(
            # Tracks per-worker (epoch, position) for checkpoint/resume.
            # Works with both shuffle=True and shuffle=False for map-style datasets.
            dataloader_state=L(DataLoaderStateCallback)(distributor_type="data_packer"),
        ),
    ),

    dataloader_train=L(DataPackerDataLoader)(
        data_source=L(load_data_source)(name="my-org/my-dataset", split="train"),
        data_packer=L(MyDataPacker)(...),
        max_tokens=16000,
        pool_size=16,
        max_batch_size=1,
        shuffle=True,            # per-epoch randperm, different order every epoch
        seed=42,                 # epoch e uses seed+e → reproducible permutations
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
    ),
    dataloader_val=None,
), flags={"allow_objects": True})

cs.store(group="experiment", package="_global_", name="my_experiment", node=my_experiment)
```

Launch:

```bash
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
    --config=cosmos_framework/configs/base/config.py -- \
    experiment=my_experiment \
    trainer.max_iter=1000
```

---

### Key parameters

| Parameter            | What it controls                                                                                                                                                                                                                                                                                                        |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_tokens`         | Token budget per batch. Packing stops when adding one more sample would exceed this. For Generator, counts video latent tokens; for Reasoner, counts `input_ids` length.                                                                                                                                                |
| `pool_size`          | Samples to buffer before bin-packing. Larger pool → better packing efficiency, more memory. Default: 16.                                                                                                                                                                                                                |
| `max_batch_size`     | Hard cap on samples per batch regardless of token budget. Use `1` for Reasoner (one image per step), `128`–`256` for action policy training.                                                                                                                                                                            |
| `shuffle`            | `True` → per-epoch `randperm` shuffle for map-style datasets (no effect on `IterableDataset`, a warning is logged). `False` → sequential, still resumable.                                                                                                                                                              |
| `seed`               | Base seed for the shuffle permutation. Epoch `e` uses `seed + e` → reproducible, different ordering every epoch. Default: `0`.                                                                                                                                                                                          |
| `name`               | Optional string that namespaces resume env vars. **Required** when multiple `DataPackerDataLoader` instances share the same process (i.e., inside `JointDataPackerDataLoader`). Each inner loader must have a unique `name` matching its key in the `dataloaders` dict. Leave empty (default) for single-loader setups. |
| `long_threshold`     | Samples with token count ≥ this are emitted as singleton batches, bypassing packing. Default: 6400.                                                                                                                                                                                                                     |
| `batching_strategy`  | `"prefer_closest"` (default) picks candidates nearest in token length. `"prefer_first"` picks the first that fits.                                                                                                                                                                                                      |
| `num_workers`        | DataLoader workers for `sft_process_sample`. Use `0` for debugging.                                                                                                                                                                                                                                                     |
| `persistent_workers` | Automatically promoted to `True` for all map-style datasets when `num_workers > 0` (required for correct resume behaviour).                                                                                                                                                                                             |

---

### Shuffle and stateful checkpoint/resume

For map-style datasets, `DataPackerDataLoader` tracks each worker's position and
resumes training from **exactly** where it left off after a checkpoint. This works
for both `shuffle=True` and `shuffle=False`.

#### How it works

1. Each epoch, a permutation is generated with `torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))` (or `list(range(n))` when `shuffle=False`).
2. Each `(dp_rank, worker_id)` pair sees a disjoint stride: `perm[stream_id :: total_streams]` where `stream_id = dp_rank * num_workers + worker_id`.
3. After each training step, `DataLoaderStateCallback` reads `sample_epoch` and `sample_index` from the batch and tracks the high-water mark per worker.
4. At checkpoint, the DCP checkpointer saves the state to `iter_XXXXXXXXX/dataloader/rank_{rank}.pkl`.
5. On resume, `load_state_dict` sets `DP_STATE_WORKER_{worker_id}_EPOCH/INDEX` env vars before workers start, and workers fast-forward past already-seen samples.

**At most `pool_size` (default 16) samples are re-processed** at each resume (they pass through `sft_process_sample` again but are trained on only once).

#### Required wiring

```python
from cosmos_framework.callbacks.dataloader_state import DataLoaderStateCallback

exp["trainer"]["callbacks"]["dataloader_state"] = L(DataLoaderStateCallback)(
    distributor_type="data_packer"
)
```

Use `ckpt_type=dcp` (the default) — not `ckpt_type=dummy` which disables all checkpointing.

#### Limitations

- **Map-style datasets only.** Stateful resume is not supported for `IterableDataset` sources.
- **`fork` start method required** (the default for Linux/CUDA). `spawn` is not supported.
- **`persistent_workers=True` required** when `num_workers > 0` (auto-enforced for all map-style datasets).

---

### Data-parallel sharding

`DataPackerDataLoader` automatically shards `data_source` across ranks **and**
DataLoader workers. Each `(dp_rank, worker_id)` pair receives a disjoint subset —
a strided slice of the (shuffled) permutation.

**If your dataset already shards internally** (like `SFTDataset`), disable its
sharding before passing it to `DataPackerDataLoader`:

```python
def get_my_dataset_no_dp(**kwargs):
    dataset = MyDataset(**kwargs)
    dataset.shard_world_size = 1   # disable internal sharding
    dataset.shard_rank = 0
    return dataset
```

**For FSDP + TP/PP**: pass `parallel_dims` so the correct DP rank is used
(global rank ≠ DP rank in these setups):

```python
DataPackerDataLoader(..., parallel_dims=parallel_dims)
```

---

## JointDataPackerDataLoader

### When to use it

`JointDataPackerDataLoader` wraps **multiple** `DataPackerDataLoader` instances
with ratio-based seeded selection. Use it when training on multiple datasets with
different modalities or formats — for example, video + action data at a 3:1 ratio.

Semantics mirror `IterativeJointDataLoader`:

- **One batch = one dataset** — samples from different datasets never share a packed batch.
- Ratios control how frequently each dataset is visited (per batch, not per sample).
- Selection is deterministic: step `i` always picks the same dataset given the same `seed`.
- Stateful checkpoint/resume: both the outer step counter (`global_id`) and each inner
  loader's per-worker position are saved and restored.

### How to wire it up

Each inner `DataPackerDataLoader` must be given a unique `name` that matches its
key in the `dataloaders` dict. The `name` namespaces the resume env vars to
prevent conflicts between concurrent loaders.

```python
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader, JointDataPackerDataLoader
from cosmos_framework.callbacks.dataloader_state import JointDataLoaderStateCallback
from cosmos_framework.utils.lazy_config import LazyCall as L

# Build the joint loader
joint_loader = JointDataPackerDataLoader(
    dataloaders={
        "video": {
            "dataloader": DataPackerDataLoader(
                data_source=MyVideoDataset(...),
                data_packer=MyVideoDataPacker(...),
                max_tokens=45056,
                shuffle=True,
                seed=0,
                name="video",          # must match the key above
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            ),
            "ratio": 3,                # video 3×, action 1×
        },
        "action": {
            "dataloader": DataPackerDataLoader(
                data_source=MyActionDataset(...),
                data_packer=MyActionDataPacker(...),
                max_tokens=999_999,
                max_batch_size=128,
                shuffle=True,
                seed=0,
                name="action",         # must match the key above
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            ),
            "ratio": 1,
        },
    },
    seed=42,   # controls outer dataset selection sequence
)

# Wire into the experiment config
exp["dataloader_train"] = joint_loader
exp["trainer"]["callbacks"]["dataloader_state"] = JointDataLoaderStateCallback(
    outer_loader=joint_loader,
    distributor_type="data_packer",
)
```

> **Reserved name**: `"global_id"` cannot be used as a dataset name — it is
> reserved by the checkpoint state format.

#### `JointDataPackerDataLoader` parameters

| Parameter     | What it controls                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `dataloaders` | Dict mapping dataset name → `{"dataloader": DataPackerDataLoader, "ratio": int}`. Entries with `ratio <= 0` are skipped. |
| `seed`        | Base seed for outer dataset selection. Step `i` uses `np.random.RandomState(seed + i)` → same sequence on every rank.    |

#### `JointDataLoaderStateCallback`

This single callback replaces the per-inner-loader `DataLoaderStateCallback`
instances. It saves:

- `global_id` — the outer step counter, which determines which dataset fires at each step on resume.
- Per-dataset, per-worker `(epoch, index)` — each inner loader's position.

All state is written to a single DCP checkpoint entry (`checkpoint_component="dataloader"`).

### Stateful checkpoint/resume

At checkpoint step `N`:

- `global_id = N` is saved.
- Each inner loader saves its per-worker `(epoch, index)` under its `name` key.

On resume:

1. `JointDataLoaderStateCallback.load_state_dict` calls `set_start_iteration(N)` on the outer loader → selection sequence resumes from step `N`.
2. Each inner `DataLoaderStateCallback.load_state_dict` sets namespaced env vars (`DP_STATE_{name}_WORKER_{id}_EPOCH/INDEX`) → workers fast-forward to the saved position.

Inner loader iterators are created lazily on the **first** `__iter__` call (not at
`__init__` time), ensuring workers fork **after** env vars have been set.

---

## Real-world examples

### Reasoner — HuggingFace image-text dataset

**File**: `cosmos_framework/configs/base/vlm/experiment/llava_ov_datapacker_experiment.py`

```
data_source:  lmms-lab/LLaVA-OneVision-Data  (streaming IterableDataset)
DataPacker:   VLMDataPacker
  sft_process_sample:  ShareGPT → OpenAI messages → Qwen3-VL processor
  compute_num_tokens:  len(input_ids)
  sft_collate_fn:      unsqueeze batch dim, keep pixel_values flat
max_batch_size: 1
max_tokens:    ~16000
shuffle:       False  (streaming IterableDataset — use .shuffle() externally)
```

### Action Policy — Robot learning (LIBERO)

**File**: `cosmos_framework/configs/base/experiment/action/posttrain_config/libero_policy_datapacker_experiment.py`

```
data_source:  LIBERODataset  (map-style Dataset, passed directly)
DataPacker:   ActionDataPacker
  sft_process_sample:  full ActionTransformPipeline (resize, tokenize, pad action)
  compute_num_tokens:  VAE video tokens + text tokens
  sft_collate_fn:      action/domain_id/sequence_plan fields + video + text
max_batch_size: 128   (token budget disabled — batch bounded by max_batch_size)
max_tokens:    999999
shuffle:       True, seed=0
```

`LIBERODataset` is a map-style `Dataset` passed directly. `shuffle=True` enables
per-epoch shuffling and stateful checkpoint/resume. This pattern (high `max_tokens`

- bounded `max_batch_size`) is standard for action policy training where you want
a fixed number of demonstrations per step.

---

## Checklist for a new dataset

### Single dataset (`DataPackerDataLoader`)

- [ ] Choose a `data_source`: map-style `Dataset` or `IterableDataset` (no plain lists/generators)
- [ ] For map-style: pass directly; use `shuffle=True, seed=<N>` for per-epoch shuffle
- [ ] For iterable: shuffle externally before passing (e.g. `.shuffle(buffer_size=N)`)
- [ ] If dataset has internal DP sharding, disable it (`shard_world_size=1`)
- [ ] Subclass `DataPacker` and implement `sft_process_sample`, `compute_num_tokens`, `sft_collate_fn`
- [ ] Choose `max_tokens` and `max_batch_size` for your modality
- [ ] Add `DataLoaderStateCallback(distributor_type="data_packer")` to the experiment's callbacks (works for both `shuffle=True` and `shuffle=False` on map-style datasets)
- [ ] Use `ckpt_type=dcp` (not `dummy`) for real checkpoint/resume
- [ ] Register in Hydra ConfigStore with `cs.store(group="experiment", ...)`
- [ ] Smoke-test with `ckpt_type=dummy trainer.max_iter=10` before a full run

### Multiple datasets (`JointDataPackerDataLoader`)

- [ ] Give each inner `DataPackerDataLoader` a unique `name` matching its key in `dataloaders`
- [ ] Set appropriate `ratio` for each dataset (controls visit frequency per batch)
- [ ] Use `JointDataLoaderStateCallback(outer_loader=joint_loader)` instead of `DataLoaderStateCallback`
- [ ] Do **not** also register standalone `DataLoaderStateCallback` for inner loaders — `JointDataLoaderStateCallback` handles all of them
- [ ] Avoid using `"global_id"` as a dataset name (reserved)
- [ ] Use `ckpt_type=dcp` for real checkpoint/resume
