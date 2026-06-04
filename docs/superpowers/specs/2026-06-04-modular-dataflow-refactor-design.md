# Modular Dataflow Refactor: DataPackerDataLoader & DataPacker

**Date:** 2026-06-04
**Status:** Design approved, pending implementation plan
**Scope:** All three goals (see below)

## Problem

The current OSS-facing training dataflow has the right *idea* (a pluggable
`DataPacker` feeding a shared packing engine) but fuses two concerns that should
be independent:

- `DataPacker` (`cosmos_framework/data/vfm/data_packer.py`) is user-pluggable for
  **sample processing** (`sft_process_sample`) and **collation** (`sft_collate_fn`),
  plus per-sample token cost (`compute_num_tokens`).
- `PackingIterableDataset` (`cosmos_framework/data/vfm/packing_iterable_dataset.py`)
  hardcodes the **batch selection strategy** — pool-based greedy bin-packing
  (`_best_fit_batch`, `_find_best_candidate_*`).
- `_DataPackerIterableDataset` (`cosmos_framework/data/vfm/data_packer_dataloader.py:265`)
  hard-wires `DataPacker` into that one engine, and also owns the
  **source/sharding/resume** logic (`_IterableWrapper`, `_ShuffledMapIterableDataset`).

Consequence: the only batching strategy a `DataPackerDataLoader` can use is
pool-based bin-packing. The selection strategy is not user-definable, and the
source-handling logic is entangled with the packing engine.

## Goals

1. **Modular dataflow + VLM migration.** Refactor so a user can independently
   define (a) how a raw item becomes a sample, (b) how samples are selected into
   a batch, (c) how a selected group is collated, and (d) how data is sharded
   across the DP group. The loader must support **both** iterable-style and
   map-style datasets. Migrate the current VLM training (`llava_ov_datapacker`)
   onto the new abstraction **behavior-preserving** (identical batches).
2. **VFM re-implementation.** Re-implement the VFM training dataflow
   (`vision_sft_nano`: `RankPartitionedDataLoader` + `PackingDataLoader` +
   `SFTDataset`, VAE-encode-in-model) on the new loader, behavior-preserving.
3. **Arbitrary user dataflows.** The four roles are public extension points; a
   user assembles a custom pipeline by subclassing the roles they need and
   reusing built-ins for the rest.

## Non-goals

- No change to the model side: VAE encoding stays inside the VFM model forward
  (`omni_mot_model.py:2721`); sequence packing and the flow-matching / CE losses
  are untouched.
- No requirement to preserve cosmos-rl `BaseDataPacker` method-name
  compatibility (the package is now independent; we borrow its pool-packing
  *design*, not its names).
- No on-disk checkpoint format change — resume stays compatible with the
  existing `DataLoaderStateCallback` / `JointDataLoaderStateCallback`.

## Architecture: four hierarchical roles

A raw item flows through four small, independently-swappable roles in a fixed
order enforced by the loader:

```
DataDistributor  (raw dataset + DP×worker sharding + shuffle + resume state)
  → RawItemProcessor  (one raw item → one sample dict)
    → SampleBatcher   (sample stream → list[sample]; owns buffer + selection strategy)
      → BatchCollator (list[sample] → batch dict for model.forward())
```

The roles have intentionally **different arities** (item→sample, stream→lists,
list→dict), which is why they are separate roles rather than a uniform "pipeline
stage" list — a uniform stage interface would be leaky, and a free-ordered stage
list would turn ordering constraints (must shard before buffering, must process
before collate) into runtime foot-guns. The fixed four-role order makes those
constraints structural.

### Role contracts (ABCs)

```python
class DataDistributor(ABC):
    @abstractmethod
    def stream(self, dp_rank: int, dp_world_size: int,
               worker_id: int, num_workers: int) -> Iterator[Any]:
        """Yield this (rank, worker)'s disjoint slice of raw items, indefinitely."""
    def state_dict(self) -> dict: return {}            # resume — optional
    def load_state_dict(self, state: dict) -> None: ...  # called before workers fork

class RawItemProcessor(ABC):
    @abstractmethod
    def process(self, item: Any) -> dict: ...

class SampleBatcher(ABC):
    @abstractmethod
    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        """Pull from `samples`, yield one list[dict] per batch."""
    def sample_size(self, sample: dict) -> int:        # overridable; packing batchers only
        raise NotImplementedError

class BatchCollator(ABC):
    @abstractmethod
    def collate(self, samples: list[dict]) -> dict: ...
```

`stream(...)` is a generator method: the loader passes in the rank/worker
coordinates (instead of the distributor digging them out of `get_worker_info()`),
and the distributor yields exactly the items that worker should see. It is the
direct generalization of today's `_IterableWrapper.__iter__` /
`_ShuffledMapIterableDataset.__iter__`.

`sample_size` lives on the `SampleBatcher` because in **both** existing paths the
size computation is already attached to the batching layer (VLM:
`VLMDataPacker.compute_num_tokens`; VFM: `PackingDataLoader._compute_num_tokens_per_sample`,
which carries config state — compression factors, patch size). It is an
overridable method (so VFM's stateful VAE formula is a clean subclass) with an
optional injected `size_fn` override for the trivial VLM `len(input_ids)` case.
Non-packing batchers (e.g. `SimpleBatcher`) never call it.

### Built-in implementations

**Distributors** (`dataflow/distributors.py`):

| Built-in | Wraps | Sharding | Shuffle | Resume | Replaces |
|---|---|---|---|---|---|
| `IterableDistributor` | iterable / `IterableDataset` | round-robin `i % total == mine` | no | no (`state_dict → {}`) | `_IterableWrapper` |
| `MapDistributor` | map-style `Dataset` | per-epoch `randperm` slice | yes | yes (env-var fast-forward) | `_ShuffledMapIterableDataset` |
| `RankPartitionedDistributor` | multiple datasets | whole ranks → datasets by ratio | per-dataset | per inner | `RankPartitionedDataLoader` |
| `MixtureDistributor` | multiple distributors | ratio-weighted sample merge into one stream | per source | per source | `PackingIterableDataset.datasets_cfg` / `_get_next_sample` |

**Batchers** (`dataflow/batchers.py`):

| Built-in | Strategy | `sample_size`? | Replaces |
|---|---|---|---|
| `SimpleBatcher` | fixed `batch_size`, pull-N | no | (new — normal torch DataLoader behavior) |
| `PoolPackingBatcher` | pool-based greedy bin-packing (reorders within buffer to minimize padding); modality segregation | yes | `PackingIterableDataset._best_fit_batch` etc. |
| `SequentialPackingBatcher` | order-preserving pull-until-budget; discards oversized | yes | VFM `PackingDataLoader.__iter__` |

**Collators** (`dataflow/collators.py`):

| Built-in | Behavior | Replaces |
|---|---|---|
| `DefaultBatchCollator` | `torch.utils.data.default_collate` (stack) | (new) |

Recipe-specific roles live with their recipes: `VLMProcessor` / `VLMCollator`
(VLM experiment dir); `SFTVideoProcessor` / `VFMListCollator` (VFM data package).

**Built-in naming convention:** built-ins use the full role word as suffix
(`VLMProcessor` *is-a* `RawItemProcessor`, `VLMCollator` *is-a* `BatchCollator`).

## Loader orchestration

`DataPackerDataLoader` keeps its name but gets new internals — a slim
orchestrator that wires the four roles in fixed order inside each worker:

```python
def __iter__(self):
    info = torch.utils.data.get_worker_info()
    worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
    raw     = self.distributor.stream(self.dp_rank, self.dp_world_size, worker_id, num_workers)
    samples = (self.processor.process(item) for item in raw)   # item → sample
    for group in self.batcher.batches(samples):                # stream → list[sample]
        yield self.collator.collate(group)                     # list → batch dict
```

Loader responsibilities (not user-pluggable), preserved from current code:

- Resolve DP coords: `parallel_dims.dp_coord` > `torch.distributed` > `(0, 1)`
  (`data_packer_dataloader.py:476-496`).
- Construct the torch `DataLoader` with `batch_size=None` (roles already yield
  collated batches) (`:520-523`).
- Enforce `persistent_workers=True` when using a stateful `MapDistributor`
  (`:468-474`).

### `batch_size` convenience sugar

The loader accepts a bare `batch_size=N`. When `batcher`/`collator` are omitted,
it auto-constructs `SimpleBatcher(N)` + `DefaultBatchCollator()`, giving exact
stock-`torch.utils.data.DataLoader` behavior (N samples per batch, stacked) while
still inheriting DP×worker sharding and (with `MapDistributor`) resume. Explicit
`batcher`/`collator` override the sugar.

### Resume threading

Preserve the existing callback contract so checkpoints stay compatible:

- On save, the callback calls `distributor.state_dict()`.
- On load, the callback calls `distributor.load_state_dict(state)` **before
  workers fork**, which (as today) sets the namespaced `DP_STATE_*` env vars the
  `MapDistributor` reads on its first `stream()` call. The env-var fast-forward
  logic (`data_packer_dataloader.py:239-254`) moves verbatim into
  `MapDistributor`.
- The on-disk DCP checkpoint format is unchanged.

## Multi-dataset joining

Three distinct join semantics exist today and map to different homes:

| Join semantic | New-design expression |
|---|---|
| sample-level mixing (homogeneous; one packing/collation) | `MixtureDistributor` — one pipeline |
| rank-level partitioning | `RankPartitionedDistributor` |
| batch-level interleaving (heterogeneous; different processor/collator per dataset) | keep `JointDataPackerDataLoader` as a slim outer wrapper composing N four-role loaders + per-loader resume routing |

Homogeneous joins dissolve into `DataDistributor` built-ins (a real
simplification — "mix multiple datasets" is just another distributor). The
heterogeneous batch-level join is a genuine higher-order concern (no single
processor/collator, per-inner-loader state routing), so `JointDataPackerDataLoader`
stays — now composing the refactored loaders.

## Migration mapping

### Goal 1 — VLM (`llava_ov_datapacker`), behavior-preserving

| Today (`llava_ov_datapacker_experiment.py`) | New role |
|---|---|
| `get_llava_ov_streaming()` → HF `IterableDataset` | `IterableDistributor(stream)` (round-robin, no resume) |
| `VLMDataPacker.sft_process_sample` | `VLMProcessor.process` (verbatim) |
| `VLMDataPacker.compute_num_tokens` → `len(input_ids)` | `PoolPackingBatcher.sample_size` (one-line override / `size_fn`) |
| pool engine (`max_tokens`, `pool_size`, `max_batch_size=1`, `long_threshold`, `apply_long_sample_halving`) | `PoolPackingBatcher(...)` |
| `VLMDataPacker.sft_collate_fn` | `VLMCollator.collate` (verbatim) |

Recipe wiring:

```python
dataloader_train = L(DataPackerDataLoader)(
    distributor = L(IterableDistributor)(L(get_llava_ov_streaming)(subset=..., split="train")),
    processor   = L(VLMProcessor)(backbone=..., processor=...),
    batcher     = L(PoolPackingBatcher)(max_tokens="${data_setting.max_tokens}",
                                        pool_size=16, max_batch_size=1, long_threshold=6400),
    collator    = L(VLMCollator)(),
)
```

Modality segregation (`_get_modality`) and `max_batch_size=1` move into
`PoolPackingBatcher` unchanged → identical data order, packing decisions, tensors.

### Goal 2 — VFM (`vision_sft_nano`), behavior-preserving

| Today (`joint_dataloader.py`, `sft_dataset.py`) | New role |
|---|---|
| `RankPartitionedDataLoader` (`:640-766`) | `RankPartitionedDistributor` |
| `SFTDataset.process_one_sample` (`sft_dataset.py:143-330`) | `SFTVideoProcessor.process` (verbatim) |
| `PackingDataLoader.__iter__` (sequential pull-until-budget, `:819-876`) | `SequentialPackingBatcher` |
| `_compute_num_tokens_per_sample` (`:325-400`, VAE formula + config) | `SequentialPackingBatcher.sample_size` (ctor args: compression factors, patch size) |
| `custom_collate_fn` (keep media as lists, `:26-89`) | `VFMListCollator` |

The new loader produces the identical batch dict (`video` as `list[Tensor]`,
`text_token_ids`, `sequence_plan`, …); VAE-encode-in-model, sequence packing, and
flow-matching loss are downstream and need **zero changes**.

This validates the abstraction: VFM uses *sequential* packing (order-preserving)
while VLM uses *pool* bin-packing (reorders to minimize padding) — genuinely
different `SampleBatcher`s satisfying one contract.

### Goal 3 — arbitrary user dataflows

Falls out for free: subclass whichever roles you need, reuse built-ins for the
rest; the loader enforces the fixed order so stages can't be misordered.

```python
DataPackerDataLoader(
    distributor = MyShardedDistributor(...),
    processor   = MyProcessor(...),
    batcher     = PoolPackingBatcher(...),   # reuse built-in
    collator    = MyCollator(...),
)
```

## File layout

```
cosmos_framework/data/vfm/
  dataflow/
    __init__.py            # re-exports the 4 ABCs + built-ins
    base.py                # DataDistributor, RawItemProcessor, SampleBatcher, BatchCollator
    distributors.py        # Iterable / Map / RankPartitioned / Mixture
    batchers.py            # Simple / PoolPacking / SequentialPacking
    collators.py           # DefaultBatchCollator
  data_packer_dataloader.py  # DataPackerDataLoader (new internals) + JointDataPackerDataLoader
```

**Deleted after migration:** `data_packer.py` (`DataPacker` ABC);
`packing_iterable_dataset.py` (logic → `PoolPackingBatcher`); private wrappers
`_IterableWrapper` / `_ShuffledMapIterableDataset` / `_DataPackerIterableDataset`.

## Testing strategy

1. **Golden-batch equality** (core safety net): fixed seed + fixed data slice →
   first N batches from old loader vs new loader, assert tensor-equality. One for
   VLM, one for VFM.
2. **Per-role unit tests:** distributors (disjoint coverage across ranks/workers;
   `MapDistributor` resume fast-forward), batchers (packing decisions, budget /
   halving, oversized discard), collators (output shapes).
3. **Resume integration test:** checkpoint mid-epoch with `MapDistributor`,
   restart, assert no duplicated/skipped samples (preserves
   `DataLoaderStateCallback` behavior).

## Implementation order

1. Build the four role ABCs + built-ins + slim orchestrator alongside existing
   code (no behavior change yet).
2. Migrate VLM → pass VLM golden test → delete `data_packer.py`,
   `packing_iterable_dataset.py`, and the private wrappers.
3. Add VFM built-ins (`RankPartitionedDistributor`, `SequentialPackingBatcher`,
   `SFTVideoProcessor`, `VFMListCollator`) → migrate VFM → pass VFM golden test.
4. Add `MixtureDistributor`; reconcile `JointDataPackerDataLoader` as the outer
   composer over refactored loaders.

## Key decisions log

- Four roles, fixed order; no `DataPacker` bundle (Option 2 — explicit
  `processor`/`batcher`/`collator`).
- `DataDistributor` fully user-pluggable (sharding + shuffle + resume).
- `sample_size` is an overridable `SampleBatcher` method (+ optional `size_fn`).
- `batch_size=N` sugar → `SimpleBatcher` + `DefaultBatchCollator`.
- Joining: `MixtureDistributor` for homogeneous; keep `JointDataPackerDataLoader`
  for heterogeneous batch-level interleaving.
- cosmos-rl name compatibility dropped; borrow design only.
- Resume/checkpoint format unchanged.
