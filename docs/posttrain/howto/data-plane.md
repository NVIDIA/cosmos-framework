# Ray/NIXL pooled data plane

**Come here when driver payloads are the bottleneck, not the model.** The opt-in `ray_nixl_pool`
backend keeps large trajectory fields in actor-owned pages instead of moving them through the
driver. It is off by default; the ordinary `bulk_store` path stays in place unless you select it.

Requires `grid.backend=ray` and an RDT-enabled Ray build. Used by
[`../guides/grpo.md`](../guides/grpo.md) and [`../guides/opd.md`](../guides/opd.md) topologies that
move large multimodal fields.

The opt-in `ray_nixl_pool` backend keeps large trajectory fields out of driver payloads. Ray actors
own persistent source and receive pages; dictionaries and trajectories carry `PoolSlice` descriptors.
The ordinary `bulk_store` path remains the default.

## Configuration

The backend requires `grid.backend=ray` and an RDT-enabled Ray build with the required
`ray.experimental` NIXL APIs.

```text
rl.data_pool.data_plane=ray_nixl_pool
rl.data_pool.page_bytes=33554432
rl.data_pool.cuda_capacity_bytes=1073741824
rl.data_pool.cpu_capacity_bytes=536870912
rl.data_pool.cuda_cache_bytes=536870912
rl.data_pool.cpu_cache_bytes=536870912
rl.data_pool.cpu_min_bytes=33554432
```

Capacities must be positive multiples of `page_bytes`. CUDA tensors always use the pool. CPU tensors
smaller than `cpu_min_bytes` remain on the ordinary codec path. Actors that Ray assigned a GPU
initialize both tiers. CPU-placed actors initialize only CPU storage, never create a CUDA context, and
fail clearly if asked to materialize a CUDA slice. Ray leaves `CUDA_VISIBLE_DEVICES` untouched for a
zero-GPU worker, so the tier decision keys on the Ray GPU assignment, not on what torch can see.

Data and weight transports are independent. `rl.data_pool` selects sample movement;
`grid.gpu_placement` and `rl.weight_sync_*` select policy-to-rollout weight movement.
Both GRPO and OPD pass the selected pool settings to every Ray actor; this includes OPD's frozen
teacher, whose large outputs can therefore remain pooled through the shared online loop.

## Flow and ownership

1. A producer actor creates one lease while packing a method result.
2. `ProducerDataPool` packs fields into producer-owned CUDA or CPU pages. Every field starts on a
   256-byte boundary so that both consumer paths can reinterpret its bytes as the original dtype in
   place; scalars are stored as one element. Fields larger than one page receive a dedicated
   contiguous extent. At publication every used page becomes its own Ray ObjectRef, one `ray.put`
   with the NIXL transport per page, so consumers fetch pages selectively. Only stock Ray
   `ray.experimental` APIs are used.
3. The payload carries `PoolSlice` metadata and one Ray page reference, not the tensor value.
4. A consumer actor receives each page into a registered slot of its persistent `ConsumerDataPool`
   and returns an owned copy of the field bytes. Slots are landing zones only: a later miss may evict
   and rewrite a slot while earlier results are still live, so a result never aliases slot storage.
   Repeated fields on the same page hit the receive cache and pay only the copy.
5. When producer and consumer share an actor, `RayStoreMover` returns a direct source-page view and
   records its provenance; it does not perform a NIXL transfer or restage that view.
6. After every synchronous GRPO/OPD step finishes all trainer reads, the driver frees upstream Ray
   results, clears nested page references, and asks every registered actor to release all live
   producer leases. Actors without a producer pool return immediately. Each producer's publication
   registry is the source of truth, so leases are released even when filtering removes every
   trajectory that referenced them. Producers free published RDT refs and wait before reusing pages;
   the hot path does not force cyclic garbage collection.
   Draws that GRPO dynamic sampling drained are covered by the same barrier: the replay host
   releases their `BulkRef` leases at egress without fetching (`ExperienceBuffer.
   lease_releasing_output_methods`), and the driver drops the result handle once that returns.
7. Held-out validation performs the same release after every rollout batch, on both the scored and
   the failed path, so a multi-batch validation pass never holds more than one batch of producer
   pages and initial validation cannot exhaust the pool before the first training step.

`owner_id`, `lease_id`, pool tier, page id, and generation form the page identity. Generation prevents
a reused page from aliasing stale cache state.

## Scope and failure behavior

- Pool sizes are fixed per actor for a run. Allocation fails loudly when no suitable extent remains.
- Oversize fields consume whole contiguous producer pages until their lease is released. A consumer
  receives them through a temporary registered target that is released as soon as the field is
  copied out.
- Release is an explicit global step barrier. The backend requires non-overlapping synchronous
  steps; asynchronous consumers need epoch-scoped ownership instead.
- Missing RDT APIs, unknown owners, double release, stale generations, and cache exhaustion fail loudly.
- `ProducerDataPool.close()` and `ConsumerDataPool.close()` deregister supported NIXL memory and drop
  persistent pages during actor teardown.

Implementation: `workers/data_pool.py`, `workers/ray.py`, and `comm/codec.py`. The focused cross-node
transfer and lifecycle test is `tests/test_workers_nixl_data_pool.py`.

## Profiling

Set `POSTTRAIN_PROFILE_LOOP=1` to emit structured phase records:

```text
benchmark profile component=<component> <dimensions...> <phase>_s=<seconds> total_s=<seconds>
```

The `grpo_step`, `rollout_generate`, `ray_invoke`, and `data_pool_publish` components separate the
driver timeline, vLLM generation, actor materialization/body/packing, and Ray page publication.
Enable the switch on the normal GRPO or OPD launch and retain the Ray worker logs. Profiling is
diagnostic and synchronous, so compare backends with the same setting and exclude initialization.

Set `POSTTRAIN_PROFILE_WEIGHT_SYNC=1` (only the exact value `1` enables it) in the controller that
calls `weight_sync` to log one record after each policy-to-rollout publication that streams bytes:

```text
benchmark weight_sync_detail <json>
```

The JSON holds latency distributions over policy cells for `plan`, `coverage`, `ready` and `total`,
the slowest cell, and per bucket the `materialize`, `gather_gpu` and `deliver` distributions with the
bucket's `meta`. `meta.staged_bytes` is the bucket's flat staging size, including 16-byte alignment
padding, not the raw tensor payload. `p50_s` is the median; `p90_s` and `p99_s` are nearest-rank.

For Qwen3-VL's producer-driven export session, `plan_s` (JSON `plan`) includes the one-time shard
state-dictionary build. Previously, each bucket's `materialize_s` included a separate build.
`ready_s` (JSON `ready`) includes export-session cleanup alongside receiver live-value verification
performed during `mark_weight_version_ready`. Compare `total_s` to distinguish reduced work from
work moving between these phases.

Byte-free shared-model publication logs nothing. The probe adds one RPC round to the policy cells per
publication and moves no tensors; when it fails or its records disagree, it logs a warning instead
of ending the step.
