# Troubleshooting

Organized by **symptom**, because that is what you arrive with. Each entry gives the cause, the
fix, and where to read more.

## The job hangs

### Every GPU drops to 0% but holds its memory, and nothing logs an error

The classic disaggregated-run failure at the first weight sync. It is **not a deadlock — it is a
segfault that never gets to die.**

Look in the Ray worker `.err` file (not the actor log) for the canary:

```
W ucx_utils.cpp:628] 8 NVIDIA GPU(s) were detected, but UCX CUDA support was not found!
E ucx_utils.cpp:582] VRAM memory is detected as host by UCX...
```

followed by `nixlAgent::registerMem -> ... -> ucp_mem_unmap -> SIGSEGV`.

**Cause:** two UCX copies in one process. The NIXL wheel vendors its own UCX; NCCL's HPC-X plugin
loads the container's. The wheel's copy loses CUDA support, treats device memory as host memory,
and faults — and the *other* UCX's error handler catches the SIGSEGV and freezes the process
instead of aborting.

**Fix:** the framework calls `warm_nixl()` before the NCCL rendezvous so the NIXL agent
initializes first. This **wins the race rather than removing it**, so it is order-fragile: any code
path that initializes NCCL earlier — a library that builds a process group at import, for instance
— brings the hang back silently and in the same shape. Full analysis, the evidence, and what to
re-check whenever the image or the NIXL pin moves:
the internal environment analysis.

> `UCX_NET_DEVICES=lo` is **not** the fix. It was introduced before the ordering bug was
> understood and merely masked it; it also cannot work cross-node.

### The run stops mid weight sync after a bucket fails

**Known limitation.** Streaming weight sync has no abort path for a mid-stream failure
(`comm/weight_transfer.py`, marked `FIXME ... HANG RISK`). If applying a bucket fails on one peer,
the others keep waiting for buckets that will never arrive, so a recoverable error presents as a
hang.

**Workaround:** none automatic. Kill the job and inspect the actor logs for the first bucket-apply
exception; that peer is the origin.

### `Prolog hung on node ...`

Normal on some clusters — container or NFS setup exceeding the prolog timeout. The job proceeds.

## Out of memory

### The policy OOMs in `log_softmax` on a colocated run

The logprob computation is deliberately done in fp32 over the full vocabulary (bf16 was measured to
drop the GRPO clip rate 17x, from 0.01493 to 0.00089 at `n_mini=8`), and at Qwen's 151k vocabulary
that tensor is roughly 1 GB per chunk. The colocated vLLM engine's reservation is what leaves too
little room.

**Fix, in order** — change one item, rerun a short smoke, and stop as soon as it fits. This order
preserves training semantics for as long as possible:

1. **Lower `rollout.gpu_memory_utilization`** (e.g. `0.40` → `0.25` → `0.20`). Reduces vLLM KV
   capacity, so throughput may drop, but training semantics are unchanged.
2. **Add or lower `rl.micro_batch_per_replica`.** A pure memory split: same trajectories, same
   single optimizer step, smaller activation footprint. See
   [`howto/batching.md`](howto/batching.md).
3. **Reduce `rollout.max_tokens` and any validation `max_tokens` together**, then
   `rollout.max_model_len` — only if a shorter reasoning and truncation contract is acceptable.
   Record every changed value with the run.
4. **Reduce train and held-out image ceilings together** for VLM runs (e.g.
   `rl.data.factory_kwargs.max_image_side` and the matching validation key). This changes visual
   resolution, so keep the two identical and record it as a distinct variant.
5. **Stop and qualify a different model or topology** rather than changing several memory axes at
   once.

### `micro_batch_per_replica` rejected as an OOM fix

`rl.micro_batch_per_replica` must be divisible by the policy's `dp_world`. With one eight-rank
policy replica, `8` is already the minimum legal value — `4`, `2`, and `1` fail the
batch-divisibility contract and are **not** OOM recovery options. Likewise `rl.batch_per_replica`
must assign whole prompt groups to each rollout data replica.

## Placement and process-group errors

| Message                                                     | Cause                                                      | Fix                                                                                                                                             |
| ----------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `ROLLOUT_RANKS=N must equal rollout.tensor_parallel_size=M` | `grid.shapes.rollout` disagrees with the engine's TP width | Set `grid.ranks` / `grid.shapes.rollout` so the rank width equals `tensor_parallel_size`                                                        |
| `POLICY_REPLICAS=N is unsupported`                          | Policy placement has no replica axis                       | Use `grid.shapes.policy=[1,POLICY_RANKS]` and configure HSDP as `dp_replicate * dp_shard` inside those ranks — see [`glossary.md`](glossary.md) |
| `grid.shapes requires grid.backend='ray'`                   | Explicit shapes under the local backend                    | Either drop the shapes or switch to `ray`                                                                                                       |
| `per-replica rendezvous not run (need ranks=TP>1 on GPU)`   | `tensor_parallel_size > 1` under the local backend         | vLLM TP>1 needs Ray; `local` is the TP=1 path                                                                                                   |
| `policy cells disagree on needs_old_logprobs()`             | Replicas were not built alike                              | A topology fault — do not retry; the run refuses rather than training some cells clipped and others not                                         |

**Two jobs on one node collide.** They share the `--ray-port` (Ray GCS / torchrun c10d rendezvous)
and contend for GPUs. Run them sequentially, or give each a distinct `--ray-port` *and*
non-overlapping nodes.

## Wrong or missing files at runtime

### `dataset has 4 item(s) but world_size=8`, or workers read stale data

**`IMAGINAIRE_OUTPUT_ROOT` is not on a shared filesystem.** The default `/tmp/imaginaire-output` is
container-local: the driver on the head node writes run artifacts there, and actors on other nodes
resolve the same path inside *their* containers, reading absent or leftover files. Prefix the
launcher's `--run` with a shared path.

The same applies to `--log-dir`: the default `/tmp` is node-local to the batch host and disappears
with the allocation.

### Intermittent `pyxis: ... File exists` during import

A multi-minute squashfs extraction can fail this way when Slurm re-fires a task launch. The
generated sbatch script retries the import up to 3 times after removing the partial rootfs; only if
all attempts fail does the job abort.

## The run aborts

### `dynamic sampling batch_step=N: replay cells [...] hold fewer informative trajectories than their quotas [...]`

`rl.min_informative_groups > 0` (or `rl.min_informative_groups_frac` set) with the default
`rl.dynamic_sampling_on_budget_exhausted=raise`:
the extra-round budget (`rl.dynamic_sampling_max_extra_rollouts`) was spent and at least one replay cell still lacks its quota of groups
whose rewards disagree (the global count split exactly across the data cells). Every listed cell saw
fewer informative groups than its quota across `1 + dynamic_sampling_max_extra_rollouts` rounds (the base draw plus the extra rounds), so
the step did not train: the reward is not
discriminating at the current policy — too hard (all zero), too easy (all one), or `rl.group_size`
too small to see disagreement.

**Fix:** look at `train/grpo_all_zero_group_count` / `grpo_all_one_group_count` on the preceding
steps and adjust the data mix, reward, or group size. Raising the budget only defers the abort when
no prompt in the source produces spread. Set `rl.dynamic_sampling_on_budget_exhausted=proceed` only
when a fixed batch shape matters more than the informative-only guarantee; the step then trains on
tied groups as zero-advantage filler and logs the shortfall. Details:
[`guides/grpo.md`](guides/grpo.md#dynamic-sampling-dapo).

## The π_old ahead pass refuses to start

### `π_old ahead-pass determinism probe` failed on a sparse-MoE policy

The message reports the share of tokens whose logprob moved between two forwards at unchanged
weights and says the forward routing was `NATURAL`. Cause: the MoE router's top-k over near-tied
scores flips with kernel jitter alone (about half the tokens change experts between two identical
forwards of Qwen3-VL-MoE-30B-A3B), so the recorded π_old does not describe the training forward.
Fix: set `rl.routing_replay=rollout`, which pins every training forward to the routing the vLLM
engine used, or run with `rl.mini_batch_per_replica` unset (`n_mini == 1`), where π_old comes from
the same forward but the train/infer routing gap remains. See the routing-replay section in
[`reference/config.md`](reference/config.md). If the message says the probe was `PINNED`, the
movement is a genuinely stochastic layer, not the router.

## Import and construction errors

| Message                                                          | Cause                                                                 | Fix                                                                                 |
| ---------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `Unable to import DataPacker factory '...'`                      | The dotted path is unavailable on the driver or a worker `PYTHONPATH` | Install or mount the extension module identically on **every** node                 |
| `DataPacker factory '...' must return a DataPacker`              | The factory returned the wrong type                                   | Return a fresh `DataPacker` subclass instance from the zero-argument factory        |
| `model_type='x' declares vision_config but has no native packer` | An unregistered VLM with the i4-native backend                        | Set `model_backend="hf"`, register a native packer, or pass an explicit one         |
| `... requires a DataPacker at construction`                      | `packer=None` passed explicitly                                       | Build one with `init_data_packer(name_or_path)`                                     |
| `packer tokenizer adds special tokens`                           | A BOS-adding tokenizer on the vLLM path                               | See the guard section in [`howto/custom-datapacker.md`](howto/custom-datapacker.md) |
| `DataPacker.setup: empty name_or_path`                           | Packer built without a model id                                       | Pass the same `name_or_path` the model or engine was built from                     |

The full table, including the objective-side symptoms, is in
[`howto/custom-datapacker.md`](howto/custom-datapacker.md).

### `MissingConfigException: Could not load 'experiment/<name>'`

Preceded by a warning like `skipped N experiment module(s) needing absent optional dependencies
['lerobot', 'psycopg2', 'torchcodec', 'zarr']`.

**Cause.** `experiment=` selection requires the experiment catalogs to be imported. Those catalogs
live in another project and carry its optional dependencies — robotics datasets, a database driver,
video and array codecs — which the posttrain runtime image does not install. Modules that cannot be
imported are skipped with that warning, so the experiments they define cannot be selected. If the
experiment you asked for, or one it inherits through `defaults`, lives in a skipped module, config
resolution fails with this error.

The public DPO and NFT recipes are not affected: they are self-contained and do
not import the inherited experiment catalog.

**Fix:** install the named dependencies into the environment, or run from an image that carries
them. The warning lists exactly which distributions are missing.

### `ModuleNotFoundError: No module named 'apex'` at optimizer construction

Raised by `utils/fused_adam.py` when a recipe sets `rl.optimizer.name="fusedadam"`. The run reaches
this only after the models have loaded, so it looks like a late failure rather than a missing
prerequisite.

**Why a recipe would need apex at all.** `fusedadam` exists to rebuild an fp32 *master copy of the
parameters inside the optimizer*. That is only necessary when the parameters are **stored** in
bf16: plain `torch.optim.AdamW` keeps its moments in the parameter dtype, and at posttrain learning
rates a bf16 update falls below bf16 resolution and rounds away.

**The usual answer is not to install apex — it is to store the parameters in fp32.** Set
`rl.master_dtype="float32"` with `rl.param_dtype="bfloat16"`: storage is fp32, compute is bf16, the
optimizer gets fp32 moments for free, and `name="adamw"` (optionally `impl="fused"` for the fused
torch kernel) is then exactly right. The public GSM8K recipe uses this configuration,
 and it matches cosmos-rl, whose LLM path never
touches apex — apex lives only on its WFM/diffusion path.

So if you hit this on an LLM recipe, check `rl.master_dtype` before reaching for apex.

**When apex is genuinely required.** Two cases, and the second is not optional:

- A recipe that deliberately keeps bf16 parameter *storage* — for example to fit a tight
  single-GPU budget — has no fp32 parameters for the optimizer to read.
- **The DPO diffusion path cannot use fp32 storage at all.** `master_dtype` is a field on
  `HFRunConfig`; `DPOConfig` extends `RunConfig` and does not have it. The two reproduction
  experiments therefore require `fusedadam`, and their docstrings record why: the first attempt ran
  plain AdamW on the bf16 net and was killed at step ~30 with roughly 7x slower loser-side drift and
  the margin plateaued at ~3 against the reference's climb to 48 — same data, same objective, same
  trainable set. Optimizer numerics alone.

So apex remains a real dependency of this package. What is avoidable is reaching for it on an **LLM**
recipe that could simply store its parameters in fp32.

> `rl.optimizer.name=adamw` **alone**, on a recipe that stores parameters in bf16, is a diagnostic
> and not a fix: it will run, but its updates round away. `config/rl.py::OptimizerConfig` says do
> not use that combination for real bf16 posttraining. Change the storage dtype too.

## Weight sync into a vLLM-Omni engine

### `cannot export the live parameters ...` when a rollout body starts or rebuilds its engine

The body maps the worker's live parameters over CUDA IPC once per worker
(`roles/rollout/vllm_omni_worker.py::OmniIpcWorkerExtension.export_live_parameters`) and refuses to
start when they could not stay in place behind the handles. The engine is closed before the error
propagates, so no worker process is left behind.

| Message                                                                                                                                         | Cause                                                                          | Fix                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `... with enable_sleep_mode=True` / `enable_cpu_offload=True` / `enable_layerwise_offload=True: they would not stay in place behind the handle` | the engine moves or re-allocates its weights                                   | Serve a weight-synced engine without sleep mode and offload (the GR00T body's engine kwargs do) |
| `... with custom_pipeline_args: re_init_pipeline re-allocates them`                                                                             | vLLM-Omni rebuilds the pipeline per request                                    | Do not pass `custom_pipeline_args` to a weight-synced engine                                    |
| `cannot export the live parameters from the 'cudaMallocAsync' allocator`                                                                        | `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` in the worker's environment  | CUDA IPC needs torch's native caching allocator; unset it                                       |
| `cannot export live parameter '...' over CUDA IPC: ... expandable_segments`                                                                     | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the worker's environment | Unset it; the weight plane needs the default allocator                                          |
| `direct parameter writes need one worker per engine; this engine runs N`                                                                        | a tensor-parallel Omni engine                                                  | The direct write addresses one worker; TP > 1 Omni engines are outside the weight plane         |
| `the live parameters were already exported by this worker; rebuild the engine for a fresh mapping`                                              | a second export on one worker (each export consumes IPC event slots)           | `rebuild_engine()`                                                                              |

### `begin_weight_update` / `finish_weight_update` refused a stream and the lane is invalidated

| Message                                                                                                        | Cause                                                                                                               | Fix                                                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `N live parameter(s) were re-allocated since the export` / `the live parameter names changed since the export` | the worker's parameters moved after the body mapped them                                                            | `rebuild_engine()` on that lane, then a full sync of the trainer's current publication version                                                                                                                        |
| `finish_weight_update(e): the open update is epoch k` / `no update is open on this worker`                     | body and worker disagree on the stream: a failed stream left another epoch open, or a stale body                    | the next `set_bucket_plan` supersedes an open epoch; if `finish` keeps failing, `rebuild_engine()`                                                                                                                    |
| `begin_weight_update(e) does not advance past epoch k`                                                         | the body's epoch counter went backwards: two bodies on one worker, or a body replaced without rebuilding the engine | one body per engine; `rebuild_engine()`                                                                                                                                                                               |
| `N live parameter(s) do not hold the bytes just copied into them`                                              | the byte-wise compare after the copies failed: the mapping does not alias the worker's memory                       | a fault, not a knob: the lane is invalidated; check the device identity (index and UUID) in the actor log, then `rebuild_engine()` and a full sync. Only publications with `verify_weight_values` on run this compare |

This path requires an operator-provisioned GR00T runtime and transport.
