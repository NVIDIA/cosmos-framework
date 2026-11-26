# 8-bit AdamW optimizers

Post-training can keep AdamW moments in 8 bits with
[bitsandbytes 0.49.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/bitsandbytes/optim/adamw.py).
This cuts optimizer-state memory by about 75% for eligible tensors. Model weights are not
quantized, and existing recipes are unchanged unless a config selects one of the names below.

| Name               | Moments                            | Memory                                                                 |
| ------------------ | ---------------------------------- | ---------------------------------------------------------------------- |
| `adamw_bnb_8bit`   | Blockwise uint8, FP32 block scales | Ordinary CUDA memory                                                   |
| `paged_adamw_8bit` | Same                               | CUDA managed (paged) memory for tensors with at least 100,000 elements |

`adamw_8bit` is rejected on purpose, so a checkpoint's optimizer name always identifies the
backend that wrote it.

## Install

The backend is an optional extra. Ordinary optimizers work without it, because imports are
lazy.

```sh
uv sync --extra quantized-optimizers
```

## Where it is supported

| Workflow                                | How to select                                                             | Update policy                                   |
| --------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------- |
| Native posttrain SFT                    | `rl.optimizer.name=adamw_bnb_8bit`                                        | `parameter` (default) or `fp32_master`          |
| GR00T Joint PPO                         | `rl.optimizer.name=adamw_bnb_8bit rl.optimizer.update_policy=fp32_master` | `fp32_master` required                          |
| Generator (VFM) SFT, e.g. action policy | `optimizer.optimizer_type=adamw_bnb_8bit optimizer.fused=False`           | `fp32_master` behaviour, always                 |
| AlpaGym CLRL                            | `--optimizer adamw_bnb_8bit`                                              | `fp32_master` (default; `parameter` is refused) |

AlpaGym writes the resolved optimizer settings to `optimizer_config_resolved.json` in the
run's own checkpoint directory (the `checkpoints/` directory it saves to), not into the
host-owned directory that holds the resolved config. The file is run metadata, rewritten by
every launch: the AlpaGym entry does not resume from its checkpoints.

Other workflows reject both names, and `update_policy=fp32_master`, at config validation, before
a model is built: only the SFT and Joint PPO trainers save and restore the master checkpoint
schema.

- **Native SFT, `parameter`:** one CUDA device, FP32 trainable storage and
  `rl.checkpoint_format=consolidated`.
- **Native SFT, `fp32_master`:** BF16 or FP32 storage, one process or one-dimensional FSDP2,
  and either checkpoint format.
- **Generator (VFM) SFT:** set `optimizer.fused=False`; the 8-bit backend has its own kernels, so
  a fused request is refused. Set `checkpoint.dcp_async_mode_enabled=false`: the rank-local
  optimizer payload isn't staged through the pinned-memory async save, so the run refuses it
  when it starts. Use the DCP `DistributedCheckpointer`; the single-file, multi-rank and
  reasoner checkpointers, and distillation checkpointing, refuse rank-local optimizer state
  before training.

Config validation (`--validate-manifest`) checks only that the pinned bitsandbytes package is
installed; it doesn't use a GPU. The CUDA kernels are probed once, on every rank, before the
model is built.

## Update policies

`rl.optimizer.update_policy` controls where the FP32 update happens.

- **`parameter`** (default): the optimizer updates the model's own parameters.
  - Trainables must be contiguous, nonempty CUDA FP32 parameters on one device.
  - BF16 trainables are rejected. Without FP32 masters, updates smaller than the BF16 spacing
    round away.
  - DDP, FSDP and DTensor are rejected.
- **`fp32_master`**: the optimizer owns one FP32 master per local shard.
  - BF16 storage gets a tracked FP32 copy; FP32 shards alias the model storage without a copy.
  - Gradients are staged after the framework's own reduction and global-norm clipping.
  - It runs one backend step and copies the rounded result back into the model shard.
  - It supports a single process, or one-dimensional FSDP2 `Shard(0)` meshes over the full world.
  - FSDP1, HSDP, TP/EP, FP16 and optimizer offload are rejected.
  - With `name=adamw` it is the matched full-precision control.
  - It requires `force_finite_grads=true` (the default) with every optimizer name: the trainer's
    rank-consistent finite check is what lets a nonfinite window be skipped instead of aborting.
  - `bnb_min_8bit_size` is compared with the logical parameter size, not the local shard, so a
    parameter gets the same moment precision on every rank and at every world size.
  - The 8-bit quantization blocks (256 elements, each with its own scale) are laid over each
    local shard. Exact 8-bit numerics therefore depend on the world size, which is why exact
    resume requires the same topology.

## Options

| Option                 | Default     | Meaning                                                                                                                                                                                                |
| ---------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bnb_min_8bit_size`    | `4096`      | Tensors with fewer elements keep FP32 moments                                                                                                                                                          |
| `bnb_fp32_state_names` | `[]`        | Exact parameter names that keep FP32 moments, as in the model state dict: activation-checkpoint and `torch.compile` wrapper segments are removed. Unknown, frozen, duplicate or tied-alias names fail. |
| `update_policy`        | `parameter` | See above                                                                                                                                                                                              |

Both `bnb_*` options are only valid with the 8-bit names. Other restrictions:

- `impl`, `fused`, `foreach`, `capturable` and `amsgrad` are rejected.
- `force_finite_grads` must stay true (also for `update_policy=fp32_master`).
- Backend percentile clipping is disabled; clipping and gradient accumulation stay with Cosmos.

Native SFT and Joint PPO: a window with any nonfinite gradient changes no moments and does not
advance the scheduler. Generator (VFM) SFT: the optimizer step is skipped on every rank without
changing masters, moments or weights, and a warning is logged; as with the legacy optimizers,
the trainer still advances the scheduler.

## Checkpointing and resume

Exact resume requires the same backend and bitsandbytes version (its quantization maps and block
layout are part of the saved format), optimizer name, update policy, parameter names and shapes,
group order and membership, and topology. The torch version is recorded but not required to match.

- **Hyperparameters:** native SFT and Joint PPO also require the same group hyperparameters. The
  generator (VFM) container behaves like its legacy optimizers instead: a changed weight decay or
  epsilon is allowed, and the configured values are re-applied after loading; the learning rate
  and betas come from the checkpoint, as with the legacy container.
- **What is saved:** moments, block scales and maps, counters and FP32 masters. The
  `fp32_master` path writes rank-owned DCP state: one `uint8` tensor per rank (format
  `cosmos_rank_master_v4`), which DCP writes as a plain tensor and never pickles. It is decoded
  with `torch.load(weights_only=True)` and a minimal allow-list, after a SHA-256 check.
- **What is checked on load:** every rank checks that the restored masters equal its model
  shard before anything live is replaced.
- **Failures:** a partial, corrupted or changed checkpoint is rejected and leaves the live
  optimizer untouched.
- **Paged state:** paged moments are restored into managed memory.
- **Changing optimizer or topology:** switching optimizer families, or changing the world size,
  needs an explicit weights-only restart. A generator run resuming from a checkpoint without this
  state, or with state written by another world size, fails under the default
  `checkpoint.strict_resume=True`. With `strict_resume=False`, or a `keys_to_skip_loading` pattern
  matching a trained parameter on warm start, it starts with fresh moments and FP32 masters
  re-seeded from the loaded model, and logs a warning. The same happens under
  `strict_resume=False` when the saved state has another structure (changed parameter grouping
  or `bnb_*` options), or when the saved masters differ from the loaded model, e.g. after a
  partial model load that kept initial values. Every rank makes the same decision.
- **Exported weights:** exported model and adapter weights don't need bitsandbytes at
  inference time.

The consolidated SFT checkpoint keeps only the newest optimizer state by default. Set
`rl.keep_optim_last=0` to keep several exact-resume points.

## Diagnostics

- `COSMOS_OPTIMIZER_STATE_REPORT=1` (generator) logs the optimizer-owned bytes once per rank.
- `rl.cuda_allocator_receipts=true` (Joint PPO) adds each rank's CUDA caching-allocator counters
  to every update receipt (`cuda_allocator`). It resets the process-wide peak counters on every
  update, so it is off by default.

## Example: native SFT

```sh
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config cosmos_framework/posttrain/configs/sft_quantized_optimizer.py -- \
  rl.optimizer.name=adamw_bnb_8bit rl.optimizer.bnb_min_8bit_size=16384
```

- **Paged state:** use `rl.optimizer.name=paged_adamw_8bit`.
- **Full-precision baseline for comparison:**
  `rl.optimizer.name=adamw rl.optimizer.impl=fused rl.optimizer.bnb_min_8bit_size=4096`.
  The last override clears the 8-bit-only option.

The GR00T recipe is `configs/gr00t_n1d7_libero_quantized_optimizer.py`.

## Measured results

These numbers come from the qualification campaign on 8×A100-80GB, with protocols locked
before each comparison. Its LIBERO-10 recipes and evidence are not part of this change.

|                                              | GR00T-N1.7 LIBERO-10 PPO | Cosmos3-Nano LIBERO-10 SFT                                   |
| -------------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| Optimizer moments per rank                   | 13.00 → 3.30 GB (−74.6%) | 6.98 → 1.77 GB (−74.6%)                                      |
| Exact reload and resume, non-paged and paged | Pass                     | Pass                                                         |
| Quality vs full precision (3 seeds)          | Within a 2-point margin  | 1.6 points lower at the final checkpoint; 95% CI [−3.3, 0.0] |
| Step time                                    | Unchanged                | Unchanged (46.2 s/iter)                                      |

Two caveats about what the savings buy:

- **Memory saving includes masters only partly.** With `fp32_master`, the FP32 masters remain;
  counting them, GR00T optimizer-owned memory falls 49.7%.
- **The freed memory does not make training faster.** On this hardware, a larger per-GPU batch
  doesn't change step time, and turning off activation recomputation doesn't fit for either
  optimizer.
