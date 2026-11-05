# DPO

Offline preference optimization on Cosmos diffusion models: train directly on winner/loser pairs,
with no rollout engine and no reward model at training time. Facts here are read from
`config/rl.py::DPOConfig`, `entrypoints/dpo.py`, `algorithm/dpo.py`, and the experiments under
`config/experiment/dpo/`.

## 1. When to choose it

Use DPO when you already have **preference pairs** — two generations for the same prompt, one
labelled better. It is the cheapest option in this framework: no rollout engine, no reward
provider, no weight-sync edge, so the whole run is one training loop over a dataset.

Choose something else when you have a scorer but no pairs ([`grpo.md`](grpo.md)), or a stronger
model to imitate ([`opd.md`](opd.md)).

DPO differs structurally from the online algorithms: it is the one entry that runs under plain
`torchrun` with no Ray, and the one whose train-time validation evaluates a **held-out preference
objective** rather than performing rollout and reward.

## 2. Quick start

The release-facing, catalog-free image recipe and its direct-container command
are documented in the [public run guide](../howto/run-posttrain.md#4-dpo). It runs on four GPUs,
initializes from the converted Cosmos3-Nano DCP through `DPO_BASE_CHECKPOINT`, and uses synthetic
preference pairs as a functional smoke.

## 3. Data contract

`rl.data` is a `PreferenceEnvConfig`. Three sources:

| Source              | Use                                                                          |
| ------------------- | ---------------------------------------------------------------------------- |
| `synthetic_pairs=N` | In-memory generated pairs. For smoke runs and CI — no files, no coordination |
| Preference JSONL    | Real pairs on a filesystem                                                   |
| Tar shards          | Preference pairs in webdataset-style tars                                    |

Custom sources go through an importable factory
([`../howto/custom-data.md`](../howto/custom-data.md)). **A custom source used with train-time
validation must expose a metadata-only `membership_ids()` hook**; this is a fail-loud contract that
keeps the leakage audit from enumerating arbitrary rows or decoding media payloads.

## 4. Topology and launch

| Axis            | Values                           | Meaning                                                                           |
| --------------- | -------------------------------- | --------------------------------------------------------------------------------- |
| `grid.backend`  | `local` (default) \| `ray`       | `local` runs under torchrun, which carries FSDP directly                          |
| `grid.grouping` | `single` (default) \| `per_role` | Whether all roles share one process, or each gets its own. `per_role` is Ray-only |

Under torchrun, FSDP width is set explicitly with
`model.config.parallelism.data_parallel_shard_degree` and must equal
nodes x GPUs-per-node.
Multinode in either mode requires operator-provided orchestration.

## 5. Key configuration

Complete table: [`../reference/config.md`](../reference/config.md). In routine use:

| Knob                           | What it controls                                                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rl.beta`                      | Preference sharpness. The reproduction experiments use large values (1000)                                                                                         |
| `rl.optimizer.lr`              | Learning rate. DPO runs use much smaller values than pretraining                                                                                                   |
| `rl.max_steps`                 | Optimizer steps                                                                                                                                                    |
| `rl.pairs_per_step`            | Preference pairs consumed per step                                                                                                                                 |
| `rl.data.synthetic_pairs`      | Non-zero switches to generated pairs and ignores any file source                                                                                                   |
| `rl.init_from`                 | Base checkpoint. An `s3://` value is pre-pulled once to a local cache and reused on every requeue                                                                  |
| `rl.init_keys_to_skip_loading` | Base-checkpoint keys the DPO model does not build; any other extra key in the base fails the load. The public recipe lists the Cosmos3-Nano sound and action heads |

## 6. Is it training correctly

`train/dpo_accuracy` is the readable signal: the fraction of pairs the policy ranks correctly. It
starts near 0.5 and should rise. `train/dpo_reward_margin` is the implicit reward gap driving it.

DPO is the **only** objective publishing both `train/` and `val/` metrics, so with validation
enabled you can compare the two directly. Full table:
[`../reference/metrics.md`](../reference/metrics.md).

## 7. Checkpoints

Shared mechanics: [`../reference/checkpoint.md`](../reference/checkpoint.md). DPO specifics:

- **The frozen reference is not saved.** Setup rebuilds the anchor from the base checkpoint, so a
  DPO checkpoint carries no `ref` component.
- **The preference cursor resumes** (`rl.data.cursor_resume`), so a requeued run does not replay
  pairs it already consumed.

## 8. Failure semantics and limits

- **Preference symmetry is enforced.** Winner and loser must differ only in generated content; a
  prompt-token mismatch between the two sides fails rather than training on a corrupted pair.
- **Validation requires an explicit held-out source.** Posttrain never takes an implicit split from
  the training data.
- **`grid.grouping=per_role` requires Ray.** Under `local` there is one process and nothing to
  split, so it is rejected.
