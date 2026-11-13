---
name: cosmos3-nano-sim-bimanual-troubleshooting
description: Diagnose Cosmos3 Nano Sim Bimanual setup, data, training, checkpoint, inference, and evaluation failures by locating the first broken product contract. Use when a supported workflow fails, produces invalid media, or cannot be reproduced.
---

# Troubleshoot Cosmos3-Nano-Sim-Bimanual

## Find the symptom

| Symptom                                                         | Likely boundary                              | Procedure                                                                     |
| --------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| Worker cannot import Framework or read shared assets            | Launch environment and placement             | [Inspect staged launch](#3-inspect-staged-launch)                             |
| First batch stalls or iteration 3 cannot load compiled packages | Decode/materialization or shared output root | [Inspect training progress](#4-inspect-training-progress)                     |
| Missing EMA keys, inherited iteration, or wrong teacher         | Checkpoint operation and stage roles         | [Check checkpoint identity and state](#5-check-checkpoint-identity-and-state) |

> **Skill note (not in the source):** the opening guidance, the symptom rows and procedures for runtime and access (1), input validation (2), generation (6) and evaluation (7), with the matching failure-playbook sections, and "Return to the workflow" are now the Sim-Bimanual inference skill's [troubleshooting reference](../../cosmos3-inference-simulation-bimanual/references/troubleshooting.md).

## 3. Inspect staged launch

1. Use the native `torchrun` commands in [post-training](../post-training/SKILL.md).
   Confirm the package origin is the intended Framework checkout on every rank.
2. Check the environment, complete bundle, local model/VAE assets and output
   root are readable at the same absolute paths on all workers.
3. For multi-node runs, follow the [Framework launcher notes](../post-training/references/multinode-launcher.md).
   Check unique node ranks, one reachable rendezvous address, GPU count and NCCL.
   Confirm these before retrying the failing stage.
4. The [failure playbook](references/failure-playbook.md#launcher-or-allocation-failure)
   covers shared-path and configuration checks.

## 4. Inspect training progress

1. For OOM or collective hangs, inspect world size, parallel degrees,
   maximum-size sample, token budget, activation checkpointing, worker/prefetch
   settings, and the first collective error. Change one supported memory
   setting at a time.
2. For a stalled first batch, inspect local media decoding, required bundle
   strata, stage-specific windows and worker progress. Fix the first failing
   local asset or transform; do not apply remote-loader watchdog settings.
3. For an iteration-3 zip/AOT load failure, check whether
   `config.job.path_local` is node-local. Set an absolute allocation-shared
   `IMAGINAIRE_OUTPUT_ROOT` and unique run name. Rerun the config check and
   restart from the stage input.
4. For a missing `libnppicc.so.13`, follow the Framework setup library checks
   and keep the locked environment. Framework discovers wheel-provided NPP
   libraries; also check the FFmpeg library search path and available `/dev/shm`
   in every target worker.
5. For non-finite loss, inspect the first offending batch, action ranges,
   masks, flow timestep, gradients, LR, and frozen-reference inputs. Keep the
   bad batch for diagnosis. Do not bypass it by silent sample dropping.
6. Require first-batch completion, finite optimization, an aligned checkpoint
   save, fresh-process readback, and matched generation after recovery. With
   compilation enabled, cross iteration 3 and finish a later finite iteration.

## 5. Check checkpoint identity and state

1. Identify the stage, exact saved iteration, regular-or-EMA variant, and
   whether the requested operation is a weight-only warm start or same-stage
   resume. Inspect the [checkpoint contract](../post-training/references/checkpoint-and-resume.md)
   before changing load strictness.
2. **If a new stage inherited iteration or optimizer state:** set
   `checkpoint.load_training_state=False` and apply that stage's exact input
   and skip-list row. Stage 2 uses Stage 1 for both net and frozen teacher.
   Stage 3 uses Stage 2 generation weights and Stage 0 reference/critic inputs.
3. **If a Stage 0 load from Cosmos3-Nano reports missing `net_ema.*`:** restore strict
   regular loading, skip the absent Cosmos3-Nano EMA tree and fresh action modules,
   then copy the completed regular tree into EMA through the source-checked
   post-load callback. Do not weaken strictness to hide unexplained keys.
4. **If full-state resume is requested:** require the same stage and compatible
   config, model, optimizer, scheduler, EMA, preprocessing, dataset identity,
   sampler, topology, and variant. Otherwise take a fresh weight-only run.
5. Use the native local-loader resume callback. Verify restored counters and
   loader state for resume, or fresh counters for a warm start. Classify every unmatched key and
   stop on a real architecture mismatch.
6. **If export finds no vision tensors:** inspect the source keyspace. The
   exporter accepts one unambiguous `visual.*` or `model.visual.*` prefix and
   normalizes it. A source containing both is invalid. This is shared exporter
   behavior, although the mismatch was first observed on Stage 3 output.

## References

- [Model contract](../../cosmos3-inference-simulation-bimanual/references/model-contract.md)
- [Environment and workflow](../shared/release-workflow.md)
- [Unified action schema](../../cosmos3-inference-simulation-bimanual/references/unified-v1-action-schema.md)
- [Multi-node Framework launcher](../post-training/references/multinode-launcher.md)
- [Failure playbook](references/failure-playbook.md)
- [Four-stage reference recipe](../post-training/references/canonical-recipe.md)
- [Checkpoint handoffs and resume](../post-training/references/checkpoint-and-resume.md)
- [Evaluation guide](../../cosmos3-inference-simulation-bimanual/references/evaluation.md)
