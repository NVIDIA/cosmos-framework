---
name: cosmos3-nano-sim-bimanual-troubleshooting
description: Diagnose Cosmos3 Nano Sim Bimanual setup, data, training, checkpoint, inference, and evaluation failures by locating the first broken product contract. Use when a supported workflow fails, produces invalid media, or cannot be reproduced.
---

# Troubleshoot Cosmos3-Nano-Sim-Bimanual

Find the first broken boundary in setup, data preparation, training,
checkpoint loading, generation, or scoring. Start with the cheapest check that
can distinguish the bounded causes. Preserve the
[model contract](../shared/model-contract.md) during recovery.

## Find the symptom

| Symptom                                                             | Likely boundary                              | Procedure                                                                     |
| ------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| Import, CUDA, or model access fails                                 | Environment or allocation                    | [Check runtime and access](#1-check-runtime-and-access)                       |
| Width, mask, timing, caption, or split is wrong                     | Data materialization                         | [Validate one input](#2-validate-one-input)                                   |
| Worker cannot import Framework or read shared assets                | Launch environment and placement             | [Inspect staged launch](#3-inspect-staged-launch)                             |
| First batch stalls or iteration 3 cannot load compiled packages     | Decode/materialization or shared output root | [Inspect training progress](#4-inspect-training-progress)                     |
| Missing EMA keys, inherited iteration, or wrong teacher             | Checkpoint operation and stage roles         | [Check checkpoint identity and state](#5-check-checkpoint-identity-and-state) |
| Missing input media, mutated prompt, static video, or mesh mismatch | Input access or generation contract          | [Check generation](#6-check-generation)                                       |
| Missing metric rows or incomparable PSNR                            | Population and target alignment              | [Check evaluation](#7-check-evaluation)                                       |

## 1. Check runtime and access

1. Preserve the exact command, code revision, resolved config, immutable
   checkpoint, selected variant, first traceback, and failing input identity.
2. From the checkout root with the locked environment active, inspect the
   package and CLI. A container package outside the staged checkout is a
   different runtime.

   ```bash
   : "${FRAMEWORK_DIR:?set the Framework checkout root}"
   cd "$FRAMEWORK_DIR"
   source .venv/bin/activate
   python -c 'import cosmos_framework, torch; print(cosmos_framework.__file__, torch.__version__, torch.version.cuda, torch.cuda.is_available())'
   python -m cosmos_framework.scripts.inference --help
   df -h /dev/shm
   ```

   Keep `COSMOS_TRAINING=0` for an inference-only installation; training uses
   `COSMOS_TRAINING=1` and the training extra.
3. **If `uv sync --locked` reports a stale lock before installation:** verify
   it with the required resolver from the checkout root:

   ```bash
   uvx --from 'uv==0.11.29' uv lock --check
   ```

   If this passes, the original command used an incompatible `uv`; repeat setup
   through the documented `uvx` command in the environment guide. Do not regenerate
   the lock with the incompatible resolver. If it fails in a clean checkout at
   the selected Framework commit, preserve the output and report the revision.
4. **If imports fail:** repair the locked environment using the
   [setup guide](../shared/release-workflow.md). **If CUDA fails:** repeat GPU
   checks inside the target allocation before replacing dependencies.
5. Check local checkpoint, tokenizer, data and media readability. HF downloads
   may need model read access. Require provider credentials only for resources
   that are actually remote. Never copy or print credentials.
   Continue when imports, help, allocation CUDA, and selected access checks
   pass. Then rerun the original boundary with one sample.

## 2. Validate one input

1. For a schema-5 input, use Framework's checker against the actual export and
   the smallest failing manifest:

   ```bash
   : "${CHECKPOINT_DIR:?set the complete local schema-5 HF export}"
   : "${INPUT_JSONL:?set the failing schema-5 manifest path}"
   python tools/check_bimanual_mixed_inputs.py \
     --checkpoint-path "$CHECKPOINT_DIR" --input "$INPUT_JSONL"
   ```

2. **If width, mask, or domain fails:** trace native calibration, geometry,
   59-value mapping, normalization, and 64-value invalid padding through the
   [action schema](../post-training/references/unified-v1-action-schema.md).
   Repair the actual mapping. Do not reshape values or relabel an embodiment.
3. **If timing fails:** inspect timestamps and resampling. Require `T`
   actions for `T+1` observations and `4k+1` output frames. Do not trim one
   modality silently.
4. **If captions or splits fail:** match captions to the selected interval
   and isolate entire episodes. Remove future-outcome leakage when the task
   conditions only on the initial scene.
5. Repeat JSON validation, the schema round trip, and loader materialization.
   Continue with finite tensors, aligned times, grounded captions, and disjoint
   episodes.
   For a local training bundle, follow the native
   [data preparation and short training check](../post-training/SKILL.md#1-prepare-and-validate-training-data).

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

## 6. Check generation

1. Run the native [checkpoint and input check](../inference/SKILL.md#1-choose-the-checkpoint-path).
   The checker verifies the export, image, and action sidecar before loading
   weights. Check decoding in the target environment as well. For user-owned remote media,
   establish access separately; checkpoint credentials do not authenticate it.
2. **If an object is confirmed absent:** choose another supported row or
   user-owned media. Preserve relative action paths. Do not cancel an
   independent healthy training run.
3. For field rejection, compare the imported package origin with the staged
   source before editing valid input fields. For a short, static, or corrupt
   video, inspect the actual checkpoint, schema, normalized actions, frame
   count, domain, prompt, seed, and decode properties.
4. For a schedule mismatch, select the matched checkpoint and command. The
   current released schema-5 model and canonical user-trained Stage 3 student
   both use four steps, but each export owns its exact ladder. Preserve the
   selected export's saved schedule and matching CLI step count.
5. For changed punctuation or resolution text, check the export's prompt mode
   and input manifest. The matched schema-5 export selects plain text. Keep the
   exact intended prompt and follow the matching native example.
6. For process-group mismatches, require the resolved world size to equal the
   launched count. Verify divisibility of DP and CP/CFG products before loading
   weights. Retest one row before scaling.
7. Continue only with one playable output per input row after the bundled
   validator fully decodes it and confirms exact saved frame count, dimensions,
   frame rate, and spatial/temporal change. Source, padded-canvas, and saved
   dimensions may differ after spatial-compression alignment. Keep unchanged
   intended conditioning, a complete metadata record, and Cosmos3-Nano, released, and
   user-trained outputs labeled with their actual checkpoint identities.

## 7. Check evaluation

1. Compare planned sample IDs with output and metric rows before aggregation.
   Keep failures, skips, duplicates, corruption, and missing targets in the
   denominator.
2. For PSNR discrepancies, match membership, target alignment, RGB/action
   preprocessing, checkpoint variant, seed, generation recipe, and metric
   implementation. Do not resize, truncate, or drop samples to force a score.
3. Recompute aggregate counts and PSNR from the saved rows. Continue only when
   accounting reconciles. Report a recipe comparison if an invariant changes.
   Follow the [evaluation guide](../inference/references/evaluation.md) for the
   saved results and comparison procedure.

## Return to the workflow

1. Record the symptom, root cause, correction, and checks rerun.
2. Return training to [post-training](../post-training/SKILL.md),
   generation to [inference](../inference/SKILL.md),
   and scoring to [evaluation](../inference/references/evaluation.md).
3. State whether existing outputs remain comparable.

## References

- [Model contract](../shared/model-contract.md)
- [Environment and workflow](../shared/release-workflow.md)
- [Unified action schema](../post-training/references/unified-v1-action-schema.md)
- [Multi-node Framework launcher](../post-training/references/multinode-launcher.md)
- [Failure playbook](references/failure-playbook.md)
- [Four-stage reference recipe](../post-training/references/canonical-recipe.md)
- [Checkpoint handoffs and resume](../post-training/references/checkpoint-and-resume.md)
- [Evaluation guide](../inference/references/evaluation.md)
