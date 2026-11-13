# Troubleshoot Cosmos3-Nano-Sim-Bimanual

> **Skill note (not in the source):** this file is the training side of the Sim-Bimanual troubleshooting sub-skill, assembled word for word (cosmos-framework `main` 45fecab): first the launch, training-progress and checkpoint symptom rows, procedures 3 to 5 and the "References" list of its SKILL.md, then, under "Failure playbook", the opening of its failure playbook and that playbook's sections on launcher or allocation, training configuration, data (the first-batch stall), training and checkpoint failures. Section numbers are the source's. What changed: link targets are re-pointed, the links between the two former files are anchors in this file, and the playbook's headings are one level down. The two notes below name the parts that are in the Sim-Bimanual inference skill.

## Find the symptom

| Symptom                                                         | Likely boundary                              | Procedure                                                                     |
| --------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| Worker cannot import Framework or read shared assets            | Launch environment and placement             | [Inspect staged launch](#3-inspect-staged-launch)                             |
| First batch stalls or iteration 3 cannot load compiled packages | Decode/materialization or shared output root | [Inspect training progress](#4-inspect-training-progress)                     |
| Missing EMA keys, inherited iteration, or wrong teacher         | Checkpoint operation and stage roles         | [Check checkpoint identity and state](#5-check-checkpoint-identity-and-state) |

> **Skill note (not in the source):** the opening guidance, the symptom rows and procedures for runtime and access (1), input validation (2), generation (6) and evaluation (7), with the matching failure-playbook sections, and "Return to the workflow" are now the Sim-Bimanual inference skill's [troubleshooting reference](../../cosmos3-inference-simulation-bimanual/references/troubleshooting.md).

## 3. Inspect staged launch

1. Use the native `torchrun` commands in [post-training](../SKILL.md).
   Confirm the package origin is the intended Framework checkout on every rank.
2. Check the environment, complete bundle, local model/VAE assets and output
   root are readable at the same absolute paths on all workers.
3. For multi-node runs, follow the [Framework launcher notes](multinode-launcher.md).
   Check unique node ranks, one reachable rendezvous address, GPU count and NCCL.
   Confirm these before retrying the failing stage.
4. The [failure playbook](#launcher-or-allocation-failure)
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
   resume. Inspect the [checkpoint contract](checkpoint-and-resume.md)
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
- [Environment and workflow](environment.md)
- [Unified action schema](../../cosmos3-inference-simulation-bimanual/references/unified-v1-action-schema.md)
- [Multi-node Framework launcher](multinode-launcher.md)
- [Failure playbook](#failure-playbook)
- [Four-stage reference recipe](canonical-recipe.md)
- [Checkpoint handoffs and resume](checkpoint-and-resume.md)
- [Evaluation guide](../../cosmos3-inference-simulation-bimanual/references/evaluation.md)

## Failure playbook

Use the [main troubleshooting steps](#find-the-symptom) to find the first failing
operation. Preserve the selected checkpoint, inputs and first traceback.

> **Skill note (not in the source):** the sections "Environment or model access", "Inference failures" and "Evaluation failures" and the first two paragraphs of "Data failures" are now the Sim-Bimanual inference skill's [troubleshooting reference](../../cosmos3-inference-simulation-bimanual/references/troubleshooting.md).

### Launcher or allocation failure

Confirm every worker uses the same checkout/environment and can read the bundle,
VAE, tokenizer, parent checkpoints and shared output root. For multi-node runs,
check unique node ranks, rendezvous reachability, GPU count and NCCL using the
[native launcher notes](multinode-launcher.md).

If tokenizer AOT reports a zip archive or missing compiled-package error around
iteration 3, inspect `config.job.path_local`. It must be job-unique and visible
at the same path on every rank. Use an allocation-shared
`IMAGINAIRE_OUTPUT_ROOT` and restart the stage.

### Training configuration failure

Use the four native TOMLs and registrations listed in the
[reference recipe](canonical-recipe.md#cosmos-framework).
Verify the resolved experiment, local bundle, VAE, stage parents and fresh/resume
operation. A nonlocal recipe with an unspecified dataloader is not a replacement
for the local-bundle TOML.

### Data failures

For a first-batch stall, inspect the completed local bundle's assets, captured
transforms, declared sources and stage-specific windows. Check decode and worker
progress before changing memory or throughput settings.

### Training failures

For OOM or collective hangs, inspect resolved world size, DP/CP layout, longest
sample, token budget, activation checkpointing and the first collective error.
Change one supported memory setting, then repeat save/readback.

For non-finite loss, retain the first bad batch and check action ranges, masks,
flow timestep, LR, gradients and frozen-reference inputs. Do not silently drop it.

### Checkpoint failures

Confirm stage, saved iteration, weight variant and warm-start versus same-run
resume using the [checkpoint contract](checkpoint-and-resume.md).
Fresh Stage 0 strictly loads Cosmos3-Nano regular keys, skips missing EMA/new action
modules, then uses `InitializeEmaFromRegular`. Do not weaken strict loading.
Stage 1 loads Stage 0; Stage 2 initializes student and teacher from Stage 1;
Stage 3 uses Stage 2 student and Stage 0 reference/critic. Every new stage starts
fresh optimization state.

For same-stage resume, keep bundle, topology, run name, output root and parents
unchanged. Check restored iteration, EMA, optimizer/scheduler and local-stream
state. Wait for all asynchronous saves to complete before a fresh-process readback.

If export sees no vision tensors, inspect the source prefix. One unambiguous
`visual.*` or `model.visual.*` source is accepted; both together are not. Preserve
the student-only export and explicit regular/EMA selection.
