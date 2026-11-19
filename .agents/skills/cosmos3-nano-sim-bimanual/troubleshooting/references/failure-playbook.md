# Failure playbook

Use the [main troubleshooting steps](../SKILL.md) to find the first failing
operation. Preserve the selected checkpoint, inputs and first traceback.

> **Skill note (not in the source):** the sections "Environment or model access", "Inference failures" and "Evaluation failures" and the first two paragraphs of "Data failures" are now the Sim-Bimanual inference skill's [troubleshooting reference](../../../cosmos3-inference-simulation-bimanual/references/troubleshooting.md).

## Launcher or allocation failure

Confirm every worker uses the same checkout/environment and can read the bundle,
VAE, tokenizer, parent checkpoints and shared output root. For multi-node runs,
check unique node ranks, rendezvous reachability, GPU count and NCCL using the
[native launcher notes](../../post-training/references/multinode-launcher.md).

If tokenizer AOT reports a zip archive or missing compiled-package error around
iteration 3, inspect `config.job.path_local`. It must be job-unique and visible
at the same path on every rank. Use an allocation-shared
`IMAGINAIRE_OUTPUT_ROOT` and restart the stage.

## Training configuration failure

Use the four native TOMLs and registrations listed in the
[reference recipe](../../post-training/references/canonical-recipe.md#cosmos-framework).
Verify the resolved experiment, local bundle, VAE, stage parents and fresh/resume
operation. A nonlocal recipe with an unspecified dataloader is not a replacement
for the local-bundle TOML.

## Data failures

For a first-batch stall, inspect the completed local bundle's assets, captured
transforms, declared sources and stage-specific windows. Check decode and worker
progress before changing memory or throughput settings.

## Training failures

For OOM or collective hangs, inspect resolved world size, DP/CP layout, longest
sample, token budget, activation checkpointing and the first collective error.
Change one supported memory setting, then repeat save/readback.

For non-finite loss, retain the first bad batch and check action ranges, masks,
flow timestep, LR, gradients and frozen-reference inputs. Do not silently drop it.

## Checkpoint failures

Confirm stage, saved iteration, weight variant and warm-start versus same-run
resume using the [checkpoint contract](../../post-training/references/checkpoint-and-resume.md).
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
