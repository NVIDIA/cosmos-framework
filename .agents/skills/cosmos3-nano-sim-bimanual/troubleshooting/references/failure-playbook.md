# Failure playbook

Use the [main troubleshooting steps](../SKILL.md) to find the first failing
operation. Preserve the selected checkpoint, inputs and first traceback.

## Environment or model access

Activate the locked Framework environment from its checkout root and check
`cosmos_framework.__file__`, the native CLI and CUDA inside the target allocation.
If a pre-install lock check reports stale state, rerun it with the resolver
required by `pyproject.toml`:
`uvx --from 'uv==0.11.29' uv lock --check`. A passing check identifies a local
resolver-version mismatch; use the required setup command rather than rewriting
the lockfile.
Keep `COSMOS_TRAINING=0` for inference-only installs. Training and the metric
entry point need their documented dependencies. Follow
[Framework setup](../../shared/release-workflow.md#cosmos-framework) for FFmpeg
and NPP libraries; changing locked packages does not repair a missing native
library. HF access is needed only to download gated assets. The local training
TOMLs use local inputs and disable W&B by default.

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

For wrong width, mask or domain, trace calibration, coordinate conversion,
59-value mapping, normalization and 64-value padding through the
[action schema](../../post-training/references/unified-v1-action-schema.md).
Do not reshape an action or relabel its domain to make loading succeed.

For off-by-one timing, require `T` actions and `T+1` observations after
resampling. Check timestamps and interpolation rather than trimming one stream.
For captions or splits, keep wording grounded and source episodes disjoint.

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

## Inference failures

Run `tools/check_bimanual_mixed_inputs.py` against the downloaded HF snapshot
and selected manifest before loading weights. Resolve missing local media or
provider access before generation.

Match input and checkpoint: the current schema-5 model uses normalized 59D
mixed-domain actions, plain text, and four steps. The user's canonical Stage 3
student uses its own matched action metadata and exact four-step ladder. A width
change is not a semantic conversion.

For short, static or corrupt output, check actions, frame count, domain, prompt,
seed and full video decode. Distinguish source dimensions, padded model canvas
and saved dimensions: a 640x360 input can yield 640x352 after spatial alignment.

For distributed inference, match launched process count and resolved model world
size and divisibility. Test one output before scaling parallelism.

## Evaluation failures

Compare planned sample IDs, prediction/target pairs, successful metric rows,
missing targets and failed/skipped outputs. The native vision scorer can trim
unequal lengths or skip missing targets, so reconcile its aggregate count with
the planned population. Follow the [evaluation guide](../../inference/references/evaluation.md)
for matched timing, geometry and complete reporting.
