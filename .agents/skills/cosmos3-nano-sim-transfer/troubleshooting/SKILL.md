---
name: cosmos3-nano-sim-transfer-troubleshooting
description: Diagnose Cosmos3-Nano-Sim-Transfer setup, checkpoint loading, media alignment, training-state and output failures in an existing workflow.
---

# Troubleshoot Cosmos3-Nano-Sim-Transfer

Start with the first failing operation and preserve the last completed
checkpoint and existing outputs. Compare the effective inputs and model
identity with the [shared contract](../shared/model-contract.md).

| Symptom                                                  | Reference                               |
| -------------------------------------------------------- | --------------------------------------- |
| Import failure, CUDA unavailable, missing library        | [Runtime](#runtime)                     |
| Unknown model alias or incompatible weights              | [Model and runtime](#model-and-runtime) |
| Missing, distorted or drifting control                   | [Media and data](#media-and-data)       |
| Ignored flags, overwritten output or unexpected geometry | [Inference](#inference)                 |
| Wrong restart state or training failure                  | [Training state](#training-state)       |

The [failure table](references/failure-playbook.md) lists specific errors.

## Runtime

Identify the checkout and interpreter used by the failing command:

```bash
cd "${FRAMEWORK_DIR:?Set the Cosmos Framework checkout root}"
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m cosmos_framework.scripts.inference --help
```

Compare with [environment requirements](../shared/environment.md).
For missing Framework imports, activate the environment installed from
`FRAMEWORK_DIR` and clear inherited `PYTHONPATH` and
`COSMOS_DISTILLATION_IMPL`. Set `COSMOS_INTERNAL=0`; native DCP config
composition also needs `COSMOS_TRAINING=1`.

Errors importing `typing.Self`, `typing.override` or `tomllib` indicate an
incompatible interpreter. Use the pinned Python version. For shared-library
failures, inspect the named library and inherited `LD_LIBRARY_PATH`, then
apply the documented native dependency setup. Do not substitute CUDA stubs.

## Model and runtime

An unknown target alias or unsupported class means the artifact's config
and runtime disagree. Use their matching versions. Renaming the model class
does not convert weights.

Follow [checkpoint selection](../inference/SKILL.md#1-select-the-matched-inference-checkpoint).
For native DCP, select EMA from the completed checkpoint. For the HF release,
pass its directory and own `config.json` without an EMA flag. Obtain the
matching model, tokenizer and video autoencoder through the
[Hugging Face setup](../shared/setup.md).

If DCP inference reports missing `COSMOS_SIM_TRANSFER_*` environment variables,
supply the Stage 0/Stage 2 parents and prepared example2 bundle from the training run.
Hydra resolves the training config before inference skips training-only
loaders and parent initialization. Use native DCP EMA checkpoints to run
inference from newly trained Stage 3 weights.

## Media and data

For a count or geometry mismatch, inspect the failing media directly:

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames:format=duration \
  -of json control.mp4
```

RGB and controls must share camera, source times and field of view. Apply
crops or retiming to all paired streams together. Depth and segmentation
need their matching renderings. `example2` has RGB, depth and segmentation
on [Hugging Face](../shared/hf-sample.md); native training and inference derive
edge and blur from RGB.

Each training draw selects one control. A numeric stratum identifies a
temporal length bucket, not a modality. Local-caption bounds are exclusive:
`[0,121)` means frames 0 through 120. Generic video-caption JSONL instead uses
inclusive ends. Follow the [local data contract](../shared/local-dataset.md).

If `local_simulation` cannot import, use the pinned Framework checkout and
install its environment. The native local data route requires
`cosmos_framework.configs.simulation.local` and
`cosmos_framework.data.generator.local_simulation.dataset`. A generic
center-crop loader does not provide the same preprocessing. Rebuild local
training data after changing source paths or captions.

## Inference

Read native `sample_args.json` and `sample_outputs.json` for the effective
request and its status. Re-running a request regenerates its video and
overwrites matching output files. Use a fresh output directory to preserve
earlier results for comparison. Start a new sequence after changing the caption.

For the Stage 3 defaults file, use
`$SIM_TRANSFER_SKILL/inference/assets/sample_args.json`.
The [inference contract](../inference/references/inference-contract.md)
describes this profile. Generic rectified-flow step and shift flags do not
replace the stored four-transition distilled sampler.

Direct resize requires `condition_stream_encode=false`. For memory failures,
identify the failed allocation and reduce the request to a shorter clip at
the supported canvas. Preserve the selected example's history policy:

- `example2`: full history, `--kv-cache-inference-size 0 --attention-sink-size 0`.
- `example1` depth/segmentation long rollout: `--kv-cache-inference-size 30
  --attention-sink-size 3`, with a 601-frame processing window and
  `max_frames: 600`. All 600 frames are retained.

With model-size output, `vision.mp4` is the generated RGB. With original-size
output, `vision_model.mp4` retains model-native RGB. `transfer_geometry`
records source, model, output and encoder-padding dimensions. See
[output files](../inference/references/output.md).

## Training state

Choose full-state resume or a new weight-only start using
[checkpoint operations](../post-training/references/checkpoint-and-resume.md).
`load_ema_to_reg=True requires load_training_state=False` means an EMA warm
start was combined with optimizer continuation. Use EMA with fresh state for
a new run, or restore regular weights and their optimizer for resume.

If a new run resumes an old iteration, its existing run identity may have
selected a latest checkpoint. Use a new run name for an intended warm start.
Model-only exports do not restore optimizer or scheduler progress. For a
local Stage 0 data-cursor issue, inspect the composed callbacks for
`local_simulation_resume` and the corresponding saved state.

The [four-stage recipe](../post-training/references/local-training.md#framework-training-entry-points)
defines Stage 2 parent bindings and separate Stage 3 student/teacher paths.
Keep parent weights available across restart because construction loads them
first. If loss becomes nonfinite, capture the first failing operation and
compare its input, precision, masks, parents and optimization with the
[reference recipe](../post-training/references/canonical-recipe.md).
Stage 3 alternates one student update and five score-model updates in each
six-iteration cycle; its teacher remains frozen.

For a support request, include the first error, checkout and interpreter,
relevant input metadata, and the correction attempted. Exclude credentials
and private media paths.

## References

- [Model contract](../shared/model-contract.md)
- [Failure playbook](references/failure-playbook.md)
- [Environment requirements](../shared/environment.md)
- [Hugging Face examples](../shared/hf-sample.md)
- [Local training data contract](../shared/local-dataset.md)
- [Inference parameters and coupled settings](../inference/references/inference-contract.md)
- [Output files](../inference/references/output.md)
- [Checkpoint operations and resume](../post-training/references/checkpoint-and-resume.md)
- [Stage configuration and training launch](../post-training/references/local-training.md)
- [Reference training recipe](../post-training/references/canonical-recipe.md)
