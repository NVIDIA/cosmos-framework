# Troubleshoot Cosmos3-Nano-Sim-Transfer

> **Skill note (not in the source):** this file holds the opening guidance and the Runtime, Model and runtime, Media and data and Inference sections of the Sim-Transfer troubleshooting skill, the matching rows of its failure playbook, and the support-request paragraph that closed "Training state" in the source (here under "Support request"). "Training state" is in the [Sim-Transfer troubleshooting skill](../../cosmos3-nano-sim-transfer/troubleshooting/SKILL.md#training-state).

Start with the first failing operation and preserve the last completed
checkpoint and existing outputs. Compare the effective inputs and model
identity with the [shared contract](model-contract.md).

| Symptom                                                  | Reference                               |
| -------------------------------------------------------- | --------------------------------------- |
| Import failure, CUDA unavailable, missing library        | [Runtime](#runtime)                     |
| Unknown model alias or incompatible weights              | [Model and runtime](#model-and-runtime) |
| Missing, distorted or drifting control                   | [Media and data](#media-and-data)       |
| Ignored flags, overwritten output or unexpected geometry | [Inference](#inference)                 |

The [failure table](#runtime-input-and-inference) lists specific errors.

## Runtime

Identify the checkout and interpreter used by the failing command:

```bash
cd "${FRAMEWORK_DIR:?Set the Cosmos Framework checkout root}"
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m cosmos_framework.scripts.inference --help
```

Compare with [environment requirements](environment.md).
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

Follow [checkpoint selection](../SKILL.md#1-select-the-matched-inference-checkpoint).
For native DCP, select EMA from the completed checkpoint. For the HF release,
pass its directory and own `config.json` without an EMA flag. Obtain the
matching model, tokenizer and video autoencoder through the
[Hugging Face setup](setup.md).

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
on [Hugging Face](hf-sample.md); native training and inference derive
edge and blur from RGB.

Each training draw selects one control. A numeric stratum identifies a
temporal length bucket, not a modality. Local-caption bounds are exclusive:
`[0,121)` means frames 0 through 120. Generic video-caption JSONL instead uses
inclusive ends. Follow the [local data contract](../../cosmos3-nano-sim-transfer/shared/local-dataset.md).

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
`$SIM_TRANSFER_SKILL/assets/sample_args.json`.
The [inference contract](inference-contract.md)
describes this profile. Generic rectified-flow step and shift flags do not
replace the stored four-transition distilled sampler.

> **Skill note (not in the source):** the path above was `$SIM_TRANSFER_SKILL/inference/assets/sample_args.json` in the source; the defaults file moved into this skill's assets folder.

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
[output files](output.md).

## Runtime, input and inference

| Symptom                                                                                                                                             | Cause                                                                       | Correction                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Unknown Cosmos target alias` or unsupported model class                                                                                            | Runtime and exported config disagree                                        | Use the matching runtime; retain the artifact's model class                                                   |
| Missing Framework module                                                                                                                            | Wrong interpreter, incomplete dependencies or an older checkout             | Follow setup; clear inherited `PYTHONPATH` and use the pinned Framework environment                           |
| Missing `COSMOS_SIM_TRANSFER_STAGE0_CHECKPOINT`, `COSMOS_SIM_TRANSFER_STAGE2_CHECKPOINT` or `COSMOS_SIM_TRANSFER_LOCAL_BUNDLE` during DCP inference | Hydra resolves the full training recipe before disabling training-only work | Set the training run's parent and data bindings from the DCP selection block                                  |
| Missing Stage 3 defaults file                                                                                                                       | `--defaults-file` points to an absent file                                  | Pass the skill's `assets/sample_args.json` to `--defaults-file`                                               |
| Missing data file                                                                                                                                   | A local media path is incorrect or the file was moved                       | Use the pinned [HF assets](hf-sample.md) and rebuild the local requests or dataset                            |
| Only one control appears in a training draw                                                                                                         | The native loader samples one modality per record                           | Keep equal edge/blur/depth/seg weights for the example2 mixture                                               |
| No eligible training window                                                                                                                         | Caption bounds, decoded metadata or length policy disagree                  | Use exclusive whole-clip bounds and the native length-index builder                                           |
| Distorted geometry or aspect check failure                                                                                                          | Wrong canvas, mismatched crop or more than 3% distortion                    | Select an eligible canvas and preserve the full field of view                                                 |
| RGB/control motion drifts                                                                                                                           | Different frame selection, FPS, timestamps or independent retiming          | Restore the shared source clock and interval                                                                  |
| `Direct Transfer resizing requires condition_stream_encode=false`                                                                                   | Streaming control encoding conflicts with the selected resize               | Pass `--no-condition-stream-encode`                                                                           |
| Denoising step/shift flags have no effect                                                                                                           | Distilled Stage 3 uses its fixed sampler                                    | Preserve the stored four-step stochastic schedule                                                             |
| CUDA out of memory on a longer input                                                                                                                | Full-history cache growth or decoded tensors                                | Identify the allocation and shorten the request; retain the selected example's history policy                 |
| example1 stops before frame 600                                                                                                                     | A reduced decoding cap or trimmed input was used                            | Restore the complete HF input, 601-frame processing window and `max_frames: 600` for depth or segmentation    |
| Unexpected output dimensions                                                                                                                        | Source-size restoration, control display or encoder padding                 | Read native `transfer_geometry`; use `vision_model.mp4` for model-size comparison when restoration is enabled |
| Previous results are overwritten by a repeated request                                                                                              | Inference regenerates requests and writes to the same matching output paths | Select a fresh output directory to retain earlier results for comparison                                      |

> **Skill note (not in the source):** the row "Missing Stage 3 defaults file" named the skill's `inference/assets/sample_args.json` in the source; the defaults file moved into this skill's assets folder.

## Support request

For a support request, include the first error, checkout and interpreter,
relevant input metadata, and the correction attempted. Exclude credentials
and private media paths.

## References

- [Model contract](model-contract.md)
- [Environment requirements](environment.md)
- [Hugging Face examples](hf-sample.md)
- [Local training data contract](../../cosmos3-nano-sim-transfer/shared/local-dataset.md)
- [Inference parameters and coupled settings](inference-contract.md)
- [Output files](output.md)
