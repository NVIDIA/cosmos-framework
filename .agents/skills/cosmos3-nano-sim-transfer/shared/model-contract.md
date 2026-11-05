# Cosmos3-Nano-Sim-Transfer model contract

Cosmos3-Nano-Sim-Transfer generates a single RGB video from a caption and one
spatial control video. For example, a depth rendering of a moving robot arm
controls its layout and motion while the caption describes the arm, materials,
and surroundings. The output is a single-camera RGB video.

This is the shared reference for [data preparation](data-contract.md),
[post-training](../post-training/SKILL.md), and
[inference](../inference/SKILL.md). Use the Framework
checkout from [setup](setup.md) and its matching model
configuration.

## One control and one target

| Control      | Meaning                                | Preparation requirement                                                                                                           |
| ------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Edge         | Three-channel rendering of image edges | Keep the edge extraction convention matched between training and inference. The reference medium Canny thresholds are 100 and 200 |
| Blur         | Low-frequency RGB appearance           | Match the reference blur/downsampling transform. An arbitrary Gaussian blur is a different input distribution                     |
| Depth        | Three-channel depth rendering          | Supply rendered depth matching the training encoding, near/far polarity and range. Keep camera and timestamps aligned             |
| Segmentation | Color-coded region masks               | Preserve palette and object partition. Integer class IDs are not interchangeable with rendered masks                              |

The model receives control first and RGB target second, with shared temporal
positions. Each pixel-video item is shaped `[3, F, H, W]`, where `F` is the
pixel-frame count. Pixels are normalized as
`pixels / 127.5 - 1`, producing floating-point values in `[-1, 1]`. Depth retains
fractional values after resizing. The caption describes the same selected
window. Training predicts RGB conditioned on the control.

Controls and targets must depict the same camera, times, and field of view.
Inspect the paired videos to confirm alignment. Prepare each intended control
using the [data-preparation guide](data-contract.md).

## Geometry

Use direct resizing that preserves the complete field of view. Select the
least-distorted trained canvas and reject aspect distortion above 3%. Width
and height depend on aspect ratio and orientation. The 480 tier includes
the following canvases:

| Tier | Square    | Landscape 4:3 | Portrait 3:4 | Landscape 16:9 | Portrait 9:16 |
| ---- | --------- | ------------- | ------------ | -------------- | ------------- |
| 480  | 640 × 640 | 736 × 544     | 544 × 736    | 832 × 480      | 480 × 832     |

RGB uses bicubic interpolation with antialiasing. Depth uses float32
bilinear interpolation with antialiasing; segmentation uses nearest neighbor.
Native training and inference construct edge and blur on the fly from
resized RGB. Preserve the full field of view without reflection padding.
Stage 0 uses 480/720; Stages 1–3 use 480.

## Time and history

The video autoencoder compresses time by four and space by sixteen. A reference
window contains `F = 1 + 4n` RGB frames, which encode to `T = 1 + n` latent
frames: 121 RGB frames become 31 latents. 201 become 51. The first latent
represents one RGB frame and each subsequent latent represents four. Keep
pixel-frame and latent-frame settings distinct.

Causal generation predicts one latent at a time. At time `t`, control queries
see controls through `t` and completed RGB before `t`. Target queries also see
their current noisy target. Future control and RGB are masked. Completed clean
RGB is processed again to populate the attention cache used for the next
prediction. Inference starts from noise with no real RGB history prefix.

The `example2` inference profile keeps all generated history with no
attention sink. The depth/segmentation long-rollout profile for `example1`
uses a 30-latent history window and three sink entries. Its 601-frame
processing window with `max_frames: 600` preserves all 600 source frames at
30 FPS. Stage 3 training uses a 51-latent window with one pinned
initial latent. Preserve each configuration's setting. For a finite window `W`
and sink `S`, `W >= 1` and `0 <= S < W`. There are `W - S - 1` recent past latents,
plus the pinned initial entries and current latent. Control and RGB together
require `2W` logical cache entries. Eviction must not reset the temporal
position clock. Full history requires `S = 0`.

The local training loader resamples the complete selected caption window to
its length bucket and shares the resulting frame indices between RGB and
control. See the [local data contract](local-dataset.md) for exclusive bounds
and effective FPS.

Decoded input timestamps and frame rate determine the reference output clock.
Changing a request's frame-rate label does not resample its media. Select the
same source times for paired videos and describe that time interval in the
caption.

## Sampling and prompt meaning

The Stage 3 method uses the saved four-step stochastic sampler with
times `[1, 15/16, 5/6, 5/8]`, followed by terminal zero. This schedule is already
shifted. Keep text guidance and control guidance at 1. A generic rectified-flow
step count or shift does not replace this distilled schedule.

The training recipe samples structured JSON 93% of the time and the same
annotation's `temporal_caption` 7% of the time after text retention. Stage 3
has zero text dropout.

For distilled Stage 3 inference, use one caption per generated sequence.
Preserve the Transfer prompt formatter and its effective duration, frame rate,
and canvas metadata. Do not insert control-emphasis text or negative metadata:
the distilled sampler has no negative-caption branch. Teacher CFG during
training uses the separate [negative-prompt policy](../post-training/references/canonical-recipe.md).
Start a new request for a different caption.

## Model and output handoffs

Use Cosmos3-Nano-Sim-Transfer weights together with the model configuration,
matching runtime, tokenizer and video autoencoder from the documented
Hugging Face downloads in [setup](setup.md).
For a native training checkpoint, select the completed exponential moving
average (EMA) weights for inference. The documented HF release export already
contains the selected weights. Load its directory with its own `config.json`
and no EMA flag, as shown in [checkpoint selection](../inference/SKILL.md#1-select-the-matched-inference-checkpoint).

Record the actual model version or content identity, selected weight variant,
input and control, prompt, seed, effective canvas, frame rate, frame count, and
sampling settings with each result. Generated tensors are RGB
`[1, 3, F, H, W]` in `[-1, 1]` before saving. Native request metadata records
the output geometry; see [output files](../inference/references/output.md).

Keep output at model dimensions separate from a copy restored to the source's
dimensions. Compare matching dimensions and preprocessing when evaluating
models. A tiled input/control preview is not the generated RGB sample.
