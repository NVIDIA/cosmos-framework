# Prepare your control videos and captions

Sim-Transfer learns RGB generation from an aligned **control video, target RGB
video and caption for the same interval**. Start with the Hugging Face
[`example2` assets](hf-sample.md) and build the native local dataset using
[dataset preparation](local-dataset.md). The HF sample supplies source videos
and prompts; local preparation creates the training bundle and length index.

For inference, RGB supplies online edge or blur controls or a comparison
reference. It does not automatically become a real-RGB history prefix.

## Collect aligned source clips

Keep source identity, capture FPS, frame count, camera and any crop or
calibration with each example. Split by source recording or episode before
creating overlapping clips. Hold validation videos and captions fixed when
comparing adaptations. Retain source resolution so each stage can render its
own model canvas.

| Control      | Meaning and preparation                                                                                                           |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Edge         | Canny contours repeated across three channels, derived on the fly from resized RGB. The inference preset uses thresholds 100/200. |
| Blur         | Low-frequency appearance derived on the fly by the native blur/downsample transform on resized RGB.                               |
| Depth        | Matching three-channel relative-depth rendering, including normalization and near/far polarity; not metric distances.             |
| Segmentation | Matching color palette and object partition; integer class IDs need conversion to the expected rendering.                         |

Include RGB, depth and segmentation for every record to reproduce an equal
four-modality mixture. The native training and inference paths derive edge and
blur from RGB; no separate edge or blur assets are required. The training loader
samples one available control per draw. Missing depth or segmentation changes
that mixture.

Paired videos must share geometry, frame count, FPS and relative timestamps.
Apply any crop or retiming identically to all streams.

## Attach captions to the exact interval

The structured annotation and its nonempty `temporal_caption` describe the
same RGB interval. After retaining text, the reference recipe selects
structured JSON 93% and its dense temporal caption 7% of the time. Caption
dropout is 0.1 in Stages 0–2 and zero in Stage 3; control dropout is zero.
Native stage overrides apply these values over the local bundle's zero-dropout
default.

For `example2`, use `example2_upsampled_prompt.json` directly as the structured
annotation and its `temporal_caption` as the dense branch. The separate
`example2_prompt.txt` is the source prompt, not that dense branch. For new clips,
the [captioning guide](../post-training/references/caption-guide.md) writes `caption.json`
and a matching dense `caption.txt`. See the [prompt guide](prompt-guide.md).

The locally created JSONL row uses this format:

```json
{"identity":"clip_0001","rgb":"videos/clip_0001.mp4","depth":"depth/clip_0001.mp4","seg":"seg/clip_0001.mp4","caption_path":"captions/clip_0001.json","caption_start":0,"caption_end":121}
```

Paths resolve relative to the manifest. Bounds are **exclusive-end** and must
span the complete decoded clip, `[0,N)`. Preparation attaches the structured
caption after native indexing and removes saved output metadata so the loader
can append sampled values. For a shorter annotation, export all paired videos
to that interval before preparation. Generic caption-converter JSONL uses a
different format and inclusive ends.

## Preserve geometry and time in the loader

Direct-resize the full field of view to the least-distorted trained canvas.
The native direct-resize guard permits aspect distortion up to 3%. Avoid
crops or reflected borders. Examples of the 480-tier canvases:

| Aspect                         | Width × height        |
| ------------------------------ | --------------------- |
| Square                         | 640 × 640             |
| Landscape 4:3 / portrait 3:4   | 736 × 544 / 544 × 736 |
| Landscape 16:9 / portrait 9:16 | 832 × 480 / 480 × 832 |

RGB uses bicubic antialiasing. Depth uses float32 bilinear antialiasing;
segmentation uses nearest-neighbor. Online edge and blur use resized uint8 RGB.

**Edge/blur preprocessing:** Resize RGB to the model canvas **before**
computing edge or blur controls. For 480-tier inference and Stages 1–3, use the
480-tier canvas above. Computing controls at source resolution and then resizing
the maps can produce suboptimal generation quality. Stage 0 retains its mixed
480/720 streams; apply the same resize-first order at the selected tier.

For an exclusive caption window `[a,b)` and target count `F > 1`, the indexed
loader uses `s = (b-a-1)/(F-1)` and indices `a + round(i*s)`. RGB and control
share the first and last source frames and every selected index. Effective FPS
is source FPS divided by `s`; fractional stride can give unequal integer gaps.
The local training policy resamples complete windows to its supported horizons,
up to 201 frames at 480 and 149 at 720.

`example2` contains 121 source frames at 30 FPS and 1920 × 1080 pixels. Frames
0–120 span 4 seconds; frame-count/FPS duration is about 4.033 seconds. Its
480-tier canvas is 832 × 480. Changing an inference `fps` field does not
resample controls.

## Model-facing data

The native loader emits control then target RGB, each `[3,F,H,W]` in `[-1,1]`,
with shared temporal positions. Temporal compression four maps `F=1+4n` to
`1+n` latent frames; 121 frames become 31 latents. Spatial compression is 16
with 48 latent channels. The loader adds sampled FPS, duration and canvas to
the chosen caption representation.

Training draws zero/one/two real-RGB prefix latents with probabilities
0.7/0.2/0.1, capped to leave at least one generated target. Supervised loss
belongs to generated RGB positions, excluding any conditioned prefix.
Consistency distillation keeps real history; self-forcing rolls out generated
history and uses the frozen teacher plus trainable score model.

Stage 0 uses separate 480/720 streams with weights 4/2 and variable packing.
Causal stages emit one pair per data-parallel rank. The native recipes own
these settings; see [local training](../post-training/references/local-training.md).
