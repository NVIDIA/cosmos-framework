# Framework requests and sampling

Use `cosmos_framework.scripts.inference` with a complete local model snapshot.
Prepare a JSON request or a JSONL file with one request per line. Asset paths
resolve relative to that input file. For the reference scene, use
`$AUTO_SAMPLE_ROOT/input-transfer-auto.jsonl`.
Keep the original files and create corrected/cropped variants in a new directory.

## Camera request

This two-camera schema illustrates custom local assets. For the published
seven-camera scene, use its supplied request without rewriting the paths.
Replace captions with descriptions of the inspected scene. Initial images may
be supplied directly; `condition_video_as_image` takes the first frame from RGB
videos. All cameras use consistent conditioning.

```json
{
  "model_mode": "image2video",
  "prompt": "Follow the per-camera controls and captions.",
  "resolution": "480",
  "fps": 30,
  "num_frames": 297,
  "guidance": 3.0,
  "shift": 10,
  "num_steps": 35,
  "seed": 0,
  "wsm": {},
  "multiview": {
    "load_caption_from_data": true,
    "condition_video_as_image": true,
    "use_first_chunk_only": false,
    "views": [
      {
        "camera_key": "camera_front_wide_120fov",
        "control_path": "wsm/camera_front_wide_120fov.mp4",
        "vision_path": "rgb/camera_front_wide_120fov.mp4",
        "caption_path": "captions/camera_front_wide_120fov.json"
      },
      {
        "camera_key": "camera_front_tele_30fov",
        "control_path": "wsm/camera_front_tele_30fov.mp4",
        "vision_path": "rgb/camera_front_tele_30fov.mp4",
        "caption_path": "captions/camera_front_tele_30fov.json"
      }
    ]
  }
}
```

For WSM text-to-video, set `model_mode` to `text2video` and omit the
`vision_path` fields and video-as-image setting. Camera identity comes from
`camera_key`, not the top-level prompt. Selected cameras retain their physical
IDs from the [identity table](../../post-training/references/data-contract.md#camera-identities).
Keep the explicit list to preserve a custom subset.

## Caption sidecars

Each camera's `caption_path` points to its own file:

```json
{
  "caption_structured": {
    "chunk_0_297": {
      "start_frame": 0,
      "end_frame": 297,
      "caption": "Camera-local description of this inspected interval."
    }
  }
}
```

Bounds are start-inclusive/end-exclusive on the RGB clock. The example body is
a schema illustration, not an authored caption. Let the runtime add camera,
resolution and FPS wrappers once. A scene-level `prompt` does not replace the
per-view files.

Prepare a synchronized crop and rebase its caption bounds to zero when selecting
another source interval. Check every camera's coverage. Transfer-Auto's
caption-loading path derives intervals from caption chunks, so changing
`num_frames` alone is insufficient. A 300/400-frame caption span can become
297/397 frames on the RGB tokenizer's `1+4k` grid. Use one aligned caption chunk
for the first comparison and inspect actual output duration.

## Joint LiDAR

The supplied daytime request already includes LiDAR. For your own camera request,
add the following top-level block. Do not add a `joint_generation` field; it is
not part of Framework's request schema:

```json
"lidar": {
  "control_path": "lidar/hdmap_rangemap.tar"
}
```

This is a fragment to insert into the object, not a standalone JSON request.
Omit the entire LiDAR block for RGB-only generation. Prepared range-map archives
preserve source sweep IDs. Supported local tensor controls also include FP32
safetensors with `frames[3,T,H,W]`: metric range, intensity and validity. Do not
substitute an MP4 rendering. The tokenizer applies its own padding/normalization.

Optional `lidar.condition_path` supplies measured initial LiDAR;
`lidar.num_conditional_sweeps` sets the target prefix length. This is separate
from the HD-map control. Check source IDs, range projection and clock before
using a prefix. Sim-Auto permits only its supported initial prefix (see its input
skill); do not transfer broader bidirectional options automatically.

At 30 FPS RGB and 10 Hz LiDAR, 297 inference frames correspond to 99 sweeps.
The bidirectional path's rounded count differs from training's endpoint-inclusive
61/21 and 397/133 windows. Inspect returned IDs and source coverage instead of
claiming identical endpoint conventions.

## Sampling

Start with these settings for the released **Transfer-Auto** model. Set them
explicitly in your request, then compare changes on your own scenes.

| Setting                    | Transfer-Auto starting point                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| Resolution / FPS           | 720p / 30 for the reference example; 480p for lower memory use                            |
| Aspect ratio               | `"16,9"`: 832×480 or 1280×720 per camera                                                  |
| Frames                     | 61 for a short check; 297 for the long example                                            |
| Steps / shift / seed       | 35 / 10 / 0                                                                               |
| Guidance                   | 3; 1.5 is a lower-guidance comparison                                                     |
| Control guidance           | 1; required by the multiview RGB-only route                                               |
| Per-view negatives         | Bundled [negative text](../assets/negative-prompt.txt) through `per_view_negative_prompt` |
| Distributed example        | CP4 / CFGP1                                                                               |
| Optional compilation/cache | Disabled initially; compare before enabling                                               |

### GPU memory and parallelism

Size the allocation for the **selected model and request**: per-GPU memory,
GPU count and interconnect, camera count, resolution, duration, RGB-only versus
joint LiDAR, and sampling settings all matter. A four-GPU command is not a
guarantee that every request fits on every four-GPU system. Transfer-Auto and
Sim-Auto have different execution and memory requirements; do not transfer a
memory or speed estimate from one to the other.

**Context parallelism (CP)** distributes sequence computation across GPUs.
Increasing `--cp-size` can reduce per-GPU activation memory and accelerate large
requests, but adds communication overhead: more CP or more GPUs does **not**
guarantee lower latency. **FSDP** shards model weights; it addresses a different
part of memory use and does not eliminate activations, tokenizer or output
buffers. Keep the Auto examples' explicit CP/CFGP settings as the starting point;
Sim-Auto requires `--cfgp-size 1`.

For a lower-memory first run, use 480p, fewer selected cameras or a shorter
aligned interval, with matching controls and caption spans. Then check the
intended full request. Compare parallelism settings using the same model, inputs,
steps and compilation/cache options. Measure both generation time and time to
saved outputs: a denoising speedup need not produce the same end-to-end speedup.

Training also stores gradients, optimizer state and backward activations, and
distillation can hold additional networks. An inference allocation is therefore
not a training-memory recommendation. Use the separate
[training memory guidance](../../shared/framework-runtime.md#training-memory).

### Resolution and aspect ratio

Both **Transfer-Auto and Sim-Auto support 480p and 720p**. Use the landscape
`aspect_ratio: "16,9"` preset (CLI `--aspect-ratio 16,9`) to best match the
training image geometry and give the model its best expected out-of-the-box
quality:

| Request resolution | Per-camera output size | Match to training                        |
| ------------------ | ---------------------- | ---------------------------------------- |
| `"480"`            | 832×480                | Training's 480p size; approximately 16:9 |
| `"720"`            | 1280×720               | Training's 720p size; exactly 16:9       |

The 480p preset is **not** 854×480: use the runtime's 832×480 size. These are
per-camera dimensions, not the dimensions of a collage. Match RGB and WSM spatial
transforms; changing aspect ratio by cropping or stretching also changes the
effective camera geometry. Preserve calibration/alignment when preparing inputs.
Prefer these training-matched landscape sizes for Auto.

720p has about **2.3× as many pixels per camera** as 480p, so plan for greater
memory and compute. Use 720p for more spatial detail, or 480p for lower resource
use. For a larger deliverable, generate at a supported size and upscale a separate
output copy.

### Generation length and long rollouts

An example's duration is not a model limit. At **30 FPS**, roughly 300 frames
cover 10 seconds and 900 cover 30 seconds. The shared sample generates **297
frames (9.9 seconds)**. Use actual saved frame counts: tokenizer alignment,
caption spans and continuation overlap can change the final duration.

**Transfer-Auto** was trained on 400-source-frame chunks at 30 FPS, about
**13.3 seconds**; the RGB tokenizer aligns a full window to 397 frames. A
10-second request is within that temporal scale and is not inherently worse
because it is shorter. A 20- or 30-second sequence goes beyond a single training
window. Use chunkwise continuation for longer RGB sequences and inspect
object identity, motion, geometry and transitions between chunks.

For **RGB-only Transfer-Auto**, Framework supports caption-chunk continuation:
provide matching chunk boundaries in every camera's caption file, controls
covering the full sequence, and `multiview.use_first_chunk_only: false`. Each
chunk is denoised bidirectionally, with generated tail frames conditioning the
next chunk. A top-level `num_frames` edit does not split a long caption into
chunks. Keep chunks near or below the training window and inspect the actual
output duration and seams. **Joint RGB/LiDAR Transfer-Auto currently generates
one caption chunk per request**. Supply one aligned caption interval, or set
`multiview.use_first_chunk_only: true` to select the first interval. Multiple
caption chunks with that option disabled are rejected for joint generation.

**Sim-Auto** generates small causal latent chunks with rolling history, so its
requested horizon need not fit one Transfer-Auto training window. Qualitative
testing has shown **rollouts lasting a few minutes without significant visible
quality loss**. Ten- and thirty-second horizons use the same rolling mechanism;
provide synchronized controls and captions for the entire horizon and inspect
late frames, object persistence and cross-view consistency, not only the start.
Budget additional compute and input/output memory for longer runs.

### Source timing, model FPS and playback FPS

These are three different quantities. The released Auto recipe uses **30 FPS
camera data**. Match the request FPS to the effective sampling rate of your inputs.

| Quantity                    | Meaning                                                                                                              | What to check                                                                                                 |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Source sampling rate        | Actual time between the RGB/WSM observations: 100 ms at 10 FPS, about 33 ms at 30 FPS                                | Timestamps, frame selection and synchronization; an MP4 header alone is insufficient                          |
| Request `fps` / CLI `--fps` | The camera rate assumed by the model; it affects temporal-position/mRoPE scaling and formatted FPS/duration metadata | Match the effective source sampling rate for correctly timed inference                                        |
| Saved-video playback rate   | How quickly the generated frames are displayed                                                                       | Framework's Auto saver currently uses the request FPS; changing playback afterward does not change generation |

There is no separate `multiview.output_fps` request field. For a playback-only
comparison, create a separate viewing copy after generation
with the same generated frames. Preserve the original outputs and request.

Changing `--fps` does not resample multiview controls: the loader selects decoded
frames by index. Do not relabel 10-FPS recordings as 30 FPS to claim compatibility.
Actual conversion requires an explicit, synchronized resampling policy for all
affected inputs; duplicated/interpolated frames do not recover missing observations.
Recompute caption frame bounds when frame indices change, preserve event times,
and check any supplied RGB conditions against the resulting control timeline.

Count and duration must agree: 297 saved frames play for 9.9 seconds at 30 FPS,
but 29.7 seconds at 10 FPS. Check resolved caption spans, frame count and saved
video metadata; a top-level frame-count edit alone may not select a new interval.

#### Diagnose a quality drop with 10-FPS camera data

1. Confirm the actual source timestamps and which frames the loader consumes.
   If an original 30-FPS recording exists, retain a correctly conditioned
   30-FPS baseline for the same physical scene interval. A comparison that
   changes both source sampling and model FPS does not isolate either cause.
2. For an **RGB-only** test, use aligned 10-FPS RGB/WSM with request `fps: 10`.
   Keep the checkpoint, cameras, spatial transforms, captions, seed and sampling
   settings fixed. Check duration and every camera's motion and geometry.
3. Optionally repeat the **same 10-FPS frame sequence** with request `fps: 30`,
   keeping the same frame count and inputs. This deliberately changes the model's
   timing assumption and derived caption metadata. Inspect the effective captions
   too; timestamped text can now conflict with that clock. View a separate copy
   at 10 FPS if needed. The model still interpreted the observed displacements
   as occurring **three times faster**; slower playback cannot undo that.
   Use this diagnostic comparison to isolate FPS conditioning, not as a quality fix.

Use a fresh output directory per case and record source sampling, requested FPS,
actual generated frame count, caption timing and viewing-copy playback rate.
For domain adaptation, use the [Transfer-Auto post-training guide](../../post-training/SKILL.md)
and evaluate the result on held-out recordings at your target sampling rate.
For Sim-Auto, adapt Transfer-Auto before causal distillation.

**Joint RGB/LiDAR needs a separate timing check.** The reference is 30-FPS
cameras with 10-Hz LiDAR; 10-Hz LiDAR does not imply 10-FPS cameras. Camera FPS
also determines sweep counts and source-frame alignment. Current joint Sim-Auto
rejects camera FPS other than 30. Do not bypass that check or apply the relabeling
experiment to joint inputs; preserve the supported sensor clocks and verify
range-control IDs.

### What guidance means

`guidance` is the strength of **classifier-free guidance (CFG)**: the sampler
compares a prediction with your positive caption against an empty or negative
caption prediction and amplifies their difference. At **1**, it uses the positive
prediction without that extra amplification; the model still reads the caption
and controls. Above 1, CFG adds a second prediction branch and extra computation.

Start at **3**. Compare **1.5** with the same seed and inputs if the result looks
overly forced or less natural. Stronger guidance can improve prompt adherence,
but it is not an across-the-board quality increase and can amplify artifacts or
prompt contradictions. Neither value fixes incorrect view IDs or misaligned WSM.
`control_guidance` is a separate control-CFG setting, not the text-guidance knob;
keep it at **1** for this baseline. WSM conditioning remains active at 1.

### What shift and steps mean

`shift` changes the **noise-level schedule** used during denoising. A larger shift
places more sampling points toward higher noise levels, changing the balance
between early structure formation and later refinement. It does not move a video
in time, change camera synchronization or add frames. Keep **shift 10** initially.

`num_steps` controls the number of denoising steps. **35** is the starting recipe;
fewer steps reduce compute. Compare step counts using the same inputs and seed.
Keep shift and steps fixed while first
comparing guidance or captions, so the cause of any change is clear.

### What a negative prompt means

A negative prompt describes an unwanted result used for the CFG comparison
branch. It is a preference signal, not a hard prohibition or a safety filter.
On the per-view Auto path, one `per_view_negative_prompt` string is shared across
selected cameras. If omitted, that branch uses empty captions. At **guidance 1**,
negative text has no CFG effect.

Read the bundled [negative prompt](../assets/negative-prompt.txt) and put its text
in the top-level `per_view_negative_prompt` field for an artifact-reduction
comparison. Plain `negative_prompt` is not equivalent on the per-view path.
Review the text first: the supplied example includes severe weather obscuration
and lens effects. Omit or revise conflicting clauses if those effects are part
of your requested scene. There is no numeric "best negative prompt"; compare
with and without a compatible negative at fixed guidance, seed and inputs.

### Sim-Auto uses a distilled schedule

For the released **Sim-Auto** student, use **guidance 1**, **control guidance 1**
and **normalize_cfg false**. Guidance behavior is already distilled into its
weights. Its checkpoint supplies the fixed four-step schedule; do not transplant
Transfer-Auto's 35-step/shift-10 recipe or expect a negative prompt to add CFG.
Follow the [Sim-Auto inference guide](../../../cosmos3-nano-sim-auto/inference/SKILL.md)
for its initial observations and rolling-generation rules.

### Compare settings on a fixed scene

First verify camera mapping, WSM alignment and
[training-aligned captions](../../shared/caption-policy.md#keep-training-and-inference-captions-aligned).
Then compare guidance 3 versus 1.5, or a compatible negative versus none, changing
one factor at a time. Keep the checkpoint, seed, crop, duration and initial-RGB
conditioning fixed; use a fresh output directory per variant. Inspect side/rear
views and temporal/cross-view consistency as well as prompt adherence. Check
`sample_args.json` for resolved values before interpreting the comparison.

An HF snapshot contains the exported weight variant. An inference EMA flag
cannot recover another variant that is absent from the artifact. Preserve the
snapshot's attention, tokenizers and caption metadata.

## Inspect outputs

Check the actual per-camera videos and `sample_args.json`/`sample_outputs.json`.
Inspect saved control videos when present. The current joint Transfer saver
does not write control videos; use the corresponding source WSM with the
same resolved frame interval and spatial transform for overlays.
Joint output includes numeric
`lidar.safetensors` with range, intensity and frame IDs. Inspect numeric data as
well as previews, checking geometry, motion and cross-view consistency.

Re-run changed settings into a fresh output directory. Compare one factor at a
time, keeping checkpoint, seed, input geometry and conditioning mode fixed.
