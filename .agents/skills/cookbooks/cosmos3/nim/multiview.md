<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Generate multiview driving video and LiDAR with the Cosmos3 Certified NIM

Use this page to generate a synchronized multi-camera driving clip, and optionally a matching LiDAR sequence, with the Cosmos3-Nano-Transfer-Auto model.

Multiview is a **Generator** model variant. It uses the same synchronous JSON
`POST /v1/infer` endpoint as [Generation](generation.md) and
[Transfer](transfer.md), with a top-level `multiview` object that carries one
World Scenario Model (WSM) control video per camera. A WSM control is a
structured rendering of the scene: lane lines, road surface, and vehicle
bounding boxes.


| Section                                                                 | Output                                                     | Driven by                                                                                                  |
| ----------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [Camera generation](#camera-generation)                                 | One MP4 per camera                                         | Per-camera WSM controls, optionally with one frame per-camera RGB conditioning                             |
| [Joint camera and LiDAR generation](#joint-camera-and-lidar-generation) | One MP4 per camera plus a numeric LiDAR range-image tensor | Per-camera WSM controls plus a numeric LiDAR control, optionally with one frame per-camera RGB conditioning |


> Multiview requires a NIM image whose manifest contains `multiview` profiles.
> Confirm availability for the selected image in `/v1/manifest` before
> planning a deployment. The other Generator variants do not accept
> `multiview` requests, and the multiview variant accepts nothing else.



## Camera rig

The model generates the eleven cameras of the MADS autonomous-vehicle rig.
Requests name cameras by `camera_key`; the NIM always generates and returns
them in this canonical order regardless of request order:


| Index | `camera_key`                  |
| ----: | ----------------------------- |
| 0     | `camera_front_wide_120fov`    |
| 1     | `camera_cross_right_120fov`   |
| 2     | `camera_rear_right_70fov`     |
| 3     | `camera_rear_tele_30fov`      |
| 4     | `camera_rear_left_70fov`      |
| 5     | `camera_cross_left_120fov`    |
| 6     | `camera_front_tele_30fov`     |
| 7     | `camera_front_fisheye_200fov` |
| 8     | `camera_left_fisheye_200fov`  |
| 9     | `camera_right_fisheye_200fov` |
| 10    | `camera_rear_fisheye_200fov`  |


The request schema accepts 1 to 11 unique cameras. The scripted examples send the seven
cameras of the shared Auto sample described in [Prepare inputs](#prepare-inputs).

## Deploy the multiview Generator

Follow [Deployment](deployment.md) for NGC authentication, image pull, and
cache preparation, then launch the Generator with
`NIM_MODEL_VARIANT=multiview`:

```bash
docker run -d --name cosmos3-multiview \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_VARIANT=multiview \
  -e NIM_PERF_PROFILE=latency \
  -e NIM_ENABLE_TEXT_GUARDRAILS=0 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

Notes for this variant:

- **Profile.** Multiview profiles are BF16 or FP8. Run the
[pre-download profile preflight](deployment.md) with
`NIM_MODEL_VARIANT=multiview` to confirm a candidate profile, then expose
the GPU count the selected profile requires.
- **Request timeout.** An eleven-camera request is long-running, and a joint
camera and LiDAR request can exceed an hour on a single GPU. Multiview
profiles therefore default `NIM_TRITON_REQUEST_TIMEOUT` to four hours instead
of the 30 minutes used by other variants; set it at launch, in microseconds,
only to change that. The examples use a two-hour client timeout (`--timeout`).
See [Generator configuration](configuration.md#generator-configuration).
- **Attention backend.** `VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND` accepts
`maskless` or `triton`. Leave it unset normally: the runtime then selects
`maskless` when its bundled FlashAttention is available and `triton`
otherwise.
- **Text guardrail.** The command disables the text guardrail
(`NIM_ENABLE_TEXT_GUARDRAILS=0`) because the sample captions used by the
examples have not been qualified against the text blocklist. The output video
guardrails stay enabled. Disabling a guardrail reduces safety protections: do it only for
an approved, isolated evaluation. See [Guardrails](#guardrails).
- **Warmup.** `NIM_ENABLE_WARMUP=1` runs a synthetic multiview request before
readiness. Warmup is off by default.
- **Unsupported for multiview.** Prompt upsampling is not used by this variant.

To serve your own exported Cosmos3-Nano-Transfer-Auto checkpoint instead of
the bundled one, add `NIM_MODEL_PATH` as described in
[Bring your own checkpoint](bring-your-own-checkpoint.md). The export's
`model_index.json` must declare `"_class_name": "Cosmos3MultiviewPipeline"` and
its `transformer/config.json` must declare
`"backbone_type": "cosmos3_multiview"` with a `multiview` block.


## Verify the service

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Metadata must report the Generator runtime, `/v1/infer`, a selected profile
ID, and model variant `multiview` before you send a request.

## Prepare inputs

Install the [client tooling](prerequisites.md#client-tooling). The examples use
the daytime sample shared by the
[Cosmos3-Nano-Transfer-Auto](../simulation/auto/README.md) simulation cookbook.
It is published on Hugging Face with
[nvidia/Cosmos3-Nano-Transfer-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main/assets/sample_data)
and described in [`assets/daytime_scene`](../simulation/auto/assets/daytime_scene/README.md):

| Sample input                                     | Contents                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `daytime_scene/wsm/<camera_key>.mp4`             | WSM control for each of seven cameras, 300 frames at 30 FPS  |
| `daytime_scene/rgb/<camera_key>.mp4`             | Synchronized RGB recording; the NIM uses its first frame     |
| `daytime_scene/captions_with_timestamp/`         | Timestamped per-camera captions, used by default             |
| `daytime_scene/captions/`                        | Original paragraph captions                                  |
| `daytime_scene/lidar/control.tar`                | HD-map LiDAR range maps, 100 sweeps at 10 Hz                 |

Download and verify the sample, and convert its LiDAR range maps to the numeric
[LiDAR control](#lidar-control):

```bash
cd .agents/skills/cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
export AUTO_SAMPLE_ROOT="$(uv run python examples/mv_sample.py | tail -n 1)"
```

`mv_sample.py` uses the Auto cookbook helpers: it downloads only the inputs the
examples need, checks them against the sample's published `SHA256SUMS`, and
writes `examples/outputs/prepared/lidar_control.safetensors`. Later runs reuse
the Hugging Face cache. The scripts verify `AUTO_SAMPLE_ROOT` again, or
download the sample themselves when it is unset. They take each camera's caption
for the requested window from the caption JSON and convert local paths to data
URLs; `--caption-folder captions` selects the original paragraph captions.

Input rules for your own data:

- **Synchronized.** Every control must cover the same scene and time interval,
starting at the same instant. WSM clips are consumed from their first frame.
- **Long enough.** Each WSM control needs at least as many frames as the
requested output; shorter controls return HTTP 422 naming the camera.
- **Encoded.** Controls use the same raw base64, MIME-aware data URL, or
allowed HTTP(S) URL contract as other Generator media, up to 100,000,000
encoded characters per video. A server-local path is not accepted.
- **Captioned.** Every camera needs its own plain-text caption in
`views[].prompt`, describing what that camera sees. A top-level `prompt`, even
an empty one, returns HTTP 422. The NIM adds camera labels itself, so a caption that
contains a camera key, a `camera_view` label, control instructions such as
"follow the WSM", or a resolution or duration template returns HTTP 422. A
JSON-object caption is also rejected.
- **Negative prompt.** `multiview.per_view_negative_prompt` is required. It is
one string applied to every camera, and an empty string disables it.

The scripts parse their command line before doing endpoint or media work.
Before submitting a request, each requires `/v1/metadata` to report the
Generator runtime, `/v1/infer`, a selected profile ID, and model variant
`multiview`. Requests are large: every control and RGB clip is inlined as
base64, and the joint case adds about 370 MB of base64 LiDAR control, so allow
for that in proxy body limits.

## Camera generation



### Run a camera request

A camera-only request uses `model_mode=text2video`. `model_mode` describes RGB
conditioning only; the WSM control is a separate per-camera modality, so a
request with WSM controls and no RGB input is still `text2video`.

This request generates three front-facing cameras from the sample's WSM controls. Each
`control` is the camera's MP4 inlined as a data URL; the body is passed on
standard input because the encoded videos are too large for a command-line
argument. Each camera has its own short caption, the negative prompt is explicit, and
omitted generation fields take the [multiview
defaults](#generation-defaults-and-limits):

```bash
WSM=$AUTO_SAMPLE_ROOT/daytime_scene/wsm

curl -fsS -X POST "$NIM_URL/v1/infer" \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  -o /tmp/cosmos3-mv-response.json <<EOF
{
  "model_mode": "text2video",
  "seed": 42,
  "multiview": {
    "output_modality": "camera",
    "per_view_negative_prompt": "Low quality, inconsistent camera views, incorrect lane geometry, vehicles outside control boxes, flicker, warped objects.",
    "views": [
      {
        "camera_key": "camera_front_wide_120fov",
        "prompt": "Forward view in daylight along a dry two-way road behind a light-gray hatchback, with tall trees on a grassy embankment to the left and a sports-ground fence to the right.",
        "control": "data:video/mp4;base64,$(base64 < "$WSM/camera_front_wide_120fov.mp4" | tr -d '\n')"
      },
      {
        "camera_key": "camera_cross_left_120fov",
        "prompt": "Left-side view in daylight of oncoming cars passing in the adjacent lane, with a low stone wall, a grassy embankment, and tall trees beyond.",
        "control": "data:video/mp4;base64,$(base64 < "$WSM/camera_cross_left_120fov.mp4" | tr -d '\n')"
      },
      {
        "camera_key": "camera_cross_right_120fov",
        "prompt": "Right-side view in daylight of a narrow sidewalk, pale railings, sports-ground netting, and a side street passing by.",
        "control": "data:video/mp4;base64,$(base64 < "$WSM/camera_cross_right_120fov.mp4" | tr -d '\n')"
      }
    ]
  }
}
EOF

python3 - <<'PY2'
import base64
import json
from pathlib import Path

result = json.loads(Path("/tmp/cosmos3-mv-response.json").read_text())["multiview"]
for video in result["videos"]:
    Path(f"{video['camera_key']}.mp4").write_bytes(base64.b64decode(video["b64_video"]))
print(result["camera_order"], result["frames_per_view"], result["fps"])
PY2
```

Add more `views` entries, up to all eleven cameras, in the same form; the
sample provides seven.

Run the complete editable case, which sends the sample's seven cameras with
their per-camera captions and the model's bundled
[negative prompt](../../../cosmos3-nano-transfer-auto/inference/assets/negative-prompt.txt),
and checks the returned camera order:

```bash
uv run python examples/mv_rgb.py --case wsm_t2v
```

It writes one MP4 per camera to
`examples/outputs/mv_rgb_wsm_t2v/<camera_key>.mp4`.

### Condition on RGB first frames (image2video)

Set `model_mode=image2video` and add a `vision` input to **every** view. Each
view's `vision` conditions the first frame of its camera, and can be either:

- **an image** (JPEG, PNG, or WebP) of that first frame. Up to 20,000,000
encoded characters. (Recommended)
- **an RGB video:** the NIM decodes it and uses only its first frame. Send
the synchronized RGB clip unchanged; its length and FPS need not match the
WSM control. Up to 100,000,000 encoded characters. Sending video insted of
image may impact performance.

Run the RGB-conditioned case, which sends the sample's RGB clips as `vision`
videos alongside its WSM controls:

```bash
uv run python examples/mv_rgb.py --case i2v
```

Outputs are written to `examples/outputs/mv_rgb_i2v/`. A missing `vision` on any
view returns HTTP 422.

Multiview does not support `video2video`; RGB video conditioning beyond the
first frame is not available.

## Joint camera and LiDAR generation

Joint generation produces the camera videos and a LiDAR range-image sequence
in a single denoising pass. The model needs a numeric LiDAR control to produce
the LiDAR target, so the request adds both:

- `multiview.output_modality="camera_lidar"`; and
- `multiview.lidar.control`, a base64 safetensors file.

`multiview.lidar` is accepted **only** with `camera_lidar`, and `camera_lidar`
requires it. Every requested camera must still supply its WSM control.

### LiDAR control

The control is a safetensors file containing one contiguous FP32 tensor named
`frames` with shape `[3, sweeps, 128, 1800]`:


| Channel | Content   | Units / range                              |
| ------: | --------- | ------------------------------------------ |
| 0       | Range     | metres                                     |
| 1       | Intensity | `[0, 1]`                                   |
| 2       | Validity  | `[0, 1]`; values ≥ 0.5 mark a valid return |


Each sweep is a 128 × 1800 range image. LiDAR runs at its checkpoint-defined
10 Hz while cameras run at a fixed 30 FPS, so a 201-frame camera clip spans
6.7 seconds and consumes the first 67 sweeps. Supply at least that many; the
backend ignores trailing sweeps. The control must come from the same scene and
time interval as the WSM controls. The sample's control converts to 100 sweeps.

All values must be finite, range must be non-negative, and intensity and
validity must lie in `[0, 1]`. 

Send the file as raw base64 or as
`data:application/octet-stream;base64,...`, up to 853,333,336 encoded
characters. Do not send a LiDAR preview image in place of the tensor. Raw
HD-map range-map archives must first be projected into this numeric layout;
`mv_sample.py` and `mv_joint.py` do this for the sample's `control.tar` with the
Auto cookbook's
[`lidar_rangemap.py`](../simulation/auto/helpers/lidar_rangemap.py) converter.

### Condition on measured LiDAR sweeps (optional)

A joint request can also continue from measured LiDAR sweeps. Add
`multiview.lidar.condition`, a safetensors file in the same layout and transport
as the control: one FP32 `frames` tensor of shape `[3, sweeps, 128, 1800]`
holding measured range, intensity, and validity.
`multiview.lidar.num_conditional_sweeps` is an optional positive integer that
selects how many of those sweeps to use. It requires `condition`, and the
service uses one sweep when it is omitted. At least one sweep of the generated
sequence must remain after the conditioning sweeps.

Conditioning is optional, and the generated sequence starts from the measured
sweeps: a request with two measured sweeps reproduced them in the first two
generated sweeps to within about 0.08 m of range on the rays valid in both. The
reconstruction is approximate rather than exact. The examples download the
sample's LiDAR control but no measured sweeps, so `mv_joint.py` takes the file
as an option:

```bash
uv run python examples/mv_joint.py \
  --lidar-condition /path/to/measured.safetensors --num-conditional-sweeps 2
```



### Run a joint request

This request generates the same three cameras plus LiDAR from the sample. It
sets `output_modality` to `camera_lidar` and adds `lidar.control`, the file
that `mv_sample.py` prepared:

```bash
WSM=$AUTO_SAMPLE_ROOT/daytime_scene/wsm
LIDAR=examples/outputs/prepared/lidar_control.safetensors

curl -fsS -X POST "$NIM_URL/v1/infer" \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  -o /tmp/cosmos3-mv-joint-response.json <<EOF
{
  "model_mode": "text2video",
  "seed": 42,
  "multiview": {
    "output_modality": "camera_lidar",
    "per_view_negative_prompt": "Low quality, inconsistent camera views, incorrect lane geometry, vehicles outside control boxes, flicker, warped objects.",
    "views": [
      {
        "camera_key": "camera_front_wide_120fov",
        "prompt": "Forward view in daylight along a dry two-way road behind a light-gray hatchback, with tall trees on a grassy embankment to the left and a sports-ground fence to the right.",
        "control": "data:video/mp4;base64,$(base64 < "$WSM/camera_front_wide_120fov.mp4" | tr -d '\n')"
      },
      {
        "camera_key": "camera_cross_left_120fov",
        "prompt": "Left-side view in daylight of oncoming cars passing in the adjacent lane, with a low stone wall, a grassy embankment, and tall trees beyond.",
        "control": "data:video/mp4;base64,$(base64 < "$WSM/camera_cross_left_120fov.mp4" | tr -d '\n')"
      },
      {
        "camera_key": "camera_cross_right_120fov",
        "prompt": "Right-side view in daylight of a narrow sidewalk, pale railings, sports-ground netting, and a side street passing by.",
        "control": "data:video/mp4;base64,$(base64 < "$WSM/camera_cross_right_120fov.mp4" | tr -d '\n')"
      }
    ],
    "lidar": {
      "control": "data:application/octet-stream;base64,$(base64 < "$LIDAR" | tr -d '\n')"
    }
  }
}
EOF

python3 - <<'PY2'
import base64
import json
from pathlib import Path

result = json.loads(Path("/tmp/cosmos3-mv-joint-response.json").read_text())["multiview"]
for video in result["videos"]:
    Path(f"{video['camera_key']}.mp4").write_bytes(base64.b64decode(video["b64_video"]))
lidar = result["lidar"]
Path("lidar.safetensors").write_bytes(base64.b64decode(lidar["b64_frames"]))
print(lidar["shape"], lidar["fps"], lidar["channels"])
PY2
```

Run the complete editable case, which sends the sample's seven cameras with
their per-camera captions and the bundled negative prompt, converts the
sample's LiDAR control, and validates the returned LiDAR header:

```bash
uv run python examples/mv_joint.py
```

It writes the camera MP4s, `lidar.safetensors`, and a `lidar.json`
descriptor to `examples/outputs/mv_joint_text2video/`.

`--mode image2video` also sends the sample's RGB clips as `vision` inputs, so
WSM, RGB first-frame conditioning, and LiDAR are generated together; the output
has the same shape. A request that cannot produce LiDAR fails explicitly rather
than returning a camera-only HTTP 200.

### Read the LiDAR output

`multiview.lidar` in the response carries:


| Field                                       | Meaning                                                                                         |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `b64_frames`                                | Base64 safetensors file with a `frames` tensor                                                  |
| `shape`                                     | Tensor shape, for example `[3, 67, 128, 1800]`                                                  |
| `channels`, `units`                         | Channel names and units, for example `range`/`intensity`/`validity` in `metres`/`unit`/`binary` |
| `fps`, `num_frames`                         | LiDAR rate (10 Hz) and sweep count                                                              |
| `apply_validity_mask`, `validity_threshold` | How to mask invalid returns using the validity channel                                          |


`mv_joint.py` saves everything except `b64_frames` as `lidar.json`. For numeric
analysis, load `frames` from `lidar.safetensors` with the `safetensors` library
in your analysis environment and mask returns whose validity channel is below
`validity_threshold`.

## Generation defaults and limits

When omitted, multiview requests use the model's native settings. Explicit
values that differ are rejected unless the table allows them:


| Field                                | Default                             | Accepted values                                                                                                     |
| ------------------------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `resolution`                         | `720` (1280 × 720 for 16:9 controls) | Any `480` or `720` key; a bare tier sizes output from the first WSM control's aspect                                |
| `num_frames`                         | `201` per camera                    | `200` through `401`; a count that is not `4k + 1` is generated at the next `4k + 1` and each camera is trimmed back |
| `num_inference_steps`                | `35`                                | `35` only                                                                                                           |
| `guidance_scale`                     | `1.5`                               | `1.0` through `7.0`                                                                                                 |
| `flow_shift`                         | `10.0`                              | Any positive value                                                                                                  |
| `multiview.per_view_negative_prompt` | None; required                      | Any string up to 20,000 characters, applied to every camera; empty disables it                                      |


The single-view frame ceilings in [Generation](generation.md#frame-counts-and-limits)
do not apply to multiview. Always set `seed` when comparing runs.

Multiview has no `fps` field. Cameras are always generated and encoded at
30 FPS, the only rate the model is trained and conditioned on, and a request
that sends `fps` is rejected. Prepare WSM controls at 30 FPS. To play a clip
back at another rate, retime the saved MP4s; LiDAR stays at 10 Hz.

Multiview requests reject top-level `prompt`, `input_reference`, `transfer`,
`action_params`, and `negative_prompt`. Per-view `control` implies WSM;
do not add `transfer.wsm`.

## Output

A successful response returns `multiview` instead of `b64_video`:


| Field                                       | Meaning                                                      |
| ------------------------------------------- | ------------------------------------------------------------ |
| `videos[]`                                  | `{camera_key, b64_video}` per requested camera, one MP4 each |
| `camera_order`                              | Camera keys in canonical order, matching `videos[]`          |
| `frames_per_view`, `fps`, `width`, `height` | Output geometry, identical for every camera                  |
| `lidar`                                     | Present only for `camera_lidar` requests                     |


There is no aggregate video: each camera is a separate base64 MP4. With eleven
videos, and LiDAR for joint requests, response bodies are large; configure
client and proxy body limits accordingly. Each MP4 holds a VP9 track; play it
with `ffplay` or re-encode it as described in
[Output and playback](generation.md#output-and-playback).

### GPU memory and `layer` profiles

Profiles tagged `offload: layer` use VRAM reduction techniques: transformer
layer offload to host memory, VAE tiling, VAE offload, grouped attention, and
host-memory SeaCache history. These reduce performance but enable inference on
GPUs with less memory than the resident profiles require. The `layer` profiles
were benchmarked at 720p with 200 frames, eleven cameras, and joint LiDAR
generation.

For smaller payloads, such as fewer cameras, 480p, fewer frames, or camera-only
output, you can turn these reductions off with override environment variables
to recover speed. The overrides keep the selected `layer` profile and change
only what runs on it; each reduction has its own variable. See
[Multiview configuration](configuration.md#multiview). For example, to keep the
transformer on the GPU and process whole frames, add these to the launch
command:

```bash
  -e NIM_MULTIVIEW_LAYER_OFFLOAD=false \
  -e NIM_MULTIVIEW_VAE_TILING=false \
```

Turning a reduction off raises peak GPU memory, and a request that no longer
fits fails with an out-of-memory error. Validate each change with your largest
expected request. At startup the Generator logs a `VRAM optimizations for
profile <id>` block that lists each reduction in effect.

## Guardrails

The launch command in this guide disables the text guardrail; the output
guardrails stay enabled. Video guardrails run once over all generated camera
frames before the response is split per camera, so face blurring and content
checks apply to every camera.

When the text guardrail is enabled, it checks every camera caption in
`views[].prompt`. `per_view_negative_prompt` only steers generation and is not
checked. A request that a text check blocks is stopped before
generation starts and returns HTTP 422. The sample captions have not been
qualified against the blocklist, so with the check enabled, review a blocked
caption and reword it. See [Guardrails](operations.md#guardrails).

## Common failures


| Symptom                                                                                | Likely cause                                                                                                                                           | Action                                                                                                                                                                |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HTTP 422, `multiview requests require a selected model variant of 'multiview'`         | The running Generator is not the multiview variant                                                                                                     | Relaunch with `NIM_MODEL_VARIANT=multiview`                                                                                                                           |
| HTTP 422, `Selected model variant 'multiview' requires a multiview field`              | A single-view request was sent to the multiview variant                                                                                                | Use a multiview request, or launch a general-purpose variant                                                                                                          |
| HTTP 422, `supports text2video and image2video requests only`                         | Another `model_mode`, such as `text2image` or `video2video`, was sent to the multiview variant                                                         | Use `text2video`, or `image2video` with `vision` on every view                                                                                                        |
| HTTP 422, camera key or duplicate                                                      | Unknown or repeated `camera_key`                                                                                                                       | Use the eleven keys in [Camera rig](#camera-rig), each once                                                                                                           |
| HTTP 422, control has insufficient frames                                              | A WSM control is shorter than the requested output                                                                                                     | Use longer controls or request fewer `num_frames`                                                                                                                     |
| HTTP 422, `per_view_negative_prompt` field required                                    | The negative prompt was omitted or sent as a top-level `negative_prompt`                                                                               | Send `multiview.per_view_negative_prompt`, using an empty string to disable it, and drop `negative_prompt`                                                            |
| HTTP 422, `views.N.prompt` field required or `Each camera requires a non-empty prompt` | A camera has no caption                                                                                                                                | Give every view its own plain-text `prompt`                                                                                                                           |
| HTTP 422, `Multiview rejects prompt`                                                   | A top-level `prompt` was sent, for example a shared scene caption                                                                                      | Drop `prompt` and put each caption in its camera's `views[].prompt`                                                                                                   |
| HTTP 422, `Camera prompt must not contain runtime camera labels`                       | A caption contains a camera key, a `camera_view` label, control instructions, or a resolution or duration template                                     | Describe only what the camera sees, in plain text                                                                                                                     |
| HTTP 422, `num_conditional_sweeps requires condition`                                  | `num_conditional_sweeps` was sent without `lidar.condition`                                                                                            | Add the measured sweeps, or drop the field                                                                                                                            |
| HTTP 422, `multiview.lidar.control` tensor, dtype, shape, or sweep count               | The safetensors file does not match the `[3, sweeps, 128, 1800]` FP32 `frames` layout, or has fewer sweeps than the request needs                      | Re-export in the [LiDAR control](#lidar-control) layout with at least the required sweeps                                                                             |
| HTTP 422, LiDAR `values must be finite` or `must be in [0, 1]`                         | NaN or Inf, negative range, or intensity or validity outside `[0, 1]` in the sweeps used                                                               | Clean or clip the values before encoding                                                                                                                              |
| HTTP 422, `num_conditional_sweeps=N must leave at least one`                           | Conditioning would cover every LiDAR sweep in the request                                                                                              | Use at most the stated maximum, or request more frames                                                                                                                |
| HTTP 422, `This request has been blocked by our AI guidelines`                         | The text guardrail is enabled and a camera caption was blocked | Reword the blocked caption, or launch with `NIM_ENABLE_TEXT_GUARDRAILS=0` for an approved, isolated evaluation                                                           |
| HTTP 422, `requires vision for every view`                                             | `image2video` view without `vision`                                                                                                                    | Add an RGB image or video to every view                                                                                                                               |
| HTTP 422, `greater than or equal to 200` or `less than or equal to 401`                | Explicit frame count outside the supported range                                                                                                       | Request 200 to 401 frames, or omit `num_frames` for 201                                                                                                               |
| HTTP 422, `fixed 30 FPS; remove fps`                                                   | `fps` sent with a multiview request                                                                                                                    | Remove `fps`; multiview always generates at 30 FPS                                                                                                                    |
| HTTP 422, `guidance_scale` `greater than or equal to 1` or `less than or equal to 7`   | Guidance outside the request bounds                                                                                                                    | Request 1.0 to 7.0, or omit `guidance_scale` for 1.5                                                                                                                  |
| HTTP 422, `text2video multiview requests do not accept vision inputs`                  | `vision` sent with `text2video`                                                                                                                        | Remove `vision`, or use `image2video`                                                                                                                                 |
| HTTP 422, `camera_lidar` requires `multiview.lidar.control`                            | Joint output without the LiDAR control                                                                                                                 | Add `multiview.lidar.control`                                                                                                                                         |
| HTTP 422, `multiview.lidar is accepted only with output_modality 'camera_lidar'`       | LiDAR control sent with camera-only output                                                                                                             | Set `output_modality` to `camera_lidar`, or remove `lidar`                                                                                                            |
| HTTP 422, `num_inference_steps` `Input should be 35`                                   | The fixed step count was overridden                                                                                                                    | Omit the field to use the default                                                                                                                                     |
| Joint request fails after admission                                                    | LiDAR control is not a valid FP32 `frames` tensor, has too few sweeps, or contains non-finite values                                                   | Check shape, dtype, and values in the control file                                                                                                                    |
| Client timeout                                                                         | Generation exceeded the client or backend timeout                                                                                                      | Raise both: the client `--timeout`, and the backend with `NIM_TRITON_REQUEST_TIMEOUT` at launch. A timed-out client does not cancel the server, which keeps computing |
| `Broken pipe`, `Connection reset by peer`, or `Server disconnected`, with nothing in the server log | A pooled client connection sat idle longer than the server keep-alive while the client built a large camera and LiDAR body, so it was reused after the server closed it | Launch with `-e NIM_HTTP_KEEP_ALIVE_TIMEOUT_SECS=120` (default 10), or open a new connection per request; images without this variable close idle connections after 5 seconds |


Do not retry only because a long request is quiet. Check the active request
using the [long-running request guidance](operations.md#long-running-requests)
first. For other service-level failures, see
[operations.md](operations.md#troubleshooting).

## Related

- [Generation](generation.md) — single-view T2I, T2V, I2V, and V2V.
- [Transfer](transfer.md) — single-view WSM, edge, blur, depth, and
segmentation control.
- [API reference](api-reference.md) — the shared Generator request envelope.
- [Cosmos3-Nano-Transfer-Auto](../simulation/auto/README.md) — the same model on
Cosmos Framework, vLLM-Omni, and SGLang, using the same daytime sample.

