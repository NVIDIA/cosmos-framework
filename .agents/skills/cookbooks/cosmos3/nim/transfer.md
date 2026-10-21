<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Transfer

Guide Generator output with edge, blur, depth, segmentation, or
world-space-map (WSM) video controls: send `POST /v1/infer` with
`model_mode=video2video` and a top-level `transfer` object.

> Confirm control and model availability for the selected image in the
> [Support Matrix](support-matrix.md).

Start a compatible Generator with [Deployment](deployment.md) and begin with
one precomputed control. To choose between V2V, Transfer, and inverse
dynamics, refer to
[Choose a Video-Conditioned Workflow](generation.md#choose-a-video-conditioned-workflow).

## Control Types

| Control | Precomputed Video | Derived by Server | Preset |
| --- | --- | --- | --- |
| `edge` | Yes | Yes | `very_low`, `low`, `medium`, `high`, `very_high` |
| `blur` | Yes | Yes | `none`, `very_low`, `low`, `medium`, `high`, `very_high` |
| `depth` | Required | No | None |
| `seg` | Required | No | None |
| `wsm` | Required | No | None |

At least one control must be enabled. Control media uses the same
[media input rules](api-reference.md#media-inputs) as `input_reference`.

## Run the Examples

The script has five precomputed cases with matching prompts, control media,
geometry, and seeds. Run one at a time, because each is an expensive
generation.

Transfer is compute-intensive: use a high-throughput discrete GPU such as an
RTX PRO 6000 Blackwell 96 GB or H100 80 GB. DGX Spark can meet the memory
floors but is much slower and not recommended. Check memory against
[Transfer Headroom](support-matrix.md#transfer-headroom).

The client allows 60 minutes per request, but the backend defaults to 30. To
match, add `-e NIM_TRITON_REQUEST_TIMEOUT=3600000000` (microseconds) to the
Generator launch and restart. Refer to
[Long-Running Requests](operations.md#long-running-requests).

Install the [client tooling](prerequisites.md#client-tooling). Then, from the
repository root, enter the cookbook directory before running a case:

```bash
cd .agents/skills/cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
uv run python examples/transfer.py --case precomputed_edge
```

For another precomputed control, replace `precomputed_edge` with
`precomputed_blur`, `precomputed_depth`, `precomputed_seg`, or
`precomputed_wsm`.

The NIM can also derive edge or blur controls from a source video. Run one
derived case:

```bash
uv run python examples/transfer.py --case derived_edge
```

Replace `derived_edge` with `derived_blur` to use a derived blur control. The
script needs a general-purpose `nano` or `super` variant and writes
`examples/outputs/transfer_<case>.mp4`.

## Precomputed Control

Put the control media inside the matching nested object. A server-local file
path is not accepted as control media:

```json
{
  "model_mode": "video2video",
  "prompt": "<compact contents of assets/edge/prompt.json>",
  "negative_prompt": "<compact contents of assets/negative_prompt.json>",
  "transfer": {
    "edge": {
      "video": "data:video/mp4;base64,<BASE64_EDGE_CONTROL>"
    },
    "control_guidance": 1.5,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "num_video_frames_per_chunk": 121
  },
  "resolution": "720_16_9",
  "num_frames": 121,
  "fps": 30.0,
  "num_inference_steps": 50,
  "guidance_scale": 3.0,
  "flow_shift": 10.0,
  "seed": 2026
}
```

Replace `edge` with `blur`, `depth`, `seg`, or `wsm` and provide the matching
control.

## Derive Edge or Blur Controls

For server-derived controls, send the source video in `input_reference` and
omit the nested control video:

```json
{
  "model_mode": "video2video",
  "prompt": "A red sports car drives through a dramatic snowy landscape.",
  "negative_prompt": "<compact contents of assets/negative_prompt.json>",
  "input_reference": "data:video/mp4;base64,<BASE64_SOURCE_VIDEO>",
  "transfer": {
    "edge": {
      "preset_edge_threshold": "medium"
    },
    "control_guidance": 1.5,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "num_video_frames_per_chunk": 121
  },
  "resolution": "720_16_9",
  "num_frames": 121,
  "fps": 30.0,
  "num_inference_steps": 50,
  "guidance_scale": 3.0,
  "seed": 2026
}
```

The `"edge": true` and `"blur": true` settings select the corresponding derived
control with preset `medium`. They require `input_reference`. A preset applies
only to a derived control and cannot accompany nested `video`.

The server cannot derive depth, segmentation, or WSM controls. Supply a
precomputed control video.

## Tune Transfer Parameters

| Field | Constraint | Current Effective Default |
| --- | --- | --- |
| `control_guidance` | Number `[0.0,10.0]` | 1.5 generally, 2.0 segmentation-only, 3.0 WSM-only |
| `num_video_frames_per_chunk` | Integer `>= 1` | 93 generally, 101 WSM-only |
| `num_conditional_frames` | Integer `>= 0` and smaller than chunk size | 1 |
| `num_first_chunk_conditional_frames` | Integer `>= 0`; bounded by chunk and output | 0 |
| `<control>.control_weight` | Finite number `>= 0`; enabled weights need a positive sum | 1.0 |

`num_first_chunk_conditional_frames > 0` requires `input_reference`.
The `control_weight` field belongs inside a control object. Refer to
[Combine Multiple Controls](#combine-multiple-controls).

### Control Guidance Behavior

When control and text guidance are both active, the control term is scaled by
`control_guidance` alone. With the edge-family defaults, the effective control
coefficient is 1.5. `control_guidance=1.0` disables the extra control-guidance
branch.

The following table lists the current defaults by family. Explicit valid
values override them. Recheck them on another image.

| Request Family | Frames | FPS | Steps | Guidance | Flow Shift | Control Guidance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge, blur, depth, or mixed | 121 | 30 | 50 | 3.0 | 10.0 | 1.5 |
| Segmentation-only | 121 | 30 | 50 | 3.0 | 10.0 | 2.0 |
| WSM-only | 101 | 10 | 50 | 1.0 | 10.0 | 3.0 |

The example script uses a 121-frame chunk, or 101 for WSM.

## Combine Multiple Controls

A `transfer` object can enable several controls. Mixed requests use the
edge-family defaults. Each control object (`edge`, `blur`, `depth`, `seg`,
`wsm`) accepts an optional relative `control_weight`, normalized across the
enabled controls. This example gives edge and depth normalized weights of 0.25
and 0.75:

```json
{
  "model_mode": "video2video",
  "prompt": "A snowy mountain road",
  "input_reference": "data:video/mp4;base64,<BASE64_SOURCE_VIDEO>",
  "guidance_scale": 3.0,
  "transfer": {
    "edge": {"control_weight": 1.0},
    "depth": {
      "video": "data:video/mp4;base64,<BASE64_DEPTH_CONTROL>",
      "control_weight": 3.0
    },
    "control_guidance": 1.5
  }
}
```

An omitted weight, `"edge": true`, and `"blur": true` use 1.0. A zero weight is
accepted only when another enabled control has a positive weight. This guide
does not validate any combination. Use one control per request unless yours is
validated, and only combine controls that are spatially and temporally
aligned.

## Transfer VRAM Admission

A ready deployment can serve generation but reject Transfer when GPU headroom
is too small. Use a larger GPU or a configuration that needs less video memory
(VRAM). For the thresholds and the `NIM_ALLOW_UNSAFE_TRANSFER` diagnostic
override, refer to [Transfer Headroom](support-matrix.md#transfer-headroom).

## Media and Output

Control videos and video `input_reference` values accept up to 100,000,000
encoded characters. Refer to [Media Inputs](api-reference.md#media-inputs) for
formats, codecs, and URL rules.

Transfer media must also meet these requirements:

- The source video and each control video must have at least `num_frames`
  frames (WSM control: `num_frames - 1`).
- Their aspect ratio must be within 5% of the requested output shape.
- The decoded source and control videos together are capped at 225,000,000
  bytes. Each single video is also capped at 75,000,000 bytes.

The response is `b64_video` plus `action: null`.

## Validation and Common Failures

Besides the failures in the table, Transfer rejects requests that contain any
of the following:

- V2V fields (`condition_frame_indexes_vision`, `condition_video_keep`) or
  `action_params`
- A nested precomputed edge/blur video combined with its preset
- A negative or non-finite `control_weight`
- Enabled weights without a positive sum

Use this table to fix common failures:

| Failure | Fix |
| --- | --- |
| HTTP 422 says derived control needs `video` | Add top-level source video or provide a nested precomputed edge/blur video |
| HTTP 422 says control video required | Add nested video for depth, segmentation, or WSM |
| Control/output lengths are incompatible | Recheck the output frame count, chunk size, and conditional frame values |
| Transfer is disabled while T2V works | Selected configuration fits T2V but GPU headroom is below Transfer overhead; use a larger GPU or lower-VRAM configuration |

For service-level failures, refer to [Troubleshooting](operations.md#troubleshooting).
