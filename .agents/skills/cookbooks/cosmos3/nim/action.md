<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Use Cosmos3 Action Capabilities Through the Cosmos3 NIM

Send forward dynamics, policy, and inverse dynamics requests to a compatible
**Generator** with synchronous JSON `POST /v1/infer`. The `model_mode` field
selects the task.

> Confirm model and domain availability for the selected image in the
> [Support Matrix](support-matrix.md).

Start the NIM with [Deployment](deployment.md). General Action models return
visual rollouts for all three modes. The Edge-DROID and Nano-DROID specialists
serve policy only and return actions without media.

## Understand an Action Sequence

Cosmos3 models an action as a transition between consecutive visual
observations:

```text
o0 --a0--> o1 --a1--> o2 ... --a(T-1)--> oT
```

A trajectory `[T,D]` covers `T` transitions across `T+1` observations, where
`T` is `action_chunk_size` and `D` the domain's action width. The AV cases send
60 rows of 9 values and get a 61-frame rollout.

Omit `num_frames`. The service sets it to `action_chunk_size + 1`.

## Choose an Action Mode

| `model_mode` | Input | Output |
| --- | --- | --- |
| `forward_dynamics` | Conditioning image plus action trajectory `[T,D]` | Rollout video; `action` response is null |
| `policy` | Conditioning image, task prompt, and optional state | Rollout video plus predicted action trajectory, or action only for a compatible specialist profile |
| `inverse_dynamics` | Conditioning video and optional task prompt | Video plus predicted action trajectory |

Action requests must not set top-level `resolution`.

## Domains and Representations

| `domain_name` | Domain ID | General `raw_action_dim` | Example Chunk |
| --- | ---: | ---: | ---: |
| `av` | 1 | 9 | 60 actions at 10 action FPS |
| `umi` | 6 | 10 | 16 actions at 20 action FPS |
| `bridge_orig_lerobot` | 7 | 10 | 16 actions at 5 action FPS |
| `droid_lerobot` | 8 | 10 | 16 actions in the general contract |

AV actions are 9D (translation plus 6D rotation). UMI and DROID actions are a
9D end-effector pose plus a gripper value. The API checks only domain and
width. Units, normalization, frames, gripper conventions, and denormalization
belong to the checkpoint, so apply its published transforms before driving
hardware.
The DROID specialists' 8-wide output is their own contract, not a truncated
DROID 10D action.

## Cadence and Visual Geometry

Top-level `fps` is the rollout video cadence, and `action_params.action_fps` is
the action-row cadence. They are separate fields, aligned in the examples. The
`action_params.image_size` field is a model setting, not a replacement for the
forbidden `resolution`.

## Run the Examples

Install the [client tooling](prerequisites.md#client-tooling), then enter the
cookbook directory:

```bash
cd .agents/skills/cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
```

The executable cases include AV, UMI, and robot examples:

| Case | Conditioning and Expected Result | Run |
| --- | --- | --- |
| `av_forward` | Supplied AV straight trajectory; rollout visualizes that trajectory | `uv run python examples/action.py --case av_forward` |
| `av_left` | Supplied AV left trajectory; ego rollout turns left | `uv run python examples/action.py --case av_left` |
| `av_right` | Supplied AV right trajectory; ego rollout turns right | `uv run python examples/action.py --case av_right` |
| `umi_forward` | First 16-action UMI chunk; rollout visualizes that chunk | `uv run python examples/action.py --case umi_forward` |
| `av_inverse_0` | Observed `av_0.mp4`; predicts its AV action trajectory | `uv run python examples/action.py --case av_inverse_0` |
| `av_inverse_1` | Observed `av_1.mp4`; predicts its AV action trajectory | `uv run python examples/action.py --case av_inverse_1` |
| `bridge_inverse` | Observed pinned robot video; predicts its Bridge action trajectory | `uv run python examples/action.py --case bridge_inverse` |
| `av_policy_left` | `av_0.jpg` plus a left-turn goal; predicts a left-turn trajectory and rollout | `uv run python examples/action.py --case av_policy_left` |
| `av_policy_right` | `av_0.jpg` plus a right-turn goal; predicts a right-turn trajectory and rollout | `uv run python examples/action.py --case av_policy_right` |

AV policy cases need a general-purpose `nano` variant. The others accept
`nano` or `super`. The `av_policy` and `policy` aliases select
`av_policy_right`. Output lands in `examples/outputs/`: `action_<case>.mp4`,
plus `action_<case>.json` with the predicted trajectory for policy and inverse
dynamics. Review the video to check direction. Inspect action data in the
[Cosmos3 Action Viewer](https://huggingface.co/spaces/nvidia/Cosmos3-Action-Viewer).

## Forward Dynamics

Forward dynamics renders the observations that follow an initial image and an
action chunk. The AV cases share `images/av_0.jpg` and differ only in the
trajectory:

```python
request = {
    "model_mode": "forward_dynamics",
    "prompt": "You are an autonomous vehicle planning system.",
    "input_reference": image_data_url,
    "action_params": {
        "domain_name": "av",
        "action_chunk_size": 60,
        "action": trajectory,  # 60 rows × 9 finite numeric values
        "raw_action_dim": 9,
        "action_space": "joint_pos",
        "image_size": "480",
        "action_fps": 10.0,
    },
    "fps": 10.0,
    "num_inference_steps": 30,
    "guidance_scale": 1.0,
    "flow_shift": 10.0,
    "seed": 0,
}
```

The `action` field needs `action_chunk_size` rows of `raw_action_dim` finite
numbers. The trajectory, not the prompt, selects the maneuver. Forward dynamics
rejects `history_length`, `use_state`, and `observation` and returns video with
no action.

`umi_forward` uses `images/umi.png` and the first 16 rows of
`actions/umi.json`. For longer rollouts, condition each next chunk on the last
generated frame and drop duplicate boundary frames when joining.

## Policy

Policy predicts an action chunk instead of receiving one:

```json
{
  "model_mode": "policy",
  "prompt": "You are an autonomous vehicle planning system. Turn right onto the road and continue driving in the rightmost lane.",
  "input_reference": "data:image/jpeg;base64,<BASE64_IMAGE>",
  "action_params": {
    "domain_name": "av",
    "action_chunk_size": 60,
    "raw_action_dim": 9,
    "action_space": "joint_pos",
    "image_size": "480",
    "action_fps": 10.0
  },
  "fps": 10.0,
  "num_inference_steps": 30,
  "guidance_scale": 1.0,
  "seed": 0
}
```

`av_policy_left` and `av_policy_right` differ only in the language goal and
return a `[60,9]` action and a 61-frame rollout. Judge direction from the
video, not raw values. A plausible rollout does not establish traffic safety or
physical suitability. Policy omits `action` and can add the policy-only state
fields in the [reference](#action-parameter-reference), except `observation`,
which only the DROID specialists accept.

### Integrate a Policy Loop

`POST /v1/infer` is stateless and is not the streaming policy server used by
the RoboLab workflow. Call it once per control step with the current
observation and state. Apply the checkpoint's transforms to the returned
actions, execute only part of the horizon, and then replan from a new
observation. Before physical execution, enforce your own joint and workspace
bounds, collision checks, timing limits, and emergency stops. A valid response
is not authorization to execute.

## Edge-DROID and Nano-DROID Policy

Edge-DROID and Nano-DROID are BF16-only specialist policy checkpoints with the
same action-only request contract. Edge-DROID is
[experimental](support-matrix.md#choose-a-model) in this release because it
has not been validated. Choose one at startup by replacing the model selector
lines of the [Generator launch](deployment.md#launch-generator) with these:

```bash
-e NIM_MODEL_TYPE=generator \
-e NIM_MODEL_VARIANT=edge-droid \
-e NIM_PRECISION=bf16 \
-e NIM_CACHE_BACKEND=none
```

Use `NIM_MODEL_VARIANT=nano-droid` for Nano-DROID. `NIM_CACHE_BACKEND=none`
disables SeaCache. It is already the Edge-DROID default, but Nano-DROID enables
SeaCache unless you keep this line. Guardrails stay enabled. Neither variant serves DROID
forward or inverse dynamics.

Its request uses the same `POST /v1/infer` API as other Generator tasks:

```json
{
  "model_mode": "policy",
  "prompt": "Remove the crumpled paper from the sink and place it in the trash can.",
  "input_reference": "data:image/jpeg;base64,<COMPOSED_DROID_OBSERVATION>",
  "action_params": {
    "domain_name": "droid_lerobot",
    "action_chunk_size": 32,
    "observation": {
      "observation/joint_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "observation/gripper_position": 0.0
    }
  },
  "seed": 0
}
```

The observation needs exactly seven finite joint positions and one finite
gripper position. The image is a 640×540 composition with the wrist view above
two exterior views.

Omit the model-owned fields `raw_action_dim`, `action_space`, `image_size`,
`action_fps`, `history_length`, and `use_state` in `action_params`, and
top-level `fps`, `num_frames`, and `negative_prompt`. The model returns 32
actions of width 8 and defaults to 4 steps, guidance `3.0`, and flow shift
`5.0`. Those defaults and `seed` are overridable.

The response is action-only, with no `b64_image` or `b64_video`. Refer to the
[response example](api-reference.md#generator-response).

After readiness, run the [policy client](examples/policy_droid.py):

```bash
uv run python examples/policy_droid.py
```

It composes the first frame of each camera in the public
[`droid_lerobot_example`](../generator/action/assets/droid_lerobot_example),
with the wrist view above the two exterior views. It then sends the matching
first-frame joint/gripper state and saves the composed image and the validated
`[32, 8]` action under `examples/outputs/`. It does not execute robot actions.

The [vLLM-Omni policy
notebook](../generator/action/run_policy_with_vllm_omni.ipynb) uses the same
sample with a different direct-vLLM `/v1/videos` API. Do not copy its payload.

## Inverse Dynamics

Inverse dynamics estimates the actions in an observed video. The local cases
use `videos/av_0.mp4` and `videos/av_1.mp4`:

```json
{
  "model_mode": "inverse_dynamics",
  "prompt": "You are an autonomous vehicle planning system.",
  "input_reference": "data:video/mp4;base64,<BASE64_AV_VIDEO>",
  "action_params": {
    "domain_name": "av",
    "action_chunk_size": 60,
    "raw_action_dim": 9,
    "action_space": "joint_pos",
    "image_size": "480",
    "action_fps": 10.0
  },
  "fps": 10.0,
  "num_inference_steps": 30,
  "guidance_scale": 1.0,
  "seed": 0
}
```

The `bridge_inverse` case fetches a public URL, so it needs URL input and
container network access. Use a data URL offline. Policy-only state fields are
rejected.

## Response Action Object

General visual policy and inverse-dynamics profiles return a video and an action
object:

```json
{
  "b64_video": "<BASE64_MP4>",
  "action": {
    "data": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "shape": [16, 10],
    "dtype": "float32",
    "raw_action_dim": 10,
    "action_mode": "inverse_dynamics",
    "domain_id": 7
  }
}
```

The `data` array is shortened here. Real responses have every row in `shape`.
Before use, check that `shape` matches `data`, that rows, width,
`raw_action_dim`, `action_mode`, and `domain_id` match the request, and that
values are finite.
[Edge-/Nano-DROID Policy](#edge-droid-and-nano-droid-policy) returns the same
object without media.

## Action Parameter Reference

| Field | Type/Default | Contract |
| --- | --- | --- |
| `domain_name` | required enum | `av`, `bridge_orig_lerobot`, `droid_lerobot`, or `umi` |
| `action_chunk_size` | required integer | Positive multiple of 4; output frames equal this value plus 1 |
| `action` | number array `[T,D]` or null | Required only for forward dynamics; `T=action_chunk_size` and `D=raw_action_dim` |
| `raw_action_dim` | integer or null | Defaults to the selected domain width; an explicit value must match. Specialist profiles can replace an omitted value. |
| `action_space` | enum; `joint_pos` | `joint_pos` or `midtrain` |
| `image_size` | enum; `480` | `256`, `480`, `704`, or `720`; distinct from top-level `resolution` |
| `action_fps` | number or null | Optional range `[1.0,60.0]` |
| `history_length` | integer or null | Policy-only and `>= 1` |
| `use_state` | boolean or null | Policy-only |
| `observation` | object or null | DROID specialist policy only; the general policy rejects it. Free-form state without a public nested schema |

## Defaults and Request Validation

General Action profiles default to `num_inference_steps=30`,
`guidance_scale=1.0`, and `fps=10.0`. Specialists can override these.

## Common Failures

| Failure | Fix |
| --- | --- |
| `action_chunk_size` is not a positive multiple of 4 | Use 60 for AV or 16 for the general UMI and robot examples |
| Action rows or width do not match | Validate `[T,D]` and finite values before sending; use the domain table above |
| Wrong conditioning media | Image for forward/policy; video for inverse dynamics |
| Configuration cannot run the action case | Confirm the selected image's action-capable model variant and supported domain |
| Client fails on missing `b64_video` | Handle action-only specialist responses separately from general visual Action responses |

For startup, out-of-memory errors, and service diagnostics, refer to
[Troubleshooting](operations.md#troubleshooting).
