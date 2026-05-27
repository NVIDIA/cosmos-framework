# Action Policy Closed-Loop Evaluation on LIBERO

The [Action Model Server](#start-the-action-model-server) uses `cosmos_framework.scripts.action_policy_server`. The [LIBERO Evaluation Client](#run-the-libero-evaluation-client) and [Optional Dataset Action Server](#optional-dataset-action-server) use `cosmos_framework.simulation.libero.closed_loop_eval` and `cosmos_framework.simulation.libero.dataset_reply_action_server`.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Set Up LIBERO](#set-up-libero)
- [Start the Action Model Server](#start-the-action-model-server)
  - [Server Options](#server-options)
  - [HTTP API Reference](#http-api-reference)
    - [GET Health Check](#get-health-check)
    - [GET Info](#get-info)
    - [POST Predict](#post-predict)
  - [Client Control Loop](#client-control-loop)
- [Run the LIBERO Evaluation Client](#run-the-libero-evaluation-client)
- [Optional Dataset Action Server](#optional-dataset-action-server)
- [Outputs](#outputs)
- [Common Options](#common-options)
- [Troubleshooting](#troubleshooting)
  - [Server Starts but Client Gets Empty Actions](#server-starts-but-client-gets-empty-actions)
  - [Success Rate Is Near Zero](#success-rate-is-near-zero)
  - [MuJoCo or OpenGL Fails to Initialize](#mujoco-or-opengl-fails-to-initialize)
  - [LIBERO Config Is Missing](#libero-config-is-missing)
  - [`Numba needs NumPy 2.2 or less. Got NumPy 2.4.`](#numba-needs-numpy-22-or-less-got-numpy-24)

______________________________________________________________________

<!--TOC-->

## Overview

LIBERO closed-loop evaluation runs as two HTTP-connected processes:

- **Action model server**: loads a Cosmos3 Action policy checkpoint on GPU and serves `POST /predict`.
- **LIBERO evaluation client**: runs the LIBERO simulator, sends rendered observations to the server, executes returned actions, and writes success metrics.

The client and server can run on the same machine or on separate machines. Use separate machines or virtual environments when your LIBERO simulator dependencies differ from the model-serving environment.

## Prerequisites

Start from the root of the released Cosmos3 repository:

```shell
git clone git@github.com:NVIDIA/cosmos-framework.git
cd cosmos-framework
```

Install Cosmos3 with the CUDA/training dependencies needed to load DCP checkpoints, plus the `libero` group for the evaluation client:

```shell
uv sync --all-extras --group=cu130-train --group=libero
source .venv/bin/activate
export LD_LIBRARY_PATH=
```

Use `--group=cu128-train` instead if your environment uses the CUDA 12.8 dependency group. See [Setup](./setup.md) for the full installation matrix. If the model server and LIBERO client need separate environments, omit `--group=libero` from the server env and create a second venv with only `--group=libero` for the client.

You also need:

- A LIBERO-compatible Cosmos3 Action policy checkpoint.
- The experiment name and config used by that checkpoint.
- Action normalization stats for the checkpoint, if the policy was trained with normalized actions.
- LIBERO simulator dependencies on the client side.

## Set Up LIBERO

LIBERO is installed via the `libero` dependency group from [Prerequisites](#prerequisites). Verify that the evaluation client can import LIBERO and resolve the benchmark paths:

```shell
python - <<'PY'
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()
task = task_suite.get_task(0)
print(f"Loaded libero_10 task 0: {task.language}")
print(f"BDDL root: {get_libero_path('bddl_files')}")
print(f"Renderer: {OffScreenRenderEnv.__name__}")
PY
```

For headless machines, choose a MuJoCo rendering backend before launching the client:

```shell
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Use `MUJOCO_GL=osmesa` for CPU rendering if EGL is not available.

## Start the Action Model Server

Run the server in the Cosmos3 environment on a GPU machine. For a checkpoint trained with the OSS
[`action_policy_sft_nano`](../examples/toml/sft_config/action_policy_sft_nano.toml) recipe (experiment SKU at
[`cosmos_framework/configs/base/experiment/sft/action_policy_sft_nano.py`](../cosmos_framework/configs/base/experiment/sft/action_policy_sft_nano.py)),
the recommended OSS inference flow is to export the training DCP checkpoint to a consolidated Hugging Face/safetensors
checkpoint first. The training output directory is the `IMAGINAIRE_OUTPUT_ROOT` env var passed to
`cosmos_framework.scripts.train` (`--sft-toml=<recipe-toml>`); the DCP checkpoint directory is under the job directory in
`${IMAGINAIRE_OUTPUT_ROOT}`.

```shell
TRAIN_OUTPUT_DIR=/path/to/train-output
DCP_CHECKPOINT_DIR=/path/to/job/checkpoints/iter_000002000
CONSOLIDATED_CHECKPOINT_DIR="${TRAIN_OUTPUT_DIR}/model"

python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "${DCP_CHECKPOINT_DIR}" \
    --config-file "${TRAIN_OUTPUT_DIR}/config.yaml" \
    -o "${CONSOLIDATED_CHECKPOINT_DIR}"

python -m cosmos_framework.scripts.action_policy_server \
    --checkpoint-path "${CONSOLIDATED_CHECKPOINT_DIR}" \
    --host 0.0.0.0 \
    --port 8000 \
    --action-chunk-size 16 \
    --max-action-dim 64 \
    --raw-action-dim 10 \
    --action-stats-path cosmos_framework/data/vfm/action/normalizers/libero_native_frame_wise_relative_rot6d.json \
    --action-normalization quantile_rot \
    --guidance 1.0 \
    --num-steps 1 \
    --fps 20
```

Notes:

- `TRAIN_OUTPUT_DIR` should be the `IMAGINAIRE_OUTPUT_ROOT` output directory passed to `cosmos_framework.scripts.train`. Its `job` symlink points to the checkpoint job directory.
- `DCP_CHECKPOINT_DIR` can point to either a local DCP checkpoint directory or a remote checkpoint path supported by `cosmos_framework.scripts.export_model`. If you use remote storage, pass `--credential-path` to the export command.
- `--checkpoint-path` can point to the consolidated checkpoint directory produced by `cosmos_framework.scripts.export_model`, or directly to the original DCP checkpoint when you want to skip export.
- `--config-file` is not needed when starting the server from the consolidated checkpoint because the exported directory contains `config.json`. When starting directly from DCP, pass the serialized `config.yaml` from the `cosmos_framework.scripts.train` `IMAGINAIRE_OUTPUT_ROOT` output directory.
- `--raw-action-dim 10` matches LIBERO frame-wise relative actions with 6D rotation: `xyz(3) + rot6d(6) + gripper(1)`.
- `--max-action-dim 64`, `--action-chunk-size 16`, `--fps 20`, and `--action-normalization quantile_rot` match the released `action_policy_sft_nano.yaml` config.
- `--action-stats-path` should match the normalization statistics used by the checkpoint. The released package includes the LIBERO frame-wise relative 6D-rotation stats at `cosmos_framework/data/vfm/action/normalizers/libero_native_frame_wise_relative_rot6d.json`.

To see all available server arguments:

```shell
python -m cosmos_framework.scripts.action_policy_server --help
```

Check the server health endpoint:

```shell
curl http://localhost:8000/
curl http://localhost:8000/info
```

If the client runs on another machine, replace `localhost` with the server host or IP address.

### Server Options

| Argument                 | Default                                   | Description                                                                                                                                |
| ------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--experiment`           | optional                                  | Run label returned by `/info`. Required when serving a DCP checkpoint with a module-backed experiment config.                              |
| `--checkpoint-path`      | required                                  | Consolidated Hugging Face/safetensors checkpoint directory, or a DCP checkpoint directory for direct DCP serving.                          |
| `--config-file`          | `cosmos_framework/configs/base/config.py` | Usually omitted for consolidated checkpoints; required for DCP checkpoints unless using a module-backed experiment config.                 |
| `--credential-path`      | `credentials/gcp_checkpoint.secret`       | Credential file for remote DCP checkpoint storage. Not used for local checkpoints or local consolidated checkpoint serving.                |
| `--checkpoint-cache-dir` | unset                                     | Local cache root for remote DCP checkpoints.                                                                                               |
| `--seed`                 | `0`                                       | Random seed for model loading and generation.                                                                                              |
| `--guidance`             | `1.0`                                     | Classifier-free guidance scale used during denoising.                                                                                      |
| `--num-steps`            | `30`                                      | Number of denoising steps per policy request.                                                                                              |
| `--fps`                  | `20`                                      | FPS metadata appended to the prompt when the checkpoint config enables duration/FPS augmentation.                                          |
| `--action-chunk-size`    | inferred, fallback `16`                   | Number of action steps predicted per request.                                                                                              |
| `--max-action-dim`       | inferred, fallback `64`                   | Padded action width expected by the model.                                                                                                 |
| `--raw-action-dim`       | inferred from stats, otherwise unset      | Unpadded action width returned to the client. Use `10` for LIBERO 6D-rotation actions.                                                     |
| `--action-stats-path`    | unset                                     | JSON stats file used to denormalize generated actions.                                                                                     |
| `--action-normalization` | `auto`                                    | Normalization to invert: `auto`, `minmax`, `meanstd`, `quantile`, or `quantile_rot`. Use `quantile_rot` for `action_policy_sft_nano.yaml`. |
| `--dump-dir`             | unset                                     | Directory for request dumps, generated videos, and predicted actions.                                                                      |
| `--dump-every`           | `1`                                       | Dump every N-th request when `--dump-dir` is set.                                                                                          |
| `--http-400-on-error`    | disabled                                  | Return HTTP 400 on request errors instead of HTTP 200 with an empty action list.                                                           |
| `--host`                 | `0.0.0.0`                                 | Host address to bind.                                                                                                                      |
| `--port`                 | `8000`                                    | Port to listen on.                                                                                                                         |

### HTTP API Reference

The released LIBERO client uses these endpoints, and custom environment clients can use the same interface.

#### GET Health Check

Health check endpoint.

Response:

```json
{"status": "ok"}
```

#### GET Info

Returns model and server configuration useful for recording reproducible evaluation metadata.

Example response:

```json
{
  "run_name": "<libero_policy_experiment>",
  "checkpoint": "/path/to/checkpoints/iter_000002000",
  "guidance": 1.0,
  "num_steps": 30,
  "fps": 20,
  "seed": 0,
  "action_chunk_size": 16,
  "max_action_dim": 64,
  "raw_action_dim": 10,
  "action_stats_path": "/path/to/libero_action_stats.json"
}
```

#### POST Predict

Runs policy inference for one observation.

Request:

```json
{
  "image": "<base64_encoded_png>",
  "prompt": "<task_description>",
  "domain_name": "libero",
  "image_size": 256
}
```

| Field         | Type     | Required | Description                                                                                               |
| ------------- | -------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `image`       | `string` | Yes      | Base64-encoded PNG observation. Multi-view clients concatenate resized views horizontally.                |
| `prompt`      | `string` | Yes      | Natural-language task description.                                                                        |
| `domain_name` | `string` | Yes      | Action domain identifier. Use `libero` for LIBERO checkpoints.                                            |
| `image_size`  | `int`    | Yes      | Observation height used by the model input. For multi-view images, width may be a multiple of this value. |

Response:

```json
{
  "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  "video": ["<base64_png_frame_0>", "<base64_png_frame_1>"]
}
```

| Field    | Type                  | Description                                                                                       |
| -------- | --------------------- | ------------------------------------------------------------------------------------------------- |
| `action` | `list[list[float]]`   | Predicted action chunk with shape `action_chunk_size x raw_action_dim`.                           |
| `video`  | `list[string]`        | Optional base64 PNG rollout frames returned by the model, usually `action_chunk_size + 1` frames. |
| `error`  | `string`              | Present when request processing fails.                                                            |

Error response:

```json
{
  "action": [],
  "error": "<error_message>",
  "request_id": 1
}
```

### Client Control Loop

The bundled LIBERO client implements the standard closed-loop pattern:

1. Wait for `GET /` to report a healthy model server.
2. Reset the simulator and load a LIBERO initial state.
3. Render the configured camera view or horizontally concatenated multi-view observation.
4. Send the PNG observation, task prompt, domain name, and image size to `POST /predict`.
5. Queue the returned action chunk.
6. Execute `--action_horizon` actions in the simulator, or the full chunk when `--action_horizon=0`.
7. Repeat prediction and execution until success, termination, error, or `--max_steps`.
8. Call `POST /next_episode` when using the dataset action server so it advances its per-task episode cursor.

Use the same loop when adapting the HTTP server to a different simulator. The action post-processing from model action vectors to environment commands is simulator-specific; for LIBERO it is already implemented in `cosmos_framework.simulation.libero.closed_loop_eval`.

## Run the LIBERO Evaluation Client

Run the client in an environment with LIBERO installed. The OSS `action_policy_sft_nano.yaml` config trains on
concatenated `agentview` and wrist observations, so evaluate it with `--camera agentview,wrist`:

```shell
python -m cosmos_framework.simulation.libero.closed_loop_eval \
    --server_url http://localhost:8000 \
    --task_suite libero_10 \
    --num_trials_per_task 20 \
    --action_horizon 16 \
    --action_dim 10 \
    --action_space frame_wise_relative \
    --rotation_space 6d \
    --domain_name libero \
    --camera agentview,wrist \
    --mujoco_gl auto \
    --output_dir outputs/libero_closed_loop/libero_10_multiview
```

For checkpoints trained with a single camera, change only the camera and output directory with flag `--camera agentview`

`--save_comparison` writes side-by-side GIFs comparing the model-predicted rollout returned by the server with the actual environment rollout.

To see all available client arguments:

```shell
python -m cosmos_framework.simulation.libero.closed_loop_eval --help
```

## Optional Dataset Action Server

Use the dataset action server to validate the LIBERO closed-loop client with ground-truth actions from a LeRobot-format LIBERO dataset. It implements the same HTTP interface as the model server, so the client command stays the same except for `--server_url`.

Start the dataset server:

```shell
python -m cosmos_framework.simulation.libero.dataset_reply_action_server \
    --repo_id libero_10 \
    --root /path/to/libero_10_lerobot \
    --action_space frame_wise_relative \
    --rotation_space 6d \
    --pose_coordinate_frame opencv \
    --action_chunk_size 16 \
    --max_action_dim 64 \
    --send_video \
    --camera_mode agentview \
    --host 0.0.0.0 \
    --port 8001
```

Then point the evaluation client at it:

```shell
python -m cosmos_framework.simulation.libero.closed_loop_eval \
    --server_url http://localhost:8001 \
    --task_suite libero_10 \
    --num_trials_per_task 3 \
    --task_ids 0 \
    --action_horizon 16 \
    --action_dim 10 \
    --action_space frame_wise_relative \
    --rotation_space 6d \
    --camera agentview \
    --output_dir outputs/libero_closed_loop/dataset_server_smoke
```

This is useful for debugging camera orientation, action-space settings, and LIBERO initial states before testing a learned policy.

## Outputs

The evaluation client prints per-episode and per-task success rates and writes:

```text
outputs/libero_closed_loop/libero_10/
+-- summary.json
+-- actions/
+-- gifs/              # only when --save_gifs is set
+-- comparisons/       # only when --save_comparison is set
```

`summary.json` contains the selected task IDs, number of episodes, per-task success rates, overall success rate, action-space settings, and per-episode errors if any.

## Common Options

| Option                  | Description                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--task_suite`          | One of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.                                 |
| `--task_ids`            | Comma-separated task IDs. Omit to run all tasks in the suite.                                                      |
| `--num_trials_per_task` | Number of LIBERO initial states to evaluate for each selected task.                                                |
| `--camera`              | `agentview`, `wrist`, or comma-separated `agentview,wrist` for concatenated multi-view input.                      |
| `--action_space`        | `frame_wise_relative` for per-step deltas, or `relative` for anchored relative actions. Must match the checkpoint. |
| `--rotation_space`      | `3d`, `6d`, `9d`, or `auto`. Must match the action representation returned by the server.                          |
| `--action_dim`          | Unpadded action width returned by the server: usually `7` for axis-angle or `10` for 6D rotation.                  |
| `--action_horizon`      | Number of actions to execute from each server response. `0` executes the full returned chunk.                      |
| `--mujoco_gl`           | `auto`, `egl`, `osmesa`, or `glfw`. Use `egl` for headless GPU rendering and `osmesa` for CPU rendering.           |
| `--initial_states_path` | `DEFAULT` uses LIBERO benchmark initial states. Pass a JSON file to use custom initial states.                     |

## Troubleshooting

### Server Starts but Client Gets Empty Actions

Check the server logs for request errors. For stricter HTTP behavior, launch the server with `--http-400-on-error` so request failures return HTTP 400 instead of an empty action list.

### Success Rate Is Near Zero

Confirm that the following settings match the checkpoint:

- client `--action_space`
- client `--rotation_space`
- client `--action_dim`
- server `--action-chunk-size`
- server `--action-stats-path`
- camera choice and image orientation flags such as `--rotate_180`

### MuJoCo or OpenGL Fails to Initialize

Try an explicit backend:

```shell
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m cosmos_framework.simulation.libero.closed_loop_eval ...
```

If EGL is unavailable, use:

```shell
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m cosmos_framework.simulation.libero.closed_loop_eval ...
```

### LIBERO Config Is Missing

Set `LIBERO_CONFIG_PATH` to a writable directory and rerun the config snippet in [Set Up LIBERO](#set-up-libero).

### `Numba needs NumPy 2.2 or less. Got NumPy 2.4.`

LIBERO pulls in `robosuite`, which depends on `numba`; current `numba` releases support only `numpy<2.3`. Cosmos3 caps `numpy<2.3` via `[tool.uv].override-dependencies` whenever the `libero` group is part of a sync. If you see this error, you most likely synced without that override active — re-resolve the lockfile and reinstall NumPy:

```shell
uv lock --upgrade-package numpy
uv sync --all-extras --group=cu130-train --group=libero --reinstall-package numpy
```
