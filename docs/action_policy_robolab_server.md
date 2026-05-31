# Action Policy RoboLab Server

Use the RoboLab server when your client uses the openpi-style WebSocket protocol. The server accepts msgpack-encoded observation dictionaries with NumPy arrays and returns msgpack-encoded dictionaries containing `action` and, when `--decode-video` is set, `video`.

The server delegates WebSocket protocol handling to OpenPI's `WebsocketPolicyServer`. Install OpenPI's lightweight server package in the Cosmos3 environment before launching:

```shell
uv sync --all-extras --group=cu130-train --group=policy-server
source .venv/bin/activate
```

The `policy-server` group installs `openpi-server`. Alternatively, install the full `Physical-Intelligence/openpi` package if you manage dependencies in a separate environment. Run `uv sync` once for a fresh checkout or when dependency groups change; you do not need to rerun it before every server launch. If your GPU driver does not support CUDA 13, use the matching CUDA group for your node, for example `--group=cu128-train`.

## Start the Server

The primary OSS flow is to serve the consolidated DROID policy checkpoint from Hugging Face.

```shell
python -m cosmos_framework.scripts.action_policy_server_robolab --port 8000
```

By default, the server uses the released DROID RoboLab serving config: `nvidia/Cosmos3-Nano-Policy-DROID` on `main`, `droid_lerobot`, `480` resolution, 15 FPS conditioning, 540x640 input images, 32 action steps, 8-dimensional `joint_pos` actions, guidance 3.0, 4 denoising steps, shift 5.0, and per-request NumPy RNG seeds initialized from seed 0.

You can also pass a local consolidated safetensors directory produced by `cosmos_framework.scripts.export_model`.

```shell
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path /path/to/consolidated/model \
    --port 8000
```

For other checkpoints, set `--resolution`, `--action-chunk-size`, and seed behavior to the values used by that policy's serving config.

The server sends an empty metadata dictionary when a client connects, matching openpi's WebSocket policy server. Each request is an observation dictionary with a `prompt`, `observation/image`, `observation/gripper_position`, and either joint state fields for `--action-space joint_pos` or end-effector pose fields for `--action-space midtrain`.

## Common Options

| Argument                                           | Default                                    | Description                                                                                                                                         |
| -------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--checkpoint-path`                                | `nvidia/Cosmos3-Nano-Policy-DROID`         | `nvidia/Cosmos3-Nano-Policy-DROID`, `Cosmos3-Nano-Policy-DROID`, or a consolidated local safetensors checkpoint directory.                          |
| `--hf-revision`                                    | `main`                                     | Hugging Face revision to download for the public DROID policy checkpoint.                                                                           |
| `--allow-dcp-checkpoint`                           | disabled                                   | Permit direct DCP checkpoint loading for parity/debugging.                                                                                          |
| `--domain-name`                                    | `droid_lerobot`                            | Action domain passed to `get_domain_id()`.                                                                                                          |
| `--decode-video`                                   | disabled                                   | Return decoded rollout video as a uint8 NumPy array.                                                                                                |
| `--action-space`                                   | `joint_pos`                                | Use `joint_pos` or `midtrain` RoboLab postprocessing.                                                                                               |
| `--resolution`                                     | `480`                                      | Action transform resolution. Use `480` for `nvidia/Cosmos3-Nano-Policy-DROID`.                                                                      |
| `--conditioning-fps`                               | `15.0`                                     | Conditioning FPS used by the action transform.                                                                                                      |
| `--action-chunk-size`                              | `32`                                       | Number of action steps to predict per request. Use `32` for `nvidia/Cosmos3-Nano-Policy-DROID`.                                                     |
| `--image-height`                                   | `540`                                      | Input observation image height before action transform preprocessing.                                                                               |
| `--image-width`                                    | `640`                                      | Input observation image width before action transform preprocessing.                                                                                |
| `--action-dim`                                     | `8`                                        | Raw action dimension. Use `8` for DROID `joint_pos`; set explicitly for other action spaces.                                                        |
| `--history-length`                                 | `1`                                        | Number of state/history action rows to trim from the generated action output.                                                                       |
| `--guidance`                                       | `3.0`                                      | Classifier-free guidance scale.                                                                                                                     |
| `--num-steps`                                      | `4`                                        | Number of denoising steps.                                                                                                                          |
| `--shift`                                          | `5.0`                                      | UniPC sampler shift.                                                                                                                                |
| `--seed`                                           | `0`                                        | Base generation seed used to initialize the request RNG.                                                                                            |
| `--deterministic-seed` / `--no-deterministic-seed` | deterministic disabled                     | Use the same seed for every request, or advance a seeded RNG per request. The default advances the RNG for RoboLab parity with the internal server. |

Health check:

```shell
curl http://localhost:8000/healthz
```
