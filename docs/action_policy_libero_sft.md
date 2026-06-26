# Cosmos3-Nano LIBERO action-policy SFT (reproduction)

Reproduces the Cosmos3-Nano LIBERO-10 result (technical report Table 20, ~97.4%
success at checkpoint 2000) as an action policy: vision + language in, action
chunks out. Full SFT (no LoRA) on the public `nvidia/Cosmos3-Nano` base.

Pieces:

| Piece | Path |
| --- | --- |
| Dataset | `cosmos_framework/data/vfm/action/datasets/libero_lerobot_dataset.py` (`LIBEROLeRobotDataset`) |
| SFT wrapper | `get_action_libero_sft_dataset` in `.../datasets/action_sft_dataset.py` |
| Norm stats | `.../datasets/stats/libero_native_frame_wise_relative_rot6d.json` |
| Experiment | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_nano.py` |
| Run TOML | `examples/toml/sft_config/action_policy_libero_repro.toml` |
| Launch | `examples/launch_sft_action_policy_libero.sh` |
| Inference server | `cosmos_framework/scripts/action_policy_server_libero.py` |
| Closed-loop eval | `cosmos_framework/simulation/libero/closed_loop_eval.py` |

## 1. Data

`LIBEROLeRobotDataset` reads a **local** LeRobot dir directly (parquet + video,
like `DROIDLeRobotDataset`) — set `LIBERO_ROOT` to it. Use NVIDIA's **20 FPS**
conversion [`nvidia/LIBERO_LeRobot_v3`](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3)
(public, OpenMDW-1.1), which is what the bundled `quantile_rot` stats and the
20 Hz eval cadence assume. It ships one subdirectory per suite, so pre-sync just
`libero_10`:

```bash
hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
  --include 'libero_10/**' --local-dir <nfs>/LIBERO_LeRobot_v3
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3/libero_10
```

**For the Table-20 number, use `libero_10` ALONE.** Training on the full suite
mix dilutes libero_10 to ~1 pass in 2000 steps (~82%); libero_10 alone is ~2.7
passes (~97%). For more suites, sync the other subdirs and add more
`datasets=dict(...)` entries to the experiment's dataloader.

It uses `frame_wise_relative` rot6d actions (10D = `pos(3) + rot6d(6) +
gripper(1)`), `concat_view` (third-person + wrist, each resized to 256×256,
concatenated horizontally → 256×512), normalized with `quantile_rot` against the
bundled stats.

**FPS-agnostic loader.** It windows by frame index and decodes video at each
frame's real timestamp (no `delta_timestamps` grid), so any LeRobot LIBERO dataset
loads regardless of its `fps` label, and `conditioning_fps` is read from the
dataset's own `meta/info.json`. Prefer the 20 FPS `nvidia/LIBERO_LeRobot_v3` so
`conditioning_fps=20` matches the stats and the eval (serve with `--fps 20`). The
community `lerobot/libero_*` repos carry the *same frames* but label them 10 FPS;
see [§5](#5-fps--stats).

**Model-input resolution = 192×320.** The 256×512 concat is aspect-2.0, so with
`resolution=None` the `ActionTransformPipeline` snaps it to the closest `"256"`
tier canvas — 16:9 → **320×192 (w×h) = 192×320 (h×w)** — by aspect-preserving
resize + bottom reflection pad. The training prompt therefore reads
`"...is of 192x320 resolution."`. Keep this; the eval server reproduces the same
snap (see §4).

## 2. Train (1 node, 8 GPUs)

```bash
export LD_LIBRARY_PATH=''                      # NGC/PyTorch container: avoid torch._C import error
export LIBERO_ROOT=/path/to/libero_10_lerobot  # libero_10 conversion ONLY
export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir>
export WAN_VAE_PATH=<Wan2.2_VAE.pth>
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root

bash examples/launch_sft_action_policy_libero.sh
```

Or drive `cosmos_framework.scripts.train` directly:

```bash
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_policy_libero_repro.toml
```

Recipe knobs live in the registered `action_policy_libero_nano` experiment (full
SFT of the generation + action heads at lr 5e-5 with a 5× LR multiplier on the
action bridge, FusedAdam, selective activation checkpointing, `quantile_rot`
actions, action heads init fresh from the base via `keys_to_skip_loading`). The
TOML sets only run-level scalars: DP=8, `max_iter=10000`, `warm_up_steps=2000`,
`grad_accum_iter=2`, `save_iter=1000`. Checkpoint 2000 is the reference. On
lower-memory GPUs reduce the per-rank batch:
`--opts dataloader_train.max_samples_per_batch=32`.

## 3. Closed-loop eval

Start the policy server on a **trained** checkpoint, then run the LIBERO
simulator client against it. (The base `nvidia/Cosmos3-Nano` DCP has no action
heads — use a checkpoint from §2.)

```bash
# Server (training venv). Loads the DCP (single-rank no_dist), denormalizes with
# quantile_rot + the bundled libero rot6d stats. The experiment supplies the VAE
# path via the override (the server loads the experiment directly, no TOML).
python -m cosmos_framework.scripts.action_policy_server_libero \
  --experiment action_policy_libero_nano \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --checkpoint-path <trained DCP dir, e.g. $OUTPUT_ROOT/.../checkpoints/iter_000002000> \
  --action-normalization quantile_rot \
  --action-stats-path cosmos_framework/data/vfm/action/datasets/stats/libero_native_frame_wise_relative_rot6d.json \
  --raw-action-dim 10 --fps 20 --port 8000
```

**Eval environment** (the LIBERO sim needs a *separate* venv — robosuite/mujoco
versions conflict with the training env, and the NGC image needs graphics
enabled). This combo is validated headless on an NVIDIA GPU:

```bash
# 1. Enable the NVIDIA graphics libs in the container (mounts host libEGL_nvidia
#    etc.); do NOT apt-install libnvidia-gl (it mismatches the mounted driver).
export NVIDIA_DRIVER_CAPABILITIES=all
apt-get install -y libegl1 libglvnd0 libgl1 libglib2.0-0 ffmpeg
mkdir -p /usr/share/glvnd/egl_vendor.d   # ICD (usually already mounted)
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 2. Separate py3.10 venv with LIBERO-compatible sim pins + torch<2.6
#    (torch>=2.6 defaults weights_only=True and breaks LIBERO init-state loads).
uv venv --python 3.10 .libenv && VV=.libenv/bin/python
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && \
  uv pip install -p $VV -e LIBERO -r LIBERO/requirements.txt
uv pip install -p $VV "robosuite==1.4.1" "mujoco==2.3.7" "torch<2.6" loguru requests scipy pillow numpy

# 3. LIBERO first-run config (avoids the interactive prompt) + robosuite macros
mkdir -p ~/.libero && touch ~/.libero/config.yaml
RS=$($VV -c "import robosuite,os;print(os.path.dirname(robosuite.__file__))")
$VV "$RS/scripts/setup_macros.py"
$VV -c "from libero.libero import set_libero_default_path; set_libero_default_path()"

# 4. Run the client (concat agentview+wrist matches the 256x512 training view).
MUJOCO_GL=egl PYTHONPATH=$PWD:$PWD/LIBERO $VV \
  cosmos_framework/simulation/libero/closed_loop_eval.py \
  --server_url http://localhost:8000 \
  --task_suite libero_10 --num_trials_per_task 10 --action_horizon 16 \
  --camera agentview,wrist --image_size 256 \
  --action_space frame_wise_relative --rotation_space 6d --action_dim 10 \
  --save_gifs --gif_fps 20 --output_dir results/libero_closed_loop_10
```

Validated end-to-end against a stub server (episode runs, `summary.json` + GIFs
written, `rc=0`); a benign `EGLError` may print during context teardown on exit.

## 4. Gotchas (from NVIDIA/cosmos-framework#50)

These cost real accuracy if missed; the shipped eval client already handles the
first two, but verify them against your checkpoint:

- **Train ↔ serve parity (resolution + prompt).** Training snaps the 256×512
  concat to a **192×320** model-input canvas (see §1) and the prompt suffix
  encodes that resolution + clip duration (`append_resolution_info` /
  `append_duration_fps_timestamps`). The server applies the *same* snap
  (`get_vision_data_resolution` + `find_closest_target_size` + reflection pad),
  so parity is automatic **as long as the client sends the same 2:1 concat
  layout** — run `closed_loop_eval` with `--camera agentview,wrist --image_size
  256` (agentview left, wrist right, matching training). A single-view client (or
  an old server that skipped the snap) sends a different aspect → different
  canvas → the reported 192×320-train vs 256×512-serve mismatch and ~62% (vs
  ~97%). This is the first thing to check if numbers are low. Note the clip
  *duration* string is computed slightly differently on each side (training's
  rounds to `0.0s`); resolution is the dominant factor — verify both against a
  `--dump_dir` server capture if accuracy is off.
- **Gripper.** The model emits gripper in `[0, 1]`; the LIBERO env wants
  `[-1, 1]` with negative = open. `closed_loop_eval._remap_gripper_to_neg1_pos1`
  applies `1 - 2·g`. If the gripper never opens, the sign is inverted for your
  data — flip it.
- **Image orientation.** Sim frames are rotated 180° relative to training;
  `closed_loop_eval` rotates them back (`img[::-1, ::-1]`).
- **Normalization.** Always start the server with `--action-normalization
  quantile_rot` and the bundled libero rot6d stats file, or actions come out at
  the wrong scale.

## 5. FPS & stats

`LIBEROLeRobotDataset` follows `DROIDLeRobotDataset`: it reads the LeRobot parquet
directly, windows by **frame index**, and decodes video at each frame's **real
timestamp** — so it never builds LeRobot's `delta_timestamps` grid and works at
any native FPS. (The earlier `delta_timestamps` port failed on the 10 FPS public
dataset because a 1/20 s grid doesn't land on 10 FPS frames.)

- **Use the 20 FPS `nvidia/LIBERO_LeRobot_v3`.** LIBERO demos are recorded at
  robosuite's default 20 Hz `control_freq`. NVIDIA's conversion labels them 20 FPS
  (correct); the community `lerobot/libero_*` repos contain the *same frames* (e.g.
  libero_10 = 379 eps / 101,469 frames in both) but label them 10 FPS. Nothing was
  subsampled — only the `fps` metadata differs.
- **Why 20 FPS is the clean choice for THIS eval.** The closed-loop harness steps
  the env at LIBERO's default 20 Hz and applies one predicted action per
  `env.step` (no action-repeat, no `control_freq` override — see `_get_libero_env`
  / `_run_episode`). So the policy's per-action cadence must be 20 Hz. Training on
  the 20 FPS dataset makes `conditioning_fps=20` (read from `meta/info.json`),
  matches the bundled `quantile_rot` stats, and lines up with the eval's 20 Hz —
  serve with `--fps 20`, no harness change.
- **The normalization gap was never the issue.** `normalize_action(quantile)` is an
  *unclamped* affine map `2(a−q01)/(q99−q01)−1`; training and the server share the
  same stats file, so any scale cancels (same reason DROID is fine at its own
  15 FPS). The real consistency requirement is the **control rate**, which the
  20 FPS dataset satisfies by construction.
- **If you must use a differently-labelled dataset**, keep cadence consistent:
  serve at the dataset's `fps`, and if its frames are genuinely sub-sampled (fewer
  frames than the 20 Hz original), either run the eval env at a matching
  `control_freq` or action-repeat. With `nvidia/LIBERO_LeRobot_v3` none of this is
  needed.
- `fps` only sets `conditioning_fps` + prompt duration; the loader always windows
  by frame index and decodes at real timestamps.
