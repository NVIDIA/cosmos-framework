# Cosmos3-Nano LIBERO-10 action-policy SFT

Full SFT of the public `nvidia/Cosmos3-Nano` base into a LIBERO-10 action
policy: vision + language in, action chunks out.

To match the LIBERO-10 SR reported in Cosmos3, we provide **two presets** (both
lr 5e-5, warmup 500, cycle 16000, gbs 2048):

- **(A) libero_10-only** — trains on `libero_10` alone; peaks by ~iter 1500
  (max_iter 2000). Fast.
  `action_policy_libero_nano` + `action_policy_libero_repro.toml` +
  `launch_sft_action_policy_libero.sh`.
- **(B) libero-all** — equal mix of all 4 LIBERO suites; needs longer training
  (max_iter 5000).
  `action_policy_libero_all_nano` + `action_policy_libero_all_repro.toml` +
  `launch_sft_action_policy_libero_all.sh`.

| Piece            | Path                                                                                                 |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| Dataset          | `cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py` (`LIBEROLeRobotDataset`) |
| SFT wrapper      | `get_action_libero_sft_dataset` in `.../datasets/action_sft_dataset.py`                              |
| Norm stats       | `.../normalizer_stats/libero_native_frame_wise_relative_rot6d.json`                                  |
| Experiment       | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_nano.py`      |
| Run TOML         | `examples/toml/sft_config/action_policy_libero_repro.toml`                                           |
| Launch           | `examples/launch_sft_action_policy_libero.sh`                                                        |
| Inference server | `cosmos_framework/scripts/action_policy_server_libero.py`                                            |
| Closed-loop eval | `cosmos_framework/simulation/libero/closed_loop_eval.py`                                             |

## 1. Data

`LIBEROLeRobotDataset` reads a local LeRobot dir. Use the 20 FPS
[`nvidia/LIBERO_LeRobot_v3`](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3),
which the bundled `quantile_rot` stats and the 20 Hz eval assume.

**Preset A (libero_10-only)** — `LIBERO_ROOT` points at the `libero_10` suite dir:

```bash
hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
  --include 'libero_10/**' --local-dir <nfs>/LIBERO_LeRobot_v3
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3/libero_10
```

**Preset B (libero-all)** — download all 4 suites; `LIBERO_ROOT` is the **parent** dir:

```bash
hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --local-dir <nfs>/LIBERO_LeRobot_v3
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3          # parent of libero_spatial/object/goal/10
```

Actions are `frame_wise_relative` rot6d (10D = pos 3 + rot6d 6 + gripper 1),
`concat_view` (third-person + wrist, each 256×256 → 256×512), `quantile_rot`
normalized. The pipeline snaps the 256×512 concat to a 192×320 model canvas; the
eval server reproduces the same snap (§4).

## 2. Train

Common env, then pick a preset launcher:

```bash
export LD_LIBRARY_PATH=''                      # NGC container: avoid torch._C import error
export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir>
export WAN_VAE_PATH=<Wan2.2_VAE.pth>
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root

# Preset A — libero_10-only (LIBERO_ROOT = the libero_10 suite dir):
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3/libero_10
bash examples/launch_sft_action_policy_libero.sh        # HSDP 2x8; set NNODES/NODE_RANK/MASTER_ADDR per node

# Preset B — libero-all 4-suite (LIBERO_ROOT = the LIBERO_LeRobot_v3 parent dir):
export LIBERO_ROOT=<nfs>/LIBERO_LeRobot_v3
bash examples/launch_sft_action_policy_libero_all.sh    # HSDP 2x8; needs ~4500 iters to converge
```

Both recipes set lr 5e-5, warmup 500, cycle 16000, `save_iter=500`, HSDP 2x8 (global
batch 2048 = `max_samples_per_batch` 128 × 16 ranks × grad_accum 1). They differ only in
`max_iter`: **2000** for libero_10-only (peaks ~iter 1500), **5000** for libero-all
(the 4-suite mix takes longer to converge on libero_10, ~iter 4500).

## 3. Closed-loop eval

Start the policy server on a **trained** checkpoint (the base DCP has no action
heads), then run the LIBERO simulator client against it. Same for both presets —
`action_policy_libero_nano` supplies the model config for either run's
checkpoint; just point `--checkpoint-path` at the one you trained.

```bash
python -m cosmos_framework.scripts.action_policy_server_libero \
  --experiment action_policy_libero_nano \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --checkpoint-path <trained DCP dir>/checkpoints/iter_000001500 \
  --action-normalization quantile_rot \
  --action-stats-path cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json \
  --raw-action-dim 10 --fps 20 --port 8000
```

The LIBERO sim needs a separate venv (robosuite/mujoco pins conflict with the
training env):

```bash
# Optional — only on a headless container without working GPU EGL:
#   export NVIDIA_DRIVER_CAPABILITIES=all
#   apt-get install -y libegl1 libglvnd0 libgl1 libglib2.0-0 ffmpeg
#   mkdir -p /usr/share/glvnd/egl_vendor.d
#   echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
#     > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

uv venv --python 3.10 .libenv && VV=.libenv/bin/python
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && \
  uv pip install -p $VV -e LIBERO -r LIBERO/requirements.txt
uv pip install -p $VV "robosuite==1.4.1" "mujoco==2.3.7" "torch<2.6" loguru requests scipy pillow numpy
mkdir -p ~/.libero && touch ~/.libero/config.yaml
RS=$($VV -c "import robosuite,os;print(os.path.dirname(robosuite.__file__))"); $VV "$RS/scripts/setup_macros.py"
$VV -c "from libero.libero import set_libero_default_path; set_libero_default_path()"

MUJOCO_GL=egl PYTHONPATH=$PWD:$PWD/LIBERO $VV \
  cosmos_framework/simulation/libero/closed_loop_eval.py \
  --server_url http://localhost:8000 \
  --task_suite libero_10 --num_trials_per_task 50 --num_envs 8 \
  --camera agentview,wrist --image_size 256 \
  --action_space frame_wise_relative --rotation_space 6d --action_dim 10 \
  --output_dir results/libero_closed_loop_10
```

## 4. Heads-up

- **Lower-memory GPUs** — reduce the per-rank batch:
  `--opts dataloader_train.max_samples_per_batch=64` (scale `replicate` to keep
  global batch 2048).

Eval parity — the client/server already handle these; verify if accuracy is low:

- **Concat layout** — run with `--camera agentview,wrist --image_size 256` so the
  256×512 concat matches training (the server snaps it to 192×320 identically).
- **Gripper** — model emits `[0, 1]`; the env wants `[-1, 1]` (negative = open).
  The client applies `1 − 2·g`; flip the sign if the gripper never opens.
- **Image orientation** — sim frames are rotated 180° vs training; the client
  rotates them back.
- **Normalization** — start the server with `--action-normalization quantile_rot`
  and the bundled rot6d stats, or actions come out at the wrong scale.
