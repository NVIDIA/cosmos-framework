<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos Neural Simulator — Post-Train Cosmos3-Nano as a Neural Simulator of a Robot, with an Inverse-Dynamics Head

> **Authors:** [Saeed Babamohamadi](https://www.linkedin.com/in/saeedbabamohamadi)
> **Organization:** [NVIDIA](https://nvidia.com/)

Post-train the public Cosmos3-Nano model into a neural simulator of a robot: give it one camera frame and 16 actions,
and it generates the 16 frames that follow. Train it on the public LIBERO-10 demonstrations with the forward-dynamics
post-training that ships in `cosmos-framework` (the DROID FD experiment, data source swapped; nothing added to the
model), measure how far the generated video drifts from the real recording over a whole episode, and know what it
costs on one GPU.

The same run can also learn the opposite mapping. A second cell mixes forward dynamics (FD) with inverse dynamics (ID)
at a 50/50 ratio and three frame rates, giving one checkpoint that generates video from actions and recovers actions
from video. This recipe measures what the mixed cell costs in video fidelity and what its inverse-dynamics head is
worth against controls.

| Model                       | Workload                | Use case                                                                                                                    |
| --------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano` (16B) | Post-training, full SFT | First frame + 16 actions in, 16 frames out (forward dynamics); 17 frames in, 16 actions out (inverse dynamics, cell 2 only) |

| ![Held-out episode 26, generated frames 1 to 16: real video, trained simulator, untrained base model](assets/neural-simulator/hero_ep026_chunk00.png) | ![Held-out episode 26, generated frames 225 to 240: real video, trained simulator, untrained base model](assets/neural-simulator/hero_ep026_chunk14.png) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| held-out episode 26, generated frames 1 to 16 (chunk 0). Rows: real video; trained simulator, 31.6 dB on this chunk; untrained base model, 22.1 dB    | same episode, generated frames 225 to 240 (chunk 14, after 14 re-conditioning steps). Rows: real; trained simulator, 20.4 dB; base model, 13.5 dB        |

*Episode 26, put both the alphabet soup and the tomato sauce in the basket, sampling seed 0; third-person view left, wrist view right. [Captions and per-chunk numbers](assets/neural-simulator/captions.md).*

**What you get, and how we judged it.** You get a neural simulator of a robot: give it one camera frame and
a sequence of actions and it shows what the camera would see next. The method applies to your own robot's
demonstrations through the framework's dataset adapter; every number in this document is from LIBERO-10,
open loop, driven by the recorded actions of held-out demonstrations. Running a policy closed-loop inside
the simulator is not part of this recipe and was not measured. We judged the simulator by holding
out 19 demonstrations the model never saw, feeding it their first frame and their real actions, and
comparing the generated video with the real recording frame by frame (PSNR, SSIM, LPIPS), with the
untrained base model as the control and three sampling seeds. That measures whether the video follows
the actions and how far it drifts over an episode. It does not measure task success inside the
simulator: a physics simulator knows where every object is, a neural simulator only produces pixels,
and deciding success from pixels needs a judge, which this recipe does not provide. So this is a
neural simulator, not an evaluator. Use it to preview and debug; do not use it to score checkpoints.

## What You'll Build

- A **Cosmos3-Nano LIBERO-10 forward-dynamics simulator** (cell 1) trained from the public base on 360 public
  demonstrations, on one GB300 (256 GB), in 1000 iterations
- A **second checkpoint trained in mixed FD+ID mode** (cell 2) at the same data, split, token budget, optimizer,
  schedule and training seed, with video sampled at 5, 10 and 20 fps
- A **fidelity evaluation** on 19 held-out episodes: full-episode autoregressive rollout, PSNR / SSIM / LPIPS, drift
  per 16-frame chunk, a gripper-event check; base model as control, three sampling seeds
- An **inverse-dynamics evaluation** of cell 2 against five controls, and a **cost card** for one GB300

## Pipeline

| Stage | What happens                                                                                   | GPU                                                     |
| ----- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| 1     | Download LIBERO-10, hold out 19 episodes, run the leakage check                                | no                                                      |
| 2     | Convert the base checkpoint to the training format; run 10 training iterations as a smoke test | minutes                                                 |
| 3     | Post-train cell 1 (FD) for 1000 iterations, checkpoint every 250                               | 26.94 GB300-hours                                       |
| 4     | Post-train cell 2 (FD+ID, mixed fps) for 1000 iterations, checkpoint every 250                 | 26.69 GB300-hours                                       |
| 5     | Export; evaluate FD fidelity at three sampling seeds; evaluate the ID head against controls    | 16 to 17 min per FD evaluation, 11 to 12 min per ID run |

## Prerequisites

Install cosmos-framework once, following the
[shared setup guide](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/docs/setup.md). The numbers below were measured
on a branch equivalent to public `c23e51f` (`c23e51f2f157ae3e51cfcd86ebfb5464850894f2`) plus the LIBERO FD experiment
files that ship in this folder under `assets/neural-simulator/` (installed in the next block). This folder ships on the
framework's `main` branch, so keep that checkout and run the recipe in a second working tree at `c23e51f`:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git     # main: holds this skill folder
cd cosmos-framework
git worktree add ../cosmos-framework-c23e51f c23e51f2f157ae3e51cfcd86ebfb5464850894f2
cd ../cosmos-framework-c23e51f                               # every command below runs from this working tree
uv sync --all-extras --group=cu130-train     # torch 2.10.0+cu130, python 3.13
export LD_LIBRARY_PATH=''                    # inside NGC containers, and in the shell that launches Python
```

### Install the recipe files into the checkout

Everything the recipe adds to `c23e51f` travels with this skill folder. Run this once from the `c23e51f` working tree
(`$SKILL` is this folder's `assets/neural-simulator/` directory in the `main` checkout next to it):

```bash
export REPO=$PWD                                             # the c23e51f2f157ae3e51cfcd86ebfb5464850894f2 working tree
export SKILL=$(cd $REPO/../cosmos-framework/.agents/skills/cosmos3-neural-simulator-libero/assets/neural-simulator && pwd)

# 1. Patch the five framework files (430 lines): configs/base/config.py registers the two LIBERO experiments;
#    data/generator/action/datasets/{libero_lerobot_dataset,action_sft_dataset,base_dataset}.py add holdout_episodes_path,
#    shuffle_within_shard, 'a+b' mode strings, video_fps_choices and mix_seed; sitecustomize.py preloads the CUDA-13 NPP/NVRTC
#    libraries so torchcodec imports.
git apply --check $SKILL/patches/modified_files_vs_c23e51f.diff && git apply $SKILL/patches/modified_files_vs_c23e51f.diff

# 2. New files: experiment modules, TOMLs, launchers, docs
cp $SKILL/configs/action_fd_libero_posttrain.py $SKILL/configs/action_fd_id_libero_posttrain.py \
   cosmos_framework/configs/base/experiment/action/posttrain_config/
cp $SKILL/toml/action_fd_libero_posttrain.toml $SKILL/toml/action_fd_id_libero_posttrain.toml examples/toml/sft_config/
cp $SKILL/launch/launch_sft_action_fd_libero_posttrain.sh $SKILL/launch/launch_sft_action_fd_id_libero_posttrain.sh examples/
cp $SKILL/configs/docs/action_fd_libero_posttrain.md $SKILL/configs/docs/action_fd_id_libero_posttrain.md docs/

# 3. Recipe scripts and the pinned held-out list (the README's commands call them as scripts/<name> from the framework root)
mkdir -p scripts data
cp $SKILL/scripts/*.py $SKILL/scripts/*.sh scripts/
cp $SKILL/holdout_19.txt data/holdout_19.txt

# 4. Check: the experiment resolves without a traceback
PYTHONPATH=. python -m cosmos_framework.scripts.train --sft-toml examples/toml/sft_config/action_fd_libero_posttrain.toml --dryrun
```

| What                      | From this folder                                                                                    | To, inside the checkout                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Framework patch (5 files) | `assets/neural-simulator/patches/modified_files_vs_c23e51f.diff`                                    | `git apply` at the repo root                                             |
| Experiment modules        | `assets/neural-simulator/configs/action_fd_libero_posttrain.py`, `action_fd_id_libero_posttrain.py` | `cosmos_framework/configs/base/experiment/action/posttrain_config/`      |
| Run configs               | `assets/neural-simulator/toml/*.toml`                                                               | `examples/toml/sft_config/`                                              |
| Launchers                 | `assets/neural-simulator/launch/*.sh`                                                               | `examples/` (they source the shipped `examples/_sft_launcher_common.sh`) |
| Docs                      | `assets/neural-simulator/configs/docs/*.md`                                                         | `docs/`                                                                  |
| Scripts                   | `assets/neural-simulator/scripts/*.py`, `*.sh`                                                      | `scripts/`                                                               |
| Held-out ids              | `assets/neural-simulator/holdout_19.txt`                                                            | `data/holdout_19.txt`                                                    |

Beyond the shared setup:

| Requirement                                                                                         | Why                                                                                                                                 |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Hugging Face login (`hf auth login`)                                                                | model, dataset and VAE are public and ungated; the login avoids download limits                                                     |
| `Wan-AI/Wan2.2-TI2V-5B`, file `Wan2.2_VAE.pth`, revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | the video tokenizer, loaded from `WAN_VAE_PATH`                                                                                     |
| `torchcodec` importable (`python -c "import torch,natten,lerobot,torchcodec"` prints nothing)       | LeRobot decodes the dataset videos with it; on CUDA 13 wheels it may need the NPP/NVRTC libraries preloaded (see Known limitations) |
| About 900 GB of disk for both cells                                                                 | 33 GB base download, 30 GB after conversion, 166 GB per training checkpoint (four per cell), 30 GB per export, 0.6 GB dataset       |

### Hardware

| Configuration          | Post-training                                                                                               | Evaluation                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1x GB300 (256 GB)      | **measured**: 95.8 s per iteration, 218.4 GB peak allocated, 26.94 GB300-hours for 1000 iterations (cell 1) | **measured**: 35.6 GB peak, 2.09 s per 16-frame chunk; ID eval 33.6 GB peak, 1.9 to 2.1 s per window |
| 8x H100 80 GB, 1x H100 | not covered                                                                                                 | not covered                                                                                          |

While training, the GB300 is full (229 to 235 GB by `nvidia-smi`, 255 GB after each save). Evaluate after training or pause it first.

### GPU you need

| Step                                     | Minimum that fits                                                                  | Measured | Time                                                         |
| ---------------------------------------- | ---------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------ |
| Training, one cell, 1000 iterations      | one 256 GB GPU (218 GB allocated); multi-GPU sharding not measured for this recipe | 1× GB300 | 27 h                                                         |
| Fidelity and inverse-dynamics evaluation | one GPU with 40 GB or more (35.6 GB peak)                                          | 1× GB300 | 16 to 17 min per seed; 11 to 12 min per inverse-dynamics run |

## Dataset

| Name                                | Source                                                                                                                                                           | Licence              | Size   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------ |
| LIBERO-10 (LIBERO-Long), LeRobot v3 | [`nvidia/LIBERO_LeRobot_v3`](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3), folder `libero_10/`, revision `e5907374380b8f96511957e6ba5582be52a1e179` | OpenMDW-1.1, ungated | 603 MB |

379 episodes at 20 fps over 10 long-horizon tabletop tasks; two cameras, third-person (`image`) and wrist
(`wrist_image`), 256x256 each; 7-D actions (end-effector delta pose + gripper). What the recipe does with it: actions
become 10-D (3 position, 6 rotation as rot6d, 1 gripper) relative to the previous frame, normalised with the framework's
bundled `quantile_rot` statistics; the two views are placed side by side into one 256x512 image, which the model snaps
to its 320x192 canvas (content area 160x320). All sizes in this document are height x width: the source is 256 high by
512 wide, the model canvas 192 high by 320 wide, the scored content 160 high by 320 wide. Every fidelity metric is
computed on that 160x320 content, not on the 256x512 source.

## Splits

| Split           | Content                                                                                                                                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train           | 360 episodes, 96,479 frames, 90,719 17-frame windows                                                                                                                                                                             |
| held-out (test) | 19 episodes (5 %), stratified 2 per task (1 for task 8), seed 0, 4,990 frames; ids pinned in `data/holdout_19.txt` (the measured list ships as [assets/neural-simulator/holdout_19.txt](assets/neural-simulator/holdout_19.txt)) |

The split is by episode. Leakage check: 0 overlapping episode ids, 0 duplicate action sequences across splits; held-out
first frames are no closer to training frames (nearest-neighbour RMSE min 0.0432, median 0.0594) than training frames are
to each other (0.0321, 0.0591). The held-out episodes are real recordings; generated frames are scored against them and
never enter the test set or the training data. The gripper-event classifier is trained on the 360 training episodes only.
The normalisation statistics are the framework's bundled file and are not train-split-only (see Known limitations).

## Post-training

Two cells, one training seed each (42). Cell 2 differs from cell 1 only in the dataset mode and the frame-rate mix.

| Setting                  | Cell 1 (FD)                                                                                                                                                          | Cell 2 (FD+ID)                                                                                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Experiment / TOML        | `action_fd_libero_posttrain`, `examples/toml/sft_config/action_fd_libero_posttrain.toml` (shipped as `assets/neural-simulator/toml/action_fd_libero_posttrain.toml`) | `action_fd_id_libero_posttrain`, `examples/toml/sft_config/action_fd_id_libero_posttrain.toml` (shipped as `assets/neural-simulator/toml/action_fd_id_libero_posttrain.toml`) |
| Dataset mode             | `forward_dynamics`                                                                                                                                                   | `forward_dynamics+inverse_dynamics`, drawn 50/50 per visit                                                                                                                    |
| Video frame rate         | 20 fps                                                                                                                                                               | `video_fps_choices [5, 10, 20]` per visit, actions dense at 20 Hz                                                                                                             |
| Input / output           | first frame + 16 actions -> 16 frames                                                                                                                                | same, plus 17 frames -> 16 actions                                                                                                                                            |
| Global batch             | 37,000 packed tokens x 16 accumulation steps = 592k tokens per step (about 1,300 windows; 1000 steps is about 14.5 passes)                                           | same token budget                                                                                                                                                             |
| Parallelism              | shard 1 x replicate 1 (one GPU)                                                                                                                                      | same                                                                                                                                                                          |
| Dataloader               | 24 workers, shuffle within shard                                                                                                                                     | same                                                                                                                                                                          |
| Iterations / checkpoints | 1000, save every 250                                                                                                                                                 | same                                                                                                                                                                          |
| Learning rate            | 1e-4, LambdaLinear 0.4 -> 0 over 1000 iterations                                                                                                                     | same                                                                                                                                                                          |
| Optimizer                | FusedAdam (0.9, 0.99), weight decay 0.05, bf16, EMA power 0.1, cfg dropout 0.1                                                                                       | same                                                                                                                                                                          |
| Action heads             | LIBERO rows start from the base checkpoint's untrained initialisation (the base's trained rows belong to DROID and are not read)                                     | same                                                                                                                                                                          |

The LIBERO experiment files (both TOMLs, both launchers, both experiment modules) and the scripts under `scripts/` used below
are not on `c23e51f`. They ship in this folder under `assets/neural-simulator/`: `scripts/` for the evaluation and data scripts,
`toml/` for the two TOMLs, `launch/` for the two launchers, `configs/` for the two experiment modules (and `configs/docs/`
for their docs), and `patches/modified_files_vs_c23e51f.diff` (430 lines) for the five framework files the experiment touches.
[PROVENANCE.md](assets/neural-simulator/PROVENANCE.md) lists every file; the "Install the recipe files" block above copies them into
the framework checkout. The LIBERO experiment is a copy of the shipped DROID FD experiment with the dataset swapped to `LIBEROLeRobotDataset`,
`straggler_detection` and `compile_tokenizer` disabled, and three dataset options added: `holdout_episodes_path`,
`shuffle_within_shard` and, for cell 2, `video_fps_choices`. Because cell 2 spends half of every step on inverse dynamics
and two thirds of its video windows at 5 or 10 fps, it sees about half as many 20 fps FD windows as cell 1 at equal budget.

### Commands, one GB300

All commands run from the framework root with `PYTHONPATH=.` (the launchers set it themselves; the scripts need it for the
framework import and the `sitecustomize.py` preload). `$WORKDIR` holds `data/`, `runs/`, `eval/` and `logs/`; the measured runs
used the layout below, which `scripts/env.sh` reproduces (`source scripts/env.sh` after setting `REPO` and `WORKDIR`).

```bash
export REPO=$PWD WORKDIR=$PWD

# 1. Data and held-out split (pinned revision; the measured download fetched the framework registry's pin ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64,
#    which was the registry default while repo main was e5907374; both give 379 episodes, 603 MB)
hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include 'libero_10/**' \
  --revision e5907374380b8f96511957e6ba5582be52a1e179 --local-dir examples/data/LIBERO_LeRobot_v3
export LIBERO_ROOT=$PWD/examples/data/LIBERO_LeRobot_v3/libero_10
python scripts/make_holdout.py --root $LIBERO_ROOT --out data/holdout_19.txt   # 19 ids, 2 per task (1 for task 8); must equal the shipped holdout_19.txt
export LIBERO_HOLDOUT_PATH=$PWD/data/holdout_19.txt
# --stats = the framework's bundled quantile_rot statistics for LIBERO (libero_native_frame_wise_relative_rot6d.json, the file
# LIBEROLeRobotDataset loads when action_stats_path is unset; find it with: find cosmos_framework -name 'libero_native_frame_wise_relative_rot6d.json')
PYTHONPATH=. python scripts/leakage_check.py --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH \
  --stats $(find cosmos_framework -name 'libero_native_frame_wise_relative_rot6d.json' | head -1) --out data/leakage_check.json

# 2. Base checkpoint and tokenizer (pinned revisions) -> training format (DCP). CPU only, 70 s, once.
#    Pass the local directory to the converter: the catalog name `Cosmos3-Nano` would fetch whatever `main` points at that day.
hf download nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --local-dir hf/Cosmos3-Nano   # 33 GB
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir examples/checkpoints/wan22_vae
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Nano --checkpoint-path $PWD/hf/Cosmos3-Nano
export BASE_CHECKPOINT_PATH=$PWD/examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth

# Smoke test: 10 iterations, one checkpoint, its own output root. A distinct job.name keeps the main run from auto-resuming from it.
OUTPUT_ROOT=$WORKDIR/runs/smoke IMAGINAIRE_OUTPUT_ROOT=$WORKDIR/runs/smoke \
EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 trainer.logging_iter=1 job.name=action_fd_libero_smoke10" \
NPROC_PER_NODE=1 bash examples/launch_sft_action_fd_libero_posttrain.sh

# 3. Cell 1: 1000 iterations, checkpoint every 250 (the launcher's defaults); output root runs/main
OUTPUT_ROOT=$WORKDIR/runs/main IMAGINAIRE_OUTPUT_ROOT=$WORKDIR/runs/main \
EXTRA_TAIL_OVERRIDES="trainer.callbacks.wandb.log_train_loss_to_console=True" \
NPROC_PER_NODE=1 bash examples/launch_sft_action_fd_libero_posttrain.sh

# 4. Cell 2: same, mixed FD+ID with video at 5/10/20 fps; output root runs/cell2, its own MASTER_PORT, per-visit draws seeded by mix_seed
OUTPUT_ROOT=$WORKDIR/runs/cell2 IMAGINAIRE_OUTPUT_ROOT=$WORKDIR/runs/cell2 MASTER_PORT=50013 \
EXTRA_TAIL_OVERRIDES="trainer.callbacks.wandb.log_train_loss_to_console=True dataloader_train.dataloader.datasets.libero.dataset.mix_seed=42" \
NPROC_PER_NODE=1 bash examples/launch_sft_action_fd_id_libero_posttrain.sh
```

Start each run in `tmux` or with `nohup`, never under `timeout`: a checkpoint is 166 GB and takes about 230 s to write,
and a run must not be stopped while `checkpoints/iter_*` is being written. A second `torchrun` needs its own `MASTER_PORT`.
Relaunching with the same output root and `job.name` resumes from `checkpoints/latest_checkpoint.txt`; when cell 2 is
relaunched after a stop, pass a different `mix_seed` (the measured run used 42000 + the resumed iteration: 42250, 42500, 42750)
so the per-visit mode and frame-rate draws are not replayed. The run directories are
`$WORKDIR/runs/main/cosmos3_action_fd/action_sft/action_fd_libero_posttrain/` and
`$WORKDIR/runs/cell2/cosmos3_action_fd/action_sft/action_fd_id_libero_posttrain/` (`job.project/group/name` from the TOMLs),
each with `config.yaml` and `checkpoints/iter_000000250 … iter_000001000`.

### Export and evaluate

```bash
RUN=$WORKDIR/runs/main/cosmos3_action_fd/action_sft/action_fd_libero_posttrain
CELL2=$WORKDIR/runs/cell2/cosmos3_action_fd/action_sft/action_fd_id_libero_posttrain
EVAL=$WORKDIR/eval

# Export: 30 GB bf16 safetensors with EMA weights, 7 shards; CPU only, 69 to 79 s, 32 GB host RAM
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m cosmos_framework.scripts.export_model --checkpoint-path $RUN/checkpoints/iter_000001000 \
  --config-file $RUN/config.yaml -o $WORKDIR/export_iter1000

# Gripper-event classifier, once: wrist-view CNN trained on the 360 training episodes only (CPU, 424 s), 93.6 % accurate on held-out real frames
PYTHONPATH=. python scripts/gripper_classifier.py train --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --out $EVAL/gripper --label-source cmd --cpu

# Forward-dynamics fidelity: autoregressive rollout over each held-out episode, one sampling seed per call.
# scripts/run_eval.sh <iter> [run_dir] [extra args] wraps the three commands of each block below (EVAL_CKPT/EVAL_CFG/EVAL_OUT/EVAL_TITLE override them).
# Cell 1, iteration 1000, seeds 0 / 1 / 2  ->  eval/iter1000, eval/iter1000_seed1, eval/iter1000_seed2
for SEED in 0 1 2; do
  OUT=$EVAL/iter1000; [ $SEED -gt 0 ] && OUT=${OUT}_seed$SEED
  PYTHONPATH=. python scripts/fd_rollout_eval.py --checkpoint-path $RUN/checkpoints/iter_000001000 --config-file $RUN/config.yaml \
    --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed $SEED --out $OUT
  PYTHONPATH=. python scripts/gripper_classifier.py apply --model $EVAL/gripper/gripper_net.pt --eval-dir $OUT   # visible open/close events, generated vs real
  PYTHONPATH=. python scripts/make_report.py --eval-dir $OUT --title "LIBERO FD eval, iteration 1000, seed $SEED" --n-videos 10   # drift curves and a real | generated review page
done

# Controls and baselines of the fidelity table: same three commands per run, only checkpoint, config, seed and output change.
# Base model, no post-training (regular weights: the base checkpoint has no EMA copy; the architecture is read from cell 1's config.yaml)
PYTHONPATH=. python scripts/fd_rollout_eval.py --checkpoint-path $BASE_CHECKPOINT_PATH --config-file $RUN/config.yaml --no-ema \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --out $EVAL/control_base
# Cell 1, iteration 250
PYTHONPATH=. python scripts/fd_rollout_eval.py --checkpoint-path $RUN/checkpoints/iter_000000250 --config-file $RUN/config.yaml \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --out $EVAL/iter0250
# Cell 2, iteration 250
PYTHONPATH=. python scripts/fd_rollout_eval.py --checkpoint-path $CELL2/checkpoints/iter_000000250 --config-file $CELL2/config.yaml \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --out $EVAL/cell2_iter0250
# Cell 2, iteration 1000, seeds 0 / 1 / 2  ->  eval/cell2_iter1000, eval/cell2_iter1000_seed1, eval/cell2_iter1000_seed2
for SEED in 0 1 2; do
  OUT=$EVAL/cell2_iter1000; [ $SEED -gt 0 ] && OUT=${OUT}_seed$SEED
  PYTHONPATH=. python scripts/fd_rollout_eval.py --checkpoint-path $CELL2/checkpoints/iter_000001000 --config-file $CELL2/config.yaml \
    --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed $SEED --out $OUT
done
# (run gripper_classifier.py apply and make_report.py on each of these directories as in the loop above)

# Fidelity table with mean ± population std over the seed directories, and the FD+ID minus FD delta row
PYTHONPATH=. python scripts/compare_cells.py --out $EVAL/cells_final_table.md \
  "control_base=$EVAL/control_base" "cell1_iter250=$EVAL/iter0250" "cell2_iter250=$EVAL/cell2_iter0250" \
  "cell1_iter1000=$EVAL/iter1000,$EVAL/iter1000_seed1,$EVAL/iter1000_seed2" \
  "cell2_iter1000=$EVAL/cell2_iter1000,$EVAL/cell2_iter1000_seed1,$EVAL/cell2_iter1000_seed2" --delta cell2_iter1000 cell1_iter1000
PYTHONPATH=. python scripts/drift_compare.py --out $EVAL/cells_iter1000_drift.png "control=$EVAL/control_base" "FD iter 1000=$EVAL/iter1000" "FD+ID iter 1000=$EVAL/cell2_iter1000"

# Inverse dynamics: 301 non-overlapping 17-frame real windows, 11 to 12 min per run. The two constant controls of the table
# (mean training action, zero motion) are analytic and computed inside every run (summary.json -> baselines; RECORD.md columns); no separate command.
# Cell 2, iteration 1000, seeds 0 / 1 / 2  ->  eval/id_cell2_iter1000, eval/id_cell2_iter1000_seed1, eval/id_cell2_iter1000_seed2
for SEED in 0 1 2; do
  OUT=$EVAL/id_cell2_iter1000; [ $SEED -gt 0 ] && OUT=${OUT}_seed$SEED
  PYTHONPATH=. python scripts/id_eval.py --checkpoint-path $CELL2/checkpoints/iter_000001000 --config-file $CELL2/config.yaml \
    --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed $SEED --out $OUT \
    --title "cell 2 (FD+ID, mixed fps) iteration 1000, seed $SEED"
done
# FD-only control: cell 1, iteration 1000 (its LIBERO ID output rows were never trained)
PYTHONPATH=. python scripts/id_eval.py --checkpoint-path $RUN/checkpoints/iter_000001000 --config-file $RUN/config.yaml \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --out $EVAL/id_cell1_iter1000 \
  --title "cell 1 (FD only) iteration 1000, seed 0"
# Base model control (regular weights)
PYTHONPATH=. python scripts/id_eval.py --checkpoint-path $BASE_CHECKPOINT_PATH --config-file $RUN/config.yaml --no-ema \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --out $EVAL/id_control_base \
  --title "control: base Cosmos3-Nano, seed 0"
# Shuffled-video control: cell 2 shown the video of a different window of the same episode
PYTHONPATH=. python scripts/id_eval.py --checkpoint-path $CELL2/checkpoints/iter_000001000 --config-file $CELL2/config.yaml \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --video-control shuffled \
  --out $EVAL/id_cell2_iter1000_shuffled --title "cell 2 iteration 1000, shuffled video, seed 0"
```

Which weights were scored: every fidelity and inverse-dynamics number in Results was produced by loading the full
training checkpoint (`checkpoints/iter_000001000`, or `iter_000000250` for the iteration-250 columns) with its `config.yaml`.
That path loads the regular weights: `fd_rollout_eval.py` and `id_eval.py` default to `use_ema_weights: true` (recorded in each
run's `run_config.json`), but with a YAML config the framework loader ignores that flag, so the reported numbers are
regular-weight numbers (records: the run-record correction of 2026-10-06; [PROVENANCE.md](assets/neural-simulator/PROVENANCE.md),
section 2). The base-model control was run with `--no-ema` (the base checkpoint has no EMA copy), so it is a regular-weight
number as well. The 30 GB export holds the EMA weights and is the artifact of record for serving; it has not been evaluated.
Confirmed from the three seed runs' configuration files.

The rollout starts from the real first frame with the real actions and re-conditions on its own generated frames for
the rest of the episode, so the drift it reports is true drift; metrics are per frame at 160x320 content with the first
(given) frame excluded. The ID evaluation feeds 301 non-overlapping 17-frame real windows on the same chunk grid with
the action input zeroed, and checks in every run that the output is not bit-identical to the real action in the batch.

## Results / Expected Output

Measured on one GB300 (256 GB, driver 595, torch 2.10.0+cu130), cosmos-framework `c23e51f` equivalent, 360 training
demonstrations, one training seed per cell. FD evaluation: 19 held-out episodes, 4,971 generated frames, 20 fps rollout
over the full episode, UniPC 30 steps, guidance 1.0, shift 3.0, regular weights of the training checkpoint (see "Which
weights were scored"), all metrics at 160x320 content. Columns with
a ± are mean ± population std over sampling seeds 0, 1, 2; other columns are one sampling seed. The control is the base
model, no post-training, under the identical rollout protocol. Every number below is listed with its source record in
[results/metrics.json](assets/neural-simulator/results/metrics.json).

### Forward-dynamics fidelity (horizon: full episode unless a chunk is named)

| Metric                                                                                                                                                                        | Base model, control (1 seed)                                                                                                                          | FD iter 250 (1 seed)  | FD+ID iter 250 (1 seed) | FD iter 1000 (3 seeds) | FD+ID iter 1000 (3 seeds) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ----------------------- | ---------------------- | ------------------------- |
| PSNR, dB                                                                                                                                                                      | 12.83                                                                                                                                                 | 17.73                 | 17.85                   | **20.58 ± 0.06**       | 20.20 ± 0.07              |
| SSIM                                                                                                                                                                          | 0.402                                                                                                                                                 | 0.711                 | 0.712                   | **0.781 ± 0.002**      | 0.773 ± 0.002             |
| LPIPS (lower is better)                                                                                                                                                       | 0.534                                                                                                                                                 | 0.211                 | 0.213                   | **0.131 ± 0.002**      | 0.137 ± 0.002             |
| PSNR chunk 0 / 5 / 10 (frames 1-16 / 81-96 / 161-176), dB                                                                                                                     | 19.42 / 12.69 / 11.96                                                                                                                                 | 27.80 / 17.66 / 16.27 | 26.75 / 17.66 / 16.66   | 31.06 / 20.56 / 19.27  | 29.88 / 20.12 / 18.90     |
| PSNR third-person / wrist view, dB                                                                                                                                            | 13.13 / 13.13 (whole frame 12.83; lower than either half because PSNR is a logarithm of the error, verified by recomputing from the per-frame errors) | 19.71 / 16.96         | 20.07 / 17.11           | 23.49 / 19.41          | 23.03 / 19.04             |
| Gripper command followed by a visible open/close within 20 frames, wrist-view classifier 93.6 % accurate, 65 command changes (real recordings: 0.692, the detector's ceiling) | 0.046                                                                                                                                                 | 0.631                 | 0.615                   | 0.600 ± 0.045          | 0.646 ± 0.038             |
| Gripper per-frame state agreement with the real video                                                                                                                         | 0.536                                                                                                                                                 | 0.860                 | 0.860                   | 0.914 ± 0.011          | 0.923 ± 0.005             |

### What the numbers look like

| ![Episode 309, chunk 14: real video, FD simulator, FD+ID simulator](assets/neural-simulator/fd_vs_fdid_ep309_fd_wins_chunk14.png)                                 | ![Episode 62, chunk 12: real video, FD simulator, FD+ID simulator](assets/neural-simulator/fd_vs_fdid_ep062_fdid_wins_chunk12.png)                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| episode 309, chunk 14: the largest lead of FD over FD+ID in the held-out set. Rows: real; FD, 21.3 dB; FD+ID, 18.8 dB. Three-seed episode means 22.4 against 21.5 | episode 62, chunk 12: the largest lead of FD+ID over FD (one of 4 episodes of 19). Rows: real; FD, 18.9 dB; FD+ID, 19.7 dB. Three-seed episode means 21.3 against 21.6 |

Sampling seed 0 for the frames; the three-seed means are from the fidelity table. [Captions](assets/neural-simulator/captions.md).

### Drift curve, FD iter 1000 (mean PSNR per 16-frame chunk, 160x320 content, one sampling seed)

| Chunk (16 frames each; chunk 14 ends about 240 frames, 12 s, into the rollout) | 0    | 1    | 2    | 4    | 6    | 8    | 10   | 14   |
| ------------------------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| PSNR, dB                                                                       | 31.0 | 27.7 | 24.1 | 21.3 | 19.4 | 18.8 | 19.2 | 16.7 |

For comparison, the base model control sits at 19.42 / 12.69 / 11.96 dB at chunks 0 / 5 / 10 and FD+ID iter 1000 at
29.88 / 20.12 / 18.90 dB. **FD+ID versus FD at iteration 1000** (three sampling seeds each, one training seed each): PSNR -0.38 dB (standard error
of the difference 0.05 dB), SSIM -0.008, LPIPS +0.006. The gap is largest right after the conditioning frame (chunk 0:
-1.18 dB), shrinks along the rollout (chunk 10: -0.36 dB) and reverses at the end (chunk 15: +0.16 dB). FD+ID is better
in 4 of 19 episodes on PSNR; gripper reproduction is equal within sampling-seed noise; at iteration 250 the cells were at
parity (17.85 vs 17.73 dB). The difference is seven times the sampling-seed noise, but each cell is one training seed and
the training-seed spread for this recipe is not measured, so read the 0.38 dB as a single-training-seed estimate.

### Inverse dynamics, iteration 1000

Protocol: 301 non-overlapping 17-frame real windows from the 19 held-out episodes, 4,816 target actions, mode
`inverse_dynamics`, UniPC 30 steps, guidance 1.0, shift 3.0, regular weights of the training checkpoint (see "Which weights
were scored"; the base model has no EMA copy). Horizon: 16 actions per window (frame-wise
error), the chunk end (16 frames integrated) and the episode end. Millimetres are command units x 11.62, a least-squares
fit on the training episodes; "degrees" are degrees of the command rotation. Controls: constant mean training action,
zero motion, the base model, the FD-only cell 1 (its ID output rows are untrained), and cell 2 shown the video of a
different window of the same episode (shuffled video). FD+ID is mean ± population std over sampling seeds 0, 1, 2; the
other columns are one sampling seed or analytic.

| Metric                                               | Mean training action | Zero motion                   | Base model                | FD only (cell 1)          | FD+ID (cell 2, 3 seeds)       | FD+ID, shuffled video         |
| ---------------------------------------------------- | -------------------- | ----------------------------- | ------------------------- | ------------------------- | ----------------------------- | ----------------------------- |
| Position error, mm per frame                         | 6.05                 | 6.13                          | 17.43                     | 17.24                     | **0.71 ± 0.00**               | 8.07                          |
| Rotation error, degrees, mean / median               | 4.81 / 3.52          | 4.80 / 3.56                   | 18.93 / 17.77             | 19.01 / 17.88             | **0.88 / 0.69**               | 7.31 / 5.78                   |
| Gripper sign accuracy                                | 0.545                | 0.545                         | 0.507                     | 0.507                     | **0.997**                     | 0.635                         |
| Gripper transitions recall / precision (±10 frames)  | 0.00 / n.a.          | 0.00 / n.a.                   | 1.00 / 0.03               | 1.00 / 0.03               | **0.97 / 0.87**               | 0.30 / 0.27                   |
| R² x / y / z / gripper                               | 0 / 0 / 0 / 0        | -0.01 / -0.03 / -0.02 / -0.84 | -8.2 / -5.8 / -6.8 / -1.0 | -7.8 / -5.7 / -6.7 / -1.0 | **0.96 / 0.99 / 0.98 / 0.99** | -1.06 / -0.60 / -0.76 / -0.47 |
| Integrated path error at chunk end / episode end, mm | 87.7 / 174.6         | 89.4 / 304.5                  | 116.7 / 579.3             | 115.5 / 531.9             | **7.5 / 48.5**                | 116.0 / 49.2                  |

Per sampling seed (0 / 1 / 2) for the FD+ID column: position error 0.72 / 0.71 / 0.71 mm per frame; rotation 0.88 / 0.88 / 0.89
degrees (median 0.68 / 0.69 / 0.69); gripper sign accuracy 0.998 / 0.998 / 0.997; transitions recall 0.97 in all three at
precision 0.88 / 0.88 / 0.84 (hence 0.87 ± 0.02); integrated error 7.6 / 7.4 / 7.5 mm at the chunk end and 48.6 / 48.4 / 48.3 mm
at the episode end; R² identical to two decimals in all three. The spread is the population std over these three runs.
Per episode, FD+ID position error ranges 0.46 to 1.24 mm. The base model and FD-only cell flip the gripper constantly, hence recall 1.00 at precision 0.03.

**Reading.** Training in mixed FD+ID mode costs 0.38 dB of 20 fps video fidelity, concentrated in the first 32 generated
frames, and yields an inverse-dynamics head with 0.71 mm per-frame position error, an order of magnitude better than the
best constant predictor (6.05 mm). FD-only training has no usable inverse-dynamics head: its 17.24 mm is as bad as the
base model. The shuffled-video control (8.07 mm per frame, 116 mm at chunk end, both worse than a constant) shows that
cell 2 reads the motion from the video over a chunk. The episode-end integrated error does not separate them (48.5 mm
against 49.2 mm for shuffled video), so the claim rests on the per-frame and chunk-end numbers only.

### Cost card, one GB300

| Item                                                                                 | Cell 1 (FD)                                                                                                            | Cell 2 (FD+ID)                                                      |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Time per iteration                                                                   | 95.8 s median                                                                                                          | 92 to 99 s                                                          |
| GPU memory during training                                                           | torch 218.4 GB peak allocated, 222.7 GB reserved; `nvidia-smi` 229 to 235 GB steady, 255 GB after each checkpoint save | torch 218.6 GB peak allocated; `nvidia-smi` not measured separately |
| GPU-hours to iteration 250 / 500 / 750 / 1000                                        | 6.79 / 13.42 / 20.28 / 26.94 (29.24 consumed including one restart)                                                    | 26.69 at 1000                                                       |
| Smoke run (10 iterations at 74k tokens x accumulation 8)                             | 95.9 s per iteration, 241 GB allocated, 254 GB by `nvidia-smi`                                                         | same launcher                                                       |
| Checkpoint write (166 GB)                                                            | 230 s                                                                                                                  | same                                                                |
| Export (30 GB, bf16 EMA, 7 shards), CPU only                                         | 69 to 79 s, 32 GB host RAM                                                                                             | same                                                                |
| Training loss (cell 1: single FD loss; cell 2: two losses, not comparable to cell 1) | FD loss 1.169 (iter 10) -> 0.595 (250) -> 0.609 (1000)                                                                 | vision loss 0.0853 -> 0.0374; action loss 0.4266 -> 0.0024          |
| FD evaluation, 19 episodes                                                           | 2.09 s per 16-frame chunk, 16 to 17 min per seed, 35.6 GB peak                                                         | same                                                                |
| ID evaluation, 301 windows                                                           | n.a. (no trained head)                                                                                                 | 1.9 to 2.1 s per window, 11 to 12 min per run, 33.6 GB peak         |

### How you know it worked

- The smoke run saves `iter_000000010` with a finite loss and the training log says `kept 360/379`.
- Cell 1's FD loss falls from about 1.17 at iteration 10 to about 0.6 by iteration 250 and stays there. Cell 2 logs a
  vision loss and an action loss separately; neither is on cell 1's scale.
- The FD evaluation at iteration 1000 is far above the base-model control on the same 19 episodes: PSNR about 20.6 dB
  against 12.83, SSIM about 0.78 against 0.402, LPIPS about 0.13 against 0.534, all at 160x320 content over the full episode.
- The first chunk is sharp (about 31 dB) and the rollout degrades smoothly rather than collapsing: chunk 10 stays near
  19 dB where the control is at 11.96 dB. Gripper commands are visibly followed in about 60 % of cases (control 4.6 %,
  real recordings 69 %).
- For cell 2, the inverse-dynamics position error is below 1 mm per frame; the FD-only cell and the base model give
  about 17 mm and the constant predictors about 6 mm.

## Prompt and agent guide

**Prompts.** The forward-dynamics model is conditioned on the first frame and the action chunk; the task instruction
string from the dataset is passed as the text input and held fixed per episode, exactly as in training. There is no
scene prompt to write: the appearance comes from the first frame, the motion from the actions. If you post-train on
your own demonstrations, pass the same instruction strings you use for your policy, so the simulator and the policy
share one vocabulary.

**Running this recipe with a coding agent.** What to give an agent that runs this recipe from the document alone:

- Pin everything the agent cannot guess: framework commit, model and dataset revisions, the exact command block,
  the output directory layout. An agent that has to choose a version will choose a different one each run.
- Give it a stop rule and a budget in GPU-hours, and the three checks under "How you know it worked". Without a
  stop rule it keeps going past a finished result; without the checks it reports a broken run as done.
- Make it write a record per run (data, split, GPU type and count, peak memory, GPU-hours, result with seeds,
  control, leakage check) before it reports. A number without a record is not a result.
- Tell it what it may not do: delete checkpoints, change a pinned revision, write a secret to disk, enter
  generated data into a test set. Everything else it should fix itself and note in the record.
- Expect it to hit the deviations listed under Known limitations; they are the ones a careful reader hits too.
- For this recipe specifically: the agent must evaluate on the held-out episodes only, run the leakage check before
  training, state which weights it scored (checkpoint or export; they differ, see "Export and evaluate"), and report
  resolution and horizon next to every fidelity number.

## Known limitations

- **Not a policy evaluator yet.** Whether this neural simulator ranks policy checkpoints like the physics simulator is not
  measured; the ranking harness is designed but not run. Task success inside the neural simulator is not measured either.
- **Generation at 5 and 10 fps is not measured.** Cell 2 trained on those rates; every fidelity number here is a 20 fps rollout.
- **One training seed per cell.** FD versus FD+ID is judged against sampling-seed noise only; training-seed spread is not measured.
- **Normalisation statistics are not train-split-only.** The bundled `quantile_rot` file was computed on a four-suite
  LIBERO pool. It deviates from a recomputation on the 360 training episodes by up to 0.096 (q01) / 0.102 (q99) of the
  per-dimension range; the 19 held-out episodes shift the libero_10 quantiles by at most 0.0043 of range, so leakage
  through these constants is negligible.
- **Memory climbs after each checkpoint save.** `nvidia-smi` rose to 255 GB on the 256 GB card after every save; the run
  was stopped and relaunched after iterations 500 and 750, and iterations 501 to 566 were recomputed (2.07 h), the gap
  between 26.94 productive and 29.24 consumed GPU-hours.
- **The 0.38 dB is a cost at equal token budget, not at equal FD exposure** (cell 2 saw about half as many 20 fps FD windows).
- **Cosmos3-Super and Cosmos3-Edge as the simulator are not covered.**

## Document Information

|                  |                                                                                                                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Framework        | `cosmos-framework` `c23e51f2f157ae3e51cfcd86ebfb5464850894f2` (v1.2.2) equivalent, plus the LIBERO forward-dynamics experiment files and patch in `assets/neural-simulator/`                                         |
| Model            | `nvidia/Cosmos3-Nano` (public, ungated), revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`                                                                                                                         |
| Tokenizer        | `Wan-AI/Wan2.2-TI2V-5B`, file `Wan2.2_VAE.pth`, revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`                                                                                                                  |
| Data             | `nvidia/LIBERO_LeRobot_v3`, folder `libero_10/`, revision `e5907374380b8f96511957e6ba5582be52a1e179` (the measured download used the framework registry pin `ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64`), OpenMDW-1.1 |
| Measured on      | 1x GB300 (256 GB), October 2026                                                                                                                                                                                      |
| Exports          | not published; cell 1 iteration 1000 was exported as 30 GB bf16 EMA safetensors (7 shards) and not evaluated                                                                                                         |
| Companion recipe | [Cosmos Policy](../cosmos3-nano-robot-manipulation/README.md): post-train Cosmos3-Nano into a LIBERO-10 manipulation policy (same data, opposite direction)                                                          |
