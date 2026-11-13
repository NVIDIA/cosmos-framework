---
name: cosmos3-neural-simulator-libero
description: >
  Post-train Cosmos3-Nano into a forward-dynamics neural simulator of a LIBERO robot, optionally with an inverse-dynamics head,
  and measure held-out rollout fidelity against the untrained base model.
  Use when the user asks for a neural simulator or world model of a robot, forward- or inverse-dynamics post-training,
  recovering actions from video, or held-out rollout fidelity (PSNR, SSIM, LPIPS) on LIBERO.
metadata:
  industry: ROB
  gpu-budget: XL
---

# Cosmos Neural Simulator: Cosmos3-Nano forward dynamics on LIBERO-10 (one GB300)

One job: post-train the public `nvidia/Cosmos3-Nano` so that, given one camera frame and 16 actions, it generates the 16
frames that follow (forward dynamics, FD). Trained on 360 LIBERO-10 demonstrations with the framework's DROID FD
experiment, data source swapped; a second cell mixes in inverse dynamics (ID) so one checkpoint also recovers actions
from video. Measured on 19 held-out episodes, full-episode rollout: **PSNR 20.58 ± 0.06 dB against 12.83 for the
untrained base** (SSIM 0.781 vs 0.402, LPIPS 0.131 vs 0.534); the ID head of cell 2 reaches 0.71 mm per-frame position
error against 6.05 for the best constant predictor. About 27 GB300-hours per cell. Full write-up: [README.md](README.md).

## For an agent

> Using the `cosmos3-neural-simulator-libero` skill in this Cosmos Framework checkout: set up the framework at the
> pinned revision in a second working tree, install the recipe files, download LIBERO-10 and the base model, hold out
> 19 episodes and run the leakage check, run the 10-iteration smoke test, train cell 1 for 1000 iterations, evaluate
> forward-dynamics fidelity at three sampling seeds against the base-model control, and report the fidelity table.

Rules: run the steps in order; after each step compare the output with its "Expected"; stop at the first failure and
report the step, the error and the log path; evaluate on the 19 held-out episodes only; run the leakage check before
training; state which weights you scored (training checkpoint, regular weights) and the resolution and horizon next to
every fidelity number; never delete a checkpoint. Budget: about 30 GB300-hours for cell 1 with its evaluation; cell 2
(step 7) is a second 27 GB300-hours and optional.

## Read this first

1. **Two checkouts.** This skill folder ships on `main`; the recipe runs in a second working tree at
   `c23e51f2f157ae3e51cfcd86ebfb5464850894f2` (v1.2.2) with this skill's patch and files installed into it (step 2).
   Every command runs from the root of that working tree with `PYTHONPATH=.`. Step 1 writes the variables to
   `recipe.env` there; every later block starts with `source recipe.env`.
2. **GPU.** Training needs one 256 GB GPU (218 GB allocated; multi-GPU sharding not measured for this recipe).
   Evaluation needs 40 GB (35.6 GB peak). While training, the GB300 is full: evaluate after training.
3. **Disk.** About 900 GB for both cells (166 GB per checkpoint, four per cell); about 250 GB for cell 1 keeping only the
   last checkpoint.
4. **Long steps run detached**, never under `timeout`: a checkpoint takes about 230 s to write. A second `torchrun` on
   the same host needs its own `MASTER_PORT`.
5. **Not a policy evaluator.** The result is video fidelity to real recordings under the recorded actions. Whether the
   simulator ranks policies like the physics simulator is not measured; do not use it to score checkpoints.

## When to use this skill

- Post-train Cosmos3-Nano as a neural simulator of a robot from demonstrations, or reproduce the fidelity table
- Add an inverse-dynamics head (actions from video) in the same run and measure what it costs in video fidelity
- Measure how far generated video drifts from a real recording over a whole episode (PSNR per 16-frame chunk)
- Not for: a manipulation policy (`../cosmos3-nano-robot-manipulation/SKILL.md`); the released two-arm simulator
  checkpoint (`../cosmos3-sim-bimanual-agibot/SKILL.md`); text- or image-conditioned video (`../cosmos3-generate-media/SKILL.md`)

## Hardware (measured on one GB300, 256 GB)

| Step                                   | Time                                                 | GPU memory                              |
| -------------------------------------- | ---------------------------------------------------- | --------------------------------------- |
| Smoke test, 10 iterations              | about 16 min plus model load                         | 241 GB allocated                        |
| One cell, 1000 iterations              | 95.8 s per iteration, 26.94 GB300-hours              | 218.4 GB allocated, 255 GB after a save |
| FD evaluation, 19 episodes, one seed   | 16 to 17 min (2.09 s per 16-frame chunk)             | 35.6 GB peak                            |
| ID evaluation, 301 windows, one seed   | 11 to 12 min                                         | 33.6 GB peak                            |
| Export (CPU)                           | 69 to 79 s, 32 GB host RAM                           | none                                    |

Not covered: 8 × H100, 1 × H100, Cosmos3-Super or Edge as the simulator.

## Bundled assets

| Asset                                                                                                                                                                                                         | Used for                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [`assets/neural-simulator/patches/modified_files_vs_c23e51f.diff`](assets/neural-simulator/patches/modified_files_vs_c23e51f.diff)                                                                            | step 2: the five framework files the LIBERO experiments need (430 lines)              |
| `assets/neural-simulator/configs/`, `assets/neural-simulator/toml/`, `assets/neural-simulator/launch/`                                                                                                        | step 2: the two experiments (FD, FD+ID), their run configs, launchers and docs        |
| `assets/neural-simulator/scripts/`                                                                                                                                                                            | steps 3, 6, 7: holdout, leakage check, rollout evaluation, classifier, report, tables |
| [`assets/neural-simulator/holdout_19.txt`](assets/neural-simulator/holdout_19.txt)                                                                                                                            | step 3: the pinned held-out episode ids; `make_holdout.py` must reproduce it          |
| [`assets/neural-simulator/results/metrics.json`](assets/neural-simulator/results/metrics.json), [`assets/neural-simulator/captions.md`](assets/neural-simulator/captions.md), `assets/neural-simulator/*.png` | every number in the README with its record; what the frames show; not step inputs     |
| [`assets/neural-simulator/PROVENANCE.md`](assets/neural-simulator/PROVENANCE.md)                                                                                                                              | which file came from where                                                            |

## Steps

### 1. Framework at the pinned revision, in a second working tree (about 15 minutes)

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git     # main: holds this skill folder
cd cosmos-framework
git worktree add ../cosmos-framework-c23e51f c23e51f2f157ae3e51cfcd86ebfb5464850894f2
cd ../cosmos-framework-c23e51f                               # every command below runs from here
uv sync --all-extras --group=cu130-train                     # torch 2.10.0+cu130, python 3.13
cat > recipe.env <<'EOF'
export REPO=$PWD WORKDIR=$PWD
export SKILL=$(cd $REPO/../cosmos-framework/.agents/skills/cosmos3-neural-simulator-libero/assets/neural-simulator && pwd)
export LIBERO_ROOT=$PWD/examples/data/LIBERO_LeRobot_v3/libero_10
export LIBERO_HOLDOUT_PATH=$PWD/data/holdout_19.txt
export BASE_CHECKPOINT_PATH=$PWD/examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
export RUN=$WORKDIR/runs/main/cosmos3_action_fd/action_sft/action_fd_libero_posttrain
export CELL2=$WORKDIR/runs/cell2/cosmos3_action_fd/action_sft/action_fd_id_libero_posttrain
export EVAL=$WORKDIR/eval
export PYTHONPATH=.
export PATH=$PWD/.venv/bin:$PATH   # the launch scripts call bare `torchrun`
EOF
.venv/bin/python -c "import torch, natten, lerobot, torchcodec; print(torch.__version__)"
```

Expected: `2.10.0+cu130`. Inside an NGC container, and in the shell that launches Python, also `export LD_LIBRARY_PATH=''`.
**If the torchcodec import fails** on CUDA 13 wheels, the patch in step 2 installs a `sitecustomize.py` that preloads
the NPP/NVRTC libraries; re-run this check after step 2.

### 2. Install the recipe files into the working tree (one minute)

```bash
source recipe.env
git apply --check $SKILL/patches/modified_files_vs_c23e51f.diff && git apply $SKILL/patches/modified_files_vs_c23e51f.diff
cp $SKILL/configs/action_fd_libero_posttrain.py $SKILL/configs/action_fd_id_libero_posttrain.py \
   cosmos_framework/configs/base/experiment/action/posttrain_config/
cp $SKILL/toml/action_fd_libero_posttrain.toml $SKILL/toml/action_fd_id_libero_posttrain.toml examples/toml/sft_config/
cp $SKILL/launch/launch_sft_action_fd_libero_posttrain.sh $SKILL/launch/launch_sft_action_fd_id_libero_posttrain.sh examples/
cp $SKILL/configs/docs/action_fd_libero_posttrain.md $SKILL/configs/docs/action_fd_id_libero_posttrain.md docs/
mkdir -p scripts data
cp $SKILL/scripts/*.py $SKILL/scripts/*.sh scripts/
cp $SKILL/holdout_19.txt data/holdout_19.txt
.venv/bin/python -m cosmos_framework.scripts.train --sft-toml examples/toml/sft_config/action_fd_libero_posttrain.toml --dryrun
```

Expected: `git apply --check` prints nothing, and the dry run ends without a traceback (the experiment resolves). The
table in README "Install the recipe files" lists what each copy is.

### 3. Data, held-out split and leakage check (CPU, about 5 minutes)

```bash
source recipe.env
.venv/bin/hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include 'libero_10/**' \
  --revision e5907374380b8f96511957e6ba5582be52a1e179 --local-dir examples/data/LIBERO_LeRobot_v3     # 603 MB
.venv/bin/python scripts/make_holdout.py --root $LIBERO_ROOT --out data/holdout_19.generated.txt
diff <(grep -v '^#' data/holdout_19.generated.txt) <(grep -v '^#' data/holdout_19.txt) && echo "holdout matches the pinned list"
.venv/bin/python scripts/leakage_check.py --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH \
  --stats $(find cosmos_framework -name 'libero_native_frame_wise_relative_rot6d.json' | head -1) --out data/leakage_check.json
cat data/leakage_check.json
```

Expected: `holdout matches the pinned list` (19 ids, 2 per task, 1 for task 8, seed 0), and `leakage_check.json` with 0
overlapping episode ids and 0 duplicate action sequences; held-out first frames no closer to training frames (nearest
neighbour RMSE min about 0.043, median about 0.059) than training frames are to each other (0.032, 0.059). The recipe
reads `data/holdout_19.txt`, the pinned list, not the generated file.

### 4. Base checkpoint and tokenizer to the training format (CPU, about 10 minutes, 36 GB)

```bash
source recipe.env
.venv/bin/hf download nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --local-dir hf/Cosmos3-Nano   # 33 GB
.venv/bin/hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir examples/checkpoints/wan22_vae
CUDA_VISIBLE_DEVICES= .venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Nano --checkpoint-path $PWD/hf/Cosmos3-Nano
ls examples/checkpoints/Cosmos3-Nano
```

Expected: a DCP directory (20 to 70 s to convert). Pass the local directory to the converter, never the catalog name.

### 5. Smoke test: 10 iterations (about 20 minutes)

```bash
source recipe.env
OUTPUT_ROOT=$WORKDIR/runs/smoke IMAGINAIRE_OUTPUT_ROOT=$WORKDIR/runs/smoke \
EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 trainer.logging_iter=1 job.name=action_fd_libero_smoke10" \
NPROC_PER_NODE=1 bash examples/launch_sft_action_fd_libero_posttrain.sh
find $WORKDIR/runs/smoke -maxdepth 6 -name 'iter_000000010'
```

Expected: the training log says `kept 360/379`, about 48 to 96 s per iteration (both measured on a GB300), 218 to 241 GB allocated, finite loss, and
`iter_000000010` found. The distinct `job.name` keeps the main run from resuming from the smoke checkpoint.

### 6. Train cell 1 (FD), 1000 iterations (about 27 GB300-hours)

```bash
source recipe.env
OUTPUT_ROOT=$WORKDIR/runs/main IMAGINAIRE_OUTPUT_ROOT=$WORKDIR/runs/main \
EXTRA_TAIL_OVERRIDES="trainer.callbacks.wandb.log_train_loss_to_console=True" \
NPROC_PER_NODE=1 nohup bash examples/launch_sft_action_fd_libero_posttrain.sh > $WORKDIR/runs/main_train.log 2>&1 &
```

Expected: FD loss about 1.17 at iteration 10, about 0.6 by iteration 250 and flat after; checkpoints every 250
iterations in `$RUN/checkpoints/`. **If `nvidia-smi` reaches 255 GB after a save**, the measured run was stopped and
relaunched with the same output root (it resumes from `latest_checkpoint.txt`); iterations recomputed that way are the
difference between 26.94 productive and 29.24 consumed GPU-hours.

### 7. Optional: train cell 2 (FD+ID, mixed frame rates), 1000 iterations

```bash
source recipe.env
OUTPUT_ROOT=$WORKDIR/runs/cell2 IMAGINAIRE_OUTPUT_ROOT=$WORKDIR/runs/cell2 MASTER_PORT=50013 \
EXTRA_TAIL_OVERRIDES="trainer.callbacks.wandb.log_train_loss_to_console=True dataloader_train.dataloader.datasets.libero.dataset.mix_seed=42" \
NPROC_PER_NODE=1 nohup bash examples/launch_sft_action_fd_id_libero_posttrain.sh > $WORKDIR/runs/cell2_train.log 2>&1 &
```

Expected: a vision loss (0.085 → 0.037) and an action loss (0.43 → 0.002) logged separately; neither is on cell 1's
scale. When relaunching cell 2 after a stop, pass a different `mix_seed` (the measured run used 42000 + the resumed
iteration) so the per-visit draws are not replayed.

### 8. Export and evaluate forward-dynamics fidelity (about 1 hour for three seeds plus the control)

```bash
source recipe.env
CUDA_VISIBLE_DEVICES= .venv/bin/python -m cosmos_framework.scripts.export_model --checkpoint-path $RUN/checkpoints/iter_000001000 \
  --config-file $RUN/config.yaml -o $WORKDIR/export_iter1000
.venv/bin/python scripts/gripper_classifier.py train --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --out $EVAL/gripper --label-source cmd --cpu
for SEED in 0 1 2; do
  OUT=$EVAL/iter1000; [ $SEED -gt 0 ] && OUT=${OUT}_seed$SEED
  .venv/bin/python scripts/fd_rollout_eval.py --checkpoint-path $RUN/checkpoints/iter_000001000 --config-file $RUN/config.yaml \
    --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed $SEED --out $OUT
  .venv/bin/python scripts/gripper_classifier.py apply --model $EVAL/gripper/gripper_net.pt --eval-dir $OUT
  .venv/bin/python scripts/make_report.py --eval-dir $OUT --title "LIBERO FD eval, iteration 1000, seed $SEED" --n-videos 10
done
.venv/bin/python scripts/fd_rollout_eval.py --checkpoint-path $BASE_CHECKPOINT_PATH --config-file $RUN/config.yaml --no-ema \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --out $EVAL/control_base
.venv/bin/python scripts/compare_cells.py --out $EVAL/cells_final_table.md \
  "control_base=$EVAL/control_base" "cell1_iter1000=$EVAL/iter1000,$EVAL/iter1000_seed1,$EVAL/iter1000_seed2"
cat $EVAL/cells_final_table.md
```

Expected: the export is about 30 GB (7 shards, bf16 EMA weights; CPU, 25 to 79 s); the classifier reports about 92.5 to
93.6 % accuracy on held-out real frames; each `fd_rollout_eval.py` run takes 9 to 17 minutes at about 36 GB; the table
shows cell 1 at PSNR about 20.6 dB, SSIM about 0.78, LPIPS about 0.13 against the control's 12.83 to 12.88 dB / 0.40 /
0.534 (the control on two GB300 machines), all at 160 × 320
content over the full episode. The first chunk is about 31 dB and chunk 10 stays near 19 dB where the control is at
12 dB. With cell 2 trained, add its runs to `compare_cells.py` with `--delta cell2_iter1000 cell1_iter1000` as the
README's "Export and evaluate" block shows (measured: −0.38 dB PSNR for FD+ID).

### 9. Optional: inverse-dynamics evaluation of cell 2 (11 to 12 minutes per run)

```bash
source recipe.env
for SEED in 0 1 2; do
  OUT=$EVAL/id_cell2_iter1000; [ $SEED -gt 0 ] && OUT=${OUT}_seed$SEED
  .venv/bin/python scripts/id_eval.py --checkpoint-path $CELL2/checkpoints/iter_000001000 --config-file $CELL2/config.yaml \
    --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed $SEED --out $OUT \
    --title "cell 2 (FD+ID, mixed fps) iteration 1000, seed $SEED"
done
.venv/bin/python scripts/id_eval.py --checkpoint-path $RUN/checkpoints/iter_000001000 --config-file $RUN/config.yaml \
  --root $LIBERO_ROOT --holdout $LIBERO_HOLDOUT_PATH --num-steps 30 --guidance 1.0 --shift 3.0 --seed 0 --out $EVAL/id_cell1_iter1000 \
  --title "cell 1 (FD only) iteration 1000, seed 0"
grep -h -A3 'position' $EVAL/id_cell2_iter1000*/RECORD.md | head
```

Expected: cell 2 position error below 1 mm per frame (measured 0.71 ± 0.00), gripper sign accuracy about 0.997; the
FD-only cell about 17 mm, as bad as the base model; the constant predictors about 6 mm (computed inside every run). The
base-model and shuffled-video controls are in README "Export and evaluate".

## Verify the output

| Step | Correct when                                                                                              |
| ---- | --------------------------------------------------------------------------------------------------------- |
| 2    | `git apply --check` silent; `--dryrun` ends without a traceback                                           |
| 3    | generated holdout equals `holdout_19.txt`; leakage check: 0 overlapping ids, 0 duplicate action sequences |
| 5    | `kept 360/379` in the log; `iter_000000010` saved with a finite loss                                      |
| 6    | FD loss about 0.6 from iteration 250; four checkpoints                                                    |
| 8    | PSNR about 20.6 ± 0.1 dB over three seeds against 12.83 for the control; chunk 0 about 31 dB              |
| 9    | cell 2 ID position error below 1 mm per frame; FD-only cell about 17 mm                                   |

Every fidelity number is a regular-weight number of the training checkpoint at 160 × 320 content; say so when you
report it. One training seed per cell: the FD versus FD+ID difference is judged against sampling-seed noise only.

## Reference files

- [README.md](README.md): the full write-up, both cells' results, the drift curve, the inverse-dynamics table with
  five controls, the cost card, known limitations, and the complete evaluation block including the remaining controls
- [assets/neural-simulator/PROVENANCE.md](assets/neural-simulator/PROVENANCE.md): where every shipped file came from
- [assets/neural-simulator/results/metrics.json](assets/neural-simulator/results/metrics.json): every number with its record
- [assets/neural-simulator/captions.md](assets/neural-simulator/captions.md): the bundled frames and per-chunk numbers

## Things not obvious from the docs

- **The LIBERO experiments are not on `c23e51f`.** They travel with this skill (step 2). The patch touches five
  framework files: the experiment registry, three dataset files (holdout path, shuffle within shard, mixed modes and
  frame rates) and `sitecustomize.py` for the CUDA 13 NPP/NVRTC preload.
- **Which weights are scored.** `fd_rollout_eval.py` and `id_eval.py` default to EMA weights, but with a YAML config the
  framework loader loads the regular weights; the README numbers are regular-weight numbers. The 30 GB export holds the
  EMA weights and was not evaluated.
- **Resolution.** The 256 × 512 side-by-side input is snapped to the model's 320 × 192 canvas; every metric is computed
  on the 160 × 320 content area, not the source.
- **Cell 2 sees about half as many 20 fps FD windows** as cell 1 at equal token budget (half of every step is ID, two
  thirds of the video windows are 5 or 10 fps); the 0.38 dB is a cost at equal budget, not at equal FD exposure.
- **Normalisation statistics are the framework's bundled file**, computed on a four-suite LIBERO pool, not train-split
  only; leakage through them is negligible (the 19 held-out episodes shift the quantiles by at most 0.0043 of range).

## Related skills

| Skill                                         | When to use                                                            |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| `../cosmos3-nano-robot-manipulation/SKILL.md` | the same data in the opposite direction: a policy that outputs actions |
| `../cosmos3-sim-bimanual-agibot/SKILL.md`     | the released two-arm simulator checkpoint on AgiBot data               |
| `../cosmos3-setup/SKILL.md`                   | install groups, CUDA variants, container setup                         |
| `../cosmos3-env-troubleshoot/SKILL.md`        | environment and runtime errors during training                         |
