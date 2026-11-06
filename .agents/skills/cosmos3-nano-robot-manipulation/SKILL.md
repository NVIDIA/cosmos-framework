---
name: cosmos3-nano-robot-manipulation
description: >
  Post-train Cosmos3-Nano into a robot manipulation policy on LIBERO-10 and evaluate it closed-loop in the LIBERO simulator.
  Use when the user asks for Cosmos Policy, action SFT of Cosmos3-Nano on robot demonstrations, closed-loop LIBERO evaluation,
  the cost of post-training a policy on one GPU, or reproducing the 95.1 % LIBERO-10 result.
metadata:
  industry: ROB
  gpu-budget: XL
---

# Cosmos Policy: Cosmos3-Nano post-trained on LIBERO-10 (one GB300)

One job: full SFT of the public `nvidia/Cosmos3-Nano` on the 379 LIBERO-10 demonstrations with the framework's shipped
`action_policy_libero_nano` experiment, then a closed-loop evaluation in the LIBERO simulator (10 tasks × 50 layouts,
three seeds). Measured: the untrained base scores 0.0 %; iteration 2000 scores **95.1 ± 0.1 %** (a second training run
reached 91.1 ± 1.1 % at iteration 1250). About 62 GB300-hours of training. The full write-up with every number, the
cost card and the comparison with the Cosmos 3 technical report is [README.md](README.md).

## For an agent

> Using the `cosmos3-nano-robot-manipulation` skill in this Cosmos Framework checkout: set up the framework at the
> pinned revision, download LIBERO-10 and the base model, run the 10-iteration smoke test, train 2000 iterations,
> export the last checkpoint, evaluate it at three seeds in the LIBERO simulator and report the per-seed success rates.

Rules: run the steps in order; after each step compare the output with the step's "Expected"; stop at the first
failure and report the step, the error and the log path; do not change a pinned revision, the global batch of 2048 or
the evaluation protocol; never delete a checkpoint. Budget: about 70 GB300-hours end to end. Step 4 (10 iterations) is
the cheap check of data, checkpoint and memory; it needs the same GPU as training.

## Read this first

1. **Path convention.** Every command runs from the root of a cosmos-framework checkout at revision
   `c23e51f2f157ae3e51cfcd86ebfb5464850894f2` (v1.2.2), the revision every number was measured on. Step 1 writes the
   environment variables to `recipe.env` in that root; every later block starts with `source recipe.env`, so no block
   depends on an earlier shell. `<skill>` below is this skill's directory in the `main` checkout.
2. **Two Python environments.** Training uses the framework's `.venv` (`uv sync --group=cu130-train`). The simulator
   needs an older torch, `robosuite 1.4.1` and `mujoco 2.3.7` in a second environment, `.libenv`, built as
   [docs/action_policy_libero_posttrain.md §3](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/docs/action_policy_libero_posttrain.md)
   describes. Do not install the simulator into `.venv`.
3. **GPU.** Training needs one 256 GB GPU (248 GB peak) or 4 × 80 GB with `max_samples_per_batch=64` and
   `grad_accum_iter=8`; a single 80 or 96 GB card does not fit (optimizer and EMA state alone are about 200 GB). Serving
   and evaluation fit one 48 GB GPU. Do not evaluate while training on the same GPU.
4. **Disk.** About 1 TB keeping every checkpoint (166 GB each, eight of them), about 250 GB keeping only the last.
5. **Long steps run detached** (`nohup` or tmux), never under `timeout`: a checkpoint write takes 3 to 4 minutes and a
   kill during it corrupts the run.

## When to use this skill

- Post-train Cosmos3-Nano into a manipulation policy on LIBERO-10, or reproduce the 95.1 % result and its cost card
- Measure what a Cosmos policy post-training costs on one GPU before planning a run on your own robot data
- Evaluate a Cosmos3-Nano action checkpoint closed-loop in the LIBERO simulator (steps 6 and 7 alone)
- Not for: a neural simulator or world model of the robot (`../cosmos3-neural-simulator-libero/SKILL.md`); video
  generation (`../cosmos3-generate-media/SKILL.md`); LoRA (underfits here: 8.9 % against 75.8 % for full SFT on a quarter
  of the data, README "Known limitations")

## Hardware (measured unless marked)

| Configuration                  | Training                                                                                              | Evaluation                                          |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 1 × DGX Station GB300 (256 GB) | 102 s per iteration (113 s by iteration 1000 at the power cap), 248 GB peak, about 62 h               | three seeds concurrently in 2.7 h, 34 GB per server |
| 4 × H100 80 GB                 | 46.1 s per iteration at 64 × 8, 65.4 GB per GPU, measured over 50 iterations; about 26 h extrapolated | not measured                                        |
| 1 × RTX PRO 6000 (96 GB)       | does not fit                                                                                          | serves the policy (34 GB)                           |

Smoke variant: step 4 alone (10 iterations, one checkpoint, about 25 minutes including the 9-minute tokenizer compile).

## Bundled assets

| Asset                                                                                                                                              | Used for                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `assets/cosmos-policy/` (frames, clip, [`captions.md`](assets/cosmos-policy/captions.md), [`_manifest.json`](assets/cosmos-policy/_manifest.json)) | what a success and a failure look like (frames of the iteration-2000 policy); not inputs to any step      |
| [`scripts/inspect_libero10.py`](scripts/inspect_libero10.py)                                                                                       | step 2: prints the dataset's episode, frame and task counts from its LeRobot metadata (CPU, no framework) |

The dataset itself is not bundled: `nvidia/LIBERO_LeRobot_v3` is 602 MB, OpenMDW-1.1, downloaded in step 2.

## Steps

### 1. Framework at the pinned revision (about 15 minutes)

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git cosmos-framework-c23e51f
cd cosmos-framework-c23e51f
git checkout c23e51f2f157ae3e51cfcd86ebfb5464850894f2
uv sync --all-extras --group=cu130-train     # CUDA 13.0 needs driver r580 or newer; on r575 use cu128-train
uv pip install nvidia-cuda-nvcc nvidia-cuda-cccl   # the compiled tokenizer needs nvcc and the CCCL headers
cat > recipe.env <<'EOF'
export LIBERO_ROOT=$PWD/data/LIBERO_LeRobot_v3/libero_10
export BASE_CHECKPOINT_PATH=$PWD/examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$PWD/outputs/train
export CUDA_HOME=$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13
export RUN=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_libero/action_sft/action_policy_libero_10_nano
export MUJOCO_GL=egl
EOF
.venv/bin/python -c "import torch, natten, lerobot; print(torch.__version__)"
```

Expected: the last line prints a torch version (2.10.x). Then build the simulator environment `.libenv` as
[docs/action_policy_libero_posttrain.md §3](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/docs/action_policy_libero_posttrain.md)
describes, with `export UV_NO_CONFIG=1` set for its `uv` commands (the checkout's `[tool.uv]` forces `numpy>=2`, which
breaks the `opencv-python 4.6` that LIBERO pins), and check it: `.libenv/bin/python -c "import robosuite, mujoco; print(robosuite.__version__, mujoco.__version__)"`
prints `1.4.1 2.3.7`. Inside an NGC container also `export LD_LIBRARY_PATH=''` before any Python.

### 2. Data, base weights and video tokenizer (about 10 minutes, 38 GB)

```bash
source recipe.env
.venv/bin/hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include 'libero_10/**' \
  --revision e5907374380b8f96511957e6ba5582be52a1e179 --local-dir data/LIBERO_LeRobot_v3                        # 0.6 GB
.venv/bin/hf download nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --local-dir hf/Cosmos3-Nano   # 35 GB
.venv/bin/hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir examples/checkpoints/wan22_vae                                                                       # 2.7 GB
python3 <skill>/scripts/inspect_libero10.py --root $LIBERO_ROOT
```

Expected last line: `episodes 379  frames 101469  tasks 10  fps 20`. Model, dataset and tokenizer are public and
ungated; `.venv/bin/hf auth login` avoids download limits.

### 3. Base checkpoint to the training format (CPU, under 10 minutes, once)

```bash
source recipe.env
.venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp -o examples/checkpoints/Cosmos3-Nano --checkpoint-path $PWD/hf/Cosmos3-Nano
ls examples/checkpoints/Cosmos3-Nano
```

Expected: a DCP directory (`.distcp` shards and `.metadata`). Pass the local directory, not the catalog name
`Cosmos3-Nano`: the catalog name fetches whatever `main` points at that day. The converter still contacts the Hub.

### 4. Smoke test: 10 iterations on one GPU (about 25 minutes)

```bash
source recipe.env
ONE_GPU="model.config.parallelism.data_parallel_shard_degree=1 model.config.parallelism.data_parallel_replicate_degree=1 \
  dataloader_train.max_samples_per_batch=128 trainer.grad_accum_iter=16 job.wandb_mode=disabled \
  trainer.callbacks.compile_tokenizer.enabled=true trainer.callbacks.compile_tokenizer.warmup_resolutions=[256]"
EXTRA_TAIL_OVERRIDES="$ONE_GPU trainer.max_iter=10 checkpoint.save_iter=10" \
NPROC_PER_NODE=1 bash examples/launch_sft_action_policy_libero_10_nano.sh
ls $RUN/checkpoints
```

Expected: the log compiles 30 tokenizer variants (about 9 minutes), then iterations at about 102 s each with a finite
loss, peak memory about 248 GB, and `iter_000000010` listed. **If iterations take about 180 s**, the compile settings
are not in effect: check `model.compile.enabled` and the `compile_tokenizer` callback in the resolved config before the
long run. On 80 GB GPUs use `dataloader_train.max_samples_per_batch=64 trainer.grad_accum_iter=$((32 / NGPU))` so the
global batch stays 2048.

### 5. Train 2000 iterations (about 62 GB300-hours)

```bash
source recipe.env
ONE_GPU="model.config.parallelism.data_parallel_shard_degree=1 model.config.parallelism.data_parallel_replicate_degree=1 \
  dataloader_train.max_samples_per_batch=128 trainer.grad_accum_iter=16 job.wandb_mode=disabled \
  trainer.callbacks.compile_tokenizer.enabled=true trainer.callbacks.compile_tokenizer.warmup_resolutions=[256]"
EXTRA_TAIL_OVERRIDES="$ONE_GPU checkpoint.save_iter=250" \
NPROC_PER_NODE=1 nohup bash examples/launch_sft_action_policy_libero_10_nano.sh > outputs/train_full.log 2>&1 &
```

Expected: `checkpoints/iter_000000250` … `iter_000002000` appear every 250 iterations (each 166 GB, 200 to 260 s to
write; host RAM rises to about 410 GB after the first save). Relaunching with the same output root resumes from
`checkpoints/latest_checkpoint.txt`. Iteration 1250 (about 38 h) already scores 91 to 94 %.

### 6. Export (CPU, about 5 minutes, 30 GB)

```bash
source recipe.env
.venv/bin/python -m cosmos_framework.scripts.export_model --checkpoint-path $RUN/checkpoints/iter_000002000 \
  --config-file $RUN/config.yaml -o $RUN/export_iter2000
```

Expected: a bf16 safetensors directory of about 30 GB. The server in step 7 loads either this export or the training
checkpoint; every training-station number in the README came from the checkpoint.

### 7. Serve and evaluate at three seeds (2.7 to 3.5 hours, concurrently)

```bash
source recipe.env
mkdir -p results
for SEED in 0 1 2; do
nohup .venv/bin/python -m cosmos_framework.scripts.action_policy_server_libero \
  --experiment action_policy_libero_nano \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --checkpoint-path $RUN/checkpoints/iter_000002000 \
  --action-normalization quantile_rot \
  --action-stats-path cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json \
  --raw-action-dim 10 --fps 20 --port $((8000+SEED)) --seed $SEED \
  > results/server_iter2000_seed$SEED.log 2>&1 &
done
for SEED in 0 1 2; do
OUT=results/libero_10_iter2000_seed$SEED; PORT=$((8000+SEED)); mkdir -p $OUT
( for i in $(seq 1 240); do curl -sf -m 5 http://localhost:$PORT/ >/dev/null && break; sleep 15; done
  curl -s -m 10 http://localhost:$PORT/info > $OUT/server_info.json
  PYTHONPATH=$PWD:$PWD/LIBERO .libenv/bin/python cosmos_framework/simulation/libero/closed_loop_eval.py \
    --server_url http://localhost:$PORT --task_suite libero_10 --num_trials_per_task 50 --num_envs 8 \
    --camera agentview,wrist --image_size 256 --action_space frame_wise_relative --rotation_space 6d \
    --action_dim 10 --seed $SEED --wait_timeout 1800 --output_dir $OUT > $OUT/client_stdout.log 2>&1
  pkill -TERM -f "action_policy_server_libero.*--port $PORT " ) &
done
wait
grep -h success_rate results/libero_10_iter2000_seed*/summary.json
```

Expected: each server answers `/info` within about 14 s and uses 34 GB; each client writes `summary.json` with an
overall success rate and one line per task; 176 to 179 episodes per hour per server. A checkpoint from iteration 1500
or later scores 90 % or more over the three seeds, and no task is at 0 %. **If a late checkpoint scores under 80 %**,
check the evaluation setup before the model: the gripper sign (model outputs 0 to 1, the simulator expects −1 to 1),
the 180° image orientation the client corrects, and that the server got the normalisation statistics above. The
server's `--seed` sets the sampler; the client's `--seed` alone does not give distinct evaluation seeds.

## Verify the output

| Step | Correct when                                                                                         |
| ---- | ---------------------------------------------------------------------------------------------------- |
| 2    | `inspect_libero10.py` prints 379 episodes, 101469 frames, 10 tasks                                   |
| 4    | `iter_000000010` saved, finite loss, about 102 s per steady iteration, about 248 GB peak             |
| 5    | eight checkpoints `iter_000000250` … `iter_000002000`                                                |
| 7    | three `summary.json`; mean over seeds ≥ 90 % at iteration ≥ 1500; per-task table with no task at 0 % |

Report the mean and spread over the three seeds and the per-task table. Two training runs of this recipe differed by
2.5 points at iteration 1250; a difference between two single runs smaller than that is not a finding.

## Reference files

- [README.md](README.md): the full write-up, results of both training runs, the cost card, the comparison with the
  technical report, Cosmos3-Edge and bring-your-own-robot notes, known limitations
- [assets/cosmos-policy/captions.md](assets/cosmos-policy/captions.md): what each bundled frame and clip shows
- Framework docs at the pinned revision:
  [docs/action_policy_libero_posttrain.md](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/docs/action_policy_libero_posttrain.md),
  [examples/toml/sft_config/action_policy_libero_10_nano.toml](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/examples/toml/sft_config/action_policy_libero_10_nano.toml)

## Things not obvious from the docs

- **The recipe is a 16-GPU layout.** `ONE_GPU` keeps the global batch at 2048 (128 samples × 16 accumulation), so the
  optimizer trajectory is the shipped one; one GPU is 16× slower per iteration, same result.
- **Compile settings decide the cost card.** Without them an iteration takes about 182 s instead of 102 s.
- **One sampler seed per evaluation.** A single evaluation seed is one noise draw; weak checkpoints vary by up to 9
  points across seeds. Always report three or more seeds with per-seed values.
- **Harnesses differ by one to two points** on the same export (94.2 ± 1.6 % on 8 × H100 against 95.1 ± 0.1 % on the
  training station). Compare checkpoints on one machine, one harness, the same seeds.
- **The base model scores 0 %.** It has no action heads for this robot; iteration 250 (15.6 %) is the first useful number.
- **Normalisation statistics ship with the framework** and were computed on a four-suite LIBERO mixture; the recipe uses
  them unchanged because the evaluation server does too.

## Related skills

| Skill                                         | When to use                                                          |
| --------------------------------------------- | -------------------------------------------------------------------- |
| `../cosmos3-neural-simulator-libero/SKILL.md` | the same data in the opposite direction: actions in, video out       |
| `../cosmos3-setup/SKILL.md`                   | install groups, CUDA variants, container and `LD_LIBRARY_PATH` setup |
| `../cosmos3-env-troubleshoot/SKILL.md`        | environment and runtime errors during training                       |
