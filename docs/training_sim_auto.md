# Sim-Auto: TF → TF-dCM → joint SF-DMD

Start from **Cosmos3-Nano-Transfer-Auto**, either the released checkpoint or a
compatible checkpoint you have fine-tuned on your own domain or sensor rig.
For adaptation, follow the [Transfer-Auto training skill](../.agents/skills/cosmos3-nano-transfer-auto/post-training/SKILL.md)
first, then use that adapted checkpoint for **TF → TF-dCM → SF-DMD** below.

We recommend completing adaptation before distillation, rather than applying
ordinary supervised fine-tuning (SFT) to the released distilled Sim-Auto
checkpoint, to preserve its few-step generation behavior. Sim-Auto distillation
does not start directly from Cosmos3-Nano. Use the released Sim-Auto checkpoint
directly for [inference](inference_sim_auto.md) when no adaptation is needed.

All stages use synchronized camera RGB/WSM and paired measured-LiDAR/HD-map
inputs. Prepare your own data with the [Sim-Auto training skill](../.agents/skills/cosmos3-nano-sim-auto/post-training/SKILL.md).
A small example bundle is useful for checking execution, not for establishing
training quality.

## Prepare the environment, inputs and checkpoints

Follow [setup](setup.md) with the CUDA training group matching the GPU runtime,
system FFmpeg, and the training extras. Use the Python experiment entry point
for these joint recipes, not the generic RGB SFT TOML entry point.
The locked environment includes the FlashAttention-4 dependencies described in
[Sim-Auto runtime setup](inference_sim_auto.md). Verify the imports on the worker.
Do not enable `COSMOS_SMOKE`: it can change model architecture. The reductions
below change execution settings explicitly, not the model architecture.

Bind a new shared output directory and an already validated, complete local
bundle. `manifest.json` must identify model `auto`, format version 1, and the
480/720 strata; all referenced payloads and tokenizer resources must exist.
The native loader reads the packaged local assets, tokenizer resources and
supported transform settings; it does not fetch the original source locations.

```bash
export COSMOS_SIM_AUTO_LOCAL_BUNDLE=/absolute/path/to/auto
export HF_HOME=/absolute/path/to/cache/huggingface
export IMAGINAIRE_OUTPUT_ROOT=/absolute/path/to/new/training-output
export SIM_AUTO_RUN_GROUP=sim_auto_smoke_unique_attempt
export TRANSFER_AUTO_HF=/absolute/path/to/transfer-auto-hf
export SIM_AUTO_STAGE0_CHECKPOINT=/absolute/path/to/new/transfer-auto-dcp
export RGB_VAE=/absolute/path/to/Wan2.2_VAE.pth
export LIDAR_VAE="$TRANSFER_AUTO_HF/lidar_vae/diffusion_pytorch_model.safetensors"
export QWEN_TOKENIZER="$COSMOS_SIM_AUTO_LOCAL_BUNDLE/resources/tokenizer"
```

Set `TRANSFER_AUTO_HF` to the complete HF-format checkpoint selected above,
including all model shards and its matched `lidar_vae/config.json` and weights.
For a customer-adapted model, use your exported Transfer-Auto checkpoint and
skip the download below. To use the released model, download it with your
authorized HF login (do not put tokens in commands, logs or recipes):

```bash
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --local-dir "$TRANSFER_AUTO_HF"
```

Convert the selected released or adapted checkpoint for training:

```bash
python -m cosmos_framework.scripts.prepare_sim_auto_checkpoint \
  --checkpoint-path "$TRANSFER_AUTO_HF" \
  --output-path "$SIM_AUTO_STAGE0_CHECKPOINT"
```

Acquire `Wan2.2_VAE.pth` from `Wan-AI/Wan2.2-TI2V-5B`, pin the accepted
revision, and record its hash. LiDAR weights load separately; do not convert
the LiDAR tokenizer to DCP. The model conversion writes a weights-only
iteration root containing `model/`; preserve the original snapshot/tokenizers.
Validate DCP keys/shapes by strict native training load, not just file presence.

## Checkpoint lineage and EMA

| Stage     | Student initialization                    | Frozen teacher / score initialization              |
| --------- | ----------------------------------------- | -------------------------------------------------- |
| 1: TF     | Transfer-Auto regular `net`, weights-only | None                                               |
| 2: TF-dCM | Accepted Stage 1 regular `net`            | Same Stage 1 regular `net`                         |
| 3: SF-DMD | Accepted Stage 2 regular `net`            | Original bidirectional Transfer-Auto regular `net` |

"Original bidirectional Transfer-Auto" means the checkpoint selected at the
start of this workflow. If you adapted Transfer-Auto, retain that adapted
checkpoint for the Stage 3 teacher/score initialization too.

Never substitute EMA for these regular-network handoffs. A fresh HF weights-only
Stage 1 skips only the absent `net_ema.` history, enables EMA with rate 0.1 and
iteration shift 0. Clear that exception for full same-stage resume. A new stage
does not inherit its predecessor's optimizer/iteration state. Same-stage resume
must restore its own model, EMA, optimizer(s), scheduler(s), trainer and loader.

The full profiles preserve 4000 updates/stage, CP32/FSDP64, 1.6M model/loader
tokens, resolution-specific rigs, 60/40 resolution mixture, teacher forcing
3 latent RGB frames/chunk, 0.4-second decomposed replay and strict-past clean
controls. Stage 2 uses 48-step teacher targets and the four SDE sigma endpoints;
Stage 3 uses student frequency 6 and retains fake-score compilation.
Use the short-run settings below to check a four-GPU allocation.

## Four-GPU execution-contract smoke

Use one supported four-GPU node, CP4/FSDP4. Append the following overrides to
the native profiles. Keep the full profiles unchanged. Run each dryrun first.

```bash
set -euo pipefail
export SIM_AUTO_STAGE1_CHECKPOINT="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stage1_seed0/checkpoints/iter_000000002"
export SIM_AUTO_STAGE2_CHECKPOINT_MODEL="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stage2_seed0/checkpoints/iter_000000002/model"
export SIM_AUTO_STAGE3_TEACHER_MODEL="$SIM_AUTO_STAGE0_CHECKPOINT/model"

run_stage() {
  local stage="$1" steps="$2" mode="$3" module experiment
  case "$stage" in
    1) module=stage1_tf; experiment=sim_auto_joint_stage1_tf_all ;;
    2) module=stage2_tfdcm; experiment=sim_auto_joint_stage2_tfdcm_all ;;
    3) module=stage3_sfdmd; experiment=sim_auto_joint_stage3_sfdmd_all ;;
    *) return 2 ;;
  esac
  local -a args=(
    "experiment=$experiment" "job.group=$SIM_AUTO_RUN_GROUP" "job.name=stage${stage}_seed0"
    'job.wandb_mode=disabled' '~trainer.callbacks.wandb'
    'trainer.callbacks.norm_monitor.log_stat_wandb=false'
    'model_parallel.context_parallel_size=4'
    'model.config.parallelism.context_parallel_shard_degree=4'
    'model.config.parallelism.data_parallel_shard_degree=4'
    'model.config.max_num_tokens_after_packing=100000'
    'dataloader_train.max_sequence_length=100000' 'dataloader_train.max_samples_per_batch=1'
    'model.config.compile.enabled=false' 'trainer.logging_iter=1'
    'checkpoint.dcp_async_mode_enabled=false'
    'checkpoint.load_from_object_store.enabled=false' 'checkpoint.save_to_object_store.enabled=false'
    "trainer.max_iter=$steps" "checkpoint.save_iter=$steps" 'checkpoint.keys_to_skip_loading=[]'
  )
  if [[ "$stage" == 3 ]]; then
    args+=('~trainer.callbacks.wandb_10x' 'model.config.fake_score_compile.enabled=true')
  else
    args+=('~trainer.callbacks.wandb_2x' '~trainer.callbacks.wandb_val')
  fi
  if [[ "$stage" == 1 && "$mode" == fresh ]]; then
    args+=('checkpoint.keys_to_skip_loading=[net_ema.]')
  fi
  local resolution stream ratio
  for resolution in 480 720; do
    stream="dataloader_train.dataloaders.cam_lidar_joint_${resolution}p_separate_view_text"
    ratio=0; if [[ "$resolution" == 480 ]]; then ratio=1; fi
    args+=("$stream.ratio=$ratio"
      "$stream.dataloader.dataset.transform_overrides.num_video_frames=24"
      "$stream.dataloader.dataset.transform_overrides.joint_aligned_num_video_frames=21"
      "$stream.dataloader.dataset.transform_overrides.camera_sampler.rigs=[[camera_front_wide_120fov]]"
      "$stream.dataloader.dataset.transform_overrides.camera_sampler.weights=[1.0]")
  done
  local -a entry=(--config cosmos_framework/configs/simulation/config.py
    --experiment-module "cosmos_framework.configs.simulation.experiment.auto.$module")
  python -m cosmos_framework.scripts.train_experiment "${entry[@]}" --dryrun -- "${args[@]}"
  torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.train_experiment \
    "${entry[@]}" -- "${args[@]}"
}
check_endpoint() {
  python - "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path
import torch.distributed.checkpoint as dcp
root, steps, stage = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
parts = ['model', 'trainer']
parts += ['optim_net', 'optim_fake_score', 'scheduler_net', 'scheduler_fake_score'] if stage == 3 else ['optim', 'scheduler']
for part in parts:
    assert (root / part / '.metadata').is_file(), f'Missing completed DCP component: {part}'
for rank in range(4):
    assert (root / 'dataloader' / f'rank_{rank}.pkl').is_file(), f'Missing loader state for rank {rank}'
state = {'iteration': 0}
dcp.load(state, checkpoint_id=root / 'trainer')
assert state['iteration'] == steps, state
assert (root.parent / 'latest_checkpoint.txt').read_text().strip() == root.name
print('Verified endpoint iteration and training-state components:', root)
PY
}
run_stage 1 2 fresh
check_endpoint "$SIM_AUTO_STAGE1_CHECKPOINT" 2 1
# Also verify finite loss and exact regular loads in this job's logs.
run_stage 2 2 fresh
check_endpoint "${SIM_AUTO_STAGE2_CHECKPOINT_MODEL%/model}" 2 2
run_stage 3 6 fresh
check_endpoint "$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stage3_seed0/checkpoints/iter_000000006" 6 3
# Same job identities auto-resume their own latest saves, not the initializer.
run_stage 1 3 resume
run_stage 2 3 resume
run_stage 3 12 resume
check_endpoint "$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stage1_seed0/checkpoints/iter_000000003" 3 1
check_endpoint "$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stage2_seed0/checkpoints/iter_000000003" 3 2
check_endpoint "$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stage3_seed0/checkpoints/iter_000000012" 12 3
```

The smoke keeps canonical paired LiDAR dropout. Preflight both present and
absent-LiDAR layouts, physical camera IDs, frame alignment, caption metadata
and decoded-source/packed-batch correspondence before allocating GPUs.
Accept each handoff only after DCP metadata/readback confirms its endpoint
iteration, model and training state. Stage 3 must show finite student and critic
updates and retain both optimizer/scheduler histories on resume. Its first six
updates comprise one student and five critic updates; the resumed interval
must include the next student update. Save the resolved configurations and
checkpoint paths for resume. Set local data options under
`.dataset.transform_overrides`.

## Export the Stage 3 student

[Sim-Auto inference](inference_sim_auto.md) loads a Hugging Face export of
the Stage 3 student. Its `config.json` must record the rolling-history
profile. The training config does not contain that profile, so add it with
`--experiment-overrides` when you export. Inference refuses a student exported
without it, before the model loads. `export_model` projects the SF-DMD
training config onto the causal student model itself: it drops the teacher,
critic and training-only fields and disables compilation.

Optionally, prepare the student config first to review it before converting
weights. `prepare_student_config` applies the same projection and writes the
student config. It also fills fields missing from the saved config with the
current defaults and never overwrites a saved value, including `null` and
`false`. Finally it writes a JSON report that lists every projection change
and every default it inserted.

Run the export on a Hopper (sm90) or Blackwell (sm100/sm103) GPU with
FlashAttention-4. The causal student's maskless replay requires `COSMOS_DEVICE=cuda`;
the exporter otherwise defaults to CPU.

```bash
STAGE3_RUN="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_interactive/$SIM_AUTO_RUN_GROUP/stage3_seed0"
python -m cosmos_framework.scripts.prepare_student_config \
  --config-file "$STAGE3_RUN/config.yaml" \
  --output-config "$STAGE3_RUN/student/config.yaml" \
  --report "$STAGE3_RUN/student/report.json"
COSMOS_DEVICE=cuda python -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$STAGE3_RUN/checkpoints/$(cat "$STAGE3_RUN/checkpoints/latest_checkpoint.txt")" \
  --config-file "$STAGE3_RUN/student/config.yaml" \
  --experiment-overrides model.config.rolling_kv_cache_chunks=2 \
    model.config.rolling_kv_sink_chunks=1 model.config.rolling_prompt_policy=segmented \
  --student-only-checkpoint-metadata \
  -o "$STAGE3_RUN/model"
```

Without the preparation step, pass `--config-file "$STAGE3_RUN/config.yaml"`
to `export_model` directly. In both cases, the rolling overrides are required.

`export_model` also makes the export self-contained for inference:

- **LiDAR VAE (optional).** With `--bundle-lidar-vae`, the LiDAR tokenizer's
  VAE is copied into `lidar_vae/` (`diffusion_pytorch_model.safetensors` plus
  `config.json`), and the exported `config.json` points
  `lidar_tokenizer.vae_path` at `lidar_vae/diffusion_pytorch_model.safetensors`.
  The loader binds the bundled file, so `LIDAR_VAE` does not need to be
  reachable at inference time. Without the flag, the export keeps the
  training-time location.
- **Per-view caption layout.** Sim-Auto trains with one caption per camera.
  The exported `config.json` records this as `per_view_captions: true` and
  `per_view_caption_task: "av_joint_camera_lidar_transfer"`, which inference
  needs to pack prompts the way the weights read them. `export_model` derives
  both keys from the training dataloaders. The prepared student config keeps
  only the model and has no dataloaders, so add
  `--per-view-captions on --per-view-caption-task av_joint_camera_lidar_transfer`
  to the export command above when you use it. With
  `--config-file "$STAGE3_RUN/config.yaml"` the keys are derived automatically.
