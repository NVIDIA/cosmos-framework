# Sim-Transfer bidirectional teacher (Stage 0)

Stage 0 trains a whole-window control-to-RGB model from compatible Nano base
**regular** weights. Its new EMA output initializes Stage 1; the same Stage-0
EMA also initializes Stage 3's frozen teacher and trainable score model.
Do not substitute a converted base checkpoint for a completed Transfer teacher.

The recipe is `sim_transfer_stage0_bidir` under
`cosmos_framework/configs/simulation/config.py`. It requires
`COSMOS_SIM_TRANSFER_BASE_CHECKPOINT`, an iteration directory containing
`model/`, supplied by the user. No internal base checkpoint is hard-coded.
Base architecture, tokenizer, video/audio VAE and checkpoint key coverage must
match. A public Nano conversion is a separate compatibility check, not implied
by successful configuration composition.

The recipe retains bidirectional attention, Fused Adam with base LR 1e-4,
the linear 0.4 multiplier schedule, ten selected optimizer groups, power EMA,
BF16 compute/FP32 masters, selective activation checkpointing, language
compilation, and FSDP=8/CP=1. The two streams are 480 (fetch 2, weight 4,
maximum 201 frames) and 720 (fetch 1, weight 2, maximum 149 frames).
The token budget is 74,000. Real data must supply aligned RGB/depth/segmentation,
equal edge/blur/depth/seg draws, 10% text dropout, and the reviewed 93% structured /
7% same-window dense caption mixture. Audio/action weights are retained, but
this recipe trains exclusively on video/control examples.

## Local bundle execution

Complete [setup](setup.md) with training dependencies and native codecs.
Set the shared paths below from your Framework checkout. The bundle is the
native local Transfer format (`manifest.json`, `samples.json`, assets and
tokenizer resources). The plain-caption bundle builder selects dense captions
100% of the time; that is not the historical 93/7 recipe.

```bash
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
export COSMOS_SIM_TRANSFER_BASE_CHECKPOINT=/shared/models/nano-base/iter_000006000
export COSMOS_SIM_TRANSFER_LOCAL_BUNDLE=/shared/data/transfer
export IMAGINAIRE_OUTPUT_ROOT=/shared/outputs/my-transfer-stage0
NANO_TOKENIZER=/shared/models/nano/text_tokenizer
VIDEO_VAE=/shared/models/Wan2.2_VAE.pth
AUDIO_VAE=/shared/models/avae_48k_noncausal_25hz_64ch.ckpt
AUDIO_VAE_CONFIG=/shared/models/avae_48k_noncausal_25hz_64ch.json
ARGS=(
  --config=cosmos_framework/configs/simulation/local.py --
  experiment=sim_local_transfer_stage0_bidir
  job.name=my_transfer_stage0 job.wandb_mode=disabled
  model.config.tokenizer.bucket_name=
  model.config.tokenizer.object_store_credential_path_pretrained=
  "model.config.tokenizer.vae_path=$VIDEO_VAE"
  model.config.sound_tokenizer.bucket_name=
  model.config.sound_tokenizer.object_store_credential_path_pretrained=
  "model.config.sound_tokenizer.avae_path=$AUDIO_VAE"
  "model.config.sound_tokenizer.avae_config_path=$AUDIO_VAE_CONFIG"
  model.config.vlm_config.tokenizer.config_variant=hf
  "model.config.vlm_config.tokenizer.pretrained_model_name=$NANO_TOKENIZER"
)
python -m cosmos_framework.scripts.train_experiment --dryrun "${ARGS[@]}"
# Use this form on one eight-GPU node; multi-node torchrun needs a shared rendezvous.
torchrun --standalone --nproc-per-node=8 \
  -m cosmos_framework.scripts.train_experiment "${ARGS[@]}"
```

Use the audio checkpoint's matching JSON; the loader's fallback configuration
does not match the 48-kHz/25-Hz VAE and it does not read embedded configuration.

Use fresh run identities. A fresh stage loads base weights without optimizer,
scheduler or trainer state; rerunning the same identity/output root resumes its
own latest full checkpoint. Check restored iteration, optimizer/scheduler and
local-loader state. For a bounded smoke, append `trainer.max_iter=6
checkpoint.save_iter=6`, then restart with `trainer.max_iter=12`.
Keep other recipe settings and both resolution streams unchanged.
If an initializer contains regular weights only, explicitly skip only
`net_ema.` and retain strict loading for regular weights; fresh power EMA
with iteration shift 0 is seeded on the first update. Do not use that skip to
hide missing regular parameters.

The local path reads the same source bundle at each model canvas, preserving
the stream weights/fetch sizes and frame ceilings. Its exported longest-window
records are a bounded QA population, not the random-window sampling and
production source population of the historical recipe. Do not call successful
local smoke a quality reproduction or claim untested source-resolution coverage.
The non-local registration deliberately has missing loader targets: supply
your actual production sources rather than silently using unrelated SFT data.

## Handoffs and verification

Select a completed Stage-0 checkpoint, then set
`COSMOS_SIM_TRANSFER_STAGE0_CHECKPOINT` for native Stage 1. Stage 1 loads its
EMA into regular weights and starts fresh optimization state. Pass the new
Stage-1 EMA to Stage 2, then the new Stage-2 EMA to Stage 3 while retaining
the same Stage-0 teacher EMA. Never replace these with historical checkpoints
and still describe the run as an end-to-end lineage.

Require finite losses/gradients and actual parameter updates, completed
model/EMA plus training-state saves, and a successful fresh-process resume.
Stage 3 must exercise both student and score-model updates. Short training
checkpoints are execution evidence, not release-quality inference checkpoints.
