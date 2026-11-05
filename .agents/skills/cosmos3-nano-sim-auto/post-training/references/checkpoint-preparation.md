# Prepare the teacher and tokenizer assets

Follow the [shared setup](../../../cosmos3-nano-transfer-auto/shared/framework-runtime.md).
Use a complete, revision-pinned **Cosmos3-Nano-Transfer-Auto** snapshot or your
compatible adapted Transfer-Auto export. Keep assets outside the code checkout.

Set existing inputs and new output directories:

```bash
: "${TRANSFER_AUTO_HF:?set the complete local Transfer-Auto snapshot}"
: "${AUTO_BUNDLE:?set the validated local training bundle}"
: "${VIDEO_VAE:?set the matching Wan2.2_VAE.pth file}"
: "${SIM_AUTO_STAGE0_CHECKPOINT:?set a new DCP conversion directory}"
: "${IMAGINAIRE_OUTPUT_ROOT:?set a new shared training-output root}"
: "${SIM_AUTO_RUN_GROUP:?set a unique group for this training chain}"
export COSMOS_SIM_AUTO_LOCAL_BUNDLE="$AUTO_BUNDLE"
export RGB_VAE="$VIDEO_VAE"
export LIDAR_VAE="$TRANSFER_AUTO_HF/lidar_vae/diffusion_pytorch_model.safetensors"
export QWEN_TOKENIZER="$AUTO_BUNDLE/resources/tokenizer"
export SIM_AUTO_STAGE0_CHECKPOINT IMAGINAIRE_OUTPUT_ROOT SIM_AUTO_RUN_GROUP
export COSMOS_TRAINING=1 COSMOS_INTERNAL=0
python -m cosmos_framework.scripts.prepare_sim_auto_checkpoint \
  --checkpoint-path "$TRANSFER_AUTO_HF" \
  --output-path "$SIM_AUTO_STAGE0_CHECKPOINT"
```

The converter checks the joint architecture, trained LiDAR heads and sensor
embeddings. It writes a weights-only iteration root containing `model/`.
It creates no optimizer, scheduler, trainer or EMA history. Do not convert the
LiDAR tokenizer into model DCP: its weights load separately.

A fresh TF launch skips only the absent `net_ema.` history and starts a new EMA
with rate 0.1 and shift 0. Clear that exception for full same-stage resume.

All nodes must see the same files. Check complete HF shards, tokenizer resources
and DCP metadata before launch; strict native loading remains the compatibility
check. File existence alone is insufficient.

Do not enable `COSMOS_SMOKE`: it can change architecture. Use explicit reductions
from the [smoke guide](smoke.md).
