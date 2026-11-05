# Run the complete Sim-Transfer example

Use [example2 from Hugging Face](hf-sample.md) for prompting, data preparation,
Stage 0–3 training and inference. Its 121-frame videos share one camera and
time span. Reuse its saved structured prompt for each control.

## 1. Prepare the inputs

Complete [Framework setup](setup.md), then
[download and prepare example2](hf-sample.md#download-and-prepare-example2).
The preparation step writes one request per control. Depth and segmentation
use the HF control videos; edge and blur are generated from RGB by the native
inference pipeline.

Read the [prompt guide](prompt-guide.md) for the plain and upsampled captions.
The saved structured JSON is ready to use; no caption-service call is required.

## 2. Generate four videos

Complete [checkpoint selection](../inference/SKILL.md#1-select-the-matched-inference-checkpoint)
in the same shell to download the HF model resources and define the argument
arrays below. Run with the Framework environment active inside a GPU allocation:

```bash
cd "$FRAMEWORK_DIR"
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
export WANDB_MODE=disabled OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
for modality in edge blur depth seg; do
  python -m cosmos_framework.scripts.inference \
    -i "$SIM_TRANSFER_OUTPUT/requests/$modality.json" \
    -o "$SIM_TRANSFER_OUTPUT/generated/$modality" \
    "${SIM_TRANSFER_CHECKPOINT_ARGS[@]}" \
    --experiment-overrides "${SIM_TRANSFER_RESOURCE_OVERRIDES[@]}" \
    --defaults-file "$SIM_TRANSFER_SKILL/inference/assets/sample_args.json" \
    --autoregressive=True \
    --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
    --seed 2025 --num-outputs 1 --max-num-seqs 1 --control-guidance 1 \
    --kv-cache-inference-size 0 --attention-sink-size 0 \
    --no-use-torch-compile --no-use-cuda-graphs --no-condition-stream-encode \
    --no-diffusion-cache --prompt-upsampling=False
done
```

Each request generates 121 RGB frames at 30 FPS and 832 × 480. The Stage 3
sampler uses four distilled transitions, guidance 1, one latent per causal
chunk, full history, no attention sink and no real RGB prefix. The
[inference guide](../inference/SKILL.md) explains
history windows and source-size output.

## 3. Prepare the training bundle

Follow [training-data preparation](local-dataset.md#build-the-example2-bundle)
to build the native bundle from the same HF media and a HF tokenizer. The loader
resizes control and target together, derives edge and blur during loading, and
retains the 93% structured / 7% dense caption policy.

## 4. Train Stage 0 through Stage 3

Follow the [training walkthrough](../post-training/references/local-training.md).
It starts at Stage 0 from [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano)
and uses the checkpoints produced by your stages. Stage 0 uses 480/720;
Stages 1–3 use 480. Causal chunk size stays one.

Stage 3 initializes its student from Stage 2 EMA and its teacher and score model
from Stage 0 EMA. The teacher uses a negative prompt and CFG 3. The walkthrough
includes checkpoint saving and fresh-process resume.

## 5. Generate a long rollout

Use [example1 with depth or segmentation](../inference/SKILL.md#run-example1-long-rollout)
for 600-frame inference.
