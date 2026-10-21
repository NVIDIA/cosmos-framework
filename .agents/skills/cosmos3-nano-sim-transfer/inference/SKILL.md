---
name: cosmos3-nano-sim-transfer-inference
description: Generate RGB video with a matched Cosmos3-Nano-Sim-Transfer checkpoint. Use HF example2 for ordinary spatial-control inference and example1 for long-rollout depth or segmentation inference.
---

# Run inference with Cosmos3-Nano-Sim-Transfer

Generate RGB from one spatial control and a caption. The control specifies
motion and geometry; the caption specifies appearance. Use `example2` for the
complete workflow and `example1` only for long-rollout depth or segmentation
inference. Both are supplied through [Hugging Face](../shared/hf-sample.md).

Complete [Framework setup](../shared/setup.md) and
[prepare example2](../shared/hf-sample.md#download-and-prepare-example2).
Run the following commands in Bash with that environment active and a GPU
allocation. The saved prompts need no online upsampling.

```bash
cd "$FRAMEWORK_DIR"
unset PYTHONPATH COSMOS_DISTILLATION_IMPL
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 WANDB_MODE=disabled
```

## 1. Select the matched inference checkpoint

Download the [Sim-Transfer release](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer),
the matching [Nano text tokenizer](https://huggingface.co/nvidia/Cosmos3-Nano/tree/e59a53c25979a090fa8706c9acc0c254a6e89b92/text_tokenizer)
and the [Wan video VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth).

The commands use immutable revisions and your Hugging Face login:

```bash
export SIM_TRANSFER_CHECKPOINT="$(hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --revision 03efb0d86d004fdfc64ffc867dffd5de5df39826 --exclude 'assets/*')"
export SIM_TRANSFER_TOKENIZER_SNAPSHOT="$(hf download nvidia/Cosmos3-Nano \
  --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --include 'text_tokenizer/*')"
export SIM_TRANSFER_TEXT_TOKENIZER="$SIM_TRANSFER_TOKENIZER_SNAPSHOT/text_tokenizer"
export SIM_TRANSFER_VAE="$(hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e)"
SIM_TRANSFER_RESOURCE_OVERRIDES=(
  model.config.tokenizer.bucket_name=
  model.config.tokenizer.object_store_credential_path_pretrained=
  "model.config.tokenizer.vae_path=$SIM_TRANSFER_VAE"
  model.config.vlm_config.tokenizer.config_variant=hf
  "model.config.vlm_config.tokenizer.pretrained_model_name=$SIM_TRANSFER_TEXT_TOKENIZER"
  model.config.vlm_config.pretrained_weights.enabled=False
  model.config.teacher_forcing_frames_per_chunk=1
  model.config.kv_cache_dtype=null
)
SIM_TRANSFER_CHECKPOINT_ARGS=(
  --checkpoint-path "$SIM_TRANSFER_CHECKPOINT"
  --config-file "$SIM_TRANSFER_CHECKPOINT/config.json"
  --credential-path ""
)
```

These overrides select the downloaded HF tokenizer and VAE.
`kv_cache_dtype=null` selects BF16 cache storage. Load the release
directory with its own configuration; the export already contains its selected
weights and needs no EMA flag.

For a **Stage 3 DCP produced by the training walkthrough**, replace only the
checkpoint argument array below. Keep the HF resource overrides:

```bash
: "${SIM_TRANSFER_DCP_ROOT:?Select a completed Stage 3 checkpoint from your training run}"
export COSMOS_SIM_TRANSFER_STAGE0_CHECKPOINT="${STAGE0_CKPT_ROOT:?Use the Stage 0 parent from the training walkthrough}"
export COSMOS_SIM_TRANSFER_STAGE2_CHECKPOINT="${STAGE2_CKPT_ROOT:?Use the Stage 2 parent from the training walkthrough}"
export COSMOS_SIM_TRANSFER_LOCAL_BUNDLE="${SIM_TRANSFER_TRAINING_BUNDLE:?Use the prepared example2 training bundle}"
SIM_TRANSFER_CHECKPOINT_ARGS=(
  --checkpoint-path "$SIM_TRANSFER_DCP_ROOT"
  --config-file cosmos_framework/configs/simulation/local.py
  --experiment sim_local_transfer
  --credential-path ""
  --use-ema-weights
)
```

Framework resolves the recipe before loading a native checkpoint, so retain
its parent-checkpoint and prepared-bundle bindings. Inference then skips training
loaders, parent initialization and teacher/score-model construction. The native
route selects EMA from the checkpoint's `model/` child. Earlier stages use
their own [inference profiles](../post-training/references/canonical-recipe.md#check-intermediate-training-models).

## 2. Prepare the example2 controls and prompt

The [HF preparation step](../shared/hf-sample.md#download-and-prepare-example2)
creates four requests using the saved `example2_upsampled_prompt.json`:

| Request      | Native input                                                                  |
| ------------ | ----------------------------------------------------------------------------- |
| Depth        | HF `example2_depth.mp4` as `depth.control_path`                               |
| Segmentation | HF `example2_segmentation.mp4` as `seg.control_path`                          |
| Edge         | HF `example2.mp4` as `vision_path`, with `edge.preset_edge_threshold: medium` |
| Blur         | HF `example2.mp4` as `vision_path`, with `blur.preset_blur_strength: medium`  |

**Edge and blur preprocessing:** Resize source RGB to the 480-tier model
canvas (832 × 480 for `example2`) **before** computing edge or blur controls.
Computing controls at source resolution and then resizing the maps can produce
suboptimal generation quality. The prepared requests use the correct order.

The native pipeline resizes RGB with antialiased bicubic, depth with antialiased
bilinear and segmentation with nearest neighbor. It computes edge and blur from
resized RGB during inference, as training does. Training samples transform
parameters; these inference requests select the medium presets. RGB supplied
for control generation is not a real history prefix.

Each request uses 121 frames at 30 FPS, aspect ratio `16,9`, the 480 tier,
`direct_resize` and model-native output. The canvas is 832 × 480 with the
training-aligned 3% distortion limit. Preserve the full field of view and the
caption's time span. Read the [prompt guide](../shared/prompt-guide.md) for
appearance edits and the structured-versus-dense caption distinction. To
caption your own RGB clips, follow [caption preparation](../post-training/references/caption-guide.md).

## 3. Generate the depth example

```bash
python -m cosmos_framework.scripts.inference \
  -i "$SIM_TRANSFER_OUTPUT/requests/depth.json" \
  -o "$SIM_TRANSFER_OUTPUT/generated/depth" \
  "${SIM_TRANSFER_CHECKPOINT_ARGS[@]}" \
  --experiment-overrides "${SIM_TRANSFER_RESOURCE_OVERRIDES[@]}" \
  --defaults-file "$SIM_TRANSFER_SKILL/inference/assets/sample_args.json" \
  --autoregressive=True \
  --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
  --seed 2025 --num-outputs 1 --max-num-seqs 1 \
  --num-steps 4 --guidance 1 --control-guidance 1 \
  --kv-cache-inference-size 0 --attention-sink-size 0 \
  --no-use-torch-compile --no-use-cuda-graphs --no-condition-stream-encode \
  --no-diffusion-cache --prompt-upsampling=False
```

The generated RGB video has 121 frames at 30 FPS and 832 × 480. The native
frontend saves `vision.mp4`, the effective request and the sample status.
See [outputs](references/output.md) for artifact meanings and the
[Quick Start](../shared/quick-start.md#2-generate-four-videos) to run all four
example2 requests.

### Use optimized inference

Enable `torch.compile` and CUDA Graphs together for causal Stage 3 inference.
Use the checkpoint and HF resource arrays above, including
`model.config.kv_cache_dtype=null` for BF16 cache storage and
`model.config.teacher_forcing_frames_per_chunk=1` for one latent per step.
The command below runs on one GPU, compiles the language decoder blocks,
and uses block CUDA Graphs with the standard autoregressive path:

```bash
python -m cosmos_framework.scripts.inference \
  -i "$SIM_TRANSFER_OUTPUT/requests/depth.json" \
  -o "$SIM_TRANSFER_OUTPUT/generated/optimized/depth" \
  "${SIM_TRANSFER_CHECKPOINT_ARGS[@]}" \
  --experiment-overrides "${SIM_TRANSFER_RESOURCE_OVERRIDES[@]}" \
  --defaults-file "$SIM_TRANSFER_SKILL/inference/assets/sample_args.json" \
  --autoregressive=True \
  --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
  --seed 2025 --num-outputs 1 --max-num-seqs 1 \
  --num-steps 4 --guidance 1 --control-guidance 1 \
  --kv-cache-inference-size 0 --attention-sink-size 0 \
  --use-torch-compile --use-cuda-graphs \
  --cuda-graph-scope block --compiled-region language \
  --ar-post-saturation-mode default \
  --no-condition-stream-encode --no-diffusion-cache --prompt-upsampling=False
```

For edge, blur or segmentation, select the corresponding request JSON and
output directory. For a [prepared example1 long rollout](#run-example1-long-rollout),
use its request under `$SIM_TRANSFER_LONG_OUTPUT/requests/`, a separate output
directory, and `--kv-cache-inference-size 30 --attention-sink-size 3` in place
of the full-history settings. Keep the compilation, graph and resource settings
above in both cases.

The first generation includes compilation and graph capture. To measure warm
inference latency, append `--benchmark --warmup 1 --num-iterations 3` to the
command and choose a fresh output directory. The first warmup saves the video;
the three measured generations omit output persistence. Read their durations
from `all["OmniInference.generate_batch"]` in `benchmark.json`. These timings
include input preparation, generation and VAE decoding. Use the same input,
checkpoint and seed when comparing with the eager command above.

## 4. Sampling and output options

| Setting                                                  | Reference value     | Effect                                                                             |
| -------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------- |
| `--seed`                                                 | `2025`              | Keep fixed for comparisons                                                         |
| `--num-steps`, `--guidance`, `--control-guidance`        | `4`, `1`, `1`       | Use the Stage 3 distilled schedule; no negative-caption branch                     |
| `teacher_forcing_frames_per_chunk`                       | `1`                 | Predict one causal latent at a time                                                |
| `--kv-cache-inference-size`, `--attention-sink-size`     | `0`, `0`            | Full history with no sink; positive values select a finite history window and sink |
| `num_frames`, `num_video_frames_per_chunk`, `max_frames` | `121`, `121`, `121` | Processing window and decoded input cap; FPS alone does not retime input           |
| `transfer_output_size`                                   | `model`             | Save native canvas; `original` restores the source dimensions in the pipeline      |
| `--no-condition-stream-encode`                           | Enabled             | Use the direct-resize control path                                                 |
| `--prompt-upsampling=False`                              | Enabled             | Reuse the saved structured prompt; the formatter appends actual output metadata    |

Keep zero real RGB conditioning, shared temporal positions and the bundled
metadata templates. See the [inference contract](references/inference-contract.md)
for coupled settings. Use a separate output destination for each changed request.

## Run example1 long rollout

Use [HF example1](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/03efb0d86d004fdfc64ffc867dffd5de5df39826/assets/sample_data/example1)
only for long-rollout inference with depth or segmentation. Its controls contain
600 frames at 30 FPS and 384 × 384. The 480-tier model canvas is 640 × 640.
The preparation step extracts the `prompt` string from its upsampled JSON
envelope and writes two requests. It creates no training-input row.

Use the checkpoint and resource arrays above. Keep a separate destination so
the example2 workflow remains available:

```bash
SIM_TRANSFER_LONG_SNAPSHOT="$(hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --revision 03efb0d86d004fdfc64ffc867dffd5de5df39826 --include 'assets/sample_data/example1/*')"
SIM_TRANSFER_LONG_OUTPUT="${SIM_TRANSFER_LONG_OUTPUT:?Choose a destination for the example1 rollout}"
python "$SIM_TRANSFER_SKILL/scripts/prepare_hf_sample.py" \
  --example example1 \
  --sample-dir "$SIM_TRANSFER_LONG_SNAPSHOT/assets/sample_data/example1" \
  --output "$SIM_TRANSFER_LONG_OUTPUT"
for modality in depth seg; do
  python -m cosmos_framework.scripts.inference \
    -i "$SIM_TRANSFER_LONG_OUTPUT/requests/$modality.json" \
    -o "$SIM_TRANSFER_LONG_OUTPUT/generated/$modality" \
    "${SIM_TRANSFER_CHECKPOINT_ARGS[@]}" \
    --experiment-overrides "${SIM_TRANSFER_RESOURCE_OVERRIDES[@]}" \
    --defaults-file "$SIM_TRANSFER_SKILL/inference/assets/sample_args.json" \
    --autoregressive=True \
    --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
    --seed 2025 --num-outputs 1 --max-num-seqs 1 --control-guidance 1 \
    --kv-cache-inference-size 30 --attention-sink-size 3 \
    --no-use-torch-compile --no-use-cuda-graphs --no-condition-stream-encode \
    --no-diffusion-cache --prompt-upsampling=False
done
```

The requests use a 601-frame processing window with `max_frames: 600`. The
causal wrapper repeats the final control frame once for the VAE's `4n+1`
encoder input, then limits generated frames, control panels and streamed chunks
to the original 600-frame length. The output is 600 frames at 30 FPS and
640 × 640, with a 30-latent history window and three sink entries.

## References

- [Framework environment setup](../shared/setup.md)
- [Hugging Face examples](../shared/hf-sample.md)
- [Training method and checkpoint stages](../post-training/references/canonical-recipe.md)
- [Prompt guide](../shared/prompt-guide.md)
- [Caption preparation and review](../post-training/references/caption-guide.md)
- [Output files](references/output.md)
- [Complete example workflow](../shared/quick-start.md)
- [Inference parameters and coupled settings](references/inference-contract.md)
