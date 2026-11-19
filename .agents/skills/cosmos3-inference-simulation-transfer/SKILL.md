---
name: cosmos3-inference-simulation-transfer
description: >
  Run inference with the released Cosmos3-Nano-Sim-Transfer checkpoint: one spatial control video (edge, blur, depth
  or segmentation) plus a caption in, a 480p RGB video out, on Cosmos Framework (cosmos_framework.scripts.inference,
  eager or with torch.compile and CUDA Graphs), a vLLM-Omni server or an SGLang server. Covers Hugging Face example2
  and the 600-frame example1 long rollout. Use when the user asks "run Sim-Transfer inference", "generate a video from
  a depth, edge, blur or segmentation control with Cosmos3-Nano-Sim-Transfer", "run the Sim-Transfer long rollout",
  "serve Sim-Transfer with vLLM-Omni or SGLang" or "make Sim-Transfer inference faster". Not for training,
  distillation or inference from a checkpoint you trained (use the Sim-Transfer post-training skill), not for
  Cosmos3-Nano or Cosmos3-Super transfer (cosmos3-inference-transfer), not for Sim-Bimanual or the Auto models (start
  at cosmos3-inference-simulation).
---

# Run inference with Cosmos3-Nano-Sim-Transfer

Generate RGB from one spatial control and a caption. The control specifies
motion and geometry; the caption specifies appearance. Use `example2` for the
complete workflow and `example1` only for long-rollout depth or segmentation
inference. Both are supplied through [Hugging Face](references/hf-sample.md).

Generate RGB video with a matched Cosmos3-Nano-Sim-Transfer checkpoint. Use HF example2 for ordinary spatial-control
inference and example1 for long-rollout depth or segmentation inference.

Three variants run the released checkpoint: Cosmos Framework (the commands are in this file), a vLLM-Omni server and
an SGLang server (the converted notebooks under `references/`). The task chooser, the stage diagram and the glossary
of recipe terms are in [`references/overview.md`](references/overview.md).

## When to use this skill

- Use to run the released Cosmos3-Nano-Sim-Transfer checkpoint on Cosmos Framework (including the example1 long
  rollout and optimized inference), a vLLM-Omni server or an SGLang server.
- Not for training, distillation or inference from a checkpoint you trained (see "Adapt the model"), and not for
  Cosmos3-Nano or Cosmos3-Super transfer ([the Generator Transfer skill](../cosmos3-inference-transfer/SKILL.md)).

## Paths

This skill replaces the Sim-Transfer cookbook folder (its guide and its two notebooks) and the inference files of the
Sim-Transfer skill family:

- The cookbook guide is [`references/transfer-guide.md`](references/transfer-guide.md). The two notebooks are
  [`references/vllm-omni.md`](references/vllm-omni.md) and [`references/sglang.md`](references/sglang.md), converted
  1:1: their code cells are the fenced `python` blocks, in notebook order.
- `SIM_TRANSFER_SKILL` is this skill's folder. [`references/setup.md`](references/setup.md) exports it together with
  `FRAMEWORK_DIR`. The request defaults are `assets/sample_args.json` and the sample-preparation helper is
  `scripts/prepare_hf_sample.py`.
- Framework requests and results are in a directory you choose: the preparation step writes
  `$SIM_TRANSFER_OUTPUT/requests/` and the commands below write `$SIM_TRANSFER_OUTPUT/generated/`
  (`SIM_TRANSFER_LONG_OUTPUT` for the example1 rollout).
- `COSMOS_COOKBOOK_ROOT` is the optional override for where the two served references look for this skill. For
  `references/vllm-omni.md` it may point at this skill's folder or at `.agents/skills`. For `references/sglang.md` it
  must point at `.agents/skills`, the folder its launch command mounts into the container. Without the override both
  search upward from the working directory: "Run the notebook from anywhere inside `.agents/skills`, or set
  `COSMOS_COOKBOOK_ROOT` to it."
- The served references download the example2 inputs and write their videos under this skill's outputs folder
  (vllm_omni/transfer and sglang/transfer inside it, ignored by `.gitignore`). `COSMOS3_OUTPUT_DIR` (vLLM-Omni) and
  `TRANSFER_OUTPUT_DIR` (SGLang) override the output folder. The short client of the guide's vLLM-Omni section writes
  its MP4s under the working directory instead (outputs/vllm_omni/transfer).

## Choose a variant

All three variants run revision `main` of the model on the Hugging Face `example2` inputs; the sources document the
example1 long rollout for Cosmos Framework only. `references/environment.md`: "Generation and training require an
NVIDIA GPU from Ampere or later and a compatible driver." The sources give no GPU-memory figure for inference.

| Variant                               | Model                              | Backend                                                        | GPUs, as the sources state them                                                 | Reference                                                                      |
| ------------------------------------- | ---------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [Cosmos Framework](#cosmos-framework) | `nvidia/Cosmos3-Nano-Sim-Transfer` | `cosmos_framework.scripts.inference`, eager or optimized       | "The commands below use one GPU." (guide)                                       | this file; `references/transfer-guide.md`, `references/quick-start.md`         |
| [vLLM-Omni](#vllm-omni)               | `nvidia/Cosmos3-Nano-Sim-Transfer` | `vllm/vllm-omni:cosmos3` Docker server, `POST /v1/videos/sync` | "Linux with one NVIDIA GPU" (notebook); the launch passes `--gpus '"device=0"'` | `references/vllm-omni.md`; `references/transfer-guide.md`, section "vLLM-Omni" |
| [SGLang](#sglang)                     | `nvidia/Cosmos3-Nano-Sim-Transfer` | `lmsysorg/sglang:dev` Docker server, `POST /v1/videos`         | not stated; the launch passes `--runtime nvidia --gpus all`                     | `references/sglang.md`                                                         |

## Prerequisites

Framework checkout and environment: `../cosmos3-setup/SKILL.md`. Sim-Transfer adds the following; the served variants
state their own prerequisites in their sections.

- [Sim-Transfer setup](references/setup.md): defines `FRAMEWORK_DIR` and `SIM_TRANSFER_SKILL`, installs the matched
  environment and exports the `COSMOS_*` variables.
- Hugging Face access (same reference, "Enable Hugging Face downloads"): "Accept the license for each gated model you
  download. The default inference commands also load
  [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1) for its blocklist and face-blur resources and
  [Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) for text filtering."
- [Environment requirements](references/environment.md): system requirements, "Complete the CUDA 13.0 environment"
  (the native-library and CUDA-toolkit setup that the setup reference asks for on each host) and "Diagnostics and
  common failures".

## Cosmos Framework

Reference: this section carries the commands. [`references/transfer-guide.md`](references/transfer-guide.md), section
"Cosmos Framework", is the cookbook guide for the same example2 run.

Complete [Framework setup](references/setup.md) and
[prepare example2](references/hf-sample.md#download-and-prepare-example2).
Run the following commands in Bash with that environment active and a GPU
allocation. The saved prompts need no online upsampling.

```bash
cd "$FRAMEWORK_DIR"
unset PYTHONPATH COSMOS_DISTILLATION_IMPL
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 WANDB_MODE=disabled
```

### 1. Select the matched inference checkpoint

Download the [Sim-Transfer release](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer),
the matching [Nano text tokenizer](https://huggingface.co/nvidia/Cosmos3-Nano/tree/e59a53c25979a090fa8706c9acc0c254a6e89b92/text_tokenizer)
and the [Wan video VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth).

The commands use your Hugging Face login:

```bash
export SIM_TRANSFER_CHECKPOINT="$(hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --revision main --exclude 'assets/*')"
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

> **Skill note (not in the source):** the Stage 3 DCP route that followed here in the source (a checkpoint from your own training run) is the section [Run inference from your trained checkpoint](../cosmos3-nano-sim-transfer/post-training/SKILL.md#run-inference-from-your-trained-checkpoint) of the Sim-Transfer post-training skill. Its block reads variables of the training shell and replaces only the checkpoint argument array above.

### 2. Prepare the example2 controls and prompt

The [HF preparation step](references/hf-sample.md#download-and-prepare-example2)
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
caption's time span. Read the [prompt guide](references/prompt-guide.md) for
appearance edits and the structured-versus-dense caption distinction. To
caption your own RGB clips, follow [caption preparation](references/caption-guide.md).

### 3. Generate the depth example

```bash
# Skill note (not in the source): the defaults file is now in the assets folder of this skill. The source passed --defaults-file "$SIM_TRANSFER_SKILL/inference/assets/sample_args.json".
python -m cosmos_framework.scripts.inference \
  -i "$SIM_TRANSFER_OUTPUT/requests/depth.json" \
  -o "$SIM_TRANSFER_OUTPUT/generated/depth" \
  "${SIM_TRANSFER_CHECKPOINT_ARGS[@]}" \
  --experiment-overrides "${SIM_TRANSFER_RESOURCE_OVERRIDES[@]}" \
  --defaults-file "$SIM_TRANSFER_SKILL/assets/sample_args.json" \
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
[Quick Start](references/quick-start.md#2-generate-four-videos) to run all four
example2 requests.

#### Use optimized inference

Enable `torch.compile` and CUDA Graphs together for causal Stage 3 inference.
Use the checkpoint and HF resource arrays above, including
`model.config.kv_cache_dtype=null` for BF16 cache storage and
`model.config.teacher_forcing_frames_per_chunk=1` for one latent per step.
The command below runs on one GPU, compiles the language decoder blocks,
and uses block CUDA Graphs with the standard autoregressive path:

```bash
# Skill note (not in the source): the defaults file is now in the assets folder of this skill. The source passed --defaults-file "$SIM_TRANSFER_SKILL/inference/assets/sample_args.json".
python -m cosmos_framework.scripts.inference \
  -i "$SIM_TRANSFER_OUTPUT/requests/depth.json" \
  -o "$SIM_TRANSFER_OUTPUT/generated/optimized/depth" \
  "${SIM_TRANSFER_CHECKPOINT_ARGS[@]}" \
  --experiment-overrides "${SIM_TRANSFER_RESOURCE_OVERRIDES[@]}" \
  --defaults-file "$SIM_TRANSFER_SKILL/assets/sample_args.json" \
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

### 4. Sampling and output options

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

### Run example1 long rollout

Use [HF example1](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/main/assets/sample_data/example1)
only for long-rollout inference with depth or segmentation. Its controls contain
600 frames at 30 FPS and 384 × 384. The 480-tier model canvas is 640 × 640.
The preparation step extracts the `prompt` string from its upsampled JSON
envelope and writes two requests. It creates no training-input row.

Use the checkpoint and resource arrays above. Keep a separate destination so
the example2 workflow remains available:

```bash
SIM_TRANSFER_LONG_SNAPSHOT="$(hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --revision main --include 'assets/sample_data/example1/*')"
SIM_TRANSFER_LONG_OUTPUT="${SIM_TRANSFER_LONG_OUTPUT:?Choose a destination for the example1 rollout}"
python "$SIM_TRANSFER_SKILL/scripts/prepare_hf_sample.py" \
  --example example1 \
  --sample-dir "$SIM_TRANSFER_LONG_SNAPSHOT/assets/sample_data/example1" \
  --output "$SIM_TRANSFER_LONG_OUTPUT"
for modality in depth seg; do
  # Skill note (not in the source): the defaults file is now in the assets folder of this skill. The source passed --defaults-file "$SIM_TRANSFER_SKILL/inference/assets/sample_args.json".
  python -m cosmos_framework.scripts.inference \
    -i "$SIM_TRANSFER_LONG_OUTPUT/requests/$modality.json" \
    -o "$SIM_TRANSFER_LONG_OUTPUT/generated/$modality" \
    "${SIM_TRANSFER_CHECKPOINT_ARGS[@]}" \
    --experiment-overrides "${SIM_TRANSFER_RESOURCE_OVERRIDES[@]}" \
    --defaults-file "$SIM_TRANSFER_SKILL/assets/sample_args.json" \
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

## vLLM-Omni

Reference: [`references/vllm-omni.md`](references/vllm-omni.md), the converted notebook. Run its numbered sections in
order, from 1 (Prerequisites) to 6 (Validate and view). "Steps 1 and 2 run on the server host. Steps 3 to 6 run in
this notebook." The cookbook guide has the same recipe in short form, with its own launch command and one Python
client, in [`references/transfer-guide.md`](references/transfer-guide.md#vllm-omni), section "vLLM-Omni": "All four
examples use the published HF `example2` caption and aligned controls: **121 frames, 832 × 480, 30 FPS, seed 2025**."

Prerequisites (notebook, section 1): "**Server host:** Linux with one NVIDIA GPU, Docker with the NVIDIA Container
Toolkit, and a Hugging Face token with read access to `nvidia/Cosmos3-Nano-Sim-Transfer`." "**Notebook client:**
Python 3.10 or newer." "Install FFmpeg on the client host; the package also includes FFprobe." "**The server reads
each control video from its own filesystem.** A Sim-Transfer request names the control by absolute `control_path`; it
does not upload the file."

Server launch (notebook, section 2, verbatim). "Export `HF_TOKEN` with read access to the model, then pull the image:"

```bash
docker pull vllm/vllm-omni:cosmos3
```

"Run this shell command from the Cosmos Framework checkout root to mount the control videos at the same absolute paths
the notebook sends:"

```bash
export COSMOS3_REPO="$(pwd -P)"
```

"Start the container and leave it running while using the notebook:"

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -e HF_TOKEN \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -v "${COSMOS3_REPO}:${COSMOS3_REPO}:ro" \
  -p 127.0.0.1:8000:8000 \
  -w /app/vllm-omni \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Sim-Transfer \
    --revision main \
    --omni --model-class-name Cosmos3NanoSimTransferPipeline \
    --deploy-config vllm_omni/deploy/cosmos3_nano_sim_transfer.yaml \
    --stage-overrides '{"0":{"enforce_eager":false,"diffusion_compile_granularity":"regional","diffusion_compile_dynamic":true,"vae_fast_path":"channels_last","vae_encode_fast_path":"channels_last","model_config":{"guardrails":false,"use_cuda_graphs":true,"overlap_vae_decode":true,"clean_commit_mode":"framewise"}}}' \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

Readiness (notebook, "Verify the server"): "Wait for the server to finish starting, then confirm that it serves the
Sim-Transfer model. The cell polls for up to 30 minutes, so you can run it while the server is still loading." It
polls `GET /v1/models` through the notebook's `wait_for_model` helper. Requests (notebook, section 5): "Each example
sends one `POST /v1/videos/sync` request and receives the finished MP4." "The first request also compiles the
transformer and captures CUDA graphs, so it takes longer than the others."

## SGLang

Reference: [`references/sglang.md`](references/sglang.md), the converted notebook. Run its sections in order: "Shared
setup", "Start the server", "Prepare the input", "Preview controls", "Generate the four examples", "View generated
videos". The cookbook guide has no SGLang section; the notebook is the only source.

Prerequisites (notebook, "Shared setup"): "Install the notebook dependencies, and authenticate to Hugging Face with
read access to `nvidia/Cosmos3-Nano-Sim-Transfer`, which publishes both the checkpoint and the sample; run
`hf auth login` or set `HF_TOKEN`". "Install FFmpeg on the client host; the package also includes FFprobe." The server
"does not receive uploads: a Sim-Transfer request names each control by absolute `control_path`, and the Docker
command below mounts the cookbook root at its host path. Run the notebook from that checkout on the server host, or
use the same absolute checkout path on both hosts."

Server launch (notebook, "Start the server", verbatim). "One server runs all four examples." "Run this in a terminal
and leave it running while using the notebook:"

```bash
docker pull lmsysorg/sglang:dev
export COSMOS_ROOT=/absolute/path/to/cosmos-framework/.agents/skills
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
docker run --init --runtime nvidia --gpus all \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "${COSMOS_ROOT}:${COSMOS_ROOT}" \
  -e HF_TOKEN=$HF_TOKEN \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano-Sim-Transfer \
  --revision main \
  --host 0.0.0.0
```

"This notebook was validated against the image built from SGLang commit
[`ab580b2e`](https://github.com/sgl-project/sglang/commit/ab580b2e060059c0f8387b598a9f648dcec13f93), published as the
fixed tag `lmsysorg/sglang:nightly-dev-20261020-ab580b2e`; use that tag in place of `dev` to reproduce the validated
run."

Readiness (notebook): "The first start downloads the checkpoint into the mounted cache. The HTTP port opens only after
the weights are loaded and the warmup requests finish, so wait for the `Uvicorn running` line." "The next cell waits
until `/v1/models` lists the model." Requests and outputs (notebook, "Generate the four examples"): "Each example is
one asynchronous job that saves an MP4 in `OUTPUT_DIR`." "The latest job response is saved beside each video as
`<control>.job.json`; failed or cancelled jobs and HTTP errors show their error details."

## Check the result

Apply the entry skill's
[validation checklist](../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout), then the
Sim-Transfer checks the sources give:

- Cosmos Framework, [`references/transfer-guide.md`](references/transfer-guide.md#validate-the-rollout), section
  "Validate the rollout": "Check that `sample_outputs.json` reports success and that the video decodes to the
  requested frame count, dimensions, and FPS." "Inspect early, middle, and late frames for control adherence:
  boundaries for edge, layout and motion for blur, depth structure for depth, and region layout for segmentation."
  "Check temporal consistency and framing across the rollout."
- Expected native RGB, [`references/output.md`](references/output.md): `example2` "832 × 480, 121 frames at 30 FPS",
  `example1` long rollout "640 × 640, all 600 frames at 30 FPS"; "a successful generation has `status: success`".
- vLLM-Omni, section 6 of `references/vllm-omni.md`: "Decode each generated video completely and check its frame
  count, FPS, and canvas."
- SGLang, `references/sglang.md`: "Each request uses **121 frames, 832 × 480, 30 FPS, and seed 2025**." Check each
  saved MP4 against these values with the shared checklist.

## Troubleshooting

Start with [`references/troubleshooting.md`](references/troubleshooting.md): the symptom router (runtime, model and
runtime, media and data, inference), the failure table and what to include in a support request.
[`references/environment.md`](references/environment.md) has the environment diagnostics and their failure table. For
CUDA, import, container and server failures that are not specific to this model, use
`../cosmos3-env-troubleshoot/SKILL.md`. Training-state symptoms are in the
[Sim-Transfer troubleshooting skill](../cosmos3-nano-sim-transfer/troubleshooting/SKILL.md#training-state).

## Adapt the model

To train or distill your own Sim-Transfer checkpoint and run inference from it, use the
[Sim-Transfer post-training skill](../cosmos3-nano-sim-transfer/post-training/SKILL.md).

## Reference files

| File                                 | Read it when                                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `references/transfer-guide.md`       | You want the cookbook guide: the Framework example2 recipe, "Validate the rollout", the vLLM-Omni short form |
| `references/vllm-omni.md`            | You run the vLLM-Omni variant: the notebook's launch, client helpers, requests, validation and playback      |
| `references/sglang.md`               | You run the SGLang variant: the notebook's launch, readiness cell, asynchronous jobs and playback            |
| `references/setup.md`                | Before the first Framework command: checkout, environment, `SIM_TRANSFER_SKILL`, Hugging Face access         |
| `references/environment.md`          | You need the system requirements, the CUDA 13.0 native-library setup or the environment failure table        |
| `references/hf-sample.md`            | You download example2 or example1 and prepare the request files                                              |
| `references/quick-start.md`          | You run all four example2 controls end to end                                                                |
| `references/inference-contract.md`   | You change an inference setting: bundled defaults, geometry and time, parameter policy, runtime options      |
| `references/model-contract.md`       | You need the control definitions, geometry, time and history, sampling and prompt meaning, hand-offs         |
| `references/output.md`               | You read the generated files and their expected dimensions                                                   |
| `references/prompt-guide.md`         | You use or edit the supplied prompts, or write a prompt for your own clip                                    |
| `references/caption-guide.md`        | You caption your own RGB clip                                                                                |
| `references/local-caption-server.md` | You start the optional local captioning server                                                               |
| `references/video_captioner.txt`     | You need the captioner prompt template that the caption guide links                                          |
| `references/overview.md`             | You choose a task, or need the stage diagram and the glossary of recipe terms                                |
| `references/troubleshooting.md`      | A run fails or the output looks wrong                                                                        |
| `scripts/prepare_hf_sample.py`       | Called by the preparation commands: writes the request files, and the training-input row for example2        |
| `assets/sample_args.json`            | The Stage 3 request defaults that the commands pass with `--defaults-file`                                   |

The reference list of the source skill, carried over:

- [Framework environment setup](references/setup.md)
- [Hugging Face examples](references/hf-sample.md)
- [Training method and checkpoint stages](../cosmos3-nano-sim-transfer/post-training/references/canonical-recipe.md)
- [Prompt guide](references/prompt-guide.md)
- [Caption preparation and review](references/caption-guide.md)
- [Output files](references/output.md)
- [Complete example workflow](references/quick-start.md)
- [Inference parameters and coupled settings](references/inference-contract.md)

## Related skills

| Skill                                                 | When to use                                                                                          |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `../cosmos3-inference-simulation/SKILL.md`            | Entry skill of the Simulation row: model chooser, shared prerequisites, validation checklist         |
| `../cosmos3-inference-simulation-auto/SKILL.md`       | Transfer-Auto and Sim-Auto inference                                                                 |
| `../cosmos3-inference-simulation-bimanual/SKILL.md`   | Sim-Bimanual inference                                                                               |
| `../cosmos3-nano-sim-transfer/post-training/SKILL.md` | The Sim-Transfer post-training skill: training, distillation, inference from your trained checkpoint |
| `../cosmos3-setup/SKILL.md`                           | Framework checkout and environment                                                                   |
| `../cosmos3-env-troubleshoot/SKILL.md`                | CUDA, import, container and server failures                                                          |
| `../cosmos3-inference/SKILL.md`                       | Framework-native inference reference (arguments of `cosmos_framework.scripts.inference`)             |
| `../cosmos3-inference-transfer/SKILL.md`              | Control-to-video transfer with Cosmos3-Nano or Cosmos3-Super                                         |
