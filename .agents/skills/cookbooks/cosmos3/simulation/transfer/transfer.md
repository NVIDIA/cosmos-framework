# Cosmos3-Nano-Sim-Transfer

Cosmos3-Nano-Sim-Transfer generates a **480p video from a caption and a control
video**. It supports four control modalities: **edge**, **blur**, **depth**, and
**segmentation** (`seg`). Each sample uses one modality. The distilled causal
model generates latents autoregressively, retaining control and generated-latent
K/V history. Decoded output frames are not fed back into the transformer.

This page is the inference recipe for the published
[`nvidia/Cosmos3-Nano-Sim-Transfer`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)
artifact. For Hugging Face example2 inputs, prompt preparation, optimized inference,
and long rollouts, follow the [Sim-Transfer inference skill](../../../../cosmos3-nano-sim-transfer/inference/SKILL.md).
For the Simulation family, start at [Simulation cookbooks](../README.md). For the other use cases, see
[Cosmos3-Nano-Sim-Bimanual](../bimanual/bimanual.md),
[Cosmos3-Nano-Transfer-Auto](../auto/auto.md), and
[Cosmos3-Nano-Sim-Auto](../auto/run_sim-auto_with_cosmos_framework.ipynb).

## Supported backends

- **Cosmos Framework** for autoregressive inference with a single edge, blur,
  depth, or segmentation control video. The commands below use one GPU.
- **vLLM-Omni** for a persistent video server serving all four control modalities.
  Follow the [vLLM-Omni instructions](#vllm-omni) or the
  [runnable notebook](run_transfer_with_vllm_omni.ipynb).

## Cosmos Framework

### Prepare the environment and Hugging Face inputs

Use the native HF `example2` workflow on one NVIDIA GPU. Complete these steps
in the same Bash session:

1. Follow [Sim-Transfer setup](../../../../cosmos3-nano-sim-transfer/shared/setup.md)
   to activate the Framework environment and define `FRAMEWORK_DIR` and
   `SIM_TRANSFER_SKILL`.
2. [Download and prepare example2](../../../../cosmos3-nano-sim-transfer/shared/hf-sample.md#download-and-prepare-example2)
   to create four requests under your chosen `SIM_TRANSFER_OUTPUT` directory.
3. [Select the matched inference checkpoint](../../../../cosmos3-nano-sim-transfer/inference/SKILL.md#1-select-the-matched-inference-checkpoint).
   Run that section's HF download and array definitions for
   `SIM_TRANSFER_CHECKPOINT_ARGS` and `SIM_TRANSFER_RESOURCE_OVERRIDES`.
   They bind the release to its local HF tokenizer and video VAE and select
   BF16 K/V cache storage with one latent per autoregressive step.

### Prepare the input

The preparation step uses `nvidia/Cosmos3-Nano-Sim-Transfer` revision
`23b7738403e9b183eeecdec7e351324e8c00b613`. It writes these requests using the
saved upsampled caption:

| Request under `$SIM_TRANSFER_OUTPUT/requests/` | Input |
| --- | --- |
| `depth.json` | HF `example2_depth.mp4` as `depth.control_path` |
| `seg.json` | HF `example2_segmentation.mp4` as `seg.control_path` |
| `edge.json` | HF `example2.mp4` as `vision_path`; compute edge controls from RGB |
| `blur.json` | HF `example2.mp4` as `vision_path`; compute blur controls from RGB |

Each request generates 121 frames at 30 FPS on an 832 × 480 canvas. Native
`direct_resize` preserves the full field of view, using antialiased bicubic
for RGB, antialiased bilinear for depth, and nearest neighbor for segmentation.
The request enforces a 3% aspect-distortion limit.

For edge and blur, **resize RGB to the selected 480p-tier canvas first, then
compute the control map**. Computing maps at the source resolution and resizing
them afterward gives suboptimal conditioning. The prepared requests perform
these steps in the correct order during inference.

Each sample uses one control modality. Target generation starts from noise;
RGB used for edge or blur computation is not an initial generated-video prefix.
For your own inputs, follow the [input and prompt workflow](../../../../cosmos3-nano-sim-transfer/inference/SKILL.md#2-prepare-the-example2-controls-and-prompt).

#### Prompt upsampling

The saved HF caption is ready to use. For new captions or appearance edits,
follow the [captioning guide](../../../../cosmos3-nano-sim-transfer/post-training/references/caption-guide.md)
and [prompt guide](../../../../cosmos3-nano-sim-transfer/shared/prompt-guide.md).
Keep `prompt_upsampling: false` when the caption is already prepared.

### Run inference

With the prepared requests and checkpoint/resource arrays from the steps above:

```bash
cd "$FRAMEWORK_DIR"
unset PYTHONPATH COSMOS_DISTILLATION_IMPL
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 WANDB_MODE=disabled
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

For edge, blur, or segmentation, select its request from the table and a separate
output directory. The command uses full K/V history, no attention sinks, and the
checkpoint's fixed four-step sampler. `--no-condition-stream-encode` is required
by the direct-resize path.

For `torch.compile` and CUDA Graphs, follow
[Use optimized inference](../../../../cosmos3-nano-sim-transfer/inference/SKILL.md).
For a 600-frame depth or segmentation rollout, follow the
[HF example1 walkthrough](../../../../cosmos3-nano-sim-transfer/inference/SKILL.md#run-example1-long-rollout).

### Output

The depth example saves its 121-frame, 832 × 480 RGB video at
`$SIM_TRANSFER_OUTPUT/generated/depth/example2_depth/vision.mp4`, alongside the
effective request and sample status. See the [output guide](../../../../cosmos3-nano-sim-transfer/inference/references/output.md)
for artifact meanings. Use a separate output directory for each run you want
to preserve.

### Validate the rollout

1. Check that `sample_outputs.json` reports success and that the video decodes to
   the requested frame count, dimensions, and FPS.
2. Inspect early, middle, and late frames for control adherence: boundaries for
   edge, layout and motion for blur, depth structure for depth, and region layout
   for segmentation.
3. Check temporal consistency and framing across the rollout.

When comparing runs, keep the caption, control sequence, seed, sampling,
streaming settings, and execution environment fixed.

## vLLM-Omni

Generate videos from **depth, edge, blur, or segmentation** controls through a
persistent video server. All four examples use the published HF `example2`
caption and aligned controls: **121 frames, 832 × 480, 30 FPS, seed 2025**.
The [notebook](run_transfer_with_vllm_omni.ipynb) includes shared setup, input
previews, generation, and video playback.

The vLLM-Omni recipe uses `nvidia/Cosmos3-Nano-Sim-Transfer` at revision
`23b7738403e9b183eeecdec7e351324e8c00b613`, which supports all four control
modalities. The commands below keep this model revision and use the
`vllm/vllm-omni:cosmos3` Docker image shared by all Simulation notebooks.

| Control      | Precomputed HF asset        | What it preserves                     |
| ------------ | --------------------------- | ------------------------------------- |
| Depth        | `example2_depth.mp4`        | Scene geometry and depth structure    |
| Edge         | `example2_edge.mp4`         | Object boundaries and outlines        |
| Blur         | `example2_blur.mp4`         | Coarse appearance, layout, and motion |
| Segmentation | `example2_segmentation.mp4` | Semantic region layout                |

Use a Linux server with an NVIDIA GPU, Docker, and the NVIDIA Container Toolkit.
Export `HF_TOKEN` with read access to the model. From the Cosmos Framework checkout
root, pull the image and record the checkout path:

```bash
docker pull vllm/vllm-omni:cosmos3
export COSMOS3_REPO="$(pwd -P)"
```

All four examples use the same server. The checkout is mounted read-only at its
original absolute path so the server can read each request's `control_path`:

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
    --revision 23b7738403e9b183eeecdec7e351324e8c00b613 \
    --omni --model-class-name Cosmos3NanoSimTransferPipeline \
    --deploy-config vllm_omni/deploy/cosmos3_nano_sim_transfer.yaml \
    --no-guardrails \
    --stage-overrides '{"0":{"enforce_eager":false,"diffusion_compile_granularity":"regional","diffusion_compile_dynamic":true,"vae_fast_path":"channels_last","vae_encode_fast_path":"channels_last","model_config":{"use_cuda_graphs":true,"overlap_vae_decode":true,"clean_commit_mode":"framewise"}}}' \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

The container reuses the host Hugging Face cache and publishes port 8000 on host
loopback. For a remote notebook, forward the port with
`ssh -L 8000:127.0.0.1:8000 <server-host>`, or publish with `-p 8000:8000` and set
`VLLM_BASE_URL`. The notebook and server must use the same absolute checkout path.

### Run the four examples

In the client environment, install `huggingface_hub` and `requests`:

```bash
pip install huggingface_hub requests
```

From the Cosmos Framework checkout root on the server host, run the following
Python code. It downloads the pinned HF inputs into the mounted checkout and
saves one MP4 per control modality:

```python
from pathlib import Path
from uuid import uuid4
import json
import requests
from huggingface_hub import snapshot_download

MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Transfer"
ASSET_REVISION = "23b7738403e9b183eeecdec7e351324e8c00b613"
BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = Path("outputs/vllm_omni/transfer")
INPUT_DIR = Path(".agents/skills/cookbooks/cosmos3/simulation/outputs/vllm_omni/transfer/inputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample = Path(snapshot_download(
    MODEL_ID,
    revision=ASSET_REVISION,
    allow_patterns=["assets/sample_data/example2/*"],
    local_dir=INPUT_DIR,
)) / "assets/sample_data/example2"
prompt = json.dumps(json.loads((sample / "example2_upsampled_prompt.json").read_text()))
controls = {
    "depth": sample / "example2_depth.mp4",
    "edge": sample / "example2_edge.mp4",
    "blur": sample / "example2_blur.mp4",
    "seg": sample / "example2_segmentation.mp4",
}

for hint, control in controls.items():
    extra_params = {
        "session_id": uuid4().hex, "reset": True, "close_session": True,
        "resolution": "480", "aspect_ratio": "16,9",
        "kv_cache_inference_size": 30, "attention_sink_size": 3,
        "max_prompt_tokens": 4096,
        hint: {"control_path": str(control.resolve())},
    }
    form = {
        "model": MODEL_ID, "prompt": prompt,
        "size": "832x480", "num_frames": "121",
        "fps": "30", "seed": "2025",
        "extra_params": json.dumps(extra_params),
    }
    response = requests.post(
        f"{BASE_URL}/v1/videos/sync", data=form,
        headers={"Accept": "video/mp4"}, timeout=1800,
    )
    response.raise_for_status()
    output = OUTPUT_DIR / f"{hint}.mp4"
    output.write_bytes(response.content)
    print(output)
```

Each request waits for the completed video and closes its generation session;
the model remains loaded for the next example. The runtime uses a history window
of 30 latent frames with three attention sinks. The checkpoint supplies its
fixed four-step sampler.

### View the videos

The examples write `depth.mp4`, `edge.mp4`, `blur.mp4`, and `seg.mp4` under
`outputs/vllm_omni/transfer/`. Each contains 121 frames at 832 × 480 and
30 FPS. The
[notebook](run_transfer_with_vllm_omni.ipynb) previews the controls and plays all
four generated videos inline.

## Related

- [Simulation cookbooks](../README.md) — shared setup and validation.
- [Cosmos3-Nano-Sim-Bimanual](../bimanual/bimanual.md) — action-conditioned robotic simulation.
- [Cosmos3-Nano-Transfer-Auto](../auto/auto.md) — multi-camera driving generation.
- [Cosmos3-Nano-Sim-Auto](../auto/run_sim-auto_with_cosmos_framework.ipynb) — autoregressive
  multi-camera driving generation.
- [Cosmos3 cookbook environment setup](../../README.md)
