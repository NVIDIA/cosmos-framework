# Transfer Auto inference

Cosmos3-Nano-Transfer-Auto generates synchronized camera videos from per-camera
WSM control videos, optionally with numeric LiDAR output. Both text-to-video
and image-to-video workflows are supported; image-to-video requires a
conditioning image for every camera.

The released model is optimized for the reference camera rig. You can try a
custom rig, but expect reduced quality as camera geometry diverges. If reference
viewpoints suit your task, use the provided
[WSM retargeting tool](../.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/skills/scenario-to-rds-hq/references/hyperion-retargeting.md)
before rendering. To keep your own viewpoints, follow the
[custom-rig mapping and post-training path](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-preparation.md#choose-reference-rig-or-custom-rig-generation).
Retargeting controls does not transform recorded RGB; keep conditioning and
training images aligned with the chosen camera geometry.

For your own data, follow the [Auto inference skill](../.agents/skills/cosmos3-nano-transfer-auto/inference/SKILL.md):
use [Camera Rig Explorer](../.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/guide.md) to match cameras,
prepare [training-aligned per-view captions](../.agents/skills/cosmos3-nano-transfer-auto/shared/caption-policy.md#keep-training-and-inference-captions-aligned),
then check synchronization before generation. The
[sampling guide](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#sampling)
explains guidance, shift and negative prompts, with recommended starting values
and a one-change-at-a-time tuning workflow.

Both **480p (832×480) and 720p (1280×720)** are supported. Choose
`aspect_ratio: "16,9"` to match the training geometry. See [resolution and aspect ratio](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#resolution-and-aspect-ratio).
The training window is about **13.3 seconds at 30 FPS**. Longer RGB videos use
chunkwise continuation; see [duration and joint-generation limits](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#generation-length-and-long-rollouts)
before changing a 10-second example into a 20- or 30-second request.

For 10-FPS or other lower-rate recordings, read
[source timing versus model FPS versus playback FPS](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#source-timing-model-fps-and-playback-fps).
The reference camera rate is 30 FPS. `--fps` affects model conditioning and does
not resample controls. Match it to the source sampling rate.

Start with the [shared daytime example](#download-the-shared-example) below.
The [native Framework notebook](../.agents/skills/cookbooks/cosmos3/simulation/auto/run_auto_with_cosmos_framework.ipynb)
uses the same published sample and sampling settings, with timestamped-caption
selection. To generate WSM controls from a scene, use the independently installable
[WSM renderer](../.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/README.md).

## Environment and checkpoint

Follow [framework setup](setup.md) for platform and wheel-index requirements.
Install the full model dependencies from this framework checkout:

```bash
uv sync --locked --python 3.13 --group cu130-train --extra train --extra guardrail
source .venv/bin/activate
export COSMOS_TRAINING=0
export COSMOS_INTERNAL=0
```

The `train` dependency extra is needed by model utilities imported during startup;
the environment variables above keep execution in inference mode. The `guardrail`
extra provides the dependencies of the [guardrails](inference.md#guardrails),
which inference enables by default. To run without guardrails, leave the extra
out and add `--no-guardrails` to the inference command. Choose `cu130-train` or
`cu128-train` according to the driver requirements in [setup](setup.md).
Preserve the CUDA compatibility library paths required by your environment.
Check available GPU memory for your selected view count, resolution and output
length before launching.

Download the released model from
[`nvidia/Cosmos3-Nano-Transfer-Auto`](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto).
Authenticate with read access if required, then download a complete snapshot:

```bash
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --local-dir checkpoints/Cosmos3-Nano-Transfer-Auto
export TRANSFER_AUTO_CHECKPOINT=checkpoints/Cosmos3-Nano-Transfer-Auto
```

Pass that local directory through `--checkpoint-path`. The notebook accepts
the same directory through `COSMOS_AUTO_CHECKPOINT`.

The loader accepts native HF exports and indexed Diffusers exports. It loads
`rig_view_embed.weight` and prefers the snapshot's bundled
`lidar_vae/diffusion_pytorch_model.safetensors` over a training-time LiDAR VAE
location, so a complete bundle does not need training storage credentials.
A `checkpoint_type: "dcp"` in metadata can describe the source checkpoint format;
the actual safetensors layout determines the inference format. Keep metadata
and weights from the same snapshot.

## Download the shared example

Transfer-Auto and Sim-Auto share one
[published daytime sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/ab50792f104af5304bce8b4cb7c27c096b33e63f/assets/sample_data).
It contains seven synchronized camera videos, WSM v4.0.1 controls, per-camera
captions, LiDAR controls and targets, native inference requests, and a training
index. Download the complete folder, preserving its relative paths:

```bash
export AUTO_SAMPLE_REVISION=ab50792f104af5304bce8b4cb7c27c096b33e63f
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --repo-type model --revision "$AUTO_SAMPLE_REVISION" \
  --include 'assets/sample_data/*' \
  --local-dir ./auto-example
export AUTO_SAMPLE_ROOT="$PWD/auto-example/assets/sample_data"
(cd "$AUTO_SAMPLE_ROOT" && sha256sum -c SHA256SUMS)
```

This revision pins the **sample**, not the model checkpoint. The sample download
does not include model weights. No manifest conversion or additional loader is
needed for inference; pass the downloaded JSONL directly to the native CLI.

- `input-transfer-auto.jsonl`: bidirectional joint RGB/LiDAR generation, with
  guidance 3, 35 denoising steps and shift 10.
- `input-sim-auto.jsonl`: causal joint RGB/LiDAR generation, with guidance 1 and
  the student checkpoint's fixed four-step sampler; see [Sim-Auto](inference_sim_auto.md).

Both requests use first-frame RGB conditioning, seven named cameras, 720p,
30 FPS, 297 output frames and seed 0. Source videos and captions cover 300 frames;
the output is 297 frames and 99 LiDAR sweeps. Keep the camera keys and their
corresponding RGB, WSM and caption files together; filenames use physical camera
names, not positional `c0`, `c1`, etc.

## Run the shared example

From the Framework repository root, after the downloads above:

```bash
AUTO_NEGATIVE_PROMPT="$(cat .agents/skills/cosmos3-nano-transfer-auto/inference/assets/negative-prompt.txt)"
torchrun --standalone --nproc-per-node=4 \
  -m cosmos_framework.scripts.inference \
  -i "$AUTO_SAMPLE_ROOT/input-transfer-auto.jsonl" \
  -o outputs/transfer_auto \
  --checkpoint-path "$TRANSFER_AUTO_CHECKPOINT" \
  --resolution 720 --num-steps 35 \
  --per-view-negative-prompt "${AUTO_NEGATIVE_PROMPT:?Run from the Framework root and check the bundled negative-prompt.txt}" \
  --cp-size 4 --cfgp-size 1 \
  --no-use-torch-compile --no-diffusion-cache
```

This matches the native notebook and inference skill: **720p, guidance 3,
shift 10, 35 steps**, with the same bundled per-view negative prompt. Guidance
and shift come from the supplied JSONL; set `"guidance": 3` and `"shift": 10`
in custom requests too. The CLI sets resolution, steps and negative text without
modifying the downloaded request. Camera inputs and selected caption files stay unchanged.

For RGB-only generation, make a copy of the request alongside the original
JSONL and remove its `lidar` object. Use a different output directory for each
variant, because both requests name the sample `daytime_scene`.

## Input contract and outputs

Each camera needs a unique `camera_key`, its WSM `control_path`, and caption data;
image conditioning adds `vision_path`. Relative paths in an input specification
resolve against that specification's directory, not the working directory.
For text-to-video, use `model_mode: "text2video"` without `vision_path` on any
camera and without initial RGB conditioning; keep the WSM controls and captions.

Models with rig embeddings require the supported
[physical camera identities](../.agents/skills/cosmos3-nano-transfer-auto/post-training/references/data-contract.md#camera-identities);
`CAMERA_VIEW_MAPPING` supplies their learned IDs. Subsets and reordered views
retain those IDs. Models without rig embeddings keep the existing camera-key behavior.
For per-camera text models, `negative_prompt` alone is ignored by the per-view
caption path. Use `per_view_negative_prompt` only when a shared negative caption
is intended for every camera.

Local LiDAR controls can be FP32 safetensors containing only
`frames[3,T,H,W]`: range in meters, intensity in [0,1], and validity in [0,1].
The tokenizer applies its own padding and normalization. Prepared `.pt`, range-map
`.tar`, and sparse range-view directory inputs remain supported. With a 30 FPS
camera clock and 10 Hz LiDAR, safetensors controls receive frame IDs 0, 3, 6, ...,
which are saved beside the generated range and intensity arrays.

The shared example writes to `outputs/transfer_auto/daytime_scene/`. Inspect
`sample_outputs.json` for success, seven 1280 × 720 camera videos with 297 frames
at 30 FPS, and `lidar.safetensors` with finite `range_m` and `intensity` arrays
of shape `[99,128,1800]` and frame IDs `0,3,...,294`. Inspect `sample_args.json`
to confirm the resolved settings.

When changing duration, keep caption coverage, controls and initial observations
aligned with the requested horizon.

## Optional VAE compilation

The native CLI accepts `--use-vae-encode-compile` and
`--use-vae-decode-compile`. Both default to false and are independent of the
main model's `--use-torch-compile`. They require the Wan 2.2 4×16×16 RGB tokenizer;
the LiDAR tokenizer is unchanged.

Set `--resolution 480 --aspect-ratio 16,9` for 832×480, or
`--resolution 720 --aspect-ratio 16,9` for 1280×720.
Encoder compilation runs at startup for that canvas; other input sizes use
eager encoding. Decoder compilation occurs on the first use of each steady-state
slice shape. Reuse a process for multiple samples to amortize startup overhead.
Encoder compilation cannot be combined with
`--use-separate-pipeline-vision-decode-gpu`, which moves weights to another GPU.
