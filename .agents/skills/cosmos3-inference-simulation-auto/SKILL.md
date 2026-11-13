---
name: cosmos3-inference-simulation-auto
description: >
  Run Cosmos3 Auto simulation inference: synchronized multi-camera driving video, optionally with LiDAR, from
  per-camera WSM controls and camera-local captions, with Cosmos3-Nano-Transfer-Auto (bidirectional) or
  Cosmos3-Nano-Sim-Auto (four-step causal student) on Cosmos Framework, vLLM-Omni or SGLang. Also checks camera
  mapping, controls and per-view captions, writes av-per-view-caption captions and renders WSM controls from your own
  scene. Use when the user asks "run Transfer-Auto", "run Sim-Auto", "generate multi-camera driving video from WSM
  controls", "generate cameras and LiDAR together", "serve Transfer-Auto or Sim-Auto with vLLM-Omni or SGLang", "check
  my camera rig, controls or captions" or "render WSM for my scene". Not for fine-tuning Transfer-Auto or distilling
  Sim-Auto (use cosmos3-post-train-simulation-auto), Sim-Bimanual or Sim-Transfer inference, or control-to-video
  transfer with Cosmos3-Nano or Cosmos3-Super.
---

# Cosmos3 Simulation Inference: Auto (Transfer-Auto and Sim-Auto)

The Auto block of the Cosmos3 Simulation row: "Run Cosmos3-Nano-Transfer-Auto for bidirectional generation, or Cosmos3-Nano-Sim-Auto for few-step causal generation. Both use camera-local captions and synchronized WSM controls. Joint generation also needs numerical HD-map LiDAR controls." ([Auto README](references/auto-readme.md); its "Choose a notebook" table and "Setup and sample" notes are not repeated here.)

This skill replaces the inference skills `cosmos3-nano-transfer-auto-inference` and `cosmos3-nano-sim-auto-inference` and the Auto cookbook folder. The two skill bodies are the Cosmos Framework sections below, word for word apart from what is marked in place or listed here: three blocks moved to [`references/start-here-and-sampling.md`](references/start-here-and-sampling.md), headings are two levels down, the ones the two bodies shared and "Check the result" carry the model name and the two titles "(Cosmos Framework)", link targets are re-pointed, each old description is the paragraph after its "Reference:" line, and the two "References" lists are the opening words of their rows in [Reference files](#reference-files); the cookbook guide and notebooks are the files under `references/`, migrated 1:1. The row's entry skill is `../cosmos3-inference-simulation/SKILL.md`.

## When to use this skill

- Use to run Cosmos3-Nano-Transfer-Auto or Cosmos3-Nano-Sim-Auto on Cosmos Framework, vLLM-Omni or SGLang, to check a camera rig, controls or per-view captions, or to render WSM controls from your own scene.
- Not for fine-tuning Transfer-Auto or distilling Sim-Auto (see "Adapt the model"), and not for Sim-Bimanual, Sim-Transfer or control-to-video transfer with Cosmos3-Nano or Cosmos3-Super (see "Related skills").

## Paths

- The cookbook's Auto folder is this skill folder: helpers in `scripts/helpers`, the vLLM-Omni request templates in `assets/daytime_scene`, the WSM renderer package in `scripts/wsm`, the Transfer-Auto negative prompt in `assets/negative-prompt.txt`. Where a reference says "this Auto folder" or "its Auto cookbook", read this skill folder; the "cookbook root" that the SGLang references mount is `.agents/skills`, the parent of this folder.
- `COSMOS_COOKBOOK_ROOT` keeps its name. Set it to this skill folder, to `.agents/skills` or to the repository root; left unset, the first code cell of each converted notebook searches upward from the working directory, so start anywhere inside the checkout.
- The converted notebooks expect this folder at `.agents/skills/cosmos3-inference-simulation-auto` inside a complete Cosmos Framework checkout: the native and WSM references take the repository root (three levels up) as `FRAMEWORK_DIR`, and the vLLM-Omni references need the `cosmos_framework` package above this folder ("Run this notebook from a complete Cosmos Framework checkout").
- Outputs. The two Cosmos Framework run blocks below write to the `AUTO_OUTPUT` you set. The native and WSM references write under the repository root (`$PWD` being that root): `$PWD/outputs/transfer_auto`, `$PWD/outputs/sim_auto` and `$PWD/outputs/wsm`, one timestamped folder per run. The served references write under this skill folder (`$PWD` being this folder): `$PWD/outputs/vllm_omni/auto`, `$PWD/outputs/vllm_omni/sim_auto`, `$PWD/outputs/sglang/auto` and `$PWD/outputs/sglang/sim_auto` (ignored by the skill's `.gitignore`; `COSMOS3_OUTPUT_DIR` overrides them). For SGLang keep that folder inside the mounted `.agents/skills`: "the server reads every prepared input from it".

## Choose a variant

Pick the model and the backend, read the reference named in the row, then follow the section. The Auto README's [Choose a notebook](references/auto-readme.md#choose-a-notebook) table lists each example's cameras, frame count and resolution.

| Model                      | Backend          | GPUs, as the sources state them                                                            | Section                                                                                                                             | Reference                                                                                      |
| -------------------------- | ---------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Cosmos3-Nano-Transfer-Auto | Cosmos Framework | "The distributed example requires four suitable GPUs" (four ranks, `--cp-size 4`)          | [Run inference with Cosmos3-Nano-Transfer-Auto (Cosmos Framework)](#run-inference-with-cosmos3-nano-transfer-auto-cosmos-framework) | [`references/transfer-auto-cosmos-framework.md`](references/transfer-auto-cosmos-framework.md) |
| Cosmos3-Nano-Transfer-Auto | vLLM-Omni        | "at least one NVIDIA GPU"; "The command uses one GPU"                                      | [Serve Cosmos3-Nano-Transfer-Auto with vLLM-Omni](#serve-cosmos3-nano-transfer-auto-with-vllm-omni)                                 | [`references/transfer-auto-vllm-omni.md`](references/transfer-auto-vllm-omni.md)               |
| Cosmos3-Nano-Transfer-Auto | SGLang           | "at least one NVIDIA GPU"; the launch passes `--gpus all`                                  | [Serve Cosmos3-Nano-Transfer-Auto with SGLang](#serve-cosmos3-nano-transfer-auto-with-sglang)                                       | [`references/transfer-auto-sglang.md`](references/transfer-auto-sglang.md)                     |
| Controls for both models   | WSM renderer     | "a graphics-capable NVIDIA driver" (NVIDIA EGL); no model inference                        | [Render WSM controls from your own scene](#render-wsm-controls-from-your-own-scene)                                                 | [`references/generate-wsm.md`](references/generate-wsm.md)                                     |
| Cosmos3-Nano-Sim-Auto      | Cosmos Framework | "a four-GPU example is not a guarantee of fit on every device" (four ranks, `--cp-size 4`) | [Run inference with Cosmos3-Nano-Sim-Auto (Cosmos Framework)](#run-inference-with-cosmos3-nano-sim-auto-cosmos-framework)           | [`references/sim-auto-cosmos-framework.md`](references/sim-auto-cosmos-framework.md)           |
| Cosmos3-Nano-Sim-Auto      | vLLM-Omni        | "one NVIDIA GPU with enough memory for tensor parallel size 1"                             | [Serve Cosmos3-Nano-Sim-Auto with vLLM-Omni](#serve-cosmos3-nano-sim-auto-with-vllm-omni)                                           | [`references/sim-auto-vllm-omni.md`](references/sim-auto-vllm-omni.md)                         |
| Cosmos3-Nano-Sim-Auto      | SGLang           | "one NVIDIA GPU with enough memory for the model"                                          | [Serve Cosmos3-Nano-Sim-Auto with SGLang](#serve-cosmos3-nano-sim-auto-with-sglang)                                                 | [`references/sim-auto-sglang.md`](references/sim-auto-sglang.md)                               |

No source states a required GPU model or a GPU-memory figure for any variant: fit is unknown until measured on your hardware.

## Prerequisites

Framework checkout, `uv` and checkpoint download: `../cosmos3-setup/SKILL.md` and the repository [setup guide](../../../docs/setup.md). The row's shared prerequisites (images and tags, the `--defaults-file` note): [Simulation README](../cosmos3-inference-simulation/references/simulation-readme.md#prerequisites). Then, per variant, as the sources state them:

- **Access:** a Hugging Face token with read access to `nvidia/Cosmos3-Nano-Transfer-Auto`, which also publishes the shared daytime sample, and for Sim-Auto also to `nvidia/Cosmos3-Nano-Sim-Auto`: "run `hf auth login` or set `HF_TOKEN`".
- **Cosmos Framework, both models:** [environment and model assets](references/framework-runtime.md): the locked `cu130-train` environment with the `train` and `guardrail` extras, a complete local snapshot in `TRANSFER_AUTO_HF` or `SIM_AUTO_HF`, and `COSMOS_TRAINING=0` for inference. The run blocks use four GPUs. The native references also need FFmpeg/FFprobe on PATH and `huggingface_hub`, `numpy`, `safetensors`, `IPython` and `matplotlib` in the kernel.
- **Sim-Auto on Cosmos Framework:** "Sim-Auto additionally requires FlashAttention-4, included in the documented CUDA training environment." (`references/sim-auto-cosmos-framework.md`, section 1); the checkout's lockfile "already specifies FlashAttention-4" (`references/framework-runtime.md`).
- **vLLM-Omni and SGLang:** a server host with Docker and the NVIDIA Container Toolkit (images `vllm/vllm-omni:cosmos3` and `lmsysorg/sglang:dev`); a client with Python 3.10 or newer, the `pip install` line of the reference's section 1, and FFmpeg with FFprobe. SGLang opens inputs by path, so "the cookbook checkout, including the output folder this notebook writes, must be on the server host or on a filesystem it shares with this notebook, at one absolute path".
- **WSM renderer:** your own RDS-HQ scene, and the renderer installed "following [WSM setup](scripts/wsm/README.md#install) and its NVIDIA EGL instructions. The renderer needs a graphics-capable NVIDIA driver; the notebook kernel needs NumPy, IPython and FFmpeg/FFprobe."

## Transfer-Auto

Cosmos3-Nano-Transfer-Auto, the bidirectional model: the Cosmos Framework path first, then the two served backends and WSM rendering for your own scene.

### Run inference with Cosmos3-Nano-Transfer-Auto (Cosmos Framework)

Reference: the steps below are the body of the Transfer-Auto inference skill that this skill replaces. The native notebook for the same model is [`references/transfer-auto-cosmos-framework.md`](references/transfer-auto-cosmos-framework.md) (converted 1:1; run its sections 1 to 5 in order), the cookbook guide's CLI recipe is [`references/transfer-auto-guide.md`](references/transfer-auto-guide.md#cosmos-framework), and the camera keys with their learned IDs are in [`references/camera-identities.md`](references/camera-identities.md).

Prepare or run Cosmos3-Nano-Transfer-Auto in Cosmos Framework on camera WSM controls and per-view captions, with optional initial RGB and joint LiDAR. Use for camera mapping, WSM/renderer and synchronization checks, supplied-caption review, av-per-view-caption generation or upsampling, targeted-KPI captions, native requests, sampling settings and output inspection. Input-checking and caption-only requests stop after the requested workflow.

Generate one video per selected camera from that camera's WSM control and caption.
Add an HD-map range control for joint RGB/LiDAR generation. Use initial camera
images for image-to-video, or omit them for text-to-video with WSM.

The released model is optimized for the reference camera rig. You can try a
custom rig, but expect reduced quality as camera placement, viewing direction,
FOV and lens projection diverge. Choose between
[retargeting WSM to the reference rig and retaining your own rig](references/input-preparation.md#choose-reference-rig-or-custom-rig-generation);
the latter may require [post-training](../cosmos3-post-train-simulation-auto/SKILL.md#quickstart-fine-tune-the-released-model).

Before the first run, read [Start here](references/start-here-and-sampling.md#start-here-transfer-auto): try the reference scene first, choose and match your cameras, prepare camera-local captions, then generate, inspect and tune.

#### Choose the task (Transfer-Auto)

- **Check inputs or existing captions:** read [input preparation](references/input-preparation.md)
  and complete only the requested checks. Existing-caption checks do not invoke
  authoring or rewrite the original.
- **Generate or upsample captions (`av-per-view-caption`):** read
  [caption guide](references/caption-guide.md), including its caption policy.
  Follow its options, defaults and WSM-inspection requirements; finish after the
  requested caption deliverables.
- **Explain options or defaults:** read the relevant guide and explain them
  without requiring media or starting work.
- **Run generation:** follow Prepare, Run, and Sampling and output checks below.

For model/configuration questions, read the [model contract](references/model-contract.md).
An explanation request does not authorize jobs, uploads or changes to source assets.

#### Prepare (Transfer-Auto)

1. Follow [environment and snapshots](references/framework-runtime.md). Set
   `TRANSFER_AUTO_HF` to a complete local Transfer-Auto snapshot.
2. Download and checksum-verify the
   [shared reference scene](references/reference-scene.md#get-the-scene), or prepare
   your own scene.
3. Run [input preparation](references/input-preparation.md), which invokes
   [per-view captioning](references/caption-guide.md) when requested.
4. For the reference scene, set
   `AUTO_REQUEST="$AUTO_SAMPLE_ROOT/input-transfer-auto.jsonl"`.
   It selects paragraph captions by default; follow
   [Choose a caption style](references/reference-scene.md#choose-a-caption-style)
   to use the recommended timestamped alternative.
   For your own data, prepare a Framework JSON/JSONL request using the
   [input and sampling guide](references/input-and-execution.md).
   Set `"guidance": 3` and `"shift": 10` in the request, as the supplied sample
   and notebook do; these settings are read from JSON in inference mode.
   Set `AUTO_OUTPUT` to a fresh output directory.

The prepared artifact is a request plus camera-local caption sidecars and
synchronized controls/optional initial observations. A camera-to-text dictionary
alone is not executable.

Supply scene-caption bodies only. Framework automatically adds the selected
camera rig and current view's direction/lens/FOV description, plus the WSM-control
task instruction as a system prompt. Do not duplicate either in your caption
files. See [automatic prefixes](references/caption-policy.md#caption-text-and-automatic-prefixes).

#### Run (Transfer-Auto)

From the Framework root, inside your GPU allocation:

```bash
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0
# Skill note (not in the source): the negative prompt is now this skill's assets/negative-prompt.txt; this was .agents/skills/cosmos3-nano-transfer-auto/inference/assets/negative-prompt.txt.
AUTO_NEGATIVE_PROMPT="$(cat .agents/skills/cosmos3-inference-simulation-auto/assets/negative-prompt.txt)"
torchrun --standalone --nproc-per-node=4 \
  -m cosmos_framework.scripts.inference \
  -i "${AUTO_REQUEST:?set a prepared request}" \
  -o "${AUTO_OUTPUT:?set a fresh output directory}" \
  --checkpoint-path "${TRANSFER_AUTO_HF:?set the local snapshot}" \
  --resolution 720 --num-steps 35 \
  --per-view-negative-prompt "${AUTO_NEGATIVE_PROMPT:?Run from the Framework root and check the bundled negative-prompt.txt}" \
  --cp-size 4 --cfgp-size 1 \
  --no-use-torch-compile --no-diffusion-cache
```

The request supplies guidance 3 and shift 10; the command sets 720p, 35 steps
and the same bundled negative text as the native Framework notebook. Only these
CLI options override the request; the downloaded JSONL stays unchanged. Cameras,
captions, controls, initial images, FPS, frame count and seed also come from the request.

Use explicit physical views in the request; do not add a canonical-view-count
override when preserving a custom subset. Model snapshots contain their own
configuration; no training experiment selector is needed for this command.
The distributed example requires four suitable GPUs; memory fit and speed depend
on the request and hardware. Read
[GPU memory and parallelism](references/input-and-execution.md#gpu-memory-and-parallelism)
before changing the allocation. Keep the installed runtime's guardrail policy; any
benchmark-specific disabling must be explicit.

#### Sampling and output checks

The sampling notes of this section (a short first check, 480p and 720p, generation length, 10-FPS inputs, the per-view negative text) are in [Sampling](references/start-here-and-sampling.md#sampling-transfer-auto).

Inspect `sample_args.json`, `sample_outputs.json`, all camera/control videos,
and `lidar.safetensors` for joint output. Check view identity, actual duration,
numeric range/intensity and frame IDs. Save 50/50 WSM/RGB overlays using matching
crops and times. A clean overlay checks geometry, not the correctness of a
physical camera alias.

Use a new output directory when changing inputs or settings.

### Serve Cosmos3-Nano-Transfer-Auto with vLLM-Omni

Reference: [`references/transfer-auto-vllm-omni.md`](references/transfer-auto-vllm-omni.md) (the vLLM-Omni notebook, converted 1:1). Run its sections in order: "1. Prerequisites", "2. Start the server", "3. Set up the client", "Download and verify the sample", "Verify the server", then "4. WSM-only example" or "5. Joint RGB and LiDAR example" ("The examples are independent. To run only the LiDAR example, complete steps 1 to 3, then go to step 5."). The request contract is in [`references/transfer-auto-guide.md`](references/transfer-auto-guide.md#vllm-omni), section "vLLM-Omni".

Server launch, from section 2: "Export `HF_TOKEN` with read access to the model, then pull the image" (`docker pull vllm/vllm-omni:cosmos3`). "Start the container and leave it running while using the notebook":

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -e HF_TOKEN \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  -e VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=1800 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 127.0.0.1:8000:8000 \
  -w /app/vllm-omni \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Transfer-Auto \
    --revision main \
    --omni --model-class-name Cosmos3MultiviewPipeline \
    --diffusion-compile-granularity regional --no-diffusion-compile-dynamic \
    --no-guardrails \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

Readiness, from "Verify the server": "Wait for the server to finish starting, then confirm that it serves the Transfer-Auto model. The cell polls for up to 30 minutes, so you can run it while the server is still loading." The guide: "When startup finishes, `curl http://127.0.0.1:8000/v1/models` must return the served model before you submit generation requests." "Both requests upload their files, so the notebook can run on any host that reaches the server."

### Serve Cosmos3-Nano-Transfer-Auto with SGLang

Reference: [`references/transfer-auto-sglang.md`](references/transfer-auto-sglang.md) (the SGLang notebook, converted 1:1). Run its sections in order: "1. Prerequisites", "2. Start the server", "3. Shared setup" with "Download and verify the sample" and "Verify the server", then "4. WSM-only example" or "5. Joint RGB and LiDAR example" ("The examples are independent: run **Shared setup**, then either one."). The SGLang request contract is in [`references/transfer-auto-guide.md`](references/transfer-auto-guide.md#send-the-multipart-request), sections "Send the multipart request" and "Joint RGB and LiDAR with SGLang".

Server launch, from section 2: "On the server host, set `COSMOS_ROOT` to the absolute cookbook root (`.agents/skills`), which step 3 prints as `Cookbook root`." "Pull the image, then run the server in a terminal and leave it running while using the notebook; both examples use it":

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
  --model-path nvidia/Cosmos3-Nano-Transfer-Auto \
  --revision main \
  --host 0.0.0.0 --port 30000
```

Readiness, from section 2: "The HTTP port opens only after the weights are loaded and the warmup request finishes, so wait for the `Uvicorn running` line." From "Verify the server": "The cell retries connection errors and server errors for up to 30 minutes, so you can run it while the server is still loading." "SGLang opens the inputs from paths the server can read, so the notebook writes every prepared input under its output folder inside the mounted cookbook root and uploads nothing."

### Render WSM controls from your own scene

Reference: [`references/generate-wsm.md`](references/generate-wsm.md) (the WSM notebook, converted 1:1). Run its sections in order: "1. Environment and inputs", "2. Render controls", "3. Check the rendered views", "4. Prepare or review per-view captions", optionally "5. Optional: render same-scene LiDAR controls", then "6. Generate RGB or joint RGB/LiDAR", which hands the exported request to the native reference of either model.

"The published HF daytime sample contains already-rendered WSM videos, not raw RDS-HQ inputs; use it directly in the inference notebooks if you only want to try the models." The renderer is the standalone package in `scripts/wsm` (its own [README](scripts/wsm/README.md)); "Convert your own source with the [RDS-HQ schema](scripts/wsm/docs/rds-hq.md) and [conversion skill](scripts/wsm/skills/scenario-to-rds-hq/SKILL.md)." Section 1 sets `AUTO_RDS_HQ_ROOT`, `AUTO_RDS_HQ_CLIP` and `WSM_PYTHON` "before running the configuration cell"; "This notebook supports **F-theta scenes, supported Auto camera keys, and 30 Hz poses**." Inspect the rendered controls with [WSM control checks](references/wsm-conventions.md).

## Sim-Auto

Cosmos3-Nano-Sim-Auto, the four-step causal student of Transfer-Auto: the Cosmos Framework path first, then the two served backends. For controls from your own scene, see [Render WSM controls from your own scene](#render-wsm-controls-from-your-own-scene).

### Run inference with Cosmos3-Nano-Sim-Auto (Cosmos Framework)

Reference: the steps below are the body of the Sim-Auto inference skill that this skill replaces. The native notebook for the same model is [`references/sim-auto-cosmos-framework.md`](references/sim-auto-cosmos-framework.md) (converted 1:1; run its sections 1 to 5 in order); the model settings to preserve are in [`references/model-contract.md`](references/model-contract.md#sim-auto), section "Sim-Auto".

Prepare inputs or run Cosmos3-Nano-Sim-Auto camera-only or joint camera/LiDAR generation with a released or exported student in Cosmos Framework. Check causal conditioning, per-camera captions, rolling metadata and saved outputs.

Use the four-step causal student to generate synchronized camera views and
optional LiDAR from initial RGB, per-camera captions and WSM/HD-map controls.

The released model is optimized for the reference camera rig. You can try a
custom rig, but camera geometry differences can degrade quality. Follow the
[reference-rig retargeting or custom-rig workflow](references/input-preparation.md#choose-reference-rig-or-custom-rig-generation).
Retargeting WSM does not reproject initial RGB images. For adaptation, fine-tune
Transfer-Auto on your rig's paired data, then distill it into Sim-Auto.

Before the first run, read [Start here](references/start-here-and-sampling.md#start-here-sim-auto): new rig, captions, sampling, timing, size and duration.

#### Choose the task (Sim-Auto)

- For input preparation only, follow the
  [input-preparation guide](references/sim-auto-inputs.md) and
  [caption guidance](references/sim-auto-inputs.md#camera-local-captions). Return the checked media inventory
  and per-view caption sidecars without launching generation.
- To generate videos and optional LiDAR, continue below.

#### Prepare (Sim-Auto)

1. Follow [Framework setup](references/framework-runtime.md).
   Set `SIM_AUTO_HF` to the downloaded student directory, or a complete student
   export from [distillation](../cosmos3-post-train-simulation-auto/SKILL.md#distill-sim-auto).
2. Follow [input preparation](references/sim-auto-inputs.md). Both Auto models share
   the [reference scene](references/reference-scene.md#get-the-scene);
   download the [sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/main/assets/sample_data)
   and verify its checksums.
3. Set `AUTO_REQUEST="$AUTO_SAMPLE_ROOT/input-sim-auto.jsonl"` for the
   reference scene, or prepare your own JSON/JSONL request using
   [these fields](references/sim-auto-inputs.md#request-fields). Set a fresh `AUTO_OUTPUT` directory.
   The sample request selects paragraph captions; use
   [caption selection](references/reference-scene.md#choose-a-caption-style)
   for the recommended timestamped alternative.

Both models use `daytime_scene`: seven cameras, WSM v4.0.1 controls and paired
RGB/LiDAR targets. `AUTO_SAMPLE_ROOT` points to the downloaded
`assets/sample_data` directory; Sim-Auto uses its `input-sim-auto.jsonl` request.

Supply scene-caption bodies only. Framework automatically adds the camera-rig
and current-view direction/lens/FOV description, plus the WSM-control task
instruction as a system prompt. Do not paste these into your caption files;
see [automatic prefixes](references/caption-policy.md#caption-text-and-automatic-prefixes).

Use an exported model snapshot, not a training experiment or raw DCP directory.
The artifact must include per-view-caption metadata, matching tokenizers and
`rolling_kv_cache_chunks`, `rolling_kv_sink_chunks`, `rolling_prompt_policy`.
Do not guess missing history settings from the model name.

#### Run (Sim-Auto)

```bash
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0
torchrun --standalone --nproc-per-node=4 \
  -m cosmos_framework.scripts.inference \
  -i "${AUTO_REQUEST:?set a prepared request}" \
  -o "${AUTO_OUTPUT:?set a fresh output directory}" \
  --checkpoint-path "${SIM_AUTO_HF:?set the local student snapshot}" \
  --cp-size 4 --cfgp-size 1 \
  --no-use-torch-compile --no-diffusion-cache
```

The request selects camera-only versus joint generation by omitting or including
`lidar.control_path`; there is no separate `--joint-generation` flag here.
The fixed student schedule comes from its model configuration. Do not add
Transfer-Auto's 35-step/shift/negative-CFG recipe.

The causal frontend runs eagerly with serial CFG. It disables unsupported
compilation/CUDA-graph/cache options and reports adjustments. Rolling history
comes from checkpoint metadata; older generic KV/sink-size flags are rejected.
Memory fit and speed depend on the request and hardware. Read
[GPU memory and parallelism](references/input-and-execution.md#gpu-memory-and-parallelism)
before changing the allocation; a four-GPU example is not a guarantee of fit on
every device.

#### Check the result (Sim-Auto)

Inspect the sample output directory:

- `vision_view<NN>_<camera_key>.mp4` and corresponding control videos.
- `caption_chunks.json` and `rolling_prompt.json`: text and transitions actually used.
- `sample_args.json` and `sample_outputs.json`: resolved request and completion.
- Joint only: `lidar.safetensors` with `range_m`, `intensity` and `frame_ids`,
  plus decoded `lidar.pt`.

At 297 camera frames and 30 FPS, joint inference generates 99 sweeps at 10 Hz.
Check source IDs and numeric ranges, not only a rendered point cloud. Inspect all
camera views and 50/50 WSM overlays; compare caption transitions over the full
requested horizon.

For a preview, the Framework provides
`python -m cosmos_framework.scripts.multiview_collage "$AUTO_OUTPUT" --tile-width 832`.
Preserve original numeric outputs and videos. Use
[troubleshooting](references/troubleshooting.md#sim-auto) for failures.

### Serve Cosmos3-Nano-Sim-Auto with vLLM-Omni

Reference: [`references/sim-auto-vllm-omni.md`](references/sim-auto-vllm-omni.md) (the vLLM-Omni notebook, converted 1:1). Run its sections in order: "1. Prerequisites", "2. Start the server", "3. Set up the client", "Download and verify the sample", "Verify the server", then "4. RGB example" or "5. Joint RGB and LiDAR example" ("The examples are independent. To run only the joint example, complete steps 1 to 3, then go to step 5.").

Server launch, from section 2: "Export `HF_TOKEN` with read access to the model, then pull the image" (`docker pull vllm/vllm-omni:cosmos3`). "Start the container and leave it running while using the notebook":

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -e HF_TOKEN \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 \
  -e VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 127.0.0.1:8000:8000 \
  -w /app/vllm-omni \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Sim-Auto \
    --revision main \
    --omni --model-class-name Cosmos3NanoSimAutoPipeline \
    --deploy-config vllm_omni/deploy/cosmos3_nano_sim_auto.yaml \
    --tensor-parallel-size 1 \
    --no-guardrails \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

Readiness, from "Verify the server": "Wait for the server to finish starting, then confirm that it serves the Sim-Auto model. The cell polls for up to 30 minutes, so you can run it while the server is still loading. `/v1/models` reports the model name, not its revision; the `--revision` in the server command pins that."

### Serve Cosmos3-Nano-Sim-Auto with SGLang

Reference: [`references/sim-auto-sglang.md`](references/sim-auto-sglang.md) (the SGLang notebook, converted 1:1). Run its sections in order: "1. Prerequisites", "2. Start the server", "3. Set up the client" with "Download and verify the sample" and "Verify the server", "4. Prepare the inputs", "5. Generate", "6. Validate and view". "This notebook runs the RGB example of the shared **daytime sample**"; joint RGB and LiDAR for Sim-Auto is in the vLLM-Omni and Cosmos Framework sections.

Server launch, from section 2: "On the server host, set `COSMOS_ROOT` to the absolute cookbook root (`.agents/skills`), which step 3 prints as `Cookbook root`." "Pull the image, then run the server in a terminal and leave it running while using the notebook":

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
  --model-path nvidia/Cosmos3-Nano-Sim-Auto \
  --revision main \
  --host 0.0.0.0 --port 30000
```

Readiness, from section 2: "The HTTP port opens only after the weights are loaded and the warmup request finishes, so wait for the `Uvicorn running` line. The start-up log also prints `Cosmos3-Nano-Sim-Auto pipeline stages created`, followed by the checkpoint's chunk, ring, and sampler settings." From "Verify the server": "The cell polls for up to 30 minutes, so you can run it while the server is still loading."

## Check the result

Apply the row's [validation checklist](../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout) to every sample, then the checks of the model and backend you ran:

- **Transfer-Auto, Cosmos Framework:** [Sampling and output checks](#sampling-and-output-checks) above, the guide's three checks for a joint multiview rollout, [Validate the rollout](references/transfer-auto-guide.md#validate-the-rollout), and "5. Inspect the result" of the native reference.
- **Sim-Auto, Cosmos Framework:** [Check the result (Sim-Auto)](#check-the-result-sim-auto) above and [5. Inspect the result](references/sim-auto-cosmos-framework.md#5-inspect-the-result) of the native reference.
- **vLLM-Omni and SGLang:** the validation cells of the reference you ran: "Validate and view" in the two vLLM-Omni references and the Sim-Auto SGLang reference; "Split and view the camera videos" and "Validate generated outputs and display cameras" in the Transfer-Auto SGLang reference.
- **Rendered WSM controls:** "3. Check the rendered views" of [`references/generate-wsm.md`](references/generate-wsm.md) and [WSM control checks](references/wsm-conventions.md).

## Troubleshooting

Inference symptoms of both models, with how to collect one failing example and report it: [`references/troubleshooting.md`](references/troubleshooting.md) ([Transfer-Auto](references/troubleshooting.md#transfer-auto), [Sim-Auto](references/troubleshooting.md#sim-auto)). Served requests: [vLLM-Omni troubleshooting](references/transfer-auto-guide.md#vllm-omni-troubleshooting) in the guide and the "Next steps" section of each served reference. Environment, CUDA, import and container failures: `../cosmos3-env-troubleshoot/SKILL.md`. Training, checkpoint, resume and distillation symptoms are in the troubleshooting reference of the Auto post-train skill: [Transfer-Auto](../cosmos3-post-train-simulation-auto/references/troubleshooting.md#troubleshoot-cosmos3-nano-transfer-auto) and [Sim-Auto](../cosmos3-post-train-simulation-auto/references/troubleshooting.md#troubleshoot-cosmos3-nano-sim-auto).

## Adapt the model

To adapt to your own rig or data, fine-tune Transfer-Auto with the [Transfer-Auto post-training skill](../cosmos3-post-train-simulation-auto/SKILL.md#fine-tune-transfer-auto), then distill it into a Sim-Auto student with the [Sim-Auto post-training skill](../cosmos3-post-train-simulation-auto/SKILL.md#distill-sim-auto).

## Reference files

Every file under `references/`, `assets/` and `scripts/helpers`, and the entry points of the renderer package in `scripts/wsm`. The converted guide and notebooks replace the cookbook files: do not look for the notebooks.

| File                                                                                                                                                                             | When to read it                                                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/auto-readme.md`                                                                                                                                                      | The Auto cookbook README: notebook chooser, "Setup and sample", sampling and custom data, rendering WSM from your own scene                                                                          |
| `references/transfer-auto-guide.md`                                                                                                                                              | The cookbook guide `auto.md`: supported backends, the Cosmos Framework CLI recipe and request shape, the vLLM-Omni and SGLang request contracts, "Validate the rollout", "vLLM-Omni troubleshooting" |
| `references/transfer-auto-cosmos-framework.md`                                                                                                                                   | `run_auto_with_cosmos_framework.ipynb`: native Transfer-Auto, sample download through output inspection                                                                                              |
| `references/transfer-auto-vllm-omni.md`                                                                                                                                          | `run_auto_with_vllm_omni.ipynb`: server, client, WSM-only and joint RGB and LiDAR examples                                                                                                           |
| `references/transfer-auto-sglang.md`                                                                                                                                             | `run_auto_with_sglang.ipynb`: server, shared setup, WSM-only and joint RGB and LiDAR examples                                                                                                        |
| `references/sim-auto-cosmos-framework.md`                                                                                                                                        | `run_sim-auto_with_cosmos_framework.ipynb`: native Sim-Auto, sample download through output inspection                                                                                               |
| `references/sim-auto-vllm-omni.md`                                                                                                                                               | `run_sim_auto_with_vllm_omni.ipynb`: server, client, RGB and joint RGB and LiDAR examples                                                                                                            |
| `references/sim-auto-sglang.md`                                                                                                                                                  | `run_sim-auto_with_sglang.ipynb`: server, client, the RGB example                                                                                                                                    |
| `references/generate-wsm.md`                                                                                                                                                     | `generate_wsm.ipynb`: render WSM v4.0.1 controls from your RDS-HQ scene, caption them, export a native request                                                                                       |
| `references/start-here-and-sampling.md`                                                                                                                                          | "Start here" of both models and the Transfer-Auto sampling notes, moved out of the two skill bodies                                                                                                  |
| `references/input-preparation.md`                                                                                                                                                | Camera, WSM and caption input checks: reference rig or custom rig, camera-to-identity mapping, controls and synchronization, packaging                                                               |
| `references/caption-guide.md`                                                                                                                                                    | Per-view caption generation and policy (`av-per-view-caption`): options and defaults, WSM inspection, validation                                                                                     |
| `references/caption-policy.md`                                                                                                                                                   | Training-aligned caption style, authoring or reviewing captions, the automatic prefixes                                                                                                              |
| `references/input-and-execution.md`                                                                                                                                              | Request fields, sampling and outputs: camera request, caption sidecars, joint LiDAR, GPU memory and parallelism, resolution, length, FPS, guidance                                                   |
| `references/sim-auto-inputs.md`                                                                                                                                                  | Input preparation, captions and request fields for Sim-Auto: the causal rules and the deltas over the Transfer-Auto guide                                                                            |
| `references/model-contract.md`                                                                                                                                                   | Model contract: inputs that must stay consistent, model settings, workflows and handoffs; section "Sim-Auto" for the student's settings                                                              |
| `references/framework-runtime.md`                                                                                                                                                | Framework environment and model assets: the `uv sync` environment, model snapshots, execution modes                                                                                                  |
| `references/reference-scene.md`                                                                                                                                                  | Shared Auto reference scene: download and checksum, files, caption styles, binding `AUTO_REQUEST`                                                                                                    |
| `references/camera-identities.md`                                                                                                                                                | Camera keys and learned camera IDs, and the rig rules that go with them                                                                                                                              |
| `references/wsm-conventions.md`                                                                                                                                                  | WSM control checks: what the WSM v4.0.1 colors mean (objects, signs, signals, road markings, lane lines), renderer evidence and geometry                                                             |
| `references/troubleshooting.md`                                                                                                                                                  | Inference symptoms and first checks for both models; collecting one failing example                                                                                                                  |
| `assets/negative-prompt.txt`                                                                                                                                                     | The Transfer-Auto per-view negative prompt read by the run block and the references                                                                                                                  |
| `assets/daytime_scene/README.md`                                                                                                                                                 | What the four request templates are, the caption styles, the LiDAR formats                                                                                                                           |
| `assets/daytime_scene/transfer_auto_wsm.json`, `assets/daytime_scene/transfer_auto_joint.json`, `assets/daytime_scene/sim_auto.json`, `assets/daytime_scene/sim_auto_joint.json` | The vLLM-Omni request templates the served references load                                                                                                                                           |
| `assets/workflow.svg`                                                                                                                                                            | Diagram "Transfer-Auto end-to-end workflow" from the cookbook assets                                                                                                                                 |
| `scripts/helpers/daytime_sample.py`, `scripts/helpers/auto_helpers.py`                                                                                                           | Imported by the references: sample download with checksum verification and request preparation; CPU-only request, media, HTTP and validation helpers                                                 |
| `scripts/helpers/sim_auto_helpers.py`, `scripts/helpers/sim_auto_vllm_omni_helpers.py`, `scripts/helpers/sim_auto_sglang_helpers.py`                                             | Output checks and previews for native runs; request helpers for the Sim-Auto vLLM-Omni and SGLang references                                                                                         |
| `scripts/helpers/lidar_rangemap.py`, `scripts/helpers/lidar_visualization.py`                                                                                                    | LiDAR: convert a sparse HD-map range-map archive to the numeric control; CPU previews                                                                                                                |
| `scripts/helpers/wsm_helpers.py`, `scripts/helpers/__init__.py`                                                                                                                  | Imported by the WSM reference: inspect an RDS-HQ scene and export native requests; the package marker                                                                                                |
| `scripts/wsm/README.md`                                                                                                                                                          | The standalone WSM renderer package in `scripts/wsm`: get, install, NVIDIA EGL check, render, verify                                                                                                 |
| `scripts/wsm/skills/scenario-to-rds-hq/SKILL.md`                                                                                                                                 | The renderer package's conversion skill: bring a scenario in another format into RDS-HQ                                                                                                              |
| `scripts/wsm/docs/rds-hq.md`, `scripts/wsm/docs/scenario-to-rds-hq.md`, `scripts/wsm/docs/camera-rig/guide.md`, `scripts/wsm/docs/lidar-rangemap.md`, `scripts/wsm/docs/wsm4.md` | Renderer documentation: RDS-HQ format, using the conversion skill, Camera Rig Explorer, LiDAR range maps, WSM4 rendering                                                                             |

## Related skills

| Skill                                               | When to use                                                                                  |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `../cosmos3-inference-simulation/SKILL.md`          | Entry skill of the Simulation row: model chooser, shared prerequisites, validation checklist |
| `../cosmos3-inference-simulation-bimanual/SKILL.md` | Sim-Bimanual inference                                                                       |
| `../cosmos3-inference-simulation-transfer/SKILL.md` | Sim-Transfer inference                                                                       |
| `../cosmos3-inference-transfer/SKILL.md`            | Control-to-video transfer with Cosmos3-Nano or Cosmos3-Super                                 |
| `../cosmos3-setup/SKILL.md`                         | Framework installation and checkpoint download                                               |
| `../cosmos3-inference/SKILL.md`                     | Framework-native inference reference: arguments and parallelism                              |
| `../cosmos3-env-troubleshoot/SKILL.md`              | CUDA, import, container and download failures                                                |
| `../cosmos3-post-train-simulation-auto/SKILL.md`    | Transfer-Auto and Sim-Auto post-training: fine-tune Transfer-Auto, distill Sim-Auto          |
