---
name: cosmos3-nano-transfer-auto-inference
description: Prepare or run Cosmos3-Nano-Transfer-Auto in Cosmos Framework on camera WSM controls and per-view captions, with optional initial RGB and joint LiDAR. Use for camera mapping, WSM/renderer and synchronization checks, supplied-caption review, av-per-view-caption generation or upsampling, targeted-KPI captions, native requests, sampling settings and output inspection. Input-checking and caption-only requests stop after the requested workflow.
---

# Run inference with Cosmos3-Nano-Transfer-Auto

Generate one video per selected camera from that camera's WSM control and caption.
Add an HD-map range control for joint RGB/LiDAR generation. Use initial camera
images for image-to-video, or omit them for text-to-video with WSM.

The released model is optimized for the reference camera rig. You can try a
custom rig, but expect reduced quality as camera placement, viewing direction,
FOV and lens projection diverge. Choose between
[retargeting WSM to the reference rig and retaining your own rig](references/input-preparation.md#choose-reference-rig-or-custom-rig-generation);
the latter may require [post-training](../post-training/SKILL.md#quickstart-fine-tune-the-released-model).

## Start here

1. **Try the reference scene first.** Use its complete request and the released
   model to establish a baseline before substituting your own inputs.
2. **Choose and match your cameras.** Follow the
   [rig workflow](references/input-preparation.md#choose-reference-rig-or-custom-rig-generation), then use the
   [Camera Rig Explorer](../../cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/guide.md) and
   [mapping guide](references/input-preparation.md#2-map-physical-cameras-to-learned-identities)
   to choose the closest supported identities for your rig. Keep each camera's
   RGB, WSM and caption together; `camera_key` selects its learned view embedding.
3. **Prepare camera-local captions.** Describe the requested scene in the
   [training-aligned caption style](../shared/caption-policy.md#keep-training-and-inference-captions-aligned),
   then check synchronization and agreement across views.
4. **Generate, inspect, then tune.** The reference recipe uses **720p**, guidance
   **3**, shift **10**, **35 steps** and the bundled [negative prompt](assets/negative-prompt.txt).
   Read [what these settings do](references/input-and-execution.md#sampling)
   before changing them; inspect every camera, not only the front view.

## Choose the task

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

For model/configuration questions, read the [model contract](../shared/model-contract.md).
An explanation request does not authorize jobs, uploads or changes to source assets.

## Prepare

1. Follow [environment and snapshots](../shared/framework-runtime.md). Set
   `TRANSFER_AUTO_HF` to a complete local Transfer-Auto snapshot.
2. Download and checksum-verify the
   [shared reference scene](../shared/reference-scene.md#get-the-scene), or prepare
   your own scene.
3. Run [input preparation](references/input-preparation.md), which invokes
   [per-view captioning](references/caption-guide.md) when requested.
4. For the reference scene, set
   `AUTO_REQUEST="$AUTO_SAMPLE_ROOT/input-transfer-auto.jsonl"`.
   It selects paragraph captions by default; follow
   [Choose a caption style](../shared/reference-scene.md#choose-a-caption-style)
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
files. See [automatic prefixes](../shared/caption-policy.md#caption-text-and-automatic-prefixes).

## Run

From the Framework root, inside your GPU allocation:

```bash
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0
AUTO_NEGATIVE_PROMPT="$(cat .agents/skills/cosmos3-nano-transfer-auto/inference/assets/negative-prompt.txt)"
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

## Sampling and output checks

The reference request generates 297 frames at 720p/30 FPS. For a short first
check, change the command to `--resolution 480` and prepare a 61-frame interval.
Match caption spans and control coverage to the selected interval:
caption-driven Transfer-Auto output length can differ from a top-level override.

Both 480p and 720p are supported. Use `aspect_ratio: "16,9"` for the
training-matched 832×480 or 1280×720 sizes; see
[resolution and aspect ratio](references/input-and-execution.md#resolution-and-aspect-ratio).
Transfer-Auto trains on approximately 13.3-second chunks and supports longer
RGB sequences through chunkwise continuation. Read the
[duration guide](references/input-and-execution.md#generation-length-and-long-rollouts)
before extending a request, especially for joint RGB/LiDAR.

For 10-FPS RGB-only inputs, follow the
[recommended inference configuration](references/input-and-execution.md#recommended-inference-configuration-for-10-fps-inputs):
keep the input frames unchanged, use `--fps 30` for model conditioning, and
encode the generated frames at 10 FPS for playback. Joint RGB/LiDAR requires
separate clock/alignment handling; do not apply this configuration unchanged.

For separate per-view captions, route the supplied
[negative text](assets/negative-prompt.txt) through `per_view_negative_prompt`,
not `negative_prompt`. Preserve the user's explicit settings in comparisons.
The [sampling guide](references/input-and-execution.md#sampling) explains the
guidance choices and joint LiDAR timing.

Inspect `sample_args.json`, `sample_outputs.json`, all camera/control videos,
and `lidar.safetensors` for joint output. Check view identity, actual duration,
numeric range/intensity and frame IDs. Save 50/50 WSM/RGB overlays using matching
crops and times. A clean overlay checks geometry, not the correctness of a
physical camera alias.

Use a new output directory when changing inputs or settings.

## References

- [Camera, WSM and caption input checks](references/input-preparation.md)
- [Per-view caption generation and policy](references/caption-guide.md)
- [Model contract](../shared/model-contract.md)
- [Framework environment and model assets](../shared/framework-runtime.md)
- [Shared Auto reference scene](../shared/reference-scene.md)
- [Request fields, sampling and outputs](references/input-and-execution.md)
