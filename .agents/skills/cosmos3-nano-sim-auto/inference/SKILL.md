---
name: cosmos3-nano-sim-auto-inference
description: Prepare inputs or run Cosmos3-Nano-Sim-Auto camera-only or joint camera/LiDAR generation with a released or exported student in Cosmos Framework. Check causal conditioning, per-camera captions, rolling metadata and saved outputs.
---

# Run inference with Cosmos3-Nano-Sim-Auto

Use the four-step causal student to generate synchronized camera views and
optional LiDAR from initial RGB, per-camera captions and WSM/HD-map controls.

The released model is optimized for the reference camera rig. You can try a
custom rig, but camera geometry differences can degrade quality. Follow the
[reference-rig retargeting or custom-rig workflow](../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md#choose-reference-rig-or-custom-rig-generation).
Retargeting WSM does not reproject initial RGB images. For adaptation, fine-tune
Transfer-Auto on your rig's paired data, then distill it into Sim-Auto.

## Start here

- **New rig:** compare it with the reference cameras in
  [Camera Rig Explorer](../../cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/guide.md), then follow the
  shared [camera-to-embedding mapping](../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md#2-map-physical-cameras-to-learned-identities).
  Map each camera to the closest supported identity and keep RGB, WSM and caption
  assignments consistent. For domain/rig adaptation, post-train Transfer-Auto
  first and then [distill it](../post-training/SKILL.md).
- **Captions:** use the same
  [camera-local, training-aligned style](../../cosmos3-nano-transfer-auto/shared/caption-policy.md#keep-training-and-inference-captions-aligned)
  as Transfer-Auto, checking appearance and motion across views and time.
- **Sampling:** use guidance **1**, control guidance **1** and the checkpoint's
  **four-step** schedule. Read [why Sim-Auto settings differ](../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#sim-auto-uses-a-distilled-schedule);
  guidance 1 still uses captions and controls, but a negative prompt adds no CFG effect.
- **Timing:** use aligned 30-FPS cameras for the reference recipe; joint Sim-Auto
  requires that camera rate. Read the [FPS guide](../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#source-timing-model-fps-and-playback-fps)
  before using lower-rate recordings. Changing playback speed does not correct
  model timing or resample controls.
- **Size and duration:** both 480p and 720p are supported; use the
  [training-matched landscape preset](../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#resolution-and-aspect-ratio).
  Sim-Auto supports long rolling generation, with qualitatively observed
  multi-minute rollouts. Read the [duration guide](../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#generation-length-and-long-rollouts)
  for quality expectations and full-horizon input requirements.

## Choose the task

- For input preparation only, follow the
  [input-preparation guide](references/input-preparation.md) and
  [caption guidance](references/input-preparation.md#camera-local-captions). Return the checked media inventory
  and per-view caption sidecars without launching generation.
- To generate videos and optional LiDAR, continue below.

## Prepare

1. Follow [Framework setup](../../cosmos3-nano-transfer-auto/shared/framework-runtime.md).
   Set `SIM_AUTO_HF` to the downloaded student directory, or a complete student
   export from [distillation](../post-training/SKILL.md).
2. Follow [input preparation](references/input-preparation.md). Both Auto models share
   the [reference scene](../../cosmos3-nano-transfer-auto/shared/reference-scene.md#get-the-scene);
   download the [sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/ab50792f104af5304bce8b4cb7c27c096b33e63f/assets/sample_data)
   at the pinned revision and verify its checksums.
3. Set `AUTO_REQUEST="$AUTO_SAMPLE_ROOT/input-sim-auto.jsonl"` for the
   reference scene, or prepare your own JSON/JSONL request using
   [these fields](references/input-preparation.md#request-fields). Set a fresh `AUTO_OUTPUT` directory.
   The sample request selects paragraph captions; use
   [caption selection](../../cosmos3-nano-transfer-auto/shared/reference-scene.md#choose-a-caption-style)
   for the recommended timestamped alternative.

Both models use `daytime_scene`: seven cameras, WSM v4.0.1 controls and paired
RGB/LiDAR targets. `AUTO_SAMPLE_ROOT` points to the downloaded
`assets/sample_data` directory; Sim-Auto uses its `input-sim-auto.jsonl` request.

Supply scene-caption bodies only. Framework automatically adds the camera-rig
and current-view direction/lens/FOV description, plus the WSM-control task
instruction as a system prompt. Do not paste these into your caption files;
see [automatic prefixes](../../cosmos3-nano-transfer-auto/shared/caption-policy.md#caption-text-and-automatic-prefixes).

Use an exported model snapshot, not a training experiment or raw DCP directory.
The artifact must include per-view-caption metadata, matching tokenizers and
`rolling_kv_cache_chunks`, `rolling_kv_sink_chunks`, `rolling_prompt_policy`.
Do not guess missing history settings from the model name.

## Run

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
[GPU memory and parallelism](../../cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#gpu-memory-and-parallelism)
before changing the allocation; a four-GPU example is not a guarantee of fit on
every device.

## Check the result

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
[troubleshooting](../troubleshooting/SKILL.md) for failures.

## References

- [Input preparation, captions and request fields](references/input-preparation.md)
- [Framework environment and model assets](../../cosmos3-nano-transfer-auto/shared/framework-runtime.md)
- [Shared Auto reference scene](../../cosmos3-nano-transfer-auto/shared/reference-scene.md)
