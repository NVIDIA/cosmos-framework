---
name: cosmos3-transfer-inference
description: >
  Guide users through Cosmos3 transfer inference, conditioned transfer, control
  transfer, and multi-transfer generation with edge, blur, depth, segmentation,
  or WSM controls. Use when the user asks to transfer a source image or video,
  use multiple conditioning signals, run edge/blur/depth/seg/wsm controls, or
  make transfer work on Cosmos3-Nano or Cosmos3-Super.
---

# Cosmos3 transfer inference

## Goal

Run Cosmos3 transfer generation through the normal inference entry point while making the transfer-specific JSON fields, launch mode, and output checks explicit.

## Path convention

All paths below are relative to the Cosmos Framework repo root. Run commands from the directory containing `pyproject.toml`.
Input paths inside JSON/YAML specs are resolved relative to the spec file's directory.

## Source facts

- Transfer is activated when a sample has at least one hint field: `edge`, `blur`, `depth`, `seg`, or `wsm`.
- The cookbook transfer specs use local structured caption files via `prompt_path`, local structured negative caption files via `negative_prompt_file`, and precomputed control videos via each hint's `control_path`.
- `prompt_path` may point to `.json` for transfer; the framework loads and compacts that JSON into the prompt string. `negative_prompt_file` also loads JSON into `negative_prompt`.
- Cookbook prompt and negative-prompt JSON files are dense scene descriptions, not one-line prompts. They should cover subjects, background, lighting, composition, camera, actions or segments, temporal caption, audio, resolution, aspect ratio, duration, and fps when relevant.
- `edge` and `blur` can also be computed on the fly from `vision_path`; use that as a local convenience smoke-test path, not as the default cookbook-parity transfer spec. `depth`, `seg`, and `wsm` require a precomputed `control_path`.
- Multi-transfer is not a separate command. Put multiple hint objects in one sample. The model packs `[ctrl_1, ..., ctrl_N, target]`, normalises non-negative per-hint `weight` values, and uses weighted multi-control attention.
- Single-hint transfer defaults for `guidance`, `control_guidance`, and `shift` are applied only when exactly one hint is present. For multi-transfer, set those fields explicitly.
- Transfer samples are generated one sample at a time. Do not mix transfer and non-transfer samples in one batch. You may pass multiple transfer input files to one command so the model loads once.
- Outputs are written under `<output_dir>/<sample_name>/`: `vision.jpg` or `vision.mp4`, `control_<hint>.jpg` or `control_<hint>.mp4`, `sample_args.json`, and `sample_outputs.json`.
- The public Cosmos transfer notebook runs the same checked-in `specs/<control>.json` files for Nano and Super. Its Nano cells use `python` with `--parallelism-preset=latency`; its Super cells use `torchrun` with `--parallelism-preset=latency`.

## Default workflow

1. Read `AGENTS.md`, `docs/inference.md`, `cosmos_framework/inference/args.py`, `cosmos_framework/inference/transfer.py`, and `cosmos_framework/inference/inference.py` before changing commands or docs.
2. Verify the environment with `uv run --all-extras --group=cu130-train python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.
3. Choose model launch:
   - Nano: use a single process and `--parallelism-preset=latency` for cookbook transfer specs.
   - Super: use `torchrun` across visible ranks and `--parallelism-preset=latency` for cookbook transfer specs.
4. Build or reuse a transfer input JSON with `model_mode`, `prompt_path`, `negative_prompt_file`, and one or more transfer hints. Prefer precomputed `control_path` assets when matching cookbook behaviour.
5. Preflight every source and control media file. Record image dimensions and video width, height, frame count, frame rate, and duration before running inference.
6. Set `seed`, `resolution`, `aspect_ratio`, `num_frames`, `fps`, `shift`, `guidance`, `control_guidance`, `num_video_frames_per_chunk`, `num_conditional_frames`, `num_first_chunk_conditional_frames`, and `share_vision_temporal_positions`.
7. Run inference with `--no-guardrails` only when the task is validation/smoke testing and guardrail downloads would obscure transfer behaviour.
8. Check generated media with `find`, `ffprobe`, and image inspection. Report source media, control media, transferred output, command, dimensions, video length, frame count, and any warnings.

## JSON patterns

Cookbook-aligned video edge transfer:

```json
{
  "name": "transfer_edge",
  "model_mode": "video2video",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 121,
  "fps": 30,
  "shift": 10.0,
  "num_steps": 50,
  "seed": 2026,
  "num_video_frames_per_chunk": 121,
  "num_conditional_frames": 1,
  "num_first_chunk_conditional_frames": 0,
  "share_vision_temporal_positions": true,
  "negative_metadata_mode": "none",
  "negative_prompt_keep_metadata": false,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "negative_prompt_file": "../assets/negative_prompt.json",
  "prompt_path": "../assets/edge/prompt.json",
  "edge": {
    "control_path": "../assets/edge/control_edge.mp4",
    "preset_edge_threshold": "medium"
  }
}
```

Cookbook-aligned WSM transfer uses the same structure, but the WSM single-hint default shape is different:

```json
{
  "name": "transfer_wsm",
  "model_mode": "video2video",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 101,
  "fps": 10,
  "shift": 10.0,
  "num_steps": 50,
  "seed": 2026,
  "num_video_frames_per_chunk": 101,
  "num_conditional_frames": 1,
  "num_first_chunk_conditional_frames": 0,
  "share_vision_temporal_positions": true,
  "negative_metadata_mode": "none",
  "negative_prompt_keep_metadata": false,
  "guidance": 1.0,
  "control_guidance": 3.0,
  "negative_prompt_file": "../assets/negative_prompt.json",
  "prompt_path": "../assets/wsm/prompt.json",
  "wsm": {
    "control_path": "../assets/wsm/control_wsm.mp4"
  }
}
```

Multi-transfer with precomputed controls from the same source clip:

```json
{
  "name": "transfer_edge_blur",
  "model_mode": "video2video",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 121,
  "fps": 30,
  "shift": 10.0,
  "num_steps": 50,
  "seed": 2026,
  "num_video_frames_per_chunk": 121,
  "num_conditional_frames": 1,
  "num_first_chunk_conditional_frames": 0,
  "share_vision_temporal_positions": true,
  "negative_metadata_mode": "none",
  "negative_prompt_keep_metadata": false,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "negative_prompt_file": "../assets/negative_prompt.json",
  "prompt_path": "../assets/same_scene/prompt.json",
  "edge": {
    "control_path": "../assets/same_scene/control_edge.mp4",
    "preset_edge_threshold": "medium",
    "weight": 0.5
  },
  "blur": {
    "control_path": "../assets/same_scene/control_blur.mp4",
    "preset_blur_strength": "medium",
    "weight": 0.5
  }
}
```

Single-frame transfer with precomputed control:

```json
{
  "name": "transfer_edge_image",
  "model_mode": "image2image",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 1,
  "shift": 10.0,
  "num_steps": 50,
  "seed": 2026,
  "negative_metadata_mode": "none",
  "negative_prompt_keep_metadata": false,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "negative_prompt_file": "../assets/negative_prompt.json",
  "prompt_path": "../assets/edge_image/prompt.json",
  "edge": {
    "control_path": "../assets/edge_image/control_edge.jpg",
    "preset_edge_threshold": "medium"
  }
}
```

Local convenience smoke test using source media to derive edge or blur:

```json
{
  "name": "edge_blur_from_source",
  "model_mode": "video2video",
  "prompt_path": "prompts/source_scene.json",
  "negative_prompt_file": "prompts/negative_prompt.json",
  "vision_path": "inputs/source.mp4",
  "resolution": "256",
  "aspect_ratio": "16,9",
  "num_frames": 17,
  "fps": 24,
  "shift": 10.0,
  "num_steps": 20,
  "seed": 23,
  "num_video_frames_per_chunk": 17,
  "num_conditional_frames": 1,
  "num_first_chunk_conditional_frames": 0,
  "share_vision_temporal_positions": true,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "show_input": true,
  "show_control_condition": true,
  "edge": {
    "preset_edge_threshold": "medium",
    "weight": 0.7
  },
  "blur": {
    "preset_blur_strength": "medium",
    "weight": 0.3
  }
}
```

## Prompt files

- Prefer `prompt_path` over inline `prompt` for transfer. Use a local `.json` file for structured prompts.
- Prefer `negative_prompt_file` over inline `negative_prompt` for transfer. Use a local `.json` file for structured negative captions.
- Do not reduce transfer prompts to one sentence unless the user explicitly asks for a tiny smoke test. Include visual content, timing, camera, lighting, action, and output metadata.
- For single-frame prompts, the framework removes video-only metadata when formatting JSON prompts; still keep the source JSON structured.
- `prompt_path` and `negative_prompt_file` must be local files. If using public cookbook assets, clone or copy them locally before running.
- For multi-transfer, create all control files from the same source image or video. Do not combine unrelated cookbook assets just because their hint names differ.

## Media preflight

- Before inference, inspect every `vision_path` and every hint `control_path`. Do this for generated outputs too.
- For video files, record width, height, frame count, frame rate, and duration:

```shell
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames,duration \
  -show_entries format=duration \
  -of default=noprint_wrappers=1 input.mp4
```

- For image files, record width and height:

```shell
python - <<'PY'
from pathlib import Path
from PIL import Image

for path in ["input_or_control.jpg"]:
    with Image.open(Path(path)) as image:
        print(f"{path}: {image.width}x{image.height}")
PY
```

- If multiple control videos are active, their duration, frame count, and fps should match or be intentionally trimmed to the same range. If they differ, stop and report the mismatch before generation.
- If a control image/video has a different aspect ratio from the requested spec, state that the framework will resize/crop to the requested `resolution` and `aspect_ratio`; do not hide that transformation in the report.

## Launch templates

Nano:

```shell
LD_LIBRARY_PATH= uv run --all-extras --group=cu130-train \
  python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i transfer.json \
  -o outputs/transfer_nano \
  --checkpoint-path Cosmos3-Nano \
  --seed=17
```

Super on `N` GPUs:

```shell
LD_LIBRARY_PATH= torchrun --nproc-per-node=N -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i transfer.json \
  -o outputs/transfer_super \
  --checkpoint-path Cosmos3-Super \
  --seed=17
```

Use a local checkpoint directory in `--checkpoint-path` when the model is already downloaded.

To mirror the public cookbook exactly, set `SPEC` to a checked-in `specs/<control>.json` file and set the output directory to either `Cosmos3-Nano` or `Cosmos3-Super`; change only the launcher and `--checkpoint-path`.

## Conditioning rules

- Any non-empty subset of `edge`, `blur`, `depth`, `seg`, and `wsm` is valid if required paths exist.
- Use precomputed `control_path` for cookbook parity. Use `vision_path` only when intentionally deriving `edge` or `blur` controls from the source media.
- Without `vision_path`, every active hint must provide `control_path`; aspect ratio is auto-detected from the first control when `aspect_ratio` is not set.
- Keep controls aligned in time and content. The loader resizes controls to the requested resolution/aspect ratio, but it does not fix semantic mismatches.
- `weight` must be non-negative and the total weight must be positive. Equal weights are implied when omitted.
- Hint order is deterministic by enum order: `edge`, `blur`, `depth`, `seg`, `wsm`, not JSON insertion order.
- For multi-transfer, explicitly set `guidance`, `control_guidance`, and `shift`; the code applies task-tuned defaults only for single-hint transfer.

## Validation checklist

- Run one single-frame transfer and confirm `vision.jpg` plus `control_<hint>.jpg`.
- Run one video transfer and confirm `vision.mp4` plus `control_<hint>.mp4`.
- Run one multi-transfer sample with at least two hints and confirm every `control_<hint>` output exists.
- For Nano and Super, validate the same JSON schema; change only the launcher and `--checkpoint-path`.
- Record command, source media, control media, output media, image dimensions, video dimensions, video duration, frame count, fps, and any CUDA/checkpoint warnings.
