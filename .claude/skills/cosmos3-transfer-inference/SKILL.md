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
Local `prompt_path`, `vision_path`, and hint `control_path` values inside JSON/YAML specs are resolved relative to the spec file's directory.
`negative_prompt_file` is a plain string: when its loader is reached, it checks the process working directory first and then `cosmos_framework/inference/defaults/<basename>`.

## Source facts

- Transfer is activated when a sample has at least one hint field: `edge`, `blur`, `depth`, `seg`, or `wsm`.
- The cookbook transfer specs use local structured caption files via `prompt_path`, declare negative captions via `negative_prompt_file`, and use either precomputed controls or source-media-derived edge/blur controls.
- `prompt_path` may point to `.json` for transfer; the framework loads and compacts that JSON into the prompt string.
- A user `negative_prompt_file` is normally ignored because modality defaults populate `negative_prompt` before the transfer loader runs. Video-to-video therefore normally uses the packaged `neg_prompts.json`; image-to-image defaults to an empty negative prompt. Use an inline `negative_prompt` when a custom value is required.
- Cookbook prompt and negative-prompt JSON files are dense scene descriptions, not one-line prompts. They should cover subjects, background, lighting, composition, camera, actions or segments, temporal caption, audio, resolution, aspect ratio, duration, and fps when relevant.
- `edge` and `blur` can also be computed on the fly from `vision_path`; the cookbook uses that shape for multi-control, while its single-control specs use precomputed controls. `depth`, `seg`, and `wsm` require a precomputed `control_path`.
- Multi-transfer is not a separate command. Put multiple hint objects in one sample. The model packs `[ctrl_1, ..., ctrl_N, target]`, normalises non-negative per-hint `weight` values, and uses weighted multi-control attention.
- Single-hint transfer defaults for `guidance`, `control_guidance`, and `shift` are applied only when exactly one hint is present. For multi-transfer, set those fields explicitly.
- Transfer samples are generated one sample at a time. Do not mix transfer and non-transfer samples in one batch. Multiple transfer files may be passed to one command so the model loads once, but retain the default `--max-num-seqs=1` and leave `--max-model-len` unset.
- Outputs are written under `<output_dir>/<sample_name>/`: `vision.jpg` or `vision.mp4`, separate `control_<hint>.jpg` or `control_<hint>.mp4` files, `sample_args.json`, and `sample_outputs.json`.
- The public [Cosmos transfer cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/transfer) runs the same checked-in single-control `specs/<control>.json` files for Nano and Super. Its Nano cells use `python` with `--parallelism-preset=latency`; its Super cells use `torchrun` with `--parallelism-preset=latency`. The notebook's multi-control section is Nano-only.

## Default workflow

1. Read `AGENTS.md`, `docs/inference.md`, `cosmos_framework/inference/args.py`, `cosmos_framework/inference/transfer.py`, and `cosmos_framework/inference/inference.py` before changing commands or docs.
2. Select the installation group for the host's NVIDIA driver: `export UV_GROUP=cu130-train` for CUDA 13.x, or `export UV_GROUP=cu128-train` for CUDA 12.8. Verify it with `uv run --all-extras --group="$UV_GROUP" python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.
3. Choose model launch:
   - Nano: use a single process and `--parallelism-preset=latency` for cookbook transfer specs.
   - Super: use `torchrun` across visible ranks and `--parallelism-preset=latency` for cookbook transfer specs.
4. Build or reuse a transfer input JSON with `model_mode`, `prompt_path`, and one or more transfer hints. Use inline `negative_prompt` for a custom negative caption. Prefer precomputed `control_path` assets when matching single-control cookbook behaviour.
5. Preflight every source and control media file. A nonexistent local `control_path` is rejected while JSON/YAML is parsed, but recheck it before a long run in case it has moved or disappeared. Record image dimensions and video width, height, frame count, frame rate, and duration.
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
  "prompt_path": "../assets/edge/prompt.json",
  "emphasize_control_in_prompt": false,
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
  "prompt_path": "../assets/wsm/prompt.json",
  "emphasize_control_in_prompt": false,
  "wsm": {
    "control_path": "../assets/wsm/control_wsm.mp4"
  }
}
```

Multi-transfer adapted from the cookbook's checked-in `specs/multi_control.json`. Edge and blur are derived on the fly from the same `vision_path`:

```json
{
  "name": "transfer_multi_control",
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
  "prompt_path": "../assets/multi_control/prompt.json",
  "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_pouring.mp4",
  "emphasize_control_in_prompt": false,
  "edge": {
    "preset_edge_threshold": "medium",
    "weight": 0.5
  },
  "blur": {
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
  "num_frames": 1,
  "shift": 10.0,
  "num_steps": 50,
  "seed": 2026,
  "negative_metadata_mode": "none",
  "negative_prompt_keep_metadata": false,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "prompt_path": "../assets/edge_image/prompt.json",
  "emphasize_control_in_prompt": false,
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
  "vision_path": "inputs/source.mp4",
  "resolution": "256",
  "aspect_ratio": "16,9",
  "num_frames": 25,
  "max_frames": 25,
  "fps": 24,
  "shift": 10.0,
  "num_steps": 20,
  "seed": 2026,
  "num_video_frames_per_chunk": 25,
  "num_conditional_frames": 1,
  "num_first_chunk_conditional_frames": 0,
  "share_vision_temporal_positions": true,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "show_input": true,
  "show_control_condition": true,
  "emphasize_control_in_prompt": false,
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

This smoke example saves a four-pane composite in `vision.mp4`: input + edge + blur + generated. In general, the composite contains one generated pane, one pane for each of `K` active controls when `show_control_condition` is true, and an optional input pane when `show_input` is true and `vision_path` exists. Separate `control_<hint>.*` files are always saved. Set both display flags to false for a clean generated output.

## Prompt files

- Prefer `prompt_path` over inline `prompt` for transfer. Use a local `.json` file for structured prompts.
- For a custom negative caption, use inline `negative_prompt`. A user `negative_prompt_file` is normally ignored because modality defaults have already populated `negative_prompt`; treat that precedence bug as a runtime follow-up.
- Video-to-video normally receives the packaged `cosmos_framework/inference/defaults/neg_prompts.json`; image-to-image defaults to an empty negative prompt. The public cookbook's video negative-prompt asset is byte-identical to the packaged default, so those examples appear correct despite the ignored field.
- If the `negative_prompt_file` loader is reached, it checks the process working directory and then `cosmos_framework/inference/defaults/<basename>`. Unlike the typed media and prompt fields, it is not resolved relative to the spec.
- Do not reduce transfer prompts to one sentence unless the user explicitly asks for a tiny smoke test. Include visual content, timing, camera, lighting, action, and output metadata.
- For single-frame prompts, the framework removes video-only metadata when formatting JSON prompts; still keep the source JSON structured.
- `prompt_path` must be a local file. If using public cookbook assets, clone or copy them locally before running.
- For multi-transfer, derive every control from the same source image or video. Do not combine unrelated cookbook assets just because their hint names differ.

## Media preflight

- Before inference, inspect every `vision_path` and every hint `control_path`. Normal JSON/YAML parsing rejects a nonexistent local `control_path`; edge/blur on-the-fly computation is selected by intentionally omitting `control_path` while supplying `vision_path`. Recheck validated files before a long run and inspect generated outputs too.
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
- With `vision_path`, the framework honours an explicit `aspect_ratio` and resizes/crops source and controls to it at the requested `resolution`.
- Without `vision_path`, an explicit sample `aspect_ratio` is ignored: the framework detects the first precomputed control's ratio and uses it for every control at the requested `resolution`. Record the detected ratio and any crop applied to the other controls.

## Launch templates

Define the environment and cookbook paths before running from the Cosmos Framework repo root:

```shell
export UV_GROUP=cu130-train  # Use cu128-train when the driver supports CUDA 12.8, not 13.x.
export TRANSFER_ROOT=/absolute/path/to/cosmos/cookbooks/cosmos3/generator/transfer
export SPEC="$TRANSFER_ROOT/specs/edge.json"
export OUT_DIR="$TRANSFER_ROOT/outputs/Cosmos3-Nano"
```

`TRANSFER_ROOT/specs` and `TRANSFER_ROOT/assets` belong to the external [Cosmos transfer cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/transfer), not this repository. `SPEC` is an absolute cookbook spec path; `OUT_DIR` is a result directory, while `Cosmos3-Nano` and `Cosmos3-Super` below are checkpoint identifiers.

In an NGC/PyTorch container only, bundled libtorch paths may conflict with Triton/CUDA. Run `unset LD_LIBRARY_PATH` before the commands in that environment; do not clear it unconditionally on a normal host.

Nano:

```shell
uv run --all-extras --group="$UV_GROUP" \
  python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Nano
```

Super on `N` GPUs:

```shell
export OUT_DIR="$TRANSFER_ROOT/outputs/Cosmos3-Super"
uv run --all-extras --group="$UV_GROUP" \
  torchrun --nproc-per-node=N -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$SPEC" \
  -o "$OUT_DIR" \
  --checkpoint-path Cosmos3-Super
```

These commands retain the spec's `seed: 2026`. Use a local checkpoint directory in `--checkpoint-path` when the model is already downloaded.

For Super, keep the default full-world FSDP sharding unless the complete checkpoint fits in each smaller shard group. The latency preset overlays context and CFG parallelism on those same ranks, so do not add `--dp-shard-size=1` by default. This matches the public cookbook at commit [`bebca763`](https://github.com/NVIDIA/cosmos/tree/bebca76311266941d06c5f5572fb601184ba24fa/cookbooks/cosmos3/generator/transfer): its Super README and notebook launches use `torchrun --parallelism-preset=latency` without a data-parallel shard override.

## Conditioning rules

- Any non-empty subset of `edge`, `blur`, `depth`, `seg`, and `wsm` is valid if required paths exist.
- Use precomputed `control_path` for single-control cookbook parity. Intentionally omit `control_path` and provide `vision_path` to derive edge or blur controls on the fly, as in the cookbook multi-control spec.
- A nonexistent local `control_path` is rejected while a normal JSON/YAML spec is parsed; it does not silently select on-the-fly control. Preflight again before a long run in case a validated file has since disappeared.
- Without `vision_path`, every active hint must provide `control_path`; the first control determines aspect ratio even when the sample sets `aspect_ratio`. With `vision_path`, an explicit `aspect_ratio` is honoured. `resolution` is applied in both cases.
- Keep `num_first_chunk_conditional_frames: 0` when `vision_path` is absent. A positive value requires source frames and raises an error in a control-only run.
- Keep controls aligned in time and content. The loader can resize/crop them consistently, but it does not fix semantic mismatches.
- `weight` must be non-negative and the total weight must be positive. Equal weights are implied when omitted.
- Hint order is deterministic by enum order: `edge`, `blur`, `depth`, `seg`, `wsm`, not JSON insertion order.
- For multi-transfer, explicitly set `guidance`, `control_guidance`, and `shift`; the code applies task-tuned defaults only for single-hint transfer.
- `emphasize_control_in_prompt` defaults to true and appends a sentence requiring shape, contour, silhouette, position, and motion to follow the active controls. Set it to false for exact prompt fidelity, cookbook parity, clean baselines, or ablations.
- For video smoke tests, use at least 24 frames and a `4n+1` temporal shape. The example's 25 frames avoids the quality warning; 17 is also `4n+1` and is not rounded up, but it is below the recommended range. `max_frames` performs the actual source/control trim.
- Transfer batching supports one sample at a time. Retain the default `--max-num-seqs=1` and leave `--max-model-len` unset. Raising `max_num_seqs` can pack multiple transfers and reach the single-sample assertion; setting `max_model_len` while the default sequence budget remains set instead violates the packer's mutually exclusive budget assertion.

Verified single-hint defaults, applied only when the corresponding fields are omitted:

| hint                | guidance | control guidance | shift | additional defaults                                          |
| ------------------- | -------: | ---------------: | ----: | ------------------------------------------------------------ |
| edge / blur / depth |      3.0 |              1.5 |  10.0 | none                                                         |
| seg                 |      3.0 |              2.0 |  10.0 | none                                                         |
| wsm                 |      1.0 |              3.0 |  10.0 | `num_frames=101`, `fps=10`, `num_video_frames_per_chunk=101` |

## Validation checklist

- Run one single-frame transfer and confirm `vision.jpg` plus `control_<hint>.jpg`.
- Run one video transfer and confirm `vision.mp4` plus `control_<hint>.mp4`.
- Run one multi-transfer sample with at least two hints and confirm every `control_<hint>` output exists.
- For Nano and Super, validate the same JSON schema; change only the launcher, output directory, and `--checkpoint-path`.
- Record command, source media, control media, output media, image dimensions, video dimensions, video duration, frame count, fps, and any CUDA/checkpoint warnings.
