# Cosmos3 SFT Examples

Runnable artifacts for Cosmos3 supervised fine-tuning. The end-to-end walkthrough — data preparation, base-checkpoint conversion, launch, outputs, export to safetensors, and evaluation — lives in **[docs/training.md](../docs/training.md)**. Start there.

This directory contains:

- `toml/sft_config/` — TOML recipes consumed by `cosmos_framework.scripts.train --sft-toml=…`. One file per recipe. The TOML is validated against the pydantic schema at [`cosmos_framework/configs/toml_config/sft_config.py`](../cosmos_framework/configs/toml_config/sft_config.py) at load time.
- `launch_sft_*.sh` — paired launch shells. Each declares `TOML_FILE` plus `: "${DATASET_PATH:=…}"` / `: "${BASE_CHECKPOINT_PATH:=…}"` defaults (full repo-relative paths, matching what [`docs/training.md`](../docs/training.md) shows) and sources [`_sft_launcher_common.sh`](./_sft_launcher_common.sh), which sets the `torchrun` flags and forwards into `cosmos_framework.scripts.train`. `export`ing those vars in your shell before launching wins over the defaults; otherwise just run the shell after Steps 1+2 of `docs/training.md`.
- `inference.py`, `inference_pipeline.py` — runnable inference helpers; see [docs/inference.md](../docs/inference.md).

## Recipe → launch shell

| Recipe                                       | Launch shell                          |
| -------------------------------------------- | ------------------------------------- |
| Vision SFT (Cosmos3-Nano)                    | `launch_sft_vision_nano.sh`           |
| Vision SFT LoRA (Cosmos3-Super)              | `launch_sft_vision_super.sh`          |
| Vision SFT (Cosmos3-Edge)                    | `launch_sft_vision_edge.sh`           |
| Reasoner Alignment SFT                       | `launch_sft_llava_ov.sh`              |
| Reasoner Alignment SFT (Cosmos3-Nano)        | `launch_sft_videophy2_nano.sh`        |
| Reasoner Alignment SFT (Cosmos3-Super)       | `launch_sft_videophy2_super.sh`       |
| Reasoner Alignment SFT (Cosmos3-Edge)        | `launch_sft_videophy2_edge.sh`        |
| Video-conversation SFT (Cosmos3-Nano)        | `launch_sft_video.sh`                 |

## Video SFT recipes

`launch_sft_video.sh` uses `toml/sft_config/video_sft_nano.toml`. Export
`COSMOS_VIDEO_TRAIN_ANNOTATION`, `COSMOS_VIDEO_TRAIN_MEDIA`,
`COSMOS_VIDEO_VAL_ANNOTATION`, `COSMOS_VIDEO_VAL_MEDIA`, and
`VLM_SAFETENSORS_PATH` before launching. Annotations are JSON arrays of video
paths and ShareGPT/LLaVA conversation turns; media variables identify their roots.
Optional `COSMOS_VIDEO_TRAIN_LIMIT` and `COSMOS_VIDEO_VAL_LIMIT` bound the sample counts.

Use `video_sft_edge.toml` with an Edge checkpoint for the corresponding Edge
recipe. `video_reasoning_sft.toml` instead accepts the task-aware
`cosmos-video-reasoning-v1.0` format through the native Framework dataset loader:
set `COSMOS_VIDEO_{TRAIN,VAL}_ANNOTATIONS` to paths or JSON path arrays and
`COSMOS_VIDEO_{TRAIN,VAL}_MEDIA_ROOTS` to a shared root or matching root array.
It interleaves answer/reasoning training targets and validates answer targets.
These recipes can be passed directly to `torchrun -m cosmos_framework.scripts.train
--sft-toml=examples/toml/sft_config/<recipe>.toml` with the desired topology.

The conversation recipes retain their 20-step, two-epoch smoke schedule; the
task-aware recipe retains its distinct 10-step, one-epoch schedule and learning
rate. Adjust the recipes for production datasets; they are not full-run presets.
