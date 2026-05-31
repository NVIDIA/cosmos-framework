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
| Reasoner Alignment SFT                       | `launch_sft_llava_ov.sh`              |
| Reasoner Alignment SFT (Cosmos3-Nano)        | `launch_sft_videophy2_nano.sh`        |
