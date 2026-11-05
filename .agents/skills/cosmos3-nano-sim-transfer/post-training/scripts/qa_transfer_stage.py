#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Print native Framework training arguments with local assets and stage parents.

Run from FRAMEWORK_DIR. Capture stdout to a file, then read it with Bash mapfile;
see ../references/local-training.md. Native Framework factories own the recipes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXPERIMENTS: tuple[str, ...] = (
    "sim_local_transfer_stage0_bidir",
    "sim_local_transfer_stage1_tf",
    "sim_local_transfer_stage2_tfdcm",
    "sim_local_transfer",
)
LOCAL_CONFIG: str = "cosmos_framework/configs/simulation/local.py"


def local_path(name: str, *, directory: bool = False) -> Path:
    """Read an explicit existing path shared by every training rank."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name}; see references/local-training.md.")
    # Preserve snapshot symlink filenames: native loaders infer the format from
    # extensions such as .pth, while Hugging Face blob targets have no extension.
    path = Path(os.path.abspath(Path(value).expanduser()))
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"{name}: {path}")
    return path


def checkpoint_root(name: str) -> str:
    """Require DCP metadata, leaving tensor compatibility to the native loader."""
    root = local_path(name, directory=True)
    if not (root / "model/.metadata").is_file():
        raise FileNotFoundError(f"{name} must contain model/.metadata: {root}")
    return str(root)


def training_arguments(stage: int) -> list[str]:
    """Bind paths without copying or registering model, loader or loss recipes."""
    if stage not in range(4):
        raise ValueError("Stage must be 0, 1, 2 or 3.")
    framework = local_path("FRAMEWORK_DIR", directory=True)
    if not (framework / LOCAL_CONFIG).is_file():
        raise FileNotFoundError(framework / LOCAL_CONFIG)
    bundle = local_path("TRANSFER_BUNDLE_PATH", directory=True)
    for filename in ("manifest.json", "samples.json"):
        if not (bundle / filename).is_file():
            raise FileNotFoundError(bundle / filename)
    nano = local_path("NANO_HF_DIR", directory=True)
    tokenizer = nano / "text_tokenizer"
    if not tokenizer.is_dir():
        raise FileNotFoundError(tokenizer)

    values: dict[str, str | bool | int | list[str]] = {
        "model.config.tokenizer.bucket_name": "",
        "model.config.tokenizer.object_store_credential_path_pretrained": "",
        "model.config.tokenizer.vae_path": str(local_path("VIDEO_VAE_PATH")),
        "model.config.vlm_config.tokenizer.config_variant": "hf",
        "model.config.vlm_config.tokenizer.pretrained_model_name": str(tokenizer),
        "model.config.vlm_config.pretrained_weights.enabled": False,
        "checkpoint.load_from_object_store.enabled": False,
        "checkpoint.save_to_object_store.enabled": False,
        "checkpoint.load_training_state": False,
        "checkpoint.load_ema_to_reg": stage in (1, 2),
    }
    streams = (
        ("video_transfer_4modality_data_480", "video_transfer_4modality_data_720") if stage == 0 else ("local_data",)
    )
    for stream in streams:
        values[f"dataloader_train.dataloaders.{stream}.dataloader.dataset.bundle_path"] = str(bundle)

    if stage == 0:
        values.update(
            {
                "model.config.sound_tokenizer.bucket_name": "",
                "model.config.sound_tokenizer.object_store_credential_path_pretrained": "",
                "model.config.sound_tokenizer.avae_path": str(local_path("AUDIO_VAE_PATH")),
                "model.config.sound_tokenizer.avae_config_path": str(local_path("AUDIO_VAE_CONFIG_PATH")),
                "checkpoint.load_path": checkpoint_root("BASE_CKPT_ROOT"),
            }
        )
        if os.environ.get("TRANSFER_QA_BASE_MODEL_ONLY", "0") == "1":
            # A converted Nano DCP contains net.*, not net_ema.*. Keep every
            # regular-weight key strict; the native first-update beta=0 seeds
            # fresh EMA. The checkpointer ignores skips on same-job resume.
            values["checkpoint.keys_to_skip_loading"] = ["net_ema."]
            values["model.config.ema.iteration_shift"] = 0
    elif stage == 1:
        values["checkpoint.load_path"] = checkpoint_root("STAGE0_CKPT_ROOT")
    elif stage == 2:
        parent = checkpoint_root("STAGE1_CKPT_ROOT")
        values["checkpoint.load_path"] = parent
        values["model.config.teacher_forcing_dcm_teacher_load_from.load_path"] = f"{parent}/model"
        values["model.config.teacher_forcing_dcm_teacher_load_from.credentials"] = ""
    else:
        values["checkpoint.load_path"] = ""
        values["model.config.student_load_from.load_path"] = f"{checkpoint_root('STAGE2_CKPT_ROOT')}/model"
        values["model.config.student_load_from.credentials"] = ""
        values["model.config.teacher_load_from.load_path"] = f"{checkpoint_root('STAGE0_CKPT_ROOT')}/model"
        values["model.config.teacher_load_from.credentials"] = ""

    # JSON strings preserve spaces and Hydra punctuation when each line is
    # forwarded as one argv item; no shell evaluation or environment injection.
    return [
        f"--config={LOCAL_CONFIG}",
        "--",
        f"experiment={EXPERIMENTS[stage]}",
        *(f"++{key}={json.dumps(value)}" for key, value in values.items()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    try:
        print("\n".join(training_arguments(args.stage)))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Local Transfer argument binding failed: {error}\n")


if __name__ == "__main__":
    main()
