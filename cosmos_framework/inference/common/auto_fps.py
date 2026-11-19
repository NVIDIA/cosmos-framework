# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Repository-owned camera FPS policy for Auto inference; no checkpoint edits required."""

from typing import Any

from omegaconf import DictConfig, open_dict

from cosmos_framework.inference.common.multiview_captions import JOINT_CAM_LIDAR_TASK, exported_caption_task


def auto_camera_fps(metadata: Any, *, repository: str | None = None) -> int | None:
    """Identify Auto by its existing task metadata, or an exact official HF repository ID.

    Both released Auto snapshots record the joint task, including when used for RGB-only
    generation. The plain multiview task covers camera-only Auto exports. A local directory
    name, view count, LiDAR tokenizer or base_fps alone does not identify an Auto checkpoint.
    This policy lives in inference code, not in the immutable checkpoint's config.json.
    """
    if exported_caption_task(metadata) in {JOINT_CAM_LIDAR_TASK, "av_multiview_transfer"} or repository in {
        "nvidia/Cosmos3-Nano-Transfer-Auto",
        "nvidia/Cosmos3-Nano-Sim-Auto",
    }:
        return 30
    return None


def attach_auto_camera_fps(model_config: Any, camera_fps: int | None) -> None:
    """Record the inference-only Auto camera clock on the runtime model config.

    Published checkpoints load into a plain config object, so a new attribute can simply be set. A config exported
    from a training run is a struct-mode OmegaConf node typed by the training config class, which refuses unknown
    keys (``Key '_auto_camera_fps' not in 'SelfForcingDMDConfig'``); the key is added with struct mode lifted for
    this assignment only. The checkpoint files are never modified.
    """
    if isinstance(model_config, DictConfig):
        with open_dict(model_config):
            model_config["_auto_camera_fps"] = camera_fps
        return
    setattr(model_config, "_auto_camera_fps", camera_fps)
