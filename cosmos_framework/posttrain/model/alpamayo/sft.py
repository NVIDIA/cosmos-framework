# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Alpamayo-R1 trajectory supervision through the Cosmos SFT lifecycle.

The public NVlabs model owns trajectory tokenization, action-space conversion,
noise/time sampling and flow loss. No AlpaGym runtime or external trainer is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cosmos_framework.posttrain.model.sft_source import register_flow_sft_packer, register_native_sft_source

BUILDER = "cosmos_framework.posttrain.model.alpamayo.sft.build_alpamayo_sft_policy"
REFERENCE_REVISION = "11a0e01c13a5622377c45ee37d653351453ec43b"
PHYSICALAI_PROFILE = "physicalai_four_camera_v1"
PHYSICALAI_CAMERAS = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)


@dataclass
class AlpamayoSFTModelConfig:
    processor_path: str
    activation_checkpointing: str = "vision"
    min_pixels: int = 163840
    max_pixels: int = 196608
    observation_profile: str = "front_frames_v1"

    def __post_init__(self):
        if self.observation_profile not in ("front_frames_v1", PHYSICALAI_PROFILE):
            raise ValueError("Alpamayo observation_profile requires front_frames_v1 or physicalai_four_camera_v1")
        if self.activation_checkpointing not in ("none", "vision", "full"):
            raise ValueError("Alpamayo SFT supports activation_checkpointing=none|vision|full")
        if not self.processor_path or not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("Alpamayo SFT requires a local processor and positive ordered pixel bounds")


def require_runtime():
    import transformers

    if transformers.__version__ != "4.57.6":
        raise ValueError(
            "Alpamayo SFT binds Transformers 4.57.6 and the pinned public alpamayo-r1 source; use the robotics runtime"
        )


@register_native_sft_source(BUILDER)
def resolve_source(spec):
    from cosmos_framework.utils.lazy_config import instantiate

    cfg = instantiate(spec.model_config)
    cfg.__post_init__()
    if json.loads((Path(spec.model_id) / "config.json").read_text()).get("model_type") != "alpamayo_r1":
        raise ValueError("Alpamayo SFT currently binds the R1 checkpoint architecture")
    if not (Path(cfg.processor_path) / "config.json").is_file():
        raise ValueError("stage the pinned public Qwen3-VL-8B processor/config required by Alpamayo-R1")
    return spec.model_id, False


def source_config(model_id, config):
    require_runtime()
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1Config

    metadata = json.loads((Path(model_id) / "config.json").read_text())
    expected_ids = metadata.get("traj_token_ids")
    expected_vocab = metadata.get("vocab_size")
    metadata.update(
        vlm_name_or_path=config.processor_path,
        attn_implementation="sdpa",
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    result = AlpamayoR1Config(**metadata)
    if result.traj_token_ids != expected_ids or result.vocab_size != expected_vocab:
        raise ValueError(
            "Alpamayo processor produces different trajectory token IDs/vocabulary than the base checkpoint"
        )
    return result


class AlpamayoSFTPacker:
    def __init__(self, model_id, config, *, training_prefix, action_dim):
        import hydra.utils

        if training_prefix != 64 or action_dim != 2:
            raise ValueError("Alpamayo R1 SFT supervises all 64 acceleration/curvature waypoints")
        self.config = config
        self.model_config = source_config(model_id, config)
        self.processor = self.model_config._build_processor()
        self.action_space = hydra.utils.instantiate(self.model_config.action_space_cfg)

    def validation_semantics_identity(self):
        from cosmos_framework.posttrain.checkpoint.adapters import snapshot_identity

        return {
            "processor": snapshot_identity(self.config.processor_path, weights=False),
            "profile": "alpamayo-r1-trajectory-only-v1",
            "history_steps": 16,
            "future_steps": 64,
            "dt": 0.1,
            "min_pixels": self.config.min_pixels,
            "max_pixels": self.config.max_pixels,
            "prompt": "official-driving-system-history-with-empty-cot-and-future-start",
            "reasoning_supervision": False,
            "image_order": "ordered_front_camera_frames"
            if self.config.observation_profile == "front_frames_v1"
            else "camera_major_then_timestamp",
            **(
                {
                    "observation_profile": PHYSICALAI_PROFILE,
                    "camera_order": list(PHYSICALAI_CAMERAS),
                    "camera_indices": [0, 1, 2, 6],
                    "frames_per_camera": 4,
                    "coordinate_frame": "vehicle_at_t0",
                    "calibration": "source_native_camera_intrinsics_and_sensor_extrinsics",
                }
                if self.config.observation_profile == PHYSICALAI_PROFILE
                else {}
            ),
            "action_space": self.model_config.action_space_cfg,
        }

    def __call__(self, trajectories):
        import numpy as np
        import torch
        from alpamayo_r1.helper import create_message
        from PIL import Image

        rows = [t for trajectory in trajectories for t in trajectory.transitions]
        # A single physical demonstration per rank avoids an unqualified padding
        # and mutable-cache batch contract. Data parallelism remains independent.
        if len(rows) != 1:
            raise ValueError("Alpamayo SFT currently requires micro/validation batch_size=1 per rank")
        step = rows[0]
        sample = step.obs.representations["flow_sample"]
        obs = sample["observation"]
        if self.config.observation_profile == PHYSICALAI_PROFILE:
            frames = physicalai_frames(sample)
        else:
            frames = sample["media"].get("front")
            if not isinstance(frames, list) or not frames:
                raise ValueError("Alpamayo demonstrations require ordered real front-camera frames")
        images = []
        for path in frames:
            with Image.open(path) as image:
                images.append(torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1))
        messages = create_message(torch.stack(images))
        messages[-1]["content"][0]["text"] = "<|cot_start|><|cot_end|><|traj_future_start|>"
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        history = {
            name: torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
            for name in ("ego_history_xyz", "ego_history_rot")
        }
        future = torch.tensor(sample["actions"], dtype=torch.float32).unsqueeze(0)
        future_rot = torch.tensor(obs["ego_future_rot"], dtype=torch.float32).unsqueeze(0)
        if (
            history["ego_history_xyz"].shape != (1, 16, 3)
            or history["ego_history_rot"].shape != (1, 16, 3, 3)
            or future.shape != (1, 64, 3)
            or future_rot.shape != (1, 64, 3, 3)
        ):
            raise ValueError("Alpamayo requires 16 history and 64 future vehicle-frame poses at dt=0.1")
        if not all(torch.isfinite(x).all() for x in [*history.values(), future, future_rot]):
            raise ValueError("Alpamayo demonstration poses must be finite")
        actions = self.action_space.traj_to_action(
            history["ego_history_xyz"], history["ego_history_rot"], future, future_rot
        ).float()
        mask = torch.full_like(actions, float(step.algo_extra.get("validation_weight", 1.0)))
        return {
            "tokenized": dict(tokenized),
            "history": {name: value.unsqueeze(1) for name, value in history.items()},
            "actions": actions,
            "action_mask": mask,
        }


def physicalai_frames(sample):
    """Validate the new native four-camera contract without changing legacy rows."""
    import numpy as np

    obs, media = sample["observation"], sample["media"]
    if (
        set(media) != set(PHYSICALAI_CAMERAS)
        or obs.get("camera_order") != list(PHYSICALAI_CAMERAS)
        or obs.get("camera_indices") != [0, 1, 2, 6]
    ):
        raise ValueError("PhysicalAI Alpamayo requires exactly camera indices [0,1,2,6] in native order")
    groups = [media[name] for name in PHYSICALAI_CAMERAS]
    if any(not isinstance(g, list) or len(g) != 4 or len(set(g)) != 4 for g in groups):
        raise ValueError("PhysicalAI Alpamayo requires four distinct ordered frames per camera")
    timestamps = np.asarray(obs.get("frame_timestamps_us"))
    t0 = obs.get("t0_us")
    if (
        timestamps.shape != (4, 4)
        or not np.issubdtype(timestamps.dtype, np.integer)
        or np.any(np.diff(timestamps, axis=1) < 0)
        or np.any(timestamps < 0)
        or type(t0) is not int
    ):
        raise ValueError("PhysicalAI camera timestamps require a nondecreasing nonnegative sequence and integer t0_us")
    if obs.get("coordinate_frame") != "vehicle_at_t0" or obs.get("trajectory_dt_seconds") != 0.1:
        raise ValueError("PhysicalAI trajectories require the native vehicle frame at t0 and dt=0.1 seconds")
    calibration = obs.get("calibration")
    if (
        not isinstance(calibration, dict)
        or set(calibration) != {"camera_intrinsics", "sensor_extrinsics"}
        or any(not isinstance(v, dict) or not v for v in calibration.values())
    ):
        raise ValueError("PhysicalAI observations require source camera intrinsic and extrinsic calibration records")
    intrinsics = {"width", "height", "cx", "cy", *[f"{side}_poly_{i}" for side in ("bw", "fw") for i in range(5)]}
    extrinsics = {"qx", "qy", "qz", "qw", "x", "y", "z"}
    for camera in PHYSICALAI_CAMERAS:
        for kind, fields in (("camera_intrinsics", intrinsics), ("sensor_extrinsics", extrinsics)):
            value = calibration[kind].get(camera)
            if not isinstance(value, dict) or not fields.issubset(value):
                raise ValueError(f"PhysicalAI calibration missing {kind} fields for {camera}")
            numbers = np.asarray([value[key] for key in sorted(fields)], dtype=np.float64)
            if not np.isfinite(numbers).all():
                raise ValueError(f"PhysicalAI calibration is nonfinite: {kind}/{camera}")
        intrinsic = calibration["camera_intrinsics"][camera]
        if any(intrinsic[key] <= 0 or int(intrinsic[key]) != intrinsic[key] for key in ("width", "height")):
            raise ValueError(f"PhysicalAI camera dimensions must be positive integers: {camera}")
        extrinsic = calibration["sensor_extrinsics"][camera]
        if not np.isclose(sum(extrinsic[key] ** 2 for key in ("qx", "qy", "qz", "qw")), 1.0, atol=1e-5, rtol=0):
            raise ValueError(f"PhysicalAI calibration quaternion must have unit norm: {camera}")
    requested = t0 + np.arange(-3, 1) * 100_000
    if np.any(timestamps > requested[None, :]):
        raise ValueError("PhysicalAI frames must precede their native requested timestamps")
    return [frame for group in groups for frame in group]


@register_flow_sft_packer(BUILDER)
def build_packer(spec, rl):
    from cosmos_framework.utils.lazy_config import instantiate

    return AlpamayoSFTPacker(
        spec.model_id,
        instantiate(spec.model_config),
        training_prefix=rl.flow_training_prefix,
        action_dim=rl.flow_action_dim,
    )


def build_alpamayo_sft_policy(model_id, config, pdims=None, **kwargs):
    from cosmos_framework.posttrain.model.alpamayo.policy import build_policy

    return build_policy(model_id, config, pdims, **kwargs)
