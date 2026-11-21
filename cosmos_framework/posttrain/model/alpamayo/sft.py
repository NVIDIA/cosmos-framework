# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Alpamayo-R1 trajectory supervision through the Cosmos SFT lifecycle.

The public NVlabs model owns trajectory tokenization, action-space conversion,
noise/time sampling and flow loss. No AlpaGym runtime or external trainer is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
class AlpamayoBackboneAdaptation:
    """Mutable native recipe fields; the installer freezes a validated copy."""

    mode: str = "nf4_lora"
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self):
        from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig

        FamilyBackboneConfig(**vars(self))


@dataclass
class AlpamayoSFTModelConfig:
    checkpoint_path: str
    processor_path: str
    backbone_adaptation: AlpamayoBackboneAdaptation = field(default_factory=AlpamayoBackboneAdaptation)
    activation_checkpointing: str = "full"
    train_action_head: bool = True
    checkpoint_revision: str = "dd4a24cacefc9a6477a6dfc7354de2443401409d"
    processor_revision: str = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    inference_steps: int = 10
    min_pixels: int = 163840
    max_pixels: int = 196608
    observation_profile: str = PHYSICALAI_PROFILE

    def __post_init__(self):
        self.backbone_adaptation.__post_init__()
        if self.observation_profile != PHYSICALAI_PROFILE:
            raise ValueError("Alpamayo QLoRA requires the native PhysicalAI four-camera profile")
        if not self.checkpoint_path or self.checkpoint_revision != "dd4a24cacefc9a6477a6dfc7354de2443401409d":
            raise ValueError("Alpamayo QLoRA requires the pinned Alpamayo-R1-10B checkpoint")
        if self.processor_revision != "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b":
            raise ValueError("Alpamayo QLoRA requires the pinned Qwen3-VL-8B processor")
        if type(self.train_action_head) is not bool or (
            not self.train_action_head and self.backbone_adaptation.mode.endswith("_frozen")
        ):
            raise ValueError("frozen Alpamayo backbone requires train_action_head=True")
        if type(self.inference_steps) is not int or self.inference_steps != 10:
            raise ValueError("Alpamayo native open-loop evaluation binds ten Euler steps")
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
        from cosmos_framework.posttrain.model.alpamayo.source import snapshot_identity

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

        from cosmos_framework.posttrain.data.flow_media import open_flow_image

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
            with open_flow_image(path) as image:
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
        if not torch.allclose(
            history["ego_history_xyz"][:, -1], torch.zeros(1, 3), atol=1e-4, rtol=0
        ) or not torch.allclose(history["ego_history_rot"][:, -1], torch.eye(3).unsqueeze(0), atol=1e-4, rtol=0):
            raise ValueError("Alpamayo history must end at the vehicle-at-t0 origin and orientation")
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
