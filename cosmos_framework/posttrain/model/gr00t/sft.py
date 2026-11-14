# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Full GR00T-N1.7 adapter SFT; the existing action-head-only RL seat is unchanged."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cosmos_framework.posttrain.model.sft_source import (
    register_flow_sft_gradient_window,
    register_flow_sft_packer,
    register_native_sft_source,
)

BUILDER = "cosmos_framework.posttrain.model.gr00t.sft.build_gr00t_sft_policy"
REFERENCE_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


@dataclass
class Gr00tSFTModelConfig:
    backbone_path: str
    activation_checkpointing: str = "full"
    embodiment: str = "libero_sim"
    action_convention: str = "libero_environment"

    def __post_init__(self):
        if self.activation_checkpointing not in ("none", "full"):
            raise ValueError("GR00T SFT activation_checkpointing must be none|full")
        if self.embodiment != "libero_sim" or self.action_convention != "libero_environment":
            raise ValueError("GR00T SFT currently binds LIBERO environment actions and libero_sim statistics")
        if not self.backbone_path:
            raise ValueError("GR00T requires its exact local backbone config/processor snapshot")


def require_runtime():
    import transformers

    if transformers.__version__ != "4.57.6":
        raise ValueError("GR00T adapter SFT requires the pinned Isaac source and Transformers 4.57.6")


@register_native_sft_source(BUILDER)
def resolve_source(spec):
    from cosmos_framework.utils.lazy_config import instantiate

    config = instantiate(spec.model_config)
    config.__post_init__()
    root = Path(spec.model_id)
    payload = json.loads((root / "config.json").read_text())
    if payload.get("model_type") != "Gr00tN1d7":
        raise ValueError("GR00T adapter SFT binds Gr00tN1d7, not other GR00T architectures")
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        if not (root / name).is_file():
            raise ValueError(f"incomplete GR00T checkpoint: {name} is required")
    if not (Path(config.backbone_path) / "config.json").is_file():
        raise ValueError(
            "stage the exact GR00T backbone config/tokenizer/processor using authorized Hub access "
            "or posttrain.scripts.prepare_gr00t_backbone's verified identical public metadata; "
            f"checkpoint requires {payload.get('model_name', 'nvidia/Cosmos-Reason2-2B')}; "
            "another Qwen checkpoint is not an interchangeable dependency"
        )
    return spec.model_id, False


class Gr00tSFTPacker:
    def __init__(self, model_id, config, *, training_prefix, action_dim):
        from cosmos_framework.posttrain.model.gr00t.policy import load_processor

        require_runtime()
        if action_dim != 7 or not 1 <= training_prefix <= 16:
            raise ValueError("GR00T LIBERO SFT requires action_dim=7 and training_prefix in [1,16]")
        self.config, self.model_id = config, model_id
        self.training_prefix, self.action_dim = training_prefix, action_dim
        self.processor = load_processor(model_id, model_name=config.backbone_path)
        # Deterministic checkpoint-owned eval transforms for both train/validation;
        # model-owned diffusion/state dropout remains active during training.
        self.processor.eval()
        self.processor.state_dropout_prob = 0.0
        self.processor.collator.processor.tokenizer.padding_side = "left"

    def validation_semantics_identity(self):
        from cosmos_framework.posttrain.model.gr00t.sft_policy import source_identity

        return {
            "processor": source_identity(self.model_id, self.config)["processor"],
            "profile": "gr00t-libero-offline-v1",
            "training_prefix": self.training_prefix,
            "action_dim": self.action_dim,
            "state": "xyz-axisangle-two-finger-qpos",
            "images": "prepared-upright-RGB-checkpoint-eval-crop-resize",
            "action_convention": self.config.action_convention,
            "gripper_to_native": "(1-environment_gripper)/2",
            "augmentation": "checkpoint-eval-image-transform; processor-state-dropout-disabled",
        }

    def controlled_probe(self, batch, seed):
        return {**batch, "control_seed": seed}

    def __call__(self, trajectories):
        import numpy as np
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType, VLAStepData

        from cosmos_framework.posttrain.data.flow_media import open_flow_image
        from cosmos_framework.posttrain.model.gr00t.packer import action_mask_for

        features = []
        keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        for trajectory in trajectories:
            for step in trajectory.transitions:
                sample = step.obs.representations["flow_sample"]
                obs = sample["observation"]
                state = np.asarray(obs["state"], dtype=np.float32)
                action = np.asarray(sample["actions"], dtype=np.float32)
                if state.shape != (8,) or action.ndim != 2 or action.shape[1] != 7:
                    raise ValueError(
                        "GR00T LIBERO requires state xyz/axis-angle/finger-qpos (8) and physical actions (7)"
                    )
                if action.shape[0] < self.training_prefix or action.shape[0] > 16:
                    raise ValueError("LIBERO action window does not fit the checkpoint's 16-step embodiment")
                if not np.isfinite(state).all() or not np.isfinite(action).all() or np.abs(action[:, -1]).max() > 1:
                    raise ValueError("GR00T demonstration requires finite values and environment gripper in [-1,1]")
                action = action.copy()
                action[:, -1] = (1 - action[:, -1]) / 2
                images = {}
                for camera in ("image", "wrist_image"):
                    with open_flow_image(sample["media"][camera]) as image:
                        images[camera] = [np.asarray(image.convert("RGB")).copy()]
                native = VLAStepData(
                    images=images,
                    states={key: state[None, i : i + 1] if i < 6 else state[None, 6:] for i, key in enumerate(keys)},
                    actions={key: action[:, i : i + 1] for i, key in enumerate(keys)},
                    text=obs["prompt"],
                    embodiment=EmbodimentTag.resolve("libero_sim"),
                )
                item = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": native}])
                item["action_mask"] = action_mask_for(
                    item["action"], action_dim=7, training_prefix=self.training_prefix
                ) * float(step.algo_extra.get("validation_weight", 1.0))
                item["vlm_content"].pop("conversation", None)
                features.append(item)
        inputs = dict(self.processor.collator(features)["inputs"])
        return {"inputs": inputs, "actions": inputs["action"], "action_mask": inputs["action_mask"]}


@register_flow_sft_packer(BUILDER)
def build_packer(spec, rl):
    from cosmos_framework.utils.lazy_config import instantiate

    return Gr00tSFTPacker(
        spec.model_id,
        instantiate(spec.model_config),
        training_prefix=rl.flow_training_prefix,
        action_dim=rl.flow_action_dim,
    )


def build_gr00t_sft_policy(model_id, config, pdims=None, **kwargs):
    from cosmos_framework.posttrain.model.gr00t.sft_policy import build_policy

    return build_policy(model_id, config, pdims, **kwargs)


@register_flow_sft_gradient_window(BUILDER)
def build_gradient_window(policy, optimizer, rl):
    from cosmos_framework.posttrain.model.gr00t.sft_policy import build_gradient_window as install

    return install(policy, optimizer, rl)
