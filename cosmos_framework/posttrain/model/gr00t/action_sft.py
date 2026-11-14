# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T action SFT factories for the shared supervised-task entrypoint.

LIBERO preparation follows retained build 48f6aa082b38a598a1fe59735e36a543f8834a02:
checkpoint-owned transforms/statistics, upright RGB cameras, native gripper conversion.
The original action flow objective and four-step action sampler are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Gr00tActionDataConfig:
    train_path: str = ""
    validation_path: str = ""
    action_convention: str = "libero_environment"


class Gr00tActionPacker:
    def __init__(self, model_id, config, *, training_prefix, action_dim):
        from cosmos_framework.posttrain.model.gr00t.policy import load_processor

        if action_dim != 7 or not 1 <= training_prefix <= 16:
            raise ValueError("GR00T LIBERO SFT requires action_dim=7 and training_prefix in [1,16]")
        self.config, self.model_id = config, model_id
        self.training_prefix, self.action_dim = training_prefix, action_dim
        self.processor = load_processor(model_id, model_name=config.backbone_adaptation.backbone_path)
        # Deterministic checkpoint-owned eval transforms for both train/validation;
        # model-owned diffusion/state dropout remains active during training.
        self.processor.eval()
        self.processor.state_dropout_prob = 0.0
        self.processor.collator.processor.tokenizer.padding_side = "left"

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
        return inputs


def build_policy(config, *, device, output):
    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy
    from cosmos_framework.utils.lazy_config import instantiate

    model = instantiate(config.model)
    model.__post_init__()
    if model.embodiment_tag != "libero_sim" or model.action_dim != 7:
        raise ValueError("GR00T action SFT initially binds only the LIBERO 7-axis embodiment")
    policy = Gr00tN1d7TrainPolicy(model, device=device, save_cfg_dir=Path(output) / "model-config")
    policy.packer = Gr00tActionPacker(
        model.checkpoint_path, model, training_prefix=model.training_prefix, action_dim=model.action_dim
    )
    return policy


class Gr00tActionDataset:
    def __init__(self, path):
        from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset

        self.rows = FlowSFTDataset(path)
        self.identity = {"format": "flow-sft-v1", "sha256": self.rows.fingerprint, "family": "gr00t-libero"}
        self.groups, self.ids, self.media_hashes = self.rows.episodes, self.rows.ids, self.rows.media_hashes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

        row = self.rows[index]
        return Trajectory(
            id=row["id"],
            status=Status.COMPLETED,
            transitions=[Transition(obs=ObsBundle(representations={"flow_sample": row}), action=row["actions"])],
        )


def build_dataset(config, split, policy):
    from cosmos_framework.utils.lazy_config import instantiate

    data = instantiate(config.data)
    if data.action_convention != "libero_environment":
        raise ValueError("GR00T data must record LIBERO environment actions")
    if split not in {"train", "validation"}:
        raise ValueError("unknown GR00T action SFT data split")
    return Gr00tActionDataset(getattr(data, f"{split}_path"))


def build_objective(policy, config):
    from cosmos_framework.posttrain.algorithm.flow_bc import make_objective

    return make_objective(
        policy.packer, training_prefix=policy.config.training_prefix, action_dim=policy.config.action_dim
    )


def evaluate_actions(policy, batch, config):
    """Four-step sampled actions, distinct from teacher-forced velocity loss or rollout success."""
    import numpy as np

    inputs = policy.packer(batch)
    expected = inputs["action"]
    mask = inputs["action_mask"]
    # Upstream interprets an action input as RTC conditioning. Never condition an evaluation
    # prediction on its demonstration's ground-truth action chunk.
    inference_inputs = {key: value for key, value in inputs.items() if key not in {"action", "action_mask"}}
    prediction = policy.predict_actions(inference_inputs)["action_pred"]
    expected, mask = expected.to(prediction.device), mask.to(prediction.device)
    squared = ((prediction.float() - expected.float()).square() * mask.float()).sum()
    result = {"normalized_action_mse": {"sum": float(squared), "count": float(mask.sum())}}
    tag = policy.config.resolve_embodiment_tag()
    decoded = policy.packer.processor.decode_action(prediction.float().cpu().numpy(), tag)
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    action = np.concatenate([decoded[key] for key in keys], axis=-1)
    action[..., -1] = 1 - 2 * action[..., -1]
    prefix = policy.config.training_prefix
    truth = np.stack([np.asarray(t.transitions[0].action, dtype=np.float32)[:prefix] for t in batch])
    error = np.abs(action[:, :prefix] - truth)
    result["action_mae_environment_units"] = {"sum": float(error.sum()), "count": int(error.size)}
    for index, key in enumerate(keys):
        result[f"action_mae_{key}_environment_units"] = {
            "sum": float(error[..., index].sum()),
            "count": int(error[..., index].size),
        }
    return result


def load_artifact(config, artifact, *, device, output):
    """Fresh-process native action policy loading, including all trained projector/head tensors."""
    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    policy = build_policy(config, device=device, output=output)
    load_adapter_weights(artifact, policy.family_state.codec, manager=CheckpointManager(checkpoint_format="dcp"))
    policy.net.eval()
    return policy
