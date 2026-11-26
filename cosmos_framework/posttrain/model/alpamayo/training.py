# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""PhysicalAI trajectory data and native driving metrics for action_sft."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlpamayoDataConfig:
    train_path: str
    validation_path: str


class AlpamayoDataset:
    def __init__(self, path):
        from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset
        from cosmos_framework.posttrain.model.alpamayo.sft import physicalai_frames

        self.dataset = FlowSFTDataset(path)
        self.identity = {"format": "physicalai-alpamayo-flow-v1", "sha256": self.dataset.fingerprint}
        self.groups = self.dataset.episodes
        self.ids = self.dataset.ids
        self.media_hashes = self.dataset.media_hashes
        for sample in self.dataset:
            physicalai_frames(sample)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from cosmos_framework.posttrain.schema.trajectory import ObsBundle, Trajectory, Transition

        row = self.dataset[index]
        return Trajectory(
            id=row["id"],
            session=row["episode_id"],
            group_id=row["episode_id"],
            transitions=[
                Transition(obs=ObsBundle(id=row["id"], representations={"flow_sample": row}), targets=row["actions"])
            ],
        )


def build_data(config, split, policy):
    from cosmos_framework.utils.lazy_config import instantiate

    data = instantiate(config.data)
    if split not in {"train", "validation"}:
        raise ValueError("Alpamayo action data split must be train or validation")
    return AlpamayoDataset(getattr(data, f"{split}_path"))


def build_objective(policy, config):
    from cosmos_framework.posttrain.algorithm.flow_bc import make_objective

    if config.rl.micro_batch_size != 1:
        raise ValueError("Alpamayo native trajectory flow currently requires micro_batch_size=1")
    objective = make_objective(policy.packer, training_prefix=64, action_dim=2)
    objective.identity = {**objective.identity, "task": policy.packer.validation_semantics_identity()}
    return objective


def trajectory_metrics(predicted, expected):
    """Equal physical-trajectory weighting; planar distances in meters at native dt=.1s."""
    import torch

    if predicted.shape != expected.shape or predicted.ndim != 3 or predicted.shape[1:] != (64, 3):
        raise ValueError("Alpamayo open-loop metrics require paired Bx64x3 vehicle-frame trajectories")
    if not torch.isfinite(predicted).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite Alpamayo trajectory metric input")
    error = torch.linalg.vector_norm(predicted[..., :2].float() - expected[..., :2].float(), dim=-1)
    result = {}
    for steps, label in ((10, "1s"), (30, "3s"), (64, "6p4s")):
        result[f"open_loop_ade_{label}_meters"] = {
            "sum": float(error[:, :steps].mean(-1).sum()),
            "count": error.shape[0],
        }
        result[f"open_loop_fde_{label}_meters"] = {"sum": float(error[:, steps - 1].sum()), "count": error.shape[0]}
    return result


def evaluate_trajectories(policy, batch, config):
    import torch

    from cosmos_framework.posttrain.model.alpamayo.policy import sample_trajectories

    predicted = sample_trajectories(policy, policy.packer(batch))
    expected = torch.tensor(
        [step.obs.representations["flow_sample"]["actions"] for trajectory in batch for step in trajectory.transitions],
        device=predicted.device,
        dtype=torch.float32,
    )
    return trajectory_metrics(predicted, expected)
