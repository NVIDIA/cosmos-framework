# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Physical demonstrations over the existing resumable SFT cursor and padding."""

from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


class FlowSFTDatasetEnv(SFTDatasetEnv):
    def _trajectory(self, index, *, weight=None):
        row = self.dataset[index]
        transition = Transition(obs=ObsBundle(representations={"flow_sample": row}), action=row["actions"])
        if weight is not None:
            transition.algo_extra["validation_weight"] = weight
        return Trajectory(id=f"{self.source_fingerprint}:{index}", transitions=[transition], status=Status.COMPLETED)


def build_flow_sft_env(config, seed, **kwargs):
    if not config.path or config.factory or config.path.startswith("s3://"):
        raise ValueError("flow_bc currently requires a staged local demonstration JSONL")
    return FlowSFTDatasetEnv(FlowSFTDataset(config.path), seed=seed, shuffle=config.shuffle, **kwargs)
