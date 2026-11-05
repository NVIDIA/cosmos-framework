# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small on-disk GR00T sources and a source-backed continuous policy for CPU lifecycles."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy
from cosmos_framework.posttrain.model.gr00t.source import Gr00tPPOSource, load_ppo_source
from cosmos_framework.posttrain.tests.helpers.ppo_continuous import GaussianPolicy


def write_source(directory: Path, *, sharded: bool = True) -> Path:
    directory.mkdir(parents=True)
    dependency = directory.parent / "processor-source"
    dependency.mkdir(exist_ok=True)
    for name, value in {
        "config.json": {"model_type": "qwen3_vl"},
        "tokenizer.json": {"version": "1.0", "vocabulary": ["robot", "cup"]},
        "preprocessor_config.json": {"size": 32},
    }.items():
        (dependency / name).write_text(json.dumps(value))
    for name, value in {
        "config.json": {
            "model_name": str(dependency),
            "action_horizon": 40,
            "max_action_dim": 132,
            "num_inference_timesteps": 4,
        },
        "processor_config.json": {
            "processor_kwargs": {
                "model_name": str(dependency),
                "modality_configs": {"libero_sim": {"action": {"delta_indices": list(range(16))}}},
            }
        },
        "statistics.json": {"scale": 1.0},
        "embodiment_id.json": {"libero_sim": 0},
    }.items():
        (directory / name).write_text(json.dumps(value))
    tensors = {"weight": torch.eye(2), "bias": torch.zeros(2)}
    if sharded:
        weight_map = {name: f"model-{index:05d}-of-00002.safetensors" for index, name in enumerate(tensors, 1)}
        for name, tensor in tensors.items():
            save_file({name: tensor}, str(directory / weight_map[name]))
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(tensors, str(directory / "model.safetensors"))
    return directory


def load_backbone(source: Gr00tPPOSource) -> torch.nn.Linear:
    directory = source.checkpoint
    if (directory / "model.safetensors").is_file():
        files = [directory / "model.safetensors"]
    else:
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        files = [directory / name for name in sorted(set(index["weight_map"].values()))]
    tensors = {key: value for path in files for key, value in load_file(str(path)).items()}
    with torch.random.fork_rng(devices=[]):
        backbone = torch.nn.Linear(2, 2).requires_grad_(False)
    backbone.load_state_dict(tensors)
    return backbone


class SourceGaussianPolicy(GaussianPolicy):
    """Use production source binding and policy identity with real small file-backed weights."""

    ppo_identity = Gr00tN1d7TrainPolicy.ppo_identity

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.config = make_model_config(str(checkpoint))

        def load(source: Gr00tPPOSource) -> tuple[torch.nn.Linear, dict]:
            return load_backbone(source), json.loads((source.checkpoint / "statistics.json").read_text())

        (self.backbone, self.statistics), self._ppo_source_identity = load_ppo_source(self.config, load)

    def infer_batch(self, observations: list[dict], **kwargs) -> list[dict]:
        values = torch.tensor([[row["position"], 1.0] for row in observations])
        positions = self.backbone(values)[:, 0] * self.statistics["scale"]
        return super().infer_batch([{"position": float(value)} for value in positions], **kwargs)
