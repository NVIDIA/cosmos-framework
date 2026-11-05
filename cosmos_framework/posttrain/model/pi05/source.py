# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Content identity and strict source inventory for the local RLinf OpenPI checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from cosmos_framework.posttrain.model.pi05.config import (
    RLINF_REVISION,
    SAMPLER_IDENTITY,
    SAMPLER_NOISE_LEVEL,
    Pi05ModelConfig,
)
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
from cosmos_framework.posttrain.utils.object_store import file_sha256


def _checkpoint_file(directory: Path) -> Path:
    weights = sorted(directory.glob("*.safetensors"))
    if [path.name for path in weights] != ["model.safetensors"]:
        raise ValueError("pi05.checkpoint_path requires exactly one model.safetensors file")
    return weights[0]


def model_identity(config: Pi05ModelConfig, source: dict[str, str]) -> dict:
    """Bind model arithmetic separately from relocatable checkpoint and processor bytes."""
    return {
        "model": "pi05-openpi-libero-v1",
        "reference": RLINF_REVISION,
        "sampler": SAMPLER_IDENTITY,
        "action_chunk": config.action_chunk,
        "denoise_steps": config.denoise_steps,
        "noise_level": SAMPLER_NOISE_LEVEL,
        "model_horizon": 10,
        "latent_dim": 32,
        "discrete_state_input": False,
        "precision": "openpi-selected-bf16-fp32-v1",
        "matmul_precision": "high",
        "value_initialization_seed": 42,
        "training_replay": "preprocessed-uint8-frozen-prefix-fixed-microbatch-v2",
        "source": dict(source),
    }


@dataclass(frozen=True)
class Pi05Source:
    checkpoint: Path
    tokenizer: Path
    normalization: Path

    @classmethod
    def resolve(cls, config: Pi05ModelConfig) -> Pi05Source:
        directory = Path(config.checkpoint_path).expanduser().resolve()
        result = cls(
            _checkpoint_file(directory),
            Path(config.tokenizer_path).expanduser().resolve(),
            Path(config.norm_stats_path).expanduser().resolve(),
        )
        for name in ("checkpoint", "tokenizer", "normalization"):
            if not getattr(result, name).is_file():
                raise FileNotFoundError(f"pi05 {name}: {getattr(result, name)}")
        return result

    def identity(self) -> dict[str, str]:
        if _checkpoint_file(self.checkpoint.parent) != self.checkpoint:
            raise ValueError("pi05 checkpoint inventory differs from its resolved source")
        manifest = {
            name: file_sha256(str(getattr(self, name))) for name in ("checkpoint", "tokenizer", "normalization")
        }
        return {
            "protocol": "pi05-openpi-source-v1",
            "sha256": canonical_json_sha256(manifest, context="pi05 source"),
        }


def load_model_weights(model: torch.nn.Module, source: Pi05Source) -> None:
    """Validate the unfiltered checkpoint inventory before the shared loader writes any tensor.

    Safetensors serializes one name of the tied PaliGemma embedding/head. A raw
    SFT source may omit the entire value head; a partially saved head is invalid.
    """
    import re

    import torch
    from safetensors import safe_open

    from cosmos_framework.posttrain.model.specs import LoadPlan
    from cosmos_framework.posttrain.model.weight_loader import load_weights

    state = model.state_dict()
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    with safe_open(str(source.checkpoint), framework="pt", device="cpu") as checkpoint:
        names = set(checkpoint.keys())
        unexpected = names - state.keys()
        if unexpected:
            raise ValueError(f"pi05 checkpoint has unexpected tensors: {sorted(unexpected)}")
        for name in sorted(names):
            shape = tuple(checkpoint.get_slice(name).get_shape())
            if shape != tuple(state[name].shape):
                raise ValueError(f"pi05 checkpoint {name}: shape {shape} differs from {tuple(state[name].shape)}")
        missing = state.keys() - names
        permitted: set[str] = set()
        for group in aliases.values():
            present = sorted(names.intersection(group))
            if present:
                permitted.update(set(group) - names)
                for name in present[1:]:
                    if not torch.equal(checkpoint.get_tensor(present[0]), checkpoint.get_tensor(name)):
                        raise ValueError(f"pi05 checkpoint tied tensors disagree: {present[0]!r}, {name!r}")
        value_names = {name for name in state if name.startswith("value_head.")}
        if not names.intersection(value_names):
            permitted.update(value_names)
        if missing != permitted:
            raise ValueError(f"pi05 checkpoint is incomplete: {sorted(missing - permitted)}")
    # No persistent views outlive loading; only exact absent aliases/head are exempt.
    del state
    load_weights(
        model,
        str(source.checkpoint),
        LoadPlan(skip_patterns=tuple(f"^{re.escape(name)}$" for name in sorted(permitted))),
    )
