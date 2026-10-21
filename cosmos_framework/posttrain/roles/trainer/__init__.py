# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Trainer engine bodies; algorithms arrive through the objective contract."""

from cosmos_framework.posttrain.roles.trainer.base import (
    BaseTrainer,
    NonFiniteUpdateError,
    ObjectiveValidatingTrainerProtocol,
    PartialUpdateError,
    TrainerProtocol,
)
from cosmos_framework.posttrain.roles.trainer.diffusion import CopyInventoryPolicy, DiffusionTrainer
from cosmos_framework.posttrain.roles.trainer.llm import AuxiliaryModelResidency, LLMTrainer
from cosmos_framework.posttrain.roles.trainer.local import LocalTrainer
from cosmos_framework.posttrain.roles.trainer.ppo import PPOPolicyTrainer

__all__ = [
    "AuxiliaryModelResidency",
    "BaseTrainer",
    "DiffusionTrainer",
    "CopyInventoryPolicy",
    "LLMTrainer",
    "LocalTrainer",
    "NonFiniteUpdateError",
    "ObjectiveValidatingTrainerProtocol",
    "PartialUpdateError",
    "PPOPolicyTrainer",
    "TrainerProtocol",
]
