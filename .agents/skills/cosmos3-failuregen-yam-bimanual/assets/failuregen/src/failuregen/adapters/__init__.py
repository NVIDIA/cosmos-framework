# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Forward-dynamics model adapters. The base protocol imports nothing heavy; ``cosmos`` does."""

from .base import ForwardDynamicsModel, load_adapter

__all__ = ["ForwardDynamicsModel", "load_adapter"]
