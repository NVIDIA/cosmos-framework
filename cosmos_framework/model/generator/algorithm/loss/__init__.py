# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Loss functions used by VFM (rectified flow) and VLM (next-token CE) training paths."""

__all__: list[str] = []

from cosmos_framework.model.generator.algorithm.loss.cross_entropy import cross_entropy_loss, weighted_cross_entropy_loss

__all__ += ["cross_entropy_loss", "weighted_cross_entropy_loss"]

from cosmos_framework.model.generator.algorithm.loss.load_balancing import compute_load_balancing_loss

__all__ += ["compute_load_balancing_loss"]

from cosmos_framework.model.generator.algorithm.loss.time_weight import TrainTimeWeight

__all__ += ["TrainTimeWeight"]

from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss

__all__ += ["compute_flow_matching_loss"]
