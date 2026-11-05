# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Control and history messages for causal action-conditioned inference."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StreamedActionChunk:
    actions: torch.Tensor
    domain_ids: torch.Tensor


@dataclass(frozen=True)
class ARHistorySnapshot:
    next_frame_idx: int
    conditional: tuple[tuple[int, ...], ...]
    unconditional: tuple[tuple[int, ...], ...]
