# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in BatchCollator implementations."""

from __future__ import annotations

import torch.utils.data

from cosmos_framework.data.vfm.dataflow.base import BatchCollator


class DefaultBatchCollator(BatchCollator):
    """Stacks samples with torch's default_collate — stock DataLoader behavior."""

    def collate(self, samples: list[dict]) -> dict:
        return torch.utils.data.default_collate(samples)
