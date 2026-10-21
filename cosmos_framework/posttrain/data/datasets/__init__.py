# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""data.datasets — concrete dataset implementations (storage/decode/snap/shuffle/
pack). The data layer (DESIGN §1.3): pipeline-agnostic sample producers, consumed by
the `roles/env/` source nodes. No `Trajectory`/role/placement semantics here."""

from cosmos_framework.posttrain.data.datasets.dpo_preference import DPOPreferenceDataset
from cosmos_framework.posttrain.data.datasets.nft_prompt import NFTPromptImageDataLoader

__all__ = ["DPOPreferenceDataset", "NFTPromptImageDataLoader"]
