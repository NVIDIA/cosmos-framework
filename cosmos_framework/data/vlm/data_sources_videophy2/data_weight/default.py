# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Default weight mix for the VideoPhy-2 dataset (single source)."""

from cosmos_framework.data.vlm.data_sources_videophy2.videophy2 import DATAINFO, url_to_category

# Single-source SFT mix — used for both data_train and data_val. The dataloader
# registrar picks the right split via the LocalDataSource.manifest_path mapping.
data_weight_train = {"videophy2_train": 1}
data_weight_val = {"videophy2_val": 1}

__all__ = ["DATAINFO", "url_to_category", "data_weight_train", "data_weight_val"]
