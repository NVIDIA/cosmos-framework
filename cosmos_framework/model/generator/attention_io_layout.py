# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Attention-boundary layouts supported by interactive models."""

from typing import Literal

AttentionIOLayout = Literal["sequence_sharded", "replicated"]
