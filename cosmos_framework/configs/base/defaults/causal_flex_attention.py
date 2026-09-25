# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Interactive-only FlexAttention configuration."""

import attrs

from cosmos_framework.configs.base.defaults.multiview_attention import (
    MultiviewAttentionConfig,
    MultiviewAttentionMaskConfig,
)


@attrs.define(slots=False)
class CausalFlexAttentionMaskConfig(MultiviewAttentionMaskConfig):
    """Keep interactive Flex mask defaults separate from replay policy."""


@attrs.define(slots=False)
class CausalFlexAttentionConfig(MultiviewAttentionConfig):
    """Interactive FlexAttention config; replay connectivity is backend-neutral."""

    mask: CausalFlexAttentionMaskConfig = CausalFlexAttentionMaskConfig()
