# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Interactive-only FlexAttention configuration."""

import attrs

from cosmos_framework.configs.base.defaults.flex_attention import (
    FlexAttentionConfig,
    FlexAttentionMaskConfig,
)


@attrs.define(slots=False)
class CausalFlexAttentionMaskConfig(FlexAttentionMaskConfig):
    """Keep interactive Flex mask defaults separate from replay policy."""


@attrs.define(slots=False)
class CausalFlexAttentionConfig(FlexAttentionConfig):
    """Interactive FlexAttention config; replay connectivity is backend-neutral."""

    mask: CausalFlexAttentionMaskConfig = CausalFlexAttentionMaskConfig()
