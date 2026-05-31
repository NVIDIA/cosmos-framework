# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""VLM freeze config (read by ``vlm_model._apply_freeze_config``)."""

import attrs

from cosmos_framework.utils.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class VLMFreezeConfig:
    """Selects which parts of a VLM stay trainable.

    Applied at model construction, before the optimizer is built; the optimizer
    only sees the resulting ``requires_grad`` state.
    """

    # Named freeze flags. Supported architectures: Qwen2.5-VL,
    # Qwen3-VL (dense + MoE), InternVL3_5.
    freeze_vision_encoder: bool = False
    freeze_mm_projector: bool = False
    freeze_llm: bool = False

    # Regex-based freeze (mutually exclusive with each other).
    # trainable_params: whitelist — only matching params are trainable.
    # frozen_params:    blacklist — matching params get frozen.
    trainable_params: list[str] | None = None
    frozen_params: list[str] | None = None

    def __attrs_post_init__(self) -> None:
        if self.trainable_params is not None and self.frozen_params is not None:
            raise ValueError("VLMFreezeConfig: set at most one of trainable_params or frozen_params, not both.")
