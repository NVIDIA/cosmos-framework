# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Thin transformers config for the `cosmos3_omni` model_type."""

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


class Cosmos3OmniConfig(Qwen3VLConfig):
    model_type = "cosmos3_omni"

    def __post_init__(self, **kwargs):
        kwargs.pop("model", None)
        self.allow_patterns_overrides = kwargs.pop("allow_patterns_overrides", None)

        super().__post_init__(**kwargs)
