# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""model — model backends behind narrow capability protocols (DESIGN §6, §1.4).

`FlowMatchingPolicy` and `VLMPolicy` (`protocol`) are the contracts; backends register builders in
`registry`. Importing this package wires the built-in `cosmos3` OmniMoT adapter. Concrete causal
VLM families stay in their explicit submodules so importing a protocol does not eagerly import a
model stack. The toy `FlowMatchingPolicy` used by tests is constructed directly from
`tests/helpers/toy_model.py`, not registered as a backend.
"""

from cosmos_framework.posttrain.model import (
    cosmos3,  # noqa: F401  (registers "cosmos3"; no native policy imports at load time)
)
from cosmos_framework.posttrain.model.protocol import (
    EMAFlowMatchingPolicy,
    FlowCondition,
    FlowConditionMask,
    FlowMatchingPolicy,
    PreferenceFlowPolicy,
    VLMPolicy,
    VLMPolicyOutput,
)
from cosmos_framework.posttrain.model.registry import build_model, register_model
from cosmos_framework.posttrain.model.value import DenseValueModel

__all__ = [
    "EMAFlowMatchingPolicy",
    "DenseValueModel",
    "FlowCondition",
    "FlowConditionMask",
    "FlowMatchingPolicy",
    "PreferenceFlowPolicy",
    "VLMPolicy",
    "VLMPolicyOutput",
    "build_model",
    "register_model",
]
