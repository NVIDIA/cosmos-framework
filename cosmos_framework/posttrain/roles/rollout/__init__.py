# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""roles.rollout — rollout variants (DESIGN §1.2). `local` (reference) + `hf` +
`vllm` (real inference engine, #6); SGLang / TRT-LLM drop in as further siblings.

`vllm.py` imports the `vllm` package only inside `generate()` (call time), so importing
`VLLMRollout` here is safe without vLLM installed — the package is needed only to run it.
"""

from cosmos_framework.posttrain.roles.rollout.base import Rollout, ShardSpec
from cosmos_framework.posttrain.roles.rollout.hf import HFRollout
from cosmos_framework.posttrain.roles.rollout.local import LocalRollout
from cosmos_framework.posttrain.roles.rollout.ppo import PPOVLLMRollout
from cosmos_framework.posttrain.roles.rollout.validation import ValidationRolloutContext
from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout
from cosmos_framework.posttrain.roles.rollout.wfm import WFMOnlineRollout

__all__ = [
    "Rollout",
    "ShardSpec",
    "ValidationRolloutContext",
    "LocalRollout",
    "HFRollout",
    "VLLMRollout",
    "PPOVLLMRollout",
    "WFMOnlineRollout",
    "PolicyBody",
]


def __getattr__(name):
    # Lazy: cosmos3.py defers heavy i4/OSS policy imports to __init__/methods, but
    # keep this re-export lazy too so importing roles.rollout stays cheap.
    if name == "PolicyBody":
        from cosmos_framework.posttrain.roles.rollout import cosmos3

        return getattr(cosmos3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
