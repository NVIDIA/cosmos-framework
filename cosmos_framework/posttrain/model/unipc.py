# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native Cosmos3 UniPC construction for online flow-policy adapters."""

from __future__ import annotations

from typing import Any


def build_unipc_sampler(policy: Any) -> Any:
    """Build native UniPC with base shift 1; rollout applies runtime shift once."""
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig

    model = policy.omnimot
    inference = model.config.rectified_flow_inference_config
    scheduler_type = str(inference.scheduler_type).lower()
    if scheduler_type != "unipc":
        raise ValueError(f"build_unipc_sampler needs scheduler_type='unipc', got {inference.scheduler_type!r}")
    config = UniPCSamplerConfig(
        num_train_timesteps=int(inference.num_train_timesteps),
        shift=1.0,
        use_dynamic_shifting=False,
    )
    return UniPCSampler(cfg=config, tensor_kwargs=model.tensor_kwargs)
