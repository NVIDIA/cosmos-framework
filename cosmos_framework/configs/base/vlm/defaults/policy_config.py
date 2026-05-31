# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Union

import attrs

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.ema import EMAConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.vlm import VLMConfig
from cosmos_framework.configs.base.vlm.freeze_config import VLMFreezeConfig


@attrs.define(slots=False)
class PolicyConfig:
    # VLM backbone identity, shared with OmniMoTModelConfig.vlm_config.
    backbone: VLMConfig = VLMConfig()
    # The maximum length for training, longer than this will be ignored for training stability
    model_max_length: int = 16000

    # The maximum length for video tokens, only applied to qwen model
    qwen_max_video_token_length: int = 8000

    # Extra model config
    lora: Union[str, None] = None
    enable_liger_kernel: bool = False
    trainable_map: Union[str, None] = None
    monkey_patch_for_text_only_data: bool = False

    # HF attention impl. Default "cosmos" routes through imaginaire.attention
    # (NATTEN/blackwell-fmha on GB200). Override to "flash_attention_2",
    # "sdpa", or "eager" for fallback.
    attn_implementation: str = "cosmos"


@attrs.define(slots=False)
class VLMModelConfig:
    """Config for VLM model."""

    # Training infrastructure: parallelism mesh, torch.compile, activation
    # checkpointing, and FSDP / dtype precision. Consumed by VLMModel at
    # construction time.
    parallelism: ParallelismConfig = ParallelismConfig()
    compile: CompileConfig = CompileConfig()
    activation_checkpointing: ActivationCheckpointingConfig = ActivationCheckpointingConfig()
    precision: str = "bfloat16"

    policy: PolicyConfig = PolicyConfig()
    # Applied at model construction, before the optimizer is built.
    freeze: VLMFreezeConfig = VLMFreezeConfig()
    ema: EMAConfig = EMAConfig(enabled=False)

    # Force deterministic kernels in Flash-Attention init (slower; required for
    # parity bit-exactness)
    deterministic: bool = False
