# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Union

import attrs

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.ema import EMAConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.reasoner import SoundUnderstandingConfig, VLMConfig
from cosmos_framework.configs.base.reasoner.freeze_config import VLMFreezeConfig


@attrs.define(slots=False)
class PolicyConfig:
    # VLM backbone identity, shared with OmniMoTModelConfig.vlm_config.
    backbone: VLMConfig = VLMConfig()
    # The maximum length for training, longer than this will be ignored for training stability
    model_max_length: int = 16000

    # The maximum length for video tokens, only applied to qwen model
    qwen_max_video_token_length: int = 8000

    use_weighted_ce: bool = False
    # Controls the interpolation between per-token (0) and per-sample (1) loss:
    #   exponent=1 -> per-sample loss: every sample contributes equally to the global loss
    #   exponent=0 -> per-token loss: every token contributes equally to the global loss
    #   0 < exponent < 1 -> interpolation; e.g. exponent=0.5 gives square-root per-token loss (Qwen3-VL)
    weighted_ce_exponent: float = 1.0

    # Parameter-efficient fine-tuning.  These fields intentionally live on
    # the VLM policy instead of reusing the VFM/MoT model config: the two
    # backends have different module names and checkpoint layouts.
    lora_enabled: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    lora_bias: str = "none"
    lora_use_rslora: bool = False
    lora_modules_to_save: str = ""
    lora_precision: str | None = None

    # Legacy free-form field retained for config compatibility. New recipes
    # must use the explicit fields above so PEFT equivalence can be validated.
    lora: Union[str, None] = None
    enable_liger_kernel: bool = False
    trainable_map: Union[str, None] = None
    monkey_patch_for_text_only_data: bool = False

    # HF attention impl. Default "cosmos" routes through cosmos_framework.model.attention
    # (NATTEN/blackwell-fmha on GB200). Override to "flash_attention_2",
    # "sdpa", or "eager" for fallback.
    attn_implementation: str = "cosmos"

    # Qwen3-VL's patch projection is a non-overlapping Conv3d and is therefore
    # algebraically equivalent to a linear projection. ``auto`` selects the
    # linear implementation on A100 (SM80), where large FSDP runs have exposed
    # a cuDNN Conv3d backward failure.  No environment-time monkey patch is
    # required.
    qwen3_vl_patch_embed: str = "auto"

    def __attrs_post_init__(self) -> None:
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError("lora_bias must be one of: none, all, lora_only")
        if self.lora_precision not in {None, "float32", "float16", "bfloat16"}:
            raise ValueError("lora_precision must be float32, float16, bfloat16, or unset")
        if self.qwen3_vl_patch_embed not in {"auto", "linear", "conv3d"}:
            raise ValueError("qwen3_vl_patch_embed must be auto, linear, or conv3d")


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
    # Optional Parakeet inputs for standalone Reasoner CE/SFT, disabled by default.
    sound_und: bool = False
    sound_und_config: SoundUnderstandingConfig = SoundUnderstandingConfig()
    # Applied at model construction, before the optimizer is built.
    freeze: VLMFreezeConfig = VLMFreezeConfig()
    ema: EMAConfig = EMAConfig(enabled=False)

    # Force deterministic kernels in Flash-Attention init (slower; required for
    # parity bit-exactness). VLM-only knob — consumed by VLMModel.__init__ via
    # init_flash_attn_meta.
    deterministic: bool = False
