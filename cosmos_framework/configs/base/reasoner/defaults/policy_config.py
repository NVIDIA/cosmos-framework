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
    # Opt-in objective change: normalize CE once over the full gradient-accumulation window
    # instead of averaging independently normalized microbatch ratios.
    normalize_weighted_ce_over_accumulation_window: bool = False

    # Extra model config
    lora: Union[str, None] = None
    enable_liger_kernel: bool = False
    trainable_map: Union[str, None] = None
    monkey_patch_for_text_only_data: bool = False

    # HF attention impl. Default "cosmos" routes through cosmos_framework.model.attention
    # (NATTEN/blackwell-fmha on GB200). Override to "flash_attention_2",
    # "sdpa", or "eager" for fallback.
    attn_implementation: str = "cosmos"


@attrs.define(slots=False)
class LBLConfig:
    """MoE load-balancing auxiliary loss for the Qwen3-VL-MoE backbone.

    The routing statistics are collected every step regardless (the patched MoE block
    stashes them either way, see ``monkey_patch.patch_qwen3_vl_moe_grouped_mm_experts``);
    these knobs only control whether a loss term is built from them.
    """

    # "local" balances each rank's own token counts; "global" sums the counts across the
    # DP mesh first, balancing the global batch at the cost of a collective per step.
    method: str = attrs.field(default="local", validator=attrs.validators.in_({"local", "global"}))

    # Multiplier on the load-balancing loss added to the CE objective. None disables the
    # term entirely, which is the default so existing recipes are unchanged.
    coeff: float | None = None


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

    # MoE load-balancing auxiliary loss (Qwen3-VL-MoE only), disabled by default.
    lbl: LBLConfig = LBLConfig()

    # Optional audio inputs for standalone Reasoner CE/SFT, disabled by default.
    sound_und: bool = False
    sound_und_config: SoundUnderstandingConfig = SoundUnderstandingConfig()
    # Applied at model construction, before the optimizer is built.
    freeze: VLMFreezeConfig = VLMFreezeConfig()
    ema: EMAConfig = EMAConfig(enabled=False)

    # Force deterministic kernels in Flash-Attention init (slower; required for
    # parity bit-exactness). VLM-only knob — consumed by VLMModel.__init__ via
    # init_flash_attn_meta.
    deterministic: bool = False
