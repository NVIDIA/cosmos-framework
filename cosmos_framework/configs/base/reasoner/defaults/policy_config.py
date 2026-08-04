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

    # LoRA (parameter-efficient fine-tuning). When ``lora_enabled=True``,
    # ``VLMModel._init_vlm`` injects the custom LoRA adapters BEFORE FSDP wrap on
    # the meta-device HF backbone, then re-initializes lora_A/lora_B after the
    # meta tensors are materialized and the base weights are loaded. Pair with
    # ``optimizer.keys_to_select=["lora_"]``.
    #
    # ``lora_target_modules`` is matched by EXACT child name (see
    # ``_inject_lora_inplace``), not substring. The default targets the four
    # Qwen3-VL LLM attention projections; the vision tower names its projections
    # ``qkv`` / ``proj`` / ``linear_fc1`` / ``linear_fc2``, so the ViT is never
    # touched. Other model families (e.g. cosmos3_edge) may need different names.
    #
    # ``lora_exclude_path_regex`` is searched against each candidate module's
    # dotted path and skips matches. Needed when the two towers of a VLM share
    # projection names: Cosmos3-Edge names its LLM projections q/k/v/o_proj and
    # its SigLIP2 vision projections q/k/v/out_proj, so three of four collide and
    # name matching alone would inject adapters into the (frozen) vision tower.
    lora_enabled: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    lora_exclude_path_regex: str = ""

    # Extra model config
    enable_liger_kernel: bool = False
    trainable_map: Union[str, None] = None
    monkey_patch_for_text_only_data: bool = False

    # HF attention impl. Default "cosmos" routes through cosmos_framework.model.attention
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
