# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Load the understanding tower of a Cosmos3 checkpoint."""

import re

from transformers import Qwen3VLForConditionalGeneration

from transformers_cosmos3.config import Cosmos3OmniConfig

DROP_PATTERNS: tuple[str, ...] = (
    # Generation
    r"_moe_gen",
    r"^llm2vae\.",
    r"^vae2llm\.",
    r"^proj_in\.",
    r"^proj_out\.",
    r"^time_embedder\.",
    r"^latent_pos_embed$",
    r"\.self_attn\.add_[qkv]_proj\.",
    r"\.self_attn\.to_add_out\.",
    r"\.self_attn\.norm_added_[qk]\.",
    # Sound / audio
    r"^llm2sound\.",
    r"^sound2llm\.",
    r"^sound_modality_embed$",
    r"^audio_proj_(in|out)\.",
    r"^audio_modality_embed$",
    # Action
    r"^llm2action\.",
    r"^action2llm\.",
    r"^action_modality_embed$",
    r"^action_proj_(in|out)\.",
)
"""Drop patterns (regex, matched via `re.search`). Shared with vllm-cosmos3."""

KEY_MAPPING: dict[str, str] = {
    # Diffusers-style attention names -> HF Qwen3-VL text names.
    r"\.self_attn\.to_q\.": ".self_attn.q_proj.",
    r"\.self_attn\.to_k\.": ".self_attn.k_proj.",
    r"\.self_attn\.to_v\.": ".self_attn.v_proj.",
    r"\.self_attn\.to_out\.": ".self_attn.o_proj.",
    r"\.self_attn\.norm_q\.": ".self_attn.q_norm.",
    r"\.self_attn\.norm_k\.": ".self_attn.k_norm.",
    # vLLM/HF task-head variants -> HF top-level task head.
    r"^model\.lm_head\.": "lm_head.",
    r"^language_model\.lm_head\.": "lm_head.",
    # vLLM language model namespace -> HF Qwen3-VL language model namespace.
    r"^language_model\.model\.(.+)$": r"model.language_model.\1",
    # Flat Qwen3 text keys -> nested HF Qwen3-VL language model.
    r"^model\.(?!language_model\.|visual\.)(embed_tokens\.|layers\.|norm\.)(.*)$": r"model.language_model.\1\2",
    r"^language_model\.(?!model\.|lm_head\.)(embed_tokens\.|layers\.|norm\.)(.*)$": r"model.language_model.\1\2",
    r"^(embed_tokens\.|layers\.|norm\.)(.*)$": r"model.language_model.\1\2",
    # Flat Qwen3-VL vision component -> nested HF Qwen3-VL.
    r"^visual\.(.+)$": r"model.visual.\1",
    r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)(.*)$": r"model.visual.\1\2",
}
"""Cosmos3 checkpoint key renames into HF Qwen3-VL names. Shared with vllm-cosmos3."""


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Match the root model_type. Qwen3VLConfig.from_pretrained otherwise selects
    # Cosmos's nested vision_config (also tagged qwen3_vl) as the whole config.
    config_class = Cosmos3OmniConfig

    # Drop-pattern keys don't match any model parameter after rename -- the
    # loader skips them; these patterns silence the resulting warning.
    _keys_to_ignore_on_load_unexpected = list(DROP_PATTERNS)

    def _get_key_renaming_mapping(
        self,
        checkpoint_keys: list[str],
        key_mapping: dict[str, str] | None = None,
        loading_base_model_from_task_state_dict: bool = False,
        loading_task_model_from_base_state_dict: bool = False,
    ) -> dict[str, str]:
        """Compose attention and namespace renames on Transformers 4.57.

        Its default implementation stops after the first matching rule. A flat
        ``layers.N.self_attn.to_q`` key needs both the projection rename and the
        ``model.language_model`` prefix, including for ModelOpt scale buffers.
        Cosmos supplies the complete task checkpoint (including lm_head), even
        with flat names, so these mappings already include the full prefix.
        """
        mapping = {}
        for key in checkpoint_keys:
            renamed, _ = self._fix_state_dict_key_on_load(key)
            for pattern, replacement in (key_mapping or KEY_MAPPING).items():
                renamed = re.sub(pattern, replacement, renamed)
            mapping[key] = renamed
        return mapping

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "Cosmos3ForConditionalGeneration":
        # `_checkpoint_conversion_mapping` is a global model_type -> mapping
        # registry, so subclassing doesn't register new renames. Inject via
        # the per-call `key_mapping=` kwarg instead, letting callers override.
        merged = {**KEY_MAPPING, **(kwargs.pop("key_mapping", None) or {})}
        kwargs["key_mapping"] = merged
        return super().from_pretrained(*args, **kwargs)
