# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Configuration for the framework-native Cosmos3-Edge VLM (``model_type="cosmos3_edge"``).

Parses the renewed ``nvidia/Cosmos3-Edge`` root ``config.json`` (native schema, no
remote code): ``text_config`` has 28 paired layers, each one attention block followed
by one MLP block. The ported modeling keeps the Nemotron-H 56-block layout (even
index = attention, odd = MLP) so state-dict keys stay
``model.language_model.layers.{0..55}.*``; the 56-block view is derived
(``layers_block_type``), never stored, so serialization round-trips the native
schema. ``vision_start_token_id``/``vision_end_token_id`` default to 20/21 (the old
remote-code values) when absent from ``config.json``.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig


class Cosmos3EdgeVisionConfig(Siglip2VisionConfig):
    """SigLIP2 vision-tower config under the renewed repo's native model_type."""

    model_type = "cosmos3_edge_vision"


class Cosmos3EdgeProjectorConfig(PretrainedConfig):
    """PatchMerger projector config (native schema: ``merger_intermediate_size``)."""

    model_type = "cosmos3_edge_projector"

    def __init__(
        self,
        input_hidden_size: int = 1152,
        merger_intermediate_size: int = 11520,
        out_hidden_size: int = 2048,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        **kwargs,
    ) -> None:
        self.input_hidden_size = input_hidden_size
        self.merger_intermediate_size = merger_intermediate_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        super().__init__(**kwargs)

    @property
    def merger_intermedia(self) -> int:
        """Old remote-code field name; the vendored ``PatchMerger`` reads this."""
        return self.merger_intermediate_size


class Cosmos3EdgeTextConfig(PretrainedConfig):
    """Text config in the renewed native schema (28 paired layers).

    Old-remote-code aliases needed by the ported modeling and the reused
    ``MultiModalRotaryEmbedding`` (``rope_theta``, ``mrope_section``,
    ``layers_block_type``, ``enable_mrope``) are exposed as read-only properties so
    they never serialize.
    """

    model_type = "cosmos3_edge_text"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2048,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "relu2",
        attention_bias: bool = False,
        mlp_bias: bool = False,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        max_position_embeddings: int = 131072,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rope_parameters: dict | None = None,
        sliding_window: int | None = None,
        residual_in_fp32: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 11,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.rope_parameters = (
            dict(rope_parameters)
            if rope_parameters is not None
            else {
                "mrope_section": [24, 20, 20],
                "rope_theta": 100000000,
                "rope_type": "default",
            }
        )
        self.sliding_window = sliding_window
        self.residual_in_fp32 = residual_in_fp32
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rope_theta(self) -> float:
        return self.rope_parameters.get("rope_theta", 100000000)

    @property
    def mrope_section(self) -> list[int]:
        return self.rope_parameters.get("mrope_section", [24, 20, 20])

    @property
    def layers_block_type(self) -> list[str]:
        """56-block Nemotron-H view of the 28 paired native layers ("*-" * 28)."""
        return ["attention", "mlp"] * self.num_hidden_layers

    @property
    def enable_mrope(self) -> bool:
        """Architecture constant (old config.json pinned enable_rope/enable_mrope=true)."""
        return True


class Cosmos3EdgeConfig(PretrainedConfig):
    """Top-level Cosmos3-Edge VLM config: SigLIP2 tower + PatchMerger + Nemotron-H dense LM."""

    model_type = "cosmos3_edge"
    sub_configs = {
        "vision_config": Cosmos3EdgeVisionConfig,
        "projector_config": Cosmos3EdgeProjectorConfig,
        "text_config": Cosmos3EdgeTextConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projector_config=None,
        image_token_id: int = 19,
        video_token_id: int = 18,
        vision_start_token_id: int = 20,
        vision_end_token_id: int = 21,
        projector_hidden_size: int = 11520,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        if isinstance(projector_config, dict):
            self.projector_config = self.sub_configs["projector_config"](**projector_config)
        elif projector_config is None:
            self.projector_config = self.sub_configs["projector_config"]()
        else:
            self.projector_config = projector_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.projector_hidden_size = projector_hidden_size
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = [
    "Cosmos3EdgeConfig",
    "Cosmos3EdgeProjectorConfig",
    "Cosmos3EdgeTextConfig",
    "Cosmos3EdgeVisionConfig",
]
