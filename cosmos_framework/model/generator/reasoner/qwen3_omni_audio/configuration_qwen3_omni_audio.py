# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Source Repository: https://github.com/huggingface/transformers
# This is adapted from src/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py.
# Transformers Version: v4.57.1
# Commit Hash: 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc
"""Configuration for the Qwen3-Omni Thinking audio frontend."""

from numbers import Integral
from typing import Any, ClassVar

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)


def get_qwen3_omni_thinking_audio_encoder_config() -> Qwen3OmniMoeAudioEncoderConfig:
    """Return the released Qwen3-Omni-30B-A3B-Thinking audio-tower config."""
    return Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=50,
        output_dim=2048,
        n_window_infer=800,
        conv_chunksize=500,
        downsample_hidden_size=480,
        attn_implementation="flash_attention_2",
    )


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class Qwen3OmniAudioConfig(PretrainedConfig):
    """Configuration for a Qwen3-Omni audio tower plus Cosmos projector."""

    model_type: ClassVar[str] = "qwen3_omni_thinking_audio"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {"encoder_config": Qwen3OmniMoeAudioEncoderConfig}

    def __init__(
        self,
        encoder_config: Qwen3OmniMoeAudioEncoderConfig | dict[str, Any] | None = None,
        projection_hidden_size: int = 4096,
        out_hidden_size: int = 2688,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if encoder_config is None:
            encoder_config = get_qwen3_omni_thinking_audio_encoder_config()
        elif isinstance(encoder_config, dict):
            encoder_config = Qwen3OmniMoeAudioEncoderConfig(**encoder_config)
        elif not isinstance(encoder_config, Qwen3OmniMoeAudioEncoderConfig):
            raise TypeError(
                "encoder_config must be a Qwen3OmniMoeAudioEncoderConfig, a configuration dictionary, or None, "
                f"got {type(encoder_config).__name__}"
            )

        _validate_positive_integer("encoder_config.output_dim", encoder_config.output_dim)
        _validate_positive_integer("projection_hidden_size", projection_hidden_size)
        _validate_positive_integer("out_hidden_size", out_hidden_size)

        self.encoder_config = encoder_config
        self.projection_hidden_size = int(projection_hidden_size)
        self.out_hidden_size = int(out_hidden_size)


__all__ = ["Qwen3OmniAudioConfig", "get_qwen3_omni_thinking_audio_encoder_config"]
