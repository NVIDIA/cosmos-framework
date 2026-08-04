# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Configuration for Parakeet audio understanding components."""

from numbers import Integral
from typing import Any, ClassVar

from transformers import ParakeetEncoderConfig
from transformers.configuration_utils import PretrainedConfig


def get_nemotron_parakeet_config() -> ParakeetEncoderConfig:
    """Return the audio-encoder configuration used by Nemotron 3 Nano Omni."""
    return ParakeetEncoderConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_act="silu",
        attention_bias=False,
        conv_kernel_size=9,
        convolution_bias=False,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        num_mel_bins=128,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.1,
        dropout_positions=0.0,
        layerdrop=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=5000,
        scale_input=False,
        initializer_range=0.02,
    )


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class ParakeetAudioConfig(PretrainedConfig):
    """Configuration for a Parakeet encoder plus its audio projector.

    The default encoder matches Nemotron 3 Nano Omni, while the fresh projector
    follows the Qwen-VL merger architecture. ``out_hidden_size`` is configurable
    so the same audio encoder can feed different reasoner hidden sizes.
    """

    model_type: ClassVar[str] = "parakeet_audio"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {"encoder_config": ParakeetEncoderConfig}

    def __init__(
        self,
        encoder_config: ParakeetEncoderConfig | dict[str, Any] | None = None,
        projection_hidden_size: int = 4096,
        out_hidden_size: int = 2688,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if encoder_config is None:
            encoder_config = get_nemotron_parakeet_config()
        elif isinstance(encoder_config, dict):
            encoder_config = ParakeetEncoderConfig(**encoder_config)
        elif not isinstance(encoder_config, ParakeetEncoderConfig):
            raise TypeError(
                "encoder_config must be a ParakeetEncoderConfig, a configuration dictionary, or None, "
                f"got {type(encoder_config).__name__}"
            )

        _validate_positive_integer("encoder_config.hidden_size", encoder_config.hidden_size)
        _validate_positive_integer("projection_hidden_size", projection_hidden_size)
        _validate_positive_integer("out_hidden_size", out_hidden_size)
        self.encoder_config = encoder_config
        self.projection_hidden_size = int(projection_hidden_size)
        self.out_hidden_size = int(out_hidden_size)


__all__ = ["ParakeetAudioConfig", "get_nemotron_parakeet_config"]
