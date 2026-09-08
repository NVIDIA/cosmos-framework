# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Source Repository: https://github.com/huggingface/transformers
# This is adapted from src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py.
# Transformers Version: v4.57.1
# Commit Hash: 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc
"""Qwen3-Omni Thinking audio tower adapted to the Cosmos audio ABI."""

import math

import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

from cosmos_framework.model.generator.reasoner.audio.projector import AudioProjector
from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.configuration_qwen3_omni_audio import (
    Qwen3OmniAudioConfig,
    get_qwen3_omni_thinking_audio_encoder_config,
)


def get_qwen3_omni_audio_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute the native output length for each packed mel sequence."""
    input_lengths_leave = input_lengths % 100
    feature_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _build_sinusoidal_positions(
    length: int,
    channels: int,
    *,
    device: torch.device | None,
) -> torch.Tensor:
    """Rebuild the native nonpersistent sinusoidal position buffer."""
    log_timescale_increment = math.log(10_000.0) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32, device=device)
    )
    scaled_time = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1) * inv_timescales.unsqueeze(0)
    return torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=1)


class Qwen3OmniThinkingAudioEncoder(nn.Module):
    """Adapt the native packed audio tower to padded ``[batch, time, 128]`` inputs."""

    config: Qwen3OmniMoeAudioEncoderConfig
    encoder: Qwen3OmniMoeAudioEncoder

    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig | None = None,
        encoder: Qwen3OmniMoeAudioEncoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else get_qwen3_omni_thinking_audio_encoder_config()
        self.encoder = encoder if encoder is not None else Qwen3OmniMoeAudioEncoder(self.config)

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Return the native encoded length for each input feature sequence."""
        return get_qwen3_omni_audio_output_lengths(input_lengths)

    def reset_nonpersistent_buffers(self, buffer_device: torch.device | None = None) -> None:
        """Rebuild the sinusoidal position buffer after meta materialization."""
        positional_embedding = _build_sinusoidal_positions(
            self.config.max_source_positions,
            self.config.d_model,
            device=buffer_device,
        )
        self.encoder.positional_embedding.register_buffer(
            "positional_embedding",
            positional_embedding,
            persistent=False,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded log-mels and return padded native audio features."""
        packed_features = torch.cat(
            [input_features[index, : int(length)].transpose(0, 1) for index, length in enumerate(input_lengths)],
            dim=1,
        ).contiguous()
        output_lengths = self.get_output_lengths(input_lengths)
        use_bf16_autocast = (
            self.config._attn_implementation == "flash_attention_2" and packed_features.device.type == "cuda"
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
            flat_features = self.encoder(
                input_features=packed_features,
                feature_lens=input_lengths,
            ).last_hidden_state
        padded_features = nn.utils.rnn.pad_sequence(
            flat_features.split(output_lengths.tolist()),
            batch_first=True,
        )
        return padded_features, output_lengths


class Qwen3OmniAudioProjector(AudioProjector):
    """Project native Qwen audio features into a reasoner's hidden space."""

    def __init__(self, config: Qwen3OmniAudioConfig) -> None:
        super().__init__(
            input_hidden_size=config.encoder_config.output_dim,
            projection_hidden_size=config.projection_hidden_size,
            out_hidden_size=config.out_hidden_size,
        )


class Qwen3OmniAudioModel(nn.Module):
    """Composable Qwen3-Omni Thinking encoder and Cosmos projector."""

    def __init__(
        self,
        config: Qwen3OmniAudioConfig | None = None,
        encoder: Qwen3OmniThinkingAudioEncoder | None = None,
        projector: Qwen3OmniAudioProjector | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else Qwen3OmniAudioConfig()
        self.encoder = encoder if encoder is not None else Qwen3OmniThinkingAudioEncoder(self.config.encoder_config)
        self.projector = projector if projector is not None else Qwen3OmniAudioProjector(self.config)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Restore nonpersistent buffers and initialize the fresh projector."""
        self.encoder.reset_nonpersistent_buffers(buffer_device)
        self.projector.reset_parameters()

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
        audio_token_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume the common audio processor ABI and return projected features."""
        encoded_features, output_lengths = self.encoder(audio_features, audio_feature_lengths)
        processor_lengths = audio_token_lengths.to(device=output_lengths.device, dtype=output_lengths.dtype)
        if not torch.equal(processor_lengths, output_lengths):
            raise ValueError(
                "audio_token_lengths do not match Qwen3-Omni encoder output lengths: "
                f"processor={processor_lengths.tolist()}, encoder={output_lengths.tolist()}"
            )
        encoded_features = encoded_features.to(dtype=self.projector.linear_fc1.weight.dtype)
        return self.projector(encoded_features), output_lengths


__all__ = [
    "Qwen3OmniAudioModel",
    "Qwen3OmniAudioProjector",
    "Qwen3OmniThinkingAudioEncoder",
    "get_qwen3_omni_audio_output_lengths",
]
