# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Audio encoder backends supported by Cosmos Reasoner models."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioEncoderBackend:
    """Factories shared by Reasoner model and data construction."""

    build_model: Callable[[int, int], Any]
    build_processor: Callable[[], Any]


def _build_parakeet_model(projection_hidden_size: int, out_hidden_size: int) -> Any:
    from cosmos_framework.model.generator.reasoner.parakeet.configuration_parakeet import ParakeetAudioConfig
    from cosmos_framework.model.generator.reasoner.parakeet.parakeet import ParakeetAudioModel

    return ParakeetAudioModel(
        ParakeetAudioConfig(
            projection_hidden_size=projection_hidden_size,
            out_hidden_size=out_hidden_size,
        )
    )


def _build_parakeet_processor() -> Any:
    from cosmos_framework.data.generator.processors.parakeet_audio_processor import ParakeetAudioProcessor

    return ParakeetAudioProcessor()


def _build_qwen3_omni_thinking_model(projection_hidden_size: int, out_hidden_size: int) -> Any:
    from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.configuration_qwen3_omni_audio import (
        Qwen3OmniAudioConfig,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.qwen3_omni_audio import Qwen3OmniAudioModel

    return Qwen3OmniAudioModel(
        Qwen3OmniAudioConfig(
            projection_hidden_size=projection_hidden_size,
            out_hidden_size=out_hidden_size,
        )
    )


def _build_qwen3_omni_thinking_processor() -> Any:
    from cosmos_framework.data.generator.processors.qwen3_omni_audio_processor import Qwen3OmniAudioProcessor

    return Qwen3OmniAudioProcessor()


_AUDIO_ENCODER_BACKENDS = {
    "parakeet": AudioEncoderBackend(
        build_model=_build_parakeet_model,
        build_processor=_build_parakeet_processor,
    ),
    "qwen3_omni_thinking": AudioEncoderBackend(
        build_model=_build_qwen3_omni_thinking_model,
        build_processor=_build_qwen3_omni_thinking_processor,
    ),
}


def get_audio_encoder_backend(audio_encoder_type: str) -> AudioEncoderBackend:
    """Return the model and processor factories for an audio encoder type."""
    backend = _AUDIO_ENCODER_BACKENDS.get(audio_encoder_type)
    if backend is None:
        raise ValueError(f"Unsupported audio encoder type: {audio_encoder_type!r}")
    return backend
