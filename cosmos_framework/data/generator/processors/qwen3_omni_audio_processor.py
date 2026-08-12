# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Source Repository: https://github.com/huggingface/transformers
# This uses the Qwen3-Omni processor's WhisperFeatureExtractor parameters and
# intentionally disables the Transformers 4.57.1 frontend's 30-second truncation.
# Transformers Version: v4.57.1
# Commit Hash: 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc
"""Raw-audio preprocessing for the Qwen3-Omni Thinking audio tower."""

from collections.abc import Sequence
from typing import ClassVar, Protocol

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.qwen3_omni_audio import (
    get_qwen3_omni_audio_output_lengths,
)

AudioClip = np.ndarray | torch.Tensor
AudioInput = AudioClip | Sequence[AudioClip]


class _FeatureExtractor(Protocol):
    chunk_length: int
    feature_size: int
    hop_length: int
    n_fft: int
    sampling_rate: int

    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, torch.Tensor]: ...


class Qwen3OmniAudioProcessor:
    """Convert mono 16 kHz waveforms into Qwen3-Omni log-mel features."""

    model_input_names: ClassVar[list[str]] = ["audio_features", "audio_feature_lengths", "audio_token_lengths"]

    sampling_rate: int = 16_000
    num_mel_bins: int = 128
    hop_length: int = 160
    n_fft: int = 400
    chunk_length: int = 300
    position_id_per_seconds: int = 13

    feature_extractor: _FeatureExtractor

    def __init__(self, feature_extractor: _FeatureExtractor | None = None) -> None:
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else WhisperFeatureExtractor(
                feature_size=self.num_mel_bins,
                sampling_rate=self.sampling_rate,
                hop_length=self.hop_length,
                chunk_length=self.chunk_length,
                n_fft=self.n_fft,
                padding_value=0.0,
                dither=0.0,
                return_attention_mask=True,
                padding_side="right",
            )
        )
        expected_attributes = {
            "sampling_rate": self.sampling_rate,
            "feature_size": self.num_mel_bins,
            "hop_length": self.hop_length,
        }
        for attribute, expected_value in expected_attributes.items():
            actual_value = getattr(self.feature_extractor, attribute, None)
            if actual_value != expected_value:
                raise ValueError(f"feature_extractor.{attribute} must be {expected_value}, got {actual_value}")

    @staticmethod
    def _as_waveform_array(audio: AudioClip, clip_index: int) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            if audio.ndim != 1:
                raise ValueError(f"Audio clip {clip_index} must be a one-dimensional mono waveform")
            if not audio.dtype.is_floating_point:
                raise TypeError(f"Audio clip {clip_index} must have a floating-point dtype, got {audio.dtype}")
            waveform = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
        elif isinstance(audio, np.ndarray):
            if audio.ndim != 1:
                raise ValueError(f"Audio clip {clip_index} must be a one-dimensional mono waveform")
            if not np.issubdtype(audio.dtype, np.floating):
                raise TypeError(f"Audio clip {clip_index} must have a floating-point dtype, got {audio.dtype}")
            waveform = np.asarray(audio, dtype=np.float32)
        else:
            raise TypeError(
                f"Audio clip {clip_index} must be a NumPy array or torch tensor, got {type(audio).__name__}"
            )

        if waveform.size == 0:
            raise ValueError(f"Audio clip {clip_index} must contain at least one sample")
        if not np.isfinite(waveform).all():
            raise ValueError(f"Audio clip {clip_index} contains NaN or infinite samples")
        return waveform

    @staticmethod
    def _normalize_audio_input(audios: AudioInput) -> list[AudioClip]:
        if isinstance(audios, (np.ndarray, torch.Tensor)):
            return [audios]
        clips = list(audios)
        if not clips:
            raise ValueError("audios must contain at least one clip")
        return clips

    @classmethod
    def get_token_timestamps(cls, audio_feature_length: int) -> list[float]:
        """Return timestamps on Qwen3-Omni's released logical audio clock."""
        num_tokens = int(get_qwen3_omni_audio_output_lengths(torch.tensor([audio_feature_length]))[0])
        return [token_index / cls.position_id_per_seconds for token_index in range(num_tokens)]

    def __call__(
        self,
        audios: AudioInput,
        *,
        sampling_rate: int,
    ) -> dict[str, torch.Tensor]:
        """Extract right-padded log-mels without implicit truncation or resampling.

        The released checkpoint frontend parameters are retained, while a
        300-second chunk window and ``truncation=False`` preserve long audio
        instead of the 30-second runtime fallback in Transformers 4.57.1.
        """
        if int(sampling_rate) != self.sampling_rate:
            raise ValueError(
                f"Qwen3OmniAudioProcessor requires {self.sampling_rate} Hz mono audio and does not resample; "
                f"got {sampling_rate} Hz. Resample before preprocessing."
            )

        clips = self._normalize_audio_input(audios)
        waveforms = [self._as_waveform_array(clip, clip_index) for clip_index, clip in enumerate(clips)]
        extracted = self.feature_extractor(
            waveforms,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=False,
        )
        input_features = torch.as_tensor(extracted["input_features"], dtype=torch.float32)
        attention_mask = torch.as_tensor(extracted["attention_mask"])
        audio_feature_lengths = attention_mask.sum(dim=1, dtype=torch.long)
        audio_features = input_features.transpose(1, 2).contiguous()
        return {
            "audio_features": audio_features,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_token_lengths": get_qwen3_omni_audio_output_lengths(audio_feature_lengths),
        }


__all__ = ["Qwen3OmniAudioProcessor"]
