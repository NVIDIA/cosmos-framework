# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Source Repository: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
# This is adapted from processing.py and wraps transformers.ParakeetFeatureExtractor.
# Commit Hash: 24e67ea000b7c2837fc8f9488aa2008524fac8ba
"""Standalone raw-audio preprocessing for the Nemotron Parakeet encoder."""

from collections.abc import Sequence
from typing import ClassVar, Protocol

import numpy as np
import torch
from transformers import ParakeetFeatureExtractor

from cosmos_framework.model.generator.reasoner.parakeet.configuration_parakeet import get_nemotron_parakeet_config
from cosmos_framework.model.generator.reasoner.parakeet.parakeet import get_subsampled_lengths

_PARAKEET_ENCODER_CONFIG = get_nemotron_parakeet_config()

AudioClip = np.ndarray | torch.Tensor
AudioInput = AudioClip | Sequence[AudioClip]


class _FeatureExtractor(Protocol):
    """Subset of ``ParakeetFeatureExtractor`` used by this processor."""

    feature_size: int
    hop_length: int
    sampling_rate: int

    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: str,
    ) -> dict[str, torch.Tensor]: ...


class ParakeetAudioProcessor:
    """Convert mono 16 kHz waveforms into features for Parakeet.

    This processor deliberately does not load audio files or resample audio.
    Each input clip must be a one-dimensional floating-point NumPy array or
    torch tensor that has already been mixed down to mono and sampled at 16 kHz.
    """

    model_input_names: ClassVar[list[str]] = ["audio_features", "audio_feature_lengths", "audio_token_lengths"]

    sampling_rate: int = 16_000
    num_mel_bins: int = 128
    hop_length: int = 160
    subsampling_factor: int = 8

    feature_extractor: _FeatureExtractor

    def __init__(self, feature_extractor: _FeatureExtractor | None = None) -> None:
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else ParakeetFeatureExtractor(
                sampling_rate=self.sampling_rate,
                feature_size=self.num_mel_bins,
            )
        )
        self._validate_feature_extractor()

    def _validate_feature_extractor(self) -> None:
        """Ensure an injected extractor matches the released audio frontend."""
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
        """Validate one mono waveform and copy it to a CPU float32 array."""
        if isinstance(audio, torch.Tensor):
            if audio.ndim != 1:
                raise ValueError(
                    f"Audio clip {clip_index} must be a one-dimensional mono waveform; "
                    f"got tensor shape {tuple(audio.shape)}. Mix channels to mono before preprocessing."
                )
            if not audio.dtype.is_floating_point:
                raise TypeError(f"Audio clip {clip_index} must have a floating-point dtype, got {audio.dtype}")
            waveform = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
        elif isinstance(audio, np.ndarray):
            if audio.ndim != 1:
                raise ValueError(
                    f"Audio clip {clip_index} must be a one-dimensional mono waveform; "
                    f"got array shape {audio.shape}. Mix channels to mono before preprocessing."
                )
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
        if waveform.size < 2 * ParakeetAudioProcessor.hop_length:
            raise ValueError(
                f"Audio clip {clip_index} must contain at least {2 * ParakeetAudioProcessor.hop_length} samples "
                "for stable feature normalization"
            )
        return waveform

    @staticmethod
    def _normalize_audio_input(audios: AudioInput) -> list[AudioClip]:
        """Normalize a single clip or an ordered clip sequence to a list."""
        if isinstance(audios, (np.ndarray, torch.Tensor)):
            return [audios]
        if not isinstance(audios, Sequence) or isinstance(audios, (str, bytes)):
            raise TypeError("audios must be a waveform array/tensor or a sequence of waveform arrays/tensors")
        clips = list(audios)
        if not clips:
            raise ValueError("audios must contain at least one clip")
        return clips

    @classmethod
    def get_token_timestamps(cls, audio_feature_length: int) -> list[float]:
        """Return native token start times for one unpadded mel sequence."""
        num_tokens = int(
            get_subsampled_lengths(
                torch.tensor([audio_feature_length + 1]),
                _PARAKEET_ENCODER_CONFIG,
            )[0]
        )
        token_stride_seconds = cls.hop_length * cls.subsampling_factor / cls.sampling_rate
        return [token_index * token_stride_seconds for token_index in range(num_tokens)]

    def __call__(
        self,
        audios: AudioInput,
        *,
        sampling_rate: int,
    ) -> dict[str, torch.Tensor]:
        """Extract a padded batch of log-mel features from raw waveforms.

        Args:
            audios: One mono floating-point waveform or a sequence of such
                waveforms. Every clip must be one-dimensional; two-dimensional
                arrays are not interpreted as either stereo audio or a padded
                batch.
            sampling_rate: Sampling rate shared by all clips. Only 16 kHz is
                accepted. Callers must resample before invoking this processor.

        Returns:
            ``audio_features`` with shape ``[num_clips, max_frames, 128]`` and
            one-dimensional ``audio_feature_lengths`` and
            ``audio_token_lengths`` tensors.
        """
        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, (int, np.integer)):
            raise TypeError(f"sampling_rate must be an integer, got {type(sampling_rate).__name__}")
        if int(sampling_rate) != self.sampling_rate:
            raise ValueError(
                f"ParakeetAudioProcessor requires {self.sampling_rate} Hz mono audio and does not resample; "
                f"got {sampling_rate} Hz. Resample before preprocessing."
            )

        clips = self._normalize_audio_input(audios)
        waveforms = [self._as_waveform_array(clip, clip_index) for clip_index, clip in enumerate(clips)]
        sample_lengths = torch.tensor([waveform.shape[0] for waveform in waveforms], dtype=torch.long)
        natural_feature_lengths = torch.div(sample_lengths, self.hop_length, rounding_mode="floor") + 1

        extracted = self.feature_extractor(
            waveforms,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        )
        audio_features = torch.as_tensor(extracted["input_features"], dtype=torch.float32)
        expected_shape = (len(waveforms), int(natural_feature_lengths.max()), self.num_mel_bins)
        if tuple(audio_features.shape) != expected_shape:
            raise RuntimeError(
                "ParakeetFeatureExtractor returned an unexpected input_features shape: "
                f"expected {expected_shape}, got {tuple(audio_features.shape)}"
            )
        if not torch.isfinite(audio_features).all():
            raise RuntimeError("ParakeetFeatureExtractor returned NaN or infinite input_features")

        attention_mask = torch.as_tensor(extracted["attention_mask"])
        if tuple(attention_mask.shape) != audio_features.shape[:2]:
            raise RuntimeError(
                "ParakeetFeatureExtractor returned an unexpected attention_mask shape: "
                f"expected {tuple(audio_features.shape[:2])}, got {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(torch.long)
        elif attention_mask.dtype.is_floating_point or attention_mask.dtype.is_complex:
            raise TypeError(f"Parakeet attention_mask must use an integer or bool dtype, got {attention_mask.dtype}")
        if torch.any((attention_mask != 0) & (attention_mask != 1)):
            raise ValueError("Parakeet attention_mask must contain only zeros and ones")
        audio_feature_lengths = attention_mask.sum(dim=1, dtype=torch.long)
        if not torch.equal(audio_feature_lengths + 1, natural_feature_lengths):
            raise RuntimeError(
                "ParakeetFeatureExtractor attention lengths do not match its center-padded feature shape: "
                f"mask={audio_feature_lengths.tolist()}, expected={natural_feature_lengths.tolist()}"
            )

        # Transformers masks the final center-padding mel frame, while the
        # released Omni merge path keeps the corresponding encoded position.
        # Keep both lengths: mask lengths drive encoder attention; +1 lengths
        # drive dynamic placeholder expansion and valid output selection.
        audio_token_lengths = get_subsampled_lengths(
            audio_feature_lengths + 1,
            _PARAKEET_ENCODER_CONFIG,
        ).to(dtype=audio_feature_lengths.dtype)
        return {
            "audio_features": audio_features.contiguous(),
            "audio_feature_lengths": audio_feature_lengths,
            "audio_token_lengths": audio_token_lengths,
        }
