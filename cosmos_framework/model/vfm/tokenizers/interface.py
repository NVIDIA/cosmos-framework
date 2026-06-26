# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch

from cosmos_framework.utils.env_parsers.cred_env_parser import CRED_ENVS


class VideoTokenizerInterface(ABC):
    def __init__(self, object_store_credential_path_pretrained: Optional[str] = None):
        assert object_store_credential_path_pretrained is None or isinstance(
            object_store_credential_path_pretrained, str
        )
        if object_store_credential_path_pretrained is None:
            self.backend_args = None
        elif os.path.exists(object_store_credential_path_pretrained) or CRED_ENVS.APP_ENV in ["prod", "dev", "stg"]:
            self.backend_args = {
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": object_store_credential_path_pretrained,
            }
        else:
            raise FileNotFoundError(
                f"Invalid object_store_credential_path_pretrained: {object_store_credential_path_pretrained} and APP_ENV is not prod/dev/stg"
            )

    @abstractmethod
    def reset_dtype(self):
        """
        Reset the dtype of the model to the dtype its weights were trained with or quantized to.
        """
        pass

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        pass

    @abstractmethod
    def get_pixel_num_frames(self, num_latent_frames: int, **kwargs) -> int:
        pass

    def get_latent_temporal_positions(
        self,
        num_pixel_frames: int,
        resolution: str | None = None,
        num_latent_frames: int | None = None,
    ) -> torch.Tensor | None:
        """Return per-latent temporal coordinates when the tokenizer has nonuniform time semantics.

        The default ``None`` preserves legacy latent-index RoPE behavior. Tokenizers
        with boundary or overlap latents can override this to expose one coordinate
        per latent frame.
        """
        del num_pixel_frames, resolution, num_latent_frames
        return None

    @property
    @abstractmethod
    def spatial_compression_factor(self) -> int:
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self) -> int:
        pass

    @property
    @abstractmethod
    def spatial_resolution(self) -> int:
        pass

    @property
    @abstractmethod
    def pixel_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_ch(self) -> int:
        pass

    def compile_encode(
        self,
        warmup_resolutions: Sequence[str],
        output_dir: str,
        aspect_ratio: str | None = None,
        backend: str | None = None,
        mode: str | None = None,
        fullgraph: bool | None = None,
        dynamic: bool | None = None,
    ) -> None:
        """Compile the tokenizer for the given resolutions.

        Subclasses that support AOT compilation should override this method.
        The default raises ``NotImplementedError``.

        Args:
            warmup_resolutions: Resolution keys to compile for.
            output_dir: Root directory where compiled artifacts are stored
                (typically ``config.job.path_local``).
            aspect_ratio: If given, only compile this single aspect ratio.
            --- Only used if the tokenizer does not support AOT compilation ---
            backend: Backend to use for compilation.
            mode: Mode to use for compilation.
            fullgraph: Whether to compile the full graph.
            dynamic: Whether to compile the dynamic graph.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support compilation")

    @property
    def is_chunk_overlap(self):
        return False

    @property
    def is_causal(self) -> bool:
        # Subclasses set self._causal in their __init__ via the `causal` constructor argument.
        return getattr(self, "_causal", True)


class AudioTokenizerInterface(ABC):
    """Abstract interface for audio tokenizers."""

    def __init__(self, object_store_credential_path_pretrained: Optional[str] = None):
        assert object_store_credential_path_pretrained is None or isinstance(
            object_store_credential_path_pretrained, str
        )
        if not object_store_credential_path_pretrained:
            self.backend_args = None
        elif os.path.exists(object_store_credential_path_pretrained) or CRED_ENVS.APP_ENV in ["prod", "dev", "stg"]:
            self.backend_args = {
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": object_store_credential_path_pretrained,
            }
        else:
            raise FileNotFoundError(
                f"Invalid object_store_credential_path_pretrained: {object_store_credential_path_pretrained} and APP_ENV is not prod/dev/stg"
            )

    @abstractmethod
    def reset_dtype(self):
        """
        Reset the dtype of the model to the dtype its weights were trained with or quantized to.
        """
        pass

    @abstractmethod
    def encode(self, audio: torch.Tensor, force_pad: bool = False) -> torch.Tensor:
        """
        Encode audio waveform to latent representation.

        Args:
            audio: Input audio tensor of shape [B, C, T] where:
                   B = batch size, C = audio channels, T = time samples
            force_pad: Whether to force padding to match compression factor

        Returns:
            Latent tensor of shape [B, latent_ch, T']
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio waveform.

        Args:
            latent: Latent tensor of shape [B, latent_ch, T']

        Returns:
            Audio tensor of shape [B, C, T]
        """
        pass

    @abstractmethod
    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        """
        Calculate the number of latent time samples from audio samples.

        Args:
            num_audio_samples: Number of audio samples

        Returns:
            Number of latent time samples
        """
        pass

    @abstractmethod
    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        """
        Calculate the number of audio samples from latent samples.

        Args:
            num_latent_samples: Number of latent time samples

        Returns:
            Number of audio samples
        """
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self) -> int:
        """
        Temporal compression factor (downsampling ratio).
        audio_samples = latent_samples * temporal_compression_factor
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        pass

    @property
    @abstractmethod
    def audio_channels(self) -> int:
        """Number of audio channels (e.g., 1 for mono, 2 for stereo)."""
        pass

    @property
    @abstractmethod
    def latent_ch(self) -> int:
        """Number of latent channels."""
        pass

    @property
    def is_causal(self) -> bool:
        """Whether the model is causal (for streaming)."""
        return False
