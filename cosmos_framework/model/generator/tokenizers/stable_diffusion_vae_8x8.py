# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stable Diffusion VAE tokenizer wrapper for DiT image pretraining."""

from typing import Any

import torch

from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Resolve a string dtype name to a torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _DTYPE_BY_NAME:
        raise ValueError(f"Unsupported SD VAE dtype '{dtype}'. Supported values: {sorted(_DTYPE_BY_NAME)}.")
    return _DTYPE_BY_NAME[dtype]


def _default_device() -> torch.device:
    """Return the current CUDA device when available."""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _config_value(config: Any, key: str, default: Any) -> Any:
    """Fetch a diffusers config value from object-like or dict-like configs."""
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


class StableDiffusionVAEInterface(VideoTokenizerInterface):
    """Stable Diffusion AutoencoderKL adapter using the shared video tokenizer interface."""

    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str | None = None,
        vae_path: str = "stabilityai/sd-vae-ft-ema",
        subfolder: str | None = None,
        scaling_factor: float | None = 0.18215,
        sample_posterior: bool = True,
        dtype: str | torch.dtype = "float32",
        device: str | None = None,
        chunk_duration: int = 1,
        spatial_compression_factor: int = 8,
        temporal_compression_factor: int = 1,
    ) -> None:
        super().__init__(object_store_credential_path_pretrained=object_store_credential_path_pretrained)
        self.vae_path = vae_path
        self.subfolder = subfolder
        self.sample_posterior = sample_posterior
        self._dtype = _resolve_dtype(dtype)
        self.device = torch.device(device) if device is not None else _default_device()
        self.chunk_duration = chunk_duration
        self.spatial_compression = spatial_compression_factor
        self.temporal_compression = temporal_compression_factor

        resolved_vae_path = self._resolve_vae_path(bucket_name=bucket_name, vae_path=vae_path)
        self.model = self._load_model(vae_path=resolved_vae_path, subfolder=subfolder)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device=self.device, dtype=self._dtype)

        model_config = getattr(self.model, "config", None)
        self.scaling_factor = float(
            scaling_factor if scaling_factor is not None else _config_value(model_config, "scaling_factor", 0.18215)
        )
        self._latent_ch = int(_config_value(model_config, "latent_channels", 4))
        self._spatial_resolution = int(_config_value(model_config, "sample_size", 256))

    def _resolve_vae_path(self, bucket_name: str, vae_path: str) -> str:
        """Resolve internal pretrained paths while leaving Hugging Face repo ids unchanged."""
        if vae_path.startswith("pretrained/") and bucket_name:
            return f"s3://{bucket_name}/{vae_path}"
        return vae_path

    def _load_model(self, vae_path: str, subfolder: str | None) -> torch.nn.Module:
        """Load a diffusers AutoencoderKL model."""
        try:
            from diffusers import AutoencoderKL
        except ImportError as error:
            raise ImportError(
                "StableDiffusionVAEInterface requires diffusers. Install diffusers or use wan2pt1_tokenizer."
            ) from error

        kwargs: dict[str, object] = {"torch_dtype": self._dtype}
        if subfolder is not None:
            kwargs["subfolder"] = subfolder
        if vae_path.startswith("s3://"):
            raise ValueError(
                "StableDiffusionVAEInterface expects a Hugging Face repo id or local diffusers VAE directory, "
                f"not an S3 path: {vae_path}"
            )
        return AutoencoderKL.from_pretrained(vae_path, **kwargs)

    @property
    def dtype(self) -> torch.dtype:
        """Model compute dtype."""
        return self._dtype

    def reset_dtype(self) -> None:
        """Reset the dtype of the model."""
        self.model.to(device=self.device, dtype=self._dtype)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode normalized pixels in [-1, 1] to scaled SD VAE latents."""
        if state.dim() != 5:
            raise ValueError(f"Expected state tensor [B,3,T,H,W], got shape {tuple(state.shape)}.")
        batch_size, channels, num_frames, height, width = state.shape
        if channels != 3:
            raise ValueError(f"Expected 3 input channels, got {channels}.")

        frames = state.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,3,H,W]
        frames = frames.view(batch_size * num_frames, channels, height, width)  # [B*T,3,H,W]
        frames = frames.to(device=self.device, dtype=self._dtype)  # [B*T,3,H,W]

        posterior = self.model.encode(frames).latent_dist
        if self.sample_posterior:
            latents_2d = posterior.sample()  # [B*T,4,H//8,W//8]
        else:
            latents_2d = posterior.mode()  # [B*T,4,H//8,W//8]
        latents_2d = latents_2d * self.scaling_factor  # [B*T,4,H//8,W//8]

        latent_channels = latents_2d.shape[1]
        latent_height = latents_2d.shape[2]
        latent_width = latents_2d.shape[3]
        latents = latents_2d.view(
            batch_size, num_frames, latent_channels, latent_height, latent_width
        )  # [B,T,4,H//8,W//8]
        latents = latents.permute(0, 2, 1, 3, 4).contiguous()  # [B,4,T,H//8,W//8]
        return latents

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode scaled SD VAE latents to normalized pixels in [-1, 1]."""
        if latent.dim() != 5:
            raise ValueError(f"Expected latent tensor [B,4,T,H,W], got shape {tuple(latent.shape)}.")
        batch_size, channels, num_frames, height, width = latent.shape
        if channels != self._latent_ch:
            raise ValueError(f"Expected {self._latent_ch} latent channels, got {channels}.")

        latents_2d = latent.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,4,H,W]
        latents_2d = latents_2d.view(batch_size * num_frames, channels, height, width)  # [B*T,4,H,W]
        latents_2d = latents_2d.to(device=self.device, dtype=self._dtype) / self.scaling_factor  # [B*T,4,H,W]

        decoded_2d = self.model.decode(latents_2d).sample  # [B*T,3,H*8,W*8]
        decoded_height = decoded_2d.shape[2]
        decoded_width = decoded_2d.shape[3]
        decoded = decoded_2d.view(batch_size, num_frames, 3, decoded_height, decoded_width)  # [B,T,3,H*8,W*8]
        decoded = decoded.permute(0, 2, 1, 3, 4).contiguous()  # [B,3,T,H*8,W*8]
        return decoded

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        """Get number of latent frames from pixel frames."""
        return num_pixel_frames

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        """Get number of pixel frames from latent frames."""
        return num_latent_frames

    @property
    def spatial_compression_factor(self) -> int:
        """Spatial compression factor."""
        return self.spatial_compression

    @property
    def temporal_compression_factor(self) -> int:
        """Temporal compression factor."""
        return self.temporal_compression

    @property
    def spatial_resolution(self) -> int:
        """Spatial resolution."""
        return self._spatial_resolution

    @property
    def pixel_chunk_duration(self) -> int:
        """Pixel chunk duration."""
        return self.chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        """Latent chunk duration."""
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self) -> int:
        """Number of latent channels."""
        return self._latent_ch

    @property
    def name(self) -> str:
        """Name of the tokenizer."""
        return "sd_vae_tokenizer"

    def count_param(self) -> int:
        """Count the number of parameters in the model."""
        return sum(parameter.numel() for parameter in self.model.parameters())
