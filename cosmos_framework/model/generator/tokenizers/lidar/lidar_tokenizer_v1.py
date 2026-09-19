# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LiDAR TransformerVAE inference interface (range + intensity + mask)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface
from cosmos_framework.model.generator.tokenizers.lidar.checkpoint import (
    load_model_checkpoint,
    parse_lidar_checkpoint_stats,
    resolve_artifact_path,
)
from cosmos_framework.model.generator.tokenizers.lidar.dtypes import as_torch_dtype
from cosmos_framework.model.generator.tokenizers.lidar.network.transformer_vae import TransformerVAE
from cosmos_framework.model.generator.tokenizers.lidar.normalization import (
    metric_lidar_to_network,
    network_lidar_to_metric_clip,
)
from cosmos_framework.model.generator.tokenizers.lidar.paths import (
    is_remote_uri,
    resolve_credential_path,
)
from cosmos_framework.model.generator.tokenizers.lidar.postprocessing import (
    resolved_validity_channel,
    validate_validity_threshold,
    validity_probability,
)
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    INVALID_NORMALIZED_VALUE,
    MAX_RANGE_METERS,
    MIN_RANGE_METERS,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig

# =============================================================================
# Defaults (3-channel range + intensity + mask tokenizer)
# =============================================================================

DEFAULT_EXPERIMENT_NAME = "lidar_tokenizer_v1"
DEFAULT_INPUT_RESOLUTION = (128, 3600)  # (H, W) native range-map width
DEFAULT_LATENT_SPATIAL = (16, 225)  # H/8, W/16
DEFAULT_SPATIAL_COMPRESSION = (8, 16)  # (H, W)
DEFAULT_IN_CHANNELS = 3  # range, intensity, mask
DEFAULT_OUT_CHANNELS = 3  # range, intensity, mask (mask channel trained as logits)
DEFAULT_LATENT_CH = 128

# Spatial: patch 2x4 + 2 merges -> 8x16. Temporal: no downsample -> 1x.
DEFAULT_NETWORK_CONFIG: dict[str, Any] = {
    "resolution": list(DEFAULT_INPUT_RESOLUTION),
    "in_channels": DEFAULT_IN_CHANNELS,
    "out_channels": DEFAULT_OUT_CHANNELS,
    "z_dim": DEFAULT_LATENT_CH,
    "base_channels": 128,
    "patch_size": [2, 4],
    "window_size": [5, 45],
    "depths": [3, 3, 3],
    "num_heads": [4, 4, 8],
    "dilation": [1, 1, 1],
    "temporal_downsample": [False, False],
    "temporal_upsample": [False, False],
    "mlp_ratio": 3.0,
    "mapping_depth": 2,
    "positional_embedding": "learnable_embedding",
    "formulation": "VAE",
    # Mask is already channel 2 of the input tensor (not derived inside encode).
    # Reconstructed as output channel 2 (logits); no extra validity head.
    "mask_as_input": False,
    "bottleneck_3d": True,
    "bottleneck_3d_max_t": 32,
    "bottleneck_3d_causal_time": True,
    "bottleneck_3d_rope": True,
}


def infer_lidar_compression(network_config: Mapping[str, Any]) -> tuple[tuple[int, int], int]:
    """Infer ``((height, width), time)`` compression from a TransformerVAE config."""
    patch_size = tuple(int(value) for value in network_config["patch_size"])
    depths = tuple(network_config["depths"])
    temporal_downsample = tuple(bool(value) for value in network_config["temporal_downsample"])
    if len(patch_size) != 2 or any(value <= 0 for value in patch_size):
        raise ValueError(f"patch_size must contain two positive values, got {patch_size}")
    if not depths:
        raise ValueError("depths must contain at least one level")
    expected_merges = len(depths) - 1
    if len(temporal_downsample) != expected_merges:
        raise ValueError(
            "temporal_downsample must have one entry per encoder merge, got "
            f"{len(temporal_downsample)} entries for {len(depths)} levels"
        )
    spatial_merge_factor = 2**expected_merges
    spatial = (patch_size[0] * spatial_merge_factor, patch_size[1] * spatial_merge_factor)
    temporal = 2 ** sum(temporal_downsample)
    return spatial, temporal


# =============================================================================
# Interface
# =============================================================================


class LidarTokenizerV1Interface(VideoTokenizerInterface):
    """V1 LiDAR range+intensity+mask tokenizer (H/8 × W/16 spatial / 1× temporal).

    Input: ``[B, 3, T, 128, 3600]`` (metric range, unit intensity, mask).
    Latent: ``[B, 128, T, 16, 225]`` (H/8, W/16).
    Decode: ``[B, 3, T, 128, 3600]`` (metric range, unit intensity, mask).

    The decoded mask channel matches the input convention: ``{0, 1}`` once the
    validity cut is applied, and the underlying probability when it is not. The
    raw logits never leave this class.
    """

    def __init__(
        self,
        vae_path: str | None = None,
        *,
        object_store_credential_path_pretrained: str | None = None,
        bucket_name: str = "",
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = torch.float32,
        sample_posterior: bool = False,
        apply_validity_mask: bool = True,
        network_config: Mapping[str, Any] | None = None,
        load_checkpoint: bool = True,
        spatial_compression_factor: int | None = None,
        spatial_compression: Sequence[int] | None = None,
        temporal_compression_factor: int | None = None,
        range_projection: LidarRangeProjectionConfig | Mapping[str, Any] | None = None,
        streaming_chunk_frames: int | None = None,
        streaming_context_frames: int | None = None,
    ) -> None:
        # Resolve object-store credentials when loading remote artifacts.
        needs_remote = load_checkpoint and vae_path is not None and is_remote_uri(vae_path)
        if object_store_credential_path_pretrained is None and needs_remote:
            object_store_credential_path_pretrained = resolve_credential_path()
        super().__init__(object_store_credential_path_pretrained)

        self.device = torch.device(device)
        self.dtype = as_torch_dtype(dtype)
        self.sample_posterior = sample_posterior
        self.apply_validity_mask = apply_validity_mask
        if streaming_chunk_frames is not None and streaming_chunk_frames < 1:
            raise ValueError(f"streaming_chunk_frames must be positive, got {streaming_chunk_frames}")
        if streaming_context_frames is not None and streaming_context_frames < 1:
            raise ValueError(f"streaming_context_frames must be positive, got {streaming_context_frames}")
        if (
            streaming_chunk_frames is not None
            and streaming_context_frames is not None
            and streaming_context_frames < streaming_chunk_frames
        ):
            raise ValueError("streaming_context_frames must be at least streaming_chunk_frames")
        self.streaming_chunk_frames = streaming_chunk_frames
        self.streaming_context_frames = streaming_context_frames

        # Derive compression from the effective architecture. Explicit metadata
        # remains available to config interpolation, but cannot silently drift.
        config = dict(DEFAULT_NETWORK_CONFIG)
        if network_config is not None:
            config.update(network_config)
        inferred_spatial, inferred_temporal = infer_lidar_compression(config)
        input_resolution = tuple(int(value) for value in config["resolution"])
        if len(input_resolution) != 2 or any(value <= 0 for value in input_resolution):
            raise ValueError(f"resolution must contain two positive values, got {input_resolution}")
        if any(size % factor for size, factor in zip(input_resolution, inferred_spatial, strict=True)):
            raise ValueError(
                f"resolution {input_resolution} must be divisible by spatial compression {inferred_spatial}"
            )
        self._input_resolution = input_resolution
        self._latent_spatial = tuple(
            size // factor for size, factor in zip(input_resolution, inferred_spatial, strict=True)
        )
        spatial_compression = inferred_spatial if spatial_compression is None else tuple(spatial_compression)
        spatial_compression_factor = (
            inferred_spatial[0] if spatial_compression_factor is None else int(spatial_compression_factor)
        )
        temporal_compression_factor = (
            inferred_temporal if temporal_compression_factor is None else int(temporal_compression_factor)
        )
        if (
            spatial_compression_factor != inferred_spatial[0]
            or spatial_compression != inferred_spatial
            or temporal_compression_factor != inferred_temporal
        ):
            raise ValueError(
                "Compression metadata does not match the network architecture: "
                f"expected factor={inferred_spatial[0]}, spatial={inferred_spatial}, temporal={inferred_temporal}; "
                f"got factor={spatial_compression_factor}, spatial={spatial_compression}, "
                f"temporal={temporal_compression_factor}"
            )

        # VideoTokenizerInterface / DiT bookkeeping.
        self._causal = True
        self._spatial_compression_factor = spatial_compression_factor
        self._spatial_compression = spatial_compression
        self._temporal_compression_factor = temporal_compression_factor
        self._pixel_chunk_duration = 9

        # Build network (optionally override architecture for experiments).
        self.model = TransformerVAE(**config)

        # Checkpoint is optional so smoke tests can run with random weights.
        checkpoint_payload: Mapping[str, Any] | None = None
        if load_checkpoint:
            if vae_path is None:
                raise ValueError("vae_path is required when load_checkpoint=True")
            checkpoint_payload = load_model_checkpoint(
                self.model,
                resolve_artifact_path(vae_path, bucket_name),
                backend_args=self.backend_args,
                error_prefix="Incompatible LiDAR tokenizer checkpoint",
            )
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=self.dtype)

        # Latent mean/std ride in the checkpoint, so the affine and the weights
        # it belongs to can never be paired wrongly. Without a checkpoint the
        # affine is the identity, for untrained / smoke-test runs.
        if load_checkpoint:
            assert checkpoint_payload is not None
            latent_mean, latent_std, stored_min_range, stored_max_range = parse_lidar_checkpoint_stats(
                checkpoint_payload
            )
        else:
            latent_mean = torch.zeros(self.latent_ch)
            latent_std = torch.ones(self.latent_ch)
            stored_min_range = stored_max_range = None

        if isinstance(range_projection, Mapping):
            range_projection = LidarRangeProjectionConfig.from_dict(range_projection)
        if range_projection is None:
            min_range = stored_min_range
            max_range = stored_max_range
        else:
            min_range = range_projection.min_range_m
            max_range = range_projection.max_range_m
            if stored_min_range is not None and stored_min_range != min_range:
                raise ValueError(
                    f"Configured min_range_m {min_range} does not match checkpoint value {stored_min_range}"
                )
            if stored_max_range is not None and stored_max_range != max_range:
                raise ValueError(
                    f"Configured max_range_m {max_range} does not match checkpoint value {stored_max_range}"
                )
        if min_range is None or max_range is None:
            if load_checkpoint:
                warnings.warn(
                    f"LiDAR range metadata is absent; assuming the default "
                    f"[{MIN_RANGE_METERS}, {MAX_RANGE_METERS}] m span. "
                    "Pass range_projection for checkpoints trained on another span.",
                    stacklevel=2,
                )
            min_range = MIN_RANGE_METERS if min_range is None else min_range
            max_range = MAX_RANGE_METERS if max_range is None else max_range
        if range_projection is None:
            range_projection = LidarRangeProjectionConfig(
                semantic_width=input_resolution[1],
                model_width=input_resolution[1],
                min_range_m=min_range,
                max_range_m=max_range,
            )
        if (range_projection.native_height, range_projection.model_width) != input_resolution:
            raise ValueError(
                "Range projection dimensions do not match the tokenizer architecture: "
                f"projection={(range_projection.native_height, range_projection.model_width)}, "
                f"network={input_resolution}"
            )
        self.range_projection = range_projection
        self.min_range = range_projection.min_range_m
        self.max_range = range_projection.max_range_m
        self.validity_threshold = validate_validity_threshold(range_projection.validity_threshold)

        if latent_mean.numel() != self.latent_ch or latent_std.numel() != self.latent_ch:
            raise ValueError(
                "Latent statistics must have one value per channel: "
                f"expected {self.latent_ch}, got mean={latent_mean.numel()} and std={latent_std.numel()}"
            )
        if torch.any(latent_std <= 0):
            raise ValueError("Latent standard deviations must be positive")

        stats_shape = (1, self.latent_ch, 1, 1, 1)
        self.latent_mean = latent_mean.to(device=self.device, dtype=self.dtype).reshape(stats_shape)
        self.latent_std = latent_std.to(device=self.device, dtype=self.dtype).reshape(stats_shape)

    # -------------------------------------------------------------------------
    # Encode / decode
    # -------------------------------------------------------------------------

    def reset_dtype(self) -> None:
        self.model.to(device=self.device, dtype=self.dtype)
        self.latent_mean = self.latent_mean.to(device=self.device, dtype=self.dtype)
        self.latent_std = self.latent_std.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode metric range, unit intensity, and mask to normalized latents."""
        if self.streaming_chunk_frames is not None:
            return self.encode_streaming(state, chunk_frames=self.streaming_chunk_frames)
        state = state.to(device=self.device, dtype=self.dtype)
        normalized, _ = metric_lidar_to_network(state, min_range=self.min_range, max_range=self.max_range)
        return self.encode_normalized(normalized)

    @torch.inference_mode()
    def encode_normalized(self, state: torch.Tensor) -> torch.Tensor:
        """Encode legacy normalized ``[B,3,T,H,W]`` tensors."""
        if (
            state.ndim != 5
            or state.shape[1] != DEFAULT_IN_CHANNELS
            or tuple(state.shape[-2:]) != self._input_resolution
        ):
            raise ValueError(
                f"Expected [B,{DEFAULT_IN_CHANNELS},T,"
                f"{self._input_resolution[0]},{self._input_resolution[1]}], "
                f"got {tuple(state.shape)}"
            )

        state = state.to(device=self.device, dtype=self.dtype)
        sample, (posterior_mean, _) = self.model.encode(state)
        latent = sample if self.sample_posterior else posterior_mean
        latent = latent.to(dtype=self.dtype)

        return (latent - self.latent_mean) / self.latent_std

    @torch.inference_mode()
    def encode_streaming(self, state: torch.Tensor, *, chunk_frames: int | None = None) -> torch.Tensor:
        """Chunked metric-input encode for long videos."""
        state = state.to(device=self.device, dtype=self.dtype)
        normalized, _ = metric_lidar_to_network(state, min_range=self.min_range, max_range=self.max_range)
        return self.encode_streaming_normalized(normalized, chunk_frames=chunk_frames)

    @torch.inference_mode()
    def encode_streaming_normalized(self, state: torch.Tensor, *, chunk_frames: int | None = None) -> torch.Tensor:
        """Chunked encode for legacy normalized tensors."""
        if (
            state.ndim != 5
            or state.shape[1] != DEFAULT_IN_CHANNELS
            or tuple(state.shape[-2:]) != self._input_resolution
        ):
            raise ValueError(
                f"Expected [B,{DEFAULT_IN_CHANNELS},T,"
                f"{self._input_resolution[0]},{self._input_resolution[1]}], "
                f"got {tuple(state.shape)}"
            )
        chunk = self.pixel_chunk_duration if chunk_frames is None else int(chunk_frames)
        state = state.to(device=self.device, dtype=self.dtype)
        latent = self.model.encode_streaming(
            state,
            chunk_frames=chunk,
            context_frames=self.streaming_context_frames,
            sample_posterior=self.sample_posterior,
        )
        return (latent.to(dtype=self.dtype) - self.latent_mean) / self.latent_std

    def _resolve_validity_threshold(self, override: float | None) -> float:
        """Per-call probability cut, falling back to the configured one."""
        return self.validity_threshold if override is None else validate_validity_threshold(override)

    def _split_decode_output(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split decoder output into data(+mask logits) and sigmoid validity."""
        if output.shape[1] < 3:
            raise ValueError(f"Expected >=3 decode channels, got {tuple(output.shape)}")
        validity = validity_probability(output[:, 2:3].to(dtype=output.dtype))
        return output, validity

    @torch.inference_mode()
    def decode_normalized(
        self,
        latent: torch.Tensor,
        *,
        return_validity: bool = False,
        apply_mask: bool | None = None,
        validity_threshold: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Decode to the legacy normalized range/intensity representation."""
        if latent.ndim != 5 or latent.shape[1] != self.latent_ch:
            raise ValueError(f"Expected [B,{self.latent_ch},T,H,W], got {tuple(latent.shape)}")

        latent = latent.to(device=self.device, dtype=self.dtype)
        denormalized = latent * self.latent_std + self.latent_mean

        # Align decode crop target with this latent's T (1x temporal compression).
        self.model._input_T = int(latent.shape[2])
        output = self.model.decode(denormalized, return_validity=False)
        output, validity = self._split_decode_output(output)

        should_mask = self.apply_validity_mask if apply_mask is None else apply_mask
        threshold = self._resolve_validity_threshold(validity_threshold)
        output = output.clone()
        if should_mask:
            output[:, :2] = output[:, :2].masked_fill(validity < threshold, INVALID_NORMALIZED_VALUE)
        output[:, 2:3] = resolved_validity_channel(validity, should_mask=should_mask, threshold=threshold)

        if return_validity:
            return output, validity
        return output

    @torch.inference_mode()
    def decode(
        self,
        latent: torch.Tensor,
        *,
        return_validity: bool = False,
        apply_mask: bool | None = None,
        validity_threshold: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Decode to metric range, unit intensity, and resolved mask."""
        if self.streaming_chunk_frames is not None:
            return self.decode_streaming(
                latent,
                chunk_frames=self.streaming_chunk_frames,
                return_validity=return_validity,
                apply_mask=apply_mask,
                validity_threshold=validity_threshold,
            )
        normalized, validity = self.decode_normalized(latent, return_validity=True, apply_mask=False)
        assert validity is not None
        should_mask = self.apply_validity_mask if apply_mask is None else apply_mask
        threshold = self._resolve_validity_threshold(validity_threshold)
        output = network_lidar_to_metric_clip(
            normalized,
            validity,
            min_range=self.min_range,
            max_range=self.max_range,
            apply_validity_mask=should_mask,
            validity_threshold=threshold,
        )
        if return_validity:
            return output, validity
        return output

    @torch.inference_mode()
    def decode_streaming_normalized(
        self,
        latent: torch.Tensor,
        *,
        chunk_frames: int | None = None,
        return_validity: bool = False,
        apply_mask: bool | None = None,
        validity_threshold: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Chunked decode to the legacy normalized representation."""
        if latent.ndim != 5 or latent.shape[1] != self.latent_ch:
            raise ValueError(f"Expected [B,{self.latent_ch},T,H,W], got {tuple(latent.shape)}")
        chunk = self.latent_chunk_duration if chunk_frames is None else int(chunk_frames)
        latent = latent.to(device=self.device, dtype=self.dtype)
        denormalized = latent * self.latent_std + self.latent_mean
        self.model._input_T = int(latent.shape[2])
        output = self.model.decode_streaming(
            denormalized,
            chunk_frames=chunk,
            context_frames=self.streaming_context_frames,
            return_validity=False,
        )
        output, validity = self._split_decode_output(output)
        should_mask = self.apply_validity_mask if apply_mask is None else apply_mask
        threshold = self._resolve_validity_threshold(validity_threshold)
        output = output.clone()
        if should_mask:
            output[:, :2] = output[:, :2].masked_fill(validity < threshold, INVALID_NORMALIZED_VALUE)
        output[:, 2:3] = resolved_validity_channel(validity, should_mask=should_mask, threshold=threshold)
        if return_validity:
            return output, validity
        return output

    @torch.inference_mode()
    def decode_streaming(
        self,
        latent: torch.Tensor,
        *,
        chunk_frames: int | None = None,
        return_validity: bool = False,
        apply_mask: bool | None = None,
        validity_threshold: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Chunked decode to metric range, unit intensity, and resolved mask."""
        normalized, validity = self.decode_streaming_normalized(
            latent,
            chunk_frames=chunk_frames,
            return_validity=True,
            apply_mask=False,
        )
        assert validity is not None
        should_mask = self.apply_validity_mask if apply_mask is None else apply_mask
        threshold = self._resolve_validity_threshold(validity_threshold)
        output = network_lidar_to_metric_clip(
            normalized,
            validity,
            min_range=self.min_range,
            max_range=self.max_range,
            apply_validity_mask=should_mask,
            validity_threshold=threshold,
        )
        if return_validity:
            return output, validity
        return output

    # -------------------------------------------------------------------------
    # Frame / compression bookkeeping
    # -------------------------------------------------------------------------

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return num_pixel_frames

    def get_pixel_num_frames(self, num_latent_frames: int, **kwargs: Any) -> int:
        del kwargs
        return num_latent_frames

    @property
    def spatial_compression_factor(self) -> int:
        """Height compression (interface API is a single int). Prefer :meth:`spatial_compression`."""
        return self._spatial_compression_factor

    @property
    def spatial_compression(self) -> tuple[int, int]:
        """``(H, W)`` spatial compression factors (8, 16)."""
        return self._spatial_compression

    @property
    def temporal_compression_factor(self) -> int:
        return self._temporal_compression_factor

    @property
    def spatial_resolution(self) -> int:
        return self._input_resolution[0]

    @property
    def input_resolution(self) -> tuple[int, int]:
        return self._input_resolution

    @property
    def latent_spatial(self) -> tuple[int, int]:
        return self._latent_spatial

    @property
    def pixel_chunk_duration(self) -> int:
        return self._pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self._pixel_chunk_duration

    @property
    def latent_ch(self) -> int:
        return DEFAULT_LATENT_CH

    @property
    def name(self) -> str:
        return DEFAULT_EXPERIMENT_NAME
