# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dense wrapper runtime for frozen tokenizer encode/decode."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos_framework.model.tokenizer.models.dense_backends import (
    DenseResolvedBackend,
    DenseRuntimeBackend,
    resolve_dense_backend,
    run_batched_block_stack,
    run_varlen_block_stack,
)
from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import LearnedPositionEmbedder
from cosmos_framework.model.tokenizer.models.sparse_autoencoder import AutoencoderKL, SparseTransformerBase


@dataclass(frozen=True)
class DenseTemporalChunkSpec:
    """Temporal chunk configuration for the dense runtime."""

    raw_frames: int
    patch_frames: int


@dataclass(frozen=True)
class DenseGridMetadata:
    """Precomputed dense-grid metadata shared across chunk executions."""

    batch_size: int
    temporal_patches: int
    height_patches: int
    width_patches: int
    learned_pe: torch.Tensor | None
    rope_freqs_cis: torch.Tensor | None
    cu_seqlens: torch.Tensor
    q_seqlen: list[int]
    max_seq_len: int


DenseGridMetadataKey = tuple[str, int, int, int, int, str, str]


class DenseDiagonalGaussianDistribution:
    """Diagonal Gaussian posterior for dense channels-last latent tensors."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        """Initialize the dense posterior from `[mean, logvar]` moments."""
        if parameters.ndim not in (4, 5):
            raise ValueError(
                "DenseDiagonalGaussianDistribution expects 4D/5D channels-last moments, "
                f"got shape {tuple(parameters.shape)}."
            )
        self.original_dtype = parameters.dtype
        self.parameters = parameters.to(torch.float32)
        self.mean, self.logvar = torch.chunk(self.parameters, 2, dim=-1)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )

    def sample(self) -> torch.Tensor:
        """Sample a dense channels-last latent tensor."""
        sample = torch.randn_like(self.mean)
        return (self.mean + self.std * sample).to(self.original_dtype)

    def kl(self, other: "DenseDiagonalGaussianDistribution" | None = None) -> torch.Tensor:
        """Compute KL divergence per latent token, matching sparse scaling."""
        reduce_dims = (-1,)
        if self.deterministic:
            num_tokens = math.prod(self.mean.shape[:-1])
            return torch.zeros(num_tokens, device=self.parameters.device, dtype=self.parameters.dtype)
        if other is None:
            kl = 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=reduce_dims)
        else:
            kl = 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=reduce_dims,
            )
        return kl.reshape(-1)


class DenseAutoencoderRuntime(nn.Module):
    """Dense frozen-runtime wrapper around an existing sparse autoencoder.

    The wrapper intentionally holds the original ``AutoencoderKL`` as a single
    registered submodule and only exposes compile-friendly dense orchestration.
    Backend math is added incrementally in follow-up changes.
    """

    autoencoder: AutoencoderKL
    backend: DenseRuntimeBackend
    _metadata_cache: dict[DenseGridMetadataKey, DenseGridMetadata]

    def __init__(
        self,
        autoencoder: AutoencoderKL,
        backend: DenseRuntimeBackend = "auto",
        pad_frames: int = 0,
        pixel_trim: bool = True,
        chunk_size: int = 16,
    ) -> None:
        """Initialize the dense runtime wrapper.

        Args:
            autoencoder: The sparse autoencoder to wrap.
            backend: Backend selection for block-stack execution.
            pad_frames: Number of boundary frames to replicate at each end of
                every temporal chunk before encoding.  Must be divisible by
                ``patch_size[0]``.  Set ``0`` to disable boundary padding;
                set ``>0`` (typically one temporal patch, e.g. ``4``) to give
                the non-causal encoder additional context across chunk edges,
                eliminating the per-chunk-boundary PSNR dip.
            pixel_trim: When ``True`` and ``pad_frames > 0``, boundary latents
                are kept in the encoded output and trimmed in pixel space after
                decoding.  When ``False``, boundary latents are trimmed
                immediately after encoding.  ``True`` should always be used
                for the best reconstruction quality.
            chunk_size: Number of *raw* frames consumed by the encoder per
                temporal chunk.  Forwarded to
                ``autoencoder.num_sample_frames_batch_size`` and used to
                slice the input video into encode batches.  Must satisfy
                ``2 * pad_frames < chunk_size``.  Default ``16``.
        """
        super().__init__()
        self.autoencoder = autoencoder
        self.backend = backend
        autoencoder.num_sample_frames_batch_size = chunk_size
        if pad_frames < 0:
            raise ValueError(f"pad_frames must be non-negative, got {pad_frames}.")
        if 2 * pad_frames >= chunk_size:
            raise ValueError(f"pad_frames must be less than chunk_size / 2, got {pad_frames=}, {chunk_size=}.")
        self.pad_frames = pad_frames
        self.pixel_trim = pixel_trim
        self._metadata_cache: dict[DenseGridMetadataKey, DenseGridMetadata] = {}
        self.cg_compiled = False

    @classmethod
    def from_autoencoder(
        cls,
        autoencoder: AutoencoderKL,
        backend: DenseRuntimeBackend = "auto",
        pad_frames: int = 0,
        pixel_trim: bool = True,
        chunk_size: int = 16,
    ) -> "DenseAutoencoderRuntime":
        """Build a dense runtime from a supported sparse autoencoder."""
        cls._validate_autoencoder(autoencoder)
        return cls(
            autoencoder=autoencoder,
            backend=backend,
            pad_frames=pad_frames,
            pixel_trim=pixel_trim,
            chunk_size=chunk_size,
        )

    @staticmethod
    def _validate_autoencoder(autoencoder: AutoencoderKL) -> None:
        """Validate that the sparse autoencoder fits the dense-runtime V1 scope."""
        if not hasattr(autoencoder, "decoder"):
            raise ValueError("Dense runtime V1 requires use_decoder=True.")

        encoder = autoencoder.encoder
        decoder = autoencoder.decoder

        if encoder.concat_latent is not None:
            raise ValueError("Dense runtime V1 does not support concat_latent.")
        if autoencoder.use_dual_latent:
            raise ValueError("Dense runtime V1 does not support dual latent.")
        if decoder.multiscale is not None or decoder.multiscale_outputs is not None:
            raise ValueError("Dense runtime V1 does not support decoder multiscale outputs.")
        if any(getattr(block, "multiscale", None) is not None for block in encoder.blocks):
            raise ValueError("Dense runtime V1 does not support encoder multiscale blocks.")
        if any(getattr(block, "multiscale", None) is not None for block in decoder.blocks):
            raise ValueError("Dense runtime V1 does not support decoder multiscale blocks.")
        if encoder.pe_mode not in {"joint", "learned"}:
            raise ValueError(f"Dense runtime V1 currently requires encoder learned/joint PE, got {encoder.pe_mode}.")
        if decoder.pe_mode not in {"joint", "learned"}:
            raise ValueError(f"Dense runtime V1 currently requires decoder learned/joint PE, got {decoder.pe_mode}.")

    @property
    def patch_size(self) -> tuple[int, int, int]:
        """Return the tokenizer patch size."""
        patch_size = self.autoencoder.patch_size
        return int(patch_size[0]), int(patch_size[1]), int(patch_size[2])

    @property
    def patch_volume(self) -> int:
        """Return the tokenizer patch volume."""
        return math.prod(self.patch_size)

    @property
    def encoder_chunk_spec(self) -> DenseTemporalChunkSpec:
        """Return the fixed encoder chunk configuration used in eval mode."""
        raw_frames = int(self.autoencoder.num_sample_frames_batch_size)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    @property
    def decoder_window_spec(self) -> DenseTemporalChunkSpec:
        """Return the decoder inference window configuration."""
        raw_frames = int(self.autoencoder.inference_num_sample_frames_batch_size)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    @property
    def decoder_stride_spec(self) -> DenseTemporalChunkSpec:
        """Return the decoder inference stride configuration."""
        raw_frames = int(self.autoencoder.inference_num_sample_frames_stride)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    @property
    def decoder_cache_spec(self) -> DenseTemporalChunkSpec:
        """Return the decoder inference cache configuration."""
        raw_frames = int(self.autoencoder.inference_kv_cache_size)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    def resolve_backend(self, use_compile: bool = False) -> DenseResolvedBackend:
        """Resolve the backend for the current execution mode."""
        return resolve_dense_backend(self.backend, use_compile=use_compile)

    def clear_metadata_cache(self) -> None:
        """Drop cached dense-grid metadata."""
        self._metadata_cache.clear()

    def encode(
        self,
        dense_video: torch.Tensor,
        sample_posterior: bool = False,
        pad_to: int | None = None,
        chunk_raw_frames: int | None = None,
        encode_chunk_batch_size: int = 1,
    ) -> torch.Tensor:
        """Encode a dense video tensor into latent moments or posterior samples."""
        moments = self.encode_moments(
            dense_video,
            chunk_raw_frames=chunk_raw_frames,
            pad_to=pad_to,
            encode_chunk_batch_size=encode_chunk_batch_size,
        )
        if not sample_posterior:
            return moments
        return self._sample_dense_posterior(moments)

    def encode_moments(
        self,
        video: torch.Tensor,
        chunk_raw_frames: int | None = None,
        pad_to: int | None = None,
        encode_chunk_batch_size: int = 1,
    ) -> torch.Tensor:
        """Encode a dense video tensor into `[B, T_p, H_p, W_p, 2C]` latent moments.

        Args:
            video: Dense channels-last video tensor ``[B, T, H, W, 3]``.
            chunk_raw_frames: Number of raw frames per encoder chunk.  Defaults
                to ``self.encoder_chunk_spec.raw_frames``.
            pad_to: Sequence-length padding target for the ``batched_with_padding``
                backend (reduces CUDA graph recapture).
            encode_chunk_batch_size: Number of full temporal chunks to encode
                together. Only supported for the ``batched`` backend; defaults
                to ``1`` (sequential encoding).

        Shapes (example):
            Config: ``patch_size = (1, 16, 16)``, ``chunk_size = 16``,
            ``pad_frames = 1`` (1 raw frame replicated on each chunk edge).
            Whole-video input: ``[B=1, T=28, H=480, W=832, 3]``.

            Per-chunk pipeline (loop slices the 28 frames into 2 chunks of
            ``chunk_raw_frames = 16 - 2*1 = 14``):

              ::

                step                              shape                          notes
                ---------------------------------------------------------------------------------------
                1. raw chunk                      [1, 14, 480, 832, 3]           1 of 2 chunks
                2. after input padding            [1, 16, 480, 832, 3]           1 pre + 14 raw + 1 post
                3. after encoding (latent)        [1, 16,  30,  52, 2C]          T_p=16/1, H_p=480/16, W_p=832/16
                4. after decoding                 [1, 16, 480, 832, 3]
                5. after pixel trim               [1, 14, 480, 832, 3]           drops pad_frames=1 pixel frame
                                                                                 on each end

            Across both chunks the concatenated pixel-space output is
            ``[1, 28, 480, 832, 3]``; the latent fed to a downstream DiT is
            ``[1, 32, 30, 52, 2C]``.

            For images (``T = 1``) the input is repeated to one temporal patch
            (``T = patch_time``) and ``latents_per_boundary = 0``, so the
            DiT-facing shape is ``[B, 1, H_p, W_p, 2C]``.
        """
        if video.ndim != 5:
            raise ValueError(f"Dense runtime expects 5D video tensor, got {video.ndim}D")
        if video.shape[4] != 3:
            raise ValueError(f"Dense runtime expects video tensor with 3 channels, got {video.shape[4]}")

        batch_size, raw_frames, height, width, _ = video.shape
        patch_time, patch_height, patch_width = self.patch_size
        assert batch_size == 1 or encode_chunk_batch_size == 1, (
            "Dense runtime with batching currently only supports batch size 1"
        )

        if chunk_raw_frames is None:
            chunk_raw_frames = self.encoder_chunk_spec.raw_frames
            chunk_raw_frames = chunk_raw_frames - 2 * self.pad_frames
            assert chunk_raw_frames > 0, (
                f"Padding frames must be less than chunk_raw_frames, got {chunk_raw_frames=}, {self.pad_frames=}."
            )
        if chunk_raw_frames <= 0:
            raise ValueError(f"chunk_raw_frames must be positive, got {chunk_raw_frames}.")
        if encode_chunk_batch_size < 1:
            raise ValueError(f"encode_chunk_batch_size must be positive, got {encode_chunk_batch_size}.")
        if encode_chunk_batch_size > 1 and self.backend != "batched":
            raise ValueError(
                f"encode_chunk_batch_size > 1 is only supported for the batched backend, got backend={self.backend!r}."
            )

        # if input is an image, we pad to form single temporal patch
        if raw_frames == 1:
            is_image = True
            video = video.repeat(1, patch_time, 1, 1, 1)
            raw_frames = patch_time
        else:
            is_image = False

        if (chunk_raw_frames + 2 * self.pad_frames) % patch_time != 0:
            raise ValueError(
                f"chunk_raw_frames + 2 * pad_frames must be divisible by patch_size[0]={patch_time}, got {chunk_raw_frames=}, {self.pad_frames=}."
            )

        if not is_image:
            # Noncausal scheme: first frame is its own chunk; remaining must fill complete regular chunks.
            remaining_frames = raw_frames - 1
            remainder = remaining_frames % chunk_raw_frames
            if remainder != 0 and (remainder + 2 * self.pad_frames) % patch_time != 0:
                raise ValueError(
                    f"Dense runtime requires (frame_count - 1) equal to "
                    f"chunk_raw_frames * N + patch_time - 2 * pad_frames, "
                    f"got {raw_frames=}, {chunk_raw_frames=}, {self.pad_frames=}, {patch_time=}."
                )
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                "Dense runtime requires spatial dimensions divisible by patch size "
                f"{(patch_height, patch_width)}, got {(height, width)}."
            )
        pad_frames = self.pad_frames
        if not is_image:
            latents_per_boundary = pad_frames // patch_time
        else:
            latents_per_boundary = 0

        del batch_size

        # preserve the chunk size to reduce number of captured cuda graphs
        if self.backend == "batched_with_padding" and pad_to is None and self.cg_compiled:
            width_patches = width // patch_width
            height_patches = height // patch_height
            padded_chunk_frames = chunk_raw_frames + 2 * pad_frames
            temporal_patches = padded_chunk_frames // patch_time
            pad_to = width_patches * height_patches * temporal_patches

        use_chunk_batching = self.backend == "batched" and encode_chunk_batch_size > 1

        def _pad_video_chunk(video_chunk: torch.Tensor) -> torch.Tensor:
            if pad_frames > 0 and not is_image:
                # UniAE chunk-wise encoding suffers a PSNR dip at chunk boundaries
                # because the non-causal encoder lacks context beyond the chunk edges.
                # Padding each chunk with pad_frames replicated boundary frames on both
                # sides gives the encoder that context, eliminating the boundary dip.
                # In practice pad_frames=4 (one temporal patch) is used.
                # The corresponding boundary latents are trimmed after decoding
                # (see pixel_trim / latents_per_boundary below).
                pre = video_chunk[:, 0:1].expand(-1, pad_frames, -1, -1, -1)  # [B,pad,H,W,3]
                post = video_chunk[:, -1:].expand(-1, pad_frames, -1, -1, -1)  # [B,pad,H,W,3]
                video_chunk = torch.cat([pre, video_chunk, post], dim=1)  # [B,t+2*pad,H,W,3]
            return video_chunk

        def _trim_boundary_latents(encoded_chunk: torch.Tensor) -> torch.Tensor:
            if latents_per_boundary > 0 and not self.pixel_trim:
                t_latent = encoded_chunk.shape[1]
                encoded_chunk = encoded_chunk[:, latents_per_boundary : t_latent - latents_per_boundary]
            return encoded_chunk

        def _encode_padded_chunks(padded_chunks: list[torch.Tensor]) -> list[torch.Tensor]:
            if len(padded_chunks) == 1:
                encoded = self._encode_video_chunk(padded_chunks[0], pad_to=pad_to)
                return [_trim_boundary_latents(encoded)]

            batched_video = torch.cat(padded_chunks, dim=0)  # [B*G,t_pad,H,W,3]
            encoded = self._encode_video_chunk(batched_video, pad_to=pad_to)  # [B*G,T_lat,Hp,Wp,2C]
            per_video_batch = padded_chunks[0].shape[0]
            return list(_trim_boundary_latents(encoded).split(per_video_batch, dim=0))

        encoded_chunks: list[torch.Tensor] = []

        if not is_image:
            # Noncausal first chunk: encode frame 0 alone, padded to patch_time copies
            # at the head so the encoder sees exactly patch_time frames → 1 latent L₁.
            # pad_to=None: this chunk has 1 temporal patch, not the regular chunk shape.
            first_frame = video[:, 0:1]
            first_chunk = first_frame.expand(-1, patch_time, -1, -1, -1).contiguous()
            encoded_chunks.append(self._encode_video_chunk(first_chunk, pad_to=None))

        chunk_specs = [
            (
                start_frame,
                end_frame := min(start_frame + chunk_raw_frames, raw_frames),
                end_frame - start_frame == chunk_raw_frames,
            )
            for start_frame in range(0 if is_image else 1, raw_frames, chunk_raw_frames)
        ]

        pending_full_chunks: list[torch.Tensor] = []
        for start_frame, end_frame, is_full_chunk in chunk_specs:
            padded_chunk = _pad_video_chunk(video[:, start_frame:end_frame])  # [B,t_pad,H,W,3]

            if not use_chunk_batching or not is_full_chunk:
                if pending_full_chunks:
                    encoded_chunks.extend(_encode_padded_chunks(pending_full_chunks))
                    pending_full_chunks = []
                encoded_chunks.extend(_encode_padded_chunks([padded_chunk]))
                continue

            pending_full_chunks.append(padded_chunk)
            if len(pending_full_chunks) == encode_chunk_batch_size:
                encoded_chunks.extend(_encode_padded_chunks(pending_full_chunks))
                pending_full_chunks = []

        if pending_full_chunks:
            encoded_chunks.extend(_encode_padded_chunks(pending_full_chunks))

        return torch.cat(encoded_chunks, dim=1)

    def decode(
        self,
        dense_latent: torch.Tensor,
        chunk_raw_frames: int | None = None,
    ) -> torch.Tensor:
        """Decode a dense latent grid into a dense channels-last video tensor.

        When ``pixel_trim`` is enabled and ``pad_frames > 0``, the latent
        contains boundary tokens from encoding.  After decoding, the
        corresponding boundary pixel frames are trimmed from each chunk.

        **Output shape contract**:
        - Video (``temporal_patches > 1``): ``[B, T, H, W, C]`` where T is the
          total number of decoded pixel frames across all chunks (after trim).
        - Image (``temporal_patches == 1``): ``[B, 1, H, W, C]``.  The image
          latent is decoded into ``patch_time`` identical frames (it was encoded
          from ``patch_time`` copies of the same frame); only the last frame is
          kept.  This differs from pre-``dense_runtime`` behaviour where the
          full ``[B, patch_time, H, W, C]`` was returned.
        """
        if self.decoder_cache_spec.patch_frames != 0:
            raise NotImplementedError("Dense runtime decoder V1 does not support KV cache.")

        latent = self._canonicalize_dense_latent(dense_latent)
        temporal_patches = latent.shape[1]
        if chunk_raw_frames is None:
            chunk_patch_frames = self.decoder_window_spec.patch_frames
        else:
            if chunk_raw_frames <= 0:
                raise ValueError(f"chunk_raw_frames must be positive, got {chunk_raw_frames}.")
            if chunk_raw_frames % self.patch_size[0] != 0:
                raise ValueError(
                    f"chunk_raw_frames must be divisible by patch_size[0]={self.patch_size[0]}, got {chunk_raw_frames}."
                )
            chunk_patch_frames = chunk_raw_frames // self.patch_size[0]

        pad_frames = self.pad_frames
        trim_pixel = self.pixel_trim and pad_frames > 0

        # Images were encoded as a single latent (no noncausal first chunk).
        # Videos have temporal_patches > 1: latent[0] is the noncausal first frame.
        is_image = temporal_patches == 1

        # Patch 0 is always a single-latent chunk — either the noncausal first
        # frame (video) or the sole image latent.  Both were encoded from
        # [frame × patch_time] copies, so all decoded frames are equivalent;
        # keep the last one.  For images temporal_patches == 1, so the loop
        # below is empty and this is the only chunk.
        decoded_chunks: list[torch.Tensor] = []
        decoded_first = self._decode_latent_chunk(latent[:, 0:1])  # [B, patch_time, H, W, C]
        decoded_chunks.append(decoded_first[:, -1:])

        for start_patch in range(1, temporal_patches, chunk_patch_frames):
            end_patch = min(start_patch + chunk_patch_frames, temporal_patches)
            latent_chunk = latent[:, start_patch:end_patch]
            decoded_chunk = self._decode_latent_chunk(latent_chunk)
            # Images have no boundary padding, so pixel trim only applies to video chunks.
            if trim_pixel and not is_image:
                decoded_chunk = decoded_chunk[:, pad_frames:-pad_frames]
            decoded_chunks.append(decoded_chunk)
        return torch.cat(decoded_chunks, dim=1)

    def _metadata_cache_key(
        self,
        module_name: str,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DenseGridMetadataKey:
        """Build a stable metadata-cache key for one dense grid shape."""
        return (
            module_name,
            int(batch_size),
            int(temporal_patches),
            int(height_patches),
            int(width_patches),
            str(device),
            str(dtype),
        )

    def _raw_frames_to_patch_frames(self, raw_frames: int) -> int:
        """Convert raw video frames into temporal patch steps."""
        patch_time = self.patch_size[0]
        if raw_frames % patch_time != 0:
            raise ValueError(
                f"Dense runtime requires raw frame counts divisible by patch_size[0]={patch_time}, got {raw_frames}."
            )
        return raw_frames // patch_time

    def _canonicalize_dense_latent(self, dense_latent: torch.Tensor) -> torch.Tensor:
        """Normalize latent tensors to channels-last `[B, T_p, H_p, W_p, C]` format."""
        expected_channels = self.autoencoder.latent_channels
        patch_time = self.patch_size[0]
        if dense_latent.ndim == 5:
            channels_last_match = dense_latent.shape[-1] == expected_channels
            channels_first_match = dense_latent.shape[1] == expected_channels
            if channels_last_match and channels_first_match:
                raise ValueError(
                    "Dense runtime cannot infer 5D latent layout when both the channel-last and "
                    "channel-first dimensions match the expected channel count "
                    f"{expected_channels}; got shape {tuple(dense_latent.shape)}."
                )
            if channels_last_match:
                latent = dense_latent
            elif channels_first_match:
                latent = rearrange(dense_latent, "b c t h w -> b t h w c")
            else:
                raise ValueError(
                    "Dense runtime expects 5D latents in `[B, T, H, W, C]` or `[B, C, T, H, W]` format, "
                    f"got shape {tuple(dense_latent.shape)} with expected channels={expected_channels}."
                )
        elif dense_latent.ndim == 4:
            if patch_time != 1:
                raise ValueError(
                    "Dense runtime image latents are only supported when patch_size[0] == 1, "
                    f"got patch_size[0]={patch_time}."
                )
            channels_last_match = dense_latent.shape[-1] == expected_channels
            channels_first_match = dense_latent.shape[1] == expected_channels
            if channels_last_match and channels_first_match:
                raise ValueError(
                    "Dense runtime cannot infer 4D latent layout when both the channel-last and "
                    "channel-first dimensions match the expected channel count "
                    f"{expected_channels}; got shape {tuple(dense_latent.shape)}."
                )
            if channels_last_match:
                latent = dense_latent.unsqueeze(1)
            elif channels_first_match:
                latent = rearrange(dense_latent, "b c h w -> b 1 h w c")
            else:
                raise ValueError(
                    "Dense runtime expects 4D latents in `[B, H, W, C]` or `[B, C, H, W]` format, "
                    f"got shape {tuple(dense_latent.shape)} with expected channels={expected_channels}."
                )
        else:
            raise ValueError(
                "Dense runtime expects latent inputs with 4 or 5 dimensions, "
                f"got tensor with shape {tuple(dense_latent.shape)}."
            )
        return latent.contiguous()

    def _encode_video_chunk(
        self,
        dense_video_chunk: torch.Tensor,
        pad_to: int | None = None,
    ) -> torch.Tensor:
        """Encode one dense video chunk into projected latent moments."""
        assert pad_to is None or self.backend == "batched_with_padding", (
            "pad_to is only supported for batched_with_padding backend"
        )

        batch_size, raw_frames, height, width, _ = dense_video_chunk.shape
        patch_time, patch_height, patch_width = self.patch_size
        temporal_patches = raw_frames // patch_time
        height_patches = height // patch_height
        width_patches = width // patch_width
        seq_len = temporal_patches * height_patches * width_patches

        patch_feats = self._patchify_dense_video(dense_video_chunk)
        metadata = self._get_or_build_grid_metadata(
            module_name="encoder",
            module=self.autoencoder.encoder,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=patch_feats.device,
            dtype=self.autoencoder.encoder.input_layer.weight.dtype,
        )

        learned_pe = metadata.learned_pe
        rope_freqs_cis = metadata.rope_freqs_cis

        needs_padding = pad_to is not None and pad_to > seq_len
        if pad_to is not None and pad_to < seq_len:
            raise ValueError(f"pad_to ({pad_to}) must be >= sequence length ({seq_len}).")
        if needs_padding:
            if batch_size != 1:
                raise ValueError(
                    f"pad_to requires batch_size=1 for correct varlen masking, got batch_size={batch_size}."
                )
            pad_amount = pad_to - seq_len
            patch_feats = F.pad(patch_feats, (0, 0, 0, pad_amount))
            if learned_pe is not None:
                learned_pe = F.pad(learned_pe, (0, 0, 0, pad_amount))
            if rope_freqs_cis is not None:
                rope_pad = torch.zeros(
                    pad_amount,
                    rope_freqs_cis.shape[-1],
                    dtype=rope_freqs_cis.dtype,
                    device=rope_freqs_cis.device,
                )
                rope_freqs_cis = torch.cat([rope_freqs_cis, rope_pad], dim=0)

        moments = self._encode_chunk_core(
            patch_feats,
            learned_pe=learned_pe,
            rope_freqs_cis=rope_freqs_cis,
            q_seqlen=metadata.q_seqlen,
            cu_seqlens_q=metadata.cu_seqlens,
            max_q_seqlen=metadata.max_seq_len if not needs_padding else pad_to,
        )

        if needs_padding:
            moments = moments[:, :seq_len]

        if self.cg_compiled:
            moments = moments.clone()
        return moments.reshape(batch_size, temporal_patches, height_patches, width_patches, -1)

    def _encode_chunk_core(
        self,
        patch_feats: torch.Tensor,
        learned_pe: torch.Tensor | None,
        rope_freqs_cis: torch.Tensor | None,
        q_seqlen: list[int] | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        max_q_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Encode one dense `[B, S, patch_dim]` chunk into projected latent moments."""
        encoder = self.autoencoder.encoder
        input_dtype = encoder.input_layer.weight.dtype
        if patch_feats.dtype != input_dtype:
            patch_feats = patch_feats.to(input_dtype)
        feats = F.linear(patch_feats, encoder.input_layer.weight, encoder.input_layer.bias)
        if learned_pe is not None:
            feats = feats + learned_pe

        block_param = next(encoder.blocks.parameters(), None)
        block_dtype = block_param.dtype if block_param is not None else feats.dtype
        if feats.dtype != block_dtype:
            feats = feats.to(block_dtype)

        feats = self._run_block_stack(
            blocks=encoder.blocks,
            feats=feats,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            rope_freqs_cis=rope_freqs_cis,
        )
        feats = encoder.post_layernorm(feats)
        return F.linear(feats, self.autoencoder.proj.weight, self.autoencoder.proj.bias)

    def _patchify_dense_video(self, dense_video: torch.Tensor) -> torch.Tensor:
        """Patchify a dense channels-last video chunk into `[B, S, patch_dim]`."""
        batch_size, raw_frames, height, width, channels = dense_video.shape
        patch_time, patch_height, patch_width = self.patch_size
        temporal_patches = raw_frames // patch_time
        height_patches = height // patch_height
        width_patches = width // patch_width
        return rearrange(
            dense_video,
            "b (nt pt) (nh ph) (nw pw) c -> b (nt nh nw) (pt ph pw c)",
            b=batch_size,
            nt=temporal_patches,
            nh=height_patches,
            nw=width_patches,
            pt=patch_time,
            ph=patch_height,
            pw=patch_width,
            c=channels,
        )

    def _decode_latent_chunk(self, dense_latent_chunk: torch.Tensor) -> torch.Tensor:
        """Decode one dense latent chunk into a dense channels-last video chunk."""
        batch_size, temporal_patches, height_patches, width_patches, _ = dense_latent_chunk.shape
        feats = dense_latent_chunk.reshape(batch_size, temporal_patches * height_patches * width_patches, -1)
        metadata = self._get_or_build_grid_metadata(
            module_name="decoder",
            module=self.autoencoder.decoder,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=feats.device,
            dtype=self.autoencoder.decoder.input_layer.weight.dtype,
        )
        patch_feats = self._decode_chunk_core(
            feats,
            learned_pe=metadata.learned_pe,
            rope_freqs_cis=metadata.rope_freqs_cis,
            q_seqlen=metadata.q_seqlen,
            cu_seqlens_q=metadata.cu_seqlens,
            max_q_seqlen=metadata.max_seq_len,
        )
        return self._unpatchify_dense_video_chunk(
            patch_feats,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
        )

    def _decode_chunk_core(
        self,
        feats: torch.Tensor,
        learned_pe: torch.Tensor | None,
        rope_freqs_cis: torch.Tensor | None,
        q_seqlen: list[int] | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        max_q_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Decode one dense `[B, S, latent_dim]` chunk into patch-space features."""
        decoder = self.autoencoder.decoder
        input_dtype = decoder.input_layer.weight.dtype
        if feats.dtype != input_dtype:
            feats = feats.to(input_dtype)

        feats = F.linear(feats, decoder.input_layer.weight, decoder.input_layer.bias)
        if learned_pe is not None:
            feats = feats + learned_pe

        block_param = next(decoder.blocks.parameters(), None)
        block_dtype = block_param.dtype if block_param is not None else feats.dtype
        if feats.dtype != block_dtype:
            feats = feats.to(block_dtype)

        feats = self._run_block_stack(
            blocks=decoder.blocks,
            feats=feats,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            rope_freqs_cis=rope_freqs_cis,
        )
        feats = decoder.out_norm(feats)
        return F.linear(feats, decoder.out_layer.weight, decoder.out_layer.bias)

    def _unpatchify_dense_video_chunk(
        self,
        patch_feats: torch.Tensor,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
    ) -> torch.Tensor:
        """Unpatchify dense decoder outputs into channels-last video chunks."""
        patch_time, patch_height, patch_width = self.patch_size
        patch_volume = patch_time * patch_height * patch_width
        if self.autoencoder.out_channels % patch_volume != 0:
            raise ValueError(
                f"Autoencoder out_channels={self.autoencoder.out_channels} is not divisible by patch volume {patch_volume}."
            )
        output_channels = self.autoencoder.out_channels // patch_volume
        return rearrange(
            patch_feats,
            "b (nt nh nw) (pt ph pw c) -> b (nt pt) (nh ph) (nw pw) c",
            b=batch_size,
            nt=temporal_patches,
            nh=height_patches,
            nw=width_patches,
            pt=patch_time,
            ph=patch_height,
            pw=patch_width,
            c=output_channels,
        )

    def _run_block_stack(
        self,
        blocks: nn.ModuleList,
        feats: torch.Tensor,
        q_seqlen: list[int] | None,
        cu_seqlens_q: torch.Tensor | None,
        max_q_seqlen: int | None,
        rope_freqs_cis: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one backend-specific transformer block stack over `[B, S, D]` features."""
        backend = self.resolve_backend(use_compile=torch.compiler.is_compiling())
        if backend == "varlen":
            if q_seqlen is None or cu_seqlens_q is None or max_q_seqlen is None:
                raise ValueError("Varlen dense backend requires q_seqlen, cu_seqlens_q, and max_q_seqlen metadata.")
            return run_varlen_block_stack(
                blocks,
                feats,
                q_seqlen=q_seqlen,
                cu_seqlens_q=cu_seqlens_q,
                max_q_seqlen=max_q_seqlen,
                q_freqs_cis=rope_freqs_cis,
            )
        if backend == "batched":
            return run_batched_block_stack(
                blocks,
                feats,
                q_freqs_cis=rope_freqs_cis,
            )
        if backend == "batched_with_padding":
            assert feats.shape[0] == 1, (
                "batched_with_padding backend only supports batch_size=1, due to varlen kernel requirements."
            )
            return run_batched_block_stack(
                blocks,
                feats,
                cu_seqlens_q=cu_seqlens_q,
                max_q_seqlen=max_q_seqlen,
                q_freqs_cis=rope_freqs_cis,
            )
        raise ValueError(f"Unsupported dense runtime backend: {backend}")

    def _get_or_build_grid_metadata(
        self,
        module_name: str,
        module: SparseTransformerBase,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DenseGridMetadata:
        """Fetch or create dense-grid metadata for one uniform chunk shape."""
        key = self._metadata_cache_key(
            module_name,
            batch_size,
            temporal_patches,
            height_patches,
            width_patches,
            device,
            dtype,
        )
        cached = self._metadata_cache.get(key)
        if cached is not None:
            return cached

        metadata = self._build_grid_metadata(
            module=module,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        self._metadata_cache[key] = metadata
        return metadata

    def _build_grid_metadata(
        self,
        module: SparseTransformerBase,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> DenseGridMetadata:
        """Precompute dense-grid metadata for one uniform chunk shape."""
        seq_len = temporal_patches * height_patches * width_patches
        q_seqlen = [seq_len] * batch_size
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device=device,
        )
        learned_pe = self._build_learned_position_embeddings(
            module,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        rope_freqs_cis = self._build_rope_freqs_cis(
            module,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        return DenseGridMetadata(
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            learned_pe=learned_pe,
            rope_freqs_cis=rope_freqs_cis,
            cu_seqlens=cu_seqlens,
            q_seqlen=q_seqlen,
            max_seq_len=seq_len,
        )

    def _build_learned_position_embeddings(
        self,
        module: SparseTransformerBase,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build broadcastable learned spatial embeddings for one uniform chunk."""
        if module.pe_mode not in {"joint", "learned"}:
            return None
        if not isinstance(module.pos_embedder, LearnedPositionEmbedder):
            raise ValueError(
                "Dense runtime V1 expects LearnedPositionEmbedder for learned/joint PE, "
                f"got {type(module.pos_embedder).__name__}."
            )

        pos_embedder = module.pos_embedder
        positional_embeddings = pos_embedder.position_embedding.weight.reshape(
            pos_embedder.position_embedding_size,
            pos_embedder.position_embedding_size,
            -1,
        )
        spatial_embeddings = pos_embedder._get_interpolated_position_embedding(
            positional_embeddings,
            target_height=height_patches,
            target_width=width_patches,
            target_device=device,
        ).to(dtype=positional_embeddings.dtype)
        spatial_flat = spatial_embeddings.reshape(height_patches * width_patches, -1)
        temporal_flat = spatial_flat.repeat(temporal_patches, 1)
        return temporal_flat.unsqueeze(0)

    def _build_rope_freqs_cis(
        self,
        module: SparseTransformerBase,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build RoPE frequencies for one regular dense patch grid."""
        blocks_with_rope = [block for block in module.blocks if getattr(block.attn, "use_rope", False)]
        if not blocks_with_rope:
            return None

        rope_configs = {
            (
                block.attn.rope.head_dim,
                block.attn.rope.pos_cls_token,
            )
            for block in blocks_with_rope
        }
        if len(rope_configs) != 1:
            raise ValueError("Dense runtime V1 requires uniform RoPE configuration across blocks.")

        positions = self._build_regular_patch_positions(
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        if batch_size > 1:
            positions = positions.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * positions.shape[0], -1)
        return blocks_with_rope[0].attn.rope.compute_freqs_cis(positions, has_special_tokens=False)

    def _build_regular_patch_positions(
        self,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build regular `[S, 4]` patch coordinates in `(t, h, w, z)` order."""
        return (
            torch.stack(
                torch.meshgrid(
                    torch.arange(temporal_patches, device=device),
                    torch.arange(height_patches, device=device),
                    torch.arange(width_patches, device=device),
                    torch.arange(1, device=device),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 4)
            .to(dtype=torch.int32)
        )

    def _sample_dense_posterior(self, moments: torch.Tensor) -> torch.Tensor:
        """Sample the dense latent posterior from `[mean, logvar]` moments."""
        original_dtype = moments.dtype
        mean, logvar = torch.chunk(moments.to(torch.float32), 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(std)
        return sample.to(original_dtype)
