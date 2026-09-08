# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""What a single billable Cosmos3 VFM training sample is, in tokens.

A "sample" is one image or one video clip as it enters the generation tower.
Cost scales with the token count that sample contributes to the packed
sequence, so this module converts a human-facing description (modality,
resolution bucket, aspect ratio, clip length) into the token counts the
FLOP model consumes.

The pipeline mirrored here is:

    pixels [3, T, H, W]
      -> Wan 2.2 VAE  (T -> 1 + (T-1)//4,  H -> H/16,  W -> W/16)
      -> patchify     (H_lat -> ceil(H_lat/2), W_lat -> ceil(W_lat/2))
      -> vision generation tokens

Stdlib only, on purpose: the token arithmetic is the part most likely to carry
an off-by-one, and keeping it importable without torch means it can be tested
anywhere. The FLOP and price layers live in ``estimator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from cosmos_framework.data.generator.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

# Wan 2.2 VAE (``wan2pt2_vae_4x16x16``) compression, and the generation tower's
# spatial patch size (``diffusion_expert_config.patch_spatial``). Together these
# give an effective 32x spatial stride from pixels to tokens, except where the
# latent grid is odd and patchify pads it (see ``vision_tokens``).
SPATIAL_COMPRESSION = 16
TEMPORAL_COMPRESSION = 4
PATCH_SPATIAL = 2

Modality = Literal["image", "video"]


def resolve_pixel_shape(modality: Modality, resolution: str, aspect_ratio: str = "16,9") -> tuple[int, int]:
    """Look up the padded pixel canvas ``(height, width)`` for a resolution bucket.

    The dataset tables store buckets as ``(width, height)``; this returns
    ``(height, width)`` to match the ``[C, T, H, W]`` tensor convention used
    everywhere downstream.

    Args:
        modality: ``"image"`` or ``"video"`` -- the two tables differ (video
            carries extra buckets such as the ``"2,3"`` BEHAVIOR-1K canvas).
        resolution: Resolution tier key, e.g. ``"256"``, ``"480"``, ``"720"``.
        aspect_ratio: Bucket key within the tier, e.g. ``"16,9"``.

    Returns:
        ``(height, width)`` in pixels.

    Raises:
        KeyError: If the tier or aspect ratio is not a defined bucket.
    """
    table = IMAGE_RES_SIZE_INFO if modality == "image" else VIDEO_RES_SIZE_INFO
    if resolution not in table:
        raise KeyError(f"Unknown {modality} resolution '{resolution}'. Available: {sorted(table)}")
    buckets = table[resolution]
    if aspect_ratio not in buckets:
        raise KeyError(
            f"Unknown aspect ratio '{aspect_ratio}' for {modality} resolution '{resolution}'. "
            f"Available: {sorted(buckets)}"
        )
    width, height = buckets[aspect_ratio]
    return height, width


def latent_frames(num_pixel_frames: int, temporal_compression: int = TEMPORAL_COMPRESSION) -> int:
    """Latent frame count produced by the causal VAE for a pixel clip.

    Matches ``Wan2pt2VAEInterface.get_latent_num_frames``: the first frame is
    encoded on its own and every subsequent group of ``temporal_compression``
    frames collapses to one latent frame.

    Args:
        num_pixel_frames: Pixel-space frame count (``1`` for a still image).
        temporal_compression: VAE temporal stride.

    Returns:
        Number of latent frames.
    """
    if num_pixel_frames < 1:
        raise ValueError(f"num_pixel_frames must be >= 1, got {num_pixel_frames}")
    return 1 + (num_pixel_frames - 1) // temporal_compression


def vision_tokens(
    height: int,
    width: int,
    num_pixel_frames: int,
    spatial_compression: int = SPATIAL_COMPRESSION,
    temporal_compression: int = TEMPORAL_COMPRESSION,
    patch_spatial: int = PATCH_SPATIAL,
) -> int:
    """Generation-tower vision tokens for one sample.

    The patch grid is CEIL-divided, not floor-divided, because
    ``Cosmos3VFMNetwork.patchify_and_pack_latents`` zero-pads the latent grid up
    to a multiple of ``patch_spatial`` before patchifying. Those pad patches are
    real tokens: they occupy sequence positions and are attended over. The 720p
    bucket is exactly this case -- a 1280x720 canvas is a 45x80 latent, which
    patchifies to 23x40 = 920 tokens per frame rather than 22x40 = 880.

    Args:
        height: Pixel canvas height.
        width: Pixel canvas width.
        num_pixel_frames: Pixel-space frame count (``1`` for a still image).
        spatial_compression: VAE spatial stride.
        temporal_compression: VAE temporal stride.
        patch_spatial: Generation-tower patch size applied to the latent grid.

    Returns:
        Token count contributed by this sample's vision stream.
    """
    if height % spatial_compression or width % spatial_compression:
        raise ValueError(f"Canvas {height}x{width} is not divisible by the VAE spatial stride {spatial_compression}.")
    latent_height = height // spatial_compression
    latent_width = width // spatial_compression
    tokens_per_frame = math.ceil(latent_height / patch_spatial) * math.ceil(latent_width / patch_spatial)
    return latent_frames(num_pixel_frames, temporal_compression) * tokens_per_frame


@dataclass(frozen=True)
class SampleSpec:
    """One billable training sample.

    Attributes:
        modality: ``"image"`` or ``"video"``.
        resolution: Resolution tier key, e.g. ``"480"``.
        aspect_ratio: Bucket key within the tier, e.g. ``"16,9"``.
        num_pixel_frames: Pixel frames. Must be ``1`` for images; for video the
            VAE consumes ``1 + 4k`` frames, so other lengths are rejected rather
            than silently rounded.
        text_tokens: Caption tokens this sample contributes to the understanding
            pathway. Attention is quadratic in the packed length, so the text
            side is not free even though the loss only supervises generation.
        sound_tokens: Audio-latent tokens, when sound generation is enabled.
        spatial_compression: VAE spatial stride.
        temporal_compression: VAE temporal stride.
        patch_spatial: Generation-tower patch size.
    """

    modality: Modality
    resolution: str
    aspect_ratio: str = "16,9"
    num_pixel_frames: int = 1
    text_tokens: int = 512
    sound_tokens: int = 0
    spatial_compression: int = SPATIAL_COMPRESSION
    temporal_compression: int = TEMPORAL_COMPRESSION
    patch_spatial: int = PATCH_SPATIAL

    def __post_init__(self) -> None:
        if self.modality not in ("image", "video"):
            raise ValueError(f"modality must be 'image' or 'video', got {self.modality!r}")
        if self.modality == "image" and self.num_pixel_frames != 1:
            raise ValueError(f"image samples must have num_pixel_frames=1, got {self.num_pixel_frames}")
        if self.modality == "video":
            if self.num_pixel_frames < 1 + self.temporal_compression:
                raise ValueError(
                    f"video samples need at least {1 + self.temporal_compression} frames, got {self.num_pixel_frames}"
                )
            if (self.num_pixel_frames - 1) % self.temporal_compression:
                raise ValueError(
                    f"video frame count must be 1 + {self.temporal_compression}k to encode without VAE padding, "
                    f"got {self.num_pixel_frames}"
                )
        if self.text_tokens < 0 or self.sound_tokens < 0:
            raise ValueError("text_tokens and sound_tokens must be non-negative")

    @property
    def pixel_shape(self) -> tuple[int, int]:
        """Padded pixel canvas ``(height, width)``."""
        return resolve_pixel_shape(self.modality, self.resolution, self.aspect_ratio)

    @property
    def height(self) -> int:
        return self.pixel_shape[0]

    @property
    def width(self) -> int:
        return self.pixel_shape[1]

    @property
    def latent_frames(self) -> int:
        """Latent frames after VAE temporal compression."""
        return latent_frames(self.num_pixel_frames, self.temporal_compression)

    @property
    def vision_tokens(self) -> int:
        """Vision generation tokens contributed by this sample."""
        height, width = self.pixel_shape
        return vision_tokens(
            height,
            width,
            self.num_pixel_frames,
            self.spatial_compression,
            self.temporal_compression,
            self.patch_spatial,
        )

    @property
    def gen_tokens(self) -> int:
        """Total generation-pathway tokens (vision + sound)."""
        return self.vision_tokens + self.sound_tokens

    @property
    def total_tokens(self) -> int:
        """Packed sequence length this sample occupies (understanding + generation)."""
        return self.text_tokens + self.gen_tokens

    @property
    def name(self) -> str:
        """Stable identifier used as the bucket key in reports and calibration files."""
        if self.modality == "image":
            return f"image_{self.resolution}"
        return f"video_{self.resolution}_{self.num_pixel_frames}f"

    def describe(self) -> str:
        """One-line human summary."""
        height, width = self.pixel_shape
        shape = f"{width}x{height}"
        if self.modality == "video":
            shape += f"x{self.num_pixel_frames}f"
        return f"{self.name} ({shape}, {self.total_tokens} tokens: {self.gen_tokens} gen + {self.text_tokens} text)"
