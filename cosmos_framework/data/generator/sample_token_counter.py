# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""One token accounting implementation for worker mixtures and the parent packer."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from cosmos_framework.model.generator.tokenizers.uniae.frame_math import (
    get_uniae_chunk_frames,
    get_uniae_latent_num_frames,
    normalize_uniae_chunk_frames,
)
from cosmos_framework.utils.generator.data_utils import read_positive_int_metadata
from cosmos_framework.utils.generator.spatial_patch import normalize_spatial_patch_hw


class SampleTokenCounter:
    """Price decoded media, text, references and controls before sequence packing."""

    tokenizer_spatial_compression_factor: int
    tokenizer_temporal_compression_factor: int
    patch_spatial: int
    lidar_spatial_compression: tuple[int, ...] | None
    lidar_temporal_compression_factor: int | None
    radar_spatial_compression: tuple[int, ...] | None
    radar_temporal_compression_factor: int | None
    lidar_patch_spatial_hw: tuple[int, int]
    radar_patch_spatial_hw: tuple[int, int]
    sound_latent_fps: float
    audio_sample_rate: int
    uniae_pad_frames: int | None
    uniae_chunk_frames: int | dict[str, int] | None

    def __init__(
        self,
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        lidar_spatial_compression: Sequence[int] | None = None,
        lidar_temporal_compression_factor: int | None = None,
        radar_spatial_compression: Sequence[int] | None = None,
        radar_temporal_compression_factor: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        lidar_patch_spatial_hw: int | tuple[int, int] | None = None,
        radar_patch_spatial_hw: int | tuple[int, int] | None = None,
    ) -> None:
        self.tokenizer_spatial_compression_factor = tokenizer_spatial_compression_factor
        self.tokenizer_temporal_compression_factor = tokenizer_temporal_compression_factor
        self.lidar_spatial_compression = (
            tuple(int(factor) for factor in lidar_spatial_compression)
            if lidar_spatial_compression is not None
            else None
        )
        if self.lidar_spatial_compression is not None and (
            len(self.lidar_spatial_compression) != 2 or any(factor <= 0 for factor in self.lidar_spatial_compression)
        ):
            raise ValueError(
                "lidar_spatial_compression must contain two positive factors "
                f"(height, width), got {self.lidar_spatial_compression}"
            )
        self.lidar_temporal_compression_factor = (
            int(lidar_temporal_compression_factor) if lidar_temporal_compression_factor is not None else None
        )
        if self.lidar_temporal_compression_factor is not None and self.lidar_temporal_compression_factor <= 0:
            raise ValueError(
                f"lidar_temporal_compression_factor must be positive, got {self.lidar_temporal_compression_factor}"
            )
        self.radar_spatial_compression = (
            tuple(int(factor) for factor in radar_spatial_compression)
            if radar_spatial_compression is not None
            else None
        )
        if self.radar_spatial_compression is not None and (
            len(self.radar_spatial_compression) != 2 or any(factor <= 0 for factor in self.radar_spatial_compression)
        ):
            raise ValueError(
                "radar_spatial_compression must contain two positive factors "
                f"(height, width), got {self.radar_spatial_compression}"
            )
        self.radar_temporal_compression_factor = (
            int(radar_temporal_compression_factor) if radar_temporal_compression_factor is not None else None
        )
        if self.radar_temporal_compression_factor is not None and self.radar_temporal_compression_factor <= 0:
            raise ValueError(
                f"radar_temporal_compression_factor must be positive, got {self.radar_temporal_compression_factor}"
            )
        self.patch_spatial = patch_spatial
        self.lidar_patch_spatial_hw: tuple[int, int] = normalize_spatial_patch_hw(
            patch_spatial if lidar_patch_spatial_hw is None else lidar_patch_spatial_hw
        )
        self.radar_patch_spatial_hw: tuple[int, int] = normalize_spatial_patch_hw(
            patch_spatial if radar_patch_spatial_hw is None else radar_patch_spatial_hw
        )
        self.sound_latent_fps = sound_latent_fps
        self.audio_sample_rate = audio_sample_rate
        self.uniae_pad_frames = int(uniae_pad_frames) if uniae_pad_frames is not None else None
        self.uniae_chunk_frames = self._normalize_uniae_chunk_frames(uniae_chunk_frames)

    def count_raw_sample(self, sample: Mapping[str, Any]) -> int:
        """Match the packer's single-sample representation without copying tensors.

        Collation and the parent's sample splitter wrap bare text/action/sound
        tensors in lists. Media already accept either bare tensors or lists.
        """
        data = dict(sample)
        for key in ("text_token_ids", "action", "sound"):
            value = data.get(key)
            if value is not None and not isinstance(value, list):
                data[key] = [value]
        return self._compute_num_tokens_per_sample(data)

    def _num_lidar_tokens(self, data_batch: Mapping[str, Any]) -> int:
        """Cost the sample's LiDAR range clips with the LiDAR VAE's own compression.

        The LiDAR VAE does not compress time, so costing a sweep clip with the camera's 4x
        would undercount it fourfold and silently over-pack the batch.
        """
        clips = data_batch.get("lidar")
        if not clips:
            return 0
        if self.lidar_spatial_compression is None or self.lidar_temporal_compression_factor is None:
            raise ValueError(
                "This batch carries a LiDAR stream, but the loader has no LiDAR compression factors. "
                "Set lidar_spatial_compression and lidar_temporal_compression_factor."
            )
        spatial_h, spatial_w = self.lidar_spatial_compression
        num_tokens = 0
        for clip in clips:
            _, T, H, W = clip.shape
            patch_h = math.ceil(H // spatial_h / self.lidar_patch_spatial_hw[0])
            patch_w = math.ceil(W // spatial_w / self.lidar_patch_spatial_hw[1])
            latent_t = 1 + (T - 1) // self.lidar_temporal_compression_factor
            num_tokens += patch_h * patch_w * latent_t
        return num_tokens

    def _num_radar_tokens(self, data_batch: Mapping[str, Any]) -> int:
        """Cost the sample's radar BEV clips with the radar VAE's own compression.

        Same arithmetic as the sweeps over a square BEV grid rather than a range image: the
        radar VAE does not compress time either, so one scan is one latent frame and the
        camera's 4x would undercount a clip fourfold. Radar is the more expensive of the two
        sensors per second of clip -- it cycles at ~20 Hz against LiDAR's 10 and each scan
        carries more latents -- so an underpriced clip over-packs the batch by more here.
        """
        clips = data_batch.get("radar")
        if not clips:
            return 0
        if self.radar_spatial_compression is None or self.radar_temporal_compression_factor is None:
            raise ValueError(
                "This batch carries a radar stream, but the loader has no radar compression factors. "
                "Set radar_spatial_compression and radar_temporal_compression_factor."
            )
        spatial_h, spatial_w = self.radar_spatial_compression
        num_tokens = 0
        for clip in clips:
            _, T, H, W = clip.shape
            patch_h = math.ceil(H // spatial_h / self.radar_patch_spatial_hw[0])
            patch_w = math.ceil(W // spatial_w / self.radar_patch_spatial_hw[1])
            latent_t = 1 + (T - 1) // self.radar_temporal_compression_factor
            num_tokens += patch_h * patch_w * latent_t
        return num_tokens

    def _normalize_uniae_chunk_frames(
        self, uniae_chunk_frames: int | Mapping[str, int] | None
    ) -> int | dict[str, int] | None:
        return normalize_uniae_chunk_frames(
            uniae_chunk_frames,
            pad_frames=self.uniae_pad_frames,
            temporal_compression_factor=self.tokenizer_temporal_compression_factor,
            temporal_divisibility_name="tokenizer_temporal_compression_factor",
        )

    def _get_uniae_chunk_frames(self, spatial_shape: tuple[int, int]) -> int:
        assert self.uniae_chunk_frames is not None
        return get_uniae_chunk_frames(self.uniae_chunk_frames, spatial_shape=spatial_shape)

    def _compute_vision_latent_t_shape(self, T: int, H: int, W: int) -> int:
        if T < 1:
            raise ValueError(f"Vision media must contain at least one frame, got {T}.")
        if T == 1 or self.uniae_chunk_frames is None:
            return 1 + (T - 1) // self.tokenizer_temporal_compression_factor

        assert self.uniae_pad_frames is not None
        return get_uniae_latent_num_frames(
            T,
            self.uniae_chunk_frames,
            pad_frames=self.uniae_pad_frames,
            temporal_compression_factor=self.tokenizer_temporal_compression_factor,
            spatial_shape=(H, W),
        )

    def _compute_num_tokens_per_sample(self, data_batch: Mapping[str, Any]) -> int:
        """
        This function computes the number of tokens per sample in the data batch.
        This includes text + vision generation tokens + action tokens.

        Args:
            data_batch (dict): The data batch containing the text tokens.

        Returns:
            int: The number of tokens per sample.
        """
        und_tokens, gen_tokens = self._compute_token_split_per_sample(data_batch)
        return und_tokens + gen_tokens

    def _compute_token_split_per_sample(self, data_batch: Mapping[str, Any]) -> tuple[int, int]:
        """
        Split one sample's packed length into the understanding and generation pathways.

        The two towers of the MoT attend over the same packed sequence but cost
        different amounts per token, so the cost model prices them separately. The
        two returned counts always sum to ``_compute_num_tokens_per_sample``.

        The three structural markers (``<eos>``, ``<vision_start>``, ``<vision_end>``)
        are attributed to the stream they delimit rather than broken out, which keeps
        that sum exact at the cost of a handful of tokens landing on one side or the
        other.

        Args:
            data_batch (dict): The data batch containing the text tokens.

        Returns:
            tuple[int, int]: ``(understanding tokens, generation tokens)``.
        """

        # The token sequence we have is
        # <text tokens> <eos> <vision_start> <image tokens> <vision_end> [<action tokens>]
        # The spatial dimension of image tokens is compressed by
        # vae spatial downsampling factor + pathification
        # The temporal dimension of image tokens is compressed by
        # vae temporal downsampling factor
        # Action tokens have 1 token per time step (no spatial dimension)

        has_text_tokens = "text_token_ids" in data_batch
        und_tokens = 0
        gen_tokens = 0
        if has_text_tokens:
            text_token_ids = data_batch["text_token_ids"]
            if isinstance(text_token_ids, list):
                # Price every independently tokenized per-view caption. The packing MR
                # will use these boundaries as separate UND segments; accounting for the
                # wrapper per segment here prevents the batch builder from underpacking.
                und_tokens = sum(int(tokens.shape[0]) + 1 for tokens in text_token_ids)
            else:
                num_text_tokens = text_token_ids.shape[1]
                und_tokens = num_text_tokens + 1

        # Vision part. A LiDAR-only sample has no camera stream, so the loop below prices
        # nothing and the whole generation cost comes from the sweeps.
        is_image_batch = "images" in data_batch
        # Absent rather than empty for a LiDAR-only sample, and tested for explicitly: the loop
        # below also accepts a bare tensor, and `or []` would ask that tensor for its truth
        # value before reaching that branch.
        input_images_or_videos = data_batch.get("images" if is_image_batch else "video")
        if input_images_or_videos is None:
            input_images_or_videos = []
        if "enable_per_camera_vae_encoding" in data_batch:
            sample_n_views_values = read_positive_int_metadata(data_batch, "sample_n_views", expected_count=1)
            frames_per_view_values = read_positive_int_metadata(
                data_batch,
                "num_video_frames_per_view",
                expected_count=1,
            )
            if sample_n_views_values is None or frames_per_view_values is None:
                raise ValueError(
                    "sample_n_views and num_video_frames_per_view are required when per-camera VAE encoding is enabled."
                )
            sample_n_views = sample_n_views_values[0]
            frames_per_view = frames_per_view_values[0]
        else:
            sample_n_views = None
            frames_per_view = None

        # iterate over all the media in the batch
        for media in input_images_or_videos if isinstance(input_images_or_videos, list) else [input_images_or_videos]:
            if is_image_batch:
                _, H, W = media.shape
                T = 1
            else:
                _, T, H, W = media.shape

            latent_h_shape = H // self.tokenizer_spatial_compression_factor
            latent_w_shape = W // self.tokenizer_spatial_compression_factor
            patch_h_shape = math.ceil(latent_h_shape / self.patch_spatial)
            patch_w_shape = math.ceil(latent_w_shape / self.patch_spatial)
            if sample_n_views is not None and frames_per_view is not None and not is_image_batch:
                # The multiview dataset resizes every camera to the same H/W and
                # selects the same synchronized frame count before concatenating
                # the camera-major clips along T.
                expected_frames = sample_n_views * frames_per_view
                if T != expected_frames:
                    raise ValueError(
                        "Multiview vision length must equal sample_n_views * num_video_frames_per_view: "
                        f"got T={T}, sample_n_views={sample_n_views}, num_video_frames_per_view={frames_per_view}."
                    )
                latent_t_shape = sample_n_views * self._compute_vision_latent_t_shape(frames_per_view, H, W)
            else:
                latent_t_shape = self._compute_vision_latent_t_shape(T, H, W)

            num_vision_tokens = patch_h_shape * patch_w_shape * latent_t_shape
            if has_text_tokens:
                num_vision_tokens += 2
            gen_tokens += num_vision_tokens

        # LiDAR part: its own VAE, hence its own compression factors. Charged to
        # gen_tokens so the iteration-time cost model sees the sweeps.
        gen_tokens += self._num_lidar_tokens(data_batch)

        # Radar part: a third VAE and a third pair of factors, charged the same way. A joint
        # camera + radar sample carries no sweeps and a joint camera + LiDAR one no scans, so
        # each of the two calls prices nothing on the other's recipe.
        gen_tokens += self._num_radar_tokens(data_batch)

        # Action part: each action time step is 1 token.
        # Action tensor shape is (T_action, D) per sample; stored as a single-element list.
        if "action" in data_batch:
            list_of_actions = data_batch["action"]
            for action in list_of_actions:
                # skip None actions
                if action is None:
                    continue
                num_action_tokens = action.shape[0]
                gen_tokens += num_action_tokens

        # Sound part — estimate sound tokens from audio waveform length
        if self.sound_latent_fps > 0 and "sound" in data_batch:
            sound_data = data_batch["sound"]
            if isinstance(sound_data, list) and len(sound_data) > 0:
                first_sound = sound_data[0]
                # Unwrap nested list if needed
                if isinstance(first_sound, list):
                    first_sound = first_sound[0]
                if first_sound is not None and isinstance(first_sound, torch.Tensor):
                    num_audio_samples = first_sound.shape[-1]
                    audio_duration = num_audio_samples / self.audio_sample_rate
                    num_sound_tokens = int(audio_duration * self.sound_latent_fps)
                    gen_tokens += num_sound_tokens

        return und_tokens, gen_tokens
