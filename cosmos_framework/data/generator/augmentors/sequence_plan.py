# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Augmentor for creating sequence plans with random conditional frames.

Supports two sampling strategies:
- weighted dict (``conditioning_config``): named mode or legacy prefix count → probability pairs
- uniform (``uniform_conditioning=True``): k ~ Uniform{0, T_latent-1}, where T_latent
  is computed from the actual video length using the VAE temporal compression factor
  or UniAE chunking parameters when provided.
"""

import random
from collections.abc import Mapping
from typing import Optional

import torch

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.tokenizers.uniae.frame_math import (
    get_uniae_chunk_frames,
    get_uniae_latent_num_frames,
    normalize_uniae_chunk_frames,
)

_NAMED_CONDITIONING_FRAME_COUNTS = {
    "t2v": 0,
    "i2v": 1,
    "v2v": 2,
}
_FRAME_INTERPOLATION_MODE = "frame_interpolation"
_VALID_NAMED_CONDITIONING_MODES = {*_NAMED_CONDITIONING_FRAME_COUNTS, _FRAME_INTERPOLATION_MODE}


class SequencePlanAugmentor(Augmentor):
    """Augmentor that creates SequencePlan with random conditional frames.

    Samples conditioning frame indexes and writes them into the SequencePlan.
    Downstream packing code reads this field to set condition_mask.

    Args:
        input_keys: List of input keys (not used, but required by Augmentor interface).
        output_keys: List of output keys (not used, but required by Augmentor interface).
        args: Dictionary containing:
            - "conditioning_config" (dict[str | int, float], optional): Weighted
              distribution mapping named modes or legacy prefix frame counts to
              unnormalized probabilities. Named modes are ``"t2v"``, ``"i2v"``,
              ``"v2v"``, and ``"frame_interpolation"``.
            - "uniform_conditioning" (bool, default False): When True, samples
              k ~ Uniform{0, T_latent-1}. Takes precedence over conditioning_config when
              both are set. At least one of uniform_conditioning or conditioning_config
              must be provided.
            - "temporal_compression_factor" (int, default 4): VAE temporal compression
              factor used to convert pixel frame count N to T_latent = 1 + (N-1) // tcf.
            - "uniae_chunk_frames" / "uniae_pad_frames" (optional): When provided,
              use UniAE's non-causal first-frame plus padded-chunk latent count.
              ``uniae_chunk_frames`` may be a scalar or a resolution-keyed mapping.
            - "resolution" (str, optional): Target dataset resolution key. Preferred over
              the current tensor shape when selecting a resolution-keyed UniAE chunk.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        if args is None:
            args = {}

        self.conditioning_config: dict[str | int, float] | None = args.get("conditioning_config")
        self.uniform_conditioning = args.get("uniform_conditioning", False)
        self.temporal_compression_factor = args.get("temporal_compression_factor", 4)
        self.target_resolution_key = None if args.get("resolution") is None else str(args["resolution"])
        self.uniae_pad_frames = None if args.get("uniae_pad_frames") is None else int(args["uniae_pad_frames"])
        self.uniae_chunk_frames = self._normalize_uniae_chunk_frames(args.get("uniae_chunk_frames"))

        if self.conditioning_config is None and not self.uniform_conditioning:
            raise ValueError("args must provide 'conditioning_config' or set 'uniform_conditioning=True'")

        # Validate and normalize probabilities
        if self.conditioning_config is not None:
            for mode, prob in self.conditioning_config.items():
                if isinstance(mode, int):
                    if mode < 0:
                        raise ValueError(f"integer conditioning_config keys must be non-negative, got {mode!r}")
                elif not isinstance(mode, str) or mode not in _VALID_NAMED_CONDITIONING_MODES:
                    raise ValueError(
                        "conditioning_config keys must be non-negative integers or one of "
                        f"{sorted(_VALID_NAMED_CONDITIONING_MODES)}, got {mode!r}"
                    )
                if not isinstance(prob, (int, float)) or prob < 0:
                    raise ValueError(f"conditioning_config values must be non-negative numbers, got {prob}")

            # Normalize probabilities to sum to 1.0
            total_prob = sum(self.conditioning_config.values())
            if total_prob <= 0:
                raise ValueError("conditioning_config probabilities must sum to a positive number")

            self.normalized_config = {k: v / total_prob for k, v in self.conditioning_config.items()}
        else:
            self.normalized_config = {"t2v": 1.0}

    def _normalize_uniae_chunk_frames(
        self, uniae_chunk_frames: int | Mapping[str, int] | None
    ) -> int | dict[str, int] | None:
        return normalize_uniae_chunk_frames(
            uniae_chunk_frames,
            pad_frames=self.uniae_pad_frames,
            temporal_compression_factor=self.temporal_compression_factor,
        )

    def _get_uniae_chunk_frames(self, spatial_shape: tuple[int, int] | None = None) -> int:
        assert self.uniae_chunk_frames is not None
        return get_uniae_chunk_frames(
            self.uniae_chunk_frames,
            spatial_shape=spatial_shape,
            target_resolution_key=self.target_resolution_key,
        )

    def _get_num_frames_and_spatial_shape(
        self,
        data_dict: dict,
        video: object,
    ) -> tuple[int | None, tuple[int, int] | None]:
        del data_dict
        if isinstance(video, torch.Tensor):
            assert video.ndim == 4, "video should be a tensor with shape (C, T, H, W)"
            return video.shape[1], (video.shape[2], video.shape[3])  # video: [C,T,H,W]

        # If video is not a tensor or dict, we can't determine the exact number
        # Use a conservative approach - will be limited by max available frames
        return None, None

    def _get_latent_frame_count(self, num_frames: int | None, spatial_shape: tuple[int, int] | None = None) -> int:
        if num_frames is None:
            return 1
        if num_frames < 1:
            raise ValueError(f"video must contain at least one frame, got {num_frames}")
        if num_frames == 1:
            return 1
        if self.uniae_chunk_frames is None:
            return 1 + (num_frames - 1) // self.temporal_compression_factor

        assert self.uniae_pad_frames is not None
        return get_uniae_latent_num_frames(
            num_frames,
            self.uniae_chunk_frames,
            pad_frames=self.uniae_pad_frames,
            temporal_compression_factor=self.temporal_compression_factor,
            spatial_shape=spatial_shape,
            target_resolution_key=self.target_resolution_key,
        )

    def __call__(self, data_dict: dict) -> dict:
        """Create a SequencePlan with random conditional frames.

        Args:
            data_dict: Input data dictionary. Should contain "video" key to determine
                the number of frames available.

        Returns:
            data_dict: Output dictionary with "sequence_plan" key added.
        """
        # Get video to determine available frames
        video = data_dict.get("video")
        if video is None:
            # This is an image batch
            sequence_plan = SequencePlan(
                has_text=True,  # Has text prompt!
                has_vision=True,
                condition_frame_indexes_vision=[],  # No conditioning frames!
            )
            data_dict["sequence_plan"] = sequence_plan
            return data_dict

        # Determine number of frames
        # Video should be a tensor with shape (C, T, H, W) by this point in the pipeline
        num_frames, spatial_shape = self._get_num_frames_and_spatial_shape(data_dict, video)

        T_latent = self._get_latent_frame_count(num_frames, spatial_shape)

        if self.uniform_conditioning:
            num_conditional_frames = random.randint(0, max(0, T_latent - 1))
            condition_frame_indexes_vision = list(range(num_conditional_frames))
        else:
            modes = list(self.normalized_config.keys())
            weights = list(self.normalized_config.values())
            selected_mode = random.choices(modes, weights=weights, k=1)[0]
            if isinstance(selected_mode, int):
                num_conditional_frames = selected_mode
                num_conditional_frames = min(num_conditional_frames, T_latent - 1) if num_frames is not None else 0
                condition_frame_indexes_vision = list(range(num_conditional_frames))
            elif selected_mode == _FRAME_INTERPOLATION_MODE:
                if T_latent == 1:
                    # Conditioning the only latent frame would leave nothing to noise or supervise.
                    condition_frame_indexes_vision = []
                elif T_latent == 2:
                    # Two conditions would leave no useful interpolation target.
                    condition_frame_indexes_vision = [0]
                else:
                    second_frame_index = random.randint(2, T_latent - 1)
                    condition_frame_indexes_vision = [0, second_frame_index]
            else:
                num_conditional_frames = _NAMED_CONDITIONING_FRAME_COUNTS[selected_mode]
                num_conditional_frames = min(num_conditional_frames, T_latent - 1) if num_frames is not None else 0
                # Prefix conditioning is used for T2V, I2V, and V2V.
                condition_frame_indexes_vision = list(range(num_conditional_frames))

        # Create SequencePlan
        sequence_plan = SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=condition_frame_indexes_vision,
        )

        # Add sequence plan to data dict
        data_dict["sequence_plan"] = sequence_plan

        return data_dict


class MultiviewSequencePlanAugmentor(SequencePlanAugmentor):
    """Sequence planner for camera-major multiview T2V/I2V samples.

    Multiview video-only datasets pack selected camera clips as
    ``[C, num_views * frames_per_view, H, W]`` before VAE encoding. The
    SequencePlan should still store per-view-local latent frame indexes, e.g.
    ``[0]`` for one I2V conditioning frame per camera. This subclass derives
    T_latent from ``num_video_frames_per_view`` so the sampled conditioning
    count is clamped by per-camera length, not the flattened camera-major
    length. Downstream sequence packing expands those local indexes across all
    selected cameras using the per-camera VAE metadata.
    """

    num_frames_key = "num_video_frames_per_view"

    def _get_num_frames_and_spatial_shape(
        self,
        data_dict: dict,
        video: object,
    ) -> tuple[int | None, tuple[int, int] | None]:
        _, spatial_shape = super()._get_num_frames_and_spatial_shape(data_dict, video)
        if self.num_frames_key not in data_dict:
            raise ValueError(f"{self.num_frames_key} is required for multiview sequence planning.")

        num_frames_value = data_dict[self.num_frames_key]
        if isinstance(num_frames_value, torch.Tensor):
            flattened_num_frames = num_frames_value.reshape(-1)  # [N]
            return int(flattened_num_frames[0].item()), spatial_shape
        return int(num_frames_value), spatial_shape
