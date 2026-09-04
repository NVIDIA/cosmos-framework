# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Augmentor that builds a SequencePlan for sound-enabled training.

This augmentor creates a SequencePlan based on the presence of sound data
in the sample, following the same pattern as Action's ActionTransformPipeline
which builds sequence plans for action-enabled training.

Placed at the END of the augmentor pipeline (after video/audio extraction
and text transforms) so that all data shapes are known.
"""

import random
from typing import Optional

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.sound_data_utils import (
    VALID_SOUND_MODES,
    build_sequence_plan_for_sound,
    get_sound_condition_frame_indexes,
)


class SoundSequencePlanBuilder(Augmentor):
    """Builds a SequencePlan for sound-enabled samples.

    Inspects the data dict for sound data and creates an appropriate
    SequencePlan. If no sound is present, creates a video-only plan.

    Args:
        input_keys: Not used (reads from data_dict directly)
        output_keys: Not used
        args: Dictionary with:
            - mode: Generation mode ("t2vs", "tv2s", "ts2v", "ti2sv", "vs2vs"). Default: "t2vs"
            - video_key: Key to find video data. Default: "video"
            - sound_key: Key to find sound data. Default: "sound"
            - temporal_compression_factor: Video tokenizer temporal compression factor. Default: 4
            - sound_latent_fps: Sound tokenizer latent rate in Hz. Default: 25
            - sound_prefix_conditioning_prob: Probability that an eligible "vs2vs" sample
              conditions on the synchronized audio prefix (vs full-audio generation). Default: 1.0
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.mode = args.get("mode", "t2vs")
        self.video_key = args.get("video_key", "video")
        self.sound_key = args.get("sound_key", "sound")
        self.temporal_compression_factor: int = args.get("temporal_compression_factor", 4)
        self.sound_latent_fps: float = args.get("sound_latent_fps", 25.0)
        self.sound_prefix_conditioning_prob: float = args.get("sound_prefix_conditioning_prob", 1.0)

        assert self.mode in VALID_SOUND_MODES, f"Invalid mode: {self.mode}. Must be one of {VALID_SOUND_MODES}"

    def __call__(self, data_dict: dict) -> dict | None:
        """Add sound fields to the existing SequencePlan.

        Only modifies ``has_sound`` and ``condition_frame_indexes_sound``.
        All other fields (vision conditioning, action conditioning, etc.) set
        by upstream augmentors are preserved.

        If no upstream plan exists, creates a minimal one for modes that do not
        require vision-prefix conditioning. ``vs2vs`` requires the upstream plan's
        ``condition_frame_indexes_vision`` to be the contiguous prefix ``list(range(k))``;
        the synchronized sound prefix is derived from ``k``.
        """
        video = data_dict.get(self.video_key)
        sound = data_dict.get(self.sound_key)

        if video is None:
            return None  # Can't proceed without video

        if not hasattr(video, "shape"):
            return None

        video_length = video.shape[1]  # (C, T, H, W) → T

        existing_plan = data_dict.get("sequence_plan")

        if existing_plan is not None:
            # Update only the sound fields on the existing plan
            if sound is not None and hasattr(sound, "shape"):
                sound_plan = build_sequence_plan_for_sound(
                    mode=self.mode,
                    video_latent_length=video_length,
                    sound_latent_length=0,
                )
                sound_condition_frame_indexes = sound_plan.condition_frame_indexes_sound
                vision_prefix_indexes = existing_plan.condition_frame_indexes_vision
                vision_prefix_length = len(vision_prefix_indexes)
                if self.mode == "vs2vs" and vision_prefix_indexes != list(range(vision_prefix_length)):
                    raise ValueError(
                        "sound_generation_mode='vs2vs' requires a contiguous vision prefix starting at latent 0, "
                        f"got condition_frame_indexes_vision={vision_prefix_indexes}"
                    )
                if (
                    self.mode == "vs2vs"
                    and vision_prefix_length >= 2
                    and random.random() < self.sound_prefix_conditioning_prob
                ):
                    sound_condition_frame_indexes = get_sound_condition_frame_indexes(
                        vision_prefix_length=vision_prefix_length,
                        video_temporal_compression_factor=self.temporal_compression_factor,
                        conditioning_fps=float(data_dict["conditioning_fps"]),
                        sound_num_samples=int(sound.shape[-1]),
                        audio_sample_rate=int(data_dict["audio_sample_rate"]),
                        sound_latent_fps=self.sound_latent_fps,
                    )
                    if sound_condition_frame_indexes is None:
                        # Guard, not an expected data path: the v3 pipeline zero-fills audio to the full
                        # clip duration and the vision prefix leaves >= 1 latent frame, so this only fires
                        # for other callers or degenerate fps metadata. Debug level to keep
                        # num_workers * world_size writers out of the training log.
                        log.debug(
                            "VS2VS sample audio is too short for a synchronized prefix; skipping sample",
                            rank0_only=False,
                        )
                        return None
                existing_plan.has_sound = sound_plan.has_sound
                existing_plan.condition_frame_indexes_sound = sound_condition_frame_indexes
            else:
                existing_plan.has_sound = False
                existing_plan.condition_frame_indexes_sound = []
        else:
            if self.mode == "vs2vs":
                raise ValueError("sound_generation_mode='vs2vs' requires an upstream vision sequence plan")
            # No upstream plan — build a complete one from scratch
            if sound is not None and hasattr(sound, "shape"):
                data_dict["sequence_plan"] = build_sequence_plan_for_sound(
                    mode=self.mode,
                    video_latent_length=video_length,
                    sound_latent_length=0,
                )
            else:
                from cosmos_framework.data.generator.sequence_packing import SequencePlan

                data_dict["sequence_plan"] = SequencePlan(
                    has_text=True,
                    has_vision=True,
                    has_sound=False,
                )

        return data_dict
