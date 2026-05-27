# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Augmentor that builds a SequencePlan for sound-enabled training.

This augmentor creates a SequencePlan based on the presence of sound data
in the sample, following the same pattern as Action's ActionTransformPipeline
which builds sequence plans for action-enabled training.

Placed at the END of the augmentor pipeline (after video/audio extraction
and text transforms) so that all data shapes are known.
"""

from typing import Optional

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.data.vfm.sound_data_utils import VALID_SOUND_MODES, build_sequence_plan_for_sound


class SoundSequencePlanBuilder(Augmentor):
    """Builds a SequencePlan for sound-enabled samples.

    Inspects the data dict for sound data and creates an appropriate
    SequencePlan. If no sound is present, creates a video-only plan.

    Args:
        input_keys: Not used (reads from data_dict directly)
        output_keys: Not used
        args: Dictionary with:
            - mode: Generation mode ("t2vs", "tv2s", "ts2v", "ti2sv"). Default: "t2vs"
            - video_key: Key to find video data. Default: "video"
            - sound_key: Key to find sound data. Default: "sound"
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.mode = args.get("mode", "t2vs")
        self.video_key = args.get("video_key", "video")
        self.sound_key = args.get("sound_key", "sound")

        assert self.mode in VALID_SOUND_MODES, f"Invalid mode: {self.mode}. Must be one of {VALID_SOUND_MODES}"

    def __call__(self, data_dict: dict) -> dict | None:
        """Add sound fields to the existing SequencePlan.

        Only modifies ``has_sound`` and ``condition_frame_indexes_sound``.
        All other fields (vision conditioning, action conditioning, etc.) set
        by upstream augmentors are preserved.

        If no upstream plan exists, creates a minimal one with sensible defaults.
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
                existing_plan.has_sound = sound_plan.has_sound
                existing_plan.condition_frame_indexes_sound = sound_plan.condition_frame_indexes_sound
            else:
                existing_plan.has_sound = False
                existing_plan.condition_frame_indexes_sound = []
        else:
            # No upstream plan — build a complete one from scratch
            if sound is not None and hasattr(sound, "shape"):
                data_dict["sequence_plan"] = build_sequence_plan_for_sound(
                    mode=self.mode,
                    video_latent_length=video_length,
                    sound_latent_length=0,
                )
            else:
                from cosmos_framework.data.vfm.sequence_packing import SequencePlan

                data_dict["sequence_plan"] = SequencePlan(
                    has_text=True,
                    has_vision=True,
                    has_sound=False,
                )

        return data_dict
