# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.data.generator.augmentors.audio_caption import AudioCaptionAppender

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_normalizes_background_recording_conditions_label() -> None:
    augmentor = AudioCaptionAppender(
        input_keys=["metas", "ai_caption"],
        args={"separator": "\n\nAudio description:\n\n"},
    )
    data = {
        "ai_caption": "A person speaks to the camera.",
        "metas": {
            "caption_audio": (
                "Speech: A person speaks.\n\nBackground and recording conditions: Quiet indoor room tone."
            )
        },
        "sound": object(),
    }

    assert augmentor(data) == {
        "ai_caption": (
            "A person speaks to the camera.\n\n"
            "Audio description:\n\nSpeech: A person speaks.\n\n"
            "Background audio and recording conditions: Quiet indoor room tone."
        ),
        "sound": data["sound"],
    }
