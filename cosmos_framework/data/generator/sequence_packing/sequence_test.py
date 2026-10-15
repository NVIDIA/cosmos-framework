# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("fps", "seconds_per_frame", "known"), [(30.0, 4 / 30.0, True), (None, 1.0, False)])
def test_pack_vision_tokens_records_whether_the_frame_rate_was_known(
    fps: float | None, seconds_per_frame: float, known: bool
) -> None:
    builder = PackedSequenceBuilder()
    builder.begin_sample(0)
    builder.pack_vision_tokens(
        torch.zeros(1, 4, 2, 4, 4),  # [1,C,T,H,W]
        condition_frame_indexes_vision=[0],
        input_timestep=0.5,
        latent_patch_size=2,
        vision_fps=fps,
        enable_fps_modulation=False,
        base_fps=24.0,
        temporal_compression_factor=4,
        vision_temporal_positions=None,
        temporal_position_period=None,
    )

    assert builder.vision is not None
    assert builder.vision.seconds_per_frame == pytest.approx([seconds_per_frame])
    assert builder.vision.seconds_per_frame_known == [known]
