# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import (
    _can_encode_frame_zero_policy_batch,
    _encode_frame_zero_conditioned_video,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _CausalTokenizer:
    def __init__(self) -> None:
        self.encoded_frames: list[int] = []

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // 4

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        self.encoded_frames.append(int(state.shape[2]))
        latent_frames = self.get_latent_num_frames(int(state.shape[2]))
        latent = state.new_full((state.shape[0], 2, latent_frames, 2, 2), 7)
        latent[:, :, 0].copy_(state[:, :2, 0, :2, :2])
        return latent


def test_encode_frame_zero_conditioned_video_preserves_shape_and_prefix() -> None:
    tokenizer = _CausalTokenizer()
    state = torch.arange(1 * 3 * 33 * 2 * 2).reshape(1, 3, 33, 2, 2).float()
    full = tokenizer.encode(state)
    tokenizer.encoded_frames.clear()

    actual = _encode_frame_zero_conditioned_video(tokenizer, state)

    assert tokenizer.encoded_frames == [1]
    assert actual.shape == (1, 2, 9, 2, 2)
    torch.testing.assert_close(actual[:, :, :1], full[:, :, :1])
    assert torch.count_nonzero(actual[:, :, 1:]) == 0


def test_encode_frame_zero_conditioned_video_keeps_single_frame_inputs() -> None:
    tokenizer = _CausalTokenizer()
    state = torch.ones((1, 3, 1, 2, 2))

    actual = _encode_frame_zero_conditioned_video(tokenizer, state)

    assert tokenizer.encoded_frames == [1]
    assert actual.shape == (1, 2, 1, 2, 2)


@pytest.mark.parametrize(
    ("modes", "conditioned_frames", "model_training", "multi_item", "causal", "expected"),
    [
        (["policy"], [[0]], False, False, True, True),
        (["policy"], [[0]], True, False, True, False),
        (["policy"], [[0]], False, True, True, False),
        (["policy"], [[0]], False, False, False, False),
        (["inverse_dynamics"], [[0]], False, False, True, False),
        (["policy"], [[0, 1]], False, False, True, False),
        (["policy", "policy"], [[0]], False, False, True, False),
    ],
)
def test_can_encode_frame_zero_policy_batch(
    modes: list[str],
    conditioned_frames: list[list[int]],
    model_training: bool,
    multi_item: bool,
    causal: bool,
    expected: bool,
) -> None:
    plans = [SimpleNamespace(condition_frame_indexes_vision=frames) for frames in conditioned_frames]

    assert (
        _can_encode_frame_zero_policy_batch(
            {"mode": modes, "sequence_plan": plans},
            batch_size=1,
            model_training=model_training,
            has_multiple_vision_items=multi_item,
            tokenizer_is_causal=causal,
        )
        is expected
    )
