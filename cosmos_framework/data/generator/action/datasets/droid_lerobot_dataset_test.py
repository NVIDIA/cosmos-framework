# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

_IMAGE_FEATURES = {"wrist": "wrist_key", "left": "left_key", "right": "right_key"}


def _dataset(use_image_augmentation: bool, jitter_after_compose: bool = False) -> DROIDLeRobotDataset:
    """Build a bare instance exercising only ``_compose_multi_view``.

    ``__init__`` opens LeRobot datasets from disk, which this test does not need:
    the method under test reads three tensors out of ``sample`` and the
    augmentation attributes, and touches nothing else.
    """
    dataset = object.__new__(DROIDLeRobotDataset)
    dataset._image_features = _IMAGE_FEATURES
    dataset._use_image_augmentation = use_image_augmentation
    dataset._jitter_after_compose = jitter_after_compose
    dataset._image_augmentor = None
    dataset._color_augmentor = None
    return dataset


def _sample(wrist: float, left: float, right: float, t: int = 2, h: int = 16, w: int = 32) -> dict:
    return {
        "wrist_key": torch.full((t, 3, h, w), wrist),
        "left_key": torch.full((t, 3, h, w), left),
        "right_key": torch.full((t, 3, h, w), right),
    }


@pytest.mark.L0
def test_compose_multi_view_tiles_wrist_over_left_and_right() -> None:
    """Wrist spans the top, left and right tile the bottom at half scale."""
    composite = _dataset(use_image_augmentation=False)._compose_multi_view(_sample(0.1, 0.5, 0.9))

    assert composite.shape == (2, 3, 24, 32)  # [T,C,3H/2,W]
    torch.testing.assert_close(composite[:, :, :16, :], torch.full((2, 3, 16, 32), 0.1))
    torch.testing.assert_close(composite[:, :, 16:, :16], torch.full((2, 3, 8, 16), 0.5))
    torch.testing.assert_close(composite[:, :, 16:, 16:], torch.full((2, 3, 8, 16), 0.9))


@pytest.mark.L0
@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_compose_multi_view_keeps_layout_under_augmentation(jitter_after_compose: bool) -> None:
    """Augmentation must not change the composite's shape or tiling."""
    torch.manual_seed(0)
    dataset = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    composite = dataset._compose_multi_view(_sample(0.1, 0.5, 0.9))

    assert composite.shape == (2, 3, 24, 32)
    # Each region is still uniform, so the tiles did not bleed into each other.
    for region in (composite[:, :, :16, :], composite[:, :, 16:, :16], composite[:, :, 16:, 16:]):
        assert torch.allclose(region, region.flatten()[0].expand_as(region), atol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_color_jitter_is_shared_across_the_three_views(jitter_after_compose: bool) -> None:
    """One colour draw covers the whole composite.

    Jittering the views independently would cut colour diversity across the
    frame and make the three cameras disagree on lighting, so identical input
    views have to stay identical after augmentation.
    """
    torch.manual_seed(0)
    dataset = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    composite = dataset._compose_multi_view(_sample(0.4, 0.4, 0.4))

    wrist_px = composite[:, :, :16, :].flatten()[0]
    torch.testing.assert_close(composite[:, :, 16:, :16], torch.full((2, 3, 8, 16), wrist_px))
    torch.testing.assert_close(composite[:, :, 16:, 16:], torch.full((2, 3, 8, 16), wrist_px))


@pytest.mark.L0
@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_augmentation_actually_changes_the_composite(jitter_after_compose: bool) -> None:
    """Guards against the augmentor silently becoming a no-op."""
    sample = _sample(0.4, 0.6, 0.8)
    plain = _dataset(use_image_augmentation=False)._compose_multi_view(sample)
    torch.manual_seed(0)
    dataset = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    augmented = dataset._compose_multi_view(sample)

    assert not torch.allclose(plain, augmented)


@pytest.mark.L0
def test_default_keeps_jitter_inside_the_per_view_pipeline() -> None:
    """``jitter_after_compose`` defaults to the original ordering.

    Off: ColorJitter stays bundled in the per-view Compose and nothing runs on
    the composite, byte-identical to the pre-flag pipeline. On: ColorJitter
    leaves the per-view Compose and runs on the composite instead.
    """
    default = _dataset(use_image_augmentation=True)
    default._compose_multi_view(_sample(0.1, 0.5, 0.9))
    assert default._color_augmentor is None
    assert any(isinstance(t, T.ColorJitter) for t in default._image_augmentor.transforms)

    reordered = _dataset(use_image_augmentation=True, jitter_after_compose=True)
    reordered._compose_multi_view(_sample(0.1, 0.5, 0.9))
    assert isinstance(reordered._color_augmentor, T.ColorJitter)
    assert not any(isinstance(t, T.ColorJitter) for t in reordered._image_augmentor.transforms)
