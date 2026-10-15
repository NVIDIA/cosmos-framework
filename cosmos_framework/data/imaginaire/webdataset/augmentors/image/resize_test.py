# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.imaginaire.webdataset.augmentors.image.resize import ResizeToSize
from cosmos_framework.utils import log

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    "source_size,target_size,accepted",
    [
        ((100, 100), (103, 100), True),
        ((100, 100), (100, 103), True),
        ((103, 100), (100, 100), True),
        ((100, 103), (100, 100), True),
        ((100, 100), (104, 100), False),
        ((100, 100), (100, 104), False),
        ((100, 100), (206, 200), True),
        ((160, 90), (832, 480), True),
        ((90, 160), (480, 832), True),
        ((240, 100), (832, 480), False),
    ],
)
def test_resize_filters_by_symmetric_axis_scaling_without_mutating_rejected_samples(
    source_size: tuple[int, int], target_size: tuple[int, int], accepted: bool
) -> None:
    source_w, source_h = source_size
    target_w, target_h = target_size
    source = torch.zeros(3, source_h, source_w)  # [3,H_source,W_source]
    sample = {"video": source, "aspect_ratio": "bucket"}
    resize = ResizeToSize(
        input_keys=["video"],
        output_keys=["resized"],
        args={"size": {"bucket": target_size}, "max_aspect_ratio_distortion": 0.03},
    )

    output = resize(sample)

    if accepted:
        assert output is not None
        assert output["resized"].shape == (3, target_h, target_w)
        assert output["image_size"].tolist() == [target_h, target_w, target_h, target_w]
        assert "video" not in output
    else:
        assert output is None
        assert sample["video"] is source
        assert "resized" not in sample
        assert "image_size" not in sample


def test_direct_resize_without_a_distortion_limit_remains_unfiltered() -> None:
    source = torch.zeros(3, 10, 20)  # [3,10,20]
    resize = ResizeToSize(input_keys=["video"], args={"size": {"bucket": (10, 10)}})

    output = resize({"video": source, "aspect_ratio": "bucket"})

    assert output is not None
    assert output["video"].shape == (3, 10, 10)


def test_aspect_filter_reports_bounded_per_bucket_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(log, "info", lambda message, **kwargs: messages.append(message))
    resize = ResizeToSize(
        input_keys=["video"], args={"size": {"square": (10, 10)}, "max_aspect_ratio_distortion": 0.03}
    )
    for index in range(1000):
        # Alternate accepted and rejected geometry while retaining the metadata label.
        source = torch.zeros(3, 10, 10 if index % 2 else 20)  # [3,10,W_source]
        resize({"video": source, "aspect_ratio": "square"})

    assert resize.aspect_ratio_stats.counts == {("square", (10, 10)): (1000, 500)}
    assert len(messages) == 2
    assert "checked=1000, accepted=500, rejected=500, acceptance=50.00%" in messages[-1]
    assert "worker-local" in messages[-1]
    assert "aspect=square" in messages[-1]


def test_aspect_filter_worker_counts_reset_after_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    resize = ResizeToSize(
        input_keys=["video"], args={"size": {"square": (10, 10)}, "max_aspect_ratio_distortion": 0.03}
    )
    resize({"video": torch.zeros(3, 10, 20), "aspect_ratio": "square"})  # [3,10,20]
    monkeypatch.setattr("cosmos_framework.data.imaginaire.webdataset.augmentors.image.resize.os.getpid", lambda: -1)
    resize({"video": torch.zeros(3, 10, 10), "aspect_ratio": "square"})  # [3,10,10]
    assert resize.aspect_ratio_stats.counts == {("square", (10, 10)): (1, 0)}
