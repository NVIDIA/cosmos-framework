# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.data.generator.action.utils.episode_filters import normalize_episode_filter_keys

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    "key",
    [
        "episode-1",
        "dataset/episode-1",
    ],
)
def test_filter_ranges_do_not_depend_on_storage_location(key: str) -> None:
    ranges = [[0, 12], [20, 35]]
    assert normalize_episode_filter_keys({key: ranges}) == {key: ranges}


def test_uri_filter_keys_preserve_nested_episode_ids() -> None:
    key = "s3://example-bucket/dataset/day-1/episode-1/recordings/MP4--s3://example-bucket/dataset/day-1/episode-1/trajectory.h5"
    assert normalize_episode_filter_keys({key: [[0, 12]]}, source_prefix="s3://example-bucket/dataset") == {
        "day-1/episode-1": [[0, 12]]
    }


def test_uri_filter_keys_require_an_operator_prefix() -> None:
    with pytest.raises(ValueError, match="source_prefix is required"):
        normalize_episode_filter_keys({"s3://example-bucket/episode-1": [[0, 12]]})


def test_conflicting_episode_ranges_are_refused() -> None:
    with pytest.raises(ValueError, match="Conflicting filter ranges"):
        normalize_episode_filter_keys(
            {"episode-1": [[0, 12]], "s3://example-bucket/dataset/episode-1": [[0, 24]]},
            source_prefix="s3://example-bucket/dataset",
        )


def test_matching_duplicate_episode_ranges_are_preserved_once() -> None:
    assert normalize_episode_filter_keys(
        {"s3://example-bucket/dataset/episode-1": [[0, 12]], "episode-1": [[0, 12]]},
        source_prefix="s3://example-bucket/dataset",
    ) == {"episode-1": [[0, 12]]}
