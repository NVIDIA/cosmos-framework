# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression tests for action normalization on the LeRobot dataset base.

``action_normalization`` is a silent setting: a dataset that ignores it still
returns plausible floats, and every consumer downstream -- transforms, the
serving JSON, the model -- accepts them without complaint.  Nothing raises, so
these tests are the only thing standing between a refactor and actions that are
tens of times off in scale.
"""

import json

import pytest
import torch

from cosmos_framework.data.generator.action.action_processing import (
    load_action_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    BaseActionLeRobotDataset,
)
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset


class _FakeDataset(BaseActionLeRobotDataset):
    """Exercise ``_build_result`` without touching LeRobot data or video files."""

    def __init__(self, normalizer=None) -> None:
        # Bypass __init__: it opens a dataset root. Only the attributes
        # _build_result reads are needed here.
        self._action_normalizer = normalizer
        self._skip_video_loading = True
        self._pose_convention = None
        self._rotation_format = None

    def _compute_idle_frames(self, action):  # noqa: D102 - fixture stub
        return None


def _droid_stub() -> DROIDLeRobotDataset:
    """A DROID instance without __init__, for path resolution only.

    The three attributes are what the inherited filename convention reads, so
    that a dataset which has lost its override still resolves to a concrete
    path and fails on the missing file rather than on a missing attribute.
    """
    stub = object.__new__(DROIDLeRobotDataset)
    stub._embodiment_type = DROIDLeRobotDataset.EMBODIMENT_TYPE
    stub._pose_convention = "backward_framewise"
    stub._rotation_format = "rot6d"
    return stub


def _stats(q01: list[float], q99: list[float]) -> dict[str, torch.Tensor]:
    return {
        "q01": torch.tensor(q01, dtype=torch.float32),
        "q99": torch.tensor(q99, dtype=torch.float32),
    }


def test_build_result_applies_the_configured_normalizer() -> None:
    """A configured normalizer must reach the returned sample.

    Regression: the normalizer used to be constructed and then never applied,
    so ``action_normalization`` silently did nothing.
    """
    normalizer = resolve_action_normalization("quantile", _stats([-1.0, 0.0], [1.0, 2.0]))
    action = torch.tensor([[-1.0, 0.0], [1.0, 2.0], [0.0, 1.0]], dtype=torch.float32)

    result = _FakeDataset(normalizer)._build_result(mode="wam", video=None, action=action, ai_caption="")

    # quantile maps [q01, q99] onto [-1, 1].
    torch.testing.assert_close(
        result["action"],
        torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
    )


def test_build_result_leaves_action_untouched_without_a_normalizer() -> None:
    action = torch.tensor([[0.3, -0.7]], dtype=torch.float32)
    result = _FakeDataset(None)._build_result(mode="wam", video=None, action=action, ai_caption="")
    torch.testing.assert_close(result["action"], action)


def test_droid_bundled_stats_file_exists() -> None:
    """The DROID stats path must resolve to a file that ships with the package.

    Regression: DROID inherited the base filename convention, which resolves to
    ``datasets/normalizers/droid_lerobot_backward_framewise_rot6d.json`` -- a
    directory that does not exist -- so requesting normalization raised
    FileNotFoundError instead of normalizing.
    """
    path = _droid_stub()._normalizer_path()
    assert path.is_file(), f"bundled DROID stats missing at {path}"

    stats = load_action_stats(str(path))
    assert set(stats) >= {"q01", "q99"}
    assert len(stats["q01"]) == len(stats["q99"]) == 10  # pos(3) + rot6d(6) + gripper(1)


@pytest.mark.parametrize("method", ["quantile", "quantile_rot"])
def test_droid_stats_load_for_every_supported_method(method: str) -> None:
    """Both DROID normalization modes must resolve against the bundled stats."""
    path = _droid_stub()._normalizer_path()
    stats_key = "global_raw" if method == "quantile_rot" else "global"
    raw = load_action_stats(str(path), stats_key=stats_key)
    stats = {key: torch.from_numpy(value).float() for key, value in raw.items()}
    normalizer = resolve_action_normalization(method, stats, apply_forward_clamp=True)

    # A gripper action at the q99 upper quantile normalizes to +1.
    action = torch.zeros((1, 10), dtype=torch.float32)
    action[0, 9] = float(json.loads(path.read_text())["q99"][9])
    assert normalizer.normalize_action(action)[0, 9].item() == pytest.approx(1.0)
