"""Physical controls retain geometry and pool occupied returns without interpolation."""

import numpy as np
import pytest
from safetensors.numpy import load_file

from cosmos_auto.lidar.control import save_control

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_physical_control_pooling_and_empty_rays(tmp_path):
    ranges = np.zeros((2, 128, 3600), dtype=np.float32)
    ranges[0, 4, :4] = [0, 12, 18, 7]
    path = tmp_path / "lidar_control.safetensors"
    save_control(ranges, [0, 3], path, "scene")
    frames = load_file(path)["frames"]
    assert frames.shape == (3, 2, 128, 1800)
    assert frames.dtype == np.float32
    np.testing.assert_array_equal(frames[0, 0, 4, :3], [12, 7, 0])
    np.testing.assert_array_equal(frames[1], frames[2])
    np.testing.assert_array_equal(frames[2], frames[0] > 0)
    assert not frames[:, 1].any()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_control_rejects_nonphysical_ranges(tmp_path, bad):
    ranges = np.zeros((1, 128, 1800), dtype=np.float32)
    ranges[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite, nonnegative"):
        save_control(ranges, [0], tmp_path / "bad.safetensors", "scene")
