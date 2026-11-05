import unittest

import numpy as np
import pytest
from cosmos_auto.wm.camera.base import encoder_safe_resolution
from cosmos_auto.wm.camera.ftheta import FThetaCamera


@pytest.mark.L0
@pytest.mark.CPU
class SourceResolutionTest(unittest.TestCase):
    def test_even_intrinsic_resolution_is_preserved(self):
        self.assertEqual(
            encoder_safe_resolution((1924, 1084)),
            (1924, 1084),
        )

    def test_odd_intrinsic_resolution_is_rounded_up(self):
        self.assertEqual(
            encoder_safe_resolution((1719, 1080)),
            (1720, 1080),
        )

    def test_ftheta_rescale_to_matches_anisotropic_image_scaling(self):
        intrinsic = np.asarray(
            [
                859.60636,
                539.7604,
                1719.0,
                1080.0,
                0.0,
                0.0027604087,
                1.1222094e-7,
                -2.1748277e-9,
                1.6450183e-12,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        camera = FThetaCamera.from_numpy(intrinsic, device="cpu")
        rays = np.asarray(
            [[0.0, 0.0, 1.0], [0.1, -0.2, 1.0], [-0.3, 0.15, 0.8]],
            dtype=np.float32,
        )
        original_pixels = camera.ray2pixel(rays)
        original_backprojected, original_valid = camera.pixel2ray(original_pixels[1:])

        camera.rescale_to(1720, 1080)
        scaled_pixels = camera.ray2pixel(rays)
        scaled_backprojected, scaled_valid = camera.pixel2ray(scaled_pixels[1:])

        expected = original_pixels * np.asarray([1720 / 1719, 1.0])
        np.testing.assert_allclose(scaled_pixels, expected, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(scaled_backprojected, original_backprojected, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(scaled_valid, original_valid)
        self.assertEqual((camera.width, camera.height), (1720, 1080))
        np.testing.assert_allclose(camera.intrinsics[2:4], [1720, 1080])


if __name__ == "__main__":
    unittest.main()
