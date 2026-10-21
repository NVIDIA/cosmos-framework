import os
import unittest

import numpy as np
import pytest
from cosmos_auto.wm.camera.ftheta import FThetaCamera
from cosmos_auto.wm.camera.pinhole import PinholeCamera
from cosmos_auto.wm.config import CosmosAVConfig
from cosmos_auto.wm.graphics_utils import WorldSpaceLines, WorldSpaceTriangles, render_geometries
from cosmos_auto.wm.occupancy_utils import points_below_ground_plane, precompute_ground_planes


@pytest.mark.L0
@pytest.mark.CPU
class WMV4BelowLevelFilterTest(unittest.TestCase):
    @staticmethod
    def _make_gradual_ramp():
        frame_count = 121
        camera_height = 1.5
        angles = np.deg2rad(np.linspace(0.0, 6.0, frame_count))
        x = np.linspace(0.0, 60.0, frame_count)
        z = np.zeros(frame_count)
        poses = np.repeat(np.eye(4)[None], frame_count, axis=0)
        normals = []
        for index, angle in enumerate(angles):
            if index:
                dx = x[index] - x[index - 1]
                z[index] = z[index - 1] + dx * np.tan(angle)
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.array([[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]])
            normal = rotation @ np.array([0.0, 0.0, 1.0])
            normals.append(normal)
            poses[index, :3, :3] = rotation
            poses[index, :3, 3] = np.array([x[index], 0.0, z[index]]) + camera_height * normal
        return poses, x, z, np.asarray(normals)

    def test_filter_is_below_only_and_capped_with_distance(self):
        plane = {
            "ground_point": np.zeros(3),
            "normal": np.array([0.0, 0.0, 1.0]),
        }
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -4.0],
                [0.0, 0.0, 5.0],
                [100.0, 0.0, -1.4],
                [100.0, 0.0, -1.6],
                [100.0, 0.0, -2.9],
                [100.0, 0.0, -3.1],
                [82.5, 0.0, -2.2],
                [82.5, 0.0, -2.3],
            ]
        )
        np.testing.assert_array_equal(
            points_below_ground_plane(points, plane),
            [False, True, False, False, False, False, True, False, True],
        )

        crossing_segment = points[[1, 0]]
        self.assertFalse(points_below_ground_plane(crossing_segment, plane).all())

    def test_smoothed_pose_planes_follow_a_gradual_ramp(self):
        poses, x, z, _ = self._make_gradual_ramp()

        planes = precompute_ground_planes(poses, poses[0], smooth_window=10)
        for index in (30, 60, 90):
            plane = planes[index]
            road_point = np.array([x[index], 0.0, z[index]])
            lower_point = road_point - 5.0 * plane["normal"]
            self.assertFalse(points_below_ground_plane(road_point[None], plane)[0])
            self.assertTrue(points_below_ground_plane(lower_point[None], plane)[0])

    def test_gradual_ramp_keeps_lookahead_road(self):
        poses, x, z, normals = self._make_gradual_ramp()
        frame_count = len(poses)

        planes = precompute_ground_planes(poses, poses[0], smooth_window=10)
        for frame in (30, 60, 90):
            lookahead = np.minimum(frame + np.array([10, 20, 40]), frame_count - 1)
            road = np.stack([x[lookahead], np.zeros(3), z[lookahead]], axis=1)
            self.assertFalse(points_below_ground_plane(road, planes[frame]).any())
            lower = road - 5.0 * normals[lookahead]
            self.assertTrue(points_below_ground_plane(lower, planes[frame]).all())

    def test_worldspace_geometry_reuses_buffers_while_plane_changes(self):
        camera = PinholeCamera(100.0, 100.0, 64.0, 64.0, 128, 128)
        segments = np.array([[[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]]])
        geometry = WorldSpaceLines(
            segments,
            np.ones((2, 3)),
            4.0,
            camera,
            below_level_filter=True,
        )
        self.assertIsNone(geometry._level_positions)
        original_positions = geometry._vertex_positions.copy()
        first = {"ground_point": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
        second = {
            "ground_point": np.array([10.0, 0.0, 2.0]),
            "normal": np.array([-0.1, 0.0, 1.0]),
        }
        geometry.set_level_plane(first)
        first_point = geometry._level_ground_point.copy()
        geometry.set_level_plane(second)
        np.testing.assert_allclose(geometry._vertex_positions, original_positions)
        self.assertFalse(np.allclose(geometry._level_ground_point, first_point))
        self.assertAlmostEqual(np.linalg.norm(geometry._level_ground_normal), 1.0, places=6)

    @unittest.skipUnless(os.environ.get("RUN_GPU_TESTS") == "1", "requires an EGL GPU")
    def test_gpu_filter_compiles_and_preserves_overpasses(self):
        plane = {"ground_point": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
        segments = np.array(
            [
                [[-2.0, -0.4, 10.0], [2.0, -0.4, 10.0]],
                [[-2.0, 0.0, 5.0], [2.0, 0.0, 5.0]],
                [[-2.0, 0.4, 15.0], [2.0, 0.4, 15.0]],
            ]
        )
        level_positions = np.array(
            [
                [[-2.0, -0.4, 0.0], [2.0, -0.4, 0.0]],
                [[-2.0, 0.0, -5.0], [2.0, 0.0, -5.0]],
                [[-2.0, 0.4, 5.0], [2.0, 0.4, 5.0]],
            ]
        )
        colors = np.repeat(np.eye(3), 2, axis=0)
        cameras = [PinholeCamera(100.0, 100.0, 64.0, 64.0, 128, 128)]
        ftheta_intrinsic = np.array(
            [64.0, 64.0, 128.0, 128.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        cameras.append(FThetaCamera.from_numpy(ftheta_intrinsic, device="cpu"))

        for camera in cameras:
            with self.subTest(camera=type(camera).__name__):
                enabled = WorldSpaceLines(
                    segments,
                    colors,
                    6.0,
                    camera,
                    below_level_filter=True,
                    level_positions=level_positions,
                )
                enabled.set_world_to_camera(np.eye(4))
                enabled.set_level_plane(plane)
                image = render_geometries(
                    [enabled],
                    128,
                    128,
                    75.0,
                    depth_gradient=False,
                    multi_sample=1,
                    use_cache=False,
                )
                self.assertGreater(image[..., 0].max(), 0)
                self.assertEqual(image[..., 1].max(), 0)
                self.assertGreater(image[..., 2].max(), 0)

                enabled = WorldSpaceLines(
                    segments,
                    colors,
                    6.0,
                    camera,
                    below_level_filter=True,
                    level_positions=level_positions,
                )
                enabled.set_world_to_camera(np.eye(4))
                enabled.set_level_plane(plane)
                disabled = WorldSpaceLines(
                    segments,
                    colors,
                    6.0,
                    camera,
                    below_level_filter=False,
                    level_positions=level_positions,
                )
                disabled.set_world_to_camera(np.eye(4))
                triangles = WorldSpaceTriangles(
                    vertices=np.array(
                        [
                            [-1.0, -1.0, 12.0],
                            [1.0, -1.0, 12.0],
                            [0.0, 1.0, 12.0],
                            [-1.0, -1.0, 7.0],
                            [1.0, -1.0, 7.0],
                            [0.0, 1.0, 7.0],
                        ]
                    ),
                    colors=np.tile([1.0, 1.0, 0.0], (6, 1)),
                    camera_model=camera,
                    below_level_filter=True,
                    level_positions=np.array(
                        [
                            [-1.0, -1.0, 2.0],
                            [1.0, -1.0, 2.0],
                            [0.0, 1.0, 2.0],
                            [-1.0, -1.0, -5.0],
                            [1.0, -1.0, -5.0],
                            [0.0, 1.0, -5.0],
                        ]
                    ),
                )
                triangles.set_world_to_camera(np.eye(4))
                triangles.set_level_plane(plane)
                image = render_geometries(
                    [enabled, disabled, triangles],
                    128,
                    128,
                    75.0,
                    depth_gradient=False,
                    multi_sample=1,
                    use_cache=False,
                )
                self.assertGreater(image[..., 1].max(), 0)

                triangle_only = WorldSpaceTriangles(
                    vertices=np.array(
                        [
                            [-1.0, -1.0, 12.0],
                            [1.0, -1.0, 12.0],
                            [0.0, 1.0, 12.0],
                            [-1.0, -1.0, 7.0],
                            [1.0, -1.0, 7.0],
                            [0.0, 1.0, 7.0],
                        ]
                    ),
                    colors=np.concatenate([np.tile([0.0, 0.0, 1.0], (3, 1)), np.tile([0.0, 1.0, 0.0], (3, 1))]),
                    camera_model=camera,
                    below_level_filter=True,
                    level_positions=np.array(
                        [
                            [-1.0, -1.0, 2.0],
                            [1.0, -1.0, 2.0],
                            [0.0, 1.0, 2.0],
                            [-1.0, -1.0, -5.0],
                            [1.0, -1.0, -5.0],
                            [0.0, 1.0, -5.0],
                        ]
                    ),
                )
                triangle_only.set_world_to_camera(np.eye(4))
                triangle_only.set_level_plane(plane)
                image = render_geometries(
                    [triangle_only],
                    128,
                    128,
                    75.0,
                    depth_gradient=False,
                    multi_sample=1,
                    use_cache=False,
                )
                self.assertEqual(image[..., 1].max(), 0)
                self.assertGreater(image[..., 2].max(), 0)

    def test_filter_is_enabled_by_default(self):
        self.assertTrue(CosmosAVConfig.__dataclass_fields__["filter_below_ego_level"].default)


if __name__ == "__main__":
    unittest.main()
