import unittest
from unittest import mock

import numpy as np
import pytest
from cosmos_auto.wm import laneline_utils, minimap_utils, occupancy_utils

EMPTY_LABEL = {"labelFamily": "EMPTY"}


def cuboid_label():
    return {
        "labelFamily": "SHAPE3D",
        "labelData": {
            "shape3d": {
                "cuboid3d": {
                    "vertices": [
                        [-1, -1, 11],
                        [1, -1, 11],
                        [-1, -1, 10],
                        [1, -1, 10],
                        [-1, 1, 11],
                        [1, 1, 11],
                        [-1, 1, 10],
                        [1, 1, 10],
                    ]
                },
                "attributes": [],
            }
        },
    }


def elevated_cuboid_label(level, feature_id="feature", sign_class=None):
    vertices = np.array(
        [
            [-0.1, -0.5, -0.5],
            [-0.1, 0.5, -0.5],
            [0.1, 0.5, -0.5],
            [0.1, -0.5, -0.5],
            [-0.1, -0.5, 0.5],
            [-0.1, 0.5, 0.5],
            [0.1, 0.5, 0.5],
            [0.1, -0.5, 0.5],
        ]
    )
    vertices += np.array([0.0, 0.0, level])
    attributes = [{"name": "feature_id", "text": feature_id}]
    if sign_class is not None:
        attributes.append({"name": "sign_class", "text": sign_class})
    return {
        "labelFamily": "SHAPE3D",
        "labelData": {
            "shape3d": {
                "cuboid3d": {"vertices": vertices.tolist()},
                "attributes": attributes,
            }
        },
    }


class IdentityCamera:
    def get_xy_and_depth(self, vertices, camera_pose):
        return np.asarray(vertices, dtype=float)


class FixedDepthCamera:
    def get_xy_and_depth(self, vertices, camera_pose):
        vertices = np.asarray(vertices, dtype=float)
        return np.column_stack([vertices[:, :2], np.full(len(vertices), 10.0)])


@pytest.mark.L0
@pytest.mark.CPU
class WMV4SingleLabelComponentTest(unittest.TestCase):
    def test_laneline_distinguishes_real_label_from_empty_sentinel(self):
        real_label = {
            "labelFamily": "SHAPE3D",
            "labelData": {
                "shape3d": {
                    "polyline3d": {"vertices": [[0, 0, 10], [1, 0, 10]]},
                    "attributes": [
                        {"name": "colors", "enumsList": {"enumsList": ["WHITE"]}},
                        {"name": "styles", "enumsList": {"enumsList": ["SOLID_SINGLE"]}},
                    ],
                }
            },
        }

        for labels, expected_count in (([real_label], 1), ([EMPTY_LABEL], 0)):
            with (
                self.subTest(labels=labels),
                mock.patch.object(
                    laneline_utils,
                    "get_sample",
                    return_value={"lanelines.json": {"labels": labels}},
                ),
            ):
                result = laneline_utils.prepare_laneline_type_geometry_data("unused.tar")
                self.assertEqual(len(result), expected_count)

    def test_sign_and_light_distinguish_real_label_from_empty_sentinel(self):
        camera = IdentityCamera()
        pose = np.eye(4)
        real_label = cuboid_label()

        real_sign = minimap_utils.create_traffic_sign_geometry_objects_from_data_v4(
            [real_label], pose, camera, {"back": [40, 40, 40]}, [100, 100, 100]
        )
        empty_sign = minimap_utils.create_traffic_sign_geometry_objects_from_data_v4(
            [EMPTY_LABEL], pose, camera, {"back": [40, 40, 40]}, [100, 100, 100]
        )

        tl_palette = {
            "back": [30, 30, 30],
            "edge": [200, 200, 200],
            "unknown": [80, 80, 80],
        }
        real_light = minimap_utils.create_traffic_light_status_geometry_objects_from_data_v4(
            [real_label], {}, 0, pose, camera, tl_palette
        )
        empty_light = minimap_utils.create_traffic_light_status_geometry_objects_from_data_v4(
            [EMPTY_LABEL], {}, 0, pose, camera, tl_palette
        )

        self.assertGreater(len(real_sign), 0)
        self.assertEqual(empty_sign, [])
        self.assertGreater(len(real_light), 0)
        self.assertEqual(empty_light, [])

    def test_sign_and_light_filter_only_wholly_below_cuboids(self):
        camera = FixedDepthCamera()
        pose = np.eye(4)
        plane = {"ground_point": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
        below = elevated_cuboid_label(-5.0)
        stricter_than_ground_geometry = elevated_cuboid_label(-1.0)
        crossing = elevated_cuboid_label(0.0)
        above = elevated_cuboid_label(3.0)
        sign_colors = {"back": [40, 40, 40]}
        light_colors = {
            "back": [30, 30, 30],
            "edge": [200, 200, 200],
            "unknown": [80, 80, 80],
        }

        for builder, args in (
            (
                minimap_utils.create_traffic_sign_geometry_objects_from_data_v4,
                (pose, camera, sign_colors, [100, 100, 100]),
            ),
            (
                minimap_utils.create_traffic_light_status_geometry_objects_from_data_v4,
                ({}, 0, pose, camera, light_colors),
            ),
        ):
            with self.subTest(builder=builder.__name__):
                self.assertGreater(len(builder([below], *args)), 0)
                self.assertEqual(builder([below], *args, level_plane=plane), [])
                self.assertEqual(
                    builder([stricter_than_ground_geometry], *args, level_plane=plane),
                    [],
                )
                self.assertGreater(len(builder([crossing], *args, level_plane=plane)), 0)
                self.assertGreater(len(builder([above], *args, level_plane=plane)), 0)

    def test_traffic_light_filter_preserves_original_status_index(self):
        camera = FixedDepthCamera()
        pose = np.eye(4)
        plane = {"ground_point": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
        labels = [
            elevated_cuboid_label(-5.0, "below"),
            elevated_cuboid_label(3.0, "above"),
        ]
        states = {
            "0": {"feature_id": "below", "state": ["green"]},
            "1": {"feature_id": "above", "state": ["red"]},
        }
        palette = {
            "back": [30, 30, 30],
            "edge": [200, 200, 200],
            "unknown": [80, 80, 80],
            "red": [255, 0, 0],
            "green": [0, 255, 0],
        }
        geometry = minimap_utils.create_traffic_light_status_geometry_objects_from_data_v4(
            labels, states, 0, pose, camera, palette, level_plane=plane
        )
        triangle_colors = geometry[0].base_color
        self.assertTrue(np.any(np.all(triangle_colors == [1.0, 0.0, 0.0], axis=1)))
        self.assertFalse(np.any(np.all(triangle_colors == [0.0, 1.0, 0.0], axis=1)))

    def test_traffic_sign_filter_preserves_parallel_metadata(self) -> None:
        camera = FixedDepthCamera()
        pose = np.eye(4)
        plane = {"ground_point": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
        labels = [
            elevated_cuboid_label(-5.0, sign_class="TRAFFIC_SIGN_REGULATORY_R1_STOP"),
            elevated_cuboid_label(3.0, sign_class="TRAFFIC_SIGN_REGULATORY_R1_YIELD"),
        ]
        palette = {
            "back": [40, 40, 40],
            "stop": [255, 0, 0],
            "yield": [0, 255, 0],
        }
        geometry = minimap_utils.create_traffic_sign_geometry_objects_from_data_v4(
            labels,
            pose,
            camera,
            palette,
            [100, 100, 100],
            level_plane=plane,
        )
        triangle_colors = geometry[0].base_color
        self.assertTrue(np.any(np.all(triangle_colors == [0.0, 1.0, 0.0], axis=1)))
        self.assertFalse(np.any(np.all(triangle_colors == [1.0, 0.0, 0.0], axis=1)))

    def test_occupancy_culling_distinguishes_real_label_from_empty_sentinel(self):
        self.assertEqual(
            occupancy_utils._cuboid_vertices_from_labels([cuboid_label()]).shape,
            (1, 8, 3),
        )
        self.assertEqual(
            occupancy_utils._cuboid_vertices_from_labels([EMPTY_LABEL]).shape,
            (0, 8, 3),
        )


if __name__ == "__main__":
    unittest.main()
