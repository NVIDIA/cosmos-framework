import unittest

import numpy as np
import pytest
from cosmos_auto.wm.bbox_utils import (
    _camera_inside_object_box,
    create_bbox_geometry_objects_for_frame,
    create_bbox_projection,
    interpolate_bbox,
)
from cosmos_auto.wm.camera.ftheta import FThetaCamera
from cosmos_auto.wm.graphics_utils import FthetaCurvedEdges
from scipy.spatial.transform import Rotation


def object_info(center, lwh=(4.0, 2.0, 2.0), rotation=None):
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3) if rotation is None else rotation
    transform[:3, 3] = center
    return {
        "object_to_world": transform.tolist(),
        "object_lwh": list(lwh),
        "object_is_moving": True,
        "object_type": "Car",
    }


def camera_pose(center):
    pose = np.eye(4)
    pose[:3, 3] = center
    return pose


def ftheta_camera():
    intrinsic = np.array([64.0, 64.0, 128.0, 128.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0])
    return FThetaCamera.from_numpy(intrinsic, device="cpu")


class RecordingCamera:
    def __init__(self):
        self.polylines = []

    def draw_line_depth(self, pose, polylines, radius, colors):
        self.polylines.append(polylines)
        return np.zeros((1, 2, 2, 3), dtype=np.uint8)


@pytest.mark.L0
@pytest.mark.CPU
class BBoxEgoSuppressionTest(unittest.TestCase):
    def test_containment_uses_oriented_object_coordinates(self):
        rotation = Rotation.from_euler("z", 37, degrees=True).as_matrix()
        center = np.array([8.0, -3.0, 1.0])
        info = object_info(center, rotation=rotation)
        camera_center = center + rotation @ np.array([1.5, 0.7, -0.4])
        self.assertTrue(_camera_inside_object_box(info, camera_pose(camera_center)))

        for local_center in ([2.01, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.01]):
            with self.subTest(local_center=local_center):
                world_center = center + rotation @ np.asarray(local_center)
                self.assertFalse(_camera_inside_object_box(info, camera_pose(world_center)))

    def test_boundary_allows_only_numerical_epsilon(self):
        info = object_info([0.0, 0.0, 0.0])
        self.assertTrue(_camera_inside_object_box(info, camera_pose([2.0, 1.0, 1.0])))
        self.assertTrue(_camera_inside_object_box(info, camera_pose([2.0 + 5e-7, 0.0, 0.0])))
        self.assertFalse(_camera_inside_object_box(info, camera_pose([2.0 + 2e-6, 0.0, 0.0])))

    def test_real_clip_frame_74_contains_frontwide_camera(self):
        info = {
            "object_to_world": [
                [0.9998260168643893, -0.017919832170659473, -0.0051783796756448095, 1.7768907892381647],
                [0.017920610501086794, 0.9998394075630042, 0.00010393906450356515, -0.03705497896473225],
                [0.005175685496441129, -0.0001967207060531852, 0.9999865867003446, 0.9216583552097496],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "object_lwh": [4.725716590881348, 2.0045511722564697, 1.776111125946045],
        }
        pose = np.array(
            [
                [-0.009132727058959305, -0.013779481467868372, 0.9998633298068345, 2.040672287992455],
                [-0.999048908828742, 0.04275999159566101, -0.008535997106694348, -0.05314193184672871],
                [-0.04263652589171368, -0.9989903237601789, -0.014156891154573555, 1.591223358126054],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.assertTrue(_camera_inside_object_box(info, pose))

    def test_modern_builder_drops_only_camera_containing_box(self):
        geometry = create_bbox_geometry_objects_for_frame(
            {
                "ego": object_info([0.0, 0.0, 0.0], lwh=(6.0, 4.0, 4.0)),
                "other": object_info([0.0, 0.0, 10.0]),
            },
            np.eye(4),
            ftheta_camera(),
            fill_face="none",
        )
        edges = [item for item in geometry if isinstance(item, FthetaCurvedEdges)]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].edge_endpoints.shape, (12, 2, 3))
        self.assertAlmostEqual(edges[0].edge_endpoints[..., 2].min(), 9.0)
        self.assertAlmostEqual(edges[0].edge_endpoints[..., 2].max(), 11.0)

    def test_oldest_cpu_projection_drops_only_camera_containing_box(self):
        camera = RecordingCamera()
        create_bbox_projection(
            {
                "000000.all_object_info.json": {
                    "ego": object_info([0.0, 0.0, 0.0], lwh=(6.0, 4.0, 4.0)),
                    "other": object_info([0.0, 0.0, 10.0]),
                }
            },
            np.eye(4)[None],
            [0],
            camera,
        )
        self.assertEqual([len(items) for items in camera.polylines], [1, 0, 0, 0, 0])
        remaining = np.asarray(camera.polylines[0][0])
        self.assertAlmostEqual(remaining[:, 2].min(), 9.0)
        self.assertAlmostEqual(remaining[:, 2].max(), 11.0)

    def test_interpolated_box_is_checked_at_rendered_frame(self):
        frames = {
            "000000.all_object_info.json": {"ego": object_info([-1.0, 0.0, 0.0])},
            "000002.all_object_info.json": {"ego": object_info([1.0, 0.0, 0.0])},
        }
        interpolated = interpolate_bbox(frames, [1])
        info = interpolated["000001.all_object_info.json"]["ego"]
        self.assertTrue(_camera_inside_object_box(info, np.eye(4)))
        self.assertEqual(
            create_bbox_geometry_objects_for_frame({"ego": info}, np.eye(4), ftheta_camera(), fill_face="none"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
