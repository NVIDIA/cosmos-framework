"""Bounding boxes must use the selected camera's calibration, including pinhole."""

import numpy as np
import pytest

from cosmos_auto.wm.bbox_utils import (
    build_cuboid_bounding_box,
    create_bbox_geometry_objects_for_frame,
)
from cosmos_auto.wm.camera.ftheta import FThetaCamera
from cosmos_auto.wm.camera.pinhole import PinholeCamera
from cosmos_auto.wm.graphics_utils import (
    EDGE_INDICES,
    FthetaCurvedEdges,
    TriangleList2D,
    WorldSpaceLines,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _box(center, camera_pose=None):
    transform = np.eye(4)
    transform[:3, 3] = center
    if camera_pose is not None:
        transform = camera_pose @ transform
    return {
        "object_to_world": transform.tolist(),
        "object_lwh": [3.0, 2.0, 2.0],
        "object_type": "Car",
    }


def _camera():
    # Unequal focal lengths and an off-center principal point catch accidental
    # replacement by a symmetric or fitted camera model.
    return PinholeCamera(132.5, 91.25, 67.0, 43.0, 160, 96, device="cpu")


@pytest.mark.parametrize("center", [None, [0, 0, -10], [0, 0, 0]])
def test_pinhole_empty_behind_and_camera_containing_boxes(center):
    objects = {} if center is None else {"box": _box(center)}
    assert create_bbox_geometry_objects_for_frame(objects, np.eye(4), _camera(), fill_face="none") == []


@pytest.mark.parametrize(
    "fill_face,style",
    [("none", "solid"), ("all", "solid"), ("front_and_back", "diagonal")],
)
def test_pinhole_edges_use_original_intrinsics_and_camera_space_once(fill_face, style):
    camera = _camera()
    intrinsics = camera.intrinsics.copy()
    theta = np.deg2rad(23)
    pose = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 12],
            [np.sin(theta), np.cos(theta), 0, -3],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ]
    )
    local = np.eye(4)
    local[:3, 3] = [0.5, -0.25, 12]
    objects = {"box": _box(local[:3, 3], pose)}
    geometries = create_bbox_geometry_objects_for_frame(
        objects, pose, camera, fill_face=fill_face, fill_face_style=style
    )
    assert not any(isinstance(g, FthetaCurvedEdges) for g in geometries)
    edges = [g for g in geometries if isinstance(g, WorldSpaceLines)]
    assert len(edges) == (2 if style == "diagonal" else 1)
    assert edges[0].num_segments == 12
    if style == "diagonal":
        assert edges[1].num_segments == 4
    expected = build_cuboid_bounding_box(3.0, 2.0, 2.0, local)[np.asarray(EDGE_INDICES)]
    np.testing.assert_allclose(edges[0]._vertex_positions.reshape(-1, 2, 3), expected, atol=1e-6)
    for edge in edges:
        assert edge.camera_type == "pinhole"
        assert edge._cam_params == {"fx": 132.5, "fy": 91.25, "cx": 67.0, "cy": 43.0}
        np.testing.assert_array_equal(edge._w2c_col_major, np.eye(4, dtype=np.float32))
        points = edge._vertex_positions
        expected_pixels = np.column_stack(
            (
                132.5 * points[:, 0] / points[:, 2] + 67,
                91.25 * points[:, 1] / points[:, 2] + 43,
            )
        )
        np.testing.assert_allclose(camera.ray2pixel(points), expected_pixels, atol=1e-5)
        assert np.isfinite(edge._vertex_positions).all()
    fills = [g for g in geometries if isinstance(g, TriangleList2D)]
    assert len(fills) == (0 if fill_face == "none" else 1)
    np.testing.assert_array_equal(camera.intrinsics, intrinsics)
    assert not hasattr(camera, "_fw_poly")


def test_pinhole_edges_crossing_near_plane_are_clipped_before_projection():
    camera = PinholeCamera(20, 15, 64, 48, 128, 96, device="cpu")
    geometries = create_bbox_geometry_objects_for_frame({"box": _box([3, 0, 0.5])}, np.eye(4), camera, fill_face="none")
    edges = [g for g in geometries if isinstance(g, WorldSpaceLines)]
    assert len(edges) == 1
    points = edges[0]._vertex_positions
    assert np.all(points[:, 2] > 0)
    assert np.isfinite(camera.ray2pixel(points)).all()


def test_ftheta_still_uses_original_curved_edge_geometry():
    intrinsic = np.array([64.0, 64.0, 128.0, 128.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0])
    camera = FThetaCamera.from_numpy(intrinsic, device="cpu")
    geometry = create_bbox_geometry_objects_for_frame(
        {"box": _box([0, 0, 10])},
        np.eye(4),
        camera,
        fill_face="front_and_back",
        curve_subdivide=7,
    )
    edges = [g for g in geometry if isinstance(g, FthetaCurvedEdges)]
    assert len(edges) == 2
    assert edges[0].edge_endpoints.shape == (12, 2, 3)
    assert edges[1].edge_endpoints.shape == (4, 2, 3)
    assert edges[0].ftheta_params["n_steps"] == 7
    assert edges[1].depth_bias == -1e-4
    np.testing.assert_array_equal(edges[0].ftheta_params["c"], camera._fw_poly.coef)


@pytest.mark.parametrize("fail", [False, True])
def test_per_frame_pinhole_buffers_are_released_even_on_render_failure(monkeypatch, fail):
    geometries = create_bbox_geometry_objects_for_frame(
        {"box": _box([0, 0, 10])}, np.eye(4), _camera(), fill_face="none"
    )
    edge = next(g for g in geometries if isinstance(g, WorldSpaceLines))
    released = []

    class Buffer:
        def release(self):
            released.append(self)

    def render(self, ctx, program, **kwargs):
        self._pos_vbo = Buffer()
        self._color_vbo = Buffer()
        if fail:
            raise RuntimeError("render failed")

    monkeypatch.setattr(WorldSpaceLines, "render", render)
    if fail:
        with pytest.raises(RuntimeError, match="render failed"):
            edge.render(None, None, image_width=160)
    else:
        edge.render(None, None, image_width=160)
    assert len(released) == 2
    assert edge._pos_vbo is None
    assert edge._color_vbo is None
