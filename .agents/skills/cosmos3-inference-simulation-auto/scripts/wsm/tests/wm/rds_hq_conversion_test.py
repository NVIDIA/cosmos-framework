"""Exercise converted archives through the WSM4 readers and geometry builders."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from cosmos_auto.wm.bbox_utils import fix_static_objects, interpolate_bbox
from cosmos_auto.wm.camera.pinhole import PinholeCamera
from cosmos_auto.wm.minimap_utils import (
    create_traffic_light_status_geometry_objects_from_data_v4,
    prepare_traffic_light_status_data,
)
from cosmos_auto.wm.wds_utils import get_sample

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(scope="module")
def io_helper() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "skills/scenario-to-rds-hq/scripts/rds_hq_io.py"
    spec = importlib.util.spec_from_file_location("rds_hq_conversion_io", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def converted_clip(io_helper: ModuleType, tmp_path: Path) -> dict[str, Any]:
    camera_pose = np.asarray([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]])  # [4,4]
    poses = np.tile(camera_pose, (7, 1, 1))  # [7,4,4]
    intrinsic = np.asarray([500, 500, 320, 240, 640, 480])  # [6]
    object_pose = np.eye(4)  # [4,4]
    object_pose[:3, 3] = [12, 0, 0.8]  # [3]
    objects = [{} for _ in range(7)]
    objects[3] = {
        "actor": {
            "object_type": "Car",
            "object_is_moving": True,
            "object_lwh": [4.5, 1.8, 1.6],
            "object_to_world": object_pose.tolist(),  # [4,4]
        }
    }
    corners = np.asarray(
        [
            [-0.1, -0.3, -0.5],
            [-0.1, 0.3, -0.5],
            [0.1, 0.3, -0.5],
            [0.1, -0.3, -0.5],
            [-0.1, -0.3, 0.5],
            [-0.1, 0.3, 0.5],
            [0.1, 0.3, 0.5],
            [0.1, -0.3, 0.5],
        ]
    )  # [8,3]
    corners = corners * [-1, -1, 1] + [10, 0, 3]  # [8,3], front faces the camera
    layers = {layer: {"labels": []} for layer in io_helper.MAP_GEOMETRY}
    layers["3d_traffic_lights"]["labels"] = [
        {
            "labelData": {
                "shape3d": {
                    "cuboid3d": {"vertices": corners.tolist()},  # [8,3]
                    "attributes": [{"name": "feature_id", "text": "single-signal"}],
                }
            }
        }
    ]
    report = io_helper.write_clip(
        tmp_path,
        "converted_clip",
        {"camera_front": poses},
        {"camera_front": intrinsic},
        objects,
        layers,
        {
            "state_to_legacy_render_state": {"red_solid": "red"},
            "traffic_light_states": {"0": {"feature_id": "single-signal", "state": ["red_solid"] * 7}},
        },
        camera_type="pinhole",
    )
    assert report["map_labels"]["3d_traffic_lights"] == 1
    return {"root": tmp_path, "clip_id": "converted_clip"}


def test_single_converted_signal_reaches_wsm4_geometry(converted_clip: dict[str, Any]) -> None:
    root, clip_id = converted_clip["root"], converted_clip["clip_id"]
    labels, states, palette, state_map = prepare_traffic_light_status_data(
        root / "3d_traffic_lights" / f"{clip_id}.tar",
        root / "3d_traffic_lights_lightnet_status" / f"{clip_id}.tar",
        traffic_light_color_version="v4",
    )
    assert len(labels) == 1
    poses = get_sample(root / "pose" / f"{clip_id}.tar")
    intrinsics = get_sample(root / "pinhole_intrinsic" / f"{clip_id}.tar")
    camera = PinholeCamera.from_numpy(intrinsics["pinhole_intrinsic.camera_front.npy"], device="cpu")
    geometry = create_traffic_light_status_geometry_objects_from_data_v4(
        labels,
        states,
        0,
        poses["000000.pose.camera_front.npy"],
        camera,
        palette,
        legacy_state_map=state_map,
    )
    assert geometry
    red = np.asarray(palette["red"]) / 255.0  # [3]
    assert np.any(np.all(np.isclose(geometry[0].base_color, red), axis=1))


def test_sparse_empty_boundaries_preserve_track_lifetime(io_helper: ModuleType, converted_clip: dict[str, Any]) -> None:
    root, clip_id = converted_clip["root"], converted_clip["clip_id"]
    path = root / "all_object_info" / f"{clip_id}.tar"
    records = io_helper.read_archive(path, clip_id)
    sparse = {f"{frame:06d}.all_object_info.json": records[f"{frame:06d}.all_object_info.json"] for frame in (0, 3, 6)}
    path.unlink()
    io_helper.write_archive(path, clip_id, sparse)
    assert io_helper.validate_clip(root, clip_id, camera_type="pinhole")["object_frames"] == 3
    objects = interpolate_bbox(fix_static_objects(get_sample(path)), list(range(7)))
    assert list(objects["000003.all_object_info.json"]) == ["actor"]
    assert all(objects[f"{frame:06d}.all_object_info.json"] == {} for frame in (0, 1, 2, 4, 5, 6))
