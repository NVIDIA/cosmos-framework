"""Target-format regression checks for the skill's RDS-HQ writer and validator."""

import copy
import importlib.util
import io
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(scope="module")
def helper() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "rds_hq_io.py"
    spec = importlib.util.spec_from_file_location("scenario_to_rds_hq_io", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clip() -> dict[str, Any]:
    poses = np.tile(np.eye(4), (3, 1, 1))  # [3,4,4]
    poses[:, 0, 3] = np.arange(3)  # [3]
    intrinsic = np.asarray([320, 240, 640, 480, 0, 0.002, 0, 0, 0, 0, 1])  # [11]
    corners = [
        [-0.1, -0.2, -0.3],
        [-0.1, 0.2, -0.3],
        [0.1, 0.2, -0.3],
        [0.1, -0.2, -0.3],
        [-0.1, -0.2, 0.3],
        [-0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, -0.2, 0.3],
    ]
    layers = {
        name: {"labels": []}
        for name in (
            "3d_lanelines",
            "3d_road_boundaries",
            "3d_crosswalks",
            "3d_wait_lines",
            "3d_poles",
            "3d_traffic_signs",
            "3d_road_markings",
            "3d_traffic_lights",
        )
    }
    for feature_id in ("signal-a", "signal-b"):
        layers["3d_traffic_lights"]["labels"].append(
            {
                "labelData": {
                    "shape3d": {
                        "attributes": [{"name": "feature_id", "text": feature_id}],
                        "cuboid3d": {"vertices": copy.deepcopy(corners)},
                    }
                },
            }
        )
    return {
        "clip_id": "test_clip",
        "camera_poses": {"camera_front": poses, "camera_side": poses.copy()},  # each [3,4,4]
        "intrinsics": {"camera_front": intrinsic, "camera_side": intrinsic.copy()},  # each [11]
        "objects": [
            {
                "actor": {
                    "object_type": "Car",
                    "object_is_moving": True,
                    "object_lwh": [4.5, 1.8, 1.6],
                    "object_to_world": pose.tolist(),  # [4,4]
                }
            }
            for pose in poses  # pose: [4,4]
        ],
        "layers": layers,
        "traffic_light_states": {
            "state_to_legacy_render_state": {"red_solid": "red", "unknown": "unknown"},
            "traffic_light_states": {
                "0": {"feature_id": "signal-a", "state": ["red_solid"] * 3},
                "1": {"feature_id": "signal-b", "state": ["unknown"] * 3},
            },
        },
    }


def archive_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.glob("*/*.tar"))}


def test_round_trip_is_deterministic(helper: ModuleType, clip: dict[str, Any], tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    report = helper.write_clip(first, **clip)
    assert helper.write_clip(second, **clip) == report
    assert archive_bytes(first) == archive_bytes(second)
    assert report["pose_frames"] == report["object_frames"] == 3
    assert report["signal_state_observations"] == {"red_solid": 3, "unknown": 3}
    assert len(archive_bytes(first)) == 12
    poses = helper.read_archive(first / "pose" / "test_clip.tar", "test_clip")
    for camera, expected in clip["camera_poses"].items():  # expected: [3,4,4]
        actual = np.stack([poses[f"{index:06d}.pose.{camera}.npy"] for index in range(3)])  # [3,4,4]
        np.testing.assert_array_equal(actual, expected)
    objects = helper.read_archive(first / "all_object_info" / "test_clip.tar", "test_clip")
    assert [objects[f"{index:06d}.all_object_info.json"] for index in range(3)] == clip["objects"]


def test_existing_output_is_preserved(helper: ModuleType, clip: dict[str, Any], tmp_path: Path) -> None:
    helper.write_clip(tmp_path, **clip)
    before = archive_bytes(tmp_path)
    clip["camera_poses"]["camera_front"][:, 0, 3] += 20  # [3]
    with pytest.raises(ValueError):
        helper.write_clip(tmp_path, **clip)
    assert archive_bytes(tmp_path) == before


@pytest.mark.parametrize("field", ["pose", "intrinsic", "dimensions", "object_pose", "map_vertices"])
def test_numeric_strings_are_rejected(helper: ModuleType, clip: dict[str, Any], tmp_path: Path, field: str) -> None:
    if field == "pose":
        clip["camera_poses"]["camera_front"] = clip["camera_poses"]["camera_front"].astype(str)  # [3,4,4]
    elif field == "intrinsic":
        clip["intrinsics"]["camera_front"] = clip["intrinsics"]["camera_front"].astype(str)  # [11]
    elif field == "dimensions":
        clip["objects"][0]["actor"]["object_lwh"] = ["4.5", "1.8", "1.6"]
    elif field == "object_pose":
        clip["objects"][0]["actor"]["object_to_world"] = np.eye(4).astype(str).tolist()  # [4,4]
    else:
        cuboid = clip["layers"]["3d_traffic_lights"]["labels"][0]["labelData"]["shape3d"]["cuboid3d"]
        cuboid["vertices"] = np.asarray(cuboid["vertices"]).astype(str).tolist()  # [8,3]
    with pytest.raises(ValueError):
        helper.write_clip(tmp_path, **clip)
    assert not list(tmp_path.rglob("*.tar"))


@pytest.mark.parametrize("camera", ["camera_FRONT", "camera.front", "camera/front"])
def test_unstable_member_identifiers_are_rejected(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, camera: str
) -> None:
    clip["camera_poses"][camera] = clip["camera_poses"].pop("camera_front")  # [3,4,4]
    clip["intrinsics"][camera] = clip["intrinsics"].pop("camera_front")  # [11]
    with pytest.raises(ValueError):
        helper.write_clip(tmp_path, **clip)
    assert not list(tmp_path.rglob("*.tar"))


@pytest.mark.parametrize("case", ["short_objects", "long_objects", "short_camera"])
def test_writer_requires_shared_complete_timelines(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, case: str
) -> None:
    if case == "short_objects":
        clip["objects"].pop()
    elif case == "long_objects":
        clip["objects"].append({})
    else:
        clip["camera_poses"]["camera_side"] = clip["camera_poses"]["camera_side"][:2]  # [2,4,4]
    with pytest.raises(ValueError):
        helper.write_clip(tmp_path, **clip)
    assert not list(tmp_path.rglob("*.tar"))


@pytest.mark.parametrize("case", ["scale", "reflection", "homogeneous_row"])
def test_nonrigid_poses_are_rejected(helper: ModuleType, clip: dict[str, Any], tmp_path: Path, case: str) -> None:
    pose = clip["camera_poses"]["camera_front"][1]  # [4,4]
    if case == "homogeneous_row":
        pose[3, 0] = 1
    else:
        pose[0, 0] = 2 if case == "scale" else -1
    with pytest.raises(ValueError):
        helper.write_clip(tmp_path, **clip)
    assert not list(tmp_path.rglob("*.tar"))


@pytest.mark.parametrize("case", ["order", "timeline", "unmapped_state"])
def test_signal_states_must_match_geometry_and_timeline(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, case: str
) -> None:
    states = clip["traffic_light_states"]["traffic_light_states"]
    if case == "order":
        states["0"], states["1"] = states["1"], states["0"]
    elif case == "timeline":
        states["0"]["state"].pop()
    else:
        states["0"]["state"][1] = "green_solid"
    with pytest.raises(ValueError):
        helper.write_clip(tmp_path, **clip)


def test_signal_id_attributes_reject_source_metadata(helper: ModuleType, clip: dict[str, Any], tmp_path: Path) -> None:
    attribute = clip["layers"]["3d_traffic_lights"]["labels"][0]["labelData"]["shape3d"]["attributes"][0]
    attribute["source"] = "annotated"
    with pytest.raises(ValueError, match="feature_id must contain only name and text"):
        helper.write_clip(tmp_path, **clip)
    assert not list(tmp_path.rglob("*.tar"))


@pytest.mark.parametrize("layer", ["3d_crosswalks", "3d_road_markings"])
@pytest.mark.parametrize(
    "vertices",
    [
        [[0, 0, 0], [3, 4, 0], [0, 3, 0], [2, 0, 0]],  # Crossing edges with nonzero signed area
        [[0, 0, 0], [3, 0, 0], [1, 0, 0], [3, 3, 0], [0, 3, 0]],  # Overlapping edges
        [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 0, 0]],  # Repeated endpoint
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # Zero-area XY projection
    ],
)
def test_invalid_polygon_perimeters_are_rejected_before_publication(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, layer: str, vertices: list[list[int]]
) -> None:
    clip["layers"][layer]["labels"] = [
        {
            "labelData": {
                "shape3d": {
                    "surface": {"vertices": vertices},
                    "attributes": [{"name": "road_marking_class", "text": "ROI_POLYGON_TEXT_PAINT"}],
                }
            }
        }
    ]
    with pytest.raises(ValueError, match="simple, nonzero-area XY perimeter"):
        helper.write_clip(tmp_path, **clip)
    assert not list(tmp_path.rglob("*.tar"))


@pytest.mark.parametrize("reverse", [False, True])
def test_sloping_concave_polygons_preserve_vertices(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, reverse: bool
) -> None:
    vertices = [[0, 0, 2], [3, 0, 2.3], [3, 3, 2.9], [1, 1, 2.3], [0, 3, 2.6]]
    if reverse:
        vertices.reverse()
    clip["layers"]["3d_crosswalks"]["labels"] = [{"labelData": {"shape3d": {"surface": {"vertices": vertices}}}}]
    report = helper.write_clip(tmp_path, **clip, require_planar_polygons=True)
    assert report["nonplanar_polygons"] == {}
    sample = helper.read_archive(tmp_path / "3d_crosswalks" / "test_clip.tar", "test_clip")
    assert sample["crosswalks.json"]["labels"][0]["labelData"]["shape3d"]["surface"]["vertices"] == vertices


@pytest.mark.parametrize("layer", ["3d_crosswalks", "3d_road_markings"])
def test_nonplanar_polygons_are_reported_or_rejected_explicitly(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, layer: str
) -> None:
    vertices = [[0, 0, 0], [3, 0, 0], [3, 3, 1], [0, 3, 0]]
    clip["layers"][layer]["labels"] = [
        {
            "labelData": {
                "shape3d": {
                    "surface": {"vertices": vertices},
                    "attributes": [{"name": "road_marking_class", "text": "ROI_POLYGON_TEXT_PAINT"}],
                }
            }
        }
    ]
    report = helper.write_clip(tmp_path / "preserved", **clip)
    assert report["nonplanar_polygons"][f"{layer}/0"] > 0.1
    sample = helper.read_archive(tmp_path / "preserved" / layer / "test_clip.tar", "test_clip")
    assert sample[f"{layer[3:]}.json"]["labels"][0]["labelData"]["shape3d"]["surface"]["vertices"] == vertices
    with pytest.raises(ValueError, match="non-planar polygon"):
        helper.validate_clip(tmp_path / "preserved", "test_clip", require_planar_polygons=True)
    with pytest.raises(ValueError, match="non-planar polygon"):
        helper.write_clip(tmp_path / "strict", **clip, require_planar_polygons=True)
    assert not list((tmp_path / "strict").rglob("*.tar"))


@pytest.mark.parametrize("case", ["misplaced_corner", "reversed_winding", "shear"])
def test_cuboids_require_consistent_order_and_orthogonal_edges(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, case: str
) -> None:
    cuboid = clip["layers"]["3d_traffic_lights"]["labels"][0]["labelData"]["shape3d"]["cuboid3d"]
    vertices = np.asarray(cuboid["vertices"])  # [8,3]
    if case == "misplaced_corner":
        vertices[6] += [3, 4, 5]  # [3]
    elif case == "reversed_winding":
        vertices[:, 0] *= -1  # [8]
    else:
        vertices[:, 0] += vertices[:, 1]  # [8]
    cuboid["vertices"] = vertices.tolist()  # [8,3]
    with pytest.raises(ValueError):
        helper.write_clip(tmp_path, **clip)


@pytest.mark.parametrize("case", ["duplicate", "missing_frame", "wrong_sample", "directory_prefix"])
def test_validator_rejects_corrupt_archive_members(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, case: str
) -> None:
    helper.write_clip(tmp_path, **clip)
    path = tmp_path / "pose" / "test_clip.tar"
    with tarfile.open(path) as archive:
        members = []
        for member in archive:
            stream = archive.extractfile(member)
            assert stream is not None
            members.append((copy.copy(member), stream.read()))
    if case == "duplicate":
        members.append(members[0])
    elif case == "missing_frame":
        del members[0]
    elif case == "wrong_sample":
        members[0][0].name = members[0][0].name.replace("test_clip", "other_clip", 1)
    else:
        members[0][0].name = "folder/" + members[0][0].name
    with tarfile.open(path, "w") as archive:
        for member, payload in members:
            archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError):
        helper.validate_clip(tmp_path, "test_clip")


def test_sparse_object_archives_remain_supported(helper: ModuleType, clip: dict[str, Any], tmp_path: Path) -> None:
    helper.write_clip(tmp_path, **clip)
    path = tmp_path / "all_object_info" / "test_clip.tar"
    records = helper.read_archive(path, "test_clip")
    del records["000001.all_object_info.json"]
    path.unlink()
    helper.write_archive(path, "test_clip", records)
    assert helper.validate_clip(tmp_path, "test_clip")["object_frames"] == 2


@pytest.mark.parametrize("missing_frames", [[0], [2], [0, 2]])
def test_sparse_object_archives_require_boundary_records(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, missing_frames: list[int]
) -> None:
    helper.write_clip(tmp_path, **clip)
    path = tmp_path / "all_object_info" / "test_clip.tar"
    records = helper.read_archive(path, "test_clip")
    for frame in missing_frames:
        del records[f"{frame:06d}.all_object_info.json"]
    path.unlink()
    helper.write_archive(path, "test_clip", records)
    with pytest.raises(ValueError, match="first and last pose frames"):
        helper.validate_clip(tmp_path, "test_clip")


def test_empty_object_boundary_records_are_valid(helper: ModuleType, clip: dict[str, Any], tmp_path: Path) -> None:
    clip["objects"][0] = {}
    clip["objects"][-1] = {}
    report = helper.write_clip(tmp_path, **clip)
    assert report["object_frames"] == 3
    assert report["object_class_observations"] == {"Car": 1}


def test_publication_collision_preserves_competing_output(
    helper: ModuleType, clip: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_link = helper.os.link
    publications: list[Path] = []
    competing_content = b"another conversion published this archive"

    def collide_on_second_archive(source: Path, destination: Path) -> None:
        publications.append(destination)
        if len(publications) == 2:
            destination.write_bytes(competing_content)
            raise FileExistsError(destination)
        native_link(source, destination)

    monkeypatch.setattr(helper.os, "link", collide_on_second_archive)
    with pytest.raises(FileExistsError):
        helper.write_clip(tmp_path, **clip)
    assert len(publications) == 2
    assert not publications[0].exists()
    assert publications[1].read_bytes() == competing_content
    assert list(tmp_path.rglob("*.tar")) == [publications[1]]
