"""Write and structurally validate the RDS-HQ contract used by WSM4.

This helper accepts target-format data. Source interpretation, semantic mapping,
and inference belong in the conversion adapter and its report.
"""

import argparse
import io
import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

MAP_GEOMETRY: dict[str, str] = {
    "3d_lanelines": "polyline3d",
    "3d_road_boundaries": "polyline3d",
    "3d_crosswalks": "surface",
    "3d_wait_lines": "polyline3d",
    "3d_poles": "polyline3d",
    "3d_traffic_signs": "cuboid3d",
    "3d_road_markings": "surface",
    "3d_traffic_lights": "cuboid3d",
}
OBJECT_CLASSES: set[str] = {"Car", "Truck", "Pedestrian", "Cyclist", "Others"}
BASIC_STATES: set[str] = {"red", "yellow", "green", "unknown"}
POLYGON_PLANAR_TOLERANCE_M: float = 1e-3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_clip_id(clip_id: str) -> None:
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", clip_id)), "Clip ID must use letters, digits, '_' or '-'")


def write_archive(path: Path, clip_id: str, members: dict[str, Any]) -> None:
    """Write one sample without overwriting an existing archive."""
    check_clip_id(clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream, tarfile.open(fileobj=stream, mode="w") as archive:
        for key, value in sorted(members.items()):
            require("/" not in key and "\\" not in key, f"Invalid member key: {key}")
            if key.endswith(".npy"):
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                payload = buffer.getvalue()
            elif key.endswith(".json"):
                payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
            else:
                raise ValueError(f"Unsupported member: {key}")
            member = tarfile.TarInfo(f"{clip_id}.{key}")
            member.size = len(payload)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(payload))


def write_clip(
    root: Path,
    clip_id: str,
    camera_poses: dict[str, np.ndarray],  # each [T,4,4]
    intrinsics: dict[str, np.ndarray],  # each [6], [11], or [14]
    objects: list[dict[str, Any]],
    layers: dict[str, dict[str, Any]],
    traffic_light_states: dict[str, Any],
    camera_type: str = "ftheta",
    *,
    require_planar_polygons: bool = False,
) -> dict[str, Any]:
    """Serialize an already normalized clip and return its structural report."""
    check_clip_id(clip_id)
    require(camera_type in {"pinhole", "ftheta"}, "Unsupported camera model")
    require(set(layers) == set(MAP_GEOMETRY), "Supply all eight map layers, including empty layers")
    require(bool(camera_poses) and set(camera_poses) == set(intrinsics), "Camera names must match")
    require(
        all(re.fullmatch(r"[a-z0-9_-]+", camera) for camera in camera_poses),
        "Camera names must be lowercase identifiers",
    )
    require(
        len(objects) == len(next(iter(camera_poses.values()))), "write_clip requires one object record per pose frame"
    )
    sample_members: dict[str, dict[str, Any]] = {
        "pose": {
            f"{index:06d}.pose.{camera}.npy": pose
            for camera, poses in camera_poses.items()
            for index, pose in enumerate(poses)  # pose: [4,4]
        },
        f"{camera_type}_intrinsic": {
            f"{camera_type}_intrinsic.{camera}.npy": intrinsic
            for camera, intrinsic in intrinsics.items()  # intrinsic: [6], [11], or [14]
        },
        "all_object_info": {f"{index:06d}.all_object_info.json": frame for index, frame in enumerate(objects)},
        "3d_traffic_lights_lightnet_status": {"aggregated_states.json": traffic_light_states},
    }
    sample_members.update({layer: {f"{layer[3:]}.json": data} for layer, data in layers.items()})
    destinations = {layer: root / layer / f"{clip_id}.tar" for layer in sample_members}
    require(not any(path.exists() for path in destinations.values()), "Output clip already exists; use a fresh root")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rds-hq-", dir=root) as temporary:
        staged_root = Path(temporary)
        for layer, members in sample_members.items():
            write_archive(staged_root / layer / f"{clip_id}.tar", clip_id, members)
        report = validate_clip(staged_root, clip_id, camera_type, require_planar_polygons=require_planar_polygons)
        installed: list[Path] = []
        try:
            for layer, destination in destinations.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Linking on the same filesystem publishes each complete archive
                # without overwriting a clip created by a concurrent conversion.
                os.link(staged_root / layer / f"{clip_id}.tar", destination)
                installed.append(destination)
        except Exception:
            for destination in installed:
                destination.unlink()
            raise
    return report


def read_archive(path: Path, clip_id: str) -> dict[str, Any]:
    """Decode in memory without extracting paths from the archive."""
    data: dict[str, Any] = {}
    prefix = f"{clip_id}."
    with tarfile.open(path) as archive:
        for member in archive:
            require(member.isfile(), f"{path}: unexpected non-file member")
            require(member.name.startswith(prefix) and "/" not in member.name, f"{path}: wrong sample key")
            key = member.name[len(prefix) :]
            require(key not in data, f"{path}: duplicate member {key}")
            stream = archive.extractfile(member)
            require(stream is not None, f"{path}: unreadable member {key}")
            payload = stream.read()
            if key.endswith(".npy"):
                data[key] = np.load(io.BytesIO(payload), allow_pickle=False)  # array shape checked by the caller
            elif key.endswith(".json"):
                data[key] = json.loads(payload)
            else:
                raise ValueError(f"{path}: unsupported member {key}")
    require(bool(data), f"{path}: an empty archive is not an empty layer")
    return data


def numeric_array(value: Any, context: str) -> np.ndarray:  # Returns a real numeric array; shape checked by caller
    array = np.asarray(value)  # [...]
    require(array.dtype.kind in "iuf", f"{context}: use real numbers, not strings, booleans, or complex values")
    require(bool(np.isfinite(array).all()), f"{context}: non-finite values")
    return array  # [...]


def check_transform(value: Any, context: str) -> None:
    matrix = numeric_array(value, context)  # [4,4]
    require(matrix.shape == (4, 4) and bool(np.isfinite(matrix).all()), f"{context}: invalid transform")
    rotation = matrix[:3, :3]  # [3,3]
    require(bool(np.allclose(matrix[3], [0, 0, 0, 1])), f"{context}: invalid homogeneous row")
    require(bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)), f"{context}: non-rigid rotation")
    require(bool(np.isclose(np.linalg.det(rotation), 1, atol=1e-3)), f"{context}: improper rotation")


def attributes(shape: dict[str, Any]) -> dict[str, Any]:
    records = shape.get("attributes", [])
    require(isinstance(records, list), "Shape attributes must be a list")
    require(len({record["name"] for record in records}) == len(records), "Duplicate attribute names")
    return {record["name"]: record for record in records}


def validate_clip(
    root: Path, clip_id: str, camera_type: str = "ftheta", *, require_planar_polygons: bool = False
) -> dict[str, Any]:
    """Check serialized structure; semantic accuracy still needs source/render checks."""
    check_clip_id(clip_id)
    require(camera_type in {"pinhole", "ftheta"}, "Unsupported camera model")
    layers = ["pose", f"{camera_type}_intrinsic", "all_object_info", *MAP_GEOMETRY, "3d_traffic_lights_lightnet_status"]
    samples = {layer: read_archive(root / layer / f"{clip_id}.tar", clip_id) for layer in layers}
    cameras: dict[str, list[int]] = {}
    for key, pose in samples["pose"].items():  # pose: [4,4]
        match = re.fullmatch(r"(\d{6})\.pose\.([a-z0-9_-]+)\.npy", key)
        require(match is not None, f"Invalid pose key: {key}")
        cameras.setdefault(match[2], []).append(int(match[1]))
        check_transform(pose, key)
    frame_count = len(next(iter(cameras.values())))
    for camera, indices in cameras.items():
        require(
            sorted(indices) == list(range(frame_count)), f"{camera}: pose frames must be contiguous and synchronized"
        )
    expected_intrinsics = {f"{camera_type}_intrinsic.{camera}.npy" for camera in cameras}
    require(set(samples[f"{camera_type}_intrinsic"]) == expected_intrinsics, "Pose/intrinsic camera mismatch")
    for key, value in samples[f"{camera_type}_intrinsic"].items():
        array = numeric_array(value, key)  # [6], [11], or [14]
        lengths = {6} if camera_type == "pinhole" else {11, 14}
        require(
            array.ndim == 1 and len(array) in lengths and bool(np.isfinite(array).all()), f"{key}: invalid intrinsic"
        )
        width, height = array[4:6] if camera_type == "pinhole" else array[2:4]
        require(
            width > 0 and height > 0 and width == int(width) and height == int(height), f"{key}: invalid image size"
        )
        if camera_type == "pinhole":
            require(bool((array[:2] > 0).all()), f"{key}: focal lengths must be positive")
        else:
            require(array[10] in (0, 1), f"{key}: polynomial direction must be 0 or 1")
            require(bool(np.any(array[5:10] != 0)), f"{key}: constant projection polynomial")
            if len(array) == 14:
                require(abs(array[11] - array[12] * array[13]) > 1e-12, f"{key}: singular linear correction")

    object_classes: Counter[str] = Counter()
    track_ids: set[str] = set()
    object_indices: list[int] = []
    for key, frame in samples["all_object_info"].items():
        match = re.fullmatch(r"(\d{6})\.all_object_info\.json", key)
        require(match is not None, f"Invalid object key: {key}")
        object_indices.append(int(match[1]))
        require(isinstance(frame, dict), f"{key}: object frame must be a dictionary")
        for track_id, record in frame.items():
            context = f"{key}/{track_id}"
            require(isinstance(track_id, str) and bool(track_id), f"{context}: invalid track ID")
            require(record["object_type"] in OBJECT_CLASSES, f"{context}: use a canonical object class")
            require(isinstance(record["object_is_moving"], bool), f"{context}: moving flag must be boolean")
            dimensions = numeric_array(record["object_lwh"], context)  # [3]
            require(
                dimensions.shape == (3,) and bool(np.isfinite(dimensions).all()) and bool((dimensions > 0).all()),
                f"{context}: invalid box dimensions",
            )
            check_transform(record["object_to_world"], context)
            object_classes[record["object_type"]] += 1
            track_ids.add(track_id)
    require(
        bool(object_indices) and min(object_indices) >= 0 and max(object_indices) < frame_count,
        "Object frame indices lie outside the pose timeline",
    )
    require(
        0 in object_indices and frame_count - 1 in object_indices,
        "Object records must include the first and last pose frames; use {} when no objects are present",
    )

    map_counts: dict[str, int] = {}
    nonplanar_polygons: dict[str, float] = {}
    signal_ids: list[str] = []
    for layer, geometry in MAP_GEOMETRY.items():
        key = f"{layer[3:]}.json"
        require(set(samples[layer]) == {key}, f"{layer}: expected one clip-static {key}")
        labels = samples[layer][key]["labels"]
        require(isinstance(labels, list), f"{layer}: labels must be a list")
        map_counts[layer] = len(labels)
        for index, label in enumerate(labels):
            context = f"{layer}/{index}"
            shape = label["labelData"]["shape3d"]
            vertices = numeric_array(shape[geometry]["vertices"], context)  # [N,3]
            minimum = {"polyline3d": 2, "surface": 3, "cuboid3d": 8}[geometry]
            require(
                vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= minimum,
                f"{context}: invalid vertices",
            )
            require(bool(np.isfinite(vertices).all()), f"{context}: non-finite geometry")
            require(len(np.unique(vertices, axis=0)) >= minimum, f"{context}: repeated/degenerate vertices")
            if geometry == "surface":
                polygon = Polygon(vertices[:, :2])  # Perimeter projected to [N,2], as in the renderer
                require(
                    polygon.is_valid and polygon.area > 1e-8 and not np.allclose(vertices[0], vertices[-1]),
                    f"{context}: polygon must have a simple, nonzero-area XY perimeter without a repeated endpoint",
                )
                centered = vertices - vertices.mean(axis=0)  # [N,3]
                _, _, axes = np.linalg.svd(centered, full_matrices=False)  # [N,3], [3], [3,3]
                deviations = np.abs(centered @ axes[-1])  # [N], distance to the best-fit plane in metres
                deviation = float(deviations.max())
                if deviation > POLYGON_PLANAR_TOLERANCE_M:
                    require(
                        not require_planar_polygons,
                        f"{context}: non-planar polygon ({deviation:.6g} m from its best-fit plane)",
                    )
                    nonplanar_polygons[context] = deviation
            if geometry == "cuboid3d":
                require(len(vertices) == 8, f"{context}: cuboid requires eight vertices")
                edges = np.stack(
                    [vertices[3] - vertices[0], vertices[1] - vertices[0], vertices[4] - vertices[0]]
                )  # [3,3]
                require(np.linalg.det(edges) > 1e-10, f"{context}: cuboid has zero volume or reversed winding")
                corners = np.array(
                    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]
                )  # [8,3]
                expected = vertices[0] + corners @ edges  # [8,3]
                axes = edges / np.linalg.norm(edges, axis=1)[:, None]  # [3,3]
                require(
                    bool(np.allclose(vertices, expected, atol=1e-4, rtol=1e-6)),
                    f"{context}: invalid cuboid corner order",
                )
                require(
                    bool(np.allclose(axes @ axes.T, np.eye(3), atol=1e-3)),
                    f"{context}: cuboid edges are not perpendicular",
                )
            attrs = attributes(shape)
            if layer == "3d_lanelines":
                for name in ("colors", "styles"):
                    values = attrs[name]["enumsList"]["enumsList"]
                    require(
                        isinstance(values, list) and bool(values) and isinstance(values[0], str),
                        f"{context}: missing {name}",
                    )
            if layer == "3d_wait_lines":
                require(isinstance(attrs["is_implicit"]["flag"], bool), f"{context}: invalid implicit flag")
            for target_layer, name in (
                ("3d_road_boundaries", "boundary_type"),
                ("3d_traffic_signs", "sign_class"),
                ("3d_road_markings", "road_marking_class"),
                ("3d_traffic_lights", "feature_id"),
            ):
                if layer == target_layer:
                    require(isinstance(attrs[name]["text"], str), f"{context}: {name} must be a string")
            if layer == "3d_traffic_lights":
                require(
                    set(attrs["feature_id"]) == {"name", "text"},
                    f"{context}: feature_id must contain only name and text, as required by the renderer",
                )
                signal_ids.append(attrs["feature_id"]["text"])
    require(len(set(signal_ids)) == len(signal_ids), "Traffic-light feature IDs must be unique")
    aggregate = samples["3d_traffic_lights_lightnet_status"]["aggregated_states.json"]
    signals = aggregate["traffic_light_states"]
    require(
        set(signals) == {str(index) for index in range(len(signal_ids))}, "Signal indices must match geometry order"
    )
    mapping = aggregate["state_to_legacy_render_state"]
    require(
        isinstance(mapping, dict) and all(value in BASIC_STATES for value in mapping.values()),
        "Invalid basic-state mapping",
    )
    states: Counter[str] = Counter()
    for index, feature_id in enumerate(signal_ids):
        record = signals[str(index)]
        require(record["feature_id"] == feature_id, f"Signal {index}: feature ID mismatch")
        require(
            isinstance(record["state"], list) and len(record["state"]) == frame_count, f"Signal {index}: wrong timeline"
        )
        for state in record["state"]:
            require(isinstance(state, str) and state in mapping, f"Signal {index}: unmapped state {state}")
            states[state] += 1
    return {
        "clip_id": clip_id,
        "cameras": sorted(cameras),
        "pose_frames": frame_count,
        "object_frames": len(object_indices),
        "tracks": len(track_ids),
        "object_class_observations": dict(object_classes),
        "map_labels": map_counts,
        "nonplanar_polygons": nonplanar_polygons,
        "polygon_planarity_tolerance_m": POLYGON_PLANAR_TOLERANCE_M,
        "require_planar_polygons": require_planar_polygons,
        "signal_state_observations": dict(states),
        "archives": len(layers),
        "validation": "structural; source semantics and rendered alignment require separate checks",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("clip_id")
    parser.add_argument("--camera-type", choices=("pinhole", "ftheta"), default="ftheta")
    parser.add_argument(
        "--require-planar-polygons",
        action="store_true",
        help="Reject polygons more than 1 mm from their best-fit plane instead of reporting their deviations",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = json.dumps(
        validate_clip(args.root, args.clip_id, args.camera_type, require_planar_polygons=args.require_planar_polygons),
        indent=2,
    )
    if args.report:
        args.report.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
