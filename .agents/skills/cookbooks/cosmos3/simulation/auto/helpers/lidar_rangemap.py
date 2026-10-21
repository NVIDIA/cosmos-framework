# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert a sparse HD-map LiDAR rangemap archive to the vLLM-Omni numeric LiDAR control.

The archive stores one ``<clip>.<frame>.lidar_{row,col,range}.npz`` triple per sweep on the
128 x 1800 or 128 x 3600 Pandar128 grid, recorded in ``<clip>.lidar_metadata.json``.
Unversioned archives retain the legacy 3600-column contract.
vLLM-Omni accepts only dense FP32 ``frames[3,T,128,1800]``
safetensors: metric range, unit intensity, and validity. The conversion matches Cosmos
Framework's native reader for a control without return intensity: rasterize each sweep,
keep the nearest nonzero return when pooling 3600 to 1800 columns, and give occupied rays unit
intensity and validity. Ranges stay physical; the server owns range filtering, circular
padding, and normalization.

Usage, from the ``auto`` folder::

    python -m helpers.lidar_rangemap <sample_data>/daytime_scene/lidar/control.tar \
        outputs/prepared/lidar_control.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

NATIVE_SHAPE = (128, 3600)
SEMANTIC_WIDTH = 1800
_MEMBERS = {"row": ".lidar_row.npz", "col": ".lidar_col.npz", "range": ".lidar_range.npz"}


def _read_npz(archive: tarfile.TarFile, name: str) -> np.ndarray:
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Cannot read archive member {name!r}.")
    with np.load(io.BytesIO(member.read()), allow_pickle=False) as payload:
        if payload.files != ["arr_0"]:
            raise ValueError(f"Expected one array 'arr_0' in {name!r}, got {payload.files}.")
        return np.asarray(payload["arr_0"])


def index_sweeps(archive: tarfile.TarFile) -> tuple[str, dict[int, dict[str, str]]]:
    """Group sparse members by source frame ID; require one clip key and complete triples."""
    clips, sweeps = set(), {}
    for name in archive.getnames():
        if name.endswith(".lidar_metadata.json"):
            continue
        kind = next((key for key, suffix in _MEMBERS.items() if name.endswith(suffix)), None)
        if kind is None:
            raise ValueError(f"Unexpected member {name!r}; a control archive carries only row, col, and range.")
        clip, frame = name.removesuffix(_MEMBERS[kind]).rsplit(".", 1)
        clips.add(clip)
        members = sweeps.setdefault(int(frame), {})
        if kind in members:
            raise ValueError(f"Duplicate {kind} member for frame {frame}.")
        members[kind] = name
    if len(clips) != 1:
        raise ValueError(f"Expected one clip key, got {sorted(clips)}.")
    for frame, members in sweeps.items():
        if members.keys() != _MEMBERS.keys():
            raise ValueError(f"Frame {frame} is missing {sorted(_MEMBERS.keys() - members.keys())}.")
    return clips.pop(), dict(sorted(sweeps.items()))


def _tar_width(archive: tarfile.TarFile, clip_keys: set[str]) -> int:
    """Match Framework's v2 metadata rules without requiring its installation."""
    suffix = ".lidar_metadata.json"
    members = [member for member in archive.getmembers() if member.name.endswith(suffix)]
    if not members:
        return NATIVE_SHAPE[1]
    if len(members) != 1:
        raise ValueError("Expected one LiDAR metadata file per archive")
    member = members[0]
    if not member.isfile() or clip_keys != {member.name.removesuffix(suffix)}:
        raise ValueError("LiDAR metadata does not match the archive's clip")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError("Cannot read LiDAR metadata")
    with stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("LiDAR metadata must be a JSON object")
    width = metadata.get("n_columns")
    if (
        metadata.get("format_version") != 2
        or metadata.get("n_rows") != NATIVE_SHAPE[0]
        or type(width) is not int
        or width not in (1800, 3600)
    ):
        raise ValueError(f"Unsupported LiDAR metadata: {metadata}")
    return width


def rasterize(rows: np.ndarray, cols: np.ndarray, ranges: np.ndarray, *, width: int = 3600) -> np.ndarray:
    """Scatter sparse returns into an FP32 range map at the source width."""
    if not rows.shape == cols.shape == ranges.shape or rows.ndim != 1:
        raise ValueError("Sparse row, col, and range arrays must be one-dimensional and equally long.")
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= NATIVE_SHAPE[0] or cols.min() < 0 or cols.max() >= width):
        raise ValueError(f"Sparse coordinates fall outside the source 128 x {width} grid.")
    if len(set(zip(rows.tolist(), cols.tolist()))) != rows.size:
        raise ValueError("Sparse coordinates repeat a pixel.")
    ranges = ranges.astype(np.float32)
    if not np.isfinite(ranges).all() or (ranges <= 0).any():
        raise ValueError("Sparse ranges must be finite and positive.")
    dense = np.zeros((NATIVE_SHAPE[0], width), dtype=np.float32)
    dense[rows, cols] = ranges
    return dense


def pool_nearest_return(dense: np.ndarray, width: int = SEMANTIC_WIDTH) -> np.ndarray:
    """Keep the smallest nonzero range of each group of adjacent azimuth columns."""
    height, native_width = dense.shape
    if native_width % width:
        raise ValueError(f"Native width {native_width} is not divisible by {width}.")
    groups = dense.reshape(height, width, native_width // width)
    sentinel = np.finfo(np.float32).max
    nearest = np.where(groups == 0, sentinel, groups).min(axis=2)
    return np.where(nearest == sentinel, 0, nearest).astype(np.float32)


def convert_control(tar_path: Path, output_path: Path, *, camera_fps: float = 30.0, lidar_fps: float = 10.0) -> dict:
    """Write ``frames[3,T,128,1800]`` for every sweep, in source-frame order, and return a summary."""
    from safetensors.numpy import save_file

    tar_path, output_path = Path(tar_path), Path(output_path)
    stride = camera_fps / lidar_fps
    with tarfile.open(tar_path) as archive:
        clip, sweeps = index_sweeps(archive)
        width = _tar_width(archive, {clip})
        frame_ids = list(sweeps)
        if frame_ids != [round(index * stride) for index in range(len(frame_ids))]:
            raise ValueError(f"Sweeps must start at camera frame 0 and advance by {stride:g} frames.")
        frames = np.zeros((3, len(frame_ids), *NATIVE_SHAPE[:1], SEMANTIC_WIDTH), dtype=np.float32)
        for index, members in enumerate(sweeps.values()):
            ranges = pool_nearest_return(rasterize(*(_read_npz(archive, members[key]) for key in _MEMBERS), width=width))
            valid = (ranges > 0).astype(np.float32)
            frames[:, index] = ranges, valid, valid
    source = {
        "archive": tar_path.name,
        "archive_sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
        "clip": clip,
        "camera_frame_ids": frame_ids,
        "conversion": f"nearest-return pooling {width}->1800 columns; intensity = validity = range > 0",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    save_file({"frames": frames}, str(temporary), metadata={"source": json.dumps(source)})
    temporary.replace(output_path)
    valid = frames[2] > 0
    return {
        "shape": list(frames.shape),
        "clip": clip,
        "frame_ids": frame_ids,
        "range_m": [float(frames[0][valid].min()), float(frames[0][valid].max())],
        "valid_fraction": float(valid.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("archive", type=Path, help="Sparse HD-map rangemap archive, such as lidar/control.tar")
    parser.add_argument("output", type=Path, help="Output .safetensors path")
    args = parser.parse_args()
    print(json.dumps(convert_control(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
