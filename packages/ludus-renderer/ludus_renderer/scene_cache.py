# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""3-level scene cache: L1 (CUDA) / L2 (CPU RAM) / L3 (Disk/LMDB).

Provides non-blocking scene loading for training loops via:
- L3: LMDB key-value store on disk (fast mmap reads, concurrent-reader safe)
     Falls back to per-file .pt cache if LMDB is unavailable.
- L2: In-memory LRU cache of ClipgtGpuScene on CPU (byte-based capacity)
- L1: GPU-resident CUDA tensor LRU cache of ClipgtGpuScene (byte-based capacity)

GL buffer management is handled by the caller (e.g. LiveRenderingIterator)
which uploads scenes directly via CUDA-GL interop after loading to L1.

Cache directory can be set via:
1. ``cache_dir`` constructor arg (highest priority)
2. ``LUDUS_CACHE_DIR`` environment variable
3. ``~/.cache/ludus`` (default fallback)

Cache data is stored under a version subdirectory (e.g. ``v1/``) so that
different code versions coexist without conflict. Bump ``CACHE_VERSION``
when the serialization format changes.

Usage:
    db = SceneDatabase(
        scene_paths=paths,           # all known scene tar paths
        cache_dir="/fast/cache",     # L3 disk cache directory
        max_cpu_bytes=16 * 1024**3,  # L2 capacity in bytes
        max_gpu_bytes=4 * 1024**3,   # L1 capacity in bytes
    )
    # During training:
    scene = db._ensure_l1(key)             # L3→L2→L1 transparent promotion
    db.prefetch(next_step_keys)            # async warm L2 for next step
    db.prefetch_l1(next_step_keys)         # async L2→L1 on CUDA prefetch stream
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_log = logging.getLogger(__name__)

try:
    import lmdb as _lmdb
except ImportError:
    _lmdb = None

from ._ops.primitives import (
    CubePool,
    FThetaCamera,
    TimestampedPolygonPool,
    TimestampedPolylinePool,
    TimestampedScene,
)
from .clipgt import ClipgtGpuScene, EgoTrackData, load_av2_scene

CACHE_VERSION = 2

LUDUS_CACHE_DIR_ENV = "LUDUS_CACHE_DIR"
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/ludus")


def compute_format_hash() -> str:
    """Compute a hash that changes when the packed buffer layout changes.

    Combines C++ struct sizes (from pybind) with Python packing constants.
    Any struct or packing change auto-invalidates cache.
    """
    parts = [
        f"cache_version={CACHE_VERSION}",
        "vertex_stride=4",
        "triangle_stride=4",
        "aabb_per_element=6",
        "cube_float_layout=trans3_quat4_scale3_color6",
    ]
    try:
        from ._ops._ludus_gl import get_struct_sizes

        sizes = get_struct_sizes()
        for name in sorted(sizes.keys()):
            parts.append(f"sizeof_{name}={sizes[name]}")
    except ImportError:
        parts.append("cpp_sizes=unavailable")
    sig = ",".join(parts)
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


_FORMAT_HASH: Optional[str] = None


def get_format_hash() -> str:
    """Lazy-initialized format hash (avoids importing C++ at module load)."""
    global _FORMAT_HASH
    if _FORMAT_HASH is None:
        _FORMAT_HASH = compute_format_hash()
    return _FORMAT_HASH


def _resolve_cache_dir(cache_dir: Optional[str] = None) -> Path:
    """Resolve the L3 disk cache root directory.

    Priority: explicit arg > env var > default (~/.cache/ludus).
    Uses ``v{CACHE_VERSION}_{FORMAT_HASH}`` subdir so that any C++
    struct layout or Python packing change auto-invalidates the cache.
    """
    if cache_dir is not None:
        base = Path(cache_dir)
    else:
        base = Path(os.environ.get(LUDUS_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR))
    fh = get_format_hash()
    return base / f"v{CACHE_VERSION}_{fh}"


# ---------------------------------------------------------------------------
# L3: Disk cache — serialize / deserialize ClipgtGpuScene
# ---------------------------------------------------------------------------


def scene_key(scene_path: str) -> str:
    """Deterministic cache key from a scene path."""
    return hashlib.sha256(scene_path.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str) -> Path:
    """Two-level directory structure to avoid huge flat dirs.

    ``cache_dir`` should already include the version subdirectory.
    """
    return cache_dir / key[:2] / f"{key}.pt"


def serialize_scene(scene: ClipgtGpuScene) -> dict:
    """Convert a ClipgtGpuScene to a dict suitable for torch.save.

    All tensors are moved to CPU. Non-tensor fields are stored as-is.
    """

    def _polyline_pool_to_dict(p: TimestampedPolylinePool) -> dict:
        return {
            "timestamps_us": _ensure_cpu(p.timestamps_us),
            "timestamped_varrays_prefix_sum": _ensure_cpu(p.timestamped_varrays_prefix_sum),
            "varrays_prefix_sum": _ensure_cpu(p.varrays_prefix_sum),
            "vertices": _ensure_cpu(p.vertices),
            "prim_type_id": p.prim_type_id,
        }

    def _polygon_pool_to_dict(p: TimestampedPolygonPool) -> dict:
        return {
            "timestamps_us": _ensure_cpu(p.timestamps_us),
            "timestamped_varrays_prefix_sum": _ensure_cpu(p.timestamped_varrays_prefix_sum),
            "varrays_prefix_sum": _ensure_cpu(p.varrays_prefix_sum),
            "triangle_prefix_sum": _ensure_cpu(p.triangle_prefix_sum),
            "vertices": _ensure_cpu(p.vertices),
            "triangles": _ensure_cpu(p.triangles),
            "prim_type_id": p.prim_type_id,
        }

    def _cube_pool_to_dict(p: CubePool) -> dict:
        return {
            "timestamps_us": _ensure_cpu(p.timestamps_us),
            "cube_ts_prefix_sum": _ensure_cpu(p.cube_ts_prefix_sum),
            "track_timestamps_us": _ensure_cpu(p.track_timestamps_us),
            "translations": _ensure_cpu(p.translations),
            "quaternions": _ensure_cpu(p.quaternions),
            "scales": _ensure_cpu(p.scales),
            "colors": _ensure_cpu(p.colors),
            "prim_type_id": p.prim_type_id,
            "render_flags": p.render_flags,
        }

    def _camera_to_dict(c: FThetaCamera) -> dict:
        d = {
            "principal_point": _ensure_cpu(c.principal_point),
            "image_size": _ensure_cpu(c.image_size),
            "fw_poly": _ensure_cpu(c.fw_poly),
            "max_ray_angle": c.max_ray_angle,
            "depth_max": c.depth_max,
        }
        if c.linear_distortion is not None:
            d["linear_distortion"] = _ensure_cpu(c.linear_distortion)
        return d

    ts = scene.timestamped_scene
    packed = pack_scene(scene)
    return {
        "version": CACHE_VERSION,
        "format_hash": get_format_hash(),
        "polyline_pools": [_polyline_pool_to_dict(p) for p in ts.polyline_pools],
        "polygon_pools": [_polygon_pool_to_dict(p) for p in ts.polygon_pools],
        "cube_pools": [_cube_pool_to_dict(p) for p in (ts.cube_pools or [])],
        "cameras": [_camera_to_dict(c) for c in scene.cameras],
        "camera_name_to_id": scene.camera_name_to_id,
        "sensor_to_rig": {k: _ensure_cpu(v) if isinstance(v, Tensor) else v for k, v in scene.sensor_to_rig.items()},
        "ego_timestamps": _ensure_cpu(scene.ego_track.timestamps),
        "ego_poses_tquat": _ensure_cpu(scene.ego_track.poses_tquat),
        "packed_timestamps": packed.timestamps,
        "packed_int32": packed.int32_data,
        "packed_vertices": packed.vertices,
        "packed_triangles": packed.triangles,
        "packed_floats": packed.float_data,
        "packed_polyline_meta": packed.polyline_meta,
        "packed_polygon_meta": packed.polygon_meta,
        "packed_cube_meta": packed.cube_meta,
    }


def deserialize_scene(data: dict, device: torch.device = torch.device("cpu")) -> ClipgtGpuScene:
    """Reconstruct a ClipgtGpuScene from a serialized dict."""

    def _to(t, dev=device):
        return t.to(dev) if isinstance(t, Tensor) else t

    polyline_pools = [
        TimestampedPolylinePool(
            timestamps_us=_to(p["timestamps_us"]),
            timestamped_varrays_prefix_sum=_to(p["timestamped_varrays_prefix_sum"]),
            varrays_prefix_sum=_to(p["varrays_prefix_sum"]),
            vertices=_to(p["vertices"]),
            prim_type_id=p["prim_type_id"],
        )
        for p in data["polyline_pools"]
    ]

    polygon_pools = [
        TimestampedPolygonPool(
            timestamps_us=_to(p["timestamps_us"]),
            timestamped_varrays_prefix_sum=_to(p["timestamped_varrays_prefix_sum"]),
            varrays_prefix_sum=_to(p["varrays_prefix_sum"]),
            triangle_prefix_sum=_to(p["triangle_prefix_sum"]),
            vertices=_to(p["vertices"]),
            triangles=_to(p["triangles"]),
            prim_type_id=p["prim_type_id"],
        )
        for p in data["polygon_pools"]
    ]

    cube_pools = [
        CubePool(
            timestamps_us=_to(p["timestamps_us"]),
            cube_ts_prefix_sum=_to(p["cube_ts_prefix_sum"]),
            track_timestamps_us=_to(p["track_timestamps_us"]),
            translations=_to(p["translations"]),
            quaternions=_to(p["quaternions"]),
            scales=_to(p["scales"]),
            colors=_to(p["colors"]),
            prim_type_id=p["prim_type_id"],
            render_flags=p["render_flags"],
        )
        for p in data["cube_pools"]
    ]

    cameras = []
    for c in data["cameras"]:
        cam = FThetaCamera(
            principal_point=_to(c["principal_point"]),
            image_size=_to(c["image_size"]),
            fw_poly=_to(c["fw_poly"]),
            max_ray_angle=c["max_ray_angle"],
            linear_distortion=_to(c.get("linear_distortion")),
            depth_max=c.get("depth_max", 100.0),
        )
        cameras.append(cam)

    sensor_to_rig = {k: _to(v) for k, v in data["sensor_to_rig"].items()}

    packed = None
    if "packed_timestamps" in data:
        packed = PackedSceneBuffers(
            timestamps=_to(data["packed_timestamps"]),
            int32_data=_to(data["packed_int32"]),
            vertices=_to(data["packed_vertices"]),
            triangles=_to(data["packed_triangles"]),
            float_data=_to(data["packed_floats"]),
            polyline_meta=data["packed_polyline_meta"],
            polygon_meta=data["packed_polygon_meta"],
            cube_meta=data["packed_cube_meta"],
        )

    return ClipgtGpuScene(
        timestamped_scene=TimestampedScene(
            polyline_pools=polyline_pools,
            polygon_pools=polygon_pools,
            cube_pools=cube_pools,
        ),
        cameras=cameras,
        camera_name_to_id=data["camera_name_to_id"],
        sensor_to_rig=sensor_to_rig,
        ego_track=EgoTrackData(
            timestamps=_to(data["ego_timestamps"]),
            poses_tquat=_to(data["ego_poses_tquat"]),
        ),
        device=device,
        packed=packed,
    )


def scene_to_device(scene: ClipgtGpuScene, device: torch.device) -> ClipgtGpuScene:
    """Move all tensors in a ClipgtGpuScene to *device* without serialize/deserialize.

    Returns a new ClipgtGpuScene; original tensors are not freed (PyTorch
    ref-counting handles that).
    """
    if scene.device == device:
        return scene

    def _mv(t):
        return t.to(device, non_blocking=True) if isinstance(t, Tensor) else t

    polyline_pools = [
        TimestampedPolylinePool(
            timestamps_us=_mv(p.timestamps_us),
            timestamped_varrays_prefix_sum=_mv(p.timestamped_varrays_prefix_sum),
            varrays_prefix_sum=_mv(p.varrays_prefix_sum),
            vertices=_mv(p.vertices),
            prim_type_id=p.prim_type_id,
        )
        for p in scene.timestamped_scene.polyline_pools
    ]

    polygon_pools = [
        TimestampedPolygonPool(
            timestamps_us=_mv(p.timestamps_us),
            timestamped_varrays_prefix_sum=_mv(p.timestamped_varrays_prefix_sum),
            varrays_prefix_sum=_mv(p.varrays_prefix_sum),
            triangle_prefix_sum=_mv(p.triangle_prefix_sum),
            vertices=_mv(p.vertices),
            triangles=_mv(p.triangles),
            prim_type_id=p.prim_type_id,
        )
        for p in scene.timestamped_scene.polygon_pools
    ]

    cube_pools = [
        CubePool(
            timestamps_us=_mv(p.timestamps_us),
            cube_ts_prefix_sum=_mv(p.cube_ts_prefix_sum),
            track_timestamps_us=_mv(p.track_timestamps_us),
            translations=_mv(p.translations),
            quaternions=_mv(p.quaternions),
            scales=_mv(p.scales),
            colors=_mv(p.colors),
            prim_type_id=p.prim_type_id,
            render_flags=p.render_flags,
        )
        for p in (scene.timestamped_scene.cube_pools or [])
    ]

    cameras = [
        FThetaCamera(
            principal_point=_mv(c.principal_point),
            image_size=_mv(c.image_size),
            fw_poly=_mv(c.fw_poly),
            max_ray_angle=c.max_ray_angle,
            linear_distortion=_mv(c.linear_distortion),
            depth_max=c.depth_max,
        )
        for c in scene.cameras
    ]

    moved_packed = None
    if scene.packed is not None:
        p = scene.packed
        moved_packed = PackedSceneBuffers(
            timestamps=_mv(p.timestamps),
            int32_data=_mv(p.int32_data),
            vertices=_mv(p.vertices),
            triangles=_mv(p.triangles),
            float_data=_mv(p.float_data),
            polyline_meta=p.polyline_meta,
            polygon_meta=p.polygon_meta,
            cube_meta=p.cube_meta,
        )

    return ClipgtGpuScene(
        timestamped_scene=TimestampedScene(
            polyline_pools=polyline_pools,
            polygon_pools=polygon_pools,
            cube_pools=cube_pools,
        ),
        cameras=cameras,
        camera_name_to_id=scene.camera_name_to_id,
        sensor_to_rig={k: _mv(v) for k, v in scene.sensor_to_rig.items()},
        ego_track=EgoTrackData(
            timestamps=_mv(scene.ego_track.timestamps),
            poses_tquat=_mv(scene.ego_track.poses_tquat),
        ),
        device=device,
        packed=moved_packed,
    )


# ---------------------------------------------------------------------------
# Pre-packed GPU upload buffers
# ---------------------------------------------------------------------------


@dataclass
class PackedSceneBuffers:
    """Pre-packed flat tensors ready for direct GL buffer upload.

    All heavy computation (vertex padding, AABB calculation, tensor
    concatenation) is done once at pack time. Upload only needs to
    recompute lightweight pool headers with correct global offsets.
    """

    timestamps: Tensor  # int64, all pools concatenated
    int32_data: Tensor  # int32, prefix sums concatenated
    vertices: Tensor  # float32 [N, 4], padded (x,y,z,0)
    triangles: Tensor  # int32 [N, 4], padded (i0,i1,i2,0)
    float_data: Tensor  # float32, AABBs + cube translations/quaternions/scales/colors
    polyline_meta: List[dict]
    polygon_meta: List[dict]
    cube_meta: List[dict]


def _ensure_cpu(t: Tensor) -> Tensor:
    """Return tensor on CPU, avoiding a copy when already there."""
    return t if t.device.type == "cpu" else t.cpu()


def _compute_element_aabbs(vertices: Tensor, prefix_sum: Tensor) -> Tensor:
    """Compute per-element AABBs: flat [n_elements * 6] float32 tensor.

    CPU: numpy reduceat for single-pass segmented min/max.
    GPU: segment IDs + scatter_reduce.
    """
    n_elem = len(prefix_sum)
    if n_elem == 0 or len(vertices) == 0:
        return torch.zeros(0, dtype=torch.float32, device=vertices.device)

    if vertices.device.type == "cpu":
        verts = vertices.numpy().astype(np.float32, copy=False)
        if verts.ndim == 2 and verts.shape[1] > 3:
            verts = verts[:, :3]

        starts = np.empty(n_elem, dtype=np.intp)
        starts[0] = 0
        starts[1:] = prefix_sum[:-1].numpy()

        e_min = np.minimum.reduceat(verts, starts, axis=0)
        e_max = np.maximum.reduceat(verts, starts, axis=0)

        return torch.from_numpy(np.concatenate([e_min, e_max], axis=-1).reshape(-1).copy())

    verts3 = vertices[:, :3].float() if vertices.shape[1] > 3 else vertices.float()
    starts = torch.zeros(n_elem, dtype=torch.long, device=vertices.device)
    starts[1:] = prefix_sum[:-1]
    counts = prefix_sum.clone()
    counts[1:] = prefix_sum[1:] - prefix_sum[:-1]
    seg_ids = torch.arange(len(verts3), device=vertices.device)
    seg_ids = torch.bucketize(seg_ids, starts, right=True) - 1

    e_min = torch.full((n_elem, 3), float("inf"), device=vertices.device)
    e_max = torch.full((n_elem, 3), float("-inf"), device=vertices.device)
    e_min.scatter_reduce_(0, seg_ids.unsqueeze(1).expand_as(verts3), verts3, reduce="amin")
    e_max.scatter_reduce_(0, seg_ids.unsqueeze(1).expand_as(verts3), verts3, reduce="amax")

    return torch.cat([e_min, e_max], dim=-1).reshape(-1)


def pack_scene(scene: ClipgtGpuScene) -> PackedSceneBuffers:
    """Pre-pack a ClipgtGpuScene into flat tensors for fast GL upload.

    Computation runs on the scene's device (CPU or GPU). Final packed
    buffers are always on CPU for GL upload.
    """
    ts = scene.timestamped_scene
    dev = scene.device
    timestamps_list: list[Tensor] = []
    int32_list: list[Tensor] = []
    vertices_list: list[Tensor] = []
    triangles_list: list[Tensor] = []
    float_list: list[Tensor] = []

    ts_offset = 0
    int32_offset = 0
    vert_offset = 0
    tri_offset = 0
    float_offset = 0

    polyline_meta: list[dict] = []
    polygon_meta: list[dict] = []
    cube_meta: list[dict] = []

    for pool in ts.polyline_pools:
        n_ts = pool.timestamps_us.shape[0]
        n_varrays = pool.varrays_prefix_sum.shape[0]
        n_verts = pool.vertices.shape[0]

        aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum)
        n_aabb_floats = len(aabbs)

        polyline_meta.append(
            {
                "n_ts": n_ts,
                "n_varrays": n_varrays,
                "n_verts": n_verts,
                "prim_type_id": pool.prim_type_id,
                "ts_offset": ts_offset,
                "int32_offset": int32_offset,
                "vert_offset": vert_offset,
                "float_offset": float_offset,
                "n_aabb_floats": n_aabb_floats,
            }
        )

        timestamps_list.append(pool.timestamps_us)
        int32_list.append(pool.timestamped_varrays_prefix_sum.to(torch.int32))
        int32_list.append(pool.varrays_prefix_sum.to(torch.int32))

        vertices_list.append(F.pad(pool.vertices.float(), (0, 1)))

        float_list.append(aabbs)

        ts_offset += n_ts
        int32_offset += n_ts + n_varrays
        vert_offset += n_verts
        float_offset += n_aabb_floats

    for pool in ts.polygon_pools:
        n_ts = pool.timestamps_us.shape[0]
        n_varrays = pool.varrays_prefix_sum.shape[0]
        n_verts = pool.vertices.shape[0]
        n_tris = pool.triangles.shape[0]

        aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum)
        n_aabb_floats = len(aabbs)

        polygon_meta.append(
            {
                "n_ts": n_ts,
                "n_varrays": n_varrays,
                "n_verts": n_verts,
                "n_tris": n_tris,
                "prim_type_id": pool.prim_type_id,
                "ts_offset": ts_offset,
                "int32_offset": int32_offset,
                "vert_offset": vert_offset,
                "tri_offset": tri_offset,
                "float_offset": float_offset,
                "n_aabb_floats": n_aabb_floats,
            }
        )

        timestamps_list.append(pool.timestamps_us)
        int32_list.append(pool.timestamped_varrays_prefix_sum.to(torch.int32))
        int32_list.append(pool.varrays_prefix_sum.to(torch.int32))
        int32_list.append(pool.triangle_prefix_sum.to(torch.int32))

        vertices_list.append(F.pad(pool.vertices.float(), (0, 1)))
        triangles_list.append(F.pad(pool.triangles.to(torch.int32), (0, 1)))

        float_list.append(aabbs)

        ts_offset += n_ts
        int32_offset += n_ts + 2 * n_varrays
        vert_offset += n_verts
        tri_offset += n_tris
        float_offset += n_aabb_floats

    for pool in ts.cube_pools or []:
        n_global_ts = pool.timestamps_us.shape[0]
        n_cubes = pool.scales.shape[0]
        n_track_poses = pool.translations.shape[0]

        cube_meta.append(
            {
                "n_cubes": n_cubes,
                "n_global_ts": n_global_ts,
                "n_track_poses": n_track_poses,
                "prim_type_id": pool.prim_type_id,
                "render_flags": pool.render_flags,
                "ts_offset": ts_offset,
                "int32_offset": int32_offset,
                "float_offset": float_offset,
            }
        )

        timestamps_list.append(pool.timestamps_us)
        timestamps_list.append(pool.track_timestamps_us)
        int32_list.append(pool.cube_ts_prefix_sum.to(torch.int32))
        float_list.append(pool.translations.reshape(-1))
        float_list.append(pool.quaternions.reshape(-1))
        float_list.append(pool.scales.reshape(-1))
        float_list.append(pool.colors.reshape(-1))

        ts_offset += n_global_ts + n_track_poses
        int32_offset += n_cubes
        float_offset += n_track_poses * 7 + n_cubes * 9

    empty_dev = dev if dev.type != "cpu" else torch.device("cpu")
    packed_timestamps = (
        torch.cat(timestamps_list) if timestamps_list else torch.empty(0, dtype=torch.int64, device=empty_dev)
    )
    packed_int32 = torch.cat(int32_list) if int32_list else torch.empty(0, dtype=torch.int32, device=empty_dev)
    packed_vertices = (
        torch.cat(vertices_list) if vertices_list else torch.empty((0, 4), dtype=torch.float32, device=empty_dev)
    )
    packed_triangles = (
        torch.cat(triangles_list) if triangles_list else torch.empty((0, 4), dtype=torch.int32, device=empty_dev)
    )
    packed_floats = torch.cat(float_list) if float_list else torch.empty(0, dtype=torch.float32, device=empty_dev)

    return PackedSceneBuffers(
        timestamps=_ensure_cpu(packed_timestamps),
        int32_data=_ensure_cpu(packed_int32),
        vertices=_ensure_cpu(packed_vertices),
        triangles=_ensure_cpu(packed_triangles),
        float_data=_ensure_cpu(packed_floats),
        polyline_meta=polyline_meta,
        polygon_meta=polygon_meta,
        cube_meta=cube_meta,
    )


def packed_bytes(packed: PackedSceneBuffers) -> int:
    """Estimate memory footprint of pre-packed buffers."""
    total = 0
    for t in (packed.timestamps, packed.int32_data, packed.vertices, packed.triangles, packed.float_data):
        total += t.nelement() * t.element_size()
    return total


def save_scene_to_disk(scene: ClipgtGpuScene, path: Path) -> None:
    """Serialize and write a scene to disk (per-file fallback)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serialize_scene(scene), path)


def load_scene_from_disk(path: Path, device: torch.device = torch.device("cpu")) -> ClipgtGpuScene:
    """Load a serialized scene from disk (per-file fallback)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data.get("version", 0)
    if v not in (1, CACHE_VERSION):
        raise ValueError(f"Cache version mismatch: got {v}, expected {CACHE_VERSION}")
    scene = deserialize_scene(data, device=device)
    if scene.packed is None:
        scene.packed = pack_scene(scene)
    return scene


def _serialize_to_bytes(scene: ClipgtGpuScene) -> bytes:
    """Serialize a scene to raw bytes (for LMDB storage)."""
    buf = io.BytesIO()
    torch.save(serialize_scene(scene), buf)
    return buf.getvalue()


def _deserialize_from_bytes(data: bytes, device: torch.device = torch.device("cpu")) -> ClipgtGpuScene:
    """Deserialize a scene from raw bytes (from LMDB storage)."""
    buf = io.BytesIO(data)
    d = torch.load(buf, map_location="cpu", weights_only=False)
    v = d.get("version", 0)
    if v not in (1, CACHE_VERSION):
        raise ValueError(f"Cache version mismatch: got {v}, expected {CACHE_VERSION}")
    scene = deserialize_scene(d, device=device)
    if scene.packed is None:
        scene.packed = pack_scene(scene)
    return scene


# ---------------------------------------------------------------------------
# L3: LMDB scene store
# ---------------------------------------------------------------------------


class LMDBSceneStore:
    """LMDB-backed key-value store for serialized scenes.

    Single file, memory-mapped reads, concurrent-reader safe. Ideal for
    large scene datasets on network filesystems (e.g. Lustre) where
    per-file I/O is expensive.
    """

    def __init__(self, path: str | Path, map_size: int = 500 * 1024**3, readonly: bool = False):
        if _lmdb is None:
            raise ImportError("lmdb package is required for LMDBSceneStore. Install with: pip install lmdb")
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._env = _lmdb.open(
            str(self._path),
            map_size=map_size,
            readonly=readonly,
            readahead=False,
            meminit=False,
            max_dbs=0,
            lock=not readonly,
        )

    def get(self, key: str, device: torch.device = torch.device("cpu")) -> Optional[ClipgtGpuScene]:
        with self._env.begin(write=False) as txn:
            data = txn.get(key.encode())
            if data is None:
                return None
        return _deserialize_from_bytes(data, device=device)

    def put(self, key: str, scene: ClipgtGpuScene) -> None:
        data = _serialize_to_bytes(scene)
        with self._env.begin(write=True) as txn:
            txn.put(key.encode(), data)

    def put_bytes(self, key: str, data: bytes) -> None:
        """Write pre-serialized bytes (avoids double-serialization in preprocessing)."""
        with self._env.begin(write=True) as txn:
            txn.put(key.encode(), data)

    def contains(self, key: str) -> bool:
        with self._env.begin(write=False) as txn:
            return txn.get(key.encode()) is not None

    def count(self) -> int:
        with self._env.begin(write=False) as txn:
            return txn.stat()["entries"]

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _lmdb_path(cache_dir: Path) -> Path:
    """Path for the LMDB environment within the versioned cache dir."""
    return cache_dir / "scenes.lmdb"


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------


def _tensor_bytes(t) -> int:
    return t.nelement() * t.element_size() if isinstance(t, Tensor) else 0


def scene_bytes(scene: ClipgtGpuScene) -> int:
    """Estimate the memory footprint of a ClipgtGpuScene in bytes."""
    total = 0
    ts = scene.timestamped_scene
    for p in ts.polyline_pools:
        total += _tensor_bytes(p.timestamps_us)
        total += _tensor_bytes(p.timestamped_varrays_prefix_sum)
        total += _tensor_bytes(p.varrays_prefix_sum)
        total += _tensor_bytes(p.vertices)
    for p in ts.polygon_pools:
        total += _tensor_bytes(p.timestamps_us)
        total += _tensor_bytes(p.timestamped_varrays_prefix_sum)
        total += _tensor_bytes(p.varrays_prefix_sum)
        total += _tensor_bytes(p.triangle_prefix_sum)
        total += _tensor_bytes(p.vertices)
        total += _tensor_bytes(p.triangles)
    for p in ts.cube_pools or []:
        total += _tensor_bytes(p.timestamps_us)
        total += _tensor_bytes(p.cube_ts_prefix_sum)
        total += _tensor_bytes(p.track_timestamps_us)
        total += _tensor_bytes(p.translations)
        total += _tensor_bytes(p.quaternions)
        total += _tensor_bytes(p.scales)
        total += _tensor_bytes(p.colors)
    total += _tensor_bytes(scene.ego_track.timestamps)
    total += _tensor_bytes(scene.ego_track.poses_tquat)
    for c in scene.cameras:
        total += _tensor_bytes(c.principal_point)
        total += _tensor_bytes(c.image_size)
        total += _tensor_bytes(c.fw_poly)
        total += _tensor_bytes(c.linear_distortion)
    for v in scene.sensor_to_rig.values():
        total += _tensor_bytes(v)
    if scene.packed is not None:
        total += packed_bytes(scene.packed)
    return total


# ---------------------------------------------------------------------------
# L2: CPU RAM LRU cache
# ---------------------------------------------------------------------------


class CPUSceneCache:
    """Thread-safe LRU cache for ClipgtGpuScene objects on CPU.

    Eviction is based on total memory usage in bytes.
    """

    def __init__(self, max_bytes: int = 16 * 1024**3):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[str, ClipgtGpuScene] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_bytes: int = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[ClipgtGpuScene]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, scene: ClipgtGpuScene) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            size = scene_bytes(scene)
            while self._total_bytes + size > self._max_bytes and self._cache:
                evict_key, _ = self._cache.popitem(last=False)
                self._total_bytes -= self._sizes.pop(evict_key, 0)
            self._cache[key] = scene
            self._sizes[key] = size
            self._total_bytes += size

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def remove(self, key: str) -> Optional[ClipgtGpuScene]:
        with self._lock:
            scene = self._cache.pop(key, None)
            if scene is not None:
                self._total_bytes -= self._sizes.pop(key, 0)
            return scene

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())


# ---------------------------------------------------------------------------
# L1: CUDA tensor cache (GPU-resident scenes)
# ---------------------------------------------------------------------------


class CUDASceneCache:
    """LRU cache for ClipgtGpuScene objects stored as CUDA tensors.

    Eviction is based on total GPU memory usage in bytes.  Accessed only
    from the main thread, so no lock is needed.
    """

    def __init__(self, max_bytes: int = 4 * 1024**3):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[str, ClipgtGpuScene] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_bytes: int = 0
        self._evictions: int = 0

    def get(self, key: str) -> Optional[ClipgtGpuScene]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, scene: ClipgtGpuScene) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        size = scene_bytes(scene)
        while self._total_bytes + size > self._max_bytes and self._cache:
            self._evict_one()
        self._cache[key] = scene
        self._sizes[key] = size
        self._total_bytes += size

    def _evict_one(self) -> None:
        evict_key, _ = self._cache.popitem(last=False)
        self._total_bytes -= self._sizes.pop(evict_key, 0)
        self._evictions += 1

    def contains(self, key: str) -> bool:
        return key in self._cache

    def remove(self, key: str) -> Optional[ClipgtGpuScene]:
        scene = self._cache.pop(key, None)
        if scene is not None:
            self._total_bytes -= self._sizes.pop(key, 0)
        return scene

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def evictions(self) -> int:
        return self._evictions

    def keys(self) -> List[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


def _fmt_bytes(b: float) -> str:
    if b < 1024:
        return f"{b:.0f}B"
    if b < 1024**2:
        return f"{b / 1024:.1f}KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f}MB"
    return f"{b / 1024**3:.2f}GB"


@dataclass
class CacheStats:
    # L2: CPU RAM cache
    l2_size: int = 0
    l2_bytes: int = 0
    l2_max_bytes: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    # L1: CUDA tensor cache
    l1_size: int = 0
    l1_bytes: int = 0
    l1_max_bytes: int = 0
    l1_hits: int = 0
    l1_misses: int = 0
    l1_evictions: int = 0
    # L3: Disk cache
    l3_hits: int = 0
    l3_misses: int = 0
    # Prefetch
    prefetch_queued: int = 0
    prefetch_completed: int = 0

    @property
    def l2_hit_rate(self) -> float:
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l3_hit_rate(self) -> float:
        total = self.l3_hits + self.l3_misses
        return self.l3_hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats("
            f"L1={self.l1_size} {_fmt_bytes(self.l1_bytes)}/{_fmt_bytes(self.l1_max_bytes)} "
            f"hit={self.l1_hit_rate:.1%} evict={self.l1_evictions}, "
            f"L2={self.l2_size} {_fmt_bytes(self.l2_bytes)}/{_fmt_bytes(self.l2_max_bytes)} "
            f"hit={self.l2_hit_rate:.1%}, "
            f"L3 hit={self.l3_hit_rate:.1%}, "
            f"prefetch={self.prefetch_completed}/{self.prefetch_queued})"
        )


# ---------------------------------------------------------------------------
# SceneDatabase: unified multi-level cache
# ---------------------------------------------------------------------------


class SceneDatabase:
    """Multi-level scene cache: L1 (CUDA) / L2 (CPU RAM) / L3 (Disk).

    Thread-safe for concurrent prefetch and access.
    """

    def __init__(
        self,
        scene_paths: Optional[Dict[str, str]] = None,
        cache_dir: Optional[str] = None,
        max_cpu_bytes: int = 16 * 1024**3,
        max_gpu_bytes: int = 4 * 1024**3,
        prefetch_workers: int = 0,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        """
        Args:
            scene_paths: {cache_key: tar_path} mapping. If None, keys are
                derived from paths via scene_key().
            cache_dir: Directory for L3 disk cache. Falls back to
                ``LUDUS_CACHE_DIR`` env var, then ``~/.cache/ludus``.
                A version subdirectory (``v{CACHE_VERSION}/``) is appended
                automatically.
            max_cpu_bytes: Maximum bytes for L2 CPU cache (default 16 GB).
            max_gpu_bytes: Maximum bytes for L1 CUDA tensor cache (default 4 GB).
            prefetch_workers: Number of background threads for async prefetch.
                0 disables async prefetch (prefetch becomes synchronous).
            device: Default device for loaded scenes (CPU recommended;
                GPU transfer happens at upload time).
        """
        self._path_map: Dict[str, str] = scene_paths or {}
        self._cache_dir = None if cache_dir == "" else _resolve_cache_dir(cache_dir)
        self._device = device

        # L2: CPU RAM
        self._cpu_cache = CPUSceneCache(max_bytes=max_cpu_bytes)

        # L1: CUDA tensor cache
        self._cuda_cache = CUDASceneCache(max_bytes=max_gpu_bytes)

        # Stats
        self._stats = CacheStats(
            l2_max_bytes=max_cpu_bytes,
            l1_max_bytes=max_gpu_bytes,
        )

        # L3: LMDB (preferred) with per-file fallback
        self._lmdb: Optional[LMDBSceneStore] = None
        if self._cache_dir is not None and _lmdb is not None:
            try:
                self._lmdb = LMDBSceneStore(_lmdb_path(self._cache_dir))
            except Exception:
                pass

        # Prefetch (L2)
        self._prefetch_workers = prefetch_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_futures: Dict[str, Future] = {}
        self._prefetch_lock = threading.Lock()
        if prefetch_workers > 0:
            self._executor = ThreadPoolExecutor(
                max_workers=prefetch_workers,
                thread_name_prefix="scene_prefetch",
            )

        # Async L1 prefetch (CUDA stream)
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._l1_pending: Set[str] = set()

    # -- Key management ------------------------------------------------

    def register_scenes(self, paths: List[str]) -> List[str]:
        """Register scene tar paths and return their cache keys."""
        keys = []
        for p in paths:
            k = scene_key(p)
            self._path_map[k] = p
            keys.append(k)
        return keys

    def key_for_path(self, path: str) -> str:
        k = scene_key(path)
        if k not in self._path_map:
            self._path_map[k] = path
        return k

    # -- L3: Disk cache (LMDB primary, per-file fallback) ---------------

    def _disk_path(self, key: str) -> Path:
        return _cache_path(self._cache_dir, key)

    def _load_from_disk(self, key: str) -> Optional[ClipgtGpuScene]:
        if self._cache_dir is None:
            return None
        if self._lmdb is not None:
            try:
                scene = self._lmdb.get(key)
                if scene is not None:
                    return scene
            except Exception:
                pass
        dp = self._disk_path(key)
        if not dp.exists():
            return None
        try:
            return load_scene_from_disk(dp, device=torch.device("cpu"))
        except Exception:
            return None

    def _save_to_disk(self, key: str, scene: ClipgtGpuScene) -> None:
        if self._cache_dir is None:
            return
        if self._lmdb is not None:
            try:
                self._lmdb.put(key, scene)
                return
            except Exception:
                pass
        try:
            save_scene_to_disk(scene, self._disk_path(key))
        except Exception:
            pass

    # -- L3/origin: Load scene -----------------------------------------

    def _load_from_origin(self, key: str) -> ClipgtGpuScene:
        tar_path = self._path_map.get(key)
        if tar_path is None:
            raise KeyError(f"No tar path registered for key {key}")
        return load_av2_scene(tar_path, device="cpu")

    # -- Single scene load (L3 → L2) ----------------------------------

    def _ensure_cpu_one(self, key: str) -> ClipgtGpuScene:
        """Load a single scene into L2, going through L3/origin as needed."""
        # L2 hit
        scene = self._cpu_cache.get(key)
        if scene is not None:
            self._stats.l2_hits += 1
            return scene

        self._stats.l2_misses += 1

        # Check for in-flight prefetch
        with self._prefetch_lock:
            fut = self._prefetch_futures.pop(key, None)
        if fut is not None:
            scene = fut.result()
            if scene is not None:
                self._cpu_cache.put(key, scene)
                return scene

        # L3 hit
        scene = self._load_from_disk(key)
        if scene is not None:
            self._stats.l3_hits += 1
            self._cpu_cache.put(key, scene)
            return scene

        self._stats.l3_misses += 1

        # Origin load + populate L3
        scene = self._load_from_origin(key)
        self._save_to_disk(key, scene)
        self._cpu_cache.put(key, scene)
        return scene

    # -- Batch CPU load ------------------------------------------------

    def ensure_cpu(self, keys: List[str]) -> Dict[str, ClipgtGpuScene]:
        """Ensure all scenes are in L2 CPU cache. Returns {key: scene}."""
        result = {}
        missing = []
        for k in keys:
            scene = self._cpu_cache.get(k)
            if scene is not None:
                self._stats.l2_hits += 1
                result[k] = scene
            else:
                missing.append(k)

        if not missing:
            return result

        # Load missing scenes in parallel using prefetch executor or inline
        if self._executor is not None and len(missing) > 1:
            futures = {k: self._executor.submit(self._ensure_cpu_one, k) for k in missing}
            for k, fut in futures.items():
                result[k] = fut.result()
        else:
            for k in missing:
                result[k] = self._ensure_cpu_one(k)

        return result

    # -- L1: CUDA tensor promotion (L2 → L1) ----------------------------

    def _ensure_l1(self, key: str) -> ClipgtGpuScene:
        """Ensure a scene is in the L1 CUDA cache. Returns GPU-resident scene.

        If the key has an in-flight async L1 prefetch, synchronizes the
        prefetch stream first so the data is ready.
        """
        if key in self._l1_pending:
            if self._prefetch_stream is not None:
                self._prefetch_stream.synchronize()
            self._l1_pending.clear()

        scene = self._cuda_cache.get(key)
        if scene is not None:
            self._stats.l1_hits += 1
            return scene

        self._stats.l1_misses += 1

        cpu_scene = self._ensure_cpu_one(key)
        gpu_scene = scene_to_device(cpu_scene, torch.device("cuda", torch.cuda.current_device()))
        self._cuda_cache.put(key, gpu_scene)
        return gpu_scene

    # -- Prefetch (async L3 → L2) -------------------------------------

    def prefetch(self, keys: List[str]) -> None:
        """Async: warm L2 for upcoming scenes. Non-blocking."""
        if self._executor is None:
            return

        with self._prefetch_lock:
            for k in keys:
                if k in self._prefetch_futures:
                    continue
                if self._cpu_cache.contains(k):
                    continue
                self._stats.prefetch_queued += 1
                fut = self._executor.submit(self._prefetch_one, k)
                self._prefetch_futures[k] = fut

    def _prefetch_one(self, key: str) -> Optional[ClipgtGpuScene]:
        try:
            scene = self._load_from_disk(key)
            if scene is None:
                scene = self._load_from_origin(key)
                self._save_to_disk(key, scene)
            self._cpu_cache.put(key, scene)
            self._stats.prefetch_completed += 1
            return scene
        except Exception:
            return None
        finally:
            with self._prefetch_lock:
                self._prefetch_futures.pop(key, None)

    # -- Async L1 prefetch (L2 → L1 on CUDA stream) --------------------

    def prefetch_l1(self, keys: List[str]) -> None:
        """Asynchronously promote scenes from L2 to L1 on a dedicated CUDA stream.

        Only promotes scenes that are already in L2 (skips cache misses).
        The transfers are non-blocking and overlap with rendering on the
        default stream. Call this with next-iteration keys right after
        issuing a render call.
        """
        if not torch.cuda.is_available():
            return

        to_promote = []
        for k in keys:
            if self._cuda_cache.contains(k):
                continue
            scene = self._cpu_cache.get(k)
            if scene is not None:
                to_promote.append((k, scene))

        if not to_promote:
            return

        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream()

        with torch.cuda.stream(self._prefetch_stream):
            for k, cpu_scene in to_promote:
                gpu_scene = scene_to_device(cpu_scene, torch.device("cuda", torch.cuda.current_device()))
                self._cuda_cache.put(k, gpu_scene)
                self._l1_pending.add(k)

    # -- Convenience ---------------------------------------------------

    def get_scene(self, key: str) -> Optional[ClipgtGpuScene]:
        """Get a scene from L2 if available (no loading)."""
        return self._cpu_cache.get(key)

    def load_scene_gpu(self, key: str, device: Optional[torch.device] = None) -> ClipgtGpuScene:
        """Load a scene to GPU without retaining in L1 cache.

        Goes through L2/L3/origin as needed. Caller owns the returned scene;
        GPU memory is freed when the reference is dropped.

        Args:
            key: Scene cache key.
            device: Target CUDA device. Defaults to current CUDA device.

        Returns:
            GPU-resident ClipgtGpuScene.
        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        scene = self._ensure_cpu_one(key)
        if scene.device != device:
            scene = scene_to_device(scene, device)
        return scene

    @property
    def stats(self) -> CacheStats:
        # L2
        self._stats.l2_size = self._cpu_cache.size
        self._stats.l2_bytes = self._cpu_cache.total_bytes
        # L1
        self._stats.l1_size = self._cuda_cache.size
        self._stats.l1_bytes = self._cuda_cache.total_bytes
        self._stats.l1_evictions = self._cuda_cache.evictions
        return self._stats

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._lmdb is not None:
            self._lmdb.close()
            self._lmdb = None
