# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Model-independent LiDAR generation metrics on local XYZ arrays and features.

NumPy implementations of spatial, temporal, and distributional metrics.
No feature extractor is loaded here: FUD and Utonia MMD require externally
computed, identically preprocessed Utonia features.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, jensenshannon


def load_cloud(path: Path) -> np.ndarray:
    """Read finite floating-point sensor-coordinate XYZ, in meters."""
    points = np.load(path, allow_pickle=False)  # [N,3]
    if points.ndim != 2 or points.shape[1] != 3 or not np.issubdtype(points.dtype, np.floating):
        raise ValueError(f"{path}: expected floating-point [N,3] XYZ")
    if not np.isfinite(points).all():
        raise ValueError(f"{path}: XYZ must be finite")
    # Use the same float32 point representation as saved inference artifacts.
    return points.astype(np.float32)  # [N,3]


def load_poses(path: Path, frames: int) -> np.ndarray:
    """Read ordered sensor-to-world transforms shared by generated and GT."""
    poses = np.load(path, allow_pickle=False).astype(np.float64)  # [T,4,4]
    if poses.shape != (frames, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("lidar_to_world must be a finite [T,4,4] array")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("lidar_to_world must have homogeneous final rows")
    rotations = poses[:, :3, :3]  # [T,3,3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-2):
        raise ValueError("lidar_to_world must contain rigid rotations")
    if np.any(np.linalg.det(rotations) <= 0):
        raise ValueError("lidar_to_world rotations must be proper")
    return poses


def load_clips(manifest: Path) -> list[dict[str, Any]]:
    """Validate clip identities and ordered paired cloud lists."""
    clips = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(clips, list) or not clips:
        raise ValueError("manifest must be a nonempty JSON list")
    identifiers = [entry["clip_id"] for entry in clips]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(clips):
        raise ValueError("clip_id must be a unique nonempty string")
    for entry in clips:
        if not entry["generated"] or len(entry["generated"]) != len(entry["reference"]):
            raise ValueError("generated/reference lists must have equal positive lengths")
    return clips


def bev_counts(points: np.ndarray) -> np.ndarray:
    """100x100 counts: 3D radius [5,100), each XY coordinate [-100,100)."""
    points = points.astype(np.float32)  # [N,3]
    radius = np.linalg.norm(points, axis=1)  # [N]
    xy = points[(radius >= 5) & (radius < 100), :2]  # [M,2]
    cells = np.floor((xy + 100) / 2).astype(np.int64)  # [M,2]
    cells = cells[((cells >= 0) & (cells < 100)).all(axis=1)]  # [K,2]
    return np.bincount(cells[:, 0] * 100 + cells[:, 1], minlength=10_000)  # [10000]


def js_distance(generated: np.ndarray, reference: np.ndarray) -> float | None:
    """Natural-log Jensen-Shannon distance; unavailable for an empty population."""
    if generated.sum() == 0 or reference.sum() == 0:
        return None
    return float(jensenshannon(generated.astype(np.float64), reference.astype(np.float64)))


def bev_mmd2(generated: np.ndarray, reference: np.ndarray) -> float:
    """Biased RBF MMD² over per-frame PDFs, including within-population diagonal."""
    generated = generated.astype(np.float32)  # [T,10000]
    reference = reference.astype(np.float32)  # [T,10000]
    generated = generated / generated.sum(axis=1, keepdims=True).clip(min=1e-12)  # [T,10000]
    reference = reference / reference.sum(axis=1, keepdims=True).clip(min=1e-12)  # [T,10000]
    xx = np.exp(-cdist(generated, generated, metric="sqeuclidean") / 0.5)  # [T,T]
    yy = np.exp(-cdist(reference, reference, metric="sqeuclidean") / 0.5)  # [T,T]
    xy = np.exp(-cdist(generated, reference, metric="sqeuclidean") / 0.5)  # [T,T]
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def chamfer(first: np.ndarray, second: np.ndarray) -> float:
    """Half-symmetric mean exact Euclidean nearest-neighbor distance, meters."""
    if not len(first) or not len(second):
        raise ValueError("Chamfer requires two nonempty point clouds")
    forward = KDTree(second).query(first, k=1, eps=0, p=2, workers=1)[0]  # [N]
    backward = KDTree(first).query(second, k=1, eps=0, p=2, workers=1)[0]  # [M]
    return float(0.5 * (forward.mean() + backward.mean()))


def _temporal_score(frames: list[np.ndarray], poses: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    world_frames: list[np.ndarray] = []
    for points, pose in zip(frames, poses, strict=True):
        points = points.astype(np.float64)  # [N,3]
        if len(points) > 30_000:
            points = points[rng.choice(len(points), 30_000, replace=False)]  # [30000,3]
        world_frames.append(points @ pose[:3, :3].T + pose[:3, 3])
    return math.fsum(chamfer(current, previous) for previous, current in zip(world_frames, world_frames[1:])) / (
        len(frames) - 1
    )


def ctc_clip(
    generated: list[np.ndarray], reference: list[np.ndarray], poses: np.ndarray, clip_id: str
) -> dict[str, float] | None:
    """Adjacent-frame CTC in shared GT world coordinates, cap 30k/frame, seed 0."""
    if any(len(points) == 0 for points in generated + reference):
        return None
    if len(generated) < 2:
        raise ValueError("CTC requires at least two ordered frames")
    material = f"lidar-bench-ctc\0{0}\0{clip_id}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return {
        "generated": _temporal_score(generated, poses, seed),
        "reference": _temporal_score(reference, poses, seed),
    }


def corpus_temporal(scores: list[dict[str, float]]) -> dict[str, float]:
    """Equal clip means, followed by the ratio of those corpus means."""
    generated = math.fsum(item["generated"] for item in scores) / len(scores)
    reference = math.fsum(item["reference"] for item in scores) / len(scores)
    if reference <= 0:
        raise ValueError("CTC GT corpus mean must be positive; its ratio is undefined")
    return {"generated": generated, "reference": reference, "ratio": generated / reference}


def evaluate_clouds(manifest: Path) -> dict[str, object]:
    """Score paired clips; corpus CTC is available only if every clip has it."""
    generated_total = np.zeros(10_000, dtype=np.int64)  # [10000]
    reference_total = np.zeros(10_000, dtype=np.int64)  # [10000]
    results: list[dict[str, Any]] = []
    for entry in load_clips(manifest):
        generated = [load_cloud(manifest.parent / path) for path in entry["generated"]]
        reference = [load_cloud(manifest.parent / path) for path in entry["reference"]]
        generated_counts = np.stack([bev_counts(points) for points in generated])  # [T,10000]
        reference_counts = np.stack([bev_counts(points) for points in reference])  # [T,10000]
        generated_sum = generated_counts.sum(axis=0)  # [10000]
        reference_sum = reference_counts.sum(axis=0)  # [10000]
        generated_total += generated_sum  # [10000]
        reference_total += reference_sum  # [10000]
        temporal = None
        if "lidar_to_world" in entry:
            poses = load_poses(manifest.parent / entry["lidar_to_world"], len(generated))  # [T,4,4]
            temporal = ctc_clip(generated, reference, poses, entry["clip_id"])
        results.append(
            {
                "clip_id": entry["clip_id"],
                "frame_count": len(generated),
                "bev_jsd_local": js_distance(generated_sum, reference_sum),
                "bev_mmd2_local_sigma0p5": bev_mmd2(generated_counts, reference_counts),
                "ctc": temporal,
            }
        )
    temporal_scores = [result["ctc"] for result in results if result["ctc"] is not None]
    return {
        "bev_jsd_global": js_distance(generated_total, reference_total),
        "bev_jsd_local_macro": (
            math.fsum(result["bev_jsd_local"] for result in results) / len(results)
            if all(result["bev_jsd_local"] is not None for result in results)
            else None
        ),
        "bev_mmd2_local_sigma0p5_macro": math.fsum(result["bev_mmd2_local_sigma0p5"] for result in results)
        / len(results),
        "ctc": corpus_temporal(temporal_scores) if len(temporal_scores) == len(results) else None,
        "ctc_available_clip_count": len(temporal_scores),
        "clip_count": len(results),
        "clips": results,
    }


def _validate_features(features: np.ndarray) -> None:
    if not np.issubdtype(features.dtype, np.number) or np.issubdtype(features.dtype, np.complexfloating):
        raise ValueError("features must contain real numeric values")
    if features.ndim != 2 or min(features.shape) < 1 or features.shape[0] < 2 or not np.isfinite(features).all():
        raise ValueError("features must be finite [N,D], N >= 2 and D >= 1")


def frechet_distance(generated: np.ndarray, reference: np.ndarray) -> float:
    """Unbiased-covariance FUD with float64 PSD or low-rank cross trace."""
    for features in (generated, reference):
        _validate_features(features)
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("feature dimensions must match")
    generated = generated.astype(np.float64)  # [Ng,D]
    reference = reference.astype(np.float64)  # [Nr,D]
    delta = generated.mean(axis=0) - reference.mean(axis=0)  # [D]
    generated = generated - generated.mean(axis=0)  # [Ng,D]
    reference = reference - reference.mean(axis=0)  # [Nr,D]
    ng, nr = len(generated) - 1, len(reference) - 1
    traces = (generated * generated).sum() / ng + (reference * reference).sum() / nr
    if max(len(generated), len(reference)) < generated.shape[1]:
        cross = generated @ reference.T / math.sqrt(ng * nr)  # [Ng,Nr]
        cross_trace = np.linalg.svd(cross, compute_uv=False).sum()
    else:
        cg = generated.T @ generated / ng  # [D,D]
        cr = reference.T @ reference / nr  # [D,D]
        values, vectors = np.linalg.eigh(cg)  # [D],[D,D]
        root = (vectors * np.sqrt(values.clip(min=0))) @ vectors.T  # [D,D]
        sandwich = root @ cr @ root  # [D,D]
        cross_trace = np.sqrt(np.linalg.eigvalsh((sandwich + sandwich.T) / 2).clip(min=0)).sum()
    return float(max(0.0, delta @ delta + traces - 2 * cross_trace))


def polynomial_mmd2(generated: np.ndarray, reference: np.ndarray, keys: list[str]) -> float:
    """Unbiased (xᵀy/D+1)³ MMD², SHA-key selected cap 4096; may be negative."""
    for features in (generated, reference):
        _validate_features(features)
    if generated.shape != reference.shape or len(keys) != len(generated):
        raise ValueError("UMMD requires paired same-shape features and one key per row")
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("UMMD sample keys must be unique nonempty strings")
    indices = sorted(range(len(keys)), key=lambda i: (hashlib.sha256(keys[i].encode()).digest(), keys[i]))[:4096]
    generated = generated[indices].astype(np.float64)  # [M,D]
    reference = reference[indices].astype(np.float64)  # [M,D]
    dimension, count = generated.shape[1], len(indices)
    xx = (generated @ generated.T / dimension + 1) ** 3  # [M,M]
    yy = (reference @ reference.T / dimension + 1) ** 3  # [M,M]
    xy = (generated @ reference.T / dimension + 1) ** 3  # [M,M]
    return float((xx.sum() - np.trace(xx) + yy.sum() - np.trace(yy)) / (count * (count - 1)) - 2 * xy.mean())


def evaluate_features(generated: Path, reference: Path, keys: Path) -> dict[str, float | int]:
    """Score identically extracted Utonia features; never invent model features."""
    generated_features = np.load(generated, allow_pickle=False)  # [N,D]
    reference_features = np.load(reference, allow_pickle=False)  # [N,D]
    sample_keys = json.loads(keys.read_text(encoding="utf-8"))
    if not isinstance(sample_keys, list):
        raise ValueError("sample keys must be a JSON list")
    return {
        "fud": frechet_distance(generated_features, reference_features),
        "utonia_mmd2": polynomial_mmd2(generated_features, reference_features, sample_keys),
        "fud_frame_count": len(generated_features),
        "utonia_mmd2_frame_count": min(4096, len(generated_features)),
    }
