# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic, bounded-frustum retrieval over logical latent-frame indices."""

import math
from typing import Any

import torch


class FrustumHistorySelector:
    """Select sinks, recent history, and older frames with high approximate IoU.

    Geometry must already be aligned to latent frames. Poses use OpenCV camera
    coordinates (+x right, +y down, +z forward); intrinsics are in pixels and
    image_size is (WIDTH, HEIGHT). The image rectangle is [0, width] x [0, height].
    Intrinsics may include skew, but must be upper triangular with last row
    [0, 0, 1]. Explicit near/far planes bound depth along camera +z.

    Each frustum uses a fixed 8^3 midpoint grid: u/v are uniform in pixels and
    z^3 is uniform between near^3 and far^3, hence samples are uniform in volume.
    Intersection is the mean of the two directional inside fractions multiplied
    by their source volumes, clamped to the smaller volume before computing IoU.
    This is a deterministic approximation, not exact polyhedral intersection;
    small intersections can be missed and rankings can change at grid boundaries.

    Detached float64 geometry and samples are prepared once on CPU. History is
    scored in batches, one query at a time: storage is O(T * 512), not O(T^2).
    Returned IDs are original logical indices, never compacted cache positions.
    This helper neither stores nor evicts K/V; callers retain the full archive.
    """

    _GRID_SIZE: int = 8
    _HISTORY_BATCH_SIZE: int = 64
    _poses: torch.Tensor
    _intrinsics: torch.Tensor
    _world_points: torch.Tensor
    _volumes: torch.Tensor
    _near: float
    _far: float
    _width: int
    _height: int
    sink_size: int
    retrieved_size: int
    recent_size: int

    def __init__(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        near: float,
        far: float,
        *,
        sink_size: int = 8,
        retrieved_size: int = 32,
        recent_size: int = 32,
    ) -> None:
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or c2w.shape[0] == 0:
            raise ValueError("c2w must have shape [T, 4, 4] with T > 0")
        count = c2w.shape[0]
        if intrinsics.shape not in ((3, 3), (count, 3, 3)):
            raise ValueError("intrinsics must have shape [3, 3] or [T, 3, 3]")
        if c2w.is_complex() or intrinsics.is_complex():
            raise ValueError("geometry must be real and finite")
        if len(image_size) != 2 or any(type(size) is not int or size <= 0 for size in image_size):
            raise ValueError("image_size must contain positive integer WIDTH, HEIGHT")
        if not math.isfinite(near) or not math.isfinite(far) or not 0 < near < far:
            raise ValueError("depth bounds must be finite and satisfy 0 < near < far")
        if any(type(size) is not int or size < 0 for size in (sink_size, retrieved_size, recent_size)):
            raise ValueError("selection sizes must be nonnegative integers")

        self._poses = c2w.detach().to(device="cpu", dtype=torch.float64).clone()
        self._intrinsics = intrinsics.detach().to(device="cpu", dtype=torch.float64)
        if self._intrinsics.ndim == 2:
            self._intrinsics = self._intrinsics.expand(count, -1, -1)
        self._intrinsics = self._intrinsics.clone()
        if not torch.isfinite(self._poses).all() or not torch.isfinite(self._intrinsics).all():
            raise ValueError("geometry must be real and finite")
        last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(count, -1)
        if not torch.allclose(self._poses[:, 3], last_row, atol=1e-6, rtol=0):
            raise ValueError("c2w must have homogeneous last row [0, 0, 0, 1]")
        rotation = self._poses[:, :3, :3]
        identity = torch.eye(3, dtype=torch.float64).expand(count, -1, -1)
        if not torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=1e-4, rtol=0) or not (
            torch.allclose(torch.linalg.det(rotation), torch.ones(count, dtype=torch.float64), atol=1e-4, rtol=0)
        ):
            raise ValueError("c2w rotation must be orthonormal with determinant +1")
        k = self._intrinsics
        k_last_row = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(count, -1)
        if not torch.allclose(k[:, 2], k_last_row, atol=1e-8, rtol=0) or not torch.allclose(
            k[:, 1, 0], torch.zeros(count, dtype=torch.float64), atol=1e-8, rtol=0
        ):
            raise ValueError("intrinsics must be upper triangular with last row [0, 0, 1]")
        if (k[:, 0, 0] <= 0).any() or (k[:, 1, 1] <= 0).any():
            raise ValueError("intrinsics must have positive focal lengths")

        self._width, self._height = image_size
        self._near, self._far = near, far
        self.sink_size, self.retrieved_size, self.recent_size = sink_size, retrieved_size, recent_size
        quantiles = (torch.arange(self._GRID_SIZE, dtype=torch.float64) + 0.5) / self._GRID_SIZE
        # Cubing tensors also lets the finite-volume check reject overflowing depths.
        near_cubed = torch.tensor(near, dtype=torch.float64).pow(3)
        depth_volume = torch.tensor(far, dtype=torch.float64).pow(3) - near_cubed
        self._volumes = (self._width * self._height / (k[:, 0, 0] * k[:, 1, 1])) * depth_volume / 3
        if not torch.isfinite(self._volumes).all() or (self._volumes <= 0).any():
            raise ValueError("frustum volumes must be finite and positive")
        u, v, z = torch.meshgrid(
            quantiles * self._width,
            quantiles * self._height,
            (near_cubed + quantiles * depth_volume).pow(1 / 3),
            indexing="ij",
        )
        pixels = torch.stack((u.flatten(), v.flatten(), torch.ones(u.numel(), dtype=torch.float64)), dim=-1)
        camera_points = (pixels @ torch.linalg.inv(k).transpose(1, 2)) * z.flatten()[None, :, None]
        self._world_points = camera_points @ rotation.transpose(1, 2) + self._poses[:, None, :3, 3]
        if not torch.isfinite(self._world_points).all():
            raise ValueError("frustum samples must be finite")

    @classmethod
    def from_rgb_geometry(
        cls,
        geometry: dict[str, Any],
        num_latent_frames: int,
        temporal_compression_factor: int,
        fps: float,
    ) -> "FrustumHistorySelector":
        """Build the retrieval baseline from RGB poses and explicit camera parameters."""
        required = {"c2w", "image_size", "near", "far", "fps"}
        missing = required - geometry.keys()
        if missing:
            raise ValueError(f"Frustum geometry is missing fields: {sorted(missing)}")
        if float(geometry["fps"]) != fps:
            raise ValueError(f"Frustum geometry FPS {geometry['fps']} does not match inference FPS {fps}")
        c2w = torch.as_tensor(geometry["c2w"], dtype=torch.float64, device="cpu")
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError("Frustum c2w must have shape [RGB_frames, 4, 4]")
        # Causal temporal compression: frame 0 stands alone, and latent f>0 ends
        # at RGB frame f*tcf. Do not silently resample poses or invent calibration.
        rgb_indices = torch.arange(num_latent_frames) * temporal_compression_factor
        if c2w.shape[0] <= int(rgb_indices[-1]):
            raise ValueError("Frustum trajectory is shorter than the generated RGB-frame horizon")
        image_size = geometry["image_size"]
        if (
            not isinstance(image_size, (list, tuple))
            or len(image_size) != 2
            or any(type(size) is not int or size <= 0 for size in image_size)
        ):
            raise ValueError("Frustum image_size must be [width, height] in positive integer pixels")
        if "intrinsics" in geometry and "horizontal_fov_degrees" in geometry:
            raise ValueError("Specify intrinsics or horizontal_fov_degrees, not both")
        if "intrinsics" in geometry:
            intrinsics = torch.as_tensor(geometry["intrinsics"], dtype=torch.float64, device="cpu")
        elif "horizontal_fov_degrees" in geometry:
            fov = geometry["horizontal_fov_degrees"]
            if type(fov) not in (int, float) or not math.isfinite(fov) or not 0 < fov < 180:
                raise ValueError("horizontal_fov_degrees must be finite and strictly between 0 and 180")
            width, height = image_size
            # Estimated centered pinhole camera with square pixels. Vertical FOV
            # follows from the real (unpadded) aspect ratio, not the model canvas.
            focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
            intrinsics = torch.tensor(
                [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float64
            )
        else:
            raise ValueError("Frustum geometry is missing intrinsics or horizontal_fov_degrees")
        if intrinsics.ndim == 3:
            if intrinsics.shape[0] != c2w.shape[0]:
                raise ValueError("Per-frame frustum intrinsics must match the RGB trajectory length")
            intrinsics = intrinsics[rgb_indices]
        return cls(
            c2w[rgb_indices],
            intrinsics,
            (image_size[0], image_size[1]),
            near=float(geometry["near"]),
            far=float(geometry["far"]),
            sink_size=geometry.get("sink_size", 8),
            retrieved_size=geometry.get("retrieved_size", 32),
            recent_size=geometry.get("recent_size", 32),
        )

    def _validate_indices(self, indices: list[int]) -> None:
        if any(type(index) is not int or not 0 <= index < self._poses.shape[0] for index in indices):
            raise ValueError("frame indices must be integers in [0, T)")

    def _inside_fraction(self, points: torch.Tensor, target_indices: list[int]) -> torch.Tensor:
        """Return inside fractions for broadcast-compatible [B, 512, 3] points."""
        poses = self._poses[target_indices]
        camera = (points - poses[:, None, :3, 3]) @ poses[:, :3, :3]
        pixels = camera @ self._intrinsics[target_indices].transpose(1, 2)
        depth = camera[..., 2]
        # Homogeneous pixel inequalities avoid division by zero behind the camera.
        inside = (
            (depth >= self._near)
            & (depth <= self._far)
            & (pixels[..., 0] >= 0)
            & (pixels[..., 0] <= self._width * depth)
            & (pixels[..., 1] >= 0)
            & (pixels[..., 1] <= self._height * depth)
        )
        return inside.to(torch.float64).mean(dim=-1)

    def overlap_scores(self, query_indices: list[int], history_indices: list[int]) -> torch.Tensor:
        """Return CPU float64 [H] approximate IoUs, averaged over query frames.

        History order is preserved. Debug scoring allows arbitrary valid frame
        indices (including self-comparisons); only select enforces past-only IDs.
        Empty history returns an empty tensor; the query must be nonempty.
        """
        if not query_indices:
            raise ValueError("query_indices must not be empty")
        self._validate_indices(query_indices)
        self._validate_indices(history_indices)
        scores = torch.zeros(len(history_indices), dtype=torch.float64)
        for start in range(0, len(history_indices), self._HISTORY_BATCH_SIZE):
            history = history_indices[start : start + self._HISTORY_BATCH_SIZE]
            history_volume = self._volumes[history]
            batch_scores = torch.zeros(len(history), dtype=torch.float64)
            for query in query_indices:
                query_volume = self._volumes[query]
                query_inside = self._inside_fraction(self._world_points[query : query + 1], history)
                history_inside = self._inside_fraction(self._world_points[history], [query])
                intersection = (query_inside * query_volume + history_inside * history_volume) / 2
                intersection = torch.minimum(intersection, torch.minimum(query_volume, history_volume))
                batch_scores += (intersection / (query_volume + history_volume - intersection)).clamp(0, 1)
            scores[start : start + len(history)] = batch_scores / len(query_indices)
        return scores

    def select(self, frame_idx: int, chunk_end: int | None = None) -> list[int]:
        """Return chronological sinks + recent + retrieved IDs from [0, frame_idx).

        Query poses are [frame_idx, chunk_end), defaulting to just frame_idx.
        Sinks take priority over recent frames; retrieval uses only remaining
        older candidates. Equal scores prefer newer logical IDs. There is no
        eviction or remapping, and short histories simply use available frames.
        """
        self._validate_indices([frame_idx])
        if chunk_end is None:
            chunk_end = frame_idx + 1
        if type(chunk_end) is not int or not frame_idx < chunk_end <= self._poses.shape[0]:
            raise ValueError("chunk_end must satisfy frame_idx < chunk_end <= T")
        sink_end = min(self.sink_size, frame_idx)
        recent_start = max(sink_end, frame_idx - self.recent_size)
        sinks = list(range(sink_end))
        recent = list(range(recent_start, frame_idx))
        candidates = list(range(sink_end, recent_start))
        if len(candidates) <= self.retrieved_size:
            retrieved = candidates
        elif self.retrieved_size:
            scores = self.overlap_scores(list(range(frame_idx, chunk_end)), candidates).tolist()
            ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: (item[1], item[0]), reverse=True)
            retrieved = [index for index, _ in ranked[: self.retrieved_size]]
        else:
            retrieved = []
        return sorted(sinks + recent + retrieved)
