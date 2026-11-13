# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU LiDAR previews using Framework's shared Pandar128 calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
from safetensors import safe_open

from cosmos_framework.model.generator.tokenizers.lidar.geometry import PANDAR128_ELEVATIONS_DEG


@dataclass
class LidarSweepPreview:
    range_m: np.ndarray
    intensity: np.ndarray
    valid: np.ndarray
    xyz: np.ndarray
    min_range: float
    max_range: float


def prepare_lidar_sweep(
    sweep: np.ndarray, descriptor: dict, *, smooth_geometry: bool = True,
) -> LidarSweepPreview:
    """Resolve validity and convert one physical sweep into calibrated XYZ."""
    if sweep.shape != (3, 128, 1800):
        raise ValueError(f"Expected one [3,128,1800] sweep, got {sweep.shape}.")

    projection = descriptor["range_projection"]
    if projection.get("sensor", "pandar128") != "pandar128":
        raise ValueError("This calibration supports Pandar128 only.")
    if projection.get("coordinate_system", "x_forward_y_left_z_up") != "x_forward_y_left_z_up":
        raise ValueError("The LiDAR coordinate system does not match this calibration.")
    range_m, intensity, validity = sweep
    min_range = float(projection.get("min_range_m", 5.0))
    max_range = float(projection.get("max_range_m", 100.0))
    valid = np.isfinite(sweep).all(axis=0) & (validity >= descriptor["validity_threshold"])
    valid &= (range_m > 0) & (range_m >= min_range) & (range_m <= max_range)
    geometry_range = np.where(valid, range_m, 0.0)
    geometry_valid = valid.copy()

    # Filter within each beam row. Azimuth wraps; elevations are never averaged.
    if smooth_geometry:
        weighted_range = np.zeros_like(geometry_range)
        total_weight = np.zeros_like(geometry_range)
        range_sigma = 0.2 + 0.03 * geometry_range
        for offset in range(-2, 3):
            neighbor_range = np.roll(geometry_range, offset, axis=1)
            neighbor_valid = np.roll(valid, offset, axis=1)
            spatial_weight = np.exp(-0.5 * (offset / 1.5) ** 2)
            range_weight = np.exp(-0.5 * ((neighbor_range - geometry_range) / range_sigma) ** 2)
            weight = neighbor_valid * spatial_weight * range_weight
            weighted_range += weight * neighbor_range
            total_weight += weight
        geometry_valid &= (total_weight - valid.astype(np.float32)) >= 0.45
        geometry_range = weighted_range / np.maximum(total_weight, 1e-12)

    # Unproject physical ranges with the calibrated beam elevations and azimuths.
    elevation = np.deg2rad(np.asarray(PANDAR128_ELEVATIONS_DEG))[:, None]
    azimuth = np.deg2rad(np.linspace(
        projection.get("azimuth_start_degrees", 180.0),
        projection.get("azimuth_end_degrees", -180.0),
        range_m.shape[1], endpoint=projection.get("azimuth_endpoint", False),
    ))[None, :]
    directions = np.stack(np.broadcast_arrays(
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ), axis=-1)
    xyz = (geometry_range[..., None] * directions)[geometry_valid]
    return LidarSweepPreview(range_m, intensity, valid, xyz, min_range, max_range)


def draw_lidar_sweep(
    preview: LidarSweepPreview,
    descriptor: dict,
    sweep_index: int,
    *,
    smooth_geometry: bool = True,
    max_plot_points: int = 20_000,
    bev_radius_m: float = 80.0,
    bev_size_px: int = 512,
    height_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
    dpi: float = 100,
) -> Figure:
    """Plot range, intensity, metric BEV, and a point cloud for one sweep.

    Supplied height and Z bounds keep video colors and the camera stable.
    """
    if max_plot_points < 1 or bev_size_px < 8:
        raise ValueError("max_plot_points must be positive and bev_size_px must be at least 8.")
    if not math.isfinite(bev_radius_m) or bev_radius_m <= 0 or not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("bev_radius_m and dpi must be positive and finite.")
    range_m, intensity, valid, xyz = preview.range_m, preview.intensity, preview.valid, preview.xyz
    min_range, max_range = preview.min_range, preview.max_range
    if height_limits is None:
        height_lo, height_hi = np.percentile(xyz[:, 2], [2, 98]) if len(xyz) else (-2.0, 2.0)
    else:
        height_lo, height_hi = height_limits
    if height_hi - height_lo < 1e-3:
        height_lo, height_hi = -2.0, 2.0
    height_norm = plt.Normalize(height_lo, height_hi)

    # Rasterize a metric top-down view; the nearest XY return wins each pixel.
    bev = np.zeros((bev_size_px, bev_size_px, 3), dtype=np.uint8)
    inside = (np.abs(xyz[:, :2]) <= bev_radius_m).all(axis=1)
    bev_points = xyz[inside]
    columns = np.floor((bev_radius_m - bev_points[:, 1]) / (2 * bev_radius_m) * bev_size_px).astype(int)
    rows = np.floor((bev_radius_m - bev_points[:, 0]) / (2 * bev_radius_m) * bev_size_px).astype(int)
    inside_pixels = (columns >= 0) & (columns < bev_size_px) & (rows >= 0) & (rows < bev_size_px)
    columns, rows, bev_points = columns[inside_pixels], rows[inside_pixels], bev_points[inside_pixels]
    if len(bev_points):
        pixels = rows * bev_size_px + columns
        order = np.lexsort((np.linalg.norm(bev_points[:, :2], axis=1), pixels))
        _, first = np.unique(pixels[order], return_index=True)
        nearest = order[first]
        colors = plt.get_cmap("turbo")(height_norm(bev_points[nearest, 2]))[:, :3]
        bev[rows[nearest], columns[nearest]] = np.rint(colors * 255).astype(np.uint8)

    figure = plt.figure(figsize=(16, 10), dpi=dpi)
    figure.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.90, wspace=0.25, hspace=0.30)
    time_seconds = descriptor["start_time_seconds"] + sweep_index / descriptor["fps"]
    figure.suptitle(f"LiDAR sweep {sweep_index} at {time_seconds:.2f} s — synthetic output")
    range_axis = figure.add_subplot(2, 2, 1)
    range_image = range_axis.imshow(np.ma.array(range_m, mask=~valid), aspect="auto",
                                    cmap=plt.get_cmap("viridis").with_extremes(bad="black"),
                                    vmin=min_range, vmax=max_range)
    range_axis.set(title="Range — original", xlabel="Azimuth column", ylabel="Beam row")
    figure.colorbar(range_image, ax=range_axis, label="Range (m)")
    intensity_axis = figure.add_subplot(2, 2, 2)
    intensity_image = intensity_axis.imshow(np.ma.array(intensity, mask=~valid), aspect="auto",
                                            cmap=plt.get_cmap("gray").with_extremes(bad="black"),
                                            vmin=0, vmax=1)
    intensity_axis.set(title="Intensity — original", xlabel="Azimuth column", ylabel="Beam row")
    figure.colorbar(intensity_image, ax=intensity_axis, label="Unit intensity")
    bev_axis = figure.add_subplot(2, 2, 3)
    bev_axis.imshow(bev, extent=(bev_radius_m, -bev_radius_m, -bev_radius_m, bev_radius_m))
    geometry_label = "smoothed" if smooth_geometry else "raw"
    bev_axis.set(title=f"Top-down BEV — {geometry_label}", xlabel="Y left (m)", ylabel="X forward (m)")
    cloud_axis = figure.add_subplot(2, 2, 4, projection="3d")
    plot_xyz = xyz[inside]
    selected = np.linspace(0, len(plot_xyz) - 1, min(len(plot_xyz), max_plot_points), dtype=int)
    cloud = cloud_axis.scatter(*plot_xyz[selected].T, c=plot_xyz[selected, 2], cmap="turbo", norm=height_norm,
                               s=0.5, linewidths=0, rasterized=True)
    cloud_axis.view_init(elev=35, azim=-135)
    cloud_axis.set(title=f"Point cloud — {geometry_label}", xlabel="X forward (m)",
                   ylabel="Y left (m)", zlabel="Z up (m)")
    z_lo = min(-5.0, float(plot_xyz[:, 2].min())) if len(plot_xyz) else -5.0
    z_hi = max(5.0, float(plot_xyz[:, 2].max())) if len(plot_xyz) else 5.0
    if z_limits is not None:
        z_lo, z_hi = z_limits
    cloud_axis.set(xlim=(-bev_radius_m, bev_radius_m), ylim=(-bev_radius_m, bev_radius_m), zlim=(z_lo, z_hi))
    cloud_axis.set_box_aspect((2 * bev_radius_m, 2 * bev_radius_m, z_hi - z_lo))
    for axis in (cloud_axis.xaxis, cloud_axis.yaxis, cloud_axis.zaxis):
        axis.set_major_locator(MaxNLocator(3))
    cloud_axis.tick_params(labelsize=8)
    figure.colorbar(cloud, ax=[bev_axis, cloud_axis], label="Height (m)")
    return figure


def render_lidar_video(
    lidar_path: Path,
    descriptor: dict,
    output_path: Path,
    *,
    smooth_geometry: bool = True,
    max_plot_points: int = 20_000,
    bev_radius_m: float = 80.0,
    bev_size_px: int = 512,
    dpi: float = 80,
) -> Path:
    """Stream every sweep as a four-panel H.264 MP4 at the artifact's FPS.

    A first pass estimates clip-wide height percentiles from at most 2048
    uniformly selected points per sweep and computes camera Z bounds from all
    visible points. Each pass reads and processes one sweep at a time. Original
    tensors are unchanged; failed encodes leave an existing video intact.
    """
    fps = float(descriptor["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("LiDAR FPS must be positive and finite.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install FFmpeg and add it to PATH to render LiDAR video.")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("The LiDAR preview output must use the .mp4 extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".part.mp4")
    encoder: subprocess.Popen | None = None
    try:
        with safe_open(lidar_path, framework="numpy") as handle, tempfile.TemporaryFile() as errors:
            frames = handle.get_slice("frames")
            shape = frames.get_shape()
            if len(shape) != 4 or shape[0] != 3 or shape[1] < 1 or shape[2:] != [128, 1800]:
                raise ValueError(f"Expected nonempty [3,T,128,1800] LiDAR, got {shape}.")
            count = shape[1]
            samples: list[np.ndarray] = []
            z_lo, z_hi = -5.0, 5.0
            print(f"Preparing fixed height colors and camera bounds for {count} sweeps.", flush=True)
            for index in range(count):
                preview = prepare_lidar_sweep(frames[:, index], descriptor, smooth_geometry=smooth_geometry)
                if len(preview.xyz):
                    selected = np.linspace(0, len(preview.xyz) - 1, min(len(preview.xyz), 2048), dtype=int)
                    samples.append(preview.xyz[selected, 2])
                    visible = preview.xyz[(np.abs(preview.xyz[:, :2]) <= bev_radius_m).all(axis=1)]
                    if len(visible):
                        z_lo = min(z_lo, float(visible[:, 2].min()))
                        z_hi = max(z_hi, float(visible[:, 2].max()))
            height_lo, height_hi = np.percentile(np.concatenate(samples), [2, 98]) if samples else (-2.0, 2.0)
            del samples
            height_limits = (float(height_lo), float(height_hi))
            width, height = 0, 0
            for index in range(count):
                preview = prepare_lidar_sweep(frames[:, index], descriptor, smooth_geometry=smooth_geometry)
                figure = draw_lidar_sweep(
                    preview, descriptor, index, smooth_geometry=smooth_geometry,
                    max_plot_points=max_plot_points, bev_radius_m=bev_radius_m,
                    bev_size_px=bev_size_px, height_limits=height_limits,
                    z_limits=(z_lo, z_hi), dpi=dpi,
                )
                try:
                    canvas = FigureCanvasAgg(figure)
                    canvas.draw()
                    if encoder is None:
                        width, height = canvas.get_width_height()
                        if width % 2 or height % 2:
                            raise ValueError("Video dimensions must be even; choose an integer dpi.")
                        encoder = subprocess.Popen(
                            [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
                             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(temporary)],
                            stdin=subprocess.PIPE, stderr=errors,
                        )
                    if canvas.get_width_height() != (width, height):
                        raise RuntimeError("The LiDAR preview dimensions changed during encoding.")
                    assert encoder.stdin is not None
                    try:
                        encoder.stdin.write(np.asarray(canvas.buffer_rgba())[:, :, :3].tobytes())
                    except BrokenPipeError as error:
                        encoder.wait()
                        errors.seek(0)
                        raise RuntimeError("FFmpeg failed: " + errors.read().decode(errors="replace")) from error
                finally:
                    plt.close(figure)
                if (index + 1) % 10 == 0 or index + 1 == count:
                    print(f"Rendered {index + 1}/{count} sweeps.", flush=True)
            assert encoder is not None and encoder.stdin is not None
            encoder.stdin.close()
            if encoder.wait() != 0:
                errors.seek(0)
                raise RuntimeError("FFmpeg failed: " + errors.read().decode(errors="replace"))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty LiDAR preview.")
        temporary.replace(output_path)
        print(f"Saved {output_path} — {count} frames at {fps:g} FPS ({count / fps:.2f} s).", flush=True)
        return output_path
    finally:
        if encoder is not None:
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
            if encoder.stdin is not None and not encoder.stdin.closed:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
        temporary.unlink(missing_ok=True)
