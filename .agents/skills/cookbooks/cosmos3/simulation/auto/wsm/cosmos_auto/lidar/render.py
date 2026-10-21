# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Render HD-map (control) and measured LiDAR range maps for an RDS-HQ clip.

Pipeline for the HD-map range map (per sweep):

1. gather world points: dense samples of the static map layers (+ optional
   road-surface fill) and of the frame's object cuboids;
2. optional *prefilter* (off by default): project with the sweep-start pose
   and unproject the surviving cells (toolkit step 3.1, thins tens of millions of
   points to at most one per lattice cell). Thinning at a single pose is lossy
   once the ego moves during the sweep (10-24% of cells on a clip driven at
   11 m/s), so it is an opt-in speed trade-off, not the default;
3. rolling-shutter solve: find each point's firing time from the sweep start/end
   poses and express it in the moving sensor frame
   (:func:`cosmos_auto.lidar.shutter.world_points_to_shutter_sensor_frame`);
4. project to the range map, nearest return wins.

Sweep poses come from :mod:`cosmos_auto.lidar.pose_source`; with the default
``rds_hq_pose`` source no LiDAR data is read for the control signal.

The measured LiDAR range map (``render_lidar=True``) is built from ``lidar_raw``
alone with the recorded sweep poses; when it is rendered without the HD-map map
no ego-pose or rig component is opened.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cosmos_auto.lidar import lidar_raw as lidar_raw_module
from cosmos_auto.lidar.hdmap_points import (
    DEFAULT_DENSITY_M,
    DEFAULT_MINIMAP_TYPES,
    DEFAULT_RADIUS_M,
    bbox_points_for_frames,
    load_object_info,
    load_static_map_points,
)
from cosmos_auto.lidar.pose_source import PoseSource, SweepPoses, load_sweep_poses
from cosmos_auto.lidar.rangemap import (
    DEFAULT_MAX_RANGE_M,
    points_to_range_map,
    range_map_to_points,
    save_range_maps_to_tar,
)
from cosmos_auto.lidar.sensor_model import DEFAULT_PARAMETERS_PATH, SpinningLidarModel
from cosmos_auto.lidar.shutter import world_points_to_shutter_sensor_frame

HDMAP_OUTPUT_DIR = "hdmap_rangemap"
LIDAR_OUTPUT_DIR = "lidar_rangemap"
METADATA_OUTPUT_DIR = "lidar_rangemap_metadata"
RENDER_OUTPUT_DIR = "render"


@dataclass
class RangeMapRenderConfig:
    """Options for :func:`render_clip`."""

    input_root: str
    output_root: str
    # Where the sweep poses for the HD-map range map come from (see pose_source.py).
    pose_source: PoseSource = "rds_hq_pose"
    camera_name: str = "camera_front_wide_120fov"
    input_pose_fps: int = 30
    lidar_fps: int = 10
    # Sensor lattice: "auto" = sensor_collection/<clip>.txt, then lidar_sensor_config/<clip>.json,
    # then the bundled default; "default" = bundled default; otherwise a path to a parameters JSON.
    sensor_model: str = "auto"
    lidar_sensor_name: str | None = None
    # Range-map width. None = take ``n_columns`` from the parameters JSON when it has one,
    # otherwise 3600 (sensor_collection files do not store it).
    n_columns: int | None = None
    allow_default_extrinsic: bool = False
    # Which maps to write.
    render_hdmap: bool = True
    render_lidar: bool = False  # measured LiDAR range map; needs lidar_raw/<clip>.tar
    # HD-map content.
    minimap_types: tuple[str, ...] = DEFAULT_MINIMAP_TYPES
    density: float = DEFAULT_DENSITY_M
    radius: float = DEFAULT_RADIUS_M
    fill_in_road: bool = False
    # Opt-in sweep-start thinning (toolkit step 3.1): several times faster per sweep but
    # drops 10-24% of cells under ego motion. Off by default so the control signal is exact.
    prefilter: bool = False
    max_range: float = DEFAULT_MAX_RANGE_M
    max_frames: int | None = None
    export_control: bool = False
    # Visualisation.
    vis: bool = True
    cmap: str = "Spectral"
    vis_fps: int = 10
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    verbose: bool = True


@dataclass
class ClipRenderResult:
    clip_id: str
    frame_ids: list[int]
    outputs: dict[str, str]
    provenance: dict[str, str]
    seconds: float


def check_sensor_consistency(
    model: SpinningLidarModel, extrinsic_provenance: str, clip_id: str
) -> None:
    """Refuse to pair the lattice of one LiDAR with the mounting pose of another.

    The lattice (ring calibration) may come from ``sensor_collection`` while the
    extrinsic comes from ``rig``; when both name their sensor, the names must agree.
    Unnamed sources (bundled default lattice, ``lidar_raw`` or default extrinsic) are
    not checked.
    """
    if model.sensor_name is None or ":" not in extrinsic_provenance:
        return
    source, extrinsic_name = extrinsic_provenance.split(":", 1)
    if source not in {"rig", "sensor_collection"}:
        return
    if extrinsic_name != model.sensor_name:
        raise ValueError(
            f"[{clip_id}] lattice calibration is for {model.sensor_name!r} but the extrinsic from {source} is for "
            f"{extrinsic_name!r}; pass --lidar-sensor-name to select one sensor for both"
        )


def resolve_sensor_model(
    cfg: RangeMapRenderConfig, clip_id: str
) -> tuple[SpinningLidarModel, str]:
    """Pick the lattice calibration for a clip (see ``RangeMapRenderConfig.sensor_model``).

    ``"auto"`` tries ``sensor_collection/<clip>.txt`` first; a collection that cannot
    supply a lattice (no usable lidar block or no ``line_parameters``) is skipped
    with one printed line and the search continues with
    ``lidar_sensor_config/<clip>.json`` and then the bundled default. An explicit
    ``cfg.n_columns`` is forwarded to every constructor; ``None`` lets a parameters
    JSON supply its own value (3600 otherwise).

    Returns ``(model, provenance)`` where ``provenance`` names the file used.
    """
    if cfg.sensor_model == "default":
        return _default_model(cfg), "default"
    if cfg.sensor_model == "auto":
        collection = Path(cfg.input_root) / "sensor_collection" / f"{clip_id}.txt"
        if collection.exists():
            try:
                model = SpinningLidarModel.from_sensor_collection(
                    collection,
                    cfg.lidar_sensor_name,
                    cfg.n_columns if cfg.n_columns is not None else 3600,
                )
            except (ValueError, KeyError) as error:
                print(
                    f"[{clip_id}] sensor_collection skipped for the lattice ({error}); trying the next source"
                )
            else:
                return model, str(collection)
        config_json = Path(cfg.input_root) / "lidar_sensor_config" / f"{clip_id}.json"
        if config_json.exists():
            return SpinningLidarModel.from_parameters_json(
                config_json, cfg.n_columns
            ), str(config_json)
        return _default_model(cfg), "default"
    return SpinningLidarModel.from_parameters_json(
        cfg.sensor_model, cfg.n_columns
    ), cfg.sensor_model


def _default_model(cfg: RangeMapRenderConfig) -> SpinningLidarModel:
    """Bundled default calibration, honouring an explicit ``cfg.n_columns``."""
    if cfg.n_columns is None:
        return SpinningLidarModel.default()
    return SpinningLidarModel.from_parameters_json(
        DEFAULT_PARAMETERS_PATH, cfg.n_columns
    )


def build_static_points(
    cfg: RangeMapRenderConfig,
    clip_id: str,
    sweep_poses: SweepPoses,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Dense world-frame points of all static map layers (+ optional road fill)."""
    layers = load_static_map_points(
        cfg.input_root, clip_id, cfg.minimap_types, cfg.density, cfg.radius
    )
    counts = {name: int(points.shape[0]) for name, points in layers.items()}
    tensors = [
        torch.as_tensor(points, dtype=torch.float32)
        for points in layers.values()
        if points.shape[0] > 0
    ]
    static_points = (
        torch.cat(tensors, dim=0)
        if tensors
        else torch.zeros((0, 3), dtype=torch.float32)
    )
    if cfg.fill_in_road:
        from cosmos_auto.lidar.road_surface import estimate_road_surface_in_world

        lane = torch.as_tensor(
            layers.get("lanelines", np.zeros((0, 3))), dtype=torch.float32
        )
        boundaries = torch.as_tensor(
            layers.get("road_boundaries", np.zeros((0, 3))), dtype=torch.float32
        )
        ego = torch.as_tensor(sweep_poses.ego_positions(), dtype=torch.float32)
        road = estimate_road_surface_in_world(
            boundaries.to(device),
            torch.cat([ego, lane], dim=0).to(device),
            block_size=[100.0, 100.0],
            voxel_sizes=[0.5, 0.5, 0.2],
            fine_voxel_sizes=[0.04, 0.04, 0.2],
            verbose=cfg.verbose,
        ).cpu()
        counts["road_surface"] = int(road.shape[0])
        static_points = torch.cat([road, static_points], dim=0)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return static_points, counts


def render_hdmap_range_map_for_sweep(
    model: SpinningLidarModel,
    world_points: torch.Tensor,
    lidar_to_world_start: np.ndarray,
    lidar_to_world_end: np.ndarray,
    max_range: float = DEFAULT_MAX_RANGE_M,
    prefilter: bool = False,
    ray_directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """HD-map range map ``[n_rows, n_columns]`` for one sweep from world-frame points."""
    device = world_points.device
    start = torch.as_tensor(lidar_to_world_start, dtype=torch.float32, device=device)
    points = world_points.float()
    if prefilter:
        world_to_start = torch.linalg.inv(start)
        points_start = points @ world_to_start[:3, :3].T + world_to_start[:3, 3]
        coarse_map, _ = points_to_range_map(points_start, model, max_range=max_range)
        if ray_directions is None:
            ray_directions = torch.as_tensor(
                model.ray_directions(), dtype=torch.float32, device=device
            )
        points_start = range_map_to_points(coarse_map, ray_directions)
        points = points_start @ start[:3, :3].T + start[:3, 3]
    shutter = world_points_to_shutter_sensor_frame(
        model,
        points,
        start,
        torch.as_tensor(lidar_to_world_end, dtype=torch.float32, device=device),
    )
    range_map, _ = points_to_range_map(
        shutter.points_sensor, model, max_range=max_range
    )
    return range_map


def render_hdmap_range_maps(
    model: SpinningLidarModel,
    sweep_poses: SweepPoses,
    static_points: torch.Tensor,
    bbox_points: list[np.ndarray],
    device: torch.device,
    max_range: float = DEFAULT_MAX_RANGE_M,
    prefilter: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """``[T, n_rows, n_columns]`` HD-map range maps, one per sweep, streamed frame by frame."""
    static_points = static_points.to(device)
    ray_directions = torch.as_tensor(
        model.ray_directions(), dtype=torch.float32, device=device
    )
    maps: list[np.ndarray] = []
    started = time.time()
    for idx in range(len(sweep_poses)):
        frame_boxes = torch.as_tensor(
            bbox_points[idx], dtype=torch.float32, device=device
        )
        world_points = torch.cat([frame_boxes, static_points], dim=0)
        range_map = render_hdmap_range_map_for_sweep(
            model,
            world_points,
            sweep_poses.lidar_to_world_start[idx],
            sweep_poses.lidar_to_world_end[idx],
            max_range=max_range,
            prefilter=prefilter,
            ray_directions=ray_directions,
        )
        maps.append(range_map.cpu().numpy())
        if verbose and (idx % 50 == 0 or idx == len(sweep_poses) - 1):
            print(
                f"  hdmap sweep {idx + 1}/{len(sweep_poses)} ({time.time() - started:.1f}s)"
            )
    return (
        np.stack(maps, axis=0)
        if maps
        else np.zeros((0, *model.shape), dtype=np.float32)
    )


def render_lidar_range_maps(
    lidar_sample: dict,
    model: SpinningLidarModel,
    device: torch.device,
    max_range: float = DEFAULT_MAX_RANGE_M,
) -> tuple[SweepPoses, np.ndarray, np.ndarray]:
    """Measured LiDAR range + intensity maps from ``lidar_raw`` (recorded sweep poses)."""
    from cosmos_auto.lidar.rangemap import project_frames_to_range_maps

    sweep_poses, points, intensities = lidar_raw_module.sweep_poses_and_points(
        lidar_sample, model, device
    )
    range_maps, intensity_maps = project_frames_to_range_maps(
        points, model, device, intensities, max_range=max_range
    )
    assert intensity_maps is not None
    return sweep_poses, range_maps, intensity_maps


def save_metadata(
    path: Path,
    sweep_poses: SweepPoses,
    provenance: dict[str, str],
    model: SpinningLidarModel,
) -> None:
    """Toolkit-compatible ``metadata/<clip>.npz`` (``pose_list``, ``timestamps_list``, ``frame_indices``) plus provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pose_list=np.stack(
            [sweep_poses.lidar_to_world_start, sweep_poses.lidar_to_world_end], axis=1
        ),
        timestamps_list=np.stack(
            [sweep_poses.start_timestamps_us, sweep_poses.end_timestamps_us], axis=1
        ),
        frame_indices=np.asarray([f"{f:06d}" for f in sweep_poses.frame_ids]),
        pose_source=np.asarray(sweep_poses.source),
        provenance=np.asarray(json.dumps(provenance)),
        n_rows=np.asarray(model.n_rows),
        n_columns=np.asarray(model.n_columns),
    )


def render_clip(clip_id: str, cfg: RangeMapRenderConfig) -> ClipRenderResult:
    """Render and save the requested range maps for one clip. Returns output paths.

    * ``render_hdmap``: sweep poses from :func:`load_sweep_poses` (RDS-HQ ego poses +
      rig extrinsic by default), HD-map range map, ``lidar_rangemap_metadata`` from
      those sweep poses.
    * ``render_lidar``: measured range + intensity map from ``lidar_raw`` with the
      recorded sweep poses. Without ``render_hdmap`` nothing else is opened and the
      metadata carries the recorded poses (``pose_source == "lidar_raw"``).

    ``frame_ids`` of the result are the HD-map sweep frames when that map was
    rendered, otherwise the measured sweep frames.
    """
    if not (cfg.render_hdmap or cfg.render_lidar):
        raise ValueError("nothing to render: set render_hdmap and/or render_lidar")
    if cfg.export_control and not cfg.render_hdmap:
        raise ValueError(
            "--export-control requires HD-map rendering; measured LiDAR is not a geometry control"
        )
    started = time.time()
    device = torch.device(cfg.device)
    model, model_provenance = resolve_sensor_model(cfg, clip_id)
    if cfg.verbose:
        print(
            f"[{clip_id}] sensor model {model.shape} from {model_provenance}; pose source {cfg.pose_source}"
        )
    outputs: dict[str, str] = {}
    provenance: dict[str, str] = {"sensor_model": model_provenance}

    lidar_sample: dict | None = None
    if cfg.render_lidar or (cfg.render_hdmap and cfg.pose_source == "lidar_raw"):
        lidar_sample = lidar_raw_module.load_lidar_raw(cfg.input_root, clip_id)

    hdmap_poses: SweepPoses | None = None
    hdmap_maps: np.ndarray | None = None
    if cfg.render_hdmap:
        hdmap_poses, hdmap_maps = _render_hdmap(
            clip_id, cfg, model, device, lidar_sample, provenance, outputs
        )

    lidar_poses: SweepPoses | None = None
    lidar_maps: np.ndarray | None = None
    intensity_maps: np.ndarray | None = None
    if cfg.render_lidar:
        assert lidar_sample is not None
        lidar_poses, lidar_maps, intensity_maps = _render_lidar(
            clip_id, cfg, model, device, lidar_sample, outputs
        )
        if hdmap_poses is None:
            provenance.update({"poses": "lidar_raw", "extrinsic": "lidar_raw"})

    # Metadata follows the HD-map sweep poses when that map exists, else the recorded sweep poses.
    metadata_poses = hdmap_poses if hdmap_poses is not None else lidar_poses
    assert metadata_poses is not None
    metadata_path = Path(cfg.output_root) / METADATA_OUTPUT_DIR / f"{clip_id}.npz"
    save_metadata(metadata_path, metadata_poses, provenance, model)
    outputs["metadata"] = str(metadata_path)

    if cfg.vis:
        from cosmos_auto.lidar.visualize import write_range_map_video

        video_path = Path(cfg.output_root) / RENDER_OUTPUT_DIR / f"{clip_id}.mp4"
        write_range_map_video(
            video_path,
            hdmap_maps=hdmap_maps,
            hdmap_frames=hdmap_poses.frame_ids if hdmap_poses is not None else None,
            lidar_maps=lidar_maps,
            intensity_maps=intensity_maps,
            lidar_frames=lidar_poses.frame_ids if lidar_poses is not None else None,
            cmap=cfg.cmap,
            fps=cfg.vis_fps,
        )
        outputs["render"] = str(video_path)

    seconds = time.time() - started
    if cfg.verbose:
        print(f"[{clip_id}] done in {seconds:.1f}s -> {outputs}")
    return ClipRenderResult(
        clip_id=clip_id,
        frame_ids=metadata_poses.frame_ids,
        outputs=outputs,
        provenance=provenance,
        seconds=seconds,
    )


def _render_hdmap(
    clip_id: str,
    cfg: RangeMapRenderConfig,
    model: SpinningLidarModel,
    device: torch.device,
    lidar_sample: dict | None,
    provenance: dict[str, str],
    outputs: dict[str, str],
) -> tuple[SweepPoses, np.ndarray]:
    """HD-map range map: load sweep poses, gather map + object points, render, save the tar."""
    sweep_poses, pose_provenance = load_sweep_poses(
        cfg.input_root,
        clip_id,
        cfg.pose_source,
        camera_name=cfg.camera_name,
        input_pose_fps=cfg.input_pose_fps,
        lidar_fps=cfg.lidar_fps,
        lidar_sensor_name=cfg.lidar_sensor_name,
        allow_default_extrinsic=cfg.allow_default_extrinsic,
        lidar_sample=lidar_sample,
    )
    provenance.update(pose_provenance)
    check_sensor_consistency(model, pose_provenance.get("extrinsic", ""), clip_id)
    if cfg.max_frames is not None:
        sweep_poses = _truncate(sweep_poses, cfg.max_frames)
    if cfg.verbose:
        print(
            f"[{clip_id}] {len(sweep_poses)} sweeps, frames {sweep_poses.frame_ids[0]:06d}..{sweep_poses.frame_ids[-1]:06d}; {pose_provenance}"
        )
    static_points, counts = build_static_points(cfg, clip_id, sweep_poses, device)
    bbox_points = bbox_points_for_frames(
        load_object_info(cfg.input_root, clip_id),
        sweep_poses.frame_ids,
        cfg.density,
        cfg.radius,
    )
    if cfg.verbose:
        print(f"[{clip_id}] static points {int(static_points.shape[0]):,} {counts}")
    hdmap_maps = render_hdmap_range_maps(
        model,
        sweep_poses,
        static_points,
        bbox_points,
        device,
        cfg.max_range,
        cfg.prefilter,
        cfg.verbose,
    )
    hdmap_tar = Path(cfg.output_root) / HDMAP_OUTPUT_DIR / f"{clip_id}.tar"
    save_range_maps_to_tar(hdmap_maps, sweep_poses.frame_ids, clip_id, hdmap_tar)
    outputs["hdmap_rangemap"] = str(hdmap_tar)
    if cfg.export_control:
        from cosmos_auto.lidar.control import save_control

        control_path = (
            Path(cfg.output_root) / "controls" / clip_id / "lidar_control.safetensors"
        )
        save_control(hdmap_maps, sweep_poses.frame_ids, control_path, clip_id)
        outputs["lidar_control"] = str(control_path)
    return sweep_poses, hdmap_maps


def _render_lidar(
    clip_id: str,
    cfg: RangeMapRenderConfig,
    model: SpinningLidarModel,
    device: torch.device,
    lidar_sample: dict,
    outputs: dict[str, str],
) -> tuple[SweepPoses, np.ndarray, np.ndarray]:
    """Measured range + intensity maps from ``lidar_raw`` with the recorded sweep poses; saves the tar."""
    lidar_poses, lidar_maps, intensity_maps = render_lidar_range_maps(
        lidar_sample, model, device, cfg.max_range
    )
    if cfg.max_frames is not None:
        lidar_poses = _truncate(lidar_poses, cfg.max_frames)
        lidar_maps, intensity_maps = (
            lidar_maps[: len(lidar_poses)],
            intensity_maps[: len(lidar_poses)],
        )
    if cfg.verbose:
        print(f"[{clip_id}] {len(lidar_poses)} measured sweeps from lidar_raw")
    lidar_tar = Path(cfg.output_root) / LIDAR_OUTPUT_DIR / f"{clip_id}.tar"
    save_range_maps_to_tar(
        lidar_maps, lidar_poses.frame_ids, clip_id, lidar_tar, intensity_maps
    )
    outputs["lidar_rangemap"] = str(lidar_tar)
    return lidar_poses, lidar_maps, intensity_maps


def _truncate(sweep_poses: SweepPoses, max_frames: int) -> SweepPoses:
    keep = slice(0, max_frames)
    return SweepPoses(
        frame_ids=sweep_poses.frame_ids[keep],
        lidar_to_world_start=sweep_poses.lidar_to_world_start[keep],
        lidar_to_world_end=sweep_poses.lidar_to_world_end[keep],
        start_timestamps_us=sweep_poses.start_timestamps_us[keep],
        end_timestamps_us=sweep_poses.end_timestamps_us[keep],
        source=sweep_poses.source,
    )
