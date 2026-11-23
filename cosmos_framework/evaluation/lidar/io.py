# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert Cosmos Framework metric range-view outputs to local evaluation arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.evaluation.lidar.tokenizer import load_projection, range_to_points


def _crop(array: np.ndarray, projection: dict[str, Any]) -> np.ndarray:
    """Remove explicitly declared symmetric model padding, never resize rays."""
    width = int(projection.get("semantic_width", array.shape[-1]))
    if array.shape[-1] == width:
        return array
    model_width = int(projection.get("model_width", width))
    if (
        array.shape[-1] != model_width
        or model_width <= width
        or (model_width - width) % 2
        or projection.get("model_width_transform", "circular_pad") != "circular_pad"
    ):
        raise ValueError("Range-view width differs from projection; declare padding or pool GT explicitly")
    left = (model_width - width) // 2
    return array[..., left : left + width]  # [T,H,W_semantic]


def read_rangeview(path: Path, projection: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read Framework safetensors or explicit NPZ, preserving physical range and source IDs."""
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        arrays = load_file(path)
        id_key = "frame_ids"
        valid = np.asarray(arrays["range_m"]) > 0  # [T,H,W]
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        id_key = "sweep_ids"
        valid = np.asarray(arrays["valid"])  # [T,H,W]
    else:
        raise ValueError(f"Expected .safetensors or .npz range-view, got {path}")
    ranges = np.asarray(arrays["range_m"], dtype=np.float32)  # [T,H,W]
    ids = np.asarray(arrays[id_key])  # [T]
    if ranges.ndim != 3 or min(ranges.shape) < 1 or not np.isfinite(ranges).all():
        raise ValueError("range_m must be finite nonempty metric [T,H,W]")
    if valid.dtype != np.bool_ or valid.shape != ranges.shape:
        raise ValueError("valid must be boolean with the same shape as range_m")
    if ids.shape != (len(ranges),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Source frame IDs must be integer [T]")
    if len(ids) > 1 and np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Generation frame IDs must be strictly increasing; remove artificial duplicates explicitly")
    result = {"range_m": _crop(ranges, projection), "valid": _crop(valid, projection), "sweep_ids": ids}
    if "intensity" in arrays:
        intensity = np.asarray(arrays["intensity"], dtype=np.float32)  # [T,H,W]
        if intensity.shape != ranges.shape or not np.isfinite(intensity).all():
            raise ValueError("intensity must be finite and match range_m")
        if np.any((intensity < 0) | (intensity > 1)):
            raise ValueError("intensity must be measured/predicted unit intensity in [0,1]")
        result["intensity"] = _crop(intensity, projection)
    return result


def _read_framework_tar(path: Path, projection: dict[str, Any]) -> dict[str, np.ndarray]:
    """Use the installed Framework local tar adapter; this import never needs a generator model."""
    from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
        list_rangemap_sweep_ids,
        load_prepared_lidar_video_3ch,
    )
    from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig

    if projection.get("sensor", "pandar128") != "pandar128" or "elevation_degrees" in projection:
        raise ValueError(
            "Framework sparse tar loading supports its native Pandar128 projection; use NPZ for custom rigs"
        )
    fields = {field.name for field in LidarRangeProjectionConfig.__attrs_attrs__}
    configured = {key: value for key, value in projection.items() if key in fields}
    configured["model_width"] = configured.get("semantic_width", 1800)
    configured.setdefault("semantic_width", 1800)
    configured["model_width_transform"] = "circular_pad"
    native_projection = LidarRangeProjectionConfig.from_dict(configured)
    ids = list_rangemap_sweep_ids(str(path))
    if not ids:
        raise ValueError(f"No sweeps in {path}")
    video, selected = load_prepared_lidar_video_3ch(  # video:[3,T,128,W], selected:[T]
        str(path),
        min_source_frame_id=min(ids),
        max_source_frame_id=max(ids),
        prepend_first_frame=False,
        range_projection=native_projection,
    )
    values = video.detach().float().cpu().numpy()  # [3,T,128,W]
    return {
        "range_m": values[0],
        "intensity": values[1],
        "valid": values[2] > 0.5,
        "sweep_ids": np.asarray(selected, dtype=np.int64),
    }


def _pool_reference(reference: dict[str, np.ndarray], width: int) -> dict[str, np.ndarray]:
    """Keep the nearest valid return per adjacent azimuth group, including its intensity."""
    ranges, valid = reference["range_m"], reference["valid"]  # [T,H,W_native]
    if ranges.shape[-1] % width:
        raise ValueError("Reference width must be an integer multiple of generated width")
    factor = ranges.shape[-1] // width
    shape = (*ranges.shape[:-1], width, factor)
    candidates = np.where(valid.reshape(shape), ranges.reshape(shape), np.inf)  # [T,H,W,factor]
    index = candidates.argmin(axis=-1, keepdims=True)  # [T,H,W,1]
    selected = np.take_along_axis(candidates, index, axis=-1)[..., 0]  # [T,H,W]
    kept = np.isfinite(selected)  # [T,H,W]
    result = {
        "range_m": np.where(kept, selected, 0).astype(np.float32),
        "valid": kept,
        "sweep_ids": reference["sweep_ids"],
    }
    if "intensity" in reference:
        intensity = np.take_along_axis(reference["intensity"].reshape(shape), index, axis=-1)[..., 0]  # [T,H,W]
        result["intensity"] = np.where(kept, intensity, 0).astype(np.float32)
    return result


def prepare_framework_outputs(
    manifest: Path,
    output_dir: Path,
    projection_path: Path | None = None,
    select_reference_frames: bool = False,
    pool_reference: bool = False,
) -> dict[str, Any]:
    """Export aligned XYZ/XYZI and NPZ from native Framework output/GT pairs without resmoothing."""
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Input must be a nonempty JSON list of {clip_id,generated,reference}")
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")
    projection = load_projection(projection_path)
    output_dir.mkdir(parents=True)
    clouds: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        clip_id = entry["clip_id"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise ValueError("clip_id must be a unique nonempty string")
        if Path(clip_id).name != clip_id or clip_id in {".", ".."}:
            raise ValueError("clip_id must be one path component")
        seen.add(clip_id)
        generated_path = manifest.parent / entry["generated"]
        reference_path = manifest.parent / entry["reference"]
        generated = read_rangeview(generated_path, projection)
        if reference_path.suffix == ".tar" and not pool_reference:
            raise ValueError("Sparse GT tar uses Framework nearest-return projection; explicitly pass --pool-reference")
        reference_projection = (
            {
                key: value
                for key, value in projection.items()
                if key
                not in {
                    "semantic_width",
                    "model_width",
                }
            }
            if pool_reference
            else projection
        )
        reference = (
            _read_framework_tar(reference_path, {**projection, "semantic_width": generated["range_m"].shape[-1]})
            if reference_path.suffix == ".tar"
            else read_rangeview(reference_path, reference_projection)
        )
        if select_reference_frames:
            lookup = {int(value): index for index, value in enumerate(reference["sweep_ids"])}
            if any(int(value) not in lookup for value in generated["sweep_ids"]):
                raise ValueError(f"{clip_id}: reference lacks generated source frame IDs")
            indices = [lookup[int(value)] for value in generated["sweep_ids"]]
            reference = {key: value[indices] for key, value in reference.items()}
        if not np.array_equal(generated["sweep_ids"], reference["sweep_ids"]):
            raise ValueError(
                f"{clip_id}: exact source frame IDs differ; use explicit reference selection if appropriate"
            )
        if reference["range_m"].shape != generated["range_m"].shape:
            if not pool_reference or reference["range_m"].shape[:-1] != generated["range_m"].shape[:-1]:
                raise ValueError(f"{clip_id}: range grids differ; explicitly pool native GT to generated resolution")
            reference = _pool_reference(reference, generated["range_m"].shape[-1])
        minimum, maximum = float(projection.get("min_range_m", 5.0)), float(projection.get("max_range_m", 100.0))
        if np.any(reference["valid"] & ((reference["range_m"] < minimum) | (reference["range_m"] > maximum))):
            raise ValueError(f"{clip_id}: GT valid rays fall outside the declared metric band")
        cloud: dict[str, Any] = {"clip_id": clip_id, "generated": [], "reference": []}
        for arm, values in (("generated", generated), ("reference", reference)):
            arm_dir = output_dir / clip_id / arm
            arm_dir.mkdir(parents=True)
            np.savez_compressed(arm_dir.parent / f"{arm}.npz", **values)
            for index, (ranges, valid) in enumerate(zip(values["range_m"], values["valid"], strict=True)):
                xyz = range_to_points(ranges, valid, projection)  # [N,3]
                path = arm_dir / f"frame_{index:06d}.npy"
                np.save(path, xyz, allow_pickle=False)
                cloud[arm].append(str(path.relative_to(output_dir)))
                if "intensity" in values:
                    mask = valid & (ranges > 0)  # [H,W]
                    xyzi = np.column_stack((xyz, values["intensity"][index][mask]))  # [N,4]
                    xyzi_dir = output_dir / clip_id / f"{arm}_xyzi"
                    xyzi_dir.mkdir(exist_ok=True)
                    np.save(xyzi_dir / path.name, xyzi.astype(np.float32), allow_pickle=False)
        for key in ("lidar_to_world", "annotations"):
            if key in entry:
                cloud[key] = str((manifest.parent / entry[key]).resolve())
        cloud["frame_ids"] = generated["sweep_ids"].tolist()
        clouds.append(cloud)
    (output_dir / "clouds.json").write_text(json.dumps(clouds, indent=2) + "\n")
    (output_dir / "projection.json").write_text(json.dumps(projection, indent=2) + "\n")
    metadata: dict[str, Any] = {
        "clips": len(clouds),
        "frames": sum(len(clip["generated"]) for clip in clouds),
        "reference_selected_by_source_ids": select_reference_frames,
        "reference_nearest_pooling_requested": pool_reference,
        "safetensors_are_masked_generation_artifacts": True,
        "tokenizer_warning": "Framework saved generation ranges are masked/smoothed: do not use them for tokenizer fidelity",
        "clouds_manifest": str((output_dir / "clouds.json").resolve()),
    }
    (output_dir / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
