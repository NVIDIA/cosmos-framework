# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare exported 20D YAM EEF intervals for fine-tuning existing domain 16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import EEF_CONTRACT, FPS, file_sha256
from .prepare import prepare as prepare_joints


def check_export_manifest(manifest: dict, dataset: Path):
    expected = {
        "status": "relative_geometry_complete_checkpoint_contract_unverified",
        "pose_convention": "delta_T[t] = inverse(T[t-1]) @ T[t] at sampled observation times",
        "reference_point": "tcp_site at gripper mount (flange-origin candidate)",
        "rotation_encoding": "first two matrix columns concatenated: R00,R10,R20,R01,R11,R21",
        "arm_order": ["left", "right"],
        "position_units": "meters",
        "source_fps": 30,
        "fps": 5,
        "horizon": 16,
        "gripper": "continuous recorded state at destination time t; no differencing or remapping",
        "columns": [
            f"{arm}_{field}"
            for arm in ("left", "right")
            for field in ("dx", "dy", "dz", "R00", "R10", "R20", "R01", "R11", "R21", "gripper_t")
        ],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"EEF export {key!r} differs from the supported 5 FPS relative convention")
    domain = manifest.get("target_domain", {})
    if domain.get("name") != "xdof_yam" or domain.get("id") != 16:
        raise ValueError("Expected xdof_yam/domain-16 EEF export")
    if manifest.get("normalization", {}).get("applied") is not False:
        raise ValueError("Expected unnormalized EEF actions; do not normalize twice")
    # Source keys may be absolute macOS paths. Compare provenance by suffix;
    # never try to open those machine-specific paths on the cluster.
    info_hashes = [value for key, value in manifest.get("source_hashes", {}).items() if key.endswith("/meta/info.json")]
    if info_hashes != [file_sha256(dataset / "meta/info.json")]:
        raise ValueError("EEF export provenance does not match this dataset's metadata")


def poses(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError("Expected finite (N, 20) EEF vectors")
    result = np.broadcast_to(np.eye(4), (len(values), 2, 4, 4)).copy()
    for arm, offset in enumerate((0, 10)):
        a, b = values[:, offset + 3 : offset + 6], values[:, offset + 6 : offset + 9]
        rotation = np.stack((a, b, np.cross(a, b)), axis=-1)
        if not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.allclose(
            np.linalg.det(rotation), 1, atol=1e-6, rtol=0
        ):
            raise ValueError("EEF rotation columns do not form proper rotations")
        result[:, arm, :3, :3] = rotation
        result[:, arm, :3, 3] = values[:, offset : offset + 3]
    return result


def validate_arrays(data, joints, indices, timestamps):
    n = len(indices)
    expected = {
        "observation_frame_index": indices,
        "observation_video_timestamp": indices / 30,
        "observation_recorded_timestamp": timestamps,
        "action_start_frame_index": indices[:-1],
        "action_end_frame_index": indices[1:],
        "action_timestamp": indices[1:] / 30,
    }
    for key, value in expected.items():
        actual = np.asarray(data[key])
        if actual.shape != value.shape or not np.allclose(actual, value, atol=1e-9, rtol=0):
            raise ValueError(f"EEF {key} is not aligned with the recorded camera/state rows")
    observations, actions = data["observations_eef_20d"], data["actions_raw"]
    if observations.shape != (n, 20) or actions.shape != (n - 1, 20):
        raise ValueError("EEF export must have N observations and N-1 actions")
    absolute, relative = poses(observations), poses(actions)
    if not np.allclose(absolute[:-1] @ relative, absolute[1:], atol=1e-7, rtol=0):
        raise ValueError("EEF action direction/convention fails pose reconstruction")
    if not np.allclose(observations[:, [9, 19]], joints[:, [6, 13]], atol=1e-7, rtol=0):
        raise ValueError("EEF gripper observations differ from recorded joint-state grippers")
    if not np.array_equal(actions[:, [9, 19]], observations[1:, [9, 19]]):
        raise ValueError("EEF grippers must use destination observations")


def prepare(
    dataset: Path, output: Path, *, eef_root: Path | None = None, splits_path: Path | None = None, seed: int = 42
) -> dict:
    dataset = dataset.resolve()
    eef_root = (eef_root or dataset / ".labeling/cosmos_yam_report_5fps").resolve()
    export_path = eef_root / "manifest.json"
    export = json.loads(export_path.read_text())
    check_export_manifest(export, dataset)
    exports = {entry["episode"]: entry for entry in export["episodes"]}
    if len(exports) != len(export["episodes"]):
        raise ValueError("Duplicate episode IDs in EEF manifest")
    selected = json.loads((dataset / "failuregen_annotations.json").read_text())["episodes"]
    for key in selected:
        index = int(key.split("_")[-1])
        if index not in exports:
            raise ValueError(f"Missing EEF export for selected episode {index}")
        entry = exports[index]
        source = eef_root / entry["output"]
        if source.resolve().parent != eef_root or file_sha256(source) != entry["output_sha256"]:
            raise ValueError(f"EEF episode {index} checksum/path differs from its manifest")

    # Reuse the tested episode splits, unknown-footage filtering and source-row
    # checks. Only the prepared sidecar is replaced with EEF intervals.
    manifest = prepare_joints(dataset, output, seed=seed, splits_path=splits_path)
    manifest.update(
        status="in_progress",
        contract=EEF_CONTRACT,
        eef_export_manifest_sha256=file_sha256(export_path),
        eef_export_root=str(eef_root),
        pretrained_preprocessing_verified=False,
    )
    destination = output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    train_actions = []
    for episode in manifest["episodes"]:
        source = eef_root / exports[episode["episode"]]["output"]
        path = output / episode["states"]
        with np.load(path, allow_pickle=False) as original:
            joints = original["states"]
            indices, timestamps = original["source_indices"], original["source_timestamps"]
        with np.load(source, allow_pickle=False) as arrays:
            validate_arrays(arrays, joints, indices, timestamps)
            actions = np.asarray(arrays["actions_raw"], dtype=np.float32)
            observations = np.asarray(arrays["observations_eef_20d"], dtype=np.float32)
        np.savez_compressed(
            path,
            actions_raw=actions,
            observations_eef_20d=observations,
            source_indices=indices,
            source_timestamps=timestamps,
        )
        episode.update(states_sha256=file_sha256(path), eef_source=source.name, eef_source_sha256=file_sha256(source))
        if episode["split"] == "train":
            valid = np.array(
                [
                    not any(
                        i / FPS < segment["end"] and (i + 1) / FPS >= segment["start"]
                        for segment in episode["excluded_intervals"]
                    )
                    for i in range(len(actions))
                ]
            )
            train_actions.append(actions[valid])
    raw = np.concatenate(train_actions)
    lo, hi = np.quantile(raw, [0.01, 0.99], axis=0)
    center, span = (lo + hi) / 2, np.maximum(hi - lo, 1e-3)
    stats = {
        "contract": EEF_CONTRACT,
        "train_episodes": manifest["splits"]["train"],
        "train_transitions": len(raw),
        "minimum_quantile_range": 1e-3,
        "q01": (center - span / 2).tolist(),
        "q99": (center + span / 2).tolist(),
    }
    stats_path = output / "action_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    manifest.update(status="prepared", stats_sha256=file_sha256(stats_path))
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eef-root", type=Path)
    parser.add_argument("--splits", type=Path)
    args = parser.parse_args()
    result = prepare(args.dataset, args.output, eef_root=args.eef_root, splits_path=args.splits)
    print(json.dumps({"contract": result["contract"], "windows": result["window_counts"]}, indent=2))


if __name__ == "__main__":
    main()
