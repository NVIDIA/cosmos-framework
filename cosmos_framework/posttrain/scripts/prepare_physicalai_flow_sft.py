# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Lossless, restartable PhysicalAI clip preparation for native Alpamayo SFT.

The pinned Alpamayo loader owns frame selection, egomotion interpolation and
vehicle-frame conversion. Each committed clip retains its calibration, source
identity, decoded timestamps, poses and lossless RGB bytes. A caller assembles a
finite split only after every expected clip has a verified completion record.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from cosmos_framework.posttrain.model.alpamayo.sft import PHYSICALAI_CAMERAS, PHYSICALAI_PROFILE, physicalai_frames
from cosmos_framework.posttrain.utils.object_store import file_sha256

FORMAT = "cosmos-physicalai-flow-preparation-v1"
NATIVE_LOADER_SHA256 = "43abc6bcda9a5a384aea780ed29d2e40e2179a166ca1487cccafba2676d50840"


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def verify_clip(directory, identity):
    """A directory without its final marker never becomes a usable source row."""
    root = Path(directory)
    marker = root / "complete.json"
    if not marker.is_file():
        raise ValueError(f"incomplete PhysicalAI clip: {root}")
    record = json.loads(marker.read_text())
    if record.get("format") != FORMAT or record.get("identity") != identity:
        raise ValueError(f"PhysicalAI clip source/profile identity differs: {root}")
    expected = {"row.json", *[f"{camera}-{i}.png" for camera in PHYSICALAI_CAMERAS for i in range(4)]}
    if set(record.get("files", {})) != expected:
        raise ValueError(f"incomplete PhysicalAI clip file inventory: {root}")
    for name, digest in record["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or file_sha256(str(root / path)) != digest:
            raise ValueError(f"PhysicalAI prepared clip bytes differ: {root}/{name}")
    row = json.loads((root / "row.json").read_text())
    physicalai_frames(row)
    return row, record


def write_clip(sample, calibration, destination, identity):
    """Commit one native-loader result; preserve incomplete attempts for diagnosis."""
    root = Path(destination)
    if root.exists():
        return verify_clip(root, identity)
    clip = identity["clip_id"]
    if not re.fullmatch(r"[a-zA-Z0-9-]+", clip) or sample["clip_id"] != clip:
        raise ValueError("invalid or inconsistent PhysicalAI clip ID")
    frames = sample["image_frames"].cpu().numpy()
    if frames.ndim != 5 or frames.shape[:3] != (4, 4, 3) or frames.dtype != np.uint8:
        raise ValueError("native PhysicalAI loader must supply uint8 [4,4,3,H,W] images")
    if sample["camera_indices"].tolist() != [0, 1, 2, 6] or sample["t0_us"] != identity["t0_us"]:
        raise ValueError("native PhysicalAI camera or keyframe identity differs")
    fields = {
        name: sample[name].cpu().numpy()
        for name in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
    }
    expected = {
        "ego_history_xyz": (1, 1, 16, 3),
        "ego_history_rot": (1, 1, 16, 3, 3),
        "ego_future_xyz": (1, 1, 64, 3),
        "ego_future_rot": (1, 1, 64, 3, 3),
    }
    if any(value.shape != expected[name] or not np.isfinite(value).all() for name, value in fields.items()):
        raise ValueError("native PhysicalAI poses must be finite 16-history/64-future vehicle-frame samples")
    if not np.allclose(fields["ego_history_xyz"][0, 0, -1], 0, atol=1e-6, rtol=0) or not np.allclose(
        fields["ego_history_rot"][0, 0, -1], np.eye(3), atol=1e-6, rtol=0
    ):
        raise ValueError("PhysicalAI history must end at the vehicle-frame origin at t0")
    observation = {
        "ego_history_xyz": fields["ego_history_xyz"][0, 0].tolist(),
        "ego_history_rot": fields["ego_history_rot"][0, 0].tolist(),
        "ego_future_rot": fields["ego_future_rot"][0, 0].tolist(),
        "camera_order": list(PHYSICALAI_CAMERAS),
        "camera_indices": [0, 1, 2, 6],
        "frame_timestamps_us": sample["absolute_timestamps"].tolist(),
        "t0_us": sample["t0_us"],
        "coordinate_frame": "vehicle_at_t0",
        "trajectory_dt_seconds": 0.1,
        "calibration": calibration,
    }
    row = {
        "id": f"physicalai-{clip}-{sample['t0_us']}",
        "episode_id": f"physicalai-{identity['source_revision']}-{clip}",
        "observation": observation,
        "actions": fields["ego_future_xyz"][0, 0].tolist(),
        "media": {cam: [f"clips/{clip}/{cam}-{i}.png" for i in range(4)] for cam in PHYSICALAI_CAMERAS},
    }
    physicalai_frames(row)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(f".{root.name}.incomplete-{uuid.uuid4().hex}")
    temporary.mkdir()
    for c, camera in enumerate(PHYSICALAI_CAMERAS):
        intrinsics = calibration["camera_intrinsics"][camera]
        if (intrinsics["height"], intrinsics["width"]) != frames.shape[-2:]:
            raise ValueError(f"PhysicalAI decoded dimensions differ from source calibration: {camera}")
        for i in range(4):
            name = f"{camera}-{i}.png"
            rgb = frames[c, i].transpose(1, 2, 0)
            Image.fromarray(rgb).save(temporary / name, compress_level=1)
            with Image.open(temporary / name) as image:
                if not np.array_equal(np.asarray(image), rgb):
                    raise ValueError("PhysicalAI PNG roundtrip changed RGB bytes")
    _json(temporary / "row.json", row)
    record = {
        "format": FORMAT,
        "identity": identity,
        "files": {p.name: file_sha256(str(p)) for p in sorted(temporary.iterdir())},
        "decoded_rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
    }
    _json(temporary / "complete.json", record)
    # A same-clip concurrent producer cannot replace an existing committed tree.
    os.rename(temporary, root)
    return row, record


def prepare_clip(avdi, clip_id, out, *, source_revision, source_files, t0_us=5_100_000):
    """Prepare one declared clip using the exact public native dataset reader."""
    import inspect

    import pandas as pd
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    if (
        importlib.metadata.version("physical-ai-av") != "0.2.3"
        or file_sha256(inspect.getfile(load_physical_aiavdataset)) != NATIVE_LOADER_SHA256
    ):
        raise ValueError("PhysicalAI preparation requires SDK0.2.3 and the pinned Alpamayo native dataset loader")
    if avdi.revision != source_revision or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("PhysicalAI source must be an immutable, matching dataset revision")
    required = ["egomotion", "camera_intrinsics", "sensor_extrinsics", *PHYSICALAI_CAMERAS]
    archives = {}
    for feature in required:
        if avdi._get_clip_feature_source(clip_id, feature) is not avdi:
            raise ValueError(f"PhysicalAI preparation requires one immutable source; redirected feature {feature}")
        if not bool(avdi.feature_presence.loc[clip_id, feature]):
            raise ValueError(f"PhysicalAI clip {clip_id} has no required {feature}")
        name = avdi._get_clip_feature_filename(clip_id, feature)
        item = source_files.get(name)
        if (
            not isinstance(item, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
            or item.get("bytes", 0) < 1
        ):
            raise ValueError(f"PhysicalAI source manifest lacks immutable size/hash: {name}")
        archives[feature] = {"path": name, **item}
    identity = {
        "format": FORMAT,
        "profile": PHYSICALAI_PROFILE,
        "clip_id": clip_id,
        "source_repository": avdi.repo_id,
        "source_revision": source_revision,
        "t0_us": t0_us,
        "native_loader_sha256": NATIVE_LOADER_SHA256,
        "sdk": "0.2.3",
        "source_archives": archives,
        "source_read_mode": "native SDK remote range reads; ZIP CRC validation; source archive hashes are repository-declared, prepared bytes verified",
    }
    destination = Path(out) / "clips" / clip_id
    if destination.exists():
        return verify_clip(destination, identity)
    # Preserve exact raw records, including source quaternion conventions. These
    # records are metadata; pose conversion remains in the unchanged native loader.
    calibration = {}
    for feature in ("camera_intrinsics", "sensor_extrinsics"):
        with avdi.open_file(archives[feature]["path"], maybe_stream=True) as stream:
            frame = pd.read_parquet(stream).loc[clip_id]
        calibration[feature] = {str(name): {str(k): float(v) for k, v in row.items()} for name, row in frame.iterrows()}
    sample = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi, maybe_stream=True)
    return write_clip(sample, calibration, destination, identity)


def main():
    import argparse

    import physical_ai_av

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, help="JSON repository/revision/files size and SHA256 map")
    parser.add_argument("--clip-ids", required=True, help="JSON list of declared clip IDs, fixed before processing")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.source_manifest).read_text())
    clips = json.loads(Path(args.clip_ids).read_text())
    if (
        not isinstance(clips, list)
        or not clips
        or any(not isinstance(c, str) for c in clips)
        or len(set(clips)) != len(clips)
    ):
        raise ValueError("PhysicalAI clip IDs must be a nonempty unique string list")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        repo_id=manifest["repository"],
        revision=manifest["revision"],
        cache_dir=args.cache_dir,
        split_name="alpamayo_aligned",
        confirm_download_threshold_gb=float("inf"),
    )
    for index, clip in enumerate(clips):
        prepare_clip(avdi, clip, args.out, source_revision=manifest["revision"], source_files=manifest["files"])
        print(json.dumps({"completed": index + 1, "expected": len(clips), "clip_id": clip}), flush=True)


if __name__ == "__main__":
    main()
