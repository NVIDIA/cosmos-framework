# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Package user-owned, already rendered/synchronized Auto media for local training.

This does not render WSM, synthesize measured LiDAR, or label target captions.
It supplies the storage adapter; the training preflight checks semantic alignment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.local_simulation.core import FORMAT_VERSION
from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_VIEW_MAPPING
from cosmos_framework.data.generator.multiview.rig_catalog import (
    MADS_AML_ALT_RIG,
    MADS_MODERATE_PLUS_RIGS,
    MADS_MODERATE_PLUS_RIGS_MAX_6_VIEWS,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import V1P2_TRANSFER_RANGE_PROJECTION


def default_transforms(cameras: list[str], height: int) -> dict[str, Any]:
    """Portable final-stage member schema; each stage overrides its own policy."""
    if height not in (480, 720):
        raise ValueError("Reference Auto streams are 480p or 720p")
    if not cameras or len(set(cameras)) != len(cameras) or set(cameras) - set(CAMERA_VIEW_MAPPING):
        raise ValueError("Supply unique, calibrated physical camera keys")
    if height == 480:
        rigs = MADS_MODERATE_PLUS_RIGS
        weights = [0.6 if rig == MADS_AML_ALT_RIG else 0.3 if len(rig) == 11 else 0.1 / (len(rigs) - 2) for rig in rigs]
    else:
        rigs = [*[list(rig) for rig in combinations(MADS_AML_ALT_RIG, 6)], *MADS_MODERATE_PLUS_RIGS_MAX_6_VIEWS]
        weights = [0.9 / 7] * 7 + [0.1 / len(MADS_MODERATE_PLUS_RIGS_MAX_6_VIEWS)] * len(
            MADS_MODERATE_PLUS_RIGS_MAX_6_VIEWS
        )
    # Missing physical cameras cannot be fabricated. Project the reference
    # catalog onto the available rig and normalize the eligible probabilities.
    eligible = [(rig, weight) for rig, weight in zip(rigs, weights, strict=True) if set(rig) <= set(cameras)]
    if not eligible:
        eligible = [(cameras, 1.0)]
    total = sum(weight for _, weight in eligible)
    return dict(
        dataset_name="local_auto",
        resolution_hw=[height, 832 if height == 480 else 1280],
        camera_keys=cameras,
        camera_sampler={
            "_target_": "cosmos_framework.data.generator.multiview.rig_sampler.ExplicitRigSampler",
            "rigs": [rig for rig, _ in eligible],
            "weights": [weight / total for _, weight in eligible],
        },
        num_cameras_to_keep=None,
        camera_video_key_mapping={key: f"video_{key}" for key in cameras},
        camera_caption_key_mapping={key: f"caption_{key}" for key in set(cameras) | {"camera_front_wide_120fov"}},
        camera_control_key_mapping={key: f"wsm_{key}" for key in cameras},
        camera_attribute_mapping=None,
        caption_probability={"caption_structured": 1.0},
        conditioning_config={"__mapping__": [[0, 0.5], [1, 0.5]]},
        cfg_dropout_rate=0.1,
        per_view_caption_dropout_rate=0.1,
        num_video_frames=400,
        fps_downsample_factor=1,
        single_caption_camera_name=None,
        single_caption_probability=1.0,
        separate_view_text_tokenization=True,
        add_view_prefix_to_caption=True,
        use_system_prompt=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        enable_per_camera_vae_encoding=True,
        share_vision_temporal_positions=True,
        first_caption_paragraph_only=False,
        require_full_num_frames=False,
        lidar_transfer=False,
        load_lidar=True,
        joint_cam_lidar_transfer=True,
        joint_cam_lidar_fixed_camera_frames=None,
        joint_aligned_num_video_frames=None,
        lidar_native_caption_window=True,
        lidar_tokenizer_v1=True,
        lidar_range_projection={
            "_target_": "cosmos_framework.model.generator.tokenizers.lidar.range_projection.LidarRangeProjectionConfig",
            **V1P2_TRANSFER_RANGE_PROJECTION.to_dict(),
        },
        lidar_dropout_rate=0.1,
        lidar_prefix_dropout_rate=0.0,
        control_dropout_rate=0.0,
        lidar_conditioning_probability=0.0,
        rgb_view_completion_probability=0.0,
        lidar_fps=10.0,
    )


def build_bundle(index: Path, tokenizer: Path, output: Path, cameras: list[str], *, hardlink: bool = False) -> None:
    """Create a new bundle; never overwrite an output or modify the input files."""
    index = index.resolve()
    records = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("The input JSONL must contain samples")
    policies = {str(height): default_transforms(cameras, height) for height in (480, 720)}
    identities = set()
    required = {f"{prefix}_{key}" for key in cameras for prefix in ("video", "wsm", "caption")}
    required.add("caption_camera_front_wide_120fov")
    prepared = []
    for record in records:
        identity = record["id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Every sample needs a unique nonempty id")
        identities.add(identity)
        stratum = str(record["stratum"])
        if stratum not in policies:
            raise ValueError("Each sample stratum must be 480 or 720")
        assets = record["assets"]
        if not isinstance(assets, dict) or required - set(assets):
            raise ValueError(f"Missing camera/control/caption members for sample {identity}")
        for key in ("lidar/lidar_rangemap", "lidar/hdmap_rangemap"):
            if key not in assets:
                raise ValueError(f"Joint Auto bundle needs {key}")
        files = {key: (index.parent / relative).resolve() for key, relative in assets.items()}
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        for camera in set(cameras) | {"camera_front_wide_120fov"}:
            metadata = json.loads(files[f"caption_{camera}"].read_text())
            if not isinstance(metadata, dict) or "caption_structured" not in metadata:
                raise ValueError("Per-camera captions require caption_structured chunk metadata")
        prepared.append((identity, stratum, files))
    if not tokenizer.is_dir():
        raise FileNotFoundError(tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    (output / "resources").mkdir()
    shutil.copytree(tokenizer, output / "resources/tokenizer")
    encoded = []
    for sample_index, (identity, stratum, files) in enumerate(prepared):
        members = {}
        for asset_index, (key, path) in enumerate(sorted(files.items())):
            relative = f"assets/{sample_index:06d}_{asset_index:03d}{path.suffix}"
            target = output / relative
            if hardlink:
                os.link(path, target)
            else:
                shutil.copyfile(path, target)
            members[key] = {"path": relative, "name": path.name}
        encoded.append(dict(identity=identity, stratum=stratum, source_uri=f"local://{identity}", assets=members))
    (output / "samples.json").write_text(json.dumps(encoded, sort_keys=True))
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                format_version=FORMAT_VERSION,
                model="auto",
                sample_count=len(encoded),
                complete=True,
                transforms=policies,
            ),
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-keys", nargs="+", required=True)
    parser.add_argument(
        "--hardlink", action="store_true", help="Same filesystem only; linked inputs must remain immutable"
    )
    args = parser.parse_args()
    build_bundle(args.index, args.tokenizer, args.output, args.camera_keys, hardlink=args.hardlink)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
