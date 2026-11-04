# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run one Cosmos3 multiview request that generates cameras and LiDAR jointly."""

import argparse
import base64
import json
import os
import struct
from pathlib import Path

import requests
from common import require_generator_profile
from mv_rgb import (
    NEGATIVE_PROMPT_PATH,
    NUM_FRAMES,
    OUTPUTS,
    SAMPLE_SCENE,
    add_sample_arguments,
    build_views,
    fetch_sample,
    save_cameras,
)

# mv_rgb puts the Auto cookbook helpers on sys.path.
from helpers.lidar_rangemap import convert_control  # noqa: E402

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
LIDAR_CONTROL = OUTPUTS / "prepared" / "lidar_control.safetensors"
MODES = ("text2video", "image2video")


def lidar_data_url(path: Path) -> str:
    """Encode the numeric LiDAR control; it is safetensors, not a media file."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:application/octet-stream;base64,{encoded}"


def prepare_lidar_control(sample_root: Path) -> Path:
    """Convert the sample's HD-map range-map archive to the numeric FP32 control."""
    summary = convert_control(sample_root / SAMPLE_SCENE / "lidar" / "control.tar", LIDAR_CONTROL)
    print(f"Prepared LiDAR control {summary['shape']} at {LIDAR_CONTROL}")
    return LIDAR_CONTROL


def build_request(
    mode: str,
    sample_root: Path,
    caption_folder: str,
    lidar_condition: Path | None = None,
    num_conditional_sweeps: int | None = None,
) -> dict:
    views = build_views(sample_root, caption_folder, with_vision=mode == "image2video")
    lidar = {"control": lidar_data_url(prepare_lidar_control(sample_root))}
    if lidar_condition is not None:
        # Optional measured sweeps that the generated LiDAR continues from.
        lidar["condition"] = lidar_data_url(lidar_condition)
        if num_conditional_sweeps is not None:
            lidar["num_conditional_sweeps"] = num_conditional_sweeps
    return {
        "model_mode": mode,
        "resolution": "480",
        "num_frames": NUM_FRAMES,
        "seed": 42,
        "multiview": {
            "output_modality": "camera_lidar",
            "per_view_negative_prompt": NEGATIVE_PROMPT_PATH.read_text(encoding="utf-8").strip(),
            "views": views,
            "lidar": lidar,
        },
    }


def save_lidar(lidar: dict, output_dir: Path) -> None:
    data = base64.b64decode(lidar["b64_frames"], validate=True)
    # A safetensors file starts with a little-endian header length and JSON header.
    header_len = struct.unpack("<Q", data[:8])[0]
    frames = json.loads(data[8 : 8 + header_len])["frames"]
    if frames["shape"] != lidar["shape"]:
        raise ValueError(f"LiDAR tensor {frames['shape']} != reported {lidar['shape']}")
    (output_dir / "lidar.safetensors").write_bytes(data)
    descriptor = {key: value for key, value in lidar.items() if key != "b64_frames"}
    (output_dir / "lidar.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    print(
        f"Saved LiDAR {frames['dtype']} {frames['shape']} at {lidar['fps']} Hz "
        f"({', '.join(lidar['channels'])}) to {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="text2video")
    add_sample_arguments(parser)
    parser.add_argument(
        "--lidar-condition",
        type=Path,
        help="Optional measured-LiDAR safetensors file with a [3, sweeps, 128, 1800] frames tensor.",
    )
    parser.add_argument(
        "--num-conditional-sweeps",
        type=int,
        help="Measured sweeps to condition on; requires --lidar-condition (the service defaults to 1).",
    )
    args = parser.parse_args()
    if args.num_conditional_sweeps is not None and args.lidar_condition is None:
        parser.error("--num-conditional-sweeps requires --lidar-condition")
    mode = args.mode
    require_generator_profile(NIM_URL, allowed_variants=("multiview",))
    sample_root = fetch_sample(args.sample_root)
    request = build_request(
        mode,
        sample_root,
        args.caption_folder,
        args.lidar_condition,
        args.num_conditional_sweeps,
    )

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=args.timeout)
    response.raise_for_status()

    multiview = response.json()["multiview"]
    if not isinstance(multiview.get("lidar"), dict):
        raise ValueError("Joint response is missing LiDAR output")
    output_dir = OUTPUTS / f"mv_joint_{mode}"
    save_cameras(multiview, request, output_dir)
    save_lidar(multiview["lidar"], output_dir)


if __name__ == "__main__":
    main()
