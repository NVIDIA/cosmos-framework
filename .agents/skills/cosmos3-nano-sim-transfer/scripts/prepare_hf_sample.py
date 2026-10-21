#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare native inference requests and an example2 training row from HF assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(example: str, sample_dir: Path, output: Path) -> None:
    sample_dir = sample_dir.resolve()
    prompt_file = sample_dir / f"{example}_upsampled_prompt.json"
    if example == "example1":
        prompt = json.loads(prompt_file.read_text())["prompt"]
        modalities = ("depth", "seg")
        frames, window, aspect_ratio = 600, 601, "1,1"
    else:
        prompt = prompt_file.read_text()
        modalities = ("edge", "blur", "depth", "seg")
        frames, window, aspect_ratio = 121, 121, "16,9"

    defaults_file = Path(__file__).resolve().parents[1] / "inference/assets/sample_args.json"
    defaults = json.loads(defaults_file.read_text())
    requests = output / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    rgb = str(sample_dir / f"{example}.mp4")
    depth = str(sample_dir / f"{example}_depth.mp4")
    segmentation = str(sample_dir / f"{example}_segmentation.mp4")
    for modality in modalities:
        request = {
            **defaults,
            "name": f"{example}_{modality}",
            "model_mode": "video2video",
            "prompt": prompt,
            "fps": 30.0,
            "num_frames": window,
            "num_video_frames_per_chunk": window,
            "max_frames": frames,
            "aspect_ratio": aspect_ratio,
            "seed": 2025,
        }
        if modality == "edge":
            request.update(vision_path=rgb, edge={"preset_edge_threshold": "medium"})
        elif modality == "blur":
            request.update(vision_path=rgb, blur={"preset_blur_strength": "medium"})
        elif modality == "depth":
            request["depth"] = {"control_path": depth}
        else:
            request["seg"] = {"control_path": segmentation}
        (requests / f"{modality}.json").write_text(json.dumps(request, indent=2) + "\n")

    if example == "example2":
        row = {
            "identity": example,
            "rgb": rgb,
            "depth": depth,
            "seg": segmentation,
            "caption_path": str(prompt_file),
            "caption_start": 0,
            "caption_end": frames,
        }
        (output / "training.jsonl").write_text(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("example1", "example2"), default="example2")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.example, args.sample_dir, args.output)


if __name__ == "__main__":
    main()
