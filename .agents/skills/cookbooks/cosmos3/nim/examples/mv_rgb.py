# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run one Cosmos3 multiview camera request on the shared Auto daytime sample."""

import argparse
import os
import sys
from pathlib import Path

import requests
from common import decode_video, media_to_data_url, require_generator_profile

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
SKILLS_ROOT = Path(__file__).resolve().parents[4]
AUTO_ROOT = SKILLS_ROOT / "cookbooks" / "cosmos3" / "simulation" / "auto"
NEGATIVE_PROMPT_PATH = (
    SKILLS_ROOT / "cosmos3-nano-transfer-auto" / "inference" / "assets" / "negative-prompt.txt"
)
OUTPUTS = Path(__file__).parent / "outputs"
NUM_FRAMES = 201
# One seven-camera joint request can run longer than an hour on a single GPU.
DEFAULT_TIMEOUT_SECONDS = 7200

# Reuse the Auto cookbook's sample download, checksum check, and caption windows.
sys.path.insert(0, str(AUTO_ROOT))
from helpers.daytime_sample import (  # noqa: E402
    CAPTION_FOLDERS,
    SAMPLE_CAMERAS,
    SAMPLE_SCENE,
    fetch_sample,
    read_caption,
)

CASES = ("wsm_t2v", "i2v")


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=os.environ.get("AUTO_SAMPLE_ROOT"),
        help="Local copy of the daytime sample; downloaded from Hugging Face when unset.",
    )
    parser.add_argument(
        "--caption-folder", choices=CAPTION_FOLDERS, default="captions_with_timestamp"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)


def build_views(sample_root: Path, caption_folder: str, *, with_vision: bool) -> list[dict]:
    scene = sample_root / SAMPLE_SCENE
    views = []
    for camera in SAMPLE_CAMERAS:
        view = {
            "camera_key": camera,
            "control": media_to_data_url(scene / "wsm" / f"{camera}.mp4"),
            "prompt": read_caption(scene / caption_folder / f"{camera}.json", NUM_FRAMES),
        }
        if with_vision:
            # The NIM conditions on the first decoded frame of each RGB clip.
            view["vision"] = media_to_data_url(scene / "rgb" / f"{camera}.mp4")
        views.append(view)
    return views


def build_request(case: str, sample_root: Path, caption_folder: str) -> dict:
    # Captions come only from views[].prompt, and the negative applies to every
    # camera; multiview rejects a top-level prompt or negative_prompt.
    return {
        "model_mode": "image2video" if case == "i2v" else "text2video",
        "resolution": "480",
        "num_frames": NUM_FRAMES,
        "seed": 42,
        "multiview": {
            "output_modality": "camera",
            "per_view_negative_prompt": NEGATIVE_PROMPT_PATH.read_text(encoding="utf-8").strip(),
            "views": build_views(sample_root, caption_folder, with_vision=case == "i2v"),
        },
    }


def save_cameras(multiview: dict, request: dict, output_dir: Path) -> None:
    requested = [view["camera_key"] for view in request["multiview"]["views"]]
    returned = [video["camera_key"] for video in multiview["videos"]]
    if sorted(returned) != sorted(requested) or multiview["camera_order"] != returned:
        raise ValueError(f"Expected cameras {requested}, received {returned}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for video in multiview["videos"]:
        path = output_dir / f"{video['camera_key']}.mp4"
        path.write_bytes(decode_video(video["b64_video"]))
    print(
        f"Saved {len(returned)} cameras, {multiview['frames_per_view']} frames each, "
        f"{multiview['width']}x{multiview['height']} at {multiview['fps']} FPS "
        f"to {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="wsm_t2v")
    add_sample_arguments(parser)
    args = parser.parse_args()
    require_generator_profile(NIM_URL, allowed_variants=("multiview",))
    sample_root = fetch_sample(args.sample_root)
    request = build_request(args.case, sample_root, args.caption_folder)

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=args.timeout)
    response.raise_for_status()

    save_cameras(response.json()["multiview"], request, OUTPUTS / f"mv_rgb_{args.case}")


if __name__ == "__main__":
    main()
