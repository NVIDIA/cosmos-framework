# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run action-only Edge/Nano-DROID policy on the checked-in LeRobot sample."""

import argparse
import json
import os
from pathlib import Path

import av
from PIL import Image
import requests

from action import validate_action_output
from common import media_to_data_url, require_generator_profile

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "generator/action/assets/droid_lerobot_example"
)
OUTPUTS = Path(__file__).parent / "outputs"


def compose_observation() -> Image.Image:
    frames = []
    for camera in ("wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"):
        path = SAMPLE / "videos" / f"observation.image.{camera}" / "chunk-000/file-000.mp4"
        with av.open(str(path)) as video:
            frames.append(next(video.decode(video=0)).to_image().convert("RGB"))
    if any(frame.size != (640, 360) for frame in frames):
        raise ValueError("The DROID sample must have three 640x360 camera views")
    image = Image.new("RGB", (640, 540))
    image.paste(frames[0], (0, 0))
    for index, frame in enumerate(frames[1:]):
        image.paste(frame.resize((320, 180), Image.Resampling.BILINEAR), (320 * index, 360))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="Remove the crumpled paper from the sink and place it in the trash can.",
    )
    args = parser.parse_args()
    require_generator_profile(NIM_URL, allowed_variants=("edge-droid", "nano-droid"))
    OUTPUTS.mkdir(exist_ok=True)
    image_path = OUTPUTS / "droid_observation.png"
    compose_observation().save(image_path)

    request = {
        "model_mode": "policy",
        "prompt": args.prompt,
        "input_reference": media_to_data_url(image_path),
        "action_params": {
            "domain_name": "droid_lerobot",
            "action_chunk_size": 32,
            "observation": {
                # State at frame zero in the accompanying public LeRobot sample.
                "observation/joint_position": [
                    0.04376881942152977, -0.40943676233291626, 0.42356184124946594,
                    -2.431333065032959, -0.027579963207244873, 1.9793097972869873,
                    0.6358067393302917,
                ],
                "observation/gripper_position": 0.0,
            },
        },
        "seed": 0,
    }
    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()
    result = response.json()
    if result.get("b64_image") or result.get("b64_video"):
        raise ValueError("DROID policy must return actions without generated media")
    expected = {**request, "action_params": {**request["action_params"], "raw_action_dim": 8}}
    validate_action_output(result.get("action"), expected)
    output = OUTPUTS / "policy_droid.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validated [32, 8] policy action to {output}")


if __name__ == "__main__":
    main()
