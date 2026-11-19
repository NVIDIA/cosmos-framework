# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU preflight: render exactly the observation/action contract used in training."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from failuregen.yam_training.contract import (
    CAMERAS,
    CANVAS,
    CHUNK,
    FPS,
    action_space_for,
    compose_views,
    decode_frames,
    episode_actions,
    file_sha256,
    load_prepared,
    normalize,
)


def write_media(frames: np.ndarray, destination: Path, ffmpeg: str):
    command = [
        ffmpeg,
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{CANVAS}x{CANVAS}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
    ]
    if destination.suffix == ".png":
        command += ["-frames:v", "1", "-update", "1"]
    else:
        command += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    subprocess.run([*command, "-n", str(destination)], input=frames.tobytes(), capture_output=True, check=True)


def prepare(
    prepared: Path,
    dataset: Path,
    output: Path,
    episode: int,
    start_seconds: float,
    chunks: int = 1,
    ffmpeg: str = "ffmpeg",
    seed: int = 0,
) -> dict:
    manifest, stats = load_prepared(prepared)
    contract = manifest["contract"]
    action_space = action_space_for(contract)
    ep = next((ep for ep in manifest["episodes"] if ep["episode"] == episode), None)
    if ep is None or chunks < 1 or start_seconds < 0:
        raise ValueError("Episode must be selected, chunks positive, and start nonnegative")
    start = round(start_seconds * FPS)
    starts = [start + index * CHUNK for index in range(chunks)]
    if not all(index in ep["valid_starts"] for index in starts):
        raise ValueError("Replay touches unknown footage or exceeds the episode; choose a valid earlier start")
    state_path = prepared / ep["states"]
    if file_sha256(state_path) != ep["states_sha256"]:
        raise ValueError("Prepared states changed")
    with np.load(state_path) as arrays:
        state_key = "observations_eef_20d" if action_space == "eef" else "states"
        states = arrays[state_key][start : start + chunks * CHUNK + 1]
        source_times = arrays["source_timestamps"][start : start + chunks * CHUNK + 1]
        first_frame = int(arrays["source_indices"][start])
        actions = normalize(episode_actions(arrays, contract)[start : start + chunks * CHUNK], stats)
    for camera in CAMERAS:
        if (dataset / ep["videos"][camera]).stat().st_size != ep["video_sizes"][camera]:
            raise ValueError("Source video changed")
    frames = [
        decode_frames(dataset / ep["videos"][key], first_frame, chunks * CHUNK + 1, ffmpeg=ffmpeg) for key in CAMERAS
    ]
    mosaic = compose_views(*frames)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_media(mosaic[:1], output / "obs.png", ffmpeg)
    write_media(mosaic, output / "ground_truth.mp4", ffmpeg)
    for index in range(chunks):
        action_path = output / f"actions_{index:03d}.json"
        action_path.write_text(json.dumps(actions[index * CHUNK : (index + 1) * CHUNK].tolist()) + "\n")
        sample = dict(
            model_mode="forward_dynamics",
            name=f"chunk_{index:03d}",
            domain_name=contract["domain"],
            vision_path=str(output / ("obs.png" if index == 0 else f"condition_{index:03d}.png")),
            action_path=str(action_path),
            prompt=ep["caption"],
            action_chunk_size=CHUNK,
            fps=FPS,
            image_size=720,
            view_point="concat_view",
            num_steps=30,
            guidance=1.0,
            shift=10.0,
            seed=seed + index,
        )
        (output / f"sample_{index:03d}.json").write_text(json.dumps(sample, indent=2) + "\n")
    record = dict(
        contract=contract,
        episode=episode,
        split=ep["split"],
        start_seconds=start / FPS,
        chunks=chunks,
        seed=seed,
        stats_sha256=manifest["stats_sha256"],
        states_sha256=ep["states_sha256"],
        annotations_sha256=manifest["annotations_sha256"],
        timing_verified_visually=manifest["timing_verified_visually"],
    )
    (output / "replay.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "telemetry.json").write_text(
        json.dumps(
            {
                "video_seconds": (np.arange(len(states)) / FPS).tolist(),
                "source_timestamp": source_times.tolist(),
                "state": states.tolist(),
                "representation": state_key,
                "left_gripper_column": 9 if action_space == "eef" else 6,
                "right_gripper_column": 19 if action_space == "eef" else 13,
            },
            indent=2,
        )
        + "\n"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start-seconds", type=float, required=True)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.prepared,
                args.dataset,
                args.output,
                args.episode,
                args.start_seconds,
                args.chunks,
                args.ffmpeg,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
