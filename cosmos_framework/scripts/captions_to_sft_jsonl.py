# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert caption_from_video.py output directories into the SFT training JSONL format.

The SFT dataset loader (sft_dataset.py) expects each JSONL line to have:
  uuid, duration, width, height, vision_path, t2w_windows

where t2w_windows is a list of dicts with start_frame, end_frame, temporal_interval
and a caption.  This converter emits **both** caption representations per window:

* ``caption_json`` — the canonical structured-JSON caption object (read from each
  clip's ``caption.json``).  The loader prefers this and trains on it by default.
* ``caption`` — the dense narrative string (read from ``caption.txt``), kept as the
  backup the loader falls back to when ``caption_json`` is absent.

If a clip has no ``caption.json`` (e.g. produced by an older captioner), the row is
written dense-only, exactly as before.

Filters mirror what training actually consumes so dataset counts match:

* clips longer than 61 s are dropped (matches the loader's hard cap);
* windows shorter than ``max(61, num_video_frames)`` frames are dropped.  Pass
  ``--num-video-frames`` to match your training recipe.  The default (-1) applies
  only the loader's metadata minimum of 61 frames, matching the example recipe
  (``num_video_frames=-1``) so short example clips (~85 frames) are kept.

A sibling ``<output>.summary.json`` records kept/dropped counts per reason.

Usage
-----
    python -m cosmos_framework.scripts.captions_to_sft_jsonl \
        --captions-dir outputs/captions \
        --videos-dir outputs/videos \
        -o outputs/my_dataset.jsonl

    # Match a recipe that decodes a fixed number of frames per window:
    python -m cosmos_framework.scripts.captions_to_sft_jsonl \
        --captions-dir outputs/captions \
        --videos-dir outputs/videos \
        -o outputs/my_dataset.jsonl \
        --num-video-frames 93

    # With a custom dense caption key (default: caption):
    python -m cosmos_framework.scripts.captions_to_sft_jsonl \
        --captions-dir outputs/captions \
        --videos-dir outputs/videos \
        -o outputs/my_dataset.jsonl \
        --caption-key qwen3_235b_dense
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated

import tyro

from cosmos_framework.inference.structured_caption import CAPTION_JSON_KEY
from cosmos_framework.scripts.video_metadata import probe_video_metadata

_MAX_DURATION = 61.0  # seconds; matches hard-coded limit in sft_dataset.py
_MIN_FRAMES = 61  # matches the metadata min_frames=61 in get_sft_dataset()
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _find_video(videos_dir: Path, name: str) -> Path | None:
    for ext in _VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _relativize_vision_path(vision_path: str, output_jsonl: Path) -> str:
    """Rewrite ``vision_path`` relative to the output JSONL's directory.

    The SFT loader resolves relative paths against the JSONL's directory, which
    survives moving the dataset to a different mount/container.  URIs containing
    ``://`` (e.g. ``s3://bucket/key``) pass through unchanged.
    """
    if "://" in vision_path:
        return vision_path
    return os.path.relpath(vision_path, start=output_jsonl.parent)


def main(
    captions_dir: Annotated[
        Path, tyro.conf.arg(help="Directory containing per-video caption subdirectories (each with a caption.txt).")
    ],
    videos_dir: Annotated[Path, tyro.conf.arg(help="Directory containing video files named <clip_name>.<ext>.")],
    output: Annotated[Path, tyro.conf.arg(aliases=("-o",), help="Output JSONL path.")],
    caption_key: str = "caption",
    num_video_frames: Annotated[
        int,
        tyro.conf.arg(
            help="Decoded frames per window in your training recipe; windows shorter than "
            "max(61, this) are dropped. -1 (default) applies only the 61-frame metadata "
            "minimum, matching the example recipe."
        ),
    ] = -1,
    min_short_edge: Annotated[
        int, tyro.conf.arg(help="Drop clips whose shortest spatial edge is below this value. 0 disables.")
    ] = 0,
) -> None:
    """Build an SFT JSONL (caption_json + dense caption) from caption dirs and videos."""
    caption_files = sorted(captions_dir.glob("*/caption.txt"))
    # Also accept dirs that only have caption.json (no caption.txt).
    json_only = sorted(
        p.parent / "caption.txt" for p in captions_dir.glob("*/caption.json") if not (p.parent / "caption.txt").exists()
    )
    caption_files = sorted(set(caption_files) | set(json_only))
    if not caption_files:
        print(f"No caption.txt / caption.json files found under {captions_dir}", file=sys.stderr)
        sys.exit(1)

    effective_min_frames = _MIN_FRAMES if num_video_frames <= 0 else max(_MIN_FRAMES, num_video_frames)

    records = []
    drops: Counter[str] = Counter()

    for caption_path in caption_files:
        name = caption_path.parent.name
        dense = caption_path.read_text().strip() if caption_path.exists() else ""

        caption_json_path = caption_path.parent / "caption.json"
        caption_json = None
        if caption_json_path.exists():
            try:
                caption_json = json.loads(caption_json_path.read_text())
            except json.JSONDecodeError as e:
                print(f"  WARN {name}: caption.json is not valid JSON ({e}); using dense only")
        # Fall back to the JSON's temporal_caption for the dense backup if needed.
        if not dense and isinstance(caption_json, dict):
            dense = str(caption_json.get("temporal_caption", "")).strip()

        if not dense and caption_json is None:
            print(f"  SKIP {name}: no caption content")
            drops["empty_caption"] += 1
            continue

        video_path = _find_video(videos_dir, name)
        if video_path is None:
            print(f"  SKIP {name}: no video found in {videos_dir} for name '{name}'")
            drops["missing_video"] += 1
            continue

        try:
            meta = probe_video_metadata(video_path)
        except Exception as e:
            print(f"  SKIP {name}: ffprobe error — {e}")
            drops["ffprobe_error"] += 1
            continue

        if meta["duration"] > _MAX_DURATION:
            print(f"  SKIP {name}: duration {meta['duration']:.1f}s > {_MAX_DURATION}s")
            drops["duration_too_long"] += 1
            continue

        if meta["total_frames"] < effective_min_frames:
            print(f"  SKIP {name}: only {meta['total_frames']} frames < {effective_min_frames}")
            drops["too_few_frames"] += 1
            continue

        if min_short_edge > 0 and min(meta["width"], meta["height"]) < min_short_edge:
            print(f"  SKIP {name}: short edge {min(meta['width'], meta['height'])} < {min_short_edge}")
            drops["short_edge_too_small"] += 1
            continue

        window: dict = {
            "start_frame": 0,
            "end_frame": meta["total_frames"] - 1,
            "temporal_interval": 1,
        }
        if caption_json is not None:
            window[CAPTION_JSON_KEY] = caption_json  # PRIMARY (structured)
        if dense:
            window[caption_key] = dense  # BACKUP (dense)

        record = {
            "uuid": name,
            "duration": meta["duration"],
            "width": meta["width"],
            "height": meta["height"],
            "vision_path": _relativize_vision_path(str(video_path), output),
            "t2w_windows": [window],
        }
        records.append(record)
        kind = "json+dense" if caption_json is not None else "dense"
        print(f"  OK  {name}: {meta['duration']:.1f}s, {meta['total_frames']} frames, {kind}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "captions_dir": str(captions_dir),
        "videos_dir": str(videos_dir),
        "output_jsonl": str(output),
        "records_kept": len(records),
        "records_with_caption_json": sum(1 for r in records if CAPTION_JSON_KEY in r["t2w_windows"][0]),
        "records_dropped": sum(drops.values()),
        "drops_by_reason": dict(drops),
        "filters": {
            "max_duration_s": _MAX_DURATION,
            "min_window_frames": effective_min_frames,
            "min_short_edge": min_short_edge,
            "num_video_frames": num_video_frames,
        },
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nWrote {len(records)} records → {output}")
    print(f"  with caption_json: {summary['records_with_caption_json']}")
    if drops:
        print("Drops by reason:")
        for reason, count in sorted(drops.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {reason}: {count}")
    print(f"Summary: {summary_path}")
    if not records:
        print("ERROR: No valid records written.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)
