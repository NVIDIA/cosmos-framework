# SPDX-License-Identifier: OpenMDW-1.1
"""Source-vs-Lance storage comparison for the video-carrying tables.

Answers "does the re-encoded table blow up disk?": for each modality, the size of
the original video files the base loader reads vs the converted Lance table
(video + label tables — everything the Lance loader needs).

    python table_sizes.py --droid-root <root>/success --droid-uri <lance_dir> \
        --vsft-jsonl <video_dataset_file.jsonl> --vsft-uri <lance_dir>

Either pair may be omitted. Local paths only (S3 tables are byte-identical copies).
"""

from __future__ import annotations

import argparse
import json
import os


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path, followlinks=True):
        for f in files:
            p = os.path.join(root, f)
            if os.path.exists(p):
                total += os.path.getsize(p)
    return total


def _row(label: str, src: int, lance: int) -> None:
    print(f"{label:<12} source={src / 1e9:6.2f} GB   lance={lance / 1e9:6.2f} GB   ratio={lance / src:.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--droid-root", help="DROID success/ root (source videos under videos/)")
    ap.add_argument("--droid-uri", help="composed-DROID Lance dir")
    ap.add_argument("--vsft-jsonl", help="vision-SFT video_dataset_file.jsonl")
    ap.add_argument("--vsft-uri", help="vision-SFT Lance dir")
    args = ap.parse_args()

    if args.droid_root and args.droid_uri:
        _row("action", _dir_bytes(os.path.join(args.droid_root, "videos")), _dir_bytes(args.droid_uri))
    if args.vsft_jsonl and args.vsft_uri:
        base = os.path.dirname(os.path.abspath(args.vsft_jsonl))
        clips: dict[str, int] = {}
        with open(args.vsft_jsonl) as fh:
            for line in fh:
                vp = json.loads(line)["vision_path"]
                p = vp if vp.startswith("/") else os.path.join(base, vp)
                clips[p] = os.path.getsize(p)
        _row("vision-sft", sum(clips.values()), _dir_bytes(args.vsft_uri))


if __name__ == "__main__":
    main()
