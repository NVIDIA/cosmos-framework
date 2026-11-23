#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Print the episode, frame and task counts of a LeRobot v3 dataset folder from its metadata (no framework needed).

usage: inspect_libero10.py --root data/LIBERO_LeRobot_v3/libero_10
Expected for nvidia/LIBERO_LeRobot_v3 libero_10 at revision e5907374: episodes 379  frames 101469  tasks 10  fps 20
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset folder holding meta/info.json")
    args = ap.parse_args()
    info_path = Path(args.root) / "meta" / "info.json"
    if not info_path.is_file():
        print(f"no LeRobot metadata at {info_path}", file=sys.stderr)
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    cameras = sorted(k for k, v in info.get("features", {}).items() if v.get("dtype") in ("video", "image"))
    print(f"codebase_version {info.get('codebase_version')}  robot {info.get('robot_type')}  cameras {cameras}")
    print(
        f"episodes {info.get('total_episodes')}  frames {info.get('total_frames')}  "
        f"tasks {info.get('total_tasks')}  fps {info.get('fps')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
