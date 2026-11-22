# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fetch the shared Auto daytime sample and prepare its numeric LiDAR control.

The last line printed is the verified sample root, for the curl requests in
multiview.md.
"""

import argparse
import os
from pathlib import Path

from mv_joint import prepare_lidar_control
from mv_rgb import fetch_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=os.environ.get("AUTO_SAMPLE_ROOT"),
        help="Local copy of the daytime sample; downloaded from Hugging Face when unset.",
    )
    args = parser.parse_args()
    sample_root = fetch_sample(args.sample_root)
    prepare_lidar_control(sample_root)
    print(sample_root)


if __name__ == "__main__":
    main()
