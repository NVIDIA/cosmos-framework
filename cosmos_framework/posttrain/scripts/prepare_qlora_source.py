# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare the pinned adapter source on CPU, before allocating training accelerators."""

from __future__ import annotations

import argparse
import json

from cosmos_framework.posttrain.utils.adapter_source import prepare_adapter_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    identity = prepare_adapter_snapshot(args.destination)
    print(json.dumps({name: value for name, value in identity.items() if name != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
