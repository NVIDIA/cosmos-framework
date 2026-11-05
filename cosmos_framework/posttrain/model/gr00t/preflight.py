# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Allocation-side GR00T model runtime probe, invoked through the shared launcher."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if os.environ.get("GR00T_DIT_SDPA_MODE") != "math":
        raise ValueError("GR00T preflight requires GR00T_DIT_SDPA_MODE=math")
    from gr00t.model.modules.dit import _should_force_math_sdpa

    if not _should_force_math_sdpa():
        raise ValueError("the loaded Isaac DiT did not enable math SDPA")
    if torch.cuda.device_count() != int(os.environ["GPUS_PER_NODE"]):
        raise ValueError("visible GPUs differ from the actual allocation")
    if not (Path(args.checkpoint) / "config.json").is_file():
        raise ValueError("checkpoint config.json is missing")
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": args.checkpoint,
        "sdpa_mode": os.environ["GR00T_DIT_SDPA_MODE"],
        "sdpa_effective": True,
        **{package: importlib.metadata.version(package) for package in ("transformers", "vllm", "vllm-omni")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=4) + "\n")


if __name__ == "__main__":
    main()
