# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run isolated RLinf OpenPI and posttrain GPU numerical probes on one immutable request.

Use the RLinf interpreter/PYTHONPATH for ``reference`` and the posttrain environment
for ``posttrain``. ``prepare --observations`` accepts NPZ image/wrist_image uint8
[B,H,W,3], state [B,8], and prompt Unicode [B]. Without it, images are synthetic
and the result only qualifies numerical parity, never simulator performance.
The common checkpoint must include its initialized value head. Comparison is
exact until independent measurements justify an explicit tolerance contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    prepare = phases.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--observations", type=Path)
    prepare.add_argument("--batch", type=int, default=1)
    prepare.add_argument("--seed", type=int, default=1234)
    for phase in ("reference", "posttrain"):
        run = phases.add_parser(phase)
        for name in ("request", "checkpoint", "tokenizer", "norm-stats", "output"):
            run.add_argument(f"--{name}", type=Path, required=True)
        if phase == "posttrain":
            run.add_argument("--reference-trace", type=Path)
    compare = phases.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--posttrain", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch

    from cosmos_framework.posttrain.tests.helpers.pi05_parity import (
        compare_records,
        prepare_request,
        read_request,
        run_posttrain,
        run_reference,
    )

    if args.phase == "prepare":
        if args.batch < 1 or args.seed < 0:
            parser.error("--batch must be positive and --seed nonnegative")
        prepare_request(args.output, batch=args.batch, seed=args.seed, observations=args.observations)
        print(json.dumps({"request": str(args.output), "seed": args.seed}))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "compare":
        report = compare_records(
            torch.load(args.reference, map_location="cpu", weights_only=True),
            torch.load(args.posttrain, map_location="cpu", weights_only=True),
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        failed = [row for row in report["rows"] if not row["exact"]]
        print(json.dumps({"passed": report["passed"], "tensors": len(report["rows"]), "failed": len(failed)}))
        for row in failed[:20]:
            print(json.dumps(row))
        return 0 if report["passed"] else 1
    if not torch.cuda.is_available():
        raise RuntimeError(f"{args.phase} requires a real CUDA GPU")
    torch.set_float32_matmul_precision("high")
    request = read_request(args.request)
    result = (run_reference if args.phase == "reference" else run_posttrain)(args, request)
    torch.save(result, args.output)
    print(json.dumps({"phase": args.phase, "output": str(args.output), "tensors": len(result["tensors"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
