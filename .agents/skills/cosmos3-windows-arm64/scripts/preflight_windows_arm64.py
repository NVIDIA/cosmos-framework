#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Probe the native ARM64 CUDA runtime without loading a Cosmos model.

The JSON report goes to stdout (and to --output if given); progress lines go to stderr.
Budget ten minutes for a first run with --allocation-gib 72: the first torch import and CUDA
initialisation take a minute or more on a cold machine, and the allocation probe writes every
page (about 1.5 s per GiB on an RTX Spark, so about two minutes for 72 GiB).
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from typing import Any


def gib(value: int) -> float:
    return round(value / 2**30, 2)


def progress(message: str) -> None:
    print(f"preflight: {message}", file=sys.stderr, flush=True)


def emit(report: dict[str, Any], output: str | None) -> None:
    text = json.dumps(report, indent=2)
    print(text)
    if output:
        with open(output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate native Windows-on-ARM Torch and optionally probe CUDA capacity."
    )
    parser.add_argument(
        "--allocation-gib",
        type=int,
        default=0,
        help="Allocate, write, synchronize, and release this many GiB of CUDA memory "
        "(about 1.5 s per GiB on an RTX Spark; run it on an idle GPU).",
    )
    parser.add_argument("--output", default=None, help="Also write the JSON report to this file.")
    args = parser.parse_args()
    if args.allocation_gib < 0:
        parser.error("--allocation-gib must be zero or positive")

    report: dict[str, Any] = {
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "status": "FAIL",
        "checks": [],
    }
    if platform.machine().upper() != "ARM64":
        report["checks"].append("The selected interpreter is not native ARM64.")
        emit(report, args.output)
        return 1

    progress("importing torch (a first import on a cold machine can take a minute or more)")
    started = time.perf_counter()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - structured hardware preflight output
        report["checks"].append(f"Torch import failed: {type(exc).__name__}: {exc}")
        emit(report, args.output)
        return 1

    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["torch_import_seconds"] = round(time.perf_counter() - started, 1)
    progress("initialising CUDA")
    started = time.perf_counter()
    if not torch.cuda.is_available():
        report["checks"].append("torch.cuda.is_available() is false")
        emit(report, args.output)
        return 1

    device = 0
    free_before, total = torch.cuda.mem_get_info(device)
    report.update(
        {
            "device": torch.cuda.get_device_name(device),
            "cuda_total_gib": gib(total),
            "cuda_free_before_gib": gib(free_before),
            "cuda_init_seconds": round(time.perf_counter() - started, 1),
        }
    )
    if args.allocation_gib:
        buffer = None
        progress(
            f"allocating and writing {args.allocation_gib} GiB; expect about "
            f"{max(10, round(args.allocation_gib * 1.5))} s, do not interrupt"
        )
        started = time.perf_counter()
        try:
            buffer = torch.empty(args.allocation_gib * 2**30, dtype=torch.uint8, device=device)
            buffer.zero_()
            torch.cuda.synchronize(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            report["allocation_probe"] = {
                "gib": args.allocation_gib,
                "result": "PASS",
                "cuda_free_after_gib": gib(free_after),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            }
        except Exception as exc:  # pragma: no cover - hardware-dependent result
            report["allocation_probe"] = {
                "gib": args.allocation_gib,
                "result": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            emit(report, args.output)
            return 1
        finally:
            progress("releasing the buffer")
            del buffer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    report["checks"].append("Native ARM64 Torch CUDA probe passed.")
    report["status"] = "PASS"
    emit(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
