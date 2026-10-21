# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Capture one real window, then qualify fused FSDP2 replay/save/fresh-process resume.

capture uses an existing Ray cluster; continuous/save/resume require independent
8-rank torchrun invocations. compare is CPU-only. Keep publication0 throughout:
these are two optimizer passes over an immutable behavior0 window, not a second
on-policy collection. Formal training must start as its own fresh driver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("capture", "continuous", "save", "resume", "compare"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--records-root", type=Path, help="Read-only real capture directory; defaults to OUTPUT/records."
    )
    parser.add_argument(
        "--export-records", type=Path, help="Continuous phase only: export compact prepared16 before updates."
    )
    args = parser.parse_args(argv)
    if args.export_records is not None and args.phase != "continuous":
        parser.error("--export-records requires --phase continuous")
    if args.records_root is not None and args.phase in ("capture", "compare"):
        parser.error("--records-root applies only to continuous/save/resume")
    from cosmos_framework.posttrain.tests.helpers.pi05_fused_replay import (
        capture,
        compare_replays,
        qualification_config,
        replay_phase,
    )

    if args.phase == "compare":
        report = compare_replays(args.output)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    config, recipe = qualification_config(args.config, args.output)
    if args.phase == "capture":
        report = capture(config, recipe, args.output)
        print(json.dumps({"phase": "capture", "passed": report["passed"], "records": report["records"]}), flush=True)
    else:
        print(
            json.dumps(replay_phase(config, recipe, args.output, args.phase, args.records_root, args.export_records)),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
