# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stage 1's regression entry for the GR00T serving gate: version 0, an engine of its own.

The gate itself (three layers between the vLLM-Omni port and the Isaac-GR00T reference, plus layer 3 through the
engine) lives in ``tests/helpers/gr00t_parity_gate.py``; ``tests/gr00t_weight_sync_cli.py`` runs the same gate
against a HOT engine at versions 1, 2, ... This file only parses arguments, builds and closes its own body, and
applies the gate's rule (``operator GR00T validation documentation``): every cross-implementation row bitwise equal, the single
tolerance row being the serving side's own batch-invariance check. Without ``--assert`` it reports and exits 0.

Run on one GB200 in the gr00t venv; both implementations import into one process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from cosmos_framework.posttrain.tests.helpers.gr00t_parity_gate import TOLERANCES, evaluate_gate, render, run_gate

__all__ = ["TOLERANCES", "main", "run_parity"]


def run_parity(checkpoint: str, batch: int, seed: int, with_engine: bool) -> dict:
    """Version-0 gate: the reference and the port load ``checkpoint``; ``with_engine`` adds a fresh body's rows."""
    from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config

    config = make_model_config(checkpoint_path=checkpoint)
    if not with_engine:
        return run_gate(checkpoint=checkpoint, config=config, batch=batch, seed=seed)

    from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody

    body = Gr00tPolicyBody(config)
    try:
        return run_gate(checkpoint=checkpoint, config=config, batch=batch, seed=seed, body=body, weight_version=0)
    finally:
        body.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("GR00T_CKPT", "nvidia/GR00T-N1.7-DROID"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--with-engine", action="store_true", help="also compare through Gr00tPolicyBody.infer_batch")
    parser.add_argument("--json", default="", help="write the raw report here")
    parser.add_argument(
        "--assert",
        dest="do_assert",
        action="store_true",
        help="fail unless every parity row is bitwise equal and the tolerance row is within TOLERANCES",
    )
    args = parser.parse_args(argv)

    report = run_parity(args.checkpoint, args.batch, args.seed, args.with_engine)
    print(render(report))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)

    if not args.do_assert:
        return 0
    ok, failed, exact_failed = evaluate_gate(report)
    for d in failed + exact_failed:
        print(f"FAIL {d['name']}: max_abs={d['max_abs']:.3e}")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
