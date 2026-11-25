# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export adapters from a committed adapter-SFT checkpoint.

Portable adapter (no optimizer, scheduler, RNG or data cursor; used for warm starts)::

    python -m cosmos_framework.posttrain.checkpoint.tools.export_adapters \\
        --checkpoint <run>/checkpoints/iter_000100 --out <adapter_dir>

Merged Hugging Face model (adapters folded into the base weights; loads without adapter code)::

    python -m cosmos_framework.posttrain.checkpoint.tools.export_adapters \\
        --checkpoint <iter_dir> | --adapter <adapter_dir> --merged --base <hf_snapshot> --out <merged_dir> \\
        [--dtype float32|bfloat16]

``--base`` must be the local snapshot the adapters were trained on (its target weights must reproduce the recorded
base fingerprint unless ``--allow-base-mismatch``). ``--dtype`` defaults to the training storage
dtype of the base, so a bfloat16 model keeps its size; rounding to bfloat16 loses part of the small adapter update,
and ``--dtype float32`` keeps it exactly. Destinations must not exist.
"""

from __future__ import annotations

import argparse
import os
import tempfile

from cosmos_framework.posttrain.checkpoint.adapters import checkpoint_adapter_state
from cosmos_framework.utils.generator.adapter_artifacts import export_merged_adapter, write_adapter_artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="committed training checkpoint directory (iter_*)")
    source.add_argument("--adapter", help="portable adapter directory (with --merged)")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--merged", action="store_true", help="write a merged Hugging Face model instead")
    parser.add_argument("--base", help="local Hugging Face base snapshot (required with --merged)")
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="merge even if --base does not reproduce the base fingerprint recorded with the adapter",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default=None,
        help="merged weight dtype (default: training base dtype)",
    )
    args = parser.parse_args(argv)
    if args.merged != bool(args.base):
        parser.error("--merged and --base go together")
    if args.adapter and not args.merged:
        parser.error("--adapter is only used with --merged")
    if (args.dtype or args.allow_base_mismatch) and not args.merged:
        parser.error("--dtype and --allow-base-mismatch apply to --merged exports")
    if not args.merged:
        write_adapter_artifact(args.out, *checkpoint_adapter_state(args.checkpoint))
        return
    if args.adapter:
        export_merged_adapter(
            args.adapter, args.base, args.out, dtype=args.dtype, allow_base_mismatch=args.allow_base_mismatch
        )
        return
    with tempfile.TemporaryDirectory(prefix="cosmos-adapter-") as temporary:
        adapter = os.path.join(temporary, "adapter")
        write_adapter_artifact(adapter, *checkpoint_adapter_state(args.checkpoint))
        export_merged_adapter(
            adapter, args.base, args.out, dtype=args.dtype, allow_base_mismatch=args.allow_base_mismatch
        )


if __name__ == "__main__":
    main()
