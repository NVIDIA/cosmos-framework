# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export a trained native action/driving policy's effective base and complete task heads.

Use torchrun --standalone --nproc-per-node=1 in the pinned family runtime.
This creates a native policy artifact; use that family's dense inference loader.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--output", required=True)
    representation = result.add_mutually_exclusive_group(required=True)
    representation.add_argument("--merged-backbone-dtype", choices=("bfloat16", "float32"))
    representation.add_argument(
        "--unmerged",
        action="store_true",
        help="Export actual BF16 effective bases plus separate FP32 native LoRA factors; inference only",
    )
    result.add_argument("--max-shard-size-bytes", type=int, default=512 * 1024**2)
    result.add_argument("--override", action="append", default=[])
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.max_shard_size_bytes <= 0:
        raise ValueError("--max-shard-size-bytes must be positive")
    if os.environ.get("WORLD_SIZE") != "1" or os.environ.get("RANK") != "0":
        raise ValueError("native family dense export requires torchrun with one process")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    import tempfile

    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.family_dense import export_family_dense, export_family_unmerged
    from cosmos_framework.posttrain.entrypoints.action_sft import _factory, validate_config
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized, init_local_process_group
    from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.config import load_config

    read_adapter_metadata(args.checkpoint)
    config = load_config(args.config, ["--", *args.override])
    rl = validate_config(config)
    enable_determinism(strict=True)
    init_local_process_group()
    try:
        misc.set_random_seed(rl.seed, by_rank=False)
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        sidecars_factory = getattr(config, "dense_sidecars_factory", "")
        if not sidecars_factory:
            raise ValueError("native family dense export requires its family sidecar factory")
        sidecars = _factory(sidecars_factory)(config)
        record_qlora_phase("family_dense_loading_begin", reset_peak=True)
        with tempfile.TemporaryDirectory(prefix="cosmos-family-dense-config-") as temporary:
            policy = _factory(config.policy_factory)(config, device=device, output=temporary)
            record_qlora_phase("family_dense_loading_complete", reset_peak=True)
            try:
                export = export_family_unmerged if args.unmerged else export_family_dense
                representation = {} if args.unmerged else {"merged_backbone_dtype": args.merged_backbone_dtype}
                result = export(
                    policy.net,
                    checkpoint=args.checkpoint,
                    output=args.output,
                    **representation,
                    max_shard_size_bytes=args.max_shard_size_bytes,
                    sidecars=sidecars,
                    exporter_build={
                        "code_revision": os.environ.get("COSMOS_TRAINING_CODE_REVISION", ""),
                        "wheel_sha256": os.environ.get("COSMOS_TRAINING_WHEEL_SHA256", ""),
                    },
                )
            except BaseException:
                with suppress(Exception):
                    record_qlora_phase("family_dense_export_failed")
                raise
            record_qlora_phase("family_dense_export_complete")
        print(
            json.dumps(
                {
                    "output": args.output,
                    "complete": result["complete"],
                    "tensors": len(result["tensors"]),
                    "sidecars": result["sidecars"],
                    "numerical_contract": result["numerical_contract"],
                },
                sort_keys=True,
            )
        )
    finally:
        destroy_process_group_if_initialized()


if __name__ == "__main__":
    main()
