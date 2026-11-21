# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export the effective native adapter model as a new dense HF artifact.

Launch with torchrun --standalone --nproc-per-node=1 in the pinned native runtime.
Supply the original model/processor overrides and a committed DCP or portable
adapter source. This operation does not train or relax exact-resume identity.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from contextlib import suppress


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--output", required=True, help="New local directory; existing outputs are refused")
    result.add_argument("--output-dtype", choices=("bfloat16", "float32"), required=True)
    result.add_argument(
        "--max-shard-size-bytes",
        type=int,
        default=512 * 1024**2,
        help="Maximum tensor payload per HF shard (default: 512 MiB); an oversized tensor is stored alone",
    )
    result.add_argument("--override", action="append", default=[])
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.max_shard_size_bytes <= 0:
        raise ValueError("--max-shard-size-bytes must be positive")
    if os.environ.get("WORLD_SIZE") != "1" or os.environ.get("RANK") != "0":
        raise ValueError("dense native export requires torchrun with one process")
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.dense_adapter import export_dense_adapter
    from cosmos_framework.posttrain.config.build import build_backbone
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config
    from cosmos_framework.posttrain.utils.adapter_identity import adapter_processor_identity
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized, init_local_process_group
    from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.config import load_config

    read_adapter_metadata(args.checkpoint)  # Fail before allocating the base.
    config = load_config(args.config, ["--", *args.override])
    rl = validate_config(config)
    enable_determinism(strict=True)
    init_local_process_group()
    try:
        misc.set_random_seed(rl.seed, by_rank=False)
        source, trust = sft_policy_source(config)
        packer = build_sft_packer(
            source,
            max_length=rl.max_length,
            processor_kwargs=rl.processor_kwargs,
            trust_remote_code=trust,
            mask_think_opener=rl.mask_think_opener,
        )
        model = build_backbone(config)
        codec = model.net._cosmos_adapter_checkpoint_codec
        codec.bind_processor_identity(
            adapter_processor_identity(packer, source_path=source, source_identity=codec.identity["base"])
        )
        record_qlora_phase("dense_export_start", reset_peak=True)
        try:
            result = export_dense_adapter(
                model.net,
                checkpoint=args.checkpoint,
                source=source,
                output=args.output,
                output_dtype=args.output_dtype,
                max_shard_size_bytes=args.max_shard_size_bytes,
                exporter_build={
                    "code_revision": os.environ.get("COSMOS_TRAINING_CODE_REVISION", ""),
                    "wheel_sha256": os.environ.get("COSMOS_TRAINING_WHEEL_SHA256", ""),
                },
            )
        except BaseException:
            # A diagnostic write or CUDA synchronization failure must not hide
            # the original export failure (including interruption).
            with suppress(Exception):
                record_qlora_phase("dense_export_failed")
            raise
        record_qlora_phase("dense_export_complete")
        print(
            json.dumps(
                {
                    "output": args.output,
                    "complete": result["complete"],
                    "output_dtype": result["output_dtype"],
                    "tensor_count": len(result["tensors"]),
                },
                sort_keys=True,
            )
        )
    finally:
        destroy_process_group_if_initialized()


if __name__ == "__main__":
    main()
