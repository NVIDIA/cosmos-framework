# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare immutable residual LoRA factors for the native frozen NF4 base.

This explicit, single-device calibration operation never trains a model. The
training builder reads the resulting factors; it cannot rerun their decomposition.
Launch with torchrun --standalone --nproc-per-node=1 in the pinned QLoRA runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Verified local pinned BF16 source snapshot")
    parser.add_argument("--output", required=True, help="New directory for factors and commit-last manifest.json")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(name not in os.environ for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE")):
        parser.error("launch with torchrun --standalone --nproc-per-node=1")
    if os.environ["WORLD_SIZE"] != "1" or os.environ["RANK"] != "0":
        parser.error("native residual initialization supports exactly one process")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("export CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching torchrun")
    for name, size in (("COSMOS_TRAINING_CODE_REVISION", 40), ("COSMOS_TRAINING_WHEEL_SHA256", 64)):
        if re.fullmatch(r"[0-9a-f]{%d}" % size, os.environ.get(name, "")) is None:
            parser.error(f"set {name} from the installed native build receipt before preparing factors")

    from transformers import AutoConfig

    from cosmos_framework.posttrain.config.model import (
        NativeLoRAConfig,
        QuantizedBaseTrainingConfig,
        Qwen3VLModelConfig,
    )
    from cosmos_framework.posttrain.model.quantization.initialization import prepare_initialization_artifact
    from cosmos_framework.posttrain.utils.adapter_source import SOURCE_REPOSITORY, SOURCE_REVISION
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized, init_local_process_group
    from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase

    enable_determinism(strict=True)
    init_local_process_group()
    try:
        cfg = Qwen3VLModelConfig(
            name_or_path=str(Path(args.model).absolute()),
            revision=SOURCE_REVISION,
            attn_impl="sdpa",
            adapter=NativeLoRAConfig(rank=args.rank, alpha=args.alpha, initialization_seed=args.seed),
            quantized_base=QuantizedBaseTrainingConfig(),
            source_identity={"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        )
        model_config = AutoConfig.from_pretrained(cfg.name_or_path, trust_remote_code=False, local_files_only=True)
        started = time.monotonic()
        record_qlora_phase("residual_initialization_start", reset_peak=True)
        result = prepare_initialization_artifact(cfg, Path(args.output).absolute(), model_config=model_config)
        record_qlora_phase("residual_initialization_complete", reset_peak=False)
        print(json.dumps({"artifact": result, "elapsed_seconds": time.monotonic() - started}, sort_keys=True))
    finally:
        destroy_process_group_if_initialized()


if __name__ == "__main__":
    main()
