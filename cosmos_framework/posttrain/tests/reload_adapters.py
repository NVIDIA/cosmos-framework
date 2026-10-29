# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fresh-process portable/merged inference checks for qualify_adapters outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.posttrain.checkpoint.adapters import read_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy, build_qwen3_vl_model
from cosmos_framework.posttrain.tests.qualify_adapters import attest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--merged")
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--build-record", required=True)
    args = parser.parse_args()
    evidence = attest(args.wheel, args.build_record)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run)
    identity, _ = read_adapter_artifact(str(root / "portable"))
    if args.merged:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.merged, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()
        policy = Qwen3VLPolicy(model, "qwen3_vl")
    else:
        components = [
            AdapterComponentConfig(component=row["component"], targets=[row["name"]], **row["settings"])
            for row in identity["resolved"]["targets"]
        ]
        extras = [
            AdapterExtraModuleConfig(
                component=row["component"],
                targets=[row["name"].rsplit(".", 1)[0]],
                lr_multiplier=row["lr_multiplier"],
                weight_decay=row["weight_decay"],
            )
            for row in identity["resolved"]["extra_parameters"]
            if row["owner"] == row["name"]
        ]
        config = Qwen3VLModelConfig(
            name_or_path=args.base, dtype="bfloat16", attn_impl="sdpa", activation_checkpointing="none"
        )
        policy = build_qwen3_vl_model(
            config,
            adapters=AdapterConfig(components=components, extra_modules=extras, warm_start=str(root / "portable")),
            adapter_media_identity=identity["media"],
        )
    policy.net.eval()
    maximum, absolute_sum, count, argmax_equal, positions = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for path in sorted(root.glob("probes-rank*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            for batch, expected in zip(record["batches"], record["logits"]):
                actual = policy.forward_policy_batch(policy.prepare_policy_batch(batch)).logits[:, -8:].float().cpu()
                error = (actual - expected).abs()
                maximum = max(maximum, error.max().item())
                absolute_sum += error.sum().item()
                count += error.numel()
                argmax_equal += int((actual.argmax(-1) == expected.argmax(-1)).sum())
                positions += actual.shape[0] * actual.shape[1]
                # BF16 merged weights round the completed FP32 direction once.
                # Portable reload has no such rounding and uses a tighter bound.
                torch.testing.assert_close(
                    actual, expected, atol=0.05 if args.merged else 2e-5, rtol=0.02 if args.merged else 2e-5
                )
    mean = absolute_sum / count
    assert mean <= (0.005 if args.merged else 2e-6), mean
    assert argmax_equal == positions, (argmax_equal, positions)
    mode = "merged" if args.merged else "portable"
    report = {
        **evidence,
        "mode": mode,
        "maximum_logit_error": maximum,
        "mean_absolute_logit_error": mean,
        "argmax_equal": argmax_equal,
        "positions": positions,
        "status": "passed",
    }
    (root / f"{mode}-reload.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
