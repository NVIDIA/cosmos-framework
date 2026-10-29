# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fresh-process portable/merged inference checks for qualify_adapters outputs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from cosmos_framework.posttrain.checkpoint.adapters import read_adapter_artifact
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy, build_qwen3_vl_model
from cosmos_framework.posttrain.tests.adapter_numerics import (
    compare_logits,
    verify_promoted_merge,
)
from cosmos_framework.posttrain.tests.qualify_adapters import attest, logits


def portable_policy(base: str, directory: str, identity: dict):
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
        name_or_path=base, dtype=identity["base_storage_dtype"], attn_impl="sdpa", activation_checkpointing="none"
    )
    return build_qwen3_vl_model(
        config,
        adapters=AdapterConfig(components=components, extra_modules=extras, warm_start=directory),
        adapter_media_identity=identity["media"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--merged")
    parser.add_argument("--probe-run", help="reuse input probes for a different trained adapter; merged checks only")
    parser.add_argument("--report-dir", help="write new evidence separately from immutable training outputs")
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--build-record", required=True)
    args = parser.parse_args()
    if args.probe_run and not args.merged:
        parser.error("--probe-run requires --merged; portable exact reload needs its own saved reference outputs")
    evidence = attest(args.wheel, args.build_record)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run)
    identity, _ = read_adapter_artifact(str(root / "portable"))
    source = json.loads((root / "portable/adapter_manifest.json").read_text())["provenance"]
    batches, references = [], []
    for path in sorted(Path(args.probe_run or root).glob("probes-rank*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("base", "processor", "media", "base_storage_dtype"):
            if record["identity"][key] != identity[key]:
                raise ValueError(f"probe input compatibility differs in {key}; use matching media/base inputs")
        batches.extend(record["batches"])
        references.extend(record["logits"])
    if not batches:
        raise ValueError("no controlled probe inputs found")
    policy = portable_policy(args.base, str(root / "portable"), identity)
    native = logits(policy, batches)
    if not args.probe_run:
        for value, expected in zip(native, references):
            torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-5)
    control = None
    if args.merged:
        from transformers import AutoModelForImageTextToText

        marker = json.loads((Path(args.merged) / ".materialized").read_text())
        if marker["identity"].get("output_dtype") != "float32":
            raise ValueError("merged qualification requires FP32 export; re-export with the current adapter tool")
        policy.net.float()
        expected = logits(policy, batches)
        control = verify_promoted_merge(policy, identity, batches, args.merged, reference=expected)
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(
            args.merged, dtype="auto", attn_implementation="sdpa"
        ).cuda()
        assert all(p.dtype == torch.float32 for p in model.parameters())
        actual = logits(Qwen3VLPolicy(model, "qwen3_vl"), batches)
    else:
        actual, expected = native, references
    mode = "merged" if args.merged else "portable"
    report = {
        **evidence,
        "verification_protocol": "cosmos-adapter-numerics-v3-fp32-export",
        "training_artifact_provenance": source,
        "probe_run": args.probe_run or args.run,
        "mode": mode,
        **compare_logits(actual, expected),
        "fp32_merge_control": control,
        "reference_precision": "float32" if args.merged else identity["base_storage_dtype"],
        "versus_training_precision": compare_logits(actual, native) if args.merged else None,
        "status": "measured_awaiting_drift_gate",
    }
    output = Path(args.report_dir or root)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(output / f"{mode}-reload.json"), report)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(
            value, reference, atol=5e-4 if args.merged else 2e-5, rtol=2e-4 if args.merged else 2e-5
        )
    assert report["mean_absolute_logit_error"] <= (5e-5 if args.merged else 2e-6), report
    assert report["argmax_equal"] == report["positions"], report
    report["status"] = "passed"
    atomic_write_json(str(output / f"{mode}-reload.json"), report)


if __name__ == "__main__":
    main()
