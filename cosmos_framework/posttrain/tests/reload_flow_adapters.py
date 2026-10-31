# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fresh-process numerical gates for native physical-action adapter artifacts."""

import argparse
import gc
from pathlib import Path

import torch

from cosmos_framework.posttrain.checkpoint.adapters import adapter_configuration, read_adapter_artifact
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.tests.adapter_numerics import configure_qualification_numerics
from cosmos_framework.posttrain.tests.qualify_adapters import attest, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "base", "processor", "wheel", "build-record"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--merged")
    args = parser.parse_args()
    evidence = attest(args.wheel, args.build_record)
    evidence["numerical_profile"] = configure_qualification_numerics()
    root = Path(args.run)
    identity, _ = read_adapter_artifact(str(root / "portable"))
    cfg = adapter_configuration(identity)
    cfg.warm_start = str(root / "portable")
    safe_types = []
    if identity["architecture"] == "pi05":
        from cosmos_framework.posttrain.model.pi05.adapter_export import load_merged
        from cosmos_framework.posttrain.model.pi05.processing import Pi05ReplayInputs
        from cosmos_framework.posttrain.model.pi05.sft import Pi05SFTModelConfig, build_pi05_sft_policy

        safe_types = [Pi05ReplayInputs]
        model_config = Pi05SFTModelConfig(
            str(Path(args.processor) / "tokenizer.model"), str(Path(args.processor) / "norm_stats.json"), "none"
        )
        policy = build_pi05_sft_policy(args.base, model_config, adapters=cfg, adapter_media_identity=identity["media"])
    elif identity["architecture"] == "alpamayo_r1":
        from cosmos_framework.posttrain.model.alpamayo.adapter_export import load_merged, model_configuration
        from cosmos_framework.posttrain.model.alpamayo.policy import build_policy

        policy = build_policy(
            args.base,
            model_configuration(identity, args.processor),
            adapters=cfg,
            adapter_media_identity=identity["media"],
        )
    elif identity["architecture"] == "Gr00tN1d7":
        from cosmos_framework.posttrain.model.gr00t.adapter_export import load_merged
        from cosmos_framework.posttrain.model.gr00t.sft import Gr00tSFTModelConfig
        from cosmos_framework.posttrain.model.gr00t.sft_policy import build_policy

        policy = build_policy(
            args.base,
            Gr00tSFTModelConfig(args.processor, "none"),
            adapters=cfg,
            adapter_media_identity=identity["media"],
        )
    else:
        raise ValueError(f"no flow reload qualification binding for {identity['architecture']}")
    batches, references = [], []
    for path in sorted(root.glob("probes-rank*.pt")):
        with torch.serialization.safe_globals(safe_types):
            record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("numerical_profile", evidence["numerical_profile"]) != evidence["numerical_profile"]:
            raise ValueError("flow probe numerical profile differs; use the recorded qualification kernel settings")
        if record["identity"] != identity:
            raise ValueError("flow probe identity differs from its portable adapter")
        batches.extend(record["batches"])
        references.extend(record["logits"])
    if not batches:
        raise ValueError("flow reload qualification requires saved controlled probes")
    native = logits(policy, batches)
    for a, b in zip(native, references, strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    if args.merged:
        import json

        metadata = json.loads((Path(args.merged) / "merged_model.json").read_text())
        if metadata["adapter_identity"] != identity:
            raise ValueError("flow merged artifact has a different adapter identity")
        policy.net.float()
        if identity["architecture"] == "Gr00tN1d7":
            policy.autocast = False
        expected = logits(policy, batches)
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        policy = load_merged(args.merged, device="cuda")
        actual = logits(policy, batches)
    else:
        actual, expected = native, references
    errors = [(a - b).abs() for a, b in zip(actual, expected, strict=True)]
    record = {
        **evidence,
        "mode": "merged" if args.merged else "portable",
        "maximum_absolute_output_error": max(x.max().item() for x in errors),
        "mean_absolute_output_error": torch.cat(errors).mean().item(),
        "reference_precision": "float32" if args.merged else identity["base_storage_dtype"],
        "status": "measured_awaiting_gate",
    }
    report = root / ("merged-reload.json" if args.merged else "portable-reload.json")
    atomic_write_json(str(report), record)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=5e-4 if args.merged else 2e-5, rtol=2e-4 if args.merged else 2e-5)
    if record["mean_absolute_output_error"] > (5e-5 if args.merged else 2e-6):
        raise AssertionError(record)
    record["status"] = "passed"
    atomic_write_json(str(report), record)


if __name__ == "__main__":
    main()
