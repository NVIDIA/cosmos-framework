# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""MoE export oracle: native BF16 reload, stock HF FP32 inference over multiple GPUs.

The native grouped expert kernel cannot execute FP32. Keep its portable reload
as a separate mandatory gate; compare the promoted adapter and serialized merge
in stock HF, whose loader owns the checkpoint expert transpose. No expert adapter
or alternate training path is introduced.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

from cosmos_framework.posttrain.checkpoint.adapters import adapter_configuration, read_adapter_artifact
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.checkpoint.tools.adapters import validate_merged_export_metadata
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, prepare_adapter_model
from cosmos_framework.posttrain.tests.adapter_numerics import (
    HFProbePolicy,
    assert_bf16_deployment_drift,
    compare_logits,
    configure_qualification_numerics,
)
from cosmos_framework.posttrain.tests.qualify_adapters import attest, logits


def hf_portable(base, adapter, identity):
    if identity["architecture"] != "qwen3_vl_moe" or identity["base_storage_dtype"] != "bfloat16":
        raise ValueError("MoE export oracle requires native Qwen3-VL-MoE BF16 training identity")
    net = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    config = adapter_configuration(identity)
    config.warm_start = adapter
    prepare_adapter_model(
        net, config, base=base, processor=base, base_dtype="bfloat16", media=identity["media"], family="qwen3_vl_moe"
    )
    if net._posttrain_adapter_identity != identity:
        raise ValueError("stock HF MoE resolver differs from the native portable target contract")
    return net.eval()


def place_for_inference(net, *, max_memory_gib):
    from accelerate import dispatch_model, infer_auto_device_map

    if max_memory_gib <= 0 or torch.cuda.device_count() < 2:
        raise ValueError("30B FP32 MoE qualification requires at least two GPUs and a positive per-GPU memory budget")
    device_map = infer_auto_device_map(
        net,
        max_memory={i: int(max_memory_gib * 1024**3) for i in range(torch.cuda.device_count())},
        no_split_module_classes=[*net._no_split_modules, "PosttrainAdapterLinear"],
        dtype=next(net.parameters()).dtype,
    )
    if any(device in ("cpu", "disk") for device in device_map.values()):
        raise ValueError(
            f"MoE FP32 model does not fit declared GPU budgets; no implicit CPU/disk offload: {device_map}"
        )
    return dispatch_model(net, device_map=device_map).eval(), device_map


def verify_serialized(base, merged, identity, adapter_state):
    """Verify every output tensor against original rounded base + shared merge."""
    base, merged = Path(base), Path(merged)
    index = json.loads((merged / "model.safetensors.index.json").read_text())["weight_map"]
    base_index = base / "model.safetensors.index.json"
    if base_index.exists():
        source_index = json.loads(base_index.read_text())["weight_map"]
    else:
        with safe_open(str(base / "model.safetensors"), framework="pt") as handle:
            source_index = {n: "model.safetensors" for n in handle.keys()}
    if set(index) != set(source_index):
        raise ValueError("MoE merged tensor inventory differs from its base")
    targets = {row["name"] + ".weight": row for row in identity["resolved"]["targets"]}
    extras = {row["name"] for row in identity["resolved"]["extra_parameters"]}
    verified = 0
    for source_shard in sorted(set(source_index.values())):
        with safe_open(str(base / source_shard), framework="pt") as source:
            for name in source.keys():
                expected = source.get_tensor(name).to(torch.bfloat16).float()
                if name in extras:
                    expected = adapter_state[name].float()
                elif name in targets:
                    row = targets[name]
                    if row["owner"] != row["name"]:
                        raise ValueError("MoE stock export oracle does not support custom target aliases")
                    module = torch.nn.Linear(
                        row["shape"][1], row["shape"][0], bias=False, device="meta", dtype=torch.float32
                    )
                    layer = PosttrainAdapterLinear(
                        module,
                        AdapterComponentConfig(component=row["component"], targets=[row["name"]], **row["settings"]),
                        initialize=False,
                    )
                    expected = layer.merged_state_weight({name: expected, **adapter_state}, row["name"])
                with safe_open(str(merged / index[name]), framework="pt") as target:
                    actual = target.get_tensor(name)
                    if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                        raise ValueError(f"MoE serialized merged tensor differs: {name}")
                verified += 1
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "base", "merged", "wheel", "build-record"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-memory-gib", type=float, default=70)
    args = parser.parse_args()
    evidence = attest(args.wheel, args.build_record)
    evidence["numerical_profile"] = configure_qualification_numerics()
    root = Path(args.run)
    identity, adapter_state = read_adapter_artifact(str(root / "portable"))
    native_reload = json.loads((root / "portable-reload.json").read_text())
    if native_reload["status"] != "passed":
        raise ValueError("native BF16 fresh-process reload must pass before the MoE FP32 export gate")
    marker = json.loads((Path(args.merged) / ".materialized").read_text())
    if marker["identity"].get("output_dtype") != "float32":
        raise ValueError("MoE merged oracle requires the FP32 exported artifact")
    validate_merged_export_metadata(marker["identity"].get("export_metadata"), str(root / "portable"), identity)
    verified = verify_serialized(args.base, args.merged, identity, adapter_state)
    batches, native = [], []
    for path in sorted(root.glob("probes-rank*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("numerical_profile", evidence["numerical_profile"]) != evidence["numerical_profile"]:
            raise ValueError("MoE probe numerical profile differs; use the recorded qualification kernel settings")
        if record["identity"] != identity:
            raise ValueError("MoE controlled probe identity mismatch")
        batches.extend(record["batches"])
        native.extend(record["logits"])
    if not batches:
        raise ValueError("MoE oracle requires controlled text/image/video training probes")
    # Stock BF16/native BF16 comparison is reported separately from FP32 merge
    # equivalence, using the existing probability-aware BF16 drift limits.
    model = hf_portable(args.base, str(root / "portable"), identity).cuda().eval()
    stock_bf16 = logits(HFProbePolicy(model), batches)
    bf16_drift = compare_logits(stock_bf16, native)
    assert_bf16_deployment_drift(bf16_drift)
    model.cpu().float()
    model, adapter_map = place_for_inference(model, max_memory_gib=args.max_memory_gib)
    expected = logits(HFProbePolicy(model), batches)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(
        args.merged, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
    model, merged_map = place_for_inference(model, max_memory_gib=args.max_memory_gib)
    actual = logits(HFProbePolicy(model), batches)
    report = {
        **evidence,
        "verification_protocol": "cosmos-moe-adapter-numerics-v1",
        "mode": "merged",
        "serialized_tensors_verified": verified,
        "native_bf16_vs_stock_bf16": bf16_drift,
        "reference_precision": "float32-stock-hf",
        "adapter_device_map": adapter_map,
        "merged_device_map": merged_map,
        **compare_logits(actual, expected),
        "status": "measured_awaiting_gate",
    }
    path = root / "merged-reload.json"
    atomic_write_json(str(path), report)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=5e-4, rtol=2e-4)
    if report["mean_absolute_logit_error"] > 5e-5 or report["argmax_equal"] != report["positions"]:
        raise AssertionError(report)
    report["status"] = "passed"
    atomic_write_json(str(path), report)


if __name__ == "__main__":
    main()
