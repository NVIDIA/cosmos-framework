# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Separate exact adapter state/FP32 merge checks from BF16 deployment drift.

These are qualification helpers, not a second export or inference implementation.
BF16 low-rank and dense GEMMs round at different boundaries. A raw absolute-logit
threshold alone both obscures this effect and ignores its probability impact.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch


def compare_logits(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> dict:
    if not actual or len(actual) != len(expected):
        raise ValueError("logit comparison requires the same non-empty probe population")
    rows = []
    for value, reference in zip(actual, expected):
        if value.shape != reference.shape or not torch.isfinite(value).all() or not torch.isfinite(reference).all():
            raise ValueError("logit comparison requires matching shapes and finite values")
        value, reference = value.float(), reference.float()
        delta = (value - reference).abs()
        log_p, log_q = reference.log_softmax(-1), value.log_softmax(-1)
        p, q = log_p.exp(), log_q.exp()
        kl = (p * (log_p - log_q)).sum(-1).clamp_min(0)
        tv = (p - q).abs().sum(-1) / 2
        chosen = value.argmax(-1)
        regret = p.max(-1).values - p.gather(-1, chosen[..., None]).squeeze(-1)
        rows.append(
            {
                "maximum_logit_error": delta.max().item(),
                "mean_absolute_logit_error": delta.mean().item(),
                "relative_l2": ((value - reference).norm() / reference.norm().clamp_min(1e-20)).item(),
                "mean_kl": kl.mean().item(),
                "maximum_kl": kl.max().item(),
                "mean_total_variation": tv.mean().item(),
                "maximum_total_variation": tv.max().item(),
                "maximum_argmax_probability_regret": regret.max().item(),
                "argmax_equal": int((chosen == reference.argmax(-1)).sum()),
                "positions": chosen.numel(),
                "values": value.numel(),
            }
        )
    positions = sum(row["positions"] for row in rows)
    values = sum(row["values"] for row in rows)
    return {
        "probes": rows,
        "maximum_logit_error": max(row["maximum_logit_error"] for row in rows),
        "mean_absolute_logit_error": sum(row["mean_absolute_logit_error"] * row["values"] for row in rows) / values,
        "maximum_relative_l2": max(row["relative_l2"] for row in rows),
        "mean_kl": sum(row["mean_kl"] * row["positions"] for row in rows) / positions,
        "maximum_kl": max(row["maximum_kl"] for row in rows),
        "mean_total_variation": sum(row["mean_total_variation"] * row["positions"] for row in rows) / positions,
        "maximum_total_variation": max(row["maximum_total_variation"] for row in rows),
        "maximum_argmax_probability_regret": max(row["maximum_argmax_probability_regret"] for row in rows),
        "argmax_equal": sum(row["argmax_equal"] for row in rows),
        "positions": positions,
    }


def assert_bf16_deployment_drift(report: dict) -> None:
    """Fixed real-model BF16 gates; FP32 equivalence and tensor checks are mandatory too.

    Preserve confident decisions: any changed greedy token must be within two
    percentage points of the original maximum probability. Always report actual
    argmax agreement, including ties; never describe this as bitwise equivalence.
    """
    limits = {
        "maximum_logit_error": 2.0,
        "mean_absolute_logit_error": 0.2,
        "maximum_relative_l2": 0.0625,
        "mean_kl": 0.01,
        "maximum_kl": 0.05,
        "mean_total_variation": 0.025,
        "maximum_total_variation": 0.1,
        "maximum_argmax_probability_regret": 0.02,
    }
    for name, limit in limits.items():
        if report[name] > limit:
            raise AssertionError(f"BF16 merged deployment drift {name}={report[name]:.8g} exceeds {limit}")


def verify_promoted_merge(
    policy, identity: dict, batches: list[dict], exported: str, *, reference: list[torch.Tensor] | None = None
) -> dict:
    """Control a private loaded copy in FP32 and account for every exported tensor.

    Promoting a BF16 base preserves its already-rounded values. This isolates the
    algebra of each method from BF16 evaluation rounding. The production pure
    merger prepares the state; independent unmerged/native/HF forwards provide
    the numerical oracle. The caller must discard this private evaluation copy.
    """
    from accelerate import init_on_device
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForImageTextToText

    from cosmos_framework.posttrain.checkpoint.adapters import prepare_merged_state
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear
    from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy
    from cosmos_framework.posttrain.tests.qualify_adapters import logits

    device = next(policy.net.parameters()).device
    policy.net.float().eval()
    if reference is None:
        reference = logits(policy, batches)
    state = {name: value.detach().cpu() for name, value in policy.net.state_dict().items()}
    merged = prepare_merged_state(state, {**identity, "base_storage_dtype": "float32"})
    path = Path(exported)
    marker = json.loads((path / ".materialized").read_text())
    metadata = marker["identity"].get("export_metadata", {})
    if metadata.get("format") != "cosmos-merged-adapter-v1" or metadata.get("adapter_identity") != identity:
        raise ValueError(
            "merged export lacks matching adapter/base/processor/media identity; re-export with the current tool"
        )
    export_dtype = marker["identity"].get("output_dtype", identity["base_storage_dtype"])
    index = json.loads((path / "model.safetensors.index.json").read_text())
    expected_keys = set(merged)
    if policy.net.config.tie_word_embeddings and "lm_head.weight" not in index["weight_map"]:
        assert torch.equal(merged["lm_head.weight"], merged["model.language_model.embed_tokens.weight"])
        expected_keys.remove("lm_head.weight")
    assert set(index["weight_map"]) == expected_keys, "merged artifact tensor set differs from the promoted reference"
    verified = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(str(path / shard))
        for name, value in tensors.items():
            expected = merged[name].to(getattr(torch, export_dtype))
            assert value.dtype == expected.dtype and torch.equal(value, expected), f"merged tensor mismatch: {name}"
            verified += 1
        del tensors
    for name, module in list(policy.net.named_modules()):
        if isinstance(module, PosttrainAdapterLinear):
            dense = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device="meta",
                dtype=torch.float32,
            )
            dense.weight = module.weight
            if module.bias is not None:
                dense.bias = module.bias
            parent, _, leaf = name.rpartition(".")
            setattr(policy.net.get_submodule(parent), leaf, dense)
    policy.net.load_state_dict(merged, strict=True)
    native = logits(policy, batches)
    # Release GPU storage before constructing the full FP32 HF evaluation copy.
    policy.net.cpu()
    del state, module, dense
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with init_on_device("meta", include_buffers=False):
        hf = AutoModelForImageTextToText.from_config(
            AutoConfig.from_pretrained(exported), dtype=torch.float32, attn_implementation="sdpa"
        )
    hf.load_state_dict(merged, strict=True, assign=True)
    hf.tie_weights()
    hf.to(device).eval()
    actual = logits(Qwen3VLPolicy(hf, "qwen3_vl"), batches)
    result = {
        "serialized_tensors_verified": verified,
        "native_fp32": compare_logits(native, reference),
        "hf_fp32": compare_logits(actual, reference),
    }
    for outputs in (native, actual):
        for value, expected in zip(outputs, reference):
            torch.testing.assert_close(value, expected, atol=5e-4, rtol=2e-4)
        metrics = compare_logits(outputs, reference)
        assert metrics["mean_absolute_logit_error"] <= 5e-5, metrics
        assert metrics["argmax_equal"] == metrics["positions"], metrics
    return result
