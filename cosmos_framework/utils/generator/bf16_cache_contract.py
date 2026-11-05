# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared, lightweight deployment identity; no inference/training cross-imports."""

PROFILE = "qwen3_vl_bf16_canonical_cache_v1"


def validate_adapter_identity(identity: dict) -> None:
    """The deployment override cannot reinterpret legacy factorized artifacts."""
    if not isinstance(identity, dict):
        raise ValueError(f"{PROFILE} requires an adapter identity object")
    resolved, dependencies = identity.get("resolved"), identity.get("dependencies")
    if (
        not isinstance(resolved, dict)
        or not isinstance(dependencies, dict)
        or not isinstance(resolved.get("targets"), list)
        or any(not isinstance(row, dict) or not isinstance(row.get("settings"), dict) for row in resolved["targets"])
    ):
        raise ValueError(f"{PROFILE} requires valid resolved target and dependency objects")
    if (
        identity.get("format") != "cosmos-posttrain-adapter-v4"
        or identity.get("architecture") != "qwen3_vl"
        or identity.get("base_storage_dtype") != "bfloat16"
        or identity.get("resolved", {}).get("model_execution") != "merged_weight_bf16_v1"
        or not identity.get("resolved", {}).get("targets")
        or any(
            row.get("settings", {}).get("execution") != "merged_weight_bf16_v1"
            or row.get("settings", {}).get("dropout") != 0
            for row in identity["resolved"]["targets"]
        )
        or identity.get("dependencies", {}).get("torch") != "2.10.0"
        or identity.get("dependencies", {}).get("transformers") != "5.8.1"
    ):
        raise ValueError(f"{PROFILE} requires a BF16 Qwen3-VL merged_weight_bf16_v1 v4 adapter identity")
