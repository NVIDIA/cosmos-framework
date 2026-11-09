# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared, lightweight deployment identity; no inference/training cross-imports."""

PROFILE = "qwen3_vl_bf16_canonical_cache_v1"
CORRECTED_PROFILE = "qwen3_vl_bf16_canonical_cache_v2"
FAST_PROFILE = "qwen3_vl_bf16_tensorcore_prefill_v1"
QWEN3_VL_PROFILES = (PROFILE, CORRECTED_PROFILE, FAST_PROFILE)
QWEN3_5_PROFILE = "qwen3_5_bf16_canonical_cache_v1"
PROFILE_FAMILIES = {**dict.fromkeys(QWEN3_VL_PROFILES, "qwen3_vl"), QWEN3_5_PROFILE: "qwen3_5"}


def validate_adapter_identity(identity: dict, *, profile: str = PROFILE) -> None:
    """The deployment override cannot reinterpret legacy factorized artifacts."""
    if profile not in PROFILE_FAMILIES:
        raise ValueError(f"unknown canonical cache profile: {profile!r}")
    if not isinstance(identity, dict):
        raise ValueError(f"{profile} requires an adapter identity object")
    resolved, dependencies = identity.get("resolved"), identity.get("dependencies")
    if (
        not isinstance(resolved, dict)
        or not isinstance(dependencies, dict)
        or not isinstance(resolved.get("targets"), list)
        or any(not isinstance(row, dict) or not isinstance(row.get("settings"), dict) for row in resolved["targets"])
    ):
        raise ValueError(f"{profile} requires valid resolved target and dependency objects")
    if (
        identity.get("format") != "cosmos-posttrain-adapter-v4"
        or identity.get("architecture") != PROFILE_FAMILIES[profile]
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
        raise ValueError(f"{profile} requires a matching BF16 dense Qwen merged_weight_bf16_v1 v4 adapter identity")
