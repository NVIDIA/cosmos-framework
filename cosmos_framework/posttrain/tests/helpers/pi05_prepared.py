# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The shared, provenance-bound compact prepared16 input for Pi05 GPU probes."""

import json
from pathlib import Path

import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import TARGET_KEY, validate_decision
from cosmos_framework.posttrain.config.rl import JointPPOConfig, OptimizerConfig
from cosmos_framework.posttrain.model.pi05.records import TRACE_KEY, validate_ppo_decision
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.object_store import file_sha256

# Stable rows from the archived rank0 capture; the first thirteen retain all coverage.
PREPARED16_INDICES = (0, 8, 24, 138, 146, 242, 2, 3, 1, 4, 5, 6, 7, 9, 10, 11)


def parity_recipe():
    return JointPPOConfig(
        microbatch=16,
        value_lr=1e-4,
        optimizer=OptimizerConfig(lr=5e-6, impl="fused", betas=[0.9, 0.95], weight_decay=0.01, grad_clip_norm=1.0),
        dual_clip=None,
        loss_reduction="fixed_episode_mean",
        time_limit_bootstrap="zero",
    )


def record_descriptor(sample):
    step = sample.transitions[0]
    row = step.policy_info["sample_index"]
    target = step.algo_extra[TARGET_KEY]
    return {
        "sample_id": sample.id,
        "version": step.policy_info["weight_version"],
        "seed": step.policy_info["sampling_seed"],
        "row": row,
        "denoise_index": int(step.policy_info[TRACE_KEY].denoise_index[row]),
        "loss_mask": target.loss_mask,
        "loss_weight": target.loss_weight,
    }


def prepared_coverage(samples):
    rows = [record_descriptor(sample) for sample in samples]
    if len(samples) != 16 or len({row["sample_id"] for row in rows}) != 16:
        raise ValueError("prepared16 requires sixteen unique real decisions")
    if {row["version"] for row in rows} != {0}:
        raise ValueError("prepared16 must come from common_init behavior version0")
    if len({row["seed"] for row in rows}) < 2 or len({row["row"] for row in rows}) < 2:
        raise ValueError("prepared16 requires mixed invocations and original rows")
    if {(row["denoise_index"], row["loss_mask"]) for row in rows} != {
        (index, mask) for index in range(3) for mask in (False, True)
    }:
        raise ValueError("prepared16 must cover every denoise index with true and false masks")
    valid_weights = {row["loss_weight"] for row in rows if row["loss_mask"]}
    if 1.0 not in valid_weights or not any(weight > 1.0 for weight in valid_weights):
        raise ValueError("prepared16 requires valid unit and nonunit loss weights")
    return rows


def load_prepared16(path, config, assets):
    """Validate the export manifest and immutable capture provenance before unpickling."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if manifest["assets"] != assets:
        raise ValueError("prepared16 model assets differ")
    if file_sha256(path) != manifest["records_sha256"]:
        raise ValueError("prepared16 bytes changed after export")
    capture_path = Path(manifest["source_capture"]["path"])
    if file_sha256(capture_path) != manifest["source_capture"]["sha256"]:
        raise ValueError("prepared16 capture manifest changed")
    samples = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(samples, (tuple, list)) or not all(isinstance(sample, Trajectory) for sample in samples):
        raise TypeError("prepared16 must contain Trajectory records")
    for sample in samples:
        validate_ppo_decision(validate_decision(sample, 0, training=True), config)
    if prepared_coverage(samples) != manifest["selection"]:
        raise ValueError("prepared16 selection differs from its provenance")
    return list(samples), manifest
