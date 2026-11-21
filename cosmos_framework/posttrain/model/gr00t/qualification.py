# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Controlled native action outputs for artifact parity, separate from task-quality claims."""

from __future__ import annotations

import hashlib

import numpy as np


def capture_outputs(policy, batch, *, seed):
    import torch

    from cosmos_framework.posttrain.entrypoints.action_sft import preserved_evaluation
    from cosmos_framework.posttrain.model.gr00t.rollout import predict_processed

    with preserved_evaluation(policy, seed):
        inputs = policy.packer(batch)
        identity = {}
        for name, value in inputs.items():
            if not torch.is_tensor(value):
                raise TypeError(f"native GR00T processed input {name} is not a tensor")
            value = value.detach().cpu().contiguous()
            identity[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
        flow = policy.flow_matching_loss(inputs)
        normalized, native = predict_processed(policy, inputs)
        if not torch.isfinite(flow.loss):
            raise ValueError("nonfinite native GR00T qualification loss")
        return {
            "format": "gr00t-native-output-probe-v1",
            "seed": seed,
            "ids": [item.id for item in batch],
            "inputs": identity,
            "masked_elements": float(flow.action_mask.sum()),
            "flow_loss": float(flow.loss),
            "normalized_actions": normalized.cpu().tolist(),
            "native_decoded_actions": native.tolist(),
        }


def compare_outputs(reference, candidate, *, normalized_atol, decoded_atol, flow_loss_atol, flow_loss_rtol):
    """Call with prospectively declared tolerances; no default numerical/task acceptance margin."""
    for key in ("format", "seed", "ids", "inputs", "masked_elements"):
        if reference[key] != candidate[key]:
            raise ValueError(f"GR00T artifact comparison changed {key}")
    maxima = {}
    for key, tolerance in (
        ("normalized_actions", normalized_atol),
        ("native_decoded_actions", decoded_atol),
        ("flow_loss", flow_loss_atol),
    ):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("artifact tolerances must be explicit finite nonnegative values")
        left, right = np.asarray(reference[key]), np.asarray(candidate[key])
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"GR00T artifact output shape/finiteness differs for {key}")
        if key == "normalized_actions":
            if left.shape != (len(reference["ids"]), 40, 132):
                raise ValueError("GR00T normalized output requires the pinned [batch,40,132] head")
            maxima["normalized_actions_all_slots"] = float(np.max(np.abs(left - right)))
            # The checkpoint's LIBERO processor decodes sixteen seven-axis actions.
            # Padded/unexecuted head channels remain reported, not substituted for task outputs.
            left, right = left[:, :16, :7], right[:, :16, :7]
        elif key == "native_decoded_actions":
            if left.shape != (len(reference["ids"]), 16, 7):
                raise ValueError("native decoded action comparison requires [batch,16,7]")
        elif left.ndim != 0:
            raise ValueError("native flow-loss comparison requires a scalar")
        maximum = float(np.max(np.abs(left - right)))
        if maximum > tolerance:
            raise ValueError(f"GR00T {key} maximum error {maximum} exceeds declared tolerance {tolerance}")
        maxima[key] = maximum
    if not np.isfinite(flow_loss_rtol) or flow_loss_rtol < 0:
        raise ValueError("relative flow-loss tolerance must be explicit, finite and nonnegative")
    baseline = float(reference["flow_loss"])
    if baseline < 0 or float(candidate["flow_loss"]) < 0:
        raise ValueError("native flow loss must be nonnegative")
    relative = maxima["flow_loss"] / max(baseline, 1e-6)
    if relative > flow_loss_rtol:
        raise ValueError(f"GR00T flow-loss relative error {relative} exceeds declared tolerance {flow_loss_rtol}")
    left, right = np.asarray(reference["native_decoded_actions"]), np.asarray(candidate["native_decoded_actions"])
    # Native LIBERO executes -sign(2*g-1). A small continuous error can otherwise
    # conceal an opposite open/close command, including the exact midpoint case.
    if not np.array_equal(np.sign(2 * left[..., -1] - 1), np.sign(2 * right[..., -1] - 1)):
        raise ValueError("GR00T dense conversion changes a discrete gripper command")
    maxima["flow_loss_relative"] = relative
    maxima["gripper_decisions_equal"] = True
    return maxima
