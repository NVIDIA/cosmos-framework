# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L1/CPU tests for the simulator-free GR00T sample source against the REAL processor (Isaac types plus
the DROID checkpoint's processor, so gr00t-venv only; no GPU). The property worth a checkpoint load is
in-distribution-ness: the processor clips normalized actions to [-1, 1] at q01/q99, and a clipped action is
constant, which would make the loss insensitive to most of the head."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytest.importorskip("gr00t", reason="Isaac-GR00T is only installed in the gr00t venv")

from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config  # noqa: E402
from cosmos_framework.posttrain.model.gr00t.policy import load_processor  # noqa: E402
from cosmos_framework.posttrain.tests.helpers.gr00t_synthetic import (  # noqa: E402
    build_observations,
    synthetic_samples,
    synthetic_step_data,
)


@pytest.fixture(scope="module")
def config():
    return make_model_config()


@pytest.fixture(scope="module")
def processor(config):
    if "HF_HOME" not in os.environ:
        pytest.skip("HF_HOME is unset; point it at the gr00t-scratch cache holding the DROID checkpoint")
    return load_processor(config.checkpoint_path)


@pytest.mark.L1
@pytest.mark.CPU
def test_step_data_actions_are_absolute_and_rotations_are_valid(config) -> None:
    """`eef_9d` rows must be orthonormalizable: the processor converts them to relative rotations."""
    horizon = config.model_action_horizon
    step = synthetic_step_data(config, build_observations(config, 1, seed=0)[0], seed=0)

    assert sorted(step.actions) == ["eef_9d", "gripper_position", "joint_position"]
    assert step.actions["eef_9d"].shape == (horizon, 9)
    assert step.actions["gripper_position"].shape == (horizon, 1)
    assert step.actions["joint_position"].shape == (horizon, 7)
    for row in step.actions["eef_9d"]:
        first, second = row[3:6].astype(np.float64), row[6:9].astype(np.float64)
        assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
        assert np.linalg.norm(second) == pytest.approx(1.0, abs=1e-5)
        assert float(np.dot(first, second)) == pytest.approx(0.0, abs=1e-5)
    # Gripper stays off the 0/1 rails, where the absolute normalization would saturate.
    assert step.actions["gripper_position"].min() > 0.0 and step.actions["gripper_position"].max() < 1.0
    # One frame per camera: the checkpoint's delta_indices are [0], so T = 1.
    assert sorted(step.images) == sorted(config.camera_map)
    assert all(len(frames) == 1 for frames in step.images.values())


@pytest.mark.L1
@pytest.mark.CPU
def test_processed_samples_match_the_collator_contract_and_stay_in_band(processor, config) -> None:
    horizon, action_dim = config.model_action_horizon, config.action_dim
    samples = synthetic_samples(processor, config, 2, seed=1234)
    assert len(samples) == 2
    for sample in samples:
        assert set(sample) >= {"state", "action", "action_mask", "embodiment_id", "vlm_content"}
        width = sample["state"].shape[-1]
        assert sample["state"].shape == (1, width) and sample["state"].dtype == torch.float32
        assert sample["action"].shape == (horizon, width) and sample["action"].dtype == torch.float32
        assert isinstance(sample["embodiment_id"], int)
        assert isinstance(sample["vlm_content"]["text"], str) and sample["vlm_content"]["text"]
        assert len(sample["vlm_content"]["images"]) == len(config.camera_map)
        assert all(f.dtype == torch.uint8 for f in sample["vlm_content"]["images"])
        # In band and varying: a value at +/-1 was clipped by the processor's q01/q99 normalization.
        real = sample["action"][:horizon, :action_dim]
        assert torch.isfinite(real).all() and (real.abs() < 1.0).all() and real.std().item() > 0.0


@pytest.mark.L1
@pytest.mark.CPU
def test_samples_are_seed_deterministic_end_to_end(processor, config) -> None:
    first = synthetic_samples(processor, config, 2, seed=4321)
    second = synthetic_samples(processor, config, 2, seed=4321)
    for left, right in zip(first, second):
        assert torch.equal(left["action"], right["action"]) and torch.equal(left["state"], right["state"])
        assert left["vlm_content"]["text"] == right["vlm_content"]["text"]
        assert all(torch.equal(a, b) for a, b in zip(left["vlm_content"]["images"], right["vlm_content"]["images"]))
