# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native rollout conventions and artifact-output checks; no mocked simulator qualification."""

import copy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
    execution_protocol,
)
from cosmos_framework.posttrain.model.gr00t.qualification import compare_outputs
from cosmos_framework.posttrain.model.gr00t.rollout import NativeArtifactRollout, prepare_observations


def test_explicit_eight_step_profile_preserves_original_spatial_protocol():
    legacy = execution_protocol("libero_spatial")
    selected = execution_protocol("libero_spatial", GR00T_QLORA_SPATIAL_EXECUTION_PROFILE)
    assert legacy.execution_horizon == 16 and selected.execution_horizon == 8
    assert legacy.max_episode_steps == selected.max_episode_steps == 256
    assert legacy.reset_steps == selected.reset_steps == 15
    assert selected.action_transform == "binary-inverted-gripper"
    with pytest.raises(ValueError):
        execution_protocol("libero_10", GR00T_QLORA_SPATIAL_EXECUTION_PROFILE)


def test_observation_only_packing_preserves_upright_rgb_and_finger_state(monkeypatch):
    types = ModuleType("gr00t.data.types")
    types.MessageType = SimpleNamespace(EPISODE_STEP=SimpleNamespace(value="episode_step"))
    types.VLAStepData = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, types.__name__, types)
    captured = []

    class Processor:
        def __call__(self, messages):
            captured.append(messages[0]["content"])
            return {"vlm_content": {"conversation": "unused"}}

        def collator(self, items):
            assert "conversation" not in items[0]["vlm_content"]
            return {"inputs": {"state": torch.ones(1, 1, 132)}}

    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    observation = {
        "image": image,
        "wrist_image": image.copy(),
        "state": np.arange(8, dtype=np.float32),
        "prompt": "open the drawer",
    }
    policy = SimpleNamespace(
        config=SimpleNamespace(resolve_embodiment_tag=lambda: "libero_sim"),
        packer=SimpleNamespace(processor=Processor()),
    )
    inputs, states = prepare_observations(policy, [observation])
    assert "action" not in inputs and captured[0].actions == {}
    np.testing.assert_array_equal(captured[0].images["image"][0], image)
    np.testing.assert_array_equal(states["gripper"], [[[6, 7]]])
    with pytest.raises(ValueError, match="uint8"):
        prepare_observations(policy, [{**observation, "image": image.astype(np.float32)}])


def test_rollout_slices_supervised_prefix_and_never_inverts_gripper_twice(monkeypatch):
    from cosmos_framework.posttrain.model.gr00t import rollout

    config = make_config().model
    policy = SimpleNamespace(config=config, net=nn.Linear(2, 2))
    monkeypatch.setattr(rollout, "prepare_observations", lambda policy, obs: ({}, None))

    def predict(*args, **kwargs):
        result = torch.randn(1, 16, 7).numpy()
        result[..., -1] = 0.25
        return None, result

    monkeypatch.setattr(rollout, "predict_processed", predict)
    body = NativeArtifactRollout(policy)
    context = {"case": {"suite": "libero_spatial", "task_id": 2, "init_state_id": 3}, "seed": 17, "decision": 0}
    rng = torch.get_rng_state().clone()
    first, second = body.act([{}], context=context), body.act([{}], context=context)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[0].shape == (8, 7) and np.all(first[0][:, -1] == 0.25)
    assert torch.equal(torch.get_rng_state(), rng)
    different = body.act([{}], context={**context, "decision": 1})
    assert not np.array_equal(first[0], different[0])
    assert body.describe().horizon == 8


def test_response_measurement_preserves_actions_rng_and_bounds(monkeypatch):
    from cosmos_framework.posttrain.model.gr00t import rollout

    policy = SimpleNamespace(config=make_config().model, net=nn.Linear(2, 2))
    monkeypatch.setattr(rollout, "prepare_observations", lambda policy, obs: ({}, None))
    monkeypatch.setattr(rollout, "predict_processed", lambda *a, **kw: (None, torch.randn(1, 16, 7).numpy()))
    clock = iter([1.0, 1.25, 2.0, 2.5, 3.0, 3.75, 4.0])
    monkeypatch.setattr(rollout.time, "perf_counter", lambda: next(clock))
    observation = {"image": np.zeros((256, 256, 3), dtype=np.uint8), "wrist_image": np.zeros((256, 256, 3))}
    context = {"case": {"suite": "libero_spatial", "task_id": 2, "init_state_id": 3}, "seed": 17, "decision": 0}
    plain = NativeArtifactRollout(policy)
    measured = NativeArtifactRollout(policy, latency_capacity=2)
    rng = torch.get_rng_state().clone()
    expected = plain.act([observation], context=context)
    first = measured.act([observation], context=context)
    second = measured.act([observation], context=context)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    report = measured.latency_report(warmup_calls=1)
    assert report["cold_first_response_seconds"] == 0.5
    assert report["steady_seconds"]["p95"] == 0.75
    assert report["calls"][0]["media_shapes"] == [{"image": [256, 256, 3], "wrist_image": [256, 256, 3]}]
    assert measured.latency_report(warmup_calls=2)["steady_seconds"] is None
    with pytest.raises(RuntimeError, match="capacity"):
        measured.act([observation], context=context)
    with pytest.raises(ValueError, match="warmup"):
        measured.latency_report(warmup_calls=-1)


def test_actual_output_comparison_has_no_implicit_tolerance_or_input_substitution():
    reference = {
        "format": "gr00t-native-output-probe-v1",
        "seed": 17,
        "ids": ["clip"],
        "inputs": {"state": {"sha256": "bound"}},
        "masked_elements": 70,
        "flow_loss": 0.1,
        "normalized_actions": np.full((1, 40, 132), 0.2).tolist(),
        "native_decoded_actions": np.full((1, 16, 7), 0.3).tolist(),
    }
    assert compare_outputs(
        reference, reference, normalized_atol=0, decoded_atol=0, flow_loss_atol=0, flow_loss_rtol=0
    ) == {
        "flow_loss": 0,
        "normalized_actions": 0,
        "normalized_actions_all_slots": 0,
        "native_decoded_actions": 0,
        "flow_loss_relative": 0,
        "gripper_decisions_equal": True,
    }
    changed = copy.deepcopy(reference)
    changed["normalized_actions"][0][0][0] += 0.01
    with pytest.raises(ValueError, match="exceeds"):
        compare_outputs(reference, changed, normalized_atol=0.001, decoded_atol=0, flow_loss_atol=0, flow_loss_rtol=0)
    changed = copy.deepcopy(reference)
    changed["inputs"]["state"]["sha256"] = "different"
    with pytest.raises(ValueError, match="inputs"):
        compare_outputs(reference, changed, normalized_atol=1, decoded_atol=1, flow_loss_atol=1, flow_loss_rtol=1)


def test_selected_profile_uses_existing_explicit_environment_seed_branch(monkeypatch):
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LiberoCase
    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession

    class Physics:
        def __init__(self):
            self.seed_value, self.commands = None, []

        def seed(self, value):
            self.seed_value = value

        def reset(self):
            pass

        def set_init_state(self, value):
            self.initial = value

        def step(self, action):
            self.commands.append(action.copy())
            return {}, 0.0, False, {}

    session = object.__new__(LiberoChunkSession)
    physics = Physics()
    session.n_lanes, session._seed = 1, 17
    session._protocol = execution_protocol("libero_spatial", GR00T_QLORA_SPATIAL_EXECUTION_PROFILE)
    session._tasks, session._init_states, session._envs = [2], [-1], [physics]
    session._suite = SimpleNamespace(
        get_task=lambda index: SimpleNamespace(language="open drawer"),
        get_task_init_states=lambda index: [np.zeros(2), np.ones(2)],
    )
    session._rng = SimpleNamespace(
        integers=lambda *args: pytest.fail("selected profile must not draw a legacy environment seed")
    )
    monkeypatch.setattr(session, "_choose_case", lambda: LiberoCase(2, 1))
    monkeypatch.setattr(session, "_reset_statistics", lambda lane, raw: None)
    session._reset(0)
    assert physics.seed_value == 17
    assert len(physics.commands) == 15
    np.testing.assert_array_equal(physics.commands[0], [0, 0, 0, 0, 0, 0, -1])
    np.testing.assert_array_equal(physics.initial, np.ones(2))


def test_conversion_rejects_relative_loss_regression_and_discrete_gripper_flip():
    reference = {
        "format": "gr00t-native-output-probe-v1",
        "seed": 17,
        "ids": ["clip"],
        "inputs": {"state": {"sha256": "bound"}},
        "masked_elements": 70,
        "flow_loss": 0.01,
        "normalized_actions": np.full((1, 40, 132), 0.2).tolist(),
        "native_decoded_actions": [[([0.0] * 6 + [0.499]) for _ in range(16)]],
    }
    changed = copy.deepcopy(reference)
    changed["flow_loss"] = 0.011
    with pytest.raises(ValueError, match="relative error"):
        compare_outputs(
            reference, changed, normalized_atol=0.02, decoded_atol=0.02, flow_loss_atol=0.002, flow_loss_rtol=0.02
        )
    changed = copy.deepcopy(reference)
    changed["native_decoded_actions"][0][0][-1] = 0.501
    with pytest.raises(ValueError, match="discrete gripper"):
        compare_outputs(
            reference, changed, normalized_atol=0.02, decoded_atol=0.02, flow_loss_atol=0.002, flow_loss_rtol=0.02
        )


def test_conversion_reports_unexecuted_padding_without_replacing_valid_action_gate():
    reference = {
        "format": "gr00t-native-output-probe-v1",
        "seed": 17,
        "ids": ["clip"],
        "inputs": {"state": {"sha256": "bound"}},
        "masked_elements": 70,
        "flow_loss": 0.01,
        "normalized_actions": np.zeros((1, 40, 132)).tolist(),
        "native_decoded_actions": np.zeros((1, 16, 7)).tolist(),
    }
    changed = copy.deepcopy(reference)
    changed["normalized_actions"][0][20][100] = 1.0
    result = compare_outputs(reference, changed, normalized_atol=0, decoded_atol=0, flow_loss_atol=0, flow_loss_rtol=0)
    assert result["normalized_actions"] == 0 and result["normalized_actions_all_slots"] == 1
    changed["native_decoded_actions"] = [[[0.0] * 7]]
    with pytest.raises(ValueError, match="shape"):
        compare_outputs(reference, changed, normalized_atol=0, decoded_atol=0, flow_loss_atol=0, flow_loss_rtol=0)
