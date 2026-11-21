# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Observation-only preparation against the actual pinned native processor/sampler."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("alpamayo_r1")
import transformers

if transformers.__version__ != "4.57.6":
    pytest.skip("pinned Alpamayo processor/runtime", allow_module_level=True)

from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
from cosmos_framework.posttrain.entrypoints.action_sft import preserved_evaluation
from cosmos_framework.posttrain.model.alpamayo.observation import prepare_observation
from cosmos_framework.posttrain.model.alpamayo.policy import sample_trajectories
from cosmos_framework.posttrain.tests.test_action_sft import assert_state_equal

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace
    from alpamayo_r1.models.delta_tokenizer import DeltaTrajectoryTokenizer
    from PIL import Image

    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTModelConfig, AlpamayoSFTPacker
    from cosmos_framework.posttrain.tests.test_alpamayo_qlora_contract import sample
    from cosmos_framework.posttrain.tests.test_alpamayo_qlora_flow import source

    root = tmp_path_factory.mktemp("native-observation")
    net = source(root).float()
    packer = AlpamayoSFTPacker.__new__(AlpamayoSFTPacker)
    packer.config = AlpamayoSFTModelConfig(str(root / "base"), str(root / "processor"), min_pixels=16, max_pixels=64)
    packer.model_config = net.config
    packer.processor = net.config._build_processor()
    packer.processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]}
    )
    packer.processor.chat_template = "{% for message in messages %}{% for content in message['content'] %}{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endfor %}"
    packer.action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=64)
    net.action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=64)
    net.hist_traj_tokenizer = DeltaTrajectoryTokenizer(num_bins=64)
    net.diffusion.x_dims = (64, 2)
    row = sample()
    for camera in row["media"]:
        paths = []
        for index in range(4):
            path = root / f"{camera}-{index}.png"
            Image.new("RGB", (4, 2), (20 + index, 30, 40)).save(path)
            paths.append(str(path))
        row["media"][camera] = paths
    row["observation"].update(
        ego_history_xyz=[[i * 0.1, 0, 0] for i in range(-15, 1)],
        ego_history_rot=[torch.eye(3).tolist()] * 16,
        ego_future_rot=[torch.eye(3).tolist()] * 64,
    )
    row["actions"] = [[i * 0.1, 0, 0] for i in range(1, 65)]
    return SimpleNamespace(net=net, packer=packer, config=packer.config), row


def labeled_batch(row):
    return [
        SimpleNamespace(
            id="observation-fixture",
            transitions=[SimpleNamespace(obs=SimpleNamespace(representations={"flow_sample": row}), algo_extra={})],
        )
    ]


def observation(row):
    row = copy.deepcopy(row)
    del row["actions"]
    del row["observation"]["ego_future_rot"]
    return row


def test_future_free_inputs_match_native_training_preparation_without_mutation(native):
    policy, labeled = native
    sample = observation(labeled)
    before = copy.deepcopy(sample)
    rng = collect_rng_state()
    reference = policy.packer(labeled_batch(labeled))
    prepared = prepare_observation(policy, sample)
    assert set(prepared) == {"tokenized", "history"}
    assert_state_equal(prepared, {key: reference[key] for key in prepared})
    assert sample == before
    assert_state_equal(collect_rng_state(), rng)
    assert prepared["tokenized"]["image_grid_thw"].shape[0] == 16
    assert prepared["history"]["ego_history_xyz"].shape == (1, 1, 16, 3)


def test_actual_native_sampler_and_rng_match_after_removing_future_labels(native):
    policy, labeled = native
    before = collect_rng_state()
    modes = [module.training for module in policy.net.modules()]
    prepared = prepare_observation(policy, observation(labeled))
    reference = policy.packer(labeled_batch(labeled))
    results = []
    for inputs in (reference, prepared):
        with preserved_evaluation(policy, 123):
            results.append(sample_trajectories(policy, inputs))
    assert results[0].shape == (1, 64, 3)
    assert torch.isfinite(results[0]).all()
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    assert_state_equal(collect_rng_state(), before)
    assert [module.training for module in policy.net.modules()] == modes


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_history", "history"),
        ("short_history", "history"),
        ("nan_history", "finite"),
        ("wrong_origin", "origin"),
        ("wrong_rotation", "orientation"),
        ("camera", "camera"),
        ("duplicates", "distinct"),
        ("future_frame", "precede"),
        ("world_frame", "vehicle frame"),
        ("missing_media", "media and observation"),
        ("missing_file", "No such file"),
    ],
)
def test_malformed_observations_fail_explicitly_without_mutating_input(native, change, match):
    policy, labeled = native
    sample = observation(labeled)
    obs = sample["observation"]
    first_camera = obs["camera_order"][0]
    if change == "missing_history":
        del obs["ego_history_xyz"]
    elif change == "short_history":
        obs["ego_history_xyz"].pop()
    elif change == "nan_history":
        obs["ego_history_xyz"][0][0] = float("inf")
    elif change == "wrong_origin":
        obs["ego_history_xyz"][-1][0] = 1
    elif change == "wrong_rotation":
        obs["ego_history_rot"][-1] = [[0] * 3] * 3
    elif change == "camera":
        del sample["media"][first_camera]
    elif change == "duplicates":
        sample["media"][first_camera][1] = sample["media"][first_camera][0]
    elif change == "future_frame":
        obs["frame_timestamps_us"][0][-1] += 1
    elif change == "world_frame":
        obs["coordinate_frame"] = "world"
    elif change == "missing_media":
        del sample["media"]
    else:
        sample["media"][first_camera][0] = "/missing-observation-fixture.png"
    before = copy.deepcopy(sample)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        prepare_observation(policy, sample)
    assert sample == before
