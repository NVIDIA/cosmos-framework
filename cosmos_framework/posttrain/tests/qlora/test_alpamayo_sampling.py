# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Alpamayo packer inputs and trajectory sampling with the pinned public processor."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.tests.qlora.action_helpers import (
    alpamayo_source,
    assert_state_equal,
    physicalai_sample,
    require_native_action,
)

require_native_action("alpamayo_r1", transformers=True)


from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
from cosmos_framework.posttrain.entrypoints.action_common import preserved_evaluation
from cosmos_framework.posttrain.model.alpamayo.policy import sample_trajectories

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace
    from alpamayo_r1.models.delta_tokenizer import DeltaTrajectoryTokenizer
    from PIL import Image

    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTModelConfig, AlpamayoSFTPacker

    root = tmp_path_factory.mktemp("native-observation")
    net = alpamayo_source(root).float()
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
    row = physicalai_sample()
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


def test_actual_native_sampler_on_packer_inputs_preserves_rng_and_modes(native):
    policy, labeled = native
    before = collect_rng_state()
    modes = [module.training for module in policy.net.modules()]
    inputs = policy.packer(labeled_batch(labeled))
    assert inputs["tokenized"]["image_grid_thw"].shape[0] == 16
    assert inputs["history"]["ego_history_xyz"].shape == (1, 1, 16, 3)
    results = []
    for _ in range(2):
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
        ("short_history", "history"),
        ("nan_history", "finite"),
        ("wrong_origin", "origin"),
        ("wrong_rotation", "orientation"),
        ("camera", "camera"),
        ("duplicates", "distinct"),
        ("future_frame", "precede"),
        ("world_frame", "vehicle frame"),
        ("missing_file", "No such file"),
    ],
)
def test_malformed_samples_fail_explicitly_in_the_packer_without_mutating_input(native, change, match):
    policy, labeled = native
    sample = copy.deepcopy(labeled)
    obs = sample["observation"]
    first_camera = obs["camera_order"][0]
    if change == "short_history":
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
    else:
        sample["media"][first_camera][0] = "/missing-observation-fixture.png"
    before = copy.deepcopy(sample)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        policy.packer(labeled_batch(sample))
    assert sample == before
