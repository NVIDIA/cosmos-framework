# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the RoboLab WebSocket action policy server helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    for module_name in (
        "cosmos_framework.scripts.action_policy_server_utils",
        "cosmos_framework.scripts.action_policy_server_robolab",
    ):
        if module_name in sys.modules:
            del sys.modules[module_name]
    from cosmos_framework.scripts import action_policy_server_robolab as robolab_server  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_resolve_public_hf_policy_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded"
    downloaded_path.mkdir()
    calls: list[tuple[str, str]] = []

    def fake_download(checkpoint: Any) -> str:
        calls.append((checkpoint.repository, checkpoint.revision))
        return str(downloaded_path)

    monkeypatch.setattr(robolab_server.CheckpointDirHf, "download", fake_download)

    resolved = robolab_server._resolve_checkpoint_path("Cosmos3-Nano-Policy-DROID", hf_revision="test-revision")

    assert resolved == str(downloaded_path)
    assert calls == [("nvidia/Cosmos3-Nano-Policy-DROID", "test-revision")]


def test_resolve_checkpoint_keeps_existing_local_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "Cosmos3-Nano-Policy-DROID"
    checkpoint_path.mkdir()

    resolved = robolab_server._resolve_checkpoint_path(str(checkpoint_path), hf_revision="main")

    assert resolved == str(checkpoint_path)


def test_validate_checkpoint_accepts_diffusers_safetensors_index(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")

    robolab_server._validate_checkpoint(str(tmp_path), allow_dcp_checkpoint=False)


def test_load_openpi_websocket_policy_server_from_lightweight_package(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWebsocketPolicyServer:
        pass

    fake_package = type(sys)("openpi_server")
    fake_package.__path__ = []
    fake_module = type(sys)("openpi_server.websocket_policy_server")
    fake_module.WebsocketPolicyServer = FakeWebsocketPolicyServer
    monkeypatch.setitem(sys.modules, "openpi_server", fake_package)
    monkeypatch.setitem(sys.modules, "openpi_server.websocket_policy_server", fake_module)

    assert robolab_server._load_openpi_websocket_policy_server() is FakeWebsocketPolicyServer


def test_server_args_default_to_released_droid_serving_config() -> None:
    args = robolab_server.RobolabServerArgs()

    assert args.checkpoint_path == "nvidia/Cosmos3-Nano-Policy-DROID"
    assert args.hf_revision == "main"
    assert args.domain_name == "droid_lerobot"
    assert args.seed == 0
    assert args.resolution == "480"
    assert args.conditioning_fps == 15.0
    assert args.action_chunk_size == 32
    assert args.action_dim == 8
    assert args.image_height == 540
    assert args.image_width == 640
    assert args.history_length == 1
    assert args.action_space == "joint_pos"
    assert args.use_state is True
    assert args.guidance == 3.0
    assert args.num_steps == 4
    assert args.shift == 5.0
    assert args.deterministic_seed is False


def test_joint_pos_observation_preprocessing_matches_internal_layout() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service.cfg = robolab_server.RobolabPolicyConfig(
        checkpoint_path="/unused/model",
        domain_name="droid_lerobot",
        decode_video=False,
        seed=0,
        deterministic_seed=True,
        guidance=3.0,
        num_steps=4,
        shift=5.0,
        conditioning_fps=15.0,
        resolution=None,
        action_chunk_size=4,
        action_dim=8,
        image_height=4,
        image_width=5,
        action_space="joint_pos",
        use_state=True,
        history_length=2,
    )
    service._transform = lambda sample, resolution: sample

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    joint_position = np.arange(14, dtype=np.float32).reshape(2, 7)
    gripper_position = np.array([[0.2], [0.3]], dtype=np.float32)
    obs = {
        "prompt": "open the drawer",
        "observation/image": image,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
    }

    sample = robolab_server.RobolabPolicyService._build_sample(service, obs)

    assert sample["video"].shape == (3, 5, 4, 5)
    assert sample["video"].dtype == torch.uint8
    assert sample["action"].shape == (5, 8)
    np.testing.assert_allclose(sample["action"][0].numpy(), np.concatenate([joint_position[-1], [0.7]]))
    assert sample["history_action"].shape == (1, 8)
    np.testing.assert_allclose(sample["history_action"][0].numpy(), np.concatenate([joint_position[0], [0.8]]))
    assert sample["ai_caption"] == "open the drawer"
    assert sample["viewpoint"] == "concat_view"


def test_build_data_batch_wraps_multi_item_keys_like_internal_server() -> None:
    sample = {
        "video": torch.zeros((3, 2, 4, 5), dtype=torch.uint8),  # [3,T,H,W]
        "action": torch.zeros((1, 8), dtype=torch.float32),  # [T,D]
        "domain_id": torch.tensor(1, dtype=torch.long),  # []
        "conditioning_fps": torch.tensor(15, dtype=torch.long),  # []
        "ai_caption": "move",
    }

    batch = robolab_server._build_data_batch_from_sample(sample)

    assert batch["video"][0][0] is sample["video"]
    assert batch["action"][0][0] is sample["action"]
    assert batch["domain_id"][0].shape == (1,)
    assert batch["conditioning_fps"][0].shape == (1,)
    assert batch["ai_caption"] == ["move"]
