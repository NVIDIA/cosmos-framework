# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the RoboLab WebSocket action policy server helpers."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

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

    rank0_downloads: list[Any] = []

    def fake_download_on_rank0(download: Any) -> Path:
        rank0_downloads.append(download)
        return Path(download())

    monkeypatch.setattr(robolab_server.CheckpointDirHf, "download", fake_download)
    monkeypatch.setattr(robolab_server, "_download_on_rank0", fake_download_on_rank0)

    resolved = robolab_server._resolve_checkpoint_path("Cosmos3-Nano-Policy-DROID", hf_revision="test-revision")

    assert resolved == str(downloaded_path)
    assert calls == [("nvidia/Cosmos3-Nano-Policy-DROID", "test-revision")]
    assert len(rank0_downloads) == 1


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
    assert args.guidance_interval is None
    assert args.num_steps == 4
    assert args.shift == 5.0
    assert args.deterministic_seed is False
    assert args.cfg_parallel is False


def test_server_args_accept_guidance_interval() -> None:
    args = robolab_server.RobolabServerArgs(guidance_interval=(960.0, 1001.0))

    assert args.guidance_interval == (960.0, 1001.0)


@pytest.mark.parametrize(
    ("cfg_parallel", "world_size", "guidance", "expected"),
    [
        (False, 1, 1.0, {"dp_shard_size": 1}),
        (True, 2, 3.0, {"dp_shard_size": 1}),
    ],
)
def test_resolve_parallelism_overrides(
    cfg_parallel: bool,
    world_size: int,
    guidance: float,
    expected: dict[str, int],
) -> None:
    assert (
        robolab_server._resolve_parallelism_overrides(
            cfg_parallel=cfg_parallel,
            world_size=world_size,
            guidance=guidance,
        )
        == expected
    )


@pytest.mark.parametrize(("cfg_parallel", "world_size"), [(False, 2), (True, 1), (True, 4)])
def test_resolve_parallelism_overrides_rejects_unsupported_launches(
    cfg_parallel: bool,
    world_size: int,
) -> None:
    with pytest.raises(ValueError):
        robolab_server._resolve_parallelism_overrides(
            cfg_parallel=cfg_parallel,
            world_size=world_size,
            guidance=3.0,
        )


def test_resolve_parallelism_overrides_rejects_cfg_parallel_without_cfg() -> None:
    with pytest.raises(ValueError, match="guidance"):
        robolab_server._resolve_parallelism_overrides(
            cfg_parallel=True,
            world_size=2,
            guidance=1.0,
        )


def test_build_control_group_uses_gloo_with_long_idle_timeout() -> None:
    control_group = Mock()
    with (
        patch.object(robolab_server.dist, "is_available", return_value=True),
        patch.object(robolab_server.dist, "is_initialized", return_value=True),
        patch.object(robolab_server.dist, "get_world_size", return_value=2),
        patch.object(robolab_server.dist, "new_group", return_value=control_group) as new_group,
    ):
        assert robolab_server._build_control_group() is control_group

    new_group.assert_called_once_with(
        ranks=[0, 1],
        backend="gloo",
        timeout=robolab_server._CONTROL_GROUP_TIMEOUT,
    )


def test_control_messages_use_the_gloo_group() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service._control_group = Mock()
    message = {"request_id": 3, "kind": "infer"}

    with patch.object(robolab_server.dist, "broadcast_object_list") as broadcast:
        assert service._broadcast_control_message(message, src=0) is message

    broadcast.assert_called_once_with([message], src=0, group=service._control_group)


def test_invalid_launch_is_rejected_before_checkpoint_resolution() -> None:
    args = robolab_server.RobolabServerArgs(cfg_parallel=True)
    with (
        patch.object(robolab_server.torch.cuda, "is_available", return_value=True),
        patch.object(robolab_server, "maybe_init_distributed"),
        patch.object(robolab_server.dist, "is_available", return_value=True),
        patch.object(robolab_server.dist, "is_initialized", return_value=True),
        patch.object(robolab_server.dist, "get_world_size", return_value=4),
        patch.object(robolab_server, "_resolve_checkpoint_path") as resolve_checkpoint,
        pytest.raises(ValueError, match="exactly 2 ranks"),
    ):
        robolab_server.RobolabPolicyService(args)

    resolve_checkpoint.assert_not_called()


def test_infer_rejects_invalid_observation_before_distributed_dispatch() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service._lock = threading.Lock()
    service._build_sample = Mock(side_effect=ValueError("bad observation"))
    service._next_seed = Mock()
    service._send_control_request = Mock()

    with pytest.raises(ValueError, match="bad observation"):
        service.infer({"prompt": "missing image and state"})

    service._next_seed.assert_not_called()
    service._send_control_request.assert_not_called()


def test_infer_waits_for_worker_then_formats_rank0_output() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service._lock = threading.Lock()
    service._control_request_id = 0
    service._build_sample = Mock(return_value={"sample": "prepared"})
    service._next_seed = Mock(return_value=17)
    service._distributed_enabled = Mock(return_value=True)
    service._send_control_request = Mock()
    service._wait_worker_ready = Mock()
    service._generate = Mock(return_value={"samples": "generated"})
    service._format_outputs = Mock(return_value={"action": "formatted"})
    obs = {"prompt": "move"}

    assert service.infer(obs) == {"action": "formatted"}

    service._send_control_request.assert_called_once_with(
        0,
        {"kind": "infer", "obs": obs, "seed": 17},
    )
    service._wait_worker_ready.assert_called_once_with(0)
    service._generate.assert_called_once_with({"sample": "prepared"}, 17)
    service._format_outputs.assert_called_once()


def test_worker_reports_preparation_error_then_processes_next_request() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service._control_request_id = 0
    service._distributed_enabled = Mock(return_value=True)
    service._receive_control_request = Mock(
        side_effect=[
            {"kind": "infer", "obs": {"bad": True}, "seed": 7},
            {"kind": "infer", "obs": {"good": True}, "seed": 8},
            {"kind": "shutdown"},
        ]
    )
    service._build_sample = Mock(side_effect=[ValueError("bad sample"), {"sample": "valid"}])
    service._generate = Mock(return_value={})
    service._send_worker_ready = Mock()

    with patch.object(robolab_server.dist, "get_rank", return_value=1):
        service.worker_loop()

    assert service._send_worker_ready.call_args_list[0].args == (0,)
    assert service._send_worker_ready.call_args_list[0].kwargs == {"error": "ValueError: bad sample"}
    assert service._send_worker_ready.call_args_list[1].args == (1,)
    assert service._send_worker_ready.call_args_list[1].kwargs == {}
    assert service._send_worker_ready.call_args_list[2].args == (2,)
    assert service._send_worker_ready.call_args_list[2].kwargs == {}
    service._generate.assert_called_once_with({"sample": "valid"}, 8)


def test_shutdown_worker_uses_next_control_request_and_waits_for_ack() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service._lock = threading.Lock()
    service._control_request_id = 4
    service._distributed_enabled = Mock(return_value=True)
    service._send_control_request = Mock()
    service._wait_worker_ready = Mock()

    with patch.object(robolab_server.dist, "get_rank", return_value=0):
        service.shutdown_worker()

    service._send_control_request.assert_called_once_with(4, {"kind": "shutdown"})
    service._wait_worker_ready.assert_called_once_with(4)


def test_joint_pos_observation_preprocessing_matches_internal_layout() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service.cfg = robolab_server.RobolabPolicyConfig(
        checkpoint_path="/unused/model",
        domain_name="droid_lerobot",
        decode_video=False,
        seed=0,
        deterministic_seed=True,
        guidance=3.0,
        guidance_interval=None,
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
