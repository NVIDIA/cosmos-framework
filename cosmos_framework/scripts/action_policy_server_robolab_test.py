# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the RoboLab WebSocket action policy server helpers."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
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
    assert args.use_torch_compile is True
    assert args.compiled_region == "all"
    assert args.compile_dynamic is False
    assert args.use_cuda_graphs is False
    assert args.attention_backend == "default"
    assert args.allow_missing_action_heads is False
    assert args.action_head_init_seed == 0
    assert args.action_head_state_path is None
    assert args.startup_warmup_requests == 0
    assert args.startup_warmup_prompt == "Pick up the banana and place it in the bowl"


def test_edge_setup_enables_only_explicit_missing_action_heads(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"unused")
    (checkpoint / "weights.distcp").write_bytes(b"unused")
    output_dir = tmp_path / "output"
    service = object.__new__(robolab_server.RobolabPolicyService)
    args = robolab_server.RobolabServerArgs(
        checkpoint_path=str(checkpoint),
        allow_dcp_checkpoint=True,
        config_file="edge_native_config.py",
        experiment="edge_policy_native",
        output_dir=output_dir,
        allow_missing_action_heads=True,
        attention_backend="pytorch_sdpa_cudnn",
        deterministic_seed=True,
        guidance=1.0,
    )

    setup = robolab_server.RobolabPolicyService._build_setup_args(service, args)

    assert setup.checkpoint_path == str(checkpoint)
    assert setup.use_ema_weights is True
    assert setup.guardrails is False
    assert setup.keys_to_skip_loading == list(robolab_server._ACTION_HEAD_PREFIXES)
    assert setup.use_torch_compile is True
    assert setup.compiled_region == "all"
    assert setup.compile_dynamic is False
    assert setup.use_cuda_graphs is False
    assert "model.config.attention_backend=pytorch_sdpa_cudnn" in setup.experiment_overrides


def test_missing_action_head_opt_in_rejects_safetensors_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"unused")
    service = object.__new__(robolab_server.RobolabPolicyService)
    args = robolab_server.RobolabServerArgs(
        checkpoint_path=str(checkpoint),
        output_dir=tmp_path / "output",
        allow_missing_action_heads=True,
    )

    with pytest.raises(ValueError, match="only valid for an explicitly allowed DCP"):
        robolab_server.RobolabPolicyService._build_setup_args(service, args)


class _ActionHeadNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action2llm = torch.nn.Linear(2, 3)
        self.llm2action = torch.nn.Linear(3, 2)
        self.unrelated = torch.nn.Linear(2, 2)


def test_action_head_state_roundtrip_and_checksum(tmp_path: Path) -> None:
    model = SimpleNamespace(net=_ActionHeadNetwork())
    state = robolab_server._get_action_head_state(model)
    expected_digest = robolab_server._action_head_digest(state)
    path = tmp_path / "action_heads.pt"

    assert robolab_server._save_action_head_state(state, path) == expected_digest
    with torch.no_grad():
        model.net.action2llm.weight.zero_()
        model.net.llm2action.bias.fill_(7)

    assert robolab_server._load_action_head_state(model, path) == expected_digest
    assert robolab_server._action_head_digest(robolab_server._get_action_head_state(model)) == expected_digest
    assert path.with_suffix(".pt.sha256").read_text().strip() == f"{expected_digest}  action_heads.pt"


def test_action_head_state_rejects_mismatched_keys(tmp_path: Path) -> None:
    model = SimpleNamespace(net=_ActionHeadNetwork())
    path = tmp_path / "bad_action_heads.pt"
    torch.save({"action2llm.weight": torch.zeros_like(model.net.action2llm.weight)}, path)

    with pytest.raises(ValueError, match="Action-head state mismatch"):
        robolab_server._load_action_head_state(model, path)


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


def test_infer_returns_valid_action_and_server_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def generate_samples_from_batch(self, data_batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            return {"action": [torch.zeros((5, 8), dtype=torch.float32)]}

    service = object.__new__(robolab_server.RobolabPolicyService)
    service.cfg = robolab_server.RobolabPolicyConfig(
        checkpoint_path="/unused/model",
        domain_name="droid_lerobot",
        decode_video=False,
        seed=0,
        deterministic_seed=True,
        guidance=1.0,
        num_steps=4,
        shift=5.0,
        conditioning_fps=15.0,
        resolution=None,
        action_chunk_size=4,
        action_dim=8,
        image_height=4,
        image_width=5,
    )
    service._transform = lambda sample, resolution: sample
    service._lock = threading.Lock()
    service._rng = np.random.default_rng(0)
    service.model = FakeModel()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    observation = {
        "prompt": "move",
        "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
        "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
    }

    result = robolab_server.RobolabPolicyService.infer(service, observation)

    assert result["action"].shape == (4, 8)
    assert np.isfinite(result["action"]).all()
    assert set(result["timing"]) == {
        "server_preprocess_ms",
        "server_policy_inference_ms",
        "server_postprocess_ms",
        "server_total_ms",
    }
    assert result["timing"]["server_total_ms"] >= result["timing"]["server_policy_inference_ms"]
