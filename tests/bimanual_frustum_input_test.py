# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bimanual hosted-input materialization and frustum-geometry contracts."""

import json
from pathlib import Path

import pytest
import torch

import cosmos_framework.inference.args as args_module
from cosmos_framework.data.generator.action.utils.action_processing import make_camera_global_asinh_normalizer
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel
from cosmos_framework.inference.args import ActionDataArgs, OmniSampleOverrides
from cosmos_framework.inference.bimanual import (
    load_bimanual_frustum_geometry,
    recover_bimanual_camera_poses,
)


def test_frustum_path_resolves_relative_to_manifest(tmp_path: Path) -> None:
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text("{}\n")
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "example",
                "kv_cache_frustum_path": "geometry.json",
                "model_mode": "forward_dynamics",
                "autoregressive": True,
            }
        )
        + "\n"
    )

    (sample,) = OmniSampleOverrides.from_files([manifest])
    assert sample.kv_cache_frustum_path == str(geometry_path)
    resolved = ActionDataArgs.model_validate({"kv_cache_frustum_path": sample.kv_cache_frustum_path})
    assert resolved.kv_cache_frustum_path == geometry_path


def test_hf_hosted_bimanual_inputs_use_the_standard_download_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(url: str | None, output_dir: Path, output_name: str) -> str | None:
        if url is None:
            return None
        calls.append((url, output_name))
        return str(output_dir / f"{output_name}{Path(url).suffix}")

    monkeypatch.setattr(args_module, "download_file", fake_download)
    root = "https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/main"
    manifest = tmp_path / "hosted.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "agibot_001_unified_v1",
                "vision_path": f"{root}/assets/sim_bimanual/agibot/images/agibot_001_initial.png",
                "content_crop_box": [0, 0, 640, 360],
                "action_path": f"{root}/assets/sim_bimanual/agibot/actions/agibot_001_normalized_unifiedv1.json",
                "kv_cache_frustum_path": f"{root}/examples/inference/bimanual_frustum.json",
            }
        )
        + "\n"
    )
    (sample,) = OmniSampleOverrides.from_files([manifest])
    sample.download(tmp_path)

    assert set(calls) == {
        (f"{root}/assets/sim_bimanual/agibot/images/agibot_001_initial.png", "vision"),
        (f"{root}/assets/sim_bimanual/agibot/actions/agibot_001_normalized_unifiedv1.json", "action"),
        (f"{root}/examples/inference/bimanual_frustum.json", "kv_cache_frustum"),
    }
    assert sample.vision_path == str(tmp_path / "vision.png")
    assert sample.content_crop_box == (0, 0, 640, 360)
    assert sample.action_path == str(tmp_path / "action.json")
    assert sample.kv_cache_frustum_path == str(tmp_path / "kv_cache_frustum.json")


def _write_chunk16_actions(path: Path) -> torch.Tensor:
    frames = 65
    poses = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
    angle = torch.arange(frames, dtype=torch.float64) * 0.01 + 0.3
    poses[:, 0, 0] = poses[:, 2, 2] = angle.cos()
    poses[:, 0, 2] = angle.sin()
    poses[:, 2, 0] = -angle.sin()
    poses[:, 0, 3] = torch.arange(frames, dtype=torch.float64) * 0.02 + 1.0
    poses[:, 2, 3] = angle.sin() + 0.5
    raw = torch.from_numpy(
        pose_abs_to_rel(
            poses.numpy(),
            rotation_format="rot6d",
            pose_convention="backward_chunk_anchored_16f",
        )
    )
    normalizer = make_camera_global_asinh_normalizer(pose_convention="backward_chunk_anchored_16f")
    normalized = torch.zeros(frames - 1, 59)
    normalized[:, :9] = normalizer.normalize_action(raw)
    path.write_text(
        json.dumps(
            {
                "action_space": "normalized_unified_v1",
                "action": normalized.tolist(),
                "domain_names": "agibotworld",
            }
        )
    )
    return torch.linalg.inv(poses[0]) @ poses


def test_frustum_config_recovers_camera_poses_from_bimanual_sidecar(tmp_path: Path) -> None:
    actions = tmp_path / "actions.json"
    expected = _write_chunk16_actions(actions)
    config = tmp_path / "frustum.json"
    config.write_text(
        json.dumps(
            {
                "pose_source": "bimanual_unified_v1_chunk16",
                "image_size": [640, 360],
                "fps": 30,
                "horizontal_fov_degrees": 90.0,
                "near": 0.1,
                "far": 10.0,
                "sink_size": 8,
                "retrieved_size": 32,
                "recent_size": 32,
            }
        )
    )

    recovered = recover_bimanual_camera_poses(actions)
    torch.testing.assert_close(recovered, expected, atol=1e-5, rtol=1e-5)
    geometry = load_bimanual_frustum_geometry(config, actions)
    torch.testing.assert_close(geometry["c2w"], expected, atol=1e-5, rtol=1e-5)
    assert geometry["fps"] == 30
    assert "c2w" not in json.loads(config.read_text())


def test_frustum_camera_recovery_rejects_non_agibot_domains(tmp_path: Path) -> None:
    actions = tmp_path / "actions.json"
    _write_chunk16_actions(actions)
    payload = json.loads(actions.read_text())
    payload["domain_names"] = "xdof_yam"
    actions.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="AgiBot/camera"):
        recover_bimanual_camera_poses(actions)
