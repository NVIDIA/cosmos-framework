# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Actual native weights → tokenizer reconstruction → metric integration checks."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from cosmos_framework.evaluation.lidar.framework_tokenizer import evaluate_framework_tokenizer
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import (
    DEFAULT_NETWORK_CONFIG,
    LidarTokenizerV1Interface,
)
from cosmos_framework.model.generator.tokenizers.lidar.network.transformer_vae import TransformerVAE
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    enforce_tokenizer_clip_length,
    load_prepared_lidar_video_3ch,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _write_tar(path: Path) -> None:
    with tarfile.open(path, "w") as archive:
        for frame, distance in ((0, 25.0), (3, 30.0)):
            for kind, value in (("row", 64), ("col", 900), ("range", distance), ("intensity", 0.25)):
                buffer = io.BytesIO()
                np.savez(buffer, np.array([value]))
                payload = buffer.getvalue()
                member = tarfile.TarInfo(f"scene.{frame:06d}.lidar_{kind}.npz")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def _artifact(directory: Path, suffix: str) -> tuple[Path, Path, dict[str, Any], LidarRangeProjectionConfig]:
    # No local-attention blocks: this is a real TransformerVAE that runs on CPU without NATTEN kernels.
    network: dict[str, Any] = {
        **DEFAULT_NETWORK_CONFIG,
        "resolution": [128, 16],
        "base_channels": 4,
        "patch_size": [16, 16],
        "window_size": [1, 1],
        "depths": [0],
        "num_heads": [1],
        "dilation": [1],
        "temporal_downsample": [],
        "temporal_upsample": [],
        "mapping_depth": 0,
        "bottleneck_3d": False,
    }
    torch.manual_seed(7)
    model = TransformerVAE(**network)
    state = dict(model.state_dict())  # each value: [*parameter_shape]
    state["latent_mean"] = torch.zeros(128)  # [C_latent]
    state["latent_std"] = torch.ones(128)  # [C_latent]
    state["min_range"] = torch.tensor(5.0)  # [scalar]
    state["max_range"] = torch.tensor(100.0)  # [scalar]
    projection = LidarRangeProjectionConfig(semantic_width=16, model_width=16)
    directory.mkdir()
    checkpoint = directory / f"weights{suffix}"
    if suffix == ".safetensors":
        save_file(state, checkpoint)
    else:
        torch.save({"model": state}, checkpoint)
    config = {
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "network_config": network,
        "range_projection": projection.to_dict(),
    }
    (directory / "config.json").write_text(json.dumps(config))
    sample = directory.parent / "recording.tar"
    _write_tar(sample)
    manifest = directory.parent / "recordings.txt"
    manifest.write_text("recording.tar\n")
    return checkpoint, manifest, network, projection


@pytest.mark.parametrize("suffix", [".pt", ".safetensors"])
def test_native_weights_reconstruction_and_pooled_metrics(tmp_path: Path, suffix: str) -> None:
    checkpoint, manifest, network, projection = _artifact(tmp_path / "artifact", suffix)
    scores = evaluate_framework_tokenizer(tmp_path / "artifact", manifest, tmp_path / "scored", "cpu", 64)

    # Independently reconstruct through the public interface and reduce over all GT-valid rays.
    direct = LidarTokenizerV1Interface(
        vae_path=str(checkpoint),
        device="cpu",
        dtype=torch.float32,
        network_config=network,
        range_projection=projection,
        sample_posterior=False,
    )
    video, frame_ids = load_prepared_lidar_video_3ch(
        tmp_path / "recording.tar",
        min_source_frame_id=0,
        max_source_frame_id=3,
        prepend_first_frame=True,
        range_projection=projection,
    )  # video: [3,3,128,16]; frame_ids: list[int]
    video, frame_ids = enforce_tokenizer_clip_length(video, frame_ids, num_source_frames=8)  # [3,9,128,16], list[int]
    with torch.inference_mode():
        latent = direct.encode(video.unsqueeze(0))  # [1,128,9,8,1]
        decoded = direct.decode(latent, apply_mask=False)  # [1,3,9,128,16]
        assert isinstance(decoded, torch.Tensor)
        predicted = decoded[0, :, 1:]  # [3,8,128,16]
    target, valid = video[0, 1:], video[2, 1:].bool()  # each [8,128,16]
    predicted_valid = predicted[2] >= projection.validity_threshold  # [8,128,16]
    mae = (predicted[0] - target).abs()[valid].double().mean().item()  # [scalar]
    intersection = int((valid & predicted_valid).sum())  # [scalar]
    union = int((valid | predicted_valid).sum())  # [scalar]
    assert scores["micro"]["corpus"]["range_mae_m"] == pytest.approx(mae, abs=2e-6)
    assert scores["micro"]["corpus"]["validity_iou"] == pytest.approx(intersection / union)
    assert scores["sources"][0]["repeated_last_sweeps"] == 6
    assert scores["sources"][0]["scored_source_sweeps"] == [0, 3, 3, 3, 3, 3, 3, 3]
    assert scores["macro"]["frames"] == 8
    with np.load(tmp_path / "scored/00000.reference.npz") as reference:
        np.testing.assert_array_equal(reference["sweep_ids"], frame_ids)
        measured = reference["range_m"][:, 64, 4]  # [8]
        np.testing.assert_allclose(measured, [25, 30, 30, 30, 30, 30, 30, 30])


def test_weight_checksum_mismatch_fails_before_model_loading(tmp_path: Path) -> None:
    checkpoint, manifest, _, _ = _artifact(tmp_path / "artifact", ".pt")
    checkpoint.write_bytes(b"invalid weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate_framework_tokenizer(tmp_path / "artifact", manifest, tmp_path / "scored", "cpu")


def test_resize_projection_is_rejected_before_model_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tokenizer.pt"
    checkpoint.write_bytes(b"not loaded")
    projection = LidarRangeProjectionConfig(
        semantic_width=1200,
        model_width=1024,
        model_width_transform="resize",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "range_projection": projection.to_dict(),
            }
        )
    )
    with pytest.raises(ValueError, match="resize canvases are unsupported"):
        evaluate_framework_tokenizer(tmp_path, tmp_path / "unused.txt", tmp_path / "scored", "cpu")
