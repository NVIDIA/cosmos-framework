# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""L0 tests for the LiDAR tokenizer checkpoint warm-start helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cosmos_framework.model.generator.tokenizers.lidar.checkpoint import (
    resize_azimuth_embedding,
    warm_start_network,
)


class _Net(nn.Module):
    """Stand-in for the parts of the TransformerVAE the warm start touches.

    ``tokenizer`` stands for the patch embedding whose channel count changes,
    ``spatial_pe.embedding`` for a range-view embedding whose azimuth width
    changes, and ``trunk`` for the bulk of the network that transfers as-is.
    """

    def __init__(self, *, in_channels: int, width: int) -> None:
        super().__init__()
        self.tokenizer = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=False)
        self.trunk = nn.Linear(4, 4)
        self.spatial_pe = nn.Module()
        self.spatial_pe.embedding = nn.Parameter(torch.zeros(1, 2, width, 4))


def _write_checkpoint(directory: Path, network: nn.Module) -> str:
    path = directory / "source.pt"
    torch.save({"model": network.state_dict()}, path)
    return str(path)


def _randomized(network: nn.Module) -> nn.Module:
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.copy_(torch.randn_like(parameter))
    return network


@pytest.mark.L0
@pytest.mark.CPU
def test_resize_azimuth_embedding_is_a_no_op_at_equal_width() -> None:
    embedding = torch.randn(1, 3, 8, 5)
    assert resize_azimuth_embedding(embedding, 8) is embedding


@pytest.mark.L0
@pytest.mark.CPU
def test_resize_azimuth_embedding_preserves_a_constant_signal() -> None:
    # A constant is the one signal any sane resampling must reproduce exactly,
    # wrap-around included.
    embedding = torch.full((1, 2, 16, 3), 0.75)
    resized = resize_azimuth_embedding(embedding, 29)
    assert resized.shape == (1, 2, 29, 3)
    torch.testing.assert_close(resized, torch.full((1, 2, 29, 3), 0.75))


@pytest.mark.L0
@pytest.mark.CPU
def test_resize_azimuth_embedding_hits_source_bins_on_an_odd_upsample() -> None:
    # Tripling places every third target center exactly on a source center
    # (target j = 1 + 3k maps to source bin k), so those columns must come back
    # untouched. Even ratios never coincide under the bin-center convention.
    embedding = torch.randn(1, 1, 6, 2)
    resized = resize_azimuth_embedding(embedding, 18)
    torch.testing.assert_close(resized[:, :, 1::3, :], embedding)


@pytest.mark.L0
@pytest.mark.CPU
def test_resize_azimuth_embedding_wraps_rather_than_clamping() -> None:
    # A ramp whose last bin is far from its first: clamping at the edge would
    # extend the ramp, whereas wrapping blends the two ends back together.
    embedding = torch.zeros(1, 1, 4, 1)
    embedding[0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    resized = resize_azimuth_embedding(embedding, 8)
    # Target center 7 sits at source 3.25, a quarter of the way from bin 3 back
    # round to bin 0, so 3 * 0.75 + 0 * 0.25.
    torch.testing.assert_close(resized[0, 0, 7, 0], torch.tensor(2.25))


@pytest.mark.L0
@pytest.mark.CPU
def test_resize_azimuth_embedding_rejects_a_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="target width must be positive"):
        resize_azimuth_embedding(torch.randn(1, 2, 4, 3), 0)


@pytest.mark.L0
@pytest.mark.CPU
def test_warm_start_applies_each_category(tmp_path: Path) -> None:
    source = _randomized(_Net(in_channels=2, width=8))
    checkpoint = _write_checkpoint(tmp_path, source)
    # Wider azimuth plus an extra input channel: the V0 -> V1.2 shift in
    # miniature.
    target = _Net(in_channels=3, width=14)
    stale_tokenizer = target.tokenizer.weight.detach().clone()

    report = warm_start_network(target, checkpoint, backend_args=None)

    # Shapes already agree, so the trunk transfers verbatim.
    assert report["copied"] == ["trunk.bias", "trunk.weight"]
    torch.testing.assert_close(target.trunk.weight, source.trunk.weight)
    # Differs only along azimuth, so it is resampled rather than dropped.
    assert report["resized"] == ["spatial_pe.embedding"]
    assert target.spatial_pe.embedding.shape == (1, 2, 14, 4)
    # Gained an input channel, so it keeps its fresh initialization.
    assert report["skipped"] == ["tokenizer.weight"]
    assert report["missing"] == ["tokenizer.weight"]
    torch.testing.assert_close(target.tokenizer.weight, stale_tokenizer)


@pytest.mark.L0
@pytest.mark.CPU
def test_warm_start_transfers_everything_when_shapes_already_agree(tmp_path: Path) -> None:
    source = _randomized(_Net(in_channels=3, width=14))
    checkpoint = _write_checkpoint(tmp_path, source)
    target = _Net(in_channels=3, width=14)

    report = warm_start_network(target, checkpoint, backend_args=None)

    assert report["resized"] == []
    assert report["skipped"] == []
    assert report["missing"] == []
    for key, tensor in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], tensor)


@pytest.mark.L0
@pytest.mark.CPU
def test_warm_start_accepts_a_bare_state_dict_without_a_model_key(tmp_path: Path) -> None:
    # V0 stores a training wrapper with a "model" key; published weights may be
    # a bare state dict.
    source = _randomized(_Net(in_channels=3, width=14))
    path = tmp_path / "bare.pt"
    torch.save(source.state_dict(), path)
    target = _Net(in_channels=3, width=14)

    report = warm_start_network(target, str(path), backend_args=None)

    assert report["missing"] == []
    torch.testing.assert_close(target.trunk.weight, source.trunk.weight)


@pytest.mark.L0
@pytest.mark.CPU
def test_warm_start_leaves_the_network_usable_when_nothing_matches(tmp_path: Path) -> None:
    path = tmp_path / "unrelated.pt"
    torch.save({"model": {"totally.unrelated": torch.randn(3, 3)}}, path)
    target = _Net(in_channels=3, width=14)

    report = warm_start_network(target, str(path), backend_args=None)

    # Reported as absent rather than raising, so a bad path cannot silently
    # look like a successful warm start.
    assert report["copied"] == []
    assert report["resized"] == []
    assert "trunk.weight" in report["missing"]


@pytest.mark.L0
@pytest.mark.CPU
def test_warm_start_rejects_a_payload_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(TypeError, match="Checkpoint must be a mapping"):
        warm_start_network(_Net(in_channels=3, width=14), str(path), backend_args=None)


@pytest.mark.L0
@pytest.mark.CPU
def test_query_string_url_still_routes_to_the_safetensors_loader(monkeypatch) -> None:
    """Assert the routing, not just the helper.

    Testing ``_artifact_suffix`` alone leaves the decision that uses it uncovered: reverting the
    call site to ``path.endswith(...)`` would send a presigned URL down the pickle path with
    every helper test still green.
    """
    from cosmos_framework.model.generator.tokenizers.lidar import checkpoint as ckpt

    taken: list[str] = []
    monkeypatch.setattr(ckpt, "_resolve_published_artifact", lambda path: path)
    monkeypatch.setattr(
        ckpt, "_load_safetensors_artifact", lambda path, backend_args=None: taken.append("safetensors") or {}
    )
    monkeypatch.setattr(ckpt.easy_io, "load", lambda *a, **k: taken.append("pickle") or {})

    ckpt.load_artifact("https://host/a/model.safetensors?X-Amz-Signature=abc")
    assert taken == ["safetensors"], taken

    taken.clear()
    ckpt.load_artifact("s3://bucket/checkpoints/iter_000030000.pt")
    assert taken == ["pickle"], taken


@pytest.mark.L0
@pytest.mark.CPU
def test_registered_artifact_download_failure_keeps_its_cause(monkeypatch) -> None:
    """A failed download must surface, not degrade into a misleading storage error.

    Catching it would hand back the ``s3://bucket/...`` registry alias, which the loader then
    tries to read as a real object-store address -- reporting a deserialization failure instead
    of the missing credential or network error behind it.
    """
    from cosmos_framework.model.generator.tokenizers.lidar import checkpoint as ckpt

    def _boom(uri, *, check_exists=True):
        raise RuntimeError("credential file not found")

    monkeypatch.setattr("cosmos_framework.utils.checkpoint_db.download_checkpoint_v2", _boom)
    with pytest.raises(RuntimeError, match="credential file not found"):
        ckpt._resolve_published_artifact("s3://bucket/pretrained/tokenizers/lidar/x.safetensors")
