# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the in-memory synthetic preference dataset (the quick-start toy path).

The dataset replaced the entry-level fallback that wrote synthetic files to the shared
out_dir under rank-0/others-wait coordination — determinism across independent
constructions is the property that makes the file-free design correct (every rank
builds identical data with no cross-rank handshake)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.data.datasets.synthetic_preference import SyntheticPreferenceDataset


@pytest.mark.L0
@pytest.mark.CPU
def test_contract_and_shapes_video():
    ds = SyntheticPreferenceDataset(n_pairs=4, resolution="256", num_frames=9, media="video")
    assert len(ds) == 4
    prompt, winner, loser = ds[0]
    assert prompt == "a synthetic sample 0"
    w = winner["video"][0]
    assert w.dtype == torch.uint8 and w.dim() == 4 and w.shape[1] == 9  # [C,T,H,W]
    assert w.shape == loser["video"][0].shape
    assert not torch.equal(w, loser["video"][0])  # distinct sides


@pytest.mark.L0
@pytest.mark.CPU
def test_contract_image():
    ds = SyntheticPreferenceDataset(n_pairs=2, resolution="256", num_frames=1, media="image")
    _, winner, _ = ds[1]
    img = winner["images"][0]
    assert img.dtype == torch.uint8 and img.dim() == 3  # [C,H,W]


@pytest.mark.L0
@pytest.mark.CPU
def test_deterministic_across_independent_constructions():
    """Two independent instances (≙ two ranks, no coordination) yield bit-identical
    data — the invariant the no-files design rests on."""
    a = SyntheticPreferenceDataset(n_pairs=3, resolution=None, num_frames=5, cfg_dropout_rate=0.0)
    b = SyntheticPreferenceDataset(n_pairs=3, resolution=None, num_frames=5, cfg_dropout_rate=0.0)
    for i in range(3):
        pa, wa, la = a[i]
        pb, wb, lb = b[i]
        assert pa == pb
        assert torch.equal(wa["video"][0], wb["video"][0])
        assert torch.equal(la["video"][0], lb["video"][0])
    # Different data seed ⇒ different frames.
    c = SyntheticPreferenceDataset(n_pairs=3, resolution=None, num_frames=5, seed=7, cfg_dropout_rate=0.0)
    assert not torch.equal(a[0][1]["video"][0], c[0][1]["video"][0])


@pytest.mark.L0
@pytest.mark.CPU
def test_build_env_fails_loud_without_data():
    from types import SimpleNamespace

    from cosmos_framework.posttrain.entrypoints.dpo import build_env

    data = SimpleNamespace(
        synthetic_pairs=0,
        jsonl_path="",
        tar_path="",
        media="video",
        resolution=None,
        num_frames=5,
        cfg_dropout_rate=0.0,
        shuffle=True,
        seed=0,
        cursor_resume="auto",
        caption_key="prompt",
        min_score_gap=0.0,
        min_wm_gap=0.0,
        min_align_gap=0.0,
    )
    with pytest.raises(ValueError, match="no preference data configured"):
        build_env(SimpleNamespace(data=data))
