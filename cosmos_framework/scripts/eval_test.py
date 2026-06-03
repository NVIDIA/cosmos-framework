# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for cosmos_framework.scripts.eval.

Run on a node with the full env (needs torch/torchvision):
    CUDA_VISIBLE_DEVICES="" uv run python -m pytest -q --noconftest -p no:cacheprovider \
        cosmos_framework/scripts/eval_test.py

The metric tests operate on tensors directly and the file-IO path is exercised by
monkeypatching ``read_media_frames`` — so nothing here depends on the (lossy) mp4 codec.
"""

import json

import pytest
import torch

from cosmos_framework.scripts import eval as evalmod


def _clip(seed: int = 0, c: int = 3, t: int = 8, h: int = 16, w: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (c, t, h, w), generator=g, dtype=torch.uint8)


# ---- metrics ----------------------------------------------------------------


def test_psnr_identical_is_capped():
    x = _clip()
    assert evalmod.compute_psnr(x, x) == 100.0


def test_psnr_decreases_with_noise():
    x = _clip()
    noisy = (x.to(torch.int16) + 20).clamp(0, 255).to(torch.uint8)
    psnr = evalmod.compute_psnr(x, noisy)
    assert psnr < 100.0
    assert psnr > evalmod.compute_psnr(x, (x.to(torch.int16) + 60).clamp(0, 255).to(torch.uint8))


def test_ssim_identical_is_one():
    x = _clip()
    assert evalmod.compute_ssim(x, x) == pytest.approx(1.0, abs=1e-4)


def test_ssim_decreases_with_noise():
    x = _clip()
    noisy = (x.to(torch.int16) + 40).clamp(0, 255).to(torch.uint8)
    assert evalmod.compute_ssim(x, noisy) < evalmod.compute_ssim(x, x)


# ---- compute_video_metrics alignment (monkeypatched IO) ---------------------


def test_video_metrics_temporal_trim_no_error(monkeypatch):
    gt = _clip(t=8)
    # pred has one extra frame (4k+1 vs raw); first 8 frames identical → trims to 8 → PSNR capped.
    pred = torch.cat([gt, gt[:, :1]], dim=1)
    monkeypatch.setattr(evalmod, "read_media_frames", lambda path, max_frames: (pred[:, :max_frames], 5.0))
    m = evalmod.compute_video_metrics(gt, pred_path=evalmod.Path("ignored.mp4"))
    assert m["psnr"] == 100.0
    assert m["ssim"] == pytest.approx(1.0, abs=1e-4)


def test_video_metrics_spatial_mismatch_raises(monkeypatch):
    gt = _clip(h=16, w=16)
    pred_small = _clip(seed=1, h=8, w=8)  # smaller than GT → top-left crop can't fix → hard error
    monkeypatch.setattr(evalmod, "read_media_frames", lambda path, max_frames: (pred_small, 5.0))
    with pytest.raises(ValueError, match="spatial mismatch"):
        evalmod.compute_video_metrics(gt, pred_path=evalmod.Path("ignored.mp4"))


def test_video_metrics_larger_pred_is_cropped(monkeypatch):
    gt = _clip(h=16, w=16)
    # pred padded on the right/bottom; top-left 16x16 equals GT → crop → identical.
    pred = torch.nn.functional.pad(gt, (0, 4, 0, 4))  # pad W then H → (C,T,20,20)
    monkeypatch.setattr(evalmod, "read_media_frames", lambda path, max_frames: (pred, 5.0))
    assert evalmod.compute_video_metrics(gt, pred_path=evalmod.Path("ignored.mp4"))["psnr"] == 100.0


# ---- pairing ----------------------------------------------------------------


def test_derive_match_key_and_group_inference_layout(tmp_path):
    p = tmp_path / "t2v" / "episode_0" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    assert evalmod.derive_match_key_and_group(p, tmp_path) == ("episode_0", "t2v")


def test_derive_match_key_and_group_plain_file(tmp_path):
    p = tmp_path / "sub" / "foo.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    assert evalmod.derive_match_key_and_group(p, tmp_path) == ("foo", "sub")


def test_derive_match_key_and_group_not_under_root_raises(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    p = tmp_path / "preds" / "t2v" / "ep" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    with pytest.raises(ValueError, match="not under"):
        evalmod.derive_match_key_and_group(p, other)


# ---- aggregation ------------------------------------------------------------


def test_aggregate_metrics_per_mode(tmp_path):
    def _write(mode, name, psnr, ssim):
        d = tmp_path / mode / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"mode": mode, "name": name, "psnr": psnr, "ssim": ssim}))

    _write("t2v", "a", 10.0, 0.4)
    _write("t2v", "b", 20.0, 0.6)
    _write("i2v", "a", 30.0, 0.9)
    agg = evalmod.aggregate_metrics(tmp_path)
    assert agg["t2v"]["psnr"] == {"mean": 15.0, "count": 2}
    assert agg["t2v"]["ssim"]["mean"] == pytest.approx(0.5)
    assert agg["i2v"]["psnr"] == {"mean": 30.0, "count": 1}


# ---- comparison -------------------------------------------------------------


def test_format_comparison_delta_and_winner():
    baseline = {"i2v": {"psnr": {"mean": 18.0, "count": 5}}}  # dense
    current = {"i2v": {"psnr": {"mean": 20.0, "count": 5}}}  # json
    md = evalmod.format_comparison(baseline, current, baseline_label="dense", current_label="json")
    assert "+2.0000" in md
    # current (json) is higher → its label wins that row
    row = [ln for ln in md.splitlines() if ln.startswith("| i2v ")][0]
    assert row.rstrip().endswith("| json |")
