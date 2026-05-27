# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for :mod:`cosmos3.scripts.eval_utils` aggregation and score-only semantics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from cosmos_framework.inference.scripts.eval_utils import (
    aggregate_metrics,
    compute_video_metrics,
    derive_match_key_and_group,
)

# ``cosmos3.scripts.eval`` calls ``init_script()`` at import time, which raises
# if ``imaginaire`` is already loaded. The package-level ``conftest.py`` loads
# ``imaginaire.lazy_config`` during collection, so by the time this test module
# is imported, the strict check would fire. We patch the underlying
# ``_init_script`` to a no-op for the rest of this test module — the real
# init work (env-var setup, error handlers, grad disable) is irrelevant for
# unit tests that don't actually run inference.
with patch("cosmos3.common.init._init_script", lambda **kwargs: None):
    if "cosmos3.scripts.eval" in sys.modules:
        del sys.modules["cosmos3.scripts.eval"]
    from cosmos_framework.inference.dataset import DatasetArgs  # noqa: E402
    from cosmos_framework.inference.scripts.eval import (
        EvalArgs,  # noqa: E402
        eval_vision,  # noqa: E402
    )

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


# ---------------------------------------------------------------------------
# aggregate_metrics — mean / count for every scalar metric
# ---------------------------------------------------------------------------


def _write_metrics(tmp_path: Path, name: str, mode: str, values: dict) -> None:
    d = tmp_path / mode / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"mode": mode, "name": name, **values}))


def test_aggregate_metrics_empty_dir_returns_empty(tmp_path):
    assert aggregate_metrics(tmp_path) == {}


def test_aggregate_metrics_single_sample(tmp_path):
    _write_metrics(tmp_path, "s0", "vision", {"psnr": 20.0})
    out = aggregate_metrics(tmp_path)
    assert out == {"vision": {"psnr": {"mean": 20.0, "count": 1}}}


def test_aggregate_metrics_mean_correct(tmp_path):
    for i, v in enumerate([10.0, 20.0, 30.0]):
        _write_metrics(tmp_path, f"s{i}", "vision", {"psnr": v})
    out = aggregate_metrics(tmp_path)["vision"]["psnr"]
    assert out["count"] == 3
    assert math.isclose(out["mean"], 20.0)


def test_aggregate_metrics_skips_files_without_mode(tmp_path):
    d = tmp_path / "orphan"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"name": "x", "psnr": 99.0}))
    _write_metrics(tmp_path, "s0", "vision", {"psnr": 20.0})
    out = aggregate_metrics(tmp_path)
    assert set(out.keys()) == {"vision"}
    assert out["vision"]["psnr"]["count"] == 1


def test_aggregate_metrics_separates_modes(tmp_path):
    _write_metrics(tmp_path, "s0", "vision", {"psnr": 20.0})
    _write_metrics(tmp_path, "s1", "forward_dynamics", {"psnr": 24.0})
    out = aggregate_metrics(tmp_path)
    assert set(out.keys()) == {"vision", "forward_dynamics"}
    assert out["vision"]["psnr"]["mean"] == 20.0
    assert out["forward_dynamics"]["psnr"]["mean"] == 24.0


def test_aggregate_metrics_flattens_nested_dicts(tmp_path):
    # nested dicts (e.g. grouped action_mse) flatten to "k/sub_k"
    _write_metrics(tmp_path, "s0", "policy", {"group_mse": {"arm": 0.1, "gripper": 0.2}})
    _write_metrics(tmp_path, "s1", "policy", {"group_mse": {"arm": 0.3, "gripper": 0.4}})
    out = aggregate_metrics(tmp_path)["policy"]
    assert "group_mse/arm" in out and "group_mse/gripper" in out
    assert math.isclose(out["group_mse/arm"]["mean"], 0.2)
    assert math.isclose(out["group_mse/gripper"]["mean"], 0.3)


# ---------------------------------------------------------------------------
# compute_video_metrics — vision lenient T-trim vs strict (action) on mismatch
# ---------------------------------------------------------------------------


def _write_synthetic_mp4(path: Path, frames_cthw_uint8: torch.Tensor, fps: int = 5) -> None:
    """Write a (C, T, H, W) uint8 tensor as an mp4 via torchvision.

    Lossy encoding will shift pixel values slightly; tests assert structural
    properties (shape, presence of metrics) rather than exact PSNR.
    """
    import torchvision.io as tvio

    # write_video expects (T, H, W, C) uint8
    thwc = frames_cthw_uint8.permute(1, 2, 3, 0).contiguous()
    tvio.write_video(str(path), thwc, fps=fps)


def test_compute_video_metrics_vision_lenient_trims_to_min_t(tmp_path, caplog):
    """VFM mode: pred has fewer frames than GT → trim both to min(T), warn, return metrics."""
    g = torch.Generator().manual_seed(0)
    gt = torch.randint(0, 256, (3, 8, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with caplog.at_level("WARNING"):
        metrics = compute_video_metrics(gt, pred_path, mode="vision")

    assert "psnr" in metrics


def test_compute_video_metrics_vision_no_warning_on_matching_shapes(tmp_path, caplog):
    """VFM mode: matching shapes → no warning, full metrics."""
    g = torch.Generator().manual_seed(1)
    gt = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with caplog.at_level("WARNING"):
        metrics = compute_video_metrics(gt, pred_path, mode="vision")

    assert "psnr" in metrics
    assert "trimmed to" not in caplog.text.lower()


def test_compute_video_metrics_action_strict_on_t_mismatch(tmp_path):
    """forward_dynamics: T mismatch still raises ValueError (the action chunk size is fixed)."""
    g = torch.Generator().manual_seed(2)
    gt = torch.randint(0, 256, (3, 8, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_video_metrics(gt, pred_path, mode="forward_dynamics")


def test_compute_video_metrics_vision_spatial_mismatch_still_errors(tmp_path):
    """Even in vision mode, an H mismatch that survives the top-left crop is a hard error."""
    g = torch.Generator().manual_seed(3)
    gt = torch.randint(0, 256, (3, 5, 16, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 8, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with pytest.raises(ValueError, match="spatial mismatch"):
        compute_video_metrics(gt, pred_path, mode="vision")


# ---------------------------------------------------------------------------
# derive_match_key_and_group — generic path-structure-based pairing rule
# ---------------------------------------------------------------------------


def test_derive_match_key_and_group_user_tree_cosmos_nano(tmp_path):
    """Tree 1: <root>/cosmos_nano_t2w/episode_*/vision.mp4 → key=episode_*, group=cosmos_nano_t2w."""
    p = tmp_path / "cosmos_nano_t2w" / "episode_002345_clip000" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "episode_002345_clip000"
    assert group == "cosmos_nano_t2w"


def test_derive_match_key_and_group_user_tree_mixed_modality(tmp_path):
    """Tree 2: <root>/mixed_modality_*/t2v/episode_*/vision.mp4 → group=mixed_modality_*/t2v."""
    p = tmp_path / "mixed_modality_sft_8b_0507e" / "t2v" / "episode_002345_clip000" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "episode_002345_clip000"
    assert group == "mixed_modality_sft_8b_0507e/t2v"


def test_derive_match_key_and_group_flat_layout(tmp_path):
    """Flat: <root>/<key>/vision.mp4 → key=<key>, group empty string."""
    p = tmp_path / "clip0" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "clip0"
    assert group == ""


def test_derive_match_key_and_group_inference_py_output(tmp_path):
    """Canonical inference.py output: <output_dir>/<sample.name>/vision.mp4."""
    p = tmp_path / "t2v" / "episode_049683_clip000" / "vision.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "episode_049683_clip000"
    assert group == "t2v"


def test_derive_match_key_and_group_non_vision_filename_uses_stem(tmp_path):
    """If basename isn't vision.*, the filename stem becomes the key (no parent-dir drop)."""
    p = tmp_path / "sub" / "foo.mp4"
    p.parent.mkdir(parents=True)
    p.touch()
    key, group = derive_match_key_and_group(p, tmp_path)
    assert key == "foo"
    assert group == "sub"


def test_derive_match_key_and_group_rejects_path_outside_predictions_dir(tmp_path):
    other = tmp_path.parent / "elsewhere" / "vision.mp4"
    with pytest.raises(ValueError, match="not under predictions_dir"):
        derive_match_key_and_group(other, tmp_path)


# ---------------------------------------------------------------------------
# score_only end-to-end (no model, CPU)
# ---------------------------------------------------------------------------


def test_score_only_end_to_end(tmp_path, monkeypatch):
    """Build a synthetic GT dir + predictions tree, run score_only, assert sidecars + aggregate."""
    from cosmos_framework.inference.args import OmniSetupOverrides

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"
    gt_dir.mkdir()
    pred_dir.mkdir()
    out_dir.mkdir()

    # Three clips; T=5, 32x32.
    keys = ["clipA", "clipB", "clipC"]
    g = torch.Generator().manual_seed(0)
    for k in keys:
        frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
        _write_synthetic_mp4(gt_dir / f"{k}.mp4", frames)
        bucket = pred_dir / "model_x" / "t2v" / k
        bucket.mkdir(parents=True)
        # Pred = lightly-perturbed GT so PSNR is sane (codec lossy regardless).
        pred = (frames.float() + 4).clamp(0, 255).to(torch.uint8)
        _write_synthetic_mp4(bucket / "vision.mp4", pred)

    args = EvalArgs(
        setup=OmniSetupOverrides(output_dir=out_dir, checkpoint_path=""),
        dataset=DatasetArgs(model_mode="vision"),
        gt_dir=gt_dir,
        predictions_dir=pred_dir,
        predictions_glob="**/vision.mp4",
    )
    eval_vision(args)

    # Per-sample sidecars.
    for k in keys:
        m = json.loads((out_dir / "model_x/t2v" / k / "metrics.json").read_text())
        assert m["mode"] == "model_x/t2v"
        assert m["name"] == k
        assert "psnr" in m

    # Aggregate.
    agg = json.loads((out_dir / "metrics_aggregate.json").read_text())
    assert "model_x/t2v" in agg
    for metric in ("psnr",):
        entry = agg["model_x/t2v"][metric]
        assert entry["count"] == 3
        assert "mean" in entry


def test_score_only_missing_gt_logs_warning_and_skips(tmp_path, caplog):
    from cosmos_framework.inference.args import OmniSetupOverrides

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"
    gt_dir.mkdir()
    pred_dir.mkdir()
    out_dir.mkdir()

    g = torch.Generator().manual_seed(1)
    frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    # GT only has clipA; pred has both clipA + clipZ.
    _write_synthetic_mp4(gt_dir / "clipA.mp4", frames)
    for k in ("clipA", "clipZ"):
        d = pred_dir / "m" / k
        d.mkdir(parents=True)
        _write_synthetic_mp4(d / "vision.mp4", frames)

    args = EvalArgs(
        setup=OmniSetupOverrides(output_dir=out_dir, checkpoint_path=""),
        dataset=DatasetArgs(model_mode="vision"),
        gt_dir=gt_dir,
        predictions_dir=pred_dir,
        predictions_glob="**/vision.mp4",
    )
    with caplog.at_level("WARNING"):
        eval_vision(args)

    # clipA scored, clipZ skipped.
    assert (out_dir / "m" / "clipA" / "metrics.json").exists()
    assert not (out_dir / "m" / "clipZ" / "metrics.json").exists()
    agg = json.loads((out_dir / "metrics_aggregate.json").read_text())
    assert agg["m"]["psnr"]["count"] == 1


def test_score_only_single_mode_bucket(tmp_path):
    """Predictions under one mode subdir → one bucket in aggregate (no t2v/i2v/v2v assumption)."""
    from cosmos_framework.inference.args import OmniSetupOverrides

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"
    gt_dir.mkdir()
    pred_dir.mkdir()
    out_dir.mkdir()

    keys = ["clipA", "clipB"]
    g = torch.Generator().manual_seed(0)
    for k in keys:
        frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
        _write_synthetic_mp4(gt_dir / f"{k}.mp4", frames)
        d = pred_dir / "t2v" / k
        d.mkdir(parents=True)
        _write_synthetic_mp4(d / "vision.mp4", (frames.float() + 4).clamp(0, 255).to(torch.uint8))

    args = EvalArgs(
        setup=OmniSetupOverrides(output_dir=out_dir, checkpoint_path=""),
        dataset=DatasetArgs(model_mode="vision"),
        gt_dir=gt_dir,
        predictions_dir=pred_dir,
        predictions_glob="**/vision.mp4",
    )
    eval_vision(args)

    agg = json.loads((out_dir / "metrics_aggregate.json").read_text())
    assert set(agg.keys()) == {"t2v"}
    assert agg["t2v"]["psnr"]["count"] == 2


def test_score_only_flat_layout_uses_default_bucket(tmp_path):
    """Flat ``<pred_dir>/<key>/vision.mp4`` (no mode subfolder) → bucket=='default'."""
    from cosmos_framework.inference.args import OmniSetupOverrides

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"
    gt_dir.mkdir()
    pred_dir.mkdir()
    out_dir.mkdir()

    keys = ["clipA", "clipB"]
    g = torch.Generator().manual_seed(0)
    for k in keys:
        frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
        _write_synthetic_mp4(gt_dir / f"{k}.mp4", frames)
        d = pred_dir / k
        d.mkdir(parents=True)
        _write_synthetic_mp4(d / "vision.mp4", (frames.float() + 4).clamp(0, 255).to(torch.uint8))

    args = EvalArgs(
        setup=OmniSetupOverrides(output_dir=out_dir, checkpoint_path=""),
        dataset=DatasetArgs(model_mode="vision"),
        gt_dir=gt_dir,
        predictions_dir=pred_dir,
        predictions_glob="**/vision.mp4",
    )
    eval_vision(args)

    agg = json.loads((out_dir / "metrics_aggregate.json").read_text())
    assert set(agg.keys()) == {"default"}
    assert agg["default"]["psnr"]["count"] == 2
