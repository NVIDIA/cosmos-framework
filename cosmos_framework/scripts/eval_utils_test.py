# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for :mod:`cosmos_framework.scripts.eval_utils` aggregation and score-only semantics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import av
import pytest
import torch

from cosmos_framework.scripts.eval_utils import (
    _compute_action_metrics,
    aggregate_metrics,
    compute_video_metrics,
    derive_match_key_and_group,
    extract_gt_action,
    extract_gt_video,
)

# ``cosmos_framework.scripts.eval`` calls ``init_script()`` at import time, which
# initializes distributed state, error handlers and console logging and disables
# autograd for the whole process. Patch the underlying ``_init_script`` to a
# no-op while importing it: none of that is needed by unit tests that do not run
# inference.
with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    if "cosmos_framework.scripts.eval" in sys.modules:
        del sys.modules["cosmos_framework.scripts.eval"]
    from cosmos_framework.inference.dataset import DatasetArgs  # noqa: E402
    from cosmos_framework.scripts.eval import (
        EvalArgs,  # noqa: E402
        eval_vision,  # noqa: E402
    )

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_extract_gt_video_matches_latent_spatial_alignment() -> None:
    video = torch.zeros((1, 3, 5, 480, 832), dtype=torch.uint8)
    image_size = [torch.tensor([[480.0, 832.0, 360.0, 640.0]])]

    extracted = extract_gt_video(
        {"video": video, "image_size": image_size},
        spatial_compression_factor=16,
    )

    assert extracted is not None
    assert extracted.shape == (3, 5, 352, 640)


def test_extract_gt_action_uses_canonical_raw_action() -> None:
    action_raw = torch.tensor([[[-2.0, 0.5], [2.0, -0.5]]])  # [B,T,D_raw]
    action_model = action_raw.clamp(-1.0, 1.0)  # [B,T,D_model]

    extracted = extract_gt_action({"action": action_model, "action_raw": action_raw})

    assert extracted is not None
    extracted_action, valid_mask = extracted
    torch.testing.assert_close(extracted_action, action_raw[0].float())
    assert valid_mask is None


def test_action_mse_averages_only_valid_unified_slots() -> None:
    gt_action = torch.zeros(2, 4)
    pred_action = [[1.0, 0.0, 3.0, 0.0], [1.0, 0.0, 3.0, 0.0]]
    valid_mask = torch.tensor([True, False, True, False])

    metrics = _compute_action_metrics(gt_action, pred_action, valid_mask)

    assert metrics["action_mse"] == 5.0


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
    _write_metrics(tmp_path, "s0", "wam", {"group_mse": {"arm": 0.1, "gripper": 0.2}})
    _write_metrics(tmp_path, "s1", "wam", {"group_mse": {"arm": 0.3, "gripper": 0.4}})
    out = aggregate_metrics(tmp_path)["wam"]
    assert "group_mse/arm" in out and "group_mse/gripper" in out
    assert math.isclose(out["group_mse/arm"]["mean"], 0.2)
    assert math.isclose(out["group_mse/gripper"]["mean"], 0.3)


# ---------------------------------------------------------------------------
# compute_video_metrics — vision lenient T-trim vs strict (action) on mismatch
# ---------------------------------------------------------------------------


def _write_synthetic_mp4(path: Path, frames_cthw_uint8: torch.Tensor, fps: int = 5) -> None:
    """Write a (C, T, H, W) uint8 tensor as an H.264 mp4 via PyAV.

    Lossy encoding will shift pixel values slightly; tests assert structural
    properties (shape, presence of metrics) rather than exact PSNR.
    """
    thwc = frames_cthw_uint8.permute(1, 2, 3, 0).contiguous().numpy()  # [T,H,W,C]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = thwc.shape[2], thwc.shape[1], "yuv420p"
        for pixels in thwc:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def requires_torchcodec() -> None:
    """Skip tests that decode videos when torchcodec cannot load (``read_media_frames`` uses it)."""
    try:
        import torchcodec.decoders  # noqa: F401
    except Exception as error:  # ImportError, or RuntimeError when FFmpeg's libraries do not load
        pytest.skip(f"torchcodec does not load here ({type(error).__name__})")


def test_compute_video_metrics_vision_lenient_trims_to_min_t(tmp_path, caplog, requires_torchcodec):
    """VFM mode: pred has fewer frames than GT → trim both to min(T), warn, return metrics."""
    g = torch.Generator().manual_seed(0)
    gt = torch.randint(0, 256, (3, 8, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with caplog.at_level("WARNING"):
        metrics = compute_video_metrics(gt, pred_path, mode="vision")

    assert "psnr" in metrics


def test_compute_video_metrics_vision_no_warning_on_matching_shapes(tmp_path, caplog, requires_torchcodec):
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


def test_compute_video_metrics_action_strict_on_t_mismatch(tmp_path, requires_torchcodec):
    """forward_dynamics: T mismatch still raises ValueError (the action chunk size is fixed)."""
    g = torch.Generator().manual_seed(2)
    gt = torch.randint(0, 256, (3, 8, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_video_metrics(gt, pred_path, mode="forward_dynamics")


def test_compute_video_metrics_spatial_mismatch_still_errors(tmp_path, requires_torchcodec):
    """GT extraction must align spatial dimensions before metric computation."""
    g = torch.Generator().manual_seed(3)
    gt = torch.randint(0, 256, (3, 5, 24, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_frames = torch.randint(0, 256, (3, 5, 16, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    pred_path = tmp_path / "vision.mp4"
    _write_synthetic_mp4(pred_path, pred_frames)

    with pytest.raises(ValueError, match="spatial mismatch"):
        compute_video_metrics(gt, pred_path, mode="vision")


# ---------------------------------------------------------------------------
# derive_match_key_and_group — path-structure-based pairing rule
# ---------------------------------------------------------------------------


def test_derive_match_key_and_group_inference_py_output(tmp_path):
    """Canonical inference.py output: <output_dir>/<sample.name>/<output_index>/vision.mp4."""
    p = tmp_path / "t2v" / "episode_049683_clip000" / "0" / "vision.mp4"
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


def test_score_only_end_to_end(tmp_path, monkeypatch, requires_torchcodec):
    """Build a synthetic GT dir + predictions tree, run score_only, assert sidecars + aggregate."""
    from cosmos_framework.inference.args import OmniSetupOverrides

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"
    gt_dir.mkdir()
    pred_dir.mkdir()
    out_dir.mkdir()

    # Three samples and four outputs; T=5, 32x32.
    keys = ["clipA", "123", "clipC"]
    g = torch.Generator().manual_seed(0)
    for k in keys:
        frames = torch.randint(0, 256, (3, 5, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
        _write_synthetic_mp4(gt_dir / f"{k}.mp4", frames)
        bucket = pred_dir / "model_x" / "t2v" / k / "0"
        bucket.mkdir(parents=True)
        # Pred = lightly-perturbed GT so PSNR is sane (codec lossy regardless).
        pred = (frames.float() + 4).clamp(0, 255).to(torch.uint8)
        _write_synthetic_mp4(bucket / "vision.mp4", pred)
        if k == "clipA":
            second_output = bucket.parent / "1"
            second_output.mkdir()
            _write_synthetic_mp4(second_output / "vision.mp4", pred)

    args = EvalArgs(
        setup=OmniSetupOverrides(output_dir=out_dir, checkpoint_path=""),
        dataset=DatasetArgs(model_mode="vision"),
        gt_dir=gt_dir,
        predictions_dir=pred_dir,
        predictions_glob="**/vision.mp4",
    )
    eval_vision(args)

    # Per-sample sidecars.
    sidecars = [
        (out_dir / "model_x/t2v/clipA/0/metrics.json", "clipA"),
        (out_dir / "model_x/t2v/clipA/1/metrics.json", "clipA"),
        (out_dir / "model_x/t2v/123/0/metrics.json", "123"),
        (out_dir / "model_x/t2v/clipC/0/metrics.json", "clipC"),
    ]
    for sidecar, expected_name in sidecars:
        m = json.loads(sidecar.read_text())
        assert m["mode"] == "model_x/t2v"
        assert m["name"] == expected_name
        assert "psnr" in m

    # Aggregate.
    agg = json.loads((out_dir / "metrics_aggregate.json").read_text())
    assert "model_x/t2v" in agg
    for metric in ("psnr",):
        entry = agg["model_x/t2v"][metric]
        assert entry["count"] == 4
        assert "mean" in entry


def test_score_only_missing_gt_logs_warning_and_skips(tmp_path, caplog, requires_torchcodec):
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
        d = pred_dir / "m" / k / "0"
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
    assert (out_dir / "m" / "clipA" / "0" / "metrics.json").exists()
    assert not (out_dir / "m" / "clipZ" / "0" / "metrics.json").exists()
    agg = json.loads((out_dir / "metrics_aggregate.json").read_text())
    assert agg["m"]["psnr"]["count"] == 1


def test_score_only_single_mode_bucket(tmp_path, requires_torchcodec):
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
        d = pred_dir / "t2v" / k / "0"
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


def test_score_only_flat_layout_uses_default_bucket(tmp_path, requires_torchcodec):
    """No mode subfolder in ``<pred_dir>/<key>/<output_index>/vision.mp4`` → default bucket."""
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
        d = pred_dir / k / "0"
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
