"""Regression checks for the bundled PAI-Bench-C scorer patch.

Point ``PAIBENCH_EVAL_ROOT`` at a ``physical-ai-bench/conditional_generation`` checkout
with ``patches/physical-ai-bench-metric-alignment.patch`` applied (``run_paibench_c.sh``
creates one) and run ``python -m pytest tests/`` from this directory with an environment
that has the scorer's dependencies (the checkout's own ``.venv`` after ``uv sync`` works).
``tests/pytest.ini`` keeps the run self-contained. The module is skipped when the checkout
or its dependencies (torch, cv2, scipy, pycocotools, attrs, cattrs) are missing.
"""

import os
import pathlib
import sys

import pytest

ROOT = os.environ.get("PAIBENCH_EVAL_ROOT", "")
if not ROOT or not (pathlib.Path(ROOT) / "benchmark_pipelines").is_dir():
    pytest.skip(
        "set PAIBENCH_EVAL_ROOT to a patched physical-ai-bench/conditional_generation checkout",
        allow_module_level=True,
    )
sys.path.insert(0, ROOT)
np = pytest.importorskip("numpy")
pytest.importorskip("torch")
seg = pytest.importorskip("benchmark_pipelines.scores.transfer_bench.segmentation_metrics")
schema = pytest.importorskip("schemas.eff_segmentation")

ONES = np.ones((2, 4, 4), dtype=np.uint8)
ZEROS = np.zeros((2, 4, 4), dtype=np.uint8)


def _det(mask, phrase="car"):
    return schema.SAMV2Detection(phrase=phrase, segmentation_mask_rle=schema.RleMaskSAMv2.encode(mask))


@pytest.mark.parametrize("matching", ["max", "hungarian"])
@pytest.mark.parametrize("empty", ["pred", "gt", "both"])
def test_empty_tracked_masks_report_no_match(matching, empty):
    # A valid detection whose tracked mask is all zeros is filtered out before matching;
    # the metric must report an explicit no-match result rather than fail.
    gt = ZEROS if empty in ("gt", "both") else ONES
    pred = ZEROS if empty in ("pred", "both") else ONES
    assert seg.calculate_mask_iou_and_recall([_det(gt)], [_det(pred)], matching=matching) == (0.0, 0.0)


@pytest.mark.parametrize("matching", ["max", "hungarian"])
def test_identical_masks_match(matching):
    assert seg.calculate_mask_iou_and_recall([_det(ONES)], [_det(ONES)], matching=matching) == (1.0, 1.0)


@pytest.mark.parametrize("matching", ["max", "hungarian"])
def test_untracked_gt_phrase_is_dropped_before_matching(matching):
    # One GT phrase is never tracked: the filter drops it and the remaining pair still matches.
    result = seg.calculate_mask_iou_and_recall(
        [_det(ONES, "car"), _det(ZEROS, "bike")], [_det(ONES, "car")], matching=matching
    )
    assert result == (1.0, 1.0)
