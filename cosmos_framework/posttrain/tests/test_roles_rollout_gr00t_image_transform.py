# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The reference eval image transform: checkpoint-driven configuration (L0) and bit-for-bit parity
with Isaac-GR00T's own transform (L1).

``model/gr00t/serving/image_transform.py`` transcribes Isaac's albumentations pipeline rather than
importing it, so the L1 tests run the real Isaac transform beside ours and require exact equality.
They are L1 because importing Isaac-GR00T costs more than the L0 budget; no GPU is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cosmos_framework.posttrain.model.gr00t.serving.image_transform import Gr00tEvalImageTransform

# The DROID checkpoint's real values (processor_config.json -> processor_kwargs).
TARGET, CROP, EDGE, FRACTION = (256, 256), (230, 230), 256, 0.95
DROID_KWARGS: dict[str, Any] = {
    "use_albumentations": True,
    "image_target_size": list(TARGET),
    "image_crop_size": list(CROP),
    "shortest_image_edge": EDGE,
    "crop_fraction": FRACTION,
    "letter_box_transform": False,
}
_ABSENT = object()


class _Processor:
    """Stands in for the loaded processor.

    Keeps some keys with deliberately wrong values (the FILE must win) and lacks `use_albumentations`,
    as the real port object does.
    """

    image_target_size = [512, 512]
    crop_fraction = 0.5


def _checkpoint(tmp_path: Path, **overrides: Any) -> str:
    kwargs = {k: v for k, v in {**DROID_KWARGS, **overrides}.items() if v is not _ABSENT}
    (tmp_path / "processor_config.json").write_text(json.dumps({"processor_kwargs": kwargs}))
    return str(tmp_path)


# ── L0: configuration comes from the checkpoint file ─────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_reads_every_field_from_the_checkpoint_config(tmp_path: Path) -> None:
    transform = Gr00tEvalImageTransform.from_processor(_Processor(), model_path=_checkpoint(tmp_path))
    assert transform.image_target_size == (256, 256)
    assert transform.image_crop_size == (230, 230)
    assert transform.shortest_image_edge == 256
    assert transform.crop_fraction == 0.95
    assert transform.letter_box_transform is False
    assert transform.use_albumentations is True
    assert transform.name == "reference-albumentations"
    # The derived values that actually drive the pipeline: 0.95 of 256 is 243, NOT the port's 230.
    assert transform.fraction == 0.95
    assert transform.max_size == 256


@pytest.mark.L0
@pytest.mark.CPU
def test_the_file_wins_over_the_loaded_processor(tmp_path: Path) -> None:
    """The processor object is only a fallback; a stale attribute must not override the checkpoint."""
    transform = Gr00tEvalImageTransform.from_processor(_Processor(), model_path=_checkpoint(tmp_path))
    assert transform.image_target_size == (256, 256)  # not the processor's (512, 512)
    assert transform.crop_fraction == 0.95  # not the processor's 0.5


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_use_albumentations_raises_rather_than_defaulting(tmp_path: Path) -> None:
    """It selects between two pipelines that produce different pixels; the port drops it from the object."""
    with pytest.raises(KeyError, match="use_albumentations"):
        Gr00tEvalImageTransform.from_processor(
            _Processor(), model_path=_checkpoint(tmp_path, use_albumentations=_ABSENT)
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_torchvision_branch_is_named_distinctly(tmp_path: Path) -> None:
    """`describe_pipeline` reports this name, and the startup guard requires it to be present."""
    transform = Gr00tEvalImageTransform.from_processor(
        _Processor(), model_path=_checkpoint(tmp_path, use_albumentations=False)
    )
    assert transform.name == "reference-torchvision"


@pytest.mark.L0
@pytest.mark.CPU
def test_derived_fallbacks_match_isaacs_rules(tmp_path: Path) -> None:
    """With `crop_fraction`/`shortest_image_edge` unset, Isaac falls back to the size ratio."""
    transform = Gr00tEvalImageTransform.from_processor(
        _Processor(),
        model_path=_checkpoint(tmp_path, crop_fraction=None, shortest_image_edge=None),
    )
    assert transform.fraction == 230 / 256
    assert transform.max_size == 256


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_processor_kwargs_raises(tmp_path: Path) -> None:
    (tmp_path / "processor_config.json").write_text(json.dumps({"nothing": True}))
    with pytest.raises(KeyError, match="processor_kwargs"):
        Gr00tEvalImageTransform.from_processor(_Processor(), model_path=str(tmp_path))


@pytest.mark.L0
@pytest.mark.CPU
def test_rejects_a_non_uint8_frame(tmp_path: Path) -> None:
    """The port asserts uint8 downstream; failing here names the actual problem instead."""
    transform = Gr00tEvalImageTransform.from_processor(_Processor(), model_path=_checkpoint(tmp_path))
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        transform(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        transform(np.zeros((8, 8), dtype=np.uint8))


# ── L1: bit for bit against Isaac-GR00T ──────────────────────────────────────


def _images() -> dict[str, np.ndarray]:
    """Probe frames chosen for what each exposes: a gradient turns an off-by-one crop offset into a
    constant shift, a checkerboard is the worst case for area-vs-bilinear resampling, noise is the
    stress case, and a non-square 540x640 frame (robolab's shape) is the only one that exercises the
    letterbox."""
    rng = np.random.default_rng(0)
    ramp = np.linspace(0, 255, 224, dtype=np.float32)
    gradient = np.stack(
        [ramp[None, :].repeat(224, 0), ramp[:, None].repeat(224, 1), np.full((224, 224), 128.0)], -1
    ).astype(np.uint8)
    board = (((np.arange(224) // 8)[:, None] + (np.arange(224) // 8)[None, :]) % 2) * 255
    return {
        "gradient": gradient,
        "checkerboard": np.repeat(board[:, :, None], 3, axis=2).astype(np.uint8),
        "noise": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "robolab_540x640": rng.integers(0, 256, size=(540, 640, 3), dtype=np.uint8),
    }


def _isaac_eval_transform(letter_box: bool):
    """Isaac's own eval transform, built exactly as its factory builds it."""
    gr00t_aug = pytest.importorskip("gr00t.model.gr00t_n1d7.image_augmentations")
    _train, evaluation = gr00t_aug.build_image_transformations_albumentations(
        image_target_size=list(TARGET),
        image_crop_size=list(CROP),
        random_rotation_angle=0,
        color_jitter_params=None,
        shortest_image_edge=EDGE,
        crop_fraction=FRACTION,
        letter_box_transform=letter_box,
    )
    return gr00t_aug, evaluation


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.parametrize("letter_box", [False, True])
@pytest.mark.parametrize("name", ["gradient", "checkerboard", "noise", "robolab_540x640"])
def test_matches_isaac_albumentations_bit_for_bit(name: str, letter_box: bool) -> None:
    """Exact equality: both sides are deterministic integer pipelines over the same uint8 input."""
    gr00t_aug, isaac = _isaac_eval_transform(letter_box)
    image = _images()[name]

    tensors, _replay = gr00t_aug.apply_with_replay(isaac, [image])
    expected = np.asarray(tensors[0])

    ours = Gr00tEvalImageTransform(
        image_target_size=TARGET,
        image_crop_size=CROP,
        shortest_image_edge=EDGE,
        crop_fraction=FRACTION,
        letter_box_transform=letter_box,
        use_albumentations=True,
    )(image).numpy()

    assert ours.shape == expected.shape, f"{ours.shape} != {expected.shape}"
    assert ours.dtype == expected.dtype == np.uint8
    mismatched = int((ours != expected).sum())
    assert mismatched == 0, (
        f"{mismatched}/{ours.size} pixels differ (max |diff| = {int(np.abs(ours.astype(int) - expected.astype(int)).max())})"
    )


@pytest.mark.L1
@pytest.mark.CPU
def test_letterbox_only_changes_non_square_inputs() -> None:
    """The flag must be a no-op on square frames and active on robolab's 540x640 composite."""
    kwargs = dict(
        image_target_size=TARGET,
        image_crop_size=CROP,
        shortest_image_edge=EDGE,
        crop_fraction=FRACTION,
        use_albumentations=True,
    )
    off = Gr00tEvalImageTransform(letter_box_transform=False, **kwargs)
    on = Gr00tEvalImageTransform(letter_box_transform=True, **kwargs)
    images = _images()

    assert np.array_equal(off(images["noise"]).numpy(), on(images["noise"]).numpy())
    assert not np.array_equal(off(images["robolab_540x640"]).numpy(), on(images["robolab_540x640"]).numpy())


@pytest.mark.L1
@pytest.mark.CPU
def test_differs_from_the_stock_port_transform_in_the_way_measured() -> None:
    """Guard the premise: the correction must change the pixels by the known amount (~1.7 normalized)."""
    from torchvision.transforms import v2 as transforms

    image = _images()["noise"]
    ours = (
        Gr00tEvalImageTransform(
            image_target_size=TARGET,
            image_crop_size=CROP,
            shortest_image_edge=EDGE,
            crop_fraction=FRACTION,
            letter_box_transform=False,
            use_albumentations=True,
        )(image)
        .numpy()
        .astype(np.float64)
    )
    stock = (
        transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(size=list(TARGET)),
                transforms.CenterCrop(size=list(CROP)),
                transforms.Resize(size=list(TARGET)),
            ]
        )(image)
        .numpy()
        .astype(np.float64)
    )

    # Normalized units (/127.5) to match how operator GR00T validation documentation reports it; measured at ~1.7.
    max_diff = float(np.abs(ours - stock).max()) / 127.5
    assert max_diff > 1.0, f"the stock transform now agrees to {max_diff:.3f}; re-measure the fix's premise"
