# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stateless inputs and pixel-exact CPU goldens from the pinned OpenPI transform."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.model.pi05.processing import Pi05Processor, _resize_image, validate_replay_inputs
from cosmos_framework.posttrain.tests.helpers.pi05_loading import write_processor_source

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def processor(tmp_path):
    return Pi05Processor(write_processor_source(tmp_path))


def test_pi05_inputs_are_stateless_and_keep_missing_view_mask(processor):
    image = np.full((224, 224, 3), 127, dtype=np.uint8)
    row = {"image": image, "wrist_image": image, "state": np.zeros(8), "prompt": "move_the red block\n"}
    inputs = processor.prepare([row, {**row, "state": np.ones(8)}], device="cpu")
    torch.testing.assert_close(inputs.tokens[0], inputs.tokens[1], rtol=0, atol=0)
    torch.testing.assert_close(inputs.token_mask[0], inputs.token_mask[1], rtol=0, atol=0)
    assert not torch.equal(inputs.state[0], inputs.state[1])
    assert inputs.tokens.shape == (2, 200)
    assert torch.equal(inputs.image_masks, torch.tensor([[True, True, False], [True, True, False]]))
    assert torch.equal(inputs.images[2], -torch.ones_like(inputs.images[2]))
    assert all(image.is_contiguous(memory_format=torch.channels_last) for image in inputs.images)


def test_pi05_action_decode_preserves_quantile_epsilon_and_execution_prefix(processor):
    values = torch.linspace(-1, 1, 32).flip(0).expand(2, 10, -1).clone()
    actions = processor.decode(values)
    low, high = processor.quantiles["actions"]
    native = values[:, :5, :7].numpy()
    expected = (native + 1) / 2 * (high - low + 1e-6) + low
    prematurely_promoted = (native.astype(np.float64) + 1) / 2 * (high - low + 1e-6) + low
    assert np.any(expected != prematurely_promoted)
    assert actions.dtype == torch.float64 and actions.shape == (2, 5, 7)
    np.testing.assert_array_equal(actions.numpy(), expected)


def test_pi05_processor_rejects_later_invalid_observation(processor):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    row = {"image": image, "wrist_image": image, "state": np.zeros(8), "prompt": "move the block"}
    with pytest.raises(ValueError, match=r"observations\[1\]\.state"):
        processor.prepare([row, {**row, "state": np.full(8, np.nan)}], device="cpu")


def test_compact_encoding_materializes_the_original_fp32_operations_and_layout(processor):
    rng = np.random.default_rng(51)
    rows = [
        {
            "image": rng.integers(0, 256, (120, 256, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            "state": rng.normal(size=8).astype(np.float32),
            "prompt": "move_the red block\n",
        }
        for _ in range(2)
    ]
    encoded = processor.encode(rows)
    assert validate_replay_inputs(encoded) == len(rows)
    assert len(encoded.images) == 2 and all(image.dtype == torch.uint8 for image in encoded.images)
    actual = processor.materialize(encoded, device="cpu")
    expected_images = tuple(
        torch.stack([_resize_image(row[key]) for row in rows]).to(dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0
        - 1.0
        for key in ("image", "wrist_image")
    )
    for current, old in zip(actual.images[:2], expected_images, strict=True):
        torch.testing.assert_close(current, old, atol=0, rtol=0)
        assert current.stride() == old.stride() == (150528, 1, 672, 3)
        assert current.contiguous().stride() != current.stride()  # The prior NCHW conversion defect is observable.
    low, high = processor.quantiles["state"]
    old_states = np.stack([(row["state"] - low) / (high - low + 1e-6) * 2 - 1 for row in rows])
    torch.testing.assert_close(actual.state[:, :8], torch.from_numpy(old_states), atol=0, rtol=0)
    assert actual.state.dtype == torch.float64
    leaves = (*encoded.images, encoded.image_masks, encoded.tokens, encoded.token_mask, encoded.state)
    assert sum(tensor.numel() * tensor.element_size() for tensor in leaves) == len(rows) * 303115
    assert torch.equal(actual.images[2], -torch.ones_like(actual.images[2]))


@pytest.mark.parametrize(
    "shape,digest",
    [
        ((256, 256, 3), "b3910226b9bab25063d4edc04cc4ff62f65aea6eceb971061c3184638a2e5db0"),
        ((120, 256, 3), "ea7a6b18a30a1b94cede9791ee13e3924313d8a0c7c91c1ed16a839d29a94531"),
        ((224, 224, 3), "446c4d3b624b6e0e69cd40f3f8d3fd0719bcbb61c61b232fda5d15f9929d1d98"),
    ],
)
def test_pi05_resize_matches_openpi_cpu_golden(shape, digest):
    # Generated with the actual ResizeImages dependency at RLinf ce4d821780d0,
    # openpi_client.image_tools.resize_with_pad(image, 224, 224), on 2026-09-20.
    image = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
    pixels = _resize_image(image).numpy()
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == digest


def test_pi05_processor_rejects_nonfinite_normalized_state(tmp_path):
    config = write_processor_source(tmp_path)
    path = Path(config.norm_stats_path)
    stats = json.loads(path.read_text())
    stats["norm_stats"]["state"]["q01"] = [0.0] * 32
    stats["norm_stats"]["state"]["q99"] = [0.0] * 32
    path.write_text(json.dumps(stats))
    processor = Pi05Processor(config)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    row = {"image": image, "wrist_image": image, "state": np.full(8, 1e308), "prompt": "move the block"}
    assert np.isfinite(row["state"]).all()
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="inputs.state must be finite"):
        processor.prepare([row], device="cpu")
