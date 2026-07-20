# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the FPS-mixing `stride_config` primitive in VideoParsingWithFullFrames.

Validates the categorical per-video stride sampling used by the FPS-mixing ablation:
coercion of OmegaConf string keys, the realized per-video stride distribution, input
validation, and that omitting stride_config leaves the exp-decay path unchanged.
"""

from collections import Counter

import numpy as np
import pytest
from omegaconf import OmegaConf

from cosmos_framework.data.generator.augmentors.video_parsing import VideoParsingChunkedFrames

# Minimal args to instantiate the augmentor for stride-sampling only (no video decode).
_BASE_ARGS = dict(
    max_stride=2,
    min_stride=2,
    use_dynamic_fps=True,
    min_fps=10.0,
    max_fps=60.0,
    causal_vae=True,
    max_num_frames=1000,
)


def _make(stride_config=None):
    args = dict(_BASE_ARGS)
    if stride_config is not None:
        args["stride_config"] = stride_config
    return VideoParsingChunkedFrames(input_keys=["metas", "video"], output_keys=None, args=args)


@pytest.mark.L0
@pytest.mark.CPU
def test_stride_config_coercion_omegaconf_str_keys():
    # OmegaConf serializes int keys as strings; __init__ must coerce back to {int: float}.
    aug = _make(OmegaConf.create({"1": 0.5, "2": 0.5}))
    assert aug.stride_config == {1: 0.5, 2: 0.5}
    assert all(isinstance(k, int) for k in aug.stride_config)
    assert all(isinstance(v, float) for v in aug.stride_config.values())


@pytest.mark.L0
@pytest.mark.CPU
def test_stride_config_none_leaves_exp_decay_unchanged():
    aug = _make(None)
    assert aug.stride_config is None
    # exp-decay path still yields a valid stride in [min_stride, max_stride].
    s = aug._sample_stride_with_bias(aug.max_stride, aug.min_stride)
    assert aug.min_stride <= s <= aug.max_stride


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "config,expected",
    [
        ({2: 1.0}, {2: 1.0}),  # baseline: always half-FPS
        ({1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}),  # mix50: per-video 50/50 native+half
        ({1: 0.3, 2: 0.7}, {1: 0.3, 2: 0.7}),  # generic ratios
    ],
)
def test_stride_config_realized_distribution(config, expected):
    aug = _make(OmegaConf.create({str(k): v for k, v in config.items()}))
    np.random.seed(0)  # deterministic => flake-free
    n = 8000
    c = Counter(aug._sample_stride_with_bias(aug.max_stride, aug.min_stride) for _ in range(n))
    realized = {k: c[k] / n for k in config}
    for k, p in expected.items():
        assert abs(realized[k] - p) < 0.03, f"stride {k}: realized {realized[k]:.3f} vs expected {p}"
    # only strides from the config are ever sampled
    assert set(c) <= set(config)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "bad",
    [
        {},  # empty mapping
        {0: 1.0},  # stride < 1
        {1: 0.5, 2: 0.4},  # probabilities do not sum to 1
    ],
)
def test_stride_config_validation_rejects_bad_input(bad):
    with pytest.raises(AssertionError):
        _make(OmegaConf.create({str(k): v for k, v in bad.items()}))
