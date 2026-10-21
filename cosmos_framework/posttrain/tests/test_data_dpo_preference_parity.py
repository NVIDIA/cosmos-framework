# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bit-parity of the ported bucket-snap against cosmos-rl's DPOPreferenceDataset.

`data/datasets/dpo_preference.py::_snap_to_bucket` is ported from cosmos-rl; this test
imports cosmos-rl (TEST ONLY) and asserts the two produce BIT-IDENTICAL snapped frames +
image_size across aspect-ratio bins — so the posttrain data layer feeds the model exactly
what cosmos-rl's loader does (complements gate-3, which injected x0 rather than decoding).
"""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.data.datasets.dpo_preference import _snap_to_bucket


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.parametrize("h,w", [(64, 64), (90, 160), (160, 90), (80, 48), (48, 80), (100, 130)])
def test_snap_to_bucket_bit_parity_vs_cosmos_rl(h, w):
    crl = pytest.importorskip("cosmos_rl.tools.dataset.wfm.dpo_preference_dataset")
    frames = torch.randint(0, 256, (3, 4, h, w), dtype=torch.uint8)
    mine, mysize = _snap_to_bucket(frames, "256")  # default VIDEO_RES_SIZE_INFO table
    their_frames, th, tw, nh, nw = crl.DPOPreferenceDataset._snap_to_bucket(frames, "256")
    assert torch.equal(mine, their_frames), f"snapped frames differ at ({h},{w})"
    assert mysize.tolist() == [float(th), float(tw), float(nh), float(nw)], f"image_size differs at ({h},{w})"
