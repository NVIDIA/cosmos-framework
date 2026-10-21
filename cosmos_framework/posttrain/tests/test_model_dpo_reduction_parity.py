# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bit-parity of the DPO per-sample MSE reduction vs cosmos-rl.

`model/cosmos3.py::_per_sample_mse_fp32` is the core fp32 mean-MSE reduction (the
precision-critical bit of Flow-DPO). This imports cosmos-rl's `_dpo_per_sample_mse`
(TEST ONLY) and asserts the two are BIT-IDENTICAL on random preds/targets — masked
(I2V) and unmasked (T2V), including the stale leading-batch squeeze. A pure-function
guard (no model) complementing gate-3's end-to-end parity.
"""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.model.cosmos3 import _per_sample_mse_fp32


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("stale", [False, True], ids=["aligned", "stale_batch"])
def test_per_sample_mse_bit_parity_vs_cosmos_rl(masked, stale):
    crl = pytest.importorskip("cosmos_rl.policy.model.wfm.models.cosmos3")
    g = torch.Generator().manual_seed(0)
    C, T, H, W, B = 48, 3, 16, 16, 3
    targets = [torch.randn(C, T, H, W, generator=g) for _ in range(B)]
    # stale: preds carry a leading singleton batch dim [1,C,T,H,W] (OmniMoT unpatchify).
    preds = [torch.randn((1, C, T, H, W) if stale else (C, T, H, W), generator=g) for _ in range(B)]
    cms = [torch.randint(0, 2, (T, 1, 1), generator=g).float() for _ in range(B)] if masked else None

    mine = _per_sample_mse_fp32([p.clone() for p in preds], [t.clone() for t in targets], cms)
    theirs = crl._dpo_per_sample_mse([p.clone() for p in preds], [t.clone() for t in targets], cms)
    torch.testing.assert_close(mine, theirs, rtol=0, atol=0)  # bit-exact
