# SPDX-License-Identifier: OpenMDW-1.1
"""Equivalence test for the composed Action (DROID) loader vs the base DROIDLeRobotDataset.

Labels (action/pose/caption) are bit-exact; video is within one offline H.264 re-encode.
"""

from __future__ import annotations

import os

import pytest
import torch

from cosmos_framework.data.lance import LanceDROIDComposedDataset
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

AROOT = os.environ.get("DROID_COSMOS_ROOT")
ACOMP = os.environ.get("DROID_COMPOSED_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (AROOT and ACOMP and os.path.isdir(AROOT)), reason="set DROID_COSMOS_ROOT and DROID_COMPOSED_LANCE_URI"
)

_AKW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)
_IDXS = [17000, 1, 26000, 0, 123, 5000]  # unsorted: batched take must map back to the right episode


def test_action_composed():
    base = DROIDLeRobotDataset(root=AROOT, **_AKW)
    lance = LanceDROIDComposedDataset(ACOMP, decode_device="cpu", **_AKW)
    idxs = [i for i in _IDXS if i < len(base)]
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert torch.equal(b["action"], l["action"])  # labels bit-exact
        assert b["ai_caption"] == l["ai_caption"]
        mad = (b["video"].float() - l["video"].float()).abs().mean().item() / 255.0
        assert mad < 0.02  # video within H.264 re-encode tolerance
