# SPDX-License-Identifier: OpenMDW-1.1
"""Equivalence tests for the Action (DROID) loaders vs the base DROIDLeRobotDataset."""
from __future__ import annotations

import os

import pytest
import torch

from cosmos_framework.data.lance import LanceDROIDComposedDataset, LanceDROIDDataset
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

AROOT = os.environ.get("DROID_COSMOS_ROOT")
AURI = os.environ.get("DROID_LANCE_URI")
ACOMP = os.environ.get("DROID_COMPOSED_LANCE_URI")

pytestmark = pytest.mark.skipif(not (AROOT and os.path.isdir(AROOT)), reason="set DROID_COSMOS_ROOT")

_AKW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)
_IDXS = [0, 1, 123, 5000, 17000, 26000]


@pytest.fixture(scope="module")
def base():
    return DROIDLeRobotDataset(root=AROOT, **_AKW)


@pytest.mark.skipif(not AURI, reason="set DROID_LANCE_URI")
def test_action_raw_bytes(base):
    lance = LanceDROIDDataset(root=AROOT, lance_uri=AURI, decode_device="cpu", **_AKW)
    assert len(base) == len(lance)
    idxs = [i for i in _IDXS if i < len(base)]
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert torch.equal(b["video"], l["video"])  # pixel-identical (raw mp4 bytes)
        assert torch.equal(b["action"], l["action"])
        assert b["ai_caption"] == l["ai_caption"]


@pytest.mark.skipif(not ACOMP, reason="set DROID_COMPOSED_LANCE_URI")
def test_action_composed(base):
    lance = LanceDROIDComposedDataset(root=AROOT, lance_uri=ACOMP, decode_device="cpu", **_AKW)
    idxs = [i for i in _IDXS if i < len(base)]
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert torch.equal(b["action"], l["action"])
        assert b["ai_caption"] == l["ai_caption"]
        mad = (b["video"].float() - l["video"].float()).abs().mean().item() / 255.0
        assert mad < 0.02  # within H.264 re-encode tolerance
