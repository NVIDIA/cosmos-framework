# SPDX-License-Identifier: OpenMDW-1.1
"""Equivalence tests for Action (DROID) loaders."""
from __future__ import annotations

import os
import pytest
import torch

AROOT = os.environ.get("DROID_COSMOS_ROOT")
AURI = os.environ.get("DROID_LANCE_URI")
ACOMP = os.environ.get("DROID_COMPOSED_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (AROOT and os.path.isdir(AROOT)),
    reason="set DROID_COSMOS_ROOT to the prepared fixtures"
)

_AKW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)
_BATCH_IDXS = [0, 1, 123, 5000, 17000, 26000]

@pytest.fixture(scope="module")
def base():
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
    return DROIDLeRobotDataset(root=AROOT, **_AKW)

@pytest.mark.skipif(not AURI, reason="set DROID_LANCE_URI")
def test_action_raw_bytes(base):
    from cosmos_framework.data.lance import LanceDROIDDataset
    lance = LanceDROIDDataset(root=AROOT, lance_uri=AURI, decode_device="cpu", **_AKW)
    assert len(base) == len(lance)
    idxs = [i for i in _BATCH_IDXS if i < len(base)]

    # Single item
    for i in idxs:
        b, l = base[i], lance[i]
        assert torch.equal(b["video"], l["video"])
        assert torch.allclose(b["action"], l["action"], atol=0, rtol=0)
        assert b["ai_caption"] == l["ai_caption"]

    # Batch item
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        assert torch.equal(base[i]["video"], batch[j]["video"])

@pytest.mark.skipif(not ACOMP, reason="set DROID_COMPOSED_LANCE_URI")
def test_action_composed(base):
    from cosmos_framework.data.lance import LanceDROIDComposedDataset
    lance = LanceDROIDComposedDataset(root=AROOT, lance_uri=ACOMP, decode_device="cpu", **_AKW)
    idxs = [i for i in _BATCH_IDXS if i < len(base)]
    batch = lance.__getitems__(idxs)

    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert torch.equal(b["action"], l["action"])
        assert b["ai_caption"] == l["ai_caption"]
        # Composed video is within H.264 tolerance (mean|Δ|/255 < 0.02)
        mad = (b["video"].float() - l["video"].float()).abs().mean().item() / 255.0
        assert mad < 0.02
