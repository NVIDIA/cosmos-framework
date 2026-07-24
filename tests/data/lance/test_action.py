# SPDX-License-Identifier: OpenMDW-1.1
"""Equivalence test for the composed Action (DROID) loader vs the base DROIDLeRobotDataset.

Labels (action/pose/caption/idle) are bit-exact; video is within one offline H.264 re-encode.
"""

from __future__ import annotations

import os

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos_framework.data.lance import LanceDROIDComposedDataset

AROOT = os.environ.get("DROID_LEROBOT_ROOT")  # versioned dir, e.g. .../droid_plus_lerobot_320x180_20260406
ACOMP = os.environ.get("DROID_COMPOSED_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (AROOT and ACOMP and os.path.isdir(AROOT)), reason="set DROID_LEROBOT_ROOT and DROID_COMPOSED_LANCE_URI"
)

_IDXS = [17000, 1, 26000, 0, 123, 5000]  # unsorted: batched take must map back to the right episode


@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch):
    # the base loader sets HF_HUB_OFFLINE itself; pre-set it so the env guard stays clean
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.mark.parametrize("action_space,use_state", [("joint_pos", True), ("midtrain", False), ("midtrain", True)])
def test_action_composed(action_space, use_state):
    kw = dict(action_space=action_space, use_state=use_state, mode="policy", chunk_length=16)
    base = DROIDLeRobotDataset(root=AROOT, use_success_only=True, **kw)
    lance = LanceDROIDComposedDataset(ACOMP, decode_device="cpu", **kw)
    assert len(base) == len(lance)
    idxs = [i for i in _IDXS if i < len(base)]
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        b, l = base[i], batch[j]
        assert torch.equal(b["action"], l["action"])  # labels bit-exact
        assert b["ai_caption"] == l["ai_caption"]
        assert torch.equal(b["idle_frames"], l["idle_frames"])
        if "initial_pose" in b:
            assert torch.equal(b["initial_pose"], l["initial_pose"])
        mad = (b["video"].float() - l["video"].float()).abs().mean().item() / 255.0
        # One offline H.264 re-encode + the base's own decoder backend difference.
        assert mad < 0.025
