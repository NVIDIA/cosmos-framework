# SPDX-License-Identifier: OpenMDW-1.1
"""The LanceDB DROID loader must produce output identical to the base loader.

Run (after building the fixtures, see tests/data/lance/README.md):
    DROID_COSMOS_ROOT=.../droid_cosmos/success \
    DROID_LANCE_URI=.../lance/droid_video \
    pytest tests/data/lance/test_action_equivalence.py
"""
from __future__ import annotations

import os

import pytest
import torch

ROOT = os.environ.get("DROID_COSMOS_ROOT")
URI = os.environ.get("DROID_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (ROOT and URI and os.path.isdir(ROOT)),
    reason="set DROID_COSMOS_ROOT and DROID_LANCE_URI to the prepared fixtures",
)

_KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)


@pytest.fixture(scope="module")
def loaders():
    from cosmos_framework.data.lance import LanceDROIDDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    base = DROIDLeRobotDataset(root=ROOT, **_KW)
    lance = LanceDROIDDataset(root=ROOT, lance_uri=URI, decode_device="cpu", **_KW)
    return base, lance


def test_same_length(loaders):
    base, lance = loaders
    assert len(base) == len(lance)


@pytest.mark.parametrize("idx", [0, 1, 123, 5000, 17000, 26000])
def test_sample_identical(loaders, idx):
    base, lance = loaders
    b, l = base[idx], lance[idx]
    assert b.keys() == l.keys()
    # CPU torchcodec decode of the same mp4 bytes => bit-exact video.
    assert torch.equal(b["video"], l["video"]), "video differs"
    assert torch.allclose(b["action"], l["action"], atol=0, rtol=0), "action differs"
    assert int(b["idle_frames"]) == int(l["idle_frames"])
    assert int(b["domain_id"]) == int(l["domain_id"])
    assert b["ai_caption"] == l["ai_caption"]
    assert b["mode"] == l["mode"]
    assert b["viewpoint"] == l["viewpoint"]


def test_ee_pose_action_space():
    """The ee_pose layout (quantile-normalized 10-D action) must also match."""
    from cosmos_framework.data.lance import LanceDROIDDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    kw = dict(action_space="ee_pose", mode="policy", chunk_length=16)
    base = DROIDLeRobotDataset(root=ROOT, **kw)
    lance = LanceDROIDDataset(root=ROOT, lance_uri=URI, decode_device="cpu", **kw)
    for idx in (0, 2000, 20000):
        b, l = base[idx], lance[idx]
        assert torch.equal(b["video"], l["video"])
        assert torch.allclose(b["action"], l["action"], atol=1e-6)
        assert torch.allclose(b["initial_pose"], l["initial_pose"], atol=1e-6)
