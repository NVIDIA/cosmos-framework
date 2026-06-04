# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for DefaultBatchCollator."""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.dataflow.collators import DefaultBatchCollator


def test_default_collator_stacks_like_torch():
    samples = [
        {"x": torch.tensor([1.0, 2.0]), "y": 0},
        {"x": torch.tensor([3.0, 4.0]), "y": 1},
    ]
    batch = DefaultBatchCollator().collate(samples)
    assert batch["x"].shape == (2, 2)
    assert torch.equal(batch["x"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(batch["y"], torch.tensor([0, 1]))


from cosmos_framework.data.vfm.dataflow.collators import VFMListCollator


def test_vfm_list_collator_keeps_media_as_lists_and_stacks_scalars():
    s1 = {"video": torch.zeros(3, 4, 8, 8), "text_token_ids": torch.arange(5), "domain_id": 0}
    s2 = {"video": torch.zeros(3, 2, 8, 8), "text_token_ids": torch.arange(7), "domain_id": 1}
    out = VFMListCollator().collate([s1, s2])
    assert isinstance(out["video"], list) and len(out["video"]) == 2
    assert isinstance(out["text_token_ids"], list) and len(out["text_token_ids"]) == 2
    assert out["domain_id"] == [0, 1]


def test_vfm_list_collator_preserves_sparse_sound_none():
    s1 = {"video": torch.zeros(3, 1, 8, 8), "sound": torch.zeros(2)}
    s2 = {"video": torch.zeros(3, 1, 8, 8), "sound": None}
    out = VFMListCollator().collate([s1, s2])
    assert out["sound"][1] is None and out["sound"][0] is not None


def test_vfm_list_collator_drops_optional_key_missing_in_some():
    s1 = {"video": torch.zeros(3, 1, 8, 8), "extra_meta": 5}
    s2 = {"video": torch.zeros(3, 1, 8, 8)}
    out = VFMListCollator().collate([s1, s2])
    assert "extra_meta" not in out
