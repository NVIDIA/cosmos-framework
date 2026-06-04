# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for DefaultBatchCollator and VFMListCollator."""

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
    """VFMListCollator must reproduce the legacy PackingDataLoader packed structure.

    For _MULTI_ITEM_KEYS (video, text_token_ids): each sample's tensor is
    wrapped in a single-element inner list → list[list[Tensor]] nesting.
    For metadata list keys (domain_id): flat list[element].
    """
    v0 = torch.zeros(3, 4, 8, 8)
    v1 = torch.zeros(3, 2, 8, 8)
    t0 = torch.arange(5)
    t1 = torch.arange(7)
    s1 = {"video": v0, "text_token_ids": t0, "domain_id": 0}
    s2 = {"video": v1, "text_token_ids": t1, "domain_id": 1}
    out = VFMListCollator().collate([s1, s2])

    # video: list[list[Tensor]] — each inner list has exactly one element
    assert isinstance(out["video"], list) and len(out["video"]) == 2
    assert isinstance(out["video"][0], list) and len(out["video"][0]) == 1
    assert isinstance(out["video"][1], list) and len(out["video"][1]) == 1
    assert torch.equal(out["video"][0][0], v0)
    assert torch.equal(out["video"][1][0], v1)

    # text_token_ids: list[list[Tensor]] — same nesting as video
    assert isinstance(out["text_token_ids"], list) and len(out["text_token_ids"]) == 2
    assert isinstance(out["text_token_ids"][0], list) and len(out["text_token_ids"][0]) == 1
    assert isinstance(out["text_token_ids"][1], list) and len(out["text_token_ids"][1]) == 1
    assert torch.equal(out["text_token_ids"][0][0], t0)
    assert torch.equal(out["text_token_ids"][1][0], t1)

    # domain_id is a metadata list key (not in _MULTI_ITEM_KEYS) → flat list
    assert out["domain_id"] == [0, 1]


def test_vfm_list_collator_preserves_sparse_sound_none():
    """sound is in _MULTI_ITEM_KEYS; None placeholders must survive.

    Legacy _vfm_inner_collate keeps sound=None as sparse: result["sound"] = [tensor, None].
    Then _split_one for i=0 on the batch_size=1 single-sample collation:
      - sample 0 (sound=tensor): inner-collate([s]) → {"sound": [tensor]},
        split → sound=v[0:1]=[tensor] (single-elem list).
      - sample 1 (sound=None): inner-collate([s]) → {"sound": [None]} (sparse),
        split → sound is a list and key is in _MULTI_ITEM_KEYS: elem=None,
        isinstance(None, list) is False → sound=v[0:1]=[None].
    _accumulate: output_batch["sound"] = [[tensor], [None]].
    """
    snd = torch.zeros(2)
    s1 = {"video": torch.zeros(3, 1, 8, 8), "sound": snd}
    s2 = {"video": torch.zeros(3, 1, 8, 8), "sound": None}
    out = VFMListCollator().collate([s1, s2])

    # sound must be present and have 2 entries
    assert "sound" in out
    assert len(out["sound"]) == 2

    # First entry: [[tensor]] — single-element list wrapping the sound tensor
    assert isinstance(out["sound"][0], list) and len(out["sound"][0]) == 1
    assert torch.equal(out["sound"][0][0], snd)

    # Second entry: [None] — single-element list wrapping None
    assert isinstance(out["sound"][1], list) and len(out["sound"][1]) == 1
    assert out["sound"][1][0] is None


def test_vfm_list_collator_drops_optional_key_missing_in_some():
    """Optional keys not present in every sample: legacy per-sample behavior.

    In the legacy PackingDataLoader, each sample is inner-collated at batch_size=1
    independently via custom_collate_fn.  When a sample has a key, it gets included
    in its single-sample collated dict; when a sample lacks a key, that key simply
    isn't present in its single-sample dict.  _update_output_batch then accumulates
    only what each sample contributes.

    Result: extra_meta appears in the output only for samples that have it.
    For s1 (has extra_meta=5, an int): _vfm_inner_collate([s1]) → default_collate([5])
    = tensor([5]); _split_one → tensor([5])[0:1] = tensor([5]) (shape [1]);
    _accumulate → output_batch["extra_meta"] = [tensor([5])].
    For s2 (no extra_meta): not contributed → output_batch["extra_meta"] unchanged.
    Final: out["extra_meta"] = [tensor([5])].

    Contrast with the old single-pass _vfm_collate([s1, s2]) which would have
    seen extra_meta=None for s2 and dropped the key entirely.  The new
    per-sample implementation faithfully replicates what the legacy packer does.
    """
    s1 = {"video": torch.zeros(3, 1, 8, 8), "extra_meta": 5}
    s2 = {"video": torch.zeros(3, 1, 8, 8)}
    out = VFMListCollator().collate([s1, s2])
    # Legacy: extra_meta from s1 is accumulated; s2 does not contribute it.
    assert "extra_meta" in out
    assert len(out["extra_meta"]) == 1
    assert torch.equal(out["extra_meta"][0], torch.tensor([5]))


def test_vfm_list_collator_image_size_is_flat():
    """image_size is in _FLATTEN_LIST_KEYS: must be flat list[Tensor], not list[list[Tensor]].

    Legacy _update_output_batch extends (not appends) for _FLATTEN_LIST_KEYS,
    so two samples each with image_size=Tensor([H,W]) yield a flat list of 2
    tensors, not a nested list.
    """
    isz0 = torch.tensor([64, 64])
    isz1 = torch.tensor([32, 32])
    s1 = {"video": torch.zeros(3, 1, 64, 64), "image_size": isz0}
    s2 = {"video": torch.zeros(3, 1, 32, 32), "image_size": isz1}
    out = VFMListCollator().collate([s1, s2])

    # image_size: flat list[Tensor] (extended, not appended)
    assert isinstance(out["image_size"], list) and len(out["image_size"]) == 2
    # Elements are tensors, not lists
    assert isinstance(out["image_size"][0], torch.Tensor)
    assert isinstance(out["image_size"][1], torch.Tensor)
    assert torch.equal(out["image_size"][0], isz0)
    assert torch.equal(out["image_size"][1], isz1)
