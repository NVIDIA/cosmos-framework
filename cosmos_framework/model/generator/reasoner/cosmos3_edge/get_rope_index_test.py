# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Batched mRoPE position padding for Cosmos3-Edge with unequal sequence lengths."""

from types import SimpleNamespace

import torch

from cosmos_framework.model.generator.reasoner.cosmos3_edge.modeling_cosmos3_edge import (
    Cosmos3EdgeModel,
)


def _stub_model():
    return SimpleNamespace(
        config=SimpleNamespace(
            projector_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=11,
            video_token_id=12,
            vision_start_token_id=13,
        )
    )


def test_get_rope_index_pads_unequal_batch_lengths():
    # Two text-only sequences of true lengths 10 and 9 in one padded batch —
    # this crashed pad_sequence before the (L, 3) transpose fix.
    input_ids = torch.ones(2, 10, dtype=torch.long)
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    attention_mask[1, -1] = 0
    position_ids, deltas = Cosmos3EdgeModel.get_rope_index(
        _stub_model(), input_ids=input_ids, attention_mask=attention_mask
    )
    assert position_ids.shape == (3, 2, 10)
    assert deltas.shape == (2, 1)
    # Real positions are 0..L-1 per sample; sample 2's padded tail uses the pad value.
    assert position_ids[0, 0].tolist() == list(range(10))
    assert position_ids[0, 1, :9].tolist() == list(range(9))


def test_get_rope_index_equal_lengths_unchanged():
    input_ids = torch.ones(2, 6, dtype=torch.long)
    position_ids, _ = Cosmos3EdgeModel.get_rope_index(
        _stub_model(), input_ids=input_ids, attention_mask=torch.ones(2, 6, dtype=torch.long)
    )
    assert position_ids.shape == (3, 2, 6)
    assert position_ids[0, 0].tolist() == list(range(6))
