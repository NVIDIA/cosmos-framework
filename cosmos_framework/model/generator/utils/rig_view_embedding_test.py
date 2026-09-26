# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Physical view IDs survive controls, packing and unconditional text paths."""

import pytest
import torch

from cosmos_framework.data.generator.augmentors.merge_datadict import NonRigVideoMetadata
from cosmos_framework.model.generator.utils.rig_view_embedding import add_view_embeddings, vision_view_ids

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_ids_repeat_per_item_without_renumbering() -> None:
    ids = vision_view_ids(
        {"view_indices_selection": [[8], [2, 5]], "ai_caption": ["", ""]},
        batch_size=2,
        item_counts=[2, 3],
        views_per_item=[1, 1, 2, 2, 2],
        num_embeddings=12,
    )
    assert [item.tolist() for item in ids] == [[8], [8], [2, 5], [2, 5], [2, 5]]


@pytest.mark.parametrize("raw_ids", [None, [[11]], [[-2]], [[-1, 0]], [[-1, -1]], [[]]])
def test_invalid_camera_ids_fail_before_forward(raw_ids: list[list[int]] | None) -> None:
    with pytest.raises(ValueError):
        vision_view_ids(
            {"view_indices_selection": raw_ids}, batch_size=1, item_counts=[2], views_per_item=[1, 1], num_embeddings=12
        )


def test_camera_major_offsets_and_gradients() -> None:
    embedding = torch.nn.Embedding(12, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(12).reshape(12, 1).expand(12, 3))  # [12,3]
    tokens = torch.zeros(12, 3)  # [N,D]
    ids = [torch.tensor([8]), torch.tensor([2, 5])]  # per-item [V]
    actual = add_view_embeddings(tokens, [(2, 1, 2), (4, 1, 2)], ids, embedding)  # [N,D]
    assert actual[:, 0].tolist() == [8] * 4 + [2] * 4 + [5] * 4
    actual.sum().backward()
    assert embedding.weight.grad is not None
    assert embedding.weight.grad[:, 0].tolist() == [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0]


def test_zero_initialization_preserves_ga_tokens() -> None:
    embedding = torch.nn.Embedding(12, 3)
    torch.nn.init.zeros_(embedding.weight)  # [12,D]
    tokens = torch.randn(4, 3)  # [N,D]
    actual = add_view_embeddings(tokens, [(2, 1, 2)], [torch.tensor([10])], embedding)  # [N,D]
    torch.testing.assert_close(actual, tokens, rtol=0, atol=0)


def test_non_rig_video_has_no_embedding_offset_or_gradient() -> None:
    marker = NonRigVideoMetadata(input_keys=[])
    sample = marker({"ai_caption": "A clear mountain lake."})
    assert sample["ai_caption"] == "A clear mountain lake."
    # Match default tensor collation used for this metadata, including B > 1.
    raw_ids = torch.stack([sample["view_indices_selection"], torch.tensor([8])])  # [B,1]
    ids = vision_view_ids(
        {"view_indices_selection": raw_ids},
        batch_size=2,
        item_counts=[1, 2],
        views_per_item=[1, 1, 1],
        num_embeddings=12,
    )
    assert [item.tolist() for item in ids] == [[-1], [8], [8]]
    embedding = torch.nn.Embedding(12, 3)
    torch.nn.init.constant_(embedding.weight, 7.0)  # [12,D]
    tokens = torch.randn(12, 3, requires_grad=True)  # [N,D]
    actual = add_view_embeddings(tokens, [(2, 1, 2)] * 3, ids, embedding)  # [N,D]
    torch.testing.assert_close(actual[:4], tokens[:4], rtol=0, atol=0)
    torch.testing.assert_close(actual[4:], tokens[4:] + 7.0)
    actual.sum().backward()
    assert embedding.weight.grad is not None
    assert embedding.weight.grad[:, 0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0]
    torch.testing.assert_close(tokens.grad, torch.ones_like(tokens))


def test_non_rig_metadata_cannot_replace_camera_ids_or_hide_multiview_layout() -> None:
    with pytest.raises(ValueError, match="must not overwrite"):
        NonRigVideoMetadata(input_keys=[])(dict(view_indices_selection=[0]))
    with pytest.raises(ValueError, match="encoded camera count"):
        vision_view_ids(
            {"view_indices_selection": [[-1]]},
            batch_size=1,
            item_counts=[1],
            views_per_item=[2],
            num_embeddings=12,
        )
