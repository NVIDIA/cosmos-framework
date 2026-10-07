# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Physical view IDs survive controls, packing and unconditional text paths."""

import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from cosmos_framework.data.generator.augmentors.merge_datadict import NonRigVideoMetadata
from cosmos_framework.model.generator.utils.rig_view_embedding import add_view_embeddings, vision_view_ids

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _reference_add_view_embeddings(
    tokens: torch.Tensor,  # [N,D]
    token_shapes: list[tuple[int, int, int]],
    view_ids: list[torch.Tensor],  # one [V] tensor per item
    embedding: torch.nn.Embedding,
) -> torch.Tensor:  # [N,D]
    """Keep the original out-of-place calculation as an independent regression reference."""
    offsets: list[torch.Tensor] = []
    for shape, ids in zip(token_shapes, view_ids, strict=True):
        per_view = embedding(ids.clamp_min(0)) * (ids >= 0).unsqueeze(-1)  # [V,D]
        offsets.append(per_view.repeat_interleave(math.prod(shape) // ids.numel(), dim=0))  # [N_item,D]
    return tokens + torch.cat(offsets, dim=0).to(tokens.dtype)  # [N,D]


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
    original = tokens.clone()  # [N,D]
    actual = add_view_embeddings(tokens, [(2, 1, 2)], [torch.tensor([10])], embedding)  # [N,D]
    assert actual is tokens
    torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inference_inplace_matches_out_of_place_reference(dtype: torch.dtype) -> None:
    embedding = torch.nn.Embedding(12, 3)
    tokens = torch.randn(12, 3).to(dtype)  # [N,D]
    ids = [torch.tensor([8]), torch.tensor([2, 5]), torch.tensor([-1])]  # per-item [V]
    shapes = [(2, 1, 2), (2, 1, 2), (2, 1, 2)]
    with torch.enable_grad():
        expected = _reference_add_view_embeddings(tokens, shapes, ids, embedding)  # [N,D]
    inference_tokens = tokens.clone()  # [N,D]
    with torch.no_grad():
        actual = add_view_embeddings(inference_tokens, shapes, ids, embedding)  # [N,D]
    assert actual is inference_tokens
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_inference_rejects_token_count_mismatch_before_mutation() -> None:
    embedding = torch.nn.Embedding(12, 3)
    tokens = torch.randn(5, 3)  # [N,D]
    original = tokens.clone()  # [N,D]
    with torch.no_grad(), pytest.raises(ValueError, match="projected token count"):
        add_view_embeddings(tokens, [(2, 1, 2)], [torch.tensor([8])], embedding)  # [N,D]
    torch.testing.assert_close(tokens, original, rtol=0, atol=0)


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
    actual = add_view_embeddings(tokens.clone(), [(2, 1, 2)] * 3, ids, embedding)  # [N,D]
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("execution", ["eager", "checkpoint", "compiled"])
def test_training_inplace_matches_outputs_and_upstream_gradients(dtype: torch.dtype, execution: str) -> None:
    torch.manual_seed(42)
    projection = torch.nn.Linear(5, 3).to(dtype)  # weights [D,C], bias [D]
    embedding = torch.nn.Embedding(12, 3)  # weights [12,D], FP32 offsets test mixed dtypes
    features = torch.randn(24, 5).to(dtype).requires_grad_()  # [N,C]
    # Controls and targets share camera IDs, so embedding gradients must accumulate across items.
    ids = [torch.tensor([8, 2]), torch.tensor([8, 2]), torch.tensor([-1])]  # per-item [V]
    shapes = [(4, 1, 2), (4, 1, 2), (2, 2, 2)]
    parameters = (features, projection.weight, projection.bias, embedding.weight)

    def optimized_forward(inputs: torch.Tensor) -> torch.Tensor:  # inputs: [N,C], returns [N,D]
        optimized_tokens = projection(inputs)  # [N,D], fresh non-leaf buffer
        result = add_view_embeddings(optimized_tokens, shapes, ids, embedding)  # [N,D]
        if execution == "eager":
            assert result is optimized_tokens
        return result  # [N,D]

    with torch.enable_grad(), torch.autograd.detect_anomaly():
        reference_tokens = projection(features)  # [N,D]
        expected = _reference_add_view_embeddings(reference_tokens, shapes, ids, embedding)  # [N,D]
        expected_loss = expected.float().square().mean()  # scalar
        expected_gradients = torch.autograd.grad(expected_loss, parameters)  # per-parameter shapes
        if execution == "checkpoint":
            actual = checkpoint(optimized_forward, features, use_reentrant=False)  # [N,D]
        elif execution == "compiled":
            # Exercise autograd functionalization without requiring CUDA/Triton.
            actual = torch.compile(optimized_forward, backend="aot_eager", fullgraph=True)(features)  # [N,D]
        else:
            actual = optimized_forward(features)  # [N,D]
        actual_loss = actual.float().square().mean()  # scalar
        actual_gradients = torch.autograd.grad(actual_loss, parameters)  # per-parameter shapes
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tolerance = 1e-6 if dtype == torch.float32 else 0.02
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)
