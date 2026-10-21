# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L1 @GPU: real i4-native tiny dense/MoE Qwen3-VL forward/backward."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.model.qwen3_vl import (
    Qwen3VLBuildHooks,
    Qwen3VLPolicy,
    build_qwen3_vl_policy,
    override_qwen3_vl_forward,
)
from cosmos_framework.posttrain.model.qwen3_vl_moe import override_qwen3_vl_moe_for_padded_batches

_IMAGE_TOKEN_ID = 120
_VISION_START_TOKEN_ID = 118
_VISION_END_TOKEN_ID = 119


def _dense_config() -> Any:
    from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 512,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=121,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        vision_end_token_id=_VISION_END_TOKEN_ID,
        tie_word_embeddings=False,
    )


def _moe_config() -> Any:
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeConfig,
    )

    return Qwen3VLMoeConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "decoder_sparse_step": 1,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 512,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=121,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        vision_end_token_id=_VISION_END_TOKEN_ID,
        tie_word_embeddings=False,
    )


def _prepare_native(net: torch.nn.Module, *, moe: bool) -> None:
    override_qwen3_vl_forward(net.model)
    if moe:
        assert override_qwen3_vl_moe_for_padded_batches(net) == 2


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(  # [B,T]
            [[1, _VISION_START_TOKEN_ID, _IMAGE_TOKEN_ID, _VISION_END_TOKEN_ID, 2]],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),  # [B,T]
        "pixel_values": torch.randn((4, 24), dtype=torch.float32),  # [N_patch,D_patch]
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),  # [N_image,3]
    }


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.parametrize(
    ("model_type", "model_config", "moe"),
    [
        ("qwen3_vl", _dense_config, False),
        ("qwen3_vl_moe", _moe_config, True),
    ],
)
def test_real_native_tiny_qwen3_vl_forward_backward(
    model_type: str,
    model_config: Any,
    moe: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native Qwen3-VL GPU gate")
    policy = build_qwen3_vl_policy(
        model_type=model_type,
        model_config=model_config(),
        dtype=torch.bfloat16,
        hooks=Qwen3VLBuildHooks(prepare_model=lambda net: _prepare_native(net, moe=moe)),
        initialize_on_meta=False,
    )
    assert isinstance(policy, Qwen3VLPolicy)
    batch = policy.prepare_policy_batch(_batch())

    output = policy.forward_policy_batch(batch)
    loss = output.logits.float().square().mean()  # []
    loss.backward()

    assert output.logits.shape == (1, 5, 128)
    assert torch.isfinite(output.logits).all()
    gradients = [parameter.grad for parameter in policy.net.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    if moe:
        expert_gradients = [
            parameter.grad
            for name, parameter in policy.net.named_parameters()
            if ".mlp.experts." in name and parameter.grad is not None
        ]
        assert expert_gradients
        assert any(float(gradient.float().abs().sum()) > 0.0 for gradient in expert_gradients)
