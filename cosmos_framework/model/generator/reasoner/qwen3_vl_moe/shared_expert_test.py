# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU unit tests for the generation-tower shared expert."""

import pytest
import torch

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import Qwen3VLMoeTextExpertsNaive
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
    SharedExpert,
)
from cosmos_framework.utils.generator.aux_optimizer_utils import split_orthogonalizable_params

pytestmark = [pytest.mark.L0]


def _make_config(
    hidden_size: int = 64,
    moe_intermediate_size: int = 32,
    num_experts_per_tok: int = 4,
) -> Qwen3VLMoeTextConfig:
    return Qwen3VLMoeTextConfig(
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=16,
        num_experts_per_tok=num_experts_per_tok,
        hidden_act="silu",
    )


@pytest.mark.CPU
def test_shared_expert_uses_nested_state_dict_schema() -> None:
    config = _make_config()
    disabled = Qwen3VLMoeTextSparseMoeBlock(config)
    enabled = Qwen3VLMoeTextSparseMoeBlock(
        config,
        enable_shared_expert=True,
        shared_expert_intermediate_scale=2,
    )

    assert disabled.shared_expert is None
    assert isinstance(enabled.shared_expert, SharedExpert)
    assert enabled.shared_expert.intermediate_size == 2 * config.moe_intermediate_size
    assert enabled.shared_expert.gate_up_proj.weight.shape == torch.Size(
        [4 * config.moe_intermediate_size, config.hidden_size]
    )
    assert enabled.shared_expert.down_proj.weight.shape == torch.Size(
        [config.hidden_size, 2 * config.moe_intermediate_size]
    )

    persisted = set(enabled.state_dict())
    assert "shared_expert.gate_up_proj.weight" in persisted
    assert "shared_expert.down_proj.weight" in persisted
    assert "shared_gate_up_proj" not in persisted
    assert "shared_down_proj" not in persisted


@pytest.mark.CPU
def test_shared_expert_init_preserves_warm_start_and_unblocks_learning() -> None:
    torch.manual_seed(0)
    shared_expert = SharedExpert(hidden_size=16, intermediate_size=8, hidden_act="silu")
    shared_expert.init_weights(std=0.02)
    hidden_states = torch.randn(4, 16)  # [N,D]

    output = shared_expert(hidden_states)  # [N,D]
    torch.testing.assert_close(output, torch.zeros_like(output))

    output.sum().backward()
    assert shared_expert.down_proj.weight.grad is not None
    assert torch.count_nonzero(shared_expert.down_proj.weight.grad) > 0
    assert shared_expert.gate_up_proj.weight.grad is not None
    torch.testing.assert_close(
        shared_expert.gate_up_proj.weight.grad,
        torch.zeros_like(shared_expert.gate_up_proj.weight.grad),
    )


@pytest.mark.CPU
def test_shared_expert_linear_weights_are_orthogonalizable() -> None:
    shared_expert = SharedExpert(hidden_size=16, intermediate_size=8, hidden_act="silu")
    optimizer_param_ids = {id(param) for param in shared_expert.parameters()}

    orthogonalizable, adamw_params, _ = split_orthogonalizable_params(
        shared_expert,
        optimizer_param_ids,
    )

    assert set(orthogonalizable) == {
        shared_expert.gate_up_proj.weight,
        shared_expert.down_proj.weight,
    }
    assert adamw_params == []


@pytest.mark.CPU
def test_shared_expert_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError, match="intermediate_size must be positive"):
        SharedExpert(hidden_size=16, intermediate_size=0, hidden_act="silu")


@pytest.mark.CPU
@pytest.mark.parametrize("top_k", [7, 8])
def test_grouped_mm_token_reordering_uses_runtime_top_k(top_k: int) -> None:
    config = _make_config(num_experts_per_tok=8)
    block = Qwen3VLMoeTextSparseMoeBlock(config, top_k=top_k)
    num_tokens = 2
    topk_scores = torch.ones(num_tokens, top_k)
    expert_indices = torch.arange(num_tokens * top_k).reshape(num_tokens, top_k)

    _, actual_token_indices = block.experts._reorder_tokens(topk_scores, expert_indices)

    expected_token_indices = torch.arange(num_tokens).repeat_interleave(top_k)
    torch.testing.assert_close(actual_token_indices, expected_token_indices)


@pytest.mark.GPU
@pytest.mark.parametrize("top_k", [7, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm experts require CUDA")
def test_shared_expert_grouped_mm_matches_naive(top_k: int) -> None:
    torch.manual_seed(0)
    config = _make_config(
        hidden_size=512,
        moe_intermediate_size=256,
        num_experts_per_tok=8,
    )
    grouped_block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        enable_shared_expert=True,
        top_k=top_k,
    )
    grouped_block.init_weights()
    assert grouped_block.shared_expert is not None
    torch.nn.init.normal_(grouped_block.shared_expert.down_proj.weight, mean=0.0, std=0.02)

    naive_block = Qwen3VLMoeTextSparseMoeBlock(
        config,
        enable_shared_expert=True,
        top_k=top_k,
    )
    naive_block.experts = Qwen3VLMoeTextExpertsNaive(config)
    naive_block.load_state_dict(grouped_block.state_dict())

    grouped_block = grouped_block.to(device="cuda", dtype=torch.bfloat16).eval()
    naive_block = naive_block.to(device="cuda", dtype=torch.bfloat16).eval()
    hidden_states = torch.randn(128, config.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        grouped_output, _ = grouped_block(hidden_states)
        naive_output, _ = naive_block(hidden_states)

    torch.testing.assert_close(
        grouped_output.float(),
        naive_output.float(),
        atol=5e-2,
        rtol=5e-2,
    )
