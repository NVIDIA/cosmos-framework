# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time

import torch
from torch import nn

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import create_text_experts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    AuxLossFreeLoadBalancingConfig,
    CosineRouter,
    CosineRouterConfig,
    Qwen3VLMoeTextSparseMoeBlock,
)


def test_router_activation_defaults_to_softmax() -> None:
    router = CosineRouter(CosineRouterConfig(), hidden_size=2)
    router_logits = torch.tensor([[2.0, 1.0, 0.0]])  # [1,3]
    expert_bias = torch.tensor([-0.5, 0.3, 0.0])  # [3]

    router_scores = router.get_scores(router_logits)  # [1,3]
    biased_selection_scores = router.apply_selection_bias(
        router_logits=router_logits,
        router_scores=router_scores,
        expert_bias=expert_bias,
    )  # [1,3]

    torch.testing.assert_close(router_scores, torch.softmax(router_logits, dim=-1))
    torch.testing.assert_close(biased_selection_scores, router_logits + expert_bias.unsqueeze(0))


def test_sigmoid_router_bias_is_added_in_score_space() -> None:
    router = CosineRouter(CosineRouterConfig(activation="sigmoid"), hidden_size=2)
    router_logits = torch.tensor([[2.0, 1.0, 0.0]])  # [1,3]
    expert_bias = torch.tensor([-0.5, 0.3, 0.0])  # [3]
    router_scores = router.get_scores(router_logits)  # [1,3]

    biased_selection_scores = router.apply_selection_bias(
        router_logits=router_logits,
        router_scores=router_scores,
        expert_bias=expert_bias,
    )  # [1,3]
    selected_experts = torch.topk(biased_selection_scores, k=1, dim=-1).indices  # [1,1]

    torch.testing.assert_close(biased_selection_scores, torch.sigmoid(router_logits) + expert_bias.unsqueeze(0))
    assert selected_experts.item() == 1


def test_sigmoid_router_scores_are_normalized_for_lbl_and_metrics() -> None:
    router = CosineRouter(CosineRouterConfig(activation="sigmoid"), hidden_size=2)
    router_logits = torch.tensor([[2.0, 1.0, 0.0], [-1.0, 0.0, 3.0]])  # [2,3]
    router_scores = router.get_scores(router_logits)  # [2,3]

    routing_probabilities = router.normalize_scores(router_scores)  # [2,3]

    torch.testing.assert_close(routing_probabilities.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(
        routing_probabilities,
        torch.sigmoid(router_logits) / torch.sigmoid(router_logits).sum(dim=-1, keepdim=True),
    )


def test_router_rejects_unknown_activation() -> None:
    try:
        CosineRouter(CosineRouterConfig(activation="relu"), hidden_size=2)
    except ValueError as error:
        assert "Unsupported router activation" in str(error)
    else:
        raise AssertionError("CosineRouter accepted an unsupported activation")


def test_aux_loss_free_controller_uses_block_config() -> None:
    model_config = Qwen3VLMoeTextConfig(
        hidden_size=8,
        moe_intermediate_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    controller_config = AuxLossFreeLoadBalancingConfig(
        enabled=True,
        update_speed=0.25,
        max_bias=None,
    )
    block = Qwen3VLMoeTextSparseMoeBlock(
        model_config,
        aux_loss_free_load_balancing_config=controller_config,
    )
    block.tokens_per_expert.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))  # [4]

    block.update_bias()

    torch.testing.assert_close(block.expert_bias, torch.tensor([0.25, 0.25, -0.25, -0.25]))  # [4]
    assert block.aux_loss_free_load_balancing_config is controller_config
    assert "expert_bias" in block.state_dict()
    assert "tokens_per_expert" not in block.state_dict()


def run_moe(mod: nn.Module, hidden_states: torch.Tensor, topk_scores: torch.Tensor, expert_indices: torch.Tensor):
    num_warmup_iterations = 10
    num_timing_iterations = 100

    for _ in range(num_warmup_iterations):
        with torch.no_grad():
            output = mod(hidden_states, topk_scores, expert_indices)

    start_time = time.time()
    for _ in range(num_timing_iterations):
        with torch.no_grad():
            output = mod(hidden_states, topk_scores, expert_indices)
    end_time = time.time()

    time_taken = (end_time - start_time) / num_timing_iterations

    print(f"Time taken: {time_taken} seconds")
    print(f"output: {output.norm().detach().cpu().item()} {output.shape} {output.dtype} {output.device}")
    return output, time_taken


def main():
    num_tokens = 2048
    config = Qwen3VLMoeTextConfig(
        hidden_size=2048,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        hidden_act="silu",
    )

    control = create_text_experts(config, implementation_type="naive")
    exp = create_text_experts(config, implementation_type="grouped_mm")

    control.init_weights()
    exp.load_state_dict(control.state_dict())

    control = control.to(device="cuda", dtype=torch.bfloat16)
    exp = exp.to(device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn(
        num_tokens,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_scores = torch.randn(
        num_tokens,
        config.num_experts_per_tok,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    expert_indices = torch.randint(
        0,
        config.num_experts,
        (num_tokens, config.num_experts_per_tok),
        dtype=torch.int64,
        device="cuda",
    )

    print(
        f"hidden_states: {hidden_states.norm().detach().cpu().item()} {hidden_states.shape} {hidden_states.dtype} {hidden_states.device}"
    )

    control_output, control_time_taken = run_moe(control, hidden_states, topk_scores, expert_indices)
    exp_output, exp_time_taken = run_moe(exp, hidden_states, topk_scores, expert_indices)

    diff = (control_output.detach().cpu() - exp_output.detach().cpu()).norm() / control_output.detach().cpu().norm()
    print(f"Diff: {diff}")
    print(f"Speedup: {control_time_taken / exp_time_taken}")


if __name__ == "__main__":
    main()
