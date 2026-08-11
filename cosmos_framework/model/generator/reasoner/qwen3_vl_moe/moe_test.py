# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time
from collections.abc import Callable

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe import moe as moe_impl
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import create_text_experts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    ENTROPY_EPSILON,
    AuxLossFreeLoadBalancingConfig,
    CosineRouter,
    CosineRouterConfig,
    Qwen3VLMoeTextSparseMoeBlock,
    _weighted_expert_counts,
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


# -----------------------------------------------------------------------------
# Padded streams
#
# A packed sequence is padded to a block-aligned length, and the padding rows travel through
# the MoE block rather than being sliced off, so that no shape depends on how many rows are
# real. These cover the other half of that deal: the padding has to leave no trace.
# -----------------------------------------------------------------------------


def test_weighted_expert_counts_ignores_the_masked_rows() -> None:
    """int32, like the histogram it stands in for: a Triton loop bound downstream needs an int."""
    expert_indices = torch.tensor([[0, 1], [1, 2], [3, 3]])  # [3,2]

    masked = _weighted_expert_counts(expert_indices, 4, torch.tensor([1.0, 1.0, 0.0]))  # [4]
    unmasked = _weighted_expert_counts(expert_indices, 4, torch.ones(3))  # [4]

    torch.testing.assert_close(masked, torch.tensor([1, 2, 1, 0], dtype=torch.int32))
    torch.testing.assert_close(unmasked, torch.tensor([1, 2, 1, 2], dtype=torch.int32))


def test_router_batch_mean_centering_ignores_padding_rows() -> None:
    """Centering is the one router statistic taken across rows, so padding would shift it."""
    router = CosineRouter(CosineRouterConfig(input_centering="batch_mean"), hidden_size=4)
    gate = nn.Linear(4, 3, bias=False)
    real = torch.randn(6, 4)  # [6,4]
    padded = torch.cat([real, torch.zeros(2, 4)])  # [8,4]
    token_weight = torch.tensor([1.0] * 6 + [0.0] * 2)  # [8]

    padded_logits = router(padded, gate, token_weight)  # [8,3]
    real_logits = router(real, gate)  # [6,3]

    torch.testing.assert_close(padded_logits[:6], real_logits)


def test_router_ema_centering_counts_real_rows_only() -> None:
    router = CosineRouter(CosineRouterConfig(input_centering="ema"), hidden_size=4)
    gate = nn.Linear(4, 3, bias=False)
    hidden_states = torch.randn(8, 4)  # [8,4]
    token_weight = torch.tensor([1.0] * 6 + [0.0] * 2)  # [8]

    router(hidden_states, gate, token_weight)

    torch.testing.assert_close(router.router_bias_count, torch.tensor([6.0]))
    torch.testing.assert_close(router.router_bias_sum, hidden_states[:6].sum(dim=0))


def test_callback_stats_count_pairs_with_and_without_a_token_weight() -> None:
    """A mask and no mask take different branches into the same scatter_add_.

    The unmasked branch is the one every non-padded stream uses, so it needs coverage that
    does not depend on the grouped-GEMM experts and therefore on a GPU.
    """
    config = Qwen3VLMoeTextConfig(
        hidden_size=8,
        moe_intermediate_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(config)
    expert_indices = torch.tensor([[0, 1], [2, 3], [1, 1]])  # [num_tokens,K]
    routing_probabilities = torch.full((3, config.num_experts), 0.25)  # [num_tokens,N_experts]
    stats = dict(
        num_tokens_per_expert=torch.tensor([1, 3, 1, 1]),  # [N_experts]
        num_tokens=torch.tensor([3]),
        routing_probabilities=routing_probabilities,
        expert_indices=expert_indices,
    )

    block._update_moe_callback_stats(**stats)

    assert block.coactivation_counts.sum().item() == 3
    assert block.coactivation_counts[0, 1].item() == 1
    assert block.coactivation_counts[2, 3].item() == 1
    assert block.coactivation_counts[1, 1].item() == 1

    block.coactivation_counts.zero_()
    block.sum_token_entropy.zero_()

    block._update_moe_callback_stats(**stats, token_weight=torch.tensor([1.0, 1.0, 0.0]))  # [num_tokens]

    # The masked row reaches neither the pair counts nor the entropy sum.
    assert block.coactivation_counts.sum().item() == 2
    assert block.coactivation_counts[1, 1].item() == 0
    uniform_entropy = -torch.log(torch.tensor(0.25) + ENTROPY_EPSILON).item()
    assert block.sum_token_entropy.item() == pytest.approx(2 * uniform_entropy, rel=1e-5)


class _RecordingExperts(nn.Module):
    """Stand-in for the experts that records what it was asked to process."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,  # [num_tokens,hidden_size]
        topk_scores: torch.Tensor,  # [num_tokens,top_k]
        expert_indices: torch.Tensor,  # [num_tokens,top_k]
        num_tokens_per_expert: torch.Tensor,  # [num_experts]
        token_mask: torch.Tensor | None,  # [num_tokens]
    ) -> torch.Tensor:  # [num_tokens,hidden_size]
        self.calls.append(
            (
                num_tokens_per_expert.clone(),
                expert_indices.clone(),
                None if token_mask is None else token_mask.clone(),
            )
        )
        return torch.zeros_like(hidden_states)


def test_the_experts_are_asked_to_process_the_real_rows_only() -> None:
    """The counts are also the map from an expert to its slots, so they gate the GEMM's work."""
    config = Qwen3VLMoeTextConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(config)
    block.experts = _RecordingExperts()
    num_real, num_padding = 5, 3
    hidden_states = torch.randn(num_real + num_padding, config.hidden_size)  # [8,16]
    token_mask = torch.arange(num_real + num_padding) < num_real  # [8]

    block(hidden_states, token_mask)

    counts, expert_indices, forwarded_mask = block.experts.calls[-1]
    assert counts.sum().item() == num_real * config.num_experts_per_tok
    torch.testing.assert_close(
        counts,
        torch.bincount(expert_indices[:num_real].reshape(-1), minlength=config.num_experts).to(counts.dtype),
    )
    # The sentinel group belongs to the experts, so the indices they receive are still routable
    # ones: the co-activation statistics index a [num_experts, num_experts] buffer with them.
    assert int(expert_indices.max()) < config.num_experts
    assert forwarded_mask is not None and bool(torch.equal(forwarded_mask, token_mask))


def test_padding_is_excluded_from_sample_load_balancing_statistics() -> None:
    config = Qwen3VLMoeTextConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(config)
    block.experts = _RecordingExperts()
    real = torch.randn(3, config.hidden_size)  # [N_real,H]
    padded = torch.cat([real, torch.randn(2, config.hidden_size)])  # [N_padded,H]
    real_token_mask = torch.ones(real.shape[0], dtype=torch.bool)  # [N_real]
    token_mask = torch.tensor([True, True, True, False, False])  # [N_padded]
    real_sample_ids = torch.tensor([0, 0, 1])  # [N_real]
    padded_sample_ids = torch.tensor([0, 0, 1, 2, 2])  # [N_padded]

    _, real_metadata = block(
        real,
        token_mask=real_token_mask,
        sample_ids=real_sample_ids,
        num_samples=2,
    )
    _, padded_metadata = block(
        padded,
        token_mask=token_mask,
        sample_ids=padded_sample_ids,
        num_samples=2,
    )

    assert real_metadata.sample_num_tokens_per_expert is not None
    assert padded_metadata.sample_num_tokens_per_expert is not None
    assert real_metadata.sample_num_tokens is not None
    assert padded_metadata.sample_num_tokens is not None
    assert real_metadata.sample_router_prob_sum_per_expert is not None
    assert padded_metadata.sample_router_prob_sum_per_expert is not None
    torch.testing.assert_close(
        padded_metadata.sample_num_tokens_per_expert,
        real_metadata.sample_num_tokens_per_expert,
    )
    torch.testing.assert_close(padded_metadata.sample_num_tokens, real_metadata.sample_num_tokens)
    torch.testing.assert_close(
        padded_metadata.sample_router_prob_sum_per_expert,
        real_metadata.sample_router_prob_sum_per_expert,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The grouped-GEMM experts have no CPU kernel.")
def test_the_grouped_gemm_costs_the_same_padded_as_unpadded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Padding rows must cost the GEMM nothing.

    The allocation stays at its padded upper bound either way, so the saving cannot show up
    there. It shows up in the group sizes, which are what the GEMM actually iterates over.
    """
    config = Qwen3VLMoeTextConfig(
        hidden_size=64,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(config)
    block.experts.init_weights(torch.device("cuda"))
    block = block.to(device="cuda", dtype=torch.bfloat16)

    group_sizes: list[torch.Tensor] = []
    run_experts = moe_impl._run_experts_grouped_mm

    def recording_run_experts(
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        group_sizes.append(num_tokens_per_expert.clone())  # [num_experts]
        return run_experts(gate_up_proj, down_proj, act_fn, x, num_tokens_per_expert, scores)

    monkeypatch.setattr(moe_impl, "_run_experts_grouped_mm", recording_run_experts)

    num_real, num_padding = 256, 256
    real = torch.randn(num_real, config.hidden_size, device="cuda", dtype=torch.bfloat16)  # [256,64]
    padded = torch.cat([real, torch.zeros(num_padding, config.hidden_size, device="cuda", dtype=torch.bfloat16)])
    token_mask = torch.arange(num_real + num_padding, device="cuda") < num_real  # [512]

    block(real)
    block(padded, token_mask)

    real_sizes, padded_sizes = group_sizes
    torch.testing.assert_close(padded_sizes, real_sizes)
    # And the rows skipped are the padding ones, not merely some rows: every slot of every
    # padding row would otherwise have to land in a group.
    assert int(padded_sizes.sum()) < (num_real + num_padding) * config.num_experts_per_tok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The grouped-GEMM experts have no CPU kernel.")
def test_padding_leaves_the_moe_output_and_its_routing_statistics_untouched() -> None:
    config = Qwen3VLMoeTextConfig(
        hidden_size=64,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(config)
    block.experts.init_weights(torch.device("cuda"))
    block = block.to(device="cuda", dtype=torch.bfloat16)

    num_real, num_padding = 24, 8
    real = torch.randn(num_real, config.hidden_size, device="cuda", dtype=torch.bfloat16)  # [24,64]
    padded = torch.cat([real, torch.zeros(num_padding, config.hidden_size, device="cuda", dtype=torch.bfloat16)])
    token_mask = torch.arange(num_real + num_padding, device="cuda") < num_real  # [32]

    real_out, real_lbl = block(real)
    real_counts = block.get_total_tokens_per_expert()  # [8], resets the buffer
    padded_out, padded_lbl = block(padded, token_mask)
    padded_counts = block.get_total_tokens_per_expert()  # [8]

    torch.testing.assert_close(padded_out[:num_real], real_out)
    assert bool((padded_out[num_real:] == 0).all())
    torch.testing.assert_close(padded_lbl.num_tokens_per_expert, real_lbl.num_tokens_per_expert)
    torch.testing.assert_close(padded_lbl.mean_router_prob_per_expert, real_lbl.mean_router_prob_per_expert)
    torch.testing.assert_close(padded_lbl.num_tokens, real_lbl.num_tokens)
    torch.testing.assert_close(padded_counts, real_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The grouped-GEMM experts have no CPU kernel.")
def test_a_non_finite_padding_row_reaches_neither_the_output_nor_the_expert_gradients() -> None:
    """Attention leaves whatever it likes in fully-masked padding rows, including NaN."""
    config = Qwen3VLMoeTextConfig(
        hidden_size=64,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    block = Qwen3VLMoeTextSparseMoeBlock(config)
    block.experts.init_weights(torch.device("cuda"))
    block = block.to(device="cuda", dtype=torch.bfloat16)

    num_real, num_padding = 24, 8
    hidden_states = torch.randn(
        num_real + num_padding, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )  # [32,64]
    with torch.no_grad():
        hidden_states[num_real:] = torch.nan
    token_mask = torch.arange(num_real + num_padding, device="cuda") < num_real  # [32]

    output, _ = block(hidden_states, token_mask)
    output.float().sum().backward()

    assert bool(torch.isfinite(output).all())
    assert bool(torch.isfinite(block.experts.gate_up_proj.grad).all())
    assert bool(torch.isfinite(block.experts.down_proj.grad).all())
    assert bool(torch.isfinite(block.gate.weight.grad).all())


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
