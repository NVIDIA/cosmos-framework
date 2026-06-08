# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time

import torch
from torch import nn

from cosmos_framework.model.vfm.vlm.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.vfm.vlm.qwen3_vl_moe.moe import create_text_experts


def run_moe(
    mod: nn.Module,
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
):
    num_warmup_iterations = 10
    num_timing_iterations = 100

    for _ in range(num_warmup_iterations):
        with torch.no_grad():
            output = mod(hidden_states, topk_scores, expert_indices, num_tokens_per_expert)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_timing_iterations):
        with torch.no_grad():
            output = mod(hidden_states, topk_scores, expert_indices, num_tokens_per_expert)
    torch.cuda.synchronize()
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

    control.init_weights(torch.device("cpu"))
    exp.load_state_dict(control.state_dict())

    control = control.to(device="cuda", dtype=torch.bfloat16)
    exp = exp.to(device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn(
        num_tokens,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_scores = torch.rand(
        num_tokens,
        config.num_experts_per_tok,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    expert_indices = torch.stack(
        [torch.randperm(config.num_experts, device="cuda")[: config.num_experts_per_tok] for _ in range(num_tokens)]
    ).to(torch.int64)
    num_tokens_per_expert = torch.histc(
        expert_indices.to(dtype=torch.int32).view(-1),
        bins=config.num_experts,
        min=0,
        max=config.num_experts - 1,
    )

    print(
        f"hidden_states: {hidden_states.norm().detach().cpu().item()} {hidden_states.shape} {hidden_states.dtype} {hidden_states.device}"
    )

    control_output, control_time_taken = run_moe(
        control, hidden_states, topk_scores, expert_indices, num_tokens_per_expert
    )
    exp_output, exp_time_taken = run_moe(exp, hidden_states, topk_scores, expert_indices, num_tokens_per_expert)

    diff = (control_output.detach().cpu() - exp_output.detach().cpu()).norm() / control_output.detach().cpu().norm()
    print(f"Diff: {diff}")
    print(f"Speedup: {control_time_taken / exp_time_taken}")
    torch.testing.assert_close(control_output, exp_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
