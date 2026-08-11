# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import cast

import pytest
import torch
from torch.distributed.tensor.device_mesh import DeviceMesh

from cosmos_framework.model.generator.algorithm.loss.load_balancing import (
    _scale_rank_sample_mean,
    compute_load_balancing_loss,
)
from cosmos_framework.model.generator.utils.load_balancing_stats import LBLMetadata, compute_sample_lbl_stats
from cosmos_framework.data.generator.sequence_packing.runtime import prepare_sequence_pack_metadata

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_compute_sample_lbl_stats_preserves_packed_boundaries() -> None:
    routing_probabilities = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.6, 0.1, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )  # [N,E]
    expert_indices = torch.tensor(
        [
            [0, 1],
            [1, 1],
            [2, 0],
            [2, 2],
        ]
    )  # [N,K]
    sample_ids = torch.tensor([0, 1, 1, 1])  # [N]

    counts, num_tokens, probability_sums = compute_sample_lbl_stats(
        routing_probabilities,
        expert_indices,
        sample_ids,
        num_samples=2,
    )

    torch.testing.assert_close(counts, torch.tensor([[1, 1, 0], [1, 2, 3]]))  # [B,E]
    torch.testing.assert_close(num_tokens, torch.tensor([[1], [3]]))  # [B,1]
    torch.testing.assert_close(
        probability_sums,
        torch.tensor([[0.7, 0.2, 0.1], [0.8, 1.1, 1.1]]),  # [B,E]
    )


def test_compute_sample_lbl_stats_ignores_cp_padding_and_noncontiguous_samples() -> None:
    routing_probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.5, 0.5],
        ]
    )  # [N,E]
    expert_indices = torch.tensor([[0], [1], [1], [0], [0]])  # [N,K]
    sample_ids = torch.tensor([1, 0, 2, 1, 2])  # [N], sample ID 2 is the padding sentinel

    counts, num_tokens, probability_sums = compute_sample_lbl_stats(
        routing_probabilities,
        expert_indices,
        sample_ids,
        num_samples=2,
    )

    torch.testing.assert_close(counts, torch.tensor([[0, 1], [2, 0]]))  # [B,E]
    torch.testing.assert_close(num_tokens, torch.tensor([[1], [2]]))  # [B,1]
    torch.testing.assert_close(probability_sums, torch.tensor([[0.2, 0.8], [1.6, 0.4]]))  # [B,E]


def test_cp_shard_sample_stats_sum_to_unsharded_stats() -> None:
    routing_probabilities = torch.tensor(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.4, 0.6],
        ]
    )  # [N,E]
    expert_indices = torch.tensor([[0], [1], [1], [0]])  # [N,K]
    sample_ids = torch.tensor([0, 0, 0, 1])  # [N], sample 0 crosses the CP shard boundary
    expected_stats = compute_sample_lbl_stats(
        routing_probabilities,
        expert_indices,
        sample_ids,
        num_samples=2,
    )

    rank0_stats = compute_sample_lbl_stats(
        routing_probabilities[:2],
        expert_indices[:2],
        sample_ids[:2],
        num_samples=2,
    )
    rank1_stats = compute_sample_lbl_stats(
        routing_probabilities[2:],
        expert_indices[2:],
        sample_ids[2:],
        num_samples=2,
    )

    for expected, rank0, rank1 in zip(expected_stats, rank0_stats, rank1_stats):
        torch.testing.assert_close(rank0 + rank1, expected)


def test_sequence_pack_sample_ids_follow_samples_across_multiple_splits() -> None:
    metadata = prepare_sequence_pack_metadata(
        sample_lens=[4, 3],
        split_lens=[1, 1, 2, 2, 1],
        attn_modes=["causal", "full", "causal", "causal", "full"],
        packed_und_token_indexes=torch.tensor([0, 2, 3, 4, 5]),  # [N_causal]
        device=torch.device("cpu"),
    ).as_sequence_pack_fields()

    torch.testing.assert_close(metadata["_causal_sample_ids"], torch.tensor([0, 0, 0, 1, 1]))  # [N_causal]
    torch.testing.assert_close(metadata["_full_only_sample_ids"], torch.tensor([0, 1]))  # [N_full]


def test_sample_load_balancing_averages_packed_samples_equally() -> None:
    sample_router_prob_sum_per_expert = torch.tensor(
        [[[0.9, 0.1], [0.6, 2.4]]],
        requires_grad=True,
    )  # [num_layers,num_samples,num_experts]
    metadata = LBLMetadata(
        num_tokens_per_expert=torch.tensor([[1, 3]]),  # [num_layers,num_experts]
        num_tokens=torch.tensor([[4]]),  # [num_layers,1]
        mean_router_prob_per_expert=torch.tensor([[0.375, 0.625]]),  # [num_layers,num_experts]
        top_k=torch.tensor([[1]]),  # [num_layers,1]
        sample_num_tokens_per_expert=torch.tensor([[[1, 0], [0, 3]]]),  # [num_layers,num_samples,num_experts]
        sample_num_tokens=torch.tensor([[[1], [3]]]),  # [num_layers,num_samples,1]
        sample_router_prob_sum_per_expert=sample_router_prob_sum_per_expert,
    )

    loss = compute_load_balancing_loss(
        metadata,
        coeff=1.0,
        method="sample",
        device_mesh=None,
    )

    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.7))
    loss.backward()
    torch.testing.assert_close(
        sample_router_prob_sum_per_expert.grad,
        torch.tensor([[[1.0, 0.0], [0.0, 1.0 / 3.0]]]),  # [num_layers,num_samples,num_experts]
    )


def test_sample_load_balancing_scales_rank_means_for_fsdp_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDeviceMesh:
        def size(self) -> int:
            return 2

    device_mesh = cast(DeviceMesh, _FakeDeviceMesh())

    def _fake_sum_over_mesh(local_tensor: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
        assert mesh is device_mesh
        return local_tensor.new_tensor([4.0])  # [num_layers]

    monkeypatch.setattr(
        "cosmos_framework.model.generator.algorithm.loss.load_balancing._sum_over_mesh",
        _fake_sum_over_mesh,
    )

    equal_count_mean = torch.tensor([0.75])  # [num_layers]
    equal_count_scaled_mean = _scale_rank_sample_mean(
        equal_count_mean,
        torch.tensor([2.0]),  # [num_layers]
        device_mesh,
    )  # [num_layers]
    torch.testing.assert_close(equal_count_scaled_mean, equal_count_mean)

    empty_rank_scaled_mean = _scale_rank_sample_mean(
        torch.tensor([0.0]),  # [num_layers]
        torch.tensor([0.0]),  # [num_layers]
        device_mesh,
    )  # [num_layers]
    torch.testing.assert_close(empty_rank_scaled_mean, torch.tensor([0.0]))

    rank0_local_mean = torch.tensor([0.5], requires_grad=True)  # [num_layers]
    rank1_local_mean = torch.tensor([1.5], requires_grad=True)  # [num_layers]
    rank0_scaled_mean = _scale_rank_sample_mean(
        rank0_local_mean,
        torch.tensor([1.0]),  # [num_layers]
        device_mesh,
    )  # [num_layers]
    rank1_scaled_mean = _scale_rank_sample_mean(
        rank1_local_mean,
        torch.tensor([3.0]),  # [num_layers]
        device_mesh,
    )  # [num_layers]

    # Simulate FSDP's gradient average across the two ranks. The result is the
    # global sample mean: (1 * 0.5 + 3 * 1.5) / 4 = 1.25.
    fsdp_averaged_loss = (rank0_scaled_mean + rank1_scaled_mean).mean() / device_mesh.size()  # []
    torch.testing.assert_close(fsdp_averaged_loss, torch.tensor(1.25))
    fsdp_averaged_loss.backward()
    torch.testing.assert_close(rank0_local_mean.grad, torch.tensor([0.25]))  # [num_layers]
    torch.testing.assert_close(rank1_local_mean.grad, torch.tensor([0.75]))  # [num_layers]


def test_sample_load_balancing_requires_sample_metadata() -> None:
    metadata = LBLMetadata(
        num_tokens_per_expert=torch.tensor([[1, 1]]),  # [num_layers,num_experts]
        num_tokens=torch.tensor([[2]]),  # [num_layers,1]
        mean_router_prob_per_expert=torch.tensor([[0.5, 0.5]]),  # [num_layers,num_experts]
        top_k=torch.tensor([[1]]),  # [num_layers,1]
    )

    with pytest.raises(ValueError, match="requires per-sample router metadata"):
        compute_load_balancing_loss(
            metadata,
            coeff=1.0,
            method="sample",
            device_mesh=None,
        )


def test_sample_load_balancing_returns_zero_without_valid_samples() -> None:
    sample_router_prob_sum_per_expert = torch.zeros((1, 2, 2), requires_grad=True)  # [num_layers,num_samples,E]
    metadata = LBLMetadata(
        num_tokens_per_expert=torch.zeros((1, 2), dtype=torch.int64),  # [num_layers,E]
        num_tokens=torch.zeros((1, 1), dtype=torch.int64),  # [num_layers,1]
        mean_router_prob_per_expert=torch.zeros((1, 2)),  # [num_layers,E]
        top_k=torch.tensor([[1]]),  # [num_layers,1]
        sample_num_tokens_per_expert=torch.zeros((1, 2, 2), dtype=torch.int64),  # [num_layers,num_samples,E]
        sample_num_tokens=torch.zeros((1, 2, 1), dtype=torch.int64),  # [num_layers,num_samples,1]
        sample_router_prob_sum_per_expert=sample_router_prob_sum_per_expert,
    )

    loss = compute_load_balancing_loss(
        metadata,
        coeff=1.0,
        method="sample",
        device_mesh=None,
    )

    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(
        sample_router_prob_sum_per_expert.grad, torch.zeros_like(sample_router_prob_sum_per_expert)
    )


def test_sample_load_balancing_matches_local_global_top_k_scaling() -> None:
    top_k = torch.tensor([[2]])  # [num_layers,1]
    metadata = LBLMetadata(
        num_tokens_per_expert=torch.tensor([[2, 2, 0, 0]]),  # [num_layers,num_experts]
        num_tokens=torch.tensor([[2]]),  # [num_layers,1]
        mean_router_prob_per_expert=torch.tensor([[0.4, 0.3, 0.2, 0.1]]),  # [num_layers,num_experts]
        top_k=top_k,
        sample_num_tokens_per_expert=torch.tensor([[[2, 2, 0, 0]]]),  # [num_layers,num_samples,num_experts]
        sample_num_tokens=torch.tensor([[[2]]]),  # [num_layers,num_samples,1]
        sample_router_prob_sum_per_expert=torch.tensor(
            [[[0.8, 0.6, 0.4, 0.2]]]
        ),  # [num_layers,num_samples,num_experts]
    )

    sample_loss = compute_load_balancing_loss(
        metadata,
        coeff=1.0,
        method="sample",
        device_mesh=None,
    )
    local_loss = compute_load_balancing_loss(
        metadata,
        coeff=1.0,
        method="local",
        device_mesh=None,
    )

    assert sample_loss is not None
    assert local_loss is not None
    # Without / top_k the balanced assignment mass would sum to 2 and the loss would be 2.8.
    torch.testing.assert_close(sample_loss, torch.tensor(1.4))
    torch.testing.assert_close(sample_loss, local_loss)
