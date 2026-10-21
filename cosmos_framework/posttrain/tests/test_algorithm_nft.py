# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU correctness tests for the DiffusionNFT algorithm kernel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.algorithm.nft import (
    MODEL_BATCH_KEY,
    NFTObjectiveConfig,
    make_objective,
    nft_loss_terms,
    nft_objective,
    optimality_weight,
    optimality_weight_std,
    unpack_nft_batch,
)
from cosmos_framework.posttrain.comm.bulk import BulkStore
from cosmos_framework.posttrain.comm.codec import materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.model.protocol import FlowCondition, FlowConditionMask
from cosmos_framework.posttrain.schema import (
    ObsBundle,
    Status,
    Trajectory,
    Transition,
    WFMConditioningInput,
)
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel
from cosmos_framework.posttrain.utils.wfm import repeat_conditioning_batch


class _NFTForwardCountingToy(ToyFlowModel):
    """Record whether the collective-bearing NFT model stage was entered."""

    velocity_prediction_calls: int
    fail_stochastic_preparation: bool

    def __init__(self, *, fail_stochastic_preparation: bool = False) -> None:
        super().__init__(dim=2, seed=0)
        self.velocity_prediction_calls = 0
        self.fail_stochastic_preparation = fail_stochastic_preparation

    def draw_nft_sigma_epsilon(self, x0_shape: torch.Size, cond: FlowCondition) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fail_stochastic_preparation:
            raise ValueError("injected rank-local stochastic preparation failure")
        return super().draw_nft_sigma_epsilon(x0_shape, cond)

    def nft_velocity_predictions(
        self,
        *,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.velocity_prediction_calls += 1
        return super().nft_velocity_predictions(
            x0=x0,
            cond=cond,
            sigma=sigma,
            epsilon=epsilon,
            condition_mask=condition_mask,
        )


def _asymmetric_nft_preparation_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        model = _NFTForwardCountingToy(fail_stochastic_preparation=rank == 1)
        trajectories = [_trajectory(0, torch.zeros(1, 2), advantage=0.0)]  # action [1,D]
        with pytest.raises(RuntimeError) as failed:
            nft_objective(
                model,
                trajectories,
                NFTObjectiveConfig(kl_beta=0.0),
            )
        message = str(failed.value)
        assert "NFT objective preparation failed on one or more ranks" in message
        assert "rank 1: ValueError" in message
        assert "injected rank-local stochastic preparation failure" in message
        assert model.velocity_prediction_calls == 0
    finally:
        dist.destroy_process_group()


def _trajectory(
    index: int,
    latent: torch.Tensor,
    advantage: float | None = None,
    *,
    template_identity: str = "template-0",
) -> Trajectory:
    source = WFMConditioningInput(
        id="group-0",
        template_identity=template_identity,
        model_batch={"x": torch.tensor([[1.0, -1.0]])},  # [1,D]
        prompt="shared prompt",
    )
    transition = Transition(
        obs=ObsBundle(representations={MODEL_BATCH_KEY: source}, id=f"prompt-{index}"),
        action=latent,
        advantage=advantage,
    )
    return Trajectory(
        transitions=[transition],
        id=f"sample-{index}",
        group_id="group-0",
        status=Status.COMPLETED,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_optimality_weight_linear_and_sigmoid() -> None:
    advantages = torch.tensor([-9.0, -5.0, 0.0, 5.0, 9.0])  # [B]
    linear = optimality_weight(advantages, NFTObjectiveConfig())  # [B]
    torch.testing.assert_close(linear, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))

    sigmoid = optimality_weight(
        torch.tensor([-1.0, 0.0, 1.0]),  # [B]
        NFTObjectiveConfig(r_map="sigmoid", r_tau=0.5),
    )  # [B]
    torch.testing.assert_close(sigmoid, torch.sigmoid(torch.tensor([-2.0, 0.0, 2.0])))


@pytest.mark.L0
@pytest.mark.CPU
def test_sigmoid_policy_loss_scale_is_independent_of_advantage_bounds() -> None:
    inputs = {
        "x0": torch.zeros(1, 2),  # [B,D]
        "x_t": torch.ones(1, 2),  # [B,D]
        "t_effective": torch.full((1, 1), 0.5),  # [B,1]
        "v_policy": torch.tensor([[0.25, -0.5]], requires_grad=True),  # [B,D]
        "v_old": torch.tensor([[1.0, 2.0]]),  # [B,D]
        "v_ref": None,
        "advantages": torch.ones(1),  # [B]
    }
    baseline = NFTObjectiveConfig(
        kl_beta=0.0,
        policy_loss_scale=3.0,
        r_map="sigmoid",
        advantage_low=-5.0,
        advantage_high=5.0,
    )
    unrelated_bounds = NFTObjectiveConfig(
        kl_beta=0.0,
        policy_loss_scale=3.0,
        r_map="sigmoid",
        advantage_low=-50.0,
        advantage_high=-40.0,
    )
    doubled_scale = NFTObjectiveConfig(
        kl_beta=0.0,
        policy_loss_scale=6.0,
        r_map="sigmoid",
        advantage_low=-5.0,
        advantage_high=5.0,
    )

    _baseline_loss, baseline_metrics = nft_loss_terms(**inputs, config=baseline)
    _bounded_loss, bounded_metrics = nft_loss_terms(**inputs, config=unrelated_bounds)
    _scaled_loss, scaled_metrics = nft_loss_terms(**inputs, config=doubled_scale)

    torch.testing.assert_close(baseline_metrics["nft_policy_loss"], bounded_metrics["nft_policy_loss"])
    torch.testing.assert_close(scaled_metrics["nft_policy_loss"], 2.0 * baseline_metrics["nft_policy_loss"])


@pytest.mark.L0
@pytest.mark.CPU
def test_optimality_weight_std_uses_mergeable_global_moments() -> None:
    rank_local_weights = torch.tensor([0.5, 0.50390625], dtype=torch.bfloat16)  # [G]
    fp32_weights = rank_local_weights.float()  # [G]
    mean = fp32_weights.mean()  # []
    second_moment = fp32_weights.square().mean()  # []

    actual = optimality_weight_std(mean, second_moment)

    assert actual > 0.0
    assert actual == pytest.approx(float(fp32_weights.std(correction=0).item()))


@pytest.mark.L0
@pytest.mark.CPU
def test_optimality_weight_std_rejects_bad_moments_but_clamps_roundoff() -> None:
    assert optimality_weight_std(0.5, 0.25 - 1e-8) == 0.0
    with pytest.raises(ValueError, match="must be scalar"):
        optimality_weight_std(torch.tensor([0.1, 0.2]), 0.1)
    with pytest.raises(ValueError, match="must be finite"):
        optimality_weight_std(float("nan"), 0.1)
    with pytest.raises(ValueError, match="moments are inconsistent"):
        optimality_weight_std(0.8, 0.1)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": 0.0},
        {"beta": float("nan")},
        {"kl_beta": -1.0},
        pytest.param({"policy_loss_scale": 0.0}, id="policy-loss-zero-scale"),
        {"advantage_low": 1.0, "advantage_high": 1.0},
        {"r_map": "typo"},
        {"r_map": "sigmoid", "r_tau": 0.0},
    ],
)
def test_objective_config_rejects_invalid_math(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        NFTObjectiveConfig(**kwargs)


def _independent_oracle(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    v_policy: torch.Tensor,
    v_old: torch.Tensor,
    v_ref: torch.Tensor,
    advantages: torch.Tensor,
    config: NFTObjectiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent scalar-loop transcription of Cosmos3 WFM NFT."""
    losses: list[torch.Tensor] = []
    for index in range(x0.shape[0]):
        r = (advantages[index].clamp(config.advantage_low, config.advantage_high) - config.advantage_low) / (
            config.advantage_high - config.advantage_low
        )  # []
        v_pos = config.beta * v_policy[index] + (1.0 - config.beta) * v_old[index].detach()  # [D]
        v_neg = (1.0 + config.beta) * v_old[index].detach() - config.beta * v_policy[index]  # [D]
        pos_residual = x_t[index] - t[index] * v_pos - x0[index]  # [D]
        neg_residual = x_t[index] - t[index] * v_neg - x0[index]  # [D]
        pos_weight = pos_residual.detach().abs().mean().clamp_min(1e-5)  # []
        neg_weight = neg_residual.detach().abs().mean().clamp_min(1e-5)  # []
        pos_error = (pos_residual.square() / pos_weight).mean()  # []
        neg_error = (neg_residual.square() / neg_weight).mean()  # []
        losses.append((r * pos_error + (1.0 - r) * neg_error) / config.beta)
    policy = torch.stack(losses).mean() * config.policy_loss_scale  # []
    kl = (v_policy - v_ref.detach()).square().mean()  # []
    return policy + config.kl_beta * kl, policy, kl


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_loss_matches_independent_closed_form_and_gradients() -> None:
    generator = torch.Generator().manual_seed(7)
    x0 = torch.randn(3, 5, generator=generator)  # [B,D]
    epsilon = torch.randn(3, 5, generator=generator)  # [B,D]
    t = torch.tensor([[0.1], [0.5], [0.9]])  # [B,1]
    x_t = (1.0 - t) * x0 + t * epsilon  # [B,D]
    v_policy = torch.randn(3, 5, generator=generator, requires_grad=True)  # [B,D]
    v_old = torch.randn(3, 5, generator=generator, requires_grad=True)  # [B,D]
    v_ref = torch.randn(3, 5, generator=generator, requires_grad=True)  # [B,D]
    advantages = torch.tensor([-5.0, 0.0, 5.0])  # [B]
    config = NFTObjectiveConfig(beta=0.7, kl_beta=0.2, advantage_low=-3.0, advantage_high=7.0)

    loss, metrics = nft_loss_terms(
        x0=x0,
        x_t=x_t,
        t_effective=t,
        v_policy=v_policy,
        v_old=v_old,
        v_ref=v_ref,
        advantages=advantages,
        config=config,
    )
    expected, expected_policy, expected_kl = _independent_oracle(x0, x_t, t, v_policy, v_old, v_ref, advantages, config)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["nft_policy_loss"], expected_policy)
    torch.testing.assert_close(metrics["nft_kl_loss"], expected_kl)
    expected_r = optimality_weight(advantages, config).float()  # [B]
    torch.testing.assert_close(metrics["nft_r_second_moment"], expected_r.square().mean())
    assert "nft_r_min" not in metrics
    assert "nft_r_max" not in metrics
    (grad_actual,) = torch.autograd.grad(loss, v_policy, retain_graph=True)  # [B,D]
    (grad_expected,) = torch.autograd.grad(expected, v_policy, retain_graph=True)  # [B,D]
    torch.testing.assert_close(grad_actual, grad_expected)
    loss.backward()
    assert v_policy.grad is not None and torch.isfinite(v_policy.grad).all()
    assert v_old.grad is None, "old-policy predictions must remain stop-gradient"
    assert v_ref.grad is None, "reference predictions must remain stop-gradient"


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_uses_moving_old_for_direction_and_frozen_ref_only_for_kl() -> None:
    x0 = torch.zeros(1, 2)  # [B,D]
    x_t = torch.ones(1, 2)  # [B,D]
    t = torch.full((1, 1), 0.5)  # [B,1]
    v_policy = torch.tensor([[0.25, -0.5]], requires_grad=True)  # [B,D]
    v_old = torch.tensor([[1.0, 2.0]], requires_grad=True)  # [B,D]
    v_ref = torch.tensor([[-3.0, 4.0]], requires_grad=True)  # [B,D]
    changed_old = v_old + 1.0  # [B,D]
    changed_ref = v_ref + 1.0  # [B,D]
    config = NFTObjectiveConfig(kl_beta=0.5)

    _loss, metrics = nft_loss_terms(
        x0=x0,
        x_t=x_t,
        t_effective=t,
        v_policy=v_policy,
        v_old=v_old,
        v_ref=v_ref,
        advantages=torch.zeros(1),  # [B]
        config=config,
    )
    _changed_ref_loss, changed_ref_metrics = nft_loss_terms(
        x0=x0,
        x_t=x_t,
        t_effective=t,
        v_policy=v_policy,
        v_old=v_old,
        v_ref=changed_ref,
        advantages=torch.zeros(1),  # [B]
        config=config,
    )
    _changed_old_loss, changed_old_metrics = nft_loss_terms(
        x0=x0,
        x_t=x_t,
        t_effective=t,
        v_policy=v_policy,
        v_old=changed_old,
        v_ref=v_ref,
        advantages=torch.zeros(1),  # [B]
        config=config,
    )

    torch.testing.assert_close(metrics["nft_policy_loss"], changed_ref_metrics["nft_policy_loss"])
    assert not torch.equal(metrics["nft_kl_loss"], changed_ref_metrics["nft_kl_loss"])
    assert not torch.equal(metrics["nft_policy_loss"], changed_old_metrics["nft_policy_loss"])
    torch.testing.assert_close(metrics["nft_kl_loss"], changed_old_metrics["nft_kl_loss"])


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_kl_zero_needs_no_frozen_reference_prediction() -> None:
    loss, metrics = nft_loss_terms(
        x0=torch.zeros(1, 2),  # [B,D]
        x_t=torch.ones(1, 2),  # [B,D]
        t_effective=torch.full((1, 1), 0.5),  # [B,1]
        v_policy=torch.zeros(1, 2, requires_grad=True),  # [B,D]
        v_old=torch.ones(1, 2),  # [B,D]
        v_ref=None,
        advantages=torch.zeros(1),  # [B]
        config=NFTObjectiveConfig(kl_beta=0.0),
    )

    assert torch.isfinite(loss)
    assert metrics["nft_kl_loss"].item() == 0.0

    _loss_with_ref, metrics_with_ref = nft_loss_terms(
        x0=torch.zeros(1, 2),  # [B,D]
        x_t=torch.ones(1, 2),  # [B,D]
        t_effective=torch.full((1, 1), 0.5),  # [B,1]
        v_policy=torch.zeros(1, 2, requires_grad=True),  # [B,D]
        v_old=torch.ones(1, 2),  # [B,D]
        v_ref=torch.full((1, 2), float("nan")),  # [B,D]
        advantages=torch.zeros(1),  # [B]
        config=NFTObjectiveConfig(kl_beta=0.0),
    )
    assert torch.isfinite(_loss_with_ref)
    assert metrics_with_ref["nft_kl_loss"].item() == 0.0

    with pytest.raises(ValueError, match="requires a frozen reference prediction"):
        nft_loss_terms(
            x0=torch.zeros(1, 2),  # [B,D]
            x_t=torch.ones(1, 2),  # [B,D]
            t_effective=torch.full((1, 1), 0.5),  # [B,1]
            v_policy=torch.zeros(1, 2, requires_grad=True),  # [B,D]
            v_old=torch.ones(1, 2),  # [B,D]
            v_ref=None,
            advantages=torch.zeros(1),  # [B]
            config=NFTObjectiveConfig(kl_beta=1.0),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_nonfinite_propagates_to_trainer_guard() -> None:
    x0 = torch.zeros(1, 2)  # [B,D]
    x_t = torch.zeros(1, 2)  # [B,D]
    v_policy = torch.tensor([[float("nan"), 0.0]], requires_grad=True)  # [B,D]
    loss, metrics = nft_loss_terms(
        x0=x0,
        x_t=x_t,
        t_effective=torch.ones(1, 1),  # [B,1]
        v_policy=v_policy,
        v_old=torch.zeros(1, 2),  # [B,D]
        v_ref=torch.zeros(1, 2),  # [B,D]
        advantages=torch.zeros(1),  # [B]
        config=NFTObjectiveConfig(),
    )
    assert not torch.isfinite(loss)
    assert not torch.isfinite(metrics["nft_loss"])


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_condition_mask_keeps_conditioned_position_clean() -> None:
    x0 = torch.tensor([[2.0, 3.0]])  # [B,D]
    epsilon = torch.tensor([[10.0, 20.0]])  # [B,D]
    t_effective = torch.tensor([[0.0, 0.75]])  # [B,D]
    x_t = (1.0 - t_effective) * x0 + t_effective * epsilon  # [B,D]
    assert x_t[0, 0] == x0[0, 0]
    loss, _metrics = nft_loss_terms(
        x0=x0,
        x_t=x_t,
        t_effective=t_effective,
        v_policy=torch.zeros_like(x0, requires_grad=True),  # [B,D]
        v_old=torch.zeros_like(x0),  # [B,D]
        v_ref=torch.zeros_like(x0),  # [B,D]
        advantages=torch.ones(1),  # [B]
        config=NFTObjectiveConfig(),
    )
    assert torch.isfinite(loss)


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_objective_runs_split_prediction_adapter_without_kl() -> None:
    model = ToyFlowModel(dim=2, seed=0)
    with torch.no_grad():
        for parameter in model.net_old.parameters():
            parameter.add_(0.5)
    trajectories = [
        _trajectory(1, torch.tensor([[0.5, -0.5]]), advantage=-1.0),  # [1,D]
        _trajectory(2, torch.tensor([[1.0, 0.25]]), advantage=3.0),  # [1,D]
    ]
    sigma = torch.tensor([[0.25], [0.75]])  # [B,1]
    epsilon = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # [B,D]
    loss, metrics = nft_objective(
        model,
        trajectories,
        NFTObjectiveConfig(kl_beta=0.0),
        sigma=sigma,
        epsilon=epsilon,
    )
    assert torch.isfinite(loss)
    assert set(metrics) >= {"nft_loss", "nft_policy_loss", "nft_kl_loss", "nft_r_mean"}
    torch.testing.assert_close(metrics["nft_advantage_mean"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["nft_r_mean"], torch.tensor(0.6))

    zero_advantage_trajectories = [
        _trajectory(1, torch.tensor([[0.5, -0.5]]), advantage=0.0),  # [1,D]
        _trajectory(2, torch.tensor([[1.0, 0.25]]), advantage=0.0),  # [1,D]
    ]
    zero_advantage_loss, _zero_advantage_metrics = nft_objective(
        model,
        zero_advantage_trajectories,
        NFTObjectiveConfig(kl_beta=0.0),
        sigma=sigma,
        epsilon=epsilon,
    )
    assert not torch.equal(loss.detach(), zero_advantage_loss.detach())
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.net.parameters())
    assert all(parameter.grad is None for parameter in model.net_old.parameters())
    assert all(parameter.grad is None for parameter in model.net_ref.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_objective_runs_split_prediction_adapter_with_kl() -> None:
    model = ToyFlowModel(dim=2, seed=0)
    trajectories = [_trajectory(0, torch.zeros(1, 2), advantage=0.0)]  # action [1,D]
    with torch.no_grad():
        model.net_ref.weight.add_(1.0)

    loss, metrics = nft_objective(model, trajectories, NFTObjectiveConfig(kl_beta=1.0))

    assert torch.isfinite(loss)
    assert metrics["nft_kl_loss"].item() > 0.0
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.net.parameters())
    assert all(parameter.grad is None for parameter in model.net_old.parameters())
    assert all(parameter.grad is None for parameter in model.net_ref.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_objective_rejects_missing_kl_reference_before_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ToyFlowModel()
    model.net_ref = None

    def unexpected_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("positive-KL guard must run before NFT preparation or RNG draws")

    monkeypatch.setattr(model, "encode_and_condition", unexpected_call)
    monkeypatch.setattr(model, "draw_nft_sigma_epsilon", unexpected_call)
    rng_before = torch.random.get_rng_state()  # [rng_state]

    with pytest.raises(ValueError, match="independent frozen reference"):
        nft_objective(model, [], NFTObjectiveConfig(kl_beta=1.0))

    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_objective_maps_every_internal_metric_to_train_namespace() -> None:
    objective = make_objective(NFTObjectiveConfig(kl_beta=0.0))
    model = ToyFlowModel(dim=2, seed=0)
    trajectories = [
        _trajectory(0, torch.tensor([[0.5, -0.5]]), advantage=-1.0),  # [1,D]
        _trajectory(1, torch.tensor([[1.0, 0.25]]), advantage=1.0),  # [1,D]
    ]
    trajectories[0].transitions[0].reward = {"final": 2.0}
    trajectories[1].transitions[0].reward = {"final": 4.0}

    _loss, internal = objective(model, trajectories)
    public = objective.adapt_metrics(internal, "train")

    assert set(public) == {
        "train/loss_avg",
        "train/policy_loss",
        "train/kl_loss",
        "train/advantage_mean",
        "train/optimality_mean",
        "train/optimality_second_moment",
        "train/positive_error",
        "train/negative_error",
        "train/reward_mean",
    }
    torch.testing.assert_close(public["train/reward_mean"], torch.tensor(3.0))


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_objective_rejects_partial_stochastic_overrides_before_drawing() -> None:
    class _DrawCountingToy(ToyFlowModel):
        def __init__(self) -> None:
            super().__init__(dim=2, seed=0)
            self.draw_calls = 0

        def draw_nft_sigma_epsilon(
            self, x0_shape: torch.Size, cond: FlowCondition
        ) -> tuple[torch.Tensor, torch.Tensor]:  # returns sigma [B,1], epsilon [B,D]
            self.draw_calls += 1
            return super().draw_nft_sigma_epsilon(x0_shape, cond)

    model = _DrawCountingToy()
    trajectories = [_trajectory(0, torch.zeros(1, 2), advantage=0.0)]  # action [1,D]
    sigma = torch.full((1, 1), 0.5)  # [B,1]
    epsilon = torch.zeros(1, 2)  # [B,D]

    with pytest.raises(ValueError, match="supplied together"):
        nft_objective(model, trajectories, NFTObjectiveConfig(kl_beta=0.0), sigma=sigma)
    with pytest.raises(ValueError, match="supplied together"):
        nft_objective(model, trajectories, NFTObjectiveConfig(kl_beta=0.0), epsilon=epsilon)

    assert model.draw_calls == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_conditioning_batch_repetition_rejects_malformed_batch_templates() -> None:
    base = WFMConditioningInput(
        id="group-0",
        template_identity="template-0",
        model_batch={},
        prompt="shared prompt",
    )
    for value, expected in (
        (torch.tensor(1.0), "shape"),
        (torch.ones(2, 3), "shape"),  # [2,D]
        ([], "length"),
        (["first", "second"], "length"),
    ):
        with pytest.raises(ValueError, match=expected):
            repeat_conditioning_batch(replace(base, model_batch={"field": value}), 2)

    repeated = repeat_conditioning_batch(
        replace(base, model_batch={"tensor": torch.ones(1, 3), "label": "prompt"}),  # tensor [1,D]
        2,
    )
    assert repeated["tensor"].shape == (2, 3)
    assert repeated["label"] == ["prompt", "prompt"]


def _codec_round_trip(trajectories: list[Trajectory]) -> list[Trajectory]:
    store = BulkStore()
    packed = pack_payload(trajectories, store, threshold=0)
    materialized = materialize_payload(packed, LocalStoreMover(cross_process=True))
    assert isinstance(materialized, list)
    return materialized


@pytest.mark.L0
@pytest.mark.CPU
def test_conditioning_template_identity_accepts_codec_reconstructed_group() -> None:
    trajectories = [
        _trajectory(0, torch.zeros(1, 2), advantage=-1.0),  # action [1,D]
        _trajectory(1, torch.ones(1, 2), advantage=1.0),  # action [1,D]
    ]
    materialized = _codec_round_trip(trajectories)
    first = materialized[0].transitions[0].obs.representations[MODEL_BATCH_KEY]
    second = materialized[1].transitions[0].obs.representations[MODEL_BATCH_KEY]
    assert first is not second

    source_batch, x0, advantages = unpack_nft_batch(materialized)

    assert source_batch["x"].shape == (2, 2)
    assert x0.shape == (2, 2)
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0]))


@pytest.mark.L0
@pytest.mark.CPU
def test_conditioning_template_identity_rejects_mismatch_after_codec_round_trip() -> None:
    trajectories = [
        _trajectory(0, torch.zeros(1, 2), advantage=-1.0, template_identity="template-a"),  # action [1,D]
        _trajectory(1, torch.ones(1, 2), advantage=1.0, template_identity="template-b"),  # action [1,D]
    ]

    with pytest.raises(ValueError, match="one conditioning template"):
        unpack_nft_batch(_codec_round_trip(trajectories))


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_asymmetric_nft_preparation_error_converges_before_model_forward(tmp_path: Path) -> None:
    """A local preparation failure reaches every peer before the first policy/old forward."""
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "nft-preparation-rendezvous")
    mp.spawn(
        _asymmetric_nft_preparation_worker,
        args=(world_size, rendezvous_path),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_uniform_timestep_preserves_reference_draw_order() -> None:
    events: list[str] = []
    model = ToyFlowModel(dim=2, seed=0)
    trajectories = [_trajectory(0, torch.zeros(1, 2), advantage=0.0)]  # action [1,D]
    original_rand = torch.rand
    original_randn = torch.randn
    rand_calls = 0

    def record_rand(*args: Any, **kwargs: Any) -> torch.Tensor:  # returns [B,1]
        nonlocal rand_calls
        events.append("training_sigma" if rand_calls == 0 else "uniform_timestep")
        rand_calls += 1
        return original_rand(*args, **kwargs)

    def record_randn(*args: Any, **kwargs: Any) -> torch.Tensor:  # returns [B,D]
        events.append("epsilon")
        return original_randn(*args, **kwargs)

    torch.manual_seed(17)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch, "rand", record_rand)
        monkeypatch.setattr(torch, "randn", record_randn)
        nft_objective(model, trajectories, NFTObjectiveConfig(kl_beta=0.0, uniform_timestep=True))

    assert events == ["epsilon", "training_sigma", "uniform_timestep"]


@pytest.mark.L0
@pytest.mark.CPU
def test_uniform_timestep_preserves_adapter_sigma_shape_device_and_dtype() -> None:
    class _AdapterShapedSigmaToy(ToyFlowModel):
        seen_sigma: torch.Tensor | None

        def __init__(self) -> None:
            super().__init__(dim=2, seed=0)
            self.seen_sigma = None

        def draw_nft_sigma_epsilon(
            self, x0_shape: torch.Size, cond: FlowCondition
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del cond
            sigma = torch.rand((x0_shape[0], 1, 1), dtype=torch.float64)  # [B,1,1]
            epsilon = torch.randn(x0_shape)  # [B,D]
            return sigma, epsilon

        def nft_velocity_predictions(
            self,
            *,
            x0: torch.Tensor,
            cond: FlowCondition,
            sigma: torch.Tensor,
            epsilon: torch.Tensor,
            condition_mask: FlowConditionMask,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            self.seen_sigma = sigma
            return super().nft_velocity_predictions(
                x0=x0,
                cond=cond,
                sigma=sigma.squeeze(-1).to(x0),  # [B,1]
                epsilon=epsilon,
                condition_mask=condition_mask,
            )

    model = _AdapterShapedSigmaToy()
    trajectories = [_trajectory(0, torch.zeros(1, 2), advantage=0.0)]  # action [1,D]

    nft_objective(model, trajectories, NFTObjectiveConfig(kl_beta=0.0, uniform_timestep=True))

    assert model.seen_sigma is not None
    assert model.seen_sigma.shape == (1, 1, 1)
    assert model.seen_sigma.device.type == "cpu"
    assert model.seen_sigma.dtype == torch.float64
