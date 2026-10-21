# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Orchestration-parity test for the decoupled Flow-DPO objective (DESIGN §8.2 #2).

Gate step 1 (CPU, model-free): a mock `FlowMatchingPolicy` returns fixed per-sample
errors for (policy/ref × winner/loser); we assert `algorithm.dpo.dpo_loss`
reproduces cosmos-rl `compute_dpo_loss`'s EXACT closed form, transcribed verbatim
from cosmos3.py L2617-2691:

    reward_winner = -0.5 * beta * (err_pol_w - err_ref_w)
    reward_loser  = -0.5 * beta * (err_pol_l - err_ref_l)
    loss          = -logsigmoid(reward_winner - reward_loser).mean()
    margin        = reward_winner - reward_loser
    dpo_accuracy  = ((margin>0) + 0.5*(margin==0)).mean()
    dpo_raw_model_loss = 0.5*(err_pol_w.mean()+err_pol_l.mean())
    dpo_raw_ref_loss   = 0.5*(err_ref_w.mean()+err_ref_l.mean())

This pins the 0.5 factor, sign, reduction, tie-aware accuracy and every metric, so a
regression in the lifted math fails loud — independent of any model. Adapter
fidelity against the real OmniMoT primitives is gate step 2 (GPU)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from cosmos_framework.posttrain.algorithm import dpo as dpo_module
from cosmos_framework.posttrain.algorithm.dpo import dpo_objective
from cosmos_framework.posttrain.model.protocol import FlowCondition, FlowMatchingPolicy
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory, Transition
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group

BETA = 500.0


@dataclass(frozen=True)
class _MockCondition:
    side: str
    context: object = "shared"


class _MockPreferenceContext:
    def assert_same_condition_context(self, winner: FlowCondition, loser: FlowCondition) -> None:
        if not isinstance(winner, _MockCondition) or not isinstance(loser, _MockCondition):
            raise TypeError(
                "mock preference sides must both use mock conditioning, got "
                f"{type(winner).__name__} and {type(loser).__name__}"
            )
        if winner.context != loser.context:
            raise ValueError(f"mock conditioning context mismatch: {winner.context!r} vs {loser.context!r}")


class _MockFlowModel(_MockPreferenceContext):
    """FlowMatchingPolicy that returns preset per-sample errors keyed by
    (is_policy_net, side). encode_and_condition just tags the side; denoise_error
    looks the error up — so the objective's call pattern fully determines the result."""

    def __init__(self, errs: dict[tuple[bool, str], torch.Tensor]):
        self.net = torch.nn.Identity()
        self.net_ref = torch.nn.Identity()  # distinct module identity from net
        self._errs = errs

    def encode_and_condition(self, batch):
        return torch.zeros(1), _MockCondition(  # latent [B]
            side=batch["side"], context=batch.get("context", "shared")
        )

    def condition_mask(self, cond):
        return None

    def draw_sigma_epsilon(self, x0_shape, cond):
        return torch.zeros(1), torch.zeros(1)

    def prepare_denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: _MockCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: torch.Tensor | None,
    ) -> tuple[bool, str]:
        del x0, sigma, epsilon, condition_mask
        return net is self.net, cond.side

    def denoise_prediction(
        self,
        *,
        net: torch.nn.Module,
        prepared: tuple[bool, str],
    ) -> tuple[bool, str]:
        del net
        return prepared

    def finalize_denoise_error(
        self,
        *,
        prepared: tuple[bool, str],
        prediction: tuple[bool, str],
    ) -> torch.Tensor:
        assert prepared == prediction
        return self._errs[prepared]


class _AsymmetricPreparationModel(_MockPreferenceContext):
    """Fail a structural guard on one rank before any model forward is allowed."""

    def __init__(self) -> None:
        self.net = torch.nn.Identity()
        self.net_ref = torch.nn.Identity()

    def encode_and_condition(
        self,
        batch: dict[str, str],
        *,
        sample_encode_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, FlowCondition]:
        del sample_encode_seeds
        rank = torch.distributed.get_rank()
        tokens = [[2]] if rank == 1 and batch["side"] == "loser" else [[1]]
        return torch.zeros(1, 1), _MockCondition(side=batch["side"], context=tuple(tokens[0]))  # [1, 1]

    def condition_mask(self, _cond: FlowCondition) -> None:
        return None

    def draw_sigma_epsilon(self, _x0_shape: torch.Size, _cond: FlowCondition) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 1), torch.zeros(1, 1)  # [1, 1], [1, 1]

    def prepare_denoise_error(self, **_kwargs: Any) -> Any:
        raise AssertionError("model forward must not run after a rank-local preparation failure")

    def denoise_prediction(self, **_kwargs: Any) -> Any:
        raise AssertionError("model forward must not run after a rank-local preparation failure")

    def finalize_denoise_error(self, **_kwargs: Any) -> torch.Tensor:
        raise AssertionError("model forward must not run after a rank-local preparation failure")


class _AsymmetricDenoiseStageModel(_MockPreferenceContext):
    """Inject one local failure between the objective's sequential model forwards."""

    def __init__(self, failure: str) -> None:
        self.net = torch.nn.Identity()
        self.net_ref = torch.nn.Identity()
        self.failure = failure
        self.prepare_calls = 0
        self.forward_calls = 0
        self.finalize_calls = 0

    def encode_and_condition(
        self,
        batch: dict[str, str],
        *,
        sample_encode_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, FlowCondition]:
        del sample_encode_seeds
        return torch.zeros(1, 1), _MockCondition(side=batch["side"])  # [B,D]

    def condition_mask(self, _cond: FlowCondition) -> None:
        return None

    def draw_sigma_epsilon(self, _x0_shape: torch.Size, _cond: FlowCondition) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 1), torch.zeros(1, 1)  # [B,1], [B,D]

    def prepare_denoise_error(self, **_kwargs: Any) -> torch.Tensor:
        self.prepare_calls += 1
        if self.failure == "second-preparation" and self.prepare_calls == 2 and torch.distributed.get_rank() == 1:
            raise ValueError("synthetic second-stage preparation failure")
        if self.failure == "first-preparation-none" and self.prepare_calls == 1 and torch.distributed.get_rank() == 1:
            return None  # type: ignore[return-value]  # malformed adapter return under test
        return torch.zeros(1)  # [B]

    def denoise_prediction(self, *, net: torch.nn.Module, prepared: torch.Tensor) -> torch.Tensor:
        del net
        self.forward_calls += 1
        if self.failure == "first-forward-none" and self.forward_calls == 1 and torch.distributed.get_rank() == 1:
            return None  # type: ignore[return-value]  # malformed adapter return under test
        return prepared.clone()  # [B]

    def finalize_denoise_error(
        self,
        *,
        prepared: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        del prepared
        self.finalize_calls += 1
        if self.failure == "first-finalization" and self.finalize_calls == 1 and torch.distributed.get_rank() == 1:
            raise ValueError("synthetic first-stage finalization failure")
        if self.failure == "first-finalization-none" and self.finalize_calls == 1 and torch.distributed.get_rank() == 1:
            return None  # type: ignore[return-value]  # malformed adapter return under test
        if self.failure == "objective-finalization-shape" and self.finalize_calls == 4:
            size = 2 if torch.distributed.get_rank() == 1 else 1
            return torch.zeros(size)  # [B] on rank 0, invalid [B + 1] on rank 1
        return prediction  # [B]


class _SeedRecordingFlowModel(_MockPreferenceContext):
    """Record the held-out seed streams at the model capability boundary."""

    def __init__(self) -> None:
        self.net: torch.nn.Module = torch.nn.Identity()
        self.net_ref: torch.nn.Module = torch.nn.Identity()
        self.encode_seeds: dict[str, list[int] | None] = {}
        self.noise_seeds: list[int] | None = None

    def encode_and_condition(
        self,
        batch: dict[str, Any],
        *,
        sample_encode_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, FlowCondition]:
        side = str(batch["side"])
        self.encode_seeds[side] = None if sample_encode_seeds is None else list(sample_encode_seeds)
        x0 = torch.zeros(len(batch["items"]), 1)  # [B,D]
        return x0, _MockCondition(side=side)

    def condition_mask(self, _cond: FlowCondition) -> None:
        return None

    def draw_sigma_epsilon(
        self,
        x0_shape: torch.Size,
        _cond: FlowCondition,
        sample_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.noise_seeds = None if sample_seeds is None else list(sample_seeds)
        sigma = torch.zeros(x0_shape[0], 1)  # [B,1]
        epsilon = torch.zeros(x0_shape)  # [B,D]
        return sigma, epsilon

    def prepare_denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del net, cond, sigma, epsilon, condition_mask
        return x0  # [B,D]

    def denoise_prediction(self, *, net: torch.nn.Module, prepared: torch.Tensor) -> torch.Tensor:
        del net
        return prepared  # [B,D]

    def finalize_denoise_error(
        self,
        *,
        prepared: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        del prepared
        return prediction.square().mean(dim=1)  # [B]


def _asymmetric_preparation_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        model = _AsymmetricPreparationModel()
        with pytest.raises(RuntimeError) as failed:
            dpo_objective(model, {"side": "winner"}, {"side": "loser"}, BETA)
        message = str(failed.value)
        assert "DPO objective preparation failed on one or more ranks" in message
        assert "rank 1: ValueError" in message
        assert "conditioning context mismatch" in message
    finally:
        dist.destroy_process_group()


def _asymmetric_denoise_stage_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    failure: str,
) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        model = _AsymmetricDenoiseStageModel(failure)
        with pytest.raises(RuntimeError) as failed:
            dpo_objective(model, {"side": "winner"}, {"side": "loser"}, BETA)
        message = str(failed.value)
        expected_error_type = "TypeError" if failure.endswith("-none") else "ValueError"
        assert f"rank 1: {expected_error_type}" in message
        if failure in ("second-preparation", "first-preparation-none"):
            expected_label = "reference-winner" if failure == "second-preparation" else "policy-winner"
            assert f"DPO {expected_label} forward preparation failed" in message
            expected_counts = (2, 1, 1) if failure == "second-preparation" else (1, 0, 0)
            assert (model.prepare_calls, model.forward_calls, model.finalize_calls) == expected_counts
        elif failure == "first-forward-none":
            assert "DPO policy-winner model forward failed" in message
            assert (model.prepare_calls, model.forward_calls, model.finalize_calls) == (1, 1, 0)
        else:
            assert "DPO policy-winner error finalization failed" in message
            assert (model.prepare_calls, model.forward_calls, model.finalize_calls) == (1, 1, 1)
    finally:
        dist.destroy_process_group()


def _asymmetric_objective_finalization_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        model = _AsymmetricDenoiseStageModel("objective-finalization-shape")
        with pytest.raises(RuntimeError) as failed:
            dpo_objective(
                model,
                {"side": "winner"},
                {"side": "loser"},
                BETA,
                sample_weights=[1.0],
            )
        message = str(failed.value)
        assert "DPO objective finalization failed on one or more ranks" in message
        assert "rank 1: ValueError" in message
        assert "sample_weights shape (1,) does not match reference shape (2,)" in message
        assert (model.prepare_calls, model.forward_calls, model.finalize_calls) == (4, 4, 4)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loss_matches_cosmos_rl_closed_form():
    # Distinct, non-tied per-sample errors so the formula is fully exercised.
    err_pol_w = torch.tensor([0.40, 0.60, 0.55])
    err_ref_w = torch.tensor([0.50, 0.50, 0.50])
    err_pol_l = torch.tensor([0.70, 0.65, 0.45])
    err_ref_l = torch.tensor([0.50, 0.50, 0.50])
    model = _MockFlowModel(
        {
            (True, "winner"): err_pol_w,
            (False, "winner"): err_ref_w,
            (True, "loser"): err_pol_l,
            (False, "loser"): err_ref_l,
        }
    )

    loss, metrics = dpo_objective(model, {"side": "winner"}, {"side": "loser"}, BETA)

    # cosmos-rl closed form (transcribed verbatim).
    reward_w = -0.5 * BETA * (err_pol_w - err_ref_w)
    reward_l = -0.5 * BETA * (err_pol_l - err_ref_l)
    exp_loss = -F.logsigmoid(reward_w - reward_l).mean()
    margin = reward_w - reward_l
    exp_acc = ((margin > 0).float() + 0.5 * (margin == 0).float()).mean()

    assert torch.allclose(loss, exp_loss)
    assert torch.allclose(metrics["dpo_reward_margin"], margin.mean())
    assert torch.allclose(metrics["dpo_reward_winner"], reward_w.mean())
    assert torch.allclose(metrics["dpo_reward_loser"], reward_l.mean())
    assert torch.allclose(metrics["dpo_accuracy"], exp_acc)
    assert torch.allclose(metrics["dpo_err_pol_w"], err_pol_w.mean())
    assert torch.allclose(metrics["dpo_err_ref_w"], err_ref_w.mean())
    assert torch.allclose(metrics["dpo_raw_model_loss"], 0.5 * (err_pol_w.mean() + err_pol_l.mean()))
    assert torch.allclose(metrics["dpo_raw_ref_loss"], 0.5 * (err_ref_w.mean() + err_ref_l.mean()))


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_zero_margin_gives_ln2_and_half_accuracy():
    """Bit-equal ref ⇒ zero margin ⇒ loss = -logσ(0) = ln 2, tie-aware acc = 0.5.
    This is the init invariant (matches the toy step-0 observation)."""
    e = torch.tensor([0.5, 0.5])
    model = _MockFlowModel({(True, "winner"): e, (False, "winner"): e, (True, "loser"): e, (False, "loser"): e})
    loss, metrics = dpo_objective(model, {"side": "winner"}, {"side": "loser"}, BETA)
    assert torch.allclose(loss, torch.log(torch.tensor(2.0)))
    assert torch.allclose(metrics["dpo_accuracy"], torch.tensor(0.5))
    assert torch.allclose(metrics["dpo_reward_margin"], torch.tensor(0.0))


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_domain_separates_encode_seeds_but_reuses_noise_roots() -> None:
    """Tokenizer posterior streams differ by side; DPO noise stays pair-shared."""
    roots = [7, 2**64 - 1]
    model = _SeedRecordingFlowModel()

    dpo_objective(
        model,
        {"side": "winner", "items": [0, 1]},
        {"side": "loser", "items": [0, 1]},
        BETA,
        sample_seeds=roots,
    )

    assert model.noise_seeds == roots
    # Pin the pre-refactor domain+root byte layout, not a second implementation of it.
    assert model.encode_seeds["winner"] == [5274101476837730744, 13990329813063266220]
    assert model.encode_seeds["loser"] == [3854887325886136053, 5525595673280520733]
    assert model.encode_seeds["winner"] != model.encode_seeds["loser"]


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_nonfinite_propagates_without_sanitizing() -> None:
    """A non-finite error reaches the trainer's rank-consistent validity guard."""
    nan = torch.tensor([float("nan")])
    ok = torch.tensor([0.5])
    model = _MockFlowModel({(True, "winner"): nan, (False, "winner"): ok, (True, "loser"): ok, (False, "loser"): ok})
    loss, metrics = dpo_objective(model, {"side": "winner"}, {"side": "loser"}, BETA)

    assert not torch.isfinite(loss)
    assert not torch.isfinite(metrics["dpo_loss"])
    assert not torch.isfinite(metrics["dpo_reward_margin"])


@pytest.mark.L0
@pytest.mark.CPU
def test_condition_context_validation_delegates_and_fails_closed() -> None:
    """DPO delegates opaque context and foreign-type validation to the model adapter."""

    ok = torch.tensor([0.5])  # [B]
    model = _MockFlowModel({(True, "winner"): ok, (False, "winner"): ok, (True, "loser"): ok, (False, "loser"): ok})
    dpo_objective(
        model,
        {"side": "winner", "context": "same"},
        {"side": "loser", "context": "same"},
        BETA,
    )
    with pytest.raises(ValueError, match="context mismatch"):
        dpo_objective(
            model,
            {"side": "winner", "context": "prompt-a"},
            {"side": "loser", "context": "prompt-b"},
            BETA,
        )

    model.encode_and_condition = lambda _batch: (  # type: ignore[method-assign]
        torch.zeros(1),  # latent [B]
        object(),
    )
    with pytest.raises(TypeError, match="mock preference sides"):
        dpo_objective(model, {"side": "winner"}, {"side": "loser"}, BETA)


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_requires_preference_flow_capability_before_forward() -> None:
    class _FlowModelWithoutPreferenceCapability:
        def __init__(self) -> None:
            self.net = torch.nn.Identity()
            self.net_ref = torch.nn.Identity()
            self.adapter_calls = 0

        def encode_and_condition(self, batch: dict[str, str]) -> tuple[torch.Tensor, FlowCondition]:
            self.adapter_calls += 1
            return torch.zeros(1), _MockCondition(side=batch["side"])  # latent [B]

        def condition_mask(self, _condition: FlowCondition) -> None:
            self.adapter_calls += 1
            raise AssertionError("DPO must reject a missing preference capability before model preparation")

    model = _FlowModelWithoutPreferenceCapability()
    with pytest.raises(TypeError, match="PreferenceFlowPolicy"):
        dpo_objective(
            cast(FlowMatchingPolicy, model),
            {"side": "winner"},
            {"side": "loser"},
            BETA,
        )
    assert model.adapter_calls == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_preference_rejects_malformed_pair():
    """The typed supervised signal is validated at construction (production time): a pair needs
    both a winner and a loser DICT, so a missing / non-dict side raises a clear ValueError at the
    producer (`_pair_trajectory`), not deep in the objective's four forwards."""
    from cosmos_framework.posttrain.schema import Preference

    Preference(winner={"x": 1}, loser={"x": 2})  # well-formed → no raise
    for bad in ({"winner": {"x": 1}, "loser": None}, {"winner": [1, 2], "loser": {"x": 2}}):
        with pytest.raises(ValueError, match="Preference"):
            Preference(**bad)


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_asymmetric_structural_error_converges_before_model_forward(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "dpo-preparation-rendezvous")
    mp.spawn(_asymmetric_preparation_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "failure",
    [
        "second-preparation",
        "first-preparation-none",
        "first-forward-none",
        "first-finalization",
        "first-finalization-none",
    ],
)
def test_rank_asymmetric_denoise_stage_error_converges_before_next_forward(
    tmp_path: Path,
    failure: str,
) -> None:
    """A local inter-forward failure reaches every rank before any peer advances."""
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / f"dpo-{failure}-rendezvous")
    mp.spawn(
        _asymmetric_denoise_stage_worker,
        args=(world_size, rendezvous_path, failure),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_asymmetric_objective_shape_error_converges_after_all_forwards(tmp_path: Path) -> None:
    """A rank-local closed-form shape failure reaches every peer after all four forwards."""
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "dpo-objective-finalization-rendezvous")
    mp.spawn(
        _asymmetric_objective_finalization_worker,
        args=(world_size, rendezvous_path),
        nprocs=world_size,
        join=True,
    )


def _preference_trajectory(
    *,
    targets: object,
    algo_extra: dict[str, Any] | None = None,
) -> Trajectory:
    return Trajectory(
        transitions=[
            Transition(
                obs=ObsBundle(),
                targets=targets,
                algo_extra={} if algo_extra is None else algo_extra,
            )
        ]
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_unpack_preference_rejects_untyped_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectories = [_preference_trajectory(targets={"winner": {}, "loser": {}})]
    converged: list[tuple[Exception, str]] = []

    def _converge(action: Any, context: str) -> Any:
        try:
            return action()
        except Exception as error:
            converged.append((error, context))
            raise RuntimeError("synthetic rank-converged preparation failure") from error

    monkeypatch.setattr(dpo_module, "run_rank_converged", _converge)
    with pytest.raises(RuntimeError, match="synthetic rank-converged preparation failure"):
        dpo_module.make_objective(BETA).fn(object(), trajectories)

    assert len(converged) == 1
    error, context = converged[0]
    assert context == "DPO objective batch preparation"
    assert isinstance(error, TypeError)
    assert "Transition.targets must be a Preference" in str(error)
