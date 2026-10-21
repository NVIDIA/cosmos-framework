# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen-reference K3 KL anchor for the clipped GRPO objective (`rl.kl_beta > 0`).

Pins, on tiny CPU tensors and a transformers-free two-layer LM:

* the per-token math in `clipped_policy_gradient_loss` — zero penalty at `reference == current`,
  positive and equal to PPO's `sampled_k3_kl` otherwise, gradient through `current` only and
  pointing toward the reference, exact reduction to the unanchored loss at `kl_beta == 0`, and the
  same +/-20 nat log-ratio clamp PPO applies;
* the objective end-to-end — `FrozenReference` scores the same tokens as the policy (KL exactly 0 at
  identical weights, loss identical to the `kl_beta=0` objective), a drifted policy reports a
  positive `grpo_kl_mean` whose gradient pulls it back, the empty-shard path keeps the metric key
  set and still issues the reference forward (FSDP collective parity), a positive `kl_beta`
  with nothing attached is refused before any forward, the reference forward is skipped (rank-
  uniformly) while the π_old cache records or probes and runs once per training chunk, and a
  routing replay carried by the batch pins the reference's adapter exactly as it pins the policy's;
* the config guard — `kl_beta > 0` needs `ref.build_ref`, `build_ref` needs `kl_beta > 0`,
  `update_every` must be 0, `legacy_pg` refuses the reference, and the canary dotted overrides
  parse through the structured schema;
* the worker seams — `LLMTrainer.offload` parks the twin with the policy through the trainer-owned
  `AuxiliaryModelResidency` the entry registers it with (the trainer never reads the objective's
  scorer slot),
  `entrypoints/grpo.py::build_trainer` builds the twin from the policy builder exactly when the
  config asks for it, and refuses to build it on a policy with live dropout (the reference scores
  under eval() while the policy trains under train(), at any n_mini);
* the warm-restart knob — `rl.ref.model_dir` needs `build_ref` and parses through the structured
  schema at config load, must name an existing directory holding weights when `build_trainer` runs
  (never probed by the dataclass, so a saved config reloads anywhere), binds that directory's
  CONTENT identity into the objective's exact-resume identity (`reference_artifact`: another
  directory, other content at the same path, or dropping the knob refuses the resume), and makes
  `build_trainer` load the twin
  from THAT directory (qwen3_vl alias, generic i4 spec, hf) while the policy build and the
  caller's config keep the policy's own directory;
* the gradient bound — `rl.kl_coeff_clamp` keeps the k3 VALUE bit-identical while holding the
  per-token gradient coefficient at its `|z| = c` boundary value beyond the band (the exact
  `1 − exp(z)` is unbounded inside the ±20 clamp and let one token own a step), `None` reproduces
  the old loss and gradient exactly, the ±20 saturation stays, the config validates the range and
  warns when a positive `kl_beta` runs without the bound, `train/kl_max` is reported and declared
  MAX-reduced, the console line carries `grad_norm` and the KL columns, and the objective identity
  records the bound only when set.
"""

from __future__ import annotations

import copy
import math
import sys
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from cosmos_framework.posttrain.algorithm.grpo import (  # noqa: E402
    _GRPO_PUBLIC_METRICS,
    FrozenReference,
    make_clipped_objective,
)
from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import (  # noqa: E402
    TOKEN_LOG_RATIO_CLAMP,
    clipped_policy_gradient_loss,
    sampled_k3_penalty,
)
from cosmos_framework.posttrain.config.rl import GRPOConfig, GRPORefConfig, RefConfig  # noqa: E402
from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput  # noqa: E402
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition  # noqa: E402

# --------------------------------------------------------------------------- fixtures (no transformers)


class _TinyLM(nn.Module):
    """Embedding + linear head; `forward(ids) -> SimpleNamespace(logits=[B,L,V])`, counting calls."""

    def __init__(self, vocab: int = 16, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.calls: list[tuple[int, ...]] = []

    def forward(self, ids: torch.Tensor) -> Any:  # ids [B,L]
        self.calls.append(tuple(ids.shape))
        return SimpleNamespace(logits=self.head(torch.tanh(self.emb(ids))))  # [B,L,V]


class _IdsPacker:
    """Token-id packer following `DecoderOnlyLLMDataPacker`'s policy contract, without a tokenizer."""

    def get_policy_input(
        self,
        obs: Any,
        completion: Any,
        *,
        n_ignore_prefix_tokens: int = 0,
        prompt_ids: list[int] | None = None,
    ) -> RLPolicyInput:
        assert prompt_ids, "tests store prompt_ids on every transition"
        completion_ids = [int(t) for t in completion]
        return RLPolicyInput(
            input_ids=list(prompt_ids) + completion_ids,
            logprob_masks=[0] * (len(prompt_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0],
        )

    def policy_compute_max_len(self, processed: list[RLPolicyInput]) -> int:
        return max(len(x.input_ids) for x in processed)

    def policy_collate_fn(self, processed: list[RLPolicyInput], max_len: int) -> dict[str, Any]:
        ids = torch.tensor([x.input_ids + [0] * (max_len - len(x.input_ids)) for x in processed], dtype=torch.long)
        masks = torch.tensor(
            [x.logprob_masks + [0] * (max_len - len(x.logprob_masks)) for x in processed], dtype=torch.bool
        )
        return {"input_ids": ids, "logprob_masks": masks}  # [B,L], [B,L]


def _batch(rewards: list[float], completions: list[list[int]] | None = None) -> list[Trajectory]:
    """One shared group: distinct-length completions, rewards -> group-relative advantages."""
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    completions = completions or [[5, 6, 7], [8, 9], [10, 11, 12, 13]][: len(rewards)]
    trajs = []
    for k, (reward, completion) in enumerate(zip(rewards, completions)):
        tr = Transition(
            obs=ObsBundle(),
            action=torch.tensor(completion, dtype=torch.long),
            policy_info={"prompt_ids": torch.tensor([1, 2, 3 + k])},
            reward={"final": reward},
            done=True,
        )
        trajs.append(Trajectory(transitions=[tr], id=f"g_k{k}", group_id="g", status=Status.COMPLETED))
    group_relative()(trajs)
    return trajs


def _cfg(kl_beta: float) -> GRPOConfig:
    ref = RefConfig(build_ref=kl_beta > 0, update_every=0)
    return GRPOConfig(kl_beta=kl_beta, ref=ref, group_size=2, temperature=1.0)


def _perturb(model: nn.Module, scale: float = 0.5, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


# --------------------------------------------------------------------------- per-token primitive


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_penalty_is_exactly_zero_when_reference_equals_current() -> None:
    torch.manual_seed(0)
    current = torch.randn(7, dtype=torch.float32)
    behavior = current.detach()
    advantage = torch.randn(7)
    base = clipped_policy_gradient_loss(current, behavior, advantage, 0.2, 0.2, kl_beta=0.0)
    anchored = clipped_policy_gradient_loss(
        current, behavior, advantage, 0.2, 0.2, reference=current.detach().clone(), kl_beta=0.5
    )
    # expm1(0) - 0 == 0 exactly, so the two losses are bit-identical, not merely close.
    assert torch.equal(anchored, base)


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_penalty_is_positive_and_mirrors_ppo_sampled_k3_kl() -> None:
    """Loss-side KL == kl_beta * PPO's `sampled_k3_kl(reference, current)` per token, and > 0."""
    from cosmos_framework.posttrain.algorithm.ppo.losses import sampled_k3_kl

    torch.manual_seed(1)
    current = torch.randn(9, dtype=torch.float32) - 2.0
    reference = current + torch.randn(9) * 0.7
    advantage = torch.randn(9)
    kl_beta = 0.3
    stats: dict = {}
    base = clipped_policy_gradient_loss(current, current.detach(), advantage, 0.2, 0.2, kl_beta=0.0)
    anchored = clipped_policy_gradient_loss(
        current, current.detach(), advantage, 0.2, 0.2, reference=reference, kl_beta=kl_beta, stats=stats
    )
    penalty = anchored - base  # [N]
    assert bool((penalty > 0).all()), "k3 is strictly positive wherever reference != current"
    ppo = sampled_k3_kl(reference, current)  # PPO's reward-shaping form: same z, same clamp
    assert torch.allclose(penalty, kl_beta * ppo, atol=1e-6, rtol=1e-5)
    assert math.isclose(stats["kl_sum"], float(ppo.sum()), rel_tol=1e-5)


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_gradient_flows_through_current_only_and_points_toward_reference() -> None:
    """d k3 / d cur = 1 - exp(ref - cur): negative where cur < ref, positive where cur > ref, so a
    descent step moves `current` toward `reference`; the reference receives no gradient even when it
    was produced with grad enabled (the primitive detaches it)."""
    current = torch.tensor([-3.0, -1.0, -2.0, -0.5], requires_grad=True)
    reference = torch.tensor([-1.0, -3.0, -2.0, -0.5], requires_grad=True)  # cur<ref, cur>ref, equal, equal
    advantage = torch.zeros(4)  # isolate the KL term: the surrogate is identically 0
    loss = clipped_policy_gradient_loss(
        current, current.detach(), advantage, 0.2, 0.2, reference=reference, kl_beta=1.0
    )
    loss.sum().backward()
    assert reference.grad is None, "gradient must not leak into the reference"
    grad = current.grad
    assert grad[0] < 0 and grad[1] > 0, grad  # descent raises the under-weighted token, lowers the over-weighted
    assert torch.allclose(grad[2:], torch.zeros(2)), "no pull where the two already agree"
    expected = 1.0 - torch.exp(reference.detach() - current.detach())
    assert torch.allclose(grad, expected, atol=1e-6)
    with torch.no_grad():
        stepped = current - 0.1 * grad
    assert bool(((stepped - reference.detach()).abs() <= (current.detach() - reference.detach()).abs()).all())


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_beta_zero_reduces_to_the_unanchored_loss_even_with_a_reference() -> None:
    torch.manual_seed(2)
    current = torch.randn(6)
    reference = torch.randn(6)
    advantage = torch.randn(6)
    stats: dict = {}
    with_ref = clipped_policy_gradient_loss(
        current, current.detach(), advantage, 0.2, 0.2, reference=reference, kl_beta=0.0, stats=stats
    )
    without = clipped_policy_gradient_loss(current, current.detach(), advantage, 0.2, 0.2, kl_beta=0.0)
    assert torch.equal(with_ref, without)
    assert "kl_sum" not in stats
    with pytest.raises(ValueError, match="requires reference"):
        clipped_policy_gradient_loss(current, current.detach(), advantage, 0.2, 0.2, kl_beta=0.1)


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_log_ratio_is_clamped_like_ppo_so_a_collapsed_token_stays_finite() -> None:
    """z = ref - cur = +120 would overflow fp32 `exp`; the clamp caps the penalty at k3(20) with a
    zero gradient beyond it, exactly PPO's `sampled_k3_kl` bound."""
    current = torch.tensor([-130.0, -1.0], requires_grad=True)
    reference = torch.tensor([-10.0, -1.0])
    loss = clipped_policy_gradient_loss(
        current, current.detach(), torch.zeros(2), 0.2, 0.2, reference=reference, kl_beta=1.0
    )
    assert torch.isfinite(loss).all()
    cap = float(sampled_k3_penalty(torch.tensor([TOKEN_LOG_RATIO_CLAMP]), torch.tensor([0.0])))
    assert math.isclose(float(loss[0].detach()), cap, rel_tol=1e-6)
    assert math.isclose(cap, math.expm1(20.0) - 20.0, rel_tol=1e-6)  # fp32 tensor vs float64 math: close, not ==
    loss.sum().backward()
    assert float(current.grad[0]) == 0.0  # saturated: the clamp's gradient is zero outside the band


# --------------------------------------------------------------------------- objective end-to-end


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_kl_is_zero_at_identical_weights_and_loss_matches_the_unanchored_objective() -> None:
    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    packer = _IdsPacker()
    anchored = make_clipped_objective(packer, _cfg(0.1))
    assert isinstance(anchored.reference_scorer, FrozenReference) and not anchored.reference_scorer.attached
    anchored.reference_scorer.attach(reference)
    assert anchored.reference_scorer.attached and anchored.reference_scorer.net is reference
    assert not reference.training and all(not p.requires_grad for p in reference.parameters())

    loss_anchored, metrics = anchored(policy, _batch([1.0, 0.0, 1.0]))
    loss_plain, plain_metrics = make_clipped_objective(packer, _cfg(0.0))(policy, _batch([1.0, 0.0, 1.0]))
    assert torch.allclose(loss_anchored, loss_plain, atol=1e-7)
    assert metrics["grpo_kl_mean"] == 0.0 and metrics["grpo_kl_beta"] == 0.1
    assert plain_metrics["grpo_kl_mean"] == 0.0 and plain_metrics["grpo_kl_beta"] == 0.0
    # Both objectives report the complete public metric contract (metrics are packed into one vector
    # and all-reduced, so the key set is load-bearing).
    assert set(metrics) == set(_GRPO_PUBLIC_METRICS) == set(plain_metrics)
    assert reference.calls == [policy.calls[0]], "one reference forward per chunk, on the same [B,L]"


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_reports_positive_kl_after_drift_and_one_step_pulls_the_policy_back() -> None:
    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    _perturb(policy)
    kl_beta = 0.25
    objective = make_clipped_objective(_IdsPacker(), _cfg(kl_beta))
    objective.reference_scorer.attach(reference)

    # Equal rewards -> every advantage is 0 -> the loss is the KL term alone: token-mean of kl_beta*k3.
    loss, metrics = objective(policy, _batch([1.0, 1.0, 1.0]))
    assert metrics["grpo_kl_mean"] > 1e-3, metrics
    assert math.isclose(float(loss), kl_beta * metrics["grpo_kl_mean"], rel_tol=1e-5)
    loss.backward()
    assert all(p.grad is None for p in reference.parameters()), "the frozen twin must never accumulate grads"
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in policy.parameters())

    with torch.no_grad():
        for p in policy.parameters():
            if p.grad is not None:
                p.sub_(0.05 * p.grad)
    policy.zero_grad()
    _, after = objective(policy, _batch([1.0, 1.0, 1.0]))
    assert after["grpo_kl_mean"] < metrics["grpo_kl_mean"], (after["grpo_kl_mean"], metrics["grpo_kl_mean"])


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_shard_path_keeps_the_metric_keys_and_still_issues_the_reference_forward() -> None:
    """An empty completion leaves `processed` empty: the objective runs its probe forward for FSDP
    parity and must run the reference forward too, with the same metric key set as a full shard."""
    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.1))
    objective.reference_scorer.attach(reference)
    tr = Transition(
        obs=ObsBundle(),
        action=torch.tensor([], dtype=torch.long),
        policy_info={"prompt_ids": torch.tensor([1, 2, 3])},
        reward={"final": 1.0},
        done=True,
    )
    tr.advantage = 0.0
    loss, metrics = objective(policy, [Trajectory(transitions=[tr], id="g_k0", group_id="g", status=Status.COMPLETED)])
    assert float(loss) == 0.0 and metrics["grpo_n"] == 0
    assert set(metrics) == set(_GRPO_PUBLIC_METRICS)
    assert metrics["grpo_kl_mean"] == 0.0 and metrics["grpo_kl_beta"] == 0.1
    assert len(policy.calls) == 1 and reference.calls == policy.calls


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_refuses_positive_kl_beta_with_no_reference_attached_before_any_forward() -> None:
    policy = _TinyLM()
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.1))
    with pytest.raises(ValueError, match="requires a frozen reference policy"):
        objective(policy, _batch([1.0, 0.0]))
    assert policy.calls == [], "refused before the policy forward, so no rank can strand its peers"
    # kl_beta == 0 carries no slot at all: nothing to attach, nothing to offload.
    assert make_clipped_objective(_IdsPacker(), _cfg(0.0)).reference_scorer is None


@pytest.mark.L0
@pytest.mark.CPU
def test_frozen_reference_attach_freezes_once_and_refuses_a_second_anchor() -> None:
    holder = FrozenReference()
    assert holder.net is None and not holder.attached
    net = _TinyLM()
    net.train()
    assert holder.attach(net) is holder
    assert not net.training and all(not p.requires_grad for p in net.parameters())
    with pytest.raises(RuntimeError, match="already attached"):
        holder.attach(_TinyLM())
    with pytest.raises(TypeError, match="torch.nn.Module"):
        FrozenReference().attach(object())
    # An adapter exposing `.net` is unwrapped for freezing/offload but kept whole for scoring.
    adapter = SimpleNamespace(net=_TinyLM())
    wrapped = FrozenReference().attach(adapter)
    assert wrapped.model is adapter and wrapped.net is adapter.net


@pytest.mark.L0
@pytest.mark.CPU
def test_reference_forward_is_skipped_while_pi_old_is_recorded_or_probed() -> None:
    """n_mini > 1: the ahead pass and the one-shot determinism probe call the objective under no_grad
    and discard the loss, and the reference is frozen, so a reference forward there is pure cost
    (n_mini*n_micro + 1 per collected batch). The cache mode — driver-set, identical on every rank —
    gates it: no reference forward while recording or probing, exactly one per training chunk while
    replaying. The recorded π_old is the policy's own logprob, untouched by the KL term (ratio exactly
    1 against the unchanged policy: no clip, no ratio deviation)."""
    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    _perturb(policy)  # a live KL term once the anchor is scored
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.1))
    objective.reference_scorer.attach(reference)
    cache = objective.old_logp_cache
    chunks = [_batch([1.0, 0.0]), _batch([0.0, 1.0], completions=[[9, 8, 7, 6], [5, 4]])]  # n_mini*n_micro = 2

    cache.begin_record()
    with torch.no_grad():
        for chunk in chunks:
            _, recorded = objective(policy, chunk)
            assert recorded["grpo_kl_mean"] == 0.0 and recorded["grpo_kl_max"] == 0.0
            assert recorded["grpo_kl_beta"] == 0.1  # the config's anchor strength is still echoed
    cache.begin_probe()
    with torch.no_grad():
        objective(policy, chunks[0])
    cache.mode = "record"
    assert len(policy.calls) == 3 and reference.calls == [], "no reference forward while recording or probing"
    assert cache.recorded == 2

    cache.begin_replay(expected=2)
    for chunk in chunks:
        _, trained = objective(policy, chunk)
        assert trained["grpo_kl_mean"] > 1e-3 and trained["grpo_kl_max"] >= trained["grpo_kl_mean"]
        assert (
            trained["grpo_clip_frac"] == 0.0 and trained["grpo_ratio_absdev"] == 0.0
        )  # π_old is the policy's own logprob
    assert len(reference.calls) == 2 and reference.calls == policy.calls[3:], "one reference forward per training chunk"

    # Outside an ahead-pass window (n_mini == 1, mode "off") every call scores the reference, as before.
    fresh_reference = copy.deepcopy(reference)
    fresh_reference.calls.clear()  # the deepcopy carries the two recorded calls
    single = make_clipped_objective(_IdsPacker(), _cfg(0.1))
    single.reference_scorer.attach(fresh_reference)
    single(policy, chunks[0])
    assert len(fresh_reference.calls) == 1


class _DropoutLM(_TinyLM):
    """`_TinyLM` with a live `nn.Dropout`: deterministic under eval(), stochastic under train()."""

    def __init__(self, vocab: int = 16, d: int = 8, seed: int = 0) -> None:
        super().__init__(vocab=vocab, d=d, seed=seed)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, ids: torch.Tensor) -> Any:  # ids [B,L]
        self.calls.append(tuple(ids.shape))
        return SimpleNamespace(logits=self.head(self.drop(torch.tanh(self.emb(ids)))))  # [B,L,V]


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_refuses_a_frozen_reference_on_a_policy_with_live_dropout(monkeypatch, tmp_path) -> None:
    """The twin scores under eval() while the policy trains under train(), so with a live dropout the
    log-ratio is a dropout mask — non-zero at step 0 even at identical weights — at ANY n_mini. The
    screen the π_old ahead pass runs at n_mini > 1 is applied by build_trainer before the twin is
    built; without an anchor the same policy builds (the ahead pass keeps its own screen)."""
    from cosmos_framework.posttrain.entrypoints import grpo
    from cosmos_framework.posttrain.roles.trainer.llm import (
        KL_REFERENCE_DETERMINISM_REQUIREMENT,
        raise_if_forward_is_stochastic,
    )

    with pytest.raises(ValueError, match="frozen KL reference.*DETERMINISTIC forward.*drop \\(Dropout, p=0.1\\)"):
        raise_if_forward_is_stochastic(_DropoutLM(), requirement=KL_REFERENCE_DETERMINISM_REQUIREMENT)
    raise_if_forward_is_stochastic(_TinyLM(), requirement=KL_REFERENCE_DETERMINISM_REQUIREMENT)  # dropout-free passes

    built: list[Any] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built, net_factory=_DropoutLM)
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.05))
    with pytest.raises(ValueError, match="frozen KL reference.*DETERMINISTIC forward"):
        grpo.build_trainer(
            config,
            GRPOConfig(resume="off", kl_beta=0.05, ref=RefConfig(build_ref=True)),
            build_objective_fn=lambda *_: objective,
        )
    assert [entry[0] for entry in built] == ["model"], "refused after the policy and before the twin's memory is spent"
    assert not objective.reference_scorer.attached

    built.clear()
    plain = make_clipped_objective(_IdsPacker(), _cfg(0.0))
    grpo.build_trainer(config, GRPOConfig(resume="off"), build_objective_fn=lambda *_: plain)
    assert [entry[0] for entry in built] == ["model", "trainer"], "no anchor, no KL screen"


class _ReplayPacker(_IdsPacker):
    """`_IdsPacker` whose collated batch carries a routing-replay record, as the Qwen3-VL packers do."""

    def policy_collate_fn(self, processed: list[RLPolicyInput], max_len: int) -> dict[str, Any]:
        from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY

        batch = super().policy_collate_fn(processed, max_len)
        batch[FORWARD_REPLAY_BATCH_KEY] = torch.zeros((len(processed), max_len, 1, 2), dtype=torch.uint8)
        return batch


class _ReplayAdapter:
    """A `ForwardReplayPolicy`-shaped adapter over `_TinyLM`: pins from the batch it prepares (popping the
    record from a COPY, as the MoE adapter does) and records every pin verdict."""

    def __init__(self, net: nn.Module) -> None:
        self.net = net
        self.pins: list[bool] = []

    def prepare_policy_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY

        prepared = dict(batch)
        self.pins.append(prepared.pop(FORWARD_REPLAY_BATCH_KEY, None) is not None)
        return prepared

    def forward_policy_batch(self, batch: dict[str, Any]) -> Any:
        return self.net(batch["input_ids"])

    def replay_pinned(self) -> bool:
        return bool(self.pins and self.pins[-1])


@pytest.mark.L0
@pytest.mark.CPU
def test_frozen_reference_pins_to_the_batch_routing_replay_like_the_policy() -> None:
    """Under `rl.routing_replay=rollout` the reference prepares the SAME batch through its own adapter,
    so the recorded rollout routing pins its forward exactly as it pins the policy's: the KL anchor is
    π_ref under the rollout's expert dispatch, and z measures weight drift, not routing churn (this is
    the documented choice; an unpinned reference would put the routing flip back into z). The policy's
    own preparation does not consume the record before the reference sees it."""
    policy = _ReplayAdapter(_TinyLM())
    reference = _ReplayAdapter(copy.deepcopy(policy.net))
    objective = make_clipped_objective(_ReplayPacker(), _cfg(0.1))
    objective.reference_scorer.attach(reference)
    _, metrics = objective(policy, _batch([1.0, 0.0]))
    assert reference.pins == [True] and policy.pins == [True], "both adapters pinned from the one batch"
    assert reference.replay_pinned() and policy.replay_pinned()
    assert metrics["grpo_kl_mean"] == 0.0, "identical weights under identical dispatch: exactly zero KL"
    # A batch without the record runs both natural.
    natural_policy, natural_reference = _ReplayAdapter(_TinyLM()), _ReplayAdapter(_TinyLM())
    natural = make_clipped_objective(_IdsPacker(), _cfg(0.1))
    natural.reference_scorer.attach(natural_reference)
    natural(natural_policy, _batch([1.0, 0.0]))
    assert natural_reference.pins == [False] and natural_policy.pins == [False]


# --------------------------------------------------------------------------- config guard


@pytest.mark.L0
@pytest.mark.CPU
def test_config_ties_kl_beta_and_the_frozen_reference_together() -> None:
    from omegaconf import OmegaConf

    ok = GRPOConfig(kl_beta=0.05, ref=RefConfig(build_ref=True))
    assert ok.kl_beta == 0.05 and ok.ref.build_ref and ok.ref.update_every == 0
    assert GRPOConfig().ref == GRPORefConfig(build_ref=False, update_every=0)  # default: unanchored, as before
    with pytest.raises(ValueError, match="needs a frozen reference policy"):
        GRPOConfig(kl_beta=0.05)
    with pytest.raises(ValueError, match="requires kl_beta>0"):
        GRPOConfig(kl_beta=0.0, ref=RefConfig(build_ref=True))
    with pytest.raises(ValueError, match="update_every must be 0"):
        GRPOConfig(kl_beta=0.05, ref=RefConfig(build_ref=True, update_every=10))
    with pytest.raises(ValueError, match="does not consume clipped-surrogate settings.*ref.build_ref"):
        GRPOConfig(grpo_surrogate="legacy_pg", ref=RefConfig(build_ref=True))
    with pytest.raises(ValueError, match="kl_beta must be >= 0"):
        GRPOConfig(kl_beta=-0.1, ref=RefConfig(build_ref=True))
    # The canary's dotted overrides parse through the structured schema into a valid config.
    schema = OmegaConf.structured(GRPOConfig)
    merged = OmegaConf.merge(schema, OmegaConf.from_dotlist(["kl_beta=0.05", "ref.build_ref=true"]))
    resolved = OmegaConf.to_object(merged)
    assert isinstance(resolved, GRPOConfig) and resolved.kl_beta == 0.05 and resolved.ref.build_ref is True


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_still_pins_kl_beta_to_zero_and_the_reference_off() -> None:
    from cosmos_framework.posttrain.config.rl import OPDConfig

    assert OPDConfig().kl_beta == 0.0 and OPDConfig().ref.build_ref is False
    with pytest.raises(ValueError, match="kl_beta"):
        OPDConfig(kl_beta=0.1, ref=RefConfig(build_ref=True))


# --------------------------------------------------------------------------- worker seams


@pytest.mark.L0
@pytest.mark.CPU
def test_trainer_offload_parks_the_frozen_reference_as_update_only_state() -> None:
    """The registered twin parks before policy while an objective-only scorer remains undiscovered."""
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.1))
    objective.reference_scorer.attach(reference)
    trainer = LLMTrainer(
        policy,
        objective,
        torch.optim.SGD(policy.parameters(), lr=0.1),
        auxiliary_models=[SimpleNamespace(net=reference)],
    )
    seen: dict[str, tuple[list[Any], Any]] = {}

    class _Recorder:
        def __init__(self, label: str) -> None:
            self.label = label
            self.offloaded = False

        def offload(self, modules: Any, *, optimizer: Any = None, grads_of: Any = ()) -> float:
            del grads_of
            seen[self.label] = (list(modules), optimizer)
            self.offloaded = True
            return 0.0

    trainer._offload = _Recorder("policy")
    trainer._update_offload = _Recorder("update")
    trainer.offload()
    assert seen["update"] == ([reference], trainer.opt)
    assert seen["policy"] == ([policy], None)
    assert trainer.residency.modules() == [reference] and len(trainer.residency) == 1

    # Same objective, same attached scorer, nothing registered: only the policy moves. The trainer
    # does not reach into `objective.reference_scorer`.
    unregistered = LLMTrainer(policy, objective, torch.optim.SGD(policy.parameters(), lr=0.1))
    unregistered._offload = _Recorder("policy")
    unregistered._update_offload = _Recorder("update")
    unregistered.offload()
    assert seen["update"] == ([], unregistered.opt)
    assert seen["policy"] == ([policy], None) and len(unregistered.residency) == 0

    plain = LLMTrainer(
        policy, make_clipped_objective(_IdsPacker(), _cfg(0.0)), torch.optim.SGD(policy.parameters(), lr=0.1)
    )
    plain._offload = _Recorder("policy")
    plain._update_offload = _Recorder("update")
    plain.offload()
    assert seen["update"] == ([], plain.opt)
    assert seen["policy"] == ([policy], None)


@pytest.mark.L0
@pytest.mark.CPU
def test_auxiliary_model_residency_unwraps_adapters_and_refuses_duplicates_and_non_modules() -> None:
    from cosmos_framework.posttrain.roles.trainer import AuxiliaryModelResidency

    net = _TinyLM()
    residency = AuxiliaryModelResidency([SimpleNamespace(net=net)])
    assert residency.modules() == [net] and residency.labels() == ["auxiliary model"]
    raw = _TinyLM(seed=1)
    assert residency.register(raw, label="frozen reference") is raw
    assert residency.modules() == [net, raw] and residency.labels()[1] == "frozen reference"
    with pytest.raises(ValueError, match="already registered"):
        residency.register(SimpleNamespace(net=net))
    with pytest.raises(TypeError, match="must wrap a torch.nn.Module"):
        residency.register(SimpleNamespace(net=object()), label="bogus")
    assert len(residency) == 2, "a refused registration leaves the registry unchanged"


def _mock_build_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, built: list[Any], *, net_factory: Any = _TinyLM
) -> Any:
    """The lightweight fakes `tests/test_entrypoints_grpo.py` uses for build_trainer, plus a counting
    `build_hf_model` whose products carry a real module (``net_factory(seed=...)``) so
    `FrozenReference.attach` can freeze one."""
    from cosmos_framework.posttrain.config.model import DataPackerConfig, HFModelConfig

    class _FakeTrainer:
        def __init__(self, model, objective, opt, **kwargs):
            built.append(("trainer", model, objective))
            self.checkpoint_manager = kwargs["checkpoint_manager"]
            self.auxiliary_models = list(kwargs.get("auxiliary_models", ()))  # what the entry registered

    def _build_hf_model(*args, **_kwargs):
        product = SimpleNamespace(net=net_factory(seed=len(built)))
        built.append(("model", product, args[0] if args else None))  # [2] = the model config it was handed
        return product

    monkeypatch.setattr("cosmos_framework.posttrain.roles.trainer.LLMTrainer", _FakeTrainer)
    monkeypatch.setattr("cosmos_framework.posttrain.model.hf.build_hf_model", _build_hf_model)
    monkeypatch.setattr("cosmos_framework.posttrain.config.build.build_optimizer", lambda *a, **k: object())
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory", lambda *_a, **_k: object()
    )
    monkeypatch.setitem(
        sys.modules, "cosmos_framework.posttrain.workers.ray", SimpleNamespace(current_pdims=lambda: None)
    )
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda x: x)
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    return SimpleNamespace(
        job=SimpleNamespace(project="p", group="g", name="n"),
        rl=SimpleNamespace(run_ts="20260101000000"),
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path="policy/model"),
        rollout=SimpleNamespace(name_or_path="policy/model", max_tokens=16),
        data_packer=DataPackerConfig(),
    )


def _fake_model_dir(path: Any, seed: int = 0) -> Any:
    """A minimal model artifact directory (config + one weight file) whose content follows ``seed``: what
    `rl.ref.model_dir` must name, and what its content identity (`utils/artifact_identity.py`) reads."""
    import hashlib
    import json

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"architectures": ["TinyLM"], "seed": seed}))
    (path / "model.safetensors").write_bytes(hashlib.sha256(f"anchor-{seed}".encode()).digest() * 64)
    return path


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_builds_the_frozen_twin_from_the_policy_builder_only_when_asked(monkeypatch, tmp_path) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    built: list[Any] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built)
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.05))
    trainer = grpo.build_trainer(
        config,
        GRPOConfig(resume="off", kl_beta=0.05, ref=RefConfig(build_ref=True)),
        build_objective_fn=lambda *_: objective,
    )
    models = [entry[1] for entry in built if entry[0] == "model"]
    assert len(models) == 2, "policy and reference come from the same builder"
    policy_model, reference_model = models
    ((_, trainer_model, trainer_objective),) = [entry for entry in built if entry[0] == "trainer"]
    assert trainer_model is policy_model and trainer_objective is objective
    assert objective.reference_scorer.model is reference_model and objective.reference_scorer.net is reference_model.net
    assert not reference_model.net.training and all(not p.requires_grad for p in reference_model.net.parameters())
    assert all(p.requires_grad for p in policy_model.net.parameters()), "the policy itself stays trainable"
    # The SAME model goes to the trainer for residency: the objective scores through it, the trainer
    # moves it, and neither reaches into the other.
    assert trainer.auxiliary_models == [reference_model]

    built.clear()
    plain = make_clipped_objective(_IdsPacker(), _cfg(0.0))
    trainer = grpo.build_trainer(config, GRPOConfig(resume="off"), build_objective_fn=lambda *_: plain)
    assert len([e for e in built if e[0] == "model"]) == 1 and plain.reference_scorer is None
    assert trainer.auxiliary_models == [], "no reference, nothing registered"


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_refuses_a_reference_for_an_objective_without_a_slot(monkeypatch, tmp_path) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    built: list[Any] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built)
    with pytest.raises(RuntimeError, match="carries no frozen-reference slot"):
        grpo.build_trainer(
            config,
            GRPOConfig(resume="off", kl_beta=0.05, ref=RefConfig(build_ref=True)),
            build_objective_fn=lambda *_: object(),
        )


# --------------------------------------------------------------------------- rl.ref.model_dir (warm restart)


@pytest.mark.L0
@pytest.mark.CPU
def test_ref_model_dir_must_be_an_existing_directory_and_needs_build_ref(monkeypatch, tmp_path) -> None:
    """Type and `build_ref` pairing are the dataclass's checks; the directory's EXISTENCE is `build_trainer`'s,
    on every rank and before any model is built, so a saved config (tooling reloading config_resolved.yaml,
    login-node validation, a resume after the base was pruned) reconstructs on a host without the mount."""
    from omegaconf import OmegaConf

    from cosmos_framework.posttrain.entrypoints import grpo

    base = tmp_path / "base"
    base.mkdir()
    ok = GRPOConfig(kl_beta=0.05, ref=GRPORefConfig(build_ref=True, model_dir=str(base)))
    assert ok.ref.model_dir == str(base)
    assert (
        GRPORefConfig().model_dir is None and GRPOConfig().ref.model_dir is None
    )  # default: the policy's own directory
    missing = str(tmp_path / "missing")
    unmounted = GRPOConfig(resume="off", kl_beta=0.05, ref=GRPORefConfig(build_ref=True, model_dir=missing))
    assert unmounted.ref.model_dir == missing, "the dataclass never touches the filesystem"
    built: list[Any] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built)
    with pytest.raises(ValueError, match="rl.ref.model_dir=.*is not a directory"):
        grpo.build_trainer(
            config, unmounted, build_objective_fn=lambda *_: make_clipped_objective(_IdsPacker(), _cfg(0.05))
        )
    assert built == [], "refused before the policy was built, on every rank alike"
    with pytest.raises(ValueError, match="only meaningful with ref.build_ref=true"):
        GRPORefConfig(build_ref=False, model_dir=str(base))
    for bad in ("", "   ", 42):
        with pytest.raises(ValueError, match="non-empty path string"):
            GRPORefConfig(build_ref=True, model_dir=bad)  # type: ignore[arg-type]
    # The launcher's dotted override parses through the structured schema (`rl.ref.model_dir=<base>`).
    schema = OmegaConf.structured(GRPOConfig)
    merged = OmegaConf.merge(
        schema, OmegaConf.from_dotlist(["kl_beta=0.05", "ref.build_ref=true", f"ref.model_dir={base}"])
    )
    resolved = OmegaConf.to_object(merged)
    assert isinstance(resolved, GRPOConfig) and resolved.ref.model_dir == str(base)


def _record_native_builds(monkeypatch: pytest.MonkeyPatch, built: list[Any], seen: list[tuple[str, str, str]]) -> None:
    """Stand-in for `config.build.build_native_policy` that records the artifact identity it was handed."""

    def _build_native_policy(cfg: Any) -> tuple[Any, Any]:
        from cosmos_framework.posttrain.config.build import select_policy_backend

        spec = select_policy_backend(cfg).native_spec
        seen.append((spec.model_id, spec.model_config.name_or_path, spec.model_config.checkpoint_path))
        product = SimpleNamespace(net=_TinyLM(seed=len(built)))
        built.append(("model", product))
        return product, spec.model_config

    monkeypatch.setattr("cosmos_framework.posttrain.config.build.build_native_policy", _build_native_policy)


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_builds_the_twin_from_ref_model_dir_under_the_qwen3_vl_alias(monkeypatch, tmp_path) -> None:
    """The Ultra launcher's shape: model_backend=qwen3_vl, model_qwen3_vl.{name_or_path,checkpoint_path}=<policy dir>."""
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    built: list[Any] = []
    seen: list[tuple[str, str, str]] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built)
    config.model_backend = "qwen3_vl"
    config.model_qwen3_vl = Qwen3VLModelConfig(name_or_path="policy/model", checkpoint_path="policy/model")
    _record_native_builds(monkeypatch, built, seen)
    base = _fake_model_dir(tmp_path / "base")

    objective = make_clipped_objective(_IdsPacker(), _cfg(0.05))
    grpo.build_trainer(
        config,
        GRPOConfig(resume="off", kl_beta=0.05, ref=GRPORefConfig(build_ref=True, model_dir=str(base))),
        build_objective_fn=lambda *_: objective,
    )
    assert seen == [
        ("policy/model", "policy/model", "policy/model"),  # the policy: its own directory, untouched
        (str(base), str(base), str(base)),  # the twin: identity AND weights from rl.ref.model_dir
    ]
    models = [entry[1] for entry in built if entry[0] == "model"]
    assert objective.reference_scorer.model is models[1] and not models[1].net.training
    # The caller's config still names the policy's directory: the twin was built from a copy.
    assert (config.model_qwen3_vl.name_or_path, config.model_qwen3_vl.checkpoint_path) == (
        "policy/model",
        "policy/model",
    )

    seen.clear()
    built.clear()
    plain = make_clipped_objective(_IdsPacker(), _cfg(0.05))
    grpo.build_trainer(
        config,
        GRPOConfig(resume="off", kl_beta=0.05, ref=RefConfig(build_ref=True)),
        build_objective_fn=lambda *_: plain,
    )
    assert seen == [("policy/model", "policy/model", "policy/model")] * 2, (
        "no override: twin from the policy's directory"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_redirects_only_the_populated_identity_fields_of_a_generic_i4_spec(monkeypatch, tmp_path) -> None:
    """`config.model.model_id` carries the identity; an empty `name_or_path`/`checkpoint_path` stays empty."""
    from cosmos_framework.posttrain.config.model import NativePolicyBuildSpec, Qwen3VLModelConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    built: list[Any] = []
    seen: list[tuple[str, str, str]] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built)
    config.model_backend = "i4"
    config.model = NativePolicyBuildSpec(
        builder="tests.fake.build_policy", model_id="policy/model", model_config=Qwen3VLModelConfig()
    )
    _record_native_builds(monkeypatch, built, seen)
    base = _fake_model_dir(tmp_path / "base")
    warm = GRPOConfig(resume="off", kl_beta=0.05, ref=GRPORefConfig(build_ref=True, model_dir=str(base)))

    objective = make_clipped_objective(_IdsPacker(), _cfg(0.05))
    grpo.build_trainer(config, warm, build_objective_fn=lambda *_: objective)
    assert seen == [("policy/model", "", ""), (str(base), "", "")]
    assert config.model.model_id == "policy/model" and config.model.model_config.name_or_path == ""

    # A populated storage mirror of the policy's weights is redirected too: it would otherwise win over model_id.
    config.model = NativePolicyBuildSpec(
        builder="tests.fake.build_policy",
        model_id="policy/model",
        model_config=Qwen3VLModelConfig(name_or_path="policy/model", checkpoint_path="mirror/of/policy"),
    )
    seen.clear()
    built.clear()
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.05))
    grpo.build_trainer(config, warm, build_objective_fn=lambda *_: objective)
    assert seen == [("policy/model", "policy/model", "mirror/of/policy"), (str(base), str(base), str(base))]
    assert config.model.model_config.checkpoint_path == "mirror/of/policy"


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_builds_the_hf_twin_from_ref_model_dir(monkeypatch, tmp_path) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    built: list[Any] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built)
    base = _fake_model_dir(tmp_path / "base")
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.05))
    grpo.build_trainer(
        config,
        GRPOConfig(resume="off", kl_beta=0.05, ref=GRPORefConfig(build_ref=True, model_dir=str(base))),
        build_objective_fn=lambda *_: objective,
    )
    handed = [entry[2].name_or_path for entry in built if entry[0] == "model"]
    assert handed == ["policy/model", str(base)], "policy from its own directory, twin from rl.ref.model_dir"
    assert config.model_hf.name_or_path == "policy/model"


@pytest.mark.L0
@pytest.mark.CPU
def test_reference_artifact_identity_binds_the_anchor_content_on_exact_resume(monkeypatch, tmp_path) -> None:
    """The KL anchor's weights are part of the objective. `build_trainer` binds the CONTENT identity of
    `rl.ref.model_dir` into the objective's exact-resume identity, so a resume that names the same directory
    passes, while another directory, the same path re-populated with other weights, or dropping the knob is
    refused by the trainer before it continues — the path string alone would accept every one of those."""
    from cosmos_framework.posttrain.config.build import build_objective
    from cosmos_framework.posttrain.entrypoints import grpo
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer
    from cosmos_framework.posttrain.utils.artifact_identity import model_artifact_identity

    def _built_objective(model_dir: Any | None) -> Any:
        built: list[Any] = []
        config = _mock_build_trainer(monkeypatch, tmp_path, built)
        ref = (
            RefConfig(build_ref=True) if model_dir is None else GRPORefConfig(build_ref=True, model_dir=str(model_dir))
        )
        rl = GRPOConfig(resume="off", kl_beta=0.05, kl_coeff_clamp=5.0, group_size=2, ref=ref)
        # The config-level builder, so the identity carries the loss keys the artifact joins.
        objective = build_objective(rl, "unused", packer=_IdsPacker())
        grpo.build_trainer(config, rl, build_objective_fn=lambda *_: objective)
        return objective

    def _trainer(objective: Any) -> Any:
        policy = _TinyLM()
        return LLMTrainer(policy, objective, torch.optim.SGD(policy.parameters(), lr=0.1))

    base = _fake_model_dir(tmp_path / "base", seed=1)
    anchored = _built_objective(base)
    assert anchored.identity["reference_artifact"] == model_artifact_identity(base)
    assert anchored.identity["kl_beta"] == 0.05 and anchored.identity["kl_coeff_clamp"] == 5.0, (
        "the artifact joins the loss keys; it does not replace them"
    )
    saved = _trainer(anchored)._trainer_state()
    assert saved["objective_identity"]["reference_artifact"] == model_artifact_identity(base)

    # (a) An exact resume naming the same directory passes the identity check.
    _trainer(_built_objective(base))._restore_trainer_state_local(saved)
    # The path is not what is bound: the same bytes at another location pass too.
    import shutil

    moved = tmp_path / "moved" / "base"
    shutil.copytree(base, moved)
    _trainer(_built_objective(moved))._restore_trainer_state_local(saved)

    # (b) Another checkpoint directory is refused.
    other = _fake_model_dir(tmp_path / "step200", seed=2)
    with pytest.raises(RuntimeError, match="objective exact-resume identity differs"):
        _trainer(_built_objective(other))._restore_trainer_state_local(saved)
    # (b') The same path re-populated with other weights is refused: content, not the path string.
    _fake_model_dir(base, seed=3)
    with pytest.raises(RuntimeError, match="objective exact-resume identity differs"):
        _trainer(_built_objective(base))._restore_trainer_state_local(saved)
    # Dropping the knob (the anchor becomes the policy's own directory) is refused too, and the
    # reverse: a checkpoint anchored to the policy's own weights refuses a resume that adds one.
    plain = _built_objective(None)
    assert "reference_artifact" not in plain.identity, "absent means the policy's own initial weights"
    with pytest.raises(RuntimeError, match="objective exact-resume identity differs"):
        _trainer(plain)._restore_trainer_state_local(saved)
    with pytest.raises(RuntimeError, match="objective exact-resume identity differs"):
        _trainer(_built_objective(other))._restore_trainer_state_local(_trainer(plain)._trainer_state())


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_refuses_a_ref_model_dir_without_weights(monkeypatch, tmp_path) -> None:
    """A directory that exists but holds no weight file is not an anchor (half-copied or pruned base): the
    identity read refuses it on every rank before any model is built."""
    from cosmos_framework.posttrain.entrypoints import grpo

    built: list[Any] = []
    config = _mock_build_trainer(monkeypatch, tmp_path, built)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no model weight file"):
        grpo.build_trainer(
            config,
            GRPOConfig(resume="off", kl_beta=0.05, ref=GRPORefConfig(build_ref=True, model_dir=str(empty))),
            build_objective_fn=lambda *_: make_clipped_objective(_IdsPacker(), _cfg(0.05)),
        )
    assert built == [], "refused before the policy was built"


# --------------------------------------------------------------------------- bounded k3 gradient


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_coeff_clamp_bounds_the_gradient_coefficient_and_keeps_the_value() -> None:
    """VALUE: the exact k3 at every z, bit-identical to the unclamped penalty. GRADIENT: exact inside
    |z| <= c and frozen at the boundary coefficient beyond it, so at z = ±10 with c = 5 the gradient
    equals the gradient at z = ±5 — not 0 (a saturating clamp) and not the unbounded 1 − e^{±10}."""
    c = 5.0
    z = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0], dtype=torch.float32)  # z = ref − cur
    reference = torch.zeros_like(z)
    current = (-z).clone().requires_grad_(True)
    bounded = sampled_k3_penalty(reference, current, coeff_clamp=c)  # [7]
    exact_value = sampled_k3_penalty(reference, current.detach())  # [7]
    assert torch.equal(bounded.detach(), exact_value), "the logged penalty must stay the unbiased k3"
    assert bounded.requires_grad
    (grad_bounded,) = torch.autograd.grad(bounded.sum(), current)
    # d k3 / d cur = 1 − exp(z), evaluated at the clamped z.
    expected = 1.0 - torch.exp(z.clamp(-c, c))  # [7]
    assert torch.allclose(grad_bounded, expected, rtol=1e-6, atol=1e-6), (grad_bounded, expected)
    assert float(grad_bounded[0]) == float(grad_bounded[1]) != 0.0  # z = −10 vs z = −5
    assert float(grad_bounded[-1]) == float(grad_bounded[-2])  # z = +10 vs z = +5
    assert math.isclose(float(grad_bounded[-1]), 1.0 - math.exp(c), rel_tol=1e-6)  # ≈ −147.4
    current_exact = (-z).clone().requires_grad_(True)
    (grad_exact,) = torch.autograd.grad(sampled_k3_penalty(reference, current_exact).sum(), current_exact)
    inside = z.abs() <= c  # [7]
    assert torch.allclose(grad_bounded[inside], grad_exact[inside], rtol=1e-6, atol=1e-6)
    assert float(grad_exact[-1]) < -2.0e4 and float(grad_bounded[-1]) > -150.0  # the finding, bounded
    with pytest.raises(ValueError, match="positive"):
        sampled_k3_penalty(reference, current, coeff_clamp=0.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_coeff_clamp_default_none_reproduces_the_old_loss_and_gradient_exactly() -> None:
    """The default keeps today's arithmetic bit for bit: the loss with `kl_coeff_clamp=None` equals
    the pre-change formula (k3 = expm1(z) − z on the ±20-clamped log-ratio, added as kl_beta·k3) in
    value AND gradient. With the bound set the VALUE and both kl statistics are unchanged and only the
    gradient of tokens beyond |z| = c differs."""
    torch.manual_seed(3)
    current = torch.randn(8, requires_grad=True)
    offsets = torch.tensor([3.0, -3.0, 0.5, -0.5, 0.0, 6.0, -6.0, 1.9])  # z = ref − cur per token
    reference = current.detach() + offsets
    advantage = torch.randn(8)
    kl_beta = 0.3
    z = (reference - current).clamp(-TOKEN_LOG_RATIO_CLAMP, TOKEN_LOG_RATIO_CLAMP)  # [8]
    legacy = clipped_policy_gradient_loss(current, current.detach(), advantage, 0.2, 0.2, kl_beta=0.0) + kl_beta * (
        torch.expm1(z) - z
    )  # [8]
    stats: dict = {}
    today = clipped_policy_gradient_loss(
        current, current.detach(), advantage, 0.2, 0.2, reference=reference, kl_beta=kl_beta, stats=stats
    )  # [8]
    assert torch.equal(today.detach(), legacy.detach())
    (grad_today,) = torch.autograd.grad(today.sum(), current, retain_graph=True)
    (grad_legacy,) = torch.autograd.grad(legacy.sum(), current, retain_graph=True)
    assert torch.equal(grad_today, grad_legacy)
    per_token_k3 = (torch.expm1(z) - z).detach()  # [8]
    assert math.isclose(stats["kl_sum"], float(per_token_k3.sum()), rel_tol=1e-6)
    assert math.isclose(stats["kl_max"], float(per_token_k3.max()), rel_tol=1e-6)
    assert stats["kl_max"] == float(per_token_k3[5])  # z = 6 is the largest penalty

    c = 2.0
    clamped_stats: dict = {}
    clamped = clipped_policy_gradient_loss(
        current,
        current.detach(),
        advantage,
        0.2,
        0.2,
        reference=reference,
        kl_beta=kl_beta,
        stats=clamped_stats,
        kl_coeff_clamp=c,
    )  # [8]
    assert torch.equal(clamped.detach(), legacy.detach())
    assert clamped_stats["kl_sum"] == stats["kl_sum"] and clamped_stats["kl_max"] == stats["kl_max"]
    (grad_clamped,) = torch.autograd.grad(clamped.sum(), current)
    inside = offsets.abs() <= c  # [8]
    assert torch.allclose(grad_clamped[inside], grad_legacy[inside], rtol=1e-6, atol=1e-6)
    assert bool((grad_clamped[~inside] != grad_legacy[~inside]).all())
    # Beyond the band the KL part of the gradient is the boundary coefficient, kl_beta·(1 − e^{±c}).
    surrogate_only = clipped_policy_gradient_loss(current, current.detach(), advantage, 0.2, 0.2, kl_beta=0.0)
    (grad_surrogate,) = torch.autograd.grad(surrogate_only.sum(), current)
    kl_part = grad_clamped - grad_surrogate  # [8]
    expected = kl_beta * (1.0 - torch.exp(offsets.clamp(-c, c)))  # [8]
    assert torch.allclose(kl_part, expected, rtol=1e-5, atol=1e-6), (kl_part, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_coeff_clamp_keeps_the_twenty_nat_saturation_untouched() -> None:
    """The outer ±20 clamp is unchanged: a collapsed token (z = 120) still carries the capped value
    k3(20) and a ZERO gradient with the coefficient bound on, exactly as without it."""
    current = torch.tensor([-130.0, -1.0], requires_grad=True)
    reference = torch.tensor([-10.0, -1.0])
    loss = clipped_policy_gradient_loss(
        current, current.detach(), torch.zeros(2), 0.2, 0.2, reference=reference, kl_beta=1.0, kl_coeff_clamp=5.0
    )
    assert torch.isfinite(loss).all()
    cap = float(sampled_k3_penalty(torch.tensor([TOKEN_LOG_RATIO_CLAMP]), torch.tensor([0.0])))
    assert math.isclose(float(loss[0].detach()), cap, rel_tol=1e-6)
    loss.sum().backward()
    assert float(current.grad[0]) == 0.0 and float(current.grad[1]) == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_kl_max_reports_the_largest_token_penalty_and_is_declared_max_reduced() -> None:
    """`grpo_kl_max` is in the public contract, MAX-reduced (a per-shard maximum averaged or summed
    across ranks would be a number no token had), 0.0 at identical weights and on an empty shard, and
    at least the mean after drift. The trainer exposes the declaration the entry loop merges by."""
    from cosmos_framework.posttrain.algorithm.grpo import _GRPO_METRIC_ADAPTER
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    assert _GRPO_PUBLIC_METRICS["grpo_kl_max"] == "kl_max"
    # ONE declaration point for maxima: the adapter names both the KL maximum and the longest response.
    assert _GRPO_METRIC_ADAPTER.max_reduced == frozenset({"grpo_kl_max", "grpo_response_len_max", "grpo_sc_rho_max"})
    assert _GRPO_METRIC_ADAPTER.public_max_keys("train") == frozenset(
        {"train/kl_max", "train/response_len_max", "train/sc_rho_max"}
    )
    assert "grpo_kl_max" not in _GRPO_METRIC_ADAPTER.weighted_by

    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.1))
    objective.reference_scorer.attach(reference)
    assert objective.metric_max_keys("train") == frozenset(
        {"train/kl_max", "train/response_len_max", "train/sc_rho_max"}
    )
    trainer = LLMTrainer(policy, objective, torch.optim.SGD(policy.parameters(), lr=0.1))
    assert trainer.training_metric_max_keys() == frozenset(
        {"train/kl_max", "train/response_len_max", "train/sc_rho_max"}
    )

    _, same = objective(policy, _batch([1.0, 0.0, 1.0]))
    assert same["grpo_kl_max"] == 0.0 and same["grpo_kl_mean"] == 0.0
    _perturb(policy)
    _, drifted = objective(policy, _batch([1.0, 1.0, 1.0]))
    assert drifted["grpo_kl_max"] >= drifted["grpo_kl_mean"] > 0.0, drifted
    # The unanchored objective reports the key too (config-invariant key set), at exactly 0.0.
    _, plain = make_clipped_objective(_IdsPacker(), _cfg(0.0))(policy, _batch([1.0, 0.0]))
    assert plain["grpo_kl_max"] == 0.0 and set(plain) == set(_GRPO_PUBLIC_METRICS)


# --------------------------------------------------------------------------- config: the bound


@pytest.mark.L0
@pytest.mark.CPU
def test_config_kl_coeff_clamp_validation_and_missing_clamp_warning() -> None:
    import warnings

    from omegaconf import OmegaConf

    from cosmos_framework.posttrain.config.rl import KL_LOG_RATIO_CLAMP_NATS

    # The config module stays torch-free, so it mirrors the primitive's clamp; pin them equal.
    assert KL_LOG_RATIO_CLAMP_NATS == TOKEN_LOG_RATIO_CLAMP
    assert GRPOConfig().kl_coeff_clamp is None  # default: today's exact gradient
    with pytest.warns(UserWarning, match="kl_coeff_clamp unset.*unbounded below"):
        GRPOConfig(kl_beta=0.01, ref=RefConfig(build_ref=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bounded = GRPOConfig(kl_beta=0.01, kl_coeff_clamp=5.0, ref=RefConfig(build_ref=True))
        GRPOConfig()  # unanchored: nothing to warn about
    assert not [w for w in caught if "kl_coeff_clamp" in str(w.message)], [str(w.message) for w in caught]
    assert bounded.kl_coeff_clamp == 5.0
    with pytest.raises(ValueError, match="requires kl_beta>0"):
        GRPOConfig(kl_coeff_clamp=5.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            GRPOConfig(kl_beta=0.01, kl_coeff_clamp=bad, ref=RefConfig(build_ref=True))
    for wrong_type in (True, "5"):
        with pytest.raises(TypeError, match="non-bool real"):
            GRPOConfig(kl_beta=0.01, kl_coeff_clamp=wrong_type, ref=RefConfig(build_ref=True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inert"):
        GRPOConfig(kl_beta=0.01, kl_coeff_clamp=TOKEN_LOG_RATIO_CLAMP, ref=RefConfig(build_ref=True))
    with pytest.raises(ValueError, match="does not consume clipped-surrogate settings.*kl_coeff_clamp"):
        GRPOConfig(grpo_surrogate="legacy_pg", kl_coeff_clamp=5.0)
    schema = OmegaConf.structured(GRPOConfig)
    merged = OmegaConf.merge(
        schema, OmegaConf.from_dotlist(["kl_beta=0.01", "kl_coeff_clamp=5.0", "ref.build_ref=true"])
    )
    resolved = OmegaConf.to_object(merged)
    assert isinstance(resolved, GRPOConfig) and resolved.kl_coeff_clamp == 5.0
    # The objective built from a config threads the bound through (hand-built cfgs are validated too).
    objective = make_clipped_objective(_IdsPacker(), bounded)
    assert objective.reference_scorer is not None
    with pytest.raises(ValueError, match="positive finite"):
        make_clipped_objective(_IdsPacker(), SimpleNamespace(kl_beta=0.1, kl_coeff_clamp=-1.0, group_size=2))


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_identity_records_kl_coeff_clamp_only_when_set() -> None:
    """Absent means the exact gradient every earlier anchored checkpoint trained under; an
    unconditional key would refuse their exact resume on a byte carrying no information."""
    from cosmos_framework.posttrain.config.build import build_objective

    with pytest.warns(UserWarning, match="kl_coeff_clamp unset"):
        unbounded = build_objective(GRPOConfig(kl_beta=0.01, ref=RefConfig(build_ref=True)), "unused", packer=object())
    assert "kl_coeff_clamp" not in unbounded.identity and unbounded.identity["kl_beta"] == 0.01
    bounded = build_objective(
        GRPOConfig(kl_beta=0.01, kl_coeff_clamp=5.0, ref=RefConfig(build_ref=True)), "unused", packer=object()
    )
    assert bounded.identity["kl_coeff_clamp"] == 5.0
    assert set(bounded.identity) == set(unbounded.identity) | {"kl_coeff_clamp"}


# --------------------------------------------------------------------------- console line


@pytest.mark.L0
@pytest.mark.CPU
def test_console_headline_keys_add_grad_norm_and_the_kl_columns_only_when_anchored() -> None:
    """`step N/M train/loss_avg …` carries the pre-clip grad_norm always and the KL mean AND max
    when the anchor is on; the reporter prints exactly the headline keys present in the frame."""
    import logging

    from cosmos_framework.posttrain.entrypoints.grpo import console_headline_keys
    from cosmos_framework.posttrain.utils.log import logger as posttrain_logger
    from cosmos_framework.posttrain.utils.report import ConsoleReporter

    plain = console_headline_keys(SimpleNamespace(kl_beta=0.0))
    assert plain == ("train/loss_avg", "train/reward_mean", "train/grpo_informative_group_count", "train/grad_norm")
    anchored = console_headline_keys(SimpleNamespace(kl_beta=0.01))
    assert anchored == plain + ("train/kl_mean", "train/kl_max")

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    posttrain_logger.addHandler(handler)
    try:
        reporter = ConsoleReporter(total_steps=300, headline_keys=anchored)
        reporter.report(
            {
                "train/loss_avg": 0.5,
                "train/reward_mean": 0.25,
                "train/grpo_informative_group_count": 3,
                "train/grad_norm": 114.0,
                "train/kl_mean": 173.0,
                "train/kl_max": 4.8e8,
                "train/kl_beta": 0.01,
            },
            step=34,
        )
    finally:
        posttrain_logger.removeHandler(handler)
    line = records[-1].getMessage()
    assert line.startswith("step   34/300")
    for key in anchored:
        assert key in line, (key, line)
    assert "train/kl_beta" not in line  # not a headline column
