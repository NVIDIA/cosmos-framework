# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `algorithm/opd.py` — the direct OPD objective end to end.

The divergence MATH is pinned in `test_algorithm_opd_slice_divergence.py`; this file pins the wiring around it,
which is where an OPD implementation actually goes wrong:

- **Alignment.** `TeacherSignal` row `j` must meet the `j`-th scored position of the `j`-th
  sequence, through a padded batched forward and a boolean mask. Get the flattening order wrong and
  nothing raises — the student is confidently distilled against another position's teacher. The
  fixed point is the check: a signal built FROM the student's own logprobs must give exactly zero
  divergence, and swapping two sequences' signals must not.
- **Convergence.** Malformed data lands on one rank under per-sample sharding, so validation runs
  above the first forward, not between forwards.
- **Reduction.** Three `loss_type` modes, each divided by a DP-global denominator, with the
  grad_accum contract the trainer reads off `Objective.reduction`.

WHY EVERY FORWARD HERE IS SHARPENED. A randomly-initialized GPT2 is almost exactly uniform over its
50257 tokens, so the top-k support holds ~1e-4 of the mass and the partition is
`[tiny, ..., tiny, ~1.0]` on both sides no matter which position or which teacher — every
divergence lands around 1e-7 and every negative control passes for the wrong reason (measured:
reversing the teacher's top-k moved the loss by 8e-7). Dividing the logits by
`SHARPEN_TEMPERATURE` concentrates them, which is the objective's own `temperature` knob rather
than a test-only hack, and `test_fixture_is_sharp_enough_to_fail` keeps that property honest.

The value is measured, not guessed: this model's logits have std 0.11, so the top-5 support holds
1e-4 of the mass at T=1 and 0.0033 at T=0.1 — still uniform for practical purposes. T=0.01 puts it
at 0.86..1.00 with at least 1.85 nats between the first and fifth slot.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.opd import TopKSliceDivergence, make_direct_objective
from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM, Reduction
from cosmos_framework.posttrain.data.packer.decoder_only_llm import shape_policy_input
from cosmos_framework.posttrain.schema import ObsBundle, Status, TeacherSignal, Trajectory, Transition

TOP_K = 5
SHARPEN_TEMPERATURE = 0.01  # see the module docstring — without it the negative controls are vacuous


def _tiny():
    """Replicated tiny random GPT2 + gpt2 packer (posttrain tests never import each other).

    Mirrors `tests/test_algorithm_grpo_clipped.py::_tiny`. `.eval()` disables dropout so the
    reference forward below and the objective's batched forward see the same weights."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.data.packer import init_data_packer

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()
    return model, init_data_packer("gpt2")


def _objective(packer, **kwargs):
    """`make_direct_objective` with the sharpening temperature, unless a test overrides it."""
    kwargs.setdefault("temperature", SHARPEN_TEMPERATURE)
    return make_direct_objective(packer, **kwargs)


def _traj(packer, text: str, completion: str, index: int) -> Trajectory:
    """One completed trajectory, shaped the way the rollout leaves it (ids in `action`)."""
    ids = torch.tensor(packer.tokenizer(completion, add_special_tokens=False).input_ids, dtype=torch.long)
    tr = Transition(
        obs=ObsBundle(representations={"prompt": text}, id=f"p{index}"),
        action=ids,
        policy_info={"completion_text": completion},
        done=True,
    )
    return Trajectory(transitions=[tr], id=f"g0_k{index}", group_id="g0", status=Status.COMPLETED)


def _student_scored_logprobs(model, packer, traj: Trajectory):
    """The student's FULL-vocabulary logprobs at exactly the positions the loss is taken over.

    Computed the slow, obvious way — one unbatched forward, `log_softmax` over the whole vocabulary,
    then the mask — so it shares no code with the objective's batched/flattened path. That is what
    makes the fixed-point assertion meaningful rather than tautological."""
    policy_input = shape_policy_input(traj.transitions[0], packer, 0)
    ids, mask = policy_input.input_ids, policy_input.logprob_masks
    with torch.no_grad():
        logits = model(torch.tensor(ids).unsqueeze(0)).logits[0] / SHARPEN_TEMPERATURE
    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    return logprobs[torch.tensor(mask[:-1], dtype=torch.bool)]  # [n_scored, V]


def _student_topk(model, packer, traj: Trajectory, k: int = TOP_K):
    """The student's own top-k ids and their full-vocabulary logprobs — the objective's fixed point."""
    top = _student_scored_logprobs(model, packer, traj).topk(k, dim=-1)
    return top.indices.contiguous(), top.values.contiguous()


def _with_exact_signals(model, packer, trajs):
    """Attach a teacher signal that IS the student's own distribution."""
    for traj in trajs:
        ids, logprobs = _student_topk(model, packer, traj)
        traj.transitions[0].targets = TeacherSignal(topk_ids=ids, topk_logprobs=logprobs)
    return trajs


def _with_reversed_signals(model, packer, trajs):
    """A teacher that reassigns the SAME support mass across the support, in reverse rank order.

    Total top-k mass is unchanged, so the "other" bucket is identical and the divergence measures
    only the disagreement inside the support — which is what a distillation loss is for."""
    for traj in trajs:
        ids, logprobs = _student_topk(model, packer, traj)
        traj.transitions[0].targets = TeacherSignal(topk_ids=ids, topk_logprobs=logprobs.flip(-1))
    return trajs


def _ragged_batch(packer):
    """Two trajectories of DIFFERENT total length, so the batched forward genuinely pads.

    Padding is where an alignment bug hides: a right-padded row's scored positions still sit at the
    same indices, but any implementation that flattened by column, or trusted `max_len`, misplaces
    the shorter sequence's rows."""
    specs = [("3+4=", " seven exactly"), ("100+200=", " three hundred exactly now")]
    trajs = [_traj(packer, text, completion, i) for i, (text, completion) in enumerate(specs)]
    lengths = {len(shape_policy_input(t.transitions[0], packer, 0).input_ids) for t in trajs}
    assert len(lengths) == 2, f"fixture must produce unequal lengths, got {lengths}"
    return trajs


# ── the fixture itself ────────────────────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_fixture_is_sharp_enough_to_fail():
    """Guards every negative control below: the top-k support must hold real probability mass.

    Without this, a future change to the model config or the temperature could quietly return the
    fixture to near-uniform, where teacher and student agree to 1e-7 whatever you do to them and
    the alignment tests all pass while testing nothing."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    _, logprobs = _student_topk(model, packer, _ragged_batch(packer)[0])

    support_mass = logprobs.exp().sum(dim=-1)
    assert float(support_mass.min()) > 0.5, float(support_mass.min())
    spread = logprobs[:, 0] - logprobs[:, -1]  # top slot vs k-th slot, in nats
    assert float(spread.min()) > 1.0, float(spread.min())


# ── the fixed point: teacher == student ⇒ zero divergence ─────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_equal_to_student_gives_zero_loss_on_a_padded_batch():
    """The whole objective collapses to zero when the teacher IS the student.

    A generalized JSD between identical distributions is zero for every beta, so this covers the
    partition, the gather, the flattening, the per-sequence split and the reduction at once — a
    misalignment anywhere shows up as a positive loss."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    trajs = _with_exact_signals(model, packer, _ragged_batch(packer))

    for beta in (0.0, 0.5, 1.0):
        loss, metrics = _objective(packer, jsd_beta=beta)(model, trajs)
        # 1e-5, not 1e-6: this quantity is EXACTLY zero in real arithmetic, so what is being measured is float32 round-off through a log_softmax over a ~50k vocabulary. CI's x86 and our aarch64 round differently (x86 produced -2.3e-6 on the per-token form), so a 1e-6 bound tests the runner. The negative controls keep their thresholds, so the separation is unchanged.
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-5), (beta, float(loss.detach()))
        assert metrics["opd_n"] == 2
        assert metrics["opd_valid_tokens"] > 0


@pytest.mark.L0
@pytest.mark.CPU
def test_swapping_two_sequences_signals_breaks_the_fixed_point():
    """The negative control for the test above: it must be possible to fail it.

    Two sequences, each holding the OTHER's teacher signal. The rows stay individually valid (same
    k, real ids, real full-vocabulary logprobs), so nothing raises — only the pairing is wrong, and
    a loss that stayed at zero would mean the objective never looked at which row went where."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    trajs = _ragged_batch(packer)
    signals = [_student_topk(model, packer, traj) for traj in trajs]
    # Trim both sides to the shorter span so the swap is shape-legal and this test isolates
    # alignment instead of re-testing the row-count check.
    shortest = min(ids.shape[0] for ids, _ in signals)
    for traj, (ids, logprobs) in zip(trajs, reversed(signals)):
        traj.transitions[0].action = traj.transitions[0].action[:shortest]
        traj.transitions[0].policy_info.pop("completion_text", None)
        traj.transitions[0].targets = TeacherSignal(topk_ids=ids[:shortest], topk_logprobs=logprobs[:shortest])

    loss, _ = _objective(packer, jsd_beta=1.0)(model, trajs)

    assert float(loss.detach()) > 1e-2, float(loss.detach())


# ── gradient, alignment and validation ────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_divergence_is_positionwise_and_differentiable():
    """`TopKSliceDivergence` returns one value per scored position, and only the perturbed row moves.

    Tested on the source directly rather than through the objective: the objective reduces to a
    scalar, which cannot tell "row 1 diverged" from "every row diverged a little"."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    traj = _traj(packer, "3+4=", " seven exactly now", 0)
    ids, logprobs = _student_topk(model, packer, traj)

    policy_input = shape_policy_input(traj.transitions[0], packer, 0)
    scored_ids, mask = policy_input.input_ids, policy_input.logprob_masks
    logits = model(torch.tensor(scored_ids).unsqueeze(0)).logits[0] / SHARPEN_TEMPERATURE
    student_logits = logits[:-1][torch.tensor(mask[:-1], dtype=torch.bool)]
    student_logits.retain_grad()

    perturbed = logprobs.clone()
    perturbed[1] = perturbed[1].flip(0)  # same support mass, reassigned across it -> row 1 only
    signal = TeacherSignal(topk_ids=ids, topk_logprobs=perturbed)

    per_token = TopKSliceDivergence(beta=1.0).per_token(student_logits, [signal], torch.device("cpu"))

    assert per_token.shape == (ids.shape[0],)
    assert float(per_token[1].detach()) > 1e-2, float(per_token[1].detach())
    untouched = per_token[torch.arange(per_token.numel()) != 1]
    # 1e-5, not 1e-6: an untouched row's teacher IS its student, so the value is exactly zero in
    # real arithmetic and pure float32 round-off here -- a log_softmax over 50257 entries, summed
    # over the support. That round-off is architecture-dependent (CI's x86 produced -2.3e-6 where
    # aarch64 gives ~1e-7), so a 1e-6 bound tests the runner, not the code. The bound that matters
    # is the SEPARATION from the perturbed row asserted above: 1e-2 against 1e-5 leaves three
    # orders of margin, so a real misalignment still cannot hide under it.
    assert torch.allclose(untouched, torch.zeros_like(untouched), atol=1e-5), untouched

    per_token.sum().backward()
    assert student_logits.grad is not None and float(student_logits.grad.abs().sum()) > 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_gradient_and_rank_local_failures_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loss reaches parameters; peer preparation/finalization errors preempt later collectives."""
    import torch.distributed as dist

    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    trajs = _with_reversed_signals(model, packer, _ragged_batch(packer))

    model.zero_grad()
    loss, _ = _objective(packer, jsd_beta=1.0)(model, trajs)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert sum(float(g.abs().sum()) for g in grads) > 0.0

    # Emulate the healthy rank of a two-rank job. First its peer fails during rank-local collation,
    # which must stop this rank before the FSDP forward. Then its peer fails after the forward,
    # which must stop this rank before rank_mean_count's AVG collective.
    verdict_count = 0
    fail_at_verdict = 2
    peer_error = "RuntimeError: peer collate failed"
    collectives: list[object] = []
    source_calls: list[int] = []
    forward_calls = 0
    original_forward = model.forward

    def counting_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    def fake_all_reduce(tensor: torch.Tensor, op: object = None) -> None:
        nonlocal verdict_count
        collectives.append(op)
        if tensor.dtype is torch.int32:
            verdict_count += 1
            if verdict_count == fail_at_verdict:
                tensor.fill_(1)  # [] -- peer failed at this converged phase

    def fake_all_gather_object(errors: list[str | None], local_error: str | None) -> None:
        errors[:] = [local_error, peer_error]

    class _HealthySource:
        def per_token(
            self,
            student_logits: torch.Tensor,
            signals: list[TeacherSignal],
            device: torch.device,
        ) -> torch.Tensor:  # student_logits: [N,V], returns [N]
            source_calls.append(1)
            return TopKSliceDivergence(beta=1.0).per_token(student_logits, signals, device)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda *args, **kwargs: "gloo")
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(model, "forward", counting_forward)

    with pytest.raises(RuntimeError, match="peer collate failed"):
        _objective(packer, jsd_beta=1.0, divergence=_HealthySource())(model, trajs)

    assert forward_calls == 0, "a peer collate failure must converge before any rank enters the forward"
    assert source_calls == []
    assert verdict_count == 2, "trajectory preparation and forward preparation each need one verdict"
    assert dist.ReduceOp.AVG not in collectives

    verdict_count = 0
    fail_at_verdict = 4
    peer_error = "RuntimeError: peer divergence failed"
    collectives.clear()

    with pytest.raises(RuntimeError, match="peer divergence failed"):
        _objective(packer, jsd_beta=1.0, divergence=_HealthySource())(model, trajs)

    assert forward_calls == 1, "the healthy rank must enter exactly one forward before finalization"
    assert source_calls == [1], "the healthy rank must finish its post-forward local work"
    assert verdict_count == 4, "all three pre-forward phases and post-forward finalization need a verdict"
    assert dist.ReduceOp.AVG not in collectives, "a failed finalization must preempt rank_mean_count"


@pytest.mark.L0
@pytest.mark.CPU
def test_signal_row_count_must_match_the_scored_span():
    """One row per scored token, checked before the forward.

    A signal built for the full completion when `n_ignore_prefix_tokens` trimmed the span would
    pair every position with its neighbour's teacher — silent, and the loss would look healthy."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    trajs = _with_exact_signals(model, packer, _ragged_batch(packer))
    signal = trajs[0].transitions[0].targets
    trajs[0].transitions[0].targets = TeacherSignal(
        topk_ids=signal.topk_ids[:-1], topk_logprobs=signal.topk_logprobs[:-1]
    )

    with pytest.raises(ValueError, match="row.*but the objective scores"):
        _objective(packer)(model, trajs)


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_or_wrong_targets_are_refused():
    """`targets` holds whatever the algorithm put there — DPO puts a `Preference` in the same slot."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()

    with pytest.raises(ValueError, match="needs a TeacherSignal"):
        _objective(packer)(model, _ragged_batch(packer))

    chunk_only = _ragged_batch(packer)
    for traj in chunk_only:
        traj.transitions[0].targets = TeacherSignal(
            chunk_bounds=torch.zeros(2, 2, dtype=torch.long), chunk_teacher_logprob=torch.zeros(2)
        )
    with pytest.raises(ValueError, match="cross-tokenizer DPCA chunk payload"):
        _objective(packer)(model, chunk_only)


# ── reduction ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_reduction_contract_follows_loss_type():
    """Every mode defers its DP-global valid-unit divisor across micro-batches."""
    pytest.importorskip("transformers")
    _, packer = _tiny()
    objective = _objective(packer, loss_type="token-mean")
    assert objective.reduction is Reduction.TOKEN_SUM
    assert objective.metric_weight_map("train") == {
        "train/loss_avg": "train/weight_sum",
        "train/opd_divergence": "train/valid_token_count",
    }
    assert _objective(packer, loss_type="seq-mean-token-sum").reduction is Reduction.TOKEN_SUM
    assert _objective(packer, loss_type="seq-mean-token-mean").reduction is Reduction.TOKEN_SUM
    with pytest.raises(ValueError, match="loss_type must be one of"):
        _objective(packer, loss_type="mean")


@pytest.mark.L0
@pytest.mark.CPU
def test_loss_types_weight_ragged_completions_differently():
    """A semantic knob, not formatting: the three modes disagree whenever completions differ in length.

    If they agreed, the mode would be untestable from outside and a wiring mistake that ignored it
    would be invisible."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()

    def _loss(loss_type: str) -> float:
        trajs = _with_reversed_signals(model, packer, _ragged_batch(packer))
        loss, _ = _objective(packer, loss_type=loss_type)(model, trajs)
        return float(loss.detach())

    values = [_loss(mode) for mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean")]
    assert len(set(round(v, 9) for v in values)) == 3, values


@pytest.mark.L0
@pytest.mark.CPU
def test_beta_selects_the_divergence_direction():
    """Forward, reverse and interior JSD are different numbers on an asymmetric pair, and all >= 0."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()

    def _loss(beta: float) -> float:
        trajs = _with_reversed_signals(model, packer, _ragged_batch(packer))
        # A pure reversal is an involutive permutation, so the normalized forward and reverse KLs
        # are equal even though the distributions differ. Lower one teacher slot to make the two
        # categorical partitions genuinely asymmetric and keep this a beta-wiring test.
        for traj in trajs:
            signal = traj.transitions[0].targets
            teacher_logprobs = signal.topk_logprobs.clone()  # [T,K]
            teacher_logprobs[:, 0] -= 0.2  # [T,K]
            traj.transitions[0].targets = TeacherSignal(
                topk_ids=signal.topk_ids,
                topk_logprobs=teacher_logprobs,
            )
        loss, _ = _objective(packer, jsd_beta=beta)(model, trajs)
        return float(loss.detach())

    values = [_loss(b) for b in (0.0, 0.5, 1.0)]
    assert all(v >= 0.0 for v in values), values
    assert len(set(round(v, 6) for v in values)) == 3, values
    with pytest.raises(ValueError, match=r"jsd_beta must be in \[0, 1\]"):
        _objective(packer, jsd_beta=1.5)


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_completions_take_the_zeroed_probe_path():
    """A batch whose every completion is empty still runs one forward and the full collective sequence.

    Returning early instead would leave peers that DID run a forward blocked in gradient collectives
    this rank never joins — the deadlock the probe exists to prevent — and the metric keys have to
    match too, since they are packed into a fixed-width vector for reduction."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    trajs = _ragged_batch(packer)
    for traj in trajs:
        traj.transitions[0].action = torch.zeros(0, dtype=torch.long)
        traj.transitions[0].policy_info.pop("completion_text", None)

    loss, metrics = _objective(packer)(model, trajs)

    assert float(loss.detach()) == 0.0
    assert metrics == {
        "opd_loss": 0.0,
        "opd_n": 0,
        "opd_valid_tokens": 0,
        "opd_divergence": 0.0,
        OBJECTIVE_WEIGHT_SUM: 0.0,
    }
    assert _objective(packer).adapt_metrics(metrics, "train") == {
        "train/loss_avg": 0.0,
        "train/valid_sequence_count": 0,
        "train/valid_token_count": 0,
        "train/opd_divergence": 0.0,
        "train/weight_sum": 0.0,
    }
    loss.backward()  # on the graph, so the trainer's backward() does not raise


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_batch_is_refused():
    """Zero trajectories on EVERY rank is a real error, and raising on all of them cannot desync."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model, packer = _tiny()
    with pytest.raises(ValueError, match="empty batch"):
        _objective(packer)(model, [])


# ── the advantage objective ───────────────────────────────────────────────────────────────────


def _with_group_advantages(trajs):
    """Run the reward-stage estimator, as the real pipeline does before dispatch.

    Rewards {1.0, 0.0} in ONE group give normalized advantages {+1, -1} — nonzero and unequal, so
    the surrogate has something to push on. Distinct group ids would make each trajectory a
    singleton (std 0 -> advantage 0) and every gradient below would be trivially zero."""
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    for reward, traj in zip((1.0, 0.0), trajs):
        traj.transitions[0].reward = {"final": reward}
    group_relative()(list(trajs))
    return trajs


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_objective_reduces_to_clipped_grpo_when_the_penalty_is_off():
    """With `kl_penalty_coef = 0` the penalty vanishes and OPD's advantage mode IS GRPO.

    The strongest statement available about the wiring: same base advantages, same surrogate, same
    reduction, so the PARAMETER GRADIENTS must match `make_clipped_objective`'s. A sign error on the
    penalty, a mis-broadcast advantage, or a different π_old would all break this while still
    producing a plausible-looking loss. Approximate, not exact: this objective takes
    `logit - logsumexp` where the GRPO one takes `log_softmax(...).gather(...)` — same value, a few
    ulps apart.

    The coefficient, not a matched teacher, is what turns the penalty off: a zero raw-slice value
    does not imply a zero gradient because the slice omits residual vocabulary mass."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    torch.manual_seed(0)
    model, packer = _tiny()

    def _batch():
        return _with_group_advantages(_with_exact_signals(model, packer, _ragged_batch(packer)))

    def _grads(loss):
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    cfg = GRPOConfig(epsilon_low=10.0, epsilon_high=10.0, kl_beta=0.0, temperature=SHARPEN_TEMPERATURE, group_size=2)
    model.zero_grad()
    grpo_grads = _grads(make_clipped_objective(packer, cfg)(model, _batch())[0])

    model.zero_grad()
    opd_loss, metrics = make_advantage_objective(
        packer, epsilon_low=10.0, epsilon_high=10.0, temperature=SHARPEN_TEMPERATURE, kl_penalty_coef=0.0
    )(model, _batch())
    opd_grads = _grads(opd_loss)

    # 1e-5, not 1e-6: this quantity is EXACTLY zero in real arithmetic, so what is being measured is float32 round-off through a log_softmax over a ~50k vocabulary. CI's x86 and our aarch64 round differently (x86 produced -2.3e-6 on the per-token form), so a 1e-6 bound tests the runner. The negative controls keep their thresholds, so the separation is unchanged.
    assert metrics["opd_divergence"] == pytest.approx(0.0, abs=1e-5)
    assert set(opd_grads) == set(grpo_grads)
    # RELATIVE to each tensor's own scale. The two paths differ only in float32 rounding, but
    # `SHARPEN_TEMPERATURE` divides the logits by 100 and the gradient is scaled by the same
    # factor — the largest gradients here reach ~60, where a 1e-6 relative difference is 6e-5 in
    # absolute terms. A fixed atol would either fail on those or be meaningless on the embedding
    # rows, whose gradients sit around 1e-11.
    for name, grad in opd_grads.items():
        reference = grpo_grads[name]
        scale = max(float(reference.abs().max()), 1e-30)
        assert float((grad - reference).abs().max()) <= 1e-4 * scale, name


@pytest.mark.L0
@pytest.mark.CPU
def test_penalty_scales_linearly_with_kl_penalty_coef():
    """`A_t = A_group - kl_penalty_coef · D_t`, so the mean advantage is affine in the coefficient.

    Reported through the metrics rather than the loss because the loss also passes through the
    clip; the advantage is the quantity the method actually defines."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective

    torch.manual_seed(0)
    model, packer = _tiny()

    def _metrics(coef: float) -> dict:
        trajs = _with_group_advantages(_with_reversed_signals(model, packer, _ragged_batch(packer)))
        _, metrics = make_advantage_objective(
            packer, temperature=SHARPEN_TEMPERATURE, kl_penalty_coef=coef, epsilon_low=10.0, epsilon_high=10.0
        )(model, trajs)
        return metrics

    at_zero, at_one, at_two = (_metrics(c) for c in (0.0, 1.0, 2.0))
    divergence = at_one["opd_divergence"]

    assert divergence > 1e-3, divergence  # the fixture must actually disagree
    assert at_one["opd_advantage"] == pytest.approx(at_zero["opd_advantage"] - divergence, abs=1e-5)
    assert at_two["opd_advantage"] == pytest.approx(at_zero["opd_advantage"] - 2 * divergence, abs=1e-5)


@pytest.mark.L0
@pytest.mark.CPU
def test_the_divergence_reaches_the_parameters():
    """The divergence MUST stay differentiable, because that route is the entire distillation signal.

    At the on-policy ratio the loss differentiates into
    `-(A_group - c*D_t)*∇log π(a_t) + c*∇D_t`, and only the second term teaches. The first term's
    divergence part is a state-dependent baseline — `D_t` depends on the prefix, never on the
    sampled token — so `E[D_t * ∇log π] = 0` exactly, detached or not. Cutting `∇D_t` does not
    weaken the penalty, it deletes it and leaves plain GRPO plus variance.

    A `detach_divergence` config flag used to reach this state from a recipe. It was removed rather
    than documented, because no setting of it was sound. The property is still worth pinning, so
    the detaching source the flag used to build is injected here through the `divergence` seam that
    exists for DPCA — same discrimination, no foot-gun in the shipped config surface."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective

    class _DetachingSource:
        """What `detach_divergence=True` used to build."""

        def __init__(self, inner):
            self._inner = inner

        def per_token(self, student_logits, signals, device):
            return self._inner.per_token(student_logits, signals, device).detach()

    torch.manual_seed(0)
    model, packer = _tiny()

    def _grads(source):
        trajs = _with_group_advantages(_with_reversed_signals(model, packer, _ragged_batch(packer)))
        model.zero_grad()
        loss, _ = make_advantage_objective(
            packer,
            temperature=SHARPEN_TEMPERATURE,
            kl_penalty_coef=1.0,
            epsilon_low=10.0,
            epsilon_high=10.0,
            divergence=source,
        )(model, trajs)
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    live = TopKSliceDivergence(1.0)
    attached, detached = _grads(live), _grads(_DetachingSource(live))

    assert set(attached) == set(detached)
    assert any(not torch.allclose(attached[n], detached[n], atol=1e-7) for n in attached)


@pytest.mark.L0
@pytest.mark.CPU
def test_the_advantage_path_casts_to_fp32_exactly_once():
    """The divergence source must be handed the fp32 logits the ratio already built, not the raw ones.

    `_support_logprobs` casts to fp32 itself, and `.float()` on an fp32 tensor returns the SAME
    object — so passing the already-cast tensor costs nothing, while passing the model's bf16 output
    makes a SECOND `[n_tok, V]` fp32 copy that autograd holds through the backward. That is the
    objective's largest allocation (~9.3 GiB at 16k scored tokens on a 151936 vocabulary), so the
    difference is the whole memory budget, doubled, against the figure the docstring and the recipe
    both quote. Nothing raises when it regresses; the run just OOMs one batch size early.

    Checking the dtype the source RECEIVES is the honest form of the check: it is exactly the
    condition under which the second cast is a no-op."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective

    torch.manual_seed(0)
    model, packer = _tiny()
    trajs = _with_exact_signals(model, packer, _ragged_batch(packer))
    model = model.to(torch.bfloat16)  # a real training dtype: a stray `.float()` then allocates
    for traj in trajs:
        traj.transitions[0].advantage = 1.0

    seen: list[torch.dtype] = []

    class _Spy:
        def per_token(self, student_logits, signals, device):
            seen.append(student_logits.dtype)
            return TopKSliceDivergence(beta=1.0).per_token(student_logits, signals, device)

    objective = make_advantage_objective(packer, temperature=SHARPEN_TEMPERATURE, divergence=_Spy())
    objective(model, trajs)

    assert seen == [torch.float32], f"divergence source received {seen}, so the caller cast twice"


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_objective_requires_the_reward_stage_estimate():
    """It READS `Transition.advantage`; a missing one means the reward worker got no annotators."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective

    torch.manual_seed(0)
    model, packer = _tiny()
    trajs = _with_exact_signals(model, packer, _ragged_batch(packer))  # signals, but no advantages

    with pytest.raises(ValueError, match="advantage=None"):
        make_advantage_objective(packer, temperature=SHARPEN_TEMPERATURE)(model, trajs)


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_objective_refuses_knobs_it_does_not_implement():
    """Loud beats approximated. `kl_beta` needs reference logprobs this objective never computes,
    and the reference's `kl_discount_factor` rebuilds the advantages from a Python list, which
    silently detaches them — importing that quietly would change which gradient routes exist."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective

    _, packer = _tiny()
    with pytest.raises(ValueError, match="kl_beta"):
        make_advantage_objective(packer, kl_beta=0.1)
    for invalid_kl_beta in (-0.1, float("nan"), True):
        with pytest.raises(ValueError, match="kl_beta=0 exactly"):
            make_advantage_objective(packer, kl_beta=invalid_kl_beta)
    with pytest.raises(ValueError, match="kl_penalty_coef"):
        make_advantage_objective(packer, kl_penalty_coef=-0.1)
    with pytest.raises(NotImplementedError, match="kl_discount_factor"):
        make_advantage_objective(packer, kl_discount_factor=0.9)
    with pytest.raises(ValueError, match="loss_type must be one of"):
        make_advantage_objective(packer, loss_type="token-sum")


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_objective_empty_shard_metrics_match_the_full_path():
    """`_reduce_metrics` packs metrics into a fixed-width vector, so both paths must emit the same
    keys — a rank that returned three where its peers returned five would desync the reduction."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective

    torch.manual_seed(0)
    model, packer = _tiny()
    objective = make_advantage_objective(packer, temperature=SHARPEN_TEMPERATURE, epsilon_low=10.0, epsilon_high=10.0)
    assert objective.metric_weight_map("train") == {
        "train/loss_avg": "train/weight_sum",
        "train/opd_divergence": "train/valid_token_count",
        "train/opd_advantage": "train/valid_token_count",
        "train/opd_teacher_credit": "train/valid_token_count",
    }

    full = _with_group_advantages(_with_exact_signals(model, packer, _ragged_batch(packer)))
    _, full_metrics = objective(model, full)
    assert set(objective.adapt_metrics(full_metrics, "train")) == {
        "train/loss_avg",
        "train/valid_sequence_count",
        "train/valid_token_count",
        "train/opd_divergence",
        "train/opd_advantage",
        "train/opd_teacher_credit",
        "train/weight_sum",
    }

    empty = _with_group_advantages(_ragged_batch(packer))
    for traj in empty:
        traj.transitions[0].action = torch.zeros(0, dtype=torch.long)
        traj.transitions[0].policy_info.pop("completion_text", None)
    loss, empty_metrics = objective(model, empty)

    assert set(empty_metrics) == set(full_metrics)
    assert empty_metrics == dict.fromkeys(full_metrics, 0.0)
    assert float(loss.detach()) == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_a_teacher_id_outside_the_student_vocabulary_raises():
    """A wider teacher can return an id the student has no logit column for. Name it, don't gather it.

    The shipped recipe pairs Qwen2.5-1.5B (`vocab_size` 151936) with Qwen2.5-7B (152064). The
    tokenizers are identical and stop at id 151664, so the 128-id gap is padding rows that a healthy
    teacher should never rank — but nothing enforced that, and unchecked the id reaches `gather`,
    where CUDA raises an ASYNCHRONOUS device-side assert: the context is poisoned, the message names
    nothing, and the peer ranks hang in a collective until the NCCL watchdog fires. This guard
    converges the failure before any rank enters that collective.
    """
    from cosmos_framework.posttrain.algorithm.opd import _concat_support

    vocab = 151936
    outside = TeacherSignal(
        topk_ids=torch.tensor([[0, 5, vocab]], dtype=torch.int32),
        topk_logprobs=torch.zeros(1, 3, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match=r"outside the student's vocabulary"):
        _concat_support([outside], torch.device("cpu"), vocab_size=vocab)

    # The last in-range id is fine, so the bound is `>=` and not an off-by-one that rejects it.
    inside = TeacherSignal(
        topk_ids=torch.tensor([[0, 5, vocab - 1]], dtype=torch.int32),
        topk_logprobs=torch.zeros(1, 3, dtype=torch.float32),
    )
    ids, _ = _concat_support([inside], torch.device("cpu"), vocab_size=vocab)
    assert ids.dtype == torch.long and int(ids.max()) == vocab - 1
