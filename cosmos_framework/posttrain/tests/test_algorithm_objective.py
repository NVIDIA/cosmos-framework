# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the reusable objective reduction seam.

Objectives declare a `Reduction`. GRPO declares TOKEN_SUM: it sums per-token `advantage · logπ` and
divides by the DP-global completion-token count (all-reduced inside the objective), so longer
completions carry proportional weight and the loss is normalized by total tokens — pinned here by the
batch-replication invariance test. DPO declares MEAN (byte-identical); only its declaration is asserted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from cosmos_framework.posttrain.algorithm import dpo, grpo
from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    PerSampleReducer,
    Reduction,
    ValidationBatchMetadata,
    backward,
    read_validation_batch_metadata,
)
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


class _TinyLM(nn.Module):
    """Minimal causal-LM surface for the GRPO objective: `net(ids).logits` over a small vocab. No
    transformers / HF download — the objective only needs per-token logits."""

    def __init__(self, vocab: int = 16, d: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.head(self.emb(ids)))


def _net() -> _TinyLM:
    torch.manual_seed(0)  # fixed weights ⇒ deterministic per-token logp across calls
    return _TinyLM()


def _traj(group_id: str, reward: float, prompt_ids: list[int], comp_ids: list[int]) -> Trajectory:
    tr = Transition(
        obs=ObsBundle(),
        action=torch.tensor(comp_ids),
        policy_info={"prompt_ids": torch.tensor(prompt_ids)},
        reward={"final": reward},
        done=True,
    )
    return Trajectory(transitions=[tr], id="t", group_id=group_id, status=Status.COMPLETED)


def _with_advantages(trajs, unbiased: bool = False):
    """Apply the reward-stage estimator, as the real pipeline does before dispatch.

    The objective no longer computes the group-relative advantage; it reads `transition.advantage`.
    A test that hands it raw trajectories is skipping the reward stage, so it must supply what that
    stage would have written.
    """
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    group_relative(unbiased=unbiased)(list(trajs))
    return trajs


def _batch() -> list[Trajectory]:
    """Two GRPO groups (g0, g1), varied completion lengths (2/4/3/1 = 10 tokens) so token_sum weights
    them unequally.

    Advantages are applied HERE, by the same estimator the reward stage runs, because the objective
    reads `transition.advantage` rather than computing it. Groups no longer need to stay atomic
    downstream — each sample carries the value normalized over its whole group.
    """
    return _with_advantages(
        [
            _traj("g0", 1.0, [1, 2, 3], [4, 5]),
            _traj("g0", 0.0, [1, 2, 3], [6, 7, 8, 9]),
            _traj("g1", 1.0, [2, 3], [10, 11, 12]),
            _traj("g1", 0.0, [2, 3], [13]),
        ]
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_objectives_declare_reduction():
    """GRPO declares TOKEN_SUM; DPO declares MEAN."""
    grpo_objective = grpo.make_objective()
    assert grpo_objective.reduction is Reduction.TOKEN_SUM
    # Clip diagnostics are weighted by the unit they count over, so each rate is batch-global: per-token
    # (`clip_frac`, `ratio_absdev`) by valid tokens, per-response (`seq_*`) by valid sequences, and the
    # train-vs-rollout gap by the tokens that carried a recorded rollout logprob.
    assert grpo_objective.metric_weight_map("train") == {
        "train/loss_avg": "train/weight_sum",
        "train/clip_frac": "train/valid_token_count",
        "train/ratio_absdev": "train/valid_token_count",
        "train/seq_clip_frac": "train/valid_sequence_count",
        "train/seq_log_ratio_abs": "train/valid_sequence_count",
        "train/logprob_gap_mean": "train/logprob_gap_token_count",
        "train/mean_rank_logprob_gap_p99": "train/logprob_gap_token_count",
        "train/logprob_gap_gt0p5_frac": "train/logprob_gap_token_count",
        "train/kl_mean": "train/valid_token_count",
        "train/rollout_truncated_frac": "train/response_count",
        "train/response_len_mean": "train/response_count",
        # Score-centering diagnostics (rl.score_centering) by the tokens the correction covered.
        "train/sc_head_mass_q": "train/sc_token_count",
        "train/sc_head_mass_p": "train/sc_token_count",
        "train/sc_rho_mean": "train/sc_token_count",
        "train/sc_kl_head": "train/sc_token_count",
        "train/sc_correction_abs": "train/sc_token_count",
        "train/sc_token_weight": "train/sc_token_count",
        "train/sc_out_of_head_frac": "train/sc_token_count",
    }
    assert dpo.make_objective(beta=1.0).reduction is Reduction.MEAN


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_objective_utilities_are_strict_and_masked() -> None:
    adapter = MetricAdapter(
        names={"algo_loss": "loss_avg"},
        internal_prefix="algo_",
        phases=frozenset({"train", "val"}),
        ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
    )
    assert adapter(
        {"algo_loss": 1.5, OBJECTIVE_WEIGHT_SUM: 2.0, "optim/learning_rate": 1e-4},
        "val",
    ) == {"val/loss_avg": 1.5, "optim/learning_rate": 1e-4}

    weighted_adapter = MetricAdapter(
        names={"algo_loss": "loss_avg", OBJECTIVE_WEIGHT_SUM: "weight_sum"},
        internal_prefix="algo_",
        phases=frozenset({"train", "val"}),
        weighted_by={"algo_loss": OBJECTIVE_WEIGHT_SUM},
    )
    expected_weights = {"val/loss_avg": "val/weight_sum"}
    assert weighted_adapter.public_weight_map("val") == expected_weights
    declared_objective = Objective(
        fn=grpo.make_objective().fn,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=weighted_adapter,
    )
    assert declared_objective.metric_weight_map("val") == expected_weights
    undeclared_objective = Objective(
        fn=grpo.make_objective().fn,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=lambda metrics, phase: adapter(metrics, phase),
    )
    assert undeclared_objective.metric_weight_map("train") == {}
    for weighted_by in (
        {"algo_unmapped": OBJECTIVE_WEIGHT_SUM},
        {"algo_loss": "algo_unmapped_weight"},
    ):
        with pytest.raises(ValueError, match="weighted_by must reference mapped metrics"):
            MetricAdapter(
                names={"algo_loss": "loss_avg", OBJECTIVE_WEIGHT_SUM: "weight_sum"},
                internal_prefix="algo_",
                phases=frozenset({"train"}),
                weighted_by=weighted_by,
            )
    with pytest.raises(ValueError, match="unmapped internal objective metric"):
        adapter({"algo_new": 1.0}, "train")
    with pytest.raises(ValueError, match="support phases"):
        MetricAdapter(
            names={"algo_loss": "loss_avg"},
            internal_prefix="algo_",
            phases=frozenset({"train"}),
        )({"algo_loss": 1.0}, "val")

    trajectories = [
        _traj("g0", 1.0, [1], [2]),
        _traj("g0", 0.0, [1], [3]),
    ]
    for index, trajectory in enumerate(trajectories):
        trajectory.transitions[0].algo_extra.update(
            validation_weight=float(index == 0),
            validation_noise_seed=index,
        )
    assert read_validation_batch_metadata(trajectories) == ValidationBatchMetadata(
        sample_weights=[1.0, 0.0],
        sample_seeds=[0, 1],
    )

    invalid_metadata = (
        ([1.0, None], [0, 1], "validation_weight must be present"),
        ([True, 0.0], [0, 1], "validation_weight values must be non-bool real"),
        ([float("nan"), 0.0], [0, 1], "validation_weight values must be finite"),
        ([1.0, 0.0], [True, 1], "validation_noise_seed values must be non-bool integers"),
        ([1.0, 0.0], [0.5, 1], "validation_noise_seed values must be non-bool integers"),
        ([1.0, 0.0], [2**64, 1], "validation_noise_seed values must be unsigned 64-bit"),
    )
    for weights, seeds, message in invalid_metadata:
        invalid = [_traj("g0", 1.0, [1], [2]), _traj("g0", 0.0, [1], [3])]
        for trajectory, weight, seed in zip(invalid, weights, seeds, strict=True):
            if weight is not None:
                trajectory.transitions[0].algo_extra["validation_weight"] = weight
            trajectory.transitions[0].algo_extra["validation_noise_seed"] = seed
        with pytest.raises((TypeError, ValueError), match=message):
            read_validation_batch_metadata(invalid)

    values = torch.tensor([2.0, 8.0], requires_grad=True)  # [B]
    reducer = PerSampleReducer.from_weights(values, [0.5, 0.0], context="test objective")
    masked_mean = reducer.mean(values)  # []
    assert masked_mean.item() == 2.0
    assert reducer.weight_sum is not None and reducer.weight_sum.item() == 0.5
    masked_mean.backward()
    assert values.grad is not None and values.grad.tolist() == [1.0, 0.0]

    zero_values = torch.tensor([2.0, 8.0], requires_grad=True)  # [B]
    zero_reducer = PerSampleReducer.from_weights(zero_values, [0.0, 0.0], context="test objective")
    graph_zero = zero_reducer.mean(zero_values)  # []
    assert graph_zero.item() == 0.0 and graph_zero.requires_grad
    graph_zero.backward()
    assert zero_values.grad is not None and zero_values.grad.tolist() == [0.0, 0.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_token_sum_metrics_report_valid_tokens():
    """The objective reports its summed completion-token count and sequence count in metrics."""
    objective = grpo.make_objective()
    _, metrics = objective(_net(), _batch())
    assert metrics["grpo_valid_tokens"] == 10  # 2+4+3+1
    assert metrics["grpo_n"] == 4
    assert metrics[OBJECTIVE_WEIGHT_SUM] == pytest.approx(10.0)
    public = objective.adapt_metrics(metrics, "train")
    assert public["train/weight_sum"] == pytest.approx(10.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_token_sum_normalized_by_total_tokens():
    """token_sum divides by the total completion-token count, so replicating the batch leaves the loss
    unchanged (2·numerator / 2·tokens) — the loss is a per-token mean, not a per-sequence or per-batch
    sum. Group advantages are unchanged by replication (mean/std within each group_id are identical)."""
    net = _net()
    obj = grpo.make_objective()
    loss_1x, _ = obj(net, _batch())
    loss_2x, _ = obj(net, _batch() + _batch())  # same content, doubled
    assert loss_1x.item() == pytest.approx(loss_2x.item(), rel=1e-5, abs=1e-6)


@pytest.mark.L0
@pytest.mark.CPU
def test_token_sum_backprops_through_completion_logits():
    """The token_sum loss is differentiable end-to-end (the numerator is a live graph, not a detached
    scalar) — a backward populates net gradients."""
    net = _net()
    loss, _ = grpo.make_objective()(net, _batch())
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in net.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_token_empty_rank_still_emits_gradients_on_every_parameter():
    """An all-empty-completion rank must still walk the backward graph: without grads on every param it
    skips the DP gradient reduce and the other ranks block until the PG timeout."""
    net = _net()
    empty = _with_advantages([_traj("g0", 1.0, [1, 2, 3], []), _traj("g0", 0.0, [2, 3], [])])
    loss, metrics = grpo.make_objective()(net, empty)

    assert loss.item() == 0.0
    assert metrics["grpo_valid_tokens"] == 0 and metrics["grpo_n"] == 0
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())  # the anti-strand property
    assert all(bool(torch.all(p.grad == 0)) for p in net.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_rejects_empty_batch():
    """No trajectories ⇒ no forward is possible, so there is no graph to keep this rank in the
    collective: fail loudly rather than strand it."""
    with pytest.raises(ValueError, match="empty batch"):
        grpo.make_objective()(_net(), [])


@pytest.mark.L0
@pytest.mark.CPU
def test_token_sum_dp_mean_all_reduce_scales_denominator(monkeypatch):
    """The MEAN (not SUM) all-reduce of token counts is the seam's only nontrivial math: under DP the
    objective divides by the DP-mean per-rank count, so it composes with FSDP's mean-grad without a
    world_size factor. Emulate 2 ranks — this rank's 10 tokens + a fake other rank's 6 — so the
    denominator becomes (10+6)/2 = 8, i.e. the loss scales by 10/8 vs the single-rank denominator."""
    import torch.distributed as dist

    net = _net()
    obj = grpo.make_objective()
    loss_1rank, _ = obj(net, _batch())  # dist uninitialized ⇒ denominator = 10 local tokens

    ops: list = []

    def fake_all_reduce(tensor, op=None):
        # Three collectives, distinguishable by what they carry. `op` alone is not enough: the
        # failure flag and the forward count are BOTH ReduceOp.MAX, and answering the flag with a
        # count would report a phantom failure on every batch.
        if tensor.dtype is torch.int32:  # raise_if_any_rank_failed's 0-d flag
            ops.append("rank_verdict")  # leave it 0 == no rank failed
            return
        if op is dist.ReduceOp.MAX:
            # The forward-count agreement (`rank_max_count`). Emulate a peer holding the same
            # number of trajectories, so nothing pads and the token math below is untouched.
            ops.append("forward_count")
            return
        ops.append("token_count")
        tensor.copy_((tensor + 6.0) / 2.0)  # AVG this rank's count with a fake other-rank count of 6

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    # A faked process group must also fake `get_backend`: `raise_if_any_rank_failed` asks for a
    # backend-compatible device via `collective_device()`, and the real call needs a real group.
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")

    loss_2rank, _ = obj(net, _batch())  # same net + content; denominator now (10+6)/2 = 8

    # Exactly these collectives, in this ORDER. Each is unconditional and single-shot per objective
    # call -- a data-dependent count or order desyncs the process group, which is a hang rather than
    # a wrong answer, so the sequence is pinned literally rather than counted.
    #   rank_verdict (1): converge the ADVANTAGE check, so one bad rank raises everywhere rather
    #                      than aborting alone while its peers block in the collectives below.
    #   forward_count:     agree how many forwards to issue, BEFORE issuing them -- the padding it
    #                      authorizes must precede the token count, or a padding rank issues
    #                      (AVG, forward...) while its peers issue (forward..., AVG).
    #   rank_verdict (2): converge SHAPING errors, which is why all shaping is hoisted above the
    #                      first forward. Same reason as the first: rank-local data errors are hangs.
    #   token_count:       the DP-global denominator.
    assert ops == ["rank_verdict", "forward_count", "rank_verdict", "token_count"], f"collectives were {ops}"
    assert loss_2rank.item() == pytest.approx(loss_1rank.item() * 10.0 / 8.0, rel=1e-5, abs=1e-6)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "reduction,grad_accum,denom,expected_scale",
    [
        (Reduction.MEAN, 1, None, 1.0),
        (Reduction.MEAN, 4, None, 0.25),
        (Reduction.TOKEN_SUM, 1, None, 1.0),  # today's path, byte-for-byte
        (Reduction.TOKEN_SUM, 4, 8.0, 8.0),  # recover the chunk SUM
    ],
)
def test_backward_scaling_contract(reduction, grad_accum, denom, expected_scale):
    """`backward`'s scaling contract per reduction: MEAN divides by grad_accum; TOKEN_SUM at
    grad_accum=1 backprops as-is; TOKEN_SUM under accumulation with a `denom` re-multiplies to
    recover the chunk SUM before backprop."""
    x = torch.tensor([2.0], requires_grad=True)
    backward((x * 3.0).sum(), reduction, grad_accum, denom)
    assert x.grad.item() == pytest.approx(3.0 * expected_scale)


@pytest.mark.L0
@pytest.mark.CPU
def test_token_sum_accumulation_without_a_denominator_is_refused():
    """The guard narrows rather than disappears: without a denom this IS a mean-of-means."""
    x = torch.tensor([2.0], requires_grad=True)
    with pytest.raises(NotImplementedError, match="objective weight sum"):
        backward((x * 3.0).sum(), Reduction.TOKEN_SUM, 4, None)


@pytest.mark.L0
@pytest.mark.CPU
def test_metric_adapter_max_reduced_declares_public_max_keys() -> None:
    """A per-shard maximum composes by MAX, never by (weighted) mean: the adapter declares it, the
    objective exposes it per phase, and the declaration must name a mapped, unweighted metric."""
    adapter = MetricAdapter(
        names={"algo_loss": "loss_avg", "algo_peak": "kl_max", OBJECTIVE_WEIGHT_SUM: "weight_sum"},
        internal_prefix="algo_",
        phases=frozenset({"train", "val"}),
        weighted_by={"algo_loss": OBJECTIVE_WEIGHT_SUM},
        max_reduced=frozenset({"algo_peak"}),
    )
    assert adapter.public_max_keys("train") == frozenset({"train/kl_max"})
    assert adapter.public_max_keys("val") == frozenset({"val/kl_max"})
    assert adapter.public_weight_map("train") == {"train/loss_avg": "train/weight_sum"}
    declared = Objective(fn=grpo.make_objective().fn, reduction=Reduction.TOKEN_SUM, metric_adapter=adapter)
    assert declared.metric_max_keys("train") == frozenset({"train/kl_max"})
    undeclared = Objective(
        fn=grpo.make_objective().fn,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=lambda metrics, phase: adapter(metrics, phase),
    )
    assert undeclared.metric_max_keys("train") == frozenset()
    assert Objective(fn=grpo.make_objective().fn, reduction=Reduction.TOKEN_SUM).metric_max_keys("train") == frozenset()
    # Default: no maxima (DPO's adapter is untouched); GRPO declares exactly its per-token KL maximum,
    # its longest response and score centering's largest tail-mass ratio, on the one adapter both GRPO
    # objectives share -- the single declaration point for maxima.
    assert dpo.make_objective(beta=1.0).metric_adapter.max_reduced == frozenset()
    assert grpo.make_objective().metric_adapter.max_reduced == frozenset(
        {"grpo_kl_max", "grpo_response_len_max", "grpo_sc_rho_max"}
    )
    assert grpo.make_objective().metric_max_keys("train") == frozenset(
        {"train/kl_max", "train/response_len_max", "train/sc_rho_max"}
    )
    with pytest.raises(ValueError, match="max_reduced must reference mapped metrics"):
        MetricAdapter(
            names={"algo_loss": "loss_avg"},
            internal_prefix="algo_",
            phases=frozenset({"train"}),
            max_reduced=frozenset({"algo_unmapped"}),
        )
    with pytest.raises(ValueError, match="cannot both max-reduce and weight"):
        MetricAdapter(
            names={"algo_peak": "kl_max", OBJECTIVE_WEIGHT_SUM: "weight_sum"},
            internal_prefix="algo_",
            phases=frozenset({"train"}),
            weighted_by={"algo_peak": OBJECTIVE_WEIGHT_SUM},
            max_reduced=frozenset({"algo_peak"}),
        )
    with pytest.raises(ValueError, match="support phases"):
        MetricAdapter(
            names={"algo_peak": "kl_max"},
            internal_prefix="algo_",
            phases=frozenset({"train"}),
            max_reduced=frozenset({"algo_peak"}),
        ).public_max_keys("val")
    with pytest.raises(ValueError, match="phase must be"):
        declared.metric_max_keys("test")
