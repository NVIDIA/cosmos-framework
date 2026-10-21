# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GSPO: `importance_sampling_level="sequence"` on the clipped policy-gradient surrogate, pinned
against an inline transcription of the reference arithmetic (GSPO eq. 7/14, verl, cosmos-rl)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective
from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import (
    clipped_policy_gradient_loss,
    reduce_policy_gradient_loss,
)

# One group with rewards {1, 0} gets opposite-sign advantages (about +-1); distinct lengths so collation pads.
GROUP_SPECS = (("3+4=", " 7", 1.0), ("100+200=", " 300 400", 0.0))


def _reference_gspo_loss(cur, old, adv, counts, eps_low, eps_high):  # cur/old/adv: [N], counts: [B], returns [N]
    """Reference GSPO per-token loss: per-response sum/len of the log-ratio (an empty response
    contributes nothing) broadcast with ``expand`` so autograd flows THROUGH the mean, clamp(max=10),
    the epsilon clip with a negative epsilon disabling its side, ``-min(loss1, loss2)``, no dual clip."""
    delta = cur - old  # [N]
    spans, start = [], 0
    for length in counts.tolist():
        if length > 0:
            spans.append((delta[start : start + length].sum() / length).expand(length))  # [L_i]
        start += length
    ratio = torch.exp(torch.clamp(torch.cat(spans), max=10.0))  # [N]
    min_clipped = 1.0 - eps_low if eps_low >= 0.0 else None
    max_clipped = 1.0 + eps_high if eps_high >= 0.0 else None
    if min_clipped is None and max_clipped is None:
        clipped = ratio  # [N]
    else:
        clipped = torch.clamp(ratio, min=min_clipped, max=max_clipped)  # [N]
    return -torch.min(ratio * adv, clipped * adv)  # [N]


def _sequence_loss(cur, old, adv, counts, eps):  # cur/old/adv: [N], counts: [B], returns [N]
    return clipped_policy_gradient_loss(
        cur, old, adv, *eps, lower_bound_ratio=None, importance_sampling_level="sequence", token_counts=counts
    )


def _sum_grad(loss_fn, cur, *args):  # cur: [N], returns [N]
    """d(loss_fn(cur, *args).sum()) / d cur on a fresh leaf."""
    leaf = cur.detach().clone().requires_grad_(True)  # [N]
    loss_fn(leaf, *args).sum().backward()
    return leaf.grad  # [N]


def _objective_cfg(level: str, **overrides) -> SimpleNamespace:
    """Hand-built cfg with the attributes `make_clipped_objective` reads. Sequence level must carry
    lower_bound_ratio=0.0; at ratio 1 the token-level dual clip is inert, so the cfgs differ only in level."""
    cfg = SimpleNamespace(
        loss_type="seq-mean-token-mean",
        temperature=1.0,
        epsilon_low=0.2,
        epsilon_high=0.2,
        kl_beta=0.0,
        lower_bound_ratio=0.0 if level == "sequence" else 3.0,
        use_rollout_logprobs_for_loss=False,
        logprob_dtype="float32",
        importance_sampling_level=level,
    )
    cfg.__dict__.update(overrides)
    return cfg


def _tiny():
    """Tiny random GPT2 (eval: no dropout) + gpt2 packer; posttrain tests never import each other."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.data.packer import init_data_packer

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()
    return model, init_data_packer("gpt2")


def _group_trajectories(packer):
    """One GROUP_SPECS group with advantages written by the reward-stage estimator. No
    rollout_logprobs: the objective scores on-policy (pi_old = cur)."""
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    trajectories = []
    for index, (prompt, completion_text, reward) in enumerate(GROUP_SPECS):
        completion = torch.tensor(packer.tokenizer(completion_text, add_special_tokens=False).input_ids)  # [L_i]
        transition = Transition(
            obs=ObsBundle(representations={"prompt": prompt}, id=f"p{index}"),
            action=completion,
            policy_info={"completion_text": completion_text},
            reward={"final": reward},
        )
        trajectories.append(
            Trajectory(transitions=[transition], id=f"g0_k{index}", group_id="g0", status=Status.COMPLETED)
        )
    group_relative()(trajectories)
    return trajectories


@pytest.mark.L0
@pytest.mark.CPU
def test_sequence_ratio_matches_reference_formula():
    """The stop-gradient sequence ratio equals the reference expand-through-the-mean form in per-token
    loss, reduction and gradient; the one-sided clamp alone bounds a far-off-policy negative-advantage
    response (loss e^10 * |A|, zero gradient); no dual clip is applied."""
    # counts [3,1,4] with mixed-sign deltas; response means 0.0833 / 12 / -0.1 on advantages +/-/+.
    counts = torch.tensor([3, 1, 4])  # [B]
    old = torch.tensor([-1.0, -2.0, -0.5, -3.0, -1.5, -0.8, -2.5, -1.2])  # [N]
    cur = old + torch.tensor([0.3, -0.1, 0.05, 12.0, -0.4, 0.1, -0.2, 0.1])  # [N]
    adv = torch.repeat_interleave(torch.tensor([1.5, -0.7, 1.0]), counts)  # [N]
    for eps in ((0.2, 0.2), (3e-4, 4e-4)):  # nothing clips at 0.2; response 0 clips at the paper's eps
        got = _sequence_loss(cur, old, adv, counts, eps)  # [N]
        want = _reference_gspo_loss(cur, old, adv, counts, *eps)  # [N]
        assert torch.allclose(got, want, rtol=1e-6, atol=0.0)
        assert got[3].item() == pytest.approx(math.exp(10.0) * 0.7, rel=1e-6)
        for loss_type in ("seq-mean-token-mean", "token-mean"):
            num_got, units_got = reduce_policy_gradient_loss(got, counts, loss_type)  # [], units
            num_want, units_want = reduce_policy_gradient_loss(want, counts, loss_type)  # [], units
            assert units_got == units_want and torch.allclose(num_got, num_want, rtol=1e-6, atol=0.0), loss_type
        grad_sg = _sum_grad(_sequence_loss, cur, old, adv, counts, eps)  # [N]
        grad_expand = _sum_grad(_reference_gspo_loss, cur, old, adv, counts, *eps)  # [N]
        assert torch.allclose(grad_sg, grad_expand, rtol=1e-6, atol=1e-7)
        assert grad_sg[3].item() == 0.0 and grad_sg[4:].abs().min() > 0.0  # clamped: exactly 0; response 2 live
    # No dual clip: adv = -5 at a response mean log-ratio of 3 gives 5 * e^3 (a 3.0 dual clip would give 15).
    old = torch.tensor([-1.0, -2.0])  # [N]
    loss = _sequence_loss(
        old + torch.tensor([4.0, 2.0]), old, torch.full((2,), -5.0), torch.tensor([2]), (0.2, 0.2)
    )  # [N]
    assert loss[0].item() == pytest.approx(100.4277, rel=1e-4)


@pytest.mark.L0
@pytest.mark.CPU
def test_sequence_and_token_level_agree_on_policy():
    """On-policy (no rollout_logprobs, so pi_old = cur.detach()) every log-ratio is 0 at either level,
    so the objective's loss, every parameter gradient and the metric key set coincide, and both
    sequence diagnostics read 0.0."""
    torch.manual_seed(0)
    model, packer = _tiny()
    trajectories = _group_trajectories(packer)

    def run(level):
        model.zero_grad()
        loss, metrics = make_clipped_objective(packer, _objective_cfg(level)).fn(model, trajectories)  # []
        loss.backward()
        return loss.detach(), metrics, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    loss_tok, metrics_tok, grads_tok = run("token")
    loss_seq, metrics_seq, grads_seq = run("sequence")
    assert torch.allclose(loss_seq, loss_tok, atol=1e-6)
    assert grads_tok and grads_seq.keys() == grads_tok.keys()
    assert any(g.abs().max() > 1e-6 for g in grads_tok.values()), "advantages +1/-1 must move parameters"
    for name, grad in grads_tok.items():
        assert torch.allclose(grads_seq[name], grad, atol=1e-6), name
    assert metrics_seq.keys() == metrics_tok.keys()
    for metrics in (metrics_tok, metrics_seq):
        assert metrics["grpo_seq_clip_frac"] == 0.0 and metrics["grpo_seq_log_ratio_abs"] == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_sequence_level_rejections():
    """Sequence level refuses a dual-clip bound (primitive and factory), missing token_counts, an
    unknown level, and per-token (tensor) advantages; the same batch runs at token level."""
    counts = torch.tensor([2])  # [B]
    cur, old, adv = torch.tensor([-1.0, -2.0]), torch.tensor([-1.2, -1.9]), torch.ones(2)  # [N] each
    with pytest.raises(ValueError, match="takes no dual-clip bound"):
        clipped_policy_gradient_loss(
            cur, old, adv, 0.2, 0.2, lower_bound_ratio=3.0, importance_sampling_level="sequence", token_counts=counts
        )
    with pytest.raises(ValueError, match="needs token_counts"):
        clipped_policy_gradient_loss(
            cur, old, adv, 0.2, 0.2, lower_bound_ratio=None, importance_sampling_level="sequence"
        )
    with pytest.raises(ValueError, match="importance_sampling_level must be one of"):
        clipped_policy_gradient_loss(cur, old, adv, 0.2, 0.2, importance_sampling_level="response", token_counts=counts)
    with pytest.raises(ValueError, match="applies no dual clip"):
        make_clipped_objective(packer=None, algo_cfg=_objective_cfg("sequence", lower_bound_ratio=3.0))

    torch.manual_seed(0)
    model, packer = _tiny()
    trajectories = _group_trajectories(packer)
    first = trajectories[0].transitions[0]
    first.advantage = torch.full((first.action.numel(),), float(first.advantage))  # [L_0] one value per scored token
    with pytest.raises(ValueError, match="per-token advantages are not supported"):
        make_clipped_objective(packer, _objective_cfg("sequence")).fn(model, trajectories)
    loss, _ = make_clipped_objective(packer, _objective_cfg("token")).fn(model, trajectories)  # []
    assert torch.isfinite(loss)
