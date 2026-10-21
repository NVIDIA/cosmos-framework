# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""@L0 @CPU: `algorithm/grpo.py` — the `legacy_pg` GRPO objective and its shaping seam.

The ESTIMATOR that writes `transition.advantage` is tested in `test_algorithm_advantage.py`; every
test here starts from advantages already baked in, exactly as the objective sees them after the
reward stage. Keeping the split mirrors the code: the objective reads a per-token advantage and
applies the surrogate, and does not care which estimator produced it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


def _with_advantages(trajs, unbiased: bool = False):
    """Apply the reward-stage estimator, as the real pipeline does before dispatch.

    The objective reads `transition.advantage` instead of computing it, so a test that hands it raw
    trajectories is skipping the reward stage and must supply what that stage would have written.
    """
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    group_relative(unbiased=unbiased)(list(trajs))
    return trajs


def _inline_reference_objective(model, trajectories):
    """Frozen copy of the PRE-PACKER shaping — reads `policy_info['prompt_ids']` and slices the
    completion by prompt length — over the CURRENT token_sum math.

    The packer path must reproduce this bit-for-bit: same ids, same completion positions, same loss.
    Do NOT "simplify" this to call the packer; that would make the parity test tautological. Single
    process here, so the DP all-reduce in the real objective is the identity on `n_tokens`."""
    import torch

    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative as _grel

    _grel()(trajectories)
    net = getattr(model, "net", model)
    device = next(net.parameters()).device
    numerator = torch.zeros((), device=device)
    n_tokens = 0
    for traj in trajectories:
        tr = traj.transitions[0]
        prompt_ids = tr.policy_info["prompt_ids"].to(device)
        comp_ids = tr.action.to(device)
        if comp_ids.numel() == 0:
            continue
        ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0)
        logits = net(ids).logits[0]
        logp_all = torch.log_softmax(logits[:-1], dim=-1)
        tok_logp = logp_all.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        comp_logp = tok_logp[prompt_ids.shape[0] - 1 :]
        numerator = numerator + float(tr.advantage) * comp_logp.sum()
        n_tokens += comp_logp.numel()
    return -numerator / max(float(n_tokens), 1.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_packer_objective_matches_inline_bit_for_bit():
    """The packer path must reproduce the pre-packer shaping EXACTLY (bit-for-bit).

    Trajectories store `prompt_ids = tokenizer(text, add_special_tokens=False)` — precisely what
    `HFRollout` stored for GPT2 — so the packer (which re-tokenizes `obs["text"]` the same way) and
    the inline reference see identical `input_ids`; `mask[:-1]` must select the same completion
    positions as the inline `prompt_ids.shape[0] - 1 :` slice. `torch.equal`, not approx."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.data.packer import init_data_packer

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64))
    model.eval()
    packer = init_data_packer("gpt2")
    tok = packer.tokenizer

    def _mk():
        trajs = []
        for pi, text in enumerate(["3+4=", "5+2="]):
            pid = torch.tensor(tok(text, add_special_tokens=False).input_ids, dtype=torch.long)
            for k in range(2):
                comp = torch.tensor(tok(f" {k}", add_special_tokens=False).input_ids, dtype=torch.long)
                tr = Transition(
                    obs=ObsBundle(representations={"prompt": text}, id=f"p{pi}"),
                    action=comp,
                    policy_info={"prompt_ids": pid, "completion_text": f" {k}"},
                    done=True,
                )
                tr.reward = {"final": float(k)}  # variance -> nonzero advantage
                trajs.append(Trajectory(transitions=[tr], id=f"p{pi}_k{k}", group_id=f"p{pi}", status=Status.COMPLETED))
        return trajs

    loss_ref = _inline_reference_objective(model, _mk())
    loss_packer, _ = make_objective(packer).fn(model, _with_advantages(_mk()))
    assert torch.equal(loss_ref, loss_packer), (loss_ref.item(), loss_packer.item())


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_objective_uses_configured_policy_temperature() -> None:
    """The default GRPO builder must forward rl.temperature into the policy gradient."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import build_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    class _FixedPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([2.0, 0.5, -1.0]))

        def forward(self, ids: torch.Tensor) -> SimpleNamespace:
            batch, length = ids.shape
            return SimpleNamespace(logits=self.logits.expand(batch, length, -1))

    def _batch() -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        for token, reward in ((1, 0.0), (2, 1.0)):
            transition = Transition(
                obs=ObsBundle(),
                action=torch.tensor([token]),
                policy_info={"prompt_ids": torch.tensor([0])},
                done=True,
            )
            transition.reward = {"final": reward}
            trajectories.append(
                Trajectory(
                    transitions=[transition],
                    id=f"t{token}",
                    group_id="g",
                    status=Status.COMPLETED,
                )
            )
        return _with_advantages(trajectories)

    gradients: list[torch.Tensor] = []
    for temperature in (1.0, 2.0):
        model = _FixedPolicy()
        trajectories = _batch()
        loss, _metrics = build_objective(
            GRPOConfig(grpo_surrogate="legacy_pg", temperature=temperature),
            "unused",
        ).fn(model, trajectories)
        advantages = torch.tensor([float(tr.transitions[0].advantage) for tr in trajectories])
        chosen = torch.tensor([1, 2])
        expected = -(advantages * torch.log_softmax(model.logits / temperature, dim=-1)[chosen]).mean()
        assert torch.allclose(loss, expected)
        loss.backward()
        gradients.append(model.logits.grad.detach().clone())

    assert not torch.allclose(gradients[0], gradients[1])


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_requires_ids_or_packer():
    """Shaping preference (spec 2026-07-28 §10): stored `prompt_ids` win; the packer is the FALLBACK.

    So the packer is optional on the objective — a batch that carries ids needs none — but a
    trajectory with NEITHER source is unshapeable and must fail loud rather than guess."""
    torch = pytest.importorskip("torch")
    import inspect

    from cosmos_framework.posttrain.algorithm.grpo import make_objective

    assert inspect.signature(make_objective).parameters["packer"].default is None  # optional
    tr = Transition(obs=ObsBundle(), action=torch.tensor([1, 2]), policy_info={}, done=True)
    tr.reward = {"final": 1.0}
    trajs = _with_advantages([Trajectory(transitions=[tr], id="t", group_id="g", status=Status.COMPLETED)])

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 4)

        def forward(self, ids):
            raise AssertionError("should raise before any forward")

    with pytest.raises(ValueError, match="no policy_info\\['prompt_ids'\\]"):
        make_objective().fn(_Net(), trajs)


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_stored_prompt_ids_fails_loud():
    """The stored-ids branch must mirror the packer's empty-prompt guard (codex P2).

    An empty `prompt_ids` makes the mask one element LONGER than the next-token logprob vector
    (`[0]*(0-1)` is empty, so `masks == [1]*C + [0]`), so without this guard the boolean index dies
    with an opaque shape error inside the forward rather than saying what is wrong."""
    torch = pytest.importorskip("torch")

    from cosmos_framework.posttrain.algorithm.grpo import make_objective

    tr = Transition(
        obs=ObsBundle(),
        action=torch.tensor([5, 6]),
        policy_info={"prompt_ids": torch.tensor([], dtype=torch.long)},  # empty prefix
        done=True,
    )
    tr.reward = {"final": 1.0}
    trajs = _with_advantages([Trajectory(transitions=[tr], id="t", group_id="g", status=Status.COMPLETED)])

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 4)

        def forward(self, ids):
            raise AssertionError("must raise before any forward")

    with pytest.raises(ValueError, match="prompt_ids.. is empty"):
        make_objective().fn(_Net(), trajs)


@pytest.mark.L0
@pytest.mark.CPU
def test_string_completion_with_stored_ids_and_no_packer_raises():
    """That path needs a tokenizer for the completion; without one it must fail loudly, not guess."""
    from cosmos_framework.posttrain.data.packer import shape_policy_input

    tr = Transition(
        obs=ObsBundle(representations={"prompt": "3+4="}, id="p0"),
        action=" 7",
        policy_info={"prompt_ids": torch.tensor([11, 22])},
        done=True,
    )
    with pytest.raises(ValueError, match="decoded string"):
        shape_policy_input(tr, None, 0)


@pytest.mark.L0
@pytest.mark.CPU
def test_ignore_prefix_masking_identical_on_both_branches():
    """`n_ignore_prefix_tokens` must mask the SAME completion-prefix tokens whether a trajectory
    carries stored ids or falls back to the packer. It was previously read off a packer ATTRIBUTE
    that no packer defines, so the knob silently did nothing on both paths and a mixed batch could
    have trained on prefix tokens for only some trajectories."""
    pytest.importorskip("transformers")

    from cosmos_framework.posttrain.data.packer import init_data_packer, shape_policy_input

    packer = init_data_packer("gpt2")
    obs = ObsBundle(representations={"prompt": "3+4="}, id="p0")
    prompt_ids = packer.tokenizer(packer.get_rollout_input(obs), add_special_tokens=False).input_ids
    comp_ids = packer.tokenizer(" 7 8 9", add_special_tokens=False).input_ids
    assert len(comp_ids) >= 3, "need a multi-token completion to exercise the knob"

    stored_tr = Transition(obs=obs, action=comp_ids, policy_info={"prompt_ids": prompt_ids}, done=True)
    fallback_tr = Transition(obs=obs, action=comp_ids, done=True)  # no policy_info -> packer branch

    for n_ign in (0, 1, 2):
        stored = shape_policy_input(stored_tr, packer, n_ign)
        fallback = shape_policy_input(fallback_tr, packer, n_ign)
        assert stored.input_ids == fallback.input_ids, f"ids diverge at n_ignore_prefix_tokens={n_ign}"
        assert stored.logprob_masks == fallback.logprob_masks, f"masks diverge at n_ignore_prefix_tokens={n_ign}"
        # and the knob actually bites: n_ign fewer scored positions each step
        assert sum(stored.logprob_masks) == len(comp_ids) - n_ign


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_pg_objective_forward_count_is_data_independent():
    """Every trajectory gets a forward, empty completion or not — FSDP collective parity.

    FSDP hangs its all-gathers off the module forward, so the number of forwards IS the number of
    collectives a rank issues. Skipping the empty ones made that count data-dependent: a peer
    holding different data would issue a different number and desync the group. An earlier version
    guarded only the all-empty case with a trailing probe forward placed AFTER the token-count
    all-reduce, which mismatched collective ORDER as well as count (that rank issued
    (all_reduce, forward) while peers issued (forward..., all_reduce)).

    Asserts on the forward COUNT rather than on the loss, because a wrong count is invisible in the
    loss on one rank and only deadlocks in a multi-rank run, which no CPU test can reach.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.data.packer import init_data_packer

    packer = init_data_packer("gpt2")
    obs = ObsBundle(representations={"prompt": "3+4="}, id="p0")
    nonempty = torch.tensor(packer.tokenizer(" 7", add_special_tokens=False).input_ids, dtype=torch.long)
    empty = torch.tensor([], dtype=torch.long)

    def _batch(actions):
        out = []
        for k, act in enumerate(actions):
            tr = Transition(obs=obs, action=act, policy_info={"completion_text": ""}, done=True)
            out.append(Trajectory(transitions=[tr], id=f"p0_k{k}", group_id="p0", status=Status.COMPLETED))
            out[-1].transitions[0].reward = {"final": float(k)}
        return out

    for actions in ([nonempty, empty, nonempty], [empty, empty], [nonempty, nonempty]):
        model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64))
        calls = {"n": 0}
        inner = model.forward

        def _counting(*a, _inner=inner, **kw):
            calls["n"] += 1
            return _inner(*a, **kw)

        model.forward = _counting
        loss, _ = make_objective(packer).fn(model, _with_advantages(_batch(actions)))
        assert calls["n"] == len(actions), (
            f"forward count {calls['n']} != {len(actions)} trajectories — the count must not depend "
            f"on which completions are empty (actions empty-mask pattern: "
            f"{[a.numel() == 0 for a in actions]})"
        )
        # Loss must stay on the graph even with NO learnable token, or backward() raises and the
        # peers' gradient collectives are never joined.
        assert loss.requires_grad and loss.grad_fn is not None
        loss.backward()  # must not raise


# ---------------------------------------------------------------------------
# The objective weight sum lets the trainer defer dividing a micro-batch's loss by a
# DP-global denominator to the optimizer step, so the objective must report the exact divisor it
# used (`rank_mean_count`'s DP-MEAN return), not `grpo_valid_tokens` (rank-local — a
# diagnostic, never the divisor). Exercised on `make_clipped_objective` because, unlike the
# legacy_pg objective above, its surrogate is DATA-INDEPENDENT of the model at the on-policy point
# (`old = cur.detach()` => ratio == 1 exactly, which makes both the epsilon-clip and the dual-clip
# bound inert — see `clipped_policy_gradient_loss`'s docstring) — so the expected numerator can be
# computed from the advantages and token counts alone, without touching the (randomly-initialized)
# model's actual logits.
# ---------------------------------------------------------------------------


def _tiny_clipped_batch():
    """A 3-sequence toy batch sharing one `group_id`: distinct completion lengths and rewards give
    nonzero-variance (hence nonzero, non-equal) group-relative advantages — a singleton group would
    normalize every advantage to 0 and make the divisor test vacuous. No `rollout_logprobs` on any
    transition, so pi_old defaults to `cur.detach()` (on-policy, ratio == 1)."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.data.packer import init_data_packer

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()
    packer = init_data_packer("gpt2")
    tok = packer.tokenizer
    specs = [("3+4=", " 7", 1.0), ("100+200=", " 300 400", 0.0), ("5+5=", " 10 11 12", 0.5)]
    trajs = []
    for i, (text, comp_text, reward) in enumerate(specs):
        comp = torch.tensor(tok(comp_text, add_special_tokens=False).input_ids, dtype=torch.long)
        tr = Transition(
            obs=ObsBundle(representations={"prompt": text}, id=f"p{i}"),
            action=comp,
            policy_info={"completion_text": comp_text},
            done=True,
        )
        tr.reward = {"final": reward}
        trajs.append(Trajectory(transitions=[tr], id=f"g0_k{i}", group_id="g0", status=Status.COMPLETED))
    return model, packer, tok, _with_advantages(trajs), specs


def _run_clipped_objective_on_toy_batch():
    """Runs `make_clipped_objective` (default `loss_type="token-mean"`, the `Reduction.TOKEN_SUM`
    path that divides by the DP-global objective weight at grpo.py's clipped-objective return) on
    `_tiny_clipped_batch`. Returns `(loss, metrics, trajectories, tokenizer, specs)` — the last three
    are what `_expected_numerator` needs to recompute the divided quantity independently."""
    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    model, packer, tok, trajs, specs = _tiny_clipped_batch()
    # batch_per_replica must be a positive multiple of group_size; this is an objective-only
    # config (no loop), so the minimal consistent value (one prompt's worth) is enough.
    cfg = GRPOConfig(group_size=len(specs), batch_per_replica=len(specs))  # loss_type="token-mean" default
    loss, metrics = make_clipped_objective(packer, cfg).fn(model, trajs)
    return loss, metrics, trajs, tok, specs


def _expected_numerator(trajs, tok, specs) -> float:
    """The clipped surrogate's numerator, computed WITHOUT running the model.

    On-policy (`old == cur.detach()`), `clipped_policy_gradient_loss` reduces exactly to `-adv` per
    token regardless of the model's actual logits (ratio == 1 makes the clip and the dual-clip
    bound both inert), so `token-mean`'s numerator (`sum(per_token)`) is
    `sum(-adv_i for each completion token)` = `-sum(adv_i * completion_token_count_i)`. Returned
    with the OPPOSITE sign (`sum(adv_i * count_i)`) to match how the test uses it:
    `loss == -_expected_numerator() / denom`.
    """
    total = 0.0
    for traj, (_, comp_text, _) in zip(trajs, specs):
        tr = traj.transitions[0]
        n_tok = len(tok(comp_text, add_special_tokens=False).input_ids)
        total += float(tr.advantage) * n_tok
    return total


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_metrics_report_the_divisor_actually_used():
    """The shared objective weight must be the DP-mean divisor, not the rank-local token count."""
    loss, metrics, trajs, tok, specs = _run_clipped_objective_on_toy_batch()
    assert isinstance(metrics[OBJECTIVE_WEIGHT_SUM], float)
    # Single rank: the DP mean of one value is that value, so denom == local count.
    assert metrics[OBJECTIVE_WEIGHT_SUM] == pytest.approx(metrics["grpo_valid_tokens"])
    # And it is the number the loss was actually divided by.
    assert float(loss.detach()) == pytest.approx(
        -_expected_numerator(trajs, tok, specs)
        / (metrics[OBJECTIVE_WEIGHT_SUM] if metrics[OBJECTIVE_WEIGHT_SUM] else 1.0)
    )


def _run_clipped_objective_on_empty_batch():
    """One trajectory whose completion is EMPTY, so `shape_policy_input` yields an all-False mask,
    shaping filters it out of `processed`, and the clipped objective takes the `if not processed:`
    early return — the path that must still join `rank_mean_count`'s collective (its peers
    call it unconditionally) before returning, per the docstring at that return site."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig
    from cosmos_framework.posttrain.data.packer import init_data_packer

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()
    packer = init_data_packer("gpt2")
    obs = ObsBundle(representations={"prompt": "3+4="}, id="p0")
    tr = Transition(obs=obs, action=torch.tensor([], dtype=torch.long), policy_info={"completion_text": ""}, done=True)
    tr.reward = {"final": 1.0}
    trajs = _with_advantages([Trajectory(transitions=[tr], id="g0_k0", group_id="g0", status=Status.COMPLETED)])
    cfg = GRPOConfig(group_size=2)  # config validation floor; the fixture itself is a 1-trajectory shard
    return make_clipped_objective(packer, cfg).fn(model, trajs)


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_empty_batch_reports_zero_denominator():
    """The empty path returns after joining the collective; it must still report a denom."""
    loss, metrics = _run_clipped_objective_on_empty_batch()
    assert metrics[OBJECTIVE_WEIGHT_SUM] == 0.0
    assert float(loss.detach()) == 0.0
