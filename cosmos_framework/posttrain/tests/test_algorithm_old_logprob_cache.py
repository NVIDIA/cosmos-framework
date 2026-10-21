# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""`OldLogprobCache`: π_old frozen across an optimizer-step window (`algorithm/grpo.py`).

These pin the property the clipped surrogate depends on at `n_mini > 1` — that every optimizer step
in a window scores against the SAME π_old, so the ratio can leave 1 and the epsilon-clip can bind.
Without it the ratio is `exp(cur - cur.detach()) == 1` on every step and the clip is inert, which
measurably destroyed a Qwen2.5-3B GSM8K run (reward -> 0.0 at step 29/100).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.grpo import OldLogprobCache


@pytest.mark.L0
@pytest.mark.CPU
def test_inert_until_recorded_reproduces_detach():
    # n_mini == 1 never runs the ahead pass, so the cache must stay out of the way ENTIRELY:
    # same value, and `.detach()` semantics (no grad) exactly as before it existed.
    cache = OldLogprobCache()
    cur = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    old = cache.old_for(cur)
    assert cache.mode == "off"
    assert cache.recorded == 0
    assert torch.equal(old, cur.detach())
    assert not old.requires_grad


@pytest.mark.L0
@pytest.mark.CPU
def test_replay_returns_the_recorded_policy_not_the_current_one():
    """THE load-bearing property: after the policy moves, π_old must NOT move with it."""
    cache = OldLogprobCache()
    collecting = [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])]
    for chunk in collecting:
        cache.begin_record()
        cache.old_for(chunk)
    assert cache.recorded == 2

    cache.begin_replay()
    # Simulate the policy having drifted: `cur` is now far from what collected the batch.
    drifted = torch.tensor([9.0, 9.0])
    assert torch.equal(cache.old_for(drifted), collecting[0])
    assert torch.equal(cache.old_for(drifted), collecting[1])


@pytest.mark.L0
@pytest.mark.CPU
def test_ratio_can_leave_one_only_when_frozen():
    # The whole point, stated as the ratio the surrogate actually forms.
    cur_at_collect = torch.tensor([-1.0, -1.0])
    cur_after_updates = torch.tensor([-0.5, -1.5])

    # Unfrozen (today's n_mini == 1 semantics): ratio is identically 1, clip inert.
    inert = OldLogprobCache()
    ratio_unfrozen = torch.exp(cur_after_updates - inert.old_for(cur_after_updates))
    assert torch.allclose(ratio_unfrozen, torch.ones(2))

    # Frozen: ratio departs from 1 by exactly the drift, so epsilon-clipping has something to bind.
    frozen = OldLogprobCache()
    frozen.begin_record()
    frozen.old_for(cur_at_collect)
    frozen.begin_replay()
    ratio_frozen = torch.exp(cur_after_updates - frozen.old_for(cur_after_updates))
    assert torch.allclose(ratio_frozen, torch.exp(cur_after_updates - cur_at_collect))
    assert not torch.allclose(ratio_frozen, torch.ones(2))


@pytest.mark.L0
@pytest.mark.CPU
def test_over_read_raises_instead_of_silently_unfreezing():
    # A train() beyond the recorded window must FAIL, not fall back to `cur.detach()`. The fallback
    # is the unclipped update this class exists to prevent and is invisible in every metric.
    cache = OldLogprobCache()
    cache.begin_record()
    cache.old_for(torch.tensor([1.0]))
    cache.begin_replay()
    cache.old_for(torch.tensor([0.0]))
    with pytest.raises(RuntimeError, match="exhausted"):
        cache.old_for(torch.tensor([0.0]))


@pytest.mark.L0
@pytest.mark.CPU
def test_next_window_clears_the_previous_one():
    # The cache is reused across loop steps; a stale window would freeze π_old at the WRONG policy.
    cache = OldLogprobCache()
    cache.begin_record()
    cache.old_for(torch.tensor([1.0]))
    cache.begin_replay()
    cache.old_for(torch.tensor([0.0]))

    cache.begin_record()  # next batch
    cache.old_for(torch.tensor([7.0]))
    assert cache.recorded == 1, "previous window leaked into the new one"
    cache.begin_replay()
    assert torch.equal(cache.old_for(torch.tensor([0.0])), torch.tensor([7.0]))


@pytest.mark.L0
@pytest.mark.CPU
def test_begin_record_is_idempotent_within_a_pass():
    # The trainer calls `begin_record()` on EVERY ahead-pass chunk (it cannot see (mini, micro)), so
    # only the first may clear — otherwise the window would never hold more than one chunk.
    cache = OldLogprobCache()
    for v in (1.0, 2.0, 3.0):
        cache.begin_record()
        cache.old_for(torch.tensor([v]))
    assert cache.recorded == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_control_kwargs_are_stripped_for_the_ahead_pass():
    # The DISPATCH/_SIGNATURES coupling itself is already covered, and more strictly, by
    # `test_roles_dispatch_registry.py::test_every_dispatch_key_is_covered_by_signatures` — which
    # caught the missing `_SIGNATURES` row this method was first shipped without. This asserts the
    # BEHAVIOUR that registry pairing buys for this specific edge, which that test does not.
    # The body signature is `compute_old_logprobs(batch)`; mini/micro/n_mini/n_micro must be
    # consumed by the ingress exactly as they are for `train`.
    from cosmos_framework.posttrain.roles.core.dispatch import DISPATCH, bind_param_names
    from cosmos_framework.posttrain.workers.worker_group import take_control_kwargs

    assert bind_param_names("trainer", "compute_old_logprobs", (["traj"],), {})[0] == "batch"
    spec = DISPATCH[("trainer", "compute_old_logprobs", "batch")]
    body_kw, select_kw = take_control_kwargs(spec, {"mini": 0, "micro": 1, "n_mini": 4, "n_micro": 2})
    assert body_kw == {}, f"control kwargs leaked to the body: {body_kw}"
    assert select_kw == {"mini": 0, "micro": 1, "n_mini": 4, "n_micro": 2}


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_carries_a_cache_and_trainer_can_find_it():
    # The trainer discovers the cache via `getattr(self.objective, "old_logp_cache", None)`; if the
    # factory stops attaching it the ahead pass would silently no-op.
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction

    plain = Objective(lambda *a, **k: (None, {}), reduction=Reduction.MEAN)
    assert plain.old_logp_cache is None, "objectives without a ratio must not carry one"

    carried = Objective(lambda *a, **k: (None, {}), reduction=Reduction.MEAN, old_logp_cache=OldLogprobCache())
    assert isinstance(carried.old_logp_cache, OldLogprobCache)


@pytest.mark.L0
@pytest.mark.CPU
def test_ahead_pass_refuses_a_stochastic_forward():
    """Dropout makes π_old(eval) and cur(train) differ at UNCHANGED weights — the ratio becomes
    noise and the clip fires on nothing.

    This is the one hazard the ahead pass introduces that `n_mini == 1` does not have: there π_old
    is `cur.detach()` from the SAME forward, so dropout cancels exactly. cosmos-rl has the identical
    eval/train split and no such check, so mirroring it is not evidence of safety. (An earlier
    version of this docstring added "its supported models just ship dropout at 0.0" — NOT verifiable
    from cosmos-rl source, which shows the eval/train split and nothing about model forwards. Only
    Qwen2.5-3B was actually checked.)
    """
    transformers = pytest.importorskip("transformers")
    from cosmos_framework.posttrain.roles.trainer.llm import _raise_if_forward_is_stochastic

    # GPT-2 defaults: resid_pdrop / embd_pdrop / attn_pdrop all 0.1 — and it is what this repo's own
    # trainer tests instantiate, so the unguarded ahead pass would have been exercised under it.
    stochastic = transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=1, n_head=1, n_embd=8))
    with pytest.raises(ValueError, match="DETERMINISTIC forward"):
        _raise_if_forward_is_stochastic(stochastic)

    # Same architecture with dropout off must pass — the guard must not block legitimate configs.
    deterministic = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=1, n_head=1, n_embd=8, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    )
    _raise_if_forward_is_stochastic(deterministic)


@pytest.mark.L0
@pytest.mark.CPU
def test_guard_catches_functional_dropout_declared_only_in_config():
    """Qwen2 applies attention dropout FUNCTIONALLY (`F.dropout(..., training=self.training)`) with
    no `nn.Dropout` module, so a module scan alone silently misses the production model family."""
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.trainer.llm import _raise_if_forward_is_stochastic

    class _Cfg:
        def __init__(self, p):
            self.attention_dropout = p

    class _Net(nn.Module):  # no nn.Dropout module anywhere
        def __init__(self, p):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.config = _Cfg(p)

    with pytest.raises(ValueError, match="attention_dropout"):
        _raise_if_forward_is_stochastic(_Net(0.1))
    _raise_if_forward_is_stochastic(_Net(0.0))  # Qwen2.5-3B's actual value


@pytest.mark.L0
@pytest.mark.CPU
def test_ahead_pass_predicate_matches_cosmos_rl_both_clauses():
    """`needs_cached_old_logprobs` must mirror cosmos-rl's `need_compute_old_ahead` EXACTLY:
    `batch_size > per_optimize_batch_size AND NOT use_rollout_logprobs_for_loss`.

    I originally implemented only the first clause. Under rollout logprobs the objective takes the
    `gather_behavior_logprobs` branch and never consults the cache, so an ahead pass there is a
    wasted forward that ALSO applies the eval/train determinism guard to a mode with no eval-mode
    π_old — rejecting a dropout-enabled model that is perfectly valid. And because the bulk refcount
    is sized from the same decision, gating the two differently leaks BulkStore entries.
    """
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    def _cfg(mini, rollout_lp):
        c = GRPOConfig()
        c.batch_per_replica, c.group_size = 128, 16
        c.mini_batch_per_replica = mini
        c.use_rollout_logprobs_for_loss = rollout_lp
        return c

    assert _cfg(4, False).needs_cached_old_logprobs is True, "n_mini>1 without rollout logprobs must record"
    assert _cfg(4, True).needs_cached_old_logprobs is False, "rollout logprobs bypass the cache entirely"
    assert _cfg(0, False).needs_cached_old_logprobs is False, "n_mini==1 needs no frozen π_old"
    assert _cfg(0, True).needs_cached_old_logprobs is False


@pytest.mark.L0
@pytest.mark.CPU
def test_reader_doubling_follows_the_driver_decision_not_n_mini():
    """`workers/grid.py` sizes the bulk refcount from the forwarded `ahead_pass` flag, so it cannot
    disagree with the driver. Re-deriving it from `n_mini` alone leaked entries under rollout
    logprobs (driver skips the pass, refcount still doubled, nothing ever drains)."""
    from cosmos_framework.posttrain.roles.core.dispatch import DISPATCH
    from cosmos_framework.posttrain.workers.dispatch import select_chunk

    for edge in (("trainer", "train", "batch"), ("trainer", "compute_old_logprobs", "batch")):
        assert "ahead_pass" in DISPATCH[edge].select_params, f"{edge} cannot carry the routing flag"

    # Selection must stay a pure function of (mini, micro, dp): the flag is accepted and IGNORED,
    # or the two passes would not produce identical chunks and the order-keyed cache would mis-pair.
    # `select_chunk` groups by `group_id`, so the items must carry one.
    items = [SimpleNamespace(group_id=f"g{i // 2}", id=f"t{i}", idx=i) for i in range(8)]
    kw = dict(mini=0, micro=0, n_mini=2, n_micro=1, dp_rank=0, dp_world=1)
    on = [x.idx for x in select_chunk(items, ahead_pass=True, **kw)]
    off = [x.idx for x in select_chunk(items, ahead_pass=False, **kw)]
    assert on == off, "the routing flag must not perturb which chunk a call selects"
    assert on, "selection returned nothing — the assertion above would pass vacuously"


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_capability_decides_not_the_inherited_config():
    """`OPDConfig` INHERITS `needs_cached_old_logprobs`, so config alone is the wrong gate.

    OPD's DIRECT objective forms no importance ratio and carries no cache; driving the ahead pass
    for it aborted the run on its first chunk (`compute_old_logprobs` raises without a cache). The
    capability is a pure read of the objective the trainer holds, so it cannot be wrong by
    inheritance.
    """
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    def _capability(objective):
        trainer = LLMTrainer.__new__(LLMTrainer)  # capability is a pure read of the objective
        trainer.objective = objective
        return trainer.needs_old_logprobs()

    assert _capability(Objective(lambda *a, **k: (None, {}), reduction=Reduction.MEAN)) is False
    assert (
        _capability(Objective(lambda *a, **k: (None, {}), reduction=Reduction.MEAN, old_logp_cache=OldLogprobCache()))
        is True
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_probe_mode_never_enters_the_recorded_window():
    """The determinism probe captures into a scratch slot. If it appended, the window would gain a
    phantom chunk and every subsequent replay would be off by one — mis-pairing π_old with the
    wrong tokens, which is worse than not freezing at all."""
    cache = OldLogprobCache()
    cache.begin_record()
    cache.old_for(torch.tensor([1.0]))
    assert cache.recorded == 1

    cache.begin_probe()
    cache.old_for(torch.tensor([9.0]))
    assert cache.recorded == 1, "probe leaked into the recorded window"
    assert torch.equal(cache.probe, torch.tensor([9.0]))
    assert torch.equal(cache.last_recorded, torch.tensor([1.0]))

    cache.mode = "record"  # trainer restores the window after probing
    cache.old_for(torch.tensor([2.0]))
    assert cache.recorded == 2
    cache.begin_replay()
    assert [float(cache.old_for(torch.tensor([0.0]))) for _ in range(2)] == [1.0, 2.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_duplicate_or_lost_ahead_pass_record_is_fatal_not_silent():
    """A retried ahead-pass call appends π_old twice; every later chunk then replays its
    PREDECESSOR's reference. The exhaustion guard cannot see it — the count still runs out on
    schedule — so it is a silent wrong-reference update. The expected-count check is what catches it.
    """
    cache = OldLogprobCache()
    for v in (1.0, 2.0):
        cache.begin_record()
        cache.old_for(torch.tensor([v]))
    cache.old_for(torch.tensor([2.0]))  # the retry: same chunk delivered twice
    assert cache.recorded == 3

    with pytest.raises(RuntimeError, match="walked 2 chunks"):
        cache.begin_replay(expected=2)

    # A LOST call mis-pairs identically and must also refuse.
    short = OldLogprobCache()
    short.begin_record()
    short.old_for(torch.tensor([1.0]))
    with pytest.raises(RuntimeError, match="holds 1 records"):
        short.begin_replay(expected=2)

    # The honest window still replays, and `expected=None` keeps the unchecked call working.
    ok = OldLogprobCache()
    for v in (1.0, 2.0):
        ok.begin_record()
        ok.old_for(torch.tensor([v]))
    ok.begin_replay(expected=2)
    assert [float(ok.old_for(torch.tensor([0.0]))) for _ in range(2)] == [1.0, 2.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_vacuous_probe_does_not_retire_the_determinism_guard():
    """An empty shard returns before `old_for` is reached, so that rank compares NOTHING. Retiring
    the guard on it would disable the check for the whole run on a comparison that never happened —
    and the rank that does hold data would never be probed."""
    from cosmos_framework.posttrain.utils.dist import rank_any

    empty = OldLogprobCache()
    empty.begin_record()  # no old_for(): the empty-shard path returns before π_old
    empty.begin_probe()
    assert empty.last_recorded is None and empty.probe is None
    compared = empty.last_recorded is not None and empty.probe is not None
    assert compared is False, "a rank with no data must not report a comparison"
    assert rank_any(False) is False, "single-rank: nothing compared stays not-compared"
    assert rank_any(True) is True, "a real comparison anywhere certifies the model"


@pytest.mark.L0
@pytest.mark.CPU
def test_n_mini_gt_1_is_refused_without_an_explicit_determinism_attestation():
    """The ahead pass is FAIL-CLOSED for an unvetted model.

    Neither screen can prove a forward is deterministic given its weights — `LLMTrainer` accepts any
    model, and a layer stochastic only on inputs the probed chunk misses passes both. Retiring a
    general-model guard on one sampled comparison is not a correctness contract, so the contract is
    stated by whoever knows the model and the screens become defence in depth.
    """
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    def _cfg(mini, certified):
        c = GRPOConfig()
        # The gate is scoped to objectives that actually consult a cached π_old. `legacy_pg` is a
        # plain policy gradient with no ratio and no cache, so it is exempt and must
        # stay mini-batchable. The ahead pass runs on the CLIPPED objective, which is what this test
        # is about.
        c.grpo_surrogate = "clipped"
        c.batch_per_replica, c.group_size = 128, 16
        c.mini_batch_per_replica = mini
        c.certified_deterministic_forward = certified
        return c

    with pytest.raises(ValueError, match="certified_deterministic_forward"):
        _cfg(4, False).__post_init__()  # n_mini = 32 > 1, unattested

    _cfg(4, True).__post_init__()  # attested: allowed
    _cfg(0, False).__post_init__()  # n_mini == 1: π_old comes from the same forward, no attestation
    _cfg(128, False).__post_init__()  # mini == batch ⇒ n_mini == 1, likewise no attestation


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_logprob_objective_carries_no_cache_so_expectation_is_rank_invariant():
    """`needs_old_logprobs()` must answer from RANK-INVARIANT facts, not mutable cache mode.

    Under `use_rollout_logprobs_for_loss` the objective supplies π_old from `gather_behavior_logprobs`
    and never reaches `old_for`, so a cache there would sit permanently empty while still reporting
    True. The trainer derives "was an ahead pass expected?" from (n_mini > 1, cache is not None); if
    that could disagree across ranks, a rank would skip a collective its peers entered — which is
    exactly how a restarted rank hung its peers before this was fixed.
    """
    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective

    def _obj(rollout_lp):
        cfg = SimpleNamespace(
            loss_type="token-mean",
            temperature=1.0,
            epsilon_low=0.2,
            epsilon_high=0.2,
            kl_beta=0.0,
            lower_bound_ratio=3.0,
            use_rollout_logprobs_for_loss=rollout_lp,
        )
        return make_clipped_objective(packer=None, algo_cfg=cfg)

    assert _obj(False).old_logp_cache is not None, "the clipped objective consults a frozen π_old"
    assert _obj(True).old_logp_cache is None, "rollout-logprob mode never consults the cache"


@pytest.mark.L0
@pytest.mark.CPU
def test_discarded_chunks_advance_the_read_pointer_so_pairing_survives_a_poisoned_window():
    """A chunk the trainer DISCARDS must still consume its slot, or every later chunk mis-pairs.

    The window is keyed by append order: training chunk k replays `_store[k]`. `LLMTrainer.train`'s
    poisoned-window path (section 5.2) returns before the objective runs, so `old_for` never fires
    for that chunk -- while the ahead pass already recorded one. Without `skip_chunk` the pointer
    lags and chunk k replays the reference of a DIFFERENT chunk's trajectories for the rest of the
    collected batch.

    It is the silent variant: the cache UNDER-consumes, so `old_for`'s exhaustion guard cannot fire
    and `begin_replay_window`'s count check is an upper bound only. And the resulting quantity is not
    an importance ratio -- `exp(log pi(A) - log pi_old(B))` saturates the clamp and pins tokens at
    the clip boundary, which is worse than never freezing.

    Shape here is the reviewer's: n_mini=2, n_micro=2, one discard in the first window.
    """
    import torch

    cache = OldLogprobCache()
    recorded = [torch.tensor([float(i)]) for i in range(4)]  # 2 minis x 2 micros

    cache.begin_record()
    for chunk in recorded:
        cache.old_for(chunk)
    assert len(cache._store) == 4
    cache.begin_replay()

    # chunk 0 scores normally
    assert cache.old_for(torch.tensor([99.0])).item() == 0.0
    # chunk 1 is DISCARDED by the poisoned-window path: no objective call, so no `old_for`
    cache.skip_chunk()
    # chunk 2 must now see ITS OWN reference, not chunk 1's
    assert cache.old_for(torch.tensor([99.0])).item() == 2.0, (
        "a discarded chunk must consume its slot; otherwise chunk 2 replays chunk 1's reference"
    )
    assert cache.old_for(torch.tensor([99.0])).item() == 3.0


@pytest.mark.L0
@pytest.mark.CPU
def test_skip_chunk_is_inert_outside_replay():
    """No window, no pairing to maintain — `skip_chunk` must not disturb recording or the off state."""
    import torch

    cache = OldLogprobCache()
    cache.skip_chunk()  # mode "off"
    assert cache._read == 0

    cache.begin_record()
    cache.old_for(torch.tensor([1.0]))
    cache.skip_chunk()  # must not consume anything while recording
    assert cache._read == 0 and len(cache._store) == 1
