# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Golden-reference tests for the clipped policy-gradient loss used by GRPO.

Pins three properties of `clipped_policy_gradient_loss` (the pure per-token math) on tiny CPU
tensors: on-policy reduction to policy-gradient, the upper epsilon-clip binding, and the KL
penalty sign. `make_clipped_objective` is imported to guard that the public objective factory
stays importable alongside the `legacy_pg` `make_objective`.
"""

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.grpo import (  # noqa: E402
    make_clipped_objective,  # noqa: F401  (import guard: public factory stays importable)
)


def _with_advantages(trajs, unbiased: bool = False):
    """Apply the reward-stage estimator, as the real pipeline does before dispatch.

    The objective reads `transition.advantage` instead of computing it, so a test that hands it raw
    trajectories is skipping the reward stage and must supply what that stage would have written.
    """
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    group_relative(unbiased=unbiased)(list(trajs))
    return trajs


def _tiny():
    """Replicated tiny random GPT2 + gpt2 packer (posttrain tests never import each other).

    Mirrors `tests/test_roles_rollout_hf.py::_tiny`: a 2-layer random `GPT2LMHeadModel` on CPU
    plus the `DecoderOnlyLLMDataPacker` (owns the gpt2 tokenizer). `.eval()` disables dropout so
    the batched and per-sequence forwards are numerically comparable (dropout would desync them).
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.data.packer import init_data_packer

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()
    packer = init_data_packer("gpt2")  # shaping seam; owns the tokenizer (design 2026-07-28)
    return model, packer


@pytest.mark.L0
@pytest.mark.CPU
def test_clipped_objective_matches_inline_pg_gradient():
    """The batched `make_clipped_objective` gradient equals the inline PG `make_objective` gradient.

    At the on-policy point — no `rollout_logprobs` on the trajectories so `old = cur.detach()` =>
    ratio == 1, with `epsilon_{low,high}=10` so the clip never binds and `kl_beta=0` => no KL — the
    clipped surrogate reduces to policy-gradient. Both objectives then compute the SAME per-token
    gradient `-adv * d(mean completion logp)`; the model parameter gradients must match.

    This is the only test that EXECUTES the batched path end-to-end: it pins the mask alignment
    (`masks[:, :-1]`), the single batched (padded) forward, the advantage broadcast
    (`repeat_interleave`), and the token-mean-then-sequence-mean reduction — none of which the three
    pure clipped-loss helper tests above exercise. The two prompts have different token
    lengths, so the batched forward genuinely pads (right-pad + GPT2 causal attention => the
    completion-position logits are identical to the unbatched forward, up to float batch-reduction
    noise, hence `atol=1e-5` rather than exact).

    Advantages come from a SINGLE shared `group_id` with rewards {1.0, 0.0} => normalized
    advantages {+1.0, -1.0} (nonzero and non-equal). Distinct group_ids would make each trajectory a
    singleton group (std 0 => advantage 0 => trivially zero gradients), defeating the test; this
    mirrors `test_algorithm_grpo.py`'s shared-group pattern.
    """
    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig  # clip knobs are FLAT on it
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    torch.manual_seed(0)
    model, packer = _tiny()
    tok = packer.tokenizer

    def _mk():
        # distinct-length prompts (so the batched forward pads) + rewards giving advantages +1/-1.
        specs = [("3+4=", " 7", 1.0), ("100+200=", " 300", 0.0)]
        trajs = []
        for i, (text, comp_text, reward) in enumerate(specs):
            comp = torch.tensor(tok(comp_text, add_special_tokens=False).input_ids, dtype=torch.long)
            tr = Transition(
                obs=ObsBundle(representations={"prompt": text}, id=f"p{i}"),
                action=comp,
                policy_info={"completion_text": comp_text},  # no rollout_logprobs => on-policy (ratio 1)
                done=True,
            )
            tr.reward = {"final": reward}
            trajs.append(Trajectory(transitions=[tr], id=f"g0_k{i}", group_id="g0", status=Status.COMPLETED))
        # The reward stage writes the advantage now; the objective only reads it.
        return _with_advantages(trajs)

    cfg = GRPOConfig(epsilon_low=10.0, epsilon_high=10.0, kl_beta=0.0, temperature=1.0, group_size=2)
    # group_size=2 matches what the fixture builds. The objective no longer requires whole groups —
    # the advantage is baked in upstream — but the config value still describes the batch honestly.

    model.zero_grad()
    loss_clip, _ = make_clipped_objective(packer, cfg)(model, _mk())
    loss_clip.backward()
    grads_clip = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad()  # isolate the second backward from the first
    loss_pg, _ = make_objective(packer)(model, _mk())
    loss_pg.backward()
    grads_pg = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    assert grads_clip and set(grads_clip) == set(grads_pg)
    # Guard against a trivially-zero pass (would make allclose vacuous): advantages +1/-1 must move params.
    assert any(g.abs().max() > 1e-6 for g in grads_pg.values())
    for name in grads_pg:
        assert torch.allclose(grads_clip[name], grads_pg[name], atol=1e-5), name


@pytest.mark.L0
@pytest.mark.CPU
def test_pi_old_defaults_to_recomputed_current_policy():
    """π_old defaults to `cur.detach()` (ratio == 1) even when the trajectories DO carry
    `rollout_logprobs` — matching cosmos-rl's `use_rollout_logprobs_for_loss=False`.

    Regression: the engine's logprobs are raw temperature-1 values from the INFERENCE forward, so
    they equal log π_old only under the unwarped T=1 contract and carry the train/infer gap into
    every ratio otherwise; and a padded position (HF path) carries -inf, which becomes
    exp(+inf)=inf and then inf*0=NaN against a zero-advantage group (the common case early in
    training)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    assert GRPOConfig().use_rollout_logprobs_for_loss is False  # the cosmos-rl default

    packer = init_data_packer("gpt2")
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64))
    obs = ObsBundle(representations={"prompt": "3+4="}, id="p0")
    comp = packer.tokenizer(" 7", add_special_tokens=False).input_ids
    trajs = []
    for k in range(2):
        tr = Transition(
            obs=obs,
            action=torch.tensor(comp),
            # -inf is exactly what a padded / warped-out position reports. With the default this is
            # IGNORED; without it the loss would be NaN.
            policy_info={"rollout_logprobs": [float("-inf")] * len(comp)},
            done=True,
        )
        tr.reward = {"final": float(k)}
        trajs.append(Trajectory(transitions=[tr], id=f"p0_k{k}", group_id="p0", status=Status.COMPLETED))

    loss, _ = make_clipped_objective(packer, GRPOConfig(group_size=2)).fn(model, _with_advantages(trajs))
    assert torch.isfinite(loss), "default π_old path must not be poisoned by rollout_logprobs"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "loss_type",
    ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-fixed"],
)
def test_loss_type_modes_and_reduction_contract(loss_type: str) -> None:
    """Every mode defers its DP-global valid-unit divisor across micro-batches."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig
    from cosmos_framework.posttrain.data.packer import init_data_packer

    cfg = GRPOConfig(group_size=2)  # fixture builds 2-member groups
    assert GRPOConfig().loss_type == "token-mean"  # cosmos-rl parity default
    cfg.loss_type = loss_type
    max_response_length = 32 if loss_type == "seq-mean-token-sum-fixed" else None
    obj = make_clipped_objective(
        init_data_packer("gpt2"),
        cfg,
        max_response_length=max_response_length,
    )
    assert obj.reduction.value == "token_sum"


@pytest.mark.L0
@pytest.mark.CPU
def test_fixed_response_mode_scales_the_real_clipped_objective_without_dropping_metrics() -> None:
    """Dr-GRPO changes only the sequence-sum divisor inside the production objective."""
    from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM
    from cosmos_framework.posttrain.config.rl import GRPOConfig
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    torch.manual_seed(0)
    model, packer = _tiny()
    specs = [("3+4=", " 7", 1.0), ("100+200=", " 300 400", 0.0)]
    trajectories = []
    for index, (prompt, completion_text, reward) in enumerate(specs):
        completion = torch.tensor(  # [N_i]
            packer.tokenizer(completion_text, add_special_tokens=False).input_ids,
            dtype=torch.long,
        )
        transition = Transition(
            obs=ObsBundle(representations={"prompt": prompt}, id=f"p{index}"),
            action=completion,
            policy_info={"completion_text": completion_text},
            done=True,
        )
        transition.reward = {"final": reward}
        trajectories.append(
            Trajectory(
                transitions=[transition],
                id=f"g0_k{index}",
                group_id="g0",
                status=Status.COMPLETED,
            )
        )
    trajectories = _with_advantages(trajectories, unbiased=True)

    base_cfg = GRPOConfig(group_size=2, loss_type="seq-mean-token-sum")
    fixed_cfg = GRPOConfig(group_size=2, loss_type="seq-mean-token-sum-fixed", unbiased_advantage=True)
    base = make_clipped_objective(packer, base_cfg)
    fixed = make_clipped_objective(packer, fixed_cfg, max_response_length=32)

    base_loss, base_metrics = base.fn(model, trajectories)  # []
    fixed_loss, fixed_metrics = fixed.fn(model, trajectories)  # []

    assert fixed_loss.item() == pytest.approx(base_loss.item() / 32)
    assert fixed_metrics.keys() == base_metrics.keys()
    assert fixed_metrics[OBJECTIVE_WEIGHT_SUM] == base_metrics[OBJECTIVE_WEIGHT_SUM]
    assert fixed_metrics["grpo_loss"] == pytest.approx(base_metrics["grpo_loss"] / 32)
    assert fixed.reduction is base.reduction


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("max_response_length", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_fixed_response_mode_keeps_a_worker_side_bound_backstop(
    max_response_length: object,
    error: type[Exception],
) -> None:
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    cfg = GRPOConfig(group_size=2, loss_type="seq-mean-token-sum-fixed")
    with pytest.raises(error, match="rollout.max_tokens"):
        make_clipped_objective(
            object(),  # type: ignore[arg-type]
            cfg,
            max_response_length=max_response_length,  # type: ignore[arg-type]
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_dual_clip_bounds_negative_advantage_only():
    """Dual-clip (arXiv 1912.09729) caps the loss where the advantage is NEGATIVE, and only there.

    Pins the exact numbers from the cosmos-rl comparison (adv=-5, log-ratio 3): single-clip 100.43
    vs dual-clip 15.0 == 3.0*|adv|. Without a
    bound the adv<0 branch grows without limit in the ratio, so one badly off-policy token can
    dominate a batch — that unboundedness is the whole reason the term exists, and it had no test.
    """
    import torch

    from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import clipped_policy_gradient_loss

    old = torch.zeros(1)
    cur = torch.full((1,), 3.0)  # ratio = e^3 ~= 20.09, far outside the clip band
    neg = torch.full((1,), -5.0)

    single = clipped_policy_gradient_loss(cur, old, neg, 0.2, 0.2, lower_bound_ratio=None)
    dual = clipped_policy_gradient_loss(cur, old, neg, 0.2, 0.2, lower_bound_ratio=3.0)
    assert single.item() == pytest.approx(100.4277, rel=1e-4)  # unbounded single-clip
    assert dual.item() == pytest.approx(15.0, rel=1e-6)  # == lower_bound_ratio * |adv|
    assert dual.item() < single.item()

    # POSITIVE advantage: the bound must not fire — same ratio, identical with and without it.
    pos = torch.full((1,), 5.0)
    assert clipped_policy_gradient_loss(cur, old, pos, 0.2, 0.2, lower_bound_ratio=3.0).item() == pytest.approx(
        clipped_policy_gradient_loss(cur, old, pos, 0.2, 0.2, lower_bound_ratio=None).item(), rel=1e-6
    )

    # ON-POLICY (ratio == 1): inert for either sign, so enabling dual-clip cannot perturb the
    # on-policy gradient the golden-reference parity test pins.
    same = torch.zeros(1)
    for adv in (torch.full((1,), -5.0), torch.full((1,), 5.0)):
        with_dc = clipped_policy_gradient_loss(same, same, adv, 0.2, 0.2, lower_bound_ratio=3.0)
        without = clipped_policy_gradient_loss(same, same, adv, 0.2, 0.2, lower_bound_ratio=None)
        assert with_dc.item() == pytest.approx(without.item(), rel=1e-6)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("loss_type", ["seq-mean-token-sum", "seq-mean-token-mean"])
def test_seq_mean_modes_normalize_over_the_DP_GLOBAL_sequence_count(monkeypatch, loss_type):
    """mean(rank means) != mean(all sequences) once the ranks hold different numbers of sequences.

    `select_chunk` round-robins SAMPLES now, so shard sizes differ by up to one and a rank can
    hold zero. FSDP averages gradients equally across ranks, so a rank-LOCAL `.mean()` silently
    reweights: at 3 sequences over 2 ranks the lone sequence on the short rank carried 1/2 the weight
    instead of 1/3, and at 1 sequence over 2 ranks the empty rank contributed a zero mean and HALVED
    the update. Nothing raises — the loss is finite and plausible either way.

    This was correct before only because whole-group sharding plus a startup gate guaranteed equal
    counts. Both are gone, so the denominator has to be global, exactly as `token-mean`'s already is.

    Emulates two ranks IN ONE PROCESS: run the objective on each shard with `rank_mean_count`
    faked to return the DP mean, then check that averaging the two rank losses (what FSDP does)
    reproduces the loss of the whole batch on one rank.
    """
    import torch.distributed as dist
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    packer = init_data_packer("gpt2")
    tok = packer.tokenizer
    torch.manual_seed(0)  # same weights on every "rank" -- the shard is what differs, not the model
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()

    def _batch(specs):
        """One batch, advantages written ONCE over the whole thing — as the reward stage does.

        Building each shard separately would re-normalize the advantage within the shard, which is
        exactly what this refactor moved upstream to stop. The shards below are SLICES of this.
        """
        trajs = []
        for i, (text, comp_text, reward) in enumerate(specs):
            comp = torch.tensor(tok(comp_text, add_special_tokens=False).input_ids, dtype=torch.long)
            tr = Transition(
                obs=ObsBundle(representations={"prompt": text}, id=f"p{i}"),
                action=comp,
                policy_info={"completion_text": comp_text, "rollout_logprobs": [0.0] * comp.numel()},
                done=True,
            )
            tr.reward = {"final": reward}
            trajs.append(Trajectory(transitions=[tr], id=f"g0_k{i}", group_id="g0", status=Status.COMPLETED))
        return _with_advantages(trajs)

    # Distinct prompts/completion lengths so the three sequences carry genuinely different losses —
    # equal ones would make the buggy and correct denominators agree and the test prove nothing.
    specs = [("3+4=", " 7", 1.0), ("100+200=", " 300 400", 0.0), ("5+5=", " 10 11 12", 0.5)]
    cfg = GRPOConfig(
        epsilon_low=0.2, epsilon_high=0.2, kl_beta=0.0, temperature=1.0, group_size=3, batch_per_replica=6
    )  # batch must be a multiple of group_size (3); objective-only cfg
    cfg.loss_type = loss_type
    # Exercise the optional behavior-logprob verdict collective on both empty and non-empty shards.
    cfg.use_rollout_logprobs_for_loss = True
    obj = make_clipped_objective(packer, cfg)

    def _run(shard, dp_mean_n_units, peer_count):
        """One rank's loss, with the DP collectives faked: MAX -> the agreed forward count, AVG ->
        the DP-mean unit count. Both are real collectives in this objective and mean different
        things, so a fake that treats them alike would either abort the empty rank or skew the
        denominator."""
        calls: list = []

        def fake(tensor, op=None):
            calls.append(op)
            if tensor.dtype is torch.int32:
                return  # raise_if_any_rank_failed's flag: leave it 0 == no rank reported a failure
            tensor.fill_(float(peer_count if op is dist.ReduceOp.MAX else dp_mean_n_units))

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "all_reduce", fake)
        # A faked process group must also fake `get_backend`: `raise_if_any_rank_failed` asks for a
        # backend-compatible device via `collective_device()`, and the real call needs a real group.
        monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")
        try:
            loss, _ = obj.fn(model, shard)
        finally:
            monkeypatch.undo()
        assert dist.ReduceOp.AVG in calls, "the DP-global denominator collective was not issued"
        return float(loss.detach()), calls

    # Ground truth: the whole batch on ONE rank (dist uninitialized -> denominator is the local 3).
    full = _batch(specs)
    whole = float(obj.fn(model, full)[0].detach())

    # 3 sequences over 2 ranks -> shards of 2 and 1; the DP-mean unit count is 1.5 on both when the
    # unit is the SEQUENCE. Slices of `full`, so every advantage is the one the whole group implies.
    (a, calls_a) = _run(full[:2], 1.5, peer_count=2)
    (b, calls_b) = _run(full[2:], 1.5, peer_count=2)
    assert calls_a == calls_b, "uneven non-empty shards must issue identical collectives"
    assert (a + b) / 2 == pytest.approx(whole, rel=1e-5, abs=1e-6), (
        f"{loss_type}: FSDP's mean over rank losses gave {(a + b) / 2}, but the whole batch is "
        f"{whole} — the short rank's sequences are mis-weighted"
    )

    # 1 sequence over 2 ranks: one rank is EMPTY and must contribute 0 to the numerator while the
    # denominator stays global, so the surviving sequence keeps its full weight.
    one = _batch(specs[:1])
    whole_one = float(obj.fn(model, one)[0].detach())
    (a1, calls_a1) = _run(one, 0.5, peer_count=1)
    (b1, calls_b1) = _run([], 0.5, peer_count=1)
    assert calls_a1 == calls_b1, "empty and non-empty shards must issue identical collectives"
    assert (a1 + b1) / 2 == pytest.approx(whole_one, rel=1e-5, abs=1e-6), (
        f"{loss_type}: an empty shard scaled the update to {(a1 + b1) / 2} instead of {whole_one}"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_negative_epsilon_disables_that_side_of_the_clip():
    """cosmos-rl parity: a NEGATIVE epsilon means "no clip on this side", not a negative width.

    `grpo_trainer.py:216-232` builds `min_clipped = 1-eps_low if eps_low >= 0 else None` and passes
    `None` through to `torch.clamp`. The distillation configs rely on it — `epsilon_low =
    epsilon_high = -1.0` turns clipping off entirely, which is sound at `mu_iterations=1` where the
    ratio is identically 1 and the clip is inert anyway.

    Without the branch this pins, `clamp(ratio, 1-(-1), 1+(-1))` is `clamp(ratio, 2.0, 0.0)` — an
    INVERTED range, for which torch returns the max. `clipped` would collapse to 0 and the
    surrogate would silently become `-min(ratio*adv, 0)`: a wrong loss, no error, on the exact
    config an alignment run copies."""
    from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import clipped_policy_gradient_loss

    # Off-policy on purpose: at ratio == 1 every variant agrees, so the clip has to be made to bind.
    cur = torch.tensor([0.9, -0.8, 0.5])
    old = torch.zeros(3)
    adv = torch.tensor([1.0, 1.0, -1.0])
    ratio = torch.exp(cur - old)

    unclipped = clipped_policy_gradient_loss(cur, old, adv, -1.0, -1.0, lower_bound_ratio=None)
    assert torch.allclose(unclipped, -ratio * adv), (unclipped, -ratio * adv)

    # One side only: the upper clip binds on the ratio>1 rows, the lower bound is off.
    high_only = clipped_policy_gradient_loss(cur, old, adv, -1.0, 0.2, lower_bound_ratio=None)
    expected_high = -torch.min(ratio * adv, torch.clamp(ratio, max=1.2) * adv)
    assert torch.allclose(high_only, expected_high)

    # Symmetric case: the lower clip binds on the ratio<1 rows, the upper bound is off.
    low_only = clipped_policy_gradient_loss(cur, old, adv, 0.2, -1.0, lower_bound_ratio=None)
    expected_low = -torch.min(ratio * adv, torch.clamp(ratio, min=0.8) * adv)
    assert torch.allclose(low_only, expected_low)

    # And the ordinary two-sided form is untouched by the branch.
    both = clipped_policy_gradient_loss(cur, old, adv, 0.2, 0.2, lower_bound_ratio=None)
    expected_both = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    assert torch.allclose(both, expected_both)
    assert not torch.allclose(both, unclipped)


@pytest.mark.L0
@pytest.mark.CPU
def test_clip_diagnostics_report_an_inert_clip_and_a_binding_one():
    """`clip_frac`/`ratio_absdev` must distinguish a clip that binds from one that never does.

    On-policy (`old == cur`) the ratio is identically 1, so nothing is clipped and there is no
    drift — which is the state an ahead-pass-less n_mini>1 run sits in permanently while its loss
    still looks healthy. Push π_old away and the same tensors must report a bound clip.
    """
    import torch

    from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import clipped_policy_gradient_loss

    cur = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0])

    inert: dict = {}
    clipped_policy_gradient_loss(cur, cur.detach(), adv, 0.2, 0.2, stats=inert)
    assert inert["clip_count"] == 0.0
    assert inert["ratio_absdev_sum"] == 0.0

    bound: dict = {}
    # log-ratio +1 => ratio e = 2.72, above 1+eps_high for all four tokens — but only the two with a
    # POSITIVE advantage actually bind: at adv<0 the unclipped term is the smaller one, so `min`
    # selects it and the clamp is inert. Counting band membership would say 4 (see
    # `test_clip_count_ignores_out_of_band_ratios_the_min_does_not_select`).
    clipped_policy_gradient_loss(cur, cur.detach() - 1.0, adv, 0.2, 0.2, lower_bound_ratio=None, stats=bound)
    assert bound["clip_count"] == 2.0  # SUMS, so the caller can divide by a REDUCED token count
    assert bound["ratio_absdev_sum"] > 4.0


@pytest.mark.L0
@pytest.mark.CPU
def test_clip_diagnostics_are_opt_in_and_do_not_change_the_loss():
    """`stats=None` is the default and must leave the returned tensor bit-identical."""
    import torch

    from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import clipped_policy_gradient_loss

    cur, old = torch.tensor([-1.0, -2.0]), torch.tensor([-1.3, -1.7])
    adv = torch.tensor([1.0, -1.0])
    without = clipped_policy_gradient_loss(cur, old, adv, 0.2, 0.2)
    with_stats = clipped_policy_gradient_loss(cur, old, adv, 0.2, 0.2, stats={})
    assert torch.equal(without, with_stats)


@pytest.mark.L0
@pytest.mark.CPU
def test_clip_count_ignores_out_of_band_ratios_the_min_does_not_select():
    """Band membership is NOT the same as the clip biting.

    PPO's surrogate is `-min(ratio*adv, clipped*adv)`. For a POSITIVE advantage with `ratio < 1-eps`
    the unclipped term is already the smaller one, so `min` selects it and the clamp changes
    nothing; likewise a negative advantage with `ratio > 1+eps`. Counting band membership reports a
    trust region doing work it is not doing — and the diagnostic exists precisely to answer whether
    it is working, so an over-count is worse than no metric.
    """
    import torch

    from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import clipped_policy_gradient_loss

    cur = torch.tensor([-1.0, -1.0])
    # ratio = e^{-1} = 0.368, far BELOW 1-eps_low=0.8, with a POSITIVE advantage: min picks the
    # unclipped term, so the epsilon clamp is inert here.
    inert: dict = {}
    clipped_policy_gradient_loss(
        cur, cur.detach() + 1.0, torch.tensor([1.0, 1.0]), 0.2, 0.2, lower_bound_ratio=None, stats=inert
    )
    assert inert["clip_count"] == 0.0, "out-of-band ratio counted even though min chose the unclipped term"

    # Same ratio, NEGATIVE advantage: now the clipped term is the smaller one and the clamp binds.
    binds: dict = {}
    clipped_policy_gradient_loss(
        cur, cur.detach() + 1.0, torch.tensor([-1.0, -1.0]), 0.2, 0.2, lower_bound_ratio=None, stats=binds
    )
    assert binds["clip_count"] == 2.0


@pytest.mark.L0
@pytest.mark.CPU
def test_clip_count_includes_dual_clip_bindings():
    """The dual-clip lower bound also rewrites the per-token loss, so it must count as clipping."""
    import torch

    from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import clipped_policy_gradient_loss

    cur = torch.tensor([-1.0])
    # ratio = e^3 ~ 20 with adv < 0: the dual-clip bound at 3*|adv| is what selects.
    stats: dict = {}
    clipped_policy_gradient_loss(
        cur, cur.detach() - 3.0, torch.tensor([-5.0]), -1.0, -1.0, lower_bound_ratio=3.0, stats=stats
    )
    assert stats["clip_count"] == 1.0  # epsilon clip disabled, so only dual-clip can have bound


@pytest.mark.L0
@pytest.mark.CPU
def test_logprobs_are_computed_in_fp32_even_on_a_bf16_model():
    """The surrogate's ratio is a DIFFERENCE of two logprobs, so it needs more precision than the
    forward that produced them.

    cosmos-rl scores in `train.logprob_dtype` (default float32) INDEPENDENTLY of `param_dtype`, so a
    bf16 model still yields fp32 logprobs. Scoring in bf16 instead cannot resolve the small
    deviations that decide whether a token crosses the epsilon boundary — measured at n_mini=8, it
    made the clip engage on 0.089% of tokens against 1.49% in fp32, leaving the trust region
    effectively inert. Invisible at n_mini == 1, where the ratio is identically 1.
    """
    import torch

    from cosmos_framework.posttrain.config.rl import GRPOConfig

    assert GRPOConfig().logprob_dtype == "float32", "default must match cosmos-rl's logprob_dtype"

    # log_softmax must honour the dtype argument rather than the input's.
    logits = torch.randn(2, 5, 128, dtype=torch.bfloat16)
    lp32 = torch.log_softmax(logits[:, :-1], dim=-1, dtype=torch.float32)
    lp16 = torch.log_softmax(logits[:, :-1], dim=-1)
    assert lp32.dtype == torch.float32
    assert lp16.dtype == torch.bfloat16
    # and the fp32 result genuinely carries information bf16 rounds away
    assert (lp32 - lp32.to(torch.bfloat16).float()).abs().max() > 0
