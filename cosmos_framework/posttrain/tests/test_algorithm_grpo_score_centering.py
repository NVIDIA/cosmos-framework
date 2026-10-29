# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Score centering (`rl.score_centering`): the correction, its composition, the objective wiring, the config.

Pins, on tiny CPU tensors and a transformers-free two-layer LM (no tokenizer, no hub):

* the identity -- with a sampler that EQUALS the trainer (`q == p`, float32 records) the correction
  is exactly `0.0` on every token and the clipped objective's loss AND parameter gradients are
  bit-identical to the objective with centering off, for `compose` = none / tis / mis, with and
  without the KL anchor (the KL term is untouched by construction);
* the regression -- with centering OFF the objective ignores sampler records entirely: loss and
  gradients are bit-identical to a batch that never carried them, and every `sc_*` key is 0 with
  weight 0;
* the estimator -- on a 3-token vocabulary with a perturbed sampler the EXPECTED gradient of the
  centred score under the sampler is zero (exact expectation in float64 and a 10^6-sample Monte
  Carlo estimate), while plain REINFORCE drifts by exactly `p - q`; on a 7-token vocabulary with a
  3-wide head the top-k form is exact under the tail model `q̂` and its residual under the true
  sampler is the named tail term `Σ_T (q_v − ρ p_v) ∇log p_v`, smaller than the uncentred drift;
* the composition -- value and gradient of our primitive reduce to the reference
  `score_centering_loss` snippet (quoted verbatim below) for none / TIS / MIS on a random fixture;
* the wiring -- with a perturbed sampler the objective's parameter gradient equals the reference
  loss evaluated inline on the same batch (alignment of trailing head rows, ids and the sampled
  token's true logprob), the empty-shard path keeps the key set, a short record is refused;
* the producer -- the vLLM body's head parser returns `[T, k]` int32 ids / float32 logprobs (bit-exact,
  never float16) in
  rank order, keeps the sampled token's TRUE logprob when it falls outside the head, refuses a
  missing rank, and the TP-lane identity carries a digest of the record;
* the config -- defaults off, parses through the structured schema, refuses every combination the
  objective cannot compose, joins the objective identity only when on, and the engine kwargs gain
  exactly `max_logprobs` + `logprobs_mode="processed_logprobs"` and nothing else.
"""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from cosmos_framework.posttrain.algorithm.grpo import _GRPO_PUBLIC_METRICS, make_clipped_objective  # noqa: E402
from cosmos_framework.posttrain.algorithm.primitives.policy_gradient import clipped_policy_gradient_loss  # noqa: E402
from cosmos_framework.posttrain.algorithm.primitives.score_centering import (  # noqa: E402
    ScoreCenteringSpec,
    gather_head_records,
    importance_weight,
    resolve_score_centering,
    score_centering_terms,
)
from cosmos_framework.posttrain.config.rl import GRPOConfig, OPDConfig, RefConfig, ScoreCenteringConfig  # noqa: E402
from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput  # noqa: E402
from cosmos_framework.posttrain.schema import (  # noqa: E402
    FINISH_REASON_KEY,
    ROLLOUT_TOPK_IDS_KEY,
    ROLLOUT_TOPK_LOGPROBS_KEY,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
)

# --------------------------------------------------------------------------- fixtures (no transformers)

_VOCAB = 16
_COMPLETIONS: list[list[int]] = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]  # lengths 3, 2, 4
_PROMPTS: list[list[int]] = [[1, 2, 3], [1, 2, 4], [1, 2, 3, 4]]  # distinct lengths => the collate pads


class _TinyLM(nn.Module):
    """Embedding + linear head; `forward(ids) -> SimpleNamespace(logits=[B,L,V])`. Position-local, so
    padded and unpadded forwards agree exactly, which is what lets the fixture reproduce the
    objective's log-softmax bit for bit."""

    def __init__(self, vocab: int = _VOCAB, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> Any:  # ids [B,L]
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


def _batch(rewards: list[float], *, finish_reasons: list[str | None] | None = None) -> list[Trajectory]:
    """One shared group of distinct-length prompts and completions; rewards -> mean-centred advantages."""
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    trajs = []
    for k, reward in enumerate(rewards):
        policy_info: dict[str, Any] = {"prompt_ids": torch.tensor(_PROMPTS[k], dtype=torch.long)}
        if finish_reasons is not None:
            policy_info[FINISH_REASON_KEY] = finish_reasons[k]
        tr = Transition(
            obs=ObsBundle(),
            action=torch.tensor(_COMPLETIONS[k], dtype=torch.long),
            policy_info=policy_info,
            reward={"final": reward},
            done=True,
        )
        trajs.append(Trajectory(transitions=[tr], id=f"g_k{k}", group_id="g", status=Status.COMPLETED))
    group_relative(unbiased=True)(trajs)  # the deliverable recipe: A = r - mean, no std division
    return trajs


def _collate(packer: _IdsPacker, trajs: list[Trajectory]) -> dict[str, Any]:
    processed = [
        packer.get_policy_input(
            None, t.transitions[0].action, prompt_ids=t.transitions[0].policy_info["prompt_ids"].tolist()
        )
        for t in trajs
    ]
    return packer.policy_collate_fn(processed, packer.policy_compute_max_len(processed))


def _sampler_logq(
    model: nn.Module, batch: dict[str, Any], *, temperature: float, noise: float, seed: int
) -> torch.Tensor:
    """The SAMPLER's next-token log-distribution at every position, `[B,L-1,V]` float32.

    `noise == 0` reproduces the objective's own tempered log-softmax with the very ops it uses
    (`logits / T`, `log_softmax(..., dtype=float32)` over `[:, :-1]`) on the very tensor it forwards
    (the collated `input_ids`), so `q == p` holds bit for bit. `noise > 0` perturbs the logits
    before the temperature: a sampler that disagrees with the trainer everywhere.
    """
    with torch.no_grad():
        logits = model(batch["input_ids"]).logits  # [B,L,V]
        tempered = logits / temperature
        if noise:
            g = torch.Generator().manual_seed(seed)
            tempered = tempered + noise * torch.randn(tempered.shape, generator=g)
        return torch.log_softmax(tempered[:, :-1], dim=-1, dtype=torch.float32)  # [B,L-1,V]


def _stamp_sampler(
    trajs: list[Trajectory],
    model: nn.Module,
    packer: _IdsPacker,
    *,
    temperature: float,
    top_k: int,
    noise: float = 0.0,
    seed: int = 1,
    record_dtype: torch.dtype = torch.float32,
) -> list[Trajectory]:
    """Stamp what the vLLM body would: `rollout_logprobs` (log q of the sampled token, always its TRUE
    value) and the `[T, k]` head ids / logprobs in the sampler's rank order, per generated token."""
    batch = _collate(packer, trajs)
    logq = _sampler_logq(model, batch, temperature=temperature, noise=noise, seed=seed)  # [B,L-1,V]
    selection = batch["logprob_masks"][:, :-1]  # [B,L-1]
    for i, traj in enumerate(trajs):
        tr = traj.transitions[0]
        rows = logq[i][selection[i]]  # [T,V] positions whose prediction is a completion token
        assert rows.shape[0] == tr.action.numel()
        topv, topi = rows.topk(top_k, dim=-1)  # [T,k] in q's rank order
        tr.policy_info["rollout_logprobs"] = rows.gather(-1, tr.action[:, None]).squeeze(-1).tolist()
        tr.policy_info[ROLLOUT_TOPK_IDS_KEY] = topi.to(torch.int32)
        tr.policy_info[ROLLOUT_TOPK_LOGPROBS_KEY] = topv.to(record_dtype)
    return trajs


def _cfg(
    *,
    enabled: bool,
    compose: str = "none",
    top_k: int = 4,
    kl_beta: float = 0.0,
    temperature: float = 0.7,
    mask: bool = False,
) -> GRPOConfig:
    ref = RefConfig(build_ref=kl_beta > 0, update_every=0)
    return GRPOConfig(
        kl_beta=kl_beta,
        kl_coeff_clamp=5.0 if kl_beta > 0 else None,
        ref=ref,
        group_size=3,
        batch_per_replica=3,
        temperature=temperature,
        unbiased_advantage=True,
        mask_truncated_completions=mask,
        score_centering=ScoreCenteringConfig(enabled=enabled, compose=compose, top_k=top_k),
    )


def _grads(policy: nn.Module, loss: torch.Tensor) -> list[torch.Tensor]:
    policy.zero_grad()
    loss.backward()
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in policy.parameters()]


def _perturbed_copy(model: nn.Module, scale: float = 0.5, seed: int = 7) -> nn.Module:
    twin = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in twin.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return twin


# --------------------------------------------------------------------------- the reference, verbatim


def _reference_score_centering_loss(
    train_logp, samp_logp, topk_ids, sampled_token, samp_token_logp, advantage, weight_fn=torch.ones_like, eps=1e-6
):
    """The reference README snippet, copied verbatim (PyTorch flavour) as the oracle."""
    head_logp = train_logp.gather(-1, topk_ids)
    token_logp = train_logp.gather(-1, sampled_token[..., None])[..., 0]
    p_head, q_head = head_logp.exp(), samp_logp.exp()
    p_tail = (1 - p_head.sum(-1)).clamp_min(eps)
    q_tail = (1 - q_head.sum(-1)).clamp_min(eps)
    rho = q_tail / p_tail
    alpha = rho * weight_fn(1 / rho)
    head_weight = weight_fn((head_logp - samp_logp).exp())
    residual = q_head * head_weight - alpha[..., None] * p_head
    correction = (residual.detach() * head_logp).sum(-1)
    token_weight = weight_fn((token_logp - samp_token_logp).exp())
    return -advantage.detach() * (token_weight.detach() * token_logp - correction)


def _reference_tis_weight(r):
    return r.clamp_max(2.0)


def _reference_mis_weight(r):
    return torch.where((r >= 0.5) & (r <= 5.0), r, 0.0)


_REFERENCE_WEIGHT_FN = {"none": torch.ones_like, "tis": _reference_tis_weight, "mis": _reference_mis_weight}


# --------------------------------------------------------------------------- (i) identity at q == p


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("compose", ["none", "tis", "mis"])
@pytest.mark.parametrize("kl_beta", [0.0, 0.05])
def test_correction_is_exactly_zero_and_loss_bit_identical_when_sampler_equals_trainer(compose: str, kl_beta: float):
    """`q == p` => `rho == 1`, every residual `0.0`, correction `0.0`, weight `1.0`: the loss and the
    parameter gradients equal the centering-off objective bit for bit, KL anchor or not."""
    model, packer = _TinyLM(), _IdsPacker()
    trajs = _stamp_sampler(_batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=4)

    on = make_clipped_objective(packer, _cfg(enabled=True, compose=compose, kl_beta=kl_beta))
    off = make_clipped_objective(packer, _cfg(enabled=False, kl_beta=kl_beta))
    if kl_beta > 0:
        reference = _perturbed_copy(model)
        on.reference_scorer.attach(reference)
        off.reference_scorer.attach(copy.deepcopy(reference))

    loss_on, m_on = on.fn(model, copy.deepcopy(trajs))
    grads_on = _grads(model, loss_on)
    loss_off, m_off = off.fn(model, copy.deepcopy(trajs))
    grads_off = _grads(model, loss_off)

    assert torch.equal(loss_on, loss_off), (float(loss_on), float(loss_off))
    assert all(torch.equal(a, b) for a, b in zip(grads_on, grads_off, strict=True))
    assert any(g.abs().max() > 0 for g in grads_off)  # a non-vacuous comparison
    assert m_on["grpo_sc_tokens"] == m_on["grpo_valid_tokens"] == 9  # 3 + 2 + 4 completion tokens
    assert m_on["grpo_sc_correction_abs"] == 0.0
    assert m_on["grpo_sc_rho_mean"] == 1.0 and m_on["grpo_sc_rho_max"] == 1.0
    assert m_on["grpo_sc_token_weight"] == 1.0 and m_on["grpo_sc_kl_head"] == 0.0
    # Same head mass on both sides (q == p); above the uniform-head mass k/V of a near-uniform random LM.
    assert m_on["grpo_sc_head_mass_q"] == m_on["grpo_sc_head_mass_p"] > 4 / _VOCAB
    assert m_on["grpo_sc_out_of_head_frac"] >= 0.0
    if kl_beta > 0:
        assert m_on["grpo_kl_mean"] == m_off["grpo_kl_mean"] > 0.0  # the KL term is untouched, and present
    # Off: every centering key is a zero of zero weight; the key SET is config-invariant.
    assert m_off["grpo_sc_tokens"] == 0 and m_off["grpo_sc_correction_abs"] == 0.0 and m_off["grpo_sc_rho_max"] == 0.0
    assert set(m_on) == set(m_off) == set(_GRPO_PUBLIC_METRICS)
    # Under centering the gap diagnostic pairs TEMPERED logprobs on both sides: q == p => gap 0.
    assert m_on["grpo_logprob_gap_tokens"] == 9 and m_on["grpo_logprob_gap_mean"] == 0.0

    # float16 transport of the SAME q: the correction measures the rounding (non-zero) -- the reason the
    # producer must never round the head on the wire (test_vllm_rollout_records_... asserts float32, bit-exact).
    trajs16 = _stamp_sampler(
        _batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=4, record_dtype=torch.float16
    )
    _, m16 = on.fn(model, trajs16)
    assert 0.0 < m16["grpo_sc_correction_abs"] < 0.05, m16["grpo_sc_correction_abs"]
    assert abs(m16["grpo_sc_rho_mean"] - 1.0) < 0.05


# --------------------------------------------------------------------------- (iv) regression with the flag off


@pytest.mark.L0
@pytest.mark.CPU
def test_disabled_flag_ignores_sampler_records_and_keeps_the_loss_bit_identical():
    """Centering off: a batch carrying head records trains exactly like one that never did (the
    records are read by nothing but the always-on gap diagnostic, which sees `rollout_logprobs` only)."""
    model, packer = _TinyLM(), _IdsPacker()
    plain = _batch([1.0, 0.0, 1.0])
    stamped = _stamp_sampler(_batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=4, noise=0.6)
    off = make_clipped_objective(packer, _cfg(enabled=False))

    loss_plain, m_plain = off.fn(model, plain)
    grads_plain = _grads(model, loss_plain)
    loss_stamped, m_stamped = off.fn(model, stamped)
    grads_stamped = _grads(model, loss_stamped)

    assert torch.equal(loss_plain, loss_stamped)
    assert all(torch.equal(a, b) for a, b in zip(grads_plain, grads_stamped, strict=True))
    assert m_plain["grpo_sc_tokens"] == m_stamped["grpo_sc_tokens"] == 0
    for key in _GRPO_PUBLIC_METRICS:
        if key.startswith("grpo_sc_"):
            assert m_plain[key] == m_stamped[key] == 0
    # The pre-existing diagnostic behaviour is unchanged: recorded logprobs feed the gap, nothing else.
    assert m_plain["grpo_logprob_gap_tokens"] == 0 and m_stamped["grpo_logprob_gap_tokens"] == 9
    assert m_stamped["grpo_logprob_gap_mean"] > 0.0
    # `resolve_score_centering` is the one switch: None off, a validated spec on.
    assert resolve_score_centering(ScoreCenteringConfig()) is None and resolve_score_centering(None) is None
    assert resolve_score_centering(ScoreCenteringConfig(enabled=True, top_k=7, compose="tis")) == ScoreCenteringSpec(
        top_k=7, compose="tis"
    )


# --------------------------------------------------------------------------- (ii) zero expected gradient


def _grad_wrt_logits(
    theta: torch.Tensor,
    token: int,
    *,
    logq: torch.Tensor,
    topi: torch.Tensor,
    topv: torch.Tensor,
    spec: ScoreCenteringSpec | None,
) -> torch.Tensor:  # theta [V] -> [V]
    """`∇_θ L(token)` at advantage 1 through the SAME primitive composition the objective uses."""
    th = theta.clone().requires_grad_(True)
    logp = torch.log_softmax(th, -1)  # [V]
    current = logp[token : token + 1]  # [1]
    token_weight = centering = None
    if spec is not None:
        terms = score_centering_terms(logp[topi][None], topv[None], current, logq[token : token + 1], spec=spec)
        token_weight, centering = terms.token_weight, terms.correction
    loss = clipped_policy_gradient_loss(
        current,
        current.detach(),
        torch.ones(1, dtype=theta.dtype),
        0.2,
        0.2,
        lower_bound_ratio=3.0,
        token_weight=token_weight,
        centering=centering,
    )
    return torch.autograd.grad(loss.sum(), th)[0]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("compose", ["none", "tis", "mis"])
def test_centered_score_has_zero_expected_gradient_under_the_sampler_on_a_three_token_vocab(compose: str):
    """V = 3, k = 2: the tail is one token, so the tail model is exact and `E_q[∇L] = 0` for
    centering alone AND composed with TIS / MIS (float64 exact expectation, then 10^6 draws).
    Plain REINFORCE under the same sampler drifts by exactly `p − q`."""
    V, k = 3, 2
    g = torch.Generator().manual_seed(0)
    theta = torch.randn(V, generator=g, dtype=torch.float64)
    logq = torch.log_softmax(theta + 0.7 * torch.randn(V, generator=g, dtype=torch.float64), -1)  # a wrong sampler
    q, p = logq.exp(), torch.log_softmax(theta, -1).exp()
    topv, topi = logq.topk(k)
    spec = ScoreCenteringSpec(top_k=k, compose=compose)

    grads_c = torch.stack([_grad_wrt_logits(theta, v, logq=logq, topi=topi, topv=topv, spec=spec) for v in range(V)])
    grads_u = torch.stack([_grad_wrt_logits(theta, v, logq=logq, topi=topi, topv=topv, spec=None) for v in range(V)])
    drift_c = (q[:, None] * grads_c).sum(0)  # exact E_q[∇L], [V]
    drift_u = (q[:, None] * grads_u).sum(0)
    assert float(drift_c.abs().max()) < 1e-12, drift_c
    assert torch.allclose(drift_u, p - q, atol=1e-12) and float(drift_u.abs().max()) > 1e-2, drift_u

    # Many samples: the Monte-Carlo mean of the centred per-token gradient over draws from q.
    n = 1_000_000
    counts = torch.bincount(torch.multinomial(q, n, replacement=True, generator=g), minlength=V).double() / n
    mc_c = (counts[:, None] * grads_c).sum(0)
    mc_u = (counts[:, None] * grads_u).sum(0)
    assert float(mc_c.abs().max()) < 5e-3, mc_c  # ~5 standard errors of a zero-mean estimator
    assert float((mc_u - (p - q)).abs().max()) < 5e-3, mc_u


@pytest.mark.L0
@pytest.mark.CPU
def test_top_k_form_is_exact_under_the_tail_model_and_its_residual_under_the_true_sampler_is_the_tail_term():
    """V = 7, k = 3: exact under `q̂` (head = q, tail = ρ p); under the TRUE q the residual drift is
    `Σ_{v∈T} (q_v − ρ p_v) ∇log p_v`, and it is smaller than the uncentred drift `p − q`."""
    V, k = 7, 3
    g = torch.Generator().manual_seed(3)
    theta = torch.randn(V, generator=g, dtype=torch.float64)
    logq = torch.log_softmax(theta + 0.6 * torch.randn(V, generator=g, dtype=torch.float64), -1)
    q, logp = logq.exp(), torch.log_softmax(theta, -1)
    p = logp.exp()
    topv, topi = logq.topk(k)
    head = torch.zeros(V, dtype=torch.bool)
    head[topi] = True
    rho = (1 - q[head].sum()) / (1 - p[head].sum())
    q_hat = torch.where(head, q, rho * p)
    assert abs(float(q_hat.sum()) - 1.0) < 1e-12
    spec = ScoreCenteringSpec(top_k=k)

    grads_c = torch.stack([_grad_wrt_logits(theta, v, logq=logq, topi=topi, topv=topv, spec=spec) for v in range(V)])
    drift_under_q_hat = (q_hat[:, None] * grads_c).sum(0)
    assert float(drift_under_q_hat.abs().max()) < 1e-12, drift_under_q_hat

    drift_under_q = (q[:, None] * grads_c).sum(0)
    # ∇_θ log p_v = e_v − p, so the named tail term is Σ_T (q_v − ρ p_v)(e_v − p); the LOSS is −log p,
    # so the drift of the loss gradient is minus that.
    score = torch.eye(V, dtype=torch.float64) - p[None, :]  # [V,V] row v = ∇_θ log p_v
    tail_term = -(((q - rho * p) * (~head))[:, None] * score).sum(0)
    assert torch.allclose(drift_under_q, tail_term, atol=1e-12), (drift_under_q, tail_term)
    assert float(drift_under_q.abs().max()) < float((p - q).abs().max())


# --------------------------------------------------------------------------- (iii) reduces to the reference


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("compose", ["none", "tis", "mis"])
def test_composition_matches_the_reference_score_centering_loss(compose: str):
    """Value (REINFORCE form) and gradient (the objective's surrogate + hooks) equal the reference
    snippet for the identity, TIS(2.0) and MIS[0.5, 5] weight functions on a random fixture."""
    N, V, k = 6, 9, 4
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(N, V, generator=g, dtype=torch.float64).requires_grad_(True)
    train_logp = torch.log_softmax(logits, -1)
    samp_full = torch.log_softmax(logits.detach() + 0.8 * torch.randn(N, V, generator=g, dtype=torch.float64), -1)
    samp_logp, topk_ids = samp_full.topk(k, dim=-1)
    sampled = torch.multinomial(samp_full.exp(), 1, generator=g)[:, 0]  # [N] drawn from the sampler
    samp_token_logp = samp_full.gather(-1, sampled[:, None])[:, 0]
    advantage = torch.randn(N, generator=g, dtype=torch.float64)

    ref = _reference_score_centering_loss(
        train_logp, samp_logp, topk_ids, sampled, samp_token_logp, advantage, weight_fn=_REFERENCE_WEIGHT_FN[compose]
    )  # [N]

    head_logp = train_logp.gather(-1, topk_ids)  # [N,k]
    token_logp = train_logp.gather(-1, sampled[:, None])[:, 0]  # [N]
    spec = ScoreCenteringSpec(top_k=k, compose=compose)
    terms = score_centering_terms(head_logp, samp_logp, token_logp, samp_token_logp, spec=spec)
    expected_weight = _REFERENCE_WEIGHT_FN[compose]((token_logp - samp_token_logp).exp())
    assert torch.equal(terms.token_weight, expected_weight.detach())
    assert torch.equal(
        importance_weight(torch.tensor([0.1, 1.0, 3.0, 9.0], dtype=torch.float64), spec),
        _REFERENCE_WEIGHT_FN[compose](torch.tensor([0.1, 1.0, 3.0, 9.0], dtype=torch.float64)),
    )
    ours_value = -advantage * (terms.token_weight * token_logp - terms.correction)
    assert torch.allclose(ours_value, ref, atol=1e-12, rtol=0.0)

    ours_loss = clipped_policy_gradient_loss(
        token_logp,
        token_logp.detach(),
        advantage,
        0.2,
        0.2,
        lower_bound_ratio=3.0,
        token_weight=terms.token_weight,
        centering=terms.correction,
    ).sum()
    (g_ours,) = torch.autograd.grad(ours_loss, logits, retain_graph=True)
    (g_ref,) = torch.autograd.grad(ref.sum(), logits)
    assert torch.allclose(g_ours, g_ref, atol=1e-12, rtol=0.0)
    assert float(g_ref.abs().max()) > 1e-3  # non-vacuous
    if compose != "none":
        assert not torch.equal(terms.token_weight, torch.ones_like(terms.token_weight))


# --------------------------------------------------------------------------- the objective wiring


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("compose", ["none", "tis", "mis"])
def test_objective_gradient_matches_the_reference_loss_evaluated_inline_on_the_batch(compose: str):
    """Perturbed sampler through the whole objective (collate, forward, trailing head rows, ids, the
    sampled token's true logprob, token-mean reduction) equals the reference loss computed inline
    on the same positions: `Σ_t ref_t / N`."""
    model, packer = _TinyLM(), _IdsPacker()
    trajs = _stamp_sampler(_batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=4, noise=0.6, seed=5)
    objective = make_clipped_objective(packer, _cfg(enabled=True, compose=compose, top_k=4))
    loss, metrics = objective.fn(model, copy.deepcopy(trajs))
    grads = _grads(model, loss)
    assert metrics["grpo_sc_tokens"] == 9 and metrics["grpo_sc_correction_abs"] > 0.0
    assert metrics["grpo_sc_kl_head"] > 0.0 and metrics["grpo_sc_rho_mean"] != 1.0
    if compose == "none":
        assert metrics["grpo_sc_token_weight"] == 1.0

    # Inline reference on the same collated batch.
    batch = _collate(packer, trajs)
    logits = model(batch["input_ids"]).logits  # [B,L,V]
    train_logp = torch.log_softmax((logits / 0.7)[:, :-1], dim=-1, dtype=torch.float32)  # [B,L-1,V]
    selection = batch["logprob_masks"][:, :-1]
    rows_logp = train_logp[selection]  # [N,V]
    sampled = batch["input_ids"][:, 1:][selection]  # [N]
    advantages = torch.cat(
        [torch.full((t.transitions[0].action.numel(),), float(t.transitions[0].advantage)) for t in trajs]
    )  # [N]
    topk_ids = torch.cat([t.transitions[0].policy_info[ROLLOUT_TOPK_IDS_KEY].long() for t in trajs])  # [N,k]
    samp_logp = torch.cat([t.transitions[0].policy_info[ROLLOUT_TOPK_LOGPROBS_KEY] for t in trajs])  # [N,k]
    samp_token_logp = torch.tensor([v for t in trajs for v in t.transitions[0].policy_info["rollout_logprobs"]])
    ref = _reference_score_centering_loss(
        rows_logp, samp_logp, topk_ids, sampled, samp_token_logp, advantages, weight_fn=_REFERENCE_WEIGHT_FN[compose]
    )
    grads_ref = _grads(model, ref.sum() / ref.numel())  # token-mean over one rank
    for ours, theirs in zip(grads, grads_ref, strict=True):
        assert torch.allclose(ours, theirs, atol=1e-6, rtol=1e-5), (ours, theirs)
    assert any(g.abs().max() > 1e-4 for g in grads_ref)
    # The recorded head mass is the sampler's own: Σ_H q over the stamped records.
    assert metrics["grpo_sc_head_mass_q"] == pytest.approx(float(samp_logp.exp().sum(-1).mean()), abs=1e-6)


@pytest.mark.L0
@pytest.mark.CPU
def test_records_are_validated_and_the_empty_shard_path_keeps_the_key_set():
    """A missing / short / mis-shaped record is refused before the loss; every completion cut under the
    mask takes the empty-shard path with the same key set as a trained chunk."""
    model, packer = _TinyLM(), _IdsPacker()
    objective = make_clipped_objective(packer, _cfg(enabled=True, top_k=4, mask=True))

    missing = _batch([1.0, 0.0, 1.0])  # no records at all
    with pytest.raises(Exception, match="rollout_logprobs"):
        objective.fn(model, missing)
    partial = _stamp_sampler(_batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=4)
    del partial[1].transitions[0].policy_info[ROLLOUT_TOPK_IDS_KEY]
    with pytest.raises(Exception, match="carries none"):
        objective.fn(model, partial)
    short = _stamp_sampler(_batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=4)
    info = short[2].transitions[0].policy_info
    info[ROLLOUT_TOPK_IDS_KEY] = info[ROLLOUT_TOPK_IDS_KEY][:-1]
    info[ROLLOUT_TOPK_LOGPROBS_KEY] = info[ROLLOUT_TOPK_LOGPROBS_KEY][:-1]
    with pytest.raises(Exception, match="rows <"):
        objective.fn(model, short)
    narrow = _stamp_sampler(_batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=3)  # k=3 for a k=4 objective
    with pytest.raises(Exception, match=r"\[T, 4\]"):
        objective.fn(model, narrow)

    trained = _stamp_sampler(_batch([1.0, 0.0, 1.0]), model, packer, temperature=0.7, top_k=4)
    _, metrics_trained = objective.fn(model, trained)
    cut = _stamp_sampler(
        _batch([1.0, 0.0, 1.0], finish_reasons=["length", "length", "length"]), model, packer, temperature=0.7, top_k=4
    )
    loss_empty, metrics_empty = objective.fn(model, cut)
    assert float(loss_empty.detach()) == 0.0 and metrics_empty["grpo_n"] == 0
    assert set(metrics_empty) == set(metrics_trained) == set(_GRPO_PUBLIC_METRICS)
    assert metrics_empty["grpo_sc_tokens"] == 0 and metrics_trained["grpo_sc_tokens"] == 9

    # The pure gather refuses the same shapes without an objective around it.
    selection = torch.tensor([[False, True, True]])  # one sequence, 2 masked tokens
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)  # 3 rows: trailing 2 are used
    lps = torch.tensor([[-0.1, -3.0], [-0.2, -2.0], [-0.3, -1.0]], dtype=torch.float16)
    got_ids, got_lps = gather_head_records(
        [ids], [lps], selection, top_k=2, device=torch.device("cpu"), dtype=torch.float32
    )
    assert got_ids.tolist() == [[3, 4], [5, 6]] and got_ids.dtype == torch.long and got_lps.dtype == torch.float32
    with pytest.raises(ValueError, match="invalid sampler logprob"):
        gather_head_records(
            [ids], [lps.float().fill_(0.5)], selection, top_k=2, device=torch.device("cpu"), dtype=torch.float32
        )
    neg_inf = lps.float()
    neg_inf[0, 1] = float("-inf")  # a masked-out sampler token is a zero mass, not an error
    gather_head_records([ids], [neg_inf], selection, top_k=2, device=torch.device("cpu"), dtype=torch.float32)


# --------------------------------------------------------------------------- the producer (vLLM body)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_rollout_records_the_sampler_head_in_rank_order_and_keeps_the_true_sampled_logprob():
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    def lp(value: float, rank: int) -> Any:
        return SimpleNamespace(logprob=value, rank=rank, decoded_token=None)

    # Position 0: the sampled token (7) is rank 1. Position 1: the sampled token (3) is OUTSIDE the
    # head (rank 5) -- vLLM then returns k + 1 entries.
    out = SimpleNamespace(
        token_ids=[7, 3],
        logprobs=[
            {7: lp(-0.1, 1), 2: lp(-2.5, 2), 9: lp(-3.0, 3)},
            {4: lp(-0.2, 1), 8: lp(-1.9, 2), 1: lp(-2.2, 3), 3: lp(-6.0, 5)},
        ],
    )
    ids, lps = VLLMRollout._topk_head(out, 3)
    assert ids.dtype == torch.int32 and lps.dtype == torch.float32 and tuple(ids.shape) == tuple(lps.shape) == (2, 3)
    assert ids.tolist() == [[7, 2, 9], [4, 8, 1]]
    # bit-exact float32 on the wire (FAIL-on-old: the float16 producer rounded -2.5/-1.9/-2.2 and had dtype float16)
    assert torch.equal(lps, torch.tensor([[-0.1, -2.5, -3.0], [-0.2, -1.9, -2.2]], dtype=torch.float32))
    assert VLLMRollout._token_logprobs(out) == [-0.1, -6.0]  # the sampled token's TRUE logprob, head or not

    with pytest.raises(RuntimeError, match=r"expected 1\.\.3"):
        VLLMRollout._topk_head(SimpleNamespace(token_ids=[7], logprobs=[{7: lp(-0.1, 1), 9: lp(-3.0, 3)}]), 3)
    with pytest.raises(RuntimeError, match="no logprobs"):
        VLLMRollout._topk_head(SimpleNamespace(token_ids=[7], logprobs=None), 3)
    with pytest.raises(RuntimeError, match="has no rank"):
        VLLMRollout._topk_head(SimpleNamespace(token_ids=[7], logprobs=[{7: SimpleNamespace(logprob=-0.1)}]), 1)
    empty_ids, empty_lps = VLLMRollout._topk_head(SimpleNamespace(token_ids=[], logprobs=[]), 3)
    assert tuple(empty_ids.shape) == tuple(empty_lps.shape) == (0, 3)

    # The TP-lane output identity digests the record so every lane must agree before publication.
    tr = Transition(
        obs=ObsBundle(),
        action=torch.tensor([7, 3]),
        policy_info={
            "prompt_ids": torch.tensor([1, 2]),
            "completion_ids": [7, 3],
            "completion_text": "x",
            "completion_boundary": {
                "prompt_token_count": 2,
                "completion_token_count": 2,
                "full_sequence_token_count": 4,
            },
            "rollout_logprobs": [-0.1, -6.0],
            "weight_version": 0,
            FINISH_REASON_KEY: "stop",
            ROLLOUT_TOPK_IDS_KEY: ids,
            ROLLOUT_TOPK_LOGPROBS_KEY: lps,
        },
        done=True,
    )
    identity = VLLMRollout._training_output_identity(
        [Trajectory(transitions=[tr], id="t", group_id="g", status=Status.COMPLETED)]
    )
    row = identity["trajectories"][0]
    assert row["rollout_topk_sha256"] == hashlib.sha256(ids.numpy().tobytes() + lps.numpy().tobytes()).hexdigest()
    del tr.policy_info[ROLLOUT_TOPK_IDS_KEY], tr.policy_info[ROLLOUT_TOPK_LOGPROBS_KEY]
    assert (
        "rollout_topk_sha256"
        not in VLLMRollout._training_output_identity(
            [Trajectory(transitions=[tr], id="t", group_id="g", status=Status.COMPLETED)]
        )["trajectories"][0]
    )

    # Constructor knob validation (a fake packer suffices: the body only asks it about special tokens).
    fake_packer = SimpleNamespace(adds_special_tokens=lambda: False)
    with pytest.raises(ValueError, match="topk_logprobs"):
        VLLMRollout(object(), packer=fake_packer, topk_logprobs=-1)  # type: ignore[arg-type]
    head_engine = {"max_logprobs": 128, "logprobs_mode": "processed_logprobs"}
    body = VLLMRollout(object(), packer=fake_packer, topk_logprobs=128, engine_config=head_engine)  # type: ignore[arg-type]
    assert body._topk_logprobs == 128 and VLLMRollout(object(), packer=fake_packer)._topk_logprobs == 0  # type: ignore[arg-type]
    # The likelihood convention is checked at construction: no engine identity,
    # raw logprobs, or a too-small max_logprobs all refuse; the same three refusals guard PPO's behavior head.
    with pytest.raises(ValueError, match="engine_config"):
        VLLMRollout(object(), packer=fake_packer, topk_logprobs=128)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="processed_logprobs"):
        VLLMRollout(
            object(),
            packer=fake_packer,
            topk_logprobs=128,
            engine_config={"max_logprobs": 128, "logprobs_mode": "raw_logprobs"},
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_logprobs"):
        VLLMRollout(
            object(),
            packer=fake_packer,
            topk_logprobs=128,
            engine_config={"max_logprobs": 20, "logprobs_mode": "processed_logprobs"},
        )  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the config


@pytest.mark.L0
@pytest.mark.CPU
def test_config_defaults_off_parses_refuses_uncomposable_combinations_and_shapes_identity_and_engine():
    from omegaconf import OmegaConf

    from cosmos_framework.posttrain.config.build import build_objective, rollout_topk_logprobs, vllm_engine_kwargs
    from cosmos_framework.posttrain.config.rl import RolloutConfig

    default = GRPOConfig().score_centering
    assert default == ScoreCenteringConfig()
    assert (default.enabled, default.top_k, default.compose, default.tis_clip, default.mis_low, default.mis_high) == (
        False,
        128,
        "none",
        2.0,
        0.5,
        5.0,
    )
    schema = OmegaConf.structured(GRPOConfig)
    merged = OmegaConf.merge(
        schema,
        OmegaConf.from_dotlist(
            ["score_centering.enabled=true", "score_centering.compose=tis", "score_centering.top_k=64"]
        ),
    )
    resolved = OmegaConf.to_object(merged)
    assert isinstance(resolved, GRPOConfig)
    assert (resolved.score_centering.enabled, resolved.score_centering.compose, resolved.score_centering.top_k) == (
        True,
        "tis",
        64,
    )

    with pytest.raises(ValueError, match="compose"):
        ScoreCenteringConfig(compose="ppo")
    with pytest.raises(ValueError, match="tis_clip"):
        ScoreCenteringConfig(tis_clip=0.5)
    with pytest.raises(ValueError, match="mis_low/mis_high"):
        ScoreCenteringConfig(mis_low=1.5)
    with pytest.raises(ValueError, match="mis_low/mis_high"):
        ScoreCenteringConfig(mis_high=0.9)
    with pytest.raises(ValueError, match="top_k"):
        ScoreCenteringConfig(top_k=0)
    with pytest.raises(TypeError, match="enabled"):
        ScoreCenteringConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tis_clip"):
        ScoreCenteringConfig(tis_clip=float("inf"))
    with pytest.raises(ValueError, match="top_k"):
        ScoreCenteringSpec(top_k=0)

    on = ScoreCenteringConfig(enabled=True)
    with pytest.raises(ValueError, match="does not consume clipped-surrogate settings.*score_centering"):
        GRPOConfig(grpo_surrogate="legacy_pg", score_centering=on)
    with pytest.raises(ValueError, match="use_rollout_logprobs_for_loss"):
        GRPOConfig(score_centering=on, use_rollout_logprobs_for_loss=True)
    with pytest.raises(ValueError, match="importance_sampling_level"):
        GRPOConfig(score_centering=on, importance_sampling_level="sequence", lower_bound_ratio=0.0)
    with pytest.raises(ValueError, match="n_mini"):
        GRPOConfig(
            score_centering=on, batch_per_replica=32, mini_batch_per_replica=16, certified_deterministic_forward=True
        )
    with pytest.raises(ValueError, match="OPD does not consume score_centering"):
        OPDConfig(score_centering=on)
    assert OPDConfig().score_centering.enabled is False
    with pytest.raises(TypeError, match="ScoreCenteringConfig"):
        GRPOConfig(score_centering={"enabled": True})  # type: ignore[arg-type]
    GRPOConfig(score_centering=on)  # the composable default recipe is accepted

    # The objective factory backstops for a hand-built cfg that bypasses GRPOConfig.
    with pytest.raises(ValueError, match="use_rollout_logprobs_for_loss"):
        make_clipped_objective(
            _IdsPacker(),
            SimpleNamespace(
                temperature=1.0,
                epsilon_low=0.2,
                epsilon_high=0.2,
                score_centering=on,
                use_rollout_logprobs_for_loss=True,
            ),
        )

    # Identity: absent off (every earlier checkpoint), present with the shaping knobs on.
    off_obj = build_objective(GRPOConfig(), "unused", packer=object())
    on_obj = build_objective(
        GRPOConfig(score_centering=ScoreCenteringConfig(enabled=True, top_k=32, compose="mis")),
        "unused",
        packer=object(),
    )
    assert "score_centering" not in off_obj.identity
    assert on_obj.identity["score_centering"] == {
        "top_k": 32,
        "compose": "mis",
        "tis_clip": 2.0,
        "mis_low": 0.5,
        "mis_high": 5.0,
    }
    assert set(on_obj.identity) == set(off_obj.identity) | {"score_centering"}

    # Rollout plumbing: nothing requested off; k requested on; the engine gains exactly two keys.
    assert rollout_topk_logprobs(GRPOConfig()) == 0
    assert rollout_topk_logprobs(GRPOConfig(score_centering=ScoreCenteringConfig(enabled=True, top_k=64))) == 64
    base = vllm_engine_kwargs(RolloutConfig())
    assert "max_logprobs" not in base and "logprobs_mode" not in base
    assert vllm_engine_kwargs(RolloutConfig(), topk_logprobs=64) == {
        **base,
        "max_logprobs": 64,
        "logprobs_mode": "processed_logprobs",
    }
    assert vllm_engine_kwargs(RolloutConfig(), topk_logprobs=0) == base
    with pytest.raises(ValueError, match="topk_logprobs"):
        vllm_engine_kwargs(RolloutConfig(), topk_logprobs=-1)


@pytest.mark.L0
@pytest.mark.CPU
def test_out_of_vocabulary_head_ids_are_refused_inside_the_rank_converged_preparation():
    """An id >= the trainer vocabulary must fail in the converged
    preparation on every rank, never as a rank-local CUDA gather fault. FAIL-ON-OLD: without ``vocab_size``
    the same record passes ``gather_head_records`` and only the gather can trip."""
    import math as _math

    from cosmos_framework.posttrain.algorithm.primitives.score_centering import (
        completion_token_and_head_logprobs,
        gather_head_records,
    )

    selection = torch.tensor([[True, True]])  # one sequence, two masked completion tokens
    ids = [torch.tensor([[0, 1], [2, 9]], dtype=torch.int32)]  # 9 is outside a 5-token vocabulary
    lps = [torch.full((2, 2), -1.0)]
    with pytest.raises(ValueError, match=r"outside the trainer vocabulary \[0, 5\)"):
        gather_head_records(ids, lps, selection, top_k=2, device=torch.device("cpu"), dtype=torch.float32, vocab_size=5)
    got_ids, _ = gather_head_records(
        ids, lps, selection, top_k=2, device=torch.device("cpu"), dtype=torch.float32
    )  # old path
    assert got_ids.shape == (2, 2)
    logits = torch.zeros(1, 3, 5)
    input_ids = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError, match="outside the trainer vocabulary"):
        completion_token_and_head_logprobs(logits, input_ids, selection, got_ids, temperature=1.0, dtype=torch.float32)
    good = torch.tensor([[0, 1], [2, 3]])
    tok, head = completion_token_and_head_logprobs(
        logits, input_ids, selection, good, temperature=1.0, dtype=torch.float32
    )
    assert tok.shape == (2,) and head.shape == (2, 2) and torch.allclose(head, torch.full((2, 2), _math.log(0.2)))
