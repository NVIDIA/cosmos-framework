# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
MoE Stability Callback
======================
Monitors whether the MoE router is staying healthy over the course of training.
A healthy router distributes tokens reasonably evenly, keeps all experts alive,
and remains uncertain enough (high entropy) that it is still learning to route.

The following metrics are tracked per layer, per tower (und / gen):

  Dead Expert Rate
  ----------------
  Fraction of experts receiving fewer than 10% of their fair-share of tokens
  (i.e. load fraction f_i < 0.1 / N). A dead expert has been effectively shut
  out by the router — it gets no gradient signal and its capacity is wasted.
  Ideal = 0. A rising dead-expert rate in the gen tower during early training
  is a common failure mode.

  Load Imbalance Factor (LIF)
  ---------------------------
  N * max(f_i), where f_i is the fraction of tokens routed to expert i.
  Measures how much the busiest expert is overloaded relative to uniform.
  LIF = 1.0 is perfect balance; <= 1.3 is healthy; > 3.0 indicates severe
  collapse onto a small set of experts. This is the same quantity watched by
  the load-balancing loss, but measured empirically rather than from the loss
  objective.

  Router Entropy (normalized)
  ---------------------------
  Mean per-token Shannon entropy of the full routing distribution, divided by
  log(N) to put it on a [0, 1] scale. H = 1 means the router is maximally
  uncertain (uniform over all experts); H = 0 means it always picks the same
  expert. Early in training entropy is high; we want it to stay reasonably
  high (> ~0.7) so the router continues to explore. A sudden drop signals
  routing collapse.

  Soft-vs-Hard Effective Experts (normalized)
  -------------------------------------------
  Soft and hard effective experts separate what the router *considers* (full
  probability distribution, before dispatch) from what top-k dispatch *actually
  uses* (empirical token-to-expert assignment, after dispatch). Both are
  expressed as a fraction of N, so they sit on the same axis as
  router_entropy_normalized. Their lower bounds differ slightly:
    soft_eff_normalized is bounded in [1/N, 1].
    hard_eff_normalized is bounded in [K/N, 1] — top-K dispatch always engages
      at least K experts in aggregate (the floor case is when every token
      picks the same K-expert subset).

      soft_eff_normalized = mean_t exp(H(p_t)) / N
        Average per-token router perplexity, divided by N. Asks: what fraction
        of experts is the router *considering* on a typical token? Computed
        as sum_per_token_soft_eff / total_tokens / N. Note: the unnormalized
        numerator is NOT exp of the mean entropy — by Jensen,
        mean_t exp(H_t) >= exp(mean_t H_t), and the gap matters when
        per-token entropies are heterogeneous.

      hard_eff_normalized = exp(H(f)) / N
        where f_i is the empirical fraction of *expert assignments* (not
        tokens) that went to expert i: f_i = tokens_per_expert_i / (T * K).
        Perplexity of the buffer-wide dispatch distribution, divided by N.
        Asks: what fraction of experts is top-k *actually* engaging across the
        buffer? A smoother sibling of LIF: where LIF watches the busiest
        expert, hard_eff watches the spread of the whole load distribution.

  Interpretation (high/low refer to values close to 1 vs close to 1/N):

      high soft_eff, high hard_eff
        Router considers many experts; top-k dispatch also uses many experts.
        Broadly healthy routing.
      low  soft_eff, low  hard_eff
        Router is confident or collapsed in probability space; dispatch is
        also concentrated. Entropy, LIF, and hard usage all agree that
        routing is narrow.
      high soft_eff, low  hard_eff
        Router distribution is broad, but top-k dispatch is concentrated —
        the "hidden top-k concentration" case where entropy can look healthy
        while LIF and co-activation are high.
      low  soft_eff, high hard_eff
        Less common: each token has a sharp router distribution, but
        different tokens choose different experts. Per-token confidence with
        buffer-wide diversity.

  Honest Effective Experts (N_eff) and Effective Parameters
  ---------------------------------------------------------
  Uses the soft clean gate g[t, e] (the pre-dispatch softmax over experts, summing
  to 1 per token) to estimate how many experts a layer *honestly* uses, discounting
  coverage that isn't actually token-driven.

      P[e]      = mean_t g[t, e]                      # marginal usage distribution
      H_marg    = -sum_e P[e] ln P[e]                 # entropy of overall usage (nats)
      H_within  = mean_t ( -sum_e g[t,e] ln g[t,e] )  # avg per-token entropy (nats)
      N_cov     = exp(H_marg)                          # coverage, in [1, N]
      rho       = (H_marg - H_within) / H_marg         # conditionality, in [0, 1]
      N_eff     = k + (N_cov - k) * rho                # honest experts, in [~k, N]

  N_eff is logged normalized as honest_eff_normalized = N_eff / N (a fraction in
  [~k/N, 1]), sitting on the same axis as soft_eff_normalized / hard_eff_normalized.

  rho (logged as router_token_mi_normalized) is the normalized mutual information between
  token identity and expert routing:
  rho -> 1 when routing is genuinely token-dependent, rho -> 0 when every token routes
  the same way (so wide coverage that is not input-driven is not rewarded). By concavity
  of entropy H_marg >= H_within, so rho >= 0; rho is defined as 0 when H_marg = 0. N_eff
  can fall below k if usage collapses onto fewer than k experts (N_cov < k).

  Effective Parameters (per tower) summarizes N_eff at the model level:

      Effective Parameters = Shared Parameters + sum_l ( N_eff[l] * Per-Expert Parameters )

  where Shared Parameters is the non-expert LLM backbone (token embeddings, lm_head,
  final norm, and per-layer attention + norms + router gate), derived analytically from
  the generator text config; the vision tower and diffusion adapters are excluded, so the
  number reflects the effective LLM size. Per-Expert Parameters is one expert's gate_up +
  down projection size, and the sum runs over this tower's MoE layers. It interpolates
  between the active size (all N_eff == k, e.g. ~A3B / ~A22B) and the full size (all
  N_eff == N, e.g. 30B / 235B).

Buffer ownership
----------------
  This callback is fully self-contained: it reads and resets its own dedicated
  buffers (stability_tokens_per_expert, stability_total_tokens, sum_token_entropy,
  sum_per_token_soft_eff, sum_router_prob_per_expert). It does not depend on
  ExpertHeatmap's reset cycle.
"""

import math

# Fraction of uniform fair-share below which an expert is considered "dead" (e.g. 0.1 → < 10% of K/N).
DEAD_EXPERT_THRESHOLD_MULTIPLIER = 0.1

# Smoothing added inside log() to avoid log(0) for experts that received zero
# tokens in the current buffer window. Matches the constant used inside the
# MoE block when accumulating router entropy.
ENTROPY_EPSILON = 1e-9

import torch
import wandb
from torch.distributed.tensor import DTensor, Partial

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import distributed


def _effective_experts(
    sum_per_token_soft_eff: torch.Tensor,
    total_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (soft_eff, hard_eff) from already-reduced stability buffers.

    Extracted as a pure-tensor function so it can be unit-tested without
    instantiating any MoE module or distributed state.

    Args:
        sum_per_token_soft_eff: 0-d or [1] tensor holding sum_t exp(H(p_t))
            accumulated across the buffer window.
        total_tokens: 0-d or [1] tensor holding the number of tokens seen
            since the last reset.
        tokens_per_expert: [N] tensor of per-expert token counts over the
            same buffer window.

    Returns:
        soft_eff: scalar tensor, mean_t exp(H(p_t)) in [1, N].
        hard_eff: scalar tensor, exp(H(f)) over the empirical dispatch
            distribution f_i = tokens_per_expert_i / sum_i tokens_per_expert_i.
            Bounded in [K, N] (not [1, N]) because top-K dispatch always
            engages at least K experts in aggregate.

    Note on hard_eff normalization:
        tokens_per_expert is a histogram over the K top-k slots per token, so
        it sums to T * K rather than T. We must divide by its own sum (== T*K)
        to get a true probability distribution before taking entropy.
        Dividing by total_tokens (== T) instead would give a vector summing to
        K, producing exp(H) values up to (N/K)^K — orders of magnitude beyond
        the intended [K, N] range.
    """
    total = total_tokens.float().clamp(min=1)
    soft_eff = (sum_per_token_soft_eff.float() / total).squeeze()

    total_assignments = tokens_per_expert.sum().float().clamp(min=1)
    f_i = (tokens_per_expert.float() / total_assignments).clamp(min=ENTROPY_EPSILON)
    hard_entropy = -(f_i * f_i.log()).sum()
    hard_eff = hard_entropy.exp()

    return soft_eff, hard_eff


def _load_imbalance_factor(tokens_per_expert: torch.Tensor, total_tokens: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Load Imbalance Factor = N * max(f_i) / K, where f_i = tokens_per_expert / total_tokens.

    1.0 = perfectly balanced (every expert gets its K/N fair share); >3 = severe imbalance.
    Pure-tensor helper so the training callback and offline measurement share one definition.
    """
    total = total_tokens.float().clamp(min=1)
    f_i = tokens_per_expert.float() / total
    return f_i.max() * n / k


def _router_entropy_normalized(sum_token_entropy: torch.Tensor, total_tokens: torch.Tensor, n: int) -> torch.Tensor:
    """Mean per-token router entropy normalized to [0, 1] by log(N)."""
    total = total_tokens.float().clamp(min=1)
    return (sum_token_entropy.float() / total / math.log(n)).squeeze()


def _honest_effective_experts(
    sum_router_prob_per_expert: torch.Tensor,
    sum_token_entropy: torch.Tensor,
    total_tokens: torch.Tensor,
    k: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (n_eff, rho, n_cov) honest effective experts from already-reduced buffers.

    Extracted as a pure-tensor function so the same formula is shared by the
    training callback and any offline measurement (no drift), and is unit-testable
    without instantiating an MoE module or distributed state.

    N_eff = k + (N_cov - k) * rho, where N_cov = exp(H_marg) is the marginal
    coverage and rho in [0, 1] is the routing conditionality (the normalized
    mutual information between token identity and expert routing). Uses the SOFT
    marginal P[e] = mean_t g[t, e] and the soft per-token entropy H_within, both
    in nats, so rho >= 0 holds by concavity of entropy (H_marg >= H_within).

    Args:
        sum_router_prob_per_expert: [N] tensor holding sum_t g[t, e] over the window.
        sum_token_entropy: 0-d or [1] tensor holding sum_t H(p_t) in nats.
        total_tokens: 0-d or [1] tensor holding the number of tokens since reset.
        k: top-k experts dispatched per token.
        n: total number of experts.

    Returns:
        n_eff: scalar tensor, honest effective experts clamped to [0, N]. Can dip
            below k when usage collapses onto fewer than k experts (N_cov < k);
            that is itself a meaningful signal, not an error.
        rho: scalar tensor, routing conditionality in [0, 1] (defined as 0 when
            H_marg = 0).
        n_cov: scalar tensor, marginal coverage exp(H_marg) in [1, N] — how many
            experts are used in aggregate, before discounting by conditionality.
    """
    total_d = total_tokens.double().clamp(min=1.0)
    p_marg = (sum_router_prob_per_expert.double() / total_d).clamp(min=ENTROPY_EPSILON)  # [N], ~sums to 1
    h_marg = -(p_marg * p_marg.log()).sum()  # marginal entropy (nats)
    h_within = sum_token_entropy.double().squeeze() / total_d.squeeze()  # mean per-token entropy (nats)
    n_cov = h_marg.exp()  # coverage in [1, N]
    rho = torch.where(
        h_marg > ENTROPY_EPSILON,
        ((h_marg - h_within) / h_marg).clamp(0.0, 1.0),
        torch.zeros_like(h_marg),
    )
    n_eff = (k + (n_cov - k) * rho).clamp(min=0.0, max=float(n))
    return n_eff, rho, n_cov


def compute_moe_stability_metrics(vfm: torch.nn.Module) -> dict[str, dict]:
    """
    Compute per-layer MoE stability metrics for both towers.

    Iterates over all model layers, skipping any that do not use
    Qwen3VLMoeTextSparseMoeBlock (e.g. dense layers when decoder_sparse_step > 1).
    Actual model layer indices are preserved so W&B keys (layer_000, layer_042, ...)
    always refer to the correct transformer layer regardless of MoE sparsity pattern.

    Returns a dict: tower -> {
        "layer_indices":             list[int]                — actual model layer positions
        "dead_expert_rate":          Tensor[num_moe_layers]
        "lif":                       Tensor[num_moe_layers]
        "router_entropy_normalized": Tensor[num_moe_layers]
        "soft_eff_normalized":       Tensor[num_moe_layers]   — mean_t exp(H(p_t)) / N,  in [1/N, 1]
        "hard_eff_normalized":       Tensor[num_moe_layers]   — exp(H(f)) / N,           in [1/N, 1]
        "router_token_mi_normalized": Tensor[num_moe_layers]  — normalized token-routing MI in [0, 1]
        "honest_eff_normalized":     Tensor[num_moe_layers]   — N_eff / N, fraction in [~k/N, 1]
    }
    """
    with torch.no_grad():
        num_layers = len(vfm.language_model.model.layers)

        example_weight = vfm.language_model.model.layers[0].self_attn.q_proj.weight
        device_mesh = example_weight.device_mesh if isinstance(example_weight, DTensor) else None

        if device_mesh is None:
            return {}

        def _allreduce(t: torch.Tensor) -> torch.Tensor:
            return DTensor.from_local(
                t,
                device_mesh=device_mesh,
                placements=[Partial()] * device_mesh.ndim,
            ).full_tensor()

        results: dict[str, dict] = {}
        for tower in ["und", "gen"]:
            layer_indices: list[int] = []
            dead_rates: list[torch.Tensor] = []
            lifs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            soft_effs_norm: list[torch.Tensor] = []
            hard_effs_norm: list[torch.Tensor] = []
            router_token_mi_norms: list[torch.Tensor] = []
            honest_effs_norm: list[torch.Tensor] = []

            for layer_idx in range(num_layers):
                layer_module = vfm.language_model.model.layers[layer_idx]
                # "und" tower uses layer.mlp; "gen" tower uses layer.mlp_moe_gen.
                # Both attributes exist on every layer (set in unified_mot.py), but only
                # layers where (layer_idx+1) % decoder_sparse_step == 0 are MoE blocks.
                mlp_module = layer_module.mlp if tower == "und" else getattr(layer_module, "mlp_moe_gen", None)
                if not isinstance(mlp_module, Qwen3VLMoeTextSparseMoeBlock):
                    continue

                total_tokens_per_expert = _allreduce(mlp_module.get_stability_tokens_per_expert(reset=True))
                total_tokens = _allreduce(mlp_module.get_stability_total_tokens(reset=True))
                sum_token_entropy = _allreduce(mlp_module.get_sum_token_entropy(reset=True))
                sum_per_token_soft_eff = _allreduce(mlp_module.get_sum_per_token_soft_eff(reset=True))
                sum_router_prob_per_expert = _allreduce(mlp_module.get_sum_router_prob_per_expert(reset=True))

                n = mlp_module.num_experts
                total = total_tokens.float().clamp(min=1)
                f_i = total_tokens_per_expert.float() / total  # [N] load fraction per expert

                k = mlp_module.top_k

                layer_indices.append(layer_idx)
                # Uniform fair share per expert is K/N.  "Dead" = below 10% of that.
                dead_rates.append((f_i < DEAD_EXPERT_THRESHOLD_MULTIPLIER * k / n).float().mean())
                # LIF = max(f_i) * N / K.  Interpretation:
                #   1.0 = perfectly balanced (every expert gets its fair share)
                #   2.0 = busiest expert handles 2x its fair share
                #   >3.0 = severe imbalance, consider tuning load-balancing loss
                lifs.append(_load_imbalance_factor(total_tokens_per_expert, total_tokens, n, k))
                # Mean per-token entropy, normalized to [0, 1] by log(N).
                entropies.append(_router_entropy_normalized(sum_token_entropy, total_tokens, n))

                soft_eff, hard_eff = _effective_experts(
                    sum_per_token_soft_eff=sum_per_token_soft_eff,
                    total_tokens=total_tokens,
                    tokens_per_expert=total_tokens_per_expert,
                )
                soft_effs_norm.append(soft_eff / n)
                hard_effs_norm.append(hard_eff / n)

                # Honest effective experts N_eff and routing conditionality rho.
                # See _honest_effective_experts for the full derivation; it uses the
                # SOFT marginal P[e] = mean_t g[t, e] and the soft per-token entropy.
                n_eff, rho, _n_cov = _honest_effective_experts(
                    sum_router_prob_per_expert=sum_router_prob_per_expert,
                    sum_token_entropy=sum_token_entropy,
                    total_tokens=total_tokens,
                    k=k,
                    n=n,
                )
                router_token_mi_norms.append(rho)
                # Normalized to a fraction of total experts, in [~k/N, 1].
                honest_effs_norm.append(n_eff / n)

            if layer_indices:
                results[tower] = {
                    "layer_indices": layer_indices,
                    "dead_expert_rate": torch.stack(dead_rates),
                    "lif": torch.stack(lifs),
                    "router_entropy_normalized": torch.stack(entropies),
                    "soft_eff_normalized": torch.stack(soft_effs_norm),
                    "hard_eff_normalized": torch.stack(hard_effs_norm),
                    "router_token_mi_normalized": torch.stack(router_token_mi_norms),
                    "honest_eff_normalized": torch.stack(honest_effs_norm),
                }

    return results


def compute_moe_param_counts(text_config) -> tuple[int, int, int]:
    """Analytically derive ``(per_expert_params, shared_params, num_experts)`` for the
    Effective Parameters metric, purely from the LLM (generator) text config.

    Counts a standard single-tower Qwen3-VL-MoE *LLM* — token embeddings, the LM head,
    the final norm, and per-layer attention (q/k/v/o + per-head q/k RMSNorm), the two
    layernorms, and the router gate, plus a dense MLP on any ``mlp_only_layers``. The
    vision tower and the diffusion adapters (vae2llm / llm2vae / time embedders) are
    deliberately excluded; the result is the *effective LLM size* the MoT generator pulls
    from. Effective Parameters then interpolates between the active size (all N_eff == k)
    and the full size (all N_eff == N), matching the published A{active}B / {total}B
    figures.

    per_expert_params: one expert's gate_up (H x 2I) + down (I x H) = 3 * H * I.
    num_experts: experts per MoE layer (N), used to turn honest_eff_normalized back into
        a raw effective-expert count.

    Static for a given model, so callers should cache the result.
    """
    h = text_config.hidden_size
    moe_inter = text_config.moe_intermediate_size
    num_experts = int(text_config.num_experts)
    num_layers = text_config.num_hidden_layers
    n_heads = text_config.num_attention_heads
    n_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", None) or (h // n_heads)
    vocab = text_config.vocab_size
    attn_bias = bool(getattr(text_config, "attention_bias", False))

    per_expert_params = 3 * h * moe_inter

    # Identify MoE vs dense (mlp_only) layers exactly as the decoder layer does.
    mlp_only = set(getattr(text_config, "mlp_only_layers", []) or [])
    sparse_step = getattr(text_config, "decoder_sparse_step", 1)
    num_moe_layers = sum(
        1 for i in range(num_layers) if i not in mlp_only and num_experts > 0 and (i + 1) % sparse_step == 0
    )
    num_dense_layers = num_layers - num_moe_layers

    # Per-layer attention: q/o projections (n_heads), k/v projections (n_kv_heads), and
    # the per-head q/k RMSNorms. All bias-free unless attention_bias is set.
    attn = 2 * n_heads * head_dim * h + 2 * n_kv_heads * head_dim * h + 2 * head_dim
    if attn_bias:
        attn += (n_heads + 2 * n_kv_heads) * head_dim + h  # q/k/v + o projection biases
    layer_norms = 2 * h  # input_layernorm + post_attention_layernorm

    shared_params = vocab * h  # token embeddings
    if not bool(getattr(text_config, "tie_word_embeddings", False)):
        shared_params += vocab * h  # lm_head
    shared_params += h  # final norm
    shared_params += num_layers * (attn + layer_norms)
    shared_params += num_moe_layers * (h * num_experts)  # router gate per MoE layer
    shared_params += num_dense_layers * (3 * h * text_config.intermediate_size)  # dense MLP layers

    return per_expert_params, int(shared_params), num_experts


class MoEStabilityCallback(EveryN):
    """
    Logs per-layer MoE stability metrics to W&B every N training steps.

    What it captures
    ----------------
    Whether the MoE router remains in a healthy, balanced state over training.
    The metrics collectively answer: are all experts still being used
    (dead_expert_rate), is load spread evenly (lif), is the router still
    making uncertain, exploratory decisions (router_entropy_normalized), and
    do the experts the router considers (soft_eff) match the experts top-k
    dispatch actually engages (hard_eff)?

    W&B layout
    ----------
    For each per-layer metric and each tower, two kinds of series are logged:
      - moe_stability/<metric>/<tower>/layer_NNN  — per model layer time series
      - moe_stability/<metric>/<tower>/mean|max   — summary across all MoE layers
    Plus per-tower scalars:
      - moe_stability/n_eff_total/<tower>             — sum of N_eff over MoE layers
      - moe_stability/effective_params_billions/<tower> — (Shared + n_eff_total * per-expert
        params) / 1e9, i.e. in billions so a log-scale axis reads 1B, 2B, ...

    Per-layer metrics logged: dead_expert_rate, lif, router_entropy_normalized,
    soft_eff_normalized, hard_eff_normalized, router_token_mi_normalized, honest_eff_normalized.

    Typical healthy ranges:
      dead_expert_rate  → 0 (any sustained non-zero value is a concern)
      lif               → <= 1.3 (alarm at > 3.0)
      router_entropy_normalized → > 0.7 (collapse if it drops sharply)
      soft_eff_normalized, hard_eff_normalized → high; a large gap between
        them (e.g. soft high, hard low) indicates hidden top-k concentration
      router_token_mi_normalized → higher means routing is genuinely token-conditional
      honest_eff_normalized → higher (toward 1) means more experts honestly used;
        a falling value / effective_params indicates the router is collapsing toward k/N

    Args:
        every_n (int): Logging interval in training steps.
    """

    def __init__(self, every_n: int = 100):
        super().__init__(every_n=every_n)
        # Static parameter counts for the Effective Parameters metric, computed lazily
        # on the first logging step (on rank 0) and cached thereafter.
        self._per_expert_params: int | None = None
        self._shared_params: int | None = None
        self._num_experts: int | None = None

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        metrics = compute_moe_stability_metrics(model.net)

        if not (distributed.is_rank0() and wandb.run):
            return

        if metrics and self._per_expert_params is None:
            # The MoE block holds the LLM (generator) text config we derive counts from.
            text_config = next(
                (m.config for m in model.net.modules() if isinstance(m, Qwen3VLMoeTextSparseMoeBlock)),
                None,
            )
            if text_config is not None:
                self._per_expert_params, self._shared_params, self._num_experts = compute_moe_param_counts(text_config)

        log_dict: dict[str, float] = {}
        for tower, tower_metrics in metrics.items():
            layer_indices = tower_metrics.pop("layer_indices")

            # Effective Parameters (per tower) = shared LLM backbone + effective experts.
            # honest_eff_normalized is a per-layer fraction in [0, 1]; multiply by the
            # expert count N to recover raw effective experts, then sum over MoE layers.
            if self._per_expert_params is not None:
                n_eff_total = float((tower_metrics["honest_eff_normalized"].sum() * self._num_experts).item())
                log_dict[f"moe_stability/n_eff_total/{tower}"] = n_eff_total
                # Reported in billions so a log-scale W&B axis reads 1B, 2B, 5B, ...
                effective_params = self._shared_params + n_eff_total * self._per_expert_params
                log_dict[f"moe_stability/effective_params_billions/{tower}"] = effective_params / 1e9

            for metric_name, values in tower_metrics.items():
                for layer_idx, val in zip(layer_indices, values):
                    log_dict[f"moe_stability/{metric_name}/{tower}/layer_{layer_idx:03d}"] = val.item()
                log_dict[f"moe_stability/{metric_name}/{tower}/mean"] = values.mean().item()
                log_dict[f"moe_stability/{metric_name}/{tower}/max"] = values.max().item()

        wandb.log(log_dict, step=iteration)
