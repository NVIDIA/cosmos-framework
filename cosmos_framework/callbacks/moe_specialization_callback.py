# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
MoE Specialization Callback
============================
Monitors whether MoE experts are developing distinct, stable roles over training.
A well-trained MoE should have experts that specialize — each processing a different
kind of input — rather than a few generalist experts doing everything while the rest
idle.

  Expert Co-activation Rate
  -------------------------
  If two experts frequently fire together on the same token (both in the top-K
  selected), they are likely learning redundant representations. Ideally experts
  specialize on non-overlapping token types, so co-activation should stay close
  to the chance baseline of K/N (e.g. 8/128 ≈ 0.0625 for the 235B model).

  For each layer and each unique expert pair (i, j), we compute:
      CoAct(i, j) = N_{i,j} / N_i
  where N_{i,j} = number of tokens where both i and j were selected, and N_i =
  total tokens routed to expert i. We then summarize across all pairs as max and
  mean. A rising mean_coact, especially well above the chance baseline, signals
  that the router is collapsing onto a small correlated cluster of experts.

Buffer ownership
----------------
  coactivation_counts is reset here (in compute_moe_coactivation_metrics).
  Per-expert token counts are derived from coactivation_counts itself
  (row_sum + col_sum) / (K-1), so this callback is fully independent of
  ExpertHeatmap's reset cycle for total_tokens_per_expert.
"""

import torch
import wandb
from torch.distributed.tensor import DTensor, Partial

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import distributed
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock

_RANK_EPSILON = 1e-12


def _validate_router_weight(gate_weight: torch.Tensor) -> torch.Tensor:
    """Return a detached FP32 router matrix after validating its shape."""
    if gate_weight.ndim != 2:
        raise ValueError(f"Expected a 2-D router weight, got shape {tuple(gate_weight.shape)}")
    return gate_weight.detach().to(torch.float32)


def calculate_mods(gate_weight: torch.Tensor) -> torch.Tensor:
    """Compute mean off-diagonal absolute cosine similarity between router rows."""
    weight = _validate_router_weight(gate_weight)
    num_experts = weight.shape[0]
    if num_experts < 2:
        return weight.new_zeros(())
    normalized = torch.nn.functional.normalize(weight, dim=1)
    absolute_cosines = (normalized @ normalized.T).abs()
    off_diagonal_sum = absolute_cosines.sum() - absolute_cosines.diagonal().sum()
    return off_diagonal_sum / (num_experts * (num_experts - 1))


def _entropy_rank(probabilities: torch.Tensor) -> torch.Tensor:
    """Return exp(entropy(probabilities)), matching the offline router-rank implementation."""
    probabilities = probabilities.clamp_min(_RANK_EPSILON)
    return torch.exp(-(probabilities * probabilities.log()).sum())


def _calculate_router_ranks(
    gate_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute effective and spectral ranks from one router SVD."""
    weight = _validate_router_weight(gate_weight)
    singular_values = torch.linalg.svdvals(weight)
    singular_probabilities = singular_values / singular_values.sum().clamp_min(_RANK_EPSILON)
    effective_rank = _entropy_rank(singular_probabilities)
    energy = singular_values.square()
    energy_probabilities = energy / energy.sum().clamp_min(_RANK_EPSILON)
    spectral_rank = _entropy_rank(energy_probabilities)
    max_possible_rank = min(weight.shape)
    return (
        effective_rank,
        effective_rank / max_possible_rank,
        spectral_rank,
        spectral_rank / max_possible_rank,
    )


def _materialize_parameter(parameter: torch.Tensor) -> torch.Tensor:
    """Materialize one logical parameter, participating in DTensor collectives on every rank."""
    if isinstance(parameter, DTensor):
        return parameter.full_tensor()
    return parameter


def compute_moe_router_geometry_metrics(vfm: torch.nn.Module) -> dict[str, dict]:
    """Compute checkpoint-compatible router geometry metrics per layer and tower."""
    results: dict[str, dict] = {}
    with torch.no_grad():
        for tower in ("und", "gen"):
            layer_indices: list[int] = []
            mods: list[torch.Tensor] = []
            effective_ranks: list[torch.Tensor] = []
            effective_ranks_normalized: list[torch.Tensor] = []
            spectral_ranks: list[torch.Tensor] = []
            spectral_ranks_normalized: list[torch.Tensor] = []

            for layer_idx, layer in enumerate(vfm.language_model.model.layers):
                mlp = layer.mlp if tower == "und" else getattr(layer, "mlp_moe_gen", None)
                if not isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
                    continue

                gate_weight = _materialize_parameter(mlp.gate.weight)
                (
                    effective_rank,
                    effective_rank_normalized,
                    spectral_rank,
                    spectral_rank_normalized,
                ) = _calculate_router_ranks(gate_weight)
                layer_indices.append(layer_idx)
                mods.append(calculate_mods(gate_weight))
                effective_ranks.append(effective_rank)
                effective_ranks_normalized.append(effective_rank_normalized)
                spectral_ranks.append(spectral_rank)
                spectral_ranks_normalized.append(spectral_rank_normalized)

            if layer_indices:
                results[tower] = {
                    "layer_indices": layer_indices,
                    "mods": torch.stack(mods),
                    "effective_rank": torch.stack(effective_ranks),
                    "effective_rank_normalized": torch.stack(effective_ranks_normalized),
                    "spectral_rank": torch.stack(spectral_ranks),
                    "spectral_rank_normalized": torch.stack(spectral_ranks_normalized),
                }
    return results


def _get_device_mesh(vfm: torch.nn.Module):
    weight = vfm.language_model.model.layers[0].self_attn.q_proj.weight
    return weight.device_mesh if isinstance(weight, DTensor) else None


def _allreduce_dtensor(t: torch.Tensor, device_mesh) -> torch.Tensor:
    """Sum-reduce a local tensor across all FSDP ranks and return the global tensor."""
    return DTensor.from_local(
        t,
        device_mesh=device_mesh,
        placements=[Partial()] * device_mesh.ndim,
    ).full_tensor()


def compute_moe_coactivation_metrics(vfm: torch.nn.Module) -> dict[str, dict]:
    """
    Compute per-layer Expert Co-activation metrics for both towers.

    For each unique expert pair (i < j) in the upper triangle of the N×N
    coactivation matrix, computes:
        CoAct(i, j) = N_{i,j} / N_i
    where N_{i,j} is the count of tokens where both i and j were in the top-K,
    and N_i is the total token count for expert i (the row expert, i.e. the
    lower-indexed expert in the pair).

    N_i is derived directly from the co-activation matrix rather than from
    the shared total_tokens_per_expert buffer, so this metric is independent
    of ExpertHeatmap's reset cycle.  Each token routed to expert i contributes
    to (K-1) co-activation pairs, so N_i = (row_sum_i + col_sum_i) / (K-1).

    High co-activation relative to the chance baseline (K/N) indicates that
    certain expert pairs are systematically selected together — a sign of
    redundancy rather than specialization.

    Returns a dict: tower -> {
        "layer_indices":  list[int]          — actual model layer positions
        "max_coact":      Tensor[num_moe_layers]  — worst pair per layer
        "mean_coact":     Tensor[num_moe_layers]  — average over all pairs
        "chance_baseline": float             — K/N, same for all layers (reference)
    }
    """
    with torch.no_grad():
        device_mesh = _get_device_mesh(vfm)
        if device_mesh is None:
            return {}

        results: dict[str, dict] = {}
        for tower in ["und", "gen"]:
            layer_indices, max_coacts, mean_coacts, chance_baselines = [], [], [], []

            num_layers = len(vfm.language_model.model.layers)
            for layer_idx in range(num_layers):
                layer = vfm.language_model.model.layers[layer_idx]
                mlp = layer.mlp if tower == "und" else getattr(layer, "mlp_moe_gen", None)
                if not isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
                    continue

                coact_counts = _allreduce_dtensor(mlp.get_coactivation_counts(reset=True), device_mesh)  # [N, N]

                n = mlp.num_experts
                k = mlp.top_k

                # Derive per-expert token counts directly from the co-activation
                # matrix so we don't depend on ExpertHeatmap's reset cycle.
                # Each token that routes to expert i contributes (K-1) entries
                # across row i and column i of the upper-triangle matrix.
                tokens_per_expert = (coact_counts.sum(dim=1) + coact_counts.sum(dim=0)).float() / (k - 1)

                mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=coact_counts.device), diagonal=1)
                # CoAct(i, j) = N_{i,j} / N_i — normalise by how often expert i fires overall.
                denom = tokens_per_expert.unsqueeze(1).clamp(min=1)  # [N, 1]
                coact_rates = (coact_counts.float() / denom)[mask]  # [N*(N-1)/2]

                layer_indices.append(layer_idx)
                max_coacts.append(coact_rates.max())
                mean_coacts.append(coact_rates.mean())
                # Chance baseline = probability two randomly-chosen top-K slots land on the
                # same pair under uniform routing = K/N. Constant across layers and steps,
                # logged once per tower as a reference line.
                chance_baselines.append(k / n)

            if layer_indices:
                results[tower] = {
                    "layer_indices": layer_indices,
                    "max_coact": torch.stack(max_coacts),
                    "mean_coact": torch.stack(mean_coacts),
                    "chance_baseline": chance_baselines[0],  # same value for all layers
                }

    return results


class MoESpecializationCallback(EveryN):
    """
    Logs per-layer MoE specialization metrics to W&B every N training steps.

    What it captures
    ----------------
    Whether MoE experts are developing distinct routing identities:

    Expert Co-activation (logged every N steps)
      - mean_coact / max_coact per layer: how often expert pairs fire together
        relative to the chance_baseline (K/N). Values well above the baseline
        suggest the router is selecting a redundant cluster of experts rather
        than a diverse set.

    W&B layout
    ----------
    moe_specialization/coact_chance_baseline/<tower>         — flat reference (K/N)
    moe_specialization/max_coact/<tower>/layer_NNN|mean|max
    moe_specialization/mean_coact/<tower>/layer_NNN|mean|max

    Args:
        every_n (int): Logging interval in training steps.
    """

    def __init__(self, every_n: int = 250):
        super().__init__(every_n=every_n)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        vfm = model.net

        coact_results = compute_moe_coactivation_metrics(vfm)

        if not (distributed.is_rank0() and wandb.run):
            return

        log_dict: dict[str, float] = {}

        for tower, tower_metrics in coact_results.items():
            layer_indices = tower_metrics.pop("layer_indices")
            chance_baseline = tower_metrics.pop("chance_baseline")
            log_dict[f"moe_specialization/coact_chance_baseline/{tower}"] = chance_baseline
            for metric_name, values in tower_metrics.items():
                for layer_idx, val in zip(layer_indices, values):
                    log_dict[f"moe_specialization/{metric_name}/{tower}/layer_{layer_idx:03d}"] = val.item()
                log_dict[f"moe_specialization/{metric_name}/{tower}/mean"] = values.mean().item()
                log_dict[f"moe_specialization/{metric_name}/{tower}/max"] = values.max().item()

        wandb.log(log_dict, step=iteration)


class MoERouterGeometryCallback(EveryN):
    """Log per-layer router MODS, effective rank, and spectral rank."""

    def __init__(self, every_n: int = 250):
        super().__init__(every_n=every_n)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        metrics = compute_moe_router_geometry_metrics(model.net)

        if not (distributed.is_rank0() and wandb.run):
            return

        log_dict: dict[str, float] = {}
        for tower, tower_metrics in metrics.items():
            layer_indices = tower_metrics["layer_indices"]
            for metric_name, values in tower_metrics.items():
                if metric_name == "layer_indices":
                    continue
                for layer_idx, value in zip(layer_indices, values):
                    log_dict[f"moe_router_geometry/{metric_name}/{tower}/layer_{layer_idx:03d}"] = value.item()
                log_dict[f"moe_router_geometry/{metric_name}/{tower}/mean"] = values.mean().item()
                log_dict[f"moe_router_geometry/{metric_name}/{tower}/min"] = values.min().item()
                log_dict[f"moe_router_geometry/{metric_name}/{tower}/max"] = values.max().item()

        if log_dict:
            wandb.log(log_dict, step=iteration)
