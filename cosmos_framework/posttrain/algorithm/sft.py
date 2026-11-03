# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Masked next-token supervised learning through the existing policy capability."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
    read_validation_batch_metadata,
)
from cosmos_framework.posttrain.algorithm.primitives.policy_forward import (
    forward_policy,
    policy_logits,
    prepare_policy_capability,
)
from cosmos_framework.posttrain.config.sft import SFT_SERVE_CONTRACT_DEFAULT, SFT_SERVE_CONTRACTS
from cosmos_framework.posttrain.data.packer.base import DataPacker, PackerSemanticsIdentity
from cosmos_framework.posttrain.model.protocol import LoadBalancingPolicy
from cosmos_framework.posttrain.schema import SFTTarget, Trajectory
from cosmos_framework.posttrain.utils.dist import rank_sum_count, run_rank_converged, world_size_or_1
from cosmos_framework.posttrain.utils.identity import normalize_json_dict

_IGNORE_LABEL = -100
_METRIC_ADAPTER = MetricAdapter(
    internal_prefix="sft_",
    names={
        "sft_loss": "loss_avg",
        "sft_valid_tokens": "valid_token_count",
        "sft_valid_sequences": "valid_sequence_count",
        # Present only when the objective is not the plain token mean: the value the gradient descends
        # (`objective`), the load-balancing term added to it (`lbl_loss`, coefficient included) and the
        # largest expert's share of routed tokens relative to the uniform share, max over layers
        # (`moe_load_max_rel`; 1.0 = balanced). Plain DP means: identical on every rank under method
        # "global", the per-rank value otherwise.
        "sft_objective": "objective",
        "sft_lbl_loss": "lbl_loss",
        "sft_moe_load_max_rel": "moe_load_max_rel",
    },
    phases=frozenset({"train", "val"}),
    ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
    weighted_by={"sft_loss": "sft_valid_tokens"},
)


def load_balancing_loss(per_layer: list[Any], *, coeff: float, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The Switch/GShard load-balancing term over the stacked per-layer ``LBLMetadata`` of one forward.

    ``coeff * E * mean_layers(sum_e f_e * P_e)`` with ``f_e = tokens routed to expert e / (tokens * top_k)`` (the counts
    summed over the DP ranks first under ``method="global"``, rank-local under ``"local"``) and ``P_e`` the mean router
    probability of expert ``e`` over this rank's real tokens (the only graph-carrying factor: the term trains the router
    alone). The formula is ``cosmos3/algorithm/loss/load_balancing.py::compute_load_balancing_loss`` for the two methods
    the Cosmos post-training recipe uses, without its DeviceMesh. Also returns ``E * max f_e`` (1.0 = balanced).
    """
    import torch.distributed as dist

    if not per_layer:
        raise ValueError("load_balancing_loss requires at least one layer of LBL metadata")
    layers = len(per_layer)
    counts = torch.stack([m.num_tokens_per_expert.reshape(-1) for m in per_layer]).to(torch.int64)  # [L,E]
    tokens = torch.stack([m.num_tokens.reshape(-1) for m in per_layer]).to(torch.int64).reshape(layers, 1)  # [L,1]
    probs = torch.stack([m.mean_router_prob_per_expert.reshape(-1) for m in per_layer]).float()  # [L,E]
    top_k = torch.stack([m.top_k.reshape(-1) for m in per_layer]).reshape(layers, 1).float()  # [L,1]
    if method == "global" and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
    shares = counts.float() / (tokens.float().clamp(min=1) * top_k)  # [L,E], sums to 1 per layer
    num_experts = counts.shape[-1]
    lbl = torch.mean(torch.sum(shares * probs, dim=-1) * num_experts)  # []
    return lbl * float(coeff), (shares.max() * num_experts).detach()  # [], []


def make_objective(
    packer: DataPacker,
    *,
    ce_exponent: float = 0.0,
    moe_lbl_coeff: float = 0.0,
    moe_lbl_method: str = "global",
    serve_contract: str = SFT_SERVE_CONTRACT_DEFAULT,
) -> Objective:
    """Bind packer masking semantics to masked causal CE (labels are unshifted).

    ``ce_exponent=0`` (default): the loss is the local token sum divided by the DP-*mean* supervised
    token count. DP gradient averaging then gives the global token mean. TOKEN_SUM lets LLMTrainer
    undo this divisor per microbatch and normalize an accumulated window exactly once.
    ``ce_exponent=e>0``: every row contributes ``sum_tokens(CE) / n^e`` and the divisor is the DP-mean of
    ``sum_rows n^(1-e)`` (``e=0.5``: the backbone's square-root weighting, a row weighs ``sqrt(n)``; ``e=1``:
    every row weighs the same); the window algebra is unchanged because the divisor is still a sum over
    rows. ``moe_lbl_coeff>0`` adds :func:`load_balancing_loss` over the sparse blocks' routing statistics
    of the same forward, read through the model's ``LoadBalancingPolicy`` capability (``model/protocol.py``:
    ``begin_lbl_collection`` before the forward, ``collect_lbl_metadata`` after it); a policy without the
    capability is refused with ``TypeError`` before any forward, and this module imports no model module.

    ``{train,val}/loss_avg`` is a LOCAL token mean, weighted by
    ``{train,val}/valid_token_count``. Both valid counts are LOCAL sums, to be summed
    by the trainer/report reducer, not averaged. ``OBJECTIVE_WEIGHT_SUM`` is only the
    DP-mean backward divisor, never a local metric aggregation weight.

    Every rank must supply a nonempty physical batch. Validation padding is expressed
    by first-transition ``algo_extra['validation_weight']`` (0 or 1); an all-dummy
    local shard still forwards, but a globally unsupervised batch is rejected.
    """
    if not isinstance(packer, PackerSemanticsIdentity):
        raise TypeError("SFT packer must declare validation_semantics_identity(), including masking semantics")
    if (
        isinstance(ce_exponent, bool)
        or not isinstance(ce_exponent, (int, float))
        or not 0.0 <= float(ce_exponent) <= 1.0
    ):
        raise ValueError(f"SFT ce_exponent={ce_exponent!r} must be a real in [0, 1]")
    if (
        isinstance(moe_lbl_coeff, bool)
        or not isinstance(moe_lbl_coeff, (int, float))
        or not 0.0 <= float(moe_lbl_coeff) < 1.0
    ):
        raise ValueError(f"SFT moe_lbl_coeff={moe_lbl_coeff!r} must be a real in [0, 1)")
    if moe_lbl_method not in ("local", "global"):
        raise ValueError(f"SFT moe_lbl_method={moe_lbl_method!r} must be 'local' or 'global'")
    if serve_contract not in SFT_SERVE_CONTRACTS:
        raise ValueError(f"SFT serve_contract={serve_contract!r} must be one of {SFT_SERVE_CONTRACTS}")
    ce_exponent, moe_lbl_coeff = float(ce_exponent), float(moe_lbl_coeff)
    weighted = ce_exponent != 0.0
    identity = normalize_json_dict(
        {
            "algorithm": "sft",
            "loss": "masked_next_token_cross_entropy_v1",
            "ignore_label_id": _IGNORE_LABEL,
            # The two optional keys appear only when set, so every existing run's identity is unchanged.
            "reduction": (
                f"global_row_weighted_token_sum_exponent_{ce_exponent:g}"
                if weighted
                else "global_supervised_token_mean"
            ),
            **({"ce_exponent": ce_exponent} if weighted else {}),
            **({"moe_lbl": {"coeff": moe_lbl_coeff, "method": moe_lbl_method}} if moe_lbl_coeff > 0 else {}),
            # The serve contract the run declared (``rl.serve_contract``): present
            # only when it is not the default, so every existing run's identity is unchanged.
            **({"serve_contract": serve_contract} if serve_contract != SFT_SERVE_CONTRACT_DEFAULT else {}),
            "packer_class": f"{type(packer).__module__}.{type(packer).__qualname__}",
            "packer": packer.validation_semantics_identity(),
        },
        context="SFT objective identity",
    )

    def objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        def prepare() -> tuple[Any, dict[str, Any], torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
            if not trajectories:
                raise ValueError("SFT requires a nonempty physical batch on every rank")
            metadata = read_validation_batch_metadata(trajectories)
            weights = metadata.sample_weights or [1.0] * len(trajectories)
            if any(weight not in (0.0, 1.0) for weight in weights):
                raise ValueError("SFT validation_weight must be 0 or 1")
            samples = []
            for trajectory in trajectories:
                if len(trajectory.transitions) != 1:
                    raise ValueError("SFT requires exactly one transition per trajectory")
                target = trajectory.transitions[0].targets
                if not isinstance(target, SFTTarget):
                    raise TypeError("SFT requires Transition.targets to be an SFTTarget")
                samples.append(packer.sft_process_sample(target))
            batch = dict(packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples), _IGNORE_LABEL))
            for key in ("input_ids", "labels", "attention_mask"):
                value = batch.get(key)
                if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.device.type != "cpu":
                    raise ValueError(f"SFT collation requires CPU {key} [B,L] tensors")
            input_ids, labels, attention = batch["input_ids"], batch.pop("labels"), batch["attention_mask"]  # [B,L]
            if (
                input_ids.shape != labels.shape
                or input_ids.shape != attention.shape
                or input_ids.shape[0] != len(trajectories)
                or input_ids.shape[1] < 2
            ):
                raise ValueError("SFT input_ids, labels and attention_mask must have matching [B,L>=2] shapes")
            if input_ids.dtype != torch.long or labels.dtype != torch.long:
                raise TypeError("SFT input_ids and labels must have dtype torch.long")
            if not torch.all((attention == 0) | (attention == 1)):
                raise ValueError("SFT attention_mask must contain only 0 and 1")
            if torch.any((labels < 0) & (labels != _IGNORE_LABEL)):
                raise ValueError("SFT labels must be token IDs or -100")
            # Shift exactly once. Requiring both positions excludes a left-padding
            # token predicting the first real token as well as right-padding targets.
            active = torch.tensor(weights, dtype=torch.bool)[:, None]  # [B,1]
            valid = (labels[:, 1:] != _IGNORE_LABEL) & attention[:, 1:].bool() & attention[:, :-1].bool() & active
            token_count = int(valid.sum().item())
            sequence_count = int(valid.any(dim=1).sum().item())
            net = getattr(model, "net", model)
            device = next(net.parameters()).device
            batch["input_ids"] = input_ids.to(device)  # [B,L]
            prepared = prepare_policy_capability(model, batch)
            return net, prepared, input_ids, labels[:, 1:].to(device), token_count, sequence_count, valid.to(device)

        net, prepared, input_ids, labels, token_count, sequence_count, valid = run_rank_converged(
            prepare, "SFT objective preparation"
        )
        # Count in int64 before forming the DP mean; large token batches must not
        # round their denominator through the default float32 collective dtype.
        global_tokens = rank_sum_count(token_count, labels.device)
        if global_tokens == 0:
            raise ValueError("SFT batch has no globally supervised next-token labels")
        row_counts = valid.sum(dim=1)  # [B] supervised tokens per row
        if weighted:
            # sum_rows n^(1-e) over the rows with tokens, DP-mean'd in float64 (the row weights are not integers).
            local_weight = float(row_counts[row_counts > 0].double().pow(1.0 - ce_exponent).sum().item())
            denominator = _dp_mean(local_weight, labels.device)
        else:
            denominator = global_tokens / world_size_or_1()
        if moe_lbl_coeff > 0:
            if not isinstance(model, LoadBalancingPolicy):
                raise TypeError(
                    "SFT moe_lbl_coeff > 0 requires a policy with the LoadBalancingPolicy capability "
                    f"(model/protocol.py); {type(model).__module__}.{type(model).__qualname__} declares none"
                )
            blocks = model.begin_lbl_collection()
            if blocks == 0:
                raise ValueError(
                    "SFT moe_lbl_coeff > 0 requires a policy whose network holds adapted sparse blocks "
                    "(LoadBalancingPolicy.begin_lbl_collection switched 0 blocks)"
                )
        # Collective-bearing forwards are not wrapped in a recoverable-error consensus.
        output = forward_policy(model, net, prepared)
        lbl_term: torch.Tensor | None = None
        load_max_rel: torch.Tensor | None = None
        if moe_lbl_coeff > 0:
            per_layer = model.collect_lbl_metadata()
            if len(per_layer) != blocks:
                raise RuntimeError(f"SFT collected LBL metadata from {len(per_layer)} of {blocks} sparse blocks")
            # Carries the DP all-reduce of the counts under method "global": outside the consensus wrapper too.
            lbl_term, load_max_rel = load_balancing_loss(per_layer, coeff=moe_lbl_coeff, method=moe_lbl_method)

        def finalize() -> tuple[torch.Tensor, dict[str, Any]]:
            logits = policy_logits(output, input_ids, owner="SFT policy")  # [B,L,V]
            selected_logits = logits[:, :-1][valid].float()  # [N,V]
            selected_labels = labels[valid]  # [N]
            # Sum reduction on an empty local selection yields a graph-connected zero.
            per_token = F.cross_entropy(selected_logits, selected_labels, reduction="none")  # [N]
            numerator = per_token.sum()  # []
            if weighted:
                rows = torch.nonzero(valid)[:, 0]  # [N] the row of every selected token, in selection order
                row_sums = torch.zeros(valid.shape[0], dtype=per_token.dtype, device=per_token.device).scatter_add(
                    0, rows, per_token
                )  # [B]
                counts = row_counts.to(per_token.dtype)  # [B]
                objective_value = (row_sums / counts.clamp(min=1).pow(ce_exponent) * (counts > 0)).sum() / denominator
            else:
                objective_value = numerator / denominator
            metrics: dict[str, Any] = {
                "sft_loss": numerator.detach() / max(token_count, 1),
                "sft_valid_tokens": token_count,
                "sft_valid_sequences": sequence_count,
                OBJECTIVE_WEIGHT_SUM: denominator,
            }
            loss = objective_value
            if lbl_term is not None:
                loss = objective_value + lbl_term
                metrics["sft_lbl_loss"] = lbl_term.detach()
                metrics["sft_moe_load_max_rel"] = load_max_rel
            if weighted or lbl_term is not None:
                metrics["sft_objective"] = loss.detach()
            return loss, metrics

        return run_rank_converged(finalize, "SFT objective finalization")

    return Objective(objective, reduction=Reduction.TOKEN_SUM, metric_adapter=_METRIC_ADAPTER, identity=identity)


def _dp_mean(local_value: float, device: torch.device) -> float:
    """The data-parallel mean of a float, in float64; the value itself outside a process group."""
    import torch.distributed as dist

    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return float(local_value)
    total = torch.tensor([float(local_value)], dtype=torch.float64, device=device)  # [1]
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total.item()) / dist.get_world_size()
