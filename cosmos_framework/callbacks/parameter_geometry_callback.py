# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-frequency parameter geometry diagnostics for production VFM training."""

import math

import torch
import wandb
from torch.distributed.tensor import DTensor

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import distributed


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a logical tensor, participating in DTensor collectives on every rank."""
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def calculate_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute an FP32 L2/Frobenius norm of one logical tensor."""
    return torch.linalg.vector_norm(_materialize_tensor(tensor).detach().float())


def calculate_qk_logit_scale(
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q/K gamma RMS values and the existing input-independent QK scale proxy."""
    if q_gamma.ndim != 1 or k_gamma.ndim != 1:
        raise ValueError("Expected one-dimensional Q/K RMSNorm weights")
    if q_gamma.numel() != head_dim or k_gamma.numel() != head_dim:
        raise ValueError(
            f"Expected Q/K RMSNorm weights of length {head_dim}, got {q_gamma.numel()} and {k_gamma.numel()}"
        )
    q_norm = calculate_l2_norm(q_gamma)
    k_norm = calculate_l2_norm(k_gamma)
    q_rms = q_norm / math.sqrt(head_dim)
    k_rms = k_norm / math.sqrt(head_dim)
    # Matches dion2h_qk_norm_phase_portraits.py:
    # ||gamma_q||_2 * ||gamma_k||_2 / sqrt(head_dim).
    qk_logit_scale = q_norm * k_norm / math.sqrt(head_dim)
    return q_rms, k_rms, qk_logit_scale


def _parameter_at_path(module: torch.nn.Module, path: str) -> torch.Tensor | None:
    """Resolve a structural parameter path without depending on FSDP-generated names."""
    value: object = module
    for component in path.split("."):
        if component.isdigit() and isinstance(value, (torch.nn.Sequential, torch.nn.ModuleList)):
            value = value[int(component)]
        else:
            value = getattr(value, component, None)
        if value is None:
            return None
    return value if isinstance(value, torch.Tensor) else None


_BOUNDARY_PARAMETER_PATHS = (
    # Vision boundaries.
    "vae2llm.weight",
    "vae2llm.bias",
    "llm2vae.weight",
    "llm2vae.bias",
    # Sound boundaries.
    "sound2llm.weight",
    "sound2llm.bias",
    "llm2sound.weight",
    "llm2sound.bias",
    # Domain-aware action boundaries: ``fc`` and ``bias`` are embeddings whose
    # first dimension indexes embodiments.
    "action2llm.fc.weight",
    "action2llm.bias.weight",
    "llm2action.fc.weight",
    "llm2action.bias.weight",
    # Shared diffusion timestep boundary.
    "time_embedder.mlp.0.weight",
    "time_embedder.mlp.0.bias",
    "time_embedder.mlp.2.weight",
    "time_embedder.mlp.2.bias",
)


def compute_boundary_norms(vfm: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Compute norms for every boundary parameter present in this model configuration."""
    results: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for path in _BOUNDARY_PARAMETER_PATHS:
            parameter = _parameter_at_path(vfm, path)
            if parameter is not None:
                results[path] = calculate_l2_norm(parameter)
    return results


def compute_qk_geometry_metrics(vfm: torch.nn.Module) -> dict[str, torch.Tensor | list[int]]:
    """Compute generation-path Q/K gamma and logit-scale metrics per transformer layer."""
    language_model = getattr(vfm, "language_model", None)
    text_model = getattr(language_model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return {}

    layer_indices: list[int] = []
    q_gamma_rms: list[torch.Tensor] = []
    k_gamma_rms: list[torch.Tensor] = []
    qk_logit_scales: list[torch.Tensor] = []
    cross_k_gamma_rms: list[torch.Tensor] = []
    cross_qk_logit_scales: list[torch.Tensor] = []
    cross_layer_indices: list[int] = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                continue
            q_norm_module = getattr(attention, "q_norm_moe_gen", None)
            k_norm_module = getattr(attention, "k_norm_moe_gen", None)
            q_gamma = getattr(q_norm_module, "weight", None)
            k_gamma = getattr(k_norm_module, "weight", None)
            if not isinstance(q_gamma, torch.Tensor) or not isinstance(k_gamma, torch.Tensor):
                continue

            head_dim = int(attention.head_dim)
            q_rms, k_rms, qk_scale = calculate_qk_logit_scale(q_gamma, k_gamma, head_dim)
            layer_indices.append(layer_idx)
            q_gamma_rms.append(q_rms)
            k_gamma_rms.append(k_rms)
            qk_logit_scales.append(qk_scale)

            cross_k_norm_module = getattr(attention, "k_norm_und_for_gen", None)
            cross_k_gamma = getattr(cross_k_norm_module, "weight", None)
            if isinstance(cross_k_gamma, torch.Tensor):
                _, cross_k_rms, cross_qk_scale = calculate_qk_logit_scale(q_gamma, cross_k_gamma, head_dim)
                cross_layer_indices.append(layer_idx)
                cross_k_gamma_rms.append(cross_k_rms)
                cross_qk_logit_scales.append(cross_qk_scale)

    if not layer_indices:
        return {}
    results: dict[str, torch.Tensor | list[int]] = {
        "layer_indices": layer_indices,
        "q_gamma_rms": torch.stack(q_gamma_rms),
        "k_gamma_rms": torch.stack(k_gamma_rms),
        "qk_logit_scale": torch.stack(qk_logit_scales),
    }
    if cross_layer_indices:
        results["cross_layer_indices"] = cross_layer_indices
        results["cross_k_gamma_rms"] = torch.stack(cross_k_gamma_rms)
        results["cross_qk_logit_scale"] = torch.stack(cross_qk_logit_scales)
    return results


class ParameterGeometryCallback(EveryN):
    """Log QK scale proxies and multimodal boundary norms."""

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
        qk_metrics = compute_qk_geometry_metrics(model.net)
        boundary_norms = compute_boundary_norms(model.net)

        if not (distributed.is_rank0() and wandb.run):
            return

        log_dict: dict[str, float] = {
            f"parameter_geometry/boundary_norm/{path}": value.item() for path, value in boundary_norms.items()
        }

        if qk_metrics:
            layer_indices = qk_metrics["layer_indices"]
            assert isinstance(layer_indices, list)
            for metric_name in ("q_gamma_rms", "k_gamma_rms", "qk_logit_scale"):
                values = qk_metrics[metric_name]
                assert isinstance(values, torch.Tensor)
                for layer_idx, value in zip(layer_indices, values):
                    log_dict[f"parameter_geometry/{metric_name}/gen/layer_{layer_idx:03d}"] = value.item()
                log_dict[f"parameter_geometry/{metric_name}/gen/mean"] = values.mean().item()
                log_dict[f"parameter_geometry/{metric_name}/gen/min"] = values.min().item()
                log_dict[f"parameter_geometry/{metric_name}/gen/max"] = values.max().item()

            cross_layer_indices = qk_metrics.get("cross_layer_indices")
            if isinstance(cross_layer_indices, list):
                for metric_name in ("cross_k_gamma_rms", "cross_qk_logit_scale"):
                    values = qk_metrics[metric_name]
                    assert isinstance(values, torch.Tensor)
                    for layer_idx, value in zip(cross_layer_indices, values):
                        log_dict[f"parameter_geometry/{metric_name}/gen/layer_{layer_idx:03d}"] = value.item()
                    log_dict[f"parameter_geometry/{metric_name}/gen/mean"] = values.mean().item()
                    log_dict[f"parameter_geometry/{metric_name}/gen/min"] = values.min().item()
                    log_dict[f"parameter_geometry/{metric_name}/gen/max"] = values.max().item()

        if log_dict:
            wandb.log(log_dict, step=iteration)
