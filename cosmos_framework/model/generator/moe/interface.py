# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Routed experts with canonical parameters and optional inference kernels.

Native PyTorch is the default and the training fallback. Accelerated backends
operate on local, materialized BF16 experts. Call ``prepare_moe`` after loading
weights and before torch.compile / CUDA graph capture. Workspaces are specific
to the token count; packed weights are reused across eager shape changes. Compiled
inference treats weights as immutable: after direct in-place weight edits, prepare again and recapture any CUDA graphs. Scratch is
owned by each prepared layer and must not be shared by concurrent executions.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from cosmos_framework.model.generator.moe import native


@dataclass(frozen=True)
class MoEBackendConfig:
    """Inference policy; training always uses native until a backward backend is added."""

    backend: str = "native"
    precision: str = "bf16"

    def __post_init__(self) -> None:
        if self.backend not in {"native", "auto", "deep_gemm", "flash_infer_cutedsl", "flash_infer_cutlass"}:
            raise ValueError(f"Unsupported MoE backend: {self.backend}")
        if self.precision not in {"bf16", "nvfp4"}:
            raise ValueError(f"Unsupported MoE precision: {self.precision}")
        if self.precision == "nvfp4" and self.backend != "flash_infer_cutlass":
            raise ValueError("NVFP4 requires explicit flash_infer_cutlass selection")


class BackendUnavailable(ValueError):
    """A known dependency, hardware, or shape constraint excludes a backend."""


class PreparedMoE(Protocol):
    """Backend-private prepared weights and scratch; never checkpointed."""

    name: str

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T,D]
        scores: torch.Tensor,  # [T,K]
        indices: torch.Tensor,  # [T,K]
    ) -> torch.Tensor: ...  # [T,D]

    def close(self) -> None: ...


class PackedMoE(Protocol):
    """Shape-independent derived weights; workspaces have a separate lifetime."""

    name: str

    def prepare(self, num_tokens: int) -> PreparedMoE: ...


class MoEExperts(native.NativeMoEExperts):
    """Parameter-compatible replacement for the Qwen grouped-MM experts.

    The backend consumes already-selected routes without normalization. Padding is
    inert even when its input contains NaNs. Real tokens must have distinct top-k
    IDs. Native grouped MM additionally skips padding compute. Counts describe
    real assignments only, excluding masked rows.
    """

    backend_config: MoEBackendConfig
    selected_backend: str
    fallback_reason: str | None
    _prepared: PreparedMoE | None
    _prepared_tokens: int | None
    _packed: PackedMoE | None
    _packed_config: MoEBackendConfig | None
    _weight_state: tuple[tuple[int, int, torch.device, torch.dtype], ...] | None
    _requires_native: bool

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation: str = "silu",
        backend_config: MoEBackendConfig | None = None,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, num_experts, top_k, activation)
        self.backend_config = backend_config or MoEBackendConfig()
        self.selected_backend = "native"
        self.fallback_reason = None
        self._prepared = None
        self._prepared_tokens = None
        self._packed = None
        self._packed_config = None
        self._weight_state = None
        self._requires_native = False
        self.register_load_state_dict_post_hook(_invalidate_loaded_weights)

    def init_weights(self, buffer_device: torch.device) -> None:
        self.close()
        super().init_weights(buffer_device)

    def _close_workspace(self) -> None:
        if self._prepared is not None:
            self._prepared.close()
        self._prepared = None
        self._prepared_tokens = None
        self.selected_backend = "native"

    def close(self) -> None:
        """Release weights and scratch after all uses, including captured graph replays, finish."""
        self._close_workspace()
        self._packed = None
        self._packed_config = None
        self._weight_state = None
        self._requires_native = False

    def train(self, mode: bool = True) -> "MoEExperts":
        if mode != self.training:
            self.close()
        super().train(mode)
        return self

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "MoEExperts":
        self.close()
        super()._apply(fn, recurse=recurse)
        return self

    def _current_weight_state(self) -> tuple[tuple[int, int, torch.device, torch.dtype], ...]:
        # Inference tensors do not track versions; explicit preparation is required
        # after their mutation, just as for parameters in a captured CUDA graph.
        return tuple(
            (
                id(parameter),
                -1 if torch.is_inference(parameter) else parameter._version,
                parameter.device,
                parameter.dtype,
            )
            for parameter in (self.gate_up_proj, self.down_proj)
        )

    def prepare(self, num_tokens: int) -> str:
        """Prepare an exact shape, reusing unchanged packed weights.

        Explicit preparation repacks inference tensors because their mutations
        cannot be detected. Eager shape changes reuse even those packed weights;
        callers must still prepare explicitly after editing inference tensors.
        """
        return self._prepare(
            num_tokens, force_repack=torch.is_inference(self.gate_up_proj) or torch.is_inference(self.down_proj)
        )

    @torch.no_grad()
    def _prepare(self, num_tokens: int, force_repack: bool = False) -> str:
        if num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative")
        self.fallback_reason = None
        config = self.backend_config
        state = self._current_weight_state()
        if force_repack or self._packed_config != config or self._weight_state != state:
            self.close()
        else:
            self._close_workspace()
        if self.training or config.backend == "native" or num_tokens == 0:
            self.close()
            return self.selected_backend
        if hasattr(self.gate_up_proj, "placements"):
            self.close()
            self._requires_native = True
            self.fallback_reason = "Accelerated experts require local materialized weights, not DTensors"
            if config.backend != "auto":
                raise BackendUnavailable(self.fallback_reason)
            return self.selected_backend
        candidates = (
            ("deep_gemm", "flash_infer_cutedsl", "flash_infer_cutlass")
            if config.backend == "auto"
            else (config.backend,)
        )
        reasons: list[str] = []
        for backend in candidates:
            try:
                if self.gate_up_proj.device.type != "cuda" or self.gate_up_proj.dtype != torch.bfloat16:
                    raise BackendUnavailable("Accelerated experts require CUDA BF16 parameters")
                if self.activation != "silu":
                    raise BackendUnavailable("Accelerated experts require SiLU/SwiGLU")
                with torch.cuda.device(self.gate_up_proj.device):
                    if self._packed is None or self._packed.name != backend:
                        if backend == "deep_gemm":
                            from cosmos_framework.model.generator.moe.deep_gemm import pack_weights

                            kernel = "mega"
                        else:
                            from cosmos_framework.model.generator.moe.flash_infer import pack_weights

                            kernel = "cutedsl" if backend == "flash_infer_cutedsl" else "cutlass"
                        self._packed = pack_weights(
                            self.gate_up_proj, self.down_proj, self.top_k, kernel, config.precision
                        )
                    self._prepared = self._packed.prepare(num_tokens)
            except (ImportError, BackendUnavailable) as error:
                if self._packed is not None and self._packed.name == backend:
                    self._packed = None
                reasons.append(f"{backend}: {error}")
                continue
            self._prepared_tokens = num_tokens
            self._packed_config = config
            self._weight_state = state
            self.selected_backend = self._prepared.name
            return self.selected_backend
        self.fallback_reason = "; ".join(reasons)
        if config.backend != "auto":
            raise BackendUnavailable(self.fallback_reason)
        self._prepared_tokens = num_tokens
        self._packed_config = config
        self._weight_state = state
        return self.selected_backend

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T,D]
        topk_scores: torch.Tensor,  # [T,K]
        expert_indices: torch.Tensor,  # [T,K]
        num_tokens_per_expert: torch.Tensor,  # [E]
        token_mask: torch.Tensor | None = None,  # [T]
    ) -> torch.Tensor:  # [T,D]
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if topk_scores.shape != expert_indices.shape or expert_indices.shape != (hidden_states.shape[0], self.top_k):
            raise ValueError("Routing IDs and scores must both have shape [tokens, top_k]")
        if num_tokens_per_expert.shape != (self.num_experts,):
            raise ValueError("num_tokens_per_expert must have shape [num_experts]")
        if expert_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expert IDs must be int32 or int64")
        if not topk_scores.is_floating_point():
            raise ValueError("Routing scores must be floating point")
        if token_mask is not None:
            if token_mask.shape != (hidden_states.shape[0],) or token_mask.dtype != torch.bool:
                raise ValueError("token_mask must be boolean with shape [tokens]")
        needs_backward = torch.is_grad_enabled() and (
            hidden_states.requires_grad
            or topk_scores.requires_grad
            or self.gate_up_proj.requires_grad
            or self.down_proj.requires_grad
        )
        use_native = (
            self.training
            or needs_backward
            or self._requires_native
            or self.backend_config.backend == "native"
            or hidden_states.shape[0] == 0
        )
        if not use_native:
            if not torch.compiler.is_compiling():
                if (
                    self._prepared_tokens != hidden_states.shape[0]
                    or self._weight_state != self._current_weight_state()
                    or self._packed_config != self.backend_config
                ):
                    self._prepare(hidden_states.shape[0])
            if self._prepared is not None:
                if hidden_states.shape[0] != self._prepared_tokens:
                    raise ValueError("Prepare this token count before compiling or capturing accelerated MoE")
            elif self.backend_config.backend != "auto":
                raise RuntimeError("Call prepare_moe(model, num_tokens) before compiling accelerated MoE")
        prepared = None if use_native else self._prepared
        use_naive = (
            hidden_states.device.type != "cuda"
            or hidden_states.dtype not in (torch.bfloat16, torch.float16)
            or hidden_states.shape[0] == 0
        )
        # Grouped MM excludes padding before the GEMMs; only dense paths need copies.
        if token_mask is not None and (prepared is not None or use_naive):
            hidden_states = torch.where(token_mask[:, None], hidden_states, 0)  # [T,D]
            topk_scores = torch.where(token_mask[:, None], topk_scores, 0)  # [T,K]
            # Mega kernels require distinct top-k IDs, including zero-weight padding.
            padding_ids = torch.arange(self.top_k, device=expert_indices.device, dtype=expert_indices.dtype)  # [K]
            expert_indices = torch.where(token_mask[:, None], expert_indices, padding_ids)  # [T,K]
        if prepared is not None:
            return prepared.forward(hidden_states, topk_scores, expert_indices)  # [T,D]
        if use_naive:
            return native.naive(
                self.gate_up_proj, self.down_proj, self.act_fn, hidden_states, topk_scores, expert_indices
            )  # [T,D]
        return native.grouped_mm(
            self.gate_up_proj,
            self.down_proj,
            self.act_fn,
            hidden_states,
            topk_scores.to(hidden_states.dtype),
            expert_indices,
            num_tokens_per_expert,
            token_mask,
        )  # [T,D]


def _invalidate_loaded_weights(module: nn.Module, incompatible_keys: Any) -> None:
    if isinstance(module, MoEExperts):
        module.close()


def configure_moe(model: nn.Module, config: MoEBackendConfig) -> None:
    """Configure existing experts without replacing parameters or changing their paths."""
    for module in model.modules():
        if isinstance(module, MoEExperts):
            module.close()
            module.backend_config = config


def prepare_moe(model: nn.Module, num_tokens: int | Mapping[str, int]) -> dict[str, str]:
    """Prepare inference shapes, optionally mapping each module name to its tower's token count."""
    experts = {name: module for name, module in model.named_modules() if isinstance(module, MoEExperts)}
    if not isinstance(num_tokens, int) and set(num_tokens) != set(experts):
        raise ValueError("Token-count mapping must contain exactly the MoE module names")
    return {
        name: module.prepare(num_tokens if isinstance(num_tokens, int) else num_tokens[name])
        for name, module in experts.items()
    }
