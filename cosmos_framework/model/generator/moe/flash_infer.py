# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Single-GPU inference custom operator for FlashInfer a31c725 BF16 Mega MoE.

Prepare the kernel outside torch.compile and CUDA graph capture. All six mutable
buffers are explicit operator arguments; the returned output is fresh. Runtime
launch arguments are rebuilt from those arguments, including functionalization
clones, rather than cached tensor pointers. Only MEGA_NO_DIST=1, fixed shapes,
and separate top-k reduction are supported. No autograd or concurrent use of the
same workspace is supported. Keep the prepared handle alive through all compiled
calls and graph replays, then release it after synchronizing the device.

This benchmark adapter uses private APIs from the pinned FlashInfer revision.
"""

import itertools
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from cosmos_framework.model.generator.moe.interface import BackendUnavailable


@dataclass(frozen=True)
class _PreparedKernel:
    frontend: Any
    compiled: Any
    kernel: Any
    local_bytes: int
    shared_bytes: int


_HANDLES: dict[int, _PreparedKernel] = {}
_NEXT_HANDLE = itertools.count()


def prepare_flashinfer_cutedsl_moe(buffer: Any) -> int:
    """Register an already warmed kernel; do not retain its mutable tensor buffers."""
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.bf16 import MegaMoEBf16Frontend

    config = buffer._frontend.config
    if os.environ.get("MEGA_NO_DIST") != "1" or config.world_size != 1 or config.rank != 0:
        raise ValueError("The adapter requires MEGA_NO_DIST=1 and rank 0 of one GPU")
    if config.enable_in_kernel_fc2_reduce or config.in_kernel_fc2_reduce:
        raise ValueError("The adapter supports only separate top-k reduction")
    mega = buffer._frontend._mega
    if buffer._destroyed or mega is None or mega.compiled is None:
        raise ValueError("Warm the FlashInfer kernel before preparing the adapter")
    handle = next(_NEXT_HANDLE)
    _HANDLES[handle] = _PreparedKernel(
        MegaMoEBf16Frontend(config),
        mega.compiled,
        mega.kernel,
        mega.local_workspace.numel(),
        mega.shared_workspace.numel(),
    )
    return handle


def release_flashinfer_cutedsl_moe(handle: int) -> None:
    """Release a handle after its final compiled call/replay has completed."""
    del _HANDLES[handle]


@torch.library.custom_op(
    "cosmos_moe::flashinfer_cutedsl",
    mutates_args={"staging_x", "staging_ids", "staging_scores", "combine", "local_workspace", "shared_workspace"},
    device_types="cuda",
)
def flashinfer_cutedsl_moe(
    x: torch.Tensor,  # [T,D]
    ids: torch.Tensor,  # [T,K], int64
    scores: torch.Tensor,  # [T,K], float32
    w1: torch.Tensor,  # [E,D,2I], FlashInfer-interleaved
    w2: torch.Tensor,  # [E,I,D]
    staging_x: torch.Tensor,  # [T,D]
    staging_ids: torch.Tensor,  # [T,K]
    staging_scores: torch.Tensor,  # [T,K]
    combine: torch.Tensor,  # [T,K,D]
    local_workspace: torch.Tensor,  # [local_bytes], uint8
    shared_workspace: torch.Tensor,  # [shared_bytes], uint8
    handle: int,
) -> torch.Tensor:  # [T,D]
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.bf16 import MegaMoEBf16Inputs
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.comm import _CompiledMega

    prepared = _HANDLES[handle]
    config = prepared.frontend.config
    if x.shape != (config.num_tokens_per_rank, config.hidden) or x.dtype != torch.bfloat16:
        raise ValueError("Input must match the prepared BF16 token/hidden shape")
    if ids.shape != staging_ids.shape or scores.shape != staging_scores.shape:
        raise ValueError("Routing tensors must match the prepared shape")
    if ids.dtype != torch.int64 or scores.dtype != torch.float32:
        raise ValueError("Routing IDs/scores must be int64/float32")
    for workspace, size in ((local_workspace, prepared.local_bytes), (shared_workspace, prepared.shared_bytes)):
        if workspace.dtype != torch.uint8 or not workspace.is_contiguous() or workspace.numel() != size:
            raise ValueError("Expected contiguous uint8 workspace matching the prepared size")
    inputs = MegaMoEBf16Inputs(staging_x, staging_ids, staging_scores, w1, w2, combine)
    prepared.frontend._validate(inputs, x.shape[0])
    # Use this invocation's scratch pointers, including when AOTAutograd clones them.
    mega = _CompiledMega(
        compiled=prepared.compiled,
        kernel=prepared.kernel,
        local_workspace=local_workspace,
        shared_workspace=shared_workspace,
        symmetric_base=shared_workspace.data_ptr(),
        peer_offsets_list=(0,),
    )
    staging_x.copy_(x)  # [T,D]
    staging_ids.copy_(ids)  # [T,K]
    staging_scores.copy_(scores)  # [T,K]
    prepared.compiled(**prepared.frontend._runtime_kwargs(inputs, mega))
    return combine.sum(dim=1)  # [T,D], fresh output; includes top-k reduction


@flashinfer_cutedsl_moe.register_fake
def _fake_flashinfer_cutedsl_moe(
    x: torch.Tensor,  # [T,D]
    ids: torch.Tensor,  # [T,K]
    scores: torch.Tensor,  # [T,K]
    w1: torch.Tensor,  # [E,D,2I]
    w2: torch.Tensor,  # [E,I,D]
    staging_x: torch.Tensor,  # [T,D]
    staging_ids: torch.Tensor,  # [T,K]
    staging_scores: torch.Tensor,  # [T,K]
    combine: torch.Tensor,  # [T,K,D]
    local_workspace: torch.Tensor,  # [local_bytes]
    shared_workspace: torch.Tensor,  # [shared_bytes]
    handle: int,
) -> torch.Tensor:  # [T,D]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)  # [T,D], fresh contiguous output


@dataclass
class CuteDSLExperts:
    w1: torch.Tensor  # [E,D,2I], interleaved
    w2: torch.Tensor  # [E,I,D]
    buffer: Any
    handle: int
    name: str = "flash_infer_cutedsl"
    _finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._finalizer = weakref.finalize(self, _close_cutedsl, self.buffer, self.handle)
        self._finalizer.atexit = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T,D]
        scores: torch.Tensor,  # [T,K]
        indices: torch.Tensor,  # [T,K]
    ) -> torch.Tensor:  # [T,D]
        mega = self.buffer._frontend._mega
        return flashinfer_cutedsl_moe(
            hidden_states,
            indices.long(),
            scores.float(),
            self.w1,
            self.w2,
            self.buffer.x,
            self.buffer.topk_idx,
            self.buffer.topk_weights,
            self.buffer.combine_output,
            mega.local_workspace,
            mega.shared_workspace,
            self.handle,
        )  # [T,D]

    def close(self) -> None:
        self._finalizer()


def _close_cutedsl(buffer: Any, handle: int) -> None:
    torch.cuda.synchronize(buffer.x.device)
    release_flashinfer_cutedsl_moe(handle)
    buffer.destroy()


@dataclass(frozen=True)
class _CutlassKernel:
    layer: Any
    weight_names: tuple[str, ...]
    backend_key: str
    prepare_activations: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None


_CUTLASS_HANDLES: dict[int, _CutlassKernel] = {}


@torch.library.custom_op("cosmos_moe::flashinfer_cutlass", mutates_args=(), device_types="cuda")
def flashinfer_cutlass_moe(
    x: torch.Tensor,  # [T,D]
    ids: torch.Tensor,  # [T,K]
    scores: torch.Tensor,  # [T,K]
    weights: list[torch.Tensor],  # backend-packed weights/scales
    handle: int,
) -> torch.Tensor:  # [T,D]
    """Opaque inference call; FlashInfer owns its internal runner workspace."""
    from flashinfer.fused_moe import MoEActivationPack, MoEWeightPack

    prepared = _CUTLASS_HANDLES[handle]
    pack = MoEWeightPack()
    # Rebuild the view from operator inputs, never from cached weight pointers.
    pack.prepare_for(prepared.backend_key, dict(zip(prepared.weight_names, weights, strict=True)))
    scales = None
    if prepared.prepare_activations is not None:
        x, scales = prepared.prepare_activations(x)  # [T,D/2], [T,D/16]
    activations = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=scales,
        topk_ids=ids,
        topk_weights=scores,
    )
    output = prepared.layer(activations, pack)  # [T,D] or list of outputs
    if isinstance(output, list):
        output = output[0]  # [T,D]
    return output.clone()  # [T,D]; do not expose a runner-owned output allocation


@flashinfer_cutlass_moe.register_fake
def _fake_flashinfer_cutlass_moe(
    x: torch.Tensor,  # [T,D]
    ids: torch.Tensor,  # [T,K]
    scores: torch.Tensor,  # [T,K]
    weights: list[torch.Tensor],  # backend-packed weights/scales
    handle: int,
) -> torch.Tensor:  # [T,D]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)  # [T,D]


@dataclass
class CutlassExperts:
    weights: list[torch.Tensor]  # backend-packed weights/scales
    handle: int
    name: str
    _finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._finalizer = weakref.finalize(self, _close_cutlass, self.weights[0].device, self.handle)
        self._finalizer.atexit = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T,D]
        scores: torch.Tensor,  # [T,K]
        indices: torch.Tensor,  # [T,K]
    ) -> torch.Tensor:  # [T,D]
        return flashinfer_cutlass_moe(
            hidden_states.contiguous(),
            indices.int().contiguous(),
            scores.float().contiguous(),
            self.weights,
            self.handle,
        )  # [T,D]

    def close(self) -> None:
        self._finalizer()


def _close_cutlass(device: torch.device, handle: int) -> None:
    torch.cuda.synchronize(device)
    del _CUTLASS_HANDLES[handle]


@dataclass
class CuteDSLWeights:
    w1: torch.Tensor  # [E,D,2I], interleaved
    w2: torch.Tensor  # [E,I,D]
    top_k: int
    name: str = "flash_infer_cutedsl"

    def prepare(self, num_tokens: int) -> CuteDSLExperts:
        """Rebuild shape-dependent buffers and the launch handle, reusing packed weights."""
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import bf16_mega_moe, get_symm_buffer_for_bf16_mega_moe

        experts, intermediate, hidden = self.w2.shape
        buffer = get_symm_buffer_for_bf16_mega_moe(
            experts, num_tokens, self.top_k, hidden, intermediate, 0, 1, enable_in_kernel_fc2_reduce=False
        )
        try:
            buffer.x.zero_()  # [T,D]
            buffer.topk_idx.copy_(torch.arange(self.top_k, device=self.w1.device).expand(num_tokens, -1))  # [T,K]
            buffer.topk_weights.fill_(1 / self.top_k)  # [T,K]
            output = torch.empty_like(buffer.x)  # [T,D]
            bf16_mega_moe(
                output, (self.w1, None), (self.w2, None), buffer, num_tokens=num_tokens, fast_math=False
            )  # [T,D]
            handle = prepare_flashinfer_cutedsl_moe(buffer)
        except Exception:
            buffer.destroy()
            raise
        return CuteDSLExperts(self.w1, self.w2, buffer, handle)


@dataclass
class CutlassWeights:
    weights: list[torch.Tensor]  # backend-packed weights/scales
    weight_names: tuple[str, ...]
    config: Any
    backend_key: str
    prepare_activations: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
    hidden_size: int
    name: str = "flash_infer_cutlass"

    def prepare(self, num_tokens: int) -> CutlassExperts:
        """Create and warm a shape-specific runner without repacking its weights."""
        from flashinfer.fused_moe import ExecutionConfig, MoELayer

        config = replace(self.config, execution=ExecutionConfig(tune_max_num_tokens=num_tokens))
        handle = next(_NEXT_HANDLE)
        _CUTLASS_HANDLES[handle] = _CutlassKernel(
            MoELayer(config), self.weight_names, self.backend_key, self.prepare_activations
        )
        prepared = CutlassExperts(self.weights, handle, self.name)
        try:
            x = torch.zeros(
                (num_tokens, self.hidden_size), dtype=torch.bfloat16, device=self.weights[0].device
            )  # [T,D]
            top_k = config.routing.top_k
            ids = torch.arange(top_k, device=x.device).expand(num_tokens, -1).contiguous()  # [T,K]
            scores = torch.ones((num_tokens, top_k), dtype=torch.float32, device=x.device) / top_k  # [T,K]
            prepared.forward(x, scores, ids)  # [T,D]; warm the exact shape before compilation/capture
        except Exception:
            prepared.close()
            raise
        return prepared


def pack_weights(
    gate_up_proj: torch.Tensor,  # [E,D,2I]
    down_proj: torch.Tensor,  # [E,I,D]
    top_k: int,
    kernel: str,
    precision: str,
) -> CuteDSLWeights | CutlassWeights:
    """Pack the selected implementation's weights independently of its workspace."""
    experts, hidden, twice_intermediate = gate_up_proj.shape
    intermediate = twice_intermediate // 2
    capability = torch.cuda.get_device_capability(gate_up_proj.device)
    if kernel == "cutedsl":
        if precision != "bf16" or capability != (10, 0):
            raise BackendUnavailable("CuTeDSL Mega requires BF16 on SM100")
        if hidden % 32 or intermediate % 64 or top_k > 32:
            raise BackendUnavailable("CuTeDSL requires hidden % 32 = 0, intermediate % 64 = 0, and top_k <= 32")
        if os.environ.get("MEGA_NO_DIST") != "1":
            raise BackendUnavailable("Set MEGA_NO_DIST=1 before importing FlashInfer for local CuTeDSL experts")
        from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import preprocess_mega_weights
        from flashinfer.moe_ep.weights import MoEWeightPack

        w1 = gate_up_proj.detach().transpose(1, 2).contiguous()  # [E,2I,D], gate/up
        w2 = down_proj.detach().transpose(1, 2).contiguous()  # [E,D,I]
        first, second = preprocess_mega_weights(
            MoEWeightPack(w13=w1, w2=w2), intermediate_size=intermediate, hidden_size=hidden
        )  # ([E,D,2I],None), ([E,I,D],None)
        return CuteDSLWeights(first[0], second[0], top_k)
    if kernel != "cutlass":
        raise BackendUnavailable(f"Unknown FlashInfer kernel: {kernel}")
    from flashinfer.fused_moe import (
        BackendOptions,
        CutlassBf16Config,
        CutlassNvfp4Config,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    backend: Any = CutlassBf16Config() if precision == "bf16" else CutlassNvfp4Config()
    if not backend.supported(capability[0] * 10 + capability[1]):
        raise BackendUnavailable(f"CUTLASS {precision} does not support SM{capability[0]}{capability[1]}")
    if hidden % 64 or intermediate % 16:
        raise BackendUnavailable("CUTLASS requires hidden % 64 = 0 and intermediate % 16 = 0")
    gate, up = gate_up_proj.detach().chunk(2, dim=2)  # 2x[E,D,I]
    w1 = torch.cat((up, gate), dim=2).transpose(1, 2).contiguous()  # [E,2I,D], up/gate
    w2 = down_proj.detach().transpose(1, 2).contiguous()  # [E,D,I]
    activation = SwiGLU()
    fmt = QuantFormat.BF16 if precision == "bf16" else QuantFormat.NVFP4
    config = MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=top_k),
        quant=QuantConfig(weight=fmt, activation=fmt),
        experts=ExpertConfig(intermediate_size=intermediate, local_num_experts=experts),
        activation=activation,
        backend=BackendOptions(candidates=(backend,)),
    )
    view = backend.prepare_weights(
        w1, w2, num_local_experts=experts, hidden_size=hidden, intermediate_size=intermediate, activation=activation
    )
    return CutlassWeights(
        list(view.values()),
        tuple(view),
        config,
        f"cutlass_{precision}",
        backend.prepare_activations if precision == "nvfp4" else None,
        hidden,
    )


def prepare(
    gate_up_proj: torch.Tensor,  # [E,D,2I]
    down_proj: torch.Tensor,  # [E,I,D]
    top_k: int,
    num_tokens: int,
    kernel: str,
    precision: str,
) -> CuteDSLExperts | CutlassExperts:
    """Prepare the selected implementation without changing the model's parameters."""
    return pack_weights(gate_up_proj, down_proj, top_k, kernel, precision).prepare(num_tokens)
