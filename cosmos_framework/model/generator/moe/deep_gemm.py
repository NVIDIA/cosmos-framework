# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Single-GPU inference adapter for DeepGEMM 2.8.0 (commit 78b6900) BF16 Mega MoE.

Weights must already be transformed and workspace allocated outside compilation.
The workspace is explicit and mutable; all other inputs are read-only and the
returned output is fresh. This adapter does not implement EP or autograd.
"""

from dataclasses import dataclass

import deep_gemm
import torch

from cosmos_framework.model.generator.moe.interface import BackendUnavailable


@torch.library.custom_op("cosmos_moe::deepgemm_bf16", mutates_args={"workspace"}, device_types="cuda")
def bf16_mega_moe(
    x: torch.Tensor,  # [T,D]
    ids: torch.Tensor,  # [T,K]
    scores: torch.Tensor,  # [T,K]
    w1: torch.Tensor,  # [E,2I,D], DeepGEMM-transformed
    w2: torch.Tensor,  # [E,D,I]
    workspace: torch.Tensor,  # [workspace_bytes], int8
    max_tokens: int,
    fast_math: bool = False,
) -> torch.Tensor:  # [T,D]
    tokens, hidden = x.shape
    experts, _, intermediate = w2.shape
    topk = ids.shape[1]
    # The pinned extension exposes the same layout factory used by SymmBuffer.
    # Derive views from this invocation's workspace, including functionalization clones.
    required_bytes, slice_buffers = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
        1, experts, max_tokens, topk, hidden, intermediate, "bf16xbf16", "swiglu", 0
    )
    if workspace.dtype != torch.int8 or not workspace.is_contiguous() or workspace.numel() < required_bytes:
        raise ValueError("Expected a contiguous int8 Mega MoE workspace of sufficient size")
    if tokens > max_tokens:
        raise ValueError("Input exceeds the allocated Mega MoE token capacity")
    buffer_x, _, buffer_ids, buffer_scores, *_ = slice_buffers(workspace)  # [M,D], unused, [M,K], [M,K], scratch
    buffer_x[:tokens].copy_(x)  # [T,D]
    buffer_ids[:tokens].copy_(ids)  # [T,K]
    buffer_scores[:tokens].copy_(scores)  # [T,K]
    output = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)  # [T,D]
    # Single rank only: use the current workspace pointer, including after functionalization.
    deep_gemm._C.bf16_mega_moe(
        output,
        w1,
        w2,
        None,
        None,
        None,
        workspace,
        [workspace.data_ptr()],
        0,
        max_tokens,
        experts,
        topk,
        "swiglu",
        None,
        fast_math,
    )  # output: [T,D]; workspace mutated
    return output  # [T,D]


@bf16_mega_moe.register_fake
def _fake_bf16_mega_moe(
    x: torch.Tensor,  # [T,D]
    ids: torch.Tensor,  # [T,K]
    scores: torch.Tensor,  # [T,K]
    w1: torch.Tensor,  # [E,2I,D]
    w2: torch.Tensor,  # [E,D,I]
    workspace: torch.Tensor,  # [workspace_bytes]
    max_tokens: int,
    fast_math: bool = False,
) -> torch.Tensor:  # [T,D]
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)  # [T,D], fresh contiguous output


@dataclass
class DeepGemmExperts:
    w1: torch.Tensor  # [E,2I,D]
    w2: torch.Tensor  # [E,D,I]
    workspace: torch.Tensor  # [workspace_bytes]
    num_tokens: int
    name: str = "deep_gemm"

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T,D]
        scores: torch.Tensor,  # [T,K]
        indices: torch.Tensor,  # [T,K]
    ) -> torch.Tensor:  # [T,D]
        return bf16_mega_moe(hidden_states, indices, scores, self.w1, self.w2, self.workspace, self.num_tokens)  # [T,D]

    def close(self) -> None:
        # No process group or NVSHMEM allocation is needed for this local workspace.
        torch.cuda.synchronize(self.workspace.device)


@dataclass
class DeepGemmWeights:
    w1: torch.Tensor  # [E,2I,D]
    w2: torch.Tensor  # [E,D,I]
    top_k: int
    name: str = "deep_gemm"

    def prepare(self, num_tokens: int) -> DeepGemmExperts:
        """Allocate and warm a new workspace while retaining the transformed weights."""
        experts, hidden, intermediate = self.w2.shape
        size, _ = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
            1, experts, num_tokens, self.top_k, hidden, intermediate, "bf16xbf16", "swiglu", 0
        )
        workspace = torch.zeros(size, dtype=torch.int8, device=self.w1.device)  # [workspace_bytes]
        prepared = DeepGemmExperts(self.w1, self.w2, workspace, num_tokens)
        x = self.w1.new_zeros((num_tokens, hidden))  # [T,D]
        ids = torch.arange(self.top_k, device=x.device).expand(num_tokens, -1).contiguous()  # [T,K]
        scores = torch.ones((num_tokens, self.top_k), dtype=torch.float32, device=x.device) / self.top_k  # [T,K]
        prepared.forward(x, scores, ids)  # [T,D]; JIT and warmup outside compilation/capture
        return prepared


def pack_weights(
    gate_up_proj: torch.Tensor,  # [E,D,2I]
    down_proj: torch.Tensor,  # [E,I,D]
    top_k: int,
    kernel: str,
    precision: str,
) -> DeepGemmWeights:
    """Transform canonical weights once, independently of the input token count."""
    experts, hidden, twice_intermediate = gate_up_proj.shape
    intermediate = twice_intermediate // 2
    if kernel != "mega" or precision != "bf16" or torch.cuda.get_device_capability(gate_up_proj.device)[0] != 10:
        raise BackendUnavailable("DeepGEMM Mega requires BF16 on SM10x")
    if hidden % 128 or intermediate % 64 or not 1 <= top_k <= 32:
        raise BackendUnavailable("DeepGEMM requires hidden % 128 = 0, intermediate % 64 = 0, and top_k <= 32")
    if not hasattr(deep_gemm._C, "get_symm_buffer_size_for_mega_moe") or not hasattr(deep_gemm._C, "bf16_mega_moe"):
        raise BackendUnavailable("Requires the DeepGEMM 2.8.0 / 78b6900 BF16 Mega API")
    w1 = gate_up_proj.detach().transpose(1, 2).contiguous()  # [E,2I,D]
    w2 = down_proj.detach().transpose(1, 2).contiguous()  # [E,D,I]
    w1, w2 = deep_gemm.transform_weights_for_mega_moe(w1, w2)  # [E,2I,D], [E,D,I]
    return DeepGemmWeights(w1, w2, top_k)


def prepare(
    gate_up_proj: torch.Tensor,  # [E,D,2I]
    down_proj: torch.Tensor,  # [E,I,D]
    top_k: int,
    num_tokens: int,
    kernel: str,
    precision: str,
) -> DeepGemmExperts:
    """Pack canonical weights and allocate an ordinary single-rank CUDA workspace."""
    return pack_weights(gate_up_proj, down_proj, top_k, kernel, precision).prepare(num_tokens)
