# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Primitive FLOPs building blocks shared by all model-specific estimators."""

import math


def compute_linear_flops(in_features: int, out_features: int, seq_len: int, has_bias: bool = True) -> int:
    """Compute FLOPs for a linear layer.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        seq_len: Sequence length
        has_bias: Whether the layer has bias

    Returns:
        Total FLOPs for the linear layer
    """
    # Matrix multiplication: 2 * seq_len * in_features * out_features
    # (2 accounts for multiply-add operations)
    matmul_flops = 2 * seq_len * in_features * out_features

    # Bias addition if present
    bias_flops = seq_len * out_features if has_bias else 0

    return matmul_flops + bias_flops


def compute_attention_flops(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int | None = None,
    is_causal: bool = False,
    has_bias: bool = False,
) -> int:
    """Compute FLOPs for attention mechanism.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        head_dim: Dimension per head (defaults to hidden_size // num_heads)
        is_causal: If True, account for causal masking. Causal attention
            computes only ~half of the S^2 attention entries (the lower
            triangle); FlashAttention-2 with causal=True skips upper-triangle
            blocks at block granularity, so this matches both the algorithmic
            count and the kernel work for the causal case. Defaults to False
            for bidirectional attention (e.g. the vision encoder); set True
            for causal decoders.
        has_bias: Whether Q/K/V/O projections include bias terms.

    Returns:
        Total FLOPs for attention
    """
    if head_dim is None:
        head_dim = hidden_size // num_heads

    # QKV projection: 3 linear layers (but KV uses num_kv_heads)
    q_proj_flops = compute_linear_flops(hidden_size, num_heads * head_dim, seq_len, has_bias=has_bias)
    k_proj_flops = compute_linear_flops(hidden_size, num_kv_heads * head_dim, seq_len, has_bias=has_bias)
    v_proj_flops = compute_linear_flops(hidden_size, num_kv_heads * head_dim, seq_len, has_bias=has_bias)

    # Causal masking halves the work that scales with S^2 (QK^T, softmax,
    # attn @ V). The QKV / output projections are not affected.
    causal_factor = 0.5 if is_causal else 1.0

    # Q @ K^T: (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
    # = (batch, num_heads, seq_len, seq_len)
    qk_matmul_flops = int(2 * num_heads * seq_len * seq_len * head_dim * causal_factor)

    # Softmax: approximately 3 * num_heads * seq_len * seq_len (exp, sum, divide)
    softmax_flops = int(3 * num_heads * seq_len * seq_len * causal_factor)

    # Attention @ V: (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
    # = (batch, num_heads, seq_len, head_dim)
    attn_v_matmul_flops = int(2 * num_heads * seq_len * seq_len * head_dim * causal_factor)

    # Output projection
    o_proj_flops = compute_linear_flops(num_heads * head_dim, hidden_size, seq_len, has_bias=has_bias)

    total_flops = (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + qk_matmul_flops
        + softmax_flops
        + attn_v_matmul_flops
        + o_proj_flops
    )

    return total_flops


def compute_mlp_flops(seq_len: int, hidden_size: int, intermediate_size: int, use_swiglu: bool = True) -> int:
    """Compute FLOPs for MLP layer.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension
        intermediate_size: Intermediate dimension
        use_swiglu: Whether using SwiGLU activation (requires gate and up projections)

    Returns:
        Total FLOPs for MLP
    """
    if use_swiglu:
        # SwiGLU: gate_proj, up_proj, down_proj
        gate_proj_flops = compute_linear_flops(hidden_size, intermediate_size, seq_len, has_bias=False)
        up_proj_flops = compute_linear_flops(hidden_size, intermediate_size, seq_len, has_bias=False)
        down_proj_flops = compute_linear_flops(intermediate_size, hidden_size, seq_len, has_bias=False)

        # Activation (SiLU) + element-wise multiply: ~2 ops per element
        activation_flops = 2 * seq_len * intermediate_size
        multiply_flops = seq_len * intermediate_size

        total_flops = gate_proj_flops + up_proj_flops + down_proj_flops + activation_flops + multiply_flops
    else:
        # Standard MLP: fc1, activation, fc2
        fc1_flops = compute_linear_flops(hidden_size, intermediate_size, seq_len, has_bias=True)
        fc2_flops = compute_linear_flops(intermediate_size, hidden_size, seq_len, has_bias=True)
        activation_flops = seq_len * intermediate_size

        total_flops = fc1_flops + fc2_flops + activation_flops

    return total_flops


def compute_moe_flops(
    seq_len: int,
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    use_swiglu: bool = True,
) -> int:
    """Compute FLOPs for Mixture of Experts (MoE) layer.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension
        moe_intermediate_size: Intermediate dimension for each expert
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts activated per token (top-k)
        use_swiglu: Whether using SwiGLU activation (requires gate and up projections)

    Returns:
        Total FLOPs for MoE layer

    Note:
        MoE uses sparse computation - only num_experts_per_tok out of num_experts
        are activated per token. Each expert has its own gate_proj, up_proj, down_proj.
    """
    if seq_len == 0:
        return 0

    # Router FLOPs: linear projection to select experts
    router_flops = compute_linear_flops(hidden_size, num_experts, seq_len, has_bias=False)

    # Softmax over experts: ~3 ops per element (exp, sum, divide)
    softmax_flops = 3 * seq_len * num_experts

    # Top-k selection: approximate as O(k * log(n)) operations per token
    # Using conservative estimate: num_experts_per_tok * log2(num_experts) ops per token
    topk_flops = seq_len * num_experts_per_tok * int(math.log2(num_experts))

    # Expert computation FLOPs
    # Only num_experts_per_tok experts are active per token (sparse computation)
    if use_swiglu:
        # Each active expert: gate_proj, up_proj, down_proj with moe_intermediate_size
        # Note: gate_up_proj is fused in implementation but we count separately for clarity
        gate_proj_flops = compute_linear_flops(hidden_size, moe_intermediate_size, seq_len, has_bias=False)
        up_proj_flops = compute_linear_flops(hidden_size, moe_intermediate_size, seq_len, has_bias=False)
        down_proj_flops = compute_linear_flops(moe_intermediate_size, hidden_size, seq_len, has_bias=False)

        # Activation (SiLU) + element-wise multiply
        activation_flops = 2 * seq_len * moe_intermediate_size
        multiply_flops = seq_len * moe_intermediate_size

        total_expert_flops = num_experts_per_tok * (
            gate_proj_flops + up_proj_flops + down_proj_flops + activation_flops + multiply_flops
        )
    else:
        # Standard MLP-style expert
        fc1_flops = compute_linear_flops(hidden_size, moe_intermediate_size, seq_len, has_bias=True)
        fc2_flops = compute_linear_flops(moe_intermediate_size, hidden_size, seq_len, has_bias=True)
        activation_flops = seq_len * moe_intermediate_size

        total_expert_flops = num_experts_per_tok * (fc1_flops + fc2_flops + activation_flops)

    # Weighted sum of expert outputs (element-wise multiply + sum)
    # Each token combines num_experts_per_tok expert outputs
    weighted_sum_flops = seq_len * num_experts_per_tok * hidden_size

    total_flops = router_flops + softmax_flops + topk_flops + total_expert_flops + weighted_sum_flops

    return int(total_flops)


def compute_layernorm_flops(seq_len: int, hidden_size: int) -> int:
    """Compute FLOPs for layer normalization.

    Args:
        seq_len: Sequence length
        hidden_size: Hidden dimension

    Returns:
        Total FLOPs for LayerNorm
    """
    # Mean: sum + divide
    # Variance: (x - mean)^2, sum, divide
    # Normalize: (x - mean) / sqrt(var + eps)
    # Scale and shift: x * weight + bias
    # Approximately 5 operations per element
    return 5 * seq_len * hidden_size
