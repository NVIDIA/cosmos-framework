# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Core neural network components for tokenizers.

This module contains building blocks for tokenizer networks:
    - sparse_tensor: SparseTensor data structure and operations
    - sparse_ops: Linear, normalization, and activation layers for sparse tensors
    - attention: Attention mechanisms for sparse tensors
    - quantizers: Vector quantization (FSQ, LFQ)
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

from loguru import logger as logging

if TYPE_CHECKING:
    from cosmos_framework.model.tokenizer.models.modules.quantizers.fsq import FSQ as FSQ
    from cosmos_framework.model.tokenizer.models.modules.quantizers.fsq import (
        levels_from_codebook_size as levels_from_codebook_size,
    )
    from cosmos_framework.model.tokenizer.models.modules.quantizers.lfq import LFQ as LFQ
    from cosmos_framework.model.tokenizer.models.modules.quantizers.lfq import LossBreakdown as LossBreakdown
    from cosmos_framework.model.tokenizer.models.modules.quantizers.residual_vq import RQBottleneck as RQBottleneck
    from cosmos_framework.model.tokenizer.models.modules.quantizers.residual_vq import VQEmbedding as VQEmbedding
    from cosmos_framework.model.tokenizer.models.modules.sparse_ops import LayerNorm32 as LayerNorm32
    from cosmos_framework.model.tokenizer.models.modules.sparse_ops import RMSNorm32 as RMSNorm32
    from cosmos_framework.model.tokenizer.models.modules.sparse_ops import SparseLinear as SparseLinear
    from cosmos_framework.model.tokenizer.models.modules.sparse_tensor import SparseTensor as SparseTensor
    from cosmos_framework.model.tokenizer.models.modules.sparse_tensor import sparse_cat as sparse_cat
    from cosmos_framework.model.tokenizer.models.modules.sparse_tensor import sparse_unbind as sparse_unbind
    from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import (
        AbsolutePositionEmbedder as AbsolutePositionEmbedder,
    )
    from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import (
        LearnedPositionEmbedder as LearnedPositionEmbedder,
    )
    from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import (
        LearnedPositionEmbedder4D as LearnedPositionEmbedder4D,
    )
    from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import (
        SparseMultiheadAttentionPoolingHead as SparseMultiheadAttentionPoolingHead,
    )
    from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import (
        SparseTransformerBlock as SparseTransformerBlock,
    )

# Backend configuration
BACKEND: str = "pytorch"
DEBUG: bool = False

# Valid backend options
_VALID_BACKENDS = ["pytorch", "spconv", "torchsparse"]


def _init_from_env() -> None:
    """Initialize backend settings from environment variables."""
    global BACKEND, DEBUG

    env_sparse_backend = os.environ.get("SPARSE_BACKEND")
    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    env_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get("ATTN_BACKEND")

    if env_sparse_backend is not None and env_sparse_backend in _VALID_BACKENDS:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == "1"
    if env_sparse_attn is not None:
        logging.warning(
            f"Ignoring sparse tokenizer attention backend override {env_sparse_attn!r}. "
            "Tokenizer sparse attention now defers backend selection to cosmos_framework.model.attention. "
            "If you need to filter i4 backend choices, use I4_ATTN_BACKENDS instead."
        )


_init_from_env()


# Lazy loading attribute mapping
_ATTRIBUTES = {
    # SparseTensor and operations
    "SparseTensor": "sparse_tensor",
    "PureTorchSparseTensor": "sparse_tensor",
    "sparse_batch_broadcast": "sparse_tensor",
    "sparse_batch_op": "sparse_tensor",
    "sparse_cat": "sparse_tensor",
    "sparse_unbind": "sparse_tensor",
    "reconstruct_from_temporal_slices": "sparse_tensor",
    # Linear layers
    "SparseLinear": "sparse_ops",
    # Normalization layers
    "SparseGroupNorm": "sparse_ops",
    "SparseLayerNorm": "sparse_ops",
    "SparseGroupNorm32": "sparse_ops",
    "SparseLayerNorm32": "sparse_ops",
    "SparseRMSNorm32": "sparse_ops",
    "LayerNorm32": "sparse_ops",
    "GroupNorm32": "sparse_ops",
    "ChannelLayerNorm32": "sparse_ops",
    "RMSNorm": "sparse_ops",
    "RMSNorm32": "sparse_ops",
    # Activation functions
    "SparseReLU": "sparse_ops",
    "SparseSiLU": "sparse_ops",
    "SparseGELU": "sparse_ops",
    "SparseActivation": "sparse_ops",
    # Spatial operations
    "SparseDownsample": "sparse_ops",
    "SparseDownsampleKeepCoords": "sparse_ops",
    "SparseUpsample": "sparse_ops",
    "SparseUpsampleTokenSplit": "sparse_ops",
    "SparseSubdivide": "sparse_ops",
    "SparseUpsampleNoCache": "sparse_ops",
    # Attention modules
    "sparse_scaled_dot_product_attention": "attention.full_attn",
    "RotaryPositionEmbedder": "attention.modules",
    "SparseMultiHeadRMSNorm": "attention.modules",
    "SparseMultiHeadAttention": "attention.modules",
    # Quantizers
    "FSQ": "quantizers.fsq",
    "levels_from_codebook_size": "quantizers.fsq",
    "LFQ": "quantizers.lfq",
    "LossBreakdown": "quantizers.lfq",
    "RQBottleneck": "quantizers.residual_vq",
    "VQEmbedding": "quantizers.residual_vq",
    # Transformer blocks
    "AbsolutePositionEmbedder": "transformer.blocks",
    "LearnedPositionEmbedder": "transformer.blocks",
    "LearnedPositionEmbedder4D": "transformer.blocks",
    "SparseFeedForwardNet": "transformer.blocks",
    "SparseMultiheadAttentionPoolingHead": "transformer.blocks",
    "SparseTransformerBlock": "transformer.blocks",
    "SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_FULL_LAYER": "transformer.blocks",
    "SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_MLP_ONLY": "transformer.blocks",
    "SPARSE_TRANSFORMER_CHECKPOINT_SCOPES": "transformer.blocks",
    "ModulatedSparseTransformerBlock": "transformer.modulated",
    "ModulatedSparseTransformerCrossBlock": "transformer.modulated",
}

__all__ = list(_ATTRIBUTES.keys())


def __getattr__(name: str) -> Any:
    """Lazy import of module attributes."""
    if name not in globals():
        if name in _ATTRIBUTES:
            module_name = _ATTRIBUTES[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]
