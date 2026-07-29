# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transformer blocks for sparse tokenizers.

This module provides transformer block implementations:
    - blocks: SparseTransformerBlock, SparseFeedForwardNet, AbsolutePositionEmbedder
    - modulated: ModulatedSparseTransformerBlock with adaptive layer norm conditioning
"""

from cosmos_framework.model.tokenizer.models.modules.transformer.blocks import (
    SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_FULL_LAYER,
    SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_MLP_ONLY,
    SPARSE_TRANSFORMER_CHECKPOINT_SCOPES,
    AbsolutePositionEmbedder,
    LearnedPositionEmbedder,
    LearnedPositionEmbedder4D,
    SparseFeedForwardNet,
    SparseMultiheadAttentionPoolingHead,
    SparseTransformerBlock,
)
from cosmos_framework.model.tokenizer.models.modules.transformer.modulated import (
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerCrossBlock,
)

__all__ = [
    "AbsolutePositionEmbedder",
    "LearnedPositionEmbedder",
    "LearnedPositionEmbedder4D",
    "SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_FULL_LAYER",
    "SPARSE_TRANSFORMER_CHECKPOINT_SCOPE_MLP_ONLY",
    "SPARSE_TRANSFORMER_CHECKPOINT_SCOPES",
    "SparseFeedForwardNet",
    "SparseMultiheadAttentionPoolingHead",
    "SparseTransformerBlock",
    "ModulatedSparseTransformerBlock",
    "ModulatedSparseTransformerCrossBlock",
]
