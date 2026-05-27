# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 tokenizer models.

This module provides:
    - modules: Low-level building blocks (SparseTensor, quantizers, attention)
    - utils: Generic utilities (image processing, metrics, logging)
    - sparse_autoencoder: AutoencoderKL model for image/video tokenization
"""

# Generic utilities
# Metrics (moved from utils to metrics module for consolidation)
# Dense runtime
from cosmos_framework.model.tokenizer.models.dense_runtime import (
    DenseAutoencoderRuntime,
    DenseGridMetadata,
    DenseTemporalChunkSpec,
)

# Quantizer utilities
from cosmos_framework.model.tokenizer.models.modules.quantizers import levels_from_codebook_size

# Model classes (from sparse_autoencoder)
from cosmos_framework.model.tokenizer.models.sparse_autoencoder import (
    AutoencoderKL,
    AutoencoderKLConfig,
    Decoder,
    DiagonalGaussianDistribution,
    Encoder,
    SparseTransformerBase,
)
from cosmos_framework.model.tokenizer.models.utils import (
    SampleLogger,
    average_with_scatter_add,
    batch_tensor_to_sparse,
    crop_tensors_to_match,
    reconstruct_from_temporal_slices,
    resize_and_crop,
    restore_original_shape,
    sparse_to_img_list,
    split_temporal_dimension,
)

__all__ = [
    # Utils
    "average_with_scatter_add",
    "batch_tensor_to_sparse",
    "crop_tensors_to_match",
    "reconstruct_from_temporal_slices",
    "resize_and_crop",
    "restore_original_shape",
    "SampleLogger",
    "sparse_to_img_list",
    "split_temporal_dimension",
    "DenseAutoencoderRuntime",
    "DenseGridMetadata",
    "DenseTemporalChunkSpec",
    # Quantizer utilities
    "levels_from_codebook_size",
    # Model classes
    "AutoencoderKL",
    "AutoencoderKLConfig",
    "Decoder",
    "DiagonalGaussianDistribution",
    "Encoder",
    "SparseTransformerBase",
]
