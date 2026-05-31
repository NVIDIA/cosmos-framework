# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Vector quantization modules for tokenizers.

This module provides quantization implementations:
    - fsq: Finite Scalar Quantization
    - lfq: Lookup-Free Quantization
    - residual_vq: Residual Quantization (RQ)
"""

from cosmos_framework.model.tokenizer.models.modules.quantizers.fsq import FSQ, levels_from_codebook_size
from cosmos_framework.model.tokenizer.models.modules.quantizers.lfq import LFQ, LossBreakdown
from cosmos_framework.model.tokenizer.models.modules.quantizers.residual_vq import RQBottleneck, VQEmbedding

__all__ = [
    "FSQ",
    "levels_from_codebook_size",
    "LFQ",
    "LossBreakdown",
    "RQBottleneck",
    "VQEmbedding",
]
