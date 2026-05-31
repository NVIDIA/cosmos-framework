# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
AVAE Tokenizer

This module provides the AVAE tokenizer with spec_convnext encoder,
oobleck decoder, and VAE bottleneck configuration.
"""

from cosmos_framework.model.vfm.tokenizers.audio.avae_utils.env import AttrDict
from cosmos_framework.model.vfm.tokenizers.audio.avae_utils.models import LatentAutoEncoderV2, load_generator

__all__ = [
    "LatentAutoEncoderV2",
    "AttrDict",
    "load_generator",
]
