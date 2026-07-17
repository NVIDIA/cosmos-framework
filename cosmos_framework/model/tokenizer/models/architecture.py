# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared UniAE architecture specifications used by training and inference."""

from __future__ import annotations

import copy
from typing import Any

_SIGLIP2_SO400M_COMMON_ARCH: dict[str, Any] = {
    "patch_size": (4, 16, 16),
    "in_channels": 3072,
    "out_channels": 3072,
    "encoder_model_channels": 1152,
    "encoder_num_blocks": 27,
    "encoder_num_heads": 16,
    "encoder_mlp_channels": 4304,
    "encoder_pe_mode": "joint",
    "encoder_qk_rms_norm": False,
    "encoder_use_bias": True,
    "encoder_use_rms_norm": False,
    "decoder_model_channels": 1152,
    "decoder_num_blocks": 27,
    "decoder_num_heads": 16,
    "decoder_mlp_channels": 4304,
    "decoder_pe_mode": "joint",
    "decoder_qk_rms_norm": True,
    "decoder_use_bias": False,
    "decoder_use_rms_norm": True,
    "use_decoder": True,
    "quantizer_type": "rq",
    "quantizer_codebook_size": 65536,
    # Production checkpoints use four residual-quantization stages.
    "quantizer_num_codebooks": 4,
    "quantizer_chunk_size": 1,
    "use_vf_loss": False,
    "freeze_encoder": False,
    "pretrained_model_name": "google/siglip2-so400m-patch16-naflex",
    "concat_latent": None,
    "random_num_sample_frames_batch_sizes": [8, 12, 16, 20, 24],
    "inference_num_sample_frames_batch_size": 16,
    "inference_num_sample_frames_stride": 16,
    "inference_kv_cache_size": 0,
}


def get_siglip2_so400m_common_arch() -> dict[str, Any]:
    """Return an isolated copy of the shared SigLIP2-SO400M UniAE architecture."""
    return copy.deepcopy(_SIGLIP2_SO400M_COMMON_ARCH)


__all__ = ["get_siglip2_so400m_common_arch"]
