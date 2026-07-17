# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

cuDNN Backend
"""

import torch

from cosmos_framework.model.attention.utils.safe_ops import log

# Minimum cuDNN runtime version, in ``torch.backends.cudnn.version()`` encoding
# (major * 10000 + minor * 100 + patch). 92200 == cuDNN 9.22.0.
CUDNN_MIN_BACKEND_VERSION = 92200


def cudnn_supported() -> bool:
    """
    Returns whether cuDNN Attention is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of the cuDNN runtime that ships with / is linked by PyTorch, meeting the minimum
          version requirement

    The backend runs cuDNN attention through PyTorch's own SDPA cuDNN dispatch
    (``F.scaled_dot_product_attention`` / ``aten._scaled_dot_product_cudnn_attention``), so it no
    longer depends on the standalone cuDNN Python frontend package -- only on the cuDNN that PyTorch
    itself uses.
    """
    if not torch.cuda.is_available():
        log.debug("cuDNN Attention is not supported because PyTorch did not detect CUDA runtime.")
        return False

    if not torch.backends.cudnn.is_available():
        log.debug("cuDNN Attention is not supported because PyTorch reports cuDNN is unavailable.")
        return False

    backend_version = torch.backends.cudnn.version()
    if backend_version is None or backend_version < CUDNN_MIN_BACKEND_VERSION:
        log.debug(
            "cuDNN Attention is not supported due to insufficient cuDNN runtime version "
            f"{backend_version=}, expected at least {CUDNN_MIN_BACKEND_VERSION=}."
        )
        return False

    # The cuDNN SDPA ATen op is the mechanism this backend relies on; if it is missing (unexpected on
    # a CUDA build), decline rather than fail later at call time.
    if not hasattr(torch.ops.aten, "_scaled_dot_product_cudnn_attention"):
        log.debug("cuDNN Attention is not supported because torch lacks the cuDNN SDPA ATen op.")
        return False

    return True


CUDNN_SUPPORTED = cudnn_supported()


if CUDNN_SUPPORTED:
    from cosmos_framework.model.attention.cudnn.functions import cudnn_attention

else:
    from cosmos_framework.model.attention.cudnn.stubs import cudnn_attention

__all__ = ["cudnn_attention", "CUDNN_SUPPORTED"]
