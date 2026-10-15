# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v4 (flash4) Backend
"""

import torch

from cosmos_framework.model.attention.utils.safe_ops import log


def flash4_supported() -> bool:
    """
    Returns whether Flash Attention v4 is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of the `flash_attn.cute` module, which carries the CuTe-DSL
          (FA4) kernels, and both of its varlen / dense entrypoints
        * Presence of the NVIDIA CUTLASS Python DSL, which `flash_attn.cute`
          JIT-compiles through

    This check guards imports / dependencies on the Flash Attention package.

    NOTE: unlike flash2 / flash3 there is deliberately no version gate here.
    FA4 ships *inside* the `flash_attn` distribution as the `flash_attn.cute`
    subpackage, and `flash_attn.__version__` reports the Flash Attention v2
    version of the surrounding wheel (e.g. "2.7.4.post1"), which says nothing
    about whether the FA4 kernels are present. `flash_attn.cute.__version__`
    resolves to "0.0.0" unless the separate `fa4` distribution is installed.
    Importability of the entrypoints is therefore the only reliable signal.
    """
    if not torch.cuda.is_available():
        log.debug("Flash Attention v4 is not supported because PyTorch did not detect CUDA runtime.")
        return False

    try:
        # pyrefly: ignore  # missing-import
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func  # noqa: F401
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd  # noqa: F401

    except ImportError:
        log.debug(
            "Flash Attention v4 is not supported because the 'flash_attn.cute' module was not found. "
            "FA4 ships as a subpackage of the 'flash_attn' distribution; a wheel carrying only the v2 "
            "kernels will not provide it."
        )
        return False
    except Exception as e:
        log.debug(f"Flash Attention v4 is not supported because importing 'flash_attn.cute' failed: {e}")
        return False

    try:
        # pyrefly: ignore  # missing-import
        import cutlass  # noqa: F401

    except ImportError:
        log.debug(
            "Flash Attention v4 is not supported because the NVIDIA CUTLASS Python DSL "
            "('nvidia-cutlass-dsl') was not found; the FA4 kernels JIT-compile through it."
        )
        return False
    except Exception as e:
        log.debug(f"Flash Attention v4 is not supported because importing the CUTLASS DSL failed: {e}")
        return False

    return True


FLASH4_SUPPORTED = flash4_supported()


if FLASH4_SUPPORTED:
    from cosmos_framework.model.attention.flash4.functions import flash4_attention

else:
    from cosmos_framework.model.attention.flash4.stubs import flash4_attention

__all__ = ["flash4_attention", "FLASH4_SUPPORTED"]
