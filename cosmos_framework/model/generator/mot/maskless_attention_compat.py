# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Source/API prerequisites for the experimental SM120 gapped forward path.

This is not the generic Flash4 or block-sparse attention support policy.
Keep this module CUDA-free so prerequisite checks can be tested on CPU.
"""

from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, signature
from types import ModuleType

_SM120_FA4_VERSION = "4.0.0b30"
_GAPPED_FORWARD_KEYWORDS = frozenset(
    {
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "softmax_scale",
        "out",
        "lse",
        "return_lse",
    }
)


def sm120_gapped_forward_unavailable_reason(backend: ModuleType) -> str | None:
    """Reject known-unsafe releases and changed private forward APIs.

    Stock b24 and b28 predate the invalid-work-tile fix in FA4 PR #2763.
    b30 contains that fix. Do not approve unreviewed future versions or
    same-version local backports solely because the module imports.
    Distribution metadata is a compatibility check, not source attestation.
    """
    try:
        installed = version("flash-attn-4")
    except PackageNotFoundError:
        return "experimental SM120 replay needs flash-attn-4 distribution metadata"
    if installed != _SM120_FA4_VERSION:
        return (
            f"experimental SM120 replay needs flash-attn-4=={_SM120_FA4_VERSION} "
            f"with the invalid-varlen-tile fix (FA4 PR #2763), got {installed}; "
            "install its complete dependencies in an isolated environment"
        )
    forward = getattr(backend, "_flash_attn_fwd", None)
    if not callable(forward):
        return "experimental SM120 replay needs the FA4 _flash_attn_fwd entry point"
    try:
        forward_signature = signature(forward)
        parameters = forward_signature.parameters
    except (TypeError, ValueError):
        return "experimental SM120 replay could not inspect the FA4 forward API"
    for name in sorted(_GAPPED_FORWARD_KEYWORDS):
        parameter = parameters.get(name)
        if parameter is None or parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            return f"experimental SM120 replay needs the explicit FA4 forward keyword {name}"
    try:
        forward_signature.bind(None, None, None, **dict.fromkeys(_GAPPED_FORWARD_KEYWORDS))
    except TypeError:
        return "experimental SM120 replay needs the reviewed three-tensor FA4 forward calling convention"
    return None
