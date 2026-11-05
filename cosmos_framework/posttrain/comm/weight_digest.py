# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact-byte digests of tensors, small enough to cross a worker RPC.

The weight plane proves a load by comparing what a consumer holds against what the producer sent. When the
consumer's live parameters sit in another process (a vLLM engine worker, a vLLM-Omni diffusion worker), the
tensors cannot be compared in place, so each side reduces them to this metadata and only the metadata crosses.
Shared by the vLLM packed-QKV witness (`roles/rollout/vllm_weights.py`) and the vLLM-Omni bucket verification
(`roles/rollout/vllm_omni_worker.py`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TypedDict

import torch


class TensorDigestMetadata(TypedDict):
    shape: list[int]
    dtype: str
    sha256: str


def normalize_tensor_digest(value: object) -> TensorDigestMetadata:
    """Validate RPC metadata and normalize only the list/tuple representation of its shape."""
    if not isinstance(value, Mapping) or set(value) != {"shape", "dtype", "sha256"}:
        raise TypeError("malformed tensor digest metadata: expected shape, dtype and sha256 fields")
    shape, dtype, digest = value["shape"], value["dtype"], value["sha256"]
    if not isinstance(shape, (list, tuple)) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise TypeError(f"malformed tensor digest metadata: shape={shape!r}")
    if not isinstance(dtype, str) or not dtype:
        raise TypeError(f"malformed tensor digest metadata: dtype={dtype!r}")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise TypeError(f"malformed tensor digest metadata: sha256={digest!r}")
    return {"shape": list(shape), "dtype": dtype, "sha256": digest}


def tensor_digests_equal(actual: object, expected: object) -> bool:
    """Compare validated tensor digests, including shape and dtype."""
    return normalize_tensor_digest(actual) == normalize_tensor_digest(expected)


def tensor_digest_metadata(tensor: torch.Tensor) -> TensorDigestMetadata:
    """Return exact-byte SHA256 plus shape/dtype as small control metadata."""
    contiguous = tensor.detach().contiguous()
    byte_view = contiguous.view(torch.uint8).reshape(-1).cpu()  # [N_bytes]
    return {
        # JSON-shaped: this crosses worker-result aggregation and Ray, either of which may normalize
        # containers before the final equality check, so shape is a list rather than a tuple.
        "shape": [int(dim) for dim in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
    }
