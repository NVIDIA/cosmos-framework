# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stable domain-separated seed derivation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def derive_u64_seeds(root_seeds: Sequence[int], *, domain: bytes) -> list[int]:
    """Derive one unsigned 64-bit seed per root without process-local hash state.

    CI repro: trivial no-op change to trigger the posttrain CPU pipeline.
    """
    return [
        int.from_bytes(
            hashlib.sha256(domain + root_seed.to_bytes(8, "big", signed=False)).digest()[:8],
            "big",
        )
        for root_seed in root_seeds
    ]
