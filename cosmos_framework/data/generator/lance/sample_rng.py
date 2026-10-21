# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-sample random-number isolation for Lance augmentation pipelines."""

import hashlib
import json
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

LANCE_SAMPLE_RNG_SEED_KEY = "_lance_sample_rng_seed"
LANCE_SAMPLE_RESUME_CURSOR_KEY = "_lance_sample_resume_cursor"
_TORCH_SEED_MASK = (1 << 63) - 1
_NUMPY_SEED_MODULUS = 1 << 32


def derive_lance_sample_seed(
    *,
    base_seed: int,
    pass_index: int,
    slot: int,
    source_name: str,
    source_variant: str,
    sample_id: object,
) -> int:
    """Derive a stable augmentation seed from one sample's Lance stream identity."""
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}.")
    if pass_index < 0:
        raise ValueError(f"pass_index must be non-negative, got {pass_index}.")
    if slot < 0:
        raise ValueError(f"slot must be non-negative, got {slot}.")

    payload = json.dumps(
        [base_seed, pass_index, slot, source_name, source_variant, str(sample_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & _TORCH_SEED_MASK


def attach_lance_sample_seed(
    sample: dict,
    *,
    base_seed: int,
    pass_index: int,
    slot: int,
    source_name: str,
    source_variant: str,
) -> dict:
    """Attach the private deterministic augmentation seed consumed by the Lance provider."""
    sample[LANCE_SAMPLE_RNG_SEED_KEY] = derive_lance_sample_seed(
        base_seed=base_seed,
        pass_index=pass_index,
        slot=slot,
        source_name=source_name,
        source_variant=source_variant,
        sample_id=sample.get("__key__", ""),
    )
    return sample


@contextmanager
def isolated_sample_rng(seed: int) -> Iterator[None]:
    """Temporarily seed Python, NumPy, and Torch CPU RNGs for one sample."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()  # [N_rng_state]
    try:
        random.seed(seed)
        np.random.seed(seed % _NUMPY_SEED_MODULUS)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
