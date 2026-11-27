# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pickle
import random
from typing import Any

import numpy as np
import torch

from cosmos_framework.utils.safe_unpickle import safe_loads


def get_rand_state_dict() -> dict[str, torch.Tensor]:
    """
    Get the random state dictionary. used to save the random state to a checkpoint.
    """
    numpy_packed_len, numpy_packed_bytes = pack_numpy_state(np.random.get_state())
    random_packed_len, random_packed_bytes = pack_random_state(random.getstate())

    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "numpy_packed_len": torch.tensor(numpy_packed_len, dtype=torch.long),  # []
        "numpy_packed_bytes": torch.frombuffer(numpy_packed_bytes, dtype=torch.uint8),  # [MAX_PACKED_LEN]
        "random_packed_len": torch.tensor(random_packed_len, dtype=torch.long),  # []
        "random_packed_bytes": torch.frombuffer(random_packed_bytes, dtype=torch.uint8),  # [MAX_PACKED_LEN]
    }


def set_rand_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """
    Set the random state dictionary. used to restore the random state from a checkpoint.
    """
    # Decode and validate both packed states before touching any generator, so a rejected checkpoint changes nothing.
    numpy_state = unpack_numpy_state(
        state_dict["numpy_packed_len"].item(), bytes(state_dict["numpy_packed_bytes"].tolist())
    )
    random_state = unpack_random_state(
        state_dict["random_packed_len"].item(), bytes(state_dict["random_packed_bytes"].tolist())
    )
    torch.set_rng_state(state_dict["torch"])
    torch.cuda.set_rng_state(state_dict["torch_cuda"])
    np.random.set_state(numpy_state)
    random.setstate(random_state)


# MAX padding length for the random state
# numpy and python random state are based on Mersenne Twister algorithm and state has aprox 625 integers.
# When we convert this data through pickle generated output can be of variable size.
# Pad the output to a fixed size to ensure that the size of the random state is always the same.
# Fixed size buffer is required for DCP checkpoint functions as it uses pinned preallocated memory to copy the checkpoint data.
MAX_PACKED_LEN = 4096


def pad_packed_bytes(packed_bytes: bytes) -> bytearray:
    padded = packed_bytes.ljust(MAX_PACKED_LEN, b"\0")

    return bytearray(padded)


def pack_numpy_state(state: tuple[int, ...]) -> tuple[int, bytearray]:
    packed_bytes = pickle.dumps(state)
    return len(packed_bytes), pad_packed_bytes(packed_bytes)


def pack_random_state(state: tuple[int, ...]) -> tuple[int, bytearray]:
    packed_bytes = pickle.dumps(state)
    return len(packed_bytes), pad_packed_bytes(packed_bytes)


# The packed bytes come from a checkpoint and are untrusted, so they are unpickled through an allow-list:
# a pickled ``np.random.get_state()`` references only these globals (``numpy._core`` on numpy >= 2,
# ``numpy.core`` on numpy 1.x; protocol 2 encodes the key buffer with ``_codecs.encode`` and protocol 5
# rebuilds it with ``_frombuffer``), and a pickled ``random.getstate()`` references none.
NUMPY_STATE_PICKLE_GLOBALS = frozenset(
    {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("_codecs", "encode"),
    }
)

# Mersenne Twister state size in 32-bit words.
MT19937_KEY_LEN = 624


def _unpickle_packed_state(packed_len: int, packed_bytes: bytes, allowed_globals: frozenset[tuple[str, str]]) -> Any:
    max_len = min(MAX_PACKED_LEN, len(packed_bytes))
    if type(packed_len) is not int or not 0 < packed_len <= max_len:
        raise ValueError(f"Packed random state length {packed_len!r} is outside (0, {max_len}].")
    return safe_loads(bytes(packed_bytes[:packed_len]), allowed_globals)


def unpack_numpy_state(packed_len: int, packed_bytes: bytes) -> tuple[str, np.ndarray, int, int, float]:
    state = _unpickle_packed_state(packed_len, packed_bytes, NUMPY_STATE_PICKLE_GLOBALS)
    if type(state) is not tuple or len(state) != 5:
        raise ValueError("Numpy random state must be a 5-tuple.")
    algorithm, key, pos, has_gauss, cached_gaussian = state
    if type(algorithm) is not str or algorithm != "MT19937":
        raise ValueError(f"Numpy random state algorithm must be 'MT19937', got {algorithm!r}.")
    if type(key) is not np.ndarray or key.dtype != np.uint32 or key.shape != (MT19937_KEY_LEN,):
        raise ValueError(f"Numpy random state key must be a uint32 array of shape ({MT19937_KEY_LEN},).")
    if type(pos) is not int or not 0 <= pos <= MT19937_KEY_LEN:
        raise ValueError(f"Numpy random state position must be in [0, {MT19937_KEY_LEN}], got {pos!r}.")
    if type(has_gauss) is not int or has_gauss not in (0, 1):
        raise ValueError(f"Numpy random state has_gauss must be 0 or 1, got {has_gauss!r}.")
    if type(cached_gaussian) is not float:
        raise ValueError(f"Numpy random state cached gaussian must be a float, got {type(cached_gaussian)}.")
    return state


def unpack_random_state(packed_len: int, packed_bytes: bytes) -> tuple[int, tuple[int, ...], float | None]:
    state = _unpickle_packed_state(packed_len, packed_bytes, frozenset())
    if type(state) is not tuple or len(state) != 3:
        raise ValueError("Python random state must be a 3-tuple.")
    version, internal_state, gauss_next = state
    if type(version) is not int or version != 3:
        raise ValueError(f"Python random state version must be 3, got {version!r}.")
    if (
        type(internal_state) is not tuple
        or len(internal_state) != MT19937_KEY_LEN + 1
        or any(type(word) is not int or not 0 <= word < 2**32 for word in internal_state[:-1])
        or type(internal_state[-1]) is not int
        or not 0 <= internal_state[-1] <= MT19937_KEY_LEN
    ):
        raise ValueError(
            f"Python random state must hold {MT19937_KEY_LEN} uint32 words and a position in [0, {MT19937_KEY_LEN}]."
        )
    if gauss_next is not None and type(gauss_next) is not float:
        raise ValueError(f"Python random state gauss_next must be None or a float, got {type(gauss_next)}.")
    return state


if __name__ == "__main__":
    print("--- Initializing Seeds ---")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # random operations to advance the state from the initial seed
    for _ in range(10):
        _ = torch.randn(1)
        _ = np.random.standard_normal()
        _ = random.gauss(0, 1)

    print("Capturing state...")
    state_dict = get_rand_state_dict()

    # Ensure tensors are actually padded to fixed size
    assert state_dict["numpy_packed_bytes"].numel() == MAX_PACKED_LEN, (
        f"Numpy buffer size mismatch! Expected {MAX_PACKED_LEN}, got {state_dict['numpy_packed_bytes'].numel()}"
    )
    assert state_dict["random_packed_bytes"].numel() == MAX_PACKED_LEN, (
        f"Random buffer size mismatch! Expected {MAX_PACKED_LEN}, got {state_dict['random_packed_bytes'].numel()}"
    )
    print(f"State captured. Buffers verified at {MAX_PACKED_LEN} bytes.")

    # generate a sequence of random numbers immediately after saving.
    print("Generating Ground Truth Sequence (A)...")
    seq_a = []
    for _ in range(5):
        vals = (
            torch.randn(1).item(),
            np.random.rand(),
            np.random.uniform(),
            random.random(),
        )
        seq_a.append(vals)

    # modify the state by generating random numbers
    print("Modifying state (Generating junk)...")
    for _ in range(20):
        _ = torch.rand(1)
        _ = np.random.rand()
        _ = random.random()

    # restore the state
    print("Restoring random state...")
    set_rand_state_dict(state_dict)

    # If restore works, this must match Sequence A exactly.
    print("Generating Post-Restore Sequence (B)...")
    seq_b = []
    for _ in range(5):
        vals = (
            torch.randn(1).item(),
            np.random.rand(),
            np.random.uniform(),
            random.random(),
        )
        seq_b.append(vals)

    print("\n--- Verification Results ---")
    all_match = True
    for i, (truth, actual) in enumerate(zip(seq_a, seq_b)):
        # Compare all values in the tuple with small tolerance for float precision
        row_match = all(abs(t - a) < 1e-9 for t, a in zip(truth, actual))
        print(f"  Row {i}: {truth} vs {actual} - {row_match}")
        if row_match:
            print(f"Step {i}: MATCH ")
        else:
            print(f"Step {i}: MISMATCH ")
            print(f"  Expected: {truth}")
            print(f"  Actual:   {actual}")
            all_match = False

    if all_match:
        print("\n All random states (Torch, Numpy, Python) restored successfully.")
    else:
        raise RuntimeError("Test Failed: Random states did not match after restore.")
