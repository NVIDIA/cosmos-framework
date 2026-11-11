# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Distributed checkpoint capture and restoration for process RNG streams."""

from __future__ import annotations

from typing import Any


def validate_exact_rng_state(states: Any, *, device: Any) -> None:
    """Validate a same-topology checkpoint using private generators only.

    This is opt-in for finite SFT exact resume. Legacy optional-RNG restoration
    remains available to the older lifecycle contracts.
    """
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.utils.rank_state import select_rank_state

    state = select_rank_state(states, on_world_size_mismatch="raise")
    if not isinstance(state, dict):
        raise ValueError("finite SFT exact resume requires rank-complete RNG state")
    device = torch.device(device)
    required = {"python", "numpy", "torch"} | ({"cuda"} if device.type == "cuda" else set())
    if not required.issubset(state):
        raise ValueError(f"finite SFT RNG state is missing {sorted(required - set(state))}")
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    for key, generator_device in (("torch", "cpu"), ("cuda", device)):
        if key == "cuda" and device.type != "cuda":
            continue
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(f"finite SFT RNG {key} must be a one-dimensional uint8 tensor")
        torch.Generator(device=generator_device).set_state(value.cpu())


def collect_rng_state(cuda_device: Any = None) -> dict[int, dict[str, Any]]:
    """Gather every rank's Python, NumPy, Torch, and optional CUDA RNG state."""
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

    mine: dict[str, Any] | None = None
    capture_error: BaseException | None = None
    try:
        mine = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),  # [N]
        }
        if cuda_device is not None:
            device = torch.device(cuda_device)
            if device.type != "cuda":
                raise ValueError(f"CUDA RNG capture requires a CUDA device, got {device}")
            with torch.cuda.device(device):
                mine["cuda"] = torch.cuda.get_rng_state(device)  # [N]
        elif torch.cuda.is_available():
            mine["cuda"] = torch.cuda.get_rng_state()  # [N]
    except BaseException as error:  # noqa: BLE001 -- converge before rank-state all-gather
        capture_error = error
    raise_if_any_rank_failed(capture_error, "RNG state capture")
    if mine is None:
        raise AssertionError("RNG state capture converged without local state")
    return gather_rank_states(mine, context="RNG checkpoint state preparation")


def restore_rng_state(states: dict[int, dict[str, Any]] | None, cuda_device: Any = None) -> bool:
    """Restore this rank's RNG streams, or skip a missing/topology-mismatched state."""
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.utils.rank_state import select_rank_state

    if isinstance(states, dict) and any(not isinstance(state, dict) for state in states.values()):
        raise ValueError("RNG checkpoint states must be dictionaries")
    mine = select_rank_state(states, on_world_size_mismatch="skip")
    if mine is None:
        return False
    random.setstate(mine["python"])
    np.random.set_state(mine["numpy"])
    torch.set_rng_state(torch.as_tensor(mine["torch"], dtype=torch.uint8))  # [N]
    if "cuda" in mine:
        cuda_state = torch.as_tensor(mine["cuda"], dtype=torch.uint8)  # [N]
        if cuda_device is not None:
            device = torch.device(cuda_device)
            if device.type != "cuda":
                raise ValueError(f"CUDA RNG restore requires a CUDA device, got {device}")
            with torch.cuda.device(device):
                torch.cuda.set_rng_state(cuda_state, device)
        elif torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state)
    return True
