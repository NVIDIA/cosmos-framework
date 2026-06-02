# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for the model-agnostic CPU-offload core.

Covers ``ModuleOffloadManager`` / ``OffloadPipeline``: packing disjoint module
groups into pinned CPU storage, time-sharing a single GPU arena via ``stage`` /
``context``, the numerical transparency of offloading (staged forward ==
non-offloaded forward), parameter device placement across stages, and the
error/guard paths.

The Cosmos3-specific wiring (``build_omni_offload_parts`` + the reasoner/
generator split) is validated end-to-end against the real checkpoint; here we
pin down the reusable mechanics with a tiny, deterministic model.
"""

import pytest
import torch
import torch.nn as nn

from cosmos_framework.inference.offloading import ModuleOffloadManager, OffloadPipeline


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TwoTower(nn.Module):
    """Two disjoint single-layer towers exercised one at a time."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.tower_a = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.tower_b = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def run_a(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower_a(x)

    def run_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower_b(x)


def _build(device: torch.device) -> tuple[_TwoTower, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = _TwoTower().to(device=device, dtype=torch.float32).eval()
    x = torch.randn(4, 16, device=device, dtype=torch.float32)
    with torch.no_grad():
        ref = model.run_b(model.run_a(x))  # both towers resident on the device
    return model, x, ref


def test_offload_pipeline_is_numerically_transparent():
    """Staging each group and running it must reproduce the non-offloaded forward."""
    device = _device()
    model, x, ref = _build(device)

    offloader = OffloadPipeline(
        stages=("tower_a", "tower_b"),
        parts={"tower_a": model.tower_a, "tower_b": model.tower_b},
        device=device,
        pin_memory=(device.type == "cuda"),
    )
    offloader.initialize()

    with torch.no_grad():
        offloader.context("tower_a")
        out_a = model.run_a(x)
        offloader.context("tower_b")
        out_b = model.run_b(out_a)
    offloader.cleanup()

    torch.testing.assert_close(out_b, ref, atol=0.0, rtol=0.0)


def test_stage_swaps_parameter_device_placement():
    """Only the active group's tensors live on the arena; the rest are CPU-backed."""
    device = _device()
    model, _, _ = _build(device)

    offloader = OffloadPipeline(
        stages=("tower_a", "tower_b"),
        parts={"tower_a": model.tower_a, "tower_b": model.tower_b},
        device=device,
        pin_memory=(device.type == "cuda"),
    )
    offloader.initialize()

    # After initialize (nothing staged) both groups are rebound to CPU storage.
    assert model.tower_a[0].weight.device.type == "cpu"
    assert model.tower_b[0].weight.device.type == "cpu"

    offloader.context("tower_a")
    assert model.tower_a[0].weight.device.type == device.type
    assert model.tower_b[0].weight.device.type == "cpu"

    offloader.context("tower_b")
    assert model.tower_a[0].weight.device.type == "cpu"
    assert model.tower_b[0].weight.device.type == device.type

    # Re-staging the already-active group is a no-op.
    offloader.context("tower_b")
    assert model.tower_b[0].weight.device.type == device.type

    offloader.cleanup()
    assert model.tower_b[0].weight.device.type == "cpu"


def test_arena_is_shared_across_groups():
    """The GPU arena is reused: staged tensors of either group alias one buffer."""
    device = _device()
    model, _, _ = _build(device)
    offloader = OffloadPipeline(
        stages=("tower_a", "tower_b"),
        parts={"tower_a": model.tower_a, "tower_b": model.tower_b},
        device=device,
        pin_memory=(device.type == "cuda"),
    )
    offloader.initialize()

    manager = offloader.manager
    assert manager.gpu_arena is not None
    arena_ptr = manager.gpu_arena.untyped_storage().data_ptr()

    def _in_arena(t: torch.Tensor) -> bool:
        return t.untyped_storage().data_ptr() == arena_ptr

    offloader.context("tower_a")
    assert _in_arena(model.tower_a[0].weight)
    offloader.context("tower_b")
    assert _in_arena(model.tower_b[0].weight)
    offloader.cleanup()


def test_heterogeneous_group_sizes_share_one_arena():
    """A small (vae-like) group co-exists with large (tower-like) groups in one arena.

    Mirrors the Cosmos3 stage mix: the arena is sized to the largest group and the
    smaller group stages into the same buffer (only its own bytes are copied), with
    forward output unchanged.
    """
    device = _device()
    torch.manual_seed(0)
    big_a = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64)).to(device).eval()
    big_b = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64)).to(device).eval()
    small = nn.Linear(64, 64).to(device).eval()  # vae-like: much smaller
    x = torch.randn(2, 64, device=device)
    with torch.no_grad():
        ref_a, ref_b, ref_small = big_a(x), big_b(x), small(x)

    offloader = OffloadPipeline(
        stages=("big_a", "big_b", "small"),
        parts={"big_a": big_a, "big_b": big_b, "small": small},
        device=device,
        pin_memory=(device.type == "cuda"),
    )
    offloader.initialize()

    # Arena is sized to the largest group, not the sum.
    sizes = {name: layout.nbytes for name, layout in offloader.manager.layouts.items()}
    assert offloader.manager.gpu_arena.numel() == max(sizes.values())
    assert sizes["small"] < sizes["big_a"]

    with torch.no_grad():
        offloader.context("big_a")
        out_a = big_a(x)
        offloader.context("small")
        out_small = small(x)
        offloader.context("big_b")
        out_b = big_b(x)
    offloader.cleanup()

    torch.testing.assert_close(out_a, ref_a, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_b, ref_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_small, ref_small, atol=0.0, rtol=0.0)


def test_has_part_and_tolerant_staging():
    """``has_part`` + guarded ``context`` is the tolerant-staging idiom the inference
    layer uses so a uniform ``stage(part)`` call no-ops for parts that aren't offloaded."""
    device = _device()
    model, _, _ = _build(device)
    offloader = OffloadPipeline(
        stages=("tower_a",),  # only one part is offloaded
        parts={"tower_a": model.tower_a},
        device=device,
        pin_memory=(device.type == "cuda"),
    )
    offloader.initialize()

    assert offloader.has_part("tower_a")
    assert not offloader.has_part("tower_b")
    assert not offloader.has_part("vae")

    def stage(part: str):
        # The exact wrapper installed by OmniInference._create.
        return offloader.context(part) if offloader.has_part(part) else None

    stage("vae")  # absent part -> silent no-op, nothing staged
    assert offloader.manager.active_group_name is None
    stage("tower_a")  # present part -> staged
    assert offloader.manager.active_group_name == "tower_a"
    offloader.cleanup()


def test_unknown_part_raises():
    device = _device()
    model, _, _ = _build(device)
    offloader = OffloadPipeline(
        stages=("tower_a", "tower_b"),
        parts={"tower_a": model.tower_a, "tower_b": model.tower_b},
        device=device,
        pin_memory=(device.type == "cuda"),
    )
    offloader.initialize()
    with pytest.raises(KeyError):
        offloader.context("does_not_exist")
    offloader.cleanup()


def test_double_initialize_raises():
    device = _device()
    model, _, _ = _build(device)
    manager = ModuleOffloadManager(
        groups={"tower_a": model.tower_a, "tower_b": model.tower_b},
        device=device,
        pin_memory=(device.type == "cuda"),
    )
    manager.initialize()
    with pytest.raises(RuntimeError):
        manager.initialize()


def test_empty_groups_rejected():
    with pytest.raises(ValueError):
        ModuleOffloadManager(groups={}, device=_device())


def test_param_free_group_rejected():
    """A group with no parameters or buffers cannot be packed."""
    device = _device()
    manager = ModuleOffloadManager(
        groups={"empty": nn.ReLU(), "real": nn.Linear(4, 4)},
        device=device,
        pin_memory=False,
    )
    with pytest.raises(ValueError):
        manager.initialize()
