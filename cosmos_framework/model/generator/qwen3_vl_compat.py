# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Repository-owned Qwen3-VL training compatibility helpers."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _linear_patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    patch_elements = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    flat_input = hidden_states.reshape(-1, patch_elements).to(dtype=self.proj.weight.dtype)
    flat_weight = self.proj.weight.reshape(self.embed_dim, patch_elements)
    return F.linear(flat_input, flat_weight, self.proj.bias)


_linear_patch_embed_forward._tao_channels_last_3d = True  # type: ignore[attr-defined]


def should_use_linear_patch_embed(mode: str, *, device_capability: tuple[int, int] | None = None) -> bool:
    """Resolve ``auto|linear|conv3d`` without consulting environment shims."""
    if mode not in {"auto", "linear", "conv3d"}:
        raise ValueError(f"Unsupported Qwen3-VL patch-embed mode: {mode!r}")
    if mode == "linear":
        return True
    if mode == "conv3d":
        return False
    if device_capability is None and torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
    return device_capability == (8, 0)


def apply_qwen3_vl_patch_embed_compat(
    model: torch.nn.Module,
    *,
    model_type: str,
    mode: str,
    device_capability: tuple[int, int] | None = None,
) -> bool:
    """Use the A100-safe algebraic projection for Qwen3-VL patch embeds.

    Returns whether a replacement was applied. The original Conv3d parameters
    and state-dict keys are unchanged, so checkpoint compatibility is exact.
    """
    if model_type not in {"qwen3_vl", "qwen3_vl_moe"}:
        return False
    if not should_use_linear_patch_embed(mode, device_capability=device_capability):
        return False

    patched = 0
    for module in model.modules():
        required = ("in_channels", "temporal_patch_size", "patch_size", "embed_dim", "proj")
        if all(hasattr(module, name) for name in required) and isinstance(module.proj, torch.nn.Conv3d):
            module.forward = types.MethodType(_linear_patch_embed_forward, module)
            patched += 1
    if patched != 1:
        raise RuntimeError(f"Expected one Qwen3-VL PatchEmbed module, found {patched}")
    return True
