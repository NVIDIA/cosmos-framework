# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit native aliases that are not moved through the registered-buffer API."""

from __future__ import annotations


def bind_rotary_aliases(net):
    """Keep both Qwen rotary reset aliases attached to their materialized buffer.

    Transformers 4.57.6 stores original_inv_freq as a plain tensor attribute.
    Parameter/buffer streaming cannot move that alias implicitly. The pinned
    default RoPE forward does not read it; preserving the alias also makes the
    native frequency-reset state coherent after construction and dense reload.
    """
    import torch

    for name in ("vlm.model.language_model.rotary_emb", "expert.rotary_emb"):
        rotary = net.get_submodule(name)
        if not isinstance(getattr(rotary, "inv_freq", None), torch.Tensor) or not hasattr(rotary, "original_inv_freq"):
            raise ValueError(f"Alpamayo pinned rotary contract differs: {name}")
        if rotary.inv_freq.is_meta:
            raise ValueError(f"Alpamayo rotary buffer was not materialized: {name}")
        rotary.original_inv_freq = rotary.inv_freq
