# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared policy-forward contract for objectives that score a collated batch.

VLM adapters own modality-specific inputs and device movement behind
``forward_policy_batch``. Decoder-only models retain the legacy ``net(input_ids)``
fallback, but that fallback must never receive side inputs it would silently drop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from cosmos_framework.posttrain.model.protocol import SelectedSFTPolicy, SelectedSFTPolicyOutput


def selected_sft_forward(
    model: Any,
) -> Callable[[dict[str, Any], torch.Tensor], SelectedSFTPolicyOutput] | None:
    """Resolve the opt-in capability during the rank-converged input preflight."""
    mode = getattr(model, "sft_logits_mode", "full")
    if mode == "full":
        return None
    if mode != "supervised":
        raise ValueError(f"unsupported SFT logits mode {mode!r}; expected 'full' or 'supervised'")
    if not isinstance(model, SelectedSFTPolicy):
        raise TypeError("sft_logits_mode='supervised' requires the SelectedSFTPolicy capability")
    return model.forward_sft_batch


def selected_sft_logits(output: Any, expected_tokens: int) -> torch.Tensor:
    """Validate an SFT-only projection without weakening the full-logit contract."""
    if not isinstance(output, SelectedSFTPolicyOutput) or not isinstance(output.logits, torch.Tensor):
        raise TypeError("selected SFT forward must return SelectedSFTPolicyOutput with tensor logits")
    logits = output.logits
    if logits.ndim != 2 or logits.shape[0] != expected_tokens or logits.shape[1] < 1:
        raise ValueError(
            f"selected SFT forward must return logits [N,V] for {expected_tokens} tokens, got {tuple(logits.shape)}"
        )
    return logits


def forward_policy(model: Any, net: Any, batch: dict[str, Any]) -> Any:
    """Run a model-ready batch through a VLM capability or the text fallback."""
    capability = getattr(model, "forward_policy_batch", None)
    if callable(capability):
        return capability(batch)
    return net(batch["input_ids"])  # .logits [B,L,V]


def prepare_policy_capability(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Validate and prepare a complete batch before any rank enters model forward."""
    capability = getattr(model, "forward_policy_batch", None)
    model_inputs = set(batch) - {"input_ids", "logprob_masks"}
    if not callable(capability) and model_inputs:
        raise TypeError(
            "the policy forward received model-specific inputs without a VLMPolicy capability: "
            f"{sorted(model_inputs)}. Use a model that implements `forward_policy_batch`; the "
            "decoder-only fallback would silently drop multimodal conditioning and train on text alone."
        )
    validator = getattr(model, "validate_policy_batch", None)
    if callable(validator):
        validator(batch)
    preparer = getattr(model, "prepare_policy_batch", None)
    prepared = preparer(batch) if callable(preparer) else batch
    if not isinstance(prepared, dict):
        raise TypeError(f"prepare_policy_batch must return dict[str, Any], got {type(prepared).__name__}")
    return prepared


def policy_logits(
    output: Any,
    input_ids: torch.Tensor | None = None,  # [B,L] or None
    *,
    owner: str = "policy",
) -> torch.Tensor:  # returns [B,L,V]
    """Extract policy logits and optionally validate their batch/token axes."""
    logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)  # [B,L,V]
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{owner} forward returned no tensor logits: {type(output).__name__}")
    if input_ids is not None and (logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(input_ids.shape)):
        raise ValueError(
            f"{owner} forward must return logits [B,L,V] aligned with input_ids, got "
            f"{tuple(logits.shape)} for input_ids {tuple(input_ids.shape)}"
        )
    return logits


def validate_collated_policy_batch(
    batch: dict[str, Any],
    expected_batch: int,
    *,
    owner: str,
) -> tuple[torch.Tensor, torch.Tensor]:  # returns input_ids [B,L], masks [B,L]
    """Validate the common tensor contract emitted by policy packers."""
    input_ids = batch.get("input_ids")  # [B,L] or None
    masks = batch.get("logprob_masks")  # [B,L] or None
    if not isinstance(input_ids, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError(
            f"{owner} policy collation must return tensor input_ids/logprob_masks, got "
            f"{type(input_ids).__name__} and {type(masks).__name__}"
        )
    if input_ids.ndim != 2 or masks.ndim != 2:
        raise ValueError(
            f"{owner} policy collation must return rank-2 input_ids/masks, got "
            f"{tuple(input_ids.shape)} and {tuple(masks.shape)}"
        )
    if tuple(input_ids.shape) != tuple(masks.shape):
        raise ValueError(
            f"{owner} policy collation returned mismatched input_ids/masks shapes "
            f"{tuple(input_ids.shape)} and {tuple(masks.shape)}"
        )
    if input_ids.dtype != torch.long or masks.dtype != torch.bool:
        raise ValueError(
            f"{owner} policy collation must return long input_ids and bool masks, got "
            f"{input_ids.dtype} and {masks.dtype}"
        )
    if input_ids.shape[0] != expected_batch:
        raise ValueError(
            f"{owner} policy collation returned batch size {input_ids.shape[0]} for "
            f"{expected_batch} prepared sequence(s)"
        )
    return input_ids, masks
