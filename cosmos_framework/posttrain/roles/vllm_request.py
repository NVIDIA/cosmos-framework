# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded multimodal identity for collective-bearing vLLM requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cosmos_framework.posttrain.utils.identity import normalize_json_dict

_CONTENT_IDENTITY_REQUIRED_MODALITIES = frozenset({"image"})


def vllm_request_media_identity(
    request: Mapping[str, Any],
    *,
    context: str,
    require_all_content_uuids: bool = False,
) -> dict[str, Any]:
    """Project one vLLM request's ordered media semantics into finite JSON.

    Qwen image requests carry packer-computed content UUIDs, so an image without an ordered UUID is
    rejected before TP ranks enter the engine. Existing video requests do not yet expose content
    UUIDs. A TP1 caller may keep that legacy path cardinality-only; collective-bearing callers set
    ``require_all_content_uuids`` so every modality fails before vLLM unless it has ordered UUIDs.
    """
    media = request.get("multi_modal_data", {})
    if media is None:
        media = {}
    if not isinstance(media, Mapping):
        raise TypeError(f"{context} multi_modal_data must be a mapping")

    cardinality: dict[str, int] = {}
    for modality, items in media.items():
        if not isinstance(modality, str) or not modality:
            raise TypeError(f"{context} multi_modal_data keys must be non-empty strings")
        cardinality[modality] = len(items) if isinstance(items, (list, tuple)) else 1

    raw_uuids = request.get("multi_modal_uuids", {})
    if raw_uuids is None:
        raw_uuids = {}
    if not isinstance(raw_uuids, Mapping):
        raise TypeError(f"{context} multi_modal_uuids must be a mapping")
    if any(not isinstance(modality, str) or not modality for modality in raw_uuids):
        raise TypeError(f"{context} multi_modal_uuids keys must be non-empty strings")
    extra_modalities = set(raw_uuids) - set(cardinality)
    if extra_modalities:
        raise ValueError(f"{context} multi_modal_uuids has no matching media for {sorted(extra_modalities)!r}")

    ordered_uuids: dict[str, list[str]] = {}
    cardinality_only: list[str] = []
    for modality, count in cardinality.items():
        if modality not in raw_uuids:
            if require_all_content_uuids or modality in _CONTENT_IDENTITY_REQUIRED_MODALITIES:
                raise ValueError(f"{context} {modality} media requires ordered content UUIDs")
            cardinality_only.append(modality)
            continue
        values = raw_uuids[modality]
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"{context} multi_modal_uuids[{modality!r}] must be an ordered sequence")
        if len(values) != count:
            raise ValueError(
                f"{context} {modality} media/UUID cardinality mismatch: media={count}, UUIDs={len(values)}"
            )
        if any(not isinstance(value, str) or not value for value in values):
            raise TypeError(f"{context} multi_modal_uuids[{modality!r}] must contain non-empty strings")
        ordered_uuids[modality] = list(values)

    processor_kwargs = request.get("mm_processor_kwargs", {})
    if processor_kwargs is None:
        processor_kwargs = {}
    if not isinstance(processor_kwargs, Mapping):
        raise TypeError(f"{context} mm_processor_kwargs must be a mapping")
    return normalize_json_dict(
        {
            "media_cardinality": cardinality,
            "multi_modal_uuids": ordered_uuids,
            "cardinality_only_modalities": sorted(cardinality_only),
            "mm_processor_kwargs": dict(processor_kwargs),
        },
        context=f"{context} media identity",
    )
