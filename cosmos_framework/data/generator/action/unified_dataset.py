# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.data.generator.action.utils.action_processing import ActionNormalizer
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    ActionSchema,
    UnifiedActionCodec,
    get_dataset_action_mapping,
)
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY, DROP_SAMPLE_REASON_KEY


def prepare_action_sample_for_schema(
    sample: dict[str, Any],
    *,
    action_schema: ActionSchema,
    action_normalizer: ActionNormalizer | None,
    dataset: object | None,
    unified_action_normalizer: ActionNormalizer | None = None,
) -> tuple[
    dict[str, Any],
    ActionNormalizer | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    UnifiedActionCodec | None,
]:
    """Apply the shared native-to-model action boundary for one sample."""
    normalizer_schema = getattr(action_normalizer, "action_schema", "native")
    if unified_action_normalizer is not None and action_normalizer is not None:
        raise ValueError("Provide either action_normalizer or unified_action_normalizer, not both")
    if action_schema == "native":
        if unified_action_normalizer is not None or normalizer_schema != "native":
            raise ValueError("Unified action normalization requires action_schema='unified_v1'")
        return sample, action_normalizer, None, None, None, None
    if action_schema != "unified_v1":
        raise ValueError(f"Unsupported action_schema={action_schema!r}")
    if dataset is None:
        raise ValueError("dataset is required when action_schema='unified_v1'")
    if normalizer_schema == "unified_v1":
        unified_action_normalizer = action_normalizer
        action_normalizer = None
    elif normalizer_schema != "native":
        raise ValueError(f"Unsupported normalizer action_schema={normalizer_schema!r}")

    action = sample.get("action")
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"Expected tensor sample['action'], got {type(action).__name__}")
    history_action = sample.get("history_action")
    if history_action is not None and not isinstance(history_action, torch.Tensor):
        raise TypeError(f"Expected tensor history_action, got {type(history_action).__name__}")

    nonfinite_fields: list[str] = []
    for field_name, field_value in (("action", action), ("history_action", history_action)):
        if field_value is not None and not bool(torch.isfinite(field_value).all()):
            nonfinite_fields.append(field_name)
            sample = dict(sample)
            sample[field_name] = torch.nan_to_num(field_value, nan=0.0, posinf=0.0, neginf=0.0)
    if nonfinite_fields:
        sample[DROP_SAMPLE_KEY] = True
        if not isinstance(sample.get(DROP_SAMPLE_REASON_KEY), str) or not sample[DROP_SAMPLE_REASON_KEY]:
            sample[DROP_SAMPLE_REASON_KEY] = "non-finite " + " and ".join(nonfinite_fields)

    mapping = get_dataset_action_mapping(dataset)
    action_codec = UnifiedActionCodec(
        native_normalizer=None if unified_action_normalizer is not None else action_normalizer,
        mapping=mapping,
        unified_normalizer=unified_action_normalizer,
    )
    encoded_action = action_codec.encode(sample["action"])
    sample = dict(sample)
    sample["action"] = encoded_action.raw
    history_normalized_action = None
    if history_action is not None:
        encoded_history = action_codec.encode(sample["history_action"])
        sample["history_action"] = encoded_history.raw
        history_normalized_action = encoded_history.normalized
    return (
        sample,
        None,
        encoded_action.valid_mask,
        encoded_action.normalized,
        history_normalized_action,
        action_codec,
    )
