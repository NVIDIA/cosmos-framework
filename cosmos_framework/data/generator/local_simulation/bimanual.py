# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bimanual production transforms on already-resolved, local bundle windows."""

from __future__ import annotations

import math
import random
from typing import Any

from cosmos_framework.data.generator.action.canonical_lance_action_sampler import CanonicalMixedDomainActionTransform
from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    BaseLanceActionDataset,
    LanceActionSampleRequest,
)
from cosmos_framework.data.generator.action.unified_dataset import prepare_action_sample_for_schema
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionNormalizer,
    load_action_normalization_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    LEFT_OPENNESS,
    RIGHT_OPENNESS,
    UNIFIED_ACTION_DIM,
    get_dataset_action_mapping,
)


class LocalBimanualPipeline:
    """Keep FK, metric grippers, mixed-domain masks, and tokenization unchanged.

    Index/rank selection belongs to export and LocalSimulationDataLoader, not
    to this materializer. Captured source URIs are identities only.
    """

    def __init__(self, dataset: BaseLanceActionDataset, config: dict[str, Any]) -> None:
        self.dataset = dataset
        config = dict(config)
        self.action_schema = config.pop("action_schema")
        self.global_asinh = config.pop("global_asinh")
        self.metric_gripper = config.pop("metric_gripper")
        if self.action_schema != "unified_v1" or not self.global_asinh or not self.metric_gripper:
            raise ValueError("The Bimanual recipe requires unified_v1 global-asinh metric-gripper normalization.")
        self.stats = load_action_normalization_stats(
            "global_asinh", stats_path=config.pop("global_asinh_stats_path"), expected_dim=UNIFIED_ACTION_DIM
        )
        # These policies have already selected the captured rows and requests.
        for name in ("resolution", "dense_caption_probability", "row_index_cache_tag", "slow_sample_log_threshold_s"):
            config.pop(name, None)
        mapping = get_dataset_action_mapping(dataset)
        mask = mapping.valid_mask()
        self.has_grippers = bool(mask[RIGHT_OPENNESS].item() or mask[LEFT_OPENNESS].item())
        self.normalizers: dict[float | None, ActionNormalizer] = {}
        self.transform = CanonicalMixedDomainActionTransform(ActionTransformPipeline(**config))

    def _normalizer(self, sample: dict[str, Any]) -> ActionNormalizer:
        width = self.dataset.get_fully_open_gripper_width_m(sample) if self.has_grippers else None
        if self.has_grippers and (width is None or not math.isfinite(width) or width <= 0):
            raise ValueError("Metric-gripper normalization requires the audited hardware's positive opening width.")
        if width not in self.normalizers:
            stats = self.stats
            if width is not None:
                q01, q99 = stats["q01"].clone(), stats["q99"].clone()
                slots = [RIGHT_OPENNESS, LEFT_OPENNESS]
                q01[slots] /= width
                q99[slots] /= width
                stats = {**stats, "q01": q01, "q99": q99}
            self.normalizers[width] = resolve_action_normalization(
                "global_asinh", stats, pose_convention=getattr(self.dataset, "pose_convention", None)
            )
        return self.normalizers[width]

    def transform_resolved_sample(self, sample: dict[str, Any], request: LanceActionSampleRequest) -> dict[str, Any]:
        sample, normalizer, valid_mask, normalized, history_normalized, codec = prepare_action_sample_for_schema(
            sample,
            action_schema=self.action_schema,
            action_normalizer=None,
            dataset=self.dataset,
            unified_action_normalizer=self._normalizer(sample),
        )
        state = random.getstate()
        random.seed(request.aux_seed ^ 0xD1B54A32D192ED03)
        try:
            return self.transform(
                sample,
                resolution=request.resolution,
                action_normalizer=normalizer,
                action_valid_mask=valid_mask,
                normalized_action=normalized,
                history_normalized_action=history_normalized,
                action_codec=codec,
            )
        finally:
            random.setstate(state)
