# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Storage-independent per-worker batch timing aggregation."""

from typing import Any

import torch.utils.data

_TIMING_KEYS = {"_sample_time", "_aug_time", "_pre_aug_time", "_aug_step_times"}


def _aggregate_worker_timing(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract per-sample timing keys, aggregate into per-batch scalars."""
    info: dict[str, Any] = {}
    if "_sample_time" in samples[0]:
        info["_worker_batch_time"] = sum(s.get("_sample_time", 0.0) for s in samples)
    if "_aug_time" in samples[0]:
        aug_total = sum(s.get("_aug_time", 0.0) for s in samples)
        info["_worker_aug_time"] = aug_total
        if "_worker_batch_time" in info:
            info["_worker_io_time"] = info["_worker_batch_time"] - aug_total
    if "_aug_step_times" in samples[0]:
        agg: dict[str, float] = {}
        for s in samples:
            for step_name, t in s.get("_aug_step_times", {}).items():
                agg[step_name] = agg.get(step_name, 0.0) + t
        info["_worker_aug_step_times"] = agg
    worker_info = torch.utils.data.get_worker_info()
    info["_worker_id"] = worker_info.id if worker_info is not None else 0
    return info
