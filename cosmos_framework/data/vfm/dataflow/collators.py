# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in BatchCollator implementations."""

from __future__ import annotations

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from cosmos_framework.data.vfm.dataflow.base import BatchCollator


class DefaultBatchCollator(BatchCollator):
    """Stacks samples with torch's default_collate — stock DataLoader behavior."""

    def collate(self, samples: list[dict]) -> dict:
        return torch.utils.data.default_collate(samples)


# ---------------------------------------------------------------------------
# VFMListCollator — verbatim port of custom_collate_fn from joint_dataloader.py
# ---------------------------------------------------------------------------

_TIMING_KEYS = {"_sample_time", "_aug_time", "_pre_aug_time", "_aug_step_times"}
_BATCH_TIMING_KEYS = {
    "_worker_batch_time",
    "_worker_aug_time",
    "_worker_io_time",
    "_worker_aug_step_times",
    "_worker_id",
}


def _vfm_collate(batch):
    """
    Collate function that works like default_collate for all keys other than "text_token_ids", "images", and "video".
    For "text_token_ids", "images", and "video" it simply returns them in a list, instead of stacking them as a tensor.
    """
    list_collate_keys = {
        "text_token_ids",
        "images",
        "video",
        "action",
        "domain_id",
        "sequence_plan",
        "sound",
        "raw_action_dim",
        "image_size",
    }

    # Data keys where a per-sample value of ``None`` is a meaningful signal
    # (e.g. audio extraction failed for that sample → ``sound=None`` paired
    # with ``plan.has_sound=False``).  These keys must be kept as a list with
    # ``None`` placeholders so the model can align per-sample data 1:1 with
    # per-sample plans.  Dropping the entire key on any None would leave the
    # remaining sound tensors mis-aligned with the plans whose ``has_sound``
    # flag was set BEFORE collation, causing ``sequence_packing`` to index
    # past the end of ``x0_tokens_sound``.
    sparse_data_keys = {"sound"}

    # Handle the case where the batch is already a dictionary (e.g. column-wise batching)
    if isinstance(batch, dict):
        return {key: (value if key in list_collate_keys else default_collate(value)) for key, value in batch.items()}

    # Handle standard list of samples
    elem = batch[0]
    if isinstance(elem, dict):

        # Some Action datasets add optional metadata keys (for example
        # ``additional_view_description`` for concat-view captions) only for a
        # subset of samples.  PyTorch can batch such samples together when
        # DataLoader batch_size > 1; collating only elem's keys and indexing
        # every sample by that key turns the optional field into a fatal
        # KeyError.  Use the union of keys and skip optional keys that are not
        # present in every sample.  Required training keys still fail loudly via
        # downstream assertions if actually missing.
        result = {}
        keys = set().union(*(d.keys() for d in batch))
        for key in keys:
            if key in _TIMING_KEYS:
                continue
            values = [d.get(key) for d in batch]
            if any(value is None for value in values):
                # Sparse data keys keep their None placeholders to preserve
                # 1:1 alignment with sequence_plan.  Other (optional metadata)
                # keys not present in every sample are dropped.
                if key in sparse_data_keys:
                    result[key] = values
                continue
            if key in list_collate_keys:
                result[key] = values
            else:
                result[key] = default_collate(values)
        result.update(_aggregate_worker_timing(batch))
        return result
    else:
        return default_collate(batch)


def _aggregate_worker_timing(samples: list[dict]) -> dict:
    """Extract per-sample timing keys, aggregate into per-batch scalars."""
    info: dict[str, float | int] = {}
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


class VFMListCollator(BatchCollator):
    """custom_collate_fn as a BatchCollator: media kept as lists, sparse `sound`
    None placeholders preserved 1:1 with sequence_plan, optional keys dropped,
    per-worker timing aggregated. Behavior-identical to the legacy collate_fn."""

    def collate(self, samples: list[dict]) -> dict:
        return _vfm_collate(samples)
