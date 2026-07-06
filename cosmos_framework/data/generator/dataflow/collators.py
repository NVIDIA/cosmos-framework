# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in BatchCollator implementations."""

from __future__ import annotations

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from cosmos_framework.data.generator.dataflow.base import BatchCollator


class DefaultBatchCollator(BatchCollator):
    """Stacks samples with torch's default_collate — stock DataLoader behavior."""

    def collate(self, samples: list[dict]) -> dict:
        return torch.utils.data.default_collate(samples)


# ---------------------------------------------------------------------------
# VFMListCollator — reproduces the legacy PackingDataLoader packed-batch structure:
#   for each raw sample, inner-collate at batch_size=1 (custom_collate_fn copy),
#   split per _get_next_sample i=0 rules, then _update_output_batch-accumulate.
#
# This produces list[list[Tensor]] for _MULTI_ITEM_KEYS (not flat list[Tensor]).
# ---------------------------------------------------------------------------

_TIMING_KEYS = {"_sample_time", "_aug_time", "_pre_aug_time", "_aug_step_times"}
_BATCH_TIMING_KEYS = {
    "_worker_batch_time",
    "_worker_aug_time",
    "_worker_io_time",
    "_worker_aug_step_times",
    "_worker_id",
}

# Verbatim copy of JointDataLoader._MULTI_ITEM_KEYS
_MULTI_ITEM_KEYS = {"text_token_ids", "images", "video", "action", "sound"}

# Verbatim copy of JointDataLoader._FLATTEN_LIST_KEYS
_FLATTEN_LIST_KEYS = {"image_size"}


def _vfm_inner_collate(batch):
    """
    Verbatim copy of custom_collate_fn from joint_dataloader.py.

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


def _split_one(batch: dict) -> dict:
    """Port of _get_next_sample split rules for i=0 (verbatim from joint_dataloader.py lines 470-490).

    Splitting rules:
        - _BATCH_TIMING_KEYS: passed through as-is.
        - _MULTI_ITEM_KEYS with list value: elem = v[0]; if elem is a list → sample[k]=elem,
          else → sample[k]=v[0:1] (single-element list wrapping the tensor).
        - Other list values: sample[k] = v[0] (bare element, direct-indexed).
        - Non-list (tensor/scalar) values: sample[k] = v[0:1] (preserve batch dim).
    """
    sample = {}
    for k, v in batch.items():
        if k in _BATCH_TIMING_KEYS:
            sample[k] = v
        elif isinstance(v, list) and k in _MULTI_ITEM_KEYS:
            elem = v[0]
            sample[k] = elem if isinstance(elem, list) else v[0:1]
        elif isinstance(v, list):
            sample[k] = v[0]
        else:
            sample[k] = v[0:1]
    return sample


def _accumulate(output_batch: dict, output: dict) -> None:
    """Port of _update_output_batch from joint_dataloader.py lines 405-418."""
    for key, value in output.items():
        if key in _BATCH_TIMING_KEYS:
            if key not in output_batch:
                output_batch[key] = value
        elif key in _FLATTEN_LIST_KEYS and isinstance(value, list):
            if key not in output_batch:
                output_batch[key] = value
            else:
                output_batch[key].extend(value)
        elif key not in output_batch:
            output_batch[key] = [value]
        else:
            output_batch[key].append(value)


# Keep _vfm_collate as an alias for the inner collate (used by legacy callers).
_vfm_collate = _vfm_inner_collate


class VFMListCollator(BatchCollator):
    """Reproduces the legacy PackingDataLoader packed-batch structure.

    For a group of N raw SFTDataset samples, the packed output has:
      - ``_MULTI_ITEM_KEYS`` (``text_token_ids``, ``video``, ``images``,
        ``action``, ``sound``): ``list[list[Tensor]]`` — each inner list
        is a single-element list ``[tensor]``, matching the
        ``v[i:i+1]`` slice from ``_get_next_sample``.
      - Metadata list keys (``sequence_plan``, ``domain_id``,
        ``raw_action_dim``): flat ``list[element]``.
      - ``image_size`` (``_FLATTEN_LIST_KEYS``): flat ``list[Tensor]``
        (extended, not appended).
      - Non-list tensor keys: ``list[Tensor(1,...)]``.
      - ``_BATCH_TIMING_KEYS``: set once from the first sample.

    Implementation: for each sample, inner-collate at batch_size=1 via
    ``_vfm_inner_collate`` (verbatim ``custom_collate_fn``), split
    sample 0 per ``_split_one`` (verbatim ``_get_next_sample`` i=0
    rules), then accumulate via ``_accumulate`` (verbatim
    ``_update_output_batch``).  Byte-identical to the legacy packer.
    """

    def collate(self, samples: list[dict]) -> dict:
        # Reproduce the legacy PackingDataLoader packed batch: for each sample,
        # inner-collate at batch_size=1, split sample 0 per _MULTI_ITEM_KEYS /
        # list / tensor rules, then _update_output_batch-accumulate across the group.
        output_batch: dict = {}
        for s in samples:
            collated = _vfm_inner_collate([s])      # verbatim custom_collate_fn copy
            split = _split_one(collated)            # i=0 split (rules from _get_next_sample)
            _accumulate(output_batch, split)        # _update_output_batch copy
        return output_batch
