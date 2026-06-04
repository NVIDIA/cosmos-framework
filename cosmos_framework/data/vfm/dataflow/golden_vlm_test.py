# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Golden-batch equality: legacy DataPackerDataLoader+VLMDataPacker vs the new
four-role dataflow loader on the same fixed source must yield identical batches."""

from __future__ import annotations

import random

import torch

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.configs.base.vlm.experiment.llava_ov_datapacker_experiment import VLMDataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader as LegacyLoader
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader as NewLoader,
    IterableDistributor,
    PoolPackingBatcher,
)


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        n = 4 + sum(len(m["content"]) for m in messages if isinstance(m["content"], str))
        return {"input_ids": torch.arange(n)}

    def add_assistant_tokens_mask(self, input_ids):
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[len(input_ids) // 2 :] = True
        return mask


class _FixedIterable(torch.utils.data.IterableDataset):
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        yield from self._items


def _items(k):
    out = []
    for i in range(k):
        out.append({
            "image": None,
            "conversations": [
                {"from": "human", "value": "q" * (i % 5 + 1)},
                {"from": "gpt", "value": "a" * (i % 3 + 1)},
            ],
        })
    return out


def _drain(loader, n):
    it = iter(loader)
    return [next(it) for _ in range(n)]


def test_vlm_golden_batches_match():
    proc = _FakeProcessor()
    items = _items(40)

    random.seed(0)
    legacy = LegacyLoader(
        data_source=_FixedIterable(list(items)),
        data_packer=VLMDataPacker(tokenizer_config=proc, max_seq_len=200),
        max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400,
        num_workers=0,
    )
    random.seed(0)
    new = NewLoader(
        distributor=IterableDistributor(list(items)),
        processor=VLMProcessor(processor=proc),
        batcher=PoolPackingBatcher(max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400),
        collator=VLMCollator(),
        num_workers=0,
    )

    a = _drain(legacy, 10)
    b = _drain(new, 10)
    for ba, bb in zip(a, b):
        assert ba.keys() == bb.keys(), (ba.keys(), bb.keys())
        for k in ba:
            assert torch.equal(ba[k], bb[k]), f"mismatch at key {k}"
