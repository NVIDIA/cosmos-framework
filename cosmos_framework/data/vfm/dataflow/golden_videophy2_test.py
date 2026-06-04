# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Golden-batch equality: legacy DataPackerDataLoader+VideoPhy2DataPacker vs the new
four-role dataflow loader on the same fixed source must yield identical batches.

SINGLETON-scope by design: this test is intentionally constrained to
``max_batch_size=1`` (one sample per batch).  Both the VideoPhy2 recipe and the
legacy ``VideoPhy2DataPacker.sft_collate_fn`` hard-assert ``len(samples) == 1``,
so a multi-sample legacy-vs-new parity test is not constructible without forking
the legacy code.

Multi-sample ``PoolPackingBatcher`` packing behaviour is covered independently
by ``batchers_test.py::test_pool_packs_multiple_within_budget`` and related
pool-batcher tests.
"""

from __future__ import annotations

import random

import torch
import torch.utils.data

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator
from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import VideoPhy2Processor
from cosmos_framework.configs.base.vlm.experiment.videophy2_sft_nano import VideoPhy2DataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader as LegacyLoader
from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader as NewLoader,
    IterableDistributor,
    PoolPackingBatcher,
)


class _FakeProcessor:
    """Deterministic fake: produces token count proportional to text length."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # Sum text lengths across all messages for a deterministic token count.
        total = 4
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                total += len(c)
            elif isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += len(item.get("text", ""))
        return {"input_ids": torch.arange(total)}

    def add_assistant_tokens_mask(self, input_ids):
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[len(input_ids) // 2 :] = True
        return mask


class _FixedIterable(torch.utils.data.IterableDataset):
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        yield from self._items


def _make_items(k):
    out = []
    for i in range(k):
        out.append({
            "texts": [
                {"role": "user", "content": "question" * (i % 5 + 1)},
                {"role": "assistant", "content": "answer" * (i % 3 + 1)},
            ],
            "media": {},
        })
    return out


def _drain(loader, n):
    it = iter(loader)
    return [next(it) for _ in range(n)]


def test_videophy2_golden_batches_match():
    proc = _FakeProcessor()
    items = _make_items(40)

    random.seed(0)
    legacy = LegacyLoader(
        data_source=_FixedIterable(list(items)),
        data_packer=VideoPhy2DataPacker(tokenizer_config=proc, max_seq_len=200),
        max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400,
        num_workers=0,
    )
    random.seed(0)
    new = NewLoader(
        distributor=IterableDistributor(list(items)),
        processor=VideoPhy2Processor(processor=proc),
        batcher=PoolPackingBatcher(max_tokens=200, pool_size=8, max_batch_size=1, long_threshold=6400),
        collator=VLMCollator(),
        num_workers=0,
    )

    a = _drain(legacy, 8)
    b = _drain(new, 8)
    for i, (ba, bb) in enumerate(zip(a, b)):
        assert ba.keys() == bb.keys(), f"batch {i}: key mismatch {ba.keys()} vs {bb.keys()}"
        for k in ba:
            assert torch.equal(ba[k], bb[k]), f"batch {i} key {k!r}: {ba[k]} vs {bb[k]}"
