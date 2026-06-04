# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Contract tests for the four dataflow role ABCs."""

from __future__ import annotations

import inspect

import pytest

from cosmos_framework.data.vfm.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)


def test_abcs_cannot_be_instantiated():
    for cls in (DataDistributor, RawItemProcessor, SampleBatcher, BatchCollator):
        with pytest.raises(TypeError):
            cls()  # abstract


def test_distributor_state_dict_defaults_are_noops():
    class _D(DataDistributor):
        def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
            yield from ()

    d = _D()
    assert d.state_dict() == {}
    assert d.load_state_dict({"anything": 1}) is None  # no-op default


def test_role_method_signatures():
    assert list(inspect.signature(DataDistributor.stream).parameters) == [
        "self", "dp_rank", "dp_world_size", "worker_id", "num_workers",
    ]
    assert list(inspect.signature(RawItemProcessor.process).parameters) == ["self", "item"]
    assert list(inspect.signature(SampleBatcher.batches).parameters) == ["self", "samples"]
    assert list(inspect.signature(BatchCollator.collate).parameters) == ["self", "samples"]
