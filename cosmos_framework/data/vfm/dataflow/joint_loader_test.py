# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""JointCosmosDataLoader: ratio-based batch interleave + state-callback surface."""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    IterableDistributor,
    JointCosmosDataLoader,
    SimpleBatcher,
)


def _loader(tag):
    return CosmosDataLoader(
        distributor=IterableDistributor([{"x": torch.tensor([float(i)]), "tag": tag} for i in range(100000)]),
        processor=IdentityProcessor(),
        batcher=SimpleBatcher(batch_size=1),
        num_workers=0,
    )


def test_joint_tags_batches_and_interleaves_by_ratio():
    j = JointCosmosDataLoader(
        {"a": {"dataloader": _loader("a"), "ratio": 3}, "b": {"dataloader": _loader("b"), "ratio": 1}},
        seed=0,
    )
    it = iter(j)
    names = [next(it)["dataset_name"] for _ in range(400)]
    assert set(names) == {"a", "b"}
    assert 0.6 < names.count("a") / len(names) < 0.85


def test_joint_exposes_callback_surface():
    j = JointCosmosDataLoader({"a": {"dataloader": _loader("a"), "ratio": 1}}, seed=0)
    assert j._names == ["a"]
    assert hasattr(j, "set_start_iteration") and hasattr(j, "_global_id")
    j.set_start_iteration(5)
    assert j._global_id == 5
