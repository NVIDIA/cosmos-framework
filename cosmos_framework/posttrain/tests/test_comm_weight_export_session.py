# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qwen shard-state lifetime through the real producer-driven publication path."""

from __future__ import annotations

import weakref
from typing import Any

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, WeightTargetSet, push_stream_to_peer
from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLWeightMapper
from cosmos_framework.posttrain.roles.trainer.local import LocalTrainer

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _model() -> nn.Module:
    net = nn.Module()
    net.model = nn.Module()
    net.model.language_model = nn.Module()
    net.model.language_model.embed_tokens = nn.Embedding(8, 4)  # weight: [V,H]
    attention = nn.Module()
    attention.q_proj = nn.Linear(4, 4, bias=False)  # weight: [Q,H]
    attention.k_proj = nn.Linear(4, 4, bias=False)  # weight: [K,H]
    attention.v_proj = nn.Linear(4, 4, bias=False)  # weight: [V,H]
    layer = nn.Module()
    layer.self_attn = attention
    net.model.language_model.layers = nn.ModuleList([layer])
    net.lm_head = nn.Linear(4, 8, bias=False)  # weight: [V,H]
    return net


def test_export_session_preserves_values_and_refreshes_replaced_storage() -> None:
    net = _model()
    mapper = Qwen3VLWeightMapper("qwen3_vl", tie_word_embeddings=False)
    manifest = mapper.wire_manifest(net)
    for _ in range(2):
        with mapper.export_session(net):
            for entry in manifest:
                exported = mapper.export_bucket(net, [entry])  # values: one full parameter
                parameter = net.get_parameter(entry[0])  # [*weight]
                assert torch.equal(exported[entry[0]], parameter)
                assert not exported[entry[0]].requires_grad
        replacement = torch.full_like(net.lm_head.weight, 17)  # [V,H]
        net.lm_head.weight = nn.Parameter(replacement)  # [V,H], new storage for the next publication
    # Standalone exports must also read current storage after the stream has closed.
    net.lm_head.weight = nn.Parameter(torch.full_like(net.lm_head.weight, 23))  # [V,H]
    assert torch.equal(mapper.export_state(net)["lm_head.weight"], net.lm_head.weight)


@pytest.mark.parametrize("fail_first", [False, True])
def test_push_builds_state_once_and_clears_it_on_success_or_failure(fail_first: bool) -> None:
    net = _model()
    mapper = Qwen3VLWeightMapper("qwen3_vl", tie_word_embeddings=False)
    backend = WeightSync(CopyMover(), mapper=mapper, bucket_bytes=64)
    backend.ensure_plan(net)  # Binding prepares metadata before publication.
    assert backend.bucket_count() > 1
    producer = LocalTrainer(net)
    snapshots: list[dict[str, Any]] = []
    references: list[weakref.ReferenceType[torch.Tensor]] = []

    def observe_state(module: nn.Module, state: dict[str, Any], prefix: str, metadata: dict[str, Any]) -> None:
        snapshots.append(state)
        references.extend(weakref.ref(value) for value in state.values())

    handle = net.register_state_dict_post_hook(observe_state)
    try:
        original = net.lm_head.weight  # [V,H]
        failure: pytest.ExceptionInfo[ValueError] | None = None
        if fail_first:
            # Keep the manifest but replace storage with an incompatible shape: real export validation fails.
            net.lm_head.weight = nn.Parameter(original[:1].clone())  # [1,H]
            with pytest.raises(ValueError, match="shape changed") as failure:
                push_stream_to_peer(producer, WeightTargetSet(()), backend)
            assert len(snapshots) == 1 and not snapshots[0]
            net.lm_head.weight = original  # [V,H], restore the valid source for retry
        before = len(snapshots)
        push_stream_to_peer(producer, WeightTargetSet(()), backend)
        assert len(snapshots) == before + 1
        assert all(not state for state in snapshots)
        assert mapper._export_state is None
        # A retained exception may own the one failing tensor, but never the full shard dictionary.
        if failure is None:
            assert all(reference() is None for reference in references)
        assert backend.last_push_timing is not None
        assert len(backend.last_push_timing.materialize_s) == backend.bucket_count()
    finally:
        handle.remove()
