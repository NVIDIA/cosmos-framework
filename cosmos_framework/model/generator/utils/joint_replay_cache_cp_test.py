# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise persistent replay selection through real CP sequence/head exchanges."""

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.model.generator.mot.causal_context_parallel_test import _make_factored_pack
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_attention
from cosmos_framework.model.generator.mot.maskless_attention import build_replay_maskless_plan
from cosmos_framework.model.generator.utils.joint_replay_cache_test import (
    _LAYERS,
    _TEXT,
    _memory,
    _metadata,
    _pack,
    _policy,
    _ReplayPack,
)
from cosmos_framework.model.generator.utils.kv_cache import JointARMemoryValue, TFNoisyMemoryValue
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryValue
from cosmos_framework.model.generator.utils.multiview_ar import JointARCache, joint_ar_chunks
from cosmos_framework.model.generator.utils.multiview_test_utils import _dense_reference, _pass_weights
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack, from_mode_splits


def _project_shards(pack: _ReplayPack, rank: int, kv_heads: int) -> list[torch.Tensor]:
    """Distinct head projections expose either head slicing or GQA repetition errors."""
    captions = torch.sin(torch.arange(_TEXT * 8).reshape(_TEXT, 8) * 0.23)  # [U,C]
    hidden = torch.cat((captions, pack.values))  # [U+G,C]
    result: list[torch.Tensor] = []  # each [U/CP+G/CP,H,D]
    for index, heads in enumerate((4, kv_heads, kv_heads)):
        projection = torch.sin(torch.arange(8 * heads * 4).reshape(8, heads * 4) * 0.13 + index) * 0.2
        # projection: [C,H*D]
        projected = (hidden @ projection).reshape(-1, heads, 4)  # [U+G,H,D]
        result.append(torch.cat((projected[:_TEXT].chunk(2)[rank], projected[_TEXT:].chunk(2)[rank])))
        # result[-1]: [U/CP+G/CP,H,D]
    return result


def _cp_forward(
    mesh: DeviceMesh,
    tensors: list[torch.Tensor],  # each [U/CP+G/CP,H,D]
    gen_len: int,
    weights: torch.Tensor,  # [G,U+G+M]
    memory: MemoryValue,
) -> tuple[torch.Tensor, KVToStore]:  # [G/CP,H,D], global-token/local-head K/V
    packs = [
        _make_factored_pack(value[: _TEXT // 2], value[_TEXT // 2 :], _TEXT, gen_len, value.device, is_sharded=True)
        for value in tensors
    ]

    def attention(
        query: SequencePack,
        key: SequencePack,
        value: SequencePack,
        attention_mask: SplitInfo,
        *,
        memory_value: MemoryValue,
        **kwargs: Any,
    ) -> tuple[SequencePack, None]:
        del attention_mask, kwargs
        keys = torch.cat((key["causal_seq"], key["full_only_seq"])).unsqueeze(0)  # [1,U+G,H_KV/CP,D]
        values = torch.cat((value["causal_seq"], value["full_only_seq"])).unsqueeze(0)  # [1,U+G,H_KV/CP,D]
        if isinstance(memory_value, TFNoisyMemoryValue):
            keys = torch.cat((keys, memory_value.cached_clean_gen_k), dim=1)  # [1,U+G+M,H_KV/CP,D]
            values = torch.cat((values, memory_value.cached_clean_gen_v), dim=1)  # [1,U+G+M,H_KV/CP,D]
        elif isinstance(memory_value, JointARMemoryValue) and memory_value.cached_gen_k is not None:
            assert memory_value.cached_gen_v is not None
            keys = torch.cat((keys, memory_value.cached_gen_k), dim=1)  # [1,U+G+M,H_KV/CP,D]
            values = torch.cat((values, memory_value.cached_gen_v), dim=1)  # [1,U+G+M,H_KV/CP,D]
        output = _dense_reference(
            query["full_only_seq"].unsqueeze(0), keys, values, weights, torch.ones(gen_len, dtype=torch.bool)
        ).squeeze(0)  # [G,H/CP,D]
        return from_mode_splits(query["causal_seq"], output, query), None

    mask = SplitInfo(
        split_lens=[_TEXT, gen_len],
        attn_modes=["causal", "full"],
        sample_lens=[_TEXT + gen_len],
        actual_len=_TEXT + gen_len,
    )
    output, captured = context_parallel_attention(mesh, *packs, mask, attention, memory_value=memory)
    assert captured is not None
    return output["full_only_seq"], captured  # [G/CP,H,D], K/V: [1,G,H_KV/CP,D]


def _replay_cp_worker(rank: int, rendezvous: str, kv_heads: int, has_lidar: bool) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2,))
        policy = _policy("decomposed")
        chunks = joint_ar_chunks(
            vision_frames=10,
            lidar_frames=13,
            vision_seconds_per_frame=4 / 30,
            lidar_seconds_per_frame=0.1,
            frames_per_chunk=3,
        )
        reference = _pack(tuple(range(10)), tuple(range(13)) if has_lidar else (), 2)
        cache = JointARCache(reference.packed, _LAYERS)
        with torch.no_grad():
            for chunk in chunks[:-1]:
                lidar = chunk.lidar_frames if has_lidar else ()
                current = _pack(chunk.vision_frames, lidar, 2)
                cache.prepare(current.packed, chunk.vision_frames, lidar)
                metadata = cache.augment_metadata(
                    _metadata(current, policy, noisy=False), current.items, 3, noisy=False
                )
                weights = sum(_pass_weights(build_replay_maskless_plan(metadata, current.items)).values())  # [G,U+G+M]
                _, captured = _cp_forward(
                    mesh,
                    _project_shards(current, rank, kv_heads),
                    len(current.identities),
                    weights,
                    cache.memory.read_for_layer(0),
                )
                cache.memory.spans = cache.spans
                for layer in range(_LAYERS):
                    cache.memory.write_for_layer(layer, captured)
                cache.commit_metadata()

        # A shorter prefix than the allocation exposes camera-stride remapping.
        chunk = chunks[2]
        prefix = _pack(
            tuple(range(chunk.vision_prefix_end)), tuple(range(chunk.lidar_prefix_end)) if has_lidar else (), 2
        )
        expected, actual = _memory(prefix, policy), _memory(prefix, policy)
        for memory in (expected, actual):
            memory.context_parallel_size = 2
            memory.num_kv_heads = max(kv_heads // 2, 1)
            memory.init(
                {
                    "causal_seq": torch.zeros(_TEXT // 2, 8),
                    "_num_causal_tokens": _TEXT,
                    "_num_full_tokens": len(prefix.identities),
                },
                torch.device("cpu"),
            )
        with torch.no_grad():
            clean_weights = sum(
                _pass_weights(build_replay_maskless_plan(_metadata(prefix, policy, noisy=False), prefix.items)).values()
            )  # [G,U+G]
            _, captured = _cp_forward(
                mesh,
                _project_shards(prefix, rank, kv_heads),
                len(prefix.identities),
                clean_weights,
                expected.read_for_layer(0),
            )
            for layer in range(_LAYERS):
                expected.write_for_layer(layer, captured)
            expected.pass_number = 2
            cache.populate_replay_memory(prefix.packed, actual)
        expected_value, actual_value = expected.read_for_layer(0), actual.read_for_layer(0)
        assert isinstance(expected_value, TFNoisyMemoryValue) and isinstance(actual_value, TFNoisyMemoryValue)
        torch.testing.assert_close(actual_value.cached_clean_gen_k, expected_value.cached_clean_gen_k)
        torch.testing.assert_close(actual_value.cached_clean_gen_v, expected_value.cached_clean_gen_v)
        assert actual_value.cached_clean_gen_k.shape[2:] == (max(kv_heads // 2, 1), 4)
        metadata = _metadata(prefix, policy, noisy=True, capacity=actual.selected_clean_gen_padded_capacity)
        weights = sum(_pass_weights(build_replay_maskless_plan(metadata, prefix.items)).values())  # [G,U+G+M]
        outputs: list[torch.Tensor] = []  # each [G/CP,H,D]
        gradients: list[list[torch.Tensor]] = []  # each [U/CP+G/CP,H,D]
        for value in (expected_value, actual_value):
            inputs = [tensor.detach().clone().requires_grad_() for tensor in _project_shards(prefix, rank, kv_heads)]
            # inputs: each [U/CP+G/CP,H,D]
            output, _ = _cp_forward(mesh, inputs, len(prefix.identities), weights, value)  # [G/CP,H,D], K/V
            output.square().mean().backward()
            outputs.append(output.detach())  # [G/CP,H,D]
            grads: list[torch.Tensor] = []  # each [U/CP+G/CP,H,D]
            for tensor in inputs:
                assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
                grads.append(tensor.grad.detach().clone())  # [U/CP+G/CP,H,D]
            gradients.append(grads)
        torch.testing.assert_close(outputs[0], outputs[1])
        for expected_grad, actual_grad in zip(gradients[0], gradients[1], strict=True):
            torch.testing.assert_close(actual_grad, expected_grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.no_xdist
@pytest.mark.parametrize("kv_heads", [1, 4])
@pytest.mark.parametrize("has_lidar", [False, True])
def test_persistent_replay_preserves_cp_outputs_and_gradients(tmp_path: Path, kv_heads: int, has_lidar: bool) -> None:
    torch.multiprocessing.start_processes(
        _replay_cp_worker,
        args=(f"file://{tmp_path / 'cp-cache-store'}", kv_heads, has_lidar),
        nprocs=2,
        start_method="fork",
    )
