# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

import cosmos_framework.model.generator.mot.inference_text_kv_memory as text_kv_memory
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    UndKVCache,
)


def _gen_pack(gen: torch.Tensor, und_offsets: torch.Tensor, gen_offsets: torch.Tensor) -> dict:
    num_samples = und_offsets.numel() - 1
    return {
        "causal_seq": gen.new_empty(0, *gen.shape[1:]),
        "full_only_seq": gen,
        "sample_offsets": torch.arange(num_samples + 1),
        "max_sample_len": 1,
        "max_causal_len": 0,
        "max_full_len": int((gen_offsets[1:] - gen_offsets[:-1]).max()),
        "_causal_indices": torch.empty(0, dtype=torch.int64),
        "_full_indices": torch.arange(gen.shape[0]),
        "_causal_seq_offsets": und_offsets,
        "_full_only_seq_offsets": gen_offsets,
        "is_sharded": False,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_sample_cached_attention_isolates_samples_and_zero_fills_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    und_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    gen_offsets = torch.tensor([0, 1, 3], dtype=torch.int32)
    cache = UndKVCache()
    cached_und = torch.tensor([10.0, 11.0, 20.0]).reshape(1, 3, 1, 1)
    cache.store(cached_und, cached_und)

    state = InferenceTextKVMemoryState([cache])
    state.init(
        {
            "_num_full_tokens": 3,
            "_causal_seq_offsets": und_offsets,
            "_full_only_seq_offsets": gen_offsets,
        },
        torch.device("cpu"),
    )
    memory_value = state.read_for_layer(0)

    q_gen = torch.tensor([1.0, 2.0, 3.0, 999.0]).reshape(4, 1, 1)
    k_gen = torch.tensor([100.0, 200.0, 201.0, 999.0]).reshape(4, 1, 1)
    v_gen = k_gen.clone()
    captured: dict[str, object] = {}

    def fake_attention(
        *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        captured.update(query=query, key=key, value=value, **kwargs)
        return query + 5

    monkeypatch.setattr(text_kv_memory, "attention", fake_attention)
    output, kv_to_store = text_kv_memory._attention_gen_with_cached_text(
        _gen_pack(q_gen, und_offsets, gen_offsets),
        _gen_pack(k_gen, und_offsets, gen_offsets),
        _gen_pack(v_gen, und_offsets, gen_offsets),
        memory_value,
    )

    assert kv_to_store is None
    assert captured["key"].flatten().tolist() == [10.0, 11.0, 100.0, 20.0, 200.0, 201.0]
    assert captured["value"].flatten().tolist() == [10.0, 11.0, 100.0, 20.0, 200.0, 201.0]
    assert torch.equal(captured["cumulative_seqlen_Q"], gen_offsets)
    assert torch.equal(captured["cumulative_seqlen_KV"], torch.tensor([0, 3, 6], dtype=torch.int32))
    assert captured["max_seqlen_Q"] == 2
    assert captured["max_seqlen_KV"] == 3
    assert output["full_only_seq"].flatten().tolist() == [6.0, 7.0, 8.0, 0.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_sample_layout_rejects_mismatched_sample_counts() -> None:
    state = InferenceTextKVMemoryState([UndKVCache()])

    with pytest.raises(RuntimeError, match="und/gen sample counts disagree"):
        state.init(
            {
                "_num_full_tokens": 2,
                "_causal_seq_offsets": torch.tensor([0, 1, 2], dtype=torch.int32),
                "_full_only_seq_offsets": torch.tensor([0, 2], dtype=torch.int32),
            },
            torch.device("cpu"),
        )
