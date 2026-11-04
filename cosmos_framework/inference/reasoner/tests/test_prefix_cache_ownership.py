# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Repeated branches preserve dynamic KV and Qwen3.5 recurrent prefix state."""

import pytest
import torch
from transformers import cache_utils

from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime


def test_hybrid_prefix_forks_isolate_recurrent_updates_and_share_append_only_kv():
    if not hasattr(cache_utils, "LinearAttentionLayer"):
        pytest.skip("hybrid DynamicCache is a Transformers 5 capability")
    from transformers import Qwen3_5TextConfig

    cache = cache_utils.DynamicCache(
        config=Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"])
    )
    conv = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    recurrent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    key = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    cache.update_conv_state(conv, 0)
    cache.update_recurrent_state(recurrent, 0)
    cache.update(key, key + 1, 1)
    for branch_number in range(3):
        branch = CosmosFrameworkRuntime._fork_dynamic_cache(cache)
        assert branch.layers[1].keys is cache.layers[1].keys
        assert branch.layers[0].conv_states.data_ptr() != cache.layers[0].conv_states.data_ptr()
        assert branch.layers[0].recurrent_states.data_ptr() != cache.layers[0].recurrent_states.data_ptr()
        # Exercise both the public updates and direct in-place kernel mutation.
        branch.update_conv_state(torch.full((1, 6, 1), branch_number + 10.0), 0)
        branch.layers[0].recurrent_states.add_(branch_number + 1)
        branch.update_recurrent_state(torch.zeros_like(recurrent), 0)
        branch.update(key[..., :1, :], key[..., :1, :] + 2, 1)
        assert branch.get_seq_length() == 5
        assert cache.get_seq_length() == 4
        torch.testing.assert_close(cache.layers[0].conv_states, conv, rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[0].recurrent_states, recurrent, rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[1].keys, key, rtol=0, atol=0)


def test_prefix_fork_rejects_unknown_in_place_cache_layers():
    cache = cache_utils.DynamicCache()
    cache.update(torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4), 0)
    cache.layers[0] = cache_utils.StaticLayer(max_cache_len=8)
    with pytest.raises(TypeError, match="mutation/ownership"):
        CosmosFrameworkRuntime._fork_dynamic_cache(cache)
