# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge attention default must match the platform's kernels."""

from importlib.metadata import PackageNotFoundError, version

from cosmos_framework.configs.base.reasoner.defaults.vlm_policy import cosmos3_edge_reasoner


def test_edge_policy_never_defaults_to_cosmos_adapter():
    # The "cosmos" NATTEN adapter is Qwen3-VL-specific and rejects Edge's
    # explicit attention mask; Edge must default to a mask-capable impl.
    assert cosmos3_edge_reasoner.attn_implementation in {"flash_attention_2", "sdpa"}


def test_edge_policy_matches_flash_attn_availability():
    try:
        installed = version("flash_attn").split(".", 1)[0] == "2"
    except PackageNotFoundError:
        installed = False
    expected = "flash_attention_2" if installed else "sdpa"
    assert cosmos3_edge_reasoner.attn_implementation == expected
