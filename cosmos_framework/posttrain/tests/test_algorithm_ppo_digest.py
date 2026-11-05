# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact wire compatibility and bounded allocation of streaming PPO digests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest
from cosmos_framework.posttrain.tests.helpers.ppo_digest import digest_values

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
GOLDEN = json.loads((Path(__file__).parent / "fixtures/ppo_canonical_digests.json").read_text())["digests"]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_streaming_digest_matches_original_wire_bytes(name):
    assert canonical_digest(digest_values()[name]) == GOLDEN[name]


def test_streaming_digest_reads_shared_tensor_tail_after_inplace_mutation():
    values = digest_values()
    tensor = values["torch_bfloat16"]
    original = canonical_digest(values["nested_records"])
    tensor.view(torch.uint8).reshape(-1)[-1] ^= 1
    assert canonical_digest(tensor) != GOLDEN["torch_bfloat16"]
    assert canonical_digest(values["nested_records"]) != original


def test_streaming_digest_preserves_mapping_order_independence_and_type_tags():
    mapping = digest_values()["mixed_mapping_keys"]
    assert canonical_digest(dict(reversed(list(mapping.items())))) == GOLDEN["mixed_mapping_keys"]
    values = [True, 1, 1.0, "1", b"1", bytearray(b"1"), memoryview(b"1"), (1,), [1]]
    assert len({canonical_digest(value) for value in values}) == len(values)


@pytest.mark.parametrize("value,error", [(float("nan"), ValueError), (float("inf"), ValueError), (object(), TypeError)])
def test_streaming_digest_rejects_unsupported_values(value, error):
    with pytest.raises(error, match="canonical digest|canonical PPO digest"):
        canonical_digest({"valid": torch.arange(4), "invalid": value})


def test_streaming_digest_does_not_allocate_tensor_sized_python_payloads():
    # An isolated process keeps tracemalloc's global state independent of the
    # runner and excludes torch/model imports from the measured interval.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, tracemalloc\n"
            "from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest\n"
            "from cosmos_framework.posttrain.tests.helpers.ppo_digest import digest_values\n"
            "payload = digest_values()['large_tensor']\n"
            "tracemalloc.start()\n"
            "digest = canonical_digest(payload)\n"
            "_, peak = tracemalloc.get_traced_memory()\n"
            "print(json.dumps({'digest': digest, 'peak': peak, 'nbytes': payload['prefix'].nbytes}))\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    measured = json.loads(result.stdout)
    assert measured["digest"] == GOLDEN["large_tensor"]
    assert measured["peak"] < measured["nbytes"] // 4
