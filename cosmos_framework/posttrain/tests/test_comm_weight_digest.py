# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The exact-byte tensor digest shared by the vLLM witness and the vLLM-Omni weight plane."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata


@pytest.mark.L0
@pytest.mark.CPU
def test_digest_is_exact_bytes_in_json_shape() -> None:
    """Equal bytes digest equal; one flipped bit, a different dtype or a different shape does not."""
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    digest = tensor_digest_metadata(t)
    assert digest["shape"] == [2, 3] and isinstance(digest["shape"], list)
    assert digest["dtype"] == "torch.float32" and len(digest["sha256"]) == 64
    assert tensor_digest_metadata(t.clone()) == digest

    flipped = t.clone()
    flipped.view(torch.int32)[0, 0] ^= 1  # the smallest possible change: one mantissa bit
    assert tensor_digest_metadata(flipped) != digest
    assert tensor_digest_metadata(t.to(torch.bfloat16)) != digest
    assert tensor_digest_metadata(t.reshape(3, 2)) != digest


@pytest.mark.L0
@pytest.mark.CPU
def test_digest_reads_logical_bytes_of_a_strided_view() -> None:
    """A transposed view digests as its contiguous copy would: the digest is of values, not of storage."""
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_digest_metadata(t.t()) == tensor_digest_metadata(t.t().contiguous())
    assert tensor_digest_metadata(t.t()) != tensor_digest_metadata(t)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_weights_reexports_the_shared_digest() -> None:
    """One owner: the witness path and the Omni path must hash the same way."""
    from cosmos_framework.posttrain.roles.rollout.vllm_weights import _tensor_digest_metadata

    assert _tensor_digest_metadata is tensor_digest_metadata


@pytest.mark.L0
@pytest.mark.CPU
def test_digest_transport_normalization_preserves_values():
    from cosmos_framework.posttrain.comm.weight_digest import normalize_tensor_digest, tensor_digests_equal

    expected = tensor_digest_metadata(torch.arange(6).reshape(2, 3))
    transported = {**expected, "shape": tuple(expected["shape"])}
    assert normalize_tensor_digest(transported) == expected
    assert tensor_digests_equal(transported, expected)
    assert isinstance(transported["shape"], tuple)
    for other in (torch.arange(6).reshape(3, 2), torch.arange(6).reshape(2, 3).float(), torch.ones(2, 3)):
        assert not tensor_digests_equal(tensor_digest_metadata(other), expected)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("invalid", ["missing", "extra", "shape_type", "shape_dimension", "dtype", "sha256"])
def test_digest_metadata_rejects_malformed_fields(invalid):
    from cosmos_framework.posttrain.comm.weight_digest import normalize_tensor_digest

    value = tensor_digest_metadata(torch.zeros(2))
    if invalid == "missing":
        del value["dtype"]
    elif invalid == "extra":
        value["unexpected"] = None
    elif invalid == "shape_type":
        value["shape"] = "2"
    elif invalid == "shape_dimension":
        value["shape"] = [True]
    elif invalid == "dtype":
        value["dtype"] = torch.float32
    else:
        value["sha256"] = value["sha256"][:-1]
    with pytest.raises(TypeError, match="malformed tensor digest metadata"):
        normalize_tensor_digest(value)
