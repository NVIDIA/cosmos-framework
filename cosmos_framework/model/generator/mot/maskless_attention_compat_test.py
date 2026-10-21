# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only checks; successful mocks do not establish SM120 kernel correctness."""

from importlib.metadata import PackageNotFoundError
from types import ModuleType
from unittest.mock import patch

import pytest

from cosmos_framework.model.generator.mot import maskless_attention_compat as compat


def _forward(
    q,
    k,
    v,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    out=None,
    lse=None,
    return_lse=False,
):
    raise AssertionError("A CPU prerequisite test must never launch a kernel")


def _backend(forward=_forward):
    backend = ModuleType("flash_attn.cute.interface")
    backend._flash_attn_fwd = forward
    return backend


@pytest.mark.parametrize("installed", ["4.0.0b24", "4.0.0b28", "4.0.0b29", "4.0.0b31", "4.1.0", "4.0.0b30+local"])
def test_unreviewed_or_unsafe_versions_are_rejected(installed):
    with patch.object(compat, "version", return_value=installed):
        reason = compat.sm120_gapped_forward_unavailable_reason(_backend())
    assert reason is not None
    assert "PR #2763" in reason


def test_b30_with_explicit_gapped_forward_api_is_accepted():
    with patch.object(compat, "version", return_value="4.0.0b30"):
        assert compat.sm120_gapped_forward_unavailable_reason(_backend()) is None


def test_missing_distribution_metadata_is_rejected():
    with patch.object(compat, "version", side_effect=PackageNotFoundError("flash-attn-4")):
        assert "metadata" in compat.sm120_gapped_forward_unavailable_reason(_backend())


@pytest.mark.parametrize("forward", [None, object(), lambda q, k, v, **kwargs: None])
def test_missing_or_implicit_api_is_rejected(forward):
    with patch.object(compat, "version", return_value="4.0.0b30"):
        assert compat.sm120_gapped_forward_unavailable_reason(_backend(forward)) is not None


def test_uninspectable_forward_is_rejected():
    with (
        patch.object(compat, "version", return_value="4.0.0b30"),
        patch.object(compat, "signature", side_effect=ValueError("opaque callable")),
    ):
        assert "inspect" in compat.sm120_gapped_forward_unavailable_reason(_backend())


@pytest.mark.parametrize("missing", sorted(compat._GAPPED_FORWARD_KEYWORDS))
def test_every_gapped_keyword_is_required(missing):
    actual = compat.signature(_forward)
    changed = actual.replace(parameters=[parameter for name, parameter in actual.parameters.items() if name != missing])
    with (
        patch.object(compat, "version", return_value="4.0.0b30"),
        patch.object(compat, "signature", return_value=changed),
    ):
        assert missing in compat.sm120_gapped_forward_unavailable_reason(_backend())


def test_positional_only_gapped_parameter_is_rejected():
    actual = compat.signature(_forward)
    parameter = compat.Parameter("out", compat.Parameter.POSITIONAL_ONLY)
    changed = actual.replace(parameters=[parameter, *[p for name, p in actual.parameters.items() if name != "out"]])
    with (
        patch.object(compat, "version", return_value="4.0.0b30"),
        patch.object(compat, "signature", return_value=changed),
    ):
        assert "out" in compat.sm120_gapped_forward_unavailable_reason(_backend())


def test_incompatible_tensor_calling_convention_is_rejected():
    actual = compat.signature(_forward)
    changed = actual.replace(parameters=[parameter for name, parameter in actual.parameters.items() if name != "v"])
    with (
        patch.object(compat, "version", return_value="4.0.0b30"),
        patch.object(compat, "signature", return_value=changed),
    ):
        assert "calling convention" in compat.sm120_gapped_forward_unavailable_reason(_backend())
