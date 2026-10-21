# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cosmos_framework.posttrain.data.packer.base import BaseDataPacker, DataPacker


class _Dummy(DataPacker):
    def get_rollout_input(self, item):
        return item

    def get_policy_input(self, sample, rollout_output, n_ignore_prefix_tokens=0):
        return rollout_output

    def policy_compute_max_len(self, processed_samples):
        return 0

    def policy_collate_fn(self, processed_samples, computed_max_len):
        return {}


@pytest.mark.L0
@pytest.mark.CPU
def test_register_and_get_roundtrip():
    BaseDataPacker.register("dummy-mt", _Dummy)
    got = BaseDataPacker.get_default_data_packer("dummy-mt")
    assert isinstance(got, _Dummy)


@pytest.mark.L0
@pytest.mark.CPU
def test_register_double_raises_and_override_ok():
    class _Other(_Dummy):
        pass

    BaseDataPacker.register("dup-mt", _Dummy)
    with pytest.raises(ValueError, match="already registered"):
        BaseDataPacker.register("dup-mt", _Other)
    BaseDataPacker.register("dup-mt", _Other, allow_override=True)  # no raise


@pytest.mark.L0
@pytest.mark.CPU
def test_get_unregistered_raises():
    with pytest.raises(ValueError, match="no data packer registered"):
        BaseDataPacker.get_default_data_packer("nonesuch-mt")


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_builds_tokenizer_with_pad():
    p = _Dummy()
    p.setup("gpt2")
    assert p.tokenizer is not None
    assert p.pad_token_id is not None  # gpt2 has no pad -> falls back to eos


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_empty_name_raises():
    with pytest.raises(ValueError, match="empty name_or_path"):
        _Dummy().setup("")


@pytest.mark.L0
@pytest.mark.CPU
def test_sft_methods_are_future_notimplemented():
    with pytest.raises(NotImplementedError, match="future work"):
        _Dummy().sft_process_sample({})
