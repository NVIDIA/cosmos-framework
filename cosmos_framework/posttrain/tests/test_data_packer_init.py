# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from cosmos_framework.posttrain.data import packer as pk


@pytest.mark.L0
@pytest.mark.CPU
def test_init_selects_registered_decoder_for_gpt2():
    p = pk.init_data_packer("gpt2")
    assert isinstance(p, pk.DecoderOnlyLLMDataPacker)
    assert p.tokenizer is not None


@pytest.mark.L0
@pytest.mark.CPU
def test_init_explicit_override_used():
    given = pk.DecoderOnlyLLMDataPacker()
    p = pk.init_data_packer("gpt2", packer=given)
    assert p is given
    assert p.tokenizer is not None  # still setup()


@pytest.mark.L0
@pytest.mark.CPU
def test_init_no_longer_needs_a_registry_entry_for_a_classifiable_model(monkeypatch):
    """An EMPTY registry is now fine for a model the probe can classify.

    This test used to assert the opposite: that an unlisted `model_type` raised, because the packer
    layer could not tell a VLM from a text model and a silent text-packer guess would have
    mis-shaped one. The probe (`_select_packer_cls`) can now tell them apart via `vision_config`,
    so the enumeration is gone and the guard narrowed to what it can still protect: an arch that is
    neither a VLM nor a registered causal LM, pinned by
    `test_unclassifiable_model_still_raises`.
    """
    monkeypatch.setattr(pk.BaseDataPacker, "_REGISTRY", {})
    p = pk.init_data_packer("gpt2")
    assert isinstance(p, pk.DecoderOnlyLLMDataPacker)


@pytest.mark.L0
@pytest.mark.CPU
def test_init_propagates_registered_packer_constructor_error(monkeypatch):
    # A registered packer whose constructor raises must PROPAGATE, not be swallowed.
    class _Boom(pk.DecoderOnlyLLMDataPacker):
        def __init__(self):
            raise ValueError("boom in constructor")

    monkeypatch.setattr(pk.BaseDataPacker, "_REGISTRY", {"gpt2": _Boom})
    with pytest.raises(ValueError, match="boom in constructor"):
        pk.init_data_packer("gpt2")


@pytest.mark.L0
@pytest.mark.CPU
def test_unclassifiable_model_still_raises():
    """The no-silent-fallback guard relaxes only where the probe is confident. Anything it cannot
    classify must still fail loud rather than be handed the text packer."""
    from cosmos_framework.posttrain.data.packer import _select_packer_cls

    class _Cfg:
        model_type = "some_encoder_decoder_arch"

    with pytest.raises(ValueError, match="cannot classify"):
        _select_packer_cls(_Cfg())


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("model_type", ["bert", "t5"])
def test_hf_text_fallback_rejects_non_causal_transformers_architectures(model_type: str) -> None:
    """Encoder-only and encoder-decoder families cannot use the decoder-only RL packer."""
    from cosmos_framework.posttrain.data.packer import _select_packer_cls

    config = type("_Config", (), {"model_type": model_type})()
    with pytest.raises(ValueError, match="cannot classify"):
        _select_packer_cls(config, model_backend="hf")
