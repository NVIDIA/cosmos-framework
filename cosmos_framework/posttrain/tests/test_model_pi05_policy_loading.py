# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Production policy construction binds resolved sources and releases failed resources."""

import gc
import json
import traceback
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.model.pi05 import policy
from cosmos_framework.posttrain.model.pi05.source import Pi05Source, load_model_weights
from cosmos_framework.posttrain.tests.helpers.pi05_loading import SmallPi05Model, write_source

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def preserve_precision():
    before = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(before)


def test_policy_processor_uses_the_resolved_source_after_symlink_redirect(tmp_path, monkeypatch):
    config = write_source(tmp_path)
    original = tmp_path / "norm_stats.json"
    link = tmp_path / "stats-link.json"
    link.symlink_to(original)
    foreign = tmp_path / "foreign-stats.json"
    foreign.write_text("{}")
    config = replace(config, norm_stats_path=str(link))

    def build(resolved, **kwargs):
        link.rename(tmp_path / "original-link.json")
        link.symlink_to(foreign)
        return SmallPi05Model(resolved, **kwargs)

    monkeypatch.setattr(policy, "Pi05Model", build)
    before = torch.random.get_rng_state()
    loaded = policy.Pi05TrainPolicy(config, device="cpu")
    assert loaded.processor.config.norm_stats_path == str(original.resolve())
    assert loaded.net.value_head is loaded.model.value_head
    assert loaded.net.expert is loaded.model.paligemma_with_expert.gemma_expert.model
    assert loaded.ppo_frozen_state()
    assert loaded.precision.transfer is None
    assert {parameter.dtype for parameter in loaded.net.parameters()} == {torch.float32, torch.bfloat16}
    assert torch.equal(torch.random.get_rng_state(), before)


def test_post_load_reference_precision_roundtrips_real_sft_norm_values_only(tmp_path, monkeypatch):
    config = write_source(tmp_path)
    fixture = json.loads((Path(__file__).parent / "fixtures/pi05_sft_norm_fp32.json").read_text())
    checkpoint = tmp_path / "model.safetensors"
    weights = load_file(str(checkpoint))
    key = "paligemma_with_expert.paligemma.input_layernorm.weight"
    raw = torch.tensor(fixture["values"], dtype=torch.float32)[: weights[key].numel()]
    weights[key] = raw
    save_file(weights, str(checkpoint))
    expected = raw.to(torch.bfloat16).float()
    assert not torch.equal(raw, expected)

    # The old construction-only cast leaves non-BF16-representable SFT values intact.
    with torch.random.fork_rng(devices=[]):
        uncorrected = SmallPi05Model(config)
    load_model_weights(uncorrected, Pi05Source.resolve(config))
    torch.testing.assert_close(uncorrected.state_dict()[key], raw, atol=0, rtol=0)
    assert not torch.equal(uncorrected.state_dict()[key], expected)

    monkeypatch.setattr(policy, "Pi05Model", SmallPi05Model)
    loaded = policy.Pi05TrainPolicy(config, device="cpu")
    torch.testing.assert_close(loaded.model.state_dict()[key], expected, atol=0, rtol=0)
    for name, tensor in loaded.model.state_dict().items():
        if not name.startswith("paligemma_with_expert."):
            assert tensor.dtype == torch.float32
            torch.testing.assert_close(tensor, weights[name], atol=0, rtol=0)


@pytest.mark.parametrize("stage", ["constructor", "processor", "provenance", "heads", "materialize"])
def test_failed_policy_construction_releases_model_with_retained_error(tmp_path, monkeypatch, stage):
    config = write_source(tmp_path)
    references = []
    original_heads = policy.Pi05Heads

    def build(*args, **kwargs):
        model = SmallPi05Model(*args, **kwargs)
        references.append(weakref.ref(model))
        references.extend(weakref.ref(parameter) for parameter in model.parameters())
        if stage == "constructor":
            raise ValueError("injected constructor failure")
        if stage == "provenance":
            (tmp_path / "norm_stats.json").write_text((tmp_path / "norm_stats.json").read_text() + "\n")
        return model

    def fail_heads(model):
        heads = original_heads(model)
        references.append(weakref.ref(heads))
        raise ValueError("injected heads failure")

    def fail_materialize(model, **kwargs):
        torch.nn.Module.to(model, **kwargs)
        raise ValueError("injected materialize failure")

    monkeypatch.setattr(policy, "Pi05Model", build)
    if stage == "processor":
        (tmp_path / "norm_stats.json").write_text("{}")
    elif stage == "heads":
        monkeypatch.setattr(policy, "Pi05Heads", fail_heads)
    elif stage == "materialize":
        monkeypatch.setattr(SmallPi05Model, "to", fail_materialize)
    error_type, message = {
        "processor": (KeyError, "norm_stats"),
        "provenance": (RuntimeError, "checkpoint changed while it was being loaded"),
    }.get(stage, (ValueError, f"injected {stage} failure"))
    with pytest.raises(error_type, match=message) as caught:
        policy.Pi05TrainPolicy(config, device="cpu")
    retained = caught.value
    del caught  # Only the retained exception should own its diagnostic/lifetime state.
    assert traceback.extract_tb(retained.__traceback__)
    gc.collect()
    assert references and all(reference() is None for reference in references)

    # Reconstruction follows the same public source/loading/materialization path.
    restored_config = write_source(tmp_path)
    monkeypatch.setattr(policy, "Pi05Model", SmallPi05Model)
    monkeypatch.setattr(policy, "Pi05Heads", original_heads)
    if stage == "materialize":
        monkeypatch.setattr(SmallPi05Model, "to", torch.nn.Module.to)
    rebuilt = policy.Pi05TrainPolicy(restored_config, device="cpu")
    assert rebuilt.net.value_head is rebuilt.model.value_head
