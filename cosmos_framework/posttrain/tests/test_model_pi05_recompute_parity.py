# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qualification-report validation; synthetic reports make no model-parity claim."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.tests import pi05_recompute_parity_cli as probe

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _reports(directory):
    # Independently list the expected measured stages so a missing stage cannot
    # disappear simultaneously from the reports and their verification fixture.
    inputs = [
        *(f"inputs.image.{i}" for i in range(3)),
        "inputs.image_masks",
        "inputs.tokens",
        "inputs.token_mask",
        "inputs.state",
    ]
    prefix = [*(f"prefix.{i}.{part}" for i in range(18) for part in ("key", "value")), "prefix.mask", "prefix.features"]
    score = ["logprobs", "values"]
    rollout = (
        inputs
        + prefix
        + score
        + ["trace.latent_before", "trace.latent_after", "trace.denoise_index", "actions.normalized", "actions.decoded"]
    )
    forward = inputs + prefix + score + ["velocity"]

    def tensors(names):
        return {
            name: {
                "shape": [1],
                "dtype": "torch.float32",
                "stride": [1],
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            for name in names
        }

    common = {
        "completed": True,
        "selection": [{"sample_id": "report-fixture-not-model-evidence"}],
        **{
            key: hashlib.sha256(key.encode()).hexdigest()
            for key in (
                "checkpoint_sha256",
                "tokenizer_sha256",
                "norm_stats_sha256",
                "request_sha256",
                "records_sha256",
            )
        },
    }
    update = {
        "optimizer": {"impl": "fused"},
        "actor_lr": 5e-6,
        "value_lr": 1e-4,
        "loss": 0.25,
        "gradient_norm": 0.5,
        "unused_head_unchanged": True,
        "gradients": tensors(f"parameter.{i}" for i in range(216)),
        "parameters": tensors(f"parameter.{i}" for i in range(216)),
    }
    reports = {
        "baseline": {**common, "b8": tensors(rollout), "b16": tensors(forward), "update": copy.deepcopy(update)},
        "local": {**common, "b8": tensors(rollout), "b16": tensors(forward), "update": copy.deepcopy(update)},
        "reference": {**common, "b16": tensors(forward), "update": copy.deepcopy(update)},
    }
    args = SimpleNamespace(pixel_bounds=None)
    for name, report in reports.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(report))
        setattr(args, name, path)
    return args, reports


def test_qualification_rejects_a_shared_missing_input_with_weakened_guard_control(tmp_path, monkeypatch):
    args, reports = _reports(tmp_path)
    assert probe.compare(args)["passed"]
    for name in ("baseline", "local"):
        reports[name]["b8"].pop("inputs.state")
        getattr(args, name).write_text(json.dumps(reports[name]))
    with pytest.raises(ValueError, match="B8 expected tensor inventory"):
        probe.compare(args)
    expected = probe.expected_forward_keys
    monkeypatch.setattr(
        probe,
        "expected_forward_keys",
        lambda *, rollout: expected(rollout=rollout) - ({"inputs.state"} if rollout else set()),
    )
    assert probe.compare(args)["passed"]  # The former both-sides-match check accepts incomplete evidence.


def test_pixel_comparison_uses_measured_palette_and_rejects_above_bound():
    bounds = json.loads((Path(__file__).parent / "fixtures/pi05_pixel_rounding.json").read_text())
    common = {
        "shape": [1, 3, 224, 224],
        "dtype": "torch.float32",
        "uint8_sha256": hashlib.sha256(bytes(range(256))).hexdigest(),
        "used_values": list(range(256)),
    }
    left = {**common, "palette": bounds["reference_palette"]}
    right = {**common, "palette": bounds["local_palette"]}
    assert probe.bounded_image_difference(left, right, bounds) == bounds["max_abs"]
    changed = copy.deepcopy(right)
    changed["palette"][0] = left["palette"][0] + 2 * bounds["max_abs"]
    assert probe.bounded_image_difference(left, changed, bounds) is None
    # Even a report claiming that foreign palette cannot exceed the independently
    # measured numeric bound. Removing just the final bound check breaks this.
    claimed = {**bounds, "local_palette": changed["palette"]}
    assert probe.bounded_image_difference(left, changed, claimed) is None


def test_compact_baseline_gradient_drift_fails_even_when_reference_matches_local(tmp_path):
    args, reports = _reports(tmp_path)
    reports["baseline"]["update"]["gradients"]["parameter.17"]["sha256"] = "changed"
    args.baseline.write_text(json.dumps(reports["baseline"]))
    result = probe.compare(args)
    assert not result["passed"]
    assert result["checks"]["b16_baseline_gradients"]["nonexact"] == ["parameter.17"]
    assert result["checks"]["b16_gradients"]["nonexact"] == []


def test_recompute_rejects_shared_nonfused_update_configuration(tmp_path):
    args, reports = _reports(tmp_path)
    for name, report in reports.items():
        report["update"]["optimizer"]["impl"] = "foreach"
        getattr(args, name).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="requires fused"):
        probe.compare(args)


@pytest.mark.parametrize("dtype", ["float32", "int64", "bool"])
def test_tensor_fingerprint_preserves_logical_bytes_and_original_layout(dtype):
    import numpy as np
    import torch

    from cosmos_framework.posttrain.tests.helpers.pi05_recompute_parity import fingerprint

    source = np.arange(12).reshape(3, 4).astype(dtype)
    tensor = torch.from_numpy(source).transpose(0, 1)
    result = fingerprint(tensor)
    assert result["sha256"] == hashlib.sha256(source.T.tobytes()).hexdigest()
    assert result["dtype"] == str(tensor.dtype)
    assert result["shape"] == list(source.T.shape)
    assert result["stride"] == [stride // source.itemsize for stride in source.T.strides]
    assert result["stride"] != fingerprint(tensor.contiguous())["stride"]
    assert result["sha256"] == fingerprint(tensor.contiguous())["sha256"]


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_b8_nonfinite_input_identifies_the_tensor(tmp_path, monkeypatch, nonfinite):
    import torch

    from cosmos_framework.posttrain.model.pi05.processing import Pi05Processor
    from cosmos_framework.posttrain.tests.helpers.pi05_loading import write_processor_source
    from cosmos_framework.posttrain.tests.helpers.pi05_parity import prepare_request

    config = write_processor_source(tmp_path)
    processor = Pi05Processor(config)
    request = tmp_path / "b8.npz"
    prepare_request(request, batch=8, seed=42, observations=None)
    materialize = processor.materialize

    def corrupt(*args, **kwargs):
        inputs = materialize(*args, **kwargs)
        inputs.state[0, 0] = nonfinite
        return inputs

    monkeypatch.setattr(processor, "materialize", corrupt)
    policy = SimpleNamespace(processor=processor, model=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="nonfinite") as retained:
        probe.b8_rollout(policy, SimpleNamespace(request=request))
    assert any("inputs.state" in note for note in retained.value.__notes__)


def test_prefix_nonfinite_key_identifies_its_layer():
    import torch

    from cosmos_framework.posttrain.tests.helpers.pi05_recompute_parity import record_prefix

    cache = [(torch.zeros(1), torch.ones(1)) for _ in range(18)]
    cache[-1][0].fill_(float("inf"))
    with pytest.raises(ValueError, match="nonfinite") as retained:
        record_prefix({}, torch.ones(1, dtype=torch.bool), cache, torch.zeros(1))
    assert any(f"prefix.{len(cache) - 1}.key" in note for note in retained.value.__notes__)
