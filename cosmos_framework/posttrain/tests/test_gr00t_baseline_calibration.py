# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Calibration protocol identity and BF16 Euler negative controls."""

import json

import pytest
import torch

from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LiberoManifest,
    LiberoTrial,
    LiberoTrialManifest,
    fixed_manifest,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_native_buffer_cast_matches_module_to_and_restores_aliases_on_failure() -> None:
    from copy import deepcopy

    from cosmos_framework.posttrain.model.gr00t.ppo import native_ode_backbone_buffers

    model = torch.nn.Module()
    model.register_buffer("inv_freq", 10000.0 ** (-torch.arange(0, 32, 2).float() / 32), persistent=False)
    model.original_inv_freq = model.inv_freq
    original = model.inv_freq
    native = deepcopy(model).to(torch.bfloat16)
    positions = torch.arange(128).float()[:, None]
    expected = (positions * native.inv_freq.float()).cos()
    with pytest.raises(AssertionError):
        torch.testing.assert_close((positions * original).cos(), expected, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="injected") as _retained:
        with native_ode_backbone_buffers(model):
            torch.testing.assert_close((positions * model.inv_freq.float()).cos(), expected, atol=0, rtol=0)
            assert model.original_inv_freq is original and model.inv_freq.dtype == torch.bfloat16
            raise RuntimeError("injected inference failure")
    assert model.inv_freq is original and model.original_inv_freq is original


def test_native_euler_bf16_order_rejects_the_legacy_decomposition() -> None:
    from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOCondition
    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import LinearVelocityHeads

    head = LinearVelocityHeads().to(torch.bfloat16)
    condition = Gr00tPPOCondition(
        torch.linspace(-1, 1, 8, dtype=torch.bfloat16).reshape(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 1, 132, dtype=torch.bfloat16),
        torch.zeros(1, dtype=torch.long),
    )
    rng = torch.get_rng_state().clone()
    expected = torch.randn((1, 40, 132), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1234))
    for _ in range(4):
        velocity = head.velocity(condition.backbone_features.mean(dim=1))[:, None] + expected * 0.1
        expected = expected + 0.25 * velocity * torch.ones_like(expected)
    actual = head.ode_actions(condition, seed=1234, native=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(head.ode_actions(condition, seed=1234), expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_aligned_manifest_is_complete_and_legacy_payloads_stay_readable(tmp_path):
    for suite, schema in (("libero_spatial", 1), ("libero_10", 2)):
        old = fixed_manifest("eval", suite)
        assert old.payload()["schema_version"] == schema and "profile" not in old.payload()
        path = tmp_path / f"{suite}.json"
        old.write(path)
        assert LiberoManifest.read(path).digest == old.digest
    aligned = fixed_manifest("eval", "libero_10", "aligned_200")
    explicit = LiberoTrialManifest("libero_10", tuple(LiberoTrial(case, 0) for case in aligned.cases))
    assert len(aligned.cases) == len(explicit.trials) == 200
    assert len(fixed_manifest("train", "libero_10").cases) == 500
    path = tmp_path / "aligned.json"
    aligned.write(path)
    assert LiberoManifest.read(path) == aligned
    for cases in (aligned.payload()["cases"][:-1], [*aligned.payload()["cases"], aligned.payload()["cases"][0]]):
        path.write_text(json.dumps({**aligned.payload(), "cases": cases}))
        with pytest.raises(ValueError):
            LiberoManifest.read(path)


def test_capture_restores_owned_and_inherited_attributes_after_partial_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from cosmos_framework.posttrain.tests.helpers.gr00t_diagnostics import Gr00tForwardCapture

    class Processor:
        def __init__(self):
            self._collator = torch.nn.Identity()

        @property
        def collator(self):
            return self._collator

        def decode_action(self, value):
            return value * 2

    processor = Processor()
    original = vars(processor).copy()
    backbone = torch.nn.Linear(2, 2)
    encoder, decoder = torch.nn.Identity(), torch.nn.Identity()
    model = SimpleNamespace(
        backbone=backbone, action_head=SimpleNamespace(action_encoder=encoder, action_decoder=decoder)
    )

    def reject_registration(*args, **kwargs):
        raise RuntimeError("injected decoder hook failure")

    monkeypatch.setattr(decoder, "register_forward_hook", reject_registration)
    # Retain the exception after two hooks and both attribute swaps have succeeded.
    with pytest.raises(RuntimeError, match="decoder hook") as _retained:
        with Gr00tForwardCapture(model, processor, "_collator", processor):
            pytest.fail("partial capture setup unexpectedly succeeded")
    assert vars(processor) == original and not backbone._forward_hooks and not encoder._forward_pre_hooks
    # The public collator is read-only in the pinned processor; never leave a failed replacement installed.
    with pytest.raises(AttributeError):
        with Gr00tForwardCapture(model, processor, "collator", processor):
            pytest.fail("read-only collator was mutated")
    assert vars(processor) == original
