# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for qualified numerical-golden profiles."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cosmos_framework.posttrain.tests.helpers.qualification import GoldenProfile, QualificationMismatch

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_BASE_PROFILE = GoldenProfile(
    name="cosmos3-dpo-gb200",
    algorithm="dpo",
    accelerator="B200",
    torch_version="2.9.0a0+capture",
    world_size=4,
    steps=2,
    seed=0,
    rtol=1e-5,
    atol=1e-6,
    reference=(0.7, 0.4),
)


def _profile(**overrides: object) -> GoldenProfile:
    return replace(_BASE_PROFILE, **overrides)


def test_golden_profile_accepts_only_its_capture_runtime() -> None:
    profile = _profile()

    profile.require_runtime(torch_version="2.9.0a0+capture", world_size=4, accelerator="B200")
    with pytest.raises(QualificationMismatch, match="re-capture"):
        profile.require_runtime(torch_version="2.10.0", world_size=4, accelerator="B200")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reference": (0.7,)}, "configured steps"),
        ({"world_size": 0}, "world_size"),
        ({"steps": 0, "reference": ()}, "steps"),
        ({"rtol": float("nan")}, "rtol"),
        ({"atol": -1.0}, "atol"),
        ({"reference": (0.7, float("inf"))}, "finite"),
    ],
)
def test_golden_profile_rejects_incomplete_qualification(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _profile(**overrides)
