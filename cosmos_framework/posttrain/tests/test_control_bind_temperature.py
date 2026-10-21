# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU-less objective-specific temperature/distribution contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.config.rl import GRPOConfig, OPDConfig, RolloutConfig, TeacherConfig
from cosmos_framework.posttrain.control.temperature import (
    check_temperature,
    grpo_temperature_contract,
    opd_temperature_contract,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_grpo_sampling_and_policy_temperature_mismatch_is_rejected() -> None:
    assert check_temperature(grpo_temperature_contract(GRPOConfig(), RolloutConfig())).ok

    mismatch = check_temperature(grpo_temperature_contract(GRPOConfig(temperature=0.9), RolloutConfig(temperature=1.0)))
    assert not mismatch.ok
    assert "rollout.temperature == rl.temperature" in mismatch.errors[0]

    for rollout, expected in (
        (RolloutConfig(top_p=0.9), "top_p=1.0"),
        (RolloutConfig(top_k=8), "top_k disabled"),
        (RolloutConfig(repetition_penalty=1.1), "repetition_penalty=1.0"),
    ):
        checked = check_temperature(grpo_temperature_contract(GRPOConfig(), rollout))
        assert any(expected in error for error in checked.errors)

    with pytest.raises(TypeError, match="rollout.temperature"):
        grpo_temperature_contract(GRPOConfig(), SimpleNamespace(**{**vars(RolloutConfig()), "temperature": True}))

    ignored_rollout_logprobs = check_temperature(
        grpo_temperature_contract(
            SimpleNamespace(
                **{
                    **vars(GRPOConfig(grpo_surrogate="legacy_pg")),
                    "use_rollout_logprobs_for_loss": True,
                }
            ),
            RolloutConfig(),
        )
    )
    assert any("legacy GRPO does not consume" in error for error in ignored_rollout_logprobs.errors)

    raw_rollout_logprobs = check_temperature(
        grpo_temperature_contract(
            GRPOConfig(grpo_surrogate="clipped", temperature=0.9, use_rollout_logprobs_for_loss=True),
            RolloutConfig(temperature=0.9),
        )
    )
    assert any("raw, untempered logprobs" in error for error in raw_rollout_logprobs.errors)


def test_opd_temperature_contract_is_objective_specific() -> None:
    direct = OPDConfig(distillation_mode="direct")
    with pytest.raises(ValueError, match="finite and positive"):
        opd_temperature_contract(
            direct,
            RolloutConfig(temperature=0.0, top_p=0.8, top_k=8, repetition_penalty=1.1),
        )
    assert check_temperature(
        opd_temperature_contract(
            direct,
            RolloutConfig(temperature=0.5, top_p=0.8, top_k=8, repetition_penalty=1.1),
        )
    ).ok

    teacher_mismatch = OPDConfig(
        distillation_mode="direct",
        teacher=TeacherConfig(scoring_temperature=0.5),
    )
    checked = check_temperature(opd_temperature_contract(teacher_mismatch, RolloutConfig()))
    assert not checked.ok
    assert checked.errors == (
        "top-k OPD requires teacher.scoring_temperature == rl.temperature == 1.0; "
        "got teacher=0.5, policy=1; the current forced-sequence teacher path returns untempered prompt logprobs",
    )

    matched_nonunit = OPDConfig(
        distillation_mode="direct",
        temperature=0.5,
        teacher=TeacherConfig(scoring_temperature=0.5),
    )
    checked = check_temperature(opd_temperature_contract(matched_nonunit, RolloutConfig()))
    assert checked.errors == (
        "top-k OPD requires teacher.scoring_temperature == rl.temperature == 1.0; "
        "got teacher=0.5, policy=0.5; the current forced-sequence teacher path returns untempered prompt logprobs",
    )

    advantage = OPDConfig(distillation_mode="advantage")
    checked = check_temperature(opd_temperature_contract(advantage, RolloutConfig(temperature=0.5)))
    assert any("advantage OPD requires rollout.temperature" in error for error in checked.errors)

    dpca = OPDConfig(
        distillation_mode="advantage",
        jsd_beta=None,
        teacher=TeacherConfig(input_mode="cross_token_text", signal_mode="dpca", top_k=0),
    )
    assert check_temperature(opd_temperature_contract(dpca, RolloutConfig())).ok
    checked = check_temperature(opd_temperature_contract(dpca, RolloutConfig(top_k=4)))
    assert any("DPCA requires rollout.top_k disabled" in error for error in checked.errors)


def test_vllm_subminimum_temperature_is_rejected_before_backend_rounding() -> None:
    with pytest.raises(ValueError, match=r"rewrites values in \(0, 0\.01\) to 0\.01"):
        grpo_temperature_contract(
            GRPOConfig(grpo_surrogate="clipped", temperature=0.005),
            RolloutConfig(temperature=0.005),
        )
