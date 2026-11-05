# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: fixed-schema reward-provider diagnostic reporting."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.roles.reward.diagnostics import (
    ProviderDiagnosticRecord,
    provider_diagnostics_report,
)
from cosmos_framework.posttrain.schema.metrics import MetricContribution, finalize_metric_report


@pytest.mark.L0
@pytest.mark.CPU
def test_provider_diagnostic_keys_encode_provider_and_diagnostic_as_distinct_path_segments() -> None:
    metrics, provenance = provider_diagnostics_report(
        [
            ProviderDiagnosticRecord(
                provider_name="a_b",
                provider_revision="r1",
                diagnostics={"c": MetricContribution.sum(2)},
            ),
            ProviderDiagnosticRecord(
                provider_name="a",
                provider_revision="r2",
                diagnostics={
                    "b_c": MetricContribution.sum(3),
                    "b_sample_count": MetricContribution.sum(4),
                },
            ),
        ],
        namespace="train",
    )

    assert finalize_metric_report(metrics) == {
        "train/reward_provider/a_b/c": 2,
        "train/reward_provider/a_b/sample_count": 1,
        "train/reward_provider/a/b_c": 3,
        "train/reward_provider/a/b_sample_count": 4,
        "train/reward_provider/a/sample_count": 1,
    }
    assert provenance == (("a", "r2"), ("a_b", "r1"))
