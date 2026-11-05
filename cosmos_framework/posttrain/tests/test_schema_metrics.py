# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: typed metric reduction and exact mergeable moments."""

from __future__ import annotations

import math
import pickle
import subprocess
import sys

import pytest

from cosmos_framework.posttrain.schema.metrics import (
    MetricContribution,
    MomentOutputs,
    Moments,
    finalize_metric_report,
    merge_metric_reports,
    validate_public_metric_keys,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_moments_merge_is_partition_invariant_and_population_based() -> None:
    values = [0.0, 1.0, 10.0]
    whole = Moments.from_values(values)
    partitioned = Moments.merge_many(
        [Moments.from_values(values[:2]), Moments.empty(), Moments.from_values(values[2:])]
    )
    nested = Moments.from_values(values[:1]).merge(
        Moments.from_values(values[1:2]).merge(Moments.from_values(values[2:]))
    )

    assert partitioned.count == nested.count
    assert partitioned.mean == pytest.approx(nested.mean)
    assert partitioned.m2 == pytest.approx(nested.m2)
    assert partitioned.minimum == nested.minimum
    assert partitioned.maximum == nested.maximum
    assert partitioned.count == whole.count == 3
    assert partitioned.mean == pytest.approx(11.0 / 3.0)
    assert partitioned.population_std == pytest.approx(math.sqrt(182.0 / 9.0))
    assert (partitioned.minimum, partitioned.maximum) == (0.0, 10.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_moments_are_stable_for_large_offsets() -> None:
    observations = [1.0e12 + 1.0, 1.0e12 + 2.0, 1.0e12 + 3.0]
    moments = Moments.merge_many([Moments.from_values(observations[:1]), Moments.from_values(observations[1:])])
    assert moments.mean == pytest.approx(1.0e12 + 2.0)
    assert moments.population_std == pytest.approx(math.sqrt(2.0 / 3.0))


@pytest.mark.L0
@pytest.mark.CPU
def test_weighted_mean_uses_global_numerator_and_denominator() -> None:
    report = merge_metric_reports(
        [
            {"train/loss_avg": MetricContribution.weighted_mean(2.0, weight=1.0)},
            {"train/loss_avg": MetricContribution.weighted_mean(5.0, weight=4.0)},
            {"train/loss_avg": MetricContribution.weighted_mean(0.0, weight=0.0)},
        ]
    )
    assert finalize_metric_report(report) == {"train/loss_avg": pytest.approx(22.0 / 5.0)}

    with pytest.raises(ValueError, match="no observations"):
        finalize_metric_report({"val/loss_avg": MetricContribution.weighted_mean(0.0, weight=0.0)})


@pytest.mark.L0
@pytest.mark.CPU
def test_scalar_reductions_preserve_count_integer_and_optional_key_absence() -> None:
    left = {
        "train/valid_token_count": MetricContribution.sum(2),
        "train/loss_max": MetricContribution.maximum(-1.0),
    }
    right = {
        "train/valid_token_count": MetricContribution.sum(3),
        "train/loss_max": MetricContribution.maximum(4.0),
        "train/kl_loss_max": MetricContribution.maximum(0.2),
    }
    merged = merge_metric_reports([left, right])
    finalized = finalize_metric_report(merged)

    assert finalized == {
        "train/valid_token_count": 5,
        "train/loss_max": 4.0,
        "train/kl_loss_max": 0.2,
    }
    assert isinstance(finalized["train/valid_token_count"], int)
    assert "train/missing_metric" not in finalized
    assert left["train/valid_token_count"].payload == 2  # input report was not mutated


@pytest.mark.L0
@pytest.mark.CPU
def test_identical_compares_type_and_value() -> None:
    assert finalize_metric_report(
        merge_metric_reports(
            [
                {"val/step": MetricContribution.identical(3)},
                {"val/step": MetricContribution.identical(3)},
            ]
        )
    ) == {"val/step": 3}

    with pytest.raises(ValueError, match="must be identical"):
        merge_metric_reports(
            [
                {"val/step": MetricContribution.identical(True)},
                {"val/step": MetricContribution.identical(1)},
            ]
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_moments_finalize_to_explicit_train_and_val_names() -> None:
    reward = MetricContribution.moments(Moments.from_values([0.0, 1.0, 1.0]))
    train = finalize_metric_report(
        {"train/reward": reward},
        moment_outputs={
            "train/reward": MomentOutputs(
                count="train/rollout_count",
                mean="train/reward_mean",
                std="train/reward_std",
                minimum="train/reward_min",
                maximum="train/reward_max",
            )
        },
    )
    val = finalize_metric_report(
        {"val/reward": reward},
        moment_outputs={
            "val/reward": MomentOutputs(
                count="val/rollout_count",
                mean="val/reward_avg",
                std="val/reward_std",
                minimum="val/reward_min",
                maximum="val/reward_max",
            )
        },
    )

    assert train["train/reward_mean"] == val["val/reward_avg"] == pytest.approx(2.0 / 3.0)
    assert train["train/reward_std"] == val["val/reward_std"] == pytest.approx(math.sqrt(2.0) / 3.0)
    assert train["train/rollout_count"] == val["val/rollout_count"] == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_two_level_merge_matches_one_shot_unique_shards() -> None:
    shards = [
        {
            "train/valid_token_count": MetricContribution.sum(1),
            "train/reward": MetricContribution.moments(Moments.from_values([0.0])),
        },
        {
            "train/valid_token_count": MetricContribution.sum(2),
            "train/reward": MetricContribution.moments(Moments.from_values([1.0, 2.0])),
        },
        {
            "train/valid_token_count": MetricContribution.sum(3),
            "train/reward": MetricContribution.moments(Moments.from_values([3.0])),
        },
    ]
    one_shot = merge_metric_reports(shards)
    # Rank-level reduction remains typed and unfinalized; replica-level reduction merges unique
    # rank aggregates exactly once. This catches the duplicate-global-leader integration hazard.
    two_level = merge_metric_reports([merge_metric_reports(shards[:2]), merge_metric_reports(shards[2:])])
    assert two_level == one_shot


@pytest.mark.L0
@pytest.mark.CPU
def test_invalid_metric_schema_fails_loudly() -> None:
    with pytest.raises(TypeError, match="MetricReduction"):
        MetricContribution(reduction="sum", payload=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="changed reduction"):
        merge_metric_reports(
            [
                {"train/example": MetricContribution.sum(1)},
                {"train/example": MetricContribution.maximum(2)},
            ]
        )
    with pytest.raises(TypeError, match="MOMENTS requires"):
        MetricContribution(reduction=MetricContribution.moments(Moments()).reduction, payload=1.0)
    with pytest.raises(ValueError, match="m2 must be >= 0"):
        Moments(count=1, mean=0.0, m2=-1.0, minimum=0.0, maximum=0.0)
    with pytest.raises(ValueError, match="legacy metric key"):
        validate_public_metric_keys({"grpo_loss": 1.0})
    with pytest.raises(ValueError, match="duplicate keys"):
        finalize_metric_report(
            {
                "train/reward": MetricContribution.moments(Moments.from_values([1.0])),
                "train/reward_mean": MetricContribution.identical(1.0),
            },
            moment_outputs={"train/reward": MomentOutputs(mean="train/reward_mean")},
        )
    with pytest.raises(TypeError, match="sum payload must be a real number"):
        MetricContribution.sum(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no observations"):
        MomentOutputs(mean="val/reward_avg").finalize(Moments())

    malformed_sum = MetricContribution.sum(1)
    object.__setattr__(malformed_sum, "payload", True)
    with pytest.raises(TypeError, match="sum payload must be a real number"):
        merge_metric_reports([{"train/count": malformed_sum}, {"train/count": malformed_sum}])

    malformed_scalar = MetricContribution.identical("ok")
    object.__setattr__(malformed_scalar, "payload", object())
    with pytest.raises(TypeError, match="must be an int, float, bool, or str"):
        finalize_metric_report({"val/status": malformed_scalar})

    malformed_moments = Moments.from_values([1.0])
    object.__setattr__(malformed_moments, "minimum", None)
    with pytest.raises(ValueError, match="require minimum and maximum"):
        MomentOutputs(mean="val/reward_avg").finalize(malformed_moments)


@pytest.mark.L0
@pytest.mark.CPU
def test_metric_schema_is_dependency_light_and_pickleable() -> None:
    payload = MetricContribution.moments(Moments.from_values([1.0, 2.0]))
    assert pickle.loads(pickle.dumps(payload)) == payload

    code = (
        "import sys; import cosmos_framework.posttrain.schema.metrics; "
        "assert all(name not in sys.modules for name in ('torch', 'ray', 'numpy'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
