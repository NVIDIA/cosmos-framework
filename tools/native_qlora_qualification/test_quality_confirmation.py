# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import copy

import pytest

from tools.native_qlora_qualification.common import inventory_groups
from tools.native_qlora_qualification.quality_confirmation import TRAINING_SEEDS, analyze_confirmation
from tools.native_qlora_qualification.test_quality import fixtures


def data():
    arms, inventory = fixtures()
    trained = {seed: {arm: copy.deepcopy(arms["trained_" + arm]) for arm in ("nf4", "bf16")} for seed in TRAINING_SEEDS}
    frozen = {arm: copy.deepcopy(arms["frozen_" + arm]) for arm in ("nf4", "bf16")}
    for row in frozen["nf4"]:
        row.update(
            candidate_logits={"A": 2.0, "B": 1.0}, restricted_choice_prediction="A", restricted_choice_correct=False
        )
    return trained, frozen, inventory_groups(inventory)


def test_fixed_effect_paired_weighted_bounds_and_no_release_claim():
    trained, frozen, groups = data()
    for seed in TRAINING_SEEDS:
        for row in trained[seed]["nf4"]:
            row["native_sft_ce"] *= 1.01
    result = analyze_confirmation(trained, frozen, groups, split_role="development")
    assert result["conditional_statistical_criteria_passed"] is True
    assert result["release_qualified"] is False
    assert result["artifact_validation_required_separately"] is True
    for name in ("hierarchical_primary", "crossed_group_sensitivity"):
        stats = result[name]
        assert stats["repetitions"] == 10000
        assert stats["bounds"]["relative_native_token_nll"]["interval_95"] == pytest.approx([0.01, 0.01])
        assert stats["bounds"]["accuracy_loss"]["interval_95"] == [0.0, 0.0]
        assert stats["bounds"]["gain_over_frozen_nf4"]["interval_95"] == [1.0, 1.0]


def test_preserves_negative_nll_result_and_legitimate_zero_adaptation_bound():
    trained, frozen, groups = data()
    for seed in TRAINING_SEEDS:
        for row in trained[seed]["nf4"]:
            row["native_sft_ce"] *= 1.0418
    frozen["nf4"] = copy.deepcopy(trained[17]["nf4"])
    result = analyze_confirmation(trained, frozen, groups, split_role="regression", repetitions=100)
    for name in ("hierarchical_primary", "crossed_group_sensitivity"):
        assert result[name]["criteria_passed"] == {
            "native_token_nll": False,
            "accuracy_preservation": True,
            "useful_adaptation": False,
        }
    assert result["conditional_statistical_criteria_passed"] is False


def test_seed_and_group_order_determinism_and_two_distinct_resampling_models():
    trained, frozen, groups = data()
    for seed in TRAINING_SEEDS:
        for i, row in enumerate(trained[seed]["nf4"]):
            row["native_sft_ce"] *= 1 + (seed / 123) * (i / 10)
    result = analyze_confirmation(trained, frozen, groups, split_role="regression", repetitions=200)
    reverse = {seed: trained[seed] for seed in reversed(TRAINING_SEEDS)}
    assert result == analyze_confirmation(
        reverse, frozen, dict(reversed(list(groups.items()))), split_role="regression", repetitions=200
    )
    assert result["hierarchical_primary"]["bounds"] != result["crossed_group_sensitivity"]["bounds"]
    assert result["conditional_statistical_criteria_passed"] is False


def test_token_weighted_pool_does_not_average_row_losses_or_seed_ratios():
    trained, frozen, groups = data()
    for seed in TRAINING_SEEDS:
        for i, row in enumerate(trained[seed]["nf4"]):
            row["native_sft_ce"] = 1.0 if i % 2 else 10.0
        for row in trained[seed]["bf16"]:
            row["native_sft_ce"] = float(seed)
    result = analyze_confirmation(trained, frozen, groups, split_role="development", repetitions=50)
    # Alternating token counts1,2 give weighted NF4 NLL4 rather than row mean5.5.
    assert result["pooled_trained"]["nf4"]["token_weighted_ce"] == 4.0
    expected = 4 / ((17 + 42 + 123) / 3) - 1
    assert result["point_estimates"]["relative_native_token_nll"] == pytest.approx(expected)


@pytest.mark.parametrize("damage", ["seed", "missing_arm", "frozen", "mask", "order", "group", "nan", "tokens"])
def test_incomplete_or_unpaired_inputs_fail(damage):
    trained, frozen, groups = data()
    if damage == "seed":
        trained[18] = trained.pop(17)
    elif damage == "missing_arm":
        trained[42].pop("bf16")
    elif damage == "frozen":
        frozen.pop("bf16")
    elif damage == "mask":
        trained[123]["nf4"][0]["inputs"]["labels"] = "different"
    elif damage == "order":
        trained[42]["bf16"].reverse()
    elif damage == "group":
        groups.clear()
    elif damage == "nan":
        trained[17]["nf4"][0]["native_sft_ce"] = float("nan")
    else:
        trained[42]["nf4"][0]["native_supervised_tokens"] += 1
    with pytest.raises(ValueError):
        analyze_confirmation(trained, frozen, groups, split_role="development", repetitions=10)


def test_one_untouched_group_never_produces_an_interval_or_pass():
    trained, frozen, groups = data()
    groups = dict.fromkeys(groups, "one_group")
    result = analyze_confirmation(trained, frozen, groups, split_role="untouched_diagnostic", repetitions=10000)
    assert result["conditional_statistical_criteria_passed"] is False
    for name in ("hierarchical_primary", "crossed_group_sensitivity"):
        assert result[name]["estimable"] is False
        assert "bounds" not in result[name]


def test_zero_reference_nll_is_an_explicit_undefined_contrast():
    trained, frozen, groups = data()
    for seed in TRAINING_SEEDS:
        for row in trained[seed]["bf16"]:
            row["native_sft_ce"] = 0.0
    with pytest.raises(ValueError, match="positive reference NLL"):
        analyze_confirmation(trained, frozen, groups, split_role="regression", repetitions=10)


@pytest.mark.parametrize("repetitions", [0, -1, 2.0, True])
def test_invalid_bootstrap_controls_fail(repetitions):
    trained, frozen, groups = data()
    with pytest.raises(ValueError):
        analyze_confirmation(trained, frozen, groups, split_role="regression", repetitions=repetitions)
