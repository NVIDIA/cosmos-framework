# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed source membership and paired, uneven-shard evaluation reductions."""

import json
from dataclasses import asdict

import pytest

from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LiberoManifest,
    fixed_manifest,
    load_partition,
)
from cosmos_framework.posttrain.tests.helpers.libero_analysis import episode_metrics, paired_comparison

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_fixed_partitions_reject_overlap_omissions_and_duplicates(tmp_path):
    train, evaluation = fixed_manifest("train"), fixed_manifest("eval")
    train.write(tmp_path / "train.json")
    evaluation.write(tmp_path / "eval.json")
    assert load_partition(str(tmp_path / "train.json"), str(tmp_path / "eval.json")) == (train, evaluation)
    assert len(train.cases) == 400 and len(evaluation.cases) == 100
    for cases in (train.cases[:-1], train.cases + train.cases[:1], (evaluation.cases[0], *train.cases[1:])):
        with pytest.raises(ValueError):
            LiberoManifest("train", cases)
    payload = train.payload()
    payload["cases"][-1]["extra"] = 1
    (tmp_path / "train.json").write_text(json.dumps(payload))
    with pytest.raises(TypeError):
        load_partition(str(tmp_path / "train.json"), str(tmp_path / "eval.json"))
    payload = train.payload()
    payload["schema_version"] = True
    (tmp_path / "train.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        LiberoManifest.read(tmp_path / "train.json")


def test_uneven_evaluation_shards_use_episode_counts_and_paired_cases():
    manifest = fixed_manifest("eval")
    episodes = [
        {
            **asdict(case),
            "success": case.init_state_id < case.task_id,
            "steps": 256,
            "return": float(case.init_state_id < case.task_id),
        }
        for case in manifest.cases
    ]
    shards = [episodes[rank::8] for rank in range(8)]
    merged = [episode for shard in shards for episode in shard]
    metrics = episode_metrics(merged, manifest)
    assert metrics["val/success_rate"] == 0.45
    assert metrics["val/task_9/success_rate"] == 0.9
    same = paired_comparison(episodes, list(reversed(episodes)))
    assert same["success_rate_delta"] == 0 and same["paired_ci95"] == [0, 0]
    perfect = [{**row, "success": True, "return": 1.0} for row in episodes]
    improvement = paired_comparison(episodes, perfect)
    assert improvement["success_rate_delta"] == 0.55 and improvement["paired_ci95"][0] > 0
    with pytest.raises(ValueError, match="inventory"):
        episode_metrics(merged[:-1], manifest)
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(episodes, perfect[:-1])


def test_explicit_trial_manifest_preserves_subsets_seeds_and_order(tmp_path):
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LiberoCase, LiberoTrial, LiberoTrialManifest

    source = LiberoTrialManifest(
        "libero_spatial",
        (
            LiberoTrial(LiberoCase(7, 31, "libero_spatial"), 9),
            LiberoTrial(LiberoCase(1, 4, "libero_spatial"), 0),
        ),
    )
    path = tmp_path / "trials.json"
    source.write(path)
    assert LiberoTrialManifest.read(path) == source
    assert LiberoTrialManifest(source.suite, tuple(reversed(source.trials))).digest != source.digest
    with pytest.raises(ValueError, match="duplicate"):
        LiberoTrialManifest(source.suite, source.trials + source.trials[:1])
    with pytest.raises(ValueError, match="nonempty"):
        LiberoTrialManifest(source.suite, ())
    payload = source.payload()
    payload["trials"][-1]["seed"] = -1
    with pytest.raises(ValueError, match="seed"):
        LiberoTrialManifest.from_payload(payload)


@pytest.mark.parametrize("rewards,steps", [((float("nan"),), 1), ((True,), 1), ((0.1,), 2), ((), 0)])
def test_environment_feedback_rejects_invalid_control_facts(rewards: tuple[float, ...], steps: int) -> None:
    from cosmos_framework.posttrain.schema import EnvironmentFeedback

    with pytest.raises(ValueError, match="environment feedback"):
        EnvironmentFeedback(rewards, steps)


def test_unified_libero_runtime_is_the_only_environment_and_stale_knobs_are_rejected() -> None:
    from omegaconf import OmegaConf
    from omegaconf.errors import ConfigAttributeError

    from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LIBERO_RUNTIME
    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoSimulator

    expected = LIBERO_RUNTIME.identity()
    assert {name: expected[name] for name in ("mujoco", "robosuite", "numpy")} == {
        "mujoco": "3.3.7",
        "robosuite": "1.4.1",
        "numpy": "1.26.4",
    }
    assert len(expected["requirements_sha256"]) == 64
    for config in (LiberoConfig(), LiberoConfig(suite="libero_10", execution_horizon=8, max_episode_steps=720)):
        assert config.identity()["environment"] == expected
        assert "environment" not in config.simulator_options
    # The retired rlinf/nvidia selector fails at every boundary instead of being ignored.
    with pytest.raises(TypeError):
        LiberoConfig(environment="nvidia")
    with pytest.raises(AttributeError):
        LiberoConfig().environment = "nvidia"
    with pytest.raises(ConfigAttributeError):
        OmegaConf.structured(LiberoConfig()).environment = "nvidia"
    with pytest.raises(TypeError):
        LiberoSimulator(embodiment="libero_sim", n_lanes=1, environment="nvidia")
