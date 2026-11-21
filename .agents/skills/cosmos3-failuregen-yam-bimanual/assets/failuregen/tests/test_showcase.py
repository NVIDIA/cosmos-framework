# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fixed example selection, seed routing, and immutable prepared inputs."""

import json
from pathlib import Path

import numpy as np
import pytest
from failuregen import showcase
from failuregen.plan import load
from failuregen.yam_training.contract import EEF_CONTRACT, file_sha256
from PIL import Image


@pytest.fixture
def selection():
    families = json.loads((Path(__file__).parents[1] / "configs/showcase.json").read_text())["families"]
    episodes = []
    for f in families:
        for ep in (f["donor_episode"], f["example"]["host_episode"]):
            episodes.append(
                dict(
                    episode=ep,
                    split="train",
                    outcome="success_clean",
                    valid_starts=list(range(80)),
                    states_sha256="test",
                )
            )
    return families, dict(episodes=episodes)


def test_exact_windows_are_required_without_substitution(selection):
    families, manifest = selection
    selected = showcase.select_examples(manifest, families)
    assert [(d["episode_index"], w["host"]["episode"], w["host"]["start"]) for _, d, w in selected] == [
        (42, 79, 10.8),
        (40, 43, 4.4),
        (61, 46, 7.6),
    ]
    next(ep for ep in manifest["episodes"] if ep["episode"] == 79)["valid_starts"].remove(54)
    with pytest.raises(ValueError, match="Requested donor/host window unavailable"):
        showcase.select_examples(manifest, families)


def test_unclean_host_is_rejected(selection):
    families, manifest = selection
    next(ep for ep in manifest["episodes"] if ep["episode"] == 43)["outcome"] = "failed_pickup"
    with pytest.raises(ValueError, match="clean training host"):
        showcase.select_examples(manifest, families)


def test_invalid_seed_is_rejected(selection):
    families, manifest = selection
    families[0]["example"]["seed"] = True
    with pytest.raises(ValueError, match="nonnegative integers"):
        showcase.select_examples(manifest, families)


@pytest.fixture
def bundle(tmp_path, selection, monkeypatch):
    families, _ = selection
    prepared, dataset, run_dir, checkpoint = [tmp_path / n for n in ("prepared", "dataset", "run", "checkpoint")]
    for p in (prepared, dataset, run_dir, checkpoint / "model"):
        p.mkdir(parents=True)
    (checkpoint / "model/.metadata").write_text("test checkpoint")
    stats = dict(contract=EEF_CONTRACT, q01=[-1.0] * 20, q99=[2.0] * 20)
    (prepared / "action_stats.json").write_text(json.dumps(stats))
    episodes = []
    for ep in (42, 79, 40, 43, 61, 46):
        actions = np.zeros((100, 20), dtype=np.float32)
        actions[:, [3, 7, 13, 17]] = 1
        actions[:, 0] = ep / 1000
        state = prepared / f"{ep}.npz"
        np.savez(state, actions_raw=actions, observations_eef_20d=actions)
        episodes.append(
            dict(
                episode=ep,
                states=state.name,
                states_sha256=file_sha256(state),
                caption="Move the reel.",
                split="train",
                outcome="success_clean",
                valid_starts=list(range(80)),
            )
        )
    manifest = dict(
        status="prepared",
        contract=EEF_CONTRACT,
        episodes=episodes,
        stats_sha256=file_sha256(prepared / "action_stats.json"),
        annotations="failuregen_annotations.json",
        annotations_sha256="historical-annotation-hash",
    )
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "yam_training_contract.json").write_text(
        json.dumps(
            dict(
                contract=EEF_CONTRACT,
                stats_sha256=manifest["stats_sha256"],
                prepared_manifest_sha256=file_sha256(prepared / "manifest.json"),
            )
        )
    )
    (run_dir / "config.yaml").write_text("model:\n  config:\n    num_embodiment_domains: 32\n")
    recipes = tmp_path / "recipes.json"
    recipes.write_text(json.dumps(dict(version=1, families=families)))

    def replay(prepared, dataset, directory, episode, start, **kwargs):
        directory.mkdir()
        Image.new("RGB", (960, 960), (episode, 0, 0)).save(directory / "obs.png")

    monkeypatch.setattr(showcase, "prepare_replay", replay)
    return dict(
        prepared=prepared,
        dataset=dataset,
        run_dir=run_dir,
        checkpoint=checkpoint,
        recipes=recipes,
        output=tmp_path / "showcase",
    )


def test_three_runs_share_one_seed_and_rerun_checks_inputs(bundle):
    assert not (bundle["dataset"] / "failuregen_annotations.json").exists()
    result = showcase.prepare(**bundle)
    assert result["n_clips"] == 3
    output = bundle["output"]
    assert [(j["seed"], len(load(output / j["manifest"])[0])) for j in result["jobs"]] == [(0, 3)]
    for example in result["examples"]:
        assert example["video"] == f"seeds/seed{example['seed']}/runs/{example['name']}/synth.mp4"
    assert showcase.prepare(**bundle) == result
    actions = output / result["examples"][0]["action_path"]
    actions.write_text("[]")
    with pytest.raises(ValueError, match="Showcase input changed"):
        showcase.prepare(**bundle)


def test_incomplete_preparation_and_changed_checkpoint_are_rejected(bundle):
    bundle["output"].mkdir()
    with pytest.raises(ValueError, match="preparation is incomplete"):
        showcase.prepare(**bundle)
    bundle["output"].rmdir()
    showcase.prepare(**bundle)
    (bundle["checkpoint"] / "model/.metadata").write_text("different checkpoint")
    with pytest.raises(ValueError, match="Showcase inputs changed"):
        showcase.prepare(**bundle)


def test_fixed_examples_still_require_matching_prepared_bundle(bundle):
    path = bundle["prepared"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["annotations_sha256"] = "changed preparation metadata"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from the training run"):
        showcase.prepare(**bundle)


def test_fixed_examples_still_check_state_hashes(bundle):
    (bundle["prepared"] / "42.npz").write_bytes(b"changed states")
    with pytest.raises(ValueError, match="Prepared states changed"):
        showcase.prepare(**bundle)
