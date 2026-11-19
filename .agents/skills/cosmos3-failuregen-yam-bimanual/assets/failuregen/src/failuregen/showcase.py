# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare the three fixed donor/host examples without loading Cosmos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .config import Profile
from .plan import load as load_manifest
from .prepared import load_inputs
from .yam_training.contract import CHUNK, EEF_CONTRACT, file_sha256, load_prepared, normalize
from .yam_training.failure_test import grid_index, transplant
from .yam_training.failure_types import profile_dict, read_recipes, valid_window
from .yam_training.prepare_replay import prepare as prepare_replay


def load_context(prepared: Path, run_dir: Path):
    """Validate model inputs without requiring the original annotation document.

    The frozen manifest retains its original annotation hash and eligible windows.
    This path uses explicit reference timestamps, not annotation-based selection.
    """
    manifest, stats = load_prepared(prepared)
    training = json.loads((run_dir / "yam_training_contract.json").read_text())
    if (
        manifest["contract"] != EEF_CONTRACT
        or training["contract"] != EEF_CONTRACT
        or training["stats_sha256"] != manifest["stats_sha256"]
        or training["prepared_manifest_sha256"] != file_sha256(prepared / "manifest.json")
    ):
        raise ValueError("Prepared EEF dataset/normalizer differs from the training run")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    if config["model"]["config"]["num_embodiment_domains"] != 32:
        raise ValueError("Expected the adapted 32-domain model config")
    return manifest, stats


def select_examples(manifest, families):
    """Use the supplied reference timestamps without discovering or labeling pairs."""
    by_id = {ep["episode"]: ep for ep in manifest["episodes"]}
    selected = []
    for family in families:
        example = family["example"]
        if type(example["seed"]) is not int or example["seed"] < 0:
            raise ValueError("Example seeds must be nonnegative integers")
        frame = grid_index(example["host_start"])
        donor_frame = grid_index(family["donor_start"])
        if family["chunks"] != 1:
            raise ValueError("Showcase examples require one 16-action window")
        host = by_id.get(example["host_episode"])
        donor_episode = by_id.get(family["donor_episode"])
        if (
            host is None
            or donor_episode is None
            or not valid_window(host, frame, 1)
            or not valid_window(donor_episode, donor_frame, 1)
        ):
            raise ValueError(f"Requested donor/host window unavailable: {family['id']}")
        if (
            host["episode"] == donor_episode["episode"]
            or host["split"] != "train"
            or donor_episode["split"] != "train"
            or host["outcome"] != "success_clean"
        ):
            raise ValueError("Use a distinct clean training host and training failure donor")
        window = dict(
            id=f"{family['id']}__ep{host['episode']:06d}_f{frame:06d}",
            recipe_id=family["id"],
            host=dict(episode=host["episode"], start=example["host_start"], states_sha256=host["states_sha256"]),
            selection="fixed_reference_timestamps",
        )
        if example["run_name"] != window["id"] + "__failure_donor":
            raise ValueError("Example run name differs from its donor/host window")
        donor = dict(
            source="YAM",
            episode_index=donor_episode["episode"],
            event_index=family["donor_event_index"],
            t_start=family["donor_start"],
            failure_time_s=family["event_start"],
            arm=family["score_arm"],
            label=family["label"],
            split=donor_episode["split"],
            states_sha256=donor_episode["states_sha256"],
        )
        selected.append((family, donor, window))
    names = [f["example"]["run_name"] for f, _, _ in selected]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate showcase runs")
    return selected


def prepare(
    prepared: Path,
    dataset: Path,
    run_dir: Path,
    checkpoint: Path,
    recipes: Path,
    output: Path,
    *,
    ffmpeg: str = "ffmpeg",
) -> dict:
    prepared, dataset, run_dir, checkpoint, recipes, output = (
        p.expanduser().resolve() for p in (prepared, dataset, run_dir, checkpoint, recipes, output)
    )
    manifest, stats = load_context(prepared, run_dir)
    families = read_recipes(recipes)
    selected = select_examples(manifest, families)
    needed = {d["episode_index"] for _, d, _ in selected} | {w["host"]["episode"] for _, _, w in selected}
    arrays = {}
    for episode in manifest["episodes"]:
        if episode["episode"] not in needed:
            continue
        path = prepared / episode["states"]
        if file_sha256(path) != episode["states_sha256"]:
            raise ValueError(f"Prepared states changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays[episode["episode"]] = data["actions_raw"].copy()
    request = dict(
        selection="fixed_reference_timestamps",
        prepared=str(prepared),
        dataset=str(dataset),
        run_dir=str(run_dir),
        checkpoint=str(checkpoint),
        recipes_sha256=file_sha256(recipes),
        checkpoint_metadata_sha256=file_sha256(checkpoint / "model/.metadata"),
        prepared_manifest_sha256=file_sha256(prepared / "manifest.json"),
        config_sha256=file_sha256(run_dir / "config.yaml"),
        training_contract_sha256=file_sha256(run_dir / "yam_training_contract.json"),
    )
    if output.exists():
        if not (output / "showcase.json").is_file():
            raise ValueError("Showcase preparation is incomplete; choose a fresh output directory")
        result = json.loads((output / "showcase.json").read_text())
        if result["request"] != request:
            raise ValueError("Showcase inputs changed; choose a fresh output directory")
        for relative, digest in result["files"].items():
            if file_sha256(output / relative) != digest:
                raise ValueError(f"Showcase input changed: {relative}")
        profile = Profile.from_yaml(output / "profile.yaml")
        for job in result["jobs"]:
            for run in load_manifest(output / job["manifest"])[0]:
                load_inputs(profile, run)
        return result

    output.mkdir(parents=True)
    profile = profile_dict(prepared, dataset, run_dir, manifest, families)
    Profile.from_dict(profile)
    (output / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    by_id = {ep["episode"]: ep for ep in manifest["episodes"]}
    media, runs_by_seed, donors_by_seed, examples = {}, {}, {}, []

    def source(episode, start):
        key = episode, start
        if key not in media:
            directory = output / "sources" / f"ep{episode:06d}_f{grid_index(start):06d}_c1"
            directory.parent.mkdir(exist_ok=True)
            prepare_replay(prepared, dataset, directory, episode, start, ffmpeg=ffmpeg)
            media[key] = str(directory.relative_to(output))
        return media[key]

    for family, donor, window in selected:
        example = family["example"]
        host = window["host"]
        name, seed = example["run_name"], example["seed"]
        hi, di = grid_index(host["start"]), grid_index(family["donor_start"])
        raw = transplant(
            arrays[host["episode"]][hi : hi + CHUNK],
            arrays[donor["episode_index"]][di : di + CHUNK],
            arms=tuple(family["arms"]),
        )
        host_source = source(host["episode"], host["start"])
        donor_source = source(donor["episode_index"], family["donor_start"])
        action_path = output / f"{name}.json"
        action_path.write_text(json.dumps(normalize(raw, stats).tolist(), allow_nan=False) + "\n")
        (output / f"{name}_raw.json").write_text(json.dumps(raw.tolist(), allow_nan=False) + "\n")
        image = output / host_source / "obs.png"
        provenance = dict(
            condition="failure_donor",
            family=family["label"],
            donor=donor,
            window=window,
            host_source=host_source,
            donor_source=donor_source,
            host_split="train",
            injection_frame=4,
            blend_frames=2,
            seed=seed,
            annotations_sha256=manifest["annotations_sha256"],
            manual_failure_confirmation_required=True,
        )
        run = dict(
            name=name,
            recipe=family["id"] + "__failure_donor",
            mode="perturbation",
            collection="YAM",
            host_episode=host["episode"],
            donor=f"{donor['episode_index']}:{donor['event_index']}",
            start_seconds=host["start"],
            inject_at_s=host["start"] + 0.8,
            arm=family["score_arm"],
            target_phase=family["phase"],
            caption=by_id[host["episode"]]["caption"],
            provenance=provenance,
            perturbation=dict(
                kind="eef_donor_transplant", arms=family["arms"], rows=[4, CHUNK], donor_rows=[0, CHUNK - 4]
            ),
            prepared=dict(
                vision_path=str(image.relative_to(output)),
                vision_sha256=file_sha256(image),
                action_path=action_path.name,
                action_sha256=file_sha256(action_path),
                chunks=1,
                normalization="train_quantile",
                stats_sha256=manifest["stats_sha256"],
                training_contract_sha256=request["training_contract_sha256"],
                config_sha256=request["config_sha256"],
            ),
        )
        runs_by_seed.setdefault(seed, []).append(run)
        donors_by_seed.setdefault(seed, []).append(donor)
        examples.append(
            dict(
                label=family["label"],
                name=name,
                seed=seed,
                host_source=host_source,
                donor_source=donor_source,
                action_path=action_path.name,
                video=f"seeds/seed{seed}/runs/{name}/synth.mp4",
            )
        )
    jobs = []
    for seed, runs in runs_by_seed.items():
        name = f"seed{seed}.json"
        (output / name).write_text(json.dumps(dict(n_runs=len(runs), runs=runs), indent=2) + "\n")
        (output / f"seed{seed}_donors.json").write_text(json.dumps(dict(events=donors_by_seed[seed]), indent=2) + "\n")
        jobs.append(dict(seed=seed, manifest=name, out_root=f"seeds/seed{seed}/runs"))
        for run in load_manifest(output / name)[0]:
            load_inputs(Profile.from_dict(profile), run)
    result = dict(
        request=request,
        examples=examples,
        jobs=jobs,
        n_clips=len(examples),
        files={str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()},
    )
    (output / "showcase.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "dataset", "run-dir", "checkpoint", "recipes", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    result = prepare(**vars(parser.parse_args()))
    print(f"Prepared {result['n_clips']} fixed failure examples")


if __name__ == "__main__":
    main()
