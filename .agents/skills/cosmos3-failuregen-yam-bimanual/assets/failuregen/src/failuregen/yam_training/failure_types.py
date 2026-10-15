# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare selected EEF failure families, matched controls, and donor replays."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

from failuregen.config import Profile
from failuregen.plan import load as load_manifest
from failuregen.prepared import load_inputs

from .contract import CHUNK, EEF_CONTRACT, FPS, file_sha256, normalize
from .failure_test import grid_index, load_context, transplant
from .prepare_eef import poses
from .prepare_replay import prepare as prepare_replay

LABELS = ("handover_drop", "pickup_slip", "pickup_miss", "bad_stack")
CONDITIONS = ("control", "successful_donor", "failure_donor")
INJECT_FRAME = 4
BLEND_FRAMES = 2
HOST_SPLITS = ("train", "val", "test", "heldout")


def recipe_id(family: dict) -> str:
    """Keep legacy names while allowing several explicit donors for one label."""
    return family.get("id", family["label"])


def read_recipes(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    families = data["families"]
    labels = [f["label"] for f in families]
    ids = [recipe_id(f) for f in families]
    if (
        data.get("version") != 1
        or not families
        or not set(labels) <= set(LABELS)
        or any(not isinstance(i, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", i) for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Recipes must be nonempty, use supported failure labels, and have unique safe IDs")
    for f in families:
        if (
            f["event_anchor"] not in ("start", "end")
            or type(f["chunks"]) is not int
            or f["chunks"] < 1
            or type(f["preferred_cycle"]) is not int
            or f["preferred_cycle"] < 0
            or not f["arms"]
            or len(set(f["arms"])) != len(f["arms"])
            or set(f["arms"]) - {"left", "right"}
            or f["score_arm"] not in f["arms"]
            or type(f["donor_episode"]) is not int
            or f["donor_episode"] < 0
            or not math.isfinite(f["event_offset_s"])
            or f.get("donor_split", "train") not in ("train", "val", "test")
            or not math.isfinite(f.get("event_end_margin_s", 0))
            or f.get("event_end_margin_s", 0) < 0
        ):
            raise ValueError(f"Invalid family settings: {f['label']}")
        grid_index(f["donor_start"])
        lead = f["event_start"] - f["donor_start"]
        if not BLEND_FRAMES / FPS - 1e-7 <= lead <= (f["chunks"] * CHUNK - INJECT_FRAME) / FPS - 0.8 + 1e-7:
            raise ValueError("Donor onset must follow blending and leave at least 0.8s of observation")
    return families


def valid_window(episode: dict, start: int, chunks: int) -> bool:
    valid = set(episode["valid_starts"])
    return all(start + chunk * CHUNK in valid for chunk in range(chunks))


def clean(episode: dict, annotation: dict) -> bool:
    return episode["outcome"] == "success_clean" and all(s["label"] == "normal" for s in annotation["segments"])


def pose_match(host: np.ndarray, donor: np.ndarray, arms: list[str]) -> dict:
    """Deterministic geometric distance, not a generation-success probability."""
    hp, dp = poses(np.asarray(host)[None])[0], poses(np.asarray(donor)[None])[0]
    detail = {}
    for arm in arms:
        index, offset = (0, 0) if arm == "left" else (1, 10)
        rotation = hp[index, :3, :3].T @ dp[index, :3, :3]
        detail[arm] = dict(
            position_m=float(np.linalg.norm(hp[index, :3, 3] - dp[index, :3, 3])),
            rotation_rad=float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))),
            gripper_difference=float(abs(host[offset + 9] - donor[offset + 9])),
        )
    score = sum(
        (d["position_m"] / 0.08) ** 2 + (d["rotation_rad"] / 0.6) ** 2 + (d["gripper_difference"] / 0.25) ** 2
        for d in detail.values()
    ) / len(arms)
    return dict(score=score, arms=detail)


def phase_windows(
    episode: dict, annotation: dict, family: dict, *, prefix: int, timing_adjustment_s: float = 0.0
) -> list[dict]:
    """Align the prospective event to the causal phase, including its end boundary.

    An onset can be labelled recovery/next-pick; injection belongs to the preceding
    grasp/release. Only valid whole chunks are accepted, never padded trajectories.
    """
    lead = family["event_start"] - family["donor_start"]
    phases = [p for p in annotation["phase_segments"] if p["phase"] == family["phase"]]
    valid = set(episode["valid_starts"])
    whole_starts = [
        s for s in episode["valid_starts"] if all(s + chunk * CHUNK in valid for chunk in range(family["chunks"]))
    ]
    result = []
    for cycle, phase in enumerate(phases):
        starts = [
            s
            for s in whole_starts
            if phase["start"] <= (s + prefix) / FPS < phase["end"]
            and (s + prefix) / FPS + lead <= phase["end"] - family.get("event_end_margin_s", 0) + 1e-7
        ]
        if not starts:
            continue
        target = phase[family["event_anchor"]] + family["event_offset_s"] - lead - prefix / FPS
        nearest = min(starts, key=lambda s: (abs(s / FPS - target), s))
        chosen = [s for s in starts if abs(s - nearest) / FPS <= timing_adjustment_s + 1e-7]
        for start in chosen:
            result.append(
                dict(
                    episode=episode["episode"],
                    start=start / FPS,
                    cycle=cycle,
                    phase_start=phase["start"],
                    phase_end=phase["end"],
                    states_sha256=episode["states_sha256"],
                )
            )
    return sorted(result, key=lambda w: (w["cycle"] != family["preferred_cycle"], w["cycle"]))


def select_windows(
    manifest: dict,
    annotations: dict,
    families: list[dict],
    *,
    n_hosts: int | None = 5,
    host_split: str = "test",
    observations: dict[int, np.ndarray] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Select hosts independently for each donor; None means every eligible host."""
    if (n_hosts is not None and (type(n_hosts) is not int or n_hosts < 1)) or host_split not in HOST_SPLITS:
        raise ValueError("Choose positive n_hosts or None for all, and a train/val/test/heldout host split")
    host_splits = {"val", "test"} if host_split == "heldout" else {host_split}
    episodes = sorted(manifest["episodes"], key=lambda e: e["episode"])
    by_id = {e["episode"]: e for e in episodes}
    failure_donor_ids = {f["donor_episode"] for f in families}
    windows, donors = [], []
    for f in families:
        donor = by_id.get(f["donor_episode"])
        expected_split = f.get("donor_split", "train")
        if donor is None or donor["split"] != expected_split:
            expected = "training" if expected_split == "train" else f"the declared {expected_split} split"
            raise ValueError(f"{recipe_id(f)}: failure donor must belong to {expected}")
        ann = annotations[f"episode_{donor['episode']:06d}"]
        events = [
            (i, s)
            for i, s in enumerate(ann["segments"])
            if s["label"] == f["label"] and math.isclose(s["start"], f["event_start"], abs_tol=1e-6)
        ]
        if len(events) != 1 or not valid_window(donor, grid_index(f["donor_start"]), f["chunks"]):
            raise ValueError(f"{f['label']}: missing/ambiguous failure event or invalid donor window")
        if not any(
            p["phase"] == f["phase"] and p["start"] <= f["donor_start"] < p["end"] for p in ann["phase_segments"]
        ):
            raise ValueError(f"{f['label']}: donor must start in its causal phase")
        donors.append(
            dict(
                source="YAM",
                episode_index=donor["episode"],
                event_index=events[0][0],
                t_start=f["donor_start"],
                failure_time_s=f["event_start"],
                arm=f["score_arm"],
                label=f["label"],
                split=donor["split"],
                states_sha256=donor["states_sha256"],
            )
        )
        clean_donors = []
        for other in episodes:
            other_ann = annotations[f"episode_{other['episode']:06d}"]
            if other["episode"] not in failure_donor_ids and other["split"] == "train" and clean(other, other_ann):
                clean_donors.extend(
                    phase_windows(
                        other, other_ann, f, prefix=0, timing_adjustment_s=0.8 if observations is not None else 0.0
                    )
                )
        selected = []
        for ep in episodes:
            host_ann = annotations[f"episode_{ep['episode']:06d}"]
            if ep["episode"] in failure_donor_ids or ep["split"] not in host_splits or not clean(ep, host_ann):
                continue
            candidates = phase_windows(
                ep, host_ann, f, prefix=INJECT_FRAME, timing_adjustment_s=0.8 if observations is not None else 0.0
            )
            if not candidates:
                continue

            def observation(window, prefix=0):
                return observations[window["episode"]][grid_index(window["start"]) + prefix]

            if observations is None:
                host = candidates[0]
                match = None
            else:
                donor_pose = observations[donor["episode"]][grid_index(f["donor_start"])]

                def host_match(window):
                    return pose_match(observation(window, INJECT_FRAME), donor_pose, f["arms"])

                host = min(candidates, key=lambda w: (host_match(w)["score"], w["cycle"], w["start"]))
                match = host_match(host)
            # Choose a different, successful TRAIN donor with a comparable phase/cycle.
            shams = [s for s in clean_donors if s["episode"] != ep["episode"]]
            if not shams:
                raise ValueError(f"{f['label']}: no distinct clean training donor")
            duration = host["phase_end"] - host["phase_start"]
            if observations is None:
                sham = min(
                    shams,
                    key=lambda s: (
                        s["cycle"] != host["cycle"],
                        abs(s["phase_end"] - s["phase_start"] - duration),
                        s["episode"],
                        s["start"],
                    ),
                )
            else:
                sham = min(
                    shams,
                    key=lambda s: (
                        pose_match(observation(host, INJECT_FRAME), observation(s), f["arms"])["score"],
                        s["episode"],
                        s["start"],
                    ),
                )
            window = dict(
                id=f"{recipe_id(f)}__ep{ep['episode']:06d}_f{grid_index(host['start']):06d}",
                family=f["label"],
                host_split=ep["split"],
                host=host,
                successful_donor=sham,
            )
            if "id" in f:
                window["recipe_id"] = recipe_id(f)
            if match is not None:
                window["failure_pose_match"] = match
                window["successful_pose_match"] = pose_match(
                    observation(host, INJECT_FRAME), observation(sham), f["arms"]
                )
            selected.append(window)
            if observations is None and len(selected) == n_hosts:
                break
        if observations is not None:
            selected.sort(key=lambda w: (w["failure_pose_match"]["score"], w["host"]["episode"], w["host"]["start"]))
            if n_hosts is not None:
                selected = selected[:n_hosts]
        if not selected or (n_hosts is not None and len(selected) != n_hosts):
            raise ValueError(f"{f['label']}: requested {n_hosts} clean {host_split} hosts, found {len(selected)}")
        windows.extend(selected)
    return windows, donors


def profile_dict(prepared: Path, dataset: Path, run_dir: Path, manifest: dict, families: list[dict]) -> dict:
    return {
        "robot": {
            "state_dim": 14,
            "arms": {
                "left": {"joints": [0, 6], "jaw": 6, "free": 0.022},
                "right": {"joints": [7, 13], "jaw": 13, "free": 0.996},
            },
            "hold_band": [0.15, 0.5],
            "phase_arm": {f["phase"]: f["score_arm"] for f in families},
        },
        "cameras": {
            "front": EEF_CONTRACT["camera_order"][0],
            "wrist": {"left": EEF_CONTRACT["camera_order"][1], "right": EEF_CONTRACT["camera_order"][2]},
            "front_fraction": 0.5,
        },
        "dataset": {
            "roots": {"YAM": str(dataset)},
            "annotations": manifest["annotations"],
            "source_fps": 30,
            "target_fps": FPS,
            "chunk_length": CHUNK,
            "clean_outcome": "success_clean",
            "failure_labels": list(dict.fromkeys(f["label"] for f in families)),
        },
        "model": {
            "adapter": "failuregen.adapters.cosmos:CosmosForwardDynamics",
            "domain_name": "xdof_yam",
            "action_encoding": "precomputed",
            "action_dim": 20,
            "normalization": "train_quantile",
            "normalizer_stats": str(prepared / "action_stats.json"),
            "training_contract_file": str(run_dir / "yam_training_contract.json"),
            "config_file": str(run_dir / "config.yaml"),
            "expected_num_embodiment_domains": 32,
            "view_point": "concat_view",
            "image_size": 720,
            "num_steps": 30,
            "guidance": 1.0,
            "shift": 10.0,
            "parallelism_preset": "throughput",
            "use_torch_compile": False,
            "use_cuda_graphs": False,
        },
    }


def validate_batch(output: Path) -> tuple[dict, Profile, list[dict]]:
    if not (output / "batch.json").is_file():
        raise ValueError("Batch preparation is incomplete; use a fresh output directory")
    batch = json.loads((output / "batch.json").read_text())
    for relative, digest in batch["files"].items():
        if file_sha256(output / relative) != digest:
            raise ValueError(f"Batch input changed after preparation: {relative}")
    profile = Profile.from_yaml(output / "profile.yaml")
    runs, _ = load_manifest(output / "manifest.json")
    expected = len(batch["windows"]) * len(CONDITIONS) + len(batch["families"])
    if len(runs) != expected or len({r["name"] for r in runs}) != expected:
        raise ValueError("Incomplete or duplicate runs in batch manifest")
    for run in runs:
        load_inputs(profile, run)
    return batch, profile, runs


def prepare(
    prepared: Path,
    dataset: Path,
    run_dir: Path,
    output: Path,
    checkpoint: Path,
    recipes: Path,
    *,
    n_hosts: int | None = 5,
    host_split: str = "test",
    seeds: tuple[int, ...] = (0, 1, 2),
    resume: bool = False,
    ffmpeg: str = "ffmpeg",
    match_eef_pose: bool = False,
) -> dict:
    prepared, dataset, run_dir, output, checkpoint, recipes = [
        p.resolve() for p in (prepared, dataset, run_dir, output, checkpoint, recipes)
    ]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    manifest, stats, annotations = load_context(prepared, dataset, run_dir)
    families = read_recipes(recipes)
    observations = None
    if match_eef_pose:
        observations = {}
        for ep in manifest["episodes"]:
            path = prepared / ep["states"]
            if file_sha256(path) != ep["states_sha256"]:
                raise ValueError(f"Prepared EEF states changed: episode {ep['episode']}")
            with np.load(path, allow_pickle=False) as data:
                observations[ep["episode"]] = data["observations_eef_20d"].copy()
            poses(observations[ep["episode"]])
    windows, donors = select_windows(
        manifest, annotations, families, n_hosts=n_hosts, host_split=host_split, observations=observations
    )
    request = dict(
        version=1,
        prepared=str(prepared),
        dataset=str(dataset),
        run_dir=str(run_dir),
        checkpoint=str(checkpoint),
        checkpoint_metadata_sha256=file_sha256(checkpoint / "model/.metadata"),
        prepared_manifest_sha256=file_sha256(prepared / "manifest.json"),
        training_contract_sha256=file_sha256(run_dir / "yam_training_contract.json"),
        config_sha256=file_sha256(run_dir / "config.yaml"),
        recipes_sha256=file_sha256(recipes),
        n_hosts=n_hosts,
        host_split=host_split,
        seeds=list(seeds),
        inject_frame=INJECT_FRAME,
        blend_frames=BLEND_FRAMES,
    )
    if match_eef_pose:
        request["pose_matching"] = dict(
            version=1,
            position_scale_m=0.08,
            rotation_scale_rad=0.6,
            gripper_scale=0.25,
            timing_adjustment_s=0.8,
            compare_at="host injection versus donor start",
        )
    by_id = {e["episode"]: e for e in manifest["episodes"]}
    used = {d["episode_index"] for d in donors} | {
        w[key]["episode"] for w in windows for key in ("host", "successful_donor")
    }
    arrays = {}
    for index in used:
        ep = by_id[index]
        path = prepared / ep["states"]
        if file_sha256(path) != ep["states_sha256"]:
            raise ValueError(f"Prepared EEF states changed: episode {index}")
        with np.load(path, allow_pickle=False) as data:
            arrays[index] = data["actions_raw"].copy()
    if output.exists():
        if not resume:
            raise ValueError("Output already exists; use --resume for an unchanged batch")
        batch, _, _ = validate_batch(output)
        if batch["request"] != request or batch["windows"] != windows:
            raise ValueError("Resume settings or source data differ from the prepared batch")
        return batch
    if resume:
        raise ValueError("Cannot resume a missing batch")
    output.mkdir(parents=True)
    profile = profile_dict(prepared, dataset, run_dir, manifest, families)
    Profile.from_dict(profile)
    (output / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    (output / "recipes.json").write_text(json.dumps({"version": 1, "families": families}, indent=2) + "\n")
    media = {}
    runs = []

    def source(index, start, chunks):
        key = (index, start, chunks)
        if key not in media:
            directory = output / "sources" / f"ep{index:06d}_f{grid_index(start):06d}_c{chunks}"
            directory.parent.mkdir(exist_ok=True)
            prepare_replay(prepared, dataset, directory, index, start, chunks=chunks, ffmpeg=ffmpeg)
            media[key] = directory
        return media[key]

    def raw(index, start, chunks):
        first = grid_index(start)
        result = arrays[index][first : first + chunks * CHUNK]
        if result.shape != (chunks * CHUNK, 20):
            raise ValueError("Incomplete EEF action window")
        return result

    def add_run(name, family, index, start, actions, condition, provenance, donor_id=None):
        image = source(index, start, family["chunks"]) / "obs.png"
        raw_file, action_file = output / f"{name}_raw.json", output / f"{name}.json"
        raw_file.write_text(json.dumps(actions.tolist(), allow_nan=False) + "\n")
        action_file.write_text(json.dumps(normalize(actions, stats).tolist(), allow_nan=False) + "\n")
        control = condition in ("control", "donor_replay")
        runs.append(
            dict(
                name=name,
                recipe=f"{recipe_id(family)}__{condition}",
                mode="control" if control else "perturbation",
                collection="YAM",
                host_episode=index,
                donor=donor_id,
                start_seconds=start,
                inject_at_s=start + INJECT_FRAME / FPS,
                arm=family["score_arm"],
                target_phase=family["phase"],
                caption=by_id[index]["caption"],
                perturbation=None
                if control
                else dict(
                    kind="eef_donor_transplant",
                    arms=family["arms"],
                    rows=[INJECT_FRAME, len(actions)],
                    donor_rows=[0, len(actions) - INJECT_FRAME],
                ),
                provenance=provenance,
                prepared=dict(
                    vision_path=str(image.relative_to(output)),
                    vision_sha256=file_sha256(image),
                    action_path=action_file.name,
                    action_sha256=file_sha256(action_file),
                    chunks=family["chunks"],
                    normalization="train_quantile",
                    stats_sha256=manifest["stats_sha256"],
                    training_contract_sha256=request["training_contract_sha256"],
                    config_sha256=request["config_sha256"],
                ),
            )
        )

    for f, donor in zip(families, donors):
        donor_dir = source(f["donor_episode"], f["donor_start"], f["chunks"])
        donor_actions = raw(f["donor_episode"], f["donor_start"], f["chunks"])
        common = dict(
            family=f["label"],
            donor=donor,
            contract=EEF_CONTRACT,
            annotations_sha256=manifest["annotations_sha256"],
            injection_frame=INJECT_FRAME,
            blend_frames=BLEND_FRAMES,
            donor_source=str(donor_dir.relative_to(output)),
            manual_failure_confirmation_required=True,
        )
        if "id" in f:
            common["recipe_id"] = recipe_id(f)
        add_run(
            f"{recipe_id(f)}__donor_replay",
            f,
            f["donor_episode"],
            f["donor_start"],
            donor_actions,
            "donor_replay",
            dict(
                common,
                condition="donor_replay",
                host_split=donor["split"],
                reference_event_time_s=f["event_start"] - f["donor_start"],
            ),
        )
        for w in [w for w in windows if w.get("recipe_id", w["family"]) == recipe_id(f)]:
            host, sham = w["host"], w["successful_donor"]
            host_actions = raw(host["episode"], host["start"], f["chunks"])
            sham_actions = raw(sham["episode"], sham["start"], f["chunks"])
            sham_dir = source(sham["episode"], sham["start"], f["chunks"])
            variants = dict(
                control=host_actions,
                successful_donor=transplant(host_actions, sham_actions, arms=tuple(f["arms"])),
                failure_donor=transplant(host_actions, donor_actions, arms=tuple(f["arms"])),
            )
            for condition, actions in variants.items():
                provenance = dict(
                    common,
                    condition=condition,
                    window=w,
                    host_split=w["host_split"],
                    host_source=str(source(host["episode"], host["start"], f["chunks"]).relative_to(output)),
                    successful_donor_source=str(sham_dir.relative_to(output)),
                    reference_event_time_s=INJECT_FRAME / FPS + f["event_start"] - f["donor_start"],
                )
                add_run(
                    f"{w['id']}__{condition}",
                    f,
                    host["episode"],
                    host["start"],
                    actions,
                    condition,
                    provenance,
                    f"{donor['episode_index']}:{donor['event_index']}" if condition == "failure_donor" else None,
                )
    (output / "manifest.json").write_text(json.dumps(dict(n_runs=len(runs), runs=runs), indent=2) + "\n")
    (output / "manifest_donors.json").write_text(json.dumps(dict(events=donors), indent=2) + "\n")
    batch = dict(
        request=request,
        families=families,
        windows=windows,
        n_clips=len(runs) * len(seeds),
        n_experiment_clips=len(windows) * len(CONDITIONS) * len(seeds),
        n_donor_replays=len(families) * len(seeds),
        manual_failure_confirmation_required=True,
        files={str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    # Completion marker is last; partial preparation cannot be resumed as finished.
    (output / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    validate_batch(output)
    print(f"Prepared {batch['n_experiment_clips']} matched clips + {batch['n_donor_replays']} donor replays")
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "dataset", "run-dir", "output", "checkpoint", "recipes"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--n-hosts",
        type=lambda value: None if value == "all" else int(value),
        default=5,
        help="Host count per donor, or all to use every eligible donor-host pair",
    )
    parser.add_argument(
        "--host-split",
        choices=HOST_SPLITS,
        default="test",
        help="Host source split; heldout pools val and test without changing their split labels",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--match-eef-pose",
        action="store_true",
        help="Rank phase-compatible hosts and successful donors by EEF pose and gripper state",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    print(json.dumps(prepare(**vars(parser.parse_args()))["request"], indent=2))


if __name__ == "__main__":
    main()
