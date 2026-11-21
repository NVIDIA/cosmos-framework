# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Verify selected-family renders, create a blind worksheet, and summarize labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
from collections import Counter
from pathlib import Path

import av
import numpy as np

from .contract import CHUNK, FPS
from .failure_types import CONDITIONS, INJECT_FRAME, LABELS, recipe_id, validate_batch

OBSERVED = ("none", *LABELS, "other")
FIELDS = ("clip_id", "video", "validity", "observed_failure", "failure_onset_s", "persistence", "notes")


def check_rendered(root: Path, *, allow_incomplete: bool = False) -> dict:
    batch, _, runs = validate_batch(root)
    for seed in batch["request"]["seeds"]:
        for run in runs:
            directory = root / f"seeds/seed{seed}/runs" / run["name"]
            clip = directory / "synth.mp4"
            if not clip.exists() and allow_incomplete:
                continue
            record = json.loads((directory / "record.json").read_text())
            expected = dict(
                name=run["name"],
                recipe=run["recipe"],
                mode=run["mode"],
                seed=seed,
                checkpoint=batch["request"]["checkpoint"],
                caption=run["caption"],
                fps=FPS,
                post_chunks=run["prepared"]["chunks"],
                prepared=run["prepared"],
                provenance=run["provenance"],
                perturbation=run["perturbation"],
                arm=run["arm"],
                target_phase=run["target_phase"],
                host=dict(
                    source=run["collection"],
                    episode_index=run["host_episode"],
                    t_start=run["start_seconds"],
                    t_inject=run["inject_at_s"],
                ),
            )
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(f"Rendered run {directory} differs in {key}")
            actions = json.loads(Path(run["prepared"]["action_path"]).read_text())
            fed = json.loads((directory / "actions_precomputed.json").read_text())
            if fed != [actions[i : i + CHUNK] for i in range(0, len(actions), CHUNK)]:
                raise ValueError(f"Rendered actions differ from manifest: {directory}")
            with av.open(str(clip)) as video:
                if (
                    video.streams.video[0].average_rate != FPS
                    or sum(1 for _ in video.decode(video=0)) != len(actions) + 1
                ):
                    raise ValueError(f"Incorrect frame count or FPS: {clip}")
    return batch


def review_key(batch: dict, runs: list[dict]) -> dict:
    entries = [(seed, run) for seed in batch["request"]["seeds"] for run in runs]
    random.Random(42).shuffle(entries)
    return {
        f"clip{i:04d}": dict(
            seed=seed,
            name=run["name"],
            family=run["provenance"]["family"],
            condition=run["provenance"]["condition"],
            host_episode=run["host_episode"],
            window=run["provenance"].get("window", {}).get("id"),
            duration_s=run["prepared"]["chunks"] * CHUNK / FPS,
            video=f"seeds/seed{seed}/runs/{run['name']}/synth.mp4",
        )
        for i, (seed, run) in enumerate(entries)
    }


def read_decisions(path: Path, key: dict) -> dict:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected review.csv columns")
        rows = list(reader)
    if (
        len(rows) != len(key)
        or len({r["clip_id"] for r in rows}) != len(rows)
        or {r["clip_id"] for r in rows} != set(key)
    ):
        raise ValueError("Review must contain every scheduled clip exactly once")
    decisions = {}
    for row in rows:
        clip_id = row["clip_id"]
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed review row: {clip_id}")
        if row["video"] != f"blind/{clip_id}.mp4":
            raise ValueError(f"Review video changed: {clip_id}")
        validity, observed = row["validity"], row["observed_failure"]
        if validity not in ("", "valid", "invalid", "uncertain") or observed not in ("", *OBSERVED):
            raise ValueError(f"Unknown validity or observed failure: {clip_id}")
        if not validity:
            if any(row[k] for k in ("observed_failure", "failure_onset_s", "persistence")):
                raise ValueError(f"Set validity before labeling {clip_id}")
            continue
        if validity == "valid" and not observed:
            raise ValueError(f"Valid clip needs an observed_failure (use none): {clip_id}")
        onset = row["failure_onset_s"]
        if validity == "valid" and observed not in ("", "none") and not onset:
            raise ValueError(f"Observed failure needs an onset: {clip_id}")
        if onset:
            try:
                value = float(onset)
            except ValueError:
                raise ValueError(f"Invalid onset: {clip_id}") from None
            if not math.isfinite(value) or not 0 <= value <= key[clip_id]["duration_s"]:
                raise ValueError(f"Onset outside generated clip: {clip_id}")
            if observed in ("", "none"):
                raise ValueError(f"Onset requires an observed failure: {clip_id}")
        if row["persistence"] not in ("", "transient", "persistent", "uncertain"):
            raise ValueError(f"Unknown persistence: {clip_id}")
        if validity == "valid" and observed == "bad_stack" and not row["persistence"]:
            raise ValueError(f"Bad stack needs persistence through the observation window: {clip_id}")
        decisions[clip_id] = row
    return decisions


def summarize(root: Path) -> dict:
    batch, _, runs = validate_batch(root)
    key = review_key(batch, runs)
    if json.loads((root / "review_key.json").read_text()) != key:
        raise ValueError("Review key differs from the frozen manifest")
    decisions = read_decisions(root / "review.csv", key)
    groups = {(v["window"], v["seed"], v["condition"]): k for k, v in key.items() if v["window"]}
    by_name = {run["name"]: run for run in runs}

    def matches(clip_id, label):
        row = decisions.get(clip_id, {})
        return row.get("validity") == "valid" and row.get("observed_failure") == label

    families = {}
    for label in dict.fromkeys(f["label"] for f in batch["families"]):
        ids = [k for k, v in key.items() if v["family"] == label and v["condition"] == "failure_donor"]
        complete = all(
            groups[(key[k]["window"], key[k]["seed"], condition)] in decisions for k in ids for condition in CONDITIONS
        )
        trials, hosts = [], {}
        for clip_id in ids:
            entry = key[clip_id]
            provenance = by_name[entry["name"]]["provenance"]
            control_id = groups[(entry["window"], entry["seed"], "control")]
            paired_success = (
                matches(clip_id, label)
                and matches(control_id, "none")
                and float(decisions[clip_id]["failure_onset_s"]) >= INJECT_FRAME / FPS
                and (label != "bad_stack" or decisions[clip_id]["persistence"] == "persistent")
            )
            trials.append(
                dict(
                    host_episode=entry["host_episode"],
                    seed=entry["seed"],
                    recipe_id=provenance.get("recipe_id", label),
                    donor_episode=provenance["donor"]["episode_index"],
                    donor_split=provenance["donor"]["split"],
                    all_conditions_reviewed=all(
                        groups[(entry["window"], entry["seed"], c)] in decisions for c in CONDITIONS
                    ),
                    reviewed=clip_id in decisions and control_id in decisions,
                    intended_failure_with_clean_control=paired_success,
                )
            )
            hosts.setdefault(entry["host_episode"], []).append(int(paired_success))
        successes = sum(t["intended_failure_with_clean_control"] for t in trials)
        per_host = {str(host): dict(successes=sum(values), scheduled=len(values)) for host, values in hosts.items()}
        interval = None
        if complete and len(hosts) > 1:
            # Seeds share a scene; resample whole hosts, never individual clips.
            values = np.array([np.mean(v) for v in hosts.values()])
            rng = np.random.default_rng(42)
            means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            interval = np.quantile(means, [0.025, 0.975]).tolist()
        condition_stats = {}
        for condition in CONDITIONS:
            selected = [k for k, v in key.items() if v["family"] == label and v["condition"] == condition]
            confusion = Counter(
                "unreviewed"
                if k not in decisions
                else decisions[k]["validity"]
                if decisions[k]["validity"] != "valid"
                else decisions[k]["observed_failure"]
                for k in selected
            )
            n = len(selected)
            done = all(k in decisions for k in selected)
            n_target = sum(matches(k, label) for k in selected)
            n_any = sum(
                decisions.get(k, {}).get("validity") == "valid" and decisions[k]["observed_failure"] not in ("", "none")
                for k in selected
            )
            condition_stats[condition] = dict(
                scheduled=n,
                reviewed=sum(k in decisions for k in selected),
                observed_counts=dict(confusion),
                intended_failure_rate=n_target / n if done else None,
                any_failure_rate=n_any / n if done else None,
            )
        replay_ids = [k for k, v in key.items() if v["family"] == label and v["condition"] == "donor_replay"]
        per_donor = {}
        for identifier in dict.fromkeys(t["recipe_id"] for t in trials):
            selected = [t for t in trials if t["recipe_id"] == identifier]
            done = all(t["all_conditions_reviewed"] for t in selected)
            count = sum(t["intended_failure_with_clean_control"] for t in selected)
            replays = [
                k for k in replay_ids if by_name[key[k]["name"]]["provenance"].get("recipe_id", label) == identifier
            ]
            per_donor[identifier] = dict(
                donor_episode=selected[0]["donor_episode"],
                donor_split=selected[0]["donor_split"],
                scheduled_trials=len(selected),
                complete=done,
                successes=count,
                paired_induction_rate=count / len(selected) if done else None,
                donor_replays=dict(
                    scheduled=len(replays),
                    reviewed=sum(k in decisions for k in replays),
                    intended_failure_count=sum(matches(k, label) for k in replays),
                ),
            )
        families[label] = dict(
            scheduled_trials=len(ids),
            independent_hosts=len(hosts),
            complete=complete,
            successes=successes,
            paired_induction_rate=successes / len(ids) if complete else None,
            acceptance_rule=(
                "reel remains tilted/upright/unstable after release and settling; "
                "flat and stable anywhere inside the tote is success"
                if label == "bad_stack"
                else "observed intended failure after injection"
            ),
            host_bootstrap_95_interval=interval,
            per_host=per_host,
            per_donor=per_donor,
            trials=trials,
            conditions=condition_stats,
            donor_replays=dict(
                scheduled=len(replay_ids),
                reviewed=sum(k in decisions for k in replay_ids),
                intended_failure_count=sum(matches(k, label) for k in replay_ids),
            ),
        )
    summary = dict(
        n_clips=batch["n_clips"],
        reviewed=len(decisions),
        complete=len(decisions) == len(key),
        primary_metric="valid intended failure starting at/after injection (persistent after settling for bad_stack) AND valid no-failure original control / all scheduled failure-donor trials",
        interval_method="95% percentile bootstrap over hosts, 10000 draws, seed 42; exploratory with few hosts",
        families=families,
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Failure-type induction results",
        "",
        f"Reviewed {len(decisions)} / {len(key)} clips. Rates remain pending until each family's three conditions are reviewed.",
        "",
        "| Intended type | Paired successes / scheduled | Rate | Hosts |",
        "|---|---|---|---|",
    ]
    for label, stats in families.items():
        rate = "pending" if stats["paired_induction_rate"] is None else f"{stats['paired_induction_rate']:.1%}"
        lines.append(
            f"| {label} | {stats['successes']} / {stats['scheduled_trials']} | {rate} | {stats['independent_hosts']} |"
        )
    lines += [
        "",
        "## By donor",
        "",
        "| Recipe | Donor split | Paired successes / scheduled | Rate |",
        "|---|---|---|---|",
    ]
    for stats in families.values():
        for identifier, donor in stats["per_donor"].items():
            rate = "pending" if donor["paired_induction_rate"] is None else f"{donor['paired_induction_rate']:.1%}"
            lines.append(
                f"| {identifier} | {donor['donor_split']} | {donor['successes']} / {donor['scheduled_trials']} | {rate} |"
            )
    lines += [
        "",
        "Invalid, uncertain, and wrong-type generations never count as successes. "
        "Invalid or failed original controls also make the pair unsuccessful. "
        "Failures beginning before the intervention do not count as induced. "
        "Bad stacks must persist after release and settling; transient or uncertain faults are not accepted. "
        "All scheduled failure-donor trials remain in the denominator.",
        "See summary.json for condition-specific confusion counts, control failure rates, donor replay results, "
        "and donor/host/seed breakdowns. These results describe the selected donors and hosts; "
        "the host bootstrap interval is exploratory, does not estimate donor-population uncertainty, "
        "and can collapse when every outcome agrees.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return summary


def review(root: Path, ffmpeg: str = "ffmpeg") -> dict:
    batch = check_rendered(root)
    _, _, runs = validate_batch(root)
    key = review_key(batch, runs)
    key_path, worksheet = root / "review_key.json", root / "review.csv"
    if key_path.exists() and json.loads(key_path.read_text()) != key:
        raise ValueError("Existing review key does not match this batch")
    if worksheet.exists() and not key_path.exists():
        raise ValueError("Existing review.csv has no matching review key")
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    (root / "blind").mkdir(exist_ok=True)
    for clip_id, entry in key.items():
        link = root / "blind" / f"{clip_id}.mp4"
        target = (root / entry["video"]).resolve()
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"Blind video link changed: {link}")
        elif link.exists():
            raise ValueError(f"Unexpected blind video file: {link}")
        else:
            link.symlink_to(os.path.relpath(target, link.parent))
    if not worksheet.exists():
        with worksheet.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(dict.fromkeys(FIELDS, "") | dict(clip_id=k, video=f"blind/{k}.mp4") for k in key)
    read_decisions(worksheet, key)  # Never replace or silently discard existing decisions.
    lines = [
        "# Review the selected failure types",
        "",
        "First review review.csv and the blind/ clips without opening review_key.json or comparisons. "
        "Order is shuffled and filenames hide the requested type and condition. All camera views are retained.",
        "",
        "Set validity to valid, invalid (deformation, disappearance, camera disagreement), or uncertain.",
        "For valid clips set observed_failure to none, handover_drop, pickup_slip, pickup_miss, bad_stack, or other.",
        "Handover drop: object falls during gripper-to-gripper transfer. Pickup slip: object was acquired, "
        "then lost before receiving-gripper contact. Pickup miss: grasp withdraws without the object. "
        "Bad stack: released reel remains tilted, upright, or unstable after settling. "
        "Flat and stable anywhere inside the tote is successful placement; alignment with other reels is not required.",
        "Record failure_onset_s from the generated clip, not the donor's mapped timing. For bad_stack, "
        "set persistence to transient, persistent, or uncertain over this clip's observation period.",
        "For bad_stack, inspect the tote after the gripper clears the reel. "
        "Only persistent faults are accepted; settling motion, occlusion, and insufficient post-release evidence do not establish a bad stack.",
        "Label the first failure if more than one occurs, and describe subsequent events in notes. "
        "Failures beginning before the +0.8s intervention do not count as induced.",
        "A failed or invalid original control makes its injection pair unsuccessful; invalid outputs stay in the denominator.",
        "",
        "After labeling, run python -m failuregen.yam_training.failure_types_review --root BATCH --summarize-only.",
        "",
        "## Unblinded diagnostic comparisons",
        "LEFT = original control; CENTER = successful donor transplant; RIGHT = failure donor transplant.",
        "",
        "| Type / donor / host | Seed | Comparison | Recorded host | Failure donor | Successful donor |",
        "|---|---|---|---|---|---|",
    ]
    by_name = {r["name"]: r for r in runs}
    comparisons = root / "comparisons"
    comparisons.mkdir(exist_ok=True)
    for seed in batch["request"]["seeds"]:
        for w in batch["windows"]:
            names = [f"{w['id']}__{c}" for c in CONDITIONS]
            destination = comparisons / f"{w['id']}__seed{seed}.mp4"
            temporary = destination.with_suffix(".tmp.mp4")
            inputs = [str(root / f"seeds/seed{seed}/runs" / n / "synth.mp4") for n in names]
            command = [ffmpeg, "-v", "error", "-y"]
            for path in inputs:
                command.extend(["-i", path])
            command.extend(
                [
                    "-filter_complex",
                    "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[2:v]setpts=PTS-STARTPTS[c];"
                    "[a][b][c]hstack=inputs=3[out]",
                    "-map",
                    "[out]",
                    "-an",
                    "-r",
                    str(FPS),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(temporary),
                ]
            )
            subprocess.run(command, check=True)
            temporary.replace(destination)
            p = by_name[names[0]]["provenance"]
            lines.append(
                f"| {w['family']} / donor ep{p['donor']['episode_index']} / host ep{w['host']['episode']} | {seed} | "
                f"[Watch]({destination.relative_to(root)}) | [Host]({p['host_source']}/ground_truth.mp4) | "
                f"[Failure]({p['donor_source']}/ground_truth.mp4) | [Successful]({p['successful_donor_source']}/ground_truth.mp4) |"
            )
    lines += ["", "## Generated donor replays"]
    for seed in batch["request"]["seeds"]:
        for f in batch["families"]:
            name = f"{recipe_id(f)}__donor_replay"
            lines.append(
                f"- [{f['label']}, donor ep{f['donor_episode']}, seed {seed}](seeds/seed{seed}/runs/{name}/synth.mp4)"
            )
    spans = "; ".join(
        f"{f['label']} donor ep{f['donor_episode']}: {f['chunks'] * CHUNK / FPS:g}s" for f in batch["families"]
    )
    lines += [
        "",
        f"Rollout spans: {spans}. Multiple chunks continue from the previous generated final frame. "
        "These clips assess local failure occurrence, not eventual task completion. "
        "Wrist-change scores are diagnostic only. EEF candidates are not automatically exported as labeled training data.",
    ]
    (root / "REVIEW.md").write_text("\n".join(lines) + "\n")
    return summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.allow_incomplete and not args.check_only:
        parser.error("--allow-incomplete requires --check-only")
    if args.check_only:
        check_rendered(args.root, allow_incomplete=args.allow_incomplete)
    elif args.summarize_only:
        check_rendered(args.root)
        print(json.dumps(summarize(args.root), indent=2))
    else:
        review(args.root, args.ffmpeg)


if __name__ == "__main__":
    main()
