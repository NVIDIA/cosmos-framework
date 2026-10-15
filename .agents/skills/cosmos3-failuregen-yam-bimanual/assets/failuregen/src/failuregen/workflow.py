# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Entry point for EEF donor selection and generation planning.

Recipe derivation follows failure-types/eval-v6-20261002/plan_batch.py.
Selection, motion transfer, rendering, and review use the original modules.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .yam_training.contract import EEF_CONTRACT, file_sha256, load_prepared
from .yam_training.failure_types import read_recipes, select_windows, valid_window

CAUSAL_PHASES = {
    "handover_drop": ("handover",),
    "pickup_miss": ("pick_grasp",),
    "pickup_slip": ("lift_extract", "move_to_handover"),
}


def build_recipes(manifest: dict, annotations: dict) -> tuple[list[dict], list[dict]]:
    """Derive training-donor windows with the timing rules used in job3218072."""
    families, skipped = [], []
    for episode in manifest["episodes"]:
        if episode["split"] != "train":
            continue
        index = episode["episode"]
        annotation = annotations[f"episode_{index:06d}"]
        for event in annotation["segments"]:
            label = event["label"]
            if label not in CAUSAL_PHASES:
                continue
            if not annotation.get("reviewed") or not annotation.get("phase_reviewed"):
                raise ValueError(f"Review failure and phase tracks for episode {index}")
            phases = [
                phase
                for phase in annotation["phase_segments"]
                if phase["phase"] in CAUSAL_PHASES[label] and phase["start"] < event["start"] <= phase["end"] + 1e-7
            ]
            if not phases:
                skipped.append(dict(episode=index, label=label, reason="No causal phase"))
                continue
            phase = phases[-1]
            start = math.ceil((max(event["start"] - 1.0, phase["start"]) - 1e-7) * 5) / 5
            lead = round(event["start"] - start, 8)
            if lead < 0.4 - 1e-7 or not valid_window(episode, round(start * 5), 1):
                skipped.append(
                    dict(
                        episode=index,
                        label=label,
                        reason="Onset precedes completion of the 0.4s blend within the causal phase",
                        event_start=event["start"],
                        phase=phase,
                    )
                )
                continue
            margin = 0.6 if phase["phase"] == "move_to_handover" else 0.0
            family = dict(
                id=f"{label}_eval_ep{index}",
                label=label,
                phase=phase["phase"],
                arms=["left", "right"] if label == "handover_drop" else ["left"],
                score_arm="left",
                chunks=1,
                preferred_cycle=0,
                event_anchor="end",
                event_offset_s=-margin,
                event_end_margin_s=margin,
                donor_episode=index,
                donor_split="train",
                donor_start=start,
                event_start=event["start"],
            )
            if label == "handover_drop":
                family.update(event_anchor="start", event_offset_s=1.8)
            # The reference corpus has one selected event per donor. Avoid ambiguous
            # identities if a new dataset has repeated events of the same type.
            if any(f["id"] == family["id"] for f in families):
                family["id"] += f"_event{annotation['segments'].index(event)}"
            families.append(family)
    return families, skipped


def plan(
    prepared: Path,
    dataset: Path,
    output: Path,
    *,
    labels: tuple[str, ...] = ("pickup_slip",),
    max_donors: int | None = 1,
    n_hosts: int = 1,
    seeds: tuple[int, ...] = (0,),
) -> dict:
    """Preview actual train-only, pose-matched selection without loading a model."""
    if not labels or set(labels) - set(CAUSAL_PHASES):
        raise ValueError(f"Choose failure labels from {tuple(CAUSAL_PHASES)}")
    if n_hosts < 1 or (max_donors is not None and max_donors < 1):
        raise ValueError("Host and donor limits must be positive")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    prepared, dataset, output = (p.expanduser().resolve() for p in (prepared, dataset, output))
    manifest, _ = load_prepared(prepared)
    if manifest["contract"] != EEF_CONTRACT:
        raise ValueError("Use the prepared 20-D EEF dataset, not joint-space inputs")
    annotation_path = dataset / manifest["annotations"]
    if file_sha256(annotation_path) != manifest["annotations_sha256"]:
        raise ValueError("Annotations differ from the prepared inputs; use their matching bundle")
    annotations = json.loads(annotation_path.read_text())["episodes"]
    families, skipped = build_recipes(manifest, annotations)
    families = [family for family in families if family["label"] in labels]
    if max_donors is not None:
        families = families[:max_donors]
    if not families:
        raise ValueError(f"No usable reviewed training donors for {labels}; inspect annotations and timing")
    observations = {}
    for episode in manifest["episodes"]:
        path = prepared / episode["states"]
        if file_sha256(path) != episode["states_sha256"]:
            raise ValueError(f"Prepared EEF states changed: episode {episode['episode']}")
        with np.load(path, allow_pickle=False) as arrays:
            observations[episode["episode"]] = arrays["observations_eef_20d"].copy()
    windows, donors = select_windows(
        manifest, annotations, families, n_hosts=n_hosts, host_split="train", observations=observations
    )
    report = dict(
        contract=EEF_CONTRACT,
        labels=list(labels),
        seeds=list(seeds),
        max_donors=max_donors,
        n_hosts=n_hosts,
        host_split="train",
        match_eef_pose=True,
        prepared_manifest_sha256=file_sha256(prepared / "manifest.json"),
        annotations_sha256=manifest["annotations_sha256"],
        stats_sha256=manifest["stats_sha256"],
        failure_candidates=len(seeds) * len(windows),
        matched_controls=2 * len(seeds) * len(windows),
        donor_replays=len(seeds) * len(donors),
        total_clips=len(seeds) * (3 * len(windows) + len(donors)),
        families=families,
        windows=windows,
        skipped=skipped,
    )
    recipes = dict(version=1, families=families)
    if output.exists():
        for name, expected in (("recipes.json", recipes), ("selection.json", report)):
            path = output / name
            if not path.is_file() or json.loads(path.read_text()) != expected:
                raise FileExistsError(f"Selection changed; use a fresh output directory: {output}")
        return report
    output.mkdir(parents=True)
    recipe_path = output / "recipes.json"
    recipe_path.write_text(json.dumps(recipes, indent=2) + "\n")
    read_recipes(recipe_path)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", choices=tuple(CAUSAL_PHASES), default=["pickup_slip"])
    parser.add_argument("--max-donors", type=lambda s: None if s == "all" else int(s), default=1)
    parser.add_argument("--n-hosts", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    result = plan(**vars(parser.parse_args()))
    print(
        json.dumps(
            {k: result[k] for k in ("failure_candidates", "matched_controls", "donor_replays", "total_clips")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
