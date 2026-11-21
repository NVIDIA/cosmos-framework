# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Turn recipes into a manifest of runs, each paired with a control.

Recipes are configuration, not code. The six ad-hoc planners this replaces differed only in
which donors they picked, which phase they targeted and which channels they took, so those are
now fields.

Every injection gets a control: the same host, phase, instant and arm, replayed untouched with
the same seed. This is not optional bookkeeping. An untouched clip scores 0.079 in one phase and
0.255 in another simply because the arm moves faster, so a fixed threshold reports success where
nothing happened -- which it did, at an apparent 100 percent, until controls were added.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import balance
from . import donors as donors_mod
from . import hosts as hosts_mod
from .config import Profile


@dataclass
class Recipe:
    """One family of runs: which donors, where they go, and what is taken from them."""

    name: str
    target_phases: tuple[str, ...]
    caption: str
    donor_phases: tuple[str, ...] = ()  # restrict donors to the phases they occurred in
    exclude_donor_phases: tuple[str, ...] = ()
    donor_labels: tuple[str, ...] = ()  # restrict by segment label
    donor_outcomes: tuple[str, ...] = ()  # restrict by the donor's outcome tag
    terminal_only: bool = False
    min_events_in_episode: int = 0
    dims: str = "all"
    blend_frames: int = 4
    jaw_amplitude: float = 1.0
    # None means no cap. A numeric default here is a trap: a corpus with more hosts than the
    # default silently loses the tail of its episodes, with nothing in the output to say so.
    max_donors: int | None = None
    max_hosts: int | None = None
    strategy: str = "pose"
    onset: float = 0.5
    max_windows: int = 1
    allow_pad: bool = False
    pre_chunks: int = 2
    post_chunks: int = 2
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "Recipe":
        known = {f for f in Recipe.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        for key in ("target_phases", "donor_phases", "exclude_donor_phases", "donor_labels", "donor_outcomes"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return Recipe(**kwargs, extra={k: v for k, v in d.items() if k not in known})


def _select_donors(recipe: Recipe, pool: list[donors_mod.Donor]) -> list[donors_mod.Donor]:
    sel = pool
    if recipe.donor_phases:
        sel = [d for d in sel if d.phase in recipe.donor_phases]
    if recipe.exclude_donor_phases:
        sel = [d for d in sel if d.phase not in recipe.exclude_donor_phases]
    if recipe.donor_labels:
        sel = [d for d in sel if d.label in recipe.donor_labels]
    if recipe.donor_outcomes:
        sel = [d for d in sel if d.outcome in recipe.donor_outcomes]
    if recipe.terminal_only:
        sel = [d for d in sel if d.terminal]
    if recipe.min_events_in_episode:
        sel = [d for d in sel if d.n_events_in_episode >= recipe.min_events_in_episode]
    return sel[: recipe.max_donors] if recipe.max_donors else sel


def build(
    profile: Profile,
    collection: str,
    recipes: list[Recipe],
    *,
    n_hosts: int = 3,
    donor_pool: list[donors_mod.Donor] | None = None,
    cap: int | None = None,
    per_host: int | None = None,
    require_phases: tuple[str, ...] = (),
) -> tuple[list[dict], list[donors_mod.Donor]]:
    """Expand recipes over donors and hosts, returning runs plus the donors they reference.

    ``per_host`` caps how many injections each host contributes. Without it, every recipe crosses
    every donor with every host, which over a whole corpus is thousands of runs and many variants
    of the same episode. With it, each host gets a rotating subset, so the batch is broad rather
    than deep and the same donor does not lead on every host.
    """
    pool = donor_pool if donor_pool is not None else donors_mod.find(profile, collection)
    hosts = hosts_mod.find_hosts(profile, collection, require_phases=require_phases, limit=n_hosts)
    if not hosts:
        raise ValueError(f"no clean hosts in {collection} with phases {require_phases}")

    runs: list[dict] = []
    used: dict[str, donors_mod.Donor] = {}
    for recipe in recipes:
        chosen = _select_donors(recipe, pool)
        if not chosen:
            continue
        for donor in chosen:
            for host in hosts[: recipe.max_hosts] if recipe.max_hosts else hosts:
                for phase in recipe.target_phases:
                    points = hosts_mod.injection_points(
                        profile,
                        host,
                        phase,
                        donor,
                        pre_chunks=recipe.pre_chunks,
                        post_chunks=recipe.post_chunks,
                        strategy=recipe.strategy,
                        onset=recipe.onset,
                        max_windows=recipe.max_windows,
                        allow_pad=recipe.allow_pad,
                    )
                    if not points:
                        continue
                    for point in points:
                        used[donor.id] = donor
                        runs.append(
                            {
                                "name": f"{recipe.name}__src{donor.episode}t{int(donor.t_start)}"
                                f"_c{point.window_index}__ep{host.index}",
                                "recipe": recipe.name,
                                "mode": "transplant",
                                "collection": collection,
                                "host_episode": host.index,
                                "donor": donor.id,
                                "target_phase": phase,
                                "arm": point.arm,
                                "inject_at_s": point.t,
                                "cycle": point.window_index,
                                "dims": recipe.dims,
                                "blend_frames": recipe.blend_frames,
                                "jaw_amplitude": recipe.jaw_amplitude,
                                "pre_chunks": recipe.pre_chunks,
                                "post_chunks": point.post_chunks,
                                "pose_distance": point.pose_distance,
                                "caption": recipe.caption,
                                **recipe.extra,
                            }
                        )
                    break  # the first target phase this donor and host admit

    runs = _dedupe(runs)
    if per_host is not None:
        runs = _per_host(runs, per_host)
    # Interleaved by cycle then host, so a batch that is cut short still spans the range rather
    # than finishing eleven variants of one donor.
    runs.sort(key=lambda r: (r["cycle"], r["host_episode"], r["recipe"], r["name"]))
    if cap is not None:
        runs = runs[:cap]
    named = {r["donor"] for r in runs}
    runs += _controls(runs, collection)
    for i, r in enumerate(runs):
        r["index"] = i
    return runs, [used[k] for k in sorted(used) if k in named]


def _per_host(runs: list[dict], per_host: int) -> list[dict]:
    """Keep at most ``per_host`` injections per host, rotating families and donors between hosts.

    The rotation is what turns a per-host cap into corpus-wide diversity: host 1 leads with a
    different family and donor than host 0, so a batch of four-per-host still covers every family
    and every donor window rather than the same four everywhere.
    """
    kept: list[dict] = []
    by_host: dict[int, list[dict]] = {}
    for r in runs:
        by_host.setdefault(r["host_episode"], []).append(r)
    for slot, host in enumerate(sorted(by_host)):
        kept += balance.spread(
            by_host[host], per_host, primary=lambda r: r["recipe"], secondary=lambda r: r["donor"], offset=slot
        )
    return kept


def _dedupe(runs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in runs:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        out.append(r)
    return out


def _controls(runs: list[dict], collection: str) -> list[dict]:
    """One untouched replay per distinct host, phase, instant and arm."""
    seen, out = set(), []
    for r in runs:
        key = (r["host_episode"], r["target_phase"], r["inject_at_s"], r["arm"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": f"control__{r['target_phase']}_{r['arm']}__ep{r['host_episode']}_t{r['inject_at_s']}",
                "recipe": "control",
                "mode": "control",
                "collection": collection,
                "host_episode": r["host_episode"],
                "donor": None,
                "target_phase": r["target_phase"],
                "arm": r["arm"],
                "inject_at_s": r["inject_at_s"],
                "pre_chunks": r["pre_chunks"],
                "post_chunks": r["post_chunks"],
                "caption": r.get("caption_before", ""),
            }
        )
    return out


def write(runs: list[dict], donors: list[donors_mod.Donor], manifest: Path) -> None:
    """Write the manifest and its donor file side by side, since ids are resolved across them."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"n_runs": len(runs), "runs": runs}, indent=2))
    donors_mod.write_labels(donors, manifest.with_name(manifest.stem + "_donors.json"))


def load(manifest: Path) -> tuple[list[dict], dict[str, dict]]:
    """Read a manifest plus its donors, keyed by id.

    A donor id that a manifest names but the donor file lacks is an error, not something to work
    around: silently continuing once turned 64 runs of a production batch into replays of the
    host's own opening seconds, which scored like weak failures instead of like a bug.
    """
    runs = json.loads(manifest.read_text())["runs"]
    for run in runs:
        if "prepared" in run:
            for key in ("vision_path", "action_path"):
                run["prepared"][key] = str((manifest.parent / run["prepared"][key]).resolve())
    donor_file = manifest.with_name(manifest.stem + "_donors.json")
    donors: dict[str, dict] = {}
    if donor_file.is_file():
        for event in json.loads(donor_file.read_text())["events"]:
            donors[f"{event['episode_index']}:{event['event_index']}"] = event
    missing = sorted({r["donor"] for r in runs if r.get("donor") and r["donor"] not in donors})
    if missing:
        raise ValueError(f"{donor_file} is missing donors referenced by the manifest: {missing}")
    return runs, donors
