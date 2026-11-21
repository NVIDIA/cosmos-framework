# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Command line: inspect, donors, plan, run, score, gallery.

The order is the workflow. Inspect what the annotations claim, check which claims the telemetry
corroborates, plan runs from recipes, render them, score against controls, then look at the
result. Only ``run`` needs a GPU or a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import annotations as ann
from . import donors as donors_mod
from . import plan as plan_mod
from . import review as review_mod
from . import scoring as scoring_mod
from .config import Profile


def _profile(args) -> Profile:
    return Profile.from_yaml(args.profile)


def cmd_inspect(args) -> None:
    profile = _profile(args)
    episodes = ann.load_episodes(profile, args.collection)
    dp = profile.dataset
    clean = [e for e in episodes.values() if e.is_clean(dp)]
    withfail = {i: e for i, e in episodes.items() if e.failures(dp)}
    print(
        f"{len(episodes)} episodes in {args.collection}: {len(clean)} clean, "
        f"{len(withfail)} carrying at least one {'/'.join(dp.failure_labels)} segment"
    )

    phases: dict[str, int] = {}
    for e in episodes.values():
        for w in e.phases:
            phases[w.name] = phases.get(w.name, 0) + 1
    print("\nphase windows across the collection:")
    for name, count in sorted(phases.items(), key=lambda kv: -kv[1]):
        print(f"  {name:18s} {count}")

    print("\nfailure segments by the phase they occur in:")
    by_phase: dict[str, int] = {}
    for e in withfail.values():
        for s in e.failures(dp):
            by_phase[e.phase_at(s.start) or "unknown"] = by_phase.get(e.phase_at(s.start) or "unknown", 0) + 1
    for name, count in sorted(by_phase.items(), key=lambda kv: -kv[1]):
        print(f"  {name:18s} {count}")
    absent = sorted(set(phases) - set(by_phase))
    if absent:
        print(f"\nno failure is annotated in: {', '.join(absent)}")
        print(
            "  do not target those phases unless you have a recorded donor that demonstrates "
            "the intended failure mechanism"
        )


def cmd_donors(args) -> None:
    profile = _profile(args)
    strict = donors_mod.find(profile, args.collection, require_signature=True)
    claimed = donors_mod.find(profile, args.collection, require_signature=False)
    print(f"{len(claimed)} annotated failure events, {len(strict)} with a jaw signature the telemetry corroborates\n")
    for d in strict[: args.limit]:
        print(
            f"  ep{d.episode:>4}:{d.event_index}  {d.t_start:7.2f}-{d.t_end:6.2f}s  "
            f"{str(d.phase):14s} arm={d.arm:5s} excursion {d.overshoot:.2f}"
            f"{'  terminal' if d.terminal else ''}"
        )
    if len(claimed) > len(strict):
        print(
            f"\n{len(claimed) - len(strict)} events are annotated but show no aperture movement; "
            "transplanting those carries no signature"
        )
    if args.out:
        donors_mod.write_labels(strict, Path(args.out))
        print(f"\nwrote {args.out}")


def cmd_plan(args) -> None:
    profile = _profile(args)
    recipes = [plan_mod.Recipe.from_dict(d) for d in json.loads(Path(args.recipes).read_text())]
    pool = None
    if args.donors:
        pool = [d for path in args.donors for d in donors_mod.load_labels(Path(path))]
        if args.include_annotated:
            pool += donors_mod.find(profile, args.collection)
        print(
            f"donor pool: {len(pool)} windows from {len(args.donors)} file(s)"
            + (" plus the corroborated annotations" if args.include_annotated else "")
        )
    runs, used = plan_mod.build(
        profile,
        args.collection,
        recipes,
        n_hosts=args.n_hosts,
        donor_pool=pool,
        cap=args.cap,
        per_host=args.per_host,
        require_phases=tuple(args.require_phases or ()),
    )
    plan_mod.write(runs, used, Path(args.out))
    injections = [r for r in runs if r["mode"] != "control"]
    print(f"{len(injections)} injections + {len(runs) - len(injections)} controls -> {args.out}")
    counts: dict[str, int] = {}
    for r in injections:
        counts[r["recipe"]] = counts.get(r["recipe"], 0) + 1
    for name, count in sorted(counts.items()):
        print(f"  {name:24s} {count}")
    print(f"  donors referenced: {len(used)}, hosts: {len({r['host_episode'] for r in injections})}")


def cmd_run(args) -> None:
    from .adapters.base import load_adapter
    from .runner import run_manifest

    profile = _profile(args)
    runs, donors = plan_mod.load(Path(args.manifest))
    out_root = Path(args.out_root)
    from .prepared import load_inputs

    for run in runs:
        if "prepared" in run:
            load_inputs(profile, run)  # fail before loading the GPU checkpoint
        elif profile.model.get("action_encoding") == "precomputed":
            raise ValueError("precomputed action encoding requires prepared inputs for every run")

    try:  # the reference adapter runs distributed; without it every rank writes
        import os

        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        from cosmos_framework.inference.common.init import init_script, is_rank0

        init_script()
    except Exception:

        def is_rank0():
            return True

    model = load_adapter(profile.model["adapter"], profile, checkpoint=args.checkpoint, output_dir=out_root)
    result = run_manifest(
        profile,
        model,
        runs,
        donors,
        out_root,
        seed=args.seed,
        shard=args.shard,
        shards=args.shards,
        claim_timeout=args.claim_timeout,
        is_rank0=is_rank0,
        log=lambda m: print(m, flush=True),
    )
    if result["failed"]:
        raise SystemExit(1)


def cmd_score(args) -> None:
    profile = _profile(args)
    root = Path(args.root)
    scores = scoring_mod.score_directory(profile, root)
    if not scores:
        raise SystemExit(f"no scorable runs under {root}")
    print(f"{'run':52s} {'post':>7s} {'ctl':>7s} {'ratio':>7s}  hit")
    for s in scores:
        ctl = f"{s.control:7.3f}" if s.control is not None else "      -"
        ratio = f"{s.ratio:7.2f}" if s.ratio is not None else "      -"
        print(f"{s.name[:52]:52s} {s.post:7.3f} {ctl} {ratio}  {'YES' if s.hit else '.'}")
    print(f"\nthreshold: ratio >= {scoring_mod.RATIO_THRESHOLD} against the run's own control")
    for recipe, stats in sorted(scoring_mod.summarise(scores).items()):
        print(
            f"  {recipe:24s} {stats['hits']}/{stats['scored']} median {stats['median_ratio']} "
            f"best {stats['best_ratio']}"
            + (f"  ({stats['unscorable']} without a control)" if stats["unscorable"] else "")
        )
    (root / "scores.json").write_text(
        json.dumps([s.__dict__ | {"ratio": s.ratio, "hit": s.hit} for s in scores], indent=2)
    )


def cmd_export(args) -> None:
    from . import export as export_mod

    profile = _profile(args)
    root = Path(args.root)
    out = Path(args.out)
    episodes = export_mod.select(
        profile, root, n=args.n, threshold=args.threshold, per_host=args.per_host, with_controls=not args.no_controls
    )
    print(
        f"selected {sum(1 for e in episodes if e.is_failure)} failures + "
        f"{sum(1 for e in episodes if not e.is_failure)} controls from {root} "
        f"(gate {args.threshold}x)"
    )
    if args.real_successes:
        real = export_mod.real_successes(profile, args.real_successes, require_phases=tuple(args.require_phases or ()))
        print(f"plus {len(real)} recorded successful episodes from {args.real_successes}")
        episodes += real

    normalizer = None if args.no_verify else export_mod.quantile_normalizer(profile)
    kept, rejected = export_mod.prepare(profile, episodes, normalizer)
    if rejected:
        from collections import Counter as _Counter

        print(f"dropped {len(rejected)} runs that could not be verified against what the model was fed:")
        for reason, count in _Counter(r for _, r in rejected).most_common():
            print(f"    {count:>4} {reason}")
        print("    e.g. " + ", ".join(n for n, _ in rejected[:4]))
    if not kept:
        raise SystemExit("nothing left to export")

    summary = export_mod.summarise(kept)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    collection = kept[0].record["host"]["source"]
    meta = export_mod.write(
        profile,
        kept,
        out,
        repo_id=args.repo_id,
        checkpoint=args.checkpoint or root.name,
        joint_names=export_mod.source_joint_names(profile, collection),
    )
    print(
        f"\nwrote {meta['n_episodes']} episodes ({meta['n_failures']} failures, {meta['n_controls']} controls) to {out}"
    )


def cmd_validate_export(args) -> None:
    from . import export_check

    profile = _profile(args)
    ok = export_check.validate(profile, Path(args.root))
    raise SystemExit(0 if ok else 1)


def cmd_gallery(args) -> None:
    profile = _profile(args)
    root = Path(args.root)
    scores = scoring_mod.score_directory(profile, root)
    for s in scores[: args.grids]:
        review_mod.grid(profile, root / s.name, show_donor=args.show_donor)
    out = review_mod.gallery(profile, root, scores)
    print(f"wrote {out} and per-recipe sheets, plus grids for the top {min(args.grids, len(scores))} runs")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="failuregen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--profile", required=True, help="robot/dataset profile YAML")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="what the annotations contain")
    s.add_argument("--collection", required=True)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("donors", help="failures whose telemetry corroborates the label")
    s.add_argument("--collection", required=True)
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--out", help="write a donor label file")
    s.set_defaults(func=cmd_donors)

    s = sub.add_parser("plan", help="expand recipes into a manifest")
    s.add_argument("--collection", required=True)
    s.add_argument("--recipes", required=True, help="JSON list of recipes")
    s.add_argument("--out", required=True)
    s.add_argument("--n-hosts", type=int, default=3)
    s.add_argument("--cap", type=int, help="stop after this many injections, keeping the spread")
    s.add_argument("--per-host", type=int, help="cap injections per host, rotating families and donors between hosts")
    s.add_argument(
        "--donors",
        nargs="*",
        help="donor files to draw from instead of scanning the annotations, for "
        "hand-curated windows; may be given more than once",
    )
    s.add_argument(
        "--include-annotated",
        action="store_true",
        help="also include the corroborated annotated donors alongside --donors",
    )
    s.add_argument("--require-phases", nargs="*")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("run", help="render a manifest (needs a model)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out-root", required=True)
    s.add_argument("--checkpoint")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--shard", type=int, default=0, help="this worker's index, for parallel runs")
    s.add_argument("--shards", type=int, default=1)
    s.add_argument(
        "--claim-timeout",
        type=float,
        default=900.0,
        help="seconds after which a claim with no output is treated as abandoned",
    )
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="score rendered runs against their controls")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("export", help="build a LeRobot dataset from rendered runs")
    s.add_argument("--root", required=True, help="directory of rendered run dirs")
    s.add_argument("--out", required=True, help="dataset root to write")
    s.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. you/induced_failures_v1")
    s.add_argument("--n", type=int, default=100, help="how many failures to select")
    s.add_argument(
        "--threshold",
        type=float,
        default=scoring_mod.RATIO_THRESHOLD,
        help="quality gate: how many times its control a clip must change",
    )
    s.add_argument("--per-host", type=int, help="cap failures contributed by any one host episode")
    s.add_argument(
        "--real-successes",
        metavar="COLLECTION",
        help="also include the recorded clean episodes of this collection, resampled "
        "and resized to match the generated clips",
    )
    s.add_argument("--require-phases", nargs="*", help="phase chain a recorded episode must have to be included")
    s.add_argument("--checkpoint", help="label recorded in the provenance metadata")
    s.add_argument(
        "--no-controls", action="store_true", help="omit the matched controls; the dataset then has no negatives"
    )
    s.add_argument(
        "--no-verify", action="store_true", help="skip the action round trip (needs the model framework otherwise)"
    )
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("validate-export", help="check a written dataset before publishing it")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_validate_export)

    s = sub.add_parser("gallery", help="grids and contact sheets for review")
    s.add_argument("--root", required=True)
    s.add_argument("--grids", type=int, default=12)
    s.add_argument("--show-donor", action="store_true")
    s.set_defaults(func=cmd_gallery)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
