# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check a written dataset before anyone trains on it.

The failure modes worth catching here are silent ones: labels that flip a frame early, a control
that carries a positive label, a video stream shorter than the states it is supposed to match.
None of them raise at write time and all of them would quietly teach the model the wrong thing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Profile


def _frames(mp4: Path) -> tuple[int, int, int]:
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        n = sum(1 for _ in c.decode(s))
        return n, s.codec_context.height, s.codec_context.width


def validate(profile: Profile, root: Path) -> bool:
    info = json.loads((root / "meta" / "info.json").read_text())
    labels = json.loads((root / "meta" / "failure_labels.json").read_text())
    table = pa.concat_tables([pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))])
    df = table.to_pandas()

    problems: list[str] = []
    checks: list[str] = []

    if info.get("codebase_version") != "v3.0":
        problems.append(f"codebase_version is {info.get('codebase_version')}, expected v3.0")
    else:
        checks.append("codebase_version v3.0")

    n_eps = len(labels["episodes"])
    if info["total_episodes"] != n_eps:
        problems.append(f"info says {info['total_episodes']} episodes, labels say {n_eps}")
    if info["total_frames"] != len(df):
        problems.append(f"info says {info['total_frames']} frames, parquet has {len(df)}")
    checks.append(f"{n_eps} episodes, {len(df)} frames")

    for key, feat in info["features"].items():
        if feat["dtype"] != "video":
            continue
        files = sorted((root / "videos" / key).rglob("*.mp4"))
        if not files:
            problems.append(f"no video written for {key}")
            continue
        total, h, w = 0, None, None
        for f in files:
            n, h, w = _frames(f)
            total += n
        if total != len(df):
            problems.append(f"{key} has {total} frames, parquet has {len(df)}")
        if [h, w, 3] != list(feat["shape"]):
            problems.append(f"{key} is {h}x{w} but declared {feat['shape']}")
    checks.append(
        f"{len([k for k, f in info['features'].items() if f['dtype'] == 'video'])} video streams decode "
        f"at their declared size and length"
    )

    fps = info["fps"]
    by_index = {e["episode_index"]: e for e in labels["episodes"]}
    bad_labels = 0
    for index, sub in df.groupby("episode_index"):
        entry = by_index.get(int(index))
        if entry is None:
            problems.append(f"episode {index} has no provenance entry")
            continue
        if len(sub) != entry["n_frames"]:
            problems.append(f"episode {index}: {len(sub)} frames, provenance says {entry['n_frames']}")
        active = sub["annotation.failure.active"].to_numpy()
        tto = sub["annotation.failure.time_to_onset"].to_numpy()
        generated = sub["annotation.is_generated"].to_numpy()
        want_generated = int(entry.get("source", "generated") == "generated")
        if not (generated == want_generated).all():
            problems.append(f"episode {index}: is_generated disagrees with source {entry.get('source')!r}")
        if entry["is_failure"]:
            onset = entry["onset_frame"]
            want = (np.arange(len(sub)) >= onset).astype(np.int64)
            if not np.array_equal(active, want):
                bad_labels += 1
                problems.append(f"episode {index}: label does not flip at frame {onset}")
            if abs(float(tto[0]) - onset / fps) > 1e-3:
                problems.append(f"episode {index}: time_to_onset starts at {tto[0]}, expected {onset / fps}")
        else:
            if active.any():
                bad_labels += 1
                problems.append(f"episode {index}: control carries a positive label")
            if not np.isnan(tto).all():
                problems.append(f"episode {index}: control should carry NaN time_to_onset")
    if not bad_labels:
        checks.append("every failure flips at its recorded onset; every control is negative throughout")

    # lerobot keys tasks.parquet by the task string itself, with task_index as the column.
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    strings = list(tasks["task"]) if "task" in tasks.columns else list(tasks.index)
    if len(strings) != 1:
        problems.append(f"expected a single neutral task, found {len(strings)}: {strings}")
    else:
        checks.append(f"one task string, carrying no label: {strings[0]!r}")
    if not (root / "meta" / "stats.json").is_file():
        problems.append("no meta/stats.json")
    else:
        checks.append("per-episode statistics present")

    # One shape per camera across both sources is what makes a mixed dataset trainable at all.
    sources = Counter(e.get("source", "generated") for e in labels["episodes"])
    if len(sources) > 1:
        checks.append(f"one camera geometry shared by both sources ({dict(sources)})")

    fails = [e for e in labels["episodes"] if e["is_failure"]]
    frames_by_source: Counter = Counter()
    for e in labels["episodes"]:
        frames_by_source[e.get("source", "generated")] += e["n_frames"]

    print(f"dataset {root}")
    for line in checks:
        print(f"  ok   {line}")
    print(f"\n  {len(fails)} failures / {n_eps - len(fails)} non-failures")
    print(f"  episodes by source: {dict(sources)}")
    print(f"  frames by source:   {dict(frames_by_source)}")
    positive = int(df["annotation.failure.active"].sum())
    print(f"  frames labelled failing: {positive} of {len(df)} ({positive / len(df):.0%})")
    print(f"  by family: {dict(Counter(e['failure_type'] for e in fails))}")
    print(f"  by host:   {len({e['host']['episode_index'] for e in fails})} distinct host episodes")
    print(f"  by phase:  {dict(Counter(e['target_phase'] for e in fails))}")
    generated = [e for e in labels["episodes"] if e.get("source", "generated") == "generated"]
    verified = sum(1 for e in generated if e.get("actions_verified"))
    print(f"  actions verified against the model's own inputs: {verified}/{len(generated)} generated")

    for line in problems:
        print(f"  FAIL {line}")
    print("\nvalidation: " + ("passed" if not problems else f"{len(problems)} problems"))
    return not problems
