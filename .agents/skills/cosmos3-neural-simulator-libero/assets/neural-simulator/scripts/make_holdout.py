#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Select the held-out LIBERO-10 episodes (5 %, 19 of 379) and write data/holdout_19.txt.

Selection: stratified by task_index (proportional allocation, largest remainder),
uniform random within each task, seed 0. Written once; training and eval both read
the file, so the split is fixed by the file, not by code.
"""

import argparse
import glob
import json
import random

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

eps = pd.concat(
    [
        pd.read_parquet(p, columns=["episode_index", "tasks", "length"])
        for p in sorted(glob.glob(f"{a.root}/meta/episodes/chunk-*/file-*.parquet"))
    ]
)
d = pd.concat(
    [
        pd.read_parquet(p, columns=["episode_index", "task_index"])
        for p in sorted(glob.glob(f"{a.root}/data/chunk-*/file-*.parquet"))
    ]
)
ep_task = d.groupby("episode_index").task_index.first()
assert (d.groupby("episode_index").task_index.nunique() == 1).all(), "episode spans multiple tasks"
tasks = pd.read_parquet(f"{a.root}/meta/tasks.parquet")
task_text = {int(v): str(k) for k, v in zip(tasks.index, tasks["task_index"])}
n_total = len(ep_task)
n_hold = int(round(n_total * a.frac))
# proportional allocation with largest remainder
counts = ep_task.value_counts().sort_index()
quota = counts / n_total * n_hold
alloc = np.floor(quota).astype(int)
rem = n_hold - alloc.sum()
order = (quota - alloc).sort_values(ascending=False).index.tolist()
for t in order[:rem]:
    alloc[t] += 1
assert alloc.sum() == n_hold
rng = random.Random(a.seed)
chosen = []
for t, k in alloc.items():
    ids = sorted(int(e) for e in ep_task[ep_task == t].index)
    chosen += sorted(rng.sample(ids, int(k)))
chosen = sorted(chosen)
lens = eps.set_index("episode_index")["length"]
with open(a.out, "w") as f:
    f.write(
        f"# LIBERO-10 held-out episodes: {n_hold} of {n_total} ({a.frac:.0%}), stratified by task_index, seed={a.seed}\n"
    )
    f.write(f"# root={a.root}\n# columns: episode_index task_index length task\n")
    for e in chosen:
        f.write(f"{e}\t{int(ep_task[e])}\t{int(lens[e])}\t{task_text[int(ep_task[e])]}\n")
print(
    json.dumps(
        {
            "n_total": n_total,
            "n_hold": n_hold,
            "per_task": {int(k): int(v) for k, v in alloc.items()},
            "episodes": chosen,
            "frames_heldout": int(lens[chosen].sum()),
            "frames_total": int(lens.sum()),
        }
    )
)
