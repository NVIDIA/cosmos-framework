#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Train / held-out leakage check for the LIBERO-10 FD recipe (CPU only).
Checks: (1) episode-id disjointness and window coverage of the dataset split; (2) exact duplicate action
sequences across splits (hash of the 7-D action array); (3) nearest-neighbour first-frame distance of each
held-out episode to the training first frames (and within-train baseline); (4) normalisation statistics:
bundled quantile_rot q01/q99 vs the same quantiles recomputed on the TRAIN split only (frame-wise 10-D).
Writes a JSON report."""
import argparse, glob, hashlib, json
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--holdout", required=True)
ap.add_argument("--stats", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
kw = dict(root=a.root, fps=20, chunk_length=16, mode="forward_dynamics", camera_mode="concat_view", image_size=256,
          action_space="frame_wise_relative", rotation_space="6d", pose_coordinate_frame="native",
          action_normalization="quantile_rot", holdout_episodes_path=a.holdout)
tr, va, full = (LIBEROLeRobotDataset(split=s, **kw) for s in ("train", "val", "full"))
tr_ids, va_ids = set(tr._ep_vals.tolist()), set(va._ep_vals.tolist())
rep = {"split_rule": "episode-level; 19 ids listed in data/holdout_19.txt (5 %, stratified by task, seed 0); split='train' excludes them, split='val' selects them",
       "n_train_episodes": len(tr_ids), "n_heldout_episodes": len(va_ids), "n_total_episodes": len(full._ep_vals),
       "id_overlap": sorted(tr_ids & va_ids), "union_equals_full": (tr_ids | va_ids) == set(full._ep_vals.tolist()),
       "train_windows": len(tr), "heldout_windows": len(va), "full_windows": len(full)}
# windows never span episodes: per-episode window count == length - chunk
lens = {int(e): int(np.sum(full._row_episode == e)) for e in full._ep_vals.tolist()}
rep["windows_equal_sum_len_minus_chunk"] = (sum(max(0, lens[e] - 16) for e in tr_ids) == len(tr)) and (sum(max(0, lens[e] - 16) for e in va_ids) == len(va))
rep["train_frames"] = int(sum(lens[e] for e in tr_ids)); rep["heldout_frames"] = int(sum(lens[e] for e in va_ids))
# per-task demo counts
task_of = {int(e): int(full._row_task[np.flatnonzero(full._row_episode == e)[0]]) for e in full._ep_vals.tolist()}
rep["train_demos_per_task"] = {t: sum(1 for e in tr_ids if task_of[e] == t) for t in sorted(set(task_of.values()))}
rep["heldout_demos_per_task"] = {t: sum(1 for e in va_ids if task_of[e] == t) for t in sorted(set(task_of.values()))}
# (2) duplicate action sequences across splits
def ep_hash(ds, e):
    rows = np.flatnonzero(ds._row_episode == e); return hashlib.sha1(np.ascontiguousarray(ds._row_action[rows]).tobytes()).hexdigest()
h_tr = {ep_hash(full, e): e for e in tr_ids}; dup = [(h_tr[h], e) for e in va_ids for h in [ep_hash(full, e)] if h in h_tr]
rep["duplicate_action_sequences_train_vs_heldout"] = dup
# (3) first-frame nearest neighbour (third-person + wrist concat, 64x128 grayscale) held-out -> train
def first_frame(ds, e):
    rows = np.flatnonzero(ds._row_episode == e); ts = [float(ds._row_timestamp[rows[0]])]
    v = ds._load_video(ds._episodes[int(e)], ts)[0]  # [3,256,512]
    g = v.mean(0, keepdim=True); g = torch.nn.functional.avg_pool2d(g[None], 4)[0, 0]  # [64,128]
    return g.flatten().numpy()
F_tr = {e: first_frame(full, e) for e in sorted(tr_ids)}; F_va = {e: first_frame(full, e) for e in sorted(va_ids)}
TR = np.stack([F_tr[e] for e in sorted(tr_ids)]); tr_list = sorted(tr_ids)
def nn_dist(x, pool, exclude_idx=None):
    d = np.sqrt(((pool - x) ** 2).mean(1));
    if exclude_idx is not None: d[exclude_idx] = np.inf
    j = int(np.argmin(d)); return float(d[j]), j
ho = {int(e): nn_dist(F_va[e], TR) for e in sorted(va_ids)}
same_task = {e: task_of[tr_list[j]] == task_of[e] for e, (_, j) in ho.items()}
within = [nn_dist(F_tr[e], TR, exclude_idx=i)[0] for i, e in enumerate(tr_list)]
rep["first_frame_nn_rmse_heldout_to_train"] = {e: {"rmse": round(d, 4), "nn_train_episode": tr_list[j], "nn_same_task": bool(same_task[e])} for e, (d, j) in ho.items()}
rep["first_frame_nn_rmse_summary"] = {"heldout_to_train_min": round(min(d for d, _ in ho.values()), 4), "heldout_to_train_median": round(float(np.median([d for d, _ in ho.values()])), 4),
                                     "train_to_train_min": round(min(within), 4), "train_to_train_median": round(float(np.median(within)), 4),
                                     "note": "pixel RMSE on [0,1] 64x128 grayscale; identical frames would give 0; LIBERO episodes of one task share the scene layout so small distances are expected"}
# (4) normalisation statistics: bundled vs train-only recomputation (frame-wise 10-D, raw rotation stats = 'global_raw')
bund = json.load(open(a.stats)); meta = bund.get("metadata", {}); b = bund["global_raw"]
rows_tr = np.flatnonzero(np.isin(full._row_episode, list(tr_ids)))
A = full._build_frame_wise_action(full._row_action[rows_tr]).numpy()  # [N,10]
q01, q99 = np.percentile(A, 1, axis=0), np.percentile(A, 99, axis=0)
rng_b = np.array(b["q99"]) - np.array(b["q01"]); rng_t = q99 - q01
rep["normalisation_stats"] = {
    "bundled_file": a.stats, "bundled_metadata": meta,
    "bundled_q01": [round(float(x), 5) for x in b["q01"]], "bundled_q99": [round(float(x), 5) for x in b["q99"]],
    "train_only_q01": [round(float(x), 5) for x in q01], "train_only_q99": [round(float(x), 5) for x in q99],
    "max_abs_diff_q01_over_range": round(float(np.max(np.abs(q01 - np.array(b["q01"])) / np.maximum(rng_b, 1e-8))), 5),
    "max_abs_diff_q99_over_range": round(float(np.max(np.abs(q99 - np.array(b["q99"])) / np.maximum(rng_b, 1e-8))), 5),
    "train_only_frames_used": int(len(A)),
}
json.dump(rep, open(a.out, "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("n_train_episodes", "n_heldout_episodes", "id_overlap", "union_equals_full", "train_windows", "heldout_windows", "windows_equal_sum_len_minus_chunk", "duplicate_action_sequences_train_vs_heldout", "first_frame_nn_rmse_summary")}, indent=1))
print("norm stats:", json.dumps({k: rep["normalisation_stats"][k] for k in ("bundled_metadata", "max_abs_diff_q01_over_range", "max_abs_diff_q99_over_range", "train_only_frames_used")}, indent=1))
