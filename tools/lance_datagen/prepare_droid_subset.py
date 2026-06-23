# SPDX-License-Identifier: OpenMDW-1.1
"""Materialize a small Cosmos-canonical DROID subset from the public
``lerobot/droid_1.0.1`` LeRobot v3.0 dataset.

The public release names a few features differently from what
``cosmos_framework.data.vfm.action.datasets.DROIDLeRobotDataset`` expects.
This script renames them and writes a self-contained ``<out>/success`` tree
(``meta/``, ``data/``, ``videos/``) that the base Cosmos loader reads as-is,
so the base and the LanceDB loader run on byte-identical inputs.

The (large, concatenated) source mp4s are symlinked, not copied — episode
``from_timestamp`` offsets index into them unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# public droid_1.0.1 name -> Cosmos-canonical name
VIDEO_KEY_MAP = {
    "observation.images.wrist_left": "observation.image.wrist_image_left",
    "observation.images.exterior_1_left": "observation.image.exterior_image_1_left",
    "observation.images.exterior_2_left": "observation.image.exterior_image_2_left",
}
COLUMN_MAP = {"observation.state.joint_position": "observation.state.joint_positions"}

# Reserved meta columns + the numeric features the Cosmos DROID loader uses
# (post-rename names). We prune everything else (string metadata, velocities,
# extrinsics, …) so the data parquet and info.json stay consistent and the
# downstream lerobot-lancedb converter only sees numeric + video features.
RESERVED = ["index", "episode_index", "frame_index", "task_index", "timestamp"]
NUMERIC = [
    "observation.state.cartesian_position",
    "observation.state.joint_positions",
    "observation.state.gripper_position",
    "action.joint_position",
    "action.gripper_position",
]
DATA_COLS = RESERVED + NUMERIC
# Features kept in info.json (numeric + the 3 renamed video keys).
KEEP_FEATURES = set(DATA_COLS) | set(VIDEO_KEY_MAP.values())


def _rename_info(info: dict) -> dict:
    feats = {}
    for k, v in info["features"].items():
        nk = VIDEO_KEY_MAP.get(k, COLUMN_MAP.get(k, k))
        if nk in KEEP_FEATURES:
            feats[nk] = v
    info = dict(info)
    info["features"] = feats
    return info


def _rename_table(tbl: pa.Table, mapping: dict[str, str]) -> pa.Table:
    return tbl.rename_columns([mapping.get(n, n) for n in tbl.column_names])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="droid_1.0.1 root (has meta/ data/ videos/)")
    ap.add_argument("--out", required=True, help="output root; writes <out>/success/")
    ap.add_argument("--num-episodes", type=int, default=100)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) / "success"
    n = args.num_episodes

    info = json.loads((src / "meta" / "info.json").read_text())

    # ---- data: keep episodes [0, n); keep all columns (rename only) so the
    # data parquet and info.json features stay consistent for downstream
    # converters (lerobot-lancedb iterates every declared feature). ----
    data = pq.read_table(src / "data" / "chunk-000" / "file-000.parquet")
    data = _rename_table(data, COLUMN_MAP)
    data = data.select(DATA_COLS)
    data = data.filter(pc.less(data["episode_index"], n))
    n_frames = data.num_rows

    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(data, out / "data" / "chunk-000" / "file-000.parquet")

    # ---- episode meta: keep [0, n), drop bulky stats/*, rename video keys ----
    ep = pq.read_table(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep = ep.filter(pc.less(ep["episode_index"], n))
    col_map = {}
    for old, new in VIDEO_KEY_MAP.items():
        for suf in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            col_map[f"videos/{old}/{suf}"] = f"videos/{new}/{suf}"
    keep_cols = [c for c in ep.column_names if not c.startswith("stats/")]
    ep = ep.select(keep_cols)
    ep = _rename_table(ep, col_map)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(ep, out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # ---- tasks: normalize to Cosmos schema (columns: task_index, task) ----
    tasks = pq.read_table(src / "meta" / "tasks.parquet")
    task_col = "task" if "task" in tasks.column_names else "__index_level_0__"
    tasks = pa.table(
        {"task_index": tasks["task_index"], "task": tasks[task_col].cast(pa.string())}
    )
    pq.write_table(tasks, out / "meta" / "tasks.parquet")
    info = _rename_info(info)
    info["total_episodes"] = n
    info["total_frames"] = n_frames
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # ---- videos: symlink concatenated source mp4s under Cosmos key dirs ----
    for old, new in VIDEO_KEY_MAP.items():
        dst = out / "videos" / new / "chunk-000"
        dst.mkdir(parents=True, exist_ok=True)
        srcmp4 = (src / "videos" / old / "chunk-000" / "file-000.mp4").resolve()
        link = dst / "file-000.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(srcmp4)

    print(f"wrote {out}  ({n} episodes, {n_frames} frames)")


if __name__ == "__main__":
    main()
