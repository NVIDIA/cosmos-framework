# SPDX-License-Identifier: OpenMDW-1.1
"""Build a training-optimized DROID representation for LanceDB (video + labels).

For each episode, compose the 3 camera views EXACTLY as the base loader does
(wrist on top; the two exteriors resized to half and concatenated on the
bottom -> 270x320), then re-encode that single composed clip with a tiny GOP
(all-intra by default) and store it as one per-episode large_binary row.

Alongside the video table, three label tables are written so the Lance loader
needs no LeRobot parquet tree at train time:
  {table}_frames   — per-frame labels (episode/task/timestamp + action & state
                     features), dumped verbatim from the base loader's arrays
  {table}_tasks    — task_index -> task string
  {table}_episodes — episode_index -> episode_id (for keep-ranges filtering)

Why: the base loader decodes 3 full-resolution views + resizes + concatenates
*per sample*. Decoding one pre-composed, pre-resized, short-GOP clip is far less
work — fewer pixels, one stream, no resize/concat, and short-GOP makes random
window seeks cheap. Still fully video-encoded (no per-frame JPEG / disk blowup).
The composition is byte-for-byte the base's; only the H.264 re-encode is lossy —
labels roundtrip bit-exact.

``--labels-only`` (re)writes just the label tables against an existing video table.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import lancedb
import numpy as np
import pyarrow as pa
import torch

from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset


def _encode(frames_thwc_u8: np.ndarray, fps: int, gop: int) -> bytes:
    """Raw RGB frames -> H.264 mp4 bytes via ffmpeg (short GOP, faststart).

    mp4+faststart needs seekable output, so encode to a temp file then read."""
    t, h, w, _ = frames_thwc_u8.shape
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path,
        ]
        subprocess.run(cmd, input=frames_thwc_u8.tobytes(), stdout=subprocess.DEVNULL, check=True)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


# Every per-frame feature column either action space reads ('.' -> '__' in Lance).
FEATURE_COLUMNS = [
    "action.joint_position",
    "action.gripper_position",
    "observation.state.joint_positions",
    "observation.state.gripper_position",
    "observation.state.cartesian_position",
]


def lance_col(name: str) -> str:
    return name.replace(".", "__")


def _feature_array(a: np.ndarray) -> pa.Array:
    if a.ndim == 2:
        return pa.FixedSizeListArray.from_arrays(pa.array(a.ravel(), pa.float32()), a.shape[1])
    return pa.array(a, pa.float32())


def _replace(db: lancedb.DBConnection, name: str, data, schema: pa.Schema) -> None:
    if name in db.table_names():
        db.drop_table(name)
    db.create_table(name, data=data, schema=schema)


def write_label_tables(db: lancedb.DBConnection, table: str, root: str) -> None:
    """Dump the base loader's compact label arrays verbatim (bit-exact roundtrip)."""
    jp = DROIDLeRobotDataset(root=root, action_space="joint_pos", use_state=True, mode="policy")
    ee = DROIDLeRobotDataset(root=root, action_space="ee_pose")
    feat = {**ee._feat, **jp._feat}  # union covers both action spaces

    cols = [pa.array(jp._row_episode), pa.array(jp._row_task), pa.array(jp._row_timestamp)]
    names = ["episode_index", "task_index", "timestamp"]
    for c in FEATURE_COLUMNS:
        cols.append(_feature_array(feat[c]))
        names.append(lance_col(c))
    frames = pa.table(cols, names=names)
    _replace(db, f"{table}_frames", frames, frames.schema)

    tasks = pa.table(
        [pa.array(sorted(jp._tasks), pa.int64()), pa.array([jp._tasks[k] for k in sorted(jp._tasks)], pa.string())],
        names=["task_index", "task"],
    )
    _replace(db, f"{table}_tasks", tasks, tasks.schema)

    eps = sorted(jp._episodes)
    episodes = pa.table(
        [
            pa.array(eps, pa.int64()),
            pa.array([str(jp._episodes[e].get("episode_id", "")) for e in eps], pa.string()),
        ],
        names=["episode_index", "episode_id"],
    )
    _replace(db, f"{table}_episodes", episodes, episodes.schema)
    print(
        f"wrote {table}_frames ({frames.num_rows} frames), {table}_tasks ({tasks.num_rows}), {table}_episodes ({episodes.num_rows})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Cosmos-format DROID success dir")
    ap.add_argument("--uri", required=True, help="output LanceDB dir")
    ap.add_argument("--table", default="droid_composed")
    ap.add_argument("--gop", type=int, default=1, help="keyframe interval (1=all-intra)")
    ap.add_argument("--labels-only", action="store_true", help="(re)write label tables only; keep the video table")
    args = ap.parse_args()

    if args.labels_only:
        write_label_tables(lancedb.connect(args.uri), args.table, args.root)
        return

    base = DROIDLeRobotDataset(root=args.root, action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)
    fps = int(round(base._fps))
    # video_bytes is plain large_binary. TODO: move to blob-v2 after optimizations.
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("ep_start", pa.int64()),
            pa.field("length", pa.int64()),
            pa.field("video_bytes", pa.large_binary()),
        ]
    )

    def _rows():
        for pos in range(len(base._ep_vals)):
            ep_index = int(base._ep_vals[pos])
            ep_start = int(base._ep_starts[pos])
            ep_end = ep_start + (
                int(base._ep_starts[pos + 1] - ep_start)
                if pos + 1 < len(base._ep_starts)
                else int(len(base._row_episode) - ep_start)
            )
            episode = base._episodes[ep_index]
            obs = base._window_rows(ep_start, ep_end, ep_index)
            composed = base._load_concat_video(episode, obs)  # (T, C, 270, 320) float[0,1]
            thwc = (composed.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
            thwc = np.ascontiguousarray(thwc)
            vb = _encode(thwc, fps, args.gop)
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([ep_index], pa.int64()),
                    pa.array([ep_start], pa.int64()),
                    pa.array([ep_end - ep_start], pa.int64()),
                    pa.array([vb], pa.large_binary()),
                ],
                schema=schema,
            )

    reader = pa.RecordBatchReader.from_batches(schema, _rows())
    db = lancedb.connect(args.uri)
    _replace(db, args.table, reader, schema)
    t = db.open_table(args.table)
    print(f"wrote {args.table}: {t.count_rows()} episodes (gop={args.gop}, fps={fps}) at {args.uri}")
    write_label_tables(db, args.table, args.root)


if __name__ == "__main__":
    main()
