# SPDX-License-Identifier: OpenMDW-1.1
"""Build a training-optimized DROID representation for LanceDB (video + labels).

For each episode, compose the 3 camera views EXACTLY as the base loader does
(wrist on top; the two exteriors resized to half and concatenated on the
bottom), then re-encode that single composed clip with a tiny GOP (all-intra by
default) and store it as one per-episode large_binary row.

Alongside the video table, three label tables are written so the Lance loader
needs no LeRobot tree at train time:
  {table}_frames   — per-frame labels (episode/task/timestamp + action & state
                     features), dumped verbatim from the base's LeRobot table
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
from torchcodec.decoders import VideoDecoder

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

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


def _encode(frames_thwc_u8: np.ndarray, fps: int, gop: int) -> bytes:
    """Raw RGB frames -> H.264 mp4 bytes via ffmpeg (short GOP, faststart).

    mp4+faststart needs seekable output, so encode to a temp file then read."""
    t, h, w, _ = frames_thwc_u8.shape
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "veryfast", "-g", str(gop), "-keyint_min", str(gop),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
        ]  # fmt: skip
        subprocess.run(cmd, input=frames_thwc_u8.tobytes(), stdout=subprocess.DEVNULL, check=True)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


def _feature_array(a: np.ndarray) -> pa.Array:
    if a.ndim == 2:
        return pa.FixedSizeListArray.from_arrays(pa.array(a.ravel().astype(np.float32)), a.shape[1])
    return pa.array(a.astype(np.float32))


def _replace(db: lancedb.DBConnection, name: str, data, schema: pa.Schema) -> None:
    if name in db.table_names():
        db.drop_table(name)
    db.create_table(name, data=data, schema=schema)


def _build_base(root: str) -> DROIDLeRobotDataset:
    # split="full" + joint_pos/use_state registers every episode and label column.
    return DROIDLeRobotDataset(
        root=root,
        split="full",
        use_success_only=True,
        action_space="joint_pos",
        use_state=True,
        mode="policy",
        chunk_length=16,
    )


def write_label_tables(db: lancedb.DBConnection, table: str, base: DROIDLeRobotDataset) -> None:
    """Dump the base's LeRobot label table verbatim (bit-exact roundtrip)."""
    lr = base._get_dataset(0)
    cols = lr.hf_dataset.with_format("numpy")[:]

    arrays = [
        pa.array(np.asarray(cols["episode_index"]).astype(np.int64)),
        pa.array(np.asarray(cols["task_index"]).astype(np.int64)),
        pa.array(np.asarray(cols["timestamp"]).astype(np.float64)),
    ]
    names = ["episode_index", "task_index", "timestamp"]
    for c in FEATURE_COLUMNS:
        arrays.append(_feature_array(np.asarray(cols[c])))
        names.append(lance_col(c))
    frames = pa.table(arrays, names=names)
    _replace(db, f"{table}_frames", frames, frames.schema)

    tasks_df = lr.meta.tasks  # DataFrame indexed by task string, column task_index
    tasks = pa.table(
        [pa.array(tasks_df["task_index"].astype("int64").tolist()), pa.array([str(t) for t in tasks_df.index])],
        names=["task_index", "task"],
    )
    _replace(db, f"{table}_tasks", tasks, tasks.schema)

    eps_meta = lr.meta.episodes
    n_eps = len(eps_meta)
    ep_ids = eps_meta["episode_id"] if "episode_id" in eps_meta.column_names else [""] * n_eps
    episodes = pa.table(
        [pa.array(list(range(n_eps)), pa.int64()), pa.array([str(e) for e in ep_ids], pa.string())],
        names=["episode_index", "episode_id"],
    )
    _replace(db, f"{table}_episodes", episodes, episodes.schema)
    print(
        f"wrote {table}_frames ({frames.num_rows} frames), {table}_tasks ({tasks.num_rows}), "
        f"{table}_episodes ({episodes.num_rows})"
    )


def _episode_view_frames(base: DROIDLeRobotDataset, lr, episode_id: int, feature: str) -> torch.Tensor:
    """Decode every frame of one episode's view directly from its source mp4."""
    ep = lr.meta.episodes[episode_id]
    n = int(ep["length"])
    chunk = int(ep.get(f"videos/{feature}/chunk_index", ep.get("data/chunk_index", 0)))
    fil = int(ep.get(f"videos/{feature}/file_index", ep.get("data/file_index", 0)))
    from_ts = float(ep.get(f"videos/{feature}/from_timestamp", 0.0))
    rel = lr.meta.info["video_path"].format(
        video_key=feature, chunk_index=chunk, file_index=fil, episode_chunk=chunk, episode_file=fil
    )
    dec = VideoDecoder(str(lr.root / rel), seek_mode="exact")
    ts = [from_ts + i * base._dt for i in range(n)]
    return dec.get_frames_played_at(seconds=ts).data.to(torch.float32) / 255.0  # (T,C,H,W) in [0,1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="versioned DROID LeRobot root (see droid_lerobot_dataset_config)")
    ap.add_argument("--uri", required=True, help="output LanceDB dir")
    ap.add_argument("--table", default="droid_composed")
    ap.add_argument("--gop", type=int, default=1, help="keyframe interval (1=all-intra)")
    ap.add_argument("--labels-only", action="store_true", help="(re)write label tables only; keep the video table")
    args = ap.parse_args()

    db = lancedb.connect(args.uri)
    base = _build_base(args.root)
    if args.labels_only:
        write_label_tables(db, args.table, base)
        return

    lr = base._get_dataset(0)
    fps = int(round(base._fps))
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("ep_start", pa.int64()),
            pa.field("length", pa.int64()),
            # video_bytes is plain large_binary. TODO: move to blob-v2 after optimizations.
            pa.field("video_bytes", pa.large_binary()),
        ]
    )

    def _rows():
        ep_start = 0
        for episode_id in range(len(lr.meta.episodes)):
            views = {f: _episode_view_frames(base, lr, episode_id, f) for f in base._image_features.values()}
            composed = base._compose_multi_view(views)  # (T,C,H,W) in [0,1]
            thwc = (composed.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
            vb = _encode(np.ascontiguousarray(thwc), fps, args.gop)
            n = int(lr.meta.episodes[episode_id]["length"])
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([episode_id], pa.int64()),
                    pa.array([ep_start], pa.int64()),
                    pa.array([n], pa.int64()),
                    pa.array([vb], pa.large_binary()),
                ],
                schema=schema,
            )
            ep_start += n

    reader = pa.RecordBatchReader.from_batches(schema, _rows())
    _replace(db, args.table, reader, schema)
    print(f"wrote {args.table}: {db.open_table(args.table).count_rows()} episodes (gop={args.gop}, fps={fps})")
    write_label_tables(db, args.table, base)


if __name__ == "__main__":
    main()
