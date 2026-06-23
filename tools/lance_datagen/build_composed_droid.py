# SPDX-License-Identifier: OpenMDW-1.1
"""Build a training-optimized DROID video representation for LanceDB.

For each episode, compose the 3 camera views EXACTLY as the base loader does
(wrist on top; the two exteriors resized to half and concatenated on the
bottom -> 270x320), then re-encode that single composed clip with a tiny GOP
(all-intra by default) and store it as one per-episode blob-v2 row.

Why: the base loader decodes 3 full-resolution views + resizes + concatenates
*per sample*. Decoding one pre-composed, pre-resized, short-GOP clip is far less
work — fewer pixels, one stream, no resize/concat, and short-GOP makes random
window seeks cheap. Still fully video-encoded (no per-frame JPEG / disk blowup).
The composition is byte-for-byte the base's; only the H.264 re-encode is lossy.
"""
from __future__ import annotations

import argparse
import subprocess

import lancedb
import numpy as np
import pyarrow as pa
import torch

_BLOB = {b"lance-encoding:blob": b"true"}


def _encode(frames_thwc_u8: np.ndarray, fps: int, gop: int) -> bytes:
    """Raw RGB frames -> H.264 mp4 bytes via ffmpeg (short GOP, faststart).

    mp4+faststart needs seekable output, so encode to a temp file then read."""
    import os
    import tempfile

    t, h, w, _ = frames_thwc_u8.shape
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "veryfast", "-g", str(gop), "-keyint_min", str(gop),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
        ]
        subprocess.run(cmd, input=frames_thwc_u8.tobytes(), stdout=subprocess.DEVNULL, check=True)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Cosmos-format DROID success dir")
    ap.add_argument("--uri", required=True, help="output LanceDB dir")
    ap.add_argument("--table", default="droid_composed")
    ap.add_argument("--gop", type=int, default=1, help="keyframe interval (1=all-intra)")
    args = ap.parse_args()

    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    base = DROIDLeRobotDataset(
        root=args.root, action_space="joint_pos", use_state=True, mode="policy", chunk_length=16
    )
    fps = int(round(base._fps))
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("ep_start", pa.int64()),
            pa.field("length", pa.int64()),
            pa.field("video_bytes", pa.large_binary(), metadata=_BLOB),
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
    if args.table in [t for t in db.table_names()]:
        db.drop_table(args.table)
    db.create_table(args.table, data=reader, schema=schema)
    t = db.open_table(args.table)
    print(f"wrote {args.table}: {t.count_rows()} episodes (gop={args.gop}, fps={fps}) at {args.uri}")


if __name__ == "__main__":
    main()
