# SPDX-License-Identifier: OpenMDW-1.1
"""Build a training-optimized vision-SFT representation for LanceDB.

One row per SFT clip: decode once, resize to the training resolution exactly as
SFTDataset.process_one_sample does (crop left to decode time), re-encode with a
short GOP (all-intra by default, so window seeks are exact), and store the mp4
plus caption/sizing metadata. This moves the per-epoch resize offline; only the
H.264 re-encode is lossy, and captions are stored verbatim so tokenization stays
byte-identical to the base loader.

Schema: clip_id, width/height (orig), start_frame/end_frame/temporal_interval,
enc_h/enc_w (stored size), fps, caption_json, caption, video_bytes (large_binary).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import lancedb
import numpy as np
import pyarrow as pa

from cosmos_framework.data.vfm.local_datasets.helper import (
    ffmpeg_decode_video,
    get_aspect_ratio,
    get_video_metadata,
)
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.inference.structured_caption import CAPTION_JSON_KEY


def _encode(frames_thwc_u8: np.ndarray, fps: int, gop: int) -> bytes:
    """Raw RGB frames -> H.264 mp4 bytes via ffmpeg (short GOP, faststart).

    mp4+faststart needs seekable output, so encode to a temp file then read.
    Byte-for-byte the encode path of ``build_composed_droid._encode``."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="SFT video_dataset_file.jsonl")
    ap.add_argument("--uri", required=True, help="output LanceDB dir")
    ap.add_argument("--table", default="vision_sft")
    ap.add_argument("--resolution", default="256")
    ap.add_argument("--gop", type=int, default=1, help="keyframe interval (1=all-intra)")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.jsonl))
    output_sizes = VIDEO_RES_SIZE_INFO[args.resolution]

    # video_bytes is plain large_binary. TODO: move to blob-v2 after optimizations.
    schema = pa.schema(
        [
            pa.field("clip_id", pa.string()),
            pa.field("width", pa.int64()),
            pa.field("height", pa.int64()),
            pa.field("start_frame", pa.int64()),
            pa.field("end_frame", pa.int64()),
            pa.field("temporal_interval", pa.int64()),
            pa.field("enc_h", pa.int64()),
            pa.field("enc_w", pa.int64()),
            pa.field("fps", pa.float64()),
            pa.field("caption_json", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("video_bytes", pa.large_binary()),
        ]
    )

    rows = []
    with open(args.jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            for win_idx, window in enumerate(rec["t2w_windows"]):
                rows.append((rec, win_idx, window))

    def _gen():
        for rec, win_idx, window in rows:
            vp = rec["vision_path"]
            vp = vp if ("://" in vp or vp.startswith("/")) else os.path.join(base_dir, vp)
            input_w, input_h = rec["width"], rec["height"]
            aspect_ratio = get_aspect_ratio(input_w, input_h)
            target_w, target_h = output_sizes[aspect_ratio]
            resize_ratio = max(target_w / input_w, target_h / input_h)
            resize_h, resize_w = (round(input_h * resize_ratio), round(input_w * resize_ratio))

            meta = get_video_metadata(vp)
            fps = int(round(meta["fps"]))
            # decode the WHOLE clip at training resolution (resize only; the crop is
            # done at decode time so the stored clip is a clean rectangle).
            frames = list(ffmpeg_decode_video(vp, scale_hw=(resize_h, resize_w), num_threads=2))
            thwc = np.ascontiguousarray(np.stack(frames, axis=0))  # [T, resize_h, resize_w, 3]
            vb = _encode(thwc, fps, args.gop)

            cj = window.get(CAPTION_JSON_KEY)
            cj_str = json.dumps(cj, ensure_ascii=False) if cj is not None else ""
            caption = str(window.get("caption", ""))
            clip_id = f"{rec['uuid']}_w{win_idx}"
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([clip_id], pa.string()),
                    pa.array([input_w], pa.int64()),
                    pa.array([input_h], pa.int64()),
                    pa.array([window["start_frame"]], pa.int64()),
                    pa.array([window["end_frame"]], pa.int64()),
                    pa.array([window["temporal_interval"]], pa.int64()),
                    pa.array([resize_h], pa.int64()),
                    pa.array([resize_w], pa.int64()),
                    pa.array([float(meta["fps"])], pa.float64()),
                    pa.array([cj_str], pa.string()),
                    pa.array([caption], pa.string()),
                    pa.array([vb], pa.large_binary()),
                ],
                schema=schema,
            )

    reader = pa.RecordBatchReader.from_batches(schema, _gen())
    db = lancedb.connect(args.uri)
    if args.table in [t for t in db.table_names()]:
        db.drop_table(args.table)
    db.create_table(args.table, data=reader, schema=schema)
    t = db.open_table(args.table)
    print(f"wrote {args.table}: {t.count_rows()} clips (gop={args.gop}, res={args.resolution}) at {args.uri}")


if __name__ == "__main__":
    main()
