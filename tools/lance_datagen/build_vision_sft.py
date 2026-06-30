# SPDX-License-Identifier: OpenMDW-1.1
"""Build a training-optimized vision-SFT video representation for LanceDB.

For each clip in an SFT ``video_dataset_file.jsonl`` (the official
``captions_to_sft_jsonl`` output), decode the clip once, **resize it to the
training resolution** exactly as ``SFTDataset.process_one_sample`` does (the
resize-ratio that ``VIDEO_RES_SIZE_INFO`` implies — the spatial center-crop is
left to decode time so the stored clip stays a clean rectangle), re-encode the
resized clip with a tiny GOP (all-intra by default) and store it as one per-clip
large_binary row alongside the clip's caption + sizing metadata.

Why (mirrors ``build_composed_droid.py`` for the action loader):
  * the base loader decodes each source clip at its native size, then resizes
    *per sample, every epoch*. Storing the clip already at training resolution
    moves that resize offline (do it once), so the hot path decodes fewer pixels.
  * a short GOP (``gop=1``) makes the random window seek the Lance loader does
    cheap (every frame is a keyframe -> ``seek_mode="approximate"`` is exact).
  * still fully video-encoded — no per-frame JPEG / disk blowup.

The resize is the base loader's exact op (same ``scale_hw``); only the H.264
re-encode is lossy, so the decoded frames match the base within re-encode
tolerance. The caption + window metadata are stored verbatim so tokenization on
the Lance side is byte-identical.

Schema (one row per clip):
  clip_id (str), width/height (orig int64), start_frame/end_frame/temporal_interval
  (int64), enc_h/enc_w (resized stored size int64), fps (float64),
  caption_json (str, JSON or ""), caption (str dense backup),
  video_bytes (large_binary).
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
    ap.add_argument("--jsonl", required=True, help="SFT video_dataset_file.jsonl")
    ap.add_argument("--uri", required=True, help="output LanceDB dir")
    ap.add_argument("--table", default="vision_sft")
    ap.add_argument("--resolution", default="256")
    ap.add_argument("--gop", type=int, default=1, help="keyframe interval (1=all-intra)")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.jsonl))
    output_sizes = VIDEO_RES_SIZE_INFO[args.resolution]

    # video_bytes is plain large_binary, read via the Permutation API — fastest for our small
    # (<~2MB) clips. TODO: blob-v2 is faster for larger per-row payloads (>=~8-16MB) when read
    # in parallel; switch the storage + loader together if clip sizes grow.
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
