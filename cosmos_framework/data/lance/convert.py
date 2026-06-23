# SPDX-License-Identifier: OpenMDW-1.1
"""Convert a Cosmos-format DROID LeRobot dataset to LanceDB.

Thin wrapper over the ``lerobot-lancedb`` converters (the project's
recommended drop-in path), which already implement the LanceDB-optimal
streaming ``RecordBatchReader`` writer and the inline image/video layout:

* ``jpeg`` (:func:`lerobot_lancedb.convert_to_lance`) — per-frame JPEG blobs,
  decoded with NVJPEG on GPU. Max throughput; lossy re-encode.
* ``video`` (:func:`lerobot_lancedb.convert_to_lance_video`) — original mp4
  bytes (Lance blob v2), decoded on the fly with torchcodec. Bit-exact vs the
  base loader; used for equivalence.
"""
from __future__ import annotations

from pathlib import Path


def convert(
    root: str,
    output: str,
    *,
    mode: str = "jpeg",
    table_name: str = "droid",
    jpeg_quality: int = 95,
    tolerance_s: float = 2e-4,
    overwrite: bool = True,
) -> Path:
    from lerobot_lancedb import convert_to_lance, convert_to_lance_video

    repo_id = f"local/{Path(root).parent.name}"
    if mode == "jpeg":
        return convert_to_lance(
            repo_id,
            output,
            src_root=root,
            table_name=table_name,
            jpeg_quality=jpeg_quality,
            tolerance_s=tolerance_s,
            overwrite=overwrite,
        )
    if mode == "video":
        return convert_to_lance_video(
            repo_id,
            output,
            src_root=root,
            table_name=table_name,
            tolerance_s=tolerance_s,
            overwrite=overwrite,
        )
    raise ValueError(f"mode must be 'jpeg' or 'video', got {mode!r}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Cosmos-format DROID success dir")
    ap.add_argument("--output", required=True, help="output LanceDB dir")
    ap.add_argument("--mode", choices=["jpeg", "video"], default="jpeg")
    ap.add_argument("--table", default="droid")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()
    out = convert(
        args.root, args.output, mode=args.mode, table_name=args.table, jpeg_quality=args.jpeg_quality
    )
    print(f"wrote {args.mode} table '{args.table}' at {out}")


if __name__ == "__main__":
    main()
