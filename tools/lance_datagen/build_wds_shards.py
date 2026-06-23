# SPDX-License-Identifier: OpenMDW-1.1
"""Write a LLaVA-OneVision subset as WebDataset tar shards — the canonical
cosmos VLM data format (Eagle ``wdinfo.json``-indexed tar shards, read via
``webdataset.WebLoader``). Each sample is ``{key}.png`` + ``{key}.json``.

This is the baseline that the LanceDB VLM loader replaces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import webdataset as wds
from datasets import Image, load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="figureqa(cauldron,llava_format)")
    ap.add_argument("--out", required=True, help="output dir for shard-*.tar")
    ap.add_argument("--maxcount", type=int, default=5000, help="samples per shard")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=args.subset, split="train")
    ds = ds.cast_column("image", Image(decode=False))  # raw encoded bytes

    pattern = str(out / "shard-%05d.tar")
    n = 0
    with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
        for i, rec in enumerate(ds):
            img = rec.get("image") or {}
            raw = img.get("bytes")
            if not raw:
                continue
            sink.write(
                {
                    "__key__": f"sample{i:08d}",
                    "png": raw,
                    "json": json.dumps(rec.get("conversations") or []).encode(),
                }
            )
            n += 1
    # minimal wdinfo.json (cosmos Eagle index)
    shards = sorted(p.name for p in out.glob("shard-*.tar"))
    (out / "wdinfo.json").write_text(
        json.dumps({"total_key_count": n, "shards": shards, "data_keys": ["png", "json"]}, indent=2)
    )
    print(f"wrote {n} samples across {len(shards)} shards to {out}")


if __name__ == "__main__":
    main()
