# SPDX-License-Identifier: OpenMDW-1.1
"""VLM dataloader throughput: HF streaming IterableDataset vs LanceDB map-style.

Both paths feed the SAME tokenize + image-process step (a faithful stand-in for
cosmos ``VLMProcessor``), so only the data-access layer differs:

  base-iterable  — datasets ``IterableDataset`` (sequential shards + shuffle
                   buffer, no random access) — the cosmos VLM read pattern
  lance          — LanceVLMDataset (Permutation API: O(1) random access + true
                   global shuffle, columnar batched reads)

Two measurements: raw access (no processing — isolates the access bottleneck)
and end-to-end (with tokenize+image-process — realistic training).
"""
from __future__ import annotations

import argparse
import io
import time

import torch
from PIL import Image


def _decode_image(image):
    if isinstance(image, dict):
        raw = image.get("bytes")
        return Image.open(io.BytesIO(raw)).convert("RGB") if raw else None
    return image.convert("RGB") if image is not None else None


def _sharegpt_to_messages(conversations, image):
    msgs, inserted = [], False
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"].replace("<image>", "").strip()
        if role == "user" and not inserted and image is not None:
            content = [{"type": "image", "image": image}, {"type": "text", "text": text}]
            inserted = True
        else:
            content = text
        msgs.append({"role": role, "content": content})
    return msgs


def make_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


def process(item, processor):
    """Faithful stand-in for VLMProcessor.process: ShareGPT image+convo -> tensors."""
    image = _decode_image(item.get("image"))
    messages = _sharegpt_to_messages(item.get("conversations", []), image)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    return inputs["input_ids"]


def _measure(loader, *, num_batches, warmup, batch_size):
    seen, t0 = 0, None
    for i, _ in enumerate(loader):
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += 1
        if seen >= num_batches:
            break
    dt = time.perf_counter() - t0
    return seen * batch_size / dt


def _wds_to_item(sample):
    import json as _json

    return {
        "id": sample["__key__"],
        "image": {"bytes": sample["png"]},
        "conversations": _json.loads(sample["json"]),
    }


def build_base_wds(shard_urls):
    """Canonical cosmos VLM base: webdataset tar shards (sequential reads +
    shuffle buffer). ``shard_urls`` is a brace pattern of local paths or a
    ``pipe:aws s3 cp ... -`` expression for S3."""
    import webdataset as wds

    return (
        wds.WebDataset(shard_urls, shardshuffle=True, empty_check=False)
        .shuffle(1000)
        .map(_wds_to_item)
    )


# ── base: HF IterableDataset (local cache OR S3 parquet, streaming) ─────
def build_base(name, num_workers, base_parquet=None):
    from datasets import load_dataset

    if base_parquet:
        # stream parquet shards straight from S3 (sequential shard reads, the
        # real webdataset/IterableDataset access pattern at scale)
        ds = load_dataset(
            "parquet", data_files={"train": base_parquet}, split="train", streaming=True
        )
        return ds.shuffle(seed=42, buffer_size=1000)
    ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=name, split="train")
    return ds.to_iterable_dataset(num_shards=max(1, num_workers)).shuffle(seed=42, buffer_size=1000)


_PROC = None


def _get_proc():
    global _PROC
    if _PROC is None:
        _PROC = make_processor()
    return _PROC


class Collate:
    """Module-level (picklable for spawn workers). raw -> ids; e2e -> tokenize."""

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, items):
        if self.mode == "raw":
            return [it.get("id") for it in items]
        proc = _get_proc()
        return [process(it, proc) for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="figureqa(cauldron,llava_format)")
    ap.add_argument("--lance-uri", required=True)
    ap.add_argument("--base-parquet", nargs="+", default=None,
                    help="parquet shard paths/globs (e.g. s3://...) to stream as the base; else HF hub")
    ap.add_argument("--wds-shards", default=None,
                    help="webdataset base: brace pattern of local tar paths or a 'pipe:aws s3 cp ...' expr")
    ap.add_argument("--region", default=None, help="storage_options region for an s3:// lance-uri")
    ap.add_argument("--lance-scan", action="store_true",
                    help="use chunked-shuffle sequential scan (right for S3) instead of random point-lookups")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--mode", choices=["raw", "e2e"], default="raw")
    args = ap.parse_args()

    from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset

    collate = Collate(args.mode)

    base_label = "wds-tar" if args.wds_shards else ("parquet-stream" if args.base_parquet else "hf-iterable")
    print(f"mode={args.mode} batch={args.batch_size} workers={args.num_workers} base={base_label}\n")
    print(f"{'loader':<16}{'samples/s':>12}{'speedup':>10}")

    # base: webdataset tar (canonical) | parquet stream | hf iterable
    if args.wds_shards:
        base_it = build_base_wds(args.wds_shards)
    else:
        base_it = build_base(args.subset, args.num_workers, args.base_parquet)
    base_loader = torch.utils.data.DataLoader(
        base_it, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    base_sps = _measure(base_loader, num_batches=args.num_batches, warmup=args.warmup, batch_size=args.batch_size)
    print(f"{base_label:<16}{base_sps:>12.1f}{'1.00x':>10}")

    # lance: chunked-shuffle scan (IterableDataset) OR random point-lookup
    so = {"region": args.region} if args.region else None
    if args.lance_scan:
        from cosmos_framework.data.lance.vlm_dataset import LanceVLMShuffleScan

        lance_ds = LanceVLMShuffleScan(args.lance_uri, "llava", storage_options=so, buffer_size=1000)
        lance_loader = torch.utils.data.DataLoader(
            lance_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=collate, persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        label = "lance-scan"
    else:
        lance_ds = LanceVLMDataset(args.lance_uri, "llava", storage_options=so)
        g = torch.Generator().manual_seed(42)
        sampler = torch.utils.data.RandomSampler(lance_ds, generator=g)
        lance_loader = torch.utils.data.DataLoader(
            lance_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
            collate_fn=collate, persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
        )
        label = "lance-random"
    lance_sps = _measure(lance_loader, num_batches=args.num_batches, warmup=args.warmup, batch_size=args.batch_size)
    print(f"{label:<16}{lance_sps:>12.1f}{lance_sps / base_sps:>9.2f}x")


if __name__ == "__main__":
    main()
