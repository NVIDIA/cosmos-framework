# SPDX-License-Identifier: OpenMDW-1.1
"""VLM dataloader throughput: the GENUINE cosmos VLM base vs the LanceDB loader.

The base is the shipped source factory ``get_llava_ov_streaming`` (imported, not
reconstructed) — ``lmms-lab/LLaVA-OneVision-Data`` streamed from the HuggingFace Hub
(``streaming=True`` + the same image/conversation filter), which is cosmos's actual
default VLM read pattern (sequential shard reads + a bounded shuffle buffer, no random
access). Cosmos has no local/S3 VLM base, so this is the base in every regime.

  base   — get_llava_ov_streaming(subset): HF-Hub streaming IterableDataset
  lance  — LanceVLMDataset (Permutation API: O(1) random access + true global shuffle)
           or LanceVLMShuffleScan (chunked-shuffle columnar scan — the S3 pattern)

Both paths feed the SAME tokenize+image-process step (a faithful stand-in for cosmos
``VLMProcessor``), so only the data-access layer differs. Two measurements: raw access
(no processing — isolates the access bottleneck) and end-to-end (with processing).
"""
from __future__ import annotations

import argparse
import io
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor

from cosmos_framework.configs.base.vlm.experiment.llava_ov_vlm import get_llava_ov_streaming
from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset, LanceVLMShuffleScan


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


# ── base: the genuine cosmos VLM source (HF-Hub streaming) ──────────────
class GenuineVLMBase(torch.utils.data.IterableDataset):
    """Cosmos's actual default VLM base: ``get_llava_ov_streaming`` from the shipped
    config module. Built fresh in __iter__ (the HF filter lambda isn't picklable for
    spawn workers), yielding the raw ``{id, image(PIL), conversations}`` dict."""

    def __init__(self, subset: str):
        self.subset = subset

    def __iter__(self):
        yield from get_llava_ov_streaming(subset=self.subset)


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


def _build_loader(side, a):
    """Build the (loader, label) for one side from a plain args-dict ``a``."""
    collate = Collate(a["mode"])
    kw = dict(batch_size=a["batch_size"], num_workers=a["num_workers"], collate_fn=collate,
              persistent_workers=a["num_workers"] > 0,
              prefetch_factor=4 if a["num_workers"] > 0 else None)
    so = {"region": a["region"]} if a["region"] else None
    if side == "base":
        return torch.utils.data.DataLoader(
            GenuineVLMBase(a["subset"]), multiprocessing_context="spawn" if a["num_workers"] > 0 else None, **kw
        ), "hf-stream"
    if a["lance_scan"]:
        ds = LanceVLMShuffleScan(a["lance_uri"], a["lance_table"], storage_options=so, buffer_size=1000)
        return torch.utils.data.DataLoader(ds, **kw), "lance-scan"
    ds = LanceVLMDataset(a["lance_uri"], a["lance_table"], storage_options=so)
    g = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(ds, generator=g)
    return torch.utils.data.DataLoader(
        ds, sampler=sampler, multiprocessing_context="spawn" if a["num_workers"] > 0 else None, **kw
    ), "lance-random"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="figureqa(cauldron,llava_format)",
                    help="lmms-lab/LLaVA-OneVision-Data subset for the genuine HF-streaming base")
    ap.add_argument("--lance-uri", required=True)
    ap.add_argument("--lance-table", default="llava")
    ap.add_argument("--region", default=None, help="storage_options region for an s3:// lance-uri")
    ap.add_argument("--lance-scan", action="store_true",
                    help="use chunked-shuffle sequential scan (right for S3) instead of random point-lookups")
    ap.add_argument("--side", choices=["base", "lance"], required=True,
                    help="measure ONE side per process (run twice + divide) — each backend torn down in "
                    "its own process avoids the HF/lance C++ finalization crashes of an in-process compare")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--mode", choices=["raw", "e2e"], default="raw")
    args = ap.parse_args()

    a = vars(args)
    loader, label = _build_loader(args.side, a)
    sps = _measure(loader, num_batches=args.num_batches, warmup=args.warmup, batch_size=args.batch_size)
    print(f"VLM_RESULT side={args.side} label={label} mode={args.mode} workers={args.num_workers} samples_per_s={sps:.1f}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # skip the HF/lance C++ teardown SIGABRT (result already printed)
