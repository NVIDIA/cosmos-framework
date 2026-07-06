# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference + accuracy/Pearson metrics for a VideoPhy-2-SFT'd Qwen3-VL ckpt.

Two modes share one CLI:

1. **Run + eval** — pass ``--hf_ckpt`` and ``--val_root``. The script loads the
   HF safetensors export, iterates the prepared val manifest, runs batched
   generations, writes one ``<sample_id>.json`` per sample to ``--results_dir``,
   then walks the directory and writes ``summary.json``.

2. **Eval only** — pass only ``--results_dir`` (point at a dir already filled
   by a prior run). Re-reads each JSON, recomputes ``summary.json``. Useful
   when iterating on the score parser without re-running inference.

Multi-GPU is opt-in via ``torchrun`` — every rank loads the model onto its
``LOCAL_RANK`` GPU and processes ``meta[rank::world_size]``. With no torchrun
env vars set, the script runs single-process on ``cuda:0``.

Single-GPU example::

    python -m cosmos_framework.scripts.reasoner.eval_videophy2 \\
        --hf_ckpt $HF_CKPT --val_root $VAL_ROOT --results_dir $OUT

8-GPU data-parallel example::

    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.reasoner.eval_videophy2 \\
        --hf_ckpt $HF_CKPT --val_root $VAL_ROOT --results_dir $OUT --batch_size 2

The inference path here is intentionally lightweight — it is expected to be
replaced by the upstream ``cosmos_framework.inference`` reasoner path once that
supports video conditioning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

_SCORE_RE = re.compile(r"\b([1-5])\b")


# ---------------------------------------------------------------------------
# Score parsing + metrics
# ---------------------------------------------------------------------------


def _parse_score(text):
    """Return the first standalone 1-5 digit in ``text``, or ``None``."""
    if not isinstance(text, str):
        return None
    m = _SCORE_RE.search(text)
    return int(m.group(1)) if m else None


def _pearson(xs, ys):
    """Pearson correlation; returns None if undefined."""
    if len(xs) < 2:
        return None
    try:
        from scipy.stats import pearsonr
        r = float(pearsonr(xs, ys).statistic)
    except ImportError:
        import numpy as np
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if x.std() == 0 or y.std() == 0:
            return None
        r = float(np.corrcoef(x, y)[0, 1])
    return None if math.isnan(r) else r


# ---------------------------------------------------------------------------
# Distributed helpers — opt-in via torchrun env vars
# ---------------------------------------------------------------------------


def _init_distributed():
    """Returns ``(rank, world_size, local_rank)``. Initialises the NCCL process
    group when launched under torchrun (``WORLD_SIZE`` env var > 1)."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 1, 0
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _barrier():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _prepare_sample(sample_meta, val_root):
    """Load one val sample. Returns ``(sample_id, user_turns, gt_response, video_path)``
    or ``None`` if either the media or conversation file is missing."""
    sample_id = sample_meta.get("id") or sample_meta.get("name") or "sample_unknown"
    video_rel = sample_meta.get("media") or sample_meta.get("video")
    text_rel = sample_meta.get("text") or sample_meta.get("conversation")
    video_path = val_root / video_rel
    text_path = val_root / text_rel
    if not video_path.exists() or not text_path.exists():
        return None
    conversation = json.loads(text_path.read_text())["conversations"]
    user_turns = [t for t in conversation if t.get("role") != "assistant"]
    gt_entry = next((t for t in conversation if t.get("role") == "assistant"), None)
    gt_response = (
        gt_entry["content"][0]["text"]
        if gt_entry and isinstance(gt_entry.get("content"), list)
        else None
    )
    # Resolve the "video_0" placeholder in user content to the actual file path.
    for t in user_turns:
        content = t.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "video":
                    c["video"] = str(video_path)
    return sample_id, user_turns, gt_response, video_path


def _run_inference(args, rank, world_size, local_rank):
    """Each rank loads the model onto ``cuda:<local_rank>``, processes
    ``meta[rank::world_size]`` in batches, and writes one JSON per sample
    into ``args.results_dir``."""
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

    val_root = Path(args.val_root)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((val_root / "meta.json").read_text())
    if args.n is not None:
        meta = meta[: args.n]
    shard = meta[rank::world_size]
    if rank == 0:
        print(
            f"[infer] {len(meta)} val samples; sharded across {world_size} rank(s); "
            f"rank0 owns {len(shard)} samples; batch_size={args.batch_size}",
            flush=True,
        )

    device = f"cuda:{local_rank}"
    if rank == 0:
        print(f"[infer] loading model from {args.hf_ckpt} on {device} ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.hf_ckpt, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.hf_ckpt)
    # Left-pad so newly generated tokens land at the actual sequence end.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    if rank == 0:
        print("[infer] model + processor ready", flush=True)

    batch_size = max(1, args.batch_size)
    for batch_start in range(0, len(shard), batch_size):
        batch_meta = shard[batch_start : batch_start + batch_size]
        prepared = [_prepare_sample(sm, val_root) for sm in batch_meta]
        valid = [p for p in prepared if p is not None]
        for sm, p in zip(batch_meta, prepared):
            if p is None:
                print(f"[infer] rank{rank} skip {sm.get('id')}: missing file", flush=True)
        if not valid:
            continue

        ids, conversations, gts, video_paths = zip(*valid)

        if len(conversations) == 1:
            inputs = processor.apply_chat_template(
                conversations[0],
                add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            ).to(device)
        else:
            inputs = processor.apply_chat_template(
                list(conversations),
                add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True, padding=True,
            ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        new_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        responses = processor.batch_decode(new_ids, skip_special_tokens=True)

        for sample_id, video_path, response, gt in zip(ids, video_paths, responses, gts):
            (out / f"{sample_id}.json").write_text(json.dumps({
                "id": sample_id,
                "video": str(video_path),
                "model_response": response,
                "ground_truth": gt,
            }, indent=2))

        if rank == 0:
            done = batch_start + len(valid)
            preview = responses[0].replace("\n", " ")[:60]
            print(f"[infer] rank0 {done}/{len(shard)}: {preview!r}", flush=True)

    if rank == 0:
        print(f"[infer] rank0 done -> {out}", flush=True)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def _compute_metrics(results_dir, summary_path):
    """Walk ``results_dir`` of per-sample JSONs, write ``summary.json``.

    Parse-failure policy: a sample with unparseable ground_truth is dropped
    entirely (shouldn't happen on the canonical val split). A sample with
    parseable GT but unparseable model_response counts as a miss in the
    accuracy denominator (it cannot equal the GT). Pearson is computed only
    over pairs where both sides parsed.
    """
    preds, gts = [], []
    num_pred_parse_failures = 0
    num_gt_parse_failures = 0
    num_correct = 0
    num_gt_parsed = 0
    num_samples = 0
    for json_path in sorted(results_dir.glob("*.json")):
        if json_path.resolve() == summary_path.resolve():
            continue
        sample = json.loads(json_path.read_text())
        num_samples += 1
        gt = _parse_score(sample.get("ground_truth"))
        if gt is None:
            num_gt_parse_failures += 1
            print(f"[eval] WARN unparseable ground_truth in {json_path.name}; dropping sample")
            continue
        num_gt_parsed += 1
        pred = _parse_score(sample.get("model_response"))
        if pred is None:
            num_pred_parse_failures += 1
            continue
        preds.append(pred)
        gts.append(gt)
        if pred == gt:
            num_correct += 1

    accuracy = num_correct / num_gt_parsed if num_gt_parsed else 0.0
    pearson = _pearson(preds, gts)

    summary = {
        "accuracy": accuracy,
        "pearson_correlation": pearson,
        "num_samples": num_samples,
        "num_pred_parse_failures": num_pred_parse_failures,
        "num_gt_parse_failures": num_gt_parse_failures,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    pearson_str = f"{pearson:.3f}" if pearson is not None else "n/a"
    print(
        f"[eval] acc={accuracy:.3f} ({num_correct}/{num_gt_parsed})"
        f" pearson={pearson_str} (n={len(preds)})"
        f" pred_parse_fail={num_pred_parse_failures} -> {summary_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Per-sample JSON dir — output of inference, input to metrics")
    p.add_argument("--summary", default=None,
                   help="summary.json path (default: <results_dir>/summary.json)")
    # Inference-mode args. Both required to enable the inference pass.
    p.add_argument("--hf_ckpt", default=None,
                   help="HF safetensors dir (e.g. .../hf_exports/iter_NNN/). "
                        "If set, run inference first; else just aggregate from --results_dir.")
    p.add_argument("--val_root", default=None,
                   help="VideoPhy-2 val dir with meta.json + media/ + text/. "
                        "Required when --hf_ckpt is set.")
    p.add_argument("--n", type=int, default=None,
                   help="Limit to first N val samples (default: all)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-rank generation batch size (default: 1).")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    summary_path = Path(args.summary) if args.summary else results_dir / "summary.json"

    rank, world_size, local_rank = _init_distributed()

    if args.hf_ckpt:
        if not args.val_root:
            sys.exit("--val_root is required when --hf_ckpt is set")
        _run_inference(args, rank, world_size, local_rank)
        _barrier()

    if rank == 0:
        if not results_dir.is_dir():
            sys.exit(f"[eval] results_dir not found: {results_dir}")
        _compute_metrics(results_dir, summary_path)


if __name__ == "__main__":
    main()
