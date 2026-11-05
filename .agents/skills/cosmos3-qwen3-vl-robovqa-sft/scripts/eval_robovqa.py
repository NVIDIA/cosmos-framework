#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
r"""Answer and score the RoboVQA held-out clips with a Hugging Face Qwen3-VL checkpoint.

Run it on the untouched model (the "before" score) and on the SFT export (the "after" score) with the same
arguments; only ``--model`` and ``--out`` change. It runs on plain Transformers in its own environment (Transformers
4.57.6, TorchCodec 0.10, one GPU; the skill's ``SKILL.md`` step 1), the stack the recipe's numbers were measured with.

- Frames: :func:`decode_video` keeps every round(fps / 2)-th frame from frame 0, at most 32 (evenly thinned when there
  are more), and returns the effective frame rate. ``prepare_robovqa.py frames`` writes the training clips with this
  same function, so training and evaluation see identical frames.
- Generation: greedy, at most ``--max-new-tokens`` (64) new tokens, batches of ``--batch`` (4), bfloat16, SDPA.
- Metrics, overall and per task, deterministic (no judge): ``exact`` = normalised exact match (lower case, punctuation
  to spaces, articles dropped, spaces collapsed); ``f1`` = token F1 of the normalised strings; ``binary_acc`` = on the
  yes/no tasks (tags containing ``:discrete:``) the first ``yes`` or ``no`` in the prediction against the reference,
  none counting as wrong.

Writes ``preds.jsonl``, ``scores.json`` and ``eval_meta.json`` under ``--out`` and ends with the overall line, for
example ``exact 0.548  token-F1 0.676  yes/no 0.918 (n=1000)``. ``--score-only`` re-scores an existing ``preds.jsonl``.

    python .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/eval_robovqa.py --model $MODEL \
      --data $DATA/hf/test.jsonl --ref $DATA/samples_test.jsonl --out $OUT
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from collections.abc import Sequence
from typing import Any

MAX_FRAMES = 32
TARGET_FPS = 2.0
_ARTICLES = {"a", "an", "the"}


def decode_video(path: str) -> tuple[list[Any], float]:
    """The clip's frames as PIL images and their effective frame rate: every round(fps / 2)-th frame from frame 0, at
    most ``MAX_FRAMES`` (evenly thinned when there are more). ``prepare_robovqa.py frames`` writes the training clips
    with this function, so the frames the model trains on are the frames it is scored on."""
    import numpy as np
    from PIL import Image
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(path)
    n = decoder.metadata.num_frames or 0
    fps = float(decoder.metadata.average_fps or 0.0) or 30.0
    stride = max(1, int(round(fps / TARGET_FPS)))
    indices = list(range(0, n, stride))
    if len(indices) > MAX_FRAMES:
        step = len(indices) / MAX_FRAMES
        indices = [indices[int(i * step)] for i in range(MAX_FRAMES)]
    frames = decoder.get_frames_at(indices=indices).data.permute(0, 2, 3, 1).contiguous().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames], len(indices) / n * fps


def normalize(text: str) -> str:
    """Lower case, punctuation to spaces, articles dropped, whitespace collapsed."""
    text = "".join(" " if ch in string.punctuation else ch for ch in text.lower())
    return " ".join(word for word in text.split() if word not in _ARTICLES)


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between the normalised prediction and reference (SQuAD style)."""
    pred_tokens, ref_tokens = normalize(prediction).split(), normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = sum((collections.Counter(pred_tokens) & collections.Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred_tokens), common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def first_yes_no(text: str) -> str | None:
    """The first standalone ``yes`` or ``no`` in the text, lower-cased; None when there is neither."""
    match = re.search(r"\b(yes|no)\b", text.lower())
    return match.group(1) if match else None


def score(refs: Sequence[dict[str, Any]], preds: dict[str, str]) -> dict[str, dict[str, float]]:
    """Per task tag and overall (``__all__``): ``n``, then each metric's mean and its count (``<metric>_n``). A clip
    without a prediction scores as an empty answer."""
    per_task: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for ref in refs:
        prediction, task = preds.get(ref["id"], ""), ref["task"]
        row = {
            "exact": float(normalize(prediction) == normalize(ref["answer"])),
            "f1": token_f1(prediction, ref["answer"]),
        }
        if ":discrete:" in task:
            row["binary_acc"] = float(first_yes_no(prediction) == ref["answer"].lower())
        per_task[task].append(row)
        per_task["__all__"].append(row)
    out: dict[str, dict[str, float]] = {}
    for task, rows in per_task.items():
        agg: dict[str, float] = {"n": len(rows)}
        for key in ("exact", "f1", "binary_acc"):
            values = [row[key] for row in rows if key in row]
            if values:
                agg[key] = sum(values) / len(values)
                agg[key + "_n"] = len(values)
        out[task] = agg
    return out


def write_scores(ref_path: str, pred_path: str, out_path: str) -> dict[str, Any]:
    """Score ``preds.jsonl`` against ``samples_test.jsonl`` and write ``scores.json``."""
    with open(ref_path) as handle:
        refs = [json.loads(line) for line in handle if line.strip()]
    preds: dict[str, str] = {}
    with open(pred_path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                preds[row["id"]] = row.get("prediction", "")
    missing = [ref["id"] for ref in refs if ref["id"] not in preds]
    result: dict[str, Any] = score(refs, preds)
    result["__meta__"] = {"n_ref": len(refs), "n_pred": len(preds), "missing_predictions": len(missing)}
    with open(out_path, "w", newline="\n") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"__all__": result["__all__"], "__meta__": result["__meta__"]}))
    return result


def summary_line(result: dict[str, Any]) -> str:
    overall = result["__all__"]
    return "exact %.3f  token-F1 %.3f  yes/no %.3f (n=%d)" % (
        overall["exact"],
        overall["f1"],
        overall["binary_acc"],
        overall["n"],
    )


def generate(model_path: str, data_path: str, out_dir: str, *, batch: int, max_new_tokens: int, limit: int) -> int:
    """Greedy answers for every test row into ``<out_dir>/preds.jsonl``; returns the number of rows answered."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import VideoMetadata

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = model.cuda().eval()
    with open(data_path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit:
        rows = rows[:limit]
    start = time.time()
    with open(os.path.join(out_dir, "preds.jsonl"), "w", newline="\n") as out:
        for i in range(0, len(rows), batch):
            chunk = rows[i : i + batch]
            texts, videos, metadata = [], [], []
            for row in chunk:
                frames, fps = decode_video(row["video_path"])
                user = [message for message in row["messages"] if message["role"] == "user"][0]
                content = [
                    {"type": "video"} if item["type"] == "video" else {"type": "text", "text": item["text"]}
                    for item in user["content"]
                ]
                texts.append(
                    processor.apply_chat_template(
                        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
                    )
                )
                videos.append(frames)
                metadata.append(
                    VideoMetadata(
                        total_num_frames=len(frames),
                        fps=fps,
                        duration=len(frames) / fps,
                        frames_indices=list(range(len(frames))),
                        video_backend="torchcodec",
                    )
                )
            inputs = processor(
                text=texts,
                videos=videos,
                videos_kwargs={"do_sample_frames": False, "video_metadata": metadata},
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            answers = processor.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            for row, answer in zip(chunk, answers):
                reference = row["messages"][-1]["content"][0]["text"]
                out.write(
                    json.dumps(
                        {"id": row["id"], "prediction": answer.strip(), "reference": reference, "task": row["task"]}
                    )
                    + "\n"
                )
            out.flush()
            if (i // batch) % 20 == 0:
                print(f"{i + len(chunk)}/{len(rows)} {time.time() - start:.0f}s", flush=True)
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer and score the RoboVQA held-out clips (see the module doc).")
    parser.add_argument("--model", help="Hugging Face Qwen3-VL directory to evaluate (not needed with --score-only)")
    parser.add_argument("--data", help="hf/test.jsonl written by prepare_robovqa.py split")
    parser.add_argument("--ref", required=True, help="samples_test.jsonl written by prepare_robovqa.py split")
    parser.add_argument("--out", required=True, help="output directory: preds.jsonl, scores.json, eval_meta.json")
    parser.add_argument("--batch", type=int, default=4, help="clips per generate call (default 4)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="answer length cap (default 64)")
    parser.add_argument("--limit", type=int, default=0, help="score the first N clips only (0 = all)")
    parser.add_argument("--score-only", action="store_true", help="re-score <out>/preds.jsonl without generating")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if not args.score_only:
        if not args.model or not args.data:
            parser.error("--model and --data are required unless --score-only is given")
        start = time.time()
        n = generate(
            args.model, args.data, args.out, batch=args.batch, max_new_tokens=args.max_new_tokens, limit=args.limit
        )
        elapsed = time.time() - start
        with open(os.path.join(args.out, "eval_meta.json"), "w", newline="\n") as handle:
            json.dump({"eval_seconds": elapsed, "n": n, "model": args.model}, handle)
        print(f"EVAL_DONE {n} samples in {elapsed:.0f}s")
    result = write_scores(args.ref, os.path.join(args.out, "preds.jsonl"), os.path.join(args.out, "scores.json"))
    print(summary_line(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
