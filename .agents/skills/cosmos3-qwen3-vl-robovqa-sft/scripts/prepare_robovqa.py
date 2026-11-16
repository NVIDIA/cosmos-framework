#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
r"""Write the RoboVQA split and the SFT training rows of ``cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py``.

Source: ``Tianli/robovqa`` (Apache-2.0) at revision ``eb5819490f0d9ec18a79774233d4b1c9ebd52eea``, downloaded with
``videos/*``, ``json/*`` and ``LICENSE.txt``. Only rows whose mp4 is present are usable (9,999 clips). Each row's text
holds one to nine task segments ``<task:TAG>\n<context> Q: <question>? <PRED>A: <answer>\n</PRED>``; discrete answers
are wrapped in ``<PRED:BINARY>yes|no</PRED:BINARY>``.

Three steps, each in the environment named (the skill's ``SKILL.md`` has the commands):

1. ``split`` (any Python 3.10+, CPU): shuffle the sorted video ids with ``random.Random(seed)`` and cut 80/10/10 into
   train / val / test; move every member of a group of byte-identical clips to the split of the group's smallest id;
   keep one question per clip (``random.Random(seed + int(md5(video id)[:8], 16))`` picks the segment). Writes
   ``splits/``, ``samples_{train,val,test}.jsonl`` (``{id, video, task, question, answer, split, uid}``),
   ``hf/{train,val,test}.jsonl`` (the source ``video_path`` and the conversation) and ``split_manifest.json``
   (counts, task histogram, moved duplicates, leakage check), and prints the md5 of the train and test samples.
2. ``frames`` (evaluation environment, CPU): for every training clip, decode the frames the evaluator decodes
   (``eval_robovqa.decode_video``) and write them losslessly (PNG in MOV, rgb24) at the effective frame rate under
   ``frames/``; then ``train.jsonl``, the SFT rows the recipe reads, each pointing at its file, and
   ``train.jsonl.frames.jsonl`` (per clip: frame count, rate, size, sha256 of the raw frames).
3. ``verify`` (SFT environment, CPU): for every training file, check that the training stack decodes exactly the
   recorded frames at the recorded frame rate, and that the Qwen3-VL processor called the way the SFT packer calls it
   (the path, ``do_sample_frames=False``) gives the same ``input_ids``, video grid and pixel values as the evaluator's
   call (the frames plus their ``VideoMetadata``). Writes ``train.jsonl.verify.jsonl``; exits 1 unless every row passes.

Run it as a file (it imports ``eval_robovqa`` from its own directory):

    python .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/prepare_robovqa.py split --src $SRC --out $DATA
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
_SEGMENT = re.compile(r"<task:([^>]+)>\n(.*?)<PRED>A: (.*?)\n</PRED>", re.S)
_BINARY = re.compile(r"<PRED:BINARY>(yes|no)</PRED:BINARY>")
# md5 of the samples files that `split --seed 0` writes from the pinned revision (the recipe's measured split).
SEED0_MD5 = {
    "samples_train.jsonl": "0b3739a50f5b715a6f4a5ab11081807a",
    "samples_test.jsonl": "1a9b1924eddcf34cf26ce7ffbcb69184",
}


def _default_workers() -> int:
    return min(os.cpu_count() or 1, 48)


def _md5_file(path: str | Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def parse_segments(text: str) -> list[dict[str, str]]:
    """The task segments of one RoboVQA row: ``{task, question, answer}``, a binary answer reduced to yes/no."""
    out = []
    for tag, question, answer in _SEGMENT.findall(text):
        binary = _BINARY.search(answer)
        out.append({"task": tag, "question": question.strip(), "answer": binary.group(1) if binary else answer.strip()})
    return out


# ------------------------------------------------------------------------------------------------ split --------------
def split(src: Path, out: Path, *, seed: int, fractions: Sequence[float]) -> dict[str, Any]:
    """Write the split, the samples, the conversation rows and the manifest; return the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    present = {path.name for path in (src / "videos").glob("*.mp4")}
    rows = []
    for json_path in sorted(glob.glob(str(src / "json" / "*" / "*.json"))):
        with open(json_path) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row["video"] in present:
                        rows.append(row)
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_video.setdefault(row["video"], []).append(row)
    if not all(len(group) == 1 for group in by_video.values()):
        raise ValueError("expected one source row per video")
    ids = sorted(by_video)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train, n_val = int(n * fractions[0]), int(n * fractions[1])
    splits = {"train": ids[:n_train], "val": ids[n_train : n_train + n_val], "test": ids[n_train + n_val :]}
    # Co-locate byte-identical clips: every member of an md5 group goes to the split of its lexicographically smallest
    # id (deterministic), so identical videos never straddle splits.
    md5_of = {video: _md5_file(src / "videos" / video) for video in ids}
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for video in ids:
        groups[md5_of[video]].append(video)
    where = {video: name for name, videos in splits.items() for video in videos}
    moved = []
    for group in groups.values():
        if len(group) > 1:
            target = where[min(group)]
            for video in group:
                if where[video] != target:
                    moved.append((video, where[video], target))
                    where[video] = target
    splits = {name: [video for video in ids if where[video] == name] for name in splits}

    (out / "splits").mkdir(exist_ok=True)
    samples: dict[str, list[dict[str, Any]]] = {}
    histogram: collections.Counter[tuple[str, str]] = collections.Counter()
    unparsed = 0
    for name, videos in splits.items():
        (out / "splits" / f"{name}_videos.txt").write_text("\n".join(videos) + "\n", newline="\n")
        samples[name] = []
        for video in videos:
            segments = parse_segments(by_video[video][0]["text"])
            if not segments:
                unparsed += 1
                continue
            pick = random.Random(seed + int(hashlib.md5(video.encode()).hexdigest()[:8], 16)).choice(segments)
            samples[name].append(
                {
                    "id": f"{name}_{video[:-4]}",
                    "video": video,
                    "task": pick["task"],
                    "question": pick["question"],
                    "answer": pick["answer"],
                    "split": name,
                    "uid": by_video[video][0]["uid"],
                }
            )
            histogram[(name, pick["task"])] += 1
        with open(out / f"samples_{name}.jsonl", "w", newline="\n") as handle:
            for sample in samples[name]:
                handle.write(json.dumps(sample) + "\n")

    # Leakage check: video ids, video bytes and source uids are disjoint across splits.
    split_of_md5: dict[str, set[str]] = {}
    for name, videos in splits.items():
        for video in videos:
            split_of_md5.setdefault(md5_of[video], set()).add(name)
    train, val, test = (set(splits[name]) for name in SPLITS)
    uids = {name: {sample["uid"] for sample in samples[name]} for name in SPLITS}

    (out / "hf").mkdir(exist_ok=True)
    for name in SPLITS:
        with open(out / "hf" / f"{name}.jsonl", "w", newline="\n") as handle:
            for sample in samples[name]:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": "video_0"},
                            {"type": "text", "text": sample["question"]},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
                ]
                video_path = str((src / "videos" / sample["video"]).resolve())
                row = {"id": sample["id"], "video_path": video_path, "task": sample["task"], "messages": conversation}
                handle.write(json.dumps(row) + "\n")

    manifest = {
        "source": "Tianli/robovqa (Apache-2.0 LICENSE.txt)",
        "seed": seed,
        "fractions": list(fractions),
        "rule": "shuffle sorted video ids with random.Random(seed); 80/10/10; one Q/A per video",
        "usable_videos": n,
        "unparsed_rows": unparsed,
        "counts": {name: len(samples[name]) for name in SPLITS},
        "task_hist": {f"{name}|{task}": count for (name, task), count in sorted(histogram.items())},
        "duplicates_moved": moved,
        "leakage_check": {
            "video_id_overlap": len(train & val | train & test | val & test),
            "identical_video_bytes_across_splits": sum(1 for names in split_of_md5.values() if len(names) > 1),
            "uid_overlap_train_vs_val_test": len(uids["train"] & uids["test"]) + len(uids["train"] & uids["val"]),
        },
    }
    with open(out / "split_manifest.json", "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


# ------------------------------------------------------------------------------------------------ frames -------------
def _write_clip_frames(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Decode one clip the evaluator's way and write the frames losslessly at the effective frame rate."""
    import av
    from eval_robovqa import decode_video  # this script's directory: the evaluator's own decoder

    row, frames_dir = job
    frames, fps_eff = decode_video(row["video_path"])
    rate = Fraction(fps_eff).limit_denominator(10**6)
    clip = os.path.splitext(os.path.basename(row["video_path"]))[0]
    path = os.path.join(frames_dir, f"{clip}.mov")
    tmp = path + ".tmp.mov"
    tick = Fraction(rate.denominator, rate.numerator)  # one codec tick per frame
    with av.open(tmp, "w", format="mov") as container:
        stream = container.add_stream("png", rate=rate)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "rgb24"
        stream.codec_context.time_base = tick
        for index, frame in enumerate(frames):
            video_frame = av.VideoFrame.from_image(frame)
            video_frame.pts = index
            video_frame.time_base = tick
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    os.replace(tmp, path)
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.tobytes())
    return {
        "id": row["id"],
        "src": row["video_path"],
        "mov": path,
        "T": len(frames),
        "fps_eff": fps_eff,
        "rate": f"{rate.numerator}/{rate.denominator}",
        "w": frames[0].size[0],
        "h": frames[0].size[1],
        "bytes": os.path.getsize(path),
        "frames_sha256": digest.hexdigest(),
    }


def write_frames(out: Path, *, workers: int, limit: int) -> dict[str, Any]:
    """Write ``frames/``, ``train.jsonl`` and ``train.jsonl.frames.jsonl`` from ``hf/train.jsonl``."""
    from multiprocessing import Pool

    source = out / "hf" / "train.jsonl"
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    with open(source) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit:
        rows = rows[:limit]
    with Pool(workers) as pool:
        meta = pool.map(_write_clip_frames, [(row, str(frames_dir)) for row in rows], chunksize=8)
    by_id = {clip["id"]: clip for clip in meta}
    rows_path = out / "train.jsonl"
    with open(rows_path, "w", newline="\n") as handle:
        for row in rows:
            messages = []
            for message in row["messages"]:
                parts = [
                    {"type": "video", "video": by_id[row["id"]]["mov"]}
                    if item["type"] == "video"
                    else {"type": "text", "text": item["text"]}
                    for item in message["content"]
                ]
                messages.append({"role": message["role"], "content": parts})
            handle.write(json.dumps({"messages": messages, "meta": {"item": row["id"]}}, ensure_ascii=False) + "\n")
    with open(f"{rows_path}.frames.jsonl", "w", newline="\n") as handle:
        for clip in meta:
            handle.write(json.dumps(clip) + "\n")
    return {
        "rows": len(rows),
        "src_md5": _md5_file(source),
        "rows_md5": _md5_file(rows_path),
        "bytes_total": sum(clip["bytes"] for clip in meta),
        "T_total": sum(clip["T"] for clip in meta),
    }


# ------------------------------------------------------------------------------------------------ verify -------------
_PROCESSOR: Any = None


def _verify_clip(job: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    """One training file in the training stack: same frames, same rate, same processor output as the evaluator's."""
    import torch
    from PIL import Image
    from torchcodec.decoders import VideoDecoder
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    global _PROCESSOR
    row, clip, model_path = job
    if _PROCESSOR is None:
        _PROCESSOR = AutoProcessor.from_pretrained(model_path)
    decoder = VideoDecoder(clip["mov"])
    file_meta = decoder.metadata
    data = (
        decoder.get_frames_at(indices=list(range(file_meta.num_frames))).data.permute(0, 2, 3, 1).contiguous().numpy()
    )
    digest = hashlib.sha256()
    for index in range(data.shape[0]):
        digest.update(data[index].tobytes())
    frames = [Image.fromarray(data[index]) for index in range(data.shape[0])]
    messages = [
        {
            "role": message["role"],
            "content": [
                {"type": "video"} if item["type"] == "video" else {"type": "text", "text": item["text"]}
                for item in message["content"]
            ],
        }
        for message in row["messages"]
    ]
    text = _PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    count, fps = clip["T"], clip["fps_eff"]
    evaluator_metadata = VideoMetadata(
        total_num_frames=count,
        fps=fps,
        duration=count / fps,
        frames_indices=list(range(count)),
        video_backend="torchcodec",
    )
    as_evaluated = _PROCESSOR(
        text=[text],
        videos=[frames],
        return_tensors="pt",
        videos_kwargs={"do_sample_frames": False, "video_metadata": [evaluator_metadata]},
    )
    as_trained = _PROCESSOR(
        text=[text], videos=[clip["mov"]], return_tensors="pt", videos_kwargs={"do_sample_frames": False}
    )
    return {
        "id": clip["id"],
        "T": count,
        "n_file": file_meta.num_frames,
        "sha_ok": digest.hexdigest() == clip["frames_sha256"],
        "fps_rel_err": abs(float(file_meta.average_fps) - fps) / fps,
        "ids_equal": torch.equal(as_evaluated["input_ids"], as_trained["input_ids"]),
        "grid_equal": torch.equal(as_evaluated["video_grid_thw"], as_trained["video_grid_thw"]),
        "pix_equal": torch.equal(as_evaluated["pixel_values_videos"], as_trained["pixel_values_videos"]),
        "len": int(as_trained["input_ids"].shape[1]),
    }


def verify_frames(out: Path, *, model_path: str, workers: int) -> dict[str, Any]:
    """Check every training file; write ``train.jsonl.verify.jsonl``; return the summary."""
    from multiprocessing import Pool

    rows_path = out / "train.jsonl"
    with open(rows_path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with open(f"{rows_path}.frames.jsonl") as handle:
        clips = {clip["id"]: clip for clip in (json.loads(line) for line in handle if line.strip())}
    with Pool(workers) as pool:
        results = pool.map(_verify_clip, [(row, clips[row["meta"]["item"]], model_path) for row in rows], chunksize=4)
    bad = [
        result
        for result in results
        if not (
            result["sha_ok"]
            and result["ids_equal"]
            and result["grid_equal"]
            and result["pix_equal"]
            and result["T"] == result["n_file"]
            and result["fps_rel_err"] < 1e-6
        )
    ]
    with open(f"{rows_path}.verify.jsonl", "w", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")
    return {
        "rows": len(results),
        "all_ok": not bad,
        "bad": len(bad),
        "first_bad": bad[:3],
        "max_fps_rel_err": max(result["fps_rel_err"] for result in results),
        "len_max": max(result["len"] for result in results),
        "len_sum": sum(result["len"] for result in results),
    }


# ------------------------------------------------------------------------------------------------ CLI ----------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RoboVQA split and SFT rows for sft_qwen3_vl_8b_robovqa (see the module doc)."
    )
    steps = parser.add_subparsers(dest="step", required=True)
    split_args = steps.add_parser("split", help="split RoboVQA and write samples, conversation rows and the manifest")
    split_args.add_argument("--src", required=True, help="local Tianli/robovqa download (json/, videos/)")
    split_args.add_argument("--out", required=True, help="data root (the recipe's SFT_DATA_ROOT)")
    split_args.add_argument("--seed", type=int, default=0)
    split_args.add_argument("--frac", default="0.8,0.1,0.1", help="train,val,test fractions")
    frames_args = steps.add_parser("frames", help="write the training clips' evaluation frames and train.jsonl")
    frames_args.add_argument("--out", required=True, help="data root written by split")
    frames_args.add_argument("--workers", type=int, default=_default_workers())
    frames_args.add_argument("--limit", type=int, default=0, help="first N training clips only (0 = all)")
    verify_args = steps.add_parser("verify", help="check the training files in the SFT environment")
    verify_args.add_argument("--out", required=True, help="data root written by split and frames")
    verify_args.add_argument(
        "--model", default=os.environ.get("SFT_MODEL_PATH"), help="Qwen3-VL snapshot (default $SFT_MODEL_PATH)"
    )
    verify_args.add_argument("--workers", type=int, default=_default_workers())
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()

    if args.step == "split":
        fractions = [float(value) for value in args.frac.split(",")]
        manifest = split(Path(args.src), out, seed=args.seed, fractions=fractions)
        print(json.dumps({"counts": manifest["counts"], "leakage_check": manifest["leakage_check"]}))
        for name, expected in SEED0_MD5.items():
            note = f" (seed-0 reference {expected})" if args.seed == 0 else ""
            print(f"{_md5_file(out / name)}  {name}{note}")
        return 0
    if args.step == "frames":
        print(json.dumps(write_frames(out, workers=args.workers, limit=args.limit)))
        return 0
    if not args.model:
        parser.error("verify needs --model or SFT_MODEL_PATH")
    summary = verify_frames(out, model_path=args.model, workers=args.workers)
    print(json.dumps(summary))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
