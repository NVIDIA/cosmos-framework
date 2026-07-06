# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Materialize the public videophy2 HF dataset into the canonical local SFT layout.

Public source (MIT, no gating):
    https://huggingface.co/datasets/videophysics/videophy2_train
    https://huggingface.co/datasets/videophysics/videophy2_test

Output layout:

    <out_root>/videophy2_train/
        meta.json
        media/video_<id>.mp4
        text/conversation_<id>.json
    <out_root>/videophy2_val/    # renamed from HF 'test' split
        ...

Each conversation JSON embeds a fixed ``PROMPT_TEMPLATE`` (the four-criteria
physical-plausibility scoring rubric) so the same training recipe
(``videophy2_sft_nano``) consumes the materialized output unchanged.

Run once, offline. The training process does not import this module.

Example::

    python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf \\
        --out_root examples/data/videophysics \\
        --split both
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable, Optional

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# I/O-bound HTTP GETs — saturate via threads. S3 / public CDNs comfortably
# handle this concurrency per source IP. Bump to ~64 for very fat pipes; default
# 16 is tuned for typical interactive boxes.
DEFAULT_WORKERS = 16

# The user prompt is fixed to the four-criteria physical-plausibility scoring
# rubric used by the SFT recipe. Treat this as the canonical prompt — every
# materialized conversation JSON embeds it verbatim.
PROMPT_TEMPLATE = (
    "You are a helpful video analyzer. Evaluate whether the video follows "
    "physical commonsense.\n\n"
    "Evaluation Criteria:\n"
    "1. **Object Behavior:** Do objects behave according to their expected "
    "physical properties (e.g., rigid objects do not deform unnaturally, "
    "fluids flow naturally)?\n"
    "2. **Motion and Forces:** Are motions and forces depicted in the video "
    "consistent with real-world physics (e.g., gravity, inertia, conservation "
    "of momentum)?\n"
    "3. **Interactions:** Do objects interact with each other and their "
    "environment in a plausible manner (e.g., no unnatural penetration, "
    "appropriate reactions on impact)?\n"
    "4. **Consistency Over Time:** Does the video maintain consistency across "
    "frames without abrupt, unexplainable changes in object behavior or "
    "motion?\n\n"
    "Instructions for Scoring:\n"
    "- **1:** No adherence to physical commonsense. The video contains "
    "numerous violations of fundamental physical laws.\n"
    "- **2:** Poor adherence. Some elements follow physics, but major "
    "violations are present.\n"
    "- **3:** Moderate adherence. The video follows physics for the most part "
    "but contains noticeable inconsistencies.\n"
    "- **4:** Good adherence. Most elements in the video follow physical "
    "laws, with only minor issues.\n"
    "- **5:** Perfect adherence. The video demonstrates a strong "
    "understanding of physical commonsense with no violations.\n\n"
    "Response Template:\n"
    "Analyze the video carefully and answer the question according to the "
    "following template:\n\n"
    "[Score between 1 and 5.]\n\n"
    "Example Responses:\n"
    "2\n"
)

DEFAULT_HF_REPO = "videophysics/videophy2_train"
DEFAULT_HF_TEST_REPO = "videophysics/videophy2_test"

logger = logging.getLogger("prepare_videophy2_from_hf")


# ----------------------------------------------------------------------------- args


@dataclass
class Args:
    out_root: str
    split: str  # "train" | "test" | "both"
    score_field: str = "pc"
    include_caption: bool = False
    caption_field: str = "upsampled_caption"  # falls back to "caption" if missing
    limit: Optional[int] = None
    hf_train_repo: str = DEFAULT_HF_REPO
    hf_test_repo: str = DEFAULT_HF_TEST_REPO
    media_field_name: str = "video_0"
    timeout_seconds: int = 60
    chunk_size: int = 1 << 16  # 64 KiB
    workers: int = DEFAULT_WORKERS
    extra_skip_ids: list[str] = field(default_factory=list)


def _parse_args(argv: Optional[list[str]] = None) -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out_root", required=True,
                   help="Directory to write videophy2_{train,val}/ subdirs into.")
    p.add_argument("--split", choices=["train", "test", "both"], default="both")
    p.add_argument("--score_field", default="pc", choices=["pc", "sa", "joint"],
                   help="HF column to use as the assistant target. Default: pc (Physical Commonsense).")
    p.add_argument("--include_caption", action="store_true",
                   help="Append the caption (or upsampled_caption) to the user prompt.")
    p.add_argument("--caption_field", default="upsampled_caption",
                   help="Which caption column to use when --include_caption is set.")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke-test cap: only process the first N rows of each split.")
    p.add_argument("--hf_train_repo", default=DEFAULT_HF_REPO)
    p.add_argument("--hf_test_repo", default=DEFAULT_HF_TEST_REPO)
    p.add_argument("--media_field_name", default="video_0",
                   help="Key under which the media bytes will be referenced in the conversation JSON.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Concurrent download workers per split. Default {DEFAULT_WORKERS}.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    ns = p.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(ns.verbose, 2),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    return Args(
        out_root=os.path.abspath(ns.out_root),
        split=ns.split,
        score_field=ns.score_field,
        include_caption=ns.include_caption,
        caption_field=ns.caption_field,
        limit=ns.limit,
        hf_train_repo=ns.hf_train_repo,
        hf_test_repo=ns.hf_test_repo,
        media_field_name=ns.media_field_name,
        workers=max(1, ns.workers),
    )


# ------------------------------------------------------------------------- helpers


def _make_session(timeout_seconds: int, pool_size: int = DEFAULT_WORKERS) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.request_timeout = timeout_seconds  # consumed by _stream_download
    return s


def _stream_download(session: requests.Session, url: str, dst_path: str, timeout: int, chunk: int) -> bool:
    """Stream-download `url` to `dst_path`. Returns True on success.

    Atomic via a `.part` rename and idempotent via a non-empty-size check.
    """
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return True
    tmp = dst_path + ".part"
    try:
        with session.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(tmp, "wb") as f:
                for buf in resp.iter_content(chunk_size=chunk):
                    if buf:
                        f.write(buf)
        if os.path.getsize(tmp) == 0:
            os.remove(tmp)
            logger.warning("downloaded 0 bytes from %s", url)
            return False
        os.replace(tmp, dst_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("download failed for %s: %s", url, exc)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def _conversation_for_row(row: dict, args: Args) -> Optional[dict]:
    score = row.get(args.score_field)
    if score is None:
        return None
    if args.score_field == "joint":
        answer_text = "yes" if int(bool(score)) else "no"
    else:
        try:
            answer_text = str(int(score))
        except (TypeError, ValueError):
            answer_text = str(score)

    user_text = PROMPT_TEMPLATE
    if args.include_caption:
        cap = row.get(args.caption_field) or row.get("caption")
        if cap:
            user_text = user_text + f"\nCaption: {cap}\n"

    return {
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": args.media_field_name},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            },
        ]
    }


def _load_progress(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_progress(path: str, state: dict) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


# ------------------------------------------------------------------------- main op


def _iter_rows(repo: str, split_arg: str, limit: Optional[int]) -> tuple[int, Iterable[tuple[int, dict]]]:
    """Return ``(total, iterator)`` over rows of the public videophy2 CSV on HF Hub.

    The total is returned upfront so callers (tqdm) can render an ETA without
    materializing the whole iterator.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    # The author-published files at the root of each repo:
    #   videophysics/videophy2_train: `videophy2_training.csv` (+ `.zip` for videos)
    #   videophysics/videophy2_test:  `videophy2_test.csv`
    # Both CSVs reference videos via `video_url` (public S3) — the zip is
    # an alternative we don't need.
    csv_candidates = (
        "videophy2_training.csv",
        "videophy2_test.csv",
        "data.csv",
        "videophy2.csv",
        "train.csv",
    )
    csv_path: Optional[str] = None
    for candidate in csv_candidates:
        try:
            csv_path = hf_hub_download(repo_id=repo, filename=candidate, repo_type="dataset")
            logger.info("downloaded %s/%s", repo, candidate)
            break
        except Exception as exc:  # noqa: BLE001 — fall through to the next candidate
            logger.debug("%s/%s not present: %s", repo, candidate, exc)
    if csv_path is None:
        # Last resort: use `datasets.load_dataset` (may fail on UTF-8
        # decode but at least surfaces a clear error to the user).
        from datasets import load_dataset

        ds = load_dataset(repo, split="train")
        n = len(ds) if limit is None else min(len(ds), limit)
        logger.info("loaded %s (%d rows; using %d) via datasets.load_dataset", repo, len(ds), n)

        def _gen_ds() -> Iterable[tuple[int, dict]]:
            for idx in range(n):
                yield idx, ds[idx]

        return n, _gen_ds()

    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    n = len(df) if limit is None else min(len(df), limit)
    logger.info("loaded %s (%d rows; using %d) via raw CSV (utf-8 with replace)", repo, len(df), n)

    def _gen_df() -> Iterable[tuple[int, dict]]:
        for idx in range(n):
            # Pandas row -> dict; NaN -> None so downstream `row.get(...)` works.
            row = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in df.iloc[idx].items()}
            yield idx, row

    return n, _gen_df()


def _build_split(args: Args, hf_repo: str, out_subdir: str) -> None:
    out_dir = os.path.join(args.out_root, out_subdir)
    media_dir = os.path.join(out_dir, "media")
    text_dir = os.path.join(out_dir, "text")
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    progress_path = os.path.join(out_dir, ".build_progress.json")
    progress = _load_progress(progress_path)
    done_ids = set(progress.get("done", []))

    session = _make_session(timeout_seconds=60, pool_size=args.workers)

    def _process_one(idx_row: tuple[int, dict]) -> tuple[int, str, Optional[dict]]:
        idx, row = idx_row
        sample_id = f"conversation_{idx + 1:04d}"
        media_rel = f"media/video_{idx + 1:04d}.mp4"
        text_rel = f"text/conversation_{idx + 1:04d}.json"
        media_path = os.path.join(out_dir, media_rel)
        text_path = os.path.join(out_dir, text_rel)
        entry = {"id": sample_id, "media": media_rel, "conversation": text_rel}

        if sample_id in done_ids and os.path.exists(media_path) and os.path.exists(text_path):
            return idx, "skip", entry

        video_url = row.get("video_url")
        if not video_url:
            logger.warning("row %d has no video_url; skipping", idx)
            return idx, "fail", None

        if not _stream_download(session, video_url, media_path, timeout=60, chunk=1 << 16):
            return idx, "fail", None

        conv = _conversation_for_row(row, args)
        if conv is None:
            logger.warning("row %d has no %s score; skipping", idx, args.score_field)
            try:
                os.remove(media_path)
            except OSError:
                pass
            return idx, "fail", None

        tmp = text_path + ".part"
        with open(tmp, "w") as f:
            json.dump(conv, f)
        os.replace(tmp, text_path)
        return idx, "ok", entry

    total, rows = _iter_rows(hf_repo, "train", args.limit)
    row_list = list(rows)  # small (~3k rows), cheap to materialize
    n_ok = n_skip = n_fail = 0
    results: list[tuple[int, dict]] = []  # (idx, entry) — sorted at end

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_process_one, ir) for ir in row_list]
        pbar = tqdm(total=total, desc=out_subdir, unit="vid", dynamic_ncols=True)
        try:
            for fut in as_completed(futures):
                try:
                    idx, status, entry = fut.result()
                except Exception as exc:  # noqa: BLE001 — count unexpected worker errors as fail
                    logger.warning("worker raised: %s", exc)
                    n_fail += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
                    continue
                if status == "ok":
                    n_ok += 1
                    done_ids.add(entry["id"])
                    results.append((idx, entry))
                    if n_ok % 50 == 0:
                        progress["done"] = sorted(done_ids)
                        _save_progress(progress_path, progress)
                elif status == "skip":
                    n_skip += 1
                    results.append((idx, entry))
                else:
                    n_fail += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
        finally:
            pbar.close()

    progress["done"] = sorted(done_ids)
    _save_progress(progress_path, progress)

    results.sort(key=lambda r: r[0])
    meta = [r[1] for r in results]
    meta_path = os.path.join(out_dir, "meta.json")
    tmp = meta_path + ".part"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_path)

    logger.info(
        "split=%s done: wrote %d entries to %s (ok=%d skip=%d fail=%d)",
        out_subdir, len(meta), meta_path, n_ok, n_skip, n_fail,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.out_root, exist_ok=True)

    if args.split in ("train", "both"):
        _build_split(args, args.hf_train_repo, "videophy2_train")
    if args.split in ("test", "both"):
        _build_split(args, args.hf_test_repo, "videophy2_val")

    return 0


if __name__ == "__main__":
    sys.exit(main())
