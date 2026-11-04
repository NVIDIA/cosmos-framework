#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public benchmark selection for the Cosmos3 evaluation notebook."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import time

CATALOG = {
    "perception-test": {
        "title": "Perception Test",
        "question": "What happened in the video?",
        "metric": "Multiple-choice accuracy",
        "scope": "Public validation MCQ · 19,140 questions / 5,260 videos",
        "smoke": "Five fixed questions from video_8241",
        "download": "Smoke: 225 MB archive. Full: 75.38 GB archive, plus extraction space.",
        "protocol": "16 source-cut-aware frames · aspect-preserving maximum side 768 · local top1",
        "source": "https://github.com/google-deepmind/perception_test",
        "license": "Google DeepMind / Perception Test authors · code Apache-2.0, other materials CC BY 4.0",
    },
    "videophy2": {
        "title": "VideoPhy2",
        "question": "Does the video match its caption and obey physics?",
        "metric": "Mean SA / PC Pearson correlation ×100",
        "scope": "Public test · 3,397 video-caption pairs / 6,794 rating requests",
        "smoke": "Two fixed video-caption pairs; four rating requests",
        "download": "Publisher archive: 3.93 GB, plus original public URLs for absent entries and extraction space. Smoke may also need this archive.",
        "protocol": "32 audited video frames · 224 × 224 · official categorical 1–5 ratings",
        "source": "https://github.com/Hritikbansal/videophy/tree/01d366c979e4a82875852adbf427ccc036e7373a/VIDEOPHY2",
        "license": "VideoPhy2 / Hritik Bansal and collaborators · publisher code and data MIT",
    },
    "mmsivideo-bench": {
        "title": "MMSI-Video-Bench",
        "question": "Can the model reason about space across video views?",
        "metric": "Official-extractor answer accuracy",
        "scope": "Public main benchmark · 1,106 questions",
        "smoke": "One fixed question from each of five task families",
        "download": "Both scopes: about 16.1 GB compressed canonical frames, plus references and extraction space. Original videos are not needed.",
        "protocol": "Uniform-50 canonical frames plus references · 2,048-token cap · local official scoring",
        "source": "https://github.com/InternRobotics/MMSI-Video-Bench/tree/412f5a4a9eb6d89613c60b541b17df74f11a6736",
        "license": "MMSI-Video-Bench authors · publisher CC BY 4.0; underlying third-party redistribution rights not independently cleared",
    },
}


def validate_selection(benchmark, scope, score_only=False):
    if benchmark not in CATALOG:
        raise ValueError(f"Choose BENCHMARK from {', '.join(CATALOG)}")
    if scope not in {"smoke", "full"}:
        raise ValueError("Choose SCOPE = 'smoke' or 'full'")
    if score_only and benchmark != "perception-test":
        raise ValueError("Notebook score-only mode currently supports Perception Test; use infer-and-score for the other benchmarks")


def download(benchmark, data_root, scope, metadata_only=False):
    validate_selection(benchmark, scope, score_only=metadata_only)
    data_root = Path(data_root).resolve()
    if benchmark == "perception-test":
        from vlmeval.perception_test_local import fetch_dataset

        return fetch_dataset(data_root, scope="metadata" if metadata_only else scope)
    module = importlib.import_module("videophy2" if benchmark == "videophy2" else "mmsivideo")
    return module.download(data_root, scope)


def run(benchmark, data_root, scope, work, api_nproc=1, score_only=False,
        existing_work=None, predictions=None):
    validate_selection(benchmark, scope, score_only)
    if type(api_nproc) is not int or not 1 <= api_nproc <= 4:
        raise ValueError("API_NPROC must be an integer between 1 and 4")
    data_root, work = Path(data_root).resolve(), Path(work).resolve()
    if work.exists():
        raise ValueError(f"Choose a fresh work directory; refusing to overwrite {work}")
    if work == data_root or work.is_relative_to(data_root) or data_root.is_relative_to(work):
        raise ValueError("Keep benchmark data and run outputs in separate directories")
    if score_only and (existing_work is None) == (predictions is None):
        raise ValueError("Score-only requires exactly one of EXISTING_WORK or PREDICTIONS_FILE")
    if not score_only and (existing_work is not None or predictions is not None):
        raise ValueError("Existing prediction inputs require SCORE_ONLY = True")
    if benchmark != "perception-test":
        import launchable

        launchable.endpoint(wait=1800)
        module = importlib.import_module("videophy2" if benchmark == "videophy2" else "mmsivideo")
        return module.run(data_root, scope, work, api_nproc=api_nproc)

    import launchable

    clock = time.monotonic()
    if score_only:
        native = launchable.score(data_root, scope, work=existing_work, predictions=predictions,
                                  score_dir=work)
    else:
        state = launchable.run(work, "all", scope == "smoke", False, data_root,
                               api_nproc=api_nproc, nframe=16)
        native = json.loads((work / "scores" / "summary.json").read_text())
    metric = native["primary_metric"]
    summary = {
        "benchmark": benchmark, "scope": scope, "status": "completed",
        "primary_metric": {**metric, "value": metric["value"] * 100, "unit": "percent"},
        "coverage": {"expected": native["coverage"]["selected_rows"], "scored": metric["total"]},
        "invalid_count": native["invalid_count"],
        "protocol": launchable.PROTOCOL,
        "warnings": ["Public validation split, not the held-out test leaderboard.",
                     "Adapter frame/answer-parsing protocol; no published-score parity claim."],
        "reference_parity": False, "elapsed_seconds": time.monotonic() - clock,
        "score_only": score_only,
        "per_reasoning": native["per_reasoning"],
        "native_summary": "summary.json" if score_only else "scores/summary.json",
    }
    if not score_only:
        summary["execution_evidence"] = state.get("execution_evidence")
        summary["model"] = state["model"]
        summary["model_revision"] = state["model_revision"]
    # Preserve the native score-only artifact; the notebook has its own report name.
    report = work / ("notebook-summary.json" if score_only else "summary.json")
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Notebook result: {report}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Download checksum-pinned public data")
    fetch.add_argument("--metadata-only", action="store_true")
    runner = sub.add_parser("run", help="Infer and score the selected public benchmark")
    runner.add_argument("--work-dir", type=Path, required=True)
    runner.add_argument("--api-nproc", type=int, default=1)
    runner.add_argument("--score-only", action="store_true")
    sources = runner.add_mutually_exclusive_group()
    sources.add_argument("--existing-work", type=Path)
    sources.add_argument("--predictions", type=Path)
    for command in (fetch, runner):
        command.add_argument("--benchmark", choices=CATALOG, required=True)
        command.add_argument("--scope", choices=["smoke", "full"], default="smoke")
        command.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(args.benchmark, args.data_root, args.scope, args.metadata_only)
    else:
        run(args.benchmark, args.data_root, args.scope, args.work_dir,
            api_nproc=args.api_nproc, score_only=args.score_only,
            existing_work=args.existing_work, predictions=args.predictions)


if __name__ == "__main__":
    main()
