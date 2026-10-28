#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notebook/CLI orchestration; inference and metrics remain native VLMEvalKit's."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "cosmos_eval"


def manifest():
    return json.loads((KIT / "manifest.json").read_text())


def benchmarks(value):
    available = [key.split("/", 1)[1] for key in manifest() if "/VANTAGE_" in key]
    selected = available if value == "all" else value.split(",")
    unknown = set(selected) - set(available)
    if unknown or not selected:
        raise ValueError(f"Unknown VANTAGE benchmark(s): {', '.join(sorted(unknown))}")
    return [bench for bench in available if bench in selected]


def endpoint(wait=0):
    base = os.environ.get("COSMOS_API_BASE", "http://cosmos-endpoint:8080/v1/chat/completions")
    if not base.endswith("/chat/completions"):
        raise ValueError("COSMOS_API_BASE must be the complete /v1/chat/completions URL")
    os.environ["COSMOS_API_BASE"] = base
    url = base.removesuffix("/chat/completions") + "/models"
    key = os.environ.get("COSMOS_API_KEY", "EMPTY")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    deadline = time.monotonic() + wait
    while True:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                models = json.load(response)["data"]
            if not models:
                raise ValueError("The endpoint returned no served models")
            requested = os.environ.get("COSMOS_MODEL")
            ids = [model["id"] for model in models]
            if requested and requested not in ids:
                raise ValueError(f"COSMOS_MODEL {requested!r} is not served; available: {ids}")
            if not requested and len(ids) != 1:
                raise ValueError(f"Set COSMOS_MODEL to one of {ids}")
            os.environ["COSMOS_MODEL"] = requested or ids[0]
            print(f"Endpoint ready; model: {os.environ['COSMOS_MODEL']}", flush=True)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (502, 503, 504):
                raise RuntimeError(f"Model endpoint returned HTTP {exc.code}") from exc
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Model endpoint not ready after {wait}s (HTTP {exc.code})") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Model endpoint not reachable after {wait}s") from exc
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def prepare(selected, assets_dir):
    from vlmeval.local_scoring import build_local_dataset, load_assets

    bundle = load_assets(assets_dir, selected)
    for bench in selected:
        dataset = build_local_dataset(assets_dir, bench)
        if dataset is None or dataset.data.empty:
            raise RuntimeError(f"{bench}: no labeled dataset rows")
        print(f"{bench}: {len(dataset.data)} local labeled rows ready", flush=True)
        diagnostics = bundle["manifest"]["tasks"][bench.removeprefix("VANTAGE_")].get("annotation_diagnostics")
        if diagnostics:
            print(f"{bench}: original annotation diagnostics: {json.dumps(diagnostics, sort_keys=True)}", flush=True)


def run_command(command, **kwargs):
    env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    env.update(AWS_CONFIG_FILE="/dev/null", AWS_SHARED_CREDENTIALS_FILE="/dev/null",
               AWS_EC2_METADATA_DISABLED="true", MPLBACKEND="Agg")
    subprocess.run(command, cwd=ROOT, env=env, check=True, **kwargs)


def prediction_files(work):
    return sorted(p for p in work.rglob("cosmos_*")
                  if p.parent.name.startswith("T") and p.suffix in {".tsv", ".xlsx", ".json"}
                  and p.stem in {f"cosmos_{bench}" for bench in benchmarks("all")})


def verify_predictions(work, bench, expected_rows=None):
    import pandas as pd

    files = [p for p in prediction_files(work) if p.stem == f"cosmos_{bench}"]
    if len(files) != 1:
        raise RuntimeError(f"{bench}: expected one prediction table, found {len(files)}")
    path = files[0]
    if path.suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif path.suffix == ".json":
        frame = pd.read_json(path)
    else:
        frame = pd.read_excel(path)
    if "prediction" not in frame or frame.empty:
        raise RuntimeError(f"{bench}: empty or invalid prediction output")
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(f"{bench}: expected {expected_rows} predictions, found {len(frame)}")
    text = frame["prediction"].astype(str).str.strip().str.lower()
    failed = (frame["prediction"].isna()
              | text.isin(("", "nan", "none", "null", "<na>"))
              | text.str.contains("failed to obtain answer", regex=False)
              | text.str.contains("video_load_failed", regex=False))
    if failed.any():
        raise RuntimeError(f"{bench}: {int(failed.sum())} failed/empty predictions; inspect the run log")
    print(f"{bench}: {len(frame)} predictions verified", flush=True)
    return path


def _write_state(work, state):
    (work / "run.json").write_text(json.dumps(state, indent=2) + "\n")


def run(selected, work, mode, smoke, dry_run, assets_dir, api_nproc=1):
    if api_nproc < 1:
        raise ValueError("--api-nproc must be positive")
    if smoke and selected != ["VANTAGE_VQA"]:
        raise ValueError("--smoke requires --benchmarks VANTAGE_VQA")
    if work.exists():
        raise ValueError(f"Use a fresh --work-dir (already exists: {work}) to avoid stale results")
    from vlmeval.local_scoring import dataset_config, load_assets

    # Missing labels fail before allocating inference work or contacting the model.
    bundle = load_assets(assets_dir, selected)
    local_configs = {bench: dataset_config(assets_dir, bench) for bench in selected}
    if not dry_run:
        prepare(selected, assets_dir)
        endpoint(wait=1800)
    work.mkdir(parents=True, mode=0o700)
    configs = work / "_configs"
    run_command([sys.executable, str(KIT / "run_all.py"), "--benchmarks", ",".join(selected),
                 "--export-configs", str(configs)])
    state = {"format_version": 2, "source": "local", "status": "planned" if dry_run else "running",
             "cosmos_commit": os.environ.get("COSMOS_COMMIT", "unknown"),
             "model": os.environ.get("COSMOS_MODEL", "${COSMOS_MODEL}"),
             "benchmarks": selected, "mode": mode, "smoke": smoke, "dry_run": dry_run,
             "api_nproc": api_nproc,
             "scoring_assets": {"asset_id": bundle["manifest"]["asset_id"],
                                "manifest_sha256": bundle["manifest_sha256"]},
             "work_dirs": {}, "predictions": {}, "scores": {}}
    _write_state(work, state)
    try:
        for index, bench in enumerate(selected):
            config_path = configs / f"{bench}.json"
            cfg = json.loads(config_path.read_text())
            # Recipe exports retain placeholders; supply the endpoint already checked above.
            cfg["model"]["cosmos"]["api_base"] = os.environ.get(
                "COSMOS_API_BASE", cfg["model"]["cosmos"]["api_base"])
            cfg["data"] = {bench: local_configs[bench]}
            if smoke:
                cfg["data"][bench].update(limit=5, random_state=0)
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            out = work / f"bench_{index:02d}"
            state["work_dirs"][bench] = out.name
            command = [sys.executable, str(ROOT / "run.py"), "--config", str(config_path),
                       "--mode", "infer", "--work-dir", str(out),
                       "--api-nproc", str(api_nproc), "--verbose"]
            if dry_run:
                print(" ".join(command))
                continue
            out.mkdir(mode=0o700)
            with (out / "run.log").open("w") as log:
                run_command(command, stdout=log, stderr=subprocess.STDOUT)
            path = verify_predictions(out, bench, expected_rows=5 if smoke else None)
            state["predictions"][bench] = str(path.relative_to(work))
            _write_state(work, state)
        if not dry_run:
            state["status"] = "inferred"
            _write_state(work, state)
            if mode == "all":
                score(work, assets_dir)
            else:
                results(work)
    except Exception:
        state = json.loads((work / "run.json").read_text())
        state["status"] = "failed"
        _write_state(work, state)
        raise


def score(work, assets_dir, score_dir=None):
    from vlmeval.local_scoring import load_assets, score_predictions

    state = json.loads((work / "run.json").read_text())
    if state.get("source") != "local":
        raise ValueError("Use the native score-only CLI for predictions from older or other inference runs")
    if state.get("dry_run") or state.get("status") not in {"inferred", "complete"}:
        raise ValueError("Only successful inference runs can be scored")
    selected = state["benchmarks"]
    bundle = load_assets(assets_dir, selected)
    if state.get("scoring_assets", {}).get("manifest_sha256") != bundle["manifest_sha256"]:
        raise ValueError("Scoring assets changed since inference; use the original fingerprinted bundle")
    score_dir = Path(score_dir) if score_dir is not None else work / "local-scores"
    if score_dir.exists():
        raise ValueError("Scoring output already exists; choose a fresh --score-dir")
    predictions = {bench: verify_predictions(work / state["work_dirs"][bench], bench,
                                            expected_rows=5 if state.get("smoke") else None)
                   for bench in selected}
    score_dir.mkdir(parents=True, mode=0o700)
    state["status"] = "scoring"
    state["scores"] = {}
    _write_state(work, state)
    try:
        for index, bench in enumerate(selected):
            output = score_dir / f"bench_{index:02d}"
            summary = score_predictions(bench, predictions[bench], assets_dir, output,
                                        allow_subset=bool(state.get("smoke")))
            state["scores"][bench] = {"summary": str(output / "summary.json"),
                                      "primary_metric": summary["primary_metric"],
                                      "coverage": summary["coverage"]}
            _write_state(work, state)
        state["status"] = "complete"
        _write_state(work, state)
    except Exception:
        state["status"] = "failed"
        _write_state(work, state)
        raise
    results(work)


def results(work):
    state = json.loads((work / "run.json").read_text())
    print(f"Local run: {state.get('status', 'unknown')}" +
          (" (five-question smoke subset)" if state.get("smoke") else ""))
    for bench in state["benchmarks"]:
        result = state.get("scores", {}).get(bench)
        if result:
            metric, coverage = result["primary_metric"], result["coverage"]
            print(f"{bench}: {metric['name']}={metric['value']} {metric['unit']}; "
                  f"{coverage['scored_rows']}/{coverage['expected_rows']} rows")
            print(f"  Local score: {result['summary']}")
        else:
            print(f"{bench}: predictions only; local scoring has not completed")
    print("Per-task native metrics only; no external scoring submission or unverified leaderboard parity.")


def main():
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Wait for /v1/models and verify the requested model")
    check.add_argument("--wait", type=int, default=1800)
    prep = sub.add_parser("prepare", help="Validate explicitly supplied local labeled scoring assets")
    runner = sub.add_parser("run", help="Run native inference and optionally local scoring")
    scorer = sub.add_parser("score", help="Score existing predictions locally, without a model endpoint")
    for child in (prep, runner, scorer):
        child.add_argument("--assets-dir", type=Path,
                           default=Path(os.environ.get("SCORING_ASSETS_DIR", "/scoring-assets")))
    for child in (prep, runner):
        child.add_argument("--benchmarks", default="VANTAGE_VQA")
    runner.add_argument("--mode", choices=["infer", "all"], default="all")
    runner.add_argument("--api-nproc", type=int, default=1,
                        help="Concurrent API/video workers; increase only for available host RAM")
    runner.add_argument("--smoke", action="store_true", help="Five seeded VQA samples, not a full benchmark")
    runner.add_argument("--dry-run", action="store_true")
    runner.add_argument("--work-dir", type=Path, required=True)
    scorer.add_argument("--work-dir", type=Path, required=True)
    scorer.add_argument("--score-dir", type=Path)
    report = sub.add_parser("results", help="Show verified local metric summaries")
    report.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        endpoint(args.wait)
    elif args.command == "prepare":
        prepare(benchmarks(args.benchmarks), args.assets_dir.resolve())
    elif args.command == "run":
        run(benchmarks(args.benchmarks), args.work_dir.resolve(), args.mode,
            args.smoke, args.dry_run, args.assets_dir.resolve(), args.api_nproc)
    elif args.command == "score":
        score(args.work_dir.resolve(), args.assets_dir.resolve(), args.score_dir)
    else:
        results(args.work_dir.resolve())


if __name__ == "__main__":
    main()
