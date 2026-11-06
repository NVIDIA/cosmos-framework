#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notebook/CLI orchestration for public Perception Test validation MCQ."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
SERVING_IMAGE = "vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55"
SOURCE = "perception-test-local"
PROTOCOL = {"nframe": 16, "max_frame_side": 768,
            "resize": "aspect-preserving downscale only; no crop",
            "sampling": "uniform within source cut interval; positive cut is exclusive",
            "input": "sampled images, never re-encoded video",
            "claim": "launchable prototype, not reference-score parity"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def prepare(data_root, smoke=False, require_media=True):
    from vlmeval.perception_test_local import prepare_dataset

    prepared = prepare_dataset(data_root, smoke=smoke, require_media=require_media)
    print("PerceptionTest coverage: " + json.dumps(prepared["coverage"]), flush=True)
    return prepared


def run_command(command, log_path=None):
    """Tee unbuffered child progress to Jupyter and a durable private run log."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    env.update(AWS_CONFIG_FILE="/dev/null", AWS_SHARED_CREDENTIALS_FILE="/dev/null",
               AWS_EC2_METADATA_DISABLED="true", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    log = Path(log_path).open("w") if log_path is not None else None
    try:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if log is not None:
                    log.write(line)
                    log.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
    finally:
        if log is not None:
            log.close()


def prediction_files(work):
    return sorted(path for path in work.rglob("cosmos_PerceptionTest.*")
                  if path.parent.name.startswith("T") and path.suffix in {".tsv", ".xlsx", ".json"})


def verify_predictions(path, selected_indices):
    """Reject incomplete identities/transport failures, not wrong nonempty answers."""
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        files = prediction_files(path)
        if len(files) != 1:
            raise RuntimeError(f"PerceptionTest: expected one prediction table, found {len(files)}")
        path = files[0]
    if path.suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"index": str, "prediction": str})
    elif path.suffix == ".json":
        frame = pd.DataFrame(json.loads(path.read_text()))
    elif path.suffix == ".xlsx":
        frame = pd.read_excel(path, keep_default_na=False, dtype={"index": str, "prediction": str})
    else:
        raise ValueError("Predictions must be a native TSV, XLSX, or JSON table")
    if not {"index", "prediction"}.issubset(frame.columns) or frame.empty:
        raise RuntimeError("PerceptionTest: empty or invalid prediction output")
    expected = [str(index) for index in selected_indices]
    if len(frame) != len(expected):
        raise RuntimeError(f"PerceptionTest: expected {len(expected)} predictions, found {len(frame)}")
    indices = frame["index"].astype(str)
    if indices.duplicated().any() or set(indices) != set(expected):
        raise RuntimeError("PerceptionTest: duplicate, missing, or unexpected prediction identities")
    text = frame["prediction"].astype(str).str.strip().str.lower()
    failed = (frame["prediction"].isna() | text.eq("")
              | text.str.contains("failed to obtain answer", regex=False)
              | text.str.contains("video_load_failed", regex=False))
    if failed.any():
        raise RuntimeError(f"PerceptionTest: {int(failed.sum())} failed/empty predictions; inspect the run log")
    print(f"PerceptionTest: {len(frame)} predictions verified", flush=True)
    return path


def source_provenance():
    files = ("run.py", "vlmeval/perception_test_local.py", "vlmeval/dataset/perception_test.py",
             "vlmeval/dataset/video_dataset_config.py", "vlmeval/api/cosmos_reason.py", "vlmeval/execution_evidence.py",
             "launchable/launchable.py")
    return {"framework_revision": os.environ.get("COSMOS_COMMIT", "unknown"),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in files if (ROOT / name).is_file()}}


def _write_state(work, state):
    (work / "run.json").write_text(json.dumps(state, indent=2, default=str) + "\n")


def collect_execution_evidence(work, expected):
    from vlmeval.execution_evidence import aggregate_execution_records

    records = [json.loads(path.read_text()) for path in sorted((work / "execution-responses").glob("*.json"))]
    aggregate = aggregate_execution_records(records, expected_requests=expected)
    aggregate.update(per_response="execution-responses", requested_configuration="config.json",
                     summary="execution-summary.json")
    (work / "execution-summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return aggregate


def _protocol(nframe):
    if nframe != 16:
        raise ValueError("The launchable prototype protocol requires --nframe 16")


def run(work, mode, smoke, dry_run, data_root, api_nproc=1, nframe=16):
    work, data_root = Path(work), Path(data_root)
    if api_nproc < 1:
        raise ValueError("--api-nproc must be positive")
    _protocol(nframe)
    if mode not in {"infer", "all"}:
        raise ValueError("--mode must be infer or all")
    if work.exists():
        raise ValueError(f"Use a fresh --work-dir (already exists: {work}) to avoid stale results")
    # Validate all selected media before contacting an endpoint or creating run artifacts.
    started, clock = utc_now(), time.monotonic()
    prepared = prepare(data_root, smoke, require_media=True)
    recipe = json.loads((ROOT / "cosmos_eval/models/cosmos.json").read_text())
    model = {"class": recipe["class"], **recipe["defaults"]}
    model.update(api_base=os.environ.get("COSMOS_API_BASE", "http://cosmos-endpoint:8080/v1/chat/completions"),
                 model=os.environ.get("COSMOS_MODEL", "nvidia/Cosmos3-Nano"))
    from vlmeval.execution_evidence import processor_profile_for, PINNED_SERVING_COMPATIBILITY
    model.update(execution_evidence_dir=str(work / "execution-responses"),
                 execution_processor_profile=processor_profile_for(
                     model["model"], os.environ.get("COSMOS_MODEL_REVISION", "unknown")))
    config = {"model": {"cosmos": model}, "data": {"PerceptionTest": {
        "class": "PerceptionTestLocal", "dataset": "PerceptionTest", "data_root": str(data_root),
        "nframe": nframe, "smoke": smoke}}}
    work.mkdir(parents=True, mode=0o700)
    config_path = work / "config.json"
    local = model["api_base"] == "http://cosmos-endpoint:8080/v1/chat/completions"
    state = {"format_version": 4, "benchmark": "PerceptionTest", "source": SOURCE,
             "status": "planned" if dry_run else "running", "mode": mode,
             "scope": "smoke" if smoke else "full", "smoke": smoke, "dry_run": dry_run,
             **{key: prepared[key] for key in (
                 "dataset_revision", "manifest_sha256", "annotation_manifest_sha256", "coverage",
                 "selected_indices", "media_sha256", "media_manifest_sha256")},
             "data_root": str(data_root), "nframe": nframe, "frame_protocol": PROTOCOL,
             "api_nproc": api_nproc, "config": config,
             "generation_parameters": {key: value for key, value in model.items()
                                       if key not in {"class", "api_base", "model", "execution_evidence_dir",
                                                      "execution_processor_profile"}},
             "model": model["model"],
             "model_revision": os.environ.get("COSMOS_MODEL_REVISION",
                 MODEL_REVISION if local and model["model"] == "nvidia/Cosmos3-Nano" else "unknown"),
             "serving_image": (os.environ.get("VLLM_IMAGE", SERVING_IMAGE) if local
                               else os.environ.get("COSMOS_SERVING_IMAGE", "unknown")),
             "provenance": source_provenance(), "predictions": None, "scores": None,
             "execution_evidence": None,
             "pinned_runtime_compatibility": PINNED_SERVING_COMPATIBILITY,
             "inference_timing": {"started_utc": started}}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    _write_state(work, state)
    command = [sys.executable, "-u", str(ROOT / "run.py"), "--config", str(config_path),
               "--mode", "infer", "--work-dir", str(work / "native"),
               "--api-nproc", str(api_nproc), "--verbose"]
    if dry_run:
        print(" ".join(command), flush=True)
        return state
    try:
        endpoint(wait=1800)
        model["model"] = os.environ.get("COSMOS_MODEL", model["model"])
        state["model"] = model["model"]
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        _write_state(work, state)
        run_command(command, log_path=work / "run.log")
        state["execution_evidence"] = collect_execution_evidence(work, len(prepared["selected_indices"]))
        path = verify_predictions(work / "native", prepared["selected_indices"])
        state.update(status="inferred", predictions=str(path.relative_to(work)))
        state["inference_timing"].update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - clock)
        _write_state(work, state)
        if mode == "all":
            score(data_root, state["scope"], work=work)
        else:
            results(work)
    except BaseException as exc:
        state = json.loads((work / "run.json").read_text())
        state.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        state["execution_evidence"] = collect_execution_evidence(work, len(prepared["selected_indices"]))
        if "ended_utc" not in state["inference_timing"]:
            state["inference_timing"].update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - clock)
        _write_state(work, state)
        raise
    return json.loads((work / "run.json").read_text())


def score(data_root, scope, work=None, predictions=None, score_dir=None, nframe=16):
    from vlmeval.perception_test_local import score_predictions

    if scope not in {"smoke", "full"}:
        raise ValueError("Scoring requires explicit smoke or full scope")
    if (work is None) == (predictions is None):
        raise ValueError("Supply exactly one of --work-dir or --predictions")
    _protocol(nframe)
    state = None
    if work is not None:
        work = Path(work)
        state = json.loads((work / "run.json").read_text())
        if (state.get("source") != SOURCE or state.get("benchmark") != "PerceptionTest"
                or state.get("dry_run") or state.get("status") not in {"inferred", "complete"}):
            raise ValueError("Only successful PerceptionTest inference runs can be scored")
        if scope != state["scope"]:
            raise ValueError("Scoring scope differs from inference scope")
        nframe = state["nframe"]
        _protocol(nframe)
        predictions = (work / state["predictions"]).resolve()
        if not predictions.is_relative_to(work.resolve()):
            raise ValueError("Prediction artifact must remain within its inference work directory")
        score_dir = Path(score_dir) if score_dir is not None else work / "scores"
    elif score_dir is None:
        raise ValueError("--score-dir is required for existing native prediction files")
    score_dir = Path(score_dir)
    if score_dir.exists():
        raise ValueError("Scoring output already exists; choose a fresh --score-dir")
    # Annotation/cut/scope fingerprints do not depend on video files or their hashes.
    prepared = prepare(data_root, smoke=scope == "smoke", require_media=False)
    if state is not None and any(state[key] != prepared[key] for key in
                                 ("dataset_revision", "annotation_manifest_sha256",
                                  "manifest_sha256", "selected_indices")):
        raise ValueError("Dataset identities, annotations, or cuts changed since inference")
    predictions = verify_predictions(predictions, prepared["selected_indices"])
    started, clock = utc_now(), time.monotonic()
    if state is not None:
        state.update(status="scoring", scoring_timing={"started_utc": started})
        _write_state(work, state)
    try:
        summary = score_predictions(predictions, data_root, score_dir,
                                    smoke=scope == "smoke", nframe=nframe)
        timing = {"started_utc": started, "ended_utc": utc_now(),
                  "elapsed_seconds": time.monotonic() - clock}
        summary["execution_evidence"] = state.get("execution_evidence") if state is not None else {
            "verification_counts": {"unknown": 1}, "reason": "existing predictions have no request/response evidence"}
        (score_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        # Keep controller/model provenance beside the native trusted-label score artifact.
        (score_dir / "controller.json").write_text(json.dumps({
            "source": SOURCE, "scope": scope, "timing": timing,
            "frame_protocol": PROTOCOL, "provenance": source_provenance(),
            "model": state.get("model", "unknown") if state is not None else "unknown",
            "model_revision": state.get("model_revision", "unknown") if state is not None else "unknown",
            "serving_image": state.get("serving_image", "unknown") if state is not None else "unknown",
            "generation_parameters": state.get("generation_parameters") if state is not None else None,
            "execution_evidence": summary["execution_evidence"],
            "inference_run": str(work / "run.json") if work is not None else None,
        }, indent=2) + "\n")
        if state is not None:
            state.update(status="complete", scoring_timing=timing,
                         scores={"summary": str(score_dir / "summary.json"),
                                 "primary_metric": summary["primary_metric"], "coverage": summary["coverage"]})
            _write_state(work, state)
            results(work)
        else:
            show_summary(summary, scope)
            print(f"Score artifact: {score_dir / 'summary.json'}", flush=True)
        return summary
    except BaseException as exc:
        if state is not None:
            state.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            state["scoring_timing"].update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - clock)
            _write_state(work, state)
        raise


def show_summary(summary, scope):
    print(f"PerceptionTest validation MCQ — {scope}: {summary['status']}", flush=True)
    print("Top1/counts: " + json.dumps(summary["primary_metric"]), flush=True)
    print("Coverage: " + json.dumps(summary["coverage"]), flush=True)
    print("Prototype 16-frame/aspect-preserving-768 protocol; no reference-score parity claim.", flush=True)


def results(work):
    work = Path(work)
    state = json.loads((work / "run.json").read_text())
    if state.get("scores"):
        show_summary({"status": state["status"], **state["scores"]}, state["scope"])
        print(f"Score artifact: {state['scores']['summary']}", flush=True)
    else:
        print(f"PerceptionTest: {state['status']} ({state['scope']}); coverage: "
              + json.dumps(state["coverage"]), flush=True)
    print(f"Run provenance: {work / 'run.json'}", flush=True)
    return state


def main():
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Wait for /v1/models and verify the requested model")
    check.add_argument("--wait", type=int, default=1800)
    download = sub.add_parser("download", help="Download pinned public validation inputs")
    download.add_argument("--scope", choices=["metadata", "smoke", "full"], default="smoke")
    prep = sub.add_parser("prepare", help="Validate selected annotations/cuts and media coverage")
    runner = sub.add_parser("run", help="Run native inference and optionally local top1 scoring")
    scorer = sub.add_parser("score", help="Score predictions without model, endpoint, or video files")
    for child in (download, prep, runner, scorer):
        child.add_argument("--data-root", type=Path,
                           default=Path(os.environ.get("PERCEPTION_TEST_DATA_DIR", "/perception-test")))
    for child in (prep, runner):
        child.add_argument("--smoke", action="store_true", help="Five fixed video_8241 validation questions")
    prep.add_argument("--metadata-only", action="store_true", help="Prepare without video files (score-only)")
    runner.add_argument("--mode", choices=["infer", "all"], default="all")
    runner.add_argument("--api-nproc", type=int, default=1)
    runner.add_argument("--nframe", type=int, default=16)
    runner.add_argument("--dry-run", action="store_true")
    runner.add_argument("--work-dir", type=Path, required=True)
    sources = scorer.add_mutually_exclusive_group(required=True)
    sources.add_argument("--work-dir", type=Path)
    sources.add_argument("--predictions", type=Path)
    scorer.add_argument("--scope", choices=["smoke", "full"], required=True)
    scorer.add_argument("--score-dir", type=Path)
    scorer.add_argument("--nframe", type=int, default=16)
    report = sub.add_parser("results", help="Show top1/counts, coverage, and artifact locations")
    report.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        endpoint(args.wait)
    elif args.command == "download":
        from vlmeval.perception_test_local import fetch_dataset
        fetch_dataset(args.data_root.resolve(), scope=args.scope)
    elif args.command == "prepare":
        prepare(args.data_root.resolve(), args.smoke, require_media=not args.metadata_only)
    elif args.command == "run":
        run(args.work_dir.resolve(), args.mode, args.smoke, args.dry_run,
            args.data_root.resolve(), args.api_nproc, args.nframe)
    elif args.command == "score":
        score(args.data_root.resolve(), args.scope,
              work=args.work_dir.resolve() if args.work_dir else None,
              predictions=args.predictions.resolve() if args.predictions else None,
              score_dir=args.score_dir.resolve() if args.score_dir else None, nframe=args.nframe)
    else:
        results(args.work_dir.resolve())


if __name__ == "__main__":
    main()
