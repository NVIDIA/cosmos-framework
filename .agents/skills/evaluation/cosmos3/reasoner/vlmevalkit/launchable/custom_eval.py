# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local labeled customer-video MCQ evaluation; no dataset conversion or judge."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import urllib.parse

SYSTEM_PROMPT = "Answer the multiple-choice video question with only its single uppercase option letter."


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _external(path):
    path = Path(path).expanduser().resolve()
    source = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), Path(__file__).resolve().parents[1])
    if path.is_relative_to(source) or source.is_relative_to(path):
        raise ValueError("Customer inputs and results must be outside the source checkout")
    return path


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_row(row, media_root, seen):
    """Validate one canonical annotation without changing its label or option order."""
    if not isinstance(row, dict) or set(row) - {"id", "video", "question", "candidates", "answer", "category", "start", "end"}:
        raise ValueError("Row must use only canonical fields")
    row = dict(row)
    sample_id = _text(row.get("id"), "id")
    if sample_id in seen:
        raise ValueError(f"Duplicate id: {sample_id}")
    seen.add(sample_id)
    _text(row.get("question"), "question")
    options = row.get("candidates")
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise ValueError("candidates must contain 2–26 options")
    for option in options:
        _text(option, "candidate")
    if len(set(options)) != len(options):
        raise ValueError("Duplicate candidate text")
    if not isinstance(row.get("answer"), str) or options.count(row["answer"]) != 1:
        raise ValueError("answer must equal exactly one full candidate text")
    if "category" in row:
        _text(row["category"], "category")
    relative = _text(row.get("video"), "video")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError("video must be a safe relative POSIX path within media_root")
    path = (media_root / relative).resolve()
    if not path.is_relative_to(media_root) or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing, empty or escaping media for id {sample_id}")
    _external(path)
    if ("start" in row) != ("end" in row):
        raise ValueError("start and end must be supplied together in seconds")
    if "start" in row:
        _number(row["start"], "start")
        _number(row["end"], "end")
        if row["end"] <= row["start"]:
            raise ValueError("end must exceed start")
    return row, path


def probe_video(path):
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    fps, count = float(reader.get_avg_fps()), len(reader)
    if count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video has no usable frames or FPS")
    reader.get_batch([0, count - 1])
    return {"source_frames": count, "source_fps": fps, "duration_seconds": count / fps}


def _load(config_path):
    config_path = _external(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"dataset", "manifest", "media_root", "nframes", "max_pixels", "min_pixels", "image_patch_size", "model", "model_revision", "generation", "processor_profile"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown config fields")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {"name", "revision", "source"}:
        raise ValueError("dataset requires name, revision, source")
    for key, value in dataset.items():
        _text(value, "dataset." + key)
    manifest = _external(config_path.parent / _text(config.get("manifest"), "manifest"))
    root = _external(config_path.parent / _text(config.get("media_root"), "media_root"))
    frames = _integer(config.get("nframes"), "nframes", 4)
    if frames % 2:
        raise ValueError("nframes must be even for the native video processor")
    maximum = _integer(config.get("max_pixels"), "max_pixels")
    minimum = _integer(config.get("min_pixels"), "min_pixels")
    if minimum > maximum:
        raise ValueError("min_pixels must not exceed max_pixels")
    _integer(config.get("image_patch_size"), "image_patch_size")
    model = _text(config.get("model", os.environ.get("COSMOS_MODEL")), "model or COSMOS_MODEL")
    revision = _text(config.get("model_revision", os.environ.get("COSMOS_MODEL_REVISION", "unknown")), "model_revision")
    generation = {"temperature": 0, "seed": 42, "max_tokens": 32, "enable_thinking": False}
    supplied = config.get("generation", {})
    if not isinstance(supplied, dict) or set(supplied) - set(generation):
        raise ValueError("Unknown generation fields")
    generation.update(supplied)
    _number(generation["temperature"], "temperature")
    _integer(generation["seed"], "seed", 0)
    _integer(generation["max_tokens"], "max_tokens")
    if not isinstance(generation["enable_thinking"], bool):
        raise ValueError("enable_thinking must be boolean")
    if config.get("processor_profile") is not None and not isinstance(config["processor_profile"], dict):
        raise ValueError("processor_profile must be an explicit evidence-profile object")
    rows, media, seen = [], [], set()
    lines = manifest.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        if not line.strip():
            raise ValueError("Blank JSONL lines are not canonical rows")
        row, path = validate_row(json.loads(line), root, seen)
        info = probe_video(path)
        if "end" in row and row["end"] > info["duration_seconds"]:
            raise ValueError(f"Temporal end exceeds video duration for {row['id']}")
        rows.append(row)
        media.append({"id": row["id"], "video": row["video"], "sha256": _digest(path), **info})
    if not rows:
        raise ValueError("Manifest must contain at least one labeled row")
    data = {"dataset": dataset, "rows": rows, "media": media}
    report = {"status": "validated", "benchmark": "customer-video-mcq", "scope": "custom", "dataset": dataset,
              "protocol": {"name": "customer_video_mcq_native_v1", "scorer": "strict_single_uppercase_letter",
                           "truncation_policy": "wrong_without_retry", "official_benchmark": False},
              "dataset_identity": _identity(data), "manifest_sha256": _digest(manifest), "expected": len(rows),
              "media": media, "model": {"name": model, "revision": revision},
              "generation": generation, "sampling": {k: config[k] for k in ("nframes", "min_pixels", "max_pixels", "image_patch_size")}}
    return config, root, rows, report


def validate(config_path) -> dict:
    """Validate all labels, media, IDs and temporal bounds without endpoint requests."""
    return _load(config_path)[3]


def build_payload(row, video_url, video_kwargs, settings):
    """Only video, question and candidate text cross the inference boundary."""
    question = row["question"] + "\n" + "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(row["candidates"]))
    return {"model": settings["model"]["name"], "messages": [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "video_url", "video_url": {"url": video_url}}, {"type": "text", "text": question}]}],
            **{k: settings["generation"][k] for k in ("temperature", "seed", "max_tokens")},
            "chat_template_kwargs": {"enable_thinking": settings["generation"]["enable_thinking"]},
            "mm_processor_kwargs": video_kwargs,
            "media_io_kwargs": {"video": {"num_frames": settings["sampling"]["nframes"],
                                         "min_frames": settings["sampling"]["nframes"],
                                         "max_frames": settings["sampling"]["nframes"]}},
            "return_token_ids": True}


def score_answer(row, content, finish_reason):
    """Strict single-letter exact match; malformed and truncated outputs count wrong."""
    predicted = content.strip() if isinstance(content, str) else ""
    valid = len(predicted) == 1 and "A" <= predicted < chr(65 + len(row["candidates"])) and finish_reason == "stop"
    correct = bool(valid and row["candidates"][ord(predicted) - 65] == row["answer"])
    return {"prediction": predicted if len(predicted) == 1 and "A" <= predicted <= "Z" else None,
            "valid": valid, "correct": correct, "finish_reason": finish_reason if finish_reason in ("stop", "length") else None}


def summarize(rows, results):
    """Complete coverage is mandatory; transport failures never become wrong answers."""
    by_id = {item["id"]: item for item in results}
    if len(by_id) != len(results) or set(by_id) - {row["id"] for row in rows}:
        raise ValueError("Unexpected or duplicate result IDs")
    scored = [row for row in rows if row["id"] in by_id and "error" not in by_id[row["id"]]]
    correct = sum(bool(by_id[row["id"]]["correct"]) for row in scored)
    categories = {}
    for row in scored:
        category = row.get("category", "uncategorized")
        metric = categories.setdefault(category, {"correct": 0, "total": 0})
        metric["total"] += 1
        metric["correct"] += bool(by_id[row["id"]]["correct"])
    for metric in categories.values():
        metric["accuracy_percent"] = 100 * metric["correct"] / metric["total"]
    complete = len(scored) == len(rows)
    return {"status": "completed" if complete else "failed", "coverage": {"expected": len(rows), "scored": len(scored)},
            "primary_metric": {"name": "accuracy_percent", "value": 100 * correct / len(rows) if complete else None,
                               "unit": "percent", "correct": correct, "total": len(rows)}, "categories": categories}


def run(config_path, work, api_nproc=1) -> dict:
    """Preflight every selected video before sending any request; write fresh local evidence."""
    from vlmeval.execution_evidence import aggregate_execution_records

    _integer(api_nproc, "api_nproc")
    if api_nproc > 4:
        raise ValueError("api_nproc must be at most 4")
    config, root, rows, validated = _load(config_path)
    api = os.environ.get("COSMOS_API_BASE", "http://cosmos-endpoint:8080/v1/chat/completions")
    parts = urllib.parse.urlsplit(api)
    if parts.scheme not in ("http", "https") or not parts.netloc or not parts.path.endswith("/chat/completions"):
        raise ValueError("COSMOS_API_BASE must be the complete /v1/chat/completions URL")
    timeout = float(os.environ.get("COSMOS_TIMEOUT", "300"))
    _number(timeout, "COSMOS_TIMEOUT", 0.001)
    key = os.environ.get("COSMOS_API_KEY", "EMPTY")
    work = _external(work)
    work.mkdir(parents=True, exist_ok=False)
    (work / "responses").mkdir()
    started = time.monotonic()
    summary = {**validated, **summarize(rows, []),
               "requested_execution": "requested-execution.json", "effective_execution": "responses/*.json#execution",
               "results": "results.jsonl", "workers": api_nproc}
    results = []
    _json(work / "validated.json", validated)
    try:
        import requests
        from vlmeval.api.cosmos_reason import process_video_info_to_video_url
        from vlmeval.execution_evidence import build_execution_record, processor_profile_for

        profile = config.get("processor_profile") or processor_profile_for(validated["model"]["name"], validated["model"]["revision"])
        _json(work / "requested-execution.json", {"model": validated["model"], "generation": validated["generation"],
                                                 "sampling": validated["sampling"], "processor_profile": profile})

        with tempfile.TemporaryDirectory(prefix="custom-transport-", dir=work) as temporary:
            prepared = []
            # Native decode/resize/temporal handling, fully preflighted before first request.
            for index, row in enumerate(rows):
                kwargs = {k: config[k] for k in ("nframes", "min_pixels", "max_pixels")}
                if "start" in row:
                    kwargs.update(video_start=row["start"], video_end=row["end"])
                url, video_kwargs = process_video_info_to_video_url(str(root / row["video"]), image_patch_size=config["image_patch_size"], use_cache=False, **kwargs)
                if _digest(root / row["video"]) != validated["media"][index]["sha256"]:
                    raise ValueError("Selected media changed during preflight; restore the fixed dataset")
                transport = Path(temporary) / f"{index:06d}.txt"
                transport.write_text(url, encoding="utf-8")
                prepared.append((index, row, transport, video_kwargs))

            def predict(item):
                index, row, transport, video_kwargs = item
                evidence_path = f"responses/{index:06d}.json"
                result = {"id": row["id"], "category": row.get("category", "uncategorized"), "evidence": evidence_path}
                record = {}
                try:
                    payload = build_payload(row, transport.read_text(encoding="utf-8"), video_kwargs, validated)
                    response = requests.post(api, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
                    record["http_status"] = response.status_code
                    response.raise_for_status()
                    raw = response.json()
                    if not isinstance(raw, dict):
                        raise ValueError("Non-object response")
                    record["execution"] = build_execution_record(payload, raw,
                        media={"expected_frame_slots": config["nframes"]}, processor_profile=profile,
                        requested_model_revision=validated["model"]["revision"])
                    choices = raw.get("choices")
                    if "error" in raw or (isinstance(choices, list) and any(
                            isinstance(choice, dict) and choice.get("finish_reason") in ("error", "abort")
                            for choice in choices)):
                        raise ValueError("Endpoint reported an unsuccessful generation")
                    # Successful HTTP responses with malformed answers remain in denominator.
                    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
                    message = choice.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    result.update(score_answer(row, content, choice.get("finish_reason")))
                    # Store only the selected answer, not arbitrary raw endpoint fields.
                    record["answer"] = {k: result[k] for k in ("prediction", "valid", "correct", "finish_reason")}
                except Exception as exc:
                    # Exception strings/URLs/raw responses may contain secrets or prompt echoes.
                    result["error"] = type(exc).__name__
                    record["error"] = type(exc).__name__
                _json(work / evidence_path, record)
                return result

            with concurrent.futures.ThreadPoolExecutor(max_workers=api_nproc) as pool:
                for result in pool.map(predict, prepared):
                    results.append(result)
                    with (work / "results.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
        summary.update(summarize(rows, results))
        if summary["status"] != "completed":
            raise RuntimeError("Customer evaluation incomplete; inspect summary.json and responses")
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = type(exc).__name__
        raise
    finally:
        records = [json.loads(path.read_text(encoding="utf-8")).get("execution")
                   for path in sorted((work / "responses").glob("*.json"))]
        summary["execution_evidence"] = aggregate_execution_records(records, expected_requests=len(rows))
        summary["execution_summary"] = "execution-summary.json"
        _json(work / "execution-summary.json", summary["execution_evidence"])
        summary["elapsed_seconds"] = time.monotonic() - started
        _json(work / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--config", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--work-dir", type=Path, required=True)
    execute.add_argument("--api-nproc", type=int, default=1)
    args = parser.parse_args()
    result = validate(args.config) if args.command == "validate" else run(args.config, args.work_dir, args.api_nproc)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
