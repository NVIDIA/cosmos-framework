# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned public VideoPhy2 categorical SA/PC protocol (not published Nano parity)."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import urllib.parse
import urllib.request
import zipfile

DATASET_REVISION = "90b81ffa54f565d9e40e83b7a2c247ba1dccfa2b"
DATASET_SHA256 = "85a6690b9508b7e69c592f3cbcbc4113efd3a573eb5ec69d6ae030a8ffb8a4e7"
DATASET_URL = f"https://huggingface.co/datasets/videophysics/videophy2_test/resolve/{DATASET_REVISION}/videophy2_test.csv"
ARCHIVE_REVISION = "1b920c23525d460ec47c42a135cd2eeb805a64be"
ARCHIVE_SHA256 = "69578681c7831f987e37743ad6adc979b5d72d03279632b10d103d468571a453"
ARCHIVE_URL = f"https://huggingface.co/datasets/videophysics/videophy2_test_videos/resolve/{ARCHIVE_REVISION}/videophy2_test%20.zip"
SOURCE_REVISION = "01d366c979e4a82875852adbf427ccc036e7373a"
SOURCE_URL = f"https://github.com/Hritikbansal/videophy/tree/{SOURCE_REVISION}/VIDEOPHY2"
SYSTEM_PROMPT = "You are a helpful assistant. Answer the rating question with only a single integer from 1 to 5."
PC_QUESTION = "Does this video adhere to the physical laws? Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means significant violations."
NUM_MAP = dict(zip(("zero", "one", "two", "three", "four", "five", "0", "1", "2", "3", "4", "5"), (0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5)))
MODEL_DIRS = {"hunyuan": "hunyuan", "wan": "wan", "cosmos": "cosmos", "cogvideo": "cogvideo", "ray2": "luma", "sora": "sora", "videocrafter": "videocrafter"}
SMOKE_INDICES = (0, 1)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outside_source(path):
    path = Path(path).expanduser().resolve()
    source = next((parent for parent in Path(__file__).resolve().parents
                   if (parent / ".git").exists()), Path(__file__).resolve().parents[1])
    if path.is_relative_to(source):
        raise ValueError("Benchmark data and outputs must be outside the source directory")
    return path


def _selected(rows, scope):
    if scope not in ("smoke", "full"):
        raise ValueError("scope must be smoke or full")
    return [(index, rows[index]) for index in (SMOKE_INDICES if scope == "smoke" else range(len(rows)))]


def _rows(root):
    path = root / "videophy2_test.csv"
    if _sha256(path) != DATASET_SHA256:
        raise ValueError("VideoPhy2 CSV SHA256 mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3397:
        raise ValueError("VideoPhy2 test split must contain exactly 3397 rows")
    return rows


def _fetch(url, target, expected_sha256=None):
    started = time.monotonic()
    if target.is_file():
        digest = _sha256(target)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"SHA256 mismatch: {target.name}")
        return {"sha256": digest, "bytes": target.stat().st_size, "cached": True,
                "elapsed_seconds": time.monotonic() - started}
    parts = urllib.parse.urlsplit(url)
    encoded = urllib.parse.urlunsplit(parts._replace(path=urllib.parse.quote(parts.path, safe="/%:;,@!$&'()*+=-._~")))
    temporary = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(encoded, timeout=300) as response, temporary.open("wb") as out:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                out.write(chunk)
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError(f"SHA256 mismatch: {target.name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"sha256": digest.hexdigest(), "bytes": target.stat().st_size, "cached": False,
            "elapsed_seconds": time.monotonic() - started}


def _archive_files(archive):
    files = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Unsafe entry in publisher archive")
        if info.filename.lower().endswith(".mp4") and not info.filename.startswith("__MACOSX/"):
            files.append(info.filename)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate archive member names")
    return files


def archive_match(row, files):
    """Exact URL path, then basename and original model directory; never caption-match."""
    path = urllib.parse.unquote(urllib.parse.urlsplit(row["video_url"]).path).lstrip("/")
    candidates = [name for name in files if name == path or name.endswith("/" + path)]
    if not candidates:
        candidates = [name for name in files if PurePosixPath(name).name == PurePosixPath(path).name]
        if len(candidates) > 1:
            candidates = [name for name in candidates if MODEL_DIRS[row["model_name"]] in name.lower()]
    if len(candidates) > 1:
        raise ValueError("Ambiguous publisher archive mapping")
    return candidates[0] if candidates else None


def download(data_root: Path, scope: str):
    """Download the pinned archive; absent entries use only their exact official CSV URL."""
    if scope not in ("smoke", "full"):
        raise ValueError("scope must be smoke or full")
    root = _outside_source(data_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "media").mkdir(exist_ok=True)
    started = time.monotonic()
    report = {"benchmark": "videophy2", "scope": scope, "dataset_revision": DATASET_REVISION,
              "dataset_sha256": DATASET_SHA256, "archive_revision": ARCHIVE_REVISION,
              "archive_sha256": ARCHIVE_SHA256, "sources": {}, "errors": []}
    try:
        report["sources"]["csv"] = _fetch(DATASET_URL, root / "videophy2_test.csv", DATASET_SHA256)
        rows = _rows(root)
        chosen = _selected(rows, scope)
        report["coverage"] = {"expected": len(chosen), "downloaded": 0}
        report["sources"]["archive"] = _fetch(ARCHIVE_URL, root / "official-test-videos.zip", ARCHIVE_SHA256)
        manifest_path = root / "media-manifest.jsonl"
        entries = {entry["index"]: entry for entry in (json.loads(line) for line in manifest_path.read_text().splitlines())} if manifest_path.exists() else {}
        with zipfile.ZipFile(root / "official-test-videos.zip") as archive:
            files = _archive_files(archive)
            mappings = [(index, row, archive_match(row, files)) for index, row in chosen]
            report["archive_media_files"] = len(files)
            report["selected_archive_rows"] = sum(name is not None for _, _, name in mappings)
            report["selected_original_url_rows"] = sum(name is None for _, _, name in mappings)
            for index, row, name in mappings:
                before = time.monotonic()
                target = root / "media" / f"{index:05d}.mp4"
                entry = {"index": index, "sample_id": f"videophy2_test:{index:05d}",
                         "video_url": row["video_url"], "model_name": row["model_name"],
                         "path": f"media/{index:05d}.mp4", "archive_entry": name,
                         "source": "publisher_archive" if name else "official_csv_url",
                         "dataset_sha256": DATASET_SHA256,
                         "archive_revision": ARCHIVE_REVISION if name else None,
                         "archive_sha256": ARCHIVE_SHA256 if name else None}
                old = entries.get(index, {})
                try:
                    cached = (target.is_file() and old.get("archive_entry") == name
                              and old.get("video_url") == row["video_url"]
                              and old.get("sha256") == _sha256(target) and target.stat().st_size > 0)
                    if cached:
                        entry.update(sha256=old["sha256"], bytes=target.stat().st_size, cached=True)
                    elif name is not None:
                        temporary = target.with_suffix(".partial")
                        try:
                            with archive.open(name) as source, temporary.open("wb") as out:
                                shutil.copyfileobj(source, out, 1024 * 1024)
                            temporary.replace(target)
                        finally:
                            temporary.unlink(missing_ok=True)
                        entry.update(sha256=_sha256(target), bytes=target.stat().st_size, cached=False)
                    else:
                        if target.exists():
                            raise ValueError("Unverified cached original-URL media; remove it before downloading")
                        entry.update(_fetch(row["video_url"], target))
                    if entry["bytes"] <= 0:
                        raise ValueError("Empty media")
                except Exception as exc:
                    entry["error"] = type(exc).__name__
                    report["errors"].append({"index": index, "error": type(exc).__name__})
                entry["elapsed_seconds"] = time.monotonic() - before
                entries[index] = entry
                # Append durable checkpoints; latest entry wins when resuming.
                with manifest_path.open("a") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    stream.flush()
        manifest_path.write_text("".join(json.dumps(entries[key], allow_nan=False) + "\n" for key in sorted(entries)))
        report["coverage"]["downloaded"] = sum("error" not in entries[index] for index, _ in chosen)
        if report["errors"]:
            raise RuntimeError("Selected VideoPhy2 media coverage is incomplete; see download-summary.json")
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = type(exc).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        _json(root / "download-summary.json", report)
    return report


def official_parse(output):
    """Official inference.py num_map insertion-order substring/digit fallback semantics."""
    text = output.lower().strip()
    for key, value in NUM_MAP.items():
        if key in text:
            return value, False
    digits = "".join(character for character in text if character.isdigit())
    # The official fallback emits zero when no valid rating is found.
    try:
        value = int(digits) if digits else 0
    except ValueError:
        value = 0
    return (value if value in NUM_MAP.values() else 0), True


def correlations(results):
    def pearson(task):
        actual = [row["truth"][task] for row in results]
        predicted = [row["responses"][task]["score"] for row in results]
        if len(actual) < 2:
            return None
        mean_a, mean_p = sum(actual) / len(actual), sum(predicted) / len(predicted)
        da, dp = [value - mean_a for value in actual], [value - mean_p for value in predicted]
        variance_a, variance_p = sum(value * value for value in da), sum(value * value for value in dp)
        if variance_a == 0 or variance_p == 0:
            return None
        return max(-100.0, min(100.0, 100 * sum(a * p for a, p in zip(da, dp)) / math.sqrt(variance_a * variance_p)))
    sa, pc = pearson("sa"), pearson("pc")
    return {"sa_pearson_x100": sa, "pc_pearson_x100": pc,
            "mean_sa_pc_pearson_x100": (sa + pc) / 2 if sa is not None and pc is not None else None,
            "rows": len(results)}


def _settings(api_nproc):
    base = os.environ.get("COSMOS_API_BASE", "http://cosmos-endpoint:8080/v1/chat/completions")
    parts = urllib.parse.urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.netloc or not parts.path.endswith("/chat/completions"):
        raise ValueError("COSMOS_API_BASE must be the complete /v1/chat/completions URL")
    settings = {"model": os.environ.get("COSMOS_MODEL", "nvidia/Cosmos3-Nano"),
                "model_revision": os.environ.get("COSMOS_MODEL_REVISION", "unknown"),
                "temperature": 0, "seed": 42, "max_tokens": 256,
                "timeout_seconds": float(os.environ.get("COSMOS_TIMEOUT", "300")),
                "enable_thinking": False, "workers": api_nproc}
    if isinstance(api_nproc, bool) or not isinstance(api_nproc, int) or not 1 <= api_nproc <= 4:
        raise ValueError("api_nproc must be an integer from 1 to 4")
    if settings["timeout_seconds"] <= 0 or not math.isfinite(settings["timeout_seconds"]):
        raise ValueError("COSMOS_TIMEOUT must be finite and positive")
    # Deliberately do not persist API URL or key; userinfo/query can contain credentials.
    return base, os.environ.get("COSMOS_API_KEY", "EMPTY"), settings


def sample_video(path):
    # Lazy imports keep public downloads, scoring and qualification CPU-only and lightweight.
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from vlmeval.api.cosmos_reason import tensor_video_to_base64

    reader = VideoReader(str(path), ctx=cpu(0), height=224, width=224, num_threads=1)
    total, source_fps = len(reader), float(reader.get_avg_fps())
    if total < 1 or source_fps <= 0 or not math.isfinite(source_fps):
        raise ValueError("Video has no usable frames or FPS")
    segment = float(total - 1) / 32
    start = int(segment / 2)
    indices = np.array([start + int(np.round(segment * index)) for index in range(32)])
    frames = reader.get_batch(indices).asnumpy()
    fps = 32 / (total / source_fps)
    encoded = tensor_video_to_base64(torch.from_numpy(frames).permute(0, 3, 1, 2), fps=fps, crf=0)
    return "data:video/mp4;base64," + encoded, {"source_frames": total, "source_fps": source_fps,
            "indices": indices.tolist(), "sampled_frames": 32, "width": 224, "height": 224,
            "transport_fps": fps, "crf": 0,
            "transport_base64_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def audit_response(raw, processor_profile=None):
    from vlmeval.execution_evidence import build_execution_record, nano_processor_profile

    profile = processor_profile or nano_processor_profile('nvidia/Cosmos3-Nano')
    evidence = build_execution_record(
        {'messages': [{'content': [{'type': 'video_url'}]}]}, raw,
        media={'expected_frame_slots': 32, 'expected_group_tokens': 49}, processor_profile=profile)
    if evidence['verification']['status'] != 'verified_under_explicit_profile':
        raise ValueError('32-frame geometry audit failed: require 16 bounded groups of 49 video tokens (224x224 grid)')
    count = sum(group['tokens'] for group in evidence['observed']['video_token_groups'])
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Incomplete or ambiguous response choices")
    choice = choices[0]
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("Incomplete response finish reason")
    message = choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise ValueError("Incomplete response message")
    content = message["content"]
    if content is not None and not isinstance(content, str):
        raise ValueError("Unexpected response content type")
    return content or "", choice["finish_reason"], count


def _redact(value, secrets):
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def _predict(pair, root, work, api, key, settings):
    import requests
    from vlmeval.execution_evidence import build_execution_record, processor_profile_for

    profile = processor_profile_for(settings['model'], settings['model_revision'])

    index, row = pair
    started = time.monotonic()
    result = {"index": index, "sample_id": f"videophy2_test:{index:05d}", "video_url": row["video_url"],
              "model_name": row["model_name"], "truth": {task: int(row[task]) for task in ("sa", "pc", "joint")},
              "is_hard": int(row["is_hard"]), "responses": {}}
    parts = urllib.parse.urlsplit(api)
    secrets = [key if key != "EMPTY" else "", api, parts.username, parts.password, parts.query]
    try:
        video, sampling = sample_video(root / "media" / f"{index:05d}.mp4")
        result["sampling"] = sampling
        for task in ("sa", "pc"):
            question = (f'Does this video match the description: "{row["caption"]}"? Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.' if task == "sa" else PC_QUESTION)
            payload = {"model": settings["model"], "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": [{"type": "video_url", "video_url": {"url": video}}, {"type": "text", "text": question}]}],
                       "temperature": settings["temperature"], "seed": settings["seed"], "max_tokens": settings["max_tokens"],
                       "chat_template_kwargs": {"enable_thinking": settings["enable_thinking"]},
                       "mm_processor_kwargs": {"do_sample_frames": False, "fps": sampling["transport_fps"]},
                       "media_io_kwargs": {"video": {"num_frames": 32, "fps": sampling["transport_fps"], "min_frames": 32, "max_frames": 32}},
                       "return_token_ids": True}
            evidence = work / "responses" / f"{index:05d}-{task}.json"
            request_metadata = {name: value for name, value in payload.items() if name != "messages"}
            request_metadata.update(system_prompt=SYSTEM_PROMPT, question=question, sampling=sampling)
            record = {"request": request_metadata}
            record['execution_evidence'] = build_execution_record(
                payload, {}, media=dict(sampling, expected_frame_slots=32, expected_group_tokens=49),
                processor_profile=profile,
                requested_model_revision=os.environ.get("COSMOS_MODEL_REVISION", "unknown"))
            before = time.monotonic()
            try:
                response = requests.post(api, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=settings["timeout_seconds"])
                record["http_status"] = response.status_code
                try:
                    raw = response.json()
                    record["raw"] = raw
                    record['execution_evidence'] = build_execution_record(
                        payload, raw if 200 <= response.status_code < 300 else {},
                        media=dict(sampling, expected_frame_slots=32, expected_group_tokens=49),
                        processor_profile=profile,
                        requested_model_revision=os.environ.get("COSMOS_MODEL_REVISION", "unknown"))
                except ValueError:
                    record["response_text"] = response.text
                    raise ValueError("Non-JSON endpoint response") from None
                response.raise_for_status()
                if not isinstance(raw, dict):
                    raise ValueError("Non-object endpoint response")
                if profile is None:
                    raise ValueError('VideoPhy2 geometry audit requires a processor profile: the pinned Nano '
                                     'profile or COSMOS_PROCESSOR_PROFILE for another model')
                output, finish, count = audit_response(raw, processor_profile=profile)
                rating, fallback = official_parse(output)
                issues = []
                if finish == "length":
                    issues.append("truncated")
                if output.strip() not in ("1", "2", "3", "4", "5"):
                    issues.append("invalid_categorical_answer")
                result["responses"][task] = {"score": rating, "parser_fallback": fallback, "output": output,
                    "finish_reason": finish, "visual_token_count": count, "issues": issues,
                    "execution_evidence": record['execution_evidence'],
                    "evidence": str(evidence.relative_to(work)), "seconds": time.monotonic() - before}
            except Exception as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc) if isinstance(exc, ValueError) else "Endpoint request failed; see HTTP status and raw response"}
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - before
                _json(evidence, _redact(record, secrets))
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc) if isinstance(exc, ValueError) else "Media decode or endpoint request failed; inspect response evidence"}
    result["seconds"] = time.monotonic() - started
    return _redact(result, secrets)


def _preflight(root, chosen):
    path = root / "media-manifest.jsonl"
    entries = {entry["index"]: entry for entry in (json.loads(line) for line in path.read_text().splitlines())}
    selected = []
    for index, row in chosen:
        entry = entries.get(index)
        target = root / "media" / f"{index:05d}.mp4"
        if (entry is None or "error" in entry or entry.get("video_url") != row["video_url"]
                or entry.get("sample_id") != f"videophy2_test:{index:05d}" or not target.is_file()
                or entry.get("dataset_sha256") != DATASET_SHA256
                or (entry.get("archive_entry") is not None and
                    (entry.get("archive_revision") != ARCHIVE_REVISION or entry.get("archive_sha256") != ARCHIVE_SHA256))
                or target.stat().st_size <= 0 or entry.get("bytes") != target.stat().st_size
                or entry.get("sha256") != _sha256(target)):
            raise ValueError(f"Missing or unverified VideoPhy2 media for row {index}; run download first")
        selected.append(entry)
    return selected


def run(data_root: Path, scope: str, work: Path, api_nproc: int = 4):
    """Require complete selected media, issue audited requests, and retain every rating."""
    root, work = _outside_source(data_root), _outside_source(work)
    chosen = _selected(_rows(root), scope)
    api, key, settings = _settings(api_nproc)
    work.mkdir(parents=True, exist_ok=False, mode=0o700)
    (work / "responses").mkdir(mode=0o700)
    started, started_unix = time.monotonic(), time.time()
    from vlmeval.execution_evidence import aggregate_execution_records, PINNED_SERVING_COMPATIBILITY
    protocol = {"name": "official_public_categorical_32frame_protocol_adapted", "sampling": "official get_index segment-center, decord resize",
                "frames": 32, "width": 224, "height": 224, "transport": "lossless CRF0 MP4 via native tensor_video_to_base64",
                "system_prompt": SYSTEM_PROMPT, "parser": "official num_map substring then digit fallback; invalid blank returns zero",
                "media_io_override": "num_frames=min_frames=max_frames=32; positive transport fps; do_sample_frames=false",
                "response_audit": {"required_prompt_token_id": 151656, "required_count": 784,
                                   "required_bounded_groups": 16, "required_tokens_per_group": 49,
                                   "frame_slots": 32, "decoded_frames": None, "spatial_grid_area_pixels": 50176},
                "generation": settings, "truncation_policy": "retain and score with official parser; flag generation issue; never regenerate"}
    provenance = {"benchmark": "videophy2", "dataset": {"url": DATASET_URL, "revision": DATASET_REVISION, "sha256": DATASET_SHA256, "rows": 3397},
                  "archive": {"url": ARCHIVE_URL, "revision": ARCHIVE_REVISION, "sha256": ARCHIVE_SHA256},
                  "official_source": {"url": SOURCE_URL, "revision": SOURCE_REVISION, "license": "MIT"},
                  "runner_sha256": _sha256(Path(__file__)), "framework_revision": os.environ.get("COSMOS_COMMIT", "unknown"),
                  "model": settings["model"], "model_revision": settings["model_revision"],
                  "execution_evidence": {"summary": "execution-summary.json", "per_response": "responses/*-*.json"},
                  "pinned_runtime_compatibility": PINNED_SERVING_COMPATIBILITY,
                  "reference_parity": False, "protocol": protocol, "sample_indices": [index for index, _ in chosen], "started_unix": started_unix}
    encoder_source = Path(__file__).resolve().parent.parent / "vlmeval" / "api" / "cosmos_reason.py"
    provenance["transport_encoder_sha256"] = _sha256(encoder_source)
    provenance["execution_evidence_source_sha256"] = _sha256(encoder_source.parent.parent / "execution_evidence.py")
    _json(work / "provenance.json", provenance)
    warnings = ["Protocol-adapted public categorical evaluation; not a reproduction of the published Cosmos3-Nano 45.6 reference."]
    results = []
    execution = aggregate_execution_records([], expected_requests=2 * len(chosen))
    try:
        media = _preflight(root, chosen)
        (work / "media-manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in media))
        with (work / "predictions.jsonl").open("w") as predictions, (work / "run.log").open("w") as log:
            with concurrent.futures.ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
                futures = [executor.submit(_predict, pair, root, work, api, key, settings) for pair in chosen]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    predictions.write(json.dumps(result, allow_nan=False) + "\n")
                    predictions.flush()
                    results.append(result)
                    progress = json.dumps({"index": result["index"], "completed": len(results),
                                           "expected": len(chosen), "error": result.get("error"),
                                           "seconds": result["seconds"]})
                    log.write(progress + "\n")
                    log.flush()
                    print(progress, flush=True)
        from vlmeval.execution_evidence import aggregate_execution_records
        execution = aggregate_execution_records(
            [json.loads(path.read_text()).get("execution_evidence", {}) for path in sorted((work / "responses").glob("*.json"))],
            expected_requests=2 * len(chosen))
        execution.update(summary="execution-summary.json", per_response="responses/*-*.json")
        _json(work / "execution-summary.json", execution)
        provenance["execution_evidence"] = execution
        _json(work / "provenance.json", provenance)
        complete = [row for row in results if "error" not in row and all(task in row["responses"] for task in ("sa", "pc"))]
        if len(complete) != len(chosen):
            raise RuntimeError("VideoPhy2 transport/coverage failure; evidence retained, no partial benchmark score")
        scores = correlations(complete)
        groups = {"all": scores}
        for name in sorted({row["model_name"] for row in complete}):
            groups[name] = correlations([row for row in complete if row["model_name"] == name])
        groups["unseen_prompts"] = correlations([row for row in complete if row["model_name"] in ("hunyuan", "cosmos", "cogvideo")])
        groups["unseen_video_models"] = correlations([row for row in complete if row["model_name"] not in ("hunyuan", "cosmos", "cogvideo")])
        issue_count = sum(bool(answer["issues"]) for row in complete for answer in row["responses"].values())
        if issue_count:
            warnings.append(f"{issue_count} answers had truncation or invalid categorical generation; all remain scored with the official parser.")
        if scores["mean_sa_pc_pearson_x100"] is None:
            warnings.append("One or both Pearson correlations are undefined (fewer than two rows or zero variance); null is intentional.")
        summary = {"benchmark": "videophy2", "scope": scope, "status": "completed_with_generation_issues" if issue_count else "completed",
                   "primary_metric": {"name": "mean_sa_pc_pearson_x100", "value": scores["mean_sa_pc_pearson_x100"], "unit": "pearson_x100"},
                   "coverage": {"expected": len(chosen), "scored": len(complete)}, "metrics": scores, "groups": groups,
                   "generation_issue_answers": issue_count, "protocol": protocol, "warnings": warnings,
                   "execution_evidence": execution,
                   "reference_parity": False, "model": settings["model"], "model_revision": settings["model_revision"],
                   "sample_indices": [index for index, _ in chosen], "started_unix": started_unix,
                   "ended_unix": time.time(), "elapsed_seconds": time.monotonic() - started}
        _json(work / "summary.json", summary)
        return summary
    except Exception as exc:
        scored = sum("error" not in row and all(task in row["responses"] for task in ("sa", "pc")) for row in results)
        _json(work / "execution-summary.json", execution)
        provenance["execution_evidence"] = dict(execution, summary="execution-summary.json", per_response="responses/*-*.json")
        _json(work / "provenance.json", provenance)
        _json(work / "summary.json", {"benchmark": "videophy2", "scope": scope, "status": "failed",
              "primary_metric": {"name": "mean_sa_pc_pearson_x100", "value": None, "unit": "pearson_x100"},
              "coverage": {"expected": len(chosen), "scored": scored}, "protocol": protocol,
              "execution_evidence": provenance["execution_evidence"],
              "warnings": warnings + ["Incomplete transport or media coverage; no benchmark score is reported."],
              "error": type(exc).__name__, "reference_parity": False, "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "run"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--api-nproc", type=int, default=4)
    args = parser.parse_args()
    if args.action == "run" and args.work is None:
        parser.error("run requires --work (a new output directory)")
    result = download(args.data_root, args.scope) if args.action == "download" else run(args.data_root, args.scope, args.work, args.api_nproc)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
