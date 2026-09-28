# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build fingerprinted, NVDEC-compatible video override artifacts.

This is an explicit preparation step for source streams that exceed the
declared NVDEC macroblock limit. Compatible streams remain on the GPU decoder;
the runtime's capability-scoped fallback remains a last-resort safety route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIDEO_KEYS = ("video", "video_id", "media", "media_path")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
TRANSCODE_SUFFIX = ".mp4"
TRANSCODE_VIDEO_CODEC = "libx264"
IMAGE_PROVENANCE_PATH = Path("/opt/cosmos/image-provenance.json")


def _integration_commit_from_provenance(path: Path = IMAGE_PROVENANCE_PATH) -> str | None:
    """Resolve the Framework implementation commit from image provenance."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    repositories = payload.get("repositories", {})
    if isinstance(repositories, dict):
        for name in ("cosmos-framework",):
            repository = repositories.get(name, {})
            if isinstance(repository, dict):
                commit = repository.get("commit")
                if isinstance(commit, str) and commit:
                    return commit
    for key in ("SOURCE_COMMIT", "repository_commit"):
        commit = payload.get(key)
        if isinstance(commit, str) and commit:
            return commit
    return None


def _integration_commit() -> str | None:
    """Use an explicit launch binding or the commit baked into the image."""
    return (
        os.environ.get("COSMOS_INTEGRATION_COMMIT")
        or os.environ.get("SOURCE_COMMIT")
        or _integration_commit_from_provenance()
    )


def _bundled_ffmpeg() -> str:
    """Return imageio-ffmpeg's pinned binary, ignoring the runtime override."""
    import imageio_ffmpeg

    overridden = os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    finally:
        if overridden is not None:
            os.environ["IMAGEIO_FFMPEG_EXE"] = overridden


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _media_paths(annotation_path: Path, fallback_root: Path | None) -> set[Path]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    root_value = payload.get("media_root") if isinstance(payload, dict) else None
    media_root = Path(root_value) if isinstance(root_value, str) else fallback_root
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in VIDEO_KEYS and isinstance(child, str) and Path(child).suffix.lower() in VIDEO_SUFFIXES:
                    values.add(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    resolved: set[Path] = set()
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            if media_root is None:
                raise ValueError(f"relative media path {value!r} has no media root in {annotation_path}")
            path = media_root / path
        path = path.absolute()
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.add(path)
    return resolved


def _probe(path: Path, _ffprobe: str | None = None) -> dict[str, Any]:
    """Read stream metadata through the source-baked software media stack.

    Header inspection must be CPU-safe because decoder-artifact preparation is
    intentionally independent of GPU allocation.  Some action images package
    an FFprobe whose H.264/HEVC decoder choice is CUVID-only; merely asking it
    for stream metadata can therefore require ``libnvcuvid`` or reject a stream
    that exceeds the target GPU's decode capability.  PyAV uses the image's
    separately attested software codecs and does not decode frames here.

    ``_ffprobe`` remains accepted for compatibility with existing preparation
    commands, but it cannot alter the deterministic software probe path.
    """
    import av

    container = av.open(str(path), mode="r")
    try:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise RuntimeError(f"expected a video stream in {path}, got none")
        stream = streams[0]
        codec_context = stream.codec_context
        width, height = int(codec_context.width), int(codec_context.height)
        if width < 1 or height < 1:
            raise RuntimeError(f"video stream dimensions are unavailable in {path}: {width}x{height}")

        frame_rate = None
        if stream.average_rate is not None:
            frame_rate = f"{stream.average_rate.numerator}/{stream.average_rate.denominator}"
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = str(float(stream.duration * stream.time_base))
        elif container.duration is not None:
            duration = str(float(container.duration / av.time_base))

        return {
            "codec": str(codec_context.name or "unknown"),
            "width": width,
            "height": height,
            "macroblocks": math.ceil(width / 16) * math.ceil(height / 16),
            "frames": str(stream.frames) if stream.frames else None,
            "frame_rate": frame_rate,
            "duration": duration,
        }
    finally:
        container.close()


def _requires_override(probe: dict[str, Any], max_macroblocks: int) -> bool:
    """Match the hardware limit reported by NVDEC, independent of codec."""
    return int(probe["macroblocks"]) > max_macroblocks


def _resolve_forced_media(values: list[str], media: set[Path]) -> set[Path]:
    """Resolve diagnosed runtime-incompatible streams against dataset media.

    Forced overrides are intentionally constrained to media discovered from
    the supplied annotations.  Both logical and real paths are accepted so a
    caller can preserve ``/lustre/fsw`` annotation identities while preparing
    the artifact on a compute node where that path resolves to ``/lustre/fs11``.
    """
    by_identity: dict[str, Path] = {}
    for path in media:
        by_identity[str(path)] = path
        by_identity[str(path.resolve(strict=True))] = path

    forced: set[Path] = set()
    for value in values:
        candidate = Path(value).expanduser().absolute()
        match = by_identity.get(str(candidate))
        if match is None and candidate.is_file():
            match = by_identity.get(str(candidate.resolve(strict=True)))
        if match is None:
            raise ValueError(f"forced video is not present in the supplied annotation set: {value}")
        forced.add(match)
    return forced


def _annotation_inputs(
    annotations: list[str],
    annotation_media_roots: list[list[str]],
    fallback_root: Path | None,
) -> list[tuple[Path, Path | None]]:
    """Resolve annotations with an explicit media root for each input when needed."""
    inputs: list[tuple[Path, Path | None]] = []
    inputs.extend((Path(value).expanduser().resolve(strict=True), fallback_root) for value in annotations)
    inputs.extend(
        (
            Path(annotation).expanduser().resolve(strict=True),
            Path(media_root).expanduser().resolve(strict=True),
        )
        for annotation, media_root in annotation_media_roots
    )

    deduplicated: dict[Path, Path | None] = {}
    for annotation, media_root in inputs:
        previous = deduplicated.get(annotation)
        if annotation in deduplicated and previous != media_root:
            raise ValueError(f"annotation has conflicting media roots: {annotation}: {previous} != {media_root}")
        deduplicated[annotation] = media_root
    if not deduplicated:
        raise ValueError("at least one annotation input is required")
    return list(deduplicated.items())


def _resolve_forced_annotations(
    values: list[str],
    media_by_annotation: dict[Path, set[Path]],
) -> set[Path]:
    """Resolve annotations whose complete media set must be overridden."""
    by_identity: dict[str, Path] = {}
    for annotation in media_by_annotation:
        by_identity[str(annotation)] = annotation
        by_identity[str(annotation.resolve(strict=True))] = annotation

    forced: set[Path] = set()
    for value in values:
        candidate = Path(value).expanduser().absolute()
        match = by_identity.get(str(candidate))
        if match is None and candidate.is_file():
            match = by_identity.get(str(candidate.resolve(strict=True)))
        if match is None:
            raise ValueError(f"forced annotation is not present in the supplied annotation set: {value}")
        forced.add(match)
    return forced


def _transcode_command(source: Path, temporary: Path, ffmpeg: str) -> list[str]:
    """Use only codecs and muxers packaged by the Cosmos Cosmos action image."""
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "scale=w=1920:h=1080:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v",
        TRANSCODE_VIDEO_CODEC,
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        str(temporary),
    ]


def _transcode(
    source: Path,
    output_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    max_macroblocks: int,
) -> tuple[Path, dict[str, Any]]:
    source_sha256 = _sha256(source)
    output = output_dir / f"{source_sha256}{TRANSCODE_SUFFIX}"
    if not output.is_file():
        temporary = output.with_suffix(f".{os.getpid()}.tmp{TRANSCODE_SUFFIX}")
        command = _transcode_command(source, temporary, ffmpeg)
        try:
            subprocess.run(command, check=True)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    output_probe = _probe(output, ffprobe)
    if output_probe["macroblocks"] > max_macroblocks:
        raise RuntimeError(f"transcoded output still exceeds NVDEC macroblock limit: {output}")
    return output, {
        "source_sha256": source_sha256,
        "output_sha256": _sha256(output),
        "output_probe": output_probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", action="append", default=[])
    parser.add_argument("--media-root")
    parser.add_argument(
        "--annotation-media-root",
        action="append",
        nargs=2,
        default=[],
        metavar=("ANNOTATION", "MEDIA_ROOT"),
        help=("Annotation and its fallback media root; repeat when splits or annotation groups use different roots."),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--override-map", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--processor-fingerprint", required=True)
    parser.add_argument("--max-macroblocks", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--force-video",
        action="append",
        default=[],
        help=(
            "Dataset video diagnosed as runtime-incompatible with GPU decoding; "
            "repeat for multiple paths. The path must occur in the supplied annotations."
        ),
    )
    parser.add_argument(
        "--force-annotation",
        action="append",
        default=[],
        help=(
            "Override every video referenced by this supplied annotation. "
            "Use this for deterministic validation decoding; repeat for each "
            "validation annotation."
        ),
    )
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    ffmpeg = args.ffmpeg or _bundled_ffmpeg()

    if args.max_macroblocks < 1 or args.workers < 1:
        raise ValueError("max-macroblocks and workers must be positive")
    integration_commit = _integration_commit()
    if not integration_commit:
        raise ValueError("integration commit is absent from the launch binding and image provenance")
    fallback_root = Path(args.media_root).expanduser().resolve(strict=True) if args.media_root else None
    annotation_inputs = _annotation_inputs(args.annotation, args.annotation_media_root, fallback_root)
    media_by_annotation: dict[Path, set[Path]] = {}
    media: set[Path] = set()
    for annotation, media_root in annotation_inputs:
        annotation_media = _media_paths(annotation, media_root)
        media_by_annotation[annotation] = annotation_media
        media.update(annotation_media)
    forced_annotations = _resolve_forced_annotations(args.force_annotation, media_by_annotation)
    forced_media = _resolve_forced_media(args.force_video, media)
    for annotation in forced_annotations:
        forced_media.update(media_by_annotation[annotation])
    ordered_media = sorted(media, key=str)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        probes = list(pool.map(lambda path: _probe(path, args.ffprobe), ordered_media))

    incompatible = [
        (path, probe)
        for path, probe in zip(ordered_media, probes, strict=True)
        if path in forced_media or _requires_override(probe, args.max_macroblocks)
    ]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert(item: tuple[Path, dict[str, Any]]) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
        source, source_probe = item
        output, hashes = _transcode(source, output_dir, ffmpeg, args.ffprobe, args.max_macroblocks)
        return source, source_probe, output, hashes

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        converted = list(pool.map(convert, incompatible))
    overrides: dict[str, str] = {}
    for source, _probe_value, output, _hashes in converted:
        # Dataset adapters preserve the supplied /lustre/fsw logical path,
        # while compute-node inspection resolves it to /lustre/fs11. Accept
        # both identities without changing the source annotations.
        overrides[str(source)] = str(output)
        overrides[str(source.resolve(strict=True))] = str(output)
    records = [
        {
            "source": str(source),
            "source_realpath": str(source.resolve(strict=True)),
            "source_probe": source_probe,
            "override_reasons": [
                *(["macroblock_limit"] if _requires_override(source_probe, args.max_macroblocks) else []),
                *(["forced_runtime_incompatibility"] if source in forced_media else []),
            ],
            "output": str(output),
            **hashes,
        }
        for source, source_probe, output, hashes in converted
    ]
    core = {
        "schema_version": 2,
        "dataset_fingerprint": args.dataset_fingerprint,
        "model_fingerprint": args.model_fingerprint,
        "processor_fingerprint": args.processor_fingerprint,
        "annotations": [
            {
                "path": str(annotation),
                "sha256": _sha256(annotation),
                "media_root": str(media_root) if media_root is not None else None,
                "media_count": len(media_by_annotation[annotation]),
                "force_all": annotation in forced_annotations,
            }
            for annotation, media_root in annotation_inputs
        ],
        "media_count": len(ordered_media),
        "incompatible_count": len(records),
        "forced_override_sources": sorted(str(path) for path in forced_media),
        "force_all_annotations": sorted(str(path) for path in forced_annotations),
        "max_macroblocks": args.max_macroblocks,
        "transcode": {
            "scale": "fit:1920x1080:force_divisible_by=2",
            "container": "mp4",
            "video_codec": TRANSCODE_VIDEO_CODEC,
            "preset": "slow",
            "crf": 18,
            "pixel_format": "yuv420p",
            "audio": "excluded",
            "threads": 1,
            "metadata": "excluded",
        },
        "records": records,
        "overrides": overrides,
    }
    artifact_fingerprint = _stable_sha(core)
    manifest = {
        **core,
        "artifact_fingerprint": artifact_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "integration_commit": integration_commit,
        "complete": True,
    }
    override_path = Path(args.override_map).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    override_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(override_path, overrides)
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "artifact_fingerprint": artifact_fingerprint,
                "media_count": len(ordered_media),
                "incompatible_count": len(records),
                "override_map": str(override_path),
                "override_map_sha256": _sha256(override_path),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
