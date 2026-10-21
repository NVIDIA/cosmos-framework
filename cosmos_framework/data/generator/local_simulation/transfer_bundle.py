# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a local Sim-Transfer bundle from RGB/depth/segmentation clips with plain-text captions.

The bundle feeds ``LocalSimulationDataset`` for Sim-Transfer Stage 1, 2 and 3. Each clip contributes
its longest caption window that the production length policy accepts; edge and blur controls are
computed from RGB at load time, depth and segmentation are read from the copied videos.

The ``paibench-c`` command converts the public PAI-Bench conditional-generation dataset
(``shi-labs/physical-ai-bench-conditional-generation`` on Hugging Face)::

    python -m cosmos_framework.data.generator.local_simulation.transfer_bundle paibench-c \\
        --source-root <dataset dir> --tasks 0-199 --tokenizer <Qwen3-VL tokenizer dir> --output <new dir>
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.local_simulation.core import FORMAT_VERSION
from cosmos_framework.data.generator.transfer_length_index import LengthIndexPolicy, select_indexed_video

MODALITIES = ("edge", "blur", "depth", "seg")
CONTROL_ASSETS = ("depth", "seg")
CAPTION_KEY = "plain_caption"
TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
)
REQUIRED_TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


@dataclass(frozen=True)
class TransferClip:
    """One source clip: RGB, optional aligned depth/segmentation videos and a plain-text caption."""

    identity: str
    rgb: Path
    caption: str
    depth: Path | None = None
    seg: Path | None = None


@dataclass
class _Candidate:
    """Metadata protocol consumed by the production length-index selector."""

    rgb: dict[str, Any]
    caption_chunks: dict[str, dict[str, Any]]
    persisted_by_modality: dict[str, dict[str, Any]]
    available_modalities: tuple[str, ...] = MODALITIES


def probe_video(path: Path) -> dict[str, Any]:
    """Decode every frame: the loader checks decoded counts and geometry, not container headers."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None:
            raise ValueError(f"Video has no frame rate: {path}")
        frames = 0
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (stream.width, stream.height):
                raise ValueError(f"Video geometry changes mid-stream: {path}")
            frames += 1
        metadata = {
            "width": stream.width,
            "height": stream.height,
            "framerate": float(stream.average_rate),
            "nb_frames": frames,
        }
    if any(not math.isfinite(value) or value <= 0 for value in metadata.values()):
        raise ValueError(f"Invalid video metadata: {path}")
    return metadata


def _record(
    clip: TransferClip, index: int, dataset_name: str, policy: LengthIndexPolicy, exclusions: Counter[str]
) -> dict[str, Any] | None:
    caption = clip.caption.strip()
    if not caption:
        exclusions["empty_caption"] += 1
        return None
    metadata = probe_video(clip.rgb)
    controls = {name: path for name in CONTROL_ASSETS if (path := getattr(clip, name)) is not None}
    for name, path in controls.items():
        if probe_video(path) != metadata:
            exclusions[f"{name}_not_aligned_with_rgb"] += 1
            return None
    locator = {
        "data_uri": f"local-bundle://{dataset_name}/{clip.identity}/rgb.mp4",
        "stream_id": "rgb",
        "width_px": metadata["width"],
        "height_px": metadata["height"],
        "total_frames": metadata["nb_frames"],
        "framerate": metadata["framerate"],
    }
    # The loader reads captions as structured dictionaries; a plain caption is the dense
    # ``temporal_caption`` field, which the bundle selects for every draw.
    structured = json.dumps({"temporal_caption": caption})
    chunks = {
        f"chunk_0_{metadata['nb_frames']}": {
            "start_frame": 0,
            "end_frame": metadata["nb_frames"],
            "caption": structured,
        }
    }
    candidate = _Candidate(locator, chunks, {name: dict(locator) for name in controls})
    selected = select_indexed_video(candidate, {"_rowid": index, "sample_id": clip.identity}, policy, exclusions)
    if selected is None:
        return None
    if selected.bucket < 3:
        exclusions["shorter_than_9_frames"] += 1
        return None
    assets = {"rgb": f"assets/{clip.identity}/rgb.mp4"} | {
        name: f"assets/{clip.identity}/{name}.mp4" for name in controls
    }
    entry = {
        "sample_key": clip.identity,
        "aspect_ratio": _aspect_ratio(metadata["width"], metadata["height"]),
        "caption_key": selected.caption_key,
        "caption_start": selected.caption_start,
        "caption_end": selected.caption_end,
        "caption_json": structured,
        "source_frames": selected.source_frames,
        "source_fps": selected.source_fps,
        "source_width": metadata["width"],
        "source_height": metadata["height"],
    }
    for name in ("rgb", *CONTROL_ASSETS):
        entry[f"{name}_locator_json"] = (
            json.dumps({"uri": f"local-bundle://{dataset_name}/{clip.identity}/{name}.mp4", "metadata": metadata})
            if name in assets
            else None
        )
    return {"identity": clip.identity, "stratum": str(selected.bucket), "entry": entry, "assets": assets}


def _aspect_ratio(width: int, height: int) -> str:
    gcd = math.gcd(width, height)
    return f"{width // gcd},{height // gcd}"


def build_transfer_bundle(
    clips: Iterable[TransferClip],
    output: Path,
    tokenizer: Path,
    *,
    dataset_name: str,
    resolution: str = "480",
    cfg_dropout_rate: float = 0.0,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a complete bundle; return the sample count and the per-reason exclusion counts.

    ``cfg_dropout_rate`` is the bundle default; local experiments override it per stage.
    """
    if output.exists():
        raise ValueError(f"Output must be a new directory: {output}")
    for name in REQUIRED_TOKENIZER_FILES:
        if not (tokenizer / name).is_file():
            raise ValueError(f"Missing Qwen tokenizer file: {tokenizer / name}")
    policy = LengthIndexPolicy()
    exclusions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    sources: list[TransferClip] = []
    seen: set[str] = set()
    for index, clip in enumerate(clips):
        if clip.identity in seen or not clip.identity or "/" in clip.identity or clip.identity.startswith("."):
            raise ValueError(f"Clip identities must be unique path components: {clip.identity!r}")
        seen.add(clip.identity)
        record = _record(clip, index, dataset_name, policy, exclusions)
        if record is not None:
            records.append(record)
            sources.append(clip)
    if not records:
        raise ValueError(f"No clip passed the length policy: {dict(exclusions)}")

    output.mkdir(parents=True)
    for name in TOKENIZER_FILES:
        if (tokenizer / name).is_file():
            destination = output / "resources" / "tokenizer" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tokenizer / name, destination)
    for record, clip in zip(records, sources, strict=True):
        for name, relative in record["assets"].items():
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(getattr(clip, name), destination)
    transform = {
        "dataset_name": dataset_name,
        "caption_key": CAPTION_KEY,
        "conditioning_config": {"__mapping__": [[0, 0.7], [1, 0.2], [2, 0.1]]},
        "policy": asdict(policy),
        "resolution": resolution,
        "min_num_frames": 9,
        "cfg_dropout_rate": cfg_dropout_rate,
        # Plain captions only exist as the dense temporal caption; select it for every draw.
        "dense_caption_probability": 1.0,
        "resize_on_read": True,
        "modality_weights": dict.fromkeys(MODALITIES, 1.0),
    }
    (output / "samples.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    manifest = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "model": "transfer",
        "sample_count": len(records),
        "transforms": {stratum: transform for stratum in sorted({record["stratum"] for record in records})},
        "provenance": {**(provenance or {}), "excluded": dict(exclusions)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"samples": len(records), "excluded": dict(exclusions)}


def paibench_c_clips(source_root: Path, tasks: Iterable[str]) -> list[TransferClip]:
    """PAI-Bench conditional generation: ``videos/``, ``depth_vids/``, ``sam2_vids/``, ``captions/<task>.json``."""
    clips = []
    for task in tasks:
        caption = json.loads((source_root / "captions" / f"{task}.json").read_text())["caption"]
        clips.append(
            TransferClip(
                identity=task,
                rgb=source_root / "videos" / f"{task}.mp4",
                depth=source_root / "depth_vids" / f"{task}.mp4",
                seg=source_root / "sam2_vids" / f"{task}.mp4",
                caption=caption,
            )
        )
    return clips


def parse_task_ranges(spec: str) -> list[str]:
    """``"0-199,250"`` -> ``["task_0000", ..., "task_0199", "task_0250"]``."""
    tasks: list[str] = []
    for part in spec.split(","):
        first, _, last = part.strip().partition("-")
        tasks.extend(f"task_{index:04d}" for index in range(int(first), int(last or first) + 1))
    return tasks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    paibench = commands.add_parser("paibench-c", help="Convert PAI-Bench conditional-generation clips.")
    paibench.add_argument("--source-root", type=Path, required=True)
    paibench.add_argument("--tasks", required=True, help="Task index ranges, e.g. 0-199 (the AgiBot robotics clips).")
    paibench.add_argument("--tokenizer", type=Path, required=True, help="Directory with the Qwen3-VL tokenizer files.")
    paibench.add_argument("--output", type=Path, required=True)
    paibench.add_argument("--dataset-name", default="paibench_c")
    args = parser.parse_args(argv)
    summary = build_transfer_bundle(
        paibench_c_clips(args.source_root, parse_task_ranges(args.tasks)),
        args.output,
        args.tokenizer,
        dataset_name=args.dataset_name,
        provenance={"source": "shi-labs/physical-ai-bench-conditional-generation", "tasks": args.tasks},
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
