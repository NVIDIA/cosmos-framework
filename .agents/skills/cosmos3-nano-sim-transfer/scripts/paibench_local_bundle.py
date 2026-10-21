#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare structured-caption Transfer bundles and validate the native Framework data path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from cosmos_framework.data.generator.local_simulation.transfer_bundle import TransferClip

MODALITIES = ("edge", "blur", "depth", "seg")
OUTPUT_METADATA_KEYS = ("resolution", "aspect_ratio", "duration", "fps")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    """Count decoded frames; do not trust container duration as an alignment check."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"File has no video stream: {path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if stream.average_rate is None:
            raise ValueError(f"Video has no frame rate: {path}")
        timestamps: list[Fraction] = []
        for frame in container.decode(stream):
            if (
                frame.pts is None
                or frame.time_base is None
                or (frame.width, frame.height) != (stream.width, stream.height)
            ):
                raise ValueError(f"Missing frame timestamps or changing video geometry: {path}")
            timestamps.append(frame.pts * frame.time_base)
        metadata: dict[str, Any] = {
            "width": stream.width,
            "height": stream.height,
            "framerate": float(stream.average_rate),
            "nb_frames": len(timestamps),
        }
    if any(not math.isfinite(value) or value <= 0 for value in metadata.values()):
        raise ValueError(f"Invalid media metadata: {path}")
    # Normalize exact media timestamps before converting to floats; container start times may differ.
    metadata["timestamps"] = [float(time - timestamps[0]) for time in timestamps]
    return metadata


def training_caption(opus_path: Path) -> dict[str, Any]:
    caption = json.loads(opus_path.read_text())
    if not isinstance(caption, dict) or not isinstance(caption.get("temporal_caption"), str):
        raise ValueError("Caption must be a structured JSON object with temporal_caption")
    if not caption["temporal_caption"].strip():
        raise ValueError("Caption temporal_caption is empty")
    # Framework augmentors append actual decoded duration/FPS/resolution.
    # Retain the saved prompt content separately for the inference walkthrough.
    return {key: value for key, value in caption.items() if key not in OUTPUT_METADATA_KEYS}


@dataclass
class Candidate:
    """Metadata protocol consumed by the real production length-index selector."""

    rgb: dict[str, Any]
    caption_chunks: dict[str, dict[str, Any]]
    persisted_by_modality: dict[str, dict[str, Any]]
    available_modalities: tuple[str, ...] = MODALITIES


def select_aspect_ratio(width: int, height: int) -> str:
    """Select a shared native 480/720 bucket and apply its real direct-resize guard."""
    from PIL import Image

    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
    from cosmos_framework.data.imaginaire.webdataset.augmentors.image.resize import ResizeToSize

    sizes = [VIDEO_RES_SIZE_INFO[tier] for tier in ("480", "720")]
    common = sorted(set(sizes[0]) & set(sizes[1]))
    aspect_ratio = min(
        common,
        key=lambda key: max(
            max(table[key][0] * height, table[key][1] * width) / min(table[key][0] * height, table[key][1] * width)
            for table in sizes
        ),
    )
    # Geometry-only input uses measured dimensions; actual source pixels remain untouched.
    source = Image.new("RGB", (width, height))
    for tier, table in zip(("480", "720"), sizes, strict=True):
        resize = ResizeToSize(
            input_keys=["image"],
            args={"size": table, "max_aspect_ratio_distortion": 0.03, "log_aspect_ratio_stats": False},
        )
        if resize({"image": source, "aspect_ratio": aspect_ratio}) is None:
            raise ValueError(f"Source {width}x{height} exceeds 3% aspect distortion at tier {tier}")
    return aspect_ratio


def stage0_720_exclusions(metadata: dict[str, Any], caption: dict[str, Any], identity: str) -> dict[str, int]:
    """Check the same complete caption against the native 149-frame Stage-0 ceiling."""
    from cosmos_framework.data.generator.transfer_length_index import LengthIndexPolicy, select_indexed_video

    asset = {
        "data_uri": f"local-bundle://eligibility/{identity}/rgb.mp4",
        "stream_id": "rgb",
        "width_px": metadata["width"],
        "height_px": metadata["height"],
        "total_frames": metadata["nb_frames"],
        "framerate": metadata["framerate"],
    }
    chunks = {"complete_clip": {"start_frame": 0, "end_frame": metadata["nb_frames"], "caption": json.dumps(caption)}}
    candidate = Candidate(asset, chunks, {})
    exclusions: Counter[str] = Counter()
    select_indexed_video(
        candidate, {"_rowid": 0, "sample_id": identity}, LengthIndexPolicy(max_latent_frames=38), exclusions
    )
    return dict(exclusions)


def custom_clips(
    manifest_path: Path,
    captions: dict[str, dict[str, Any]],
    aspect_ratios: dict[str, str],
    exclusions: list[dict[str, str]],
) -> Iterator[TransferClip]:
    """Validate full-clip annotations before the native builder selects temporal buckets."""
    import av

    from cosmos_framework.data.generator.local_simulation.transfer_bundle import TransferClip
    from cosmos_framework.inference.structured_caption import StructuredCaption

    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{manifest_path}:{line_number}: row must be an object")
        identity = row.get("identity")
        if (
            not isinstance(identity, str)
            or not identity
            or identity.startswith(".")
            or "/" in identity
            or identity in seen
        ):
            raise ValueError(f"{manifest_path}:{line_number}: identity must be a unique path component")
        seen.add(identity)
        paths: dict[str, Path] = {}
        for name in ("rgb", "depth", "seg", "caption_path"):
            if name in {"depth", "seg"} and row.get(name) is None:
                continue
            relative = row.get(name)
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"{identity}: missing nonempty {name} path")
            path = (manifest_path.parent / relative).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{identity}: missing {name} file: {path}")
            paths[name] = path
        try:
            caption = training_caption(paths["caption_path"])
            # Use the native permissive schema while preserving every original structured field.
            StructuredCaption.model_validate(caption)
        except ValueError as error:
            exclusions.append({"identity": identity, "reason": "invalid_caption", "detail": str(error)})
            continue
        try:
            metadata = {name: probe(path) for name, path in paths.items() if name != "caption_path"}
        except (ValueError, av.FFmpegError) as error:
            exclusions.append({"identity": identity, "reason": "invalid_media", "detail": str(error)})
            continue
        if any(value != metadata["rgb"] for value in metadata.values()):
            exclusions.append(
                {
                    "identity": identity,
                    "reason": "unaligned_control",
                    "detail": "Geometry, frames, FPS or relative timestamps differ",
                }
            )
            continue
        if (
            type(row.get("caption_start")) is not int
            or type(row.get("caption_end")) is not int
            or (row["caption_start"], row["caption_end"]) != (0, metadata["rgb"]["nb_frames"])
        ):
            exclusions.append(
                {
                    "identity": identity,
                    "reason": "caption_interval",
                    "detail": "Caption must cover exactly [0, decoded frame count)",
                }
            )
            continue
        try:
            aspect_ratios[identity] = select_aspect_ratio(metadata["rgb"]["width"], metadata["rgb"]["height"])
        except ValueError as error:
            exclusions.append({"identity": identity, "reason": "aspect_distortion", "detail": str(error)})
            continue
        temporal_exclusions = stage0_720_exclusions(metadata["rgb"], caption, identity)
        if temporal_exclusions:
            exclusions.append(
                {"identity": identity, "reason": "stage0_720_window", "detail": json.dumps(temporal_exclusions)}
            )
            continue
        captions[identity] = caption
        yield TransferClip(
            identity=identity,
            rgb=paths["rgb"],
            depth=paths.get("depth"),
            seg=paths.get("seg"),
            # The native builder accepts plain text; attach the full JSON only after indexing.
            caption=caption["temporal_caption"],
        )


def prepare_custom(manifest_path: Path, tokenizer: Path, output: Path, dataset_name: str) -> None:
    """Use the Framework builder, then retain both reviewed caption forms in each indexed record."""
    from cosmos_framework.data.generator.local_simulation.transfer_bundle import build_transfer_bundle

    captions: dict[str, dict[str, Any]] = {}
    aspect_ratios: dict[str, str] = {}
    exclusions: list[dict[str, str]] = []
    report_path = output.with_name(output.name + ".summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = build_transfer_bundle(
            custom_clips(manifest_path, captions, aspect_ratios, exclusions),
            output,
            tokenizer,
            dataset_name=dataset_name,
            resolution="480",
            cfg_dropout_rate=0.0,
            provenance={"source_manifest_sha256": file_hash(manifest_path)},
        )
    except (ValueError, FileNotFoundError) as error:
        write_json(report_path, {"status": "failed", "error": str(error), "excluded_samples": exclusions})
        raise
    manifest = json.loads((output / "manifest.json").read_text())
    # Mark incomplete while adding structured captions and their byte receipt.
    manifest["complete"] = False
    write_json(output / "manifest.json", manifest)
    records = json.loads((output / "samples.json").read_text())
    for record in records:
        caption = captions[record["identity"]]
        record["entry"]["caption_json"] = json.dumps(caption)
        record["entry"]["aspect_ratio"] = aspect_ratios[record["identity"]]
        prompt_dir = output / "prompts" / record["identity"]
        prompt_dir.mkdir(parents=True)
        write_json(prompt_dir / "training.json", caption)
        (prompt_dir / "dense.txt").write_text(caption["temporal_caption"] + "\n")
    for transform in manifest["transforms"].values():
        transform["caption_key"] = "reviewed_structured"
        transform["dense_caption_probability"] = 0.07
    write_json(output / "samples.json", records)
    counts = Counter(summary["excluded"])
    counts.update(exclusion["reason"] for exclusion in exclusions)
    manifest["provenance"].update(
        {
            "caption_mode": "structured_93_dense_7",
            "excluded": dict(counts),
            "excluded_samples": exclusions,
            "sha256": {
                path.relative_to(output).as_posix(): file_hash(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            },
        }
    )
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    write_json(
        report_path,
        {
            "status": "prepared",
            "samples": len(records),
            "excluded": dict(counts),
            "excluded_samples": exclusions,
            "model_validation": "not run",
        },
    )
    print(f"Prepared {len(records)} clips at {output}; exclusions: {dict(counts)}. Report: {report_path}")


def export_preview(path: Path, video: torch.Tensor, fps: float) -> None:  # video: [3,T,H,W]
    import av
    import torch

    pixels = ((video.detach().cpu() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)  # [3,T,H,W]
    pixels = pixels.permute(1, 2, 3, 0).contiguous()  # [T,H,W,3]
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator())
        stream.codec_context.thread_count = 1
        stream.width, stream.height, stream.pix_fmt = (
            pixels.shape[2],
            pixels.shape[1],
            "yuv444p",
        )
        stream.options = {"crf": "0"}
        for array in pixels.numpy():  # [H,W,3]
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def validate_runtime(root: Path, *, previews: bool = False, resolution: str | None = None) -> dict[str, Any]:
    """Exercise native dataset augmentation and the real config factory/one-pair packer."""
    import cv2
    import torch

    from cosmos_framework.data.generator.local_simulation.config import make_local_experiment_config
    from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset
    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
    from cosmos_framework.utils.lazy_config import instantiate

    # The assertion CLI processes one clip at a time on CPU; avoid host-sized thread pools.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    dataset = LocalSimulationDataset(root, model="transfer")
    report: dict[str, Any] = {"status": "passed", "samples": {}, "packed_batches": {}, "model_training": "not run"}
    branches = {"structured": 0.0, "dense": 1.0}
    selected_datasets: dict[tuple[str, str, str], LocalSimulationDataset] = {}
    for record in dataset.records:
        entry = record["entry"]
        caption = json.loads(entry["caption_json"])
        canvas_tier = resolution or dataset.manifest["transforms"][record["stratum"]]["resolution"]
        max_frames = 149 if canvas_tier == "720" else 201
        frames = min(1 + 4 * (int(record["stratum"]) - 1), max_frames)
        canvases = {(height, width) for width, height in VIDEO_RES_SIZE_INFO[canvas_tier].values()}
        reference: torch.Tensor | None = None  # [3,T,H,W] when set
        sample_report: dict[str, Any] = {}
        for modality in MODALITIES:
            if modality in {"depth", "seg"} and modality not in record["assets"]:
                continue
            modality_report: dict[str, Any] = {}
            for branch, probability in branches.items():
                # Force coverage only in memory. The saved runtime policy retains four equal weights.
                # Reuse each variant's manifest and tokenizer/augmentation caches across records.
                variant = (canvas_tier, modality, branch)
                if variant not in selected_datasets:
                    selected_datasets[variant] = LocalSimulationDataset(
                        root,
                        model="transfer",
                        max_num_frames=max_frames,
                        transform_overrides={
                            "modality_weights": {modality: 1.0},
                            "dense_caption_probability": probability,
                            "cfg_dropout_rate": 0.0,
                            "resolution": canvas_tier,
                        },
                    )
                sample = selected_datasets[variant].materialize(record, draw_seed=42)
                control, target = sample["video"]  # two [3,T,H,W] tensors
                for name, tensor in ((modality, control), ("rgb", target)):  # tensor: [3,T,H,W]
                    if (
                        tensor.dtype != torch.float32
                        or tuple(tensor.shape[:2]) != (3, frames)
                        or tuple(tensor.shape[2:]) not in canvases
                    ):
                        raise ValueError(f"Unexpected {name} dtype/shape: {tensor.dtype}, {tuple(tensor.shape)}")
                    if (
                        not torch.isfinite(tensor).all().item()
                        or tensor.min().item() < -1.001
                        or tensor.max().item() > 1.001
                    ):
                        raise ValueError(f"Invalid normalized {name} pixels")
                if control.shape != target.shape:
                    raise ValueError("Control and target shapes differ")
                if reference is not None and not torch.equal(reference, target):
                    raise ValueError("Changing modality or caption branch changed the aligned RGB target")
                reference = target  # [3,T,H,W]
                if not sample["sequence_plan"].share_vision_temporal_positions:
                    raise ValueError("Control and target must share temporal positions")
                if sample["sampled_chunk_key"] != entry["caption_key"]:
                    raise ValueError("Materialized caption came from another window")
                text = sample["ai_caption"]
                expected = caption["temporal_caption"] if branch == "dense" else json.dumps(caption)
                if not isinstance(text, str) or not text.startswith(expected) or sample["text_token_ids"].numel() == 0:
                    raise ValueError(f"Framework caption/tokenization differs from the {branch} annotation")
                if (
                    text.count("resolution.") - expected.count("resolution.") != 1
                    or text.count(" FPS.") - expected.count(" FPS.") != 1
                ):
                    raise ValueError("Decoded output metadata was not appended exactly once")
                modality_report[branch] = {
                    "shape": list(target.shape),
                    "dtype": str(target.dtype),
                    "caption": text,
                    "text_tokens": sample["text_token_ids"].numel(),
                }
                if previews and branch == "structured":
                    preview_root = root / "previews" / record["identity"]
                    fps = entry["source_fps"] * (frames - 1) / (entry["caption_end"] - entry["caption_start"] - 1)
                    export_preview(preview_root / f"{modality}.mp4", control, fps)
                    if modality == "edge":
                        export_preview(preview_root / "rgb.mp4", target, fps)
            sample_report[modality] = modality_report
        report["samples"][record["identity"]] = sample_report
    # Exercise the real one-sample joint packer without importing any model weights.
    source = {
        "job": {},
        "dataloader_train": {
            "tokenizer_spatial_compression_factor": 16,
            "tokenizer_temporal_compression_factor": 4,
            "patch_spatial": 2,
        },
    }
    for modality in ("edge",):
        for branch, probability in branches.items():
            config = make_local_experiment_config(
                source,
                "transfer",
                transform_overrides={
                    "modality_weights": {modality: 1.0},
                    "dense_caption_probability": probability,
                    "cfg_dropout_rate": 0.0,
                    **({"resolution": resolution} if resolution is not None else {}),
                },
            )["dataloader_train"]
            config.dataloaders.local_data.dataloader.dataset.bundle_path = str(root)
            config.dataloaders.local_data.dataloader.dataset.max_num_frames = 149 if resolution == "720" else 201
            loader = instantiate(config)
            packed = next(iter(loader))
            if packed["_num_samples"] != 1 or packed["_num_tokens"] <= 0 or len(packed["video"][0]) != 2:
                raise ValueError("Local joint packer violated the one-pair batch contract")
            report["packed_batches"][f"{modality}/{branch}"] = {
                "samples": packed["_num_samples"],
                "tokens": packed["_num_tokens"],
            }
    report["resolution_override"] = resolution
    report["max_num_frames"] = 149 if resolution == "720" else 201
    report["config_scope"] = "Native make_local_experiment_config data factory; no model or stage recipe composed"
    return report


def validate(root: Path, report_path: Path, *, previews: bool = False, resolution: str | None = None) -> None:
    """Check staged bytes, then materialize every control with the real offline Framework path."""
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["provenance"]["sha256"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Staged asset changed: {name}")
    report = validate_runtime(root, previews=previews, resolution=resolution)
    write_json(report_path, report)
    print(f"Validated available controls, both caption branches and the Framework joint packer: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    custom = commands.add_parser("prepare-custom", help="Build a structured-caption bundle from whole-clip JSONL rows.")
    custom.add_argument("--manifest", type=Path, required=True)
    custom.add_argument("--tokenizer", type=Path, required=True)
    custom.add_argument("--output", type=Path, required=True)
    custom.add_argument("--dataset-name", default="custom_transfer")
    check = commands.add_parser("validate")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--export-previews", action="store_true")
    check.add_argument(
        "--custom", action="store_true", help="Compatibility flag; all bundles use the same native validation."
    )
    check.add_argument(
        "--resolution", choices=("480", "720"), help="Override the model canvas in memory for validation."
    )
    args = parser.parse_args()
    if args.command == "prepare-custom":
        prepare_custom(args.manifest.resolve(), args.tokenizer.resolve(), args.output.resolve(), args.dataset_name)
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        try:
            validate(
                args.bundle.resolve(),
                args.report.resolve(),
                previews=args.export_previews,
                resolution=args.resolution,
            )
        except Exception as error:
            write_json(
                args.report,
                {
                    "status": "failed",
                    "bundle": str(args.bundle.resolve()),
                    "error": str(error),
                    "model_training": "not run",
                },
            )
            raise


if __name__ == "__main__":
    main()
