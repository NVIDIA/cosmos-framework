# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run native tokenizer weights with an explicit config, then score metric NPZ pairs.

This optional command needs the framework's Torch/NATTEN environment and user
tokenizer weights. Offline metrics in ``cosmos_framework.evaluation.lidar.tokenizer`` need neither.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.evaluation.lidar.tokenizer import evaluate_tokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_framework_tokenizer(
    artifact: Path,
    manifest: Path,
    output_dir: Path,
    device: str = "cuda",
    max_points: int = 50_000,
) -> dict[str, Any]:
    """Encode/decode first 8 sweeps, exclude first duplicate, and save scores.

    ``artifact`` contains ``config.json`` and native weights with embedded
    latent statistics. Config fields are ``network_config``, ``range_projection``
    and ``checkpoint_sha256``; ``checkpoint_file`` defaults to ``tokenizer.pt``.
    Both native PyTorch and safetensors weights are supported. ``manifest`` is one local tar path per line,
    resolved relative to its parent. Short clips repeat their final sweep to
    eight sweeps, and these repeats remain scored, matching the native check.
    """
    # Optional model dependencies are loaded only for this inference command.
    import torch

    from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import (
        LidarTokenizerV1Interface,
    )
    from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
        crop_range_video_width,
        enforce_tokenizer_clip_length,
        list_rangemap_sweep_ids,
        load_prepared_lidar_video_3ch,
    )
    from cosmos_framework.model.generator.tokenizers.lidar.range_projection import (
        LidarRangeProjectionConfig,
    )

    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    config = json.loads((artifact / "config.json").read_text())
    checkpoint = artifact / config.get("checkpoint_file", "tokenizer.pt")
    if _sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Export checkpoint checksum mismatch")
    if config.get("precision", "float32") != "float32":
        raise ValueError("This reference inference protocol requires an FP32 export")
    projection = LidarRangeProjectionConfig.from_dict(config["range_projection"])
    if projection.model_width_transform != "circular_pad":
        raise ValueError("Tokenizer inference requires circular_pad projection; resize canvases are unsupported")
    if int(config.get("temporal_compression", 1)) != 1:
        raise ValueError("The first-eight-sweep protocol supports temporal compression 1")
    samples = [(manifest.parent / line.strip()).resolve() for line in manifest.read_text().splitlines() if line.strip()]
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("Manifest must contain unique, nonempty local tar paths")
    for sample in samples:
        if sample.suffix != ".tar" or not sample.is_file():
            raise ValueError(f"Expected an existing local .tar recording: {sample}")
    tokenizer = LidarTokenizerV1Interface(
        vae_path=str(checkpoint),
        device=device,
        dtype=torch.float32,
        sample_posterior=False,
        network_config=config["network_config"],
        range_projection=projection,
        spatial_compression=config.get("spatial_compression"),
        temporal_compression_factor=1,
    )
    tokenizer.model.eval()
    output_dir.mkdir(parents=True)
    projection_path = output_dir / "projection.json"
    projection_path.write_text(json.dumps(config["range_projection"], indent=2) + "\n")
    pairs: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            available_ids = list_rangemap_sweep_ids(sample)
            if not available_ids:
                raise ValueError(f"No source sweeps in sample {index}")
            # Loading only the first eight avoids materializing an entire long clip.
            selected_ids = available_ids[:8]
            video, frame_ids = load_prepared_lidar_video_3ch(
                sample,
                min_source_frame_id=selected_ids[0],
                max_source_frame_id=selected_ids[-1],
                max_frames_per_chunk=8,
                prepend_first_frame=True,
                range_projection=projection,
            )  # video: [3,T,128,W_model]; frame_ids: list[int]
            original_sweeps = len(available_ids)
            video, frame_ids = enforce_tokenizer_clip_length(
                video,
                frame_ids,
                num_source_frames=8,
            )  # video: [3,9,128,W_model]; frame_ids: list[int]
            semantic = crop_range_video_width(
                video,
                semantic_width=projection.semantic_width,
            )  # [3,9,128,W_semantic]
            input_video = video.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,3,9,128,W_model]
            latent = tokenizer.encode(input_video)  # [1,C_latent,9,H_latent,W_latent]
            reconstructed = tokenizer.decode(latent, apply_mask=False)  # [1,3,9,128,W_model]
            # The native API returns a tuple only when return_validity=True.
            assert isinstance(reconstructed, torch.Tensor)
            reconstructed = crop_range_video_width(
                reconstructed,
                semantic_width=projection.semantic_width,
            )  # [1,3,9,128,W_semantic]
            if not bool(torch.isfinite(reconstructed).all()):  # [scalar]
                raise FloatingPointError(f"Non-finite reconstruction for sample {index}")
            reference = semantic[0, 1:].cpu().numpy()  # [8,128,W_semantic]
            reference_valid = semantic[2, 1:].bool().cpu().numpy()  # [8,128,W_semantic]
            predicted = reconstructed[0, 0, 1:].cpu().numpy()  # [8,128,W_semantic]
            predicted_valid = (
                (reconstructed[0, 2, 1:] >= projection.validity_threshold).cpu().numpy()
            )  # [8,128,W_semantic]
            sweep_ids = np.asarray(frame_ids, dtype=np.int64)  # [8]
            reference_name, prediction_name = f"{index:05d}.reference.npz", f"{index:05d}.prediction.npz"
            np.savez_compressed(
                output_dir / reference_name,
                range_m=reference,
                valid=reference_valid,
                sweep_ids=sweep_ids,
            )
            np.savez_compressed(
                output_dir / prediction_name,
                range_m=predicted,
                valid=predicted_valid,
                sweep_ids=sweep_ids,
            )
            clip_id = f"clip_{index:05d}"
            pairs.append({"id": clip_id, "reference": reference_name, "prediction": prediction_name})
            source_rows.append(
                {
                    "id": clip_id,
                    "source_sha256": _sha256(sample),
                    "available_source_sweeps": original_sweeps,
                    "scored_source_sweeps": frame_ids,
                    "repeated_last_sweeps": max(0, 8 - original_sweeps),
                }
            )
    pairs_path = output_dir / "pairs.json"
    pairs_path.write_text(json.dumps(pairs, indent=2) + "\n")
    result = {
        "protocol": "short-clip-first-eight; duplicate first excluded; right-padding repeats remain scored",
        "dtype": "float32",
        "posterior": "mean",
        "device": device,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "weights": config.get("weights", "unspecified"),
        "step": config.get("step"),
        "validity_threshold": projection.validity_threshold,
        "sources": source_rows,
        "macro": evaluate_tokenizer(pairs_path, "macro", projection_path, max_points),
        "micro": evaluate_tokenizer(pairs_path, "micro", projection_path, max_points),
        "numerics": "Offline reductions use float64; native FP32 pooled reductions can differ by rounding.",
    }
    (output_dir / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
