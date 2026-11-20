# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic short-clip reconstruction check for a native tokenizer export."""

import json
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.data.tokenizer.lidar import LidarTarDataset, read_manifest
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import LidarTokenizerV1Interface
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import crop_range_video_width
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig
from cosmos_framework.trainer.tokenizer.lidar import sha256


@torch.inference_mode()
def evaluate(artifact: Path, manifest: Path, output: Path, device: str = "cuda") -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((artifact / "config.json").read_text())
    checkpoint = artifact / "tokenizer.pt"
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Export checkpoint checksum mismatch")
    projection = LidarRangeProjectionConfig.from_dict(config["range_projection"])
    tokenizer = LidarTokenizerV1Interface(
        vae_path=str(checkpoint),
        device=device,
        dtype=torch.float32,
        sample_posterior=False,
        network_config=config["network_config"],
        range_projection=projection,
        spatial_compression=config["spatial_compression"],
        temporal_compression_factor=1,
    )
    samples = read_manifest(manifest)
    dataset = LidarTarDataset(samples, projection=projection)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(samples):
        batch = dataset[index]
        video = batch["video"].unsqueeze(0).to(device)  # [1,3,9,128,1808]
        reconstructed = tokenizer.decode(tokenizer.encode(video), apply_mask=False)  # [1,3,9,128,1808]
        reconstructed = crop_range_video_width(
            reconstructed, semantic_width=projection.semantic_width
        )  # [1,3,9,128,1800]
        # Exclude the duplicate first frame from scores; short input padding remains explicit.
        target = batch["target"][0, 1:].to(device)  # [8,128,1800]
        valid = batch["valid_mask"][0, 1:].to(device).bool()  # [8,128,1800]
        prediction = reconstructed[0, 0, 1:]  # [8,128,1800]
        predicted_valid = reconstructed[0, 2, 1:] >= projection.validity_threshold  # [8,128,1800]
        if not bool(torch.isfinite(reconstructed).all()):
            raise FloatingPointError(f"Non-finite reconstruction: {path}")
        errors = (prediction - target).abs()[valid]  # [N_valid]
        rows.append(
            {
                "sample_sha256": sha256(path),
                "range_abs_error_sum_m": errors.sum().item(),
                "valid_rays": errors.numel(),
                "intersection": (valid & predicted_valid).sum().item(),
                "union": (valid | predicted_valid).sum().item(),
            }
        )
    count = sum(row["valid_rays"] for row in rows)
    union = sum(row["union"] for row in rows)
    if not count or not union:
        raise ValueError("Held-out clips contain no valid rays")
    report = {
        "protocol": "short-clip-first-eight-v1",
        "dtype": "float32",
        "posterior": "mean",
        "duplicate_first_frame_scored": False,
        "validity_threshold": projection.validity_threshold,
        "weights": config["weights"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "range_mae_m": sum(row["range_abs_error_sum_m"] for row in rows) / count,
        "validity_iou": sum(row["intersection"] for row in rows) / union,
        "samples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
