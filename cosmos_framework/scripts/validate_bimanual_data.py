# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check a local Bimanual training bundle before loading model weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Bundle file paths must be nonempty relative strings")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"File path escapes the bundle: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundle file: {relative}")
    return path


def _resources(root: Path, value: Any) -> None:
    if isinstance(value, str) and value.startswith("bundle://"):
        _file(root, value.removeprefix("bundle://"))
    elif isinstance(value, dict):
        for item in value.values():
            _resources(root, item)
    elif isinstance(value, list):
        for item in value:
            _resources(root, item)


def validate_bundle(bundle: Path) -> dict[str, Any]:
    """Validate local descriptors and files without importing the training runtime."""
    root = bundle.expanduser().resolve()
    manifest = json.loads(_file(root, "manifest.json").read_text())
    records = json.loads(_file(root, "samples.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("manifest.json must declare local-bundle format_version 1")
    if manifest.get("model") != "bimanual" or manifest.get("complete") is not True:
        raise ValueError("Expected a complete Bimanual training bundle")
    if not isinstance(records, list) or not records:
        raise ValueError("samples.json must contain a nonempty array")
    if type(manifest.get("sample_count")) is not int or manifest["sample_count"] != len(records):
        raise ValueError("manifest.sample_count must match samples.json")
    transforms = manifest.get("transforms")
    if not isinstance(transforms, dict) or not transforms:
        raise ValueError("manifest.transforms must map strata to captured transforms")
    for stratum, transform in transforms.items():
        if not isinstance(stratum, str) or not stratum or not isinstance(transform, dict) or not transform:
            raise ValueError("Each stratum needs a name and a nonempty transform object")
        _resources(root, transform)
    for relative in (
        "resources/global_asinh.json",
        "resources/tokenizer/vocab.json",
        "resources/tokenizer/merges.txt",
        "resources/tokenizer/tokenizer_config.json",
    ):
        _file(root, relative)

    identities: set[str] = set()
    assets: set[Path] = set()
    strata: Counter[str] = Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"samples[{index}] must be an object")
        identity = record.get("identity")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError(f"samples[{index}] must have a unique nonempty identity")
        identities.add(identity)
        stratum = record.get("stratum")
        if not isinstance(stratum, str) or stratum not in transforms:
            raise ValueError(f"samples[{index}].stratum has no matching transform")
        strata[stratum] += 1
        if not isinstance(record.get("request"), dict) or not record["request"]:
            raise ValueError(f"samples[{index}].request must be a nonempty object")
        rows = record.get("source_rows")
        if not isinstance(rows, dict) or not rows or any(not isinstance(row, dict) for row in rows.values()):
            raise ValueError(f"samples[{index}].source_rows must map row identities to source objects")
        members = record.get("assets")
        if not isinstance(members, dict) or not members:
            raise ValueError(f"samples[{index}].assets must be a nonempty mapping")
        for identity, relative in members.items():
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"samples[{index}] has an empty asset identity")
            assets.add(_file(root, relative))
    return {"samples": len(records), "assets": len(assets), "strata": dict(strata), "materialized": False}


def materialize_batch(bundle: Path, stage: int) -> dict[str, Any]:
    """Read one packed sample using the native stage factory; never construct a model."""
    import torch

    from cosmos_framework.data.generator.local_simulation.bimanual_config import make_bimanual_local_dataloader

    loader = make_bimanual_local_dataloader(
        bundle_path=str(bundle.expanduser().resolve()),
        stage=stage,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=400_000 if stage == 0 else None,
        max_samples_per_batch=1,
        num_workers=0,
        seed=42,
    )
    batch = next(iter(loader))
    if not isinstance(batch, dict) or batch.get("_num_samples", 0) != 1 or batch.get("_num_tokens", 0) <= 0:
        raise ValueError("Native loader must return one nonempty packed sample")
    if not {"video", "action"}.issubset(batch):
        raise ValueError("Native Bimanual batch is missing video or action")
    tensors: dict[str, Any] = {}

    def inspect(value: Any, name: str) -> None:
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
                raise ValueError(f"Non-finite tensor in batch: {name}")
            tensors[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(item, f"{name}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                inspect(item, f"{name}[{index}]")

    inspect(batch, "batch")
    if not tensors:
        raise ValueError("Native batch contains no tensors")
    return {
        "stage": stage,
        "samples": int(batch["_num_samples"]),
        "tokens": int(batch["_num_tokens"]),
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--materialize", action="store_true", help="Decode and pack one sample with the training loader"
    )
    parser.add_argument(
        "--stage", type=int, choices=range(4), help="Training stage to check (required with --materialize)"
    )
    args = parser.parse_args()
    if args.materialize != (args.stage is not None):
        parser.error("Use --materialize and --stage together")
    try:
        report = validate_bundle(args.bundle)
        if args.materialize:
            report["batch"] = materialize_batch(args.bundle, args.stage)
            report["materialized"] = True
        print(json.dumps({"passed": True, **report}, indent=2))
    except (ValueError, OSError, TypeError, KeyError, RuntimeError, ImportError, StopIteration) as error:
        parser.exit(1, f"Bimanual data check failed: {error}\n")


if __name__ == "__main__":
    main()
