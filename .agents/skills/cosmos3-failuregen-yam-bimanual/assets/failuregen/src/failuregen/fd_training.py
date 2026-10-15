# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Notebook data setup and checkpoint handoff for the original EEF trainer.

Dataset identity checks follow cosmos-failure-gen/runners/slurm_yam_eval_eef.sbatch.
Training itself is implemented by the unmodified yam_training.launch module.
"""

from __future__ import annotations

import json
from pathlib import Path

from .yam_training.contract import file_sha256


def prepare_dataset_view(source: Path, bundle: Path, work: Path) -> Path:
    """Validate the frozen export against recordings before creating a dataset view."""
    source, bundle, work = source.resolve(), bundle.resolve(), work.resolve()
    identity = json.loads((bundle / "source_identity.json").read_text())
    expected = {
        source / "meta/info.json": identity["info_sha256"],
        source / "cosmos_annotations.json": identity["annotations_sha256"],
        bundle / "failuregen_annotations.json": identity["annotations_sha256"],
    }
    expected.update({source / name: digest for name, digest in identity["parquet_sha256"].items()})
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"Source differs from the frozen EEF export: {path}")
    for name, size in identity["video_sizes"].items():
        if (source / name).stat().st_size != size:
            raise ValueError(f"Video size differs from the frozen EEF export: {name}")
    for path in (bundle / "eef/manifest.json", bundle / "splits.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    view = work / "dataset"
    targets = {name: source / name for name in ("meta", "data", "videos")}
    targets["failuregen_annotations.json"] = bundle / "failuregen_annotations.json"
    for name, target in targets.items():
        link = view / name
        if not target.exists():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"Existing dataset link differs: {link}; choose a new FD_WORK")
        elif link.exists():
            raise ValueError(f"Refusing to replace {link}; choose a new FD_WORK")
    view.mkdir(parents=True, exist_ok=True)
    for name, target in targets.items():
        if not (view / name).is_symlink():
            (view / name).symlink_to(target, target_is_directory=target.is_dir())
    return view


def completed_checkpoint(run_dir: Path, prepared: Path, dataset: Path) -> Path:
    """Only hand off a completed requested budget, with its matching data contract."""
    from .yam_training.failure_test import load_context

    load_context(prepared, dataset, run_dir)
    contract = json.loads((run_dir / "yam_training_contract.json").read_text())
    latest = (run_dir / "checkpoints/latest_checkpoint.txt").read_text().strip()
    expected = f"iter_{contract['max_iter']:09d}"
    if latest != expected:
        raise ValueError(f"Training is incomplete: latest is {latest!r}, expected {expected}. Resume training first.")
    checkpoint = run_dir / "checkpoints" / latest
    if not (checkpoint / "model/.metadata").is_file():
        raise FileNotFoundError(checkpoint / "model/.metadata")
    return checkpoint
