# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local Bimanual stream selection for user-owned robot data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_FRAMEWORK_DATASET_PREFIX = "cosmos_framework.data.generator.action."
_LEGACY_DATASET_PREFIX = "projects.cosmos3.cosmos3.datasets.action."
_SUPPORTED_DATASET_TARGETS = {
    f"{prefix}canonical_lance_action_datasets.{name}"
    for prefix in (_FRAMEWORK_DATASET_PREFIX, _LEGACY_DATASET_PREFIX)
    for name in (
        "CanonicalLanceAgiBotDataset",
        "CanonicalLanceBehavior1KDataset",
        "CanonicalLanceYAMDataset",
        "CanonicalLanceembodiment_aDataset",
    )
}


def _validate_dataset_target(target: Any) -> None:
    if not isinstance(target, str) or not target:
        raise ValueError("Bimanual robot datasets require a constructor ID.")
    if target not in _SUPPORTED_DATASET_TARGETS:
        raise ValueError(f"Unsupported Bimanual robot dataset constructor: {target!r}")


def _configured_sources(manifest: dict[str, Any]) -> list[tuple[str, float]]:
    sources: list[tuple[str, float]] = []
    for stratum, captured in manifest["transforms"].items():
        entry = captured["entry"]
        _validate_dataset_target(entry["dataset"].get("_target_"))
        weight = float(entry["ratio"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid captured source weight for {stratum!r}.")
        sources.append((stratum, weight))
    return sources


def make_bimanual_local_dataloader(
    bundle_path: str,
    stage: int,
    tokenizer_spatial_compression_factor: int,
    tokenizer_temporal_compression_factor: int,
    patch_spatial: int,
    max_sequence_length: int | None,
    max_samples_per_batch: int | None,
    seed: int = 42,
    num_workers: int = 0,
) -> Any:
    """Construct native packers from the bundle's robot sources.

    A bundle containing one robot source is sufficient. When a bundle contains
    more than one supported robot source, their captured relative weights are
    retained. Stage-specific rendering choices remain a Framework concern and
    do not require users to author separate source tiers.
    """
    from cosmos_framework.data.generator.local_simulation.dataloader import (
        LocalSimulationDataLoader,
        LocalSimulationJointDataLoader,
    )
    from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset

    if stage not in range(4):
        raise ValueError("Bimanual stage must be 0, 1, 2, or 3.")
    manifest = json.loads((Path(bundle_path) / "manifest.json").read_text())
    if manifest.get("model") != "bimanual" or manifest.get("complete") is not True:
        raise ValueError("Expected a complete Bimanual local bundle.")
    entries = _configured_sources(manifest)
    if not entries:
        raise ValueError("Bimanual local training requires at least one supported robot data source.")
    group_total = math.fsum(weight for _, weight in entries)
    resolutions = {"256": 1.0, "480": 2.0} if stage in (1, 2) else {"480": 1.0}
    streams: dict[str, dict[str, Any]] = {}
    for stratum, weight in entries:
        source_ratio = weight / group_total
        for target_resolution, resolution_weight in resolutions.items():
            stream_name = f"{stratum}_{target_resolution}p" if stage in (1, 2) else stratum
            dataset = LocalSimulationDataset(
                bundle_path,
                model="bimanual",
                stratum=stratum,
                seed=seed,
                transform_overrides={"resolution": target_resolution},
            )
            streams[stream_name] = {
                "ratio": source_ratio * resolution_weight / math.fsum(resolutions.values()),
                "dataloader": LocalSimulationDataLoader(
                    dataset=dataset,
                    seed=seed,
                    num_workers=num_workers,
                ),
            }
    return LocalSimulationJointDataLoader(
        dataloaders=streams,
        tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
        tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
        patch_spatial=patch_spatial,
        max_sequence_length=max_sequence_length,
        max_samples_per_batch=max_samples_per_batch,
        lookahead_limits={},
        prewarm=False,
        lazy_initialize_child_iterators=True,
        seed=seed,
    )
