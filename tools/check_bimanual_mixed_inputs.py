# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validate the published Bimanual manifests and assets in a downloaded HF snapshot."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosmos_framework.inference.bimanual import load_bimanual_action_contract

HF_REPOSITORY = "nvidia/Cosmos3-Nano-Sim-Bimanual"
HF_RESOLVE_PREFIX = f"https://huggingface.co/{HF_REPOSITORY}/resolve/main/"
DEFAULT_MANIFESTS = (
    "examples/inference/sim_bimanual_unifiedv1_hf.jsonl",
    "examples/inference/sim_bimanual_mixed_action_hf.jsonl",
    "examples/inference/sim_bimanual_agibot_long60_hf.jsonl",
)


def _require_materialized(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"Missing downloaded {label}: {path}")
    if path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"{label} is a Git LFS pointer rather than downloaded content: {path}")
    return path


def _resolve_asset(value: str, manifest: Path, checkpoint: Path) -> Path:
    parsed = urlsplit(value)
    if parsed.scheme:
        if not value.startswith(HF_RESOLVE_PREFIX):
            raise ValueError(f"Published input must use {HF_RESOLVE_PREFIX}: {value}")
        resolved = (checkpoint / unquote(value.removeprefix(HF_RESOLVE_PREFIX))).resolve()
        if not resolved.is_relative_to(checkpoint):
            raise ValueError(f"Hosted input resolves outside the downloaded checkpoint: {value}")
        return resolved
    return (manifest.parent / value).resolve()


def _checkpoint_metadata(checkpoint: Path) -> dict:
    path = checkpoint / "checkpoint.json"
    payload = json.loads(path.read_text())
    try:
        return payload["cosmos3_nano_sim_bimanual"]
    except KeyError as error:
        raise ValueError(f"Missing cosmos3_nano_sim_bimanual metadata in {path}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Manifest to validate; repeat for multiple manifests. Defaults to all published examples.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint_path.resolve()
    metadata = _checkpoint_metadata(checkpoint)
    contract = load_bimanual_action_contract(checkpoint)
    if contract is None:
        raise ValueError("Expected a Bimanual export with schema-5 action metadata")

    schedule = metadata["fixed_step_sampler_config"]["t_list"]
    sampling_steps = len(schedule) - int(bool(schedule) and schedule[-1] == 0.0)
    if sampling_steps < 1:
        raise ValueError("The checkpoint's fixed sampling schedule contains no nonzero sigma")
    temporal_compression = metadata["temporal_compression_factor"]
    manifests = (
        [path.resolve() for path in args.input] if args.input else [checkpoint / path for path in DEFAULT_MANIFESTS]
    )
    results = []

    for manifest in manifests:
        _require_materialized(manifest, "manifest")
        names: set[str] = set()
        rows = 0
        for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            name = sample["name"]
            if name in names:
                raise ValueError(f"Duplicate sample name in {manifest}:{line_number}: {name}")
            names.add(name)

            action_count = sample["num_frames"] - 1
            if action_count <= 0 or action_count % temporal_compression:
                raise ValueError(f"Invalid frame/action alignment in {manifest}:{line_number}")
            if sample.get("action_chunk_size", action_count) != action_count:
                raise ValueError(f"action_chunk_size disagrees with num_frames in {manifest}:{line_number}")
            if sample.get("num_steps", sampling_steps) != sampling_steps:
                raise ValueError(f"num_steps disagrees with the checkpoint schedule in {manifest}:{line_number}")
            if sample.get("raw_action_dim") != contract.action_dim or sample.get("format_prompt_as_json") is not False:
                raise ValueError(f"Expected raw_action_dim=59 and a plain-text prompt in {manifest}:{line_number}")
            if not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip():
                raise ValueError(f"Expected a non-empty prompt in {manifest}:{line_number}")

            _require_materialized(_resolve_asset(sample["vision_path"], manifest, checkpoint), "conditioning image")
            action_path = _require_materialized(
                _resolve_asset(sample["action_path"], manifest, checkpoint), "action sidecar"
            )
            payload = json.loads(action_path.read_text())
            actions, domains = contract.validate(
                payload,
                action_count=action_count,
                domain_name=sample["domain_name"],
            )
            try:
                manifest_name = str(manifest.relative_to(checkpoint))
            except ValueError:
                manifest_name = str(manifest)
            results.append(
                {
                    "manifest": manifest_name,
                    "name": name,
                    "actions": [len(actions), contract.action_dim],
                    "domain_counts": dict(Counter(domains)),
                }
            )
            rows += 1
        if not rows:
            raise ValueError(f"Manifest contains no samples: {manifest}")

    print(
        json.dumps(
            {
                "checkpoint_id": metadata["checkpoint_id"],
                "sampling_steps": sampling_steps,
                "samples": results,
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
