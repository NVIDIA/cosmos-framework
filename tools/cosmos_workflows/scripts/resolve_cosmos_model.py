#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a model ID to its runtime contract and existing framework skill."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_WORKFLOW_ROOT = Path(os.environ.get("COSMOS_WORKFLOWS_ROOT", Path(__file__).resolve().parents[1]))
UNMATCHED_EXIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-root",
        type=Path,
        default=DEFAULT_WORKFLOW_ROOT,
        help="Path to the packaged Cosmos workflow bundle.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model skill name, network_arch, alias, or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--action",
        default="",
        help="Optional model action to validate and route to its owning skill.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def identifier_groups(candidate: Path, info: dict[str, Any]) -> dict[str, set[str]]:
    canonical = {
        candidate.name,
        str(info.get("network_arch", "")).strip(),
        str(info.get("model", "")).strip(),
        str(info.get("name", "")).strip(),
    }
    return {
        "huggingface_model_ids": {str(value).strip() for value in info.get("huggingface_model_ids", [])} - {""},
        "aliases": {str(value).strip() for value in info.get("aliases", [])} - {""},
        "canonical": canonical - {""},
    }


def action_contract(info: dict[str, Any], action: str) -> dict[str, Any]:
    """Look up a declared, supported model action."""
    contract = info.get("actions", {}).get(action)
    if not isinstance(contract, dict) or not contract.get("supported", True):
        reason = (
            contract.get("reason", "action is not declared") if isinstance(contract, dict) else "action is not declared"
        )
        raise ValueError(f"Unsupported model action {action!r}: {reason}")
    return contract


def resolve_model(
    workflow_root: Path,
    requested_model: str,
    *,
    action: str = "",
) -> dict[str, Any] | None:
    models_root = workflow_root.expanduser().resolve() / "models"
    if not models_root.is_dir():
        raise FileNotFoundError(f"Model skills directory not found: {models_root}")

    requested = requested_model.strip()
    if not requested:
        raise ValueError("--model must not be empty")
    requested_casefold = requested.casefold()
    matches: list[dict[str, Any]] = []

    for candidate in sorted(models_root.iterdir()):
        metadata_path = candidate / "references" / "skill_info.yaml"
        if not metadata_path.exists():
            continue
        info = load_yaml(metadata_path)
        groups = identifier_groups(candidate, info)
        matched_by = [
            group for group, values in groups.items() if requested_casefold in {value.casefold() for value in values}
        ]
        if matched_by:
            result = {
                "schema_version": 1,
                "requested_model": requested_model,
                "matched": True,
                "model": candidate.name,
                "network_arch": info.get("network_arch", candidate.name),
                "matched_by": matched_by[0],
                "metadata_path": str(metadata_path),
                "skill_path": str(
                    workflow_root.expanduser().resolve().parents[1]
                    / ".agents/skills"
                    / (
                        "cosmos3-inference"
                        if action in {"inference", "inference_microservice"}
                        else "cosmos3-post-training"
                    )
                    / "SKILL.md"
                    if candidate.name == "cosmos3-reasoner"
                    else candidate / "SKILL.md"
                ),
                "guide_path": str(candidate / ("guide.md" if candidate.name == "cosmos3-reasoner" else "SKILL.md")),
                "container_image": info.get("container_image"),
            }
            if action:
                contract = action_contract(info, action)
                result["container_image"] = contract.get("container_image", info.get("container_image"))
            if info.get("training_contract"):
                path = candidate / info["training_contract"]
                if not path.is_file():
                    raise FileNotFoundError(f"Training contract not found: {path}")
                result["training_contract_path"] = str(path)
            matches.append(result)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        owners = ", ".join(match["model"] for match in matches)
        raise ValueError(f"Model identifier '{requested}' is ambiguous; matching skills: {owners}")
    return None


def format_text(data: dict[str, Any]) -> str:
    image = data.get("container_image") or "declared by the model skill (no container image)"
    lines = [
        "Cosmos model ownership resolution:",
        f"- requested model: {data['requested_model']}",
        f"- model contract: {data['model']}",
        f"- network_arch: {data['network_arch']}",
        f"- matched by: {data['matched_by']}",
        f"- execution image: {image}",
        f"- skill: {data['skill_path']}",
        f"- metadata: {data['metadata_path']}",
    ]
    if data.get("training_contract_path"):
        lines.append(f"- training contract: {data['training_contract_path']}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        data = resolve_model(
            args.workflow_root,
            args.model,
            action=args.action,
        )
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if data is None:
        print(
            f"UNMATCHED: no packaged Cosmos model skill owns '{args.model}'",
            file=sys.stderr,
        )
        return UNMATCHED_EXIT
    output = json.dumps(data, indent=2, sort_keys=True) if args.format == "json" else format_text(data)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
