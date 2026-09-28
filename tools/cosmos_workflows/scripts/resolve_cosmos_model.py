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

DEFAULT_SKILL_BANK = Path(os.environ.get("COSMOS_WORKFLOWS_ROOT", Path(__file__).resolve().parents[1]))
UNMATCHED_EXIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-root",
        "--skill-bank",
        dest="skill_bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
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
        help="Optional action used to resolve an implementation backend.",
    )
    parser.add_argument(
        "--backend",
        default="cosmos-framework",
        choices=("auto", "cosmos-framework"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workload",
        default="",
        help="Optional workflow hint: training.",
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


def backend_contracts(candidate: Path, info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load backend contracts and their skill-owned runtime metadata.

    A legacy declaration may be a relative YAML path.  A shared frontend that
    owns backend-specific runtime knowledge uses a mapping with ``path`` and
    ``container_image``.  The referenced contract continues to own schemas,
    topology, checkpoint semantics, and actions; image pins remain centralized
    in ``references/skill_info.yaml``.
    """
    declared = info.get("backend_contracts", {})
    if not isinstance(declared, dict):
        return {}
    contracts: dict[str, dict[str, Any]] = {}
    for name, declaration in declared.items():
        metadata: dict[str, Any] = {}
        if isinstance(declaration, str):
            relative = declaration
        elif isinstance(declaration, dict):
            relative = declaration.get("path")
            metadata = declaration
        else:
            raise ValueError(f"backend_contracts.{name} must be a relative YAML path or mapping")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"backend_contracts.{name}.path must be a relative YAML path")
        path = candidate / relative
        if not path.is_file():
            raise FileNotFoundError(f"Backend contract not found: {path}")
        contract = load_yaml(path)
        if isinstance(declaration, dict):
            image = metadata.get("container_image")
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"backend_contracts.{name}.container_image must be a non-empty image reference")
            if "container_image" in contract:
                raise ValueError(
                    f"{path} must not duplicate container_image; keep backend image pins in skill_info.yaml"
                )
            contract["container_image"] = image.strip()
            contract["container_image_source"] = f"skill_info.backend_contracts.{name}.container_image"
        contract["contract_path"] = str(path)
        contracts[str(name)] = contract
    return contracts


def select_implementation_backend(
    *,
    info: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    requested_model: str,
    action: str,
    backend: str,
    workload: str,
) -> tuple[str, str] | None:
    """Require the repository-owned Framework implementation."""
    if not contracts:
        return None
    aliases = {"auto": "cosmos-framework", "framework": "cosmos-framework", "cosmos_framework": "cosmos-framework"}
    selected = aliases.get(backend.casefold(), backend.casefold())
    action = action.strip().casefold() or "train"
    if workload.strip().casefold() not in {"", "training"}:
        raise ValueError("only the Framework training workflow is supported")
    reason = "repository-owned Cosmos Framework runtime"
    if selected not in contracts:
        choices = ", ".join(sorted(contracts))
        raise ValueError(f"Backend {selected!r} is not declared; available backends: {choices}")
    action_contract = contracts[selected].get("actions", {}).get(action, {})
    if not isinstance(action_contract, dict) or not action_contract.get("supported", False):
        reason_text = (
            action_contract.get("reason", "action is unsupported")
            if isinstance(action_contract, dict)
            else "action is unsupported"
        )
        raise ValueError(f"Backend {selected!r} does not support native {action}: {reason_text}")
    return selected, reason


def resolve_model(
    skill_bank: Path,
    requested_model: str,
    *,
    action: str = "",
    backend: str = "cosmos-framework",
    workload: str = "",
) -> dict[str, Any] | None:
    models_root = skill_bank.expanduser().resolve() / "models"
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
                    skill_bank.expanduser().resolve().parents[1]
                    / ".agents/skills"
                    / (
                        "cosmos3-inference"
                        if action in {"inference", "inference_microservice", "quantize"}
                        else "cosmos3-post-training"
                    )
                    / "SKILL.md"
                    if candidate.name == "cosmos3-reasoner"
                    else candidate / "SKILL.md"
                ),
                "guide_path": str(candidate / ("guide.md" if candidate.name == "cosmos3-reasoner" else "SKILL.md")),
                "container_image": info.get("container_image"),
            }
            contracts = backend_contracts(candidate, info)
            if contracts:
                result["available_backends"] = sorted(contracts)
                result["backend_selection_policy"] = info.get("backend_selection", {})
                if action or backend != "auto" or workload:
                    selection = select_implementation_backend(
                        info=info,
                        contracts=contracts,
                        requested_model=requested_model,
                        action=action,
                        backend=backend,
                        workload=workload,
                    )
                    if selection:
                        selected, reason = selection
                        contract = contracts[selected]
                        result.update(
                            {
                                "selected_backend": selected,
                                "backend_selection_reason": reason,
                                "backend_contract_path": contract["contract_path"],
                                "container_image": contract.get("container_image"),
                            }
                        )
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
    if data.get("available_backends"):
        lines.append(f"- available backends: {', '.join(data['available_backends'])}")
    if data.get("selected_backend"):
        lines.append(f"- selected backend: {data['selected_backend']} ({data['backend_selection_reason']})")
        lines.append(f"- backend contract: {data['backend_contract_path']}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        data = resolve_model(
            args.skill_bank,
            args.model,
            action=args.action,
            backend=args.backend,
            workload=args.workload,
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
