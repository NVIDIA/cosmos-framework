#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the default Cosmos container image for a model action.

The helper reads packaged model metadata instead of relying on hand-written
prompts. Launch workflows should show this image to the user and accept an
explicit override before generating runner artifacts or submitting jobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from resolve_cosmos_model import backend_contracts, select_implementation_backend

DEFAULT_SKILL_BANK = Path(os.environ.get("COSMOS_WORKFLOWS_ROOT", Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
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
        "--network",
        dest="model",
        required=True,
        help=("Packaged model skill directory or network_arch, for example cosmos3-reasoner."),
    )
    parser.add_argument(
        "--action",
        default="train",
        help="Model action to resolve, for example train, evaluate, inference, or export.",
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
        help="Optional workload hint such as wts, aetc, automl, or hpo.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def load_model_metadata(
    skill_bank: Path,
    requested_model: str,
) -> tuple[str, Path, dict[str, Any]]:
    """Load model metadata by skill directory or network_arch alias."""
    models_root = skill_bank.expanduser() / "models"
    requested = requested_model.strip()
    exact_path = models_root / requested / "references" / "skill_info.yaml"
    if exact_path.exists():
        return requested, exact_path, load_yaml(exact_path)

    if not models_root.exists():
        raise FileNotFoundError(f"Model skills directory not found: {models_root}")

    matches: list[tuple[str, Path, dict[str, Any]]] = []
    requested_casefold = requested.casefold()
    for candidate in sorted(models_root.iterdir()):
        metadata_path = candidate / "references" / "skill_info.yaml"
        if not metadata_path.exists():
            continue
        skill_info = load_yaml(metadata_path)
        aliases = {
            candidate.name,
            str(skill_info.get("network_arch", "")).strip(),
            str(skill_info.get("model", "")).strip(),
            str(skill_info.get("name", "")).strip(),
        }
        aliases.update(str(alias).strip() for alias in skill_info.get("aliases", []))
        aliases.update(str(model_id).strip() for model_id in skill_info.get("huggingface_model_ids", []))
        aliases = {alias for alias in aliases if alias}
        if requested in aliases or requested_casefold in {alias.casefold() for alias in aliases}:
            matches.append((candidate.name, metadata_path, skill_info))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(name for name, _, _ in matches)
        raise ValueError(f"Model alias '{requested}' is ambiguous. Matching skill directories: {choices}")

    available = []
    for candidate in sorted(models_root.iterdir()):
        metadata_path = candidate / "references" / "skill_info.yaml"
        if not metadata_path.exists():
            continue
        try:
            skill_info = load_yaml(metadata_path)
        except Exception:
            continue
        network_arch = skill_info.get("network_arch", candidate.name)
        available.append(f"{candidate.name} ({network_arch})")
    suffix = "; available examples: " + ", ".join(available[:8]) if available else ""
    raise FileNotFoundError(
        f"Model metadata not found for '{requested}'. Pass a skill directory "
        f"or network_arch under {models_root}{suffix}"
    )


def resolve_image_key(skill_bank: Path, image: str) -> tuple[str, str]:
    """Resolve a versions.yaml image key to a URI when possible."""
    image = image.strip()
    if "/" in image or ":" in image:
        return image, "absolute"

    versions_path = skill_bank.expanduser() / "versions.yaml"
    versions = load_yaml(versions_path)
    cursor: Any = versions.get("images", {})
    for part in image.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return image, "unresolved_key"
        cursor = cursor[part]
    if not isinstance(cursor, str) or not cursor.strip():
        return image, "unresolved_key"
    return cursor.strip(), "versions.yaml"


def resolve_image(
    skill_bank: Path,
    model: str,
    action: str,
    *,
    backend: str = "cosmos-framework",
    workload: str = "",
) -> dict[str, Any]:
    """Resolve action-level image first, then model-level image."""
    resolved_model, metadata_path, skill_info = load_model_metadata(skill_bank, model)
    contracts = backend_contracts(metadata_path.parent.parent, skill_info)
    selected_backend = ""
    backend_reason = ""
    if contracts:
        selection = select_implementation_backend(
            info=skill_info,
            contracts=contracts,
            requested_model=model,
            action=action,
            backend=backend,
            workload=workload,
        )
        if selection:
            selected_backend, backend_reason = selection
            backend_contract = contracts[selected_backend]
            action_config = backend_contract.get("actions", {}).get(action, {})
            candidates = [
                (
                    "backend.action.container_image",
                    action_config.get("container_image") if isinstance(action_config, dict) else None,
                ),
                (
                    backend_contract.get("container_image_source", "backend.container_image"),
                    backend_contract.get("container_image"),
                ),
            ]
            for source, image in candidates:
                if isinstance(image, dict) and image.get("policy") == "repository_derived":
                    return {
                        "schema_version": 4,
                        "requested_model": model,
                        "model": resolved_model,
                        "network_arch": skill_info.get("network_arch", resolved_model),
                        "action": action,
                        "backend": selected_backend,
                        "backend_selection_reason": backend_reason,
                        "backend_contract_path": backend_contract["contract_path"],
                        "image": None,
                        "declared_image": None,
                        "resolved_from": "clean repository build required",
                        "source": source,
                        "metadata_path": str(metadata_path),
                        "build_required": True,
                        "runtime_input": image.get("runtime_input", "image_tag"),
                        "confirmation_required": True,
                        "override_key": "image_tag",
                    }
                if isinstance(image, str) and image.strip():
                    resolved_image, resolved_from = resolve_image_key(skill_bank, image)
                    return {
                        "schema_version": 3,
                        "requested_model": model,
                        "model": resolved_model,
                        "network_arch": skill_info.get("network_arch", resolved_model),
                        "action": action,
                        "backend": selected_backend,
                        "backend_selection_reason": backend_reason,
                        "backend_contract_path": backend_contract["contract_path"],
                        "image": resolved_image,
                        "declared_image": image.strip(),
                        "resolved_from": resolved_from,
                        "source": source,
                        "metadata_path": str(metadata_path),
                        "confirmation_required": True,
                        "override_key": "image",
                    }
    actions = skill_info.get("actions", {})
    if not isinstance(actions, dict):
        actions = {}

    action_config = actions.get(action)
    if action_config is None:
        available = ", ".join(sorted(actions)) if actions else "none"
        raise ValueError(
            f"Action '{action}' is not packaged for model '{resolved_model}'. Available actions: {available}"
        )
    if not isinstance(action_config, dict):
        raise ValueError(f"models/{resolved_model}/references/skill_info.yaml actions.{action} must be an object")

    candidates = [
        ("action.container_image", action_config.get("container_image")),
        ("action.image", action_config.get("image")),
        ("model.container_image", skill_info.get("container_image")),
        ("model.image", skill_info.get("image")),
    ]
    for source, image in candidates:
        if isinstance(image, str) and image.strip():
            resolved_image, resolved_from = resolve_image_key(skill_bank, image)
            return {
                "schema_version": 2,
                "requested_model": model,
                "model": resolved_model,
                "network_arch": skill_info.get("network_arch", resolved_model),
                "action": action,
                "image": resolved_image,
                "declared_image": image.strip(),
                "resolved_from": resolved_from,
                "source": source,
                "metadata_path": str(metadata_path),
                "confirmation_required": True,
                "override_key": "image",
            }

    raise ValueError(f"No container image found for model '{resolved_model}' action '{action}' in {metadata_path}")


def format_text(data: dict[str, Any]) -> str:
    """Format resolved image metadata for launch prompts."""
    lines = [
        "Cosmos container image resolution:",
        f"- requested model: {data.get('requested_model', data['model'])}",
        f"- model: {data['model']} ({data['network_arch']})",
        f"- action: {data['action']}",
        f"- default image: {data['image'] or 'none; clean repository build required'}",
        f"- declared image: {data['declared_image']}",
        f"- source: {data['source']} in {data['metadata_path']}",
        f"- resolved from: {data['resolved_from']}",
        "- confirmation: ask the user to use this image or provide image=<override> before launch",
    ]
    if data.get("backend"):
        lines.insert(3, f"- backend: {data['backend']} ({data['backend_selection_reason']})")
        lines.insert(4, f"- backend contract: {data['backend_contract_path']}")
    return "\n".join(lines)


def main() -> int:
    """Run the image resolver."""
    args = parse_args()
    data = resolve_image(
        args.skill_bank,
        args.model,
        args.action,
        backend=args.backend,
        workload=args.workload,
    )
    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(format_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
