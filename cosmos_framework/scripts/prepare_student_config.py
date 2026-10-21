# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicitly prepare a historical saved training config for student export."""

from cosmos_framework.inference.common.init import init_script

if __name__ == "__main__":
    init_script(env={"COSMOS_DEVICE": "cpu", "COSMOS_TRAINING": "1"})

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import attrs
import yaml

from cosmos_framework.inference.common.config import unstructure_config
from cosmos_framework.inference.common.distillation_export import (
    resolve_student_base_model,
    sanitize_student_model_config,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils.lazy_config.registry import convert_target_to_string


def _add_missing_defaults(
    saved: dict[str, Any],
    defaults: dict[str, Any],
    additions: dict[str, Any],
    skipped: list[str],
    prefix: str = "",
) -> None:
    """Fill absent keys without replacing saved values or crossing config classes."""
    saved_class = saved.get("_target_", saved.get("_type"))
    default_class = defaults.get("_target_", defaults.get("_type"))
    different_class = saved_class != default_class
    if different_class or any(
        key in saved and key in defaults and saved[key] != defaults[key] for key in ("_target_", "_type")
    ):
        skipped.append(prefix)
        return
    for key, value in defaults.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in saved:
            saved[key] = copy.deepcopy(value)
            additions[path] = copy.deepcopy(value)
        elif isinstance(saved[key], dict) and isinstance(value, dict):
            _add_missing_defaults(saved[key], value, additions, skipped, path)


def _describe_changes(before: dict[str, Any], after: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        if key not in after:
            # Record removed field names without copying private teacher/checkpoint payloads.
            changes[path] = {"operation": "removed"}
        elif key not in before:
            changes[path] = {"operation": "added", "after": copy.deepcopy(after[key])}
        elif isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.update(_describe_changes(before[key], after[key], path))
        elif before[key] != after[key]:
            changes[path] = {"operation": "changed", "before": before[key], "after": after[key]}
    return changes


def prepare_student_model(saved_model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project/migrate the student, then explicitly fill its missing schema defaults.

    Existing values, including ``None``, ``False``, and customized nested values,
    survive the default fill. The preceding export projection may deliberately
    change fields (for example disabling compilation); these changes are reported
    separately from inserted defaults.
    """
    model = copy.deepcopy(saved_model)
    base_model, base_config = resolve_student_base_model(model, default_base_model=OmniMoTModel)
    sanitize_student_model_config(
        model,
        base_model_target=convert_target_to_string(base_model),
        base_config_type=convert_target_to_string(base_config),
        base_config_field_names={field.name for field in attrs.fields(base_config)},
    )
    report: dict[str, Any] = {
        "original_target": saved_model.get("_target_"),
        "projected_target": model["_target_"],
        "dropped_config_fields": sorted(saved_model["config"].keys() - model["config"].keys()),
        "projection_changes": _describe_changes(saved_model, model),
        "added_defaults": {},
        "skipped_default_subtrees": [],
    }
    _add_missing_defaults(
        model["config"],
        unstructure_config(base_config()),
        report["added_defaults"],
        report["skipped_default_subtrees"],
    )
    return model, report


def prepare_student_config(config_file: Path, output_config: Path, report_file: Path) -> dict[str, Any]:
    """Extract the saved model and write its prepared config and an audit report."""
    if len({path.resolve() for path in (config_file, output_config, report_file)}) != 3:
        raise ValueError("Input config, output config, and report must be distinct files.")
    if output_config.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("Output config must have a .yaml, .yml, or .json extension.")
    source = config_file.read_bytes()
    saved = yaml.safe_load(source)
    if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
        raise TypeError("Expected a saved training config with a model dictionary under the 'model' key.")
    model, report = prepare_student_model(saved["model"])
    report.update({"input_config": str(config_file.resolve()), "input_sha256": hashlib.sha256(source).hexdigest()})
    # Do not apply public path/alias replacements to this internal saved config.
    prepared = {"model": model}
    if output_config.suffix.lower() == ".json":
        output_text = json.dumps(prepared, indent=2, sort_keys=True) + "\n"
    else:
        output_text = yaml.safe_dump(prepared, sort_keys=False)
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(output_text)
    report_file.write_text(report_text)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True, help="Original saved training YAML/JSON config.")
    parser.add_argument("--output-config", type=Path, required=True, help="Prepared student YAML/JSON config.")
    parser.add_argument("--report", type=Path, required=True, help="Projection and inserted-defaults JSON report.")
    args = parser.parse_args(argv)
    report = prepare_student_config(args.config_file, args.output_config, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
