# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate the unified-v1 schema receipt, example masks, and normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

RAW_DIM = 59
MODEL_DIM = 64
RIGHT_OPENNESS = 33
LEFT_OPENNESS = 58
REFERENCE_LOW = -1.0
REFERENCE_HIGH = 1.0
ROT6D_STARTS = (3, 12, 37)
ROT6D_INDICES = frozenset(index for start in ROT6D_STARTS for index in range(start, start + 6))
ROT6D_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain one JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no samples")
    return rows


def require_dict(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def require_list(value: Any, role: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{role} must be an array")
    return value


def require_number(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{role} must be finite")
    return result


def canonical_dimension_names() -> list[str]:
    names = ["ego_tx", "ego_ty", "ego_tz"]
    names.extend(f"ego_rot6d_c{column}_{axis}" for column in range(2) for axis in "xyz")
    names.extend(["right_wrist_tx", "right_wrist_ty", "right_wrist_tz"])
    names.extend(f"right_wrist_rot6d_c{column}_{axis}" for column in range(2) for axis in "xyz")
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        names.extend(f"right_{finger}_{axis}" for axis in "xyz")
    names.append("right_openness")
    names.extend(["left_wrist_tx", "left_wrist_ty", "left_wrist_tz"])
    names.extend(f"left_wrist_rot6d_c{column}_{axis}" for column in range(2) for axis in "xyz")
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        names.extend(f"left_{finger}_{axis}" for axis in "xyz")
    names.append("left_openness")
    if len(names) != RAW_DIM:
        raise AssertionError("canonical dimension definition must contain 59 entries")
    return names


def checked_path(root: Path, relative: Any, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{role} must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"{role} must be relative to --repo-root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{role} escapes --repo-root")
    if not resolved.is_file():
        raise ValueError(f"{role} is not a readable file: {relative}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_receipt(root: Path, receipt: dict[str, Any]) -> tuple[dict[str, Any], list[float], list[float]]:
    expected_scalars = {
        "schema_name": "unified_v1",
        "schema_receipt_version": 1,
        "raw_action_dim": RAW_DIM,
        "model_action_dim": MODEL_DIM,
        "pose_convention": "backward_chunk_anchored_16f",
        "rotation_format": "rot6d_columns",
        "action_space": "normalized_unified_v1",
    }
    for key, expected in expected_scalars.items():
        if receipt.get(key) != expected:
            raise ValueError(f"receipt {key} must be {expected!r}")
    if receipt.get("dimension_names") != canonical_dimension_names():
        raise ValueError("receipt dimension_names do not match the 59-index contract")

    normalization = require_dict(receipt.get("normalization"), "normalization")
    if normalization.get("method") != "global_asinh":
        raise ValueError("normalization.method must be global_asinh")
    stats_path = checked_path(root, normalization.get("statistics_path"), "normalization.statistics_path")
    expected_digest = normalization.get("statistics_sha256")
    if not isinstance(expected_digest, str) or sha256(stats_path) != expected_digest:
        raise ValueError("normalization statistics SHA-256 does not match the receipt")
    stats_document = require_dict(load_json(stats_path), "normalization statistics")
    metadata = require_dict(stats_document.get("metadata"), "normalization statistics metadata")
    if metadata.get("method") != "global_asinh" or metadata.get("action_schema") != "unified_v1":
        raise ValueError("normalization statistics method or action schema is incompatible")
    if metadata.get("dim") != RAW_DIM:
        raise ValueError("normalization statistics dimension must be 59")
    global_stats = require_dict(stats_document.get("global"), "normalization global statistics")
    q01 = [
        require_number(item, f"q01[{index}]") for index, item in enumerate(require_list(global_stats.get("q01"), "q01"))
    ]
    q99 = [
        require_number(item, f"q99[{index}]") for index, item in enumerate(require_list(global_stats.get("q99"), "q99"))
    ]
    if len(q01) != RAW_DIM or len(q99) != RAW_DIM:
        raise ValueError("normalization statistics must contain 59 q01 and q99 values")
    if any(high <= low for low, high in zip(q01, q99, strict=True)):
        raise ValueError("every q99 value must exceed q01")

    domains = require_dict(receipt.get("domains"), "domains")
    if not domains:
        raise ValueError("receipt must define at least one domain")
    for name, value in domains.items():
        domain = require_dict(value, f"domains.{name}")
        indices = require_list(domain.get("valid_indices"), f"domains.{name}.valid_indices")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise ValueError(f"domains.{name}.valid_indices must contain integers")
        if indices != sorted(set(indices)) or any(index < 0 or index >= RAW_DIM for index in indices):
            raise ValueError(f"domains.{name}.valid_indices must be unique sorted indices in [0,58]")
        width = require_number(domain.get("fully_open_width_m"), f"domains.{name}.fully_open_width_m")
        if width <= 0:
            raise ValueError(f"domains.{name}.fully_open_width_m must be positive")
        calibration = require_dict(domain.get("gripper_calibration"), f"domains.{name}.gripper_calibration")
        closed = require_number(calibration.get("closed_value"), f"domains.{name}.closed_value")
        opened = require_number(calibration.get("open_value"), f"domains.{name}.open_value")
        if closed == opened:
            raise ValueError(f"domains.{name} gripper endpoints must differ")
        if RIGHT_OPENNESS not in indices or LEFT_OPENNESS not in indices:
            raise ValueError(f"domains.{name} must make both gripper slots valid")
    return domains, q01, q99


def effective_stats(q01: list[float], q99: list[float], width: float) -> tuple[list[float], list[float]]:
    low = q01.copy()
    high = q99.copy()
    for index in (RIGHT_OPENNESS, LEFT_OPENNESS):
        low[index] /= width
        high[index] /= width
    return low, high


def normalize(raw: float, low: float, high: float) -> float:
    offset = (low + high) / 2.0
    scale = max((high - low) / 2.0, 1e-8)
    return math.asinh((raw - offset) / scale) / math.asinh(1.0)


def denormalize(value: float, low: float, high: float) -> float:
    offset = (low + high) / 2.0
    scale = max((high - low) / 2.0, 1e-8)
    return math.sinh(value * math.asinh(1.0)) * scale + offset


def validate_reference_band(q01: list[float], q99: list[float]) -> None:
    """Confirm that every pinned q01/q99 pair maps to the documented band."""
    for index, (low, high) in enumerate(zip(q01, q99, strict=True)):
        if not math.isclose(normalize(low, low, high), REFERENCE_LOW, abs_tol=1e-12):
            raise ValueError(f"q01[{index}] must normalize to {REFERENCE_LOW}")
        if not math.isclose(normalize(high, low, high), REFERENCE_HIGH, abs_tol=1e-12):
            raise ValueError(f"q99[{index}] must normalize to {REFERENCE_HIGH}")


def validate_rotation_columns(raw_values: list[float], valid_indices: set[int], row_index: int) -> None:
    """Reject malformed source rotations without projecting or changing them."""
    for start in ROT6D_STARTS:
        if not all(index in valid_indices for index in range(start, start + 6)):
            continue
        first = raw_values[start : start + 3]
        second = raw_values[start + 3 : start + 6]
        first_norm_squared = sum(value * value for value in first)
        second_norm_squared = sum(value * value for value in second)
        dot_product = sum(left * right for left, right in zip(first, second, strict=True))
        if (
            abs(first_norm_squared - 1.0) > ROT6D_TOLERANCE
            or abs(second_norm_squared - 1.0) > ROT6D_TOLERANCE
            or abs(dot_product) > ROT6D_TOLERANCE
        ):
            raise ValueError(f"action[{row_index}][{start}:{start + 6}] must denormalize to orthonormal rot6d columns")


def local_sidecar(manifest: Path, locator: Any) -> Path:
    if not isinstance(locator, str) or not locator:
        raise ValueError("action_path must be a non-empty string")
    parsed = urlparse(locator)
    if parsed.scheme not in {"", "file"}:
        raise ValueError("schema validation requires a materialized local action sidecar")
    candidate = Path(parsed.path) if parsed.scheme == "file" else Path(locator)
    path = candidate if candidate.is_absolute() else manifest.parent / candidate
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"action sidecar is not readable: {path}")
    return path


def validate_sidecar(
    sidecar_path: Path,
    domain_name: str,
    domain: dict[str, Any],
    q01: list[float],
    q99: list[float],
) -> tuple[int, int]:
    sidecar = require_dict(load_json(sidecar_path), "action sidecar")
    if sidecar.get("action_space") != "normalized_unified_v1":
        raise ValueError("action sidecar action_space must be normalized_unified_v1")
    actions = require_list(sidecar.get("action"), "action sidecar action")
    domains = require_list(sidecar.get("domain_names"), "action sidecar domain_names")
    if len(domains) not in {1, len(actions)} or any(name != domain_name for name in domains):
        raise ValueError("sidecar domain_names must match its manifest and action rows")
    valid_indices = set(require_list(domain.get("valid_indices"), "domain valid_indices"))
    width = require_number(domain.get("fully_open_width_m"), "fully_open_width_m")
    low, high = effective_stats(q01, q99, width)
    out_of_reference_count = 0
    for row_index, row_value in enumerate(actions):
        row = require_list(row_value, f"action[{row_index}]")
        if len(row) != RAW_DIM:
            raise ValueError(f"action[{row_index}] must contain 59 values")
        values = [require_number(item, f"action[{row_index}][{index}]") for index, item in enumerate(row)]
        raw_values = [0.0] * RAW_DIM
        for index, value in enumerate(values):
            if index not in valid_indices:
                if abs(value) > 1e-12:
                    raise ValueError(f"action[{row_index}][{index}] must be zero because its mask is false")
                continue
            if value < REFERENCE_LOW or value > REFERENCE_HIGH:
                out_of_reference_count += 1
            raw = denormalize(value, low[index], high[index])
            raw_values[index] = raw
            if index in ROT6D_INDICES and not -1.0 - 1e-6 <= raw <= 1.0 + 1e-6:
                raise ValueError(f"action[{row_index}][{index}] denormalizes outside the rot6d component envelope")
            if index in {RIGHT_OPENNESS, LEFT_OPENNESS} and not -1e-6 <= raw <= 1.0 + 1e-6:
                raise ValueError(f"action[{row_index}][{index}] denormalizes outside calibrated openness [0,1]")
            recovered = normalize(raw, low[index], high[index])
            if not math.isclose(value, recovered, rel_tol=1e-11, abs_tol=1e-11):
                raise ValueError(f"action[{row_index}][{index}] failed normalization round trip")
        validate_rotation_columns(raw_values, valid_indices, row_index)
        padded = values + [0.0] * (MODEL_DIM - RAW_DIM)
        mask = [index in valid_indices for index in range(RAW_DIM)] + [False] * (MODEL_DIM - RAW_DIM)
        if len(padded) != MODEL_DIM or len(mask) != MODEL_DIM or any(padded[index] != 0 for index in range(59, 64)):
            raise ValueError("59-to-64 padding contract failed")
    return len(actions), out_of_reference_count


def map_native(layout: str, native: list[float]) -> list[float]:
    mapped = [0.0] * RAW_DIM
    if layout == "left9,left_open,right9,right_open" and len(native) == 20:
        mapped[34:43] = native[0:9]
        mapped[LEFT_OPENNESS] = native[9]
        mapped[9:18] = native[10:19]
        mapped[RIGHT_OPENNESS] = native[19]
        return mapped
    if layout == "ego9,right9,right_open,left9,left_open" and len(native) == 29:
        mapped[0:9] = native[0:9]
        mapped[9:18] = native[9:18]
        mapped[RIGHT_OPENNESS] = native[18]
        mapped[34:43] = native[19:28]
        mapped[LEFT_OPENNESS] = native[28]
        return mapped
    raise ValueError("worked example native layout or width is unsupported")


def validate_worked_example(
    receipt: dict[str, Any], domains: dict[str, Any], q01: list[float], q99: list[float]
) -> None:
    example = require_dict(receipt.get("worked_example"), "worked_example")
    domain_name = example.get("domain")
    if not isinstance(domain_name, str) or domain_name not in domains:
        raise ValueError("worked_example.domain must name a receipt domain")
    domain = require_dict(domains[domain_name], f"domains.{domain_name}")
    native = [
        require_number(item, f"worked_example.native_raw[{index}]")
        for index, item in enumerate(require_list(example.get("native_raw"), "worked_example.native_raw"))
    ]
    mapped = map_native(str(domain.get("native_layout")), native)
    valid_indices = set(require_list(domain.get("valid_indices"), "worked example valid indices"))
    low, high = effective_stats(q01, q99, require_number(domain.get("fully_open_width_m"), "worked example width"))
    normalized = [
        normalize(value, low[index], high[index]) if index in valid_indices else 0.0
        for index, value in enumerate(mapped)
    ]
    expected = require_dict(example.get("expected_normalized_valid"), "worked_example.expected_normalized_valid")
    tolerance = require_number(example.get("normalized_tolerance"), "worked_example.normalized_tolerance")
    if {int(index) for index in expected} != valid_indices:
        raise ValueError("worked example must record every valid normalized index")
    for index_text, value in expected.items():
        index = int(index_text)
        expected_value = require_number(value, f"worked example index {index}")
        if not math.isclose(normalized[index], expected_value, abs_tol=tolerance):
            raise ValueError(f"worked example normalized value differs at index {index}")
    padded = normalized + [0.0] * (MODEL_DIM - RAW_DIM)
    recovered = [
        denormalize(normalized[index], low[index], high[index]) if index in valid_indices else 0.0
        for index in range(RAW_DIM)
    ]
    if len(padded) != MODEL_DIM or any(padded[index] != 0 for index in range(59, 64)):
        raise ValueError("worked example padding failed")
    if any(not math.isclose(mapped[index], recovered[index], abs_tol=1e-11) for index in range(RAW_DIM)):
        raise ValueError("worked example normalize/denormalize round trip failed")


def validate_examples(
    root: Path,
    receipt: dict[str, Any],
    domains: dict[str, Any],
    q01: list[float],
    q99: list[float],
    requested_manifests: list[Path],
) -> tuple[int, int, int]:
    examples = require_list(receipt.get("examples"), "examples")
    expected: dict[tuple[str, str], str] = {}
    for position, value in enumerate(examples):
        example = require_dict(value, f"examples[{position}]")
        key = (str(example.get("manifest")), str(example.get("sample")))
        domain = example.get("domain")
        if key in expected or not isinstance(domain, str) or domain not in domains:
            raise ValueError("example receipt entries must be unique and reference a declared domain")
        if example.get("validity_mask") != domain:
            raise ValueError("every example must reference its domain validity mask")
        expected[key] = domain
    if requested_manifests:
        manifests = [path.expanduser().resolve() for path in requested_manifests]
    else:
        manifests = sorted({checked_path(root, manifest, "example manifest") for manifest, _ in expected})
    sample_count = 0
    action_count = 0
    out_of_reference_count = 0
    seen: set[tuple[str, str]] = set()
    for manifest in manifests:
        try:
            relative_manifest = manifest.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("manifest must be inside --repo-root") from error
        for row in load_jsonl(manifest):
            sample = row.get("name")
            domain_name = row.get("domain_name")
            if not isinstance(sample, str) or not isinstance(domain_name, str):
                raise ValueError("manifest name and domain_name must be strings")
            key = (relative_manifest, sample)
            if key not in expected or expected[key] != domain_name:
                raise ValueError(f"sample {sample!r} is absent from or conflicts with the schema receipt")
            actions, out_of_reference = validate_sidecar(
                local_sidecar(manifest, row.get("action_path")),
                domain_name,
                require_dict(domains[domain_name], f"domains.{domain_name}"),
                q01,
                q99,
            )
            num_frames = row.get("num_frames")
            if isinstance(num_frames, bool) or not isinstance(num_frames, int) or actions != num_frames - 1:
                raise ValueError(f"sample {sample!r} must have num_frames - 1 actions")
            seen.add(key)
            sample_count += 1
            action_count += actions
            out_of_reference_count += out_of_reference
    selected_paths = {path.relative_to(root).as_posix() for path in manifests}
    expected_selected = {key for key in expected if key[0] in selected_paths}
    if seen != expected_selected:
        raise ValueError("manifest samples do not exactly match their schema receipt entries")
    return sample_count, action_count, out_of_reference_count


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    receipt = require_dict(load_json(receipt_path), "schema receipt")
    domains, q01, q99 = validate_receipt(root, receipt)
    validate_reference_band(q01, q99)
    validate_worked_example(receipt, domains, q01, q99)
    samples, actions, out_of_reference = validate_examples(root, receipt, domains, q01, q99, args.manifest)
    report = {
        "status": "passed",
        "dimensions": RAW_DIM,
        "domains": len(domains),
        "samples": samples,
        "action_rows": actions,
        "normalized_reference_band": [REFERENCE_LOW, REFERENCE_HIGH],
        "out_of_reference_values": out_of_reference,
        "round_trip": "passed",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
