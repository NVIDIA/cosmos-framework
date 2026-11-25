# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Portable adapter artifacts and merged Hugging Face exports.

A portable adapter is a directory, separate from any training checkpoint:

* ``adapter_manifest.json``: the adapter identity and the SHA-256 of the weights file;
* ``adapter.safetensors``: the float32 adapter tensors (``lora_A``/``lora_B``, DoRA magnitudes
  and the PiSSA initial factors), named as in the model state dict.

The identity holds what defines the tensors: the architecture (Hugging Face ``model_type``) and,
per target, its name, component, weight shape, ``method``, ``rank``, ``alpha`` and ``use_rslora``.
It also records the base storage dtype the adapters were trained against (``base_dtype``):
compatibility checks between artifacts ignore it, and the merged export rounds the base weights to
it before merging. ``base_fingerprint`` ties the adapters to their base: a digest of a strided
sample of every target's base weight (see ``adapters.fingerprint_sample``), checked on warm start
and merged export. Training-only settings (dropout, PiSSA initialization, learning-rate
multipliers, weight decay) are not part of the identity.

A merged export copies a local Hugging Face base snapshot with every adapted weight replaced by its
dense merge, so the result loads with stock ``from_pretrained`` without any adapter code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.generator.adapter_config import (
    ADAPTER_COMPONENTS,
    ADAPTER_METHODS,
    AdapterIdentity,
    AdapterManifest,
    adapter_architecture,
)
from cosmos_framework.utils.generator.adapters import (
    adapter_state_spec,
    base_fingerprint,
    fingerprint_sample,
    identity_differences,
    merged_target_weight,
    tensor_identity,
)

IDENTITY_FORMAT = "cosmos-adapter-v1"
MANIFEST = "adapter_manifest.json"
WEIGHTS = "adapter.safetensors"
_INDEX = "model.safetensors.index.json"
_SINGLE_SHARD = "model.safetensors"
_BASE_DTYPES = ("float32", "bfloat16")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".index.json")


def adapter_identity(manifest: AdapterManifest, *, base_dtype: str, base_fingerprint: str) -> AdapterIdentity:
    """The identity of a model with installed adapters (see the module docstring)."""
    if base_dtype not in _BASE_DTYPES:
        raise ValueError(f"adapter base storage dtype must be one of {_BASE_DTYPES}, got {base_dtype!r}")
    tensors = tensor_identity(manifest)
    return {
        "format": IDENTITY_FORMAT,
        "base_dtype": base_dtype,
        "base_fingerprint": base_fingerprint,
        "architecture": tensors["architecture"],
        "targets": tensors["targets"],
    }


def compatibility_differences(saved: Mapping[str, Any], live: Mapping[str, Any]) -> list[str]:
    """Fields in which two identities' adapter tensors differ (``base_dtype`` and ``base_fingerprint`` excluded)."""
    ignored = ("base_dtype", "base_fingerprint")
    return identity_differences(
        {k: v for k, v in saved.items() if k not in ignored}, {k: v for k, v in live.items() if k not in ignored}
    )


def validate_identity(identity: Any) -> None:
    """Refuse a malformed or foreign identity with a message naming the offending field."""
    if not isinstance(identity, dict) or identity.get("format") != IDENTITY_FORMAT:
        raise ValueError(f"unsupported adapter identity; expected format {IDENTITY_FORMAT!r}")
    adapter_architecture(identity.get("architecture"))
    if identity.get("base_dtype") not in _BASE_DTYPES:
        raise ValueError(f"adapter identity base_dtype must be one of {_BASE_DTYPES}")
    if not isinstance(identity.get("base_fingerprint"), str) or len(identity["base_fingerprint"]) != 64:
        raise ValueError("adapter identity has no valid base_fingerprint")
    targets = identity.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("adapter identity has no targets")
    for index, row in enumerate(targets):
        if not isinstance(row, dict):
            raise ValueError(f"adapter identity targets[{index}] is not an object")
        problems = [
            key
            for key, valid in (
                ("name", isinstance(row.get("name"), str) and row.get("name")),
                ("component", row.get("component") in ADAPTER_COMPONENTS),
                ("shape", isinstance(row.get("shape"), list) and len(row["shape"]) == 2),
                ("method", row.get("method") in ADAPTER_METHODS),
                ("rank", type(row.get("rank")) is int and row["rank"] > 0),
                ("alpha", type(row.get("alpha")) is float and math.isfinite(row["alpha"]) and row["alpha"] > 0),
                ("use_rslora", type(row.get("use_rslora")) is bool),
            )
            if not valid
        ]
        if problems or any(type(n) is not int or n <= 0 for n in row["shape"]):
            raise ValueError(f"adapter identity targets[{index}] has invalid fields: {problems or ['shape']}")
    names = [row["name"] for row in targets]
    if len(set(names)) != len(names):
        raise ValueError(
            f"adapter identity names a target more than once: {sorted({n for n in names if names.count(n) > 1})}"
        )


def validate_adapter_state(state: Mapping[str, torch.Tensor], identity: Mapping[str, Any]) -> None:
    """Require exactly the identity's tensors, each finite float32 with the recorded shape."""
    validate_identity(identity)
    spec = adapter_state_spec(identity)
    if set(state) != set(spec):
        raise ValueError(
            f"adapter state does not match its identity: missing={sorted(set(spec) - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - set(spec))[:5]}"
        )
    for name, shape in spec.items():
        value = state[name]
        if tuple(value.shape) != shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"invalid adapter tensor {name}: expected finite float32 {shape}, got {value.dtype} {tuple(value.shape)}"
            )


def _publish(staging: str, target: str) -> None:
    """Move a complete staging directory into place; refuse an existing destination."""
    if os.path.lexists(target):
        raise FileExistsError(f"destination already exists: {target}")
    os.rename(staging, target)
    descriptor = os.open(os.path.dirname(target), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _staging_directory(target: str) -> str:
    if os.path.lexists(target):
        raise FileExistsError(f"destination already exists: {target}")
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    return tempfile.mkdtemp(prefix=f".{os.path.basename(target)}.", dir=parent)


def write_adapter_artifact(directory: str, identity: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> None:
    """Write a portable adapter into a new directory, publishing it only once complete."""
    from safetensors.torch import save_file

    validate_adapter_state(state, identity)
    target = os.path.abspath(directory)
    staging = _staging_directory(target)
    try:
        weights = os.path.join(staging, WEIGHTS)
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}, weights)
        record = {"kind": "portable_adapter", "identity": dict(identity), "weights_sha256": _file_sha256(weights)}
        with open(os.path.join(staging, MANIFEST), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_adapter_manifest(directory: str) -> AdapterIdentity:
    """Check that ``directory`` holds a portable adapter and return its validated identity (weights not read)."""
    root = Path(directory)
    if not (root / MANIFEST).is_file() or not (root / WEIGHTS).is_file():
        raise ValueError(f"{directory!r} is not a portable adapter: {MANIFEST} and {WEIGHTS} are required")
    try:
        record = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{directory!r}: {MANIFEST} is not valid JSON") from error
    if not isinstance(record, dict) or record.get("kind") != "portable_adapter" or "identity" not in record:
        raise ValueError(f"{directory!r} is not a portable adapter (training checkpoints are exported first)")
    validate_identity(record["identity"])
    return record["identity"]


def read_adapter_artifact(directory: str) -> tuple[AdapterIdentity, dict[str, torch.Tensor]]:
    """Read and verify a portable adapter; returns ``(identity, state)``."""
    from safetensors.torch import load as load_safetensors

    identity = read_adapter_manifest(directory)
    record = json.loads((Path(directory) / MANIFEST).read_text(encoding="utf-8"))
    payload = (Path(directory) / WEIGHTS).read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("weights_sha256"):
        raise ValueError(f"adapter weights in {directory!r} do not match the manifest digest")
    state = load_safetensors(payload)
    validate_adapter_state(state, identity)
    return identity, state


def load_adapter_artifact(net: torch.nn.Module, directory: str) -> None:
    """Load a portable adapter into a model whose installed adapters (``net.adapter_identity``) match it."""
    identity, state = read_adapter_artifact(directory)
    differences = compatibility_differences(identity, net.adapter_identity)
    if differences:
        raise ValueError(
            f"adapter warm start {directory!r} is incompatible with the configured adapters: {'; '.join(differences[:10])}"
        )
    if identity["base_fingerprint"] != net.adapter_identity["base_fingerprint"]:
        raise ValueError(f"adapter warm start {directory!r} was trained on a different base (base_fingerprint differs)")
    result = net.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(state) & set(result.missing_keys):
        raise ValueError("portable adapter tensors do not match the installed adapter modules")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _base_weight_map(base: str) -> dict[str, str]:
    index = os.path.join(base, _INDEX)
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
    elif os.path.isfile(os.path.join(base, _SINGLE_SHARD)):
        from safetensors import safe_open

        with safe_open(os.path.join(base, _SINGLE_SHARD), framework="pt") as handle:
            weight_map = dict.fromkeys(handle.keys(), _SINGLE_SHARD)
    else:
        raise ValueError(f"base {base!r} has neither {_INDEX} nor {_SINGLE_SHARD}")
    for shard in set(weight_map.values()):
        if Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise ValueError(f"invalid base shard path {shard!r}")
    return weight_map


def _set_config_dtype(config: dict[str, Any], dtype: str, *, top_level: bool = True) -> None:
    """Record the exported dtype in the config and every nested sub-config (``text_config`` etc.)."""
    keys = [key for key in ("torch_dtype", "dtype") if key in config]
    for key in keys or (["torch_dtype"] if top_level else []):
        config[key] = dtype
    for value in config.values():
        if isinstance(value, dict):
            _set_config_dtype(value, dtype, top_level=False)


def export_merged_adapter(
    adapter: str, base: str, out: str, *, dtype: str | None = None, allow_base_mismatch: bool = False
) -> None:
    """Write ``base`` with the portable adapter merged into its weights, as a new HF directory.

    ``base`` must be the local Hugging Face snapshot the adapter was trained on: its target weights
    must reproduce the recorded ``base_fingerprint`` (``allow_base_mismatch`` skips that check).
    Every floating tensor is first rounded to the training storage dtype, so the export computes the
    trained function; each adapted target is then replaced by its dense merge, computed in float64.
    Non-weight files (config, tokenizer, processor) are copied. ``dtype`` defaults to the training
    storage dtype, so a bfloat16-trained model exports at its original size; ``'float32'`` keeps the
    merged weights unrounded.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    identity, state = read_adapter_artifact(adapter)
    dtype = dtype or identity["base_dtype"]
    if dtype not in _BASE_DTYPES:
        raise ValueError(f"merged export dtype must be one of {_BASE_DTYPES}, got {dtype!r}")
    if dtype == "bfloat16":
        log.info("merged weights are rounded to bfloat16; export with dtype float32 to keep the adapter update exactly")
    config_path = os.path.join(base, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"base {base!r} is not a local Hugging Face snapshot (no config.json)")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != identity["architecture"]:
        raise ValueError(
            f"base model_type {config.get('model_type')!r} differs from adapter {identity['architecture']!r}"
        )
    weight_map = _base_weight_map(base)
    rows = {f"{row['name']}.weight": row for row in identity["targets"]}
    missing = sorted(set(rows) - set(weight_map))
    if missing:
        raise ValueError(f"base snapshot lacks adapter target weights: {missing[:5]}")
    output_dtype, base_dtype = getattr(torch, dtype), getattr(torch, identity["base_dtype"])
    # Check the base before writing anything: read only the target weights first.
    samples = []
    for shard in sorted({weight_map[name] for name in rows}):
        with safe_open(os.path.join(base, shard), framework="pt") as handle:
            for name in sorted(set(handle.keys()) & set(rows)):
                tensor = handle.get_tensor(name)
                if list(tensor.shape) != rows[name]["shape"]:
                    raise ValueError(
                        f"base weight {name} has shape {list(tensor.shape)}, adapter expects {rows[name]['shape']}"
                    )
                samples.append((rows[name]["name"], fingerprint_sample(tensor.to(base_dtype))))
    unsampled = sorted(set(rows) - {f"{name}.weight" for name, _ in samples})
    if unsampled:
        raise ValueError(f"base snapshot shards lack adapter target weights listed in its index: {unsampled[:5]}")
    if base_fingerprint(samples) != identity["base_fingerprint"] and not allow_base_mismatch:
        raise ValueError(
            f"base {base!r} does not match the weights the adapter was trained on (base_fingerprint differs); "
            "pass allow_base_mismatch (--allow-base-mismatch) only if this is intended"
        )
    skip_weights = shutil.ignore_patterns(".*", *(f"*{suffix}" for suffix in _WEIGHT_SUFFIXES))
    target = os.path.abspath(out)
    staging = _staging_directory(target)
    try:
        for entry in sorted(os.listdir(base)):
            source = os.path.join(base, entry)
            if entry.startswith(".") or entry.endswith(_WEIGHT_SUFFIXES) or entry in (_INDEX, "config.json"):
                continue  # Weights and the config are rewritten below; other weight formats are not carried over.
            if os.path.isdir(source):
                shutil.copytree(source, os.path.join(staging, entry), ignore=skip_weights)
            else:
                shutil.copy2(source, os.path.join(staging, entry))
        _set_config_dtype(config, dtype)
        with open(os.path.join(staging, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        total, merged = 0, 0
        for shard in sorted(set(weight_map.values())):
            tensors = {}
            with safe_open(os.path.join(base, shard), framework="pt") as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    if tensor.is_floating_point():
                        tensor = tensor.to(base_dtype)
                    row = rows.get(name)
                    if row is not None:
                        tensor = merged_target_weight(row, tensor, state)
                        merged += 1
                    if tensor.is_floating_point():
                        tensor = tensor.to(output_dtype)
                    tensors[name] = tensor.contiguous()
            save_file(tensors, os.path.join(staging, shard), metadata={"format": "pt"})
            total += sum(t.numel() * t.element_size() for t in tensors.values())
        if merged != len(rows):
            raise ValueError(f"merged {merged} of {len(rows)} adapter targets; the base weight map is inconsistent")
        if os.path.isfile(os.path.join(base, _INDEX)):
            with open(os.path.join(staging, _INDEX), "w", encoding="utf-8") as handle:
                json.dump(
                    {"metadata": {"total_size": total}, "weight_map": weight_map}, handle, indent=2, sort_keys=True
                )
                handle.write("\n")
        _publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
