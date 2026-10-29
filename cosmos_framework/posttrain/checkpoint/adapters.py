# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Portable adapters are distinct from full resumable trainer checkpoints.

The latter continue to use CheckpointManager/LLMTrainer unchanged: they include
all model tensors (including PiSSA initial factors), optimizer, clock, scheduler,
rank RNG and data cursor. This module provides strict, atomic portable artifacts
and non-mutating merge preparation for the existing HF conversion pipeline.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.checkpoint.conversion.hf import SIDECAR_DIRS, SIDECAR_FILES
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json, fsync_directory, rename_directory_noreplace
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, canonical_name
from cosmos_framework.posttrain.utils.object_store import file_sha256

FORMAT = "cosmos-posttrain-adapter-v1"
MANIFEST = "adapter_manifest.json"
WEIGHTS = "adapter.safetensors"


def snapshot_identity(directory: str, *, weights: bool) -> dict[str, Any]:
    """Hash actual bytes, including every weight shard; same-size substitutions fail.

    Local immutable HF snapshots are the first supported identity contract. A Hub
    branch name or object-store URI must be staged before training/export.
    """
    root = Path(directory)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError(
            f"adapter identity requires a complete local HF snapshot: {directory!r}; stage a pinned revision first"
        )
    names = {"config.json"} | {n for n in SIDECAR_FILES if (root / n).is_file()}
    for directory_name in SIDECAR_DIRS:
        if (root / directory_name).is_dir():
            names.update(str(p.relative_to(root)) for p in (root / directory_name).rglob("*") if p.is_file())
    if weights:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            names.add(index.name)
            shards = set(json.loads(index.read_text())["weight_map"].values())
        else:
            shards = {"model.safetensors"}
        for name in shards:
            if Path(name).is_absolute() or ".." in Path(name).parts or not (root / name).is_file():
                raise ValueError(f"base snapshot has a missing/invalid weight shard: {name!r}")
        names.update(shards)
    files = {name: file_sha256(str(root / name)) for name in sorted(names)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def make_adapter_identity(base: str, processor: str, targets: dict, media: dict, *, base_dtype: str) -> dict[str, Any]:
    dependencies = {}
    for name in ("torch", "transformers", "safetensors", "torchcodec", "av"):
        try:
            dependencies[name] = importlib.metadata.version(name).split("+")[0]
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "format": FORMAT,
        "architecture": "qwen3_vl",
        "base_storage_dtype": base_dtype,
        "base": snapshot_identity(base, weights=True),
        "processor": snapshot_identity(processor, weights=False),
        "resolved": targets,
        "media": media,
        "dependencies": dependencies,
        "pissa_representation": "original_base_plus_trained_minus_initial_factors",
    }


def state_spec(identity: dict) -> dict[str, tuple[int, ...]]:
    required = {
        "format",
        "architecture",
        "base_storage_dtype",
        "base",
        "processor",
        "resolved",
        "media",
        "dependencies",
        "pissa_representation",
    }
    if not isinstance(identity, dict) or required - set(identity):
        missing = sorted(required - set(identity)) if isinstance(identity, dict) else sorted(required)
        raise ValueError(f"incomplete adapter identity: missing {missing}")
    if identity["base_storage_dtype"] not in ("float32", "bfloat16"):
        raise ValueError("unsupported adapter materialized base dtype")
    if identity.get("format") != FORMAT or identity.get("architecture") != "qwen3_vl":
        raise ValueError("unsupported adapter format/architecture; use a matching Cosmos adapter loader")
    if identity.get("pissa_representation") != "original_base_plus_trained_minus_initial_factors":
        raise ValueError("unsupported PiSSA residual-base representation")
    result = {}
    for row in identity["resolved"]["targets"]:
        out_features, in_features = row["shape"]
        rank = row["settings"]["rank"]
        prefix = row["name"] + "."
        result[prefix + "lora_A.weight"] = (rank, in_features)
        result[prefix + "lora_B.weight"] = (out_features, rank)
        if row["settings"]["method"] == "dora":
            result[prefix + "magnitude"] = (out_features,)
        if row["settings"]["method"] == "pissa":
            result[prefix + "initial_A"] = (rank, in_features)
            result[prefix + "initial_B"] = (out_features, rank)
    result.update({row["name"]: tuple(row["shape"]) for row in identity["resolved"]["extra_parameters"]})
    return result


def _validate_state(state: dict[str, torch.Tensor], identity: dict) -> None:
    spec = state_spec(identity)
    if set(state) != set(spec):
        raise ValueError(
            f"incomplete/extra adapter state: missing={sorted(set(spec) - set(state))}, unexpected={sorted(set(state) - set(spec))}"
        )
    for name, shape in spec.items():
        value = state[name]
        if tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(
                f"invalid adapter tensor {name}: expected finite float32 {shape}, got {value.dtype} {tuple(value.shape)}"
            )
    for row in identity["resolved"]["targets"]:
        if row["name"] != row["owner"]:
            for name in spec:
                if name.startswith(row["name"] + "."):
                    owner_name = row["owner"] + name[len(row["name"]) :]
                    if not torch.equal(state[name], state[owner_name]):
                        raise ValueError(f"tied adapter aliases disagree: {name}, {owner_name}")
    for row in identity["resolved"]["extra_parameters"]:
        if not torch.equal(state[row["name"]], state[row["owner"]]):
            raise ValueError(f"tied extra parameter aliases disagree: {row['name']}, {row['owner']}")


def write_adapter_artifact(
    directory: str, identity: dict, state: dict[str, torch.Tensor], *, provenance: dict | None = None
) -> None:
    """Write once, commit last, atomically publish using the checkpoint IO primitives."""
    _validate_state(state, identity)
    target = os.path.abspath(directory)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(target):
        raise FileExistsError(f"adapter export destination already exists: {target}")
    with tempfile.TemporaryDirectory(prefix=".adapter-stage-", dir=parent) as temporary:
        staging = os.path.join(temporary, "artifact")
        os.mkdir(staging)
        weights_path = os.path.join(staging, WEIGHTS)
        # Aliases are recorded explicitly; safetensors cannot serialize shared storage.
        save_file({n: t.detach().cpu().contiguous().clone() for n, t in state.items()}, weights_path)
        with open(weights_path, "rb") as handle:
            os.fsync(handle.fileno())
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")}
        atomic_write_json(
            os.path.join(staging, MANIFEST),
            {
                "kind": "portable_adapter",
                "identity": identity,
                "weights_sha256": file_sha256(weights_path),
                "versions": versions,
                "provenance": provenance or {},
            },
        )
        try:
            rename_directory_noreplace(staging, target)
        except OSError as error:
            if error.errno != errno.EOPNOTSUPP:
                raise
            # Lustre may not implement renameat2(RENAME_NOREPLACE). mkdir still
            # reserves the destination atomically: a competing writer can never
            # replace its owner. Readers require the commit-last manifest, so an
            # interrupted reservation is explicitly incomplete, never loadable.
            # Keep that reservation on failure; retry at a new immutable path.
            os.mkdir(target)
            fsync_directory(parent)
            os.rename(os.path.join(staging, WEIGHTS), os.path.join(target, WEIGHTS))
            fsync_directory(target)
            os.rename(os.path.join(staging, MANIFEST), os.path.join(target, MANIFEST))
            fsync_directory(target)
        fsync_directory(parent)


def read_adapter_artifact(
    directory: str, *, expected_identity: dict | None = None
) -> tuple[dict, dict[str, torch.Tensor]]:
    root = Path(directory)
    if not (root / MANIFEST).is_file() or not (root / WEIGHTS).is_file():
        raise ValueError(f"incomplete portable adapter at {directory!r}; both {MANIFEST} and {WEIGHTS} are required")
    record = json.loads((root / MANIFEST).read_text())
    if record.get("kind") != "portable_adapter" or file_sha256(str(root / WEIGHTS)) != record.get("weights_sha256"):
        raise ValueError(
            "invalid adapter artifact kind or weight digest; training checkpoints are not portable adapters"
        )
    identity = record["identity"]
    if expected_identity is not None and identity != expected_identity:
        differences = [
            key for key in set(identity) | set(expected_identity) if identity.get(key) != expected_identity.get(key)
        ]
        raise ValueError(
            f"adapter compatibility mismatch in {sorted(differences)}; use the recorded base, processor, targets and media settings"
        )
    state = load_file(str(root / WEIGHTS), device="cpu")
    _validate_state(state, identity)
    return identity, state


def load_adapter_artifact(net: torch.nn.Module, directory: str) -> None:
    """Strict warm start after structure creation/base load, before FSDP. No re-init."""
    _, state = read_adapter_artifact(directory, expected_identity=net._posttrain_adapter_identity)
    parameters = net.state_dict()
    expected_missing = set(parameters) - set(state)
    result = net.load_state_dict(state, strict=False)
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError("portable adapter reload did not match the installed model structure")


def save_adapter_artifact(net: torch.nn.Module, directory: str, *, provenance: dict | None = None) -> None:
    """Collective on distributed models; rank zero publishes an immutable snapshot."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from cosmos_framework.posttrain.checkpoint.worker import run_rank0_converged

    identity = net._posttrain_adapter_identity
    # Reuse DCP's FSDP gather and canonical name handling. Export must not unshard
    # or merge the live model in place, nor alter optimizer ownership.
    state = get_model_state_dict(net, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    run_rank0_converged(
        lambda: write_adapter_artifact(
            directory,
            identity,
            {n: state[n] for n in state_spec(identity)},
            provenance=provenance,
        ),
        "portable adapter export",
    )


def prepare_merged_state(state: dict[str, torch.Tensor], identity: dict) -> dict[str, torch.Tensor]:
    """Pure full-model state transformation for HF export; never combines adapters."""
    state = {canonical_name(n): t for n, t in state.items()}
    spec = state_spec(identity)
    _validate_state({n: state[n] for n in spec if n in state}, identity)
    # Preserve separately trained modules; remove only adapter-owned tensors.
    extra = {row["name"] for row in identity["resolved"]["extra_parameters"]}
    result = {n: t for n, t in state.items() if n not in spec or n in extra}
    for name in extra:
        # Stock HF normalization must retain its base activation boundary. A
        # merged BF16 model rounds these trained FP32 values just like merged
        # projection weights; the portable artifact retains full FP32 precision.
        result[name] = state[name].to(getattr(torch, identity["base_storage_dtype"]))
    merged = {}
    for row in identity["resolved"]["targets"]:
        name, owner = row["name"], row["owner"]
        weight = state[name + ".weight"]
        if name != owner and not torch.equal(weight, state[owner + ".weight"]):
            raise ValueError(f"tied base weights disagree at {name} and {owner}")
        if owner not in merged:
            settings = AdapterComponentConfig(component=row["component"], targets=[name], **row["settings"])
            base = torch.nn.Linear(row["shape"][1], row["shape"][0], bias=False, device="meta", dtype=weight.dtype)
            layer = PosttrainAdapterLinear(base, settings, initialize=False)
            merged[owner] = layer.merged_state_weight(state, owner)
        result[name + ".weight"] = merged[owner]
    return result


def checkpoint_adapter_state(directory: str) -> tuple[dict, dict[str, torch.Tensor], dict]:
    """Read committed consolidated or DCP checkpoints through existing IO/provenance."""
    from cosmos_framework.posttrain.checkpoint.io import load_component
    from cosmos_framework.posttrain.checkpoint.provenance import load_with_stable_provenance

    def read() -> tuple[dict, dict]:
        trainer = load_component(directory, "trainer")
        identity = (trainer or {}).get("objective_identity", {}).get("adapters")
        if identity is None:
            raise ValueError(
                "checkpoint has no adapter identity; a model-only artifact cannot reconstruct adapter semantics"
            )
        if os.path.isfile(os.path.join(directory, "model.pt")):
            model = load_component(directory, "model")["net"]
        else:
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

            with tempfile.TemporaryDirectory(prefix="cosmos-adapter-dcp-") as temporary:
                path = os.path.join(temporary, "model.pt")
                dcp_to_torch_save(os.path.join(directory, "model"), path)
                model = torch.load(path, map_location="cpu", weights_only=True)
        return identity, model

    (identity, state), provenance = load_with_stable_provenance(directory, read)
    return identity, state, provenance
