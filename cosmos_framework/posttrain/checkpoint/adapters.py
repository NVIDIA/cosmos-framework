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
from safetensors.torch import load as load_safetensors
from safetensors.torch import save_file

from cosmos_framework.posttrain.checkpoint.conversion.hf import SIDECAR_DIRS, SIDECAR_FILES
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json, fsync_directory, rename_directory_noreplace
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, canonical_name
from cosmos_framework.posttrain.utils.object_store import file_sha256

FORMAT = "cosmos-posttrain-adapter-v1"
DENSE_FORMAT = "cosmos-posttrain-adapter-v2"
MODEL_EXECUTION_FORMAT = "cosmos-posttrain-adapter-v3"
MERGED_WEIGHT_FORMAT = "cosmos-posttrain-adapter-v4"
MANIFEST = "adapter_manifest.json"
WEIGHTS = "adapter.safetensors"


def adapter_configuration(identity: dict):
    """Reconstruct exact selectors/settings from a validated portable manifest."""
    from cosmos_framework.posttrain.config.adapters import AdapterConfig, AdapterExtraModuleConfig

    # Extra-parameter owners follow the original module selection order. A
    # manifest is sorted by parameter name, which need not put each owner first.
    # Reconstruct that partial order without changing existing artifact identity
    # or choosing a different alias as owner on warm start/export validation.
    extras = identity["resolved"]["extra_parameters"]
    predecessors: dict[str, set[str]] = {}
    for row in extras:
        module, owner = row["name"].rsplit(".", 1)[0], row["owner"].rsplit(".", 1)[0]
        predecessors.setdefault(module, set()).update({owner} if owner != module else set())
    order: dict[str, int] = {}
    while predecessors:
        ready = sorted(name for name, owners in predecessors.items() if owners <= order.keys())
        if not ready:
            raise ValueError("extra module ownership is cyclic or references a missing owner; invalid adapter manifest")
        for name in ready:
            order[name] = len(order)
            del predecessors[name]

    return AdapterConfig(
        model_execution=identity["resolved"].get("model_execution", "default_v1"),
        components=[
            AdapterComponentConfig(component=row["component"], targets=[row["name"]], **row["settings"])
            for row in identity["resolved"]["targets"]
        ],
        extra_modules=[
            AdapterExtraModuleConfig(
                component=row["component"],
                targets=[row["name"].rsplit(".", 1)[0]],
                lr_multiplier=row["lr_multiplier"],
                weight_decay=row["weight_decay"],
            )
            for row in sorted(extras, key=lambda row: (order[row["name"].rsplit(".", 1)[0]], row["name"]))
        ],
    )


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


def make_adapter_identity(
    base: str,
    processor: str,
    targets: dict,
    media: dict,
    *,
    base_dtype: str,
    architecture: str = "qwen3_vl",
    source_identity: dict | None = None,
) -> dict[str, Any]:
    from cosmos_framework.posttrain.model.adapter_families import adapter_family

    adapter_family(architecture)
    dependencies = {}
    for name in ("torch", "transformers", "safetensors", "torchcodec", "av"):
        try:
            dependencies[name] = importlib.metadata.version(name).split("+")[0]
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    result = {
        "format": MERGED_WEIGHT_FORMAT
        if targets.get("model_execution") == "merged_weight_bf16_v1"
        else MODEL_EXECUTION_FORMAT
        if targets.get("model_execution", "default_v1") != "default_v1"
        else DENSE_FORMAT
        if any(row["settings"].get("execution") == "pissa_dense_v1" for row in targets["targets"])
        else FORMAT,
        "architecture": architecture,
        "base_storage_dtype": base_dtype,
        "base": snapshot_identity(base, weights=True) if source_identity is None else source_identity["base"],
        "processor": snapshot_identity(processor, weights=False)
        if source_identity is None
        else source_identity["processor"],
        "resolved": targets,
        "media": media,
        "dependencies": dependencies,
        "pissa_representation": "original_base_plus_trained_minus_initial_factors",
    }
    if source_identity is not None:
        result["model_contract"] = source_identity["model_contract"]
    return result


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
    if identity["base_storage_dtype"] not in ("float32", "bfloat16", "model_defined"):
        raise ValueError("unsupported adapter materialized base dtype")
    if identity["base_storage_dtype"] == "model_defined" and not identity.get("model_contract", {}).get(
        "parameter_dtypes"
    ):
        raise ValueError("model-defined adapter base precision requires the materialized parameter dtype map")
    if identity.get("format") not in (FORMAT, DENSE_FORMAT, MODEL_EXECUTION_FORMAT, MERGED_WEIGHT_FORMAT):
        raise ValueError("unsupported adapter format/architecture; use a matching Cosmos adapter loader")
    from cosmos_framework.posttrain.model.adapter_families import adapter_family

    adapter_family(identity["architecture"])
    if identity.get("pissa_representation") != "original_base_plus_trained_minus_initial_factors":
        raise ValueError("unsupported PiSSA residual-base representation")
    result = {}
    dense = False
    for row in identity["resolved"]["targets"]:
        execution = row["settings"].get("execution", "factorized_v1")
        if execution not in ("factorized_v1", "pissa_dense_v1", "merged_weight_bf16_v1") or (
            execution == "pissa_dense_v1" and row["settings"]["method"] != "pissa"
        ):
            raise ValueError("unsupported adapter execution semantics; use the recorded compatible runtime")
        dense |= execution == "pissa_dense_v1"
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
    profile = identity["resolved"].get("model_execution", "default_v1")
    if profile not in (
        "default_v1",
        "qwen3_vl_moe_stock_v1",
        "merged_weight_bf16_v1",
        "qwen3_vl_hf_stock_v1",
        "qwen_vl_hf_ordered_v1",
    ):
        raise ValueError("unsupported adapter model execution semantics")
    if profile == "qwen3_vl_moe_stock_v1" and (
        identity["architecture"] != "qwen3_vl_moe"
        or identity["resolved"]["extra_parameters"]
        or any(row["settings"].get("execution") != "pissa_dense_v1" for row in identity["resolved"]["targets"])
        or identity["dependencies"].get("transformers") != "5.8.1"
        or identity["dependencies"].get("torch") != "2.10.0"
    ):
        raise ValueError("stock MoE execution requires dense PiSSA and its pinned runtime identity")
    if profile == "qwen_vl_hf_ordered_v1" and (
        identity["architecture"] not in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl")
        or identity["dependencies"].get("transformers") != "5.8.1"
        or identity["dependencies"].get("torch") != "2.10.0"
    ):
        raise ValueError("ordered HF VLM execution requires Qwen2/2.5/3-VL and its pinned runtime")
    if profile == "qwen3_vl_hf_stock_v1" and (
        identity["architecture"] != "qwen3_vl"
        or identity["base_storage_dtype"] != "bfloat16"
        or any(
            row["settings"].get("execution", "factorized_v1") != "factorized_v1"
            for row in identity["resolved"]["targets"]
        )
        or identity["dependencies"].get("transformers") != "5.8.1"
        or identity["dependencies"].get("torch") != "2.10.0"
    ):
        raise ValueError("stock HF Qwen3-VL factorized execution requires BF16 storage and its pinned runtime")
    rounded = [row["settings"].get("execution") == "merged_weight_bf16_v1" for row in identity["resolved"]["targets"]]
    if profile == "merged_weight_bf16_v1":
        if (
            not rounded
            or not all(rounded)
            or identity["architecture"] not in ("qwen3_vl", "qwen3_5")
            or identity["base_storage_dtype"] != "bfloat16"
            or any(row["settings"]["dropout"] for row in identity["resolved"]["targets"])
            or identity["dependencies"].get("transformers") != "5.8.1"
            or identity["dependencies"].get("torch") != "2.10.0"
        ):
            raise ValueError(
                "merged-weight execution requires BF16 dense HF models, zero dropout and its pinned runtime"
            )
    elif any(rounded):
        raise ValueError("rounded component execution requires its complete model execution contract")
    expected_format = (
        MERGED_WEIGHT_FORMAT
        if profile == "merged_weight_bf16_v1"
        else MODEL_EXECUTION_FORMAT
        if profile != "default_v1"
        else DENSE_FORMAT
        if dense
        else FORMAT
    )
    if identity["format"] != expected_format:
        raise ValueError("adapter format/execution mismatch: preserve the recorded numerical execution contract")
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
        publish_staged_artifact(staging, target, commit_file=MANIFEST)


def publish_staged_artifact(staging: str, target: str, *, commit_file: str) -> None:
    """Shared no-replace publication for portable and native merged artifacts.

    Writers fsync payloads first; readers require the commit marker and its file
    digests. The Lustre fallback reserves a unique directory and commits last.
    Interrupted reservations stay visibly incomplete and cannot be overwritten.
    """
    parent = os.path.dirname(os.path.abspath(target))
    try:
        rename_directory_noreplace(staging, target)
    except OSError as error:
        if error.errno != errno.EOPNOTSUPP:
            raise
        os.mkdir(target)
        fsync_directory(parent)
        for name in sorted(os.listdir(staging)):
            if name != commit_file:
                os.rename(os.path.join(staging, name), os.path.join(target, name))
        fsync_directory(target)
        os.rename(os.path.join(staging, commit_file), os.path.join(target, commit_file))
        fsync_directory(target)
    fsync_directory(parent)


def read_adapter_artifact(
    directory: str, *, expected_identity: dict | None = None
) -> tuple[dict, dict[str, torch.Tensor]]:
    record, state = read_adapter_snapshot(directory, expected_identity=expected_identity)
    return record["identity"], state


def read_adapter_snapshot(
    directory: str, *, expected_identity: dict | None = None
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Capture the validated record and tensors from the same serialized payload.

    Hashing a path and then reopening it for mmap deserialization can consume a
    replacement file under the original digest. Adapter payloads exclude base
    weights; retain their bytes through verification and deserialization.
    Exporters also retain this record to bind provenance to the consumed state.
    """
    root = Path(directory)
    if not (root / MANIFEST).is_file() or not (root / WEIGHTS).is_file():
        raise ValueError(f"incomplete portable adapter at {directory!r}; both {MANIFEST} and {WEIGHTS} are required")
    record = json.loads((root / MANIFEST).read_text())
    if record.get("kind") != "portable_adapter":
        raise ValueError("invalid adapter artifact kind; training checkpoints are not portable adapters")
    payload = (root / WEIGHTS).read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("weights_sha256"):
        raise ValueError("invalid adapter weight digest; use a complete immutable portable artifact")
    identity = record["identity"]
    if expected_identity is not None and identity != expected_identity:
        differences = [
            key for key in set(identity) | set(expected_identity) if identity.get(key) != expected_identity.get(key)
        ]
        raise ValueError(
            f"adapter compatibility mismatch in {sorted(differences)}; use the recorded base, processor, targets and media settings"
        )
    state = load_safetensors(payload)
    _validate_state(state, identity)
    return record, state


def load_adapter_artifact(net: torch.nn.Module, directory: str, *, compatibility_tokenizer: Any = None) -> None:
    """Strict warm start after structure creation/base load, before FSDP. No re-init."""
    live_identity = net._posttrain_adapter_identity
    identity, state = read_adapter_artifact(directory)
    expected = live_identity
    if compatibility_tokenizer is not None and identity["media"] != live_identity["media"]:
        from cosmos_framework.posttrain.data.packer.adapter_media import relocated_warm_start_media

        expected = {
            **live_identity,
            "media": relocated_warm_start_media(compatibility_tokenizer, live_identity["media"], identity["media"]),
        }
    if identity != expected:
        differences = sorted(key for key in set(identity) | set(expected) if identity.get(key) != expected.get(key))
        raise ValueError(
            f"adapter compatibility mismatch in {differences}; use the recorded base, processor and settings"
        )
    # The source-location comparison applies only to this import. New checkpoints
    # and exports must retain the live identity so resume without warm_start agrees.
    load_adapter_state(net, live_identity, state)


def load_adapter_state(net: torch.nn.Module, identity: dict, state: dict[str, torch.Tensor]) -> None:
    """Load an already captured snapshot without reopening mutable artifact paths."""
    if identity != net._posttrain_adapter_identity:
        raise ValueError("adapter snapshot identity differs from the installed model")
    _validate_state(state, identity)
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


def prepare_merged_state(
    state: dict[str, torch.Tensor], identity: dict, *, merge_dtype: torch.dtype | None = None
) -> dict[str, torch.Tensor]:
    """Pure full-model state transformation for HF export; never combines adapters."""
    state = {canonical_name(n): t for n, t in state.items()}
    spec = state_spec(identity)
    _validate_state({n: state[n] for n in spec if n in state}, identity)
    if merge_dtype not in (None, torch.float32):
        raise ValueError("promoted merge supports float32; round deployment weights after the merge")
    if merge_dtype is not None:
        state = {name: value.float() if value.is_floating_point() else value for name, value in state.items()}
    # Preserve separately trained modules; remove only adapter-owned tensors.
    extra = {row["name"] for row in identity["resolved"]["extra_parameters"]}
    result = {n: t for n, t in state.items() if n not in spec or n in extra}
    for name in extra:
        # Stock HF normalization must retain its base activation boundary. A
        # merged BF16 model rounds these trained FP32 values just like merged
        # projection weights; the portable artifact retains full FP32 precision.
        dtype = "float32" if merge_dtype is not None else identity["base_storage_dtype"]
        if dtype == "model_defined":
            dtype = identity["model_contract"]["parameter_dtypes"][name]
        result[name] = state[name].to(getattr(torch, dtype))
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
    """Extract portable state from a committed checkpoint with bounded host memory."""
    from cosmos_framework.posttrain.checkpoint.adapter_extract import load_adapter_tensors
    from cosmos_framework.posttrain.checkpoint.io import load_component
    from cosmos_framework.posttrain.checkpoint.provenance import load_with_stable_provenance

    def read() -> tuple[dict, dict]:
        trainer = load_component(directory, "trainer")
        identity = (trainer or {}).get("objective_identity", {}).get("adapters")
        if identity is None:
            raise ValueError(
                "checkpoint has no adapter identity; a model-only artifact cannot reconstruct adapter semantics"
            )
        state = load_adapter_tensors(directory, state_spec(identity))
        _validate_state(state, identity)
        return identity, state

    (identity, state), provenance = load_with_stable_provenance(directory, read)
    return identity, state, provenance


def merged_export_metadata(
    adapter: str,
    identity: dict,
    *,
    artifact_record: dict | None = None,
    output_dtype: str = "float32",
    execution_profile: str | None = None,
) -> dict:
    """Retain the validated portable contract in the materializer's commit marker."""
    # Export uses its captured record, so later path replacement cannot label
    # already consumed tensors with another run's weight digest. Validation of
    # an existing export still compares against the currently supplied artifact.
    manifest = artifact_record if artifact_record is not None else json.loads((Path(adapter) / MANIFEST).read_text())
    if manifest.get("kind") != "portable_adapter" or manifest.get("identity") != identity:
        raise ValueError("portable adapter manifest changed during export; use an immutable artifact")
    result = {
        "format": "cosmos-merged-adapter-v1",
        "adapter_identity": identity,
        "adapter_weights_sha256": manifest["weights_sha256"],
    }
    if output_dtype == "bfloat16":
        result["deployment_precision"] = {
            "dtype": "bfloat16",
            "weight_conversion": "float32_merge_then_bfloat16",
            "equivalence": (
                "precision_specific_dense_execution_v1"
                if identity["resolved"].get("model_execution") == "merged_weight_bf16_v1"
                else "approximate_requires_task_qualification"
            ),
        }
    elif output_dtype != "float32":
        raise ValueError("adapter merged export output_dtype must be float32|bfloat16")
    if execution_profile is not None:
        from cosmos_framework.utils.generator.bf16_cache_contract import PROFILE_FAMILIES, validate_adapter_identity

        if output_dtype != "bfloat16":
            raise ValueError("execution_profile requires bfloat16 output")
        if execution_profile in PROFILE_FAMILIES:
            validate_adapter_identity(identity, profile=execution_profile)
        elif execution_profile == "qwen3_vl_bf16_residual_fp32_v1":
            if identity["resolved"].get("model_execution") == "merged_weight_bf16_v1":
                raise ValueError("merged-weight execution cannot acquire the legacy residual deployment override")
        else:
            raise ValueError(f"unknown execution_profile: {execution_profile!r}")
        result["deployment_precision"]["execution_profile"] = execution_profile
    return result
