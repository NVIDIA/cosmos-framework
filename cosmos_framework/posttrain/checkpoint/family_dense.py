# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded dense deployment artifacts for native driving/action policy families.

Merge against the effective frozen base, never the untouched source weights.
Only merged language matrices use the requested output dtype; all other native
parameters and buffers retain their dtypes, including trained FP32 action heads.
Frozen non-target aliases retain both standard weight keys on disk and share one
Parameter again after native reload; quantized targets and trainable auxiliaries
must remain unaliased.
The artifact requires its family-owned constructor/processor, not a generic HF
language-model loader. It is deliberately not a training-resume checkpoint.
The distinct unmerged format preserves actual BF16 effective weights and exact
FP32 LoRA factors, reconstructing the native separate operations for inference.
An explicit head-adapter group additionally retains its FP32 bases and biases;
that group supports only unmerged deployment.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import (
    load_adapter_weights,
    read_adapter_metadata,
    require_identity_subset,
)
from cosmos_framework.posttrain.checkpoint.dense_adapter import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    digest_json,
    effective_base_weight,
    file_record,
    merge_effective_weight,
    write_dense_shards,
)
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json, fsync_directory
from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_inventory_signature
from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
from cosmos_framework.utils.generator.lora import LoraInjectedLinear, NativeLoRAMixin

FAMILY_DENSE_MANIFEST = "cosmos_family_dense_export.json"
FAMILY_DENSE_PROTOCOL = "cosmos.native-family-dense-export.v1"
FAMILY_UNMERGED_MANIFEST = "cosmos_family_unmerged_export.json"
FAMILY_UNMERGED_PROTOCOL = "cosmos.native-family-unmerged-export.v1"
_DTYPES = {
    name: getattr(torch, name)
    for name in ("bfloat16", "float32", "float64", "float16", "int64", "int32", "int16", "int8", "uint8", "bool")
}
_SAFE_DTYPES = {
    "bfloat16": "BF16",
    "float32": "F32",
    "float64": "F64",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}


def _tree_signatures(root: Path) -> dict[str, tuple[int, ...]]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("family dense artifacts must not contain symlinks")
        info = path.stat()
        result[str(path.relative_to(root))] = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    return result


def _sidecar_sources(sidecars: Mapping[str, str] | None) -> dict[str, Path]:
    result = {}
    for name, source in (sidecars or {}).items():
        relative = Path(name)
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or any(part in {".", ".."} for part in relative.parts)
            or str(relative) != name
            or name in {FAMILY_DENSE_MANIFEST, FAMILY_UNMERGED_MANIFEST, "model.safetensors.index.json"}
            or name.endswith(".safetensors")
        ):
            raise ValueError(f"invalid family dense sidecar path: {name!r}")
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"family dense sidecar is not a regular source file: {name}")
        result[name] = path
    if any(any(str(parent) in result for parent in Path(name).parents if str(parent) != ".") for name in result):
        raise ValueError("family dense sidecar paths collide")
    return result


def head_adapter_targets(identity: Mapping[str, Any]) -> dict[str, list[int]]:
    """The optional ordinary FP32 head group is distinct from quantized language targets."""
    if "head_adapters" not in identity:
        return {}
    record = identity["head_adapters"]
    adapter = identity["adapter"]
    fixed = {
        "profile": "gr00t_n1d7_dit_attention_lora_v1",
        "rank": 32,
        "alpha": 64.0,
        "base_dtype": "float32",
        "bias": "frozen_float32",
        "parameter_dtype": "float32",
        "compute": "native_separate_gemms_ambient_autocast",
    }
    if (
        not isinstance(record, dict)
        or set(record) != {*fixed, "targets"}
        or any(record.get(key) != value for key, value in fixed.items())
        or any(adapter.get(key) != fixed[key] for key in ("rank", "alpha"))
        or adapter.get("mode") not in {"nf4_lora", "bf16_lora"}
        or identity.get("auxiliary_parameters") != []
        or not isinstance(record.get("targets"), dict)
        or not record["targets"]
    ):
        raise ValueError("unsupported FP32 head adapter identity")
    targets = record["targets"]
    if set(targets) & (set(identity["targets"]) | set(identity["quantized_tensors"])):
        raise ValueError("head adapters must be separate from quantized language targets")
    for name, shape in targets.items():
        if (
            not isinstance(name, str)
            or not name.startswith("action_head.")
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dim) is not int or dim <= 0 for dim in shape)
        ):
            raise ValueError("invalid explicit head adapter target geometry")
        for leaf, expected_shape in (("weight", shape), ("bias", [shape[0]])):
            frozen = identity["frozen_tensors"].get(f"{name}.{leaf}", {})
            if frozen.get("shape") != expected_shape or frozen.get("dtype") != "float32":
                raise ValueError("head adapter requires its complete frozen FP32 base and bias")
    return targets


def _validate_source_inventory(manifest: Mapping[str, Any], *, unmerged: bool = False) -> None:
    from cosmos_framework.posttrain.model.checkpoint import ADAPTER_CHECKPOINT_PROTOCOL

    metadata, tensors = manifest["source_metadata"], manifest["tensors"]
    identity = metadata["identity"]
    heads = head_adapter_targets(identity)
    if heads and not unmerged:
        raise ValueError(
            "action-head adapters support only unmerged export (--unmerged); merged export is not supported"
        )
    if (
        set(metadata) != {"protocol", "identity", "topology", "parameters", "optimizer"}
        or metadata["protocol"] != ADAPTER_CHECKPOINT_PROTOCOL
        or metadata["topology"] != {"strategy": "single_device", "world_size": 1}
    ):
        raise ValueError("family dense source checkpoint schema/topology differs")
    targets = {f"{name}.weight" for name in identity["targets"]}
    adapters = (
        {f"{name}.lora_{factor}.weight" for name in (*identity["targets"], *heads) for factor in ("A", "B")}
        if identity["adapter"]["mode"].endswith("_lora")
        else set()
    )
    auxiliaries = set(identity["auxiliary_parameters"])
    trained = manifest["trained_tensors"]
    if set(metadata["parameters"]) != adapters | auxiliaries or set(trained) != adapters | auxiliaries:
        raise ValueError("family dense trained inventory omits adapters or auxiliary task parameters")
    for name, record in trained.items():
        if (
            set(record) != {"shape", "dtype", "sha256"}
            or {key: record[key] for key in ("shape", "dtype")} != metadata["parameters"][name]
        ):
            raise ValueError(f"family dense trained tensor metadata differs: {name}")
        if record["dtype"] != "float32" or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
            raise ValueError(f"family dense trained tensor precision/digest differs: {name}")
    if unmerged:
        for name, (out_features, in_features) in {**identity["targets"], **heads}.items():
            for factor, shape in (
                ("A", [identity["adapter"]["rank"], in_features]),
                ("B", [out_features, identity["adapter"]["rank"]]),
            ):
                if trained[f"{name}.lora_{factor}.weight"]["shape"] != shape:
                    raise ValueError(f"unmerged adapter shape differs from its target and rank: {name}")
    expected_parameters = set(identity["frozen_tensors"]) | targets | auxiliaries | (adapters if unmerged else set())
    if {name for name, record in tensors.items() if record["kind"] == "parameter"} != expected_parameters:
        raise ValueError("family dense parameter inventory differs from the source family")
    if {name for name, record in tensors.items() if record["kind"] == "buffer"} != set(identity["base_buffers"]):
        raise ValueError("family dense buffer inventory differs from the source family")
    aliases = manifest["parameter_aliases"]
    if (
        not isinstance(aliases, list)
        or any(
            not isinstance(group, list)
            or len(group) < 2
            or any(not isinstance(name, str) for name in group)
            or group != sorted(set(group))
            for group in aliases
        )
        or aliases != sorted(aliases)
    ):
        raise ValueError("family dense parameter aliases require canonical distinct name groups")
    alias_names = [name for group in aliases for name in group]
    head_base = {f"{name}.{leaf}" for name in heads for leaf in ("weight", "bias")}
    if (
        len(alias_names) != len(set(alias_names))
        or not set(alias_names) <= set(identity["frozen_tensors"]) - targets - head_base
    ):
        raise ValueError("family dense aliases may contain only distinct frozen non-target parameters")
    for group in aliases:
        first = {key: tensors[group[0]][key] for key in ("shape", "dtype", "sha256")}
        if any(
            tensors[name]["kind"] != "parameter"
            or {key: tensors[name][key] for key in ("shape", "dtype", "sha256")} != first
            for name in group
        ):
            raise ValueError("family dense aliased parameters differ in shape, dtype or value")
    for name, record in tensors.items():
        digest = {key: record[key] for key in ("shape", "dtype", "sha256")}
        expected = (
            trained[name]
            if name in auxiliaries or (unmerged and name in adapters)
            else identity["base_buffers"][name]
            if record["kind"] == "buffer"
            else identity["frozen_tensors"].get(name)
            if name not in targets
            else None
        )
        if expected is not None and digest != {key: expected[key] for key in ("shape", "dtype", "sha256")}:
            raise ValueError(f"family dense changed a preserved head/base/buffer tensor: {name}")
        if name in targets:
            if record["shape"] != identity["targets"][name.removesuffix(".weight")] or record["dtype"] != (
                "bfloat16" if unmerged else manifest["numerical_contract"]["merged_backbone_dtype"]
            ):
                raise ValueError(f"family dense target shape/output dtype differs: {name}")
            if unmerged and identity["quantization"] is None:
                if digest != {key: identity["frozen_tensors"][name][key] for key in ("shape", "dtype", "sha256")}:
                    raise ValueError(f"unmerged BF16 target changed its frozen source values: {name}")


def _buffers(net: nn.Module) -> dict[str, tuple[nn.Module, str, torch.Tensor, bool]]:
    return {
        f"{name}.{leaf}" if name else leaf: (owner, leaf, value, leaf not in owner._non_persistent_buffers_set)
        for name, owner in net.named_modules(remove_duplicate=False)
        for leaf, value in owner._buffers.items()
        if value is not None
    }


def _parameter_aliases(parameters: Mapping[str, nn.Parameter]) -> list[list[str]]:
    owners: dict[int, list[str]] = {}
    for name, parameter in parameters.items():
        owners.setdefault(id(parameter), []).append(name)
    return sorted(sorted(names) for names in owners.values() if len(names) > 1)


def _mixed_shard_plan(inventory: Mapping[str, Mapping[str, Any]], max_bytes: int) -> dict[str, str]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be a positive integer")
    groups: list[list[str]] = []
    pending: list[str] = []
    size = 0
    for name, record in sorted(inventory.items()):
        shape, dtype = record["shape"], record["dtype"]
        if dtype not in _DTYPES or not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"unsupported dense family tensor shape/dtype: {name}")
        nbytes = math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()
        if pending and size + nbytes > max_bytes:
            groups.append(pending)
            pending, size = [], 0
        pending.append(name)
        size += nbytes
        if size >= max_bytes:
            groups.append(pending)
            pending, size = [], 0
    if pending:
        groups.append(pending)
    return {
        name: f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        for number, group in enumerate(groups, start=1)
        for name in group
    }


def _inventory(net: nn.Module, identity: Mapping[str, Any], dtype: str, *, unmerged: bool = False):
    heads = head_adapter_targets(identity)
    if heads and not unmerged:
        raise ValueError(
            "action-head adapters support only unmerged export (--unmerged); merged export is not supported"
        )
    parameters = dict(net.named_parameters(remove_duplicate=False))
    buffers = _buffers(net)
    alias_names = {name for group in _parameter_aliases(parameters) for name in group}
    allowed_aliases = set(identity["frozen_tensors"]) - {f"{name}.weight" for name in identity["targets"]}
    allowed_aliases -= {f"{name}.{leaf}" for name in heads for leaf in ("weight", "bias")}
    if not alias_names <= allowed_aliases or any(parameters[name].requires_grad for name in alias_names):
        raise ValueError(
            "family dense export aliases must be frozen non-target parameters; adapter/auxiliary aliases are unsupported"
        )
    if set(parameters) & set(buffers):
        raise ValueError("family dense parameter and buffer names overlap")
    target_shapes = {**identity["targets"], **heads}
    modules = {name: net.get_submodule(name) for name in target_shapes}
    adapted = identity["adapter"]["mode"].endswith("_lora")
    adapter_names = {f"{name}.lora_{factor}.weight" for name in modules for factor in ("A", "B")} if adapted else set()
    actual_adapters = {name for name, module in net.named_modules() if isinstance(module, NativeLoRAMixin)}
    if actual_adapters != (set(modules) if adapted else set()):
        raise ValueError("family dense adapter coverage differs from the source identity")
    if set(net._cosmos_adapter_parameters) != adapter_names | set(identity["auxiliary_parameters"]):
        raise ValueError("family dense trainables differ from the complete adapter/auxiliary inventory")
    if set(buffers) != set(identity["base_buffers"]):
        raise ValueError("family dense native buffers differ from the source identity")
    inventory = {}
    for name, parameter in parameters.items():
        if name in adapter_names and not unmerged:
            continue
        target = name.removesuffix(".weight") if name.endswith(".weight") else None
        inventory[name] = {
            "shape": list(target_shapes[target]) if target in modules else list(parameter.shape),
            "dtype": "float32"
            if target in heads
            else dtype
            if target in modules
            else str(parameter.dtype).removeprefix("torch."),
            "kind": "parameter",
        }
    for name, (_, _, value, persistent) in buffers.items():
        inventory[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "kind": "buffer",
            "persistent": persistent,
        }
    return inventory, parameters, buffers, modules


def _dense_values(
    net: nn.Module, identity: Mapping[str, Any], dtype: str, *, unmerged: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    inventory, parameters, buffers, modules = _inventory(net, identity, dtype, unmerged=unmerged)
    heads = head_adapter_targets(identity)
    for name in sorted(inventory):
        target = name.removesuffix(".weight") if name.endswith(".weight") else None
        if target in modules:
            module = modules[target]
            if isinstance(module, NativeLoRAMixin):
                effective = effective_base_weight(module)
                try:
                    if unmerged:
                        expected_dtype = torch.float32 if target in heads else torch.bfloat16
                        if effective.dtype != expected_dtype:
                            raise ValueError("unmerged export requires the declared actual language/head base dtype")
                        value = effective.to(device="cpu", copy=True)
                    else:
                        value = merge_effective_weight(
                            effective,
                            module.lora_A.weight,
                            module.lora_B.weight,
                            scale=module.lora_scale,
                            dtype=_DTYPES[dtype],
                        )
                finally:
                    del effective
            elif type(module) is nn.Linear:
                value = module.weight.detach().to(device="cpu", dtype=_DTYPES[dtype]).clone()
            else:
                from cosmos_framework.posttrain.model.quantization.bnb import NativeFrozenLinear4bit

                if not isinstance(module, NativeFrozenLinear4bit):
                    raise ValueError(f"unsupported frozen family target implementation: {target}")
                effective = module.effective_weight_for_initialization()
                try:
                    value = effective.to(device="cpu", dtype=_DTYPES[dtype]).clone()
                finally:
                    del effective
        else:
            tensor = parameters[name] if name in parameters else buffers[name][2]
            value = tensor.detach().cpu().clone()
        if list(value.shape) != inventory[name]["shape"] or not torch.isfinite(value).all().item():
            raise ValueError(f"family dense tensor is nonfinite or has the wrong shape: {name}")
        yield name, value.contiguous()
        del value


@torch.no_grad()
def _export_family_artifact(
    net: nn.Module,
    *,
    checkpoint: str,
    output: str,
    merged_backbone_dtype: str,
    exporter_build: Mapping[str, str] | None = None,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
    sidecars: Mapping[str, str] | None = None,
    unmerged: bool = False,
) -> dict[str, Any]:
    """Load a committed trained policy, stream the effective-base merge, commit last.

    CPU working memory is one output shard plus one current matrix and row-merge
    temporary. The resident policy and checkpoint's auxiliary payload remain
    additional costs. GPU dequantization holds one logical matrix at a time.
    """
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    if merged_backbone_dtype not in {"bfloat16", "float32"}:
        raise ValueError("merged_backbone_dtype must be bfloat16 or float32")
    exporter_build = dict(exporter_build or {})
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in exporter_build.items()):
        raise ValueError("exporter_build provenance must map strings to strings")
    root = Path(output).absolute()
    if root.exists():
        raise FileExistsError(root)
    if root.resolve().is_relative_to(Path(checkpoint).resolve()):
        raise ValueError("family dense output must be separate from its checkpoint")
    copied = _sidecar_sources(sidecars)
    if any(root.resolve().is_relative_to(source.parent) for source in copied.values()):
        raise ValueError("family dense output must be separate from its source sidecars")
    before = checkpoint_inventory_signature(checkpoint)
    metadata = read_adapter_metadata(checkpoint)
    identity = metadata["identity"]
    if identity.get("format") != "cosmos.family-backbone.v1" or not identity.get("task_contract"):
        raise ValueError("family dense export requires a complete native family identity")
    heads = head_adapter_targets(identity)
    if heads and not unmerged:
        raise ValueError(
            "action-head adapters support only unmerged export (--unmerged); merged export is not supported"
        )
    codec = getattr(net, "_cosmos_adapter_checkpoint_codec", None)
    if codec is None:
        raise ValueError("family dense export requires the constructed native adapter policy")
    load_adapter_weights(checkpoint, codec, manager=CheckpointManager(checkpoint_format="dcp"))
    trained = {name: tensor_digest(value) for name, value in codec.parameters.items()}
    if unmerged:
        _require_unmerged_identity(identity)
        for name in (*identity["targets"], *heads):
            module = net.get_submodule(name)
            if (
                not isinstance(module, NativeLoRAMixin)
                or module._lora_use_rslora
                or module._lora_rank != identity["adapter"]["rank"]
                or module._lora_alpha != identity["adapter"]["alpha"]
                or module._adapter_compute_dtype not in (None, torch.float32)
                or module.lora_A.weight.dtype != torch.float32
                or module.lora_B.weight.dtype != torch.float32
                or type(module.lora_dropout) not in (nn.Identity, nn.Dropout)
            ):
                raise ValueError("unmerged export requires the original ordinary native FP32-adapter arithmetic")
            if name in heads and (
                type(module) is not LoraInjectedLinear
                or module.weight.dtype != torch.float32
                or module.bias is None
                or module.bias.dtype != torch.float32
                or module.weight.requires_grad
                or module.bias.requires_grad
            ):
                raise ValueError("unmerged head adapters require ordinary frozen FP32 weights and biases")
    inventory, parameters, _, _ = _inventory(net, identity, merged_backbone_dtype, unmerged=unmerged)
    aliases = _parameter_aliases(parameters)
    weight_map = _mixed_shard_plan(inventory, max_shard_size_bytes)
    root.mkdir(parents=False, exist_ok=False)
    fsync_directory(str(root.parent))
    records, total = write_dense_shards(
        root, _dense_values(net, identity, merged_backbone_dtype, unmerged=unmerged), weight_map
    )
    tensors = {name: {**record, **inventory[name]} for name, record in records.items()}
    atomic_write_json(
        str(root / "model.safetensors.index.json"), {"metadata": {"total_size": total}, "weight_map": weight_map}
    )
    for name, source in copied.items():
        expected = file_record(source)
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            while block := reader.read(4 * 1024 * 1024):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if file_record(destination) != expected or file_record(source) != expected:
            raise RuntimeError(f"family source sidecar changed while copying: {name}")
        fsync_directory(str(destination.parent))
    codec.validate_frozen_base()
    if trained != {name: tensor_digest(value) for name, value in codec.parameters.items()}:
        raise RuntimeError("trained family parameters changed during dense export")
    if checkpoint_inventory_signature(checkpoint) != before:
        raise RuntimeError("family source checkpoint changed during dense export")
    manifest = {
        "protocol": FAMILY_UNMERGED_PROTOCOL if unmerged else FAMILY_DENSE_PROTOCOL,
        "complete": True,
        "format": "cosmos_native_family_unmerged_safetensors" if unmerged else "cosmos_native_family_safetensors",
        "source_metadata": metadata,
        "source_metadata_sha256": digest_json(metadata),
        "trained_tensors": trained,
        "parameter_aliases": aliases,
        "numerical_contract": _numerical_contract(identity, merged_backbone_dtype, unmerged=unmerged),
        "resumable_training_checkpoint": False,
        "native_loader_required": True,
        "exporter_build": exporter_build,
        "sidecars": sorted(copied),
        "sharding": {
            "policy": "sorted_greedy_payload_bytes_v1",
            "max_shard_size_bytes": max_shard_size_bytes,
            "oversized_tensor": "alone",
        },
        "tensors": tensors,
        "files": {str(path.relative_to(root)): file_record(path) for path in sorted(root.rglob("*")) if path.is_file()},
    }
    atomic_write_json(str(root / (FAMILY_UNMERGED_MANIFEST if unmerged else FAMILY_DENSE_MANIFEST)), manifest)
    fsync_directory(str(root))
    return _read_family_artifact(root, unmerged=unmerged)


def _read_family_artifact(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_manifest_sha256: str | None = None,
    unmerged: bool = False,
) -> dict[str, Any]:
    """Verify all bytes, inventory and numerical contracts before native construction.

    ``expected_identity`` is an explicit subset of the source family codec identity.
    The family loader must validate its architecture, processor and task contract;
    a self-consistent artifact alone does not select the intended policy.
    """
    from safetensors import safe_open

    root = Path(path)
    manifest_name = FAMILY_UNMERGED_MANIFEST if unmerged else FAMILY_DENSE_MANIFEST
    marker = root / manifest_name
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("family dense export is incomplete: missing regular commit manifest")
    raw = marker.read_bytes()
    if expected_manifest_sha256 is not None and sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("family dense manifest differs from the requested SHA256")
    manifest = json.loads(raw)
    expected_keys = {
        "protocol",
        "complete",
        "format",
        "source_metadata",
        "source_metadata_sha256",
        "trained_tensors",
        "parameter_aliases",
        "numerical_contract",
        "resumable_training_checkpoint",
        "native_loader_required",
        "exporter_build",
        "sharding",
        "tensors",
        "files",
        "sidecars",
    }
    if (
        set(manifest) != expected_keys
        or manifest["protocol"] != (FAMILY_UNMERGED_PROTOCOL if unmerged else FAMILY_DENSE_PROTOCOL)
        or manifest["complete"] is not True
    ):
        raise ValueError("unsupported or incomplete family dense manifest schema")
    metadata = manifest["source_metadata"]
    identity = metadata.get("identity", {})
    numerical = manifest["numerical_contract"]
    if (
        manifest["format"]
        != ("cosmos_native_family_unmerged_safetensors" if unmerged else "cosmos_native_family_safetensors")
        or manifest["resumable_training_checkpoint"] is not False
        or manifest["native_loader_required"] is not True
        or identity.get("format") != "cosmos.family-backbone.v1"
        or not identity.get("task_contract")
        or "quantization" not in identity
        or not isinstance(identity.get("adapter"), dict)
        or not isinstance(identity["adapter"].get("mode"), str)
        or digest_json(metadata) != manifest["source_metadata_sha256"]
        or numerical != _numerical_contract(identity, numerical.get("merged_backbone_dtype"), unmerged=unmerged)
        or (not unmerged and numerical["merged_backbone_dtype"] not in {"bfloat16", "float32"})
    ):
        raise ValueError("family dense source or numerical contract differs")
    if unmerged:
        _require_unmerged_identity(identity)
    if expected_identity is not None:
        require_identity_subset(identity, json.loads(json.dumps(dict(expected_identity), allow_nan=False)))
    files, tensors = manifest["files"], manifest["tensors"]
    if not isinstance(files, dict) or not isinstance(tensors, dict) or not files or not tensors:
        raise ValueError("family dense export requires complete file/tensor inventories")
    if {str(member.relative_to(root)) for member in root.rglob("*") if member.is_file()} != {
        *files,
        manifest_name,
    }:
        raise ValueError("family dense directory membership differs from its committed inventory")
    signatures = _tree_signatures(root)
    for name, record in files.items():
        member = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or str(Path(name)) != name
            or member.is_symlink()
            or not member.is_file()
            or file_record(member) != record
        ):
            raise ValueError(f"family dense file differs from its committed digest: {name}")
    for name, record in tensors.items():
        keys = {"shape", "dtype", "sha256", "file", "kind"}
        if record.get("kind") == "buffer":
            keys.add("persistent")
            if type(record.get("persistent")) is not bool:
                raise ValueError(f"family dense buffer persistence is invalid: {name}")
        elif record.get("kind") != "parameter":
            raise ValueError(f"family dense tensor kind is invalid: {name}")
        if set(record) != keys or not isinstance(name, str) or not name or any(not part for part in name.split(".")):
            raise ValueError(f"family dense tensor schema differs: {name}")
    _validate_source_inventory(manifest, unmerged=unmerged)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = {name: record["file"] for name, record in tensors.items()}
    sharding = manifest["sharding"]
    if sharding != {
        "policy": "sorted_greedy_payload_bytes_v1",
        "max_shard_size_bytes": sharding.get("max_shard_size_bytes"),
        "oversized_tensor": "alone",
    }:
        raise ValueError("family dense sharding policy differs")
    if (
        _mixed_shard_plan(tensors, sharding["max_shard_size_bytes"]) != weight_map
        or index.get("weight_map") != weight_map
    ):
        raise ValueError("family dense index differs from its deterministic tensor inventory")
    sidecars = manifest["sidecars"]
    if (
        not isinstance(sidecars, list)
        or sidecars != sorted(set(sidecars))
        or set(sidecars) & {"model.safetensors.index.json", *weight_map.values()}
    ):
        raise ValueError("family dense sidecar inventory differs")
    if set(files) != {"model.safetensors.index.json", *weight_map.values(), *sidecars}:
        raise ValueError("family dense has missing or unindexed files")
    total = sum(
        math.prod(record["shape"]) * torch.empty((), dtype=_DTYPES[record["dtype"]]).element_size()
        for record in tensors.values()
    )
    if index.get("metadata") != {"total_size": total}:
        raise ValueError("family dense index total size differs")
    groups: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        groups.setdefault(filename, []).append(name)
    for filename, names in groups.items():
        with safe_open(str(root / filename), framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(names):
                raise ValueError("family dense shard membership differs")
            for name in names:
                record, header = tensors[name], reader.get_slice(name)
                if header.get_shape() != record["shape"] or header.get_dtype() != _SAFE_DTYPES[record["dtype"]]:
                    raise ValueError(f"family dense tensor header differs: {name}")
                if tensor_digest(reader.get_tensor(name)) != {key: record[key] for key in ("shape", "dtype", "sha256")}:
                    raise ValueError(f"family dense tensor payload differs: {name}")
    if _tree_signatures(root) != signatures or marker.read_bytes() != raw:
        raise RuntimeError("family dense export changed during verification")
    return manifest


@torch.no_grad()
def _load_family_artifact(
    net: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str,
    expected_identity: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    unmerged: bool = False,
) -> dict[str, Any]:
    """Install a verified dense artifact into its native, unadapted CPU/meta policy.

    No base weights are loaded, no adapter is initialized, and no quantizer runs.
    The family constructor owns non-tensor state and must repair any unregistered
    buffer aliases and preferred layouts after this call. Serialized tensors use
    canonical contiguous values; arbitrary original strides are not an export
    parity promise. Recorded frozen non-target parameter aliases are restored as
    the same Parameter, and the constructor must already declare the same alias
    groups. This does not create a resumable trainer.
    """
    from safetensors import safe_open

    if not expected_identity:
        raise ValueError("native family dense loading requires an explicit expected identity")
    verified = _read_family_artifact(path, expected_identity=expected_identity, unmerged=unmerged)
    if manifest is not None and dict(manifest) != verified:
        raise ValueError("family dense artifact changed after native construction preflight")
    if hasattr(net, "_cosmos_adapter_checkpoint_codec") or any(
        isinstance(module, NativeLoRAMixin) for module in net.modules()
    ):
        raise ValueError("family dense loading requires a fresh unadapted native architecture")
    if unmerged:
        _install_unmerged_graph(net, verified)
    parameters = dict(net.named_parameters(remove_duplicate=False))
    buffers = _buffers(net)
    if _parameter_aliases(parameters) != verified["parameter_aliases"]:
        raise ValueError("native family parameter alias ownership differs from dense artifact")
    inventory = verified["tensors"]
    expected_parameters = {name for name, record in inventory.items() if record["kind"] == "parameter"}
    expected_buffers = set(inventory) - expected_parameters
    if set(parameters) != expected_parameters or set(buffers) != expected_buffers:
        raise ValueError("native family architecture parameter/buffer inventory differs from dense artifact")
    for name, value in parameters.items():
        if value.device.type not in {"cpu", "meta"} or list(value.shape) != inventory[name]["shape"]:
            raise ValueError(f"native dense parameter must have matching CPU/meta shape: {name}")
    for name, (_, _, value, persistent) in buffers.items():
        if (
            value.device.type not in {"cpu", "meta"}
            or list(value.shape) != inventory[name]["shape"]
            or persistent != inventory[name]["persistent"]
        ):
            raise ValueError(f"native dense buffer shape/persistence differs: {name}")
    root = Path(path)
    signatures = _tree_signatures(root)
    canonical = {name: group[0] for group in verified["parameter_aliases"] for name in group}
    restored_aliases: dict[str, nn.Parameter] = {}
    for name, record in sorted(inventory.items()):
        with safe_open(str(root / record["file"]), framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name)
        if tensor_digest(value) != {key: record[key] for key in ("shape", "dtype", "sha256")}:
            raise ValueError(f"family dense tensor changed during load: {name}")
        if name in parameters:
            owner_name = canonical.get(name, name)
            if owner_name in restored_aliases:
                parameter = restored_aliases[owner_name]
            else:
                parameter = nn.Parameter(value.to(device=device, copy=True), requires_grad=False)
                if name in canonical:
                    restored_aliases[owner_name] = parameter
            parent, _, leaf = name.rpartition(".")
            setattr(net.get_submodule(parent), leaf, parameter)
        else:
            owner, leaf, _, _ = buffers[name]
            setattr(owner, leaf, value.to(device=device, copy=True))
    if _tree_signatures(root) != signatures:
        raise RuntimeError("family dense artifact changed during native load")
    net.eval()
    if unmerged:
        net._cosmos_unmerged_family_identity = verified["source_metadata"]["identity"]
    else:
        net._cosmos_dense_family_identity = verified["source_metadata"]["identity"]
    return verified


def _require_unmerged_identity(identity):
    head_adapter_targets(identity)
    adapter = identity.get("adapter", {})
    if (
        adapter.get("mode") not in {"nf4_lora", "bf16_lora"}
        or type(adapter.get("rank")) is not int
        or adapter["rank"] < 1
        or type(adapter.get("alpha")) not in (int, float)
        or not math.isfinite(adapter["alpha"])
        or adapter["alpha"] <= 0
    ):
        raise ValueError("unmerged deployment requires an ordinary native LoRA family identity")
    expected_quantization = (
        {
            "backend": "bitsandbytes",
            "format": "nf4",
            "block_size": 64,
            "nested": True,
            "storage_dtype": "uint8",
            "compute_dtype": "bfloat16",
        }
        if adapter["mode"] == "nf4_lora"
        else None
    )
    if identity.get("quantization") != expected_quantization or identity.get("precision") != {
        "target_source_cast": "bfloat16",
        "adapter": "float32",
        "auxiliary": "float32",
    }:
        raise ValueError("unmerged deployment requires the recorded NF4/BF16 base and FP32 adapter/head policy")


def _numerical_contract(identity, merged_backbone_dtype, *, unmerged=False):
    heads = head_adapter_targets(identity)
    if heads and not unmerged:
        raise ValueError(
            "action-head adapters support only unmerged export (--unmerged); merged export is not supported"
        )
    if unmerged:
        result = {
            "base": "dequantize(Q(W))" if identity.get("quantization") else "W_bf16",
            "representation": "dequantized_unmerged_native_lora",
            "effective_backbone_dtype": "bfloat16",
            "adapter_parameter_dtype": "float32",
            "adapter_compute": "native_separate_gemms_ambient_autocast",
            "scaling": "alpha/rank",
            "addition": "base_output + (alpha/rank) * adapter_output.to(base_output.dtype)",
            "other_parameters_and_buffers": "preserve_native_dtype",
            "merged": False,
            "requantized": False,
        }
        if heads:
            result["head_adapters"] = {
                key: identity["head_adapters"][key]
                for key in ("profile", "base_dtype", "bias", "parameter_dtype", "compute")
            }
        return result
    return {
        "base": "dequantize(Q(W))" if identity.get("quantization") else "W_bf16",
        "merge": "effective_base + (alpha/rank) * B @ A"
        if identity.get("adapter", {}).get("mode", "").endswith("_lora")
        else "effective_base",
        "accumulation_dtype": "float32",
        "merge_device": "cpu",
        "merged_backbone_dtype": merged_backbone_dtype,
        "other_parameters_and_buffers": "preserve_native_dtype",
        "requantized": False,
    }


def _install_unmerged_graph(net, manifest):
    """Build only meta adapter structure, consuming no RNG and reading no base weights."""
    identity = manifest["source_metadata"]["identity"]
    heads = head_adapter_targets(identity)
    adapter = identity["adapter"]
    parameters = dict(net.named_parameters(remove_duplicate=False))
    aliases = {name for group in _parameter_aliases(parameters) for name in group}
    for name, shape in {**identity["targets"], **heads}.items():
        old = net.get_submodule(name)
        head = name in heads
        if (
            type(old) is not nn.Linear
            or (old.bias is None if head else old.bias is not None)
            or list(old.weight.shape) != shape
            or old.weight.device.type not in {"cpu", "meta"}
            or f"{name}.weight" in aliases
            or (
                head
                and (old.weight.dtype != torch.float32 or old.bias.dtype != torch.float32 or f"{name}.bias" in aliases)
            )
        ):
            raise ValueError(f"unmerged native target requires its declared unaliased CPU/meta Linear: {name}")
    for name in (*identity["targets"], *heads):
        old = net.get_submodule(name)
        head = name in heads
        meta = nn.Linear(
            old.in_features, old.out_features, bias=head, device="meta", dtype=torch.float32 if head else torch.bfloat16
        )
        # Inference-only artifact: training dropout is not part of the model identity.
        replacement = LoraInjectedLinear(meta, adapter["rank"], adapter["alpha"], adapter_dtype=torch.float32)
        replacement.train(old.training)
        parent, _, leaf = name.rpartition(".")
        net.get_submodule(parent).register_module(leaf, replacement)


def export_family_dense(
    net: nn.Module,
    *,
    checkpoint: str,
    output: str,
    merged_backbone_dtype: str,
    exporter_build: Mapping[str, str] | None = None,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
    sidecars: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Publish the merged family dense format; ``exporter_build`` is optional provenance recorded as given."""
    return _export_family_artifact(
        net,
        checkpoint=checkpoint,
        output=output,
        merged_backbone_dtype=merged_backbone_dtype,
        exporter_build=exporter_build,
        max_shard_size_bytes=max_shard_size_bytes,
        sidecars=sidecars,
    )


def export_family_unmerged(
    net: nn.Module,
    *,
    checkpoint: str,
    output: str,
    exporter_build: Mapping[str, str] | None = None,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
    sidecars: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Stream actual BF16 effective bases and exact FP32 factors/heads; no merge or requantization.

    This is an inference-only dense-base native artifact, not four-bit inference.
    GPU temporary storage is one dequantized matrix; CPU publication holds one
    output shard plus the current tensor, in addition to the resident source policy.
    """
    return _export_family_artifact(
        net,
        checkpoint=checkpoint,
        output=output,
        merged_backbone_dtype="bfloat16",
        exporter_build=exporter_build,
        max_shard_size_bytes=max_shard_size_bytes,
        sidecars=sidecars,
        unmerged=True,
    )


def read_family_dense(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return _read_family_artifact(
        path, expected_identity=expected_identity, expected_manifest_sha256=expected_manifest_sha256
    )


def read_family_unmerged(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return _read_family_artifact(
        path, expected_identity=expected_identity, expected_manifest_sha256=expected_manifest_sha256, unmerged=True
    )


def load_family_dense(
    net: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str,
    expected_identity: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _load_family_artifact(net, path, device=device, expected_identity=expected_identity, manifest=manifest)


def load_family_unmerged(
    net: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str,
    expected_identity: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct the native separate LoRA arithmetic, frozen for inference only.

    No original checkpoint, quantizer, adapter initialization, trainer codec or
    optimizer is used.
    """
    return _load_family_artifact(
        net, path, device=device, expected_identity=expected_identity, manifest=manifest, unmerged=True
    )
