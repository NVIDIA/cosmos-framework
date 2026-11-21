# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit, bounded effective-base + adapter export to dense Hugging Face weights.

This is an offline numerical conversion, never an in-place merge or a training
checkpoint. NF4 is dequantized by the attached bitsandbytes backend on its owned
CUDA device; one matrix is copied to CPU and merged in FP32 at a time. Completed
output tensors are grouped into size-bounded CPU shards. Floating-point evaluation
after merging need not be bitwise equal to the separate base/adapter operations.
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

from cosmos_framework.posttrain.checkpoint.adapter import _require_subset, load_adapter_weights, read_adapter_metadata
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json, fsync_directory, release_page_cache
from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_inventory_signature
from cosmos_framework.posttrain.model.quantization.loader import _safetensor_inventory, tensor_digest
from cosmos_framework.posttrain.utils.adapter_source import verify_adapter_snapshot
from cosmos_framework.utils.generator.lora import LoraInjectedLinear, NativeLoRAMixin

DENSE_EXPORT_MANIFEST = "cosmos_dense_export.json"
DENSE_EXPORT_PROTOCOL = "cosmos.native-adapter-dense-export.v2"
_LEGACY_DENSE_EXPORT_PROTOCOL = "cosmos.native-adapter-dense-export.v1"
DEFAULT_MAX_SHARD_SIZE_BYTES = 512 * 1024**2
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _digest_json(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    digest = sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def _directory_signatures(root: Path) -> dict[str, tuple[int, ...]]:
    return {
        p.name: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        for p in root.iterdir()
        for s in (p.stat(),)
    }


def _shard_plan(shapes: Mapping[str, list[int] | tuple[int, ...]], *, dtype: str, max_bytes: int) -> dict[str, str]:
    """Greedy, name-ordered HF packing, without materializing output tensors.

    The bound counts tensor payload bytes, excluding safetensors headers. A tensor
    larger than the bound occupies a shard alone; tensors are never split. This
    matches standard HF loading, including its support for indexed single shards.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be a positive integer")
    element_size = torch.empty((), dtype=_DTYPES[dtype]).element_size()
    groups: list[list[str]] = []
    names: list[str] = []
    size = 0
    for name, shape in sorted(shapes.items()):
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError(f"invalid dense export shape: {name}")
        nbytes = math.prod(shape) * element_size
        if names and size + nbytes > max_bytes:
            groups.append(names)
            names, size = [], 0
        names.append(name)
        size += nbytes
        if size >= max_bytes:
            groups.append(names)
            names, size = [], 0
    if names:
        groups.append(names)
    return {
        name: f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        for number, group in enumerate(groups, start=1)
        for name in group
    }


def _write_dense_shards(
    root: Path, values: Iterator[tuple[str, torch.Tensor]], weight_map: dict[str, str]
) -> tuple[dict[str, Any], int]:
    """Hold one planned CPU shard plus the current produced tensor, never the model.

    The generator's current matrix and row-merge temporaries are additional to
    retained shard tensors. A single oversized matrix is serialized alone.
    """
    from safetensors.torch import save_file

    pending: dict[str, torch.Tensor] = {}
    current: str | None = None
    tensors: dict[str, Any] = {}
    total = 0

    def flush() -> None:
        if current is None:
            return
        path = root / current
        save_file(pending, str(path), metadata={"format": "pt"})
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        release_page_cache(str(path))
        pending.clear()

    expected = iter(weight_map)
    for name, value in values:
        if name != next(expected, None):
            raise ValueError("dense export tensor order/membership differs from its shard plan")
        filename = weight_map[name]
        if filename != current:
            flush()
            current = filename
        pending[name] = value.contiguous()
        tensors[name] = {**tensor_digest(value), "file": filename}
        total += value.numel() * value.element_size()
        del value
    if next(expected, None) is not None or not tensors:
        raise ValueError("dense export is missing planned tensors")
    flush()
    return tensors, total


def _canonical_parameters(net: nn.Module) -> dict[str, nn.Parameter]:
    from cosmos_framework.posttrain.model.qwen3_vl import canonical_qwen3_vl_weight_name

    result = {}
    for raw, parameter in net.named_parameters(remove_duplicate=False):
        name = canonical_qwen3_vl_weight_name(raw)
        if name in result:
            raise ValueError(f"duplicate dense export parameter: {name}")
        result[name] = parameter
    return result


def _adapter_modules(net: nn.Module) -> dict[str, NativeLoRAMixin]:
    from cosmos_framework.posttrain.model.qwen3_vl import canonical_qwen3_vl_weight_name

    result = {}
    for raw, module in net.named_modules(remove_duplicate=False):
        if isinstance(module, NativeLoRAMixin):
            name = canonical_qwen3_vl_weight_name(raw)
            if name in result:
                raise ValueError(f"duplicate dense export adapter: {name}")
            result[name] = module
    return result


@torch.no_grad()
def merge_effective_weight(
    base: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, *, scale: float, dtype: torch.dtype
) -> torch.Tensor:
    """FP32 CPU merge in row blocks; cast only the completed sum, never its delta."""
    if dtype not in _DTYPES.values() or not math.isfinite(scale):
        raise ValueError("dense export requires a finite scale and BF16 or FP32 output")
    if (
        base.ndim != 2
        or lora_a.ndim != 2
        or lora_b.ndim != 2
        or lora_a.shape[1] != base.shape[1]
        or lora_b.shape != (base.shape[0], lora_a.shape[0])
        or base.is_meta
        or lora_a.is_meta
        or lora_b.is_meta
        or base.dtype not in (torch.bfloat16, torch.float32)
        or lora_a.dtype != torch.float32
        or lora_b.dtype != torch.float32
    ):
        raise ValueError("effective weight and FP32 adapter shapes/dtypes are incompatible")
    base = base.detach().to(device="cpu")
    lora_a, lora_b = lora_a.detach().cpu(), lora_b.detach().cpu()
    if not all(torch.isfinite(t).all().item() for t in (base, lora_a, lora_b)):
        raise ValueError("dense export refuses nonfinite base or adapter tensors")
    output = torch.empty(base.shape, dtype=dtype, device="cpu")
    with torch.autocast(device_type="cpu", enabled=False):
        for begin in range(0, base.shape[0], 256):
            end = min(begin + 256, base.shape[0])
            merged = base[begin:end].float() + scale * (lora_b[begin:end] @ lora_a)
            output[begin:end].copy_(merged)
    if not torch.isfinite(output).all().item():
        raise ValueError("dense merge produced nonfinite output")
    return output


def _effective_weight(module: NativeLoRAMixin) -> torch.Tensor:
    if isinstance(module, LoraInjectedLinear):
        return module.weight.detach()
    # Importing bitsandbytes is restricted to the actual NF4 path. CPU tests and
    # dense adapters cannot silently substitute a different quantization backend.
    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit

    if not isinstance(module, NativeLoraLinear4bit):
        raise ValueError("unsupported native adapter implementation for dense export")
    return module.effective_weight_for_initialization()


def _dense_tensors(
    net: nn.Module, identity: dict[str, Any], source: Path, dtype: torch.dtype
) -> Iterator[tuple[str, torch.Tensor]]:
    parameters, adapters = _canonical_parameters(net), _adapter_modules(net)
    inventory = _safetensor_inventory(source)
    adapter_names = {f"{name}.lora_{factor}.weight" for name in adapters for factor in ("A", "B")}
    if set(adapters) != set(identity["targets"]) or adapter_names != set(net._cosmos_adapter_parameters):
        raise ValueError("dense export adapter target membership differs from the bound identity")
    if set(parameters) - adapter_names != set(inventory):
        raise ValueError("dense export model inventory differs from its immutable base")
    if len({id(parameter) for parameter in parameters.values()}) != len(parameters):
        raise ValueError("dense export does not support shared or tied parameter aliases")
    for name, (_, shape, _) in sorted(inventory.items()):
        target = name.removesuffix(".weight") if name.endswith(".weight") else None
        if target in adapters:
            module = adapters[target]
            if tuple(identity["targets"][target]) != shape:
                raise ValueError(f"dense export logical target shape differs: {name}")
            effective = _effective_weight(module)
            try:
                value = merge_effective_weight(
                    effective, module.lora_A.weight, module.lora_B.weight, scale=module._lora_scale, dtype=dtype
                )
            finally:
                del effective
        else:
            value = parameters[name].detach().to(device="cpu", dtype=dtype).contiguous()
        if tuple(value.shape) != shape or not torch.isfinite(value).all().item():
            raise ValueError(f"dense export tensor shape or finiteness differs: {name}")
        yield name, value
        del value


def read_dense_export(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the completed HF artifact before passing it to an ordinary loader.

    Callers can pin the whole manifest or an explicit subset of its fields, e.g.
    ``base_identity``, ``processor_identity``, and ``adapter_tensors`` from the
    independently accepted source. Self-consistency alone does not select a model.
    """
    from safetensors import safe_open

    root = Path(path)
    marker = root / DENSE_EXPORT_MANIFEST
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("dense export is incomplete: missing regular commit manifest")
    marker_bytes = marker.read_bytes()
    if expected_manifest_sha256 is not None and sha256(marker_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("dense export manifest differs from the requested SHA256")
    manifest = json.loads(marker_bytes)
    if (
        manifest.get("protocol") not in (DENSE_EXPORT_PROTOCOL, _LEGACY_DENSE_EXPORT_PROTOCOL)
        or manifest.get("complete") is not True
    ):
        raise ValueError("unsupported or incomplete dense export manifest")
    if (
        manifest.get("format") != "huggingface_safetensors"
        or manifest.get("output_dtype") not in _DTYPES
        or manifest.get("accumulation_dtype") != "float32"
        or manifest.get("merge_device") != "cpu"
        or manifest.get("resumable_training_checkpoint") is not False
        or _digest_json(manifest["source_metadata"]) != manifest.get("source_metadata_sha256")
        or manifest["source_metadata"]["identity"]["base"] != manifest.get("base_identity")
        or manifest["source_metadata"]["identity"]["processor_contract"] != manifest.get("processor_identity")
    ):
        raise ValueError("dense export numerical/source contract differs")
    if expected_identity is not None:
        _require_subset(manifest, json.loads(json.dumps(dict(expected_identity), allow_nan=False)), "dense_export")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("dense export requires a committed file inventory")
    if {p.name for p in root.iterdir()} != {*files, DENSE_EXPORT_MANIFEST}:
        raise ValueError("dense export directory membership differs from the commit manifest")
    signatures = _directory_signatures(root)
    for name, expected in files.items():
        member = root / name
        if Path(name).name != name or member.is_symlink() or not member.is_file() or _file_record(member) != expected:
            raise ValueError(f"dense export file differs from its committed digest: {name}")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if config.get("torch_dtype") != manifest["output_dtype"] or any(
        c.get("tie_word_embeddings") or "quantization_config" in c for c in (config, config.get("text_config", {}))
    ):
        raise ValueError("dense export config requests incompatible dtype, tying or quantization")
    tensors = manifest["tensors"]
    if index["weight_map"] != {name: record["file"] for name, record in tensors.items()}:
        raise ValueError("dense export index differs from its committed tensor inventory")
    if set(index["weight_map"].values()) != {name for name in files if name.endswith(".safetensors")}:
        raise ValueError("dense export contains missing or unindexed shards")
    shapes = {name: record["shape"] for name, record in tensors.items()}
    if manifest["protocol"] == DENSE_EXPORT_PROTOCOL:
        sharding = manifest.get("sharding", {})
        if sharding.get("policy") != "sorted_greedy_payload_bytes_v1" or sharding.get("oversized_tensor") != "alone":
            raise ValueError("dense export requires its declared bounded shard policy")
        planned = _shard_plan(shapes, dtype=manifest["output_dtype"], max_bytes=sharding.get("max_shard_size_bytes"))
        if planned != index["weight_map"]:
            raise ValueError("dense export index differs from its deterministic shard plan")
    elif len(set(index["weight_map"].values())) != len(tensors):
        raise ValueError("legacy dense export requires one tensor per shard")
    total = (
        sum(math.prod(shape) for shape in shapes.values())
        * torch.empty((), dtype=_DTYPES[manifest["output_dtype"]]).element_size()
    )
    if index.get("metadata", {}).get("total_size") != total:
        raise ValueError("dense export index total_size differs from its tensor inventory")
    shards: dict[str, set[str]] = {}
    for name, record in tensors.items():
        if record["dtype"] != manifest["output_dtype"]:
            raise ValueError("dense export tensor dtype differs from its declared output")
        shards.setdefault(record["file"], set()).add(name)
    for filename, names in shards.items():
        with safe_open(root / filename, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != names:
                raise ValueError("dense export shard membership differs from its tensor inventory")
            for name in sorted(names):
                record = tensors[name]
                value = reader.get_slice(name)
                if (
                    value.get_shape() != record["shape"]
                    or value.get_dtype() != {"bfloat16": "BF16", "float32": "F32"}[record["dtype"]]
                ):
                    raise ValueError(f"dense export tensor header differs: {name}")
                if tensor_digest(reader.get_tensor(name)) != {k: v for k, v in record.items() if k != "file"}:
                    raise ValueError(f"dense export tensor bytes differ: {name}")
    if _directory_signatures(root) != signatures or marker.read_bytes() != marker_bytes:
        raise RuntimeError("dense export changed during verification")
    return manifest


@torch.no_grad()
def export_dense_adapter(
    net: nn.Module,
    *,
    checkpoint: str,
    source: str,
    output: str,
    output_dtype: str,
    exporter_build: dict[str, str],
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
) -> dict[str, Any]:
    """Reload a committed adapter on a fresh native model and publish a new HF directory.

    Output creation is exclusive. Files and the index are durable before the
    commit-last manifest; a failed export remains visibly incomplete and must use
    a new output path. No fallback overwrites or blesses an interrupted artifact.
    """
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    if output_dtype not in _DTYPES:
        raise ValueError("output_dtype must be bfloat16 or float32")
    if type(max_shard_size_bytes) is not int or max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be a positive integer")
    if set(exporter_build) != {"code_revision", "wheel_sha256"} or any(
        re.fullmatch(r"[0-9a-f]{%d}" % size, exporter_build[key]) is None
        for key, size in (("code_revision", 40), ("wheel_sha256", 64))
    ):
        raise ValueError("dense export requires the exact exporter revision and wheel SHA256")
    root, base = Path(output).absolute(), Path(source).resolve()
    if root.exists():
        raise FileExistsError(root)
    if root.resolve().is_relative_to(base) or root.resolve().is_relative_to(Path(checkpoint).resolve()):
        raise ValueError("dense export output must be separate from source and checkpoint")
    source_identity = verify_adapter_snapshot(base)
    source_config = json.loads((base / "config.json").read_text())
    if source_config.get("tie_word_embeddings") or source_config.get("text_config", {}).get("tie_word_embeddings"):
        raise ValueError("dense adapter export does not support tied embedding/output weights")
    if any("quantization_config" in config for config in (source_config, source_config.get("text_config", {}))):
        raise ValueError("dense adapter export requires a dense base config without quantization metadata")
    before = checkpoint_inventory_signature(checkpoint)
    saved = read_adapter_metadata(checkpoint)
    identity = saved["identity"]
    if identity.get("model_format") != "cosmos_native_qwen3_vl" or identity.get("base") != source_identity:
        raise ValueError("dense export requires the exact native Qwen3-VL immutable base identity")
    if not identity.get("processor_contract"):
        raise ValueError("dense export requires a bound processor identity")
    codec = net._cosmos_adapter_checkpoint_codec
    load_adapter_weights(checkpoint, codec, manager=CheckpointManager(checkpoint_format="dcp"))
    adapter_digests = {name: tensor_digest(p) for name, p in codec.parameters.items()}
    weight_map = _shard_plan(
        {name: shape for name, (_, shape, _) in _safetensor_inventory(base).items()},
        dtype=output_dtype,
        max_bytes=max_shard_size_bytes,
    )
    root.mkdir(parents=False, exist_ok=False)
    fsync_directory(str(root.parent))
    tensors, total = _write_dense_shards(root, _dense_tensors(net, identity, base, _DTYPES[output_dtype]), weight_map)
    for name in source_identity["files"]:
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        data = (base / name).read_bytes()
        if name == "config.json":
            config = json.loads(data)
            # Explicit HF loading dtype; all other architecture/processor semantics stay exact.
            config["torch_dtype"] = output_dtype
            if "dtype" in config:
                config["dtype"] = output_dtype
            for child in ("text_config", "vision_config"):
                if isinstance(config.get(child), dict):
                    for key in ("torch_dtype", "dtype"):
                        if key in config[child]:
                            config[child][key] = output_dtype
            data = (json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with (root / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    atomic_write_json(
        str(root / "model.safetensors.index.json"),
        {"metadata": {"total_size": total}, "weight_map": {name: record["file"] for name, record in tensors.items()}},
    )
    codec.validate_frozen_base()
    if adapter_digests != {name: tensor_digest(p) for name, p in codec.parameters.items()}:
        raise RuntimeError("adapter tensors changed during dense export")
    if before != checkpoint_inventory_signature(checkpoint) or read_adapter_metadata(checkpoint) != saved:
        raise RuntimeError("source adapter checkpoint changed during dense export")
    if verify_adapter_snapshot(base) != source_identity:
        raise RuntimeError("immutable base or processor files changed during dense export")
    manifest = {
        "protocol": DENSE_EXPORT_PROTOCOL,
        "complete": True,
        "format": "huggingface_safetensors",
        "resumable_training_checkpoint": False,
        "output_dtype": output_dtype,
        "accumulation_dtype": "float32",
        "merge_device": "cpu",
        "sharding": {
            "policy": "sorted_greedy_payload_bytes_v1",
            "max_shard_size_bytes": max_shard_size_bytes,
            "oversized_tensor": "alone",
        },
        "effective_weight": "dequantize_attached_bnb_nf4_on_cuda_plus_scaled_BA"
        if identity.get("quantization")
        else "dense_base_plus_scaled_BA",
        "merge_scope": "One effective matrix at a time; FP32 row-block merge, then one output cast. No requantization or in-place model mutation. Separate-branch evaluation is not claimed bitwise equivalent.",
        "source_metadata": saved,
        "source_metadata_sha256": _digest_json(saved),
        "source_checkpoint_signature": before,
        "adapter_tensors": adapter_digests,
        "base_identity": source_identity,
        "processor_identity": identity["processor_contract"],
        "exporter_build": dict(exporter_build),
        "tensors": tensors,
        "files": {p.name: _file_record(p) for p in sorted(root.iterdir())},
    }
    fsync_directory(str(root))
    atomic_write_json(str(root / DENSE_EXPORT_MANIFEST), manifest)
    return read_dense_export(root)
