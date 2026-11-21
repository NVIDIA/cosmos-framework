# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verify the immutable local source shared by native adapters and the SFT processor.

Preparation downloads the exact Hub revision outside training. Training only reads a
complete, content-verified local snapshot; a config revision is never a substitute
for that verification. This module deliberately imports no accelerator libraries.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from cosmos_framework.posttrain.utils.artifact_identity import _sha256

SOURCE_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
SOURCE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SOURCE_MANIFEST = "cosmos_adapter_source.json"
SOURCE_PROTOCOL = "cosmos.adapter-source.v1"
_VERIFIED: dict[tuple[str, tuple[tuple[str, int, int, int, int, int], ...]], dict[str, Any]] = {}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_files(root: Path) -> list[Path]:
    # HF sidecars are all identity-bearing, including processor and chat templates.
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name != SOURCE_MANIFEST
        and path.suffix in {".json", ".jinja", ".txt", ".model", ".safetensors"}
    )


def _require_complete(root: Path, names: set[str]) -> None:
    required = {"config.json", "tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"}
    missing = required - names
    if missing:
        raise ValueError(f"incomplete adapter source: missing {sorted(missing)}")
    if "tokenizer.json" not in names and not {"vocab.json", "merges.txt"} <= names:
        raise ValueError("adapter source requires tokenizer.json or vocab.json and merges.txt")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("adapter source must be the supported dense qwen3_vl model")
    text = config.get("text_config", {})
    expected = {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288}
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("adapter source architecture differs from the pinned Qwen3-VL-8B profile")
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("adapter source has an empty safetensors weight index")
        if any(
            not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors")
            for name in weight_map.values()
        ):
            raise ValueError("adapter source index contains an invalid shard path")
        shards = set(weight_map.values())
        if not shards <= names:
            raise ValueError(f"incomplete adapter source shards: {sorted(shards - names)}")
        if {name for name in names if name.endswith(".safetensors")} != shards:
            raise ValueError("adapter source contains unindexed safetensors shards")
    elif "model.safetensors" not in names:
        raise ValueError("adapter source requires safetensors weights and a complete shard index")


def _identity(root: Path) -> dict[str, Any]:
    paths = _source_files(root)
    _require_complete(root, {path.name for path in paths})
    files = {}
    for path in paths:
        size, digest = _sha256(path)
        files[path.name] = {"size": size, "sha256": digest}
    weights = {name: value for name, value in files.items() if name.endswith(".safetensors")}
    processor = {
        name: value
        for name, value in files.items()
        if name not in {"config.json", "generation_config.json", "model.safetensors.index.json"}
        and not name.endswith(".safetensors")
    }
    return {
        "protocol": SOURCE_PROTOCOL,
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "files": files,
        "weights_sha256": _digest(weights),
        "processor_sha256": _digest(processor),
        "config_sha256": files["config.json"]["sha256"],
    }


def prepare_adapter_snapshot(destination: str) -> dict[str, Any]:
    """Download only the pinned public source, then atomically publish its identity.

    This is an explicit CPU preparation operation, never called by model loading.
    Existing manifests are verified instead of being overwritten to bless changes.
    Without a manifest, the destination must be new/empty except for a real HF
    download-cache directory. Stale or interrupted model contents require a fresh
    destination; they must never become part of a newly blessed source identity.
    """
    root = Path(destination).expanduser().resolve()
    if (root / SOURCE_MANIFEST).exists():
        return verify_adapter_snapshot(root)
    root.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name != ".cache" or not path.is_dir() or path.is_symlink()
    )
    if unexpected:
        raise ValueError(
            "refusing unmanifested adapter source contents; use a fresh destination instead of "
            f"blessing stale or interrupted files: {unexpected}"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        local_dir=str(root),
        force_download=True,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.safetensors"],
    )
    identity = _identity(root)
    temporary = root / f".{SOURCE_MANIFEST}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / SOURCE_MANIFEST)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_adapter_snapshot(root)


def verify_adapter_snapshot(snapshot: str | Path) -> dict[str, Any]:
    """Verify a completed local source; reject changes and never access the network."""
    root = Path(snapshot).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("native adapters require a prepared local immutable snapshot")
    manifest = root / SOURCE_MANIFEST
    if not manifest.is_file():
        raise ValueError(f"missing {SOURCE_MANIFEST}; run the explicit QLoRA source preparation command first")
    expected = json.loads(manifest.read_text())
    if not isinstance(expected, dict) or expected.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("unsupported adapter source identity protocol")
    if expected.get("repository") != SOURCE_REPOSITORY or expected.get("revision") != SOURCE_REVISION:
        raise ValueError("adapter source repository/revision differs from the qualified candidate")
    files = expected.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("adapter source identity has no file inventory")
    for name, record in files.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(record, dict):
            raise ValueError("invalid adapter source file inventory")
        if set(record) != {"size", "sha256"} or type(record["size"]) is not int or record["size"] < 0:
            raise ValueError("invalid adapter source file descriptor")
        if not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("invalid adapter source content digest")
    paths = [manifest, *_source_files(root)]
    if {path.name for path in paths[1:]} != set(files):
        raise ValueError("adapter source file inventory changed")
    stamps = []
    for path in paths:
        state = path.stat()
        stamps.append((path.name, state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns))
    key = (str(root), tuple(stamps))
    if key not in _VERIFIED:
        actual = _identity(root)
        if actual != expected:
            raise ValueError("adapter source contents differ from the recorded immutable snapshot")
        _VERIFIED[key] = actual
    return copy.deepcopy(_VERIFIED[key])


def validate_adapter_source(cfg: Any) -> dict[str, Any]:
    """Native-builder gate, called before allocation with its injected local source."""
    if getattr(cfg, "revision", None) != SOURCE_REVISION:
        raise ValueError("native adapter training requires the exact pinned base revision")
    source = str(getattr(cfg, "name_or_path", ""))
    checkpoint = str(getattr(cfg, "checkpoint_path", "") or source)
    if Path(checkpoint).expanduser().resolve() != Path(source).expanduser().resolve():
        raise ValueError("native adapter checkpoint and processor sources must be identical")
    actual = verify_adapter_snapshot(source)
    declared = dict(getattr(cfg, "source_identity", {}) or {})
    for name, value in declared.items():
        if name not in actual or value != actual[name]:
            raise ValueError(f"adapter source identity mismatch: {name}")
    return actual


def prepare_sft_adapter_source(spec: Any) -> dict[str, Any] | None:
    """Bind the existing native spec and processor to one verified local snapshot."""
    cfg = spec.model_config
    if getattr(cfg, "adapter", None) is None and getattr(cfg, "quantized_base", None) is None:
        return None
    source = str(spec.model_id)
    configured = str(getattr(cfg, "name_or_path", "") or source)
    if Path(configured).expanduser().resolve() != Path(source).expanduser().resolve():
        raise ValueError("adapter native policy and model config disagree on source")
    if getattr(cfg, "revision", None) != SOURCE_REVISION:
        raise ValueError("native adapter training requires the exact pinned base revision")
    checkpoint = str(getattr(cfg, "checkpoint_path", "") or source)
    if Path(checkpoint).expanduser().resolve() != Path(source).expanduser().resolve():
        raise ValueError("native adapter checkpoint and processor sources must be identical")
    identity = verify_adapter_snapshot(source)
    for name, value in dict(getattr(cfg, "source_identity", {}) or {}).items():
        if name not in identity or identity[name] != value:
            raise ValueError(f"adapter source identity mismatch: {name}")
    cfg.source_identity = identity
    return identity
