# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint resolution for Cosmos Framework action entrypoints.

Cosmos Framework training writes PyTorch Distributed Checkpoint (DCP)
directories.  The inference-facing actions consume a self-contained Hugging
Face export.  This module bridges the two formats with the appropriate
Framework exporter and coordinates that export when an action is launched with
``torchrun``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)


def distributed_invocation_id() -> str:
    """Identify one launch/restart, shared by independent evaluation replicas.

    Direct launchers must provide a fresh COSMOS_RUN_ID for each invocation.
    torchrun's rendezvous identity plus restart generation is already shared.
    """
    run_id = os.environ.get("COSMOS_RUN_ID") or os.environ.get("TORCHELASTIC_RUN_ID")
    if not run_id or run_id == "none":
        raise ValueError("Distributed evaluation requires a fresh COSMOS_RUN_ID or torchrun rendezvous identity")
    identity = [run_id, os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"), os.environ.get("SLURM_RESTART_COUNT", "0")]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def distributed_identity() -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` without creating a process group."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(rank))))
    return rank, max(1, world_size), local_rank


def is_hf_checkpoint(path: str | Path) -> bool:
    """Return whether *path* contains a loadable HF-style model export."""
    root = Path(path)
    if not (root / "config.json").is_file():
        return False
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            return bool(shards) and all(
                (root / shard).resolve().is_relative_to(root.resolve())
                and (root / shard).is_file()
                and (root / shard).stat().st_size > 0
                for shard in shards
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False
    weights = root / "model.safetensors"
    return weights.is_file() and weights.stat().st_size > 0


def is_dcp_checkpoint(path: str | Path) -> bool:
    root = Path(path)
    return root.is_dir() and ((root / ".metadata").is_file() or any(root.rglob("*.distcp")))


def is_lora_adapter(path: str | Path) -> bool:
    """Return whether *path* is an unmerged PEFT/LoRA adapter checkpoint."""
    root = Path(path)
    return (
        root.is_dir()
        and (root / "adapter_config.json").is_file()
        and ((root / "adapter_model.safetensors").is_file() or (root / "adapter_model.bin").is_file())
    )


def _source_fingerprint(path: Path) -> str:
    """Hash export inputs, excluding separate Framework optimizer/trainer state."""
    if (path / "model" / ".metadata").is_file():
        path = path / "model"
    digest = hashlib.sha256()
    included = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        name = candidate.name
        if not (
            name.endswith((".json", ".yaml", ".jinja", ".txt", ".model", ".safetensors", ".bin", ".distcp"))
            or name in {"merges.txt", "vocab.json", ".metadata"}
        ):
            continue
        rel = candidate.relative_to(path).as_posix()
        included.append(rel)
        digest.update(rel.encode("utf-8") + b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    if not included:
        raise ValueError(f"No fingerprintable model files found in {path}")
    return digest.hexdigest()


def _infer_config_file(checkpoint_path: Path) -> Path:
    # Normal framework layout: <run>/checkpoints/epoch_N with <run>/config.yaml.
    candidates = [
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.json",
        checkpoint_path.parent / "config.yaml",
        checkpoint_path / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        "A Cosmos Framework DCP checkpoint requires model.config_file. "
        f"Could not infer config.yaml for {checkpoint_path}."
    )


def _default_export_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "hf_exports" / checkpoint_path.name
    return checkpoint_path.parent / "hf_exports" / checkpoint_path.name


def _export_is_complete(export_dir: Path) -> bool:
    return is_hf_checkpoint(export_dir) and (export_dir / "checkpoint.json").is_file()


def _check_output_path(output: Path, *sources: Path) -> None:
    for source in sources:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Export destination overlaps its source: {output}")


def _matches_marker(marker: Path, expected: dict) -> bool:
    try:
        return json.loads(marker.read_text()) == expected
    except (OSError, ValueError):
        return False


_export_calls: dict[tuple[str, str], int] = {}


def _coordinated_export(output: Path, request: dict, operation: Callable[[], str], timeout_seconds: int) -> str:
    """Only rank zero reads source weights; peers await this invocation's receipt.

    A cached export still needs a fresh content check by rank zero. Receipts must
    never substitute an earlier launch's check for that verification.
    """
    rank, world_size, _ = distributed_identity()
    if world_size == 1:
        return operation()
    invocation = distributed_invocation_id()
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    key = (invocation, request_key)
    sequence = _export_calls.get(key, 0)
    _export_calls[key] = sequence + 1
    checks = output.parent / ".cosmos_export_checks" / request_key / invocation / str(sequence)
    checks.mkdir(parents=True, exist_ok=True)
    # Fail closed if a launcher accidentally reuses an identity from a prior run.
    with (checks / f"rank_{rank}.started").open("x"):
        pass
    receipt = checks / "verified.json"
    if rank == 0:
        try:
            result = operation()
        except Exception as error:
            payload = {"error": f"{type(error).__name__}: {error}"}
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload) + "\n")
            temporary.replace(receipt)
            raise
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps({"result": result}) + "\n")
        temporary.replace(receipt)
        return result
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if receipt.is_file():
            payload = json.loads(receipt.read_text())
            if "error" in payload:
                raise RuntimeError(f"Rank zero checkpoint resolution failed: {payload['error']}")
            result = payload["result"]
            if Path(result) != output or not is_hf_checkpoint(result):
                raise RuntimeError(f"Invalid checkpoint verification receipt: {receipt}")
            return result
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for rank zero to verify/export {output}")


def ensure_hf_checkpoint(
    model_path: str,
    *,
    config_file: str | None = None,
    export_dir: str | None = None,
    vit_checkpoint_path: str | None = None,
    timeout_seconds: int = 7200,
) -> str:
    """Return an HF export, verifying/exporting DCP content on rank zero only."""
    path = Path(model_path).expanduser()
    if not path.exists() or is_hf_checkpoint(path):
        return str(path.resolve()) if path.exists() else model_path
    path = path.resolve()
    output = Path(export_dir).expanduser().resolve() if export_dir else _default_export_dir(path)
    request = {
        "checkpoint": str(path),
        "config_file": config_file,
        "export_dir": str(output),
        "vit_checkpoint_path": vit_checkpoint_path,
    }
    return _coordinated_export(
        output,
        request,
        lambda: _ensure_hf_checkpoint(
            str(path), config_file=config_file, export_dir=str(output), vit_checkpoint_path=vit_checkpoint_path
        ),
        timeout_seconds,
    )


def _ensure_hf_checkpoint(
    model_path: str,
    *,
    config_file: str | None = None,
    export_dir: str | None = None,
    vit_checkpoint_path: str | None = None,
) -> str:
    """Return an HF export path, exporting a DCP checkpoint when necessary."""
    path = Path(model_path).expanduser()
    if not path.exists():
        # Preserve HF Hub identifiers for native from_pretrained handling.
        return model_path
    path = path.resolve()
    if is_hf_checkpoint(path):
        return str(path)
    if not is_dcp_checkpoint(path):
        raise ValueError(
            f"Model path {path} is neither an HF safetensors directory nor a Cosmos Framework DCP checkpoint."
        )

    config_path = Path(config_file).expanduser().resolve() if config_file else _infer_config_file(path)
    output_path = Path(export_dir).expanduser().resolve() if export_dir else _default_export_dir(path)
    _check_output_path(output_path, path, config_path)
    expected = {
        "checkpoint": str(path),
        "checkpoint_fingerprint": _source_fingerprint(path),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "base_model": str(vit_checkpoint_path or ""),
    }
    if vit_checkpoint_path and Path(vit_checkpoint_path).is_dir():
        expected["base_model_fingerprint"] = _source_fingerprint(Path(vit_checkpoint_path))
    complete_marker = output_path / ".cosmos_export_complete"
    if _export_is_complete(output_path) and _matches_marker(complete_marker, expected):
        return str(output_path)
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from cosmos_framework.scripts.export_vlm_dcp import export_vlm_dcp, is_vlm_training_config

        with TemporaryDirectory(prefix=".cosmos-export-", dir=output_path.parent) as temporary:
            staging = Path(temporary) / "model"
            if is_vlm_training_config(config_path):
                export_vlm_dcp(
                    path, config_file=config_path, output_dir=staging, base_model_path_or_uri=vit_checkpoint_path
                )
            else:
                command = [
                    sys.executable,
                    "-m",
                    "cosmos_framework.scripts.export_model",
                    "--checkpoint-path",
                    str(path),
                    "--config-file",
                    str(config_path),
                    "-o",
                    str(staging),
                ]
                if vit_checkpoint_path:
                    command.extend(["--vit-checkpoint-path", vit_checkpoint_path])
                export_env = os.environ.copy()
                for name in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "GROUP_RANK",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                ):
                    export_env.pop(name, None)
                subprocess.run(command, check=True, env=export_env)
            if not _export_is_complete(staging):
                raise RuntimeError(f"Export did not produce a complete model at {staging}")
            (staging / complete_marker.name).write_text(json.dumps(expected) + "\n")
            staging.rename(output_path)
    else:
        raise ValueError(f"Export destination exists without matching provenance; choose a new path: {output_path}")
    return str(output_path)


def ensure_evaluation_checkpoint(
    model_path: str,
    *,
    enable_lora: bool = False,
    base_model_path: str | None = None,
    config_file: str | None = None,
    export_dir: str | None = None,
    vit_checkpoint_path: str | None = None,
    timeout_seconds: int = 7200,
) -> str:
    """Resolve dense, Framework DCP, or PEFT checkpoints for evaluation.

    Framework PEFT DCP checkpoints are merged by ``export_vlm_dcp``. Native
    Cosmos Framework adapter checkpoints are merged exactly once by rank zero; other
    data-parallel ranks wait for the completed, self-contained HF export.
    """
    path = Path(model_path).expanduser()
    if path.exists() and is_lora_adapter(path):
        output = Path(export_dir).expanduser().resolve() if export_dir else path.resolve().parent / "merged" / path.name
        request = {
            "adapter": str(path.resolve()),
            "enable_lora": enable_lora,
            "base_model_path": base_model_path,
            "export_dir": str(output),
        }
        return _coordinated_export(
            output,
            request,
            lambda: _ensure_adapter_checkpoint(
                model_path, enable_lora=enable_lora, base_model_path=base_model_path, export_dir=str(output)
            ),
            timeout_seconds,
        )
    return ensure_hf_checkpoint(
        model_path,
        config_file=config_file,
        export_dir=export_dir,
        vit_checkpoint_path=vit_checkpoint_path,
        timeout_seconds=timeout_seconds,
    )


def _ensure_adapter_checkpoint(
    model_path: str, *, enable_lora: bool, base_model_path: str | None, export_dir: str
) -> str:
    path = Path(model_path).expanduser()
    adapter = path.exists() and is_lora_adapter(path)
    if adapter and not enable_lora:
        raise ValueError(f"Model path {path} is a PEFT adapter but model.enable_lora is false.")
    if adapter:
        if not base_model_path:
            raise ValueError("model.base_model_path is required to evaluate a PEFT adapter checkpoint.")
        adapter_path = path.resolve()
        output_path = (
            Path(export_dir).expanduser().resolve()
            if export_dir
            else adapter_path.parent / "merged" / adapter_path.name
        )
        complete_marker = output_path / ".cosmos_lora_merge_complete.json"
        base_path = Path(base_model_path).expanduser().resolve()
        if not base_path.is_dir():
            raise ValueError(f"PEFT base model must be a local directory: {base_path}")
        _check_output_path(output_path, adapter_path, base_path)
        expected_provenance = {
            "adapter_checkpoint": str(adapter_path),
            "adapter_fingerprint": _source_fingerprint(adapter_path),
            "base_model": str(base_path),
            "base_model_fingerprint": _source_fingerprint(base_path),
            "format": "merged-peft-hf",
        }
        cached_provenance = None
        if complete_marker.is_file():
            try:
                cached_provenance = json.loads(complete_marker.read_text())
                cached_provenance.pop("coordination_id", None)
            except (OSError, ValueError):
                cached_provenance = None
        merge_complete = is_hf_checkpoint(output_path) and cached_provenance == expected_provenance
        if not merge_complete:
            from cosmos_framework.checkpoint.lora import merge_lora_model

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                raise ValueError(
                    f"Merge destination exists without matching provenance; choose a new path: {output_path}"
                )
            with TemporaryDirectory(prefix=".cosmos-merge-", dir=output_path.parent) as temporary:
                temporary_output = Path(temporary) / "model"
                merged = Path(
                    merge_lora_model(str(adapter_path), str(base_path), merged_model_path=str(temporary_output))
                ).resolve()
                if merged != temporary_output.resolve() or not is_hf_checkpoint(temporary_output):
                    raise RuntimeError(f"LoRA merge did not produce a complete HF model at {temporary_output}")
                (temporary_output / complete_marker.name).write_text(json.dumps(expected_provenance) + "\n")
                temporary_output.rename(output_path)
        return str(output_path)
    raise ValueError(f"Not a PEFT adapter checkpoint: {path}")
