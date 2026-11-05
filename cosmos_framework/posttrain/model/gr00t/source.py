# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Content-bound sources for the optional GR00T PPO capability.

The official loader restores every frozen tensor from the GR00T weights. Its nested
backbone config and tokenizer/processor artifacts are separate dependencies: resolve
those once and pass their immutable revision or local directory to the loaders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from traceback import clear_frames
from typing import TypeVar

from cosmos_framework.posttrain.checkpoint.conversion.hf import INDEX_FILE, SIDECAR_DIRS, SIDECAR_FILES, shard_status
from cosmos_framework.posttrain.checkpoint.manifest import is_canonical_relative_path, parse_json_object, sha256
from cosmos_framework.posttrain.checkpoint.provenance import CheckpointProvenance, load_with_stable_provenance
from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
from cosmos_framework.posttrain.utils.object_store import file_sha256

SOURCE_PROTOCOL = "gr00t-ppo-source-v1"
_Result = TypeVar("_Result")


def _json(path: Path) -> dict:
    return parse_json_object(path.read_bytes(), context=f"GR00T source {path}")


def _resolve(name: str, revision: str = "", *, dependency: bool = False) -> tuple[Path, str]:
    directory = Path(name)
    if directory.is_dir():
        if revision:
            raise ValueError(f"GR00T revision cannot pin the local directory {name!r}")
        return directory, ""
    from huggingface_hub import snapshot_download

    patterns = None
    if dependency:
        patterns = [
            "config.json",
            *SIDECAR_FILES,
            *(f"{name}/**" for name in SIDECAR_DIRS),
            "model*.safetensors",
            INDEX_FILE,
        ]
    directory = Path(snapshot_download(repo_id=name, revision=revision or None, allow_patterns=patterns))
    return directory, directory.name


def _dependency_name(config: dict, *, context: str) -> str:
    name = config.get("model_name", "nvidia/Cosmos-Reason2-2B")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{context}.model_name must be a nonempty local directory or HF repository")
    return name


def _metadata(directory: Path, *, gr00t: bool = False) -> dict[str, str]:
    required = ("config.json", "processor_config.json", "statistics.json") if gr00t else ("config.json",)
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"GR00T source requires {directory / name}")
    files = {
        name: file_sha256(str(directory / name))
        for name in sorted({*required, *SIDECAR_FILES, *(("embodiment_id.json",) if gr00t else ())})
        if (directory / name).is_file()
    }
    for name in SIDECAR_DIRS:
        for path in sorted((directory / name).rglob("*")):
            if path.is_file():
                files[path.relative_to(directory).as_posix()] = file_sha256(str(path))
    return files


def _weights(directory: Path) -> dict[str, str]:
    # HF selects the single safetensors file before an index when both exist.
    if (directory / "model.safetensors").is_file():
        shards = {"model.safetensors": None}
        files = {}
    else:
        index = directory / INDEX_FILE
        weight_map = _json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index}.weight_map must be a nonempty object")
        shards: dict[str, set[str] | None] = {}
        for key, shard in weight_map.items():
            if not key or not is_canonical_relative_path(shard):
                raise ValueError(f"{index}.weight_map[{key!r}] is invalid: {shard!r}")
            shards.setdefault(shard, set()).add(key)
        files = {INDEX_FILE: file_sha256(str(index))}
    # Validate the complete inventory before reading any large tensor payload.
    for name, keys in shards.items():
        reason = shard_status(str(directory / name), keys)
        if reason is not None:
            raise ValueError(f"GR00T weight source: {reason}")
    files.update({name: file_sha256(str(directory / name)) for name in sorted(shards)})
    return files


@dataclass(frozen=True)
class Gr00tPPOSource:
    """Resolved loader inputs; identities are captured afresh for each model instance."""

    checkpoint: Path
    backbone: Path
    backbone_revision: str
    processor: Path
    processor_revision: str
    config_sha256: str
    processor_config_sha256: str

    def identity(self) -> CheckpointProvenance:
        root = _metadata(self.checkpoint, gr00t=True)
        if (root["config.json"], root["processor_config.json"]) != (
            self.config_sha256,
            self.processor_config_sha256,
        ):
            raise RuntimeError("GR00T configuration changed after source resolution")
        metadata = {path: _metadata(path) for path in {self.backbone, self.processor}}
        manifest = {
            "checkpoint": {**root, **_weights(self.checkpoint)},
            "backbone": metadata[self.backbone],
            "backbone_revision": self.backbone_revision,
            "processor": metadata[self.processor],
            "processor_revision": self.processor_revision,
        }
        return {"protocol": SOURCE_PROTOCOL, "sha256": canonical_json_sha256(manifest, context="GR00T PPO source")}


def resolve_ppo_source(checkpoint_path: str, revision: str = "") -> Gr00tPPOSource:
    """Resolve the same root and nested dependencies that the official loaders consume."""
    checkpoint, _ = _resolve(checkpoint_path, revision)
    model_payload = (checkpoint / "config.json").read_bytes()
    processor_payload = (checkpoint / "processor_config.json").read_bytes()
    model_config = parse_json_object(model_payload, context=str(checkpoint / "config.json"))
    processor_config = parse_json_object(processor_payload, context=str(checkpoint / "processor_config.json"))
    processor_kwargs = processor_config.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        raise ValueError(f"{checkpoint / 'processor_config.json'}.processor_kwargs must be an object")
    backbone_name = _dependency_name(model_config, context=str(checkpoint / "config.json"))
    processor_name = _dependency_name(processor_kwargs, context=str(checkpoint / "processor_config.json"))
    backbone, backbone_revision = _resolve(backbone_name, dependency=True)
    processor, processor_revision = (
        (backbone, backbone_revision) if processor_name == backbone_name else _resolve(processor_name, dependency=True)
    )
    return Gr00tPPOSource(
        checkpoint,
        backbone,
        backbone_revision,
        processor,
        processor_revision,
        sha256(model_payload),
        sha256(processor_payload),
    )


def load_ppo_source(
    config: Gr00tN1d7ModelConfig,
    load: Callable[[Gr00tPPOSource], _Result],
    *,
    publish: Callable[[_Result], None] | None = None,
) -> tuple[_Result, CheckpointProvenance]:
    """Coordinate local-only loading, source verification, then rank-zero publication before FSDP.

    ``load`` must contain no collectives: one rank may fail before reaching them.
    Publication happens only after all ranks have loaded and verified unchanged sources.
    """
    from cosmos_framework.posttrain.checkpoint.worker import run_rank0_converged
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    source = run_rank_converged(
        lambda: resolve_ppo_source(config.checkpoint_path, config.revision), "GR00T PPO source resolution"
    )
    loaded = None
    try:
        loaded, identity = load_with_stable_provenance(
            str(source.checkpoint),
            lambda: run_rank_converged(lambda: load(source), "GR00T PPO source loading"),
            provenance=lambda _: source.identity(),
        )
        if publish is not None:
            run_rank0_converged(lambda: publish(loaded), "GR00T PPO configuration publication")
        return loaded, identity
    except BaseException as error:
        # The shared loader's completed frame may hold a model when its post-load
        # check fails. Retaining the exception must not retain those GPU resources.
        pending, seen = [error], set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            clear_frames(current.__traceback__)
            pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
        raise
    finally:
        loaded = None
