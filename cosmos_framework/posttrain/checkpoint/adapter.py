# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Local DCP resume and digest-bound portable artifacts for frozen-base adapters."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from cosmos_framework.posttrain.checkpoint.artifact import (
    CHECKPOINT_ARTIFACT_MANIFEST,
    read_local_artifact_manifest,
)
from cosmos_framework.posttrain.checkpoint.io import is_complete_iter_dir
from cosmos_framework.posttrain.model.checkpoint import ADAPTER_CHECKPOINT_PROTOCOL

if TYPE_CHECKING:
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec

MODEL_METADATA_COMPONENT = "model_metadata"


def _read_bound_artifact_component(path: str, name: str) -> dict[str, Any]:
    """Hash and deserialize the same opened file, honoring the native commit-last manifest."""
    import torch

    artifact = read_local_artifact_manifest(path)
    descriptor = artifact["components"].get(name, {})
    if (
        artifact["checkpoint_format"] != "consolidated"
        or artifact["checkpoint_components"] != ["model"]
        or descriptor.get("kind") != "file"
        or descriptor.get("path") != f"{name}.pt"
        or "sha256" not in descriptor
    ):
        raise RuntimeError(f"portable adapter artifact requires digest-bound consolidated {name}.pt")
    with open(os.path.join(path, descriptor["path"]), "rb") as handle:
        digest = sha256()
        size = 0
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        if size != descriptor["size"] or digest.hexdigest() != descriptor["sha256"]:
            raise ValueError(f"adapter artifact {name}.pt payload differs from its committed SHA256/size")
        handle.seek(0)
        # Only adapter tensors live here. Reading from the verified descriptor avoids path replacement
        # between hashing and loading; no original base tensors are materialized.
        return torch.load(handle, map_location="cpu", weights_only=True)


def _consolidated_adapter_state(path: str) -> dict[str, Any]:
    state = _read_bound_artifact_component(path, "model")
    if not isinstance(state, dict) or set(state) != {"net"} or not isinstance(state["net"], dict):
        raise ValueError("portable adapter model.pt requires exactly an adapter tensor mapping under net")
    return state["net"]


def _tensor_inventory(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    import torch

    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor) for name, tensor in state.items()):
        raise ValueError("adapter model state must contain only named tensors")
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
        for name, tensor in state.items()
    }


def _require_subset(actual: Any, expected: Any, path: str = "identity") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"adapter checkpoint {path} differs from the requested identity")
        for name, value in expected.items():
            if name not in actual:
                raise ValueError(f"adapter checkpoint is missing {path}.{name}")
            _require_subset(actual[name], value, f"{path}.{name}")
    elif actual != expected:
        raise ValueError(f"adapter checkpoint {path} differs from the requested identity")


def read_adapter_metadata(
    path: str,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    require_resume: bool = False,
    manager: CheckpointManager | None = None,
) -> dict[str, Any]:
    """Preflight a committed local checkpoint before allocating its base model.

    ``expected_identity`` is an explicit subset for information available before construction.
    The codec must subsequently validate the complete identity against the resolved live model.
    A partial artifact supports adapter reload only; exact resume requires optimizer, trainer,
    RNG, and data state and is limited to the original one-rank topology.
    """
    if path.startswith("s3://") or manager is not None and manager.store.is_object_store:
        raise ValueError("adapter checkpoint v1 supports local storage only")
    marker = os.path.join(path, CHECKPOINT_ARTIFACT_MANIFEST)
    if os.path.isfile(marker):
        artifact = read_local_artifact_manifest(path)
        if require_resume:
            raise ValueError("adapter-only artifacts cannot resume training")
        components = artifact["components"]
        if (
            artifact["checkpoint_format"] != "consolidated"
            or components.get("model", {}).get("kind") != "file"
            or components.get(MODEL_METADATA_COMPONENT, {}).get("kind") != "file"
        ):
            raise RuntimeError("portable adapter artifact requires committed consolidated model and model_metadata")
    elif not is_complete_iter_dir(path):
        raise RuntimeError(f"adapter checkpoint is incomplete or uncommitted: {path!r}")
    import torch

    metadata_path = os.path.join(path, f"{MODEL_METADATA_COMPONENT}.pt")
    if not os.path.isfile(metadata_path):
        raise RuntimeError("adapter checkpoint requires model_metadata.pt; dense checkpoint fallback is forbidden")
    metadata = (
        _read_bound_artifact_component(path, MODEL_METADATA_COMPONENT)
        if os.path.isfile(marker)
        else torch.load(metadata_path, map_location="cpu", weights_only=True)
    )
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"protocol", "identity", "topology", "parameters", "optimizer"}
        or metadata["protocol"] != ADAPTER_CHECKPOINT_PROTOCOL
        or metadata["topology"] != {"strategy": "single_device", "world_size": 1}
        or not isinstance(metadata["identity"], dict)
        or not metadata["identity"]
        or not isinstance(metadata["parameters"], dict)
        or not metadata["parameters"]
    ):
        raise ValueError("unsupported adapter checkpoint metadata schema or topology")
    # Reject tensors, non-JSON state, NaN, and accidental backend Python objects in the manifest.
    json.dumps(metadata, allow_nan=False)
    for name, descriptor in metadata["parameters"].items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(descriptor, dict)
            or set(descriptor) != {"shape", "dtype"}
            or descriptor["dtype"] != "float32"
            or not isinstance(descriptor["shape"], list)
            or any(type(size) is not int or size < 0 for size in descriptor["shape"])
        ):
            raise ValueError(f"invalid adapter checkpoint parameter descriptor: {name!r}")
    if expected_identity is not None:
        # Config dataclasses may expose tuples; persisted identities use their JSON array form.
        expected = json.loads(json.dumps(dict(expected_identity), allow_nan=False))
        _require_subset(metadata["identity"], expected)
    validate_adapter_tensor_inventory(path, metadata)
    if require_resume:
        if not metadata["optimizer"]:
            raise RuntimeError("adapter resume requires an optimizer identity")
        for component in ("optim/.metadata", "trainer.pt", "data.pt"):
            if not os.path.isfile(os.path.join(path, component)):
                raise RuntimeError(f"adapter resume requires complete {component} state")
    return metadata


def _dcp_inventory(path: str, component: str) -> dict[str, dict[str, Any] | None]:
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(os.path.join(path, component)).read_metadata()
    inventory = {}
    for name, descriptor in metadata.state_dict_metadata.items():
        size = getattr(descriptor, "size", None)
        inventory[name] = (
            None
            if size is None
            else {"shape": list(size), "dtype": str(descriptor.properties.dtype).removeprefix("torch.")}
        )
    return inventory


def validate_adapter_tensor_inventory(path: str, metadata: Mapping[str, Any]) -> None:
    """DCP does not reject extra source keys or dtype conversions; check both before loading."""
    inventory = (
        _tensor_inventory(_consolidated_adapter_state(path))
        if os.path.isfile(os.path.join(path, CHECKPOINT_ARTIFACT_MANIFEST))
        else _dcp_inventory(path, "model")
    )
    if inventory != metadata["parameters"]:
        raise ValueError("adapter model inventory differs from its metadata (names, shapes, or dtypes)")


def validate_dcp_target_inventory(path: str, component: str, state: dict[str, Any]) -> None:
    """Compare a parameter-only load target to every saved DCP entry without reading tensors."""
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

    planner = DefaultSavePlanner()
    planner.set_up_planner(state, is_coordinator=True)
    expected = {}
    for item in planner.create_local_plan().items:
        tensor = item.tensor_data
        expected[item.index.fqn] = (
            None
            if tensor is None
            else {"shape": list(tensor.size), "dtype": str(tensor.properties.dtype).removeprefix("torch.")}
        )
    if _dcp_inventory(path, component) != expected:
        raise ValueError(f"adapter DCP {component} inventory differs from its load target (names, shapes, or dtypes)")


def load_adapter_weights(
    path: str,
    codec: AdapterCheckpointCodec,
    *,
    manager: CheckpointManager,
    purpose: Literal["reload", "resume"] = "reload",
) -> None:
    """Reload only adapters onto the independently reconstructed, identical frozen base."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    if purpose != "reload":
        raise ValueError("use trainer.load for exact resume; load_adapter_weights only reloads the numerical model")
    metadata = read_adapter_metadata(path, manager=manager)
    codec.validate_metadata(metadata, purpose=purpose)
    if os.path.isfile(os.path.join(path, CHECKPOINT_ARTIFACT_MANIFEST)):
        state = _consolidated_adapter_state(path)
    else:
        state = get_model_state_dict(codec.view)
        validate_dcp_target_inventory(path, "model", state)
        if not manager.load_dcp_component(path, "model", state):
            raise RuntimeError("adapter checkpoint model component is missing")
    set_model_state_dict(codec.view, state)
    codec.assert_live()
    codec.validate_frozen_base()


def save_adapter_artifact(
    iter_hint: str,
    parameters: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    manager: CheckpointManager,
    publish: bool = True,
) -> str:
    """Publish adapter tensors with byte digests through the existing consolidated artifact writer."""
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    if manager.store.is_object_store:
        raise ValueError("portable adapter v1 publication requires a local checkpoint store")
    if _tensor_inventory(parameters) != metadata["parameters"]:
        raise ValueError("adapter artifact tensors differ from their metadata")
    artifact_manager = CheckpointManager(
        checkpoint_format="consolidated",
        checkpoint_store=manager.store,
        checkpoint_components=["model"],
        upload=manager.upload,
    )
    return artifact_manager.save_consolidated(
        iter_hint,
        {"model": {"net": dict(parameters)}, MODEL_METADATA_COMPONENT: {**metadata, "optimizer": None}},
        async_safe_snapshot=False,
        publish=publish,
        update_latest=False,
    )


def publish_adapter_artifact(
    source_checkpoint: str,
    output_iter_hint: str,
    *,
    manager: CheckpointManager | None = None,
    publish: bool = True,
) -> str:
    """Convert an adapter checkpoint to a portable artifact without constructing its base.

    Only the adapter tensors are allocated on CPU. The output uses the native sibling
    ``checkpoint_artifacts/iter_*`` namespace, never advances training latest, and refuses
    replacement of an already committed artifact.
    """
    import torch

    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    manager = CheckpointManager(checkpoint_format="dcp") if manager is None else manager
    metadata = read_adapter_metadata(source_checkpoint, manager=manager)
    if os.path.isfile(os.path.join(source_checkpoint, CHECKPOINT_ARTIFACT_MANIFEST)):
        state = _consolidated_adapter_state(source_checkpoint)
    else:
        state = {
            name: torch.empty(descriptor["shape"], dtype=torch.float32)
            for name, descriptor in metadata["parameters"].items()
        }
        if not manager.load_dcp_component(source_checkpoint, "model", state):
            raise RuntimeError("adapter checkpoint model component is missing")
    return save_adapter_artifact(output_iter_hint, state, metadata, manager=manager, publish=publish)


def main() -> None:
    """Publish a portable adapter artifact; this operation neither merges nor requantizes the base."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--source", required=True, help="Committed native adapter checkpoint directory")
    parser.add_argument("--output-iter", required=True, help="Iteration hint ending checkpoints/iter_NNNNNN")
    args = parser.parse_args()
    print(publish_adapter_artifact(args.source, args.output_iter))


if __name__ == "__main__":
    main()
