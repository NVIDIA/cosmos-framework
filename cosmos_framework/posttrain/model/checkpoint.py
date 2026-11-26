# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Parameter-only checkpoint views for frozen-base adapter training.

The view registers the live adapter Parameters under their canonical names. It never owns or
traverses a base module during state-dict extraction; filtering a full state dict is too late for
packed quantized parameters and for memory-constrained loading.
"""

from __future__ import annotations

import json
import os
import weakref
from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import nn

ADAPTER_CHECKPOINT_PROTOCOL = "posttrain.adapter-checkpoint.v1"
# Exact resume compares the whole identity, since any recorded field can change the continued
# trajectory. Adapter reload, inference and export compare only the fields that define the model:
# base identity, adapter rank/alpha/targets, quantization, adapter compute dtype and the
# task/processor contracts. The process environment, run history, seeds, dropout, initializer and
# memory/speed execution settings below are resume-only.
RESUME_ONLY_IDENTITY_FIELDS = frozenset(
    {
        "training_contract",
        "initial_adapters",
        "initialization",
        "strategy",
        "hardware",
        "torch_version",
        "torch_git_version",
        "dependencies",
    }
)
RESUME_ONLY_NESTED_IDENTITY_FIELDS: dict[str, frozenset[str]] = {
    "adapter": frozenset(
        {"dropout", "seed", "initialization", "initialization_seed", "initialization_artifact_sha256"}
    ),
    "head_adapters": frozenset({"dropout", "seed"}),
    "execution_policy": frozenset(
        {
            "math_precision",
            "activation_checkpointing",
            "sft_logits_mode",
            "language_compile",
            "quantized_linear_execution",
        }
    ),
}


def _model_defining_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: value for name, value in identity.items() if name not in RESUME_ONLY_IDENTITY_FIELDS}
    for field, ignored in RESUME_ONLY_NESTED_IDENTITY_FIELDS.items():
        value = result.get(field)
        if isinstance(value, Mapping):
            result[field] = {name: item for name, item in value.items() if name not in ignored}
    return result


# Set to "1" to re-hash the whole frozen base at every adapter checkpoint save and load.
VERIFY_FROZEN_BASE_ENV = "COSMOS_VERIFY_FROZEN_BASE_EVERY_CHECKPOINT"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class AdapterCheckpointCodec:
    """A one-device, FP32-adapter codec sharing the final model's live Parameters."""

    def __init__(self, net: nn.Module, *, identity: Mapping[str, Any]) -> None:
        parameters = getattr(net, "_cosmos_adapter_parameters", None)
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("adapter checkpoint codec requires a non-empty canonical parameter mapping")
        self.identity = _json_copy(dict(identity))
        if not self.identity:
            raise ValueError("adapter checkpoint identity must not be empty")
        self.parameters = dict(sorted(parameters.items()))
        self.view = nn.Module()
        self._net = weakref.ref(net)
        for name, parameter in self.parameters.items():
            if not isinstance(name, str) or any(not part for part in name.split(".")):
                raise ValueError(f"invalid canonical adapter parameter name: {name!r}")
            if not isinstance(parameter, nn.Parameter) or parameter.is_meta:
                raise ValueError(f"adapter parameter {name!r} is not a materialized Parameter")
            if parameter.dtype != torch.float32:
                raise ValueError(f"adapter parameter {name!r} must use float32, got {parameter.dtype}")
            module = self.view
            *parents, leaf = name.split(".")
            for part in parents:
                child = module._modules.get(part)
                if child is None:
                    child = nn.Module()
                    module.add_module(part, child)
                module = child
            module.register_parameter(leaf, parameter)
        self._inventory = {
            name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype).removeprefix("torch.")}
            for name, parameter in self.parameters.items()
        }
        self._devices = {name: parameter.device for name, parameter in self.parameters.items()}
        self._model_parameters = {
            id(parameter): (parameter.shape, parameter.dtype, parameter.device) for parameter in net.parameters()
        }
        self.assert_live(net)

    def assert_live(self, net: nn.Module | None = None, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Reject replacement, movement, sharding, or changed trainability before any extraction."""
        import torch.distributed as dist
        from torch.distributed.tensor import DTensor

        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("adapter checkpoints currently require world_size=1 and single-device placement")
        net = self._net() if net is None else net
        if net is None or net is not self._net():
            raise RuntimeError("adapter checkpoint codec no longer refers to its constructed model")
        current = getattr(net, "_cosmos_adapter_parameters", {})
        if set(current) != set(self.parameters) or any(current[name] is not p for name, p in self.parameters.items()):
            raise RuntimeError("adapter parameter references changed after checkpoint codec construction")
        live = {id(p): p for p in net.parameters()}
        if {key: (p.shape, p.dtype, p.device) for key, p in live.items()} != self._model_parameters:
            raise RuntimeError("model parameters changed identity, shape, dtype, or device after codec construction")
        selected = {id(p) for p in self.parameters.values()}
        if len(selected) != len(self.parameters):
            raise ValueError("adapter checkpoint names must refer to distinct Parameters")
        if {id(p) for p in live.values() if p.requires_grad} != selected:
            raise ValueError("trainable model parameters must equal the adapter checkpoint allowlist")
        for name, parameter in self.parameters.items():
            expected = self._inventory[name]
            if (
                id(parameter) not in live
                or isinstance(parameter, DTensor)
                or list(parameter.shape) != expected["shape"]
                or str(parameter.dtype).removeprefix("torch.") != expected["dtype"]
                or parameter.device != self._devices[name]
            ):
                raise RuntimeError(f"adapter parameter {name!r} changed identity, shape, dtype, or device")
        if optimizer is not None:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            if len(params) != len(selected) or {id(p) for p in params} != selected:
                raise ValueError("optimizer membership must contain exactly the adapter parameters, once each")
            if any(id(parameter) not in selected for parameter in optimizer.state):
                raise ValueError("optimizer state contains a parameter outside the adapter allowlist")

    def validate_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Require an AdamW over exactly the live adapter parameters."""
        self.assert_live(optimizer=optimizer)
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError("adapter checkpoint v1 supports torch.optim.AdamW only")

    def optimizer_identity(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Record ordered membership and static options, excluding scheduler-controlled learning rates."""
        self.validate_optimizer(optimizer)
        names = {id(parameter): name for name, parameter in self.parameters.items()}
        return {
            "class": "torch.optim.AdamW",
            "groups": [
                {
                    "parameters": [names[id(parameter)] for parameter in group["params"]],
                    "options": _json_copy(
                        {
                            key: value
                            for key, value in group.items()
                            if key not in {"params", "lr", "initial_lr", "step"}
                        }
                    ),
                }
                for group in optimizer.param_groups
            ],
        }

    def metadata(self, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        """Return small JSON metadata, independent of packed base tensor state."""
        self.assert_live(optimizer=optimizer)
        return _json_copy(
            {
                "protocol": ADAPTER_CHECKPOINT_PROTOCOL,
                "identity": self.identity,
                "topology": {"strategy": "single_device", "world_size": 1},
                "parameters": self._inventory,
                "optimizer": self.optimizer_identity(optimizer) if optimizer is not None else None,
            }
        )

    def bind_training_identity(self, identity: Mapping[str, Any]) -> None:
        """Bind immutable recipe/processor/schedule semantics before the trainer is constructed."""
        contract = _json_copy(dict(identity))
        if not contract:
            raise ValueError("adapter training identity must not be empty")
        previous = self.identity.get("training_contract")
        if previous is not None and previous != contract:
            raise ValueError("adapter checkpoint training identity is already bound to another recipe")
        self.identity["training_contract"] = contract

    def bind_processor_identity(self, identity: Mapping[str, Any]) -> None:
        """Bind processor/template/media semantics, which remain mandatory for numerical reload."""
        contract = _json_copy(dict(identity))
        if not contract:
            raise ValueError("adapter processor identity must not be empty")
        previous = self.identity.get("processor_contract")
        if previous is not None and previous != contract:
            raise ValueError("adapter checkpoint processor identity is already bound to another processor")
        self.identity["processor_contract"] = contract

    def bind_provenance(self, provenance: Mapping[str, str]) -> None:
        """Record optional build provenance; it is never compared on reload or resume."""
        if provenance:
            self.identity["provenance"] = _json_copy(dict(provenance))

    def validate_frozen_base(self) -> None:
        """Re-hash every frozen/packed base tensor against the digests recorded at build.

        This copies the whole frozen base to host, so it runs at explicit verification points
        (export, end of action training), not per checkpoint unless ``VERIFY_FROZEN_BASE_ENV`` is set.
        """
        self.assert_live()
        validate = getattr(self._net(), "_cosmos_validate_frozen_base", None)
        if validate is not None:
            validate()

    def validate_checkpoint_boundary(self) -> None:
        """The per-save/load check: cheap runtime state, or the full proof when opted in."""
        if os.environ.get(VERIFY_FROZEN_BASE_ENV) == "1":
            self.validate_frozen_base()
            return
        self.assert_live()
        validate = getattr(self._net(), "_cosmos_validate_runtime_state", None)
        if validate is not None:
            validate()

    def validate_metadata(
        self,
        metadata: Mapping[str, Any],
        optimizer: torch.optim.Optimizer | None = None,
        *,
        purpose: Literal["reload", "resume", "inference"] = "reload",
    ) -> None:
        """Compare model-defining identity for reload/inference, and everything for exact resume.

        Resume also compares the resume-only fields (``RESUME_ONLY_IDENTITY_FIELDS`` and
        ``RESUME_ONLY_NESTED_IDENTITY_FIELDS``), since any of them can change the continued trajectory.
        """
        if purpose not in ("reload", "resume", "inference"):
            raise ValueError(f"unsupported adapter checkpoint purpose: {purpose!r}")
        if purpose == "inference" and optimizer is not None:
            raise ValueError("inference validation cannot validate an optimizer or training resume")
        if purpose == "resume" and optimizer is None:
            raise ValueError("adapter resume validation requires the live optimizer")
        expected = self.metadata(optimizer)
        for key in ("protocol", "identity", "topology", "parameters"):
            saved_value, expected_value = metadata.get(key), expected[key]
            if key == "identity":
                if not isinstance(saved_value, Mapping):
                    raise ValueError("adapter checkpoint identity must be a mapping")
                saved_value = {name: value for name, value in saved_value.items() if name != "provenance"}
                expected_value = {name: value for name, value in expected_value.items() if name != "provenance"}
                if purpose != "resume":
                    saved_value, expected_value = map(_model_defining_identity, (saved_value, expected_value))
            if saved_value != expected_value:
                raise ValueError(f"adapter checkpoint {key} differs from the constructed model")
        if optimizer is not None and metadata.get("optimizer") != expected["optimizer"]:
            raise ValueError("adapter checkpoint optimizer identity differs from the configured optimizer")


def build_adapter_checkpoint_codec(net: nn.Module, *, identity: Mapping[str, Any]) -> AdapterCheckpointCodec:
    """Build after adapter initialization and all model transformations, before the optimizer."""
    return AdapterCheckpointCodec(net, identity=identity)
