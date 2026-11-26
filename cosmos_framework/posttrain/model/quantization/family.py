# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit backbone adaptation for native task policies.

Family builders own architecture validation, checkpoint provenance, preprocessing,
and task losses. This module consumes their *resolved* linear/auxiliary allowlists;
it never discovers targets by a suffix or changes a task's forward method. Qwen's
existing loader and artifact identities remain unchanged.
"""

from __future__ import annotations

import json
import math
import platform
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from cosmos_framework.posttrain.config.model import NF4_BLOCK_SIZE
from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
from cosmos_framework.posttrain.model.quantization.loader import (
    TensorSlot,
    guard_resident_movement,
    parameter_slots,
    quantized_state_record,
    require_native_bf16_gpu,
    tensor_digest,
)
from cosmos_framework.utils.generator.lora import LoraInjectedLinear


def family_tensor_record(tensor: torch.Tensor) -> dict:
    """Preserve native layouts while hashing canonical values and binding strides.

    Only noncontiguous tensors need a temporary contiguous copy. This supports
    the native GR00T vision Conv3d channels-last layout without changing it.
    Packed NF4 weights use their dedicated quantization-state representation.
    """
    return {**tensor_digest(tensor.contiguous()), "stride": list(tensor.stride())}


# Version of the family NF4/LoRA arithmetic and checkpoint layout. Bump it by hand whenever a
# change to this module, bnb.py, loader.py or the shared LoRA layer changes numerics or format.
FAMILY_NUMERICS_VERSION = "cosmos.family-backbone-numerics.v1"


@dataclass(frozen=True)
class FamilyBackboneConfig:
    """Storage/compute policy independent of a particular task architecture."""

    mode: str = "nf4_lora"
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in {"nf4_lora", "bf16_lora", "nf4_frozen", "bf16_frozen"}:
            raise ValueError("backbone mode must be nf4_lora|bf16_lora|nf4_frozen|bf16_frozen")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("backbone adapter rank must be a positive integer")
        if type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("backbone adapter alpha must be finite and positive")
        if type(self.dropout) not in (int, float) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("backbone adapter dropout must be finite and in [0, 1)")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("backbone adapter seed must be an integer in [0, 2**63)")


def _json_identity(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"family backbone requires a nonempty {name} identity")
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def require_family_runtime(device: torch.device, *, quantized: bool) -> None:
    """Reject an unsupported device, process group or bitsandbytes before any model mutation."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("family backbone adaptation currently requires one device; FSDP/DTensor is unsupported")
    if device.type == "cpu" and not quantized:
        return  # Small BF16 contract tests only; family recipes require their CUDA runtime.
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("NF4 family backbone requires CUDA; no dense/CPU fallback is permitted")
    if not torch.distributed.is_initialized():
        raise ValueError("native family CUDA training requires torchrun process-group initialization")
    if device.index != torch.cuda.current_device():
        raise ValueError("family backbone requires the framework-owned CUDA device")
    require_native_bf16_gpu(device, "native family backbone")
    if quantized:
        from cosmos_framework.posttrain.model.quantization import check_bitsandbytes_version

        check_bitsandbytes_version(version("bitsandbytes"))


def resolve_family_targets(net: nn.Module, targets: Mapping[str, Sequence[int]]) -> dict[str, nn.Linear]:
    """Validate a family-specific exact mapping before mutating any model state."""
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("family backbone targets must be an explicit nonempty shape allowlist")
    slots = parameter_slots(net)
    occurrences = Counter(id(slot.tensor) for slot in slots.values())
    modules = dict(net.named_modules(remove_duplicate=False))
    selected: dict[str, nn.Linear] = {}
    for name, shape in sorted(targets.items()):
        if not isinstance(name, str) or not name or name not in modules:
            raise ValueError(f"unmatched family backbone target: {name!r}")
        module = modules[name]
        if type(module) is not nn.Linear or module.bias is not None:
            raise ValueError(f"family backbone target requires an ordinary bias-free Linear: {name}")
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or tuple(module.weight.shape) != tuple(shape)
        ):
            raise ValueError(f"family backbone target geometry mismatch: {name}")
        if occurrences[id(module.weight)] != 1:
            raise ValueError(f"tied/shared family backbone target is unsupported: {name}")
        if module.weight.device.type not in {"cpu", "meta"} or module.weight.dtype not in {
            torch.float32,
            torch.bfloat16,
        }:
            raise ValueError(f"family backbone target must be CPU/meta BF16 or FP32 before loading: {name}")
        selected[name] = module
    return selected


def resolve_head_targets(net: nn.Module, targets: Mapping[str, Sequence[int]]) -> dict[str, nn.Linear]:
    """Validate separately declared FP32 biased head projections before any mutation."""
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("head adapters require an explicit nonempty shape allowlist")
    slots = parameter_slots(net)
    occurrences = Counter(id(slot.tensor) for slot in slots.values())
    modules = dict(net.named_modules(remove_duplicate=False))
    module_occurrences = Counter(id(module) for module in modules.values())
    selected = {}
    for name, shape in sorted(targets.items()):
        module = modules.get(name)
        if (
            not isinstance(name, str)
            or not name.startswith("action_head.")
            or type(module) is not nn.Linear
            or module.bias is None
            or module_occurrences[id(module)] != 1
        ):
            raise ValueError(f"head adapter target requires an unaliased ordinary biased Linear: {name}")
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or tuple(module.weight.shape) != tuple(shape)
        ):
            raise ValueError(f"head adapter target geometry mismatch: {name}")
        for value in (module.weight, module.bias):
            if value.device.type not in {"cpu", "meta"} or value.dtype != torch.float32 or occurrences[id(value)] != 1:
                raise ValueError(f"head adapter base and bias must be unshared CPU/meta FP32 tensors: {name}")
        selected[name] = module
    return selected


@dataclass
class FamilyBackboneState:
    config: FamilyBackboneConfig
    targets: dict[str, nn.Module]
    identity: dict[str, Any]
    codec: AdapterCheckpointCodec

    def assert_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.codec.assert_live(optimizer=optimizer)

    def validate_frozen_base(self) -> None:
        self.codec.validate_frozen_base()


@torch.no_grad()
def install_family_backbone(
    net: nn.Module,
    *,
    config: Any,
    targets: Mapping[str, Sequence[int]],
    auxiliary_parameters: Sequence[str],
    source_identity: Mapping[str, Any],
    task_identity: Mapping[str, Any],
    device: torch.device,
    read_tensor: Callable[[str], torch.Tensor] | None = None,
    head_targets: Mapping[str, Sequence[int]] | None = None,
) -> FamilyBackboneState:
    """Install before optimizer construction, on a CPU or meta native task model.

    With ``read_tensor`` a family can stream its exact source keys into meta
    slots. Otherwise an official CPU-loaded model is copied one parameter at a
    time, but source references retain its complete dense host weights until this
    function returns. That dense *host* loading peak must be measured; use meta
    streaming when the family loader supports it. Neither route creates a complete
    dense GPU backbone. Initialized buffers must already exist on CPU.

    Auxiliary parameters are an explicit complete list of concurrently trained
    task-head/projector weights. They remain in portable/resumable artifacts,
    alongside adapters, through the existing parameter-only checkpoint codec.
    Frozen feature-extractor modes intentionally contain no LoRA parameters.
    Optional head targets instead retain frozen FP32 weights/biases and train
    separate FP32 factors; they never enter the language quantization path.
    """
    cfg = FamilyBackboneConfig(**{key: getattr(config, key) for key in FamilyBackboneConfig.__dataclass_fields__})
    source = _json_identity(source_identity, "source")
    task = _json_identity(task_identity, "task")
    quantized, adapted = cfg.mode.startswith("nf4"), cfg.mode.endswith("lora")
    device = torch.device(device)
    require_family_runtime(device, quantized=quantized)
    selected = resolve_family_targets(net, targets)
    heads = {} if head_targets is None else resolve_head_targets(net, head_targets)
    if heads and (
        not adapted
        or cfg.rank != 32
        or cfg.alpha != 64
        or cfg.dropout != 0
        or auxiliary_parameters
        or set(heads) & set(selected)
    ):
        raise ValueError(
            "head adapters require separate r32/alpha64/dropout0 LoRA targets and no full-head auxiliaries"
        )
    head_source_names = {f"{name}.{leaf}" for name in heads for leaf in ("weight", "bias")}
    slots = parameter_slots(net)
    if (
        isinstance(auxiliary_parameters, (str, bytes))
        or len(set(auxiliary_parameters)) != len(auxiliary_parameters)
        or any(name not in slots for name in auxiliary_parameters)
    ):
        raise ValueError("auxiliary trainables must be distinct existing canonical parameter names")
    auxiliary = set(auxiliary_parameters)
    target_weights = {f"{name}.weight" for name in selected}
    if auxiliary & target_weights:
        raise ValueError("quantized/frozen backbone weights cannot also be auxiliary trainables")
    if not adapted and not auxiliary:
        raise ValueError("frozen backbone mode requires explicitly trainable task components")
    if hasattr(net, "_cosmos_adapter_checkpoint_codec"):
        raise ValueError("family backbone is already initialized; do not quantize or initialize on resume")
    occurrences = Counter(id(slot.tensor) for slot in slots.values())
    for name, slot in slots.items():
        if slot.tensor.device.type not in {"cpu", "meta"}:
            raise ValueError(f"family loading requires CPU/meta parameters before owned-device placement: {name}")
        if slot.tensor.is_meta and read_tensor is None:
            raise ValueError(f"family meta parameter requires a source reader: {name}")
        if name in auxiliary and (not slot.tensor.is_floating_point() or occurrences[id(slot.tensor)] != 1):
            raise ValueError(f"auxiliary trainable must be an unshared floating parameter: {name}")
    buffer_slots = {
        f"{name}.{attribute}" if name else attribute: TensorSlot(owner, attribute)
        for name, owner in net.named_modules()
        for attribute, tensor in owner._buffers.items()
        if tensor is not None
    }
    if any(slot.tensor.is_meta or slot.tensor.device.type != "cpu" for slot in buffer_slots.values()):
        raise ValueError("family loading requires initialized CPU buffers, including nonpersistent rotary state")

    adapter = SimpleNamespace(rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)
    if quantized:
        from cosmos_framework.posttrain.model.quantization.bnb import (
            NativeFrozenLinear4bit,
            NativeLoraLinear4bit,
            kernel_preflight,
        )

        # Construction copies weights under no_grad, but the NF4 kernel check
        # intentionally exercises input backward and real adapter updates.
        with torch.enable_grad():
            kernel_preflight(device, adapter)
    for parameter in net.parameters():
        parameter.requires_grad_(False)

    def source_tensor(name: str, existing: torch.Tensor) -> torch.Tensor:
        value = read_tensor(name) if existing.is_meta else existing.detach()
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or tuple(value.shape) != tuple(existing.shape)
        ):
            raise ValueError(f"family source reader returned an incompatible CPU tensor: {name}")
        if value.dtype != existing.dtype:
            raise ValueError(f"family source reader dtype differs from the initialized architecture: {name}")
        if (name in target_weights or name in auxiliary or name in head_source_names) and not torch.isfinite(
            value
        ).all().item():
            raise ValueError(f"family source contains nonfinite backbone or auxiliary values: {name}")
        return value

    installed: dict[str, nn.Module] = {}
    trainable: dict[str, nn.Parameter] = {}
    # Meta constructors neither allocate a second dense matrix nor perturb RNG. Family adapters
    # (adapter_compute_dtype=None) compute under the task policy's ambient autocast.
    for name, base in selected.items():
        value = source_tensor(f"{name}.weight", base.weight).to(dtype=torch.bfloat16)
        if base.weight.dtype != torch.bfloat16 and not torch.isfinite(value).all().item():
            raise ValueError(f"family backbone source overflows BF16 conversion: {name}.weight")
        meta = nn.Linear(base.in_features, base.out_features, bias=False, device="meta", dtype=torch.bfloat16)
        if quantized:
            replacement = (
                NativeLoraLinear4bit(meta, adapter, adapter_compute_dtype=None)
                if adapted
                else NativeFrozenLinear4bit(meta)
            )
            replacement.load_dense_weight_once(value, device)
        elif adapted:
            replacement = LoraInjectedLinear(
                meta,
                cfg.rank,
                cfg.alpha,
                dropout=cfg.dropout,
                adapter_dtype=torch.float32,
            )
            replacement.weight = nn.Parameter(value.to(device=device), requires_grad=False)
        else:
            replacement = meta
            replacement.weight = nn.Parameter(value.to(device=device), requires_grad=False)
        if adapted:
            replacement.lora_A.weight = nn.Parameter(
                torch.empty(cfg.rank, base.in_features, device=device, dtype=torch.float32)
            )
            replacement.lora_B.weight = nn.Parameter(
                torch.empty(base.out_features, cfg.rank, device=device, dtype=torch.float32)
            )
            trainable.update(
                {f"{name}.{key}": parameter for key, parameter in replacement.adapter_parameters().items()}
            )
        replacement.train(base.training)
        parent, _, attribute = name.rpartition(".")
        net.get_submodule(parent).register_module(attribute, replacement)
        installed[name] = replacement
        del value

    moved: dict[int, nn.Parameter] = {}
    for name, slot in slots.items():
        if name in target_weights:
            continue
        original = slot.tensor
        identity = id(original)
        if identity not in moved:
            value = source_tensor(name, original)
            dtype = torch.float32 if name in auxiliary else value.dtype
            converted = value.to(device=device, dtype=dtype)
            if name in auxiliary and value.dtype != dtype and not torch.isfinite(converted).all().item():
                raise ValueError(f"family auxiliary source overflows FP32 conversion: {name}")
            moved[identity] = nn.Parameter(converted, requires_grad=name in auxiliary)
        setattr(slot.owner, slot.attribute, moved[identity])
        if name in auxiliary:
            trainable[name] = moved[identity]
    for slot in buffer_slots.values():
        setattr(slot.owner, slot.attribute, slot.tensor.to(device=device))
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    for module in installed.values():
        if adapted:
            module.reset_adapter_parameters(generator=generator)
    # Head bases have already been streamed in their original FP32 storage. Build
    # only the new factors on meta, then initialize from an independent generator:
    # neither language factor initialization nor global RNG changes.
    installed_heads = {}
    head_generator = torch.Generator(device=device).manual_seed(cfg.seed) if heads else None
    for name in heads:
        base = net.get_submodule(name)
        meta = nn.Linear(base.in_features, base.out_features, bias=True, device="meta", dtype=torch.float32)
        replacement = LoraInjectedLinear(meta, cfg.rank, cfg.alpha, dropout=0.0, adapter_dtype=torch.float32)
        replacement.weight, replacement.bias = base.weight, base.bias
        replacement.lora_A.weight = nn.Parameter(
            torch.empty(cfg.rank, base.in_features, device=device, dtype=torch.float32)
        )
        replacement.lora_B.weight = nn.Parameter(
            torch.empty(base.out_features, cfg.rank, device=device, dtype=torch.float32)
        )
        replacement.reset_adapter_parameters(generator=head_generator)
        replacement.train(base.training)
        parent, _, attribute = name.rpartition(".")
        net.get_submodule(parent).register_module(attribute, replacement)
        installed_heads[name] = replacement
        trainable.update({f"{name}.{key}": value for key, value in replacement.adapter_parameters().items()})
    current = parameter_slots(net)
    packed = {name: module for name, module in installed.items() if quantized}
    frozen = {
        name: family_tensor_record(slot.tensor)
        for name, slot in current.items()
        if name not in trainable and name not in {f"{key}.weight" for key in packed}
    }
    packed_records = {name: quantized_state_record(module) for name, module in packed.items()}
    buffers = {name: family_tensor_record(slot.tensor) for name, slot in buffer_slots.items()}
    identity = {
        "format": "cosmos.family-backbone.v1",
        "numerics_version": FAMILY_NUMERICS_VERSION,
        "base": source,
        "task_contract": task,
        # Model-defining only. The adapter seed/dropout and the recorded library versions shape the
        # training trajectory and are compared on resume (training contract, top-level versions).
        "adapter": {"mode": cfg.mode, "rank": cfg.rank, "alpha": cfg.alpha},
        "targets": {name: list(shape) for name, shape in sorted(targets.items())},
        "auxiliary_parameters": sorted(auxiliary),
        "quantization": {
            "backend": "bitsandbytes",
            "format": "nf4",
            "block_size": NF4_BLOCK_SIZE,
            "nested": True,
            "storage_dtype": "uint8",
            "compute_dtype": "bfloat16",
        }
        if quantized
        else None,
        "precision": {"target_source_cast": "bfloat16", "adapter": "float32", "auxiliary": "float32"},
        "frozen_tensors": frozen,
        "quantized_tensors": packed_records,
        "base_buffers": buffers,
        # Resume-only environment, as in the Qwen loader identity.
        "hardware": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
        if device.type == "cuda"
        else {"name": device.type},
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "dependencies": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("transformers", "safetensors", *(("bitsandbytes",) if quantized else ()))
            },
        },
    }
    if heads:
        identity["head_adapters"] = {
            "profile": "gr00t_n1d7_dit_attention_lora_v1",
            "targets": {name: list(head_targets[name]) for name in heads},
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "base_dtype": "float32",
            "bias": "frozen_float32",
            "parameter_dtype": "float32",
            "compute": "native_separate_gemms_ambient_autocast",
        }

    def validate_frozen() -> None:
        for name, expected in frozen.items():
            if family_tensor_record(current[name].tensor) != expected:
                raise RuntimeError(f"frozen family tensor changed: {name}")
        for name, expected in packed_records.items():
            if quantized_state_record(packed[name]) != expected:
                raise RuntimeError(f"frozen family NF4 weight or metadata changed: {name}")
        for name, expected in buffers.items():
            if family_tensor_record(buffer_slots[name].tensor) != expected:
                raise RuntimeError(f"frozen family buffer changed: {name}")

    net._cosmos_adapter_parameters = dict(sorted(trainable.items()))
    net._native_adapter_parameter_names = tuple(sorted(trainable))
    net._cosmos_adapter_metadata = identity
    net._cosmos_validate_frozen_base = validate_frozen
    codec = AdapterCheckpointCodec(net, identity=identity)
    net._cosmos_adapter_checkpoint_codec = codec
    guard_resident_movement(net, device)
    for module in (*installed.values(), *installed_heads.values()):
        guard_resident_movement(module, device)
    return FamilyBackboneState(cfg, installed, identity, codec)
