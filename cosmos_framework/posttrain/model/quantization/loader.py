# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded native Qwen3-VL adapter loading, shared by BF16 and NF4 bases.

No full-model state_dict, dense GPU destination for quantized matrices, global
dtype conversion, or whole-checkpoint CPU dictionary is used here. Slots hold
direct owner references captured before activation-checkpoint wrappers exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from cosmos_framework.utils.generator.lora import LoraInjectedLinear, NativeLoRAMixin

PROFILE = "qwen3_vl_8b_language_v1"
REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SOURCE_ELEMENTS = 6_945_767_424


def canonical_targets() -> dict[str, tuple[int, int]]:
    """An explicit complete allowlist, never a suffix/leaf rule on the VLM."""
    shapes = {
        "self_attn.q_proj": (4096, 4096),
        "self_attn.k_proj": (1024, 4096),
        "self_attn.v_proj": (1024, 4096),
        "self_attn.o_proj": (4096, 4096),
        "mlp.gate_proj": (12288, 4096),
        "mlp.up_proj": (12288, 4096),
        "mlp.down_proj": (4096, 12288),
    }
    return {
        f"model.language_model.layers.{layer}.{projection}": shape
        for layer in range(36)
        for projection, shape in shapes.items()
    }


def resolve_targets(net: nn.Module) -> dict[str, nn.Linear]:
    """Resolve exact names/shapes and reject any alias into an excluded component."""
    targets: dict[str, nn.Linear] = {}
    aliases: dict[int, list[str]] = {}
    for name, parameter in net.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for name, shape in canonical_targets().items():
        try:
            module = net.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"native LoRA target is missing: {name}") from error
        if type(module) is not nn.Linear or tuple(module.weight.shape) != shape or module.bias is not None:
            raise ValueError(f"native LoRA target {name} must be a bias-free Linear of logical shape {shape}")
        if aliases[id(module.weight)] != [f"{name}.weight"]:
            raise ValueError(f"native LoRA target aliases another base component: {name}")
        targets[name] = module
    if len(targets) != 252 or sum(module.weight.numel() for module in targets.values()) != SOURCE_ELEMENTS:
        raise ValueError("Qwen3-VL-8B language target coverage differs from its fixed profile")
    return targets


def tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    """Hash original bytes in bounded host chunks, preserving packed storage."""
    if tensor.is_meta or not tensor.is_contiguous():
        raise ValueError("fingerprinting requires materialized contiguous tensors")
    plain = tensor.detach().as_subclass(torch.Tensor)
    raw = plain.reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), 4 * 1024 * 1024):
        chunk = raw[offset : offset + 4 * 1024 * 1024].cpu()
        digest.update(chunk.numpy().tobytes())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": digest.hexdigest(),
    }


def _quant_state_record(state: Any) -> dict[str, Any]:
    if state is None:
        raise ValueError("missing NF4 QuantState")
    result: dict[str, Any] = {
        "shape": list(state.shape) if state.shape is not None else None,
        "dtype": str(state.dtype).removeprefix("torch."),
        "quant_type": state.quant_type,
        "blocksize": state.blocksize,
        "nested": bool(state.nested),
        "absmax": tensor_digest(state.absmax),
        "code": tensor_digest(state.code),
    }
    if state.nested:
        if state.offset is None or state.state2 is None:
            raise ValueError("missing nested NF4 scale metadata")
        result["offset"] = tensor_digest(state.offset)
        result["state2"] = _quant_state_record(state.state2)
    return result


def quantized_state_record(module: Any) -> dict[str, Any]:
    module.validate_quantization_state()
    return {"weight": tensor_digest(module.weight), "quant_state": _quant_state_record(module.weight.quant_state)}


def guard_resident_movement(module: nn.Module, device: torch.device) -> None:
    """Permit no-op placement only; reject blanket casts/offload before mutation.

    Return the existing module for same-device placement, preserving Parameter
    identity and QuantState. Arbitrary _apply/to_empty transformations must never
    run over packed weights or the mixed BF16/FP32 component policy.
    """

    def guarded_apply(self: nn.Module, function: Any, recurse: bool = True) -> nn.Module:
        if getattr(function, "__qualname__", "") not in {
            "Module.to.<locals>.convert",
            "Module.cuda.<locals>.<lambda>",
        }:
            raise ValueError(
                "native adapter model is resident; blanket conversion, offload and to_empty are unsupported"
            )
        closure = dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))
        if closure.get("convert_to_format") is not None:
            raise ValueError("native adapter model does not permit changing tensor memory format")
        for dtype in (torch.uint8, torch.bfloat16, torch.float32):
            probe = torch.empty(0, device=device, dtype=dtype)
            result = function(probe)
            if result.device != device or result.dtype != dtype:
                raise ValueError("native adapter model does not permit changing component dtype or device")
        return self

    module._apply = MethodType(guarded_apply, module)


@dataclass(frozen=True)
class TensorSlot:
    owner: nn.Module
    attribute: str

    @property
    def tensor(self) -> torch.Tensor:
        return getattr(self.owner, self.attribute)


def parameter_slots(net: nn.Module) -> dict[str, TensorSlot]:
    result: dict[str, TensorSlot] = {}
    for name, owner in net.named_modules(remove_duplicate=False):
        for attribute, parameter in owner.named_parameters(recurse=False, remove_duplicate=False):
            key = f"{name}.{attribute}" if name else attribute
            if key in result:
                raise ValueError(f"duplicate canonical parameter name: {key}")
            result[key] = TensorSlot(owner, attribute)
    return result


def materialize_slots(
    parameters: dict[str, TensorSlot], buffers: list[TensorSlot], *, device: torch.device, skip: set[str]
) -> None:
    """Materialize ordinary slots once; preserve initialized nonpersistent buffers."""
    allocated: dict[int, nn.Parameter] = {}
    for name, slot in parameters.items():
        if name in skip:
            if not slot.tensor.is_meta:
                raise ValueError(f"quantized destination was materialized early: {name}")
            continue
        tensor = slot.tensor
        if not tensor.is_meta:
            raise ValueError(f"native adapter construction requires meta parameter slots: {name}")
        if id(tensor) not in allocated:
            allocated[id(tensor)] = nn.Parameter(
                torch.empty(tuple(tensor.shape), device=device, dtype=tensor.dtype), requires_grad=tensor.requires_grad
            )
        setattr(slot.owner, slot.attribute, allocated[id(tensor)])
    moved: dict[int, torch.Tensor] = {}
    for slot in buffers:
        tensor = slot.tensor
        if tensor.is_meta:
            raise ValueError("native rotary/other buffers must have initialized values before materialization")
        if id(tensor) not in moved:
            moved[id(tensor)] = tensor.to(device=device)
        setattr(slot.owner, slot.attribute, moved[id(tensor)])


def _safetensor_inventory(directory: Path) -> dict[str, tuple[Path, tuple[int, ...], str]]:
    """Read only safetensors headers; fail duplicates and corrupt index mappings."""
    from safetensors import safe_open

    result: dict[str, tuple[Path, tuple[int, ...], str]] = {}
    paths = sorted(directory.glob("*.safetensors"))
    if not paths:
        raise ValueError("native adapters require a complete local HF safetensors snapshot")
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in result:
                    raise ValueError(f"duplicate source tensor {name}")
                tensor_slice = reader.get_slice(name)
                result[name] = (path, tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())["weight_map"]
        if set(index) != set(result) or any(index[name] != path.name for name, (path, _, _) in result.items()):
            raise ValueError("safetensors index and actual source tensor inventory disagree")
    return result


def validate_runtime(cfg: Any, model_config: Any, pdims: Any) -> torch.device:
    """Reject unsupported rows before any base-model allocation."""
    adapter, quantization = cfg.adapter, cfg.quantized_base
    if adapter is None:
        raise ValueError("quantized base training requires ordinary native LoRA")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError(
            "native adapter qualification requires strict deterministic algorithms; "
            "call enable_determinism(strict=True) before CUDA initialization"
        )
    if getattr(cfg, "credential_path", ""):
        raise ValueError(
            "native adapter training loads a verified local snapshot; remote credential loading is unsupported"
        )
    if (
        getattr(model_config, "model_type", None) != "qwen3_vl"
        or cfg.dtype != "bfloat16"
        or cfg.attn_impl != "sdpa"
        or cfg.activation_checkpointing != "full"
    ):
        raise ValueError(
            "native adapter qualification requires dense Qwen3-VL, BF16, SDPA and full activation checkpointing"
        )
    if bool(getattr(model_config, "tie_word_embeddings", False)):
        raise ValueError("native adapter qualification excludes tied embeddings/output weights")
    text = model_config.text_config
    expected = {
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    for field, value in expected.items():
        if getattr(text, field, None) != value:
            raise ValueError(f"{PROFILE} requires {field}={value}")
    if (
        adapter.method != "lora"
        or adapter.parameter_dtype != "float32"
        or adapter.target_profile != PROFILE
        or adapter.bias != "none"
        or adapter.modules_to_save
        or adapter.initialization != "kaiming_A_zero_B"
        or adapter.rank < 1
        or adapter.alpha <= 0
        or not 0 <= adapter.dropout < 1
    ):
        raise ValueError("unsupported native ordinary LoRA configuration")
    if not dist.is_initialized() or dist.get_world_size() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("native adapter qualification requires torchrun with exactly one process")
    if pdims is not None:
        raise ValueError("native adapter qualification must not invoke FSDP/DTensor wrapping")
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not torch.cuda.is_available():
        raise ValueError("native adapter qualification requires Linux x86-64 and CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.current_device() != int(os.environ.get("LOCAL_RANK", "0")):
        raise ValueError("CUDA ownership differs from native torchrun LOCAL_RANK")
    props = torch.cuda.get_device_properties(device)
    if (
        torch.cuda.get_device_capability(device) != (8, 0)
        or "A100" not in props.name
        or props.total_memory < 79_000_000_000
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("initial native adapter qualification requires one A100 80GB with BF16 kernels")
    if quantization is not None:
        expected_quant = {
            "backend": "bitsandbytes",
            "backend_version": "0.49.2",
            "format": "nf4",
            "block_size": 64,
            "nested": True,
            "storage_dtype": "uint8",
            "compute_dtype": "bfloat16",
            "source_dtype": "bfloat16",
            "target_profile": PROFILE,
            "offload": "none",
        }
        if any(getattr(quantization, key, None) != value for key, value in expected_quant.items()):
            raise ValueError("unsupported native NF4 backend, representation, dtype, targets, or offload")
    return device


class NativeAdapterLifecycle:
    """Construction hooks retaining canonical names and final live parameter owners."""

    def __init__(self, cfg: Any, model_config: Any, pdims: Any) -> None:
        from cosmos_framework.posttrain.model.quantization.source import validate_adapter_source
        from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase

        record_qlora_phase("adapter_loading_start", reset_peak=True)
        self.source_identity = validate_adapter_source(cfg)
        self.device = validate_runtime(cfg, model_config, pdims)
        self.cfg = cfg
        self.quantized = cfg.quantized_base is not None
        self.canary = None
        if self.quantized:
            from cosmos_framework.posttrain.model.quantization.bnb import kernel_preflight

            self.canary = kernel_preflight(self.device, cfg.adapter, cfg.quantized_base)
        self.targets: dict[str, NativeLoRAMixin] = {}
        self.slots: dict[str, TensorSlot] = {}
        self.buffers: list[TensorSlot] = []
        self.loaded = False

    def prepare(self, net: nn.Module) -> None:
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        targets = resolve_targets(net)
        for name, base in targets.items():
            if not base.weight.is_meta:
                raise ValueError("native adapters must be inserted before base materialization")
            if self.quantized:
                from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit

                replacement = NativeLoraLinear4bit(base, self.cfg.adapter, self.cfg.quantized_base)
            else:
                replacement = LoraInjectedLinear(
                    base,
                    self.cfg.adapter.rank,
                    self.cfg.adapter.alpha,
                    dropout=self.cfg.adapter.dropout,
                    adapter_dtype=torch.float32,
                )
            parent, _, attribute = name.rpartition(".")
            net.get_submodule(parent).register_module(attribute, replacement)
            self.targets[name] = replacement
        self.slots = parameter_slots(net)
        self.buffer_slots = {
            f"{name}.{attribute}" if name else attribute: TensorSlot(owner, attribute)
            for name, owner in net.named_modules()
            for attribute, tensor in owner._buffers.items()
            if tensor is not None
        }
        self.buffers = list(self.buffer_slots.values())
        self.source_slots = {
            name: slot for name, slot in self.slots.items() if not name.endswith((".lora_A.weight", ".lora_B.weight"))
        }
        self.inventory = _safetensor_inventory(Path(self.cfg.name_or_path))
        if set(self.inventory) != set(self.source_slots):
            missing = sorted(set(self.source_slots) - set(self.inventory))[:10]
            extra = sorted(set(self.inventory) - set(self.source_slots))[:10]
            raise ValueError(f"base source tensor inventory differs: missing={missing}, extra={extra}")
        for name, (_, shape, dtype) in self.inventory.items():
            if shape != tuple(self.source_slots[name].tensor.shape) or dtype != "BF16":
                raise ValueError(f"base source logical shape/BF16 dtype mismatch: {name}: {shape}, {dtype}")

    def materialize(self, net: nn.Module) -> None:
        skip = {f"{name}.weight" for name in self.targets} if self.quantized else set()
        materialize_slots(self.slots, self.buffers, device=self.device, skip=skip)

    @torch.no_grad()
    def load(self, net: nn.Module, checkpoint_path: str, credential_path: str | None, pdims: Any) -> None:
        from safetensors import safe_open

        if self.loaded or pdims is not None or Path(checkpoint_path).resolve() != Path(self.cfg.name_or_path).resolve():
            raise ValueError("native adapter loader requires one unsharded load of its verified base snapshot")
        for path in sorted({value[0] for value in self.inventory.values()}):
            with safe_open(path, framework="pt", device="cpu") as reader:
                for name in sorted(reader.keys()):
                    source = reader.get_tensor(name)
                    module_name = name.removesuffix(".weight")
                    if self.quantized and module_name in self.targets:
                        self.targets[module_name].load_dense_weight_once(source, self.device)
                    else:
                        self.source_slots[name].tensor.copy_(source)
                    del source
        self.loaded = True

    def finalize(self, net: nn.Module) -> None:
        if not self.loaded:
            raise RuntimeError("cannot finalize adapters before every frozen base tensor is loaded")
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.adapter.initialization_seed)
        parameters: dict[str, nn.Parameter] = {}
        for name, module in sorted(self.targets.items()):
            module.reset_adapter_parameters(generator=generator)
            parameters.update({f"{name}.{key}": value for key, value in module.adapter_parameters().items()})
        expected_ids = {id(parameter) for parameter in parameters.values()}
        if len(parameters) != 504 or len(expected_ids) != 504:
            raise RuntimeError("native LoRA must have exactly 504 distinct trainable A/B parameters")
        quantized_ids = {id(module.weight) for module in self.targets.values()} if self.quantized else set()
        for parameter in net.parameters():
            is_adapter = id(parameter) in expected_ids
            expected_dtype = (
                torch.float32 if is_adapter else torch.uint8 if id(parameter) in quantized_ids else torch.bfloat16
            )
            if (
                parameter.is_meta
                or parameter.device != self.device
                or parameter.requires_grad != is_adapter
                or parameter.dtype != expected_dtype
            ):
                raise RuntimeError("native adapter final tensor dtype/device/trainability contract failed")
        for buffer in net.buffers():
            if buffer.is_meta or buffer.device != self.device:
                raise RuntimeError("native adapter construction left an unmaterialized/misplaced buffer")
        quantized_records = (
            {name: quantized_state_record(module) for name, module in sorted(self.targets.items())}
            if self.quantized
            else {}
        )
        frozen = {
            name: tensor_digest(slot.tensor)
            for name, slot in sorted(self.source_slots.items())
            if not (self.quantized and name.removesuffix(".weight") in self.targets)
        }
        buffer_records = {name: tensor_digest(slot.tensor) for name, slot in sorted(self.buffer_slots.items())}
        metadata = {
            "format": "cosmos.adapter-model.v1",
            "base": self.source_identity,
            "model_format": "cosmos_native_qwen3_vl",
            "adapter": asdict(self.cfg.adapter),
            "quantization": asdict(self.cfg.quantized_base) if self.quantized else None,
            "targets": {name: list(shape) for name, shape in sorted(canonical_targets().items())},
            "excluded_frozen_tensors": frozen,
            "base_buffers": buffer_records,
            "quantized_tensors": quantized_records,
            "initial_adapters": {name: tensor_digest(parameter) for name, parameter in parameters.items()},
            "trainable_elements": sum(parameter.numel() for parameter in parameters.values()),
            "strategy": {
                "world_size": 1,
                "kind": "single_device",
                "activation_checkpointing": "full_nonreentrant",
                "determinism_mode": "strict",
            },
            "precision": {"base_dense": "bfloat16", "adapter": "float32", "optimizer_moments": "float32"},
            "hardware": {
                "name": torch.cuda.get_device_name(self.device),
                "capability": list(torch.cuda.get_device_capability(self.device)),
            },
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "dependencies": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("torch", "transformers", "safetensors", "accelerate")},
                **({"bitsandbytes": version("bitsandbytes")} if self.quantized else {}),
            },
        }
        # Plain mappings keep references without registering a second model subtree.
        net._cosmos_adapter_parameters = parameters
        net._native_adapter_parameter_names = tuple(parameters)
        net._cosmos_adapter_metadata = metadata
        net._cosmos_quantization_canary = self.canary

        def validate_frozen_base() -> None:
            for name, expected in frozen.items():
                if tensor_digest(self.source_slots[name].tensor) != expected:
                    raise RuntimeError(f"frozen native base tensor changed during adapter training: {name}")
            for name, expected in quantized_records.items():
                if quantized_state_record(self.targets[name]) != expected:
                    raise RuntimeError(f"packed base weight or quantization state changed during training: {name}")
            for name, expected in buffer_records.items():
                if tensor_digest(self.buffer_slots[name].tensor) != expected:
                    raise RuntimeError(f"native frozen buffer changed during adapter training: {name}")

        net._cosmos_validate_frozen_base = validate_frozen_base
        guard_resident_movement(net, self.device)
        for module in self.targets.values():
            guard_resident_movement(module, self.device)
        from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec

        net._cosmos_adapter_checkpoint_codec = build_adapter_checkpoint_codec(net, identity=metadata)
        from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase

        record_qlora_phase("adapter_loading_complete", reset_peak=True)
