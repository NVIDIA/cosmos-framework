# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from cosmos_framework.posttrain.config.model import (
    NATIVE_ADAPTER_LANGUAGE_LAYERS,
    NATIVE_ADAPTER_TARGET_PROFILE,
    QWEN3_VL_8B_TEXT_ARCHITECTURE,
    quantization_config_identity,
    validate_quantized_linear_execution,
)
from cosmos_framework.utils.generator.lora import LoraInjectedLinear, NativeLoRAMixin

PROFILE = NATIVE_ADAPTER_TARGET_PROFILE


def runtime_precision_policy() -> dict[str, Any]:
    """Observe backend math policy without changing global state or initializing CUDA.

    Tensor dtypes alone do not specify FP32 matmul arithmetic, so the observed
    flags are part of the saved execution policy.
    """
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp32_precision": {
            "global": torch.backends.fp32_precision,
            "cuda_matmul": torch.backends.cuda.matmul.fp32_precision,
            "cudnn": torch.backends.cudnn.fp32_precision,
            "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
            "cudnn_rnn": torch.backends.cudnn.rnn.fp32_precision,
        },
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "environment": {
            name: os.environ.get(name) for name in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
        },
    }


MODEL_IDENTITY_FORMAT = "cosmos.adapter-model.v1"


def resolved_execution_policy(cfg: Any) -> dict[str, Any]:
    """Bind execution settings to the observed math policy."""
    from cosmos_framework.posttrain.config.model import native_adapter_execution_policy

    return {**native_adapter_execution_policy(cfg), "math_precision": runtime_precision_policy()}


def _validate_precision_record(record: Any) -> None:
    expected = runtime_precision_policy()
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError("adapter execution_policy requires complete math_precision metadata")
    fp32 = record["fp32_precision"]
    environment = record["environment"]
    if (
        record["float32_matmul_precision"] not in ("highest", "high", "medium")
        or not isinstance(fp32, dict)
        or set(fp32) != set(expected["fp32_precision"])
        or any(value not in ("none", "ieee", "tf32") for value in fp32.values())
        or any(
            type(record[name]) is not bool
            for name in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "allow_bf16_reduced_precision_reduction")
        )
        or not isinstance(environment, dict)
        or set(environment) != set(expected["environment"])
        or any(value is not None and not isinstance(value, str) for value in environment.values())
    ):
        raise ValueError("adapter execution_policy contains unsupported math_precision metadata")


def execution_config_from_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Recover explicit native model options from a saved model identity, failing closed.

    The caller must still call ``validate_execution_policy_identity`` before
    construction. Math settings are checked against the process, never silently
    changed while another model may share it.
    """
    from cosmos_framework.posttrain.config.model import (
        QuantizedBaseTrainingConfig,
        native_adapter_execution_policy,
        quantization_config_identity,
    )

    if identity.get("format") != MODEL_IDENTITY_FORMAT:
        raise ValueError(f"unsupported native adapter model identity format; expected {MODEL_IDENTITY_FORMAT}")
    validate_quantization_coverage_identity(identity)
    adapter = identity.get("adapter", {})
    if not isinstance(adapter, dict):
        raise ValueError("native adapter model identity has malformed adapter metadata")
    if adapter.get("initialization") == "nf4_residual_svd":
        from cosmos_framework.posttrain.model.quantization.initialization import METHOD, validate_initialization_record

        validate_initialization_record(identity.get("initialization"))
        if (
            adapter.get("initialization") != METHOD
            or adapter.get("initialization_artifact_sha256") != identity["initialization"]["artifact_manifest_sha256"]
            or identity.get("quantization") != quantization_config_identity(QuantizedBaseTrainingConfig())
        ):
            raise ValueError("residual initialization identity requires its exact adapter artifact and NF4 base")
    elif "initialization" in identity:
        raise ValueError("an initialization record requires residual adapter initialization")
    policy = identity.get("execution_policy")
    if not isinstance(policy, dict):
        raise ValueError("adapter execution_policy must be a complete mapping")
    try:
        options = {
            "activation_checkpointing": policy["activation_checkpointing"]["mode"],
            "adapter_compute_dtype": policy["adapter_compute_dtype"],
            "sft_logits_mode": policy["sft_logits_mode"],
            "compile_language_model": policy["language_compile"]["enabled"],
            "quantized_linear_execution": policy["quantized_linear_execution"],
        }
        if (
            options["activation_checkpointing"] not in ("none", "full", "selective")
            or options["adapter_compute_dtype"] not in ("float32", "bfloat16")
            or options["sft_logits_mode"] not in ("full", "supervised")
            or type(options["compile_language_model"]) is not bool
        ):
            raise ValueError("unsupported adapter execution options")
        if options["quantized_linear_execution"] == "checkpointed_dequant":
            if (
                identity.get("quantization") != quantization_config_identity(QuantizedBaseTrainingConfig())
                or adapter.get("initialization") != "kaiming_A_zero_B"
            ):
                raise ValueError("checkpointed_dequant requires an ordinary adapter and a full NF4 base")
            validate_quantized_linear_execution(
                SimpleNamespace(
                    **options,
                    adapter=SimpleNamespace(**adapter),
                    quantized_base=QuantizedBaseTrainingConfig(),
                )
            )
        configured = native_adapter_execution_policy(SimpleNamespace(**options))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("unsupported or incomplete adapter execution_policy") from error
    if {key: value for key, value in policy.items() if key != "math_precision"} != configured:
        raise ValueError("unsupported or incomplete adapter execution_policy")
    _validate_precision_record(policy.get("math_precision"))
    if adapter.get("initialization") == "nf4_residual_svd" and options["adapter_compute_dtype"] != "float32":
        raise ValueError("residual initialization requires FP32 adapter computation")
    return options


def validate_execution_policy_identity(cfg: Any, identity: dict[str, Any]) -> None:
    """Reject changed/missing execution or math settings before allocating a base."""
    validate_quantized_linear_execution(cfg)
    execution_config_from_identity(identity)
    expected = json.loads(json.dumps(quantization_config_identity(getattr(cfg, "quantized_base", None))))
    if identity.get("quantization") != expected:
        raise ValueError("adapter checkpoint quantization differs from configuration")
    if identity.get("execution_policy") != resolved_execution_policy(cfg):
        raise ValueError("adapter checkpoint execution policy or runtime math precision differs from configuration")


def canonical_targets() -> dict[str, tuple[int, int]]:
    """An explicit complete allowlist, never a suffix/leaf rule on the VLM."""
    text = QWEN3_VL_8B_TEXT_ARCHITECTURE
    hidden, intermediate = text["hidden_size"], text["intermediate_size"]
    query = text["num_attention_heads"] * text["head_dim"]
    key_value = text["num_key_value_heads"] * text["head_dim"]
    shapes = {
        "self_attn.q_proj": (query, hidden),
        "self_attn.k_proj": (key_value, hidden),
        "self_attn.v_proj": (key_value, hidden),
        "self_attn.o_proj": (hidden, query),
        "mlp.gate_proj": (intermediate, hidden),
        "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }
    return {
        f"model.language_model.layers.{layer}.{projection}": shape
        for layer in range(NATIVE_ADAPTER_LANGUAGE_LAYERS)
        for projection, shape in shapes.items()
    }


def quantization_coverage(quantization: Any, targets: dict[str, Any]) -> dict[str, Any]:
    """Partition complete adapter targets without changing their names or trainability."""
    layers = tuple(getattr(quantization, "excluded_language_layers", ()))
    # Revalidate even callers that changed an already-created config object.
    if quantization is not None:
        quantization_config_identity(quantization)
        layers = quantization.excluded_language_layers
    dense = set()
    canonical = canonical_targets()
    for layer in layers:
        prefix = f"model.language_model.layers.{layer}."
        expected = {name for name in canonical if name.startswith(prefix)}
        actual = {name for name in targets if name.startswith(prefix)}
        if not expected or actual != expected:
            raise ValueError("excluded language layer does not match its exact canonical targets")
        dense.update(expected)
    return {
        "protocol": "cosmos.native-quantization-coverage.v1",
        "quantized_targets": sorted(set(targets) - dense) if quantization is not None else [],
        "dense_language_targets": sorted(dense if quantization is not None else targets),
    }


def validate_quantization_coverage_identity(identity: dict[str, Any]) -> None:
    """Validate the optional record of language layers kept dense in an NF4 base."""
    from cosmos_framework.posttrain.config.model import QuantizedBaseTrainingConfig

    quantization = identity.get("quantization")
    excluded = isinstance(quantization, dict) and "excluded_language_layers" in quantization
    if not excluded:
        if "quantization_coverage" in identity:
            raise ValueError("quantization_coverage requires excluded language layers")
        return
    try:
        cfg = QuantizedBaseTrainingConfig(excluded_language_layers=quantization["excluded_language_layers"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid mixed-base quantization configuration") from error
    if not cfg.excluded_language_layers or quantization != json.loads(json.dumps(quantization_config_identity(cfg))):
        raise ValueError("mixed NF4/BF16 base requires canonical nonempty exclusions")
    expected = quantization_coverage(cfg, canonical_targets())
    frozen = identity.get("excluded_frozen_tensors")
    packed = identity.get("quantized_tensors")
    if (
        identity.get("quantization_coverage") != expected
        or identity.get("targets") != {name: list(shape) for name, shape in sorted(canonical_targets().items())}
        or not isinstance(packed, dict)
        or set(packed) != set(expected["quantized_targets"])
        or not isinstance(frozen, dict)
        or not {name + ".weight" for name in expected["dense_language_targets"]} <= set(frozen)
        or {name + ".weight" for name in expected["quantized_targets"]} & set(frozen)
        or not isinstance(identity.get("adapter"), dict)
        or identity["adapter"].get("initialization") != "kaiming_A_zero_B"
        or "initialization" in identity
    ):
        raise ValueError("mixed-base identity coverage, frozen inventory or initialization differs")


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
    """Make ``to``/``cuda``/``cpu`` no-ops on ``device`` and refuse every other conversion.

    Packed NF4 weights and the mixed BF16/FP32 component policy must never be moved, cast or
    re-materialized by a blanket ``Module`` conversion. Same-device placement returns the module
    unchanged, preserving Parameter identity and QuantState.
    """
    resident = torch.device(device)

    def same_device(target: Any) -> bool:
        if target is None:
            return True
        target = torch.device(target)
        if target.type != resident.type:
            return False
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        return target == resident

    def refuse(*_args: Any, **_kwargs: Any) -> nn.Module:
        raise ValueError("native adapter model is resident; blanket conversion, offload and to_empty are unsupported")

    def to(self: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
        target, dtype, _non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None or memory_format is not None or not same_device(target):
            raise ValueError("native adapter model does not permit changing component dtype, device or format")
        return self

    def cuda(self: nn.Module, device: Any = None) -> nn.Module:
        target = torch.device("cuda", device) if isinstance(device, int) else torch.device(device or "cuda")
        if not same_device(target):
            raise ValueError("native adapter model does not permit changing component device")
        return self

    def cpu(self: nn.Module) -> nn.Module:
        if resident.type != "cpu":
            raise ValueError("native adapter model does not permit changing component device")
        return self

    module.to = MethodType(to, module)
    module.cuda = MethodType(cuda, module)
    module.cpu = MethodType(cpu, module)
    module._apply = MethodType(refuse, module)


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


def safetensor_inventory(directory: Path) -> dict[str, tuple[Path, tuple[int, ...], str]]:
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
    validate_quantized_linear_execution(cfg)
    adapter, quantization = cfg.adapter, cfg.quantized_base
    if adapter is None:
        raise ValueError("quantized base training requires ordinary native LoRA")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError(
            "native adapter training requires strict deterministic algorithms; "
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
        or cfg.activation_checkpointing not in ("full", "none", "selective")
    ):
        raise ValueError(
            "native adapter training supports dense Qwen3-VL with BF16, SDPA and none|full|selective "
            "activation checkpointing"
        )
    if getattr(cfg, "adapter_compute_dtype", "float32") not in ("float32", "bfloat16"):
        raise ValueError("native adapter compute dtype must be float32 or bfloat16")
    if getattr(cfg, "sft_logits_mode", "full") not in ("full", "supervised"):
        raise ValueError("native adapter SFT logits mode must be full or supervised")
    if type(getattr(cfg, "compile_language_model", False)) is not bool:
        raise ValueError("native language compile must be an explicit boolean")
    if bool(getattr(model_config, "tie_word_embeddings", False)):
        raise ValueError("native adapter training does not support tied embedding/output weights")
    text = model_config.text_config
    for field, value in QWEN3_VL_8B_TEXT_ARCHITECTURE.items():
        if getattr(text, field, None) != value:
            raise ValueError(f"{PROFILE} requires {field}={value}")
    adapter.__post_init__()
    if quantization is not None:
        quantization_config_identity(quantization)
    if adapter.initialization == "nf4_residual_svd" and (
        quantization is None
        or getattr(cfg, "adapter_compute_dtype", "float32") != "float32"
        or adapter.rank > 1024
        or quantization.excluded_language_layers
    ):
        raise ValueError(
            "residual initialization requires full NF4 coverage, FP32 adapter compute and rank at most 1024"
        )
    if not dist.is_initialized() or dist.get_world_size() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("native adapter training requires torchrun with exactly one process")
    if pdims is not None:
        raise ValueError("native adapter training does not support FSDP/DTensor wrapping")
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not torch.cuda.is_available():
        raise ValueError("native adapter training requires Linux x86-64 and CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.current_device() != int(os.environ.get("LOCAL_RANK", "0")):
        raise ValueError("CUDA ownership differs from native torchrun LOCAL_RANK")
    require_native_bf16_gpu(device, "native adapter training")
    return device


def require_native_bf16_gpu(device: torch.device, purpose: str) -> None:
    """Require hardware BF16 (compute capability 8.0+, no emulation); warn off the tested A100 80GB.

    ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True`` and so also
    accepts pre-Ampere GPUs (V100/T4) that only emulate BF16.
    """
    if torch.cuda.get_device_capability(device) < (8, 0) or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError(f"{purpose} requires a CUDA GPU with compute capability 8.0+ and BF16 kernels")
    props = torch.cuda.get_device_properties(device)
    if "A100" not in props.name or props.total_memory < 79_000_000_000:
        from cosmos_framework.utils import log

        log.warning(f"{purpose} is tested on A100 80GB only; running on untested {props.name}")


class NativeAdapterLifecycle:
    """Construction hooks retaining canonical names and final live parameter owners."""

    def __init__(self, cfg: Any, model_config: Any, pdims: Any) -> None:
        validate_quantized_linear_execution(cfg)
        from cosmos_framework.posttrain.model.quantization.source import validate_adapter_source
        from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase

        record_qlora_phase("adapter_loading_start", reset_peak=True)
        self.source_identity = validate_adapter_source(cfg)
        self.device = validate_runtime(cfg, model_config, pdims)
        self.cfg = cfg
        self.adapter_compute_dtype = getattr(torch, getattr(cfg, "adapter_compute_dtype", "float32"))
        self.runtime_precision = runtime_precision_policy()
        self.execution_policy = resolved_execution_policy(cfg)
        self.quantized = cfg.quantized_base is not None
        self.initialization_artifact = None
        if cfg.adapter.initialization == "nf4_residual_svd":
            from cosmos_framework.posttrain.model.quantization.initialization import InitializationArtifact

            self.initialization_artifact = InitializationArtifact(
                cfg.adapter.initialization_artifact,
                cfg.adapter.initialization_artifact_sha256,
                adapter=cfg.adapter,
                source_identity=self.source_identity,
                quantization=cfg.quantized_base,
            )
        if self.quantized:
            from cosmos_framework.posttrain.model.quantization.bnb import kernel_preflight

            kernel_preflight(
                self.device,
                cfg.adapter,
                adapter_compute_dtype=self.adapter_compute_dtype,
                compile_model=getattr(cfg, "compile_language_model", False),
                quantized_linear_execution=getattr(cfg, "quantized_linear_execution", "bnb"),
            )
        self.targets: dict[str, NativeLoRAMixin] = {}
        self.slots: dict[str, TensorSlot] = {}
        self.buffers: list[TensorSlot] = []
        self.loaded = False

    def prepare(self, net: nn.Module) -> None:
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        targets = resolve_targets(net)
        self.coverage = quantization_coverage(self.cfg.quantized_base, targets)
        self.quantized_targets = {}
        quantized_names = set(self.coverage["quantized_targets"])
        for name, base in targets.items():
            if not base.weight.is_meta:
                raise ValueError("native adapters must be inserted before base materialization")
            if name in quantized_names:
                from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit

                replacement = NativeLoraLinear4bit(
                    base,
                    self.cfg.adapter,
                    adapter_compute_dtype=self.adapter_compute_dtype,
                    quantized_linear_execution=getattr(self.cfg, "quantized_linear_execution", "bnb"),
                )
            else:
                replacement = LoraInjectedLinear(
                    base,
                    self.cfg.adapter.rank,
                    self.cfg.adapter.alpha,
                    dropout=self.cfg.adapter.dropout,
                    adapter_dtype=torch.float32,
                    adapter_compute_dtype=self.adapter_compute_dtype,
                )
            parent, _, attribute = name.rpartition(".")
            net.get_submodule(parent).register_module(attribute, replacement)
            self.targets[name] = replacement
            if name in quantized_names:
                self.quantized_targets[name] = replacement
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
        self.inventory = safetensor_inventory(Path(self.cfg.name_or_path))
        if set(self.inventory) != set(self.source_slots):
            missing = sorted(set(self.source_slots) - set(self.inventory))[:10]
            extra = sorted(set(self.inventory) - set(self.source_slots))[:10]
            raise ValueError(f"base source tensor inventory differs: missing={missing}, extra={extra}")
        for name, (_, shape, dtype) in self.inventory.items():
            if shape != tuple(self.source_slots[name].tensor.shape) or dtype != "BF16":
                raise ValueError(f"base source logical shape/BF16 dtype mismatch: {name}: {shape}, {dtype}")

    def materialize(self, net: nn.Module) -> None:
        skip = {f"{name}.weight" for name in self.quantized_targets}
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
                    if module_name in self.quantized_targets:
                        self.targets[module_name].load_dense_weight_once(source, self.device)
                        if self.initialization_artifact is not None:
                            self.initialization_artifact.apply(module_name, self.targets[module_name], source)
                    else:
                        self.source_slots[name].tensor.copy_(source)
                    del source
        self.loaded = True

    def finalize(self, net: nn.Module) -> None:
        if not self.loaded:
            raise RuntimeError("cannot finalize adapters before every frozen base tensor is loaded")
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.adapter.initialization_seed)
        if self.initialization_artifact is not None:
            self.initialization_artifact.assert_complete()
        parameters: dict[str, nn.Parameter] = {}
        for name, module in sorted(self.targets.items()):
            if self.initialization_artifact is None:
                module.reset_adapter_parameters(generator=generator)
            parameters.update({f"{name}.{key}": value for key, value in module.adapter_parameters().items()})
        expected_ids = {id(parameter) for parameter in parameters.values()}
        expected_count = 2 * len(canonical_targets())
        if len(parameters) != expected_count or len(expected_ids) != expected_count:
            raise RuntimeError(f"native LoRA must have exactly {expected_count} distinct trainable A/B parameters")
        quantized_ids = {id(module.weight) for module in self.quantized_targets.values()}
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
        quantized_records = {
            name: quantized_state_record(module) for name, module in sorted(self.quantized_targets.items())
        }
        frozen = {
            name: tensor_digest(slot.tensor)
            for name, slot in sorted(self.source_slots.items())
            if name.removesuffix(".weight") not in self.quantized_targets
        }
        buffer_records = {name: tensor_digest(slot.tensor) for name, slot in sorted(self.buffer_slots.items())}
        from cosmos_framework.posttrain.config.model import native_adapter_config_identity

        metadata = {
            "format": MODEL_IDENTITY_FORMAT,
            "base": self.source_identity,
            "model_format": "cosmos_native_qwen3_vl",
            "adapter": native_adapter_config_identity(self.cfg.adapter),
            "quantization": quantization_config_identity(self.cfg.quantized_base),
            "targets": {name: list(shape) for name, shape in sorted(canonical_targets().items())},
            "excluded_frozen_tensors": frozen,
            "base_buffers": buffer_records,
            "quantized_tensors": quantized_records,
            "initial_adapters": {name: tensor_digest(parameter) for name, parameter in parameters.items()},
            "trainable_elements": sum(parameter.numel() for parameter in parameters.values()),
            "strategy": {
                "world_size": 1,
                "kind": "single_device",
                "activation_checkpointing": (
                    "none"
                    if self.cfg.activation_checkpointing == "none"
                    else f"{self.cfg.activation_checkpointing}_nonreentrant"
                ),
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
            "execution_policy": self.execution_policy,
        }
        if self.initialization_artifact is not None:
            metadata["initialization"] = self.initialization_artifact.identity()
        if getattr(self.cfg.quantized_base, "excluded_language_layers", ()):
            metadata["quantization_coverage"] = self.coverage
            validate_quantization_coverage_identity(json.loads(json.dumps(metadata)))
        # Plain mappings keep references without registering a second model subtree.
        net._cosmos_adapter_parameters = parameters
        net._native_adapter_parameter_names = tuple(parameters)
        net._cosmos_adapter_metadata = metadata
        net._cosmos_adapter_runtime_precision = self.runtime_precision

        def validate_runtime_state() -> None:
            if runtime_precision_policy() != self.runtime_precision:
                raise RuntimeError("native adapter runtime math precision changed after model construction")

        def validate_frozen_base() -> None:
            validate_runtime_state()
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
        net._cosmos_validate_runtime_state = validate_runtime_state
        guard_resident_movement(net, self.device)
        for module in self.targets.values():
            guard_resident_movement(module, self.device)
        from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec

        net._cosmos_adapter_checkpoint_codec = build_adapter_checkpoint_codec(net, identity=metadata)
        from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase

        record_qlora_phase("adapter_loading_complete", reset_peak=True)
