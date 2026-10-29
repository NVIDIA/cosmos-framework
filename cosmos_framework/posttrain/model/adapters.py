# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Resolved native Qwen3-VL adapters; preserves forward and base state-dict names.

LoRA and rsLoRA reuse the native implementation. DoRA adds its row magnitude;
PiSSA represents the frozen residual as W0 - s B0 A0, retaining W0 unchanged.
No generic causal-LM wrapper, forward monkey patch, or second training loop.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from fnmatch import fnmatchcase
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.utils.generator.lora import LoraInjectedLinear


def canonical_name(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


def component_of(name: str) -> str | None:
    if re.match(r"model\.language_model\.layers\.\d+\.", name) or name == "model.language_model.norm":
        return "language"
    if re.match(r"model\.visual\.(merger(?:\.|$)|deepstack_merger_list\.\d+(?:\.|$))", name):
        return "projector"
    if re.match(r"model\.visual\.blocks\.\d+\.", name):
        return "vision"
    return None


def _selected(net: nn.Module, group: Any) -> dict[str, nn.Module]:
    modules = dict(net.named_modules(remove_duplicate=False))
    selected: dict[str, nn.Module] = {}
    layers = getattr(group, "layers", None)
    for selector in group.targets:
        matches = {n: m for n, m in modules.items() if fnmatchcase(n, selector)}
        wrong = [n for n in matches if component_of(n) != group.component]
        if wrong:
            raise ValueError(f"selector {selector!r} selects outside {group.component}: {wrong}")
        if layers is not None:
            matches = {n: m for n, m in matches.items() if int(n.split(".")[3]) in layers}
        if not matches:
            raise ValueError(f"adapter selector {selector!r} matched no modules after component/layer restrictions")
        selected.update(matches)
    if layers is not None:
        missing = set(layers) - {int(n.split(".")[3]) for n in selected}
        if missing:
            raise ValueError(f"adapter layer indices matched no modules: {sorted(missing)}")
    return selected


def resolve_targets(net: nn.Module, config: AdapterConfig) -> dict[str, Any]:
    """Resolve and validate every selector and alias before mutating anything."""
    config.__post_init__()
    resolved: dict[str, dict[str, Any]] = {}
    selected_modules: dict[str, nn.Module] = {}
    for group in config.components:
        for name, module in _selected(net, group).items():
            if type(module) is not nn.Linear:
                raise ValueError(
                    f"unsupported adapter target {name}: {type(module).__name__}; requires plain nn.Linear"
                )
            if group.rank > min(module.weight.shape):
                raise ValueError(f"rank {group.rank} exceeds target dimensions at {name}: {tuple(module.weight.shape)}")
            settings = asdict(group)
            for key in ("component", "targets", "layers"):
                settings.pop(key)
            row = dict(
                component=group.component,
                name=name,
                type="torch.nn.Linear",
                shape=list(module.weight.shape),
                settings=settings,
                scaling=group.alpha / (math.sqrt(group.rank) if group.use_rslora else group.rank),
                trainable_count=group.rank * sum(module.weight.shape)
                + (module.out_features if group.method == "dora" else 0),
            )
            if name in resolved and resolved[name] != row:
                raise ValueError(f"conflicting adapter settings for {name}")
            resolved[name] = row
            selected_modules[name] = module

    # Parameter identity, not path spelling, defines ownership. All aliases must be
    # explicitly covered; injecting a tied weight through only one consumer changes semantics.
    aliases: dict[int, list[str]] = {}
    for name, parameter in net.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    owners: dict[int, str] = {}
    for name in sorted(resolved):
        module = selected_modules[name]
        names = aliases[id(module.weight)]
        if any(n.removesuffix(".weight") not in resolved or not n.endswith(".weight") for n in names):
            raise ValueError(f"target {name} shares a weight with unselected/unsupported aliases: {names}")
        owner = owners.setdefault(id(module.weight), name)
        if (
            resolved[owner]["settings"] != resolved[name]["settings"]
            or resolved[owner]["component"] != resolved[name]["component"]
        ):
            raise ValueError(f"incompatible settings for tied aliases {owner} and {name}")
        resolved[name]["owner"] = owner

    extras: dict[str, dict[str, Any]] = {}
    extra_parameter_owners: dict[int, str] = {}
    for group in config.extra_modules:
        for name, module in _selected(net, group).items():
            if not isinstance(module, nn.LayerNorm) and type(module).__name__ != "Qwen3VLTextRMSNorm":
                raise ValueError(
                    f"extra trained module {name} must be a native LayerNorm/RMSNorm; linear/embedding extras are unsupported"
                )
            parameters = list(module.named_parameters())
            if not parameters:
                raise ValueError(f"extra module {name} contains no parameters")
            for leaf, parameter in parameters:
                full_name = f"{name}.{leaf}"
                row = dict(
                    name=full_name,
                    component=group.component,
                    shape=list(parameter.shape),
                    dtype="float32",
                    lr_multiplier=group.lr_multiplier,
                    weight_decay=group.weight_decay,
                )
                if full_name in extras and {k: v for k, v in extras[full_name].items() if k != "owner"} != row:
                    raise ValueError(f"conflicting extra module settings for {full_name}")
                owner = extra_parameter_owners.setdefault(id(parameter), full_name)
                row["owner"] = owner
                extras[full_name] = row
    for parameter_id, owner in extra_parameter_owners.items():
        for alias in aliases[parameter_id]:
            if alias not in extras:
                raise ValueError(f"extra module {owner} has unselected tied alias {alias}")
            if any(extras[owner][key] != extras[alias][key] for key in ("component", "lr_multiplier", "weight_decay")):
                raise ValueError(f"incompatible extra module aliases {owner} and {alias}")
    expected = sum(row["trainable_count"] for row in resolved.values() if row["owner"] == row["name"])
    expected += sum(math.prod(row["shape"]) for row in extras.values() if row["owner"] == row["name"])
    return {
        "targets": [resolved[n] for n in sorted(resolved)],
        "extra_parameters": [extras[n] for n in sorted(extras)],
        "expected_trainable_parameters": expected,
    }


def validate_adapter_profile(cfg: Any, model_config: Any, pdims: Any, adapters: AdapterConfig, resume: bool) -> None:
    adapters.__post_init__()
    if getattr(model_config, "model_type", None) != "qwen3_vl":
        raise ValueError(
            "posttrain adapters support native dense qwen3_vl only; HF and MoE/expert projections require separate qualification"
        )
    if getattr(model_config, "quantization_config", None) is not None:
        raise ValueError("quantized adapter training is a separate contribution; use a dense BF16/FP32 base")
    if cfg.dtype not in ("float32", "bfloat16") or cfg.activation_checkpointing not in ("none", "full"):
        raise ValueError("adapters require float32/bfloat16 base storage and activation_checkpointing=none|full")
    if pdims is not None and int(pdims.dp_replicate) != 1:
        raise ValueError("adapter distributed profile supports pure FSDP2 dp_replicate=1; HSDP is not qualified")
    if resume and adapters.warm_start:
        raise ValueError(
            "exact training resume and adapter warm_start are distinct operations; clear warm_start for resume"
        )


class PosttrainAdapterLinear(LoraInjectedLinear):
    """Native LoRA plus bounded DoRA/PiSSA extensions with pure merged export."""

    def __init__(self, base: nn.Linear, config: AdapterComponentConfig, *, initialize: bool = True) -> None:
        super().__init__(
            base,
            config.rank,
            config.alpha,
            dropout=config.dropout,
            use_rslora=config.use_rslora,
            adapter_dtype=torch.float32,
        )
        self.method = config.method
        if self.method == "dora":
            self.magnitude = nn.Parameter(torch.ones(base.out_features, device=base.weight.device, dtype=torch.float32))
        if self.method == "pissa":
            self.register_buffer("initial_A", torch.zeros_like(self.lora_A.weight))
            self.register_buffer("initial_B", torch.zeros_like(self.lora_B.weight))
        # Even resume placeholders are initialized; exact restoration remains strict.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if initialize:
            self.initialize_method(config)

    @torch.no_grad()
    def initialize_method(self, config: AdapterComponentConfig) -> None:
        if self.weight.is_meta:
            raise ValueError("adapter initialization requires loaded, materialized base weights")
        if self.method == "dora":
            self.magnitude.copy_(self.weight.float().norm(dim=1))
        elif self.method == "pissa":
            with torch.autocast(device_type=self.weight.device.type, enabled=False):
                weight = self.weight.float()
                if config.initialization == "svd":
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                    u, s, v = u[:, : config.rank], s[: config.rank], vh[: config.rank].T
                else:
                    u, s, v = torch.svd_lowrank(weight, q=config.rank, niter=config.pissa_niter)
                root = (s / self._lora_scale).sqrt()
                self.lora_A.weight.copy_(root[:, None] * v.T)
                self.lora_B.weight.copy_(u * root[None, :])
                self.initial_A.copy_(self.lora_A.weight)
                self.initial_B.copy_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FSDP keeps adapter storage FP32; disable enclosing autocast for the native
        # low-rank math. The frozen base and its input retain their compute dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.method == "lora":
                return super().forward(x)
            base = F.linear(x, self.weight, self.bias)
            dropped = self.lora_dropout(x)
            delta = self.lora_B(self.lora_A(dropped.float())) * self._lora_scale
            if self.method == "pissa":
                initial = F.linear(F.linear(x.float(), self.initial_A), self.initial_B) * self._lora_scale
                return (base.float() + (delta - initial)).to(base.dtype)
            # DoRA's norm is detached as in the reference algorithm. Bias is not
            # scaled. Reuse the native merge math for the directional weight.
            direction = super().merged_weight(self.weight, self.lora_A.weight, self.lora_B.weight, dtype=torch.float32)
            norm = direction.norm(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
            ratio = self.magnitude / norm
            base_dropped = F.linear(dropped, self.weight, None).float()
            return (base.float() + (ratio - 1) * base_dropped + ratio * delta).to(base.dtype)

    @torch.no_grad()
    def merged_state_weight(self, state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        base = state[prefix + ".weight"]
        a, b = state[prefix + ".lora_A.weight"], state[prefix + ".lora_B.weight"]
        if self.method == "pissa":
            # The PiSSA paper's rank-2r conversion to the untouched base. The
            # effective scale remains s; do not reinterpret it as alpha/(2r).
            a = torch.cat((a, state[prefix + ".initial_A"]), dim=0)
            b = torch.cat((b, -state[prefix + ".initial_B"]), dim=1)
        merged = super().merged_weight(base, a, b, dtype=torch.float32)
        if self.method == "dora":
            merged = (
                merged
                * (
                    state[prefix + ".magnitude"].float() / merged.norm(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
                )[:, None]
            )
        return merged.to(base.dtype)

    def merge_adapter_(self) -> None:
        raise RuntimeError("posttrain adapters require non-mutating merged export; in-place merge is disabled")


def install_adapters(
    net: nn.Module, config: AdapterConfig, *, initialize: bool = True, distributed: bool = False
) -> dict[str, Any]:
    manifest = resolve_targets(net, config)
    if distributed:

        def fsdp_unit(name: str) -> str:
            return ".".join(name.split(".")[:4]) if component_of(name) in ("language", "vision") else "root"

        for row in [*manifest["targets"], *manifest["extra_parameters"]]:
            if fsdp_unit(row["name"]) != fsdp_unit(row["owner"]):
                raise ValueError(
                    f"tied adapter/extra parameters cross FSDP units: {row['name']}, {row['owner']}; use one GPU"
                )
    if any(p.is_meta for p in net.parameters()):
        raise ValueError("install adapters after materializing/loading base and before distributed registration")
    modules = dict(net.named_modules(remove_duplicate=False))
    installed: dict[int, PosttrainAdapterLinear] = {}
    owner_layers: dict[str, PosttrainAdapterLinear] = {}
    for row in manifest["targets"]:
        base = modules[row["name"]]
        layer = installed.get(id(base))
        if layer is None:
            settings = AdapterComponentConfig(component=row["component"], targets=[row["name"]], **row["settings"])
            layer = PosttrainAdapterLinear(base, settings, initialize=initialize and row["owner"] == row["name"])
            if row["owner"] != row["name"]:
                owner = owner_layers[row["owner"]]
                layer.lora_A, layer.lora_B = owner.lora_A, owner.lora_B
                if layer.method == "dora":
                    layer.magnitude = owner.magnitude
                if layer.method == "pissa":
                    layer.initial_A, layer.initial_B = owner.initial_A, owner.initial_B
            installed[id(base)] = layer
        owner_layers[row["name"]] = layer
        parent, _, leaf = row["name"].rpartition(".")
        setattr(net.get_submodule(parent), leaf, layer)
    net.requires_grad_(False)
    for layer in installed.values():
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
        if layer.method == "dora":
            layer.magnitude.requires_grad_(True)
    parameters = dict(net.named_parameters(remove_duplicate=False))
    for row in manifest["extra_parameters"]:
        parameter = parameters[row["name"]]
        parameter.data = parameter.data.float()
        parameter.requires_grad_(True)
    net._posttrain_adapter_manifest = manifest
    assert_trainability(net)
    return manifest


def trainable_spec(manifest: dict[str, Any]) -> dict[str, tuple[float, float]]:
    expected = {}
    for row in manifest["targets"]:
        for leaf in ["lora_A.weight", "lora_B.weight"] + (["magnitude"] if row["settings"]["method"] == "dora" else []):
            expected[row["name"] + "." + leaf] = (row["settings"]["lr_multiplier"], row["settings"]["weight_decay"])
    for row in manifest["extra_parameters"]:
        expected[row["name"]] = (row["lr_multiplier"], row["weight_decay"])
    return expected


def assert_trainability(net: nn.Module, optimizer: Any = None) -> None:
    expected = set(trainable_spec(net._posttrain_adapter_manifest))
    parameters = {canonical_name(n): p for n, p in net.named_parameters(remove_duplicate=False)}
    actual = {n for n, p in parameters.items() if p.requires_grad}
    if expected != actual:
        raise ValueError(
            f"adapter trainability mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if any(p.is_meta for p in parameters.values()):
        raise ValueError("unmaterialized parameter remains after adapter installation")
    if any(p.dtype != torch.float32 for p in parameters.values() if p.requires_grad):
        raise ValueError("adapter trainable storage must remain float32 after distributed transformation")
    count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if count != net._posttrain_adapter_manifest["expected_trainable_parameters"]:
        raise ValueError(f"adapter trainable count {count} differs from resolved target manifest")
    if optimizer is not None:
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in parameters.values() if p.requires_grad}:
            raise ValueError("optimizer membership differs from adapter trainability or contains duplicates")


def optimizer_groups(net: nn.Module, lr: float) -> tuple[Any, ...]:
    from cosmos_framework.posttrain.config.build import OptimizerParameterGroup

    assert_trainability(net)
    spec = trainable_spec(net._posttrain_adapter_manifest)
    groups: dict[tuple[float, float], list[str]] = {}
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            groups.setdefault(spec[canonical_name(name)], []).append(name)
    return tuple(
        OptimizerParameterGroup(f"adapter_{i}", tuple(names), lr * multiplier, decay)
        for i, ((multiplier, decay), names) in enumerate(groups.items())
    )
