# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Resolved architecture-bound adapters; preserves forward and base state-dict names.

LoRA and rsLoRA reuse the native implementation. DoRA adds its row magnitude;
PiSSA represents the frozen residual as W0 - s B0 A0, retaining W0 unchanged.
No generic causal-LM wrapper, forward monkey patch, or second training loop.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from fnmatch import fnmatchcase
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.model.adapter_families import AdapterFamily, adapter_family, family_for_model
from cosmos_framework.utils.generator.lora import (
    LoraInjectedLinear,
    init_lora_weights_post_materialization,
    lora_effective_weight,
)


def canonical_name(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


def component_of(name: str, family: str = "qwen3_vl") -> str | None:
    location = adapter_family(family).locate(name)
    return location[0] if location is not None else None


def _selected(net: nn.Module, group: Any, family: AdapterFamily) -> dict[str, nn.Module]:
    modules = dict(net.named_modules(remove_duplicate=False))
    selected: dict[str, nn.Module] = {}
    layers = getattr(group, "layers", None)
    for selector in group.targets:
        matches = {n: m for n, m in modules.items() if fnmatchcase(n, selector)}
        wrong = [n for n in matches if (family.locate(n) or (None,))[0] != group.component]
        if wrong:
            raise ValueError(f"selector {selector!r} selects outside {group.component}: {wrong}")
        if layers is not None:
            if any(family.locate(n)[1] is None for n in matches):
                raise ValueError(f"selector {selector!r} cannot apply layer restrictions to non-layer modules")
            matches = {n: m for n, m in matches.items() if family.locate(n)[1] in layers}
        if not matches:
            raise ValueError(f"adapter selector {selector!r} matched no modules after component/layer restrictions")
        selected.update(matches)
    if layers is not None:
        missing = set(layers) - {family.locate(n)[1] for n in selected}
        if missing:
            raise ValueError(f"adapter layer indices matched no modules: {sorted(missing)}")
    return selected


def resolve_targets(net: nn.Module, config: AdapterConfig, *, family: str | None = None) -> dict[str, Any]:
    """Resolve and validate every selector and alias before mutating anything."""
    config.__post_init__()
    binding = family_for_model(net, family)
    resolved: dict[str, dict[str, Any]] = {}
    selected_modules: dict[str, nn.Module] = {}
    for group in config.components:
        for name, module in _selected(net, group, binding).items():
            if type(module) is not nn.Linear:
                raise ValueError(
                    f"unsupported adapter target {name}: {type(module).__name__}; requires plain nn.Linear"
                )
            if group.rank > min(module.weight.shape):
                raise ValueError(f"rank {group.rank} exceeds target dimensions at {name}: {tuple(module.weight.shape)}")
            settings = asdict(group)
            for key in ("component", "targets", "layers"):
                settings.pop(key)
            # Preserve the exact identities of existing artifacts/checkpoints.
            # Absence means the original factorized execution, never dense mode.
            if settings["execution"] == "factorized_v1":
                settings.pop("execution")
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
        for name, module in _selected(net, group, binding).items():
            if not isinstance(module, nn.LayerNorm) and type(module).__name__ not in binding.normalization_types:
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
        **({"model_execution": config.model_execution} if config.model_execution != "default_v1" else {}),
    }


def validate_adapter_profile(cfg: Any, model_config: Any, pdims: Any, adapters: AdapterConfig, resume: bool) -> None:
    adapters.__post_init__()
    adapter_family(getattr(model_config, "model_type", None))
    if getattr(model_config, "quantization_config", None) is not None:
        raise ValueError("quantized adapter training is a separate contribution; use a dense BF16/FP32 base")
    modes = (
        ("none", "full", "selective") if model_config.model_type in ("qwen3_vl", "qwen3_vl_moe") else ("none", "full")
    )
    if cfg.dtype not in ("float32", "bfloat16") or cfg.activation_checkpointing not in modes:
        raise ValueError(f"adapters require float32/bfloat16 base storage and activation_checkpointing in {modes}")
    if cfg.attn_impl not in ("eager", "sdpa"):
        raise ValueError("posttrain adapters require attn_impl=eager|sdpa; other attention backends are not qualified")
    validate_adapter_parallelism(pdims)
    if resume and adapters.warm_start:
        raise ValueError(
            "exact training resume and adapter warm_start are distinct operations; clear warm_start for resume"
        )


def validate_adapter_parallelism(pdims: Any) -> None:
    """Shared DP-only contract for native/HF/physical adapter builders."""
    if pdims is not None:
        replicate, shard = int(pdims.dp_replicate), int(pdims.dp_shard)
        if replicate < 1 or shard < 1 or replicate * shard != int(pdims.world_size):
            raise ValueError("adapter DP mesh requires positive dp_replicate * dp_shard == world_size")
        if any(int(getattr(pdims, axis, 1)) != 1 for axis in ("tp", "ep", "cp", "pp", "cfgp")):
            raise ValueError("adapters support FSDP2/HSDP data parallelism only; TP/EP/CP/PP/CFG axes must be one")


def pissa_dense_weight(base, a, b, initial_a, initial_b, scale):
    """Version-1 dense execution: shared differentiable train/inference/export math.

    W0 + s * ((B-B0) A + B0 (A-A0)) represents the same PiSSA residual
    W0-s*B0*A0 without mutating W0 or materializing a second persistent base.
    FP64 products precede one FP32 rounding; the caller then applies its base
    compute dtype. At initialization the correction is exactly zero. This
    deliberately pays for dense effective weights while optimizing only A/B.
    """
    with torch.autocast(device_type=base.device.type, enabled=False):
        a64, b64 = a.double(), b.double()
        a0, b0 = initial_a.double(), initial_b.double()
        delta = (b64 - b0) @ a64 + b0 @ (a64 - a0)
        return (base.double() + delta * scale).float()


def merged_execution_weight(base, a, b, scale, *, method, magnitude=None, initial_a=None, initial_b=None):
    """Versioned new-mode construction, identical for train/portable/export.

    FP64 low-rank products and DoRA normalization avoid CPU/CUDA FP32 reduction
    noise crossing a BF16 rounding boundary. Round once to FP32 before the
    caller's declared base compute dtype. Reuse native LoRA and dense PiSSA math.
    """
    if method == "pissa":
        return pissa_dense_weight(base, a, b, initial_a, initial_b, scale)
    direction = lora_effective_weight(base.double(), a, b, scale, dtype=torch.float64)
    if method == "dora":
        norm = direction.norm(dim=1).clamp_min(torch.finfo(torch.float64).tiny).detach()
        direction = direction * (magnitude.double() / norm)[:, None]
    return direction.float()


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
        self.execution = config.execution
        if self.method == "dora":
            self.magnitude = nn.Parameter(torch.ones(base.out_features, device=base.weight.device, dtype=torch.float32))
        if self.method == "pissa":
            self.register_buffer("initial_A", torch.zeros_like(self.lora_A.weight))
            self.register_buffer("initial_B", torch.zeros_like(self.lora_B.weight))
        # Even resume placeholders are initialized; exact restoration remains strict.
        init_lora_weights_post_materialization(self)
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

    def adapter_output(self, x: torch.Tensor) -> torch.Tensor:
        # Reuse the native low-rank math, with FP32 adapter accumulation even
        # when a family owns an autocast forward around its frozen base.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().adapter_output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution == "merged_weight_bf16_v1":
            weight = self.effective_weight().to(self.weight.dtype)
            return F.linear(x, weight, self.bias)
        if self.method == "lora":
            return super().forward(x)
        if self.execution == "pissa_dense_v1":
            weight = pissa_dense_weight(
                self.weight, self.lora_A.weight, self.lora_B.weight, self.initial_A, self.initial_B, self._lora_scale
            ).to(self.weight.dtype)
            # Keep the family's Linear autocast/bias boundary. Low-rank A/B
            # remain the only trainable leaves; no merged weight is registered.
            return F.linear(x, weight, self.bias)
        # Preserve the family's base autocast boundary (e.g. FP32 GR00T master
        # weights or Alpamayo's FP32 Fourier inputs feeding BF16 projections).
        base = F.linear(x, self.weight, self.bias)
        dropped = self.lora_dropout(x)
        base_dropped = F.linear(dropped, self.weight, None) if self.method == "dora" else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = self.lora_B(self.lora_A(dropped.float())) * self._lora_scale
            if self.method == "pissa":
                initial = F.linear(F.linear(x.float(), self.initial_A), self.initial_B) * self._lora_scale
                return (base.float() + (delta - initial)).to(base.dtype)
            # DoRA's norm is detached as in the reference algorithm. Bias is not
            # scaled. Reuse the native merge math for the directional weight.
            direction = super().merged_weight(self.weight, self.lora_A.weight, self.lora_B.weight, dtype=torch.float32)
            norm = direction.norm(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
            ratio = self.magnitude / norm
            return (base.float() + (ratio - 1) * base_dropped.float() + ratio * delta).to(base.dtype)

    def effective_weight(self) -> torch.Tensor:
        """Differentiable dense values; only factors/magnitudes are trainable leaves."""
        return merged_execution_weight(
            self.weight,
            self.lora_A.weight,
            self.lora_B.weight,
            self._lora_scale,
            method=self.method,
            magnitude=getattr(self, "magnitude", None),
            initial_a=getattr(self, "initial_A", None),
            initial_b=getattr(self, "initial_B", None),
        )

    @torch.no_grad()
    def merged_state_weight(self, state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        base = state[prefix + ".weight"]
        a, b = state[prefix + ".lora_A.weight"], state[prefix + ".lora_B.weight"]
        if self.execution == "merged_weight_bf16_v1":
            return merged_execution_weight(
                base,
                a,
                b,
                self._lora_scale,
                method=self.method,
                magnitude=state.get(prefix + ".magnitude"),
                initial_a=state.get(prefix + ".initial_A"),
                initial_b=state.get(prefix + ".initial_B"),
            ).to(base.dtype)
        if self.execution == "pissa_dense_v1":
            return pissa_dense_weight(
                base, a, b, state[prefix + ".initial_A"], state[prefix + ".initial_B"], self._lora_scale
            ).to(base.dtype)
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


def _preserve_normalization_output_dtype(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
    # Native RMSNorm multiplies its FP32 trainable weight after restoring the
    # activation dtype. Preserve the base forward boundary for the next BF16
    # projection while keeping the normalization parameter/update in FP32.
    return output.to(inputs[0].dtype)


def install_adapters(
    net: nn.Module,
    config: AdapterConfig,
    *,
    initialize: bool = True,
    distributed: bool = False,
    family: str | None = None,
    fsdp_block_types: tuple[type, ...] | None = None,
) -> dict[str, Any]:
    binding = family_for_model(net, family)
    manifest = resolve_targets(net, config, family=binding.name)
    from cosmos_framework.posttrain.model.adapter_moe_execution import install_model_execution, validate_model_execution

    validate_model_execution(net, config, binding.name)
    if distributed:

        def fsdp_unit(name: str) -> str:
            if fsdp_block_types is None:
                return binding.locate(name)[2]
            roots = [
                n
                for n, m in net.named_modules(remove_duplicate=False)
                if n and isinstance(m, fsdp_block_types) and name.startswith(n + ".")
            ]
            return max(roots, key=len, default="root")

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
    extra_modules = set()
    for row in manifest["extra_parameters"]:
        parameter = parameters[row["name"]]
        parameter.data = parameter.data.float()
        parameter.requires_grad_(True)
        module = net.get_submodule(row["name"].rsplit(".", 1)[0])
        if id(module) not in extra_modules:
            if config.model_execution == "merged_weight_bf16_v1":
                from cosmos_framework.posttrain.model.adapter_bf16_execution import install_effective_norm

                install_effective_norm(module)
            else:
                module.register_forward_hook(_preserve_normalization_output_dtype)
            extra_modules.add(id(module))
    net._posttrain_adapter_manifest = manifest
    install_model_execution(net, config)
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


def prepare_adapter_model(
    net: nn.Module,
    config: AdapterConfig,
    *,
    base: str,
    processor: str,
    base_dtype: str,
    media: dict,
    resume: bool = False,
    distributed: bool = False,
    family: str | None = None,
    fsdp_block_types: tuple[type, ...] | None = None,
    source_identity: dict | None = None,
) -> None:
    """Shared loaded-base -> adapter/trainability/artifact seam, before FSDP.

    Family builders own construction, forward semantics and sharding. This seam
    performs no model forward, no training loop and no optimizer construction.
    """
    from cosmos_framework.posttrain.checkpoint.adapters import load_adapter_artifact, make_adapter_identity

    if resume and config.warm_start:
        raise ValueError("exact resume and adapter warm_start are mutually exclusive")
    if config.model_execution == "merged_weight_bf16_v1" and base_dtype != "bfloat16":
        raise ValueError("new merged-weight training requires a BF16 materialized base and FP32 adapter storage")
    binding = family_for_model(net, family)
    install_adapters(
        net,
        config,
        initialize=not resume and not config.warm_start,
        distributed=distributed,
        family=binding.name,
        fsdp_block_types=fsdp_block_types,
    )
    net._posttrain_adapter_identity = make_adapter_identity(
        base,
        processor,
        net._posttrain_adapter_manifest,
        media,
        base_dtype=base_dtype,
        architecture=binding.name,
        source_identity=source_identity,
    )
    if config.warm_start:
        load_adapter_artifact(net, config.warm_start)


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
