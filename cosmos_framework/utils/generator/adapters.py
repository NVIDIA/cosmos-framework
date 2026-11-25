# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Parameter-efficient adapters for frozen ``nn.Linear`` layers: LoRA, rsLoRA, DoRA and PiSSA.

Adapters are installed in place with :class:`AdapterLinear`. The wrapped base weight keeps its
``<path>.weight`` name, so the frozen base loads unchanged; the trainable state is the
``lora_A`` / ``lora_B`` siblings plus, per method:

* LoRA:  ``y = W x + s B A x`` with ``s = alpha / r`` (rsLoRA: ``s = alpha / sqrt(r)``).
* DoRA:  ``y = m / ||W + s B A||_row * (W + s B A) x``; ``<path>.lora_magnitude`` is trained and
  the row norm is detached, as in the DoRA paper. The bias is not rescaled.
* PiSSA: ``A``/``B`` start from the top-``r`` singular triplets of ``W``. The frozen base stays
  unmodified; the residual ``W - s B0 A0`` is represented with the initial factors kept as the
  buffers ``<path>.initial_A`` / ``<path>.initial_B``, so ``y = W x + s (B A - B0 A0) x``.

Adapter parameters are always stored and computed in float32, also when the base is bfloat16.
``AdapterLinear`` deliberately does not subclass the generic
:class:`~cosmos_framework.utils.generator.lora.LoraInjectedLinear`: the generic LoRA merge and
export helpers know only ``W + s B A`` and must not treat DoRA or PiSSA as plain LoRA.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init

from cosmos_framework.utils.generator.adapter_config import (
    AdapterArchitecture,
    AdapterComponentConfig,
    AdapterConfig,
    AdapterIdentity,
    AdapterManifest,
    AdapterResumeSettings,
    AdapterTarget,
    AdapterTargetIdentity,
    AdapterTensorIdentity,
    adapter_architecture,
)

_PISSA_OVERSAMPLE = 8  # Extra randomized-SVD columns, truncated to the rank afterwards.
_TENSOR_KEYS = ("method", "rank", "alpha", "use_rslora")


def canonical_name(name: str) -> str:
    """Strip activation-checkpoint and compile wrapper segments from a parameter/module name.

    The same rule as ``model/generator/vlm_model.py``'s helper, which cannot be imported here
    without the full VLM model stack.
    """
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


def architecture_for_model(net: nn.Module, architecture: str | None = None) -> AdapterArchitecture:
    """Resolve the binding from an explicit architecture or the model's ``config.model_type``."""
    declared = getattr(getattr(net, "config", None), "model_type", None)
    if architecture is not None and declared is not None and architecture != declared:
        raise ValueError(f"adapter architecture {architecture!r} disagrees with the model's {declared!r}")
    return adapter_architecture(architecture or declared)


def _scaling(row: Mapping[str, Any]) -> float:
    return row["alpha"] / (math.sqrt(row["rank"]) if row["use_rslora"] else row["rank"])


def _randomized_svd(
    weight: torch.Tensor, rank: int, *, niter: int, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-``rank`` singular triplets by randomized range finding (Halko et al.), oversampled then truncated.

    Like ``torch.svd_lowrank`` with power iterations, but drawing from ``generator``.
    """
    q = min(rank + _PISSA_OVERSAMPLE, *weight.shape)
    sketch = torch.randn(weight.shape[1], q, generator=generator, device=weight.device, dtype=weight.dtype)
    basis = torch.linalg.qr(weight @ sketch).Q
    for _ in range(niter):
        basis = torch.linalg.qr(weight @ torch.linalg.qr(weight.T @ basis).Q).Q
    u, s, vh = torch.linalg.svd(basis.T @ weight, full_matrices=False)
    return (basis @ u)[:, :rank], s[:rank], vh[:rank].T


class AdapterLinear(nn.Linear):
    """``nn.Linear`` with a float32 LoRA/DoRA/PiSSA adapter; the base weight stays frozen in place."""

    def __init__(self, base: nn.Linear, config: AdapterComponentConfig, *, initialize: bool = True) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None, device="meta")
        self.weight = base.weight
        self.bias = base.bias
        device = base.weight.device
        # skip_init allocates without drawing from the global RNG; initialize() or a restore fills the factors.
        self.lora_A = skip_init(
            nn.Linear, base.in_features, config.rank, bias=False, device=device, dtype=torch.float32
        )
        self.lora_B = skip_init(
            nn.Linear, config.rank, base.out_features, bias=False, device=device, dtype=torch.float32
        )
        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.config = config
        self.method = config.method
        self.scaling = config.scaling
        if self.method == "dora":
            self.lora_magnitude = nn.Parameter(torch.ones(base.out_features, device=device, dtype=torch.float32))
        if self.method == "pissa":
            self.register_buffer("initial_A", torch.zeros_like(self.lora_A.weight))
            self.register_buffer("initial_B", torch.zeros_like(self.lora_B.weight))
        if initialize:
            self.initialize()
        else:
            # Placeholders for a restored state: finite and deterministic.
            nn.init.zeros_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)

    @torch.no_grad()
    def initialize(self, generator: torch.Generator | None = None) -> None:
        """Method-specific initialization from the loaded (materialized) base weight.

        Random draws use ``generator`` (on the weight's device), or the global RNG when it is ``None``.
        """
        config = self.config
        if self.weight.is_meta:
            raise ValueError("adapter initialization requires a loaded, materialized base weight")
        if self.method != "pissa":
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5), generator=generator)
            nn.init.zeros_(self.lora_B.weight)
            if self.method == "dora":
                self.lora_magnitude.copy_(torch.linalg.vector_norm(self.weight, dim=1, dtype=torch.float32))
            return
        weight = self.weight.float()
        rank = config.rank
        if config.pissa_initialization == "svd":
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            u, s, v = u[:, :rank], s[:rank], vh[:rank].T
        else:
            u, s, v = _randomized_svd(weight, rank, niter=config.pissa_niter, generator=generator)
        root = (s / self.scaling).sqrt()
        self.lora_A.weight.copy_(root[:, None] * v.T)
        self.lora_B.weight.copy_(u * root[None, :])
        self.initial_A.copy_(self.lora_A.weight)
        self.initial_B.copy_(self.lora_B.weight)

    def adapter_tensors(self) -> list[torch.Tensor]:
        """Every tensor a portable adapter or an initialization broadcast carries, in a fixed order."""
        tensors = [self.lora_A.weight, self.lora_B.weight]
        tensors += [self.lora_magnitude] if self.method == "dora" else []
        tensors += [self.initial_A, self.initial_B] if self.method == "pissa" else []
        return tensors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dropout = self.training and not isinstance(self.lora_dropout, nn.Identity)
        base = F.linear(x, self.weight)  # Keep the base matmul in the caller's dtype/autocast context.
        dropped = self.lora_dropout(x) if dropout else x
        with torch.autocast(device_type=x.device.type, enabled=False):
            adapter_input = dropped.float()
            delta = self.lora_B(self.lora_A(adapter_input)) * self.scaling
            if self.method == "lora":
                out = base.float() + delta
            elif self.method == "pissa":
                # PiSSA requires zero dropout, so both branches share one float32 input.
                initial = F.linear(F.linear(adapter_input, self.initial_A), self.initial_B) * self.scaling
                out = base.float() + (delta - initial)
            else:
                # The row norm of W + s B A is a constant of the backward pass (DoRA, section 4.3).
                direction = (self.lora_B.weight.detach() @ self.lora_A.weight.detach()).mul_(self.scaling)
                norm = direction.add_(self.weight.detach()).norm(dim=1).detach()
                ratio = self.lora_magnitude / norm.clamp_min(torch.finfo(torch.float32).tiny)
                base_dropped = F.linear(dropped, self.weight).float() if dropout else base.float()
                out = base.float() + (ratio - 1) * base_dropped + ratio * delta
            if self.bias is not None:
                out = out + self.bias.float()
        return out.to(base.dtype)


@torch.no_grad()
def merged_target_weight(row: Mapping[str, Any], base: torch.Tensor, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Dense float32 weight equivalent to one adapted target of a resolved manifest or identity ``row``.

    ``state`` holds the adapter tensors under ``<name>.lora_A.weight`` etc. and ``base`` is the
    frozen base weight as stored for training. Accumulates in float64; nothing is mutated.
    """
    prefix, scale = row["name"], _scaling(row)
    a, b = state[f"{prefix}.lora_A.weight"].double(), state[f"{prefix}.lora_B.weight"].double()
    merged = base.double() + scale * (b @ a)
    if row["method"] == "pissa":
        merged -= scale * (state[f"{prefix}.initial_B"].double() @ state[f"{prefix}.initial_A"].double())
    if row["method"] == "dora":
        norm = merged.norm(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
        merged *= (state[f"{prefix}.lora_magnitude"].double() / norm)[:, None]
    return merged.float()


def _select(modules: Mapping[str, nn.Module], group: AdapterComponentConfig, binding: AdapterArchitecture) -> list[str]:
    located = {name: binding.locate(name) for name in modules}
    layers = group.layers
    if layers is not None and any(i < 0 for i in layers):
        count = 1 + max(
            (w[1] for w in located.values() if w and w[0] == group.component and w[1] is not None), default=-1
        )
        layers = [i + count if i < 0 else i for i in layers]
        if any(i < 0 for i in layers) or len(set(layers)) != len(layers):
            raise ValueError(f"adapter layers {group.layers} are invalid for {count} {group.component} blocks")
    selected: list[str] = []
    for selector in group.targets:
        matches = [name for name in modules if fnmatchcase(name, selector)]
        outside = [n for n in matches if located[n] is None or located[n][0] != group.component]
        if outside:
            raise ValueError(f"adapter selector {selector!r} selects modules outside {group.component}: {outside[:5]}")
        if layers is not None:
            matches = [n for n in matches if located[n][1] in layers]
        if not matches:
            raise ValueError(f"adapter selector {selector!r} matched no {group.component} modules")
        selected.extend(name for name in matches if name not in selected)
    if layers is not None:
        missing = set(layers) - {located[name][1] for name in selected}
        if missing:
            raise ValueError(f"adapter layers matched no modules: {sorted(missing)}")
    return selected


def resolve_targets(net: nn.Module, config: AdapterConfig, *, architecture: str | None = None) -> AdapterManifest:
    """Validate every selector against the model and return the resolved manifest; mutates no module.

    Each target row records its component, weight shape, the tensor-defining settings (``method``,
    ``rank``, ``alpha``, ``use_rslora``) and, under ``training``, the training-only settings.
    """
    config.validate()
    binding = architecture_for_model(net, architecture)
    modules = {canonical_name(n): m for n, m in net.named_modules(remove_duplicate=False) if n}
    aliases: dict[int, set[str]] = {}
    for name, parameter in net.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), set()).add(canonical_name(name))
    targets: dict[str, AdapterTarget] = {}
    for group in config.components:
        for name in _select(modules, group, binding):
            module = modules[name]
            if type(module) is not nn.Linear:
                raise ValueError(f"adapter target {name} is {type(module).__name__}; only plain nn.Linear is supported")
            if group.rank > min(module.weight.shape):
                raise ValueError(f"adapter rank {group.rank} exceeds target {name} shape {tuple(module.weight.shape)}")
            if len(aliases[id(module.weight)]) > 1:
                raise ValueError(f"adapter target {name} shares its weight with {sorted(aliases[id(module.weight)])}")
            if name in targets:
                raise ValueError(f"adapter target {name} is selected by more than one component group")
            targets[name] = {
                "name": name,
                "component": group.component,
                "shape": list(module.weight.shape),
                **group.tensor_settings(),
                "training": group.training_settings(),
            }
    rows = [targets[name] for name in sorted(targets)]
    return {"architecture": binding.name, "targets": rows, "trainable_parameters": sum(map(_trainable_count, rows))}


def _trainable_count(row: Mapping[str, Any]) -> int:
    out_features, in_features = row["shape"]
    return row["rank"] * (in_features + out_features) + (out_features if row["method"] == "dora" else 0)


def tensor_identity(manifest: AdapterManifest) -> AdapterTensorIdentity:
    """The part of a manifest that defines the adapter tensors: architecture and per-target identity."""
    rows: list[AdapterTargetIdentity] = [
        {
            "name": row["name"],
            "component": row["component"],
            "shape": list(row["shape"]),
            "method": row["method"],
            "rank": row["rank"],
            "alpha": row["alpha"],
            "use_rslora": row["use_rslora"],
        }
        for row in manifest["targets"]
    ]
    return {"architecture": manifest["architecture"], "targets": rows}


def identity_differences(saved: Mapping[str, Any], live: Mapping[str, Any]) -> list[str]:
    """Human-readable fields in which two adapter identities differ (empty when compatible)."""
    differences = [
        f"{key}: {saved.get(key)!r} (saved) != {live.get(key)!r} (configured)"
        for key in sorted(set(saved) | set(live))
        if key != "targets" and saved.get(key) != live.get(key)
    ]
    saved_rows = {row["name"]: row for row in saved.get("targets", [])}
    live_rows = {row["name"]: row for row in live.get("targets", [])}
    differences += [f"targets: missing {name}" for name in sorted(set(live_rows) - set(saved_rows))]
    differences += [f"targets: unexpected {name}" for name in sorted(set(saved_rows) - set(live_rows))]
    for name in sorted(set(saved_rows) & set(live_rows)):
        for key in sorted(set(saved_rows[name]) | set(live_rows[name])):
            if saved_rows[name].get(key) != live_rows[name].get(key):
                differences.append(
                    f"{name}.{key}: {saved_rows[name].get(key)!r} (saved) != {live_rows[name].get(key)!r} (configured)"
                )
    return differences


def install_adapters(
    net: nn.Module, config: AdapterConfig, *, architecture: str | None = None, initialize: bool = True
) -> AdapterManifest:
    """Replace every resolved target with an :class:`AdapterLinear` and freeze everything else.

    Call after the base weights are materialized and loaded (PiSSA and DoRA initialize from them)
    and before FSDP registers parameters. ``initialize=True`` initializes from this process's global
    RNG (single-process use); distributed builders pass ``initialize=False`` and call
    ``adapter_training.initialize_adapters``. ``initialize=False`` leaves deterministic placeholders
    for a resume or warm start that overwrites them. Returns the resolved manifest, also stored as
    ``net.adapter_manifest``.
    """
    manifest = resolve_targets(net, config, architecture=architecture)
    if any(p.is_meta for p in net.parameters()):
        raise ValueError("install adapters after materializing and loading the base weights")
    modules = {canonical_name(n): m for n, m in net.named_modules(remove_duplicate=False) if n}
    for row in manifest["targets"]:
        group = AdapterComponentConfig(
            component=row["component"],
            targets=[row["name"]],
            **{key: row[key] for key in _TENSOR_KEYS},
            **row["training"],
        )
        layer = AdapterLinear(modules[row["name"]], group, initialize=initialize)
        parent_name, _, leaf = row["name"].rpartition(".")
        setattr(modules[parent_name] if parent_name else net, leaf, layer)
    net.requires_grad_(False)
    for module in net.modules():
        if isinstance(module, AdapterLinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)
            if module.method == "dora":
                module.lora_magnitude.requires_grad_(True)
    net.adapter_manifest = manifest
    assert_trainability(net, manifest)
    return manifest


def trainable_spec(manifest: AdapterManifest) -> dict[str, tuple[float, float | None]]:
    """Trainable parameter name -> ``(lr_multiplier, weight_decay)``.

    ``weight_decay`` ``None`` inherits the optimizer's; DoRA magnitudes are never decayed.
    """
    spec: dict[str, tuple[float, float | None]] = {}
    for row in manifest["targets"]:
        training = row["training"]
        for leaf in ("lora_A.weight", "lora_B.weight"):
            spec[f"{row['name']}.{leaf}"] = (training["lr_multiplier"], training["weight_decay"])
        if row["method"] == "dora":
            spec[f"{row['name']}.lora_magnitude"] = (training["lr_multiplier"], 0.0)
    return spec


def adapter_state_spec(identity: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Name -> shape of every tensor a portable adapter carries (trainable state plus PiSSA buffers)."""
    spec: dict[str, tuple[int, ...]] = {}
    for row in identity["targets"]:
        out_features, in_features = row["shape"]
        rank, prefix = row["rank"], row["name"]
        spec[f"{prefix}.lora_A.weight"] = (rank, in_features)
        spec[f"{prefix}.lora_B.weight"] = (out_features, rank)
        if row["method"] == "dora":
            spec[f"{prefix}.lora_magnitude"] = (out_features,)
        if row["method"] == "pissa":
            spec[f"{prefix}.initial_A"] = (rank, in_features)
            spec[f"{prefix}.initial_B"] = (out_features, rank)
    return spec


def assert_trainability(net: nn.Module, manifest: AdapterManifest, optimizer: Any = None) -> None:
    """Check that exactly the adapter parameters are trainable float32 tensors (and optimized)."""
    expected = set(trainable_spec(manifest))
    parameters = {canonical_name(n): p for n, p in net.named_parameters(remove_duplicate=False)}
    actual = {name for name, p in parameters.items() if p.requires_grad}
    if actual != expected:
        raise ValueError(
            f"adapter trainability mismatch: missing={sorted(expected - actual)[:5]}, "
            f"unexpected={sorted(actual - expected)[:5]}"
        )
    if any(parameters[name].dtype != torch.float32 for name in expected):
        raise ValueError("adapter parameters must stay float32")
    count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if count != manifest["trainable_parameters"]:
        raise ValueError(f"adapter trainable parameter count {count} != {manifest['trainable_parameters']}")
    if optimizer is not None:
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in parameters.values() if p.requires_grad}:
            raise ValueError("optimizer parameters differ from the adapter parameters")


def adapter_mixed_precision_policy() -> Any:
    """FSDP2 policy for a frozen base next to float32 adapters: cast no parameter, reduce in float32."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    return MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32, cast_forward_inputs=False)


def require_adapter_calls(net: nn.Module, components: Iterable[str] = ("language",)) -> None:
    """Fail the first forward unless every adapter of ``components`` ran through its own forward.

    A target whose weight its parent reads directly (instead of calling the module) would train
    nothing. Language blocks run on every batch; vision components may legitimately be skipped by
    text-only batches, so they are not checked by default.
    """
    wanted = {row["name"] for row in installed_adapter_manifest(net)["targets"] if row["component"] in set(components)}
    seen: set[str] = set()
    handles = []
    for name, module in net.named_modules():
        name = canonical_name(name)
        if name in wanted and isinstance(module, AdapterLinear):
            handles.append(module.register_forward_pre_hook(lambda module, args, name=name: seen.add(name)))

    def check(module: nn.Module, args: Any, output: Any) -> None:
        for handle in handles:
            handle.remove()
        missing = sorted(wanted - seen)
        if missing:
            raise RuntimeError(
                f"adapter targets were not called by the model forward and would not train: {missing[:5]}"
            )

    handles.append(net.register_forward_hook(check))


_FINGERPRINT_SAMPLES = 1024


def fingerprint_sample(weight: torch.Tensor) -> bytes:
    """A deterministic, dtype-independent sample of a base weight for :func:`base_fingerprint`.

    Up to 1024 evenly strided elements, rounded to bfloat16, so float32 and bfloat16 copies of one
    base agree while a different checkpoint does not.
    """
    flat = weight.detach().reshape(-1)
    sample = flat[:: max(1, flat.numel() // _FINGERPRINT_SAMPLES)][:_FINGERPRINT_SAMPLES]
    return sample.to(device="cpu", dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()


def base_fingerprint(samples: Iterable[tuple[str, bytes]]) -> str:
    """SHA-256 over ``(target name, fingerprint_sample)`` pairs in name order."""
    import hashlib

    digest = hashlib.sha256()
    for name, sample in sorted(samples):
        digest.update(name.encode() + b"\0" + sample)
    return digest.hexdigest()


def model_base_fingerprint(net: nn.Module, manifest: AdapterManifest) -> str:
    """The base fingerprint of a model's adapted target weights (before FSDP shards them)."""
    return base_fingerprint(
        (row["name"], fingerprint_sample(net.get_submodule(row["name"]).weight)) for row in manifest["targets"]
    )


def training_identity(manifest: AdapterManifest) -> dict[str, AdapterResumeSettings]:
    """Target name -> the settings a resumed run must keep (``dropout``, ``lr_multiplier``, ``weight_decay``).

    Recorded beside the tensor identity in training checkpoints. The initialization settings do not
    matter once the adapters exist, so they are not part of it.
    """
    return {
        row["name"]: {
            "dropout": row["training"]["dropout"],
            "lr_multiplier": row["training"]["lr_multiplier"],
            "weight_decay": row["training"]["weight_decay"],
        }
        for row in manifest["targets"]
    }


def installed_adapter_manifest(net: nn.Module) -> AdapterManifest:
    """The manifest :func:`install_adapters` attached to ``net``."""
    manifest = getattr(net, "adapter_manifest", None)
    if manifest is None:
        raise ValueError(f"{type(net).__name__} has no installed adapters")
    return manifest


def installed_adapter_identity(net: nn.Module) -> AdapterIdentity:
    """The identity ``adapter_training.prepare_adapter_model`` attached to ``net``."""
    identity = getattr(net, "adapter_identity", None)
    if identity is None:
        raise ValueError(f"{type(net).__name__} has no adapter identity; prepare it with prepare_adapter_model")
    return identity
