# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Custom in-place LoRA injection for MoT-style models.

The key design choice is to subclass ``nn.Linear`` (``LoraInjectedLinear``)
rather than wrap it (as PEFT's ``LoraLayer`` does). The wrapped weight stays
at ``<path>.weight`` so checkpoints saved from a LoRA-trained model load
cleanly into either a LoRA or non-LoRA model — no key rename and no loader
alias needed. ``lora_A`` and ``lora_B`` are sibling submodules, producing
the state-dict keys ``<path>.lora_A.weight`` and ``<path>.lora_B.weight``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.utils import log


class LoraInjectedLinear(nn.Linear):
    """nn.Linear with sibling lora_A and lora_B that preserves the original ``.weight`` key.

    State-dict keys for a module at ``<path>``:
      ``<path>.weight``        — original Linear weight (unchanged key)
      ``<path>.bias``          — original Linear bias (if present)
      ``<path>.lora_A.weight`` — low-rank down-projection
      ``<path>.lora_B.weight`` — low-rank up-projection

    Forward computes ``y = base(x) + (alpha / r) * lora_B(lora_A(x))``.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        *,
        dropout: float = 0.0,
        use_rslora: bool = False,
        adapter_dtype: torch.dtype | None = None,
    ) -> None:
        # Reuse base's geometry. Inherit nn.Linear so ``super().forward(x)``
        # dispatches to the standard F.linear path.
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        # Replace nn.Linear's freshly-allocated parameters with the base's
        # existing ones. On meta device this is a no-op for memory; on a
        # real device this preserves the pretrained weight identity.
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        # Sibling submodules. Using bias=False nn.Linear gives us a single
        # ``weight`` parameter and a clean ``.lora_A.weight`` state-dict key.
        adapter_dtype = adapter_dtype or base.weight.dtype
        self.lora_A = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=adapter_dtype,
        )
        self.lora_B = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=adapter_dtype,
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self._lora_rank = int(rank)
        self._lora_alpha = float(alpha)
        self._lora_use_rslora = bool(use_rslora)

    @property
    def _lora_scale(self) -> float:
        denominator = math.sqrt(self._lora_rank) if self._lora_use_rslora else self._lora_rank
        return self._lora_alpha / denominator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        adapter_input = self.lora_dropout(x).to(dtype=self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(adapter_input)).to(dtype=base_out.dtype)
        return base_out + self._lora_scale * lora_out

    @torch.no_grad()
    def merge_adapter_(self) -> None:
        """Merge the adapter update into the base weight in place."""
        delta = self.lora_B.weight.float() @ self.lora_A.weight.float()
        self.weight.add_(delta.to(dtype=self.weight.dtype), alpha=self._lora_scale)


def _target_matches(full_child_path: str, child_name: str, target: str) -> bool:
    """Return True if ``target`` selects the child at ``full_child_path``.

    Two matching modes, chosen by whether ``target`` contains a ``.``:

    * Plain leaf name (e.g. ``q_proj_moe_gen``): matches when the child's own
      name equals ``target``. Leaf names like ``*_moe_gen`` are unique to the
      generation tower, so this cleanly selects gen-only modules.
    * Path-qualified suffix (e.g. ``mlp_moe_gen.up_proj``): matches when the
      child's full dotted path equals ``target`` or ends with ``.`` + ``target``.
      This disambiguates leaf names that are shared across towers — e.g. the
      dense-FFN ``up_proj`` / ``down_proj`` exist under BOTH the understanding
      ``mlp`` and the generation ``mlp_moe_gen``; ``mlp_moe_gen.up_proj``
      targets only the generation-tower copy.
    """
    if "." in target:
        return full_child_path == target or full_child_path.endswith("." + target)
    return child_name == target


def _inject_lora_inplace(
    network: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    use_rslora: bool = False,
    adapter_dtype: torch.dtype | None = None,
) -> int:
    """Replace each targeted ``nn.Linear`` child in-place with ``LoraInjectedLinear``.

    Targets are matched by :func:`_target_matches`: plain leaf names match by
    exact child name; names containing a ``.`` match by full-path suffix so
    tower-shared leaf names (e.g. ``mlp_moe_gen.up_proj``) can be selected
    without also wrapping the understanding tower's ``mlp.up_proj``.

    Snapshots ``named_modules()`` before mutating the tree so newly-inserted
    LoRA submodules are not re-visited.
    """
    replaced = 0
    for parent_name, parent in list(network.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_child_path = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(_target_matches(full_child_path, child_name, t) for t in target_modules):
                setattr(
                    parent,
                    child_name,
                    LoraInjectedLinear(
                        child,
                        rank,
                        alpha,
                        dropout=dropout,
                        use_rslora=use_rslora,
                        adapter_dtype=adapter_dtype,
                    ),
                )
                replaced += 1
    return replaced


def apply_lora_trainable_scope(
    network: torch.nn.Module,
    *,
    lora_target_modules: str,
    lora_bias: str = "none",
    lora_modules_to_save: str = "",
) -> dict[str, object]:
    """Reapply and report the logical LoRA trainable scope.

    The scope must be restored after model materialization/FSDP transforms,
    which may recreate parameters with ``requires_grad=True``.
    """
    target_modules = [item.strip() for item in lora_target_modules.split(",") if item.strip()]
    modules_to_save = {item.strip() for item in lora_modules_to_save.split(",") if item.strip()}
    trainable_parameters = 0
    frozen_parameters = 0
    trainable_parameter_tensors = 0
    for name, parameter in network.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        parent_name = module_name.rsplit(".", 1)[0]
        is_module_to_save = any(
            module_name == suffix
            or module_name.endswith(f".{suffix}")
            or parent_name == suffix
            or parent_name.endswith(f".{suffix}")
            for suffix in modules_to_save
        )
        is_target_bias = name.endswith(".bias") and any(
            f".{target}." in f".{name}" for target in target_modules
        )
        is_trainable_bias = lora_bias == "all" and name.endswith(".bias")
        trainable = (
            "lora_" in name
            or is_module_to_save
            or is_trainable_bias
            or (lora_bias == "lora_only" and is_target_bias)
        )
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_parameters += parameter.numel()
            trainable_parameter_tensors += 1
        else:
            frozen_parameters += parameter.numel()

    adapter_modules = sorted(
        name for name, module in network.named_modules() if isinstance(module, LoraInjectedLinear)
    )
    summary: dict[str, object] = {
        "training_mode": "peft",
        "trainable_parameters": trainable_parameters,
        "total_parameters": trainable_parameters + frozen_parameters,
        "frozen_parameters": frozen_parameters,
        "trainable_parameter_tensors": trainable_parameter_tensors,
        "adapter_module_count": len(adapter_modules),
        "adapter_modules": adapter_modules,
    }
    network._tao_peft_parameter_summary = summary
    return summary


def inject_lora_pre_fsdp(
    network: torch.nn.Module,
    *,
    lora_rank: int,
    lora_alpha: float,
    lora_target_modules: str,
    lora_dropout: float = 0.0,
    lora_bias: str = "none",
    lora_use_rslora: bool = False,
    lora_modules_to_save: str = "",
    lora_precision: str | None = None,
) -> torch.nn.Module:
    """Inject LoRA adapters into ``network`` BEFORE FSDP wrap on meta device.

    Must be called on the meta-device network (pre-FSDP) so the injector sees
    unsharded weight shapes; injecting after FSDP causes ``lora_B`` to be
    constructed with the per-rank shard size (e.g., 8192/8=1024) and triggers
    a shape mismatch at forward time.

    ``lora_A`` and ``lora_B`` parameters are left uninitialized on meta;
    the caller must initialize them AFTER
    ``to_empty(device="cuda") + init_weights(buffer_device="cuda")``
    via ``init_lora_weights_post_materialization``.

    Also freezes every non-LoRA parameter so the optimizer's
    ``keys_to_select=["lora_"]`` filter trains adapters only.
    """
    assert network is not None, "Network is not initialized"

    if lora_rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {lora_rank}")
    if lora_alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {lora_dropout}")
    if lora_bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Unsupported LoRA bias mode: {lora_bias!r}")
    if lora_precision not in {None, "float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported LoRA precision: {lora_precision!r}")

    target_modules_list = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
    if not target_modules_list:
        raise ValueError("LoRA target_modules cannot be empty")

    all_module_paths = [name for name, _ in network.named_modules()]
    leaf_names = {p.split(".")[-1] for p in all_module_paths}

    def _target_exists(t: str) -> bool:
        # Path-qualified targets (e.g. "mlp_moe_gen.up_proj") match by full-path
        # suffix; plain leaf targets match against the set of leaf names.
        if "." in t:
            return any(p == t or p.endswith("." + t) for p in all_module_paths)
        return t in leaf_names

    invalid_modules = [t for t in target_modules_list if not _target_exists(t)]
    if invalid_modules:
        log.warning(f"LoRA target modules not found in model: {invalid_modules}")

    modules_to_save = {m.strip() for m in lora_modules_to_save.split(",") if m.strip()}
    adapter_dtype = getattr(torch, lora_precision) if lora_precision else None
    log.info(
        "Injecting LoRA before FSDP: "
        f"rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
        f"rslora={lora_use_rslora}, bias={lora_bias}, targets={target_modules_list}, "
        f"modules_to_save={sorted(modules_to_save)}"
    )

    try:
        replaced = _inject_lora_inplace(
            network,
            target_modules_list,
            lora_rank,
            lora_alpha,
            dropout=lora_dropout,
            use_rslora=lora_use_rslora,
            adapter_dtype=adapter_dtype,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to inject LoRA adapters into model: {e}") from e

    if replaced == 0:
        raise ValueError(f"LoRA injection replaced 0 modules; check lora_target_modules={lora_target_modules!r}")

    summary = apply_lora_trainable_scope(
        network,
        lora_target_modules=lora_target_modules,
        lora_bias=lora_bias,
        lora_modules_to_save=lora_modules_to_save,
    )
    trainable_parameters = int(summary["trainable_parameters"])
    frozen_parameters = int(summary["frozen_parameters"])
    total_parameters = int(summary["total_parameters"])

    log.info(
        f"LoRA injection successful: {replaced} modules wrapped, "
        f"{trainable_parameters:,} trainable LoRA params, "
        f"{frozen_parameters:,} frozen base params "
        f"({100 * trainable_parameters / max(1, total_parameters):.3f}% trainable)"
    )
    return network


def init_lora_weights_post_materialization(network: torch.nn.Module) -> None:
    """Initialize LoRA params after ``to_empty + init_weights`` materializes them.

    The custom injector leaves lora_A/lora_B as uninitialized meta-device
    parameters. After ``to_empty(device=DEVICE)``, they have allocated but
    uninitialized memory. Init in-place (``lora_A ~ kaiming_uniform_(a=sqrt(5))``,
    ``lora_B = zeros``) and cast each pair to its wrapped base weight's dtype
    so ``F.linear(x, lora_A.weight)`` sees matching dtypes whether the base
    runs in fp32, bf16, or fp16.
    """
    for module in network.modules():
        if not isinstance(module, LoraInjectedLinear):
            continue
        adapter_dtype = module.lora_A.weight.dtype
        torch.nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
        module.lora_A.weight.data = module.lora_A.weight.data.to(adapter_dtype)
        torch.nn.init.zeros_(module.lora_B.weight)
        module.lora_B.weight.data = module.lora_B.weight.data.to(adapter_dtype)


@torch.no_grad()
def merge_lora_adapters_(network: torch.nn.Module) -> int:
    """Merge every injected adapter and return the number merged.

    The modules remain present so a DCP state dict keeps the same exact keys;
    callers exporting to a dense Hugging Face model should copy only base
    parameters after this operation.
    """
    merged = 0
    for module in network.modules():
        if isinstance(module, LoraInjectedLinear):
            module.merge_adapter_()
            merged += 1
    return merged


@torch.no_grad()
def merge_and_strip_lora_adapters_(network: torch.nn.Module) -> int:
    """Merge adapters and restore plain ``nn.Linear`` modules for HF export."""
    merged = 0
    for parent in list(network.modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, LoraInjectedLinear):
                continue
            child.merge_adapter_()
            dense = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            dense.weight = child.weight
            if child.bias is not None:
                dense.bias = child.bias
            setattr(parent, child_name, dense)
            merged += 1
    return merged
