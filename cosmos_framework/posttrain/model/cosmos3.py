# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""OmniMoT adapter for post-training flow-matching objectives."""

from __future__ import annotations

import copy
import hashlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Literal

import torch
from torch.distributed.tensor import DTensor, Replicate

from cosmos_framework.posttrain.model.protocol import FlowCondition, FlowConditionMask, NFTVelocityPredictions
from cosmos_framework.posttrain.model.registry import register_model
from cosmos_framework.posttrain.utils.dist import is_dtensor_sharded, run_rank_converged
from cosmos_framework.posttrain.utils.identity import canonical_config_value, canonical_json_sha256

_VISION_ITEM_SEED_DOMAIN: bytes = b"posttrain.cosmos3.vision-item-encode.v1\0"
_ROPE_RUNTIME_ATTRIBUTE_SUFFIXES: tuple[str, ...] = (
    "inv_freq",
    "max_seq_len_cached",
    "original_inv_freq",
    "long_inv_freq",
)
_MISSING_METHOD = object()
_REFERENCE_SEMANTIC_CONFIG_FIELDS: tuple[str, ...] = (
    "tokenizer",
    "load_vision_tokenizer",
    "net",
    "precision",
    "lora_enabled",
    "lora_rank",
    "lora_alpha",
    "lora_target_modules",
    "rectified_flow_inference_config",
    "rectified_flow_training_config",
    "vlm_config",
    "diffusion_expert_config",
    "input_video_key",
    "input_image_key",
    "input_caption_key",
    "state_ch",
    "state_t",
    "latent_downsample_factor",
    "resolution",
    "max_num_tokens_after_packing",
    "joint_attn_implementation",
    "multiview_attention",
    "natten_parameter_list",
    "video_temporal_causal",
    "causal_training_strategy",
    "vision_gen",
    "action_gen",
    "max_action_dim",
    "num_embodiment_domains",
    "sound_gen",
    "sound_tokenizer",
    "sound_dim",
    "sound_latent_fps",
    "enable_input_bias",
)

# This registry mirrors the exact RMSNorm classes reachable through the three variants supported by
# ``models.mot.unified_mot.LayerTypes``. Keep it fail-closed: a new variant requires an explicit numerical review.
_COSMOS3_RMSNORM_TYPES: frozenset[str] = frozenset(
    {
        "cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl.Nemotron3DenseVLRMSNorm",
        "cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl.Qwen3VLTextRMSNorm",
        "cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe.Qwen3VLMoeTextRMSNorm",
    }
)


def _module_type_name(module: torch.nn.Module) -> str:
    module_type = type(module)
    return f"{module_type.__module__}.{module_type.__name__}"


def _fp32_rmsnorm_forward(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:  # hidden_states: [*,D], returns [*,D]
    """Run the Cosmos3 post-training RMSNorm contract entirely in float32."""
    epsilon = getattr(module, "variance_epsilon", None)
    weight = getattr(module, "weight", None)  # [D] or invalid
    if isinstance(epsilon, bool) or not isinstance(epsilon, (float, int)):
        raise TypeError(f"recognized RMSNorm has invalid epsilon {epsilon!r}")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"recognized RMSNorm has invalid weight {type(weight).__name__}")
    if isinstance(weight, DTensor):
        if any(not isinstance(placement, Replicate) for placement in weight.placements):
            raise RuntimeError(
                "recognized RMSNorm DTensor weight must be replicated during forward; "
                f"got placements {weight.placements}"
            )
        weight = weight.to_local()  # [D]
    if weight.ndim != 1 or weight.numel() != hidden_states.shape[-1]:
        raise ValueError(
            "recognized RMSNorm weight must be one-dimensional and match the hidden width; "
            f"got weight shape {tuple(weight.shape)} and hidden shape {tuple(hidden_states.shape)}"
        )
    input_dtype = hidden_states.dtype
    hidden_fp32 = hidden_states.float()  # [*,D]
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)  # [*,1]
    normalized = hidden_fp32 * torch.rsqrt(variance + float(epsilon))  # [*,D]
    output = weight.float() * normalized  # [*,D]
    return output.to(input_dtype)  # [*,D]


def _enable_fp32_rmsnorms(root: torch.nn.Module) -> dict[str, str]:
    """Install the explicit fp32 RMSNorm forward on recognized Cosmos3 modules.

    The Transformers kernel decorator can replace a class's Python forward with a
    lower-precision fused implementation. Binding the post-training forward per
    instance avoids changing Cosmos3 pretraining globally and leaves state-dict
    structure untouched. Repeated calls are idempotent.
    """
    inventory: dict[str, str] = {}
    unknown: dict[str, str] = {}
    recognized: list[torch.nn.Module] = []
    for path, module in root.named_modules():
        type_name = _module_type_name(module)
        norm_like = type(module).__name__.endswith("RMSNorm") or (
            hasattr(module, "weight") and hasattr(module, "variance_epsilon")
        )
        if not norm_like:
            continue
        if type_name not in _COSMOS3_RMSNORM_TYPES:
            unknown[path or "<root>"] = type_name
            continue
        inventory[path or "<root>"] = type_name
        recognized.append(module)
    if unknown:
        raise RuntimeError(f"bf16 Cosmos3 post-training found unsupported RMSNorm modules: {unknown}")
    for module in recognized:
        if getattr(getattr(module, "forward", None), "__func__", None) is _fp32_rmsnorm_forward:
            continue
        module.forward = MethodType(_fp32_rmsnorm_forward, module)
    return inventory


@dataclass(frozen=True)
class _RuntimeAttributeState:
    """One mutable Cosmos3 runtime attribute and its registration metadata."""

    name: str
    existed: bool
    value: Any
    was_buffer: bool
    persistent: bool


@dataclass(frozen=True)
class _ModuleRuntimeState:
    """Mutable dynamic-RoPE state owned by one Cosmos3 module."""

    module: torch.nn.Module
    attributes: tuple[_RuntimeAttributeState, ...]


@dataclass(frozen=True)
class _MethodOverrideState:
    """One temporarily suppressed Cosmos3 training callback."""

    module: torch.nn.Module
    original: Any


@dataclass(frozen=True)
class _PreparedDenoise:
    """Cosmos3 inputs prepared before one distributed denoiser forward."""

    packed: Any
    targets: list[torch.Tensor]  # each [...]
    condition_mask: list[torch.Tensor] | None  # each [T,1,1]


@dataclass(frozen=True)
class _Cosmos3FlowCondition:
    """OmniMoT-owned conditioning hidden behind the generic flow capability."""

    gen_data_clean: Any
    sequence_plans: list[Any]
    text_tokens: list[list[int]]
    is_image_batch: bool
    resolutions: list[str] | None


def _require_cosmos3_condition(cond: FlowCondition) -> _Cosmos3FlowCondition:
    """Narrow the opaque flow boundary before using OmniMoT-specific fields."""

    if not isinstance(cond, _Cosmos3FlowCondition):
        raise TypeError(f"Cosmos3FlowPolicy requires _Cosmos3FlowCondition, got {type(cond).__name__}")
    return cond


def _require_cosmos3_condition_mask(condition_mask: FlowConditionMask) -> list[torch.Tensor] | None:
    """Narrow the shared mask union to the representation used by OmniMoT."""

    if condition_mask is None:
        return None
    if not isinstance(condition_mask, list) or any(not isinstance(mask, torch.Tensor) for mask in condition_mask):
        raise TypeError("Cosmos3FlowPolicy requires a list of tensor condition masks or None")
    return condition_mask


@dataclass(frozen=True)
class _PreparedNFTVelocity:
    """Cosmos3 NFT inputs prepared before ordered distributed denoiser forwards."""

    packed_policy: Any
    packed_old: Any
    packed_reference: Any | None
    x0_items: list[torch.Tensor]  # each [1,C,T,H,W]
    x_t_items: list[torch.Tensor]  # each [1,C,T,H,W]
    effective_t_items: list[torch.Tensor]  # each [T,1,1]
    timestep_fraction: torch.Tensor  # [B]
    x0_dtype: torch.dtype
    condition_mask_is_none: bool


_PersistentStateKind = Literal["parameter", "buffer"]
_TensorKind = Literal["plain", "dtensor"]
_DTensorLayout = tuple[str, tuple[int, ...], tuple[int, ...], tuple[str, ...]]
_PersistentState = dict[str, tuple[_PersistentStateKind, torch.Tensor]]


@dataclass(frozen=True)
class _PersistentStateCopyPlan:
    """One validated policy-to-snapshot persistent-state transfer."""

    name: str
    state_kind: _PersistentStateKind
    global_shape: tuple[int, ...]
    policy_dtype: str
    snapshot_dtype: str
    policy_kind: _TensorKind
    snapshot_kind: _TensorKind
    policy_layout: _DTensorLayout | None
    snapshot_layout: _DTensorLayout | None


def _dtensor_layout_signature(tensor: Any) -> _DTensorLayout:
    """Return rank-independent mesh and placement identity for one DTensor."""
    mesh = tensor.device_mesh
    mesh_ranks = mesh.mesh.detach().cpu().reshape(-1)  # [mesh_size]
    return (
        str(mesh.device_type),
        tuple(int(size) for size in mesh.mesh.shape),
        tuple(int(rank) for rank in mesh_ranks.tolist()),
        tuple(str(placement) for placement in tensor.placements),
    )


def _persistent_module_state(module: torch.nn.Module) -> _PersistentState:
    """Return parameters and persistent buffers, excluding runtime-only buffers."""
    state: _PersistentState = {
        name: ("parameter", parameter) for name, parameter in module.named_parameters(remove_duplicate=False)
    }
    for module_name, child in module.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for buffer_name, buffer in child._buffers.items():
            if buffer is None or buffer_name in child._non_persistent_buffers_set:
                continue
            name = f"{prefix}{buffer_name}"
            if name in state:
                raise RuntimeError(f"persistent module state contains duplicate name {name!r}")
            state[name] = ("buffer", buffer)
    return state


def _copy_persistent_module_state(
    policy_net: torch.nn.Module,
    snapshot_net: torch.nn.Module,
    *,
    snapshot_label: str,
) -> None:
    """Validate, then copy policy parameters and persistent buffers into one snapshot."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, distribute_tensor

    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    policy_state: _PersistentState = {}
    snapshot_state: _PersistentState = {}
    copy_plan: list[_PersistentStateCopyPlan] = []
    plan_error: Exception | None = None
    try:
        policy_state = _persistent_module_state(policy_net)
        snapshot_state = _persistent_module_state(snapshot_net)
        policy_names = set(policy_state)
        snapshot_names = set(snapshot_state)
        if policy_names != snapshot_names:
            missing = sorted(policy_names - snapshot_names)
            unexpected = sorted(snapshot_names - policy_names)
            raise RuntimeError(
                f"policy/{snapshot_label} persistent-state names differ; "
                f"missing from {snapshot_label}={missing}, unexpected in {snapshot_label}={unexpected}"
            )
        default_world = dist.get_world_size() if dist.is_initialized() else 1
        default_ranks = tuple(range(default_world))
        for name in sorted(policy_names):
            state_kind, policy = policy_state[name]
            snapshot_state_kind, snapshot = snapshot_state[name]
            if state_kind != snapshot_state_kind:
                raise RuntimeError(
                    f"policy/{snapshot_label} persistent-state kind differs for {name}: "
                    f"{state_kind} vs {snapshot_state_kind}"
                )
            policy_shape = tuple(policy.shape)
            snapshot_shape = tuple(snapshot.shape)
            if policy_shape != snapshot_shape:
                raise RuntimeError(
                    f"policy/{snapshot_label} global shape differs for {state_kind} {name}: "
                    f"{policy_shape} vs {snapshot_shape}"
                )
            if policy.dtype != snapshot.dtype and not (policy.is_floating_point() and snapshot.is_floating_point()):
                raise RuntimeError(
                    f"policy/{snapshot_label} non-floating dtype differs for {state_kind} {name}: "
                    f"{policy.dtype} vs {snapshot.dtype}"
                )
            policy_kind: _TensorKind = "dtensor" if isinstance(policy, DTensor) else "plain"
            snapshot_kind: _TensorKind = "dtensor" if isinstance(snapshot, DTensor) else "plain"
            policy_layout = _dtensor_layout_signature(policy) if isinstance(policy, DTensor) else None
            snapshot_layout = _dtensor_layout_signature(snapshot) if isinstance(snapshot, DTensor) else None
            for owner, layout in (("policy", policy_layout), (snapshot_label, snapshot_layout)):
                if layout is not None and tuple(sorted(layout[2])) != default_ranks:
                    raise RuntimeError(
                        f"{owner} {state_kind} {name} uses mesh ranks {layout[2]}, "
                        f"expected the complete default world {default_ranks}"
                    )
            if policy_layout is not None and snapshot_layout is not None and policy_layout[:3] != snapshot_layout[:3]:
                raise RuntimeError(
                    f"policy/{snapshot_label} meshes differ for {state_kind} {name}: "
                    f"{policy_layout[:3]} vs {snapshot_layout[:3]}"
                )
            copy_plan.append(
                _PersistentStateCopyPlan(
                    name=name,
                    state_kind=state_kind,
                    global_shape=policy_shape,
                    policy_dtype=str(policy.dtype),
                    snapshot_dtype=str(snapshot.dtype),
                    policy_kind=policy_kind,
                    snapshot_kind=snapshot_kind,
                    policy_layout=policy_layout,
                    snapshot_layout=snapshot_layout,
                )
            )
    except Exception as error:  # noqa: BLE001 -- converge before any per-state-tensor collective
        plan_error = error
    raise_if_any_rank_failed(plan_error, f"{snapshot_label} persistent-state copy-plan validation")

    if dist.is_initialized() and dist.get_world_size() > 1:
        plan_digest = hashlib.sha256(repr(copy_plan).encode("utf-8")).hexdigest()
        gathered_digests: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_digests, plan_digest)
        if any(digest != gathered_digests[0] for digest in gathered_digests[1:]):
            raise RuntimeError(f"{snapshot_label} persistent-state copy plans differ across ranks: {gathered_digests}")

    with torch.no_grad():
        for plan in copy_plan:
            policy = policy_state[plan.name][1]
            snapshot = snapshot_state[plan.name][1]
            if plan.snapshot_kind == "plain":
                source = policy.full_tensor() if plan.policy_kind == "dtensor" else policy.detach()  # [*global]
                target = snapshot.detach()  # [*global]
            else:
                if plan.policy_kind == "plain":
                    distributed = distribute_tensor(
                        policy.detach(),
                        snapshot.device_mesh,
                        snapshot.placements,
                        src_data_rank=0,
                    )  # DTensor [*global]
                elif plan.policy_layout == plan.snapshot_layout:
                    distributed = policy
                else:
                    distributed = policy.redistribute(
                        device_mesh=snapshot.device_mesh,
                        placements=snapshot.placements,
                    )  # DTensor [*global]
                source = distributed.to_local()  # [*target local]
                target = snapshot.to_local()  # [*target local]
            target.copy_(source)  # [*target]


def _ignore_moe_callback_stats(_module: torch.nn.Module, *_args: Any, **_kwargs: Any) -> None:
    pass


def _clone_runtime_value(value: Any, memo: dict[int, Any]) -> Any:
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()  # [...]
        memo[identity] = cloned
        return cloned
    return copy.deepcopy(value, memo)


def _mutable_rope_attribute_names(module: torch.nn.Module) -> tuple[str, ...]:
    """Return attributes mutated by Transformers' dynamic RoPE update."""
    rope_type = getattr(module, "rope_type", None)
    if isinstance(rope_type, str):
        rope_types = (("", rope_type),)
    elif isinstance(rope_type, dict):
        rope_types = tuple((f"{layer_type}_", value) for layer_type, value in rope_type.items())
    else:
        return ()
    prefixes = tuple(
        prefix for prefix, value in rope_types if isinstance(value, str) and ("dynamic" in value or value == "longrope")
    )
    if not prefixes:
        return ()
    names = [f"{prefix}{suffix}" for prefix in prefixes for suffix in _ROPE_RUNTIME_ATTRIBUTE_SUFFIXES]
    names.append("attention_scaling")
    return tuple(dict.fromkeys(names))


def _snapshot_dynamic_rope_state(
    roots: tuple[torch.nn.Module | None, ...],
) -> tuple[_ModuleRuntimeState, ...]:
    """Snapshot mutable RoPE state from the policy and frozen reference."""
    seen: set[int] = set()
    states: list[_ModuleRuntimeState] = []
    for root in roots:
        if not isinstance(root, torch.nn.Module):
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            names = _mutable_rope_attribute_names(module)
            if not names:
                continue
            memo: dict[int, Any] = {}
            attributes: list[_RuntimeAttributeState] = []
            for name in names:
                was_buffer = name in module._buffers
                existed = was_buffer or hasattr(module, name)
                attributes.append(
                    _RuntimeAttributeState(
                        name=name,
                        existed=existed,
                        value=_clone_runtime_value(getattr(module, name), memo) if existed else None,
                        was_buffer=was_buffer,
                        persistent=was_buffer and name not in module._non_persistent_buffers_set,
                    )
                )
            states.append(_ModuleRuntimeState(module=module, attributes=tuple(attributes)))
    return tuple(states)


def _restore_dynamic_rope_state(states: tuple[_ModuleRuntimeState, ...]) -> None:
    """Restore dynamic RoPE values and buffer registration exactly."""
    for state in states:
        memo: dict[int, Any] = {}
        for attribute in state.attributes:
            module = state.module
            if attribute.name in module._buffers or attribute.name in module.__dict__:
                delattr(module, attribute.name)
            if not attribute.existed:
                continue
            value = _clone_runtime_value(attribute.value, memo)
            if attribute.was_buffer:
                module.register_buffer(attribute.name, value, persistent=attribute.persistent)
            else:
                setattr(module, attribute.name, value)


def _suppress_moe_callback_stats(
    roots: tuple[torch.nn.Module | None, ...],
) -> tuple[_MethodOverrideState, ...]:
    """Disable Qwen MoE training-stat updates for a Cosmos3 validation scope."""
    overrides: list[_MethodOverrideState] = []
    seen: set[int] = set()
    try:
        for root in roots:
            if not isinstance(root, torch.nn.Module):
                continue
            for module in root.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                callback = getattr(module, "_update_moe_callback_stats", None)
                if not callable(callback):
                    continue
                original = vars(module).get("_update_moe_callback_stats", _MISSING_METHOD)
                setattr(module, "_update_moe_callback_stats", MethodType(_ignore_moe_callback_stats, module))
                overrides.append(
                    _MethodOverrideState(
                        module=module,
                        original=original,
                    )
                )
    except BaseException as primary_error:  # noqa: BLE001 -- restore every prior override
        try:
            _restore_moe_callback_stats(tuple(overrides))
        except BaseException as cleanup_error:  # noqa: BLE001 -- preserve the patching failure
            raise primary_error from cleanup_error
        raise
    return tuple(overrides)


def _restore_moe_callback_stats(states: tuple[_MethodOverrideState, ...]) -> None:
    """Restore every temporarily overridden MoE callback."""
    failures: list[str] = []
    for state in reversed(states):
        try:
            if state.original is _MISSING_METHOD:
                delattr(state.module, "_update_moe_callback_stats")
            else:
                setattr(state.module, "_update_moe_callback_stats", state.original)
        except BaseException as error:  # noqa: BLE001 -- restore every patched module
            failures.append(f"{type(state.module).__name__}: {type(error).__name__}: {error}")
    if failures:
        raise RuntimeError("; ".join(failures))


def _unbatch(value: Any) -> list[Any]:
    """Split a batched Cosmos3 tensor into singleton samples."""
    if isinstance(value, list):
        return value
    return [value[index : index + 1] for index in range(value.shape[0])]  # each [1,...]


def _nft_vision_predictions(output: object, *, role: str) -> list[torch.Tensor]:
    """Validate one local OmniMoT result after all collective-bearing NFT forwards completed."""
    if not isinstance(output, dict):
        raise TypeError(f"OmniMoT NFT {role} denoise must return {{'preds_vision': list[Tensor]}}")
    predictions = output.get("preds_vision")
    if not isinstance(predictions, list) or any(not isinstance(item, torch.Tensor) for item in predictions):
        raise TypeError(f"OmniMoT NFT {role} denoise must return {{'preds_vision': list[Tensor]}}")
    return predictions


def _rebatch_uniform(items: list[torch.Tensor], samples: list[torch.Tensor]) -> torch.Tensor:
    """Normalize uniform OmniMoT singleton outputs into one NFT batch."""
    if len(items) != len(samples):
        raise ValueError(f"prediction/sample length mismatch: {len(items)} vs {len(samples)}")
    normalized = [
        _squeeze_stale_batch(item, sample) for item, sample in zip(items, samples, strict=True)
    ]  # each [1,...]
    if not normalized:
        raise ValueError("cannot rebatch an empty prediction list")
    shapes = {tuple(item.shape[1:]) for item in normalized}
    if len(shapes) != 1 or any(item.shape[0] != 1 for item in normalized):
        raise ValueError(
            "online NFT requires uniform OmniMoT outputs with a leading singleton batch; "
            f"got {[tuple(item.shape) for item in normalized]}"
        )
    return torch.cat(normalized, dim=0)  # [B,...]


def _squeeze_stale_batch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # returns [...]
    if prediction.dim() == target.dim() + 1 and prediction.shape[0] == 1:
        return prediction.squeeze(0)  # [...]
    return prediction  # [...]


def _per_sample_mse_fp32(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    condition_mask: list[torch.Tensor] | None,
) -> torch.Tensor:  # returns [B]
    """Reduce Cosmos3 fp32 MSE per sample, excluding conditioning positions."""
    if len(predictions) != len(targets):
        raise ValueError(f"predictions/targets length mismatch: {len(predictions)} vs {len(targets)}")
    losses: list[torch.Tensor] = []
    if condition_mask is None:
        for prediction, target in zip(predictions, targets, strict=True):
            prediction = _squeeze_stale_batch(prediction, target)  # [...]
            losses.append(((prediction.float() - target.float()) ** 2).mean())  # []
        return torch.stack(losses)  # [B]
    if len(condition_mask) != len(predictions):
        raise ValueError(f"condition_mask length mismatch: {len(condition_mask)} vs {len(predictions)}")
    for prediction, target, mask in zip(predictions, targets, condition_mask, strict=True):
        prediction = _squeeze_stale_batch(prediction, target)  # [...]
        active = 1.0 - mask.to(prediction.device)  # [T,1,1]
        squared_error = (prediction.float() - target.float()) ** 2  # [...]
        denominator = active.expand_as(squared_error).sum().clamp_min(1.0)  # []
        losses.append((squared_error * active).sum() / denominator)  # []
    return torch.stack(losses)  # [B]


def _cosmos3_lora_pair_key(key: str) -> str | None:
    """Return the canonical sibling key for one Cosmos3 LoRA tensor."""
    parts = key.split(".")
    if len(parts) < 2 or parts[-1] != "weight":
        return None
    if parts[-2] == "lora_A":
        parts[-2] = "lora_B"
    elif parts[-2] == "lora_B":
        parts[-2] = "lora_A"
    else:
        return None
    return ".".join(parts)


def validate_cosmos3_cold_lora_inventory(
    *,
    source_keys: frozenset[str],
    target_only_keys: frozenset[str],
) -> None:
    """Validate that target-only copy-mode tensors are complete, cold Cosmos3 LoRA pairs."""
    source_lora = sorted(key for key in source_keys if _cosmos3_lora_pair_key(key) is not None)
    if source_lora:
        raise ValueError(
            f"cannot cold-start target LoRA because the source already contains LoRA tensors: {source_lora}"
        )
    non_lora = sorted(key for key in target_only_keys if _cosmos3_lora_pair_key(key) is None)
    if non_lora:
        raise ValueError(f"has non-LoRA target-only keys: {non_lora}")
    incomplete_lora = sorted(key for key in target_only_keys if _cosmos3_lora_pair_key(key) not in target_only_keys)
    if incomplete_lora:
        raise ValueError(f"has incomplete target-only LoRA pairs: {incomplete_lora}")


class Cosmos3FlowPolicy:
    """`FlowMatchingPolicy` over the OmniMoT (`cosmos_framework.model.generator.OmniMoTModel`)."""

    def __init__(self, omnimot: Any, build_ref: bool = True, build_old: bool = False) -> None:
        self.omnimot = omnimot
        self.net = omnimot.net
        ema_config = getattr(getattr(omnimot, "config", None), "ema", None)
        ema_enabled = bool(getattr(ema_config, "enabled", False))
        raw_ema = getattr(omnimot, "net_ema", None)
        if ema_enabled:
            worker = getattr(omnimot, "net_ema_worker", None)
            if raw_ema is None:
                raise RuntimeError("Cosmos3 EMA is enabled but the model has no net_ema")
            if not callable(getattr(worker, "update_average", None)) or not callable(
                getattr(omnimot, "ema_beta", None)
            ):
                raise RuntimeError("Cosmos3 EMA is enabled but its updater is unavailable")
        # Expose only an actively maintained OmniMoT EMA. A stale attribute from a disabled
        # configuration must not enter the trainer's initialization/checkpoint contract.
        self.net_ema: torch.nn.Module | None = raw_ema if ema_enabled else None
        # ``net_old`` is the periodically refreshed NFT behavior model. ``net_ref``
        # is an independent frozen KL anchor. DPO constructs only ``net_ref``.
        self.net_old: torch.nn.Module | None = self._build_frozen_snapshot("old-policy") if build_old else None
        self.net_ref: torch.nn.Module | None = self._build_frozen_snapshot("reference") if build_ref else None
        self._validation_depth: int = 0
        self._seeded_vision_active: bool = False
        self._fp32_rmsnorm_inventories: dict[str, dict[str, str]] = self._configure_fp32_rmsnorms()

    @contextmanager
    def validation_context(self) -> Iterator[None]:
        """Isolate Cosmos3-owned mutable state during held-out evaluation."""
        outermost = self._validation_depth == 0
        rope_states: tuple[_ModuleRuntimeState, ...] = ()
        callback_overrides: tuple[_MethodOverrideState, ...] = ()
        if outermost:
            self._validate_validation_compatibility()
            rope_states = _snapshot_dynamic_rope_state((self.net, self.net_old, self.net_ref))
            callback_overrides = _suppress_moe_callback_stats((self.net, self.net_old, self.net_ref, self.net_ema))
        self._validation_depth += 1
        try:
            yield
        finally:
            self._validation_depth -= 1
            if outermost:
                primary_error = sys.exc_info()[1]
                cleanup_failures: list[str] = []
                try:
                    _restore_moe_callback_stats(callback_overrides)
                except BaseException as error:  # noqa: BLE001 -- remaining cleanup must still run
                    cleanup_failures.append(f"training callback restore: {type(error).__name__}: {error}")
                try:
                    _restore_dynamic_rope_state(rope_states)
                except BaseException as error:  # noqa: BLE001 -- report every cleanup failure
                    cleanup_failures.append(f"dynamic RoPE restore: {type(error).__name__}: {error}")
                if cleanup_failures:
                    cleanup_error = RuntimeError("; ".join(cleanup_failures))
                    if primary_error is not None:
                        raise primary_error from cleanup_error
                    raise cleanup_error

    def reference_semantics_identity(self) -> dict[str, int | str]:
        """Identify non-tensor settings that affect encoding, packing, noise, or reference forward."""
        model_config = self.omnimot.config
        projection = {field: getattr(model_config, field) for field in _REFERENCE_SEMANTIC_CONFIG_FIELDS}
        rmsnorm_inventory = getattr(self, "_fp32_rmsnorm_inventories", {})
        return {
            "adapter": "posttrain.cosmos3-flow-policy",
            "contract_version": 2,
            "omnimot_config_sha256": canonical_json_sha256(
                canonical_config_value(projection),
                context="OmniMoT reference config",
            ),
            "rmsnorm_contract": "recognized-fp32-v1" if rmsnorm_inventory else "native-v1",
            "rmsnorm_inventory_sha256": canonical_json_sha256(
                canonical_config_value(rmsnorm_inventory),
                context="Cosmos3 post-training RMSNorm inventory",
            ),
        }

    def assert_same_condition_context(self, winner: FlowCondition, loser: FlowCondition) -> None:
        """Require two preference sides to differ only in generated content."""

        winner_condition = _require_cosmos3_condition(winner)
        loser_condition = _require_cosmos3_condition(loser)
        if len(winner_condition.text_tokens) != len(loser_condition.text_tokens):
            raise ValueError(
                f"[DPO] text_tokens length mismatch: "
                f"{len(winner_condition.text_tokens)} vs {len(loser_condition.text_tokens)}"
            )
        for index, (winner_tokens, loser_tokens) in enumerate(
            zip(winner_condition.text_tokens, loser_condition.text_tokens)
        ):
            if list(winner_tokens) != list(loser_tokens):
                raise ValueError(f"[DPO] winner/loser text_tokens differ at sample {index}; prompt must be shared")
        if winner_condition.is_image_batch != loser_condition.is_image_batch:
            raise ValueError(
                "[DPO] modality mismatch: "
                f"is_image_batch winner={winner_condition.is_image_batch} "
                f"loser={loser_condition.is_image_batch}"
            )
        if winner_condition.resolutions != loser_condition.resolutions:
            raise ValueError(
                "[DPO] winner/loser resolutions mismatch: "
                f"{winner_condition.resolutions!r} vs {loser_condition.resolutions!r}"
            )
        if len(winner_condition.sequence_plans) != len(loser_condition.sequence_plans):
            raise ValueError(
                f"[DPO] sequence_plans length mismatch: "
                f"{len(winner_condition.sequence_plans)} vs {len(loser_condition.sequence_plans)}"
            )
        for index, (winner_plan, loser_plan) in enumerate(
            zip(winner_condition.sequence_plans, loser_condition.sequence_plans)
        ):
            frames_w = list(getattr(winner_plan, "condition_frame_indexes_vision", None) or [])
            frames_l = list(getattr(loser_plan, "condition_frame_indexes_vision", None) or [])
            if frames_w != frames_l:
                raise ValueError(
                    f"[DPO] condition_frame_indexes_vision differ at sample {index}: {frames_w} vs {frames_l}"
                )
        fps_w = getattr(winner_condition.gen_data_clean, "fps_vision", None)
        fps_l = getattr(loser_condition.gen_data_clean, "fps_vision", None)
        if fps_w is None or fps_l is None:
            if fps_w is not fps_l:
                raise ValueError(f"[DPO] winner/loser fps_vision mismatch: {fps_w} vs {fps_l}")
            return
        if isinstance(fps_w, torch.Tensor) and isinstance(fps_l, torch.Tensor):
            equal = fps_w.shape == fps_l.shape and torch.equal(fps_w, fps_l)
        else:
            equal = fps_w == fps_l
        if not equal:
            raise ValueError(f"[DPO] winner/loser fps_vision mismatch: {fps_w} vs {fps_l}")

    def _configure_fp32_rmsnorms(self) -> dict[str, dict[str, str]]:
        """Keep policy, old, reference, and EMA numerically aligned in bf16 training."""
        if getattr(self.omnimot, "precision", None) is not torch.bfloat16:
            return {}
        networks = {"net": self.net, "net_old": self.net_old, "net_ref": self.net_ref, "net_ema": self.net_ema}
        inventories = {
            name: _enable_fp32_rmsnorms(network) for name, network in networks.items() if network is not None
        }
        policy_inventory = inventories.get("net", {})
        if not policy_inventory:
            raise RuntimeError(
                "bf16 Cosmos3 post-training found no recognized RMSNorm modules; "
                "refusing to run without the fp32 normalization contract"
            )
        if self.net_old is not None and inventories.get("net_old") != policy_inventory:
            raise RuntimeError(f"policy/old-policy fp32 RMSNorm inventories differ: {inventories}")
        if self.net_ref is not None and inventories.get("net_ref") != policy_inventory:
            raise RuntimeError(f"policy/reference fp32 RMSNorm inventories differ: {inventories}")
        if self.net_ema is not None and inventories.get("net_ema") != policy_inventory:
            raise RuntimeError(f"policy/EMA fp32 RMSNorm inventories differ: {inventories}")
        return inventories

    def _build_frozen_snapshot(self, snapshot_label: str) -> torch.nn.Module:
        """Build an eval-mode policy snapshot, preserving FSDP2 sharding when present."""
        net = self.net
        if is_dtensor_sharded(net):
            snapshot = self.omnimot.build_net(dtype=self.omnimot.precision)
            snapshot.requires_grad_(False)
            _copy_persistent_module_state(net, snapshot, snapshot_label=snapshot_label)
            snapshot.eval()
            return snapshot
        snapshot = copy.deepcopy(net)
        for p in snapshot.parameters():
            p.requires_grad_(False)
        snapshot.eval()
        return snapshot

    def refresh_ref(self) -> None:
        """Copy committed policy persistent state into a configured moving reference.

        Online NFT uses :meth:`refresh_old` for its behavior snapshot and keeps
        ``net_ref`` frozen when KL is enabled. This generic reference refresh remains
        available to other diffusion objectives.
        """
        if self.net_ref is None:
            raise ValueError("cannot refresh a missing reference network")
        _copy_persistent_module_state(self.net, self.net_ref, snapshot_label="reference")
        self.net_ref.eval()

    def refresh_old(self) -> None:
        """Copy committed policy persistent state into the moving NFT behavior model."""
        if self.net_old is None:
            raise ValueError("cannot refresh a missing old-policy network")
        _copy_persistent_module_state(self.net, self.net_old, snapshot_label="old-policy")
        self.net_old.eval()

    def _validate_validation_compatibility(self) -> None:
        """Reject model options whose state cannot be safely isolated for validation."""
        tokenizer = getattr(self.omnimot, "tokenizer_vision_gen", None)
        if bool(getattr(tokenizer, "keep_encoder_cache", False)):
            raise ValueError(
                "held-out objective validation requires tokenizer_vision_gen.keep_encoder_cache=False; "
                "a retained Wan2.1 encoder cache is order-dependent and cannot be restored safely"
            )
        if bool(getattr(self.net, "pad_for_cuda_graphs", False)):
            raise ValueError(
                "held-out objective validation requires net.pad_for_cuda_graphs=False; "
                "CUDA-graph sequence packing mutates process-global high-water bounds"
            )

    def _get_data_and_condition(
        self,
        batch: dict,
        sample_encode_seeds: list[int] | None = None,
    ) -> Any:
        """Encode media with optional stable per-item RNG and quiet validation telemetry."""
        if sample_encode_seeds is not None:
            item_counts = self._vision_item_counts(batch, len(sample_encode_seeds))
            with self._seeded_vision_encoder(sample_encode_seeds, item_counts):
                return self._get_data_and_condition_with_telemetry(batch)
        return self._get_data_and_condition_with_telemetry(batch)

    @contextmanager
    def _seeded_vision_encoder(self, root_seeds: list[int], item_counts: list[int]) -> Iterator[None]:
        """Give each OmniMoT vision item a stable RNG stream for validation."""
        if self._seeded_vision_active:
            raise RuntimeError("nested seeded vision encoding is not supported")
        if any(seed < 0 or seed >= 2**64 for seed in root_seeds):
            raise ValueError("root_seeds values must be unsigned 64-bit integers")
        if len(root_seeds) != len(item_counts) or any(count <= 0 for count in item_counts):
            raise ValueError("item_counts must contain one positive count per root seed")
        item_seeds = [
            int.from_bytes(
                hashlib.sha256(
                    _VISION_ITEM_SEED_DOMAIN
                    + root_seed.to_bytes(8, "big", signed=False)
                    + item_index.to_bytes(8, "big", signed=False)
                ).digest()[:8],
                "big",
            )
            for root_seed, item_count in zip(root_seeds, item_counts, strict=True)
            for item_index in range(item_count)
        ]
        owner = self.omnimot
        method_name = "_encode_vision_item"
        original = getattr(owner, method_name)
        instance_dict = getattr(owner, "__dict__", {})
        had_instance_override = method_name in instance_dict
        prior_instance_override = instance_dict.get(method_name)
        cursor = 0

        def seeded_call(*args: Any, **kwargs: Any) -> Any:
            nonlocal cursor
            if cursor >= len(item_seeds):
                raise RuntimeError(f"OmniMoT called {method_name} more than the {len(item_seeds)} seeded items")
            state = args[0]
            if state.ndim >= 5 and int(state.shape[0]) != 1:
                raise ValueError(
                    "seeded validation encoding requires one logical sample per vision item; "
                    f"got leading batch dimension {int(state.shape[0])} for shape {tuple(state.shape)}"
                )
            item_seed = item_seeds[cursor]
            cursor += 1
            device = self._vision_encoder_device(state)
            fork_devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.random.default_generator.manual_seed(item_seed)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(item_seed)
                return original(*args, **kwargs)

        self._seeded_vision_active = True
        try:
            setattr(owner, method_name, seeded_call)
            try:
                yield
            finally:
                if had_instance_override:
                    setattr(owner, method_name, prior_instance_override)
                else:
                    delattr(owner, method_name)
        finally:
            self._seeded_vision_active = False
        if cursor != len(item_seeds):
            raise RuntimeError(
                f"OmniMoT called {method_name} fewer times than seeded items: "
                f"expected {len(item_seeds)}, observed {cursor}"
            )

    def _get_data_and_condition_with_telemetry(self, batch: dict) -> Any:
        """Call OmniMoT's ``get_data_and_condition``.

        No longer needs to suppress anything during validation: OmniMoT's direct
        per-call W&B encode-time write (gated on ``iteration % log_enc_time_every_n``)
        was removed in favor of ``callbacks.iter_speed.IterSpeed``, which times VAE
        encode from its own training-step window and has no per-call side effect for
        a validation call to trip. ``get_data_and_condition`` itself no longer takes
        an ``iteration`` argument either. The method is kept (rather than calling
        ``m.get_data_and_condition`` directly at the one call site) so a future
        validation-specific concern has a single place to hook into.
        """
        return self.omnimot.get_data_and_condition(batch)

    def _vision_item_counts(self, batch: dict, sample_count: int) -> list[int]:
        """Return OmniMoT's flattened vision-item count for each logical sample."""
        m = self.omnimot
        configured_counts = batch.get("num_vision_items_per_sample")
        if configured_counts is not None:
            counts = [int(count) for count in configured_counts]
        else:
            is_image_batch = bool(m.is_image_batch(batch))
            media_key = m.input_image_key if is_image_batch else m.input_video_key
            sample_vision = batch[media_key]
            if len(sample_vision) != sample_count:
                raise ValueError(
                    "sample_encode_seeds must align with the logical vision batch: "
                    f"got {sample_count} seeds for {len(sample_vision)} samples"
                )
            has_multiple_items = any(isinstance(item, (list, tuple)) and len(item) > 1 for item in sample_vision)
            # Mirror OmniMoT's flattening rule exactly: once any logical sample is
            # multi-item, every entry is flattened and its ``len`` becomes the count.
            counts = [len(item) for item in sample_vision] if has_multiple_items else [1] * sample_count
        if len(counts) != sample_count:
            raise ValueError(
                "num_vision_items_per_sample must align with sample_encode_seeds: "
                f"got {len(counts)} counts for {sample_count} seeds"
            )
        if any(count <= 0 for count in counts):
            raise ValueError(f"vision item counts must be positive, got {counts}")
        return counts

    def _vision_encoder_device(self, state: torch.Tensor) -> torch.device:
        """Resolve the generator device used by the vision tokenizer's encode path."""
        tokenizer = getattr(self.omnimot, "tokenizer_vision_gen", None)
        declared_device = getattr(tokenizer, "device", None)
        if declared_device is not None:
            return torch.device(declared_device)
        tokenizer_model = getattr(tokenizer, "model", None)
        if isinstance(tokenizer_model, torch.nn.Module):
            parameter = next(tokenizer_model.parameters(), None)
            if parameter is not None:
                return parameter.device
            buffer = next(tokenizer_model.buffers(), None)
            if buffer is not None:
                return buffer.device
        tensor_kwargs = getattr(self.omnimot, "tensor_kwargs_fp32", None)
        if isinstance(tensor_kwargs, dict) and tensor_kwargs.get("device") is not None:
            return torch.device(tensor_kwargs["device"])
        return state.device

    # --- FlowMatchingPolicy ---
    def encode_and_condition(
        self,
        batch: dict,
        *,
        sample_encode_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, FlowCondition]:
        """Encode clean latents and build text tokens plus OmniMoT sequence plans."""
        m = self.omnimot
        gdc = self._get_data_and_condition(batch, sample_encode_seeds=sample_encode_seeds)
        if "text_token_ids" in batch:
            # iteration=1 is the upstream default; _load_and_tokenize_text_data ignores it.
            text_tokens = m._load_and_tokenize_text_data(batch, iteration=1)
        else:
            text_tokens, _ = m._get_inference_text_tokens(batch, has_negative_prompt=False)
        from cosmos_framework.data.generator.sequence_packing import build_sequence_plans_from_data_batch

        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=batch,
            input_video_key=m.input_video_key,
            input_image_key=m.input_image_key,
        )
        cond = _Cosmos3FlowCondition(
            gen_data_clean=gdc,
            sequence_plans=sequence_plans,
            text_tokens=text_tokens,
            is_image_batch=bool(gdc.is_image_batch),
            resolutions=self._resolutions(batch, gdc),
        )
        # x0 as a batched tensor for the protocol's `.shape` contract; denoise_error
        # re-splits it per sample. x0_tokens_vision is a list of [1,C,T,H,W].
        x0 = torch.cat(gdc.x0_tokens_vision, dim=0)
        return x0, cond

    def _resolutions(self, batch: dict, gdc: Any) -> list[str] | None:
        """Return per-sample keys for OmniMoT's resolution-dependent sigma shift."""
        if "image_size" not in batch:
            return None
        from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

        res = []
        for i in range(gdc.batch_size):
            img = batch["image_size"][i]
            if hasattr(img, "dim") and img.dim() == 2:
                img = img[0]
            res.append(get_vision_data_resolution((int(img[0].item()), int(img[1].item()))))
        return res

    def _pack(self, cond: FlowCondition, timesteps: torch.Tensor) -> Any:
        """Drive the real OmniMoT packer. `timesteps` must be CPU-resident `[B,1]`."""
        cosmos_condition = _require_cosmos3_condition(cond)
        return self.omnimot._pack_input_sequence(
            cosmos_condition.sequence_plans,
            cosmos_condition.text_tokens,
            cosmos_condition.gen_data_clean,
            timesteps.cpu(),
        )

    def condition_mask(self, cond: FlowCondition) -> FlowConditionMask:
        """Return packed clean-frame masks, or ``None`` when every frame is active."""
        cosmos_condition = _require_cosmos3_condition(cond)
        b = cosmos_condition.gen_data_clean.batch_size
        packed = self._pack(cond, torch.zeros(b, 1))
        if packed.vision is None or packed.vision.condition_mask is None:
            return None
        cm = packed.vision.condition_mask  # list of [T,1,1]
        if all(float(m.sum()) == 0.0 for m in cm):
            return None
        return cm

    def draw_sigma_epsilon(
        self,
        x0_shape: torch.Size,
        cond: FlowCondition,
        sample_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw OmniMoT noise fractions ``[B,1]`` and fp32 Gaussian noise."""
        m = self.omnimot
        cosmos_condition = _require_cosmos3_condition(cond)
        b = int(x0_shape[0])
        if sample_seeds is not None:
            if len(sample_seeds) != b:
                raise ValueError(f"sample_seeds length {len(sample_seeds)} does not match latent batch {b}")
            device = next(self.net.parameters()).device
            resolutions = cosmos_condition.resolutions
            sigma_rows: list[torch.Tensor] = []
            epsilon_rows: list[torch.Tensor] = []
            fork_devices = [device] if device.type == "cuda" else []
            for index, sample_seed in enumerate(sample_seeds):
                sample_resolutions = [resolutions[index]] if isinstance(resolutions, list) else resolutions
                with torch.random.fork_rng(devices=fork_devices):
                    # ``torch.manual_seed`` mutates every visible CUDA generator. Held-out
                    # objective sampling owns only CPU and the model's current device.
                    torch.random.default_generator.manual_seed(sample_seed)
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(sample_seed)
                    _timesteps, sample_sigma = m._get_train_noise_level_vision(
                        batch_size=1,
                        is_image_batch=cosmos_condition.is_image_batch,
                        num_vision_latent_frames=[int(x0_shape[2])],
                        resolutions=sample_resolutions,
                    )  # [1, 1], [1, 1]
                    if tuple(sample_sigma.shape) != (1, 1):
                        raise ValueError(
                            "Flow-DPO currently requires one noise level per sample; "
                            f"got sigma shape {tuple(sample_sigma.shape)}. "
                            "diffusion_forcing per-frame noise is not supported by the held-out objective."
                        )
                    sample_epsilon = torch.randn(
                        (1, *x0_shape[1:]),
                        device=device,
                        dtype=torch.float32,
                    )  # [1, ...]
                sigma_rows.append(sample_sigma)
                epsilon_rows.append(sample_epsilon)
            sigmas = torch.cat(sigma_rows, dim=0)  # [B, 1]
            epsilon = torch.cat(epsilon_rows, dim=0)  # [B, ...]
            return sigmas, epsilon
        _timesteps, sigmas = m._get_train_noise_level_vision(
            batch_size=b,
            is_image_batch=cosmos_condition.is_image_batch,
            num_vision_latent_frames=[int(x0_shape[2])] * b,
            resolutions=cosmos_condition.resolutions,
        )
        if tuple(sigmas.shape) != (b, 1):
            raise ValueError(
                "Flow-DPO currently requires one noise level per sample; "
                f"got sigma shape {tuple(sigmas.shape)}. "
                "diffusion_forcing per-frame noise is not supported by the held-out objective."
            )
        device = next(self.net.parameters()).device
        epsilon = torch.randn(x0_shape, device=device, dtype=torch.float32)  # [B, ...]
        return sigmas, epsilon  # sigmas: [B,1] RF fraction ∈ [0,1]

    def draw_nft_sigma_epsilon(self, x0_shape: torch.Size, cond: FlowCondition) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw NFT epsilon before sigma without changing DPO's pinned RNG trace."""
        cosmos_condition = _require_cosmos3_condition(cond)
        device = next(self.net.parameters()).device
        epsilon = torch.randn(x0_shape, device=device, dtype=torch.float32)  # [B,...]
        model = self.omnimot
        batch = int(x0_shape[0])
        _timesteps, sigmas = model._get_train_noise_level_vision(
            batch_size=batch,
            is_image_batch=cosmos_condition.is_image_batch,
            num_vision_latent_frames=[int(x0_shape[2])] * batch,
            resolutions=cosmos_condition.resolutions,
        )  # [B,1], [B,1]
        return sigmas, epsilon

    def update_ema(self, iteration: int) -> None:
        """Advance OmniMoT EMA after one applied optimizer update."""
        if self.net_ema is None:
            return
        worker = getattr(self.omnimot, "net_ema_worker", None)
        ema_beta = getattr(self.omnimot, "ema_beta", None)
        if worker is None or not callable(ema_beta):
            raise RuntimeError("Cosmos3 EMA is enabled but its updater is unavailable")
        worker.update_average(self.net, self.net_ema, beta=float(ema_beta(iteration)))
        sync_ema_buffers = getattr(self.omnimot, "sync_ema_buffers", None)
        if not callable(sync_ema_buffers):
            raise RuntimeError("Cosmos3 EMA is enabled but its persistent-buffer synchronizer is unavailable")
        sync_ema_buffers()

    def prepare_denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> _PreparedDenoise:
        """Prepare one side completely before any rank enters its FSDP forward."""
        del net
        m = self.omnimot
        is_image = _require_cosmos3_condition(cond).is_image_batch
        cosmos_condition_mask = _require_cosmos3_condition_mask(condition_mask)
        rf = m.rectified_flow_image if is_image else m.rectified_flow_video
        n_train = rf.noise_scheduler.config.num_train_timesteps

        b = int(sigma.shape[0])
        t_B = sigma[:, 0]  # [B]
        timesteps = (t_B * n_train).unsqueeze(1)  # [B,1] — matches OmniMoT σ·max_timestep
        packed = self._pack(cond, timesteps)

        # σ_eff = t·(1 - condition_mask): conditioning frames stay clean (σ=0). The
        # velocity target v* = ε - x0 is unmasked (get_interpolation's dot_x_t is
        # t-independent); masking is applied only in the loss reduction.
        if cosmos_condition_mask is None:
            sigmas_list = [t_B[i].view(1, 1, 1) for i in range(b)]
        else:
            sigmas_list = [t_B[i].view(1, 1, 1) * (1.0 - cosmos_condition_mask[i].to(t_B.device)) for i in range(b)]
        eps_list = _unbatch(epsilon)
        x0_list = _unbatch(x0)

        # RF interpolation: x_0 arg is NOISE, x_1 is clean (rectified_flow.py).
        xt_list, vt_list = rf.get_interpolation(eps_list, x0_list, sigmas_list)
        xt_list = [x.to(**m.tensor_kwargs) for x in xt_list]  # cast to model precision

        packed.vision.tokens = xt_list  # wire noised tokens (cosmos-rl _replace_clean_with_noised)
        packed.to_cuda()

        return _PreparedDenoise(packed=packed, targets=vt_list, condition_mask=cosmos_condition_mask)

    def denoise_prediction(self, *, net: torch.nn.Module, prepared: _PreparedDenoise) -> list[torch.Tensor]:
        """Run only OmniMoT's distributed denoiser on prevalidated packed inputs."""
        m = self.omnimot

        # Respect inference/no-grad validation while retaining gradients for policy training.
        grad = net is self.net and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad):
            out = m.denoise(net=net, data_batch_packed=prepared.packed)
        return out["preds_vision"]

    def finalize_denoise_error(
        self,
        *,
        prepared: _PreparedDenoise,
        prediction: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the per-sample fp32 error after every rank completed its forward."""
        return _per_sample_mse_fp32(prediction, prepared.targets, prepared.condition_mask)  # [B]

    def denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> torch.Tensor:
        """Run one complete staged denoiser call."""
        prepared = self.prepare_denoise_error(
            net=net,
            x0=x0,
            cond=cond,
            sigma=sigma,
            epsilon=epsilon,
            condition_mask=condition_mask,
        )
        prediction = self.denoise_prediction(net=net, prepared=prepared)
        return self.finalize_denoise_error(prepared=prepared, prediction=prediction)

    def nft_velocity_predictions(
        self,
        *,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> NFTVelocityPredictions:
        """Evaluate policy/old/ref velocities at one condition-aware RF point."""
        if self.net_old is None:
            raise ValueError("online NFT requires an old-policy network")

        prepared = run_rank_converged(
            lambda: self._prepare_nft_velocity(
                x0=x0,
                cond=cond,
                sigma=sigma,
                epsilon=epsilon,
                condition_mask=condition_mask,
            ),
            "Cosmos3 NFT forward preparation",
        )
        policy_out = self._nft_velocity_forward(net=self.net, packed=prepared.packed_policy)
        old_out = self._nft_velocity_forward(net=self.net_old, packed=prepared.packed_old)
        reference_out = (
            self._nft_velocity_forward(net=self.net_ref, packed=prepared.packed_reference)
            if self.net_ref is not None and prepared.packed_reference is not None
            else None
        )
        return run_rank_converged(
            lambda: self._finalize_nft_velocity(
                prepared=prepared,
                policy_out=policy_out,
                old_out=old_out,
                reference_out=reference_out,
            ),
            "Cosmos3 NFT forward finalization",
        )

    def _prepare_nft_velocity(
        self,
        *,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> _PreparedNFTVelocity:
        """Prepare every local NFT input before any rank enters a denoiser forward."""
        batch = int(x0.shape[0])
        if tuple(epsilon.shape) != tuple(x0.shape):
            raise ValueError(f"epsilon shape {tuple(epsilon.shape)} != x0 shape {tuple(x0.shape)}")
        if tuple(sigma.shape) != (batch, 1):
            raise ValueError(f"sigma must have shape ({batch}, 1), got {tuple(sigma.shape)}")

        model = self.omnimot
        is_image = _require_cosmos3_condition(cond).is_image_batch
        cosmos_condition_mask = _require_cosmos3_condition_mask(condition_mask)
        rf = model.rectified_flow_image if is_image else model.rectified_flow_video
        num_train_timesteps = rf.noise_scheduler.config.num_train_timesteps
        timestep_fraction = sigma[:, 0]  # [B]
        timesteps = (timestep_fraction * num_train_timesteps).unsqueeze(1)  # [B,1]
        packed_policy = self._pack(cond, timesteps)
        packed_old = self._pack(cond, timesteps)
        packed_reference = self._pack(cond, timesteps) if self.net_ref is not None else None

        x0_items = _unbatch(x0)  # B * [1,C,T,H,W]
        epsilon_items = _unbatch(epsilon)  # B * [1,C,T,H,W]
        if cosmos_condition_mask is None:
            effective_t_items = [timestep_fraction[index].view(1, 1, 1) for index in range(batch)]  # B * [1,1,1]
        else:
            if len(cosmos_condition_mask) != batch:
                raise ValueError(f"condition-mask length {len(cosmos_condition_mask)} != batch {batch}")
            effective_t_items = [
                timestep_fraction[index].view(1, 1, 1)
                * (1.0 - cosmos_condition_mask[index].to(timestep_fraction.device))
                for index in range(batch)
            ]  # B * [T,1,1]
        x_t_items, _targets = rf.get_interpolation(
            epsilon_items,
            x0_items,
            effective_t_items,
        )  # B * [1,C,T,H,W], B * RF targets
        x_t_model = [item.to(**model.tensor_kwargs) for item in x_t_items]  # B * [1,C,T,H,W]
        packed_policy.vision.tokens = x_t_model
        packed_old.vision.tokens = [item.clone() for item in x_t_model]  # B * [1,C,T,H,W]
        if packed_reference is not None:
            packed_reference.vision.tokens = [item.clone() for item in x_t_model]  # B * [1,C,T,H,W]
        packed_policy.to_cuda()
        packed_old.to_cuda()
        if packed_reference is not None:
            packed_reference.to_cuda()

        return _PreparedNFTVelocity(
            packed_policy=packed_policy,
            packed_old=packed_old,
            packed_reference=packed_reference,
            x0_items=x0_items,
            x_t_items=x_t_items,
            effective_t_items=effective_t_items,
            timestep_fraction=timestep_fraction,
            x0_dtype=x0.dtype,
            condition_mask_is_none=cosmos_condition_mask is None,
        )

    def _nft_velocity_forward(self, *, net: torch.nn.Module, packed: Any) -> object:
        """Run one fixed-order collective-bearing NFT denoiser forward.

        A failure inside the model's collectives is fatal and cannot be made rank-convergent after the fact.
        Keep rank-local result validation in ``_finalize_nft_velocity`` so every rank enters all configured
        policy, old-policy, and reference forwards in the same order.
        """
        grad = net is self.net and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad):
            return self.omnimot.denoise(net=net, data_batch_packed=packed)

    def _finalize_nft_velocity(
        self,
        *,
        prepared: _PreparedNFTVelocity,
        policy_out: object,
        old_out: object,
        reference_out: object | None,
    ) -> NFTVelocityPredictions:
        """Validate and rebatch ordered NFT outputs after every rank completed all forwards."""
        batch = len(prepared.x0_items)
        policy_predictions = _nft_vision_predictions(policy_out, role="policy")
        old_predictions = _nft_vision_predictions(old_out, role="old-policy")
        reference_predictions = (
            _nft_vision_predictions(reference_out, role="reference") if prepared.packed_reference is not None else None
        )

        x_t = torch.cat(prepared.x_t_items, dim=0).to(prepared.x0_dtype)  # [B,C,T,H,W]
        v_policy = _rebatch_uniform(policy_predictions, prepared.x0_items).to(prepared.x0_dtype)  # [B,C,T,H,W]
        v_old = _rebatch_uniform(old_predictions, prepared.x0_items).to(prepared.x0_dtype)  # [B,C,T,H,W]
        v_ref = (
            _rebatch_uniform(reference_predictions, prepared.x0_items).to(prepared.x0_dtype)
            if reference_predictions is not None
            else None
        )  # [B,C,T,H,W] | None
        if prepared.condition_mask_is_none:
            t_effective = prepared.timestep_fraction.view(batch, 1, 1, 1, 1).to(prepared.x0_dtype)  # [B,1,1,1,1]
        else:
            t_effective = torch.stack([item.unsqueeze(0) for item in prepared.effective_t_items], dim=0).to(
                prepared.x0_dtype
            )  # [B,1,T,1,1]
        expected_shape = tuple(x_t.shape)
        for name, prediction in (("v_policy", v_policy), ("v_old", v_old)):
            if tuple(prediction.shape) != expected_shape:
                raise ValueError(f"{name} shape {tuple(prediction.shape)} != noised latent shape {expected_shape}")
        if v_ref is not None and tuple(v_ref.shape) != expected_shape:
            raise ValueError(f"v_ref shape {tuple(v_ref.shape)} != noised latent shape {expected_shape}")
        try:
            torch.broadcast_shapes(tuple(t_effective.shape), expected_shape)
        except RuntimeError as error:
            raise ValueError(
                f"effective timestep shape {tuple(t_effective.shape)} is not broadcastable to {expected_shape}"
            ) from error
        return NFTVelocityPredictions(
            x_t=x_t,
            t_effective=t_effective,
            v_policy=v_policy,
            v_old=v_old,
            v_ref=v_ref,
        )

    def sample_final_latents(
        self,
        model_batch: dict,
        *,
        sampler: Any,
        seeds: list[int],
        num_steps: int,
        shift: float,
        guidance: float,
    ) -> list[torch.Tensor]:
        """Drive OmniMoT's native generation path and return final vision latents."""
        result = self.omnimot.generate_samples_from_batch(
            model_batch,
            net=self.net,
            sampler=sampler,
            guidance=guidance,
            seed=seeds,
            n_sample=len(seeds),
            num_steps=num_steps,
            shift=shift,
        )
        latents = result.get("vision")
        if not isinstance(latents, list) or any(not isinstance(latent, torch.Tensor) for latent in latents):
            raise TypeError("OmniMoT generation must return result['vision'] as list[Tensor]")
        if len(latents) != len(seeds):
            raise ValueError(f"OmniMoT returned {len(latents)} vision latents for {len(seeds)} seeds")
        return latents

    def decode_final_latents(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Decode one uniform rollout group with a single vision-tokenizer call."""
        if not latents:
            raise ValueError("cannot decode an empty NFT latent group")
        if any(latent.ndim != 5 or latent.shape[0] != 1 for latent in latents):
            raise ValueError(f"NFT reward decode requires [1,C,T,H,W] latents, got {[tuple(x.shape) for x in latents]}")
        shapes = {tuple(latent.shape[1:]) for latent in latents}
        if len(shapes) != 1:
            raise ValueError(f"NFT reward decode requires uniform latents, got {[tuple(x.shape) for x in latents]}")
        latent_batch = torch.cat(latents, dim=0)  # [B,C,T,H,W]
        with torch.no_grad():
            decoded = self.omnimot.decode(latent_batch)  # [B,3,T,H,W]
        if not isinstance(decoded, torch.Tensor) or decoded.ndim != 5:
            raise TypeError("OmniMoT decode must return a [B,C,T,H,W] tensor")
        return decoded

    def video_reward_decoder(self) -> _Cosmos3VideoRewardDecoder:
        """Expose the model-owned decoder with its native ``[-1, 1]`` contract."""
        return _Cosmos3VideoRewardDecoder(self)


class _Cosmos3VideoRewardDecoder:
    """Callable decoder capability consumed by the generic video reward provider."""

    output_range: Literal["neg_one_to_one"] = "neg_one_to_one"

    def __init__(self, policy: Cosmos3FlowPolicy) -> None:
        self.policy = policy

    def __call__(self, latents: list[torch.Tensor]) -> torch.Tensor:
        return self.policy.decode_final_latents(latents)


@register_model("cosmos3")
def _build_cosmos3(cfg: Any) -> Cosmos3FlowPolicy:
    """Build from the OmniMoT instance supplied by the experiment builder."""
    omnimot = getattr(cfg, "omnimot", None)
    if omnimot is None:
        raise ValueError("build cosmos3: pass a constructed OmniMoTModel via cfg.omnimot")
    return Cosmos3FlowPolicy(
        omnimot,
        build_ref=getattr(cfg, "build_ref", True),
        build_old=getattr(cfg, "build_old", False),
    )
