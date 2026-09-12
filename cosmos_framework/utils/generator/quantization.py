# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-precision quantization helpers for the Cosmos3 VFM MoT network.

Quantization is applied via torchao's :func:`apply_quantization_inplace`, which
uses the ``quantize_`` path to replace each selected weight with a quantized
tensor subclass in place. Because the live parameter becomes a tensor subclass,
this only works on unsharded (plain-tensor) params and is therefore restricted
to replicated inference (``data_parallel_shard_degree == 1``); it cannot be
applied to an FSDP-sharded model whose params are ``DTensor`` shards.

ModelOpt FP8 checkpoint loading installs the exported E4M3 weights and static
scales into TorchAO tensor subclasses without calibration or re-quantization.

This is an inference-only path: the ``quantize_`` PTQ configs have no backward
support. Module selection is delegated to the filter built by
:func:`_get_filter_fn`.

The ``int8_sim`` / ``fp8_sim`` methods are torchao-free quantize-dequantize
(Q/DQ) simulations built on :class:`QdqSimLinear`: weights are fake-quantized
once at install time, input activations on every call, and the GEMM itself stays
a dense compute-dtype (bf16) matmul. They reuse the exact same module selection
as the torchao methods, so the quantized module set is identical by construction.
"""

import gc
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn import functional as F

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.utils import log

# NOTE: ``torchao`` is imported lazily inside the functions below rather than at
# module top level. These two helpers are the only torchao consumers, but this
# module is imported transitively by the model package (e.g. during tests that
# never quantize). Keeping the imports lazy means importing this module does not
# require torchao to be installed; the imports only run — and only fail — when
# quantization is actually requested.


class _ModelOptFloat8Linear(nn.Linear):
    """Work around TorchAO 0.16 static-FP8 limitations for linear inputs."""

    # Compute dtype the FP8 weight dequantizes to. Recorded when the module is
    # swapped in on meta, where the plain E4M3 placeholder no longer carries it.
    _modelopt_high_precision_dtype: torch.dtype | None = None
    # True once real checkpoint data has been installed.
    _modelopt_weight_loaded: bool = False
    # Mixed-precision diffusion steps (see utils/generator/mixed_precision.py).
    # None until install_mixed_precision_runtime tags this module.
    _mixed_precision_runtime = None
    _mixed_precision_path: str = "generation"
    # A (N, K) dense view staged by the block provider for the current decoder
    # layer, or None. Read here, written only by the provider hooks.
    _mixed_precision_staged_weight: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_shape = (*inputs.shape[:-1], self.out_features)
        # PrototypeFloat8Tensor divides each input dimension by its block size, so
        # a zero-row input fails with 0 // 0 instead of producing an empty output.
        if inputs.numel() == 0:
            return inputs.new_empty(output_shape)
        # PrototypeFloat8Tensor requires the input and its static (1, 1) activation
        # scale to have equal rank, so flatten token dimensions before dispatch.
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        runtime = self._mixed_precision_runtime
        if runtime is not None and runtime.use_high_precision(self._mixed_precision_path):
            # W8A16: dense GEMM against the dequantized FP8 weight; the
            # activation stays in the compute dtype. The resolved weight is
            # in the runtime's configured activation_dtype, which normally
            # matches the live activations -- but defensively cast both
            # operands to the weight's dtype and cast the result back to the
            # input dtype when they differ (e.g. a fp16-activation model run
            # against a bf16-configured runtime), instead of crashing.
            weight = runtime.resolve_w8a16_weight(self)
            if weight.dtype != flat_inputs.dtype:
                output = F.linear(flat_inputs.to(weight.dtype), weight, self.bias).to(inputs.dtype)
            else:
                output = F.linear(flat_inputs, weight, self.bias)
            return output.reshape(output_shape)
        flat_outputs = F.linear(flat_inputs, self.weight, self.bias)
        return flat_outputs.reshape(output_shape)


_torchao_fsdp_support_installed = False


def _rewrap_float8(source, qdata: torch.Tensor):
    """Rebuild a static-FP8 tensor around new ``qdata``, carrying the scales over.

    Valid only because ModelOpt exports PerTensor granularity: one scalar scale
    covers the whole weight, so reshaping or slicing the data never redistributes
    it. ``block_size`` is therefore always "one block spanning everything".
    """
    return source.__class__(
        qdata,
        source.scale,
        act_quant_scale=source.act_quant_scale,
        output_act_quant_scale=source.output_act_quant_scale,
        block_size=list(qdata.shape),
        mm_config=source.mm_config,
        act_quant_kwargs=source.act_quant_kwargs,
        kernel_preference=source.kernel_preference,
        dtype=source.dtype,
        output_act_quant_kwargs=source.output_act_quant_kwargs,
    )


def install_torchao_float8_fsdp_support() -> None:
    """Teach TorchAO's static-FP8 tensor the ops and hooks FSDP2 needs.

    TorchAO 0.16 ships this tensor for single-device inference: it implements no
    ``fsdp_pre_all_gather`` / ``fsdp_post_all_gather``, and its shape-changing ops
    derive a scale layout from ``block_size`` assuming a fixed rank, so FSDP2's
    reshape/split/allocate path fails on a 2D linear weight. Sharding it is
    nonetheless well defined for a PerTensor scale — the scalar is identical on
    every shard, so only the FP8 bytes ever need to move.

    Registered once, globally, since the dispatch table is keyed by class.
    """
    global _torchao_fsdp_support_installed
    if _torchao_fsdp_support_installed:
        return
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.utils import return_and_correct_aliasing

    aten = torch.ops.aten
    implements = PrototypeFloat8Tensor.implements

    @implements([aten.view.default, aten._unsafe_view.default, aten.reshape.default])
    def _(func, types, args, kwargs):
        self, size = args[0], args[1]
        return return_and_correct_aliasing(func, args, kwargs, _rewrap_float8(self, self.qdata.reshape(*size)))

    @implements(aten.split.Tensor)
    def _(func, types, args, kwargs):
        self, size = args[0], args[1]
        dim = args[2] if len(args) > 2 else 0
        return [_rewrap_float8(self, part) for part in self.qdata.split(size, dim)]

    @implements(aten.slice.Tensor)
    def _(func, types, args, kwargs):
        self = args[0]
        return return_and_correct_aliasing(
            func, args, kwargs, _rewrap_float8(self, aten.slice.Tensor(self.qdata, *args[1:]))
        )

    @implements(aten.as_strided.default)
    def _(func, types, args, kwargs):
        self, size, stride = args[0], args[1], args[2]
        offset = args[3] if len(args) > 3 else 0
        return return_and_correct_aliasing(
            func, args, kwargs, _rewrap_float8(self, self.qdata.as_strided(size, stride, offset))
        )

    @implements(aten.new_zeros.default)
    def _(func, types, args, kwargs):
        self, size = args[0], args[1]
        return _rewrap_float8(self, self.qdata.new_zeros(size))

    @implements([aten.empty_like.default, aten.zeros_like.default])
    def _(func, types, args, kwargs):
        self = args[0]
        device = kwargs.get("device")
        qdata = func(self.qdata, **{key: value for key, value in kwargs.items() if key != "dtype"})
        # `to_empty` moves storage without data, so the scales are allocated empty
        # on the target device as well: `.to(device)` would try to copy out of a
        # meta tensor. Real scale values arrive with the checkpoint weights.
        scale = torch.empty_like(self.scale, device=device) if device is not None else self.scale
        act_quant_scale = (
            torch.empty_like(self.act_quant_scale, device=device) if device is not None else self.act_quant_scale
        )
        return self.__class__(
            qdata,
            scale,
            act_quant_scale=act_quant_scale,
            output_act_quant_scale=self.output_act_quant_scale,
            block_size=list(qdata.shape),
            mm_config=self.mm_config,
            act_quant_kwargs=self.act_quant_kwargs,
            kernel_preference=self.kernel_preference,
            dtype=self.dtype,
            output_act_quant_kwargs=self.output_act_quant_kwargs,
        )

    @implements([aten.detach.default, aten.clone.default])
    def _(func, types, args, kwargs):
        self = args[0]
        return return_and_correct_aliasing(func, args, kwargs, _rewrap_float8(self, func(self.qdata)))

    @implements(aten._to_copy.default)
    def _(func, types, args, kwargs):
        # Device moves must go through, but a dtype request must not: casting the
        # E4M3 payload to the compute dtype would silently undo the quantization.
        self = args[0]
        forwarded = {key: value for key, value in kwargs.items() if key != "dtype"}
        return return_and_correct_aliasing(func, args, kwargs, _rewrap_float8(self, func(self.qdata, **forwarded)))

    @implements(aten.copy_.default)
    def _(func, types, args, kwargs):
        destination, source = args[0], args[1]
        if isinstance(source, PrototypeFloat8Tensor):
            destination.qdata.copy_(source.qdata)
            destination.scale.copy_(source.scale)
            destination.act_quant_scale.copy_(source.act_quant_scale)
        else:
            destination.qdata.copy_(source)
        return destination

    def fsdp_pre_all_gather(self, mesh):
        # Only the FP8 bytes travel. The static scales are already identical on
        # every rank, so keeping them out of the collective saves the traffic.
        return (self.qdata,), (self.scale, self.act_quant_scale, self.dtype)

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        (qdata,) = all_gather_outputs
        scale, act_quant_scale, dtype = metadata
        if out is not None:
            return None
        gathered = self.__class__(
            qdata,
            scale,
            act_quant_scale=act_quant_scale,
            output_act_quant_scale=self.output_act_quant_scale,
            block_size=list(qdata.shape),
            mm_config=self.mm_config,
            act_quant_kwargs=self.act_quant_kwargs,
            kernel_preference=self.kernel_preference,
            dtype=param_dtype or dtype,
            output_act_quant_kwargs=self.output_act_quant_kwargs,
        )
        return gathered, (qdata,)

    PrototypeFloat8Tensor.fsdp_pre_all_gather = fsdp_pre_all_gather
    PrototypeFloat8Tensor.fsdp_post_all_gather = fsdp_post_all_gather
    _torchao_fsdp_support_installed = True
    log.info("Installed TorchAO static-FP8 FSDP support (8 aten ops + all-gather hooks)")


def is_modelopt_fp8_checkpoint(checkpoint_path: str | Path) -> bool:
    """Return whether a local checkpoint declares ModelOpt FP8 quantization."""
    quant_config_path = Path(checkpoint_path) / "hf_quant_config.json"
    if not quant_config_path.is_file():
        return False
    try:
        quant_config = json.loads(quant_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read a valid checkpoint quantization config: {quant_config_path}") from error
    if (
        not isinstance(quant_config, dict)
        or quant_config.get("quant_method") != "modelopt"
        or quant_config.get("quant_algo") != "FP8"
    ):
        raise ValueError(f"Unsupported checkpoint quantization configuration: {quant_config_path}")
    return True


def collect_modelopt_fp8_params(model: nn.Module) -> set[nn.Parameter]:
    """Return the FP8 weights of every ModelOpt linear currently in ``model``.

    Collected by module type immediately before use rather than captured earlier,
    because parallelization rebuilds parts of the network (notably the
    ``language_model`` subtree) and hands back fresh parameter objects, so any
    set gathered ahead of that no longer matches by identity.
    """
    return {module.weight for module in model.modules() if isinstance(module, _ModelOptFloat8Linear)}


def plan_modelopt_fp8_targets(
    checkpoint_path: str | Path,
    key_mapper: Callable[[str, str], str | None],
    *,
    weight_map: dict[str, str],
) -> list[str]:
    """Return the network-relative FQNs a ModelOpt FP8 checkpoint quantizes.

    Derived from the checkpoint index alone — no model instance is required — so
    the result can be threaded through the model config and consumed by
    ``build_net`` before the network exists. Only the E4M3 tensors are reported;
    weights ModelOpt intentionally left in high precision (``proj_in``,
    ``lm_head``, the vision tower, ...) are skipped.

    Args:
        checkpoint_path: Local root of the ModelOpt diffusers checkpoint.
        key_mapper: Maps a diffusers source key and shard path to a model state key.
        weight_map: Merged checkpoint weight map.

    Returns:
        Sorted target module FQNs, relative to the VFM network.
    """
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_path)
    weights_by_shard: dict[str, list[str]] = {}
    for source_key, shard_path in weight_map.items():
        if source_key.endswith(".weight"):
            weights_by_shard.setdefault(shard_path, []).append(source_key)

    target_fqns: set[str] = set()
    for shard_path, source_weight_keys in sorted(weights_by_shard.items()):
        with safe_open(checkpoint_path / shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
            for source_weight_key in sorted(source_weight_keys):
                if source_weight_key not in shard_keys:
                    continue
                if shard.get_slice(source_weight_key).get_dtype() != "F8_E4M3":
                    continue
                target_weight_key = key_mapper(source_weight_key, shard_path)
                if target_weight_key is None or not target_weight_key.endswith(".weight"):
                    continue
                target_fqns.add(target_weight_key.removesuffix(".weight"))
    return sorted(target_fqns)


def swap_modelopt_fp8_linears_on_meta(model: nn.Module, target_fqns: list[str]) -> list[str]:
    """Swap the given linears to FP8 modules while the network is still on meta.

    This must run *before* the network is parallelized and materialized. FSDP2
    wraps the whole network into a single parameter group, so a linear replaced
    after ``fully_shard`` would leave the group holding a stale parameter; and
    ``to_empty`` on the original bf16 shapes is what sets peak memory, which for
    a Super-class model exceeds a single 80 GB device. Swapping here means FSDP
    shards — and ``to_empty`` materializes — one byte per element instead of two.

    The weight is created as a real (meta) TorchAO static-FP8 tensor rather than a
    plain E4M3 placeholder, because FSDP decides at wrap time whether a parameter
    carries the ``fsdp_pre_all_gather`` extension. A plain tensor swapped to a
    subclass afterwards is already registered as an ordinary parameter and fails
    on the first all-gather. Actual weight bytes and scales arrive later, in
    :func:`apply_modelopt_fp8_checkpoint_inplace`.

    Args:
        model: Meta-device VFM network.
        target_fqns: Module FQNs to convert, as planned from the checkpoint.

    Returns:
        Sorted FQNs actually swapped; targets absent from this model variant are
        skipped, matching the loader's tolerance for task-specialized exports.
    """
    install_torchao_float8_fsdp_support()
    from torchao.float8.inference import Float8MMConfig
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization import PerTensor
    from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

    mm_config = Float8MMConfig(use_fast_accum=True)
    activation_quant_kwargs = QuantizeTensorToFloat8Kwargs(
        float8_dtype=torch.float8_e4m3fn,
        granularity=PerTensor(),
        mm_config=mm_config,
    )
    swapped_fqns: list[str] = []
    for target_module_fqn in sorted(target_fqns):
        try:
            module = model.get_submodule(target_module_fqn)
        except AttributeError:
            continue
        if not isinstance(module, nn.Linear):
            raise KeyError(f"ModelOpt FP8 target {target_module_fqn!r} is not an nn.Linear module")

        replacement = _ModelOptFloat8Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
        )
        replacement._modelopt_high_precision_dtype = module.weight.dtype
        quantized_shape = tuple(module.weight.shape)
        replacement.weight = nn.Parameter(
            PrototypeFloat8Tensor(
                torch.empty(quantized_shape, dtype=torch.float8_e4m3fn, device="meta"),
                torch.empty(1, 1, dtype=torch.float32, device="meta"),
                act_quant_scale=torch.empty(1, 1, dtype=torch.float32, device="meta"),
                block_size=list(quantized_shape),
                mm_config=mm_config,
                act_quant_kwargs=activation_quant_kwargs,
                dtype=module.weight.dtype,
            ),
            requires_grad=False,
        )
        if module.bias is not None:
            replacement.bias = module.bias
        replacement.train(module.training)

        parent_fqn, _, child_name = target_module_fqn.rpartition(".")
        parent = model.get_submodule(parent_fqn) if parent_fqn else model
        setattr(parent, child_name, replacement)
        swapped_fqns.append(target_module_fqn)

    log.info(f"Swapped {len(swapped_fqns)} linears to meta-device ModelOpt FP8 modules")
    return swapped_fqns


def apply_modelopt_fp8_checkpoint_inplace(
    model: nn.Module,
    checkpoint_path: str | Path,
    key_mapper: Callable[[str, str], str | None],
    *,
    weight_map: dict[str, str],
) -> list[str]:
    """Install ModelOpt static-FP8 checkpoint tensors as TorchAO weights.

    ModelOpt's HF export stores the already-quantized E4M3 bytes under ``weight``
    and its per-tensor dequantization scales under ``weight_scale`` and
    ``input_scale``. This adapter streams those tensors from the checkpoint and
    constructs TorchAO ``PrototypeFloat8Tensor`` weights directly, avoiding any
    dequantization, recalibration, or re-quantization.

    Args:
        model: Unsharded Cosmos3 network whose linears will be converted in place.
        checkpoint_path: Local root of the ModelOpt diffusers checkpoint.
        key_mapper: Maps a diffusers source key and shard path to a model state key.
        weight_map: Merged checkpoint weight map.

    Returns:
        Sorted fully qualified names of converted linear modules.
    """
    checkpoint_path = Path(checkpoint_path)
    if not is_modelopt_fp8_checkpoint(checkpoint_path):
        raise ValueError(f"Not a ModelOpt FP8/FP8 checkpoint: {checkpoint_path}")

    from safetensors import safe_open
    from torchao.float8.inference import Float8MMConfig
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization import PerTensor
    from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

    source_weights_by_shard: dict[str, list[str]] = {}
    for source_key, shard_path in weight_map.items():
        if source_key.endswith(".weight"):
            source_weights_by_shard.setdefault(shard_path, []).append(source_key)

    mm_config = Float8MMConfig(use_fast_accum=True)
    activation_quant_kwargs = QuantizeTensorToFloat8Kwargs(
        float8_dtype=torch.float8_e4m3fn,
        granularity=PerTensor(),
        mm_config=mm_config,
    )
    # Target module FQNs successfully replaced with static-FP8 linear modules.
    converted_fqns: list[str] = []
    # Source FP8 weights mapped to linear modules present in this model variant.
    applicable_source_weight_keys: set[str] = set()
    # Source FP8 weights actually installed; compared with the applicable set
    # after conversion to detect incomplete plans.
    converted_source_weight_keys: set[str] = set()
    # Weight shard -> validated (source weight key, target module FQN) conversions.
    conversion_plan_by_shard: dict[str, list[tuple[str, str]]] = {}
    # Scale shard -> scale keys to preload; scales may be separate from their weights.
    source_scales_by_shard: dict[str, list[str]] = {}
    # Target module FQNs reserved by the plan, used to reject duplicate mappings.
    planned_target_fqns: set[str] = set()

    # Identify and validate every applicable FP8 conversion before replacing any
    # modules. The checkpoint may contain intentionally ignored weights or
    # optional heads absent from this model variant; those must be filtered
    # before requiring their scales.
    for shard_path, source_weight_keys in sorted(source_weights_by_shard.items()):
        full_shard_path = checkpoint_path / shard_path
        with safe_open(full_shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
            missing_weight_keys = set(source_weight_keys) - shard_keys
            if missing_weight_keys:
                raise KeyError(
                    f"ModelOpt weight tensor(s) indexed in {full_shard_path} are missing: {sorted(missing_weight_keys)}"
                )
            for source_weight_key in sorted(source_weight_keys):
                tensor_slice = shard.get_slice(source_weight_key)
                if tensor_slice.get_dtype() != "F8_E4M3":
                    continue

                target_weight_key = key_mapper(source_weight_key, shard_path)
                if target_weight_key is None:
                    continue
                if not target_weight_key.endswith(".weight"):
                    raise KeyError(f"ModelOpt FP8 key {source_weight_key!r} has no Cosmos3 weight mapping")
                target_module_fqn = target_weight_key.removesuffix(".weight")
                try:
                    module = model.get_submodule(target_module_fqn)
                except AttributeError:
                    continue
                if not isinstance(module, nn.Linear):
                    raise KeyError(
                        f"ModelOpt FP8 key {source_weight_key!r} mapped to {target_module_fqn!r}, "
                        "which is not an nn.Linear module"
                    )
                if type(module.weight) is not nn.Parameter and not isinstance(module, _ModelOptFloat8Linear):
                    # Distinguish the two ways a weight stops being a plain
                    # parameter: FSDP replaced it with a shard, or something
                    # already quantized it. Reporting the former as the latter
                    # sends debugging in entirely the wrong direction. A module
                    # already swapped by `swap_modelopt_fp8_linears_on_meta` is
                    # neither — its weight may legitimately be a DTensor shard.
                    if type(module.weight).__name__ == "DTensor":
                        raise ValueError(
                            f"ModelOpt FP8 target is an FSDP shard, not a plain parameter: {target_module_fqn}. "
                            "The FP8 swap must run before the network is parallelized."
                        )
                    raise ValueError(f"ModelOpt FP8 target is already quantized: {target_module_fqn}")
                checkpoint_shape = tuple(tensor_slice.get_shape())
                if checkpoint_shape != tuple(module.weight.shape):
                    raise ValueError(
                        f"Shape mismatch for {target_module_fqn}: checkpoint has {checkpoint_shape}, "
                        f"model expects {tuple(module.weight.shape)}"
                    )
                if target_module_fqn in planned_target_fqns:
                    raise ValueError(f"Multiple ModelOpt FP8 weights map to target: {target_module_fqn}")

                source_module_key = source_weight_key.removesuffix(".weight")
                source_scale_keys = (
                    f"{source_module_key}.weight_scale",
                    f"{source_module_key}.input_scale",
                )
                missing_scale_keys = set(source_scale_keys) - weight_map.keys()
                if missing_scale_keys:
                    raise KeyError(
                        f"ModelOpt FP8 tensor {source_weight_key!r} is missing scale tensor(s) "
                        f"from the checkpoint index: {sorted(missing_scale_keys)}"
                    )
                for source_scale_key in source_scale_keys:
                    source_scales_by_shard.setdefault(weight_map[source_scale_key], []).append(source_scale_key)

                applicable_source_weight_keys.add(source_weight_key)
                planned_target_fqns.add(target_module_fqn)
                conversion_plan_by_shard.setdefault(shard_path, []).append((source_weight_key, target_module_fqn))

    if not applicable_source_weight_keys:
        raise ValueError(f"No ModelOpt FP8 linear weights were found in {checkpoint_path}")

    # ModelOpt's consolidated HF export can place the scalar scales in a
    # different shard from the corresponding FP8 weight. Read only those tiny
    # tensors up front, following the root index rather than assuming locality.
    exported_scales: dict[str, torch.Tensor] = {}
    for shard_path, source_scale_keys in sorted(source_scales_by_shard.items()):
        full_shard_path = checkpoint_path / shard_path
        with safe_open(full_shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
            missing_scale_keys = set(source_scale_keys) - shard_keys
            if missing_scale_keys:
                raise KeyError(
                    f"ModelOpt scale tensor(s) indexed in {full_shard_path} are missing: {sorted(missing_scale_keys)}"
                )
            for source_scale_key in source_scale_keys:
                scale = shard.get_tensor(source_scale_key)
                if scale.numel() != 1:
                    raise ValueError(f"ModelOpt scale tensor {source_scale_key!r} must contain exactly one value")
                exported_scales[source_scale_key] = scale

    for shard_path, conversions in sorted(conversion_plan_by_shard.items()):
        full_shard_path = checkpoint_path / shard_path
        with safe_open(full_shard_path, framework="pt", device="cpu") as shard:
            for source_weight_key, target_module_fqn in conversions:
                source_module_key = source_weight_key.removesuffix(".weight")
                source_weight_scale_key = f"{source_module_key}.weight_scale"
                source_input_scale_key = f"{source_module_key}.input_scale"
                module = model.get_submodule(target_module_fqn)
                device = module.weight.device

                if isinstance(module, _ModelOptFloat8Linear):
                    # Already swapped on meta before parallelization; the weight
                    # is an E4M3 placeholder, so the compute dtype comes from the
                    # dtype recorded at swap time rather than from the parameter.
                    replacement = module
                    high_precision_dtype = module._modelopt_high_precision_dtype
                    if high_precision_dtype is None:
                        raise ValueError(f"ModelOpt FP8 module has no recorded compute dtype: {target_module_fqn}")
                else:
                    high_precision_dtype = module.weight.dtype
                    replacement = _ModelOptFloat8Linear(
                        module.in_features,
                        module.out_features,
                        bias=False,
                        device="meta",
                        dtype=high_precision_dtype,
                    )
                    replacement.bias = module.bias
                    replacement.train(module.training)

                    parent_fqn, _, child_name = target_module_fqn.rpartition(".")
                    parent = model.get_submodule(parent_fqn) if parent_fqn else model
                    setattr(parent, child_name, replacement)
                    del module

                quantized_data = shard.get_tensor(source_weight_key)
                weight_scale = exported_scales[source_weight_scale_key].reshape(1, 1)
                input_scale = exported_scales[source_input_scale_key].reshape(1, 1)

                existing_param = replacement.weight
                if isinstance(existing_param, DTensor):
                    # FSDP already sharded the placeholder, so only this rank's
                    # slice of the checkpoint tensor is written. The scales are
                    # per-tensor and therefore identical on every rank.
                    local_weight = existing_param.to_local()
                    quantized_data = distribute_tensor(
                        quantized_data.to(device=device),
                        existing_param.device_mesh,
                        existing_param.placements,
                    ).to_local()
                elif isinstance(existing_param.data, PrototypeFloat8Tensor):
                    local_weight = existing_param.data
                else:
                    local_weight = None

                if local_weight is not None:
                    # Fill the parameter that FSDP is already tracking; rebinding
                    # `replacement.weight` here would detach it from the parameter
                    # group and strand the all-gather on a stale tensor.
                    if tuple(local_weight.qdata.shape) != tuple(quantized_data.shape):
                        raise ValueError(
                            f"Shard mismatch for {target_module_fqn}: model shard "
                            f"{tuple(local_weight.qdata.shape)}, checkpoint slice {tuple(quantized_data.shape)}"
                        )
                    local_weight.qdata.copy_(quantized_data)
                    local_weight.scale.copy_(weight_scale)
                    local_weight.act_quant_scale.copy_(input_scale)
                else:
                    # Unswapped legacy path: the module still holds a plain bf16
                    # weight, so build the quantized tensor and bind it directly.
                    replacement.weight = nn.Parameter(
                        PrototypeFloat8Tensor(
                            quantized_data.to(device=device),
                            weight_scale.to(device=device),
                            act_quant_scale=input_scale.to(device=device),
                            block_size=list(quantized_data.shape),
                            mm_config=mm_config,
                            act_quant_kwargs=activation_quant_kwargs,
                            dtype=high_precision_dtype,
                        ),
                        requires_grad=False,
                    )
                replacement._modelopt_weight_loaded = True
                converted_fqns.append(target_module_fqn)
                converted_source_weight_keys.add(source_weight_key)

    missing_conversions = applicable_source_weight_keys - converted_source_weight_keys
    if missing_conversions:
        raise ValueError(f"ModelOpt FP8 weights were not converted: {sorted(missing_conversions)}")
    converted_fqns.sort()
    sharded_count = sum(1 for fqn in converted_fqns if isinstance(model.get_submodule(fqn).weight.data, DTensor))
    log.info(
        f"Loaded {len(converted_fqns)} calibrated ModelOpt FP8 weights into TorchAO "
        f"({sharded_count} sharded / {len(converted_fqns) - sharded_count} replicated)"
    )
    log.debug(f"ModelOpt FP8 matched_fqns={converted_fqns}")
    return converted_fqns


# Symmetric INT8 keeps the range at +-127 (the -128 code is unused), matching the
# TensorRT / ModelOpt convention, so quantization is sign-symmetric.
_INT8_QMAX = 127.0
# Largest finite E4M3 magnitude; values are clamped here before the cast because
# torch's float32 -> float8_e4m3fn conversion does not saturate (it yields NaN).
_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_QDQ_SIM_METHODS = ("int8_sim", "fp8_sim")


def _qdq_scale(values: torch.Tensor, qmax: float, *, per_row: bool) -> torch.Tensor:
    """Absmax scale so that ``qmax`` maps to the largest magnitude.

    ``per_row`` reduces over the last dimension only, giving one scale per row:
    per output channel for a ``(N, K)`` weight, per token for a ``(..., K)``
    activation. Otherwise a single scale covers the whole tensor. The floor keeps
    an all-zero row from dividing by zero (its values then quantize to 0).
    """
    amax = values.abs().amax(dim=-1, keepdim=True) if per_row else values.abs().amax()
    return (amax / qmax).clamp_(min=torch.finfo(torch.float32).tiny)


def _grouped_view(values32: torch.Tensor, group_size: int) -> torch.Tensor:
    """Split the last (K) dimension into ``group_size`` blocks: ``(..., K) -> (..., K/g, g)``.

    A per-row absmax over this view is then one scale per block of ``g``
    consecutive K elements of each row (weight output channel / activation token).
    """
    k = values32.shape[-1]
    if group_size <= 0 or k % group_size:
        raise ValueError(f"Q/DQ group size {group_size} must be > 0 and divide the input dimension K={k}")
    return values32.reshape(*values32.shape[:-1], k // group_size, group_size)


def fake_quant_int8(values: torch.Tensor, *, per_row: bool = True, group_size: int = 0) -> torch.Tensor:
    """Symmetric INT8 quantize-dequantize in float32, returned in the input dtype.

    ``per_row=True`` is the common "per-channel weight / per-token activation"
    recipe (one absmax scale per row of the last dimension). ``group_size=g > 0``
    refines that to one scale per ``g`` consecutive elements of each row (K must
    be divisible by ``g``). Rounding is round-half-to-even (``torch.round``),
    the same as TensorRT / ModelOpt.
    """
    values32 = values.float()
    work = _grouped_view(values32, group_size) if group_size else values32
    scale = _qdq_scale(work, _INT8_QMAX, per_row=per_row or bool(group_size))
    quantized = torch.round(work / scale).clamp_(-_INT8_QMAX, _INT8_QMAX)
    return (quantized * scale).reshape(values.shape).to(values.dtype)


def fake_quant_fp8(values: torch.Tensor, *, per_row: bool = True, group_size: int = 0) -> torch.Tensor:
    """E4M3 quantize-dequantize in float32, returned in the input dtype.

    Scales are absmax -> 448 per row (``per_row=True``), per tensor, or per
    ``group_size`` block of each row when ``group_size > 0``; the scaled values
    are cast through ``torch.float8_e4m3fn`` (round to nearest even, mantissa
    truncated to 3 bits) and back.
    """
    values32 = values.float()
    work = _grouped_view(values32, group_size) if group_size else values32
    scale = _qdq_scale(work, _FP8_E4M3_MAX, per_row=per_row or bool(group_size))
    quantized = (work / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(values.shape).to(values.dtype)


def _fake_quant(values: torch.Tensor, method: str, *, per_row: bool, group_size: int = 0) -> torch.Tensor:
    if method == "int8_sim":
        return fake_quant_int8(values, per_row=per_row, group_size=group_size)
    if method == "fp8_sim":
        return fake_quant_fp8(values, per_row=per_row, group_size=group_size)
    raise ValueError(f"Unsupported Q/DQ simulation method: {method}")


def _qdq_granularity_name(per_row: bool, group_size: int) -> str:
    if group_size:
        return f"group{group_size}_along_K(weight_row_blocks/token_blocks)"
    return "per_channel_weight/per_token_act" if per_row else "per_tensor"


class QdqSimLinear(nn.Linear):
    """``nn.Linear`` running a quantize-dequantize (Q/DQ) simulation of W8A8.

    The weight is fake-quantized once when the module is installed (after the
    checkpoint has been loaded), the input activation is fake-quantized on every
    call with dynamically computed scales, and the GEMM is the ordinary dense
    ``F.linear`` in the compute dtype. The result therefore carries the
    rounding/clipping error of the low-precision format while needing no
    low-precision kernel, which makes it a format-only comparison baseline.

    ``qdq_per_row`` selects per-output-channel weight + per-token activation
    scales (``True``) or a single scale per tensor (``False``); ``qdq_group_size``
    > 0 blocks both operands along K in groups of that size instead.
    """

    qdq_method: str = "int8_sim"
    qdq_per_row: bool = True
    qdq_group_size: int = 0
    # ``gemm_out`` edge: also fake-quantize this linear's output (set by the installer).
    qdq_output: bool = False
    # Set by the installer after the loaded weight has been fake-quantized. A
    # forward before that would silently run the simulation against the wrong
    # (unquantized or uninitialized) weight, so it is rejected instead.
    _qdq_weight_finalized: bool = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self._qdq_weight_finalized:
            raise RuntimeError(
                "QdqSimLinear weight has not been fake-quantized; install it via apply_quantization_inplace "
                "after the checkpoint is loaded"
            )
        if inputs.numel() == 0:
            return F.linear(inputs, self.weight, self.bias)
        quantized_inputs = _fake_quant(
            inputs, self.qdq_method, per_row=self.qdq_per_row, group_size=self.qdq_group_size
        )
        outputs = F.linear(quantized_inputs, self.weight, self.bias)
        if self.qdq_output:
            outputs = _fake_quant(outputs, self.qdq_method, per_row=self.qdq_per_row, group_size=self.qdq_group_size)
        return outputs

    def extra_repr(self) -> str:
        granularity = _qdq_granularity_name(self.qdq_per_row, self.qdq_group_size)
        return f"{super().extra_repr()}, qdq={self.qdq_method}({granularity})"


def swap_qdq_sim_linears(
    model: nn.Module, target_fqns: list[str], *, method: str, per_row: bool, group_size: int = 0
) -> list[str]:
    """Replace the given loaded linears with :class:`QdqSimLinear` and fake-quantize their weights in place.

    Must run *after* the checkpoint weights are loaded: the weight tensor is
    rewritten with its Q/DQ image, so quantizing an uninitialized or later
    overwritten weight would leave a dense bf16 linear pretending to be INT8.
    The replacement reuses the original ``weight`` / ``bias`` parameter objects
    (no copies, state-dict keys unchanged); only plain-tensor (unsharded)
    parameters are supported, matching the other runtime PTQ paths.

    Returns:
        Sorted FQNs that were swapped.
    """
    if method not in _QDQ_SIM_METHODS:
        raise ValueError(f"Unsupported Q/DQ simulation method: {method}")
    swapped_fqns: list[str] = []
    for target_module_fqn in sorted(target_fqns):
        module = model.get_submodule(target_module_fqn)
        if not isinstance(module, nn.Linear):
            raise KeyError(f"Q/DQ simulation target {target_module_fqn!r} is not an nn.Linear module")
        if isinstance(module, QdqSimLinear):
            raise ValueError(f"Q/DQ simulation target is already quantized: {target_module_fqn}")
        if isinstance(module.weight, DTensor):
            raise ValueError(
                f"Q/DQ simulation target is an FSDP shard, not a plain parameter: {target_module_fqn}. "
                "Runtime quantization requires dp_shard_size == 1."
            )

        replacement = QdqSimLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
        )
        replacement.qdq_method = method
        replacement.qdq_per_row = per_row
        replacement.qdq_group_size = group_size
        replacement.weight = module.weight
        replacement.bias = module.bias
        replacement.train(module.training)
        with torch.no_grad():
            # Weight rows are output channels, so per_row here is per-output-channel
            # and group blocks run along K within each output channel.
            replacement.weight.copy_(_fake_quant(replacement.weight, method, per_row=per_row, group_size=group_size))
        replacement._qdq_weight_finalized = True

        parent_fqn, _, child_name = target_module_fqn.rpartition(".")
        parent = model.get_submodule(parent_fqn) if parent_fqn else model
        setattr(parent, child_name, replacement)
        swapped_fqns.append(target_module_fqn)
    return swapped_fqns


def _canonical_fqn(name: str) -> str:
    """Drop the ``_orig_mod`` components torch.compile inserts into module FQNs."""
    return ".".join(part for part in name.split(".") if part != "_orig_mod")


def _target_fqn_aliases(canonical_name: str) -> tuple[str, ...]:
    """Forms under which a module FQN may appear in ``QuantizationConfig.target_fqns``.

    Runtime quantization is applied to the ``OmniMoTModel`` (FQNs ``net.…``),
    while ModelOpt target lists are relative to the VFM network (``language_model.…``);
    accept both spellings so a list derived from one can drive the other.
    """
    if canonical_name.startswith("net."):
        return (canonical_name, canonical_name.removeprefix("net."))
    return (canonical_name,)


def _get_filter_fn(quantization_config: QuantizationConfig) -> Callable[[nn.Module, str], bool]:
    """Build a module-selection predicate from the quantization config.

    The returned closure captures ``include_regex`` / ``exclude_regex`` and
    implements the selection policy documented on :class:`QuantizationConfig`.
    Each key is treated as a regular expression and matched against a module's
    FQN with :func:`re.search` (a plain substring remains a valid pattern, so
    existing substring-style keys keep working). It is passed to torchao as
    ``filter_fn`` (for ``quantize_``), which expects a
    ``(module, fqn) -> bool`` signature.

    Args:
        quantization_config: Config carrying the include/exclude key lists.

    Returns:
        A predicate suitable for torchao's ``filter_fn`` / ``module_filter_fn``.
    """

    include_patterns: list[re.Pattern[str]] = []
    for pattern in quantization_config.include_regex:
        try:
            include_patterns.append(re.compile(pattern))
        except re.error as error:
            raise ValueError(f"Invalid include_regex pattern {pattern!r}: {error}") from error

    exclude_patterns: list[re.Pattern[str]] = []
    for pattern in quantization_config.exclude_regex:
        try:
            exclude_patterns.append(re.compile(pattern))
        except re.error as error:
            raise ValueError(f"Invalid exclude_regex pattern {pattern!r}: {error}") from error

    target_fqns = set(quantization_config.target_fqns)

    def _filter_fn(mod: nn.Module, name: str) -> bool:
        """Decide whether a single module should be quantized.

        When ``target_fqns`` is set the regex policy is bypassed entirely: an
        ``nn.Linear`` is selected iff its canonical FQN (or its VFM-network-
        relative alias) is in the list, which pins the module set exactly.

        Used by preflight and torchao as each walks the model recursively. A
        module is selected only when ALL of the following hold:

        1. It is an ``nn.Linear`` (the only layer type these recipes support).
        2. ``include_regex`` is empty (include everything) OR the module's FQN
           matches at least one include pattern.
        3. The module's FQN matches none of the ``exclude_regex`` patterns.

        Each include/exclude key is treated as a regular expression and matched
        against the FQN with :func:`re.search`, so the pattern can match anywhere
        in the name (a plain substring is still a valid regex, preserving the
        previous substring-match behavior, while enabling anchors like ``^``/``$``,
        alternation, character classes, etc.).

        Note the parenthesization around the include check: ``and`` binds tighter
        than ``or`` in Python, so without it the ``nn.Linear`` and exclude
        checks would not apply across both include branches.

        Args:
            mod (torch.nn.Module): The module that is being processed.
            name (str): A fully qualified name of the module that is being processed.

        Return:
            True if the module should be quantized, False otherwise.
        """
        # torch.compile inserts `_orig_mod` into FQNs; hide it from user-facing regex matching.
        canonical_name = _canonical_fqn(name)
        if target_fqns:
            return isinstance(mod, nn.Linear) and any(
                alias in target_fqns for alias in _target_fqn_aliases(canonical_name)
            )
        return (
            isinstance(mod, nn.Linear)
            and (not include_patterns or any(pattern.search(canonical_name) for pattern in include_patterns))
            and not any(pattern.search(canonical_name) for pattern in exclude_patterns)
        )

    return _filter_fn


def _get_validated_quantization_fqns(
    model: nn.Module,
    filter_fn: Callable[[nn.Module, str], bool],
    *,
    target_fqns: list[str] | None = None,
) -> list[str]:
    """Validate the selected modules and return their sorted FQNs.

    With ``target_fqns`` every listed FQN must have been matched; an entry that
    names no module, or a module that is not an ``nn.Linear``, is an error rather
    than a silent shrink of the module set.
    """
    matched_modules = sorted(
        ((name, module) for name, module in model.named_modules() if filter_fn(module, name)),
        key=lambda item: item[0],
    )
    if not matched_modules:
        raise ValueError("No nn.Linear modules matched the quantization selection")
    matched_fqns = [name for name, _ in matched_modules]
    already_quantized_fqns = [
        name
        for name, module in matched_modules
        if type(module.weight) is not nn.Parameter or isinstance(module, QdqSimLinear)
    ]
    if already_quantized_fqns:
        raise ValueError(f"Quantization targets are already quantized: {', '.join(already_quantized_fqns)}")
    if target_fqns:
        matched_aliases = {alias for name in matched_fqns for alias in _target_fqn_aliases(_canonical_fqn(name))}
        missing = sorted(set(target_fqns) - matched_aliases)
        if missing:
            all_aliases = {
                alias for name, _ in model.named_modules() for alias in _target_fqn_aliases(_canonical_fqn(name))
            }
            not_linear = [fqn for fqn in missing if fqn in all_aliases]
            not_found = [fqn for fqn in missing if fqn not in all_aliases]
            details = []
            if not_linear:
                details.append(f"not nn.Linear modules: {not_linear}")
            if not_found:
                details.append(f"not found in the model: {not_found}")
            raise ValueError(f"{len(missing)} quantization target_fqns did not match ({'; '.join(details)})")
    return matched_fqns


def _dump_matched_fqns(matched_fqns: list[str], dump_path: str | None) -> str:
    """Log a digest of the quantized module set and optionally write it to ``dump_path``.

    The digest (and the file) is what two runs are compared on to prove they
    quantized the identical set of linears; FQNs are canonicalized so a
    torch.compile-wrapped model hashes the same as a plain one.
    """
    canonical = [_canonical_fqn(name) for name in matched_fqns]
    digest = hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()[:16]
    log.info(f"Quantized module set: count={len(canonical)} sha256={digest}")
    if dump_path is not None:
        distributed = torch.distributed
        if not (distributed.is_available() and distributed.is_initialized()) or distributed.get_rank() == 0:
            path = Path(dump_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(f"{name}\n" for name in canonical), encoding="utf-8")
            log.info(f"Wrote quantized module FQNs to {path}")
    return digest


def apply_quantization_inplace(model: nn.Module, quantization_config: QuantizationConfig) -> list[str]:
    """Apply quantization in place via ``quantize_`` (replaces weights with quantized tensors).

    This is the replication path. ``quantize_`` replaces each weight with a
    quantized tensor subclass as the live parameter, which only works when the
    parameters are plain tensors. It therefore cannot be applied to an already
    FSDP-sharded model (the params are ``DTensor`` shards), so it is restricted
    to replicated inference (``data_parallel_shard_degree == 1``).

    These configs (``MXDynamicActivationMXWeightConfig`` /
    ``NVFP4DynamicActivationNVFP4WeightConfig`` /
    ``Float8DynamicActivationFloat8WeightConfig``) are inference-only (PTQ) and
    have no backward support. For the sharded case use ``apply_quantization``
    (the module-swap path) instead; both functions are currently inference
    paths, selected by whether FSDP is sharding the model.

    Returns:
        Sorted fully qualified names of the matched modules.
    """
    # No-op when quantization is disabled.
    if quantization_config.method is None:
        return []

    filter_fn = _get_filter_fn(quantization_config)
    matched_fqns = _get_validated_quantization_fqns(model, filter_fn, target_fqns=quantization_config.target_fqns)

    if quantization_config.method == "int8_speed":
        # Real INT8 GEMMs (torch._int_mm, per-channel W / per-token A): a speed proxy, not the
        # accuracy plan's group-64 scheme. Weights are stored as INT8.
        from cosmos_framework.utils.generator.int8_speed import swap_int8_speed_linears

        swap_int8_speed_linears(model, matched_fqns)
        log.info(f"Applied int8_speed (torch._int_mm W8A8, per-channel/per-token), matched_count={len(matched_fqns)}")
        _dump_matched_fqns(matched_fqns, quantization_config.matched_fqns_dump_path)
        return matched_fqns

    if quantization_config.method in _QDQ_SIM_METHODS:
        # Torchao-free Q/DQ simulation: same selection as above, dense bf16 GEMMs.
        # int8_sim is always per-channel weight / per-token activation; fp8_sim
        # follows fp8_granularity like the torchao ``fp8`` recipe does.
        per_row = quantization_config.method == "int8_sim" or quantization_config.fp8_granularity == "per_row"
        group_size = quantization_config.qdq_group_size
        if group_size and not per_row:
            raise ValueError("qdq_group_size requires fp8_granularity='per_row' (blocks refine rows, not tensors)")
        swapped = swap_qdq_sim_linears(
            model, matched_fqns, method=quantization_config.method, per_row=per_row, group_size=group_size
        )
        # Non-GEMM edges of the generation tower (residual stream, attention I/O, ...).
        from cosmos_framework.utils.generator.qdq_sim_edges import configure_sim_edges

        configure_sim_edges(
            quantization_config.sim_edges,
            method=quantization_config.method,
            group_size=group_size,
            residual_group_size=quantization_config.sim_residual_group_size,
            residual_bits=quantization_config.sim_residual_bits,
            attn_v_block_size=quantization_config.sim_attn_v_block_size,
            attn_smoothing=quantization_config.sim_attn_smoothing,
            attn_v_format=quantization_config.sim_attn_v_format,
            attn_p_format=quantization_config.sim_attn_p_format,
            attn_pv_accum=quantization_config.sim_attn_pv_accum,
            attn_k_scope=quantization_config.sim_attn_k_scope,
        )
        if "gemm_out" in quantization_config.sim_edges:
            for fqn in swapped:
                model.get_submodule(fqn).qdq_output = True
        granularity = _qdq_granularity_name(per_row, group_size)
        if quantization_config.sim_edges:
            granularity += f" + edges={sorted(quantization_config.sim_edges)}"
            if {"attn_qkv", "attn_pv"} & set(quantization_config.sim_edges):
                granularity += (
                    f" attn(Q/K per token-head, V per channel"
                    f"{'/block' + str(quantization_config.sim_attn_v_block_size) if quantization_config.sim_attn_v_block_size else ''}"
                    f", smoothing={quantization_config.sim_attn_smoothing}"
                    f", V={quantization_config.sim_attn_v_format}, P={quantization_config.sim_attn_p_format}"
                    f", PV accum={quantization_config.sim_attn_pv_accum}"
                    f", K scope={quantization_config.sim_attn_k_scope})"
                )
            if "residual" in quantization_config.sim_edges:
                granularity += (
                    f" residual=int{quantization_config.sim_residual_bits}"
                    f"/g{quantization_config.sim_residual_group_size or group_size}"
                )
        log.info(
            f"Applied runtime Q/DQ simulation method={quantization_config.method} granularity={granularity}, "
            f"matched_count={len(matched_fqns)}"
        )
        log.debug(f"Runtime PTQ matched_fqns={matched_fqns}")
        _dump_matched_fqns(matched_fqns, quantization_config.matched_fqns_dump_path)
        return matched_fqns

    from torchao.prototype.mx_formats import (
        MXDynamicActivationMXWeightConfig,
        NVFP4DynamicActivationNVFP4WeightConfig,
    )
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerRow,
        PerTensor,
        quantize_,
    )

    def _reclaim_and_log_gpu_memory(tag: str) -> None:
        # synchronize() + gc.collect() + empty_cache() are load-bearing, not passive
        # observability: torch.mem_get_info reports memory free at the CUDA-driver level.
        # synchronize() first drains any in-flight device work (e.g. async copies from the
        # checkpoint load path) so their allocations retire before we measure; otherwise the
        # baseline could over-report free memory. PyTorch's caching allocator then holds
        # freed blocks in its own pool without returning them to the driver, so empty_cache()
        # flushes that pool back to the driver.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(torch.device("cuda", torch.cuda.current_device()))
        log.info(f"GPU memory ({tag}): {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")

    _reclaim_and_log_gpu_memory("before quantization")

    if quantization_config.method == "mxfp8":
        quantize_(
            model,
            config=MXDynamicActivationMXWeightConfig(),
            filter_fn=filter_fn,
        )

    elif quantization_config.method == "nvfp4":
        # use_triton_kernel=False avoids torchao's fused NVFP4 Triton kernel, which
        # requires the external `mslk` package. Prebuilt mslk wheels are linked
        # against upstream torch and fail to load against NVIDIA's NGC custom torch
        # builds (ABI mismatch on `torch::Library::_def`), so we use torchao's
        # built-in NVFP4 path instead.
        quantize_(
            model,
            config=NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False),
            filter_fn=filter_fn,
        )

    elif quantization_config.method == "fp8":
        # Hopper-compatible FP8. Unlike mxfp8 / nvfp4 (block-scaled MX / NVFP4
        # formats whose accelerated kernels require Blackwell sm_100 tensor
        # cores), this is plain e4m3 dynamic-activation + fp8-weight quantization
        # executed via torch._scaled_mm, which is supported on Hopper (sm_90) and
        # Ada (sm_89). Scaling granularity is user-selectable: PerRow (rowwise,
        # better accuracy) or PerTensor (single scale, slightly faster); both are
        # supported on Hopper.
        granularity = PerRow() if quantization_config.fp8_granularity == "per_row" else PerTensor()
        quantize_(
            model,
            config=Float8DynamicActivationFloat8WeightConfig(granularity=granularity),
            filter_fn=filter_fn,
        )

    else:
        raise ValueError(f"Unsupported quantization method: {quantization_config.method}")

    _reclaim_and_log_gpu_memory("after quantization")
    log.info(f"Applied runtime PTQ method={quantization_config.method}, matched_count={len(matched_fqns)}")
    log.debug(f"Runtime PTQ matched_fqns={matched_fqns}")
    _dump_matched_fqns(matched_fqns, quantization_config.matched_fqns_dump_path)
    return matched_fqns
