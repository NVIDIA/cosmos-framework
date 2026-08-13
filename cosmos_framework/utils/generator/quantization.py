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
"""

import gc
import json
import re
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig

# NOTE: ``torchao`` is imported lazily inside the functions below rather than at
# module top level. These two helpers are the only torchao consumers, but this
# module is imported transitively by the model package (e.g. during tests that
# never quantize). Keeping the imports lazy means importing this module does not
# require torchao to be installed; the imports only run — and only fail — when
# quantization is actually requested.


class _ModelOptFloat8Linear(nn.Linear):
    """Work around TorchAO 0.16 static-FP8 limitations for linear inputs."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_shape = (*inputs.shape[:-1], self.out_features)
        # PrototypeFloat8Tensor divides each input dimension by its block size, so
        # a zero-row input fails with 0 // 0 instead of producing an empty output.
        if inputs.numel() == 0:
            return inputs.new_empty(output_shape)
        # PrototypeFloat8Tensor requires the input and its static (1, 1) activation
        # scale to have equal rank, so flatten token dimensions before dispatch.
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_outputs = F.linear(flat_inputs, self.weight, self.bias)
        return flat_outputs.reshape(output_shape)


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
                if type(module.weight) is not nn.Parameter:
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

                quantized_data = shard.get_tensor(source_weight_key).to(device=device)
                weight_scale = exported_scales[source_weight_scale_key].to(device=device).reshape(1, 1)
                input_scale = exported_scales[source_input_scale_key].to(device=device).reshape(1, 1)
                quantized_weight = PrototypeFloat8Tensor(
                    quantized_data,
                    weight_scale,
                    act_quant_scale=input_scale,
                    block_size=list(quantized_data.shape),
                    mm_config=mm_config,
                    act_quant_kwargs=activation_quant_kwargs,
                    dtype=high_precision_dtype,
                )
                replacement.weight = nn.Parameter(quantized_weight, requires_grad=False)
                converted_fqns.append(target_module_fqn)
                converted_source_weight_keys.add(source_weight_key)

    missing_conversions = applicable_source_weight_keys - converted_source_weight_keys
    if missing_conversions:
        raise ValueError(f"ModelOpt FP8 weights were not converted: {sorted(missing_conversions)}")
    converted_fqns.sort()
    log.info(f"Loaded {len(converted_fqns)} calibrated ModelOpt FP8 weights into TorchAO")
    log.debug(f"ModelOpt FP8 matched_fqns={converted_fqns}")
    return converted_fqns


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

    def _filter_fn(mod: nn.Module, name: str) -> bool:
        """Decide whether a single module should be quantized.

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
        canonical_name = ".".join(part for part in name.split(".") if part != "_orig_mod")
        return (
            isinstance(mod, nn.Linear)
            and (not include_patterns or any(pattern.search(canonical_name) for pattern in include_patterns))
            and not any(pattern.search(canonical_name) for pattern in exclude_patterns)
        )

    return _filter_fn


def _get_validated_quantization_fqns(model: nn.Module, filter_fn: Callable[[nn.Module, str], bool]) -> list[str]:
    """Validate the selected modules and return their sorted FQNs."""
    matched_modules = sorted(
        ((name, module) for name, module in model.named_modules() if filter_fn(module, name)),
        key=lambda item: item[0],
    )
    if not matched_modules:
        raise ValueError("No nn.Linear modules matched the quantization selection")
    matched_fqns = [name for name, _ in matched_modules]
    already_quantized_fqns = [name for name, module in matched_modules if type(module.weight) is not nn.Parameter]
    if already_quantized_fqns:
        raise ValueError(f"Quantization targets are already quantized: {', '.join(already_quantized_fqns)}")
    return matched_fqns


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
    matched_fqns = _get_validated_quantization_fqns(model, filter_fn)

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
    return matched_fqns
