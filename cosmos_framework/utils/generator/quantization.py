# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-precision quantization helpers for the Cosmos3 VFM MoT network.

Quantization is applied via torchao's :func:`apply_quantization_inplace`, which
uses the ``quantize_`` path to replace each selected weight with a quantized
tensor subclass in place. Because the live parameter becomes a tensor subclass,
this only works on unsharded (plain-tensor) params and is therefore restricted
to replicated inference (``data_parallel_shard_degree == 1``); it cannot be
applied to an FSDP-sharded model whose params are ``DTensor`` shards.

This is an inference-only path: the ``quantize_`` PTQ configs have no backward
support. Module selection is delegated to the filter built by
:func:`_get_filter_fn`.
"""

import gc
import re
from collections.abc import Callable

import torch
from torch import nn

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig

# NOTE: ``torchao`` is imported lazily inside the functions below rather than at
# module top level. These two helpers are the only torchao consumers, but this
# module is imported transitively by the model package (e.g. during tests that
# never quantize). Keeping the imports lazy means importing this module does not
# require torchao to be installed; the imports only run — and only fail — when
# quantization is actually requested.


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
