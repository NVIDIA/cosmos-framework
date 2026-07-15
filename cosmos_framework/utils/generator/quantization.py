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

import re

from torch import nn

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig

# NOTE: ``torchao`` is imported lazily inside the functions below rather than at
# module top level. These two helpers are the only torchao consumers, but this
# module is imported transitively by the model package (e.g. during tests that
# never quantize). Keeping the imports lazy means importing this module does not
# require torchao to be installed; the imports only run — and only fail — when
# quantization is actually requested.


def _get_filter_fn(quantization_config: QuantizationConfig):
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

    def _filter_fn(mod: nn.Module, name: str) -> bool:
        """Decide whether a single module should be quantized.

        Called once per module as torchao walks the model recursively. A module
        is selected only when ALL of the following hold:

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
        include_keys = quantization_config.include_regex
        exclude_keys = quantization_config.exclude_regex
        return (
            isinstance(mod, nn.Linear)
            and (len(include_keys) == 0 or any(re.search(key, name) for key in include_keys))
            and not any(re.search(key, name) for key in exclude_keys)
        )

    return _filter_fn


def apply_quantization_inplace(model: nn.Module, quantization_config: QuantizationConfig):
    """Apply quantization in place via ``quantize_`` (replaces weights with quantized tensors).

    This is the replication path. ``quantize_`` replaces each weight with a
    quantized tensor subclass as the live parameter, which only works when the
    parameters are plain tensors. It therefore cannot be applied to an already
    FSDP-sharded model (the params are ``DTensor`` shards), so it is restricted
    to replicated inference (``data_parallel_shard_degree == 1``).

    These configs (``MXDynamicActivationMXWeightConfig`` /
    ``NVFP4DynamicActivationNVFP4WeightConfig``) are inference-only (PTQ) and
    have no backward support. For the sharded case use ``apply_quantization``
    (the module-swap path) instead; both functions are currently inference
    paths, selected by whether FSDP is sharding the model.
    """
    # No-op when quantization is disabled.
    if quantization_config.method is None:
        return

    from torchao.prototype.mx_formats import (
        MXDynamicActivationMXWeightConfig,
        NVFP4DynamicActivationNVFP4WeightConfig,
    )
    from torchao.quantization import (
        quantize_,
    )

    if quantization_config.method == "mxfp8":
        # mxfp8 / nvfp4 use fixed block scales.
        quantize_(
            model,
            config=MXDynamicActivationMXWeightConfig(),
            filter_fn=_get_filter_fn(quantization_config),
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
            filter_fn=_get_filter_fn(quantization_config),
        )
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_config.method}")
