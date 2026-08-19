# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Precision helpers shared by tokenizer training and inference."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


def activation_dtype(param_dtype: torch.dtype, device_type: str = "cuda") -> torch.dtype:
    """Return the dtype activations should be staged in for weights of ``param_dtype``.

    Training keeps parameters in FP32 so the parameter is its own master copy, and takes
    reduced-precision compute from autocast instead. The parameter dtype is therefore no
    longer the dtype tensors flow in: matmuls and convolutions cast their operands down
    themselves, so staging activations at the parameter dtype would run the residual
    stream at FP32 width for no numerical benefit. Outside autocast the parameter dtype
    is the only dtype those ops accept, which covers inference against a network that has
    been cast to bf16 wholesale.

    Args:
        param_dtype: Dtype of the weights the activations are about to be fed to.
        device_type: Autocast device type to query.

    Returns:
        The autocast dtype when autocast is active, else ``param_dtype``.
    """
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return param_dtype


def model_compute_dtype(model: Any, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Return the dtype ``model`` computes in, for callers staging its input activations.

    Args:
        model: Model whose compute precision is being queried.
        default: Dtype to assume for objects without the tokenizer precision helpers,
            such as the mock models in the callback tests.
    """
    get_compute_dtype = getattr(model, "_get_compute_dtype", None)
    return get_compute_dtype() if callable(get_compute_dtype) else default


def model_compute_autocast(model: Any) -> AbstractContextManager[Any]:
    """Return the autocast context ``model``'s own step methods run under.

    Callbacks and eval probes that drive the network directly rather than through
    ``training_step``/``validation_step`` have to enter this themselves, since the
    parameters are FP32 masters and all reduced-precision compute comes from autocast.

    Args:
        model: Model the caller is about to run.
    """
    compute_autocast = getattr(model, "_compute_autocast", None)
    return compute_autocast() if callable(compute_autocast) else nullcontext()


def metric_compute_autocast(device: torch.device | str) -> AbstractContextManager[Any]:
    """Disable autocast while computing validation metrics on ``device``.

    Tokenizer model inference intentionally runs under reduced-precision autocast, but
    metric implementations historically consumed those reconstructions in FP32. A
    nested disabled context preserves that boundary for convolutional metrics such as
    SSIM, LPIPS, FID, and rFVD instead of letting the model's autocast leak into them.

    Args:
        device: Device on which the metric computation executes.
    """
    return torch.autocast(device_type=torch.device(device).type, enabled=False)
