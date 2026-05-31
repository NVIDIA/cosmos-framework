# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from contextlib import ExitStack, contextmanager
from typing import Generator

import torch

from cosmos_framework.utils.misc import timer



@contextmanager
def disable_tf32() -> Generator[None, None, None]:
    """Context manager to temporarily disable TF32 for CUDA matrix multiplications.

    This is useful for ensuring full FP32 precision in numerical computations,
    particularly when debugging or comparing results between different implementations.

    Example:
        with disable_tf32():
            result = torch.matmul(a, b)  # Uses full FP32 precision
    """
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul


@contextmanager
def data_loader_init() -> Generator[None, None, None]:
    """
    Wrap the data loader initialization with multiple context managers used for telemetry and one logger.
    """
    contexts = [
        timer("init_data_loader"),
    ]
    with ExitStack() as stack:
        yield [stack.enter_context(cm) for cm in contexts]


@contextmanager
def model_init(set_barrier: bool = False) -> Generator[None, None, None]:
    """
    Wrap the instantiation of the model with multiple context managers used for telemetry and one logger.
    """
    contexts = [
        timer("init_model"),
    ]
    with ExitStack() as stack:
        yield [stack.enter_context(cm) for cm in contexts]


@contextmanager
def distributed_init() -> Generator[None, None, None]:
    """
    Wrap the distributed initialization, used for telemetry and timers
    """
    contexts = [
        timer("init_distributed"),
    ]
    with ExitStack() as stack:
        yield [stack.enter_context(cm) for cm in contexts]
