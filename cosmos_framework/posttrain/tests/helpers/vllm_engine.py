# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Release in-process vLLM engines between GPU tests."""

from __future__ import annotations

import gc


def release_in_process_engines() -> None:
    """Free the model, KV cache and worker processes of engines a finished test left behind.

    An in-process EngineCore keeps its weights and KV cache in the pytest process, and reference
    cycles keep them alive after the test returns. Without this, consecutive tests exhaust the
    device before ``gpu_memory_utilization`` can be honoured. No-op when vLLM or CUDA is not in use.
    """
    try:
        import torch
        from vllm.distributed import cleanup_dist_env_and_memory
    except ImportError:
        return
    if not torch.cuda.is_initialized():
        return
    gc.collect()
    cleanup_dist_env_and_memory()
