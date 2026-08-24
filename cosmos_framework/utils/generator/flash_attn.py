# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Flash attention initialization for the vfm/ unified VLM training path.

This module replaces `cosmos_rl.policy.kernel.modeling_utils.init_flash_attn_meta`
in VLMModel.__init__, removing the last cosmos_rl import from projects/cosmos3/cosmos3/.

Why a stub for the FlashAttnMeta singleton part?
  HFModel uses HuggingFace's flash_attention_2 implementation
  (attn_implementation="flash_attention_2" in AutoModel.from_config), which calls
  flash_attn_func directly via the transformers.modeling_flash_attention_utils path.
  This is entirely independent of the cosmos_rl FlashAttnMeta singleton.
  Initializing that singleton has no effect on VLM training: the GPU Phase 2 smoke
  test (Qwen3-VL-8B-Instruct, 4-GPU FSDP2, 10 iters, loss 1.15→1.11) confirmed
  correct gradient flow without the singleton.

The deterministic flag IS honored here: it sets the standard PyTorch and CuDNN
determinism knobs, which is equivalent to what the old VLM trainer's
configure_training() did. VLMModel.__init__ calls this before _init_vlm(), so
determinism is configured before any compute happens.
"""

import os


def init_flash_attn_meta(deterministic: bool = False) -> None:
    """Initialize flash attention for the HFModel (VLM) training path.

    Replaces the cosmos_rl FlashAttnMeta singleton initialization (which is only
    relevant to the cosmos_rl model path, not HFModel). Also applies determinism
    settings when deterministic=True, matching what the old VLM trainer's
    configure_training() did.

    Args:
        deterministic: If True, enables PyTorch, CuDNN and FlashAttention
            deterministic modes.
    """
    if deterministic:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required for deterministic CuBLAS on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # FlashAttention-2 is a separate CUDA extension whose backward uses
        # atomics, so torch.use_deterministic_algorithms cannot police it -- it
        # reports nothing even with warn_only=False. Transformers picks FA2's
        # deterministic backward up from this environment variable in
        # modeling_flash_attention_utils, so without it two same-seed runs
        # diverge within a few steps despite every setting above.
        os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
