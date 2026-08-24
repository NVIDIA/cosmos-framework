# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mixed-precision diffusion steps for ModelOpt static-FP8 checkpoints.

The first/last N denoising steps run generation-path FP8 linears with 16-bit
activations (W8A16: the checkpoint's E4M3 weight dequantized to the compute
dtype + dense GEMM); the middle steps keep the TorchAO FP8-activation path
(W8A8). Weights are always the checkpoint's FP8 tensors — only the activation
treatment changes per step. Port of vllm-omni's Cosmos3 mixed-precision
feature (branch ``mixed-precision-diffusion-steps``) onto the framework's
TorchAO FP8 path.
"""

PRECISION_PATHS = ("reasoner", "generation")
W8A16_CACHE_MODES = ("none", "generation", "all", "cpu_block", "gpu_block")


def use_w8a16_step(first_steps: int, last_steps: int, step_index: int, num_steps: int) -> bool:
    """Return whether one denoising step runs the W8A16 path.

    ``num_steps == 1`` always selects W8A8: FSDP collective alignment pads
    slow ranks with dummy single-step samples, and those must stay on the
    cheap path (a genuine 1-step request also gets W8A8 — same open question
    as the vllm-omni reference).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if step_index < 0 or step_index >= num_steps:
        raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
    if num_steps == 1:
        return False
    return step_index < first_steps or step_index >= num_steps - last_steps
