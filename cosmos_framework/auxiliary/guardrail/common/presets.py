# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np

from cosmos_framework.auxiliary.guardrail.blocklist.blocklist import Blocklist
from cosmos_framework.auxiliary.guardrail.common.core import GuardrailRunner
from cosmos_framework.auxiliary.guardrail.face_blur_filter.face_blur_filter import RetinaFaceFilter
from cosmos_framework.auxiliary.guardrail.qwen3guard.qwen3guard import Qwen3Guard
from cosmos_framework.utils import log


def create_text_guardrail_runner(offload_model_to_cpu: bool = False) -> GuardrailRunner:
    """Create the text guardrail runner."""
    return GuardrailRunner(
        safety_models=[
            Blocklist(),
            Qwen3Guard(offload_model_to_cpu=offload_model_to_cpu),
        ]
    )


def create_video_guardrail_runner(offload_model_to_cpu: bool = False) -> GuardrailRunner:
    """Create the video guardrail runner."""
    return GuardrailRunner(
        safety_models=[
            # VideoContentSafetyFilter(offload_model_to_cpu=offload_model_to_cpu)
            # Too many false positives, add back when fixed
        ],
        postprocessors=[RetinaFaceFilter(offload_model_to_cpu=offload_model_to_cpu)],
    )


def run_text_guardrail(prompt: str, guardrail_runner: GuardrailRunner) -> bool:
    """Run the text guardrail on the prompt, checking for content safety.

    Args:
        prompt: The text prompt.
        guardrail_runner: The text guardrail runner.

    Returns:
        bool: Whether the prompt is safe.
    """
    is_safe, message = guardrail_runner.run_safety_check(prompt)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
    return is_safe


def run_video_guardrail(frames: np.ndarray, guardrail_runner: GuardrailRunner) -> np.ndarray | None:
    """Run the video guardrail on the frames, checking for content safety and applying face blur.

    Args:
        frames: The frames of the generated video.
        guardrail_runner: The video guardrail runner.

    Returns:
        The processed frames if safe, otherwise None.
    """
    is_safe, message = guardrail_runner.run_safety_check(frames)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
        return None

    frames = guardrail_runner.postprocess(frames)
    return frames
