# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Out-of-tree π0.5 RL pipeline for the pinned Omni diffusion engine."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from typing import Any

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest

from cosmos_framework.posttrain.comm.weight_validation import copy_named_tensors
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
from cosmos_framework.posttrain.model.pi05.records import map_record
from cosmos_framework.posttrain.roles.rollout.validation import isolated_validation_rng
from cosmos_framework.posttrain.roles.rollout.vllm_omni_worker import OmniIpcWorkerExtension

MODEL_ARCH = "Pi05RLPipeline"
_REGISTERED = False


class Pi05RLPipeline(torch.nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.policy = Pi05TrainPolicy(Pi05ModelConfig(**od_config.model_config["pi05"]), device="cuda")
        self.weights_sources = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return copy_named_tensors(weights, dict(self.policy.net.named_parameters()))

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        extra = req.sampling_params.extra_args or {}
        if req.request_id == DUMMY_DIFFUSION_REQUEST_ID and "observations" not in extra:
            # The engine's load-time probe has no environment observations.
            return DiffusionOutput(
                output={"actions": np.zeros((1, self.policy.config.action_chunk, 7), dtype=np.float64)}
            )
        if "observations" not in extra or "seed" not in extra:
            raise ValueError("pi05 rollout requires observations and an explicit seed")
        observations, seed = extra["observations"], extra["seed"]
        mode = extra.get("sampling_mode", "default")
        if mode not in ("default", "ode"):
            raise ValueError(f"unsupported pi05 sampling mode {mode!r}")
        with isolated_validation_rng(enabled=mode == "ode"):
            inputs = self.policy.processor.encode(observations)
            condition = self.policy.condition_from_inputs(inputs)
            if mode == "ode":
                latent = self.policy.ppo_ode_actions(condition, seed=seed)
                actions = self.policy.processor.decode(latent).cpu().numpy()
                return DiffusionOutput(output={"actions": actions}, custom_output={"sampling_mode": mode})
            sample = self.policy.ppo_sample(condition, seed=seed)
            cpu_trace = map_record(sample.trace, lambda tensor: tensor.cpu())
            return DiffusionOutput(
                output={"actions": self.policy.processor.decode(sample.normalized_actions).cpu().numpy()},
                custom_output={
                    "inputs": {field.name: getattr(inputs, field.name) for field in fields(inputs)},
                    "trace": {field.name: getattr(cpu_trace, field.name) for field in fields(cpu_trace)},
                    "logprobs": sample.scores.logprobs.cpu(),
                    "values": sample.scores.values.cpu(),
                },
            )


def register_pi05_pipeline() -> str:
    global _REGISTERED
    if not _REGISTERED:
        from vllm_omni.diffusion.registry import register_diffusion_model

        register_diffusion_model(MODEL_ARCH, __name__, Pi05RLPipeline.__name__)
        _REGISTERED = True
    return MODEL_ARCH


class Pi05OmniWorkerExtension(OmniIpcWorkerExtension):
    def _live_named_parameters(self) -> Iterable[tuple[str, torch.Tensor]]:
        return self.model_runner.pipeline.policy.net.named_parameters()

    def describe_pipeline(self) -> dict:
        pipeline = self.model_runner.pipeline
        return {
            "class": type(pipeline).__name__,
            "identity": pipeline.policy.ppo_identity(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "parameter_dtypes": {
                name: str(parameter.dtype) for name, parameter in pipeline.policy.net.named_parameters()
            },
        }


register_pi05_pipeline()
