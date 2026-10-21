# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pi0.5 requests and records on the shared Omni lifecycle and weight receiver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.source import Pi05Source, model_identity
from cosmos_framework.posttrain.roles.rollout.decision import DecisionRolloutMixin
from cosmos_framework.posttrain.roles.rollout.vllm_omni import OmniPolicyBody

MODEL_ARCH = "Pi05RLPipeline"
WORKER_EXTENSION = "cosmos_framework.posttrain.model.pi05.serving.omni_pipeline.Pi05OmniWorkerExtension"


class Pi05PolicyBody(DecisionRolloutMixin, OmniPolicyBody):
    def __init__(self, config: Pi05ModelConfig, *, output: str, **engine_kwargs: Any) -> None:
        from cosmos_framework.posttrain.model.pi05.serving.omni_pipeline import register_pi05_pipeline

        self.config = config
        self.source_identity = Pi05Source.resolve(config).identity()
        self._serving_directory = str(Path(output) / "pi05-serving")
        Path(self._serving_directory).mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(Path(self._serving_directory) / "model_index.json"), {"_class_name": MODEL_ARCH})
        register_pi05_pipeline()
        super().__init__(transfer_dtype=None, **engine_kwargs)

    def _engine_kwargs(self) -> dict[str, Any]:
        return {
            "model": self._serving_directory,
            "model_class_name": MODEL_ARCH,
            "model_config": {"pi05": asdict(self.config)},
            "worker_extension_cls": WORKER_EXTENSION,
            "max_num_seqs": 1,
            "enforce_eager": True,
        }

    def _after_engine_start(self) -> None:
        self.pipeline_info = self._worker_rpc("describe_pipeline")
        if self.pipeline_info["identity"] != model_identity(self.config, self.source_identity):
            raise ValueError("pi05 Omni worker loaded different model/processor/sampler sources")
        if self.pipeline_info["class"] != MODEL_ARCH:
            raise ValueError("pi05 Omni worker did not construct the registered RL pipeline")
        if self.pipeline_info["matmul_precision"] != "high":
            raise ValueError("pi05 Omni worker did not retain the reference FP32 matmul precision")

    def _reset_engine_caches(self) -> None:
        self.pipeline_info = None

    def _build_request(self, obs_list: Sequence[dict], seed: int | None) -> tuple[list[str], Any]:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if type(seed) is not int or seed < 0:
            raise ValueError("pi05 rollout requires an explicit nonnegative integer seed")
        return [obs_list[0]["prompt"]], OmniDiffusionSamplingParams(
            extra_args={"observations": list(obs_list), "seed": seed}
        )

    def _configure_sampling_mode(self, params: Any, mode: str) -> None:
        if mode != "ode":
            raise ValueError(f"unsupported pi05 sampling mode {mode!r}")
        params.extra_args["sampling_mode"] = mode

    def _validate_sampling_mode(self, output: Any, mode: str) -> None:
        if output.custom_output.get("sampling_mode") != mode:
            raise ValueError("pi05 worker did not attest the requested sampling mode")

    def _parse_output(self, output: Any, n: int) -> list[dict]:
        from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOTrace, validate_selected_trace
        from cosmos_framework.posttrain.model.pi05.processing import Pi05ReplayInputs, validate_replay_inputs
        from cosmos_framework.posttrain.model.pi05.records import map_record

        payload = output.custom_output
        actions = np.asarray(output.multimodal_output["actions"])
        if (
            actions.dtype != np.float64
            or actions.shape != (n, self.config.action_chunk, 7)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("pi05 Omni output has invalid action geometry or values")
        rows = [{"action": action} for action in actions]
        if payload.get("sampling_mode") == "ode":
            return rows
        inputs = map_record(Pi05ReplayInputs(**payload["inputs"]), lambda tensor: tensor.clone())
        trace = map_record(Pi05PPOTrace(**payload["trace"]), lambda tensor: tensor.clone())
        if validate_replay_inputs(inputs) != n:
            raise ValueError("pi05 Omni replay inputs differ from the requested batch")
        validate_selected_trace(trace, self.config, batch=n)
        for index, row in enumerate(rows):
            row.update(
                inputs=inputs,
                trace=trace,
                sample_index=index,
                old_logprob=float(payload["logprobs"][index]),
                old_value=float(payload["values"][index]),
            )
        return rows

    def sample_decisions(
        self, observations: list[dict], *, sources: list[dict], seed: int, weight_version: int
    ) -> list:
        from cosmos_framework.posttrain.model.pi05.records import sampled_decisions

        return sampled_decisions(
            observations,
            self.infer_batch(observations, seed=seed, weight_version=weight_version),
            sources,
            seed=seed,
            version=weight_version,
        )

    def bootstrap_values(self, observations: list[dict], *, seed: int, weight_version: int) -> tuple[float, ...]:
        return tuple(
            row["old_value"] for row in self.infer_batch(observations, seed=seed, weight_version=weight_version)
        )

    def _evaluation_actions(self, observations: list[dict], *, seed: int, weight_version: int) -> list:
        return [
            row["action"]
            for row in self.infer_batch(observations, seed=seed, weight_version=weight_version, sampling_mode="ode")
        ]

    def ppo_identity(self) -> dict:
        return self.pipeline_info["identity"]
