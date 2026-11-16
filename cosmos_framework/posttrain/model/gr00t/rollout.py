# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Observation-only rollout of a prebuilt native GR00T adapter/dense artifact.

The simulator owns image orientation and gripper execution conversion. This adapter
accepts its canonical upright RGB/axis-angle observation and returns native decoded
[0,1] gripper chunks, which LiberoChunkSession binarizes/inverts exactly once.
"""

from __future__ import annotations

import time

import numpy as np

ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def prepare_observations(policy, observations):
    from gr00t.data.types import MessageType, VLAStepData

    if not observations:
        raise ValueError("native GR00T rollout requires at least one observation")
    features, states = [], []
    for index, observation in enumerate(observations):
        state = np.asarray(observation["state"])
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("LIBERO state must be finite xyz/axis-angle/two-finger positions")
        if not isinstance(observation["prompt"], str) or not observation["prompt"]:
            raise ValueError("LIBERO rollout requires its task language instruction")
        images = {}
        for camera in ("image", "wrist_image"):
            image = observation[camera]
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"GR00T rollout {index}.{camera} requires canonical HWC RGB uint8")
            images[camera] = [image]
        state = state.astype(np.float32, copy=False)
        native_state = {key: state[None, i : i + 1] if i < 6 else state[None, 6:] for i, key in enumerate(ACTION_KEYS)}
        states.append(native_state)
        native = VLAStepData(
            images=images,
            states=native_state,
            actions={},
            text=observation["prompt"],
            embodiment=policy.config.resolve_embodiment_tag(),
        )
        item = policy.packer.processor([{"type": MessageType.EPISODE_STEP.value, "content": native}])
        item["vlm_content"].pop("conversation", None)
        features.append(item)
    inputs = dict(policy.packer.processor.collator(features)["inputs"])
    # A processor must never introduce action conditioning into observation-only inference.
    if "action" in inputs:
        raise ValueError("GR00T observation-only processor unexpectedly supplied action conditioning")
    batched_state = {key: np.stack([row[key] for row in states]) for key in ACTION_KEYS}
    return inputs, batched_state


def predict_processed(policy, inputs, *, native_states=None):
    import torch

    clean = {key: value for key, value in inputs.items() if key not in {"action", "action_mask"}}
    with torch.no_grad(), policy.compute_ctx:
        normalized = policy.gr00t.get_action(clean)["action_pred"].float()
    if not torch.isfinite(normalized).all():
        raise ValueError("native GR00T policy produced nonfinite normalized actions")
    decoded = policy.packer.processor.decode_action(
        normalized.cpu().numpy(), policy.config.resolve_embodiment_tag(), native_states
    )
    action = np.concatenate([decoded[key] for key in ACTION_KEYS], axis=-1).astype(np.float32, copy=False)
    if action.ndim != 3 or action.shape[1:] != (16, 7) or not np.isfinite(action).all():
        raise ValueError("native GR00T decoded action must be finite [B,16,7]")
    return normalized, action


class NativeArtifactRollout:
    def __init__(self, policy, *, latency_capacity=0):
        from cosmos_framework.posttrain.model.gr00t.adaptation import validate_task_config

        validate_task_config(policy.config)
        if policy.config.execution_horizon != 8 or policy.config.training_prefix < 8:
            raise ValueError("GR00T QLoRA spatial rollout requires eight supervised execution actions")
        if type(latency_capacity) is not int or latency_capacity < 0:
            raise ValueError("latency_capacity must be a nonnegative declared decision bound")
        self.policy = policy
        self.policy.net.eval()
        self.latency_capacity = latency_capacity
        self.response_timings = []

    def latency_report(self, *, warmup_calls):
        if type(warmup_calls) is not int or warmup_calls < 0:
            raise ValueError("warmup_calls must be a nonnegative prospectively selected count")
        steady = [row["response_seconds"] for row in self.response_timings[warmup_calls:]]
        return {
            "scope": "native observation preparation, transfer, four-step action sampling, decoding and RNG/mode restoration; excludes gateway IPC and simulator rendering/control",
            "warmup_calls": warmup_calls,
            "cold_first_response_seconds": self.response_timings[0]["response_seconds"]
            if self.response_timings
            else None,
            "steady_calls": len(steady),
            "steady_seconds": {
                "mean": float(np.mean(steady)),
                "p50": float(np.quantile(steady, 0.50)),
                "p95": float(np.quantile(steady, 0.95)),
                "p99": float(np.quantile(steady, 0.99)),
            }
            if steady
            else None,
            "calls": list(self.response_timings),
            "comparison_limit": "On-policy observations can diverge between artifacts; these response distributions alone do not establish matched-input inference acceleration.",
        }

    def describe(self):
        from cosmos_framework.posttrain.embodied.rollout import RolloutInfo

        return RolloutInfo(name="native_gr00t_artifact", action_dim=7, horizon=8, embodiment="libero_sim")

    def act(self, observations, *, context):
        from cosmos_framework.posttrain.embodied.sims.libero.protocol import libero_seed
        from cosmos_framework.posttrain.entrypoints.action_sft import preserved_evaluation

        started = time.perf_counter()
        if self.latency_capacity and len(self.response_timings) >= self.latency_capacity:
            raise RuntimeError("native rollout exceeded its declared response-measurement capacity")
        case = context["case"]
        if case["suite"] != "libero_spatial":
            raise ValueError("this GR00T artifact rollout supports only LIBERO spatial")
        if "scene_id" in case:
            from cosmos_framework.posttrain.embodied.sims.libero.generated import POLICY_NOISE
            from cosmos_framework.posttrain.embodied.sims.libero.protocol import GeneratedLiberoCase

            generated = GeneratedLiberoCase(**case)
            seed = libero_seed(
                "libero_spatial",
                POLICY_NOISE,
                context["seed"],
                generated.task_id,
                int(generated.scene_id, 16),
                context["decision"],
            )
        else:
            seed = libero_seed(
                "libero_spatial",
                "evaluation-policy",
                context["seed"],
                case["task_id"],
                case["init_state_id"],
                context["decision"],
            )
        import torch

        # NumPy's legacy RNG accepts uint32; preserve the full native policy seed for CUDA noise.
        with preserved_evaluation(self.policy, seed % 2**32):
            torch.manual_seed(seed)
            inputs, states = prepare_observations(self.policy, observations)
            _, actions = predict_processed(self.policy, inputs, native_states=states)
        result = [action[:8].copy() for action in actions]
        # predict_processed transfers the complete predictions to CPU before
        # decoding. The returned controller commands therefore include completed
        # GPU work; timing a CUDA enqueue alone would understate response latency.
        elapsed = time.perf_counter() - started
        if self.latency_capacity:
            self.response_timings.append(
                {
                    "case": dict(case),
                    "decision": context["decision"],
                    "seed": context["seed"],
                    "response_seconds": elapsed,
                    "batch_size": len(observations),
                    "media_shapes": [
                        {camera: list(row[camera].shape) for camera in ("image", "wrist_image")} for row in observations
                    ],
                    "decoded_horizon": 16,
                    "executed_horizon": 8,
                    "action_dimensions": 7,
                    "flow_sampling_steps": 4,
                }
            )
        return result
