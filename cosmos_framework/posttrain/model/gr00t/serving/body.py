# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rollout role -- GR00T-N1.7 policy serving on a vLLM-Omni engine.

The GR00T half of ``roles/rollout/vllm_omni.py::OmniPolicyBody``: the engine plumbing, the
weight-version gate and the ``infer``/``infer_batch`` contract live in the base class, and this module
supplies the model contract -- engine kwargs, the ``robot_obs`` request, the action chunk, and the
startup checks. Same ``infer``/``infer_batch`` shape as ``the external native policy rollout body``, so
``the external policy-evaluation entrypoint`` and ``the external simulator environment body`` drive it unchanged. Eval is
outcome-only, so there is no ``Rollout.generate()`` / per-step ``Trajectory`` here.

The full Omni engine (model in a same-card worker subprocess) is used rather than an in-process
pipeline so the stage-3 weight plane can reuse the shape of ``roles/rollout/vllm.py``. Heavy imports
are deferred into ``__init__`` / methods so the driver can import this module cheaply.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.roles.rollout.decision import DecisionRolloutMixin
from cosmos_framework.posttrain.roles.rollout.vllm_omni import OmniPolicyBody

# A STRING: importing the defining module would pull vLLM-Omni into every process that declares this
# role. The worker resolves it, and that resolution is what registers the pipeline in the worker.
WORKER_EXTENSION_CLS = "cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline.Gr00tOmniWorkerExtension"
# Restated from `omni_pipeline` so the light import path need not pull vLLM-Omni for two
# strings; `tests/test_roles_rollout_gr00t_omni_pipeline.py` pins both to their definitions.
ACTION_HEAD_PREFIX = "action_head."
MODEL_ARCH_NAME = "Gr00tN1d7Pipeline"


class Gr00tPolicyBody(DecisionRolloutMixin, OmniPolicyBody):
    """GR00T-N1.7 served through vLLM-Omni, behind the sim-eval ``infer``/``infer_batch`` contract."""

    def __init__(
        self, config: Gr00tN1d7ModelConfig, *, ppo: bool = False, ppo_save_cfg_dir: str = "", **engine_kwargs: Any
    ) -> None:
        from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import register_gr00t_rl_pipeline

        if not config.checkpoint_path:
            raise ValueError("Gr00tPolicyBody requires gr00t.checkpoint_path")
        self.config = config
        self.ppo = ppo
        self._ppo_save_cfg_dir = ppo_save_cfg_dir
        if ppo and not ppo_save_cfg_dir:
            raise ValueError("GR00T PPO rollout requires ppo_save_cfg_dir")
        # Resolve before the engine starts so a typo fails here, not deep inside the worker. Both are
        # read back by `_engine_kwargs`, which the base constructor calls. The engine is handed the
        # resolved path, never the bare repo id, so `revision` pins what is served (see
        # `Gr00tN1d7ModelConfig.resolved_checkpoint_path`); `_verify_rl_pipeline` checks the worker
        # loaded exactly that.
        self._embodiment = config.resolve_embodiment_tag()
        if ppo:
            from cosmos_framework.posttrain.model.gr00t.source import resolve_ppo_source

            source = resolve_ppo_source(config.checkpoint_path, config.revision)
            self._model_path = str(source.checkpoint)
            self._ppo_source_identity = source.identity()
        else:
            self._model_path = config.resolved_checkpoint_path()
        # The worker registers on import (that copy decides what runs); this keeps the parent's
        # registry consistent.
        register_gr00t_rl_pipeline()

        # The base starts the engine, proves it runs on this GPU, then calls `_after_engine_start`.
        super().__init__(**engine_kwargs)

    def sample_decisions(
        self, observations: list[dict], *, sources: list[dict], seed: int, weight_version: int
    ) -> list:
        from cosmos_framework.posttrain.model.gr00t.ppo import sampled_decisions

        return sampled_decisions(
            observations,
            self.infer_batch(observations, seed=seed, weight_version=weight_version),
            sources,
            seed=seed,
            version=weight_version,
        )

    def ppo_identity(self) -> dict:
        """Expose the source proved by the live worker for the learner/rollout binding check."""
        if not self.ppo:
            raise RuntimeError("GR00T PPO was not enabled when constructing this rollout")
        return {**self.config.identity(), "source": dict(self.pipeline_info["source_identity"])}

    def bootstrap_values(self, observations: list[dict], *, seed: int, weight_version: int) -> tuple[float, ...]:
        return tuple(
            row["old_value"] for row in self.infer_batch(observations, seed=seed, weight_version=weight_version)
        )

    def _evaluation_actions(self, observations: list[dict], *, seed: int, weight_version: int) -> list:
        return [
            row["action"]
            for row in self.infer_batch(
                observations, seed=seed, weight_version=weight_version, sampling_mode=self.evaluation_mode
            )
        ]

    def _after_engine_start(self) -> None:
        """Every check here asks the worker what it built; re-run after `rebuild_engine` for the new worker."""
        self.modality = self.describe_modality()
        self._verify_against_checkpoint(self.modality)
        self.pipeline_info = self._verify_rl_pipeline()

    def _reset_engine_caches(self) -> None:
        self.modality = None
        self.pipeline_info = None

    # ── model hooks ──────────────────────────────────────────────────────────
    def _engine_kwargs(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "model": self._model_path,
            # `model_class_name` is deliberately not passed: `enrich_config` overwrites it for GR00T.
            # The RL pipeline is selected by replacing the built-in registry entry instead.
            # `denoising_steps` rides in `model_config` because that dict is the only body-controlled
            # input the worker-side pipeline constructor sees; the stock constructor ignores keys it
            # does not know and `Gr00tN1d7RLPipeline` applies this one to the action head.
            "model_config": {
                "embodiment_tag": self._embodiment.name,
                "strict": True,
                "denoising_steps": self.config.denoising_steps,
                "ppo": self.ppo,
                "ppo_model_config": asdict(self.config) if self.ppo else None,
                "ppo_save_cfg_dir": self._ppo_save_cfg_dir,
            },
            "worker_extension_cls": WORKER_EXTENSION_CLS,
            # One request in, one action chunk out; vectorized envs ride a B>1 `robot_obs` in ONE
            # request rather than several batched requests.
            "max_num_seqs": 1,
            "enforce_eager": True,
        }

    def _build_request(self, obs_list: Sequence[dict], seed: int | None) -> tuple[list[str], Any]:
        """One batched ``robot_obs`` request, with the seed in ``extra_args``.

        NOT ``OmniDiffusionSamplingParams.seed``: the GR00T port's action head draws its initial noise
        with a plain ``torch.randn`` and never consumes the generator vLLM-Omni derives from that
        field, so ``Gr00tN1d7RLPipeline.forward`` reads the seed from ``extra_args`` instead. The
        field is also auto-filled by ``OmniDiffusionRequest`` when unset, so it cannot express
        "unseeded". See ``model/gr00t/serving/omni_pipeline.py``.
        """
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        robot_obs, prompt = self._build_robot_obs(obs_list)
        extra_args: dict[str, Any] = {"robot_obs": robot_obs}
        if seed is not None:
            extra_args["seed"] = int(seed)
        return [prompt], OmniDiffusionSamplingParams(extra_args=extra_args)

    def _configure_sampling_mode(self, params: Any, mode: str) -> None:
        if not self.ppo or mode not in ("ode", "native_ode"):
            raise ValueError(f"GR00T PPO does not support sampling_mode={mode!r}")
        params.extra_args["sampling_mode"] = mode

    def inference_fingerprint(self) -> dict:
        """Read the loaded worker's weights and effective storage dtypes."""
        return {
            **self._worker_rpc("inference_fingerprint", timeout=300),
            "publication_version": self.weight_version(),
        }

    def _validate_sampling_mode(self, output: Any, mode: str) -> None:
        expected = {"sampling_mode": mode}
        if mode == "native_ode":
            from cosmos_framework.posttrain.model.gr00t.ppo import NATIVE_ODE_REVISION

            expected["ode_revision"] = NATIVE_ODE_REVISION
        if mode not in ("ode", "native_ode") or output.custom_output != expected:
            raise ValueError("GR00T worker did not attest deterministic ODE evaluation")

    def _parse_output(self, output: Any, n: int) -> list[dict]:
        actions = self._split_actions(self._extract_actions(output, self.modality["action_keys"]), n)
        if self.ppo:
            import torch

            if output.custom_output.get("sampling_mode") in ("ode", "native_ode"):
                return actions
            from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOCondition, Gr00tPPOTrace

            payload = output.custom_output
            # Replaying the original invocation shape avoids bf16 GEMM differences
            # between B=1 scoring and B>1 sampling. Each decision selects one row.
            condition = Gr00tPPOCondition(**{key: value.clone() for key, value in payload["condition"].items()})
            trace = Gr00tPPOTrace(
                **{key: value.clone() if torch.is_tensor(value) else value for key, value in payload["trace"].items()}
            )
            for index, action in enumerate(actions):
                action["condition"] = condition
                action["trace"] = trace
                action["sample_index"] = index
                action["old_logprob"] = float(payload["logprobs"][index])
                action["old_value"] = float(payload["values"][index])
        return actions

    # ── startup verification ─────────────────────────────────────────────────
    def _verify_rl_pipeline(self) -> dict[str, Any]:
        """Refuse to serve unless the worker built what the recipe asked for.

        Four facts, all invisible from the output: a stock pipeline returns right-shaped actions while
        ignoring per-request seeding; the stock image transform feeds the model out-of-distribution
        crops; an action head still on the checkpoint's step count integrates a different ODE than the
        recipe's ``denoising_steps``; and an engine that loaded something other than the resolved
        checkpoint path serves an artifact ``identity_hash()`` does not describe. Ask the worker at
        startup, reading each value off the live objects rather than off the config it was given.
        """
        described = self.describe_pipeline()
        if bool(described.get("ppo", False)) != self.ppo:
            raise RuntimeError("GR00T worker PPO capability differs from its rollout binding")
        if self.ppo and described.get("sdpa_math_effective") is not True:
            raise RuntimeError("GR00T PPO worker did not enable math SDPA")
        if self.ppo and described.get("source_identity") != self._ppo_source_identity:
            raise RuntimeError("GR00T PPO worker loaded different model or processor source contents")
        # Pipeline class first: a stock pipeline also lacks the transform, and the class is the cause.
        if not described.get("is_rl_pipeline"):
            raise RuntimeError(
                "the vLLM-Omni worker built "
                f"{described.get('module')}.{described.get('class')}, not Gr00tN1d7RLPipeline, so "
                "per-request seeding is silently inactive. The RL pipeline is selected by replacing "
                f"the built-in registry entry {MODEL_ARCH_NAME!r}; check that vLLM-Omni's "
                "`OmniDiffusionConfig.enrich_config` still resolves GR00T to that name and that "
                "`worker_extension_cls` still imports `omni_pipeline` before the model is built."
            )
        if not described.get("image_transform"):
            raise RuntimeError(
                "the RL pipeline is running WITHOUT the reference eval image transform "
                f"(describe_pipeline reported image_transform={described.get('image_transform')!r}); "
                "the stock transform crops to a fixed pixel size instead of the checkpoint's "
                "crop_fraction. See model/gr00t/serving/image_transform.py."
            )
        steps = described.get("num_inference_timesteps")
        if steps != self.config.denoising_steps:
            raise RuntimeError(
                f"the worker's action head runs num_inference_timesteps={steps!r}, but the recipe "
                f"declares gr00t.denoising_steps={self.config.denoising_steps}. "
                "Gr00tN1d7RLPipeline._apply_denoising_steps sets it at construction, so a mismatch "
                "means the override did not reach the live action head."
            )
        loaded = described.get("model_path")
        if not isinstance(loaded, str) or not self._same_artifact(loaded, self._model_path):
            raise RuntimeError(
                f"the worker loaded {loaded!r}, but gr00t.checkpoint_path/revision resolved to "
                f"{self._model_path!r}; the engine is not serving the pinned artifact."
            )
        return described

    @staticmethod
    def _same_artifact(reported: str, resolved: str) -> bool:
        """Equal strings, or the same directory under two spellings; a bare repo id is a string."""
        if reported == resolved:
            return True
        return os.path.isdir(reported) and os.path.isdir(resolved) and os.path.samefile(reported, resolved)

    def _verify_against_checkpoint(self, modality: Mapping[str, Any]) -> None:
        """Reject a recipe whose modality contract disagrees with the loaded checkpoint.

        Every mismatch here (camera key, state composition, language key, horizon, tag) is silent at
        runtime: the policy simply acts worse, indistinguishable from the policy being bad.
        """
        cfg = self.config
        problems: list[str] = []

        expected_video = set(modality["video_keys"])
        if set(cfg.camera_map) != expected_video:
            problems.append(
                f"camera_map keys {sorted(cfg.camera_map)} != checkpoint video keys {sorted(expected_video)}"
            )
        expected_state = set(modality["state_keys"])
        if set(cfg.state_map) != expected_state:
            problems.append(f"state_map keys {sorted(cfg.state_map)} != checkpoint state keys {sorted(expected_state)}")
        if cfg.language_key != modality["language_key"]:
            problems.append(f"language_key {cfg.language_key!r} != checkpoint {modality['language_key']!r}")
        if cfg.execution_horizon > modality["action_horizon"]:
            problems.append(
                f"execution_horizon {cfg.execution_horizon} exceeds processor action horizon "
                f"{modality['action_horizon']}"
            )
        if "native_action_horizon" in modality and cfg.model_action_horizon != modality["native_action_horizon"]:
            problems.append("model_action_horizon differs from the loaded native action head")
        if cfg.embodiment_tag.lower() not in (modality["embodiment_tag"].lower(), self._embodiment.name.lower()):
            problems.append(f"embodiment_tag {cfg.embodiment_tag!r} != checkpoint {modality['embodiment_tag']!r}")

        if problems:
            raise ValueError(
                "GR00T recipe disagrees with the loaded checkpoint:\n  - "
                + "\n  - ".join(problems)
                + f"\nCheckpoint modality contract: {dict(modality)}"
            )

    # ── observation adaptation ───────────────────────────────────────────────
    def _build_robot_obs(self, obs_list: Sequence[dict]) -> tuple[dict[str, Any], str]:
        """Env observation dicts -> one batched GR00T ``robot_obs`` payload.

        Reads only ``self.config``, and is called directly (on a body built with ``object.__new__``,
        no engine) by ``tests/gr00t_parity_cli.py`` so the reference and serving sides are handed a
        byte-identical input. Keep it a standalone seam.

        Shape contract, from ``Gr00tPolicy.check_observation``: each modality is a DICT KEYED BY
        MODALITY KEY (the processor indexes cameras and state components by name)::

            video     {gr00t_key: uint8   [B, T, H, W, C]}
            state     {gr00t_key: float32 [B, T, D]}
            language  {language_key: [[str]] * B}

        ``T`` is 1 for this checkpoint. A missing env key raises listing the keys actually present;
        a camera value that is not already an HWC RGB uint8 frame is refused, not cast (see
        ``_require_image``).
        """
        cfg = self.config
        for i, obs in enumerate(obs_list):
            if not isinstance(obs, Mapping):
                raise TypeError(f"GR00T observation {i} must be a mapping, got {type(obs).__name__}")

        video = {
            gr00t_key: np.stack([self._require_image(o, env_key, i) for i, o in enumerate(obs_list)], axis=0)[
                :, None
            ]  # [B, T=1, H, W, C]
            for gr00t_key, env_key in cfg.camera_map.items()
        }
        state = {
            gr00t_key: np.stack(
                [np.atleast_1d(self._require(o, env_key, i)).ravel() for i, o in enumerate(obs_list)], axis=0
            )[:, None].astype(np.float32, copy=False)  # [B, T=1, D]
            for gr00t_key, env_key in cfg.state_map.items()
        }

        # One prompt per element; the stock `_normalize_observation` builds `[[prompt]]` for B=1 only.
        prompts = [str(o.get(cfg.prompt_key, "") or "") for o in obs_list]
        robot_obs = {
            "video": video,
            "state": state,
            "language": {cfg.language_key: [[p] for p in prompts]},
        }
        return robot_obs, prompts[0]

    @staticmethod
    def _require(obs: Mapping[str, Any], key: str, index: int) -> np.ndarray:
        if key not in obs:
            raise KeyError(
                f"GR00T observation {index} is missing configured key {key!r}. Present keys: "
                f"{sorted(obs)}. Fix the env-side values of gr00t.camera_map / gr00t.state_map in "
                "the recipe to match the env's actual observation schema."
            )
        return np.asarray(obs[key])

    @staticmethod
    def _require_image(obs: Mapping[str, Any], key: str, index: int) -> np.ndarray:
        """A camera value, required to already be an HWC RGB uint8 frame.

        Refused rather than corrected: an ``astype(uint8)`` on a float image, out-of-range integers or
        a CHW array truncates or wraps into plausible-looking pixels, and the eval image transform in
        the worker (which does check) would then see nothing wrong. Failing here names the env key
        and the batch element, which the worker-side check cannot.
        """
        frame = Gr00tPolicyBody._require(obs, key, index)
        if frame.dtype != np.uint8:
            raise TypeError(
                f"GR00T observation {index} camera {key!r} must be uint8, got dtype {frame.dtype}; the env "
                "must emit HWC RGB uint8 frames (a float or wider-integer image is not converted)"
            )
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"GR00T observation {index} camera {key!r} must be HWC RGB with shape (H, W, 3), got "
                f"shape {frame.shape}"
            )
        return frame

    # ── action adaptation ────────────────────────────────────────────────────
    @staticmethod
    def _extract_actions(output: Any, action_keys: Sequence[str] | None = None) -> np.ndarray:
        """``OmniRequestOutput`` -> ``[B, H, D]`` in the processor's action order."""
        multimodal = getattr(output, "multimodal_output", None)
        if not isinstance(multimodal, Mapping) or "actions" not in multimodal:
            raise RuntimeError("GR00T engine output carries no multimodal_output['actions']")
        actions = multimodal["actions"]
        if isinstance(actions, Mapping):
            if not actions:
                raise RuntimeError("GR00T engine returned an empty action dict")
            keys = tuple(actions) if action_keys is None else tuple(action_keys)
            if set(actions) != set(keys):
                raise ValueError("GR00T output action keys differ from the processor modality")
            parts = [np.asarray(actions[k], dtype=np.float32) for k in keys]
            return np.concatenate(parts, axis=-1)  # [B, H, D]
        return np.asarray(actions, dtype=np.float32)

    def _split_actions(self, actions: np.ndarray, n: int) -> list[dict]:
        if actions.ndim != 3:
            raise RuntimeError(f"expected GR00T actions shaped [B, H, D], got {actions.shape}")
        if actions.shape[0] != n:
            raise RuntimeError(f"GR00T returned {actions.shape[0]} action chunks for {n} observations")
        horizon = self.config.execution_horizon
        if self.config.action_dim and actions.shape[2] != self.config.action_dim:
            raise RuntimeError(
                f"GR00T returned action width {actions.shape[2]}, but the recipe declares "
                f"action_dim={self.config.action_dim}; a different width is a different control vector."
            )
        if actions.shape[1] < horizon:
            raise RuntimeError(
                f"GR00T returned horizon {actions.shape[1]} < execution_horizon {horizon}; the "
                "checkpoint emits fewer steps than the recipe intends to execute"
            )
        return [{"action": actions[i, :horizon]} for i in range(n)]

    # ── introspection ────────────────────────────────────────────────────────
    def describe_action_head(self) -> dict[str, dict[str, Any]]:
        """The worker's own view of the live head layout, independent of the body's cached mapping.

        The weight plane reads the mapping (`_engine_param_layout`), so this exists for reports and for the
        GPU test that must compare the mapping against a second, independent reader rather than itself.
        """
        return self._worker_rpc("describe_action_head")

    def describe_modality(self) -> dict[str, Any]:
        return self._worker_rpc("describe_modality")

    def describe_pipeline(self) -> dict[str, Any]:
        return self._worker_rpc("describe_pipeline")

    def backend(self) -> str:
        return "gr00t_omni"
