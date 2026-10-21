# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Out-of-tree vLLM-Omni pipeline for GR00T-N1.7 under posttrain RL.

Subclasses vLLM-Omni's stock ``Gr00tN1d7Pipeline``; vLLM-Omni itself is not patched. It adds what RL
needs and the stock pipeline lacks: a per-request seed (stock GR00T draws noise from the global RNG or
from the process-wide ``GR00T_NOISE_SEED``) and a weight sink (``load_weights``) for the stage-3 plane.

Two facts govern how the class gets used:

1. It is registered under the STOCK arch name, replacing the built-in pipeline. See ``MODEL_ARCH``.
2. Registration must happen inside the worker process. vLLM-Omni spawns its diffusion workers, so
   each starts with a pristine registry. ``WorkerWrapperBase`` resolves ``worker_extension_cls`` by
   qualname -- importing this module -- before it builds the worker that loads the model, so the
   import-time registration at the bottom of this module lands just in time. That is why the worker
   extension lives here rather than beside the rollout body.

Importing this module pulls in vLLM-Omni and replaces a registry entry for the whole process. Only the
rollout worker (and the parity script) should import it; never a config or schema module.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from cosmos_framework.posttrain.comm.weight_validation import copy_named_tensors
from cosmos_framework.posttrain.roles.rollout.vllm_omni_worker import OmniIpcWorkerExtension


class Gr00tObservationProcessor:
    """Isaac processor front end for PPO requests: batched ``robot_obs`` in, decoded actions out."""

    def __init__(self, processor: Any, embodiment: str, model: Any) -> None:
        from gr00t.data.embodiment_tags import EmbodimentTag

        self.processor = processor
        self.model = model
        self.processor.eval()
        self.embodiment_tag = EmbodimentTag.resolve(embodiment)
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.language_key = self.modality_configs["language"].modality_keys[0]

    def prepare(self, observation: Mapping[str, Any], device: torch.device) -> tuple[dict, dict]:
        from gr00t.data.types import MessageType, VLAStepData

        batch = next(iter(observation["video"].values())).shape[0]
        samples, states = [], []
        for index in range(batch):
            state = {key: value[index] for key, value in observation["state"].items()}
            step = VLAStepData(
                images={key: value[index] for key, value in observation["video"].items()},
                states=state,
                actions={},
                text=observation["language"][self.language_key][index][0],
                embodiment=self.embodiment_tag,
            )
            samples.append(self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
            states.append(state)
        inputs = self.processor.collator(samples)["inputs"]
        inputs = {
            key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        return inputs, {key: np.stack([state[key] for state in states]) for key in states[0]}

    def decode(self, normalized: torch.Tensor, states: dict) -> dict[str, np.ndarray]:
        return self.processor.decode_action(normalized.float().cpu().numpy(), self.embodiment_tag, states)


# Registered under the stock name, REPLACING vLLM-Omni's built-in GR00T pipeline (a documented use of
# `register_diffusion_model`). Forced, not stylistic: `OmniDiffusionConfig.enrich_config` sets
# `model_class_name = "Gr00tN1d7Pipeline"` for GR00T checkpoints with no `is None` guard, so an
# explicitly passed name is overwritten and a new registry name can never be selected -- the engine
# would silently build the stock pipeline and ignore per-request seeding. Revisit if upstream adds
# the guard.
MODEL_ARCH = "Gr00tN1d7Pipeline"
# Worker-reported parameter names carry this prefix so they compare 1:1 with the trainer's
# `net.state_dict()` keys, which hold the head under `action_head.`.
ACTION_HEAD_PREFIX = "action_head."
_REGISTERED = False


class Gr00tN1d7RLPipeline(Gr00tN1d7Pipeline):
    """Stock GR00T-N1.7 inference with a per-request seed and an RL weight sink."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        # Under GR00T_NOISE_SEED the action head takes its own `torch.Generator` branch and ignores the
        # RNG state `forward` installs: every request would share one noise draw while the caller
        # believed its seed took effect.
        if os.environ.get("GR00T_NOISE_SEED") is not None:
            raise RuntimeError(
                "GR00T_NOISE_SEED is set in this process. It makes GR00T's initial denoising noise a "
                "process-wide constant and overrides the per-request seed this pipeline installs. "
                "Unset it; pass seeds through sampling_params.extra_args['seed'] instead."
            )
        self.ppo = bool(od_config.model_config.get("ppo", False))
        if self.ppo:
            from dataclasses import replace
            from pathlib import Path

            from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
            from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy

            torch.nn.Module.__init__(self)
            self.model_path = od_config.model
            config = Gr00tN1d7ModelConfig(**od_config.model_config["ppo_model_config"])
            config = replace(config, checkpoint_path=self.model_path, revision="")
            directory = Path(od_config.model_config["ppo_save_cfg_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            self.ppo_model = Gr00tN1d7TrainPolicy(config, device="cuda", save_cfg_dir=directory, ppo=True)
            self.policy = Gr00tObservationProcessor(
                self.ppo_model.processor, config.embodiment_tag, self.ppo_model.gr00t
            )
            self._image_transform = "isaac-native-eval"
            self._denoising_steps = config.denoising_steps
            return
        super().__init__(od_config=od_config, prefix=prefix)
        # The port's torchvision eval transform does not reproduce the one the checkpoint was trained
        # with; install the reference pipeline. See `image_transform`.
        from cosmos_framework.posttrain.model.gr00t.serving.image_transform import (
            install_reference_image_transform,
        )

        self._image_transform = install_reference_image_transform(self.policy, self.model_path)
        # The body's recipe value; the stock constructor reads its own keys from `model_config` with
        # `.get` and leaves this one alone.
        self._denoising_steps = od_config.model_config.get("denoising_steps")
        self._apply_denoising_steps()

    def _apply_denoising_steps(self) -> None:
        """Make the recipe's ``denoising_steps`` the action head's Euler step count.

        The stock head fixes ``num_inference_timesteps`` from the checkpoint's ``config.json`` at
        construction and reads the attribute on every call, so overriding the live module is the only
        per-recipe path. It is a plain int outside ``state_dict``: a stage-3 weight copy leaves it
        alone. A method, not an inline assignment, so a path that rebuilds ``self.policy`` can re-apply
        it. The body verifies the effect through ``describe_pipeline``.
        """
        if self._denoising_steps is not None:
            self.policy.model.action_head.num_inference_timesteps = int(self._denoising_steps)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """vLLM-Omni's ``load_weights`` contract for this pipeline.

        Copy each ``action_head.``-prefixed tensor into the live parameter of the same name and return the
        set of names actually loaded. Whatever ``named_parameters()`` the live head has is loadable, so a
        head that grows (a value head, say) needs no change here. The weight plane itself no longer routes
        through this method: the rollout body writes straight into the live parameters the worker exported
        (``Gr00tOmniWorkerExtension``), so this is the engine's loader seam.

        An EMPTY call must succeed: the engine makes one during model loading (GR00T's weights come from
        ``Gr00tPolicy``, so ``weights_sources`` is empty). Anything not a live parameter, or a shape or
        dtype other than the live one, raises: a partial or coerced load would be a silent weight change.
        """
        consumed = list(weights)
        if not consumed:
            return set()
        for name, _tensor in consumed:
            if not name.startswith(ACTION_HEAD_PREFIX):
                raise KeyError(f"{name!r} is outside the {ACTION_HEAD_PREFIX!r} wire scope this pipeline serves")
        live = {
            f"{ACTION_HEAD_PREFIX}{name}": param for name, param in self.policy.model.action_head.named_parameters()
        }
        try:
            return copy_named_tensors(consumed, live)
        except KeyError as error:
            raise KeyError(f"{error.args[0]!r} is not a live action-head parameter of this worker") from error
        finally:
            live.clear()

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        """Stock GR00T inference, optionally under a caller-supplied seed.

        With ``GR00T_NOISE_SEED`` unset the action head's only randomness on this path is one
        ``torch.randn`` with no explicit generator, so seeding the default RNG immediately before makes
        the initial noise -- and the whole ODE -- a pure function of ``extra_args["seed"]`` while the
        sampling math stays untouched. ``fork_rng`` restores the global stream afterwards.
        """
        extra_args = req.sampling_params.extra_args or {}
        if self.ppo:
            from cosmos_framework.posttrain.roles.rollout.validation import isolated_validation_rng

            with isolated_validation_rng(enabled=extra_args.get("sampling_mode", "default") != "default"):
                return self._ppo_forward(req)
        seed = extra_args.get("seed")
        if seed is None:
            return super().forward(req, **kwargs)

        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            return super().forward(req, **kwargs)

    def _ppo_forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        import numpy as np
        from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID

        extra = req.sampling_params.extra_args or {}
        if req.request_id == DUMMY_DIFFUSION_REQUEST_ID and "robot_obs" not in extra:
            return DiffusionOutput(output={"actions": np.zeros((1, 16, 7), dtype=np.float32)})
        if "seed" not in extra or "robot_obs" not in extra:
            raise ValueError("GR00T PPO requires robot_obs and an explicit seed")
        inputs, states = self.policy.prepare(extra["robot_obs"], torch.device("cuda"))
        mode = extra.get("sampling_mode", "default")
        if mode not in ("default", "ode", "native_ode"):
            raise ValueError(f"unsupported GR00T PPO sampling mode {mode!r}")
        condition = self.ppo_model.ppo_condition(inputs, native_ode=mode == "native_ode")
        if mode in ("ode", "native_ode"):
            normalized = self.ppo_model.ppo_ode_actions(condition, seed=int(extra["seed"]), native=mode == "native_ode")
            actions = self._decode_ppo_actions(normalized, states)
            # Native inference returns decoded actions; the simulator owns control-space clipping.
            output = actions if mode == "native_ode" else np.clip(actions, -1, 1)
            attestation = {"sampling_mode": mode}
            if mode == "native_ode":
                from cosmos_framework.posttrain.model.gr00t.ppo import NATIVE_ODE_REVISION

                attestation["ode_revision"] = NATIVE_ODE_REVISION
            return DiffusionOutput(output={"actions": output}, custom_output=attestation)
        sample = self.ppo_model.ppo_sample(condition, seed=int(extra["seed"]))
        actions = self._decode_ppo_actions(sample.normalized_actions, states)
        actions = np.clip(actions + sample.action_noise.float().cpu().numpy(), -1, 1)
        return self._ppo_output(actions, condition, sample)

    def _decode_ppo_actions(self, normalized: torch.Tensor, states: Any) -> Any:
        import numpy as np

        decoded = self.policy.decode(normalized, states)
        actions = np.concatenate(
            [decoded[key] for key in self.policy.modality_configs["action"].modality_keys], axis=-1
        ).astype(np.float32)
        if actions.shape[1:] != (16, 7):
            raise ValueError(f"GR00T processor must decode [B,16,7], got {actions.shape}")
        return actions[:, : self.ppo_model.config.execution_horizon].copy()

    @staticmethod
    def _ppo_output(actions: Any, condition: Any, sample: Any) -> DiffusionOutput:
        """Use Omni's custom-output channel: the action formatter drops other output keys."""
        return DiffusionOutput(
            output={"actions": actions},
            custom_output={
                "condition": {field.name: getattr(condition, field.name).cpu() for field in fields(condition)},
                "trace": {
                    field.name: (
                        getattr(sample.trace, field.name).cpu()
                        if torch.is_tensor(getattr(sample.trace, field.name))
                        else getattr(sample.trace, field.name)
                    )
                    for field in fields(sample.trace)
                },
                "logprobs": sample.scores.logprobs.cpu(),
                "values": sample.scores.values.cpu(),
            },
        )


def register_gr00t_rl_pipeline() -> str:
    """Idempotently register this pipeline with vLLM-Omni's registry; return its arch name.

    Called at import time (the call the spawned worker depends on) and by the rollout body for its own
    process; repeated body construction in one process must not fight the registry.
    """
    global _REGISTERED
    if not _REGISTERED:
        from vllm_omni.diffusion.registry import register_diffusion_model

        register_diffusion_model(MODEL_ARCH, __name__, Gr00tN1d7RLPipeline.__name__)
        _REGISTERED = True
    return MODEL_ARCH


class Gr00tOmniWorkerExtension(OmniIpcWorkerExtension):
    """GR00T's worker extension: the generic weight-plane half plus GR00T-specific reporting.

    vLLM-Omni composes this class into the worker by dynamic inheritance, so ``self`` is the
    ``DiffusionWorker`` and the live pipeline is ``self.model_runner.pipeline``. The generic half
    (``export_live_parameters`` / ``begin_weight_update`` / ``finish_weight_update`` / ``digest_live_parameters`` /
    ``describe_device``) reaches the weights only through the ``_live_named_parameters`` hook below.
    """

    def inference_fingerprint(self) -> dict:
        import hashlib

        from gr00t.model.modules.dit import _should_force_math_sdpa

        policy = self.model_runner.pipeline.policy
        model = policy.model
        digest = hashlib.sha256()
        dtypes: dict[str, int] = {}
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + 1
        buffers = {}
        for name, buffer in model.named_buffers():
            digest.update(name.encode())
            digest.update(buffer.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            buffers[name] = str(buffer.dtype)
        return {
            "sha256": digest.hexdigest(),
            "parameter_dtypes": dtypes,
            "buffer_dtypes": buffers,
            "sdpa_math": _should_force_math_sdpa(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }

    def _live_named_parameters(self) -> Iterable[tuple[str, torch.Tensor]]:
        head = self.model_runner.pipeline.policy.model.action_head  # type: ignore[attr-defined]
        return ((f"{ACTION_HEAD_PREFIX}{name}", p) for name, p in head.named_parameters())

    def describe_action_head(self) -> dict[str, dict[str, Any]]:
        """Live action-head parameter layout (JSON-shaped; crosses a process boundary)."""
        return {
            name: {"shape": [int(d) for d in p.shape], "dtype": str(p.dtype)}
            for name, p in self._live_named_parameters()
        }

    def describe_pipeline(self) -> dict[str, Any]:
        """What the worker actually built: pipeline class, eval image transform, step count, artifact.

        A silent fallback to the stock pipeline produces right-shaped actions with no error; this makes
        that state observable. Every value is read off the live object that runs, not off the config
        the worker was given: the processor is what runs per frame, the action head's
        ``num_inference_timesteps`` is what the Euler loop reads, and ``model_path`` is what
        ``Gr00tPolicy`` loaded from. Missing pieces report ``None`` so the body can fail with a message
        instead of an AttributeError from inside the worker.
        """
        pipeline = self.model_runner.pipeline  # type: ignore[attr-defined]
        policy = getattr(pipeline, "policy", None)
        transform = getattr(getattr(policy, "processor", None), "eval_image_transform", None)
        head = getattr(getattr(policy, "model", None), "action_head", None)
        steps = getattr(head, "num_inference_timesteps", None)
        math_effective = None
        if getattr(pipeline, "ppo", False):
            from gr00t.model.modules.dit import _should_force_math_sdpa

            math_effective = _should_force_math_sdpa()
        return {
            "class": type(pipeline).__name__,
            "module": type(pipeline).__module__,
            "is_rl_pipeline": isinstance(pipeline, Gr00tN1d7RLPipeline),
            "image_transform": getattr(transform, "name", None)
            or ("isaac-native-eval" if getattr(pipeline, "ppo", False) else None),
            "ppo": getattr(pipeline, "ppo", False),
            "source_identity": pipeline.ppo_model.ppo_identity()["source"] if getattr(pipeline, "ppo", False) else None,
            "sdpa_math_effective": math_effective,
            "num_inference_timesteps": None if steps is None else int(steps),
            "model_path": getattr(pipeline, "model_path", None),
            # A hot update must keep serving from THIS process; a rebuild must not. The pid tells them apart.
            "worker_pid": os.getpid(),
        }

    def describe_modality(self) -> dict[str, Any]:
        """The loaded checkpoint's modality contract, read from the processor the worker built."""
        policy = self.model_runner.pipeline.policy  # type: ignore[attr-defined]
        configs = policy.modality_configs
        return {
            "embodiment_tag": policy.embodiment_tag.value,
            "video_keys": list(configs["video"].modality_keys),
            "state_keys": list(configs["state"].modality_keys),
            "action_keys": list(configs["action"].modality_keys),
            "language_key": policy.language_key,
            "action_horizon": len(configs["action"].delta_indices),
            "native_action_horizon": policy.model.action_head.config.action_horizon,
            "video_steps": len(configs["video"].delta_indices),
            "state_steps": len(configs["state"].delta_indices),
        }


# Import-time registration: what makes the spawned worker know about `Gr00tN1d7RLPipeline` at all.
register_gr00t_rl_pipeline()
