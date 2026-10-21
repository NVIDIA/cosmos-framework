# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T-N1.7 adapter for `LLMTrainer`: frozen backbone, trainable action head, official forward.

`flow_matching_loss` is Isaac's `Gr00tN1d7.forward` called whole, so the flow-matching math is never
re-derived here. Precision is a `Precision` preset: the default `OFFICIAL_PRECISION` is the official finetune
recipe -- fp32 parameters (backbone included) with bf16 autocast on the forward -- which is what makes the
training parity gate bitwise; `mode="fsdp"` with bf16 storage is the RLinf-style all-bf16 recipe, supported
but not gated (no official reference exists for it). `net` is the trainable scope `LLMTrainer` checkpoints
and the weight plane ships; it holds the action head only. PPO uses BF16 parameters, forward,
gradient reduction and publication for its joint action/value scope.

Isaac-GR00T is imported inside functions only (`tests/test_model_gr00t.py` asserts this in a subprocess).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from cosmos_framework.posttrain.config.build import Precision
from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.model.parallelize import parallelize
from cosmos_framework.posttrain.model.precision import cast_floating_buffers, compute_context, fsdp_param_dtype
from cosmos_framework.posttrain.model.protocol import FlowMatchingLossOutput, PPODecisionScores, PPOParameterPartition
from cosmos_framework.posttrain.model.specs import ShardPlan

if TYPE_CHECKING:
    from cosmos_framework.posttrain.model.gr00t.ppo import (
        Gr00tPPOCondition,
        Gr00tPPOSample,
        Gr00tPPOScores,
        Gr00tPPOTrace,
    )
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.schema import Transition

OFFICIAL_PRECISION = Precision(
    master="float32", param="bfloat16", reduce="float32", transfer="bfloat16", mode="autocast"
)
"""The official finetune recipe: fp32 parameters (backbone included), bf16 autocast on the forward."""

BF16_PRECISION = Precision(master="bfloat16", param="bfloat16", reduce="bfloat16", transfer="bfloat16", mode="fsdp")
"""The all-bf16 recipe: bf16 parameters (`load_bf16` for the backbone, the head cast), FSDP2 parameter cast, bf16
gradient reduction, and a wire dtype equal to storage. The same PRECISION recipe as RLinf's GR00T PPO
(`rollout.precision: ${actor.model.precision}`); it aligns nothing else -- not the sampler, the dropout policy or
the objective. Documented, not gated: `operator GR00T training documentation` records 9/539 rows bitwise against its own
single-GPU golden at dp_shard=4."""

PRECISION_PRESETS: dict[str, Precision] = {"autocast": OFFICIAL_PRECISION, "fsdp": BF16_PRECISION}
"""The `--precision-mode` vocabulary of the GR00T test launchers, keyed the way `Precision.mode` names them."""


class Gr00tHeads(nn.Module):
    """`LLMTrainer`'s `.net`: the trainable surface, shared (not copied) with the parent `Gr00tN1d7`.

    The attribute name is the wire prefix: `state_dict()` keys are `action_head.*`, matching
    `model/gr00t/serving/body.py::ACTION_HEAD_PREFIX` and the names HF's decay partition sees.

    The trainable scope is Isaac's `FinetuneConfig` default (projector + diffusion model, both inside
    `action_head`; LLM and vision frozen); `OptimizerConfig.train_keys` can narrow it further. Widening it
    to the backbone would change this wire prefix, the shard plan and the official-recipe reference at once.
    """

    def __init__(self, action_head: nn.Module) -> None:
        super().__init__()
        self.action_head = action_head


def build_official_gr00t_model(
    checkpoint: str,
    *,
    save_cfg_dir: Any,
    revision: str = "",
    precision: Precision = OFFICIAL_PRECISION,
    local_files_only: bool = False,
    ppo: bool = False,
) -> Any:
    """Build the official finetune model; PPO separates local loading from rank-zero publication.

    Flow-BC uses the official pipeline's `_create_model` unchanged. PPO uses its public HF
    loader and the same strict loading checks, avoiding `_create_model`'s internal barrier:
    all ranks must report local model/processor loading failures before any collective phase.

    Only the fields `_create_model` forwards to `from_pretrained` are set. `state_dropout_prob` comes
    from `FinetuneConfig` (0.2) because `Gr00tN1d7Config` defaults to the pretrain value (0.8).
    `precision.master` selects the storage dtype: fp32 (`load_bf16=False`, trainable backbone parameters
    kept fp32) or bf16 (`load_bf16=True` for the backbone, and the action head -- which Isaac always builds
    in fp32 -- cast to bf16, the way RLinf loads the whole model with `torch_dtype=bfloat16`). `save_cfg_dir`
    must exist: flow-BC publishes `final_model_config.json` here; the PPO source owner
    publishes it only after every rank completes model/processor loading and source checks.
    """
    if precision.master not in ("float32", "bfloat16"):
        raise ValueError(
            f"GR00T parameters are stored in float32 or bfloat16 (Isaac's load_bf16 switch), not {precision.master!r}"
        )
    from pathlib import Path

    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.finetune_config import FinetuneConfig
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    ft = FinetuneConfig(base_model_path=checkpoint, dataset_path="", embodiment_tag="")
    config = get_default_config()
    config.model.tune_llm = ft.tune_llm
    config.model.tune_visual = ft.tune_visual
    config.model.tune_projector = ft.tune_projector
    config.model.tune_diffusion_model = ft.tune_diffusion_model
    config.model.state_dropout_prob = ft.state_dropout_prob
    config.model.load_bf16 = precision.master == "bfloat16"
    config.model.backbone_trainable_params_fp32 = precision.master == "float32"
    config.model.model_revision = revision or None
    config.training.transformers_local_files_only = local_files_only
    config.training.start_from_checkpoint = checkpoint
    # Omni constructs pipelines under a CUDA default-device context. Nested HF
    # backbone loading must start on CPU: otherwise Accelerate dispatches the
    # inner model before its meta tensors have been materialized.
    with torch.device("cpu"):
        pipeline = Gr00tN1d7Pipeline(config, Path(save_cfg_dir))
        model = _load_local_ppo_model(pipeline) if ppo else pipeline._create_model()
    if precision.master == "bfloat16":
        model.action_head.to(torch.bfloat16)  # `load_bf16` reaches the backbone only
    return model


def _load_local_ppo_model(pipeline: Any) -> Any:
    """The load/check portion of Isaac 51d4c89f's setup.py::_create_model.

    Isaac does not expose that portion separately from its rank-zero write/barrier.
    Preserve the public AutoModel loader, official kwargs and optional-mask semantics;
    the source owner coordinates errors and publishes the configuration afterwards.
    """
    from transformers import AutoModel

    config = pipeline.config.model
    checkpoint = pipeline.config.training.start_from_checkpoint
    model = None
    try:
        model, loading_info = AutoModel.from_pretrained(
            checkpoint,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vlln=config.tune_vlln,
            state_dropout_prob=config.state_dropout_prob,
            backbone_trainable_params_fp32=config.backbone_trainable_params_fp32,
            load_bf16=config.load_bf16,
            transformers_loading_kwargs=pipeline.transformers_loading_kwargs,
            output_loading_info=True,
            **pipeline.transformers_loading_kwargs,
        )
        missing = loading_info.get("missing_keys", [])
        if any("mask_token" in key for key in missing) and model.action_head.mask_token is not None:
            with torch.no_grad():
                model.action_head.mask_token.data.copy_(0.02 * torch.randn_like(model.action_head.mask_token))
        other_missing = [key for key in missing if "mask_token" not in key]
        unexpected = loading_info.get("unexpected_keys", [])
        mismatched = loading_info.get("mismatched_keys", [])
        errors = []
        for label, keys in (("Missing", other_missing), ("Unexpected", unexpected), ("Mismatched", mismatched)):
            if keys:
                errors.append(f"{label} keys ({len(keys)}): {keys}")
        if errors:
            raise RuntimeError(f"Checkpoint weight mismatch for {checkpoint}:\n" + "\n".join(errors))
        return model
    finally:
        model = None


def load_processor(checkpoint: str, *, model_name: str | None = None) -> Any:
    """The checkpoint's `Gr00tN1d7Processor`, whose `.collator` the flow-BC objective collates with."""
    # Omni registers a different processor under the same HF name. Select the
    # training adapter explicitly so importing a serving worker cannot replace it.
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    if model_name is None:
        return Gr00tN1d7Processor.from_pretrained(checkpoint)
    return Gr00tN1d7Processor.from_pretrained(
        checkpoint,
        model_name=model_name,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
    )


def gr00t_shard_plan(precision: Precision) -> ShardPlan:
    """FSDP2 plan: shard the DiT / self-attention `BasicTransformerBlock`s, root-wrap the rest. `param_dtype` is
    what the wrap delivers (`model/precision.py::fsdp_param_dtype`): the compute dtype in fsdp mode, None in
    autocast mode, where bf16 compute comes from the autocast context and the wrap casts nothing."""
    from gr00t.model.modules.dit import BasicTransformerBlock

    return ShardPlan(
        block_types=(BasicTransformerBlock,),
        param_dtype=fsdp_param_dtype(precision.mode, getattr(torch, precision.param)),
        reduce_dtype=getattr(torch, precision.reduce),
    )


class Gr00tN1d7TrainPolicy:
    """Duck-typed into `LLMTrainer` like `model/cosmos3.py::Cosmos3FlowPolicy`: the trainer uses `.net`,
    the objective calls `flow_matching_loss`."""

    def __init__(
        self,
        config: Gr00tN1d7ModelConfig,
        *,
        device: Any = "cuda",
        parallel_dims: "ParallelDims | None" = None,
        save_cfg_dir: Any,
        precision: Precision | None = None,
        ppo: bool = False,
    ) -> None:
        self.config = config
        self._ppo_enabled = ppo
        self.precision = precision or (BF16_PRECISION if ppo else OFFICIAL_PRECISION)
        if ppo and self.precision != BF16_PRECISION:
            raise ValueError("GR00T PPO requires BF16_PRECISION; other PPO precision settings are unsupported")
        if ppo and os.environ.get("GR00T_DIT_SDPA_MODE") != "math":
            raise ValueError("GR00T PPO requires GR00T_DIT_SDPA_MODE=math before model construction")
        # The official recipe runs under TrainingArguments(tf32=True); set both switches explicitly.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if ppo:
            from cosmos_framework.posttrain.model.gr00t.source import Gr00tPPOSource, load_ppo_source

            def load(source: Gr00tPPOSource) -> tuple[Any, Any]:
                model = None
                try:
                    model = build_official_gr00t_model(
                        str(source.checkpoint),
                        save_cfg_dir=save_cfg_dir,
                        # The root is already local. This revision pins the nested
                        # backbone config and collator read by the official pipeline.
                        revision=source.backbone_revision,
                        precision=self.precision,
                        local_files_only=True,
                        ppo=True,
                    )
                    processor = load_processor(str(source.checkpoint), model_name=str(source.processor))
                    return model, processor
                finally:
                    model = None

            def publish(loaded: tuple[Any, Any]) -> None:
                from pathlib import Path

                from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes

                atomic_write_bytes(
                    str(Path(save_cfg_dir) / "final_model_config.json"), loaded[0].config.to_filtered_json().encode()
                )

            (self.gr00t, self.processor), self._ppo_source_identity = load_ppo_source(config, load, publish=publish)
        else:
            self.gr00t = build_official_gr00t_model(
                config.checkpoint_path,
                save_cfg_dir=save_cfg_dir,
                revision=config.revision,
                precision=self.precision,
            )
        if config.backbone_buffer_dtype is not None:
            # Match Module.to's buffer bindings without casting trainable parameters or alias attributes.
            cast_floating_buffers(self.gr00t.backbone, getattr(torch, config.backbone_buffer_dtype))
        if ppo:
            from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOHeads
            from cosmos_framework.posttrain.model.ppo import prepare_deterministic_ppo_update

            self.net = Gr00tPPOHeads(
                self.gr00t.action_head,
                score_horizon=config.execution_horizon,
            )
            prepare_deterministic_ppo_update(self.net)
            self.ppo_parameters = PPOParameterPartition(
                actor=tuple(
                    name
                    for name, parameter in self.net.named_parameters()
                    if parameter.requires_grad and not name.startswith("action_head.value_head.")
                ),
                value=tuple(
                    name
                    for name, parameter in self.net.named_parameters()
                    if parameter.requires_grad and name.startswith("action_head.value_head.")
                ),
            )
        else:
            self.net = Gr00tHeads(self.gr00t.action_head)
        self.gr00t.to(device)
        self.backbone = self.gr00t.backbone
        if ppo:
            # PPO enters the action/value container's public forward; all trainable transforms
            # and the newly bound value head belong to that FSDP root.
            parallelize(self.net, gr00t_shard_plan(self.precision), parallel_dims)
        else:
            # Wrap `action_head`, not the container: FSDP2 all-gathers in the wrapped module's
            # pre-forward hook, and the official flow-BC forward calls `action_head(...)` directly.
            parallelize(self.net.action_head, gr00t_shard_plan(self.precision), parallel_dims)
        # Built once; `flow_matching_loss` re-enters it. The device type follows `device`, so a GPU gets
        # `torch.autocast("cuda", ...)` as before while the same code is observable on CPU in tests.
        self.compute_ctx = compute_context(
            self.precision.mode, getattr(torch, self.precision.param), torch.device(device).type
        )

    ppo_logprob_tolerance: float = 0.02

    def ppo_identity(self) -> dict:
        return {**self.config.identity(), "source": dict(self._ppo_source_identity)}

    def ppo_frozen_state(self) -> dict[str, str]:
        if self.config.backbone_buffer_dtype is None:
            return {}
        import hashlib

        from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest

        digest = hashlib.sha256()
        for kind, tensors in (
            ("parameter", self.backbone.named_parameters()),
            ("buffer", self.backbone.named_buffers()),
        ):
            for name, tensor in tensors:
                digest.update(canonical_digest((kind, name, str(tensor.dtype), tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return {"backbone": digest.hexdigest()}

    def flow_matching_loss(self, inputs: dict) -> FlowMatchingLossOutput:
        """Isaac's `Gr00tN1d7.forward` under the preset's compute context, narrowed to the loss and the
        supervision mask (its `action_loss`, `backbone_features` and `state_features` are dropped).

        The mode is forwarded because HF calls `train()` on the whole model and the official forward's
        `set_frozen_modules_to_eval_mode` branches on it, while `LLMTrainer` only touches `.net`.
        """
        if self._ppo_enabled:
            raise RuntimeError("flow-BC requires a policy constructed without the PPO capability")
        self.gr00t.train(self.net.training)
        if self.precision.mode == "fsdp":
            # The parameter-cast mode: floating inputs follow the parameter dtype, as FSDP2's
            # `cast_forward_inputs` does for the wrapped head; unwrapped (one GPU) nothing else casts them.
            dtype = getattr(torch, self.precision.param)
            inputs = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in inputs.items()}
        with self.compute_ctx:
            out = self.gr00t(inputs)
        return FlowMatchingLossOutput(loss=out["loss"], action_mask=out["action_mask"])

    def ppo_condition(self, inputs: dict, *, native_ode: bool = False) -> Gr00tPPOCondition:
        """Prepare only frozen backbone outputs; the action/value forward owns trainable features."""
        if not self._ppo_enabled:
            raise RuntimeError("GR00T PPO was not enabled when constructing this model")
        from contextlib import nullcontext

        from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOCondition, native_ode_backbone_buffers

        self.backbone.eval()
        buffers = native_ode_backbone_buffers(self.backbone) if native_ode else nullcontext()
        with torch.no_grad(), self.compute_ctx, buffers:
            backbone_inputs, action_inputs = self.gr00t.prepare_input(dict(inputs))
            backbone = self.backbone(backbone_inputs)
        return Gr00tPPOCondition(
            backbone.backbone_features,
            backbone.image_mask,
            backbone.backbone_attention_mask,
            action_inputs.state,
            action_inputs.embodiment_id,
        )

    def ppo_score(self, condition: Gr00tPPOCondition, trace: Gr00tPPOTrace) -> Gr00tPPOScores:
        """Score the selected transition and state value through the FSDP public forward."""
        if not self._ppo_enabled:
            raise RuntimeError("GR00T PPO was not enabled when constructing this model")
        with self.compute_ctx:
            return self.net(condition, trace)

    def validate_ppo_record(self, step: Transition) -> None:
        from cosmos_framework.posttrain.model.gr00t.ppo import validate_ppo_decision

        validate_ppo_decision(step, score_horizon=self.config.execution_horizon)

    def score_ppo_decision(self, step: Transition) -> PPODecisionScores:
        from cosmos_framework.posttrain.model.gr00t.ppo import CONDITION_KEY, TRACE_KEY, to_device

        device = next(self.net.parameters()).device
        scores = self.ppo_score(
            to_device(step.obs.representations[CONDITION_KEY], device),
            to_device(step.policy_info[TRACE_KEY], device),
        )
        row = step.policy_info["sample_index"]
        return PPODecisionScores(
            scores.logprobs[row : row + 1],
            scores.values[row : row + 1],
            (scores.logprobs.sum() + scores.values.sum()) * 0,
        )

    def ppo_sample(self, condition: Gr00tPPOCondition, *, seed: int) -> Gr00tPPOSample:
        """Sample one fixed invocation batch without changing the process RNG streams."""
        if not self._ppo_enabled:
            raise RuntimeError("GR00T PPO was not enabled when constructing this model")
        with torch.no_grad(), self.compute_ctx:
            return self.net(condition, seed=seed)

    def ppo_ode_actions(self, condition: Gr00tPPOCondition, *, seed: int, native: bool = False) -> torch.Tensor:
        """Evaluate an unsharded rollout head and restore every module's previous mode."""
        modes = [(module, module.training) for module in self.net.modules()]
        try:
            self.net.eval()
            with torch.no_grad(), self.compute_ctx:
                return self.net.ode_actions(condition, seed=seed, native=native)
        finally:
            for module, training in modes:
                module.training = training
