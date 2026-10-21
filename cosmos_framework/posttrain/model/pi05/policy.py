# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""RLinf OpenPI model adapter over the existing decision PPO and FSDP contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from traceback import TracebackException, clear_frames
from typing import TYPE_CHECKING

import torch
from transformers.models.gemma.modeling_gemma import GemmaMLP

from cosmos_framework.posttrain.checkpoint.provenance import load_with_stable_provenance
from cosmos_framework.posttrain.config.build import Precision
from cosmos_framework.posttrain.model.parallelize import parallelize
from cosmos_framework.posttrain.model.pi05.compat import Pi05GemmaAttention
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
from cosmos_framework.posttrain.model.pi05.ppo import (
    Pi05Heads,
    Pi05PPOCondition,
    Pi05PPOSample,
    Pi05PPOScores,
    Pi05PPOTrace,
    validate_trace,
)
from cosmos_framework.posttrain.model.pi05.processing import Pi05Processor, Pi05ReplayInputs
from cosmos_framework.posttrain.model.pi05.records import (
    INPUTS_KEY,
    TRACE_KEY,
    map_record,
    pack_ppo_decisions,
    validate_ppo_decision,
)
from cosmos_framework.posttrain.model.pi05.source import Pi05Source, load_model_weights, model_identity
from cosmos_framework.posttrain.model.ppo import prepare_deterministic_ppo_forward
from cosmos_framework.posttrain.model.protocol import PPODecisionScores, PPOParameterPartition
from cosmos_framework.posttrain.model.specs import ShardPlan
from cosmos_framework.posttrain.schema import Transition
from cosmos_framework.posttrain.utils.dist import run_rank_converged

if TYPE_CHECKING:
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims


# No blanket parameter/input/transfer cast: OpenPI owns FP32 norms and projections
# alongside BF16 expert weights. FSDP and the weight plane preserve each dtype.
NATIVE_PRECISION = Precision(master=None, param=None, reduce=None, transfer=None)


def _clear_exception_frames(error: BaseException) -> None:
    """Preserve diagnostics without retaining tensor-owning forward closures."""
    original = "".join(TracebackException.from_exception(error, capture_locals=False).format(chain=True))
    error.add_note("Original exception stack before releasing retained frames:\n" + original)
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        clear_frames(current.__traceback__)
        # Torch's forward frame can retain its function's closure even after
        # clear_frames; that closure owns the packed inputs through ``args``.
        current.__traceback__ = None
        pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)


class Pi05TrainPolicy:
    ppo_logprob_tolerance: float = 1e-5
    ppo_padding: Transition | None = None

    def __init__(
        self, config: Pi05ModelConfig, *, device: torch.device | str = "cuda", parallel_dims: ParallelDims | None = None
    ) -> None:
        self.config = config
        self.precision = NATIVE_PRECISION
        # OpenPI enables TF32 for its remaining FP32 projections; both learner
        # and Omni construct this same policy before any model forward.
        torch.set_float32_matmul_precision("high")
        source = run_rank_converged(lambda: Pi05Source.resolve(config), "pi05 source resolution")
        resolved = replace(
            config,
            checkpoint_path=str(source.checkpoint.parent),
            tokenizer_path=str(source.tokenizer),
            norm_stats_path=str(source.normalization),
        )

        def load() -> tuple[Pi05Model, Pi05Processor]:
            # Construction and missing value-head initialization use an isolated RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                model = Pi05Model(resolved, device="cpu")
                load_model_weights(model, source)
                model.apply_reference_precision()
            return model, Pi05Processor(resolved)

        def materialize() -> None:
            self.model.to(device=device)
            self.net = Pi05Heads(self.model)
            prepare_deterministic_ppo_forward(self.model)
            self.ppo_parameters = PPOParameterPartition(
                actor=tuple(
                    name
                    for name, parameter in self.net.named_parameters()
                    if parameter.requires_grad and not name.startswith("value_head.")
                ),
                value=tuple(
                    name
                    for name, parameter in self.net.named_parameters()
                    if parameter.requires_grad and name.startswith("value_head.")
                ),
            )

        self.model, self.net, self.processor = None, None, None
        try:
            (self.model, self.processor), self.source_identity = load_with_stable_provenance(
                str(source.checkpoint.parent),
                lambda: run_rank_converged(load, "pi05 model loading"),
                provenance=lambda _: source.identity(),
            )
            run_rank_converged(materialize, "pi05 device materialization")
            parallelize(self.net, ShardPlan((Pi05GemmaAttention, GemmaMLP), None, None), parallel_dims)
        except BaseException as error:
            # Drop the borrowed heads before their full-model owner. Completed
            # loader/materialization frames can otherwise retain either tree.
            self.net, self.model, self.processor = None, None, None
            _clear_exception_frames(error)
            raise error

    def ppo_identity(self) -> dict:
        return model_identity(self.config, self.source_identity)

    def ppo_frozen_state(self) -> dict[str, str]:
        digest = hashlib.sha256()
        for name, tensor in (
            *self.model.paligemma_with_expert.paligemma.named_parameters(),
            *self.model.paligemma_with_expert.paligemma.named_buffers(),
        ):
            digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return {"frozen_vlm": digest.hexdigest()}

    def ppo_condition(self, observations: list[dict]) -> Pi05PPOCondition:
        return self.condition_from_inputs(self.processor.encode(observations))

    def condition_from_inputs(self, inputs: Pi05ReplayInputs) -> Pi05PPOCondition:
        device = next(self.model.parameters()).device
        return self.model.prepare_condition(self.processor.materialize(inputs, device=device))

    def validate_ppo_record(self, step: Transition) -> None:
        validate_ppo_decision(step, self.config)

    def score_ppo_decision(self, step: Transition) -> PPODecisionScores:
        device = next(self.net.parameters()).device

        def prepare() -> tuple[Pi05PPOCondition, Pi05PPOTrace]:
            self.validate_ppo_record(step)
            return (
                self.condition_from_inputs(step.obs.representations[INPUTS_KEY]),
                map_record(step.policy_info[TRACE_KEY], lambda tensor: tensor.to(device=device)),
            )

        condition = trace = scores = None
        try:
            condition, trace = run_rank_converged(prepare, "pi05 scalar decision preparation")
            scores = self.net(condition, trace)
            row = step.policy_info["sample_index"]
            return PPODecisionScores(
                scores.logprobs[row : row + 1],
                scores.values[row : row + 1],
                (scores.logprobs.sum() + scores.values.sum()) * 0,
            )
        except BaseException as error:
            condition = trace = scores = None
            _clear_exception_frames(error)
            raise error

    def score_ppo_decisions(self, steps: Sequence[Transition]) -> PPODecisionScores:
        def prepare() -> tuple[Pi05PPOCondition, Pi05PPOTrace]:
            if not steps:
                raise ValueError("pi05 PPO scoring requires a nonempty decision batch")
            for step in steps:
                self.validate_ppo_record(step)
            inputs, trace = pack_ppo_decisions(steps, device=next(self.net.parameters()).device)
            return self.condition_from_inputs(inputs), trace

        condition = trace = scores = None
        try:
            condition, trace = run_rank_converged(prepare, "pi05 batched decision preparation")
            scores = self.net(condition, trace)
            return PPODecisionScores(scores.logprobs, scores.values, (scores.logprobs.sum() + scores.values.sum()) * 0)
        except BaseException as error:
            condition = trace = scores = None
            _clear_exception_frames(error)
            raise error

    def ppo_score(self, condition: Pi05PPOCondition, trace: Pi05PPOTrace) -> Pi05PPOScores:
        run_rank_converged(lambda: validate_trace(condition, trace, self.config), "pi05 trace validation")
        return self.net(condition, trace)

    @torch.no_grad()
    def ppo_sample(self, condition: Pi05PPOCondition, *, seed: int) -> Pi05PPOSample:
        return self.net(condition, seed=seed)

    @torch.no_grad()
    def ppo_ode_actions(self, condition: Pi05PPOCondition, *, seed: int) -> torch.Tensor:
        return self.net.ode_actions(condition, seed=seed)
