# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model capability protocols (DESIGN §1.4 / §6 — model ↔ algorithm decoupling).

A model exposes a small, modality-typed capability protocol; algorithms
(`algorithm/`) are written against the protocol, never against a concrete model.
So a new model needs only an adapter implementing the protocol — no new trainer,
no new objective — and a new objective works with every model that satisfies the
protocol. This is the decoupling that removes cosmos-rl's "one trainer .py per
model" coupling (where `compute_dpo_loss` was a method *on* the model).

`FlowMatchingPolicy` is the general staged-denoising contract for diffusion/flow
objectives. `PreferenceFlowPolicy` adds only the pair-context validation required by
DPO, while `OnlineFlowPolicy` adds the final-sample operations required by NFT.
Production implementations live in `model/` (`cosmos3` for the i4-internal OmniMoT,
`hf` later); a toy fixture for tests lives in `tests/helpers/toy_model.py`.

`SupervisedFlowMatchingPolicy` is the sibling contract for a model that owns its supervised flow-matching
loss (behaviour-cloning fine-tuning, `model/gr00t/policy.py`): the objective calls `flow_matching_loss` and never
re-derives the flow-matching math.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.build import Precision
    from cosmos_framework.posttrain.roles.reward.video import VideoDecoder
    from cosmos_framework.posttrain.schema import Transition


class FlowCondition(Protocol):
    """Opaque model-owned conditioning passed back without algorithm inspection."""


FlowConditionMask = torch.Tensor | list[torch.Tensor] | None


@dataclass(frozen=True)
class PPODecisionScores:
    """Selected decisions and the complete replay's zero-contribution graph."""

    logprobs: torch.Tensor  # [B], B=1 for score_ppo_decision
    values: torch.Tensor  # [B]
    zero_loss: torch.Tensor  # scalar, retains all replayed parameter dependencies


@dataclass(frozen=True)
class PPOParameterPartition:
    """Model-owned parameter names, shared by optimizer groups and update metrics."""

    actor: tuple[str, ...]
    value: tuple[str, ...]


class DecisionPPOPolicy(Protocol):
    """Model-owned replay of a recorded sampling invocation.

    Record validation is rank-local and precedes collective-bearing scoring.
    Scalar scoring preserves the original invocation batch, then selects the recorded
    decision. Padding uses a genuine cached record and the complete zero graph.
    """

    net: torch.nn.Module
    ppo_padding: Transition | None
    ppo_parameters: PPOParameterPartition
    precision: Precision

    ppo_logprob_tolerance: float

    def ppo_identity(self) -> dict: ...
    def ppo_frozen_state(self) -> dict[str, str]: ...

    def validate_ppo_record(self, step: Transition) -> None: ...

    def score_ppo_decision(self, step: Transition) -> PPODecisionScores: ...


@runtime_checkable
class BatchedDecisionPPOPolicy(Protocol):
    """Optional replay of selected decisions in one fixed-size model forward.

    The caller supplies an ordered batch, including genuine padding records. The
    adapter selects each record's original row and preserves that order in [B]
    scores. It converges all rank-local validation and collation failures before
    entering a collective-bearing forward. ``zero_loss`` retains every replayed
    parameter dependency, including when the caller masks the entire batch.

    The adapter's identity binds this scoring layout; the unchanged-weight gate
    uses the same batch size as optimization and the declared scoring tolerance.
    """

    def score_ppo_decisions(self, steps: Sequence[Transition]) -> PPODecisionScores: ...


@dataclass(frozen=True)
class ForwardNoiseTolerance:
    """How far two forwards at UNCHANGED weights may disagree before the model counts as stochastic.

    The trainer's π_old determinism probe re-scores the first ahead chunk in ``train()`` mode and
    compares per-token logprobs against the ``eval()`` record. A token has *moved* when its logprob
    differs by more than ``per_token_nats``; the probe fails when more than ``moved_fraction`` of
    the mesh-wide compared tokens moved. ``moved_fraction == 0`` is the strict gate (equivalent to
    a max-abs bound at ``per_token_nats``).

    Validated at construction because the trainer trusts the model's declaration: a NaN in either
    field makes both ``delta > per_token_nats`` and ``fraction > moved_fraction`` false and would
    silently disable the fail-closed probe.

    Raises:
        ValueError: a value is not finite, ``per_token_nats < 0``, or ``moved_fraction`` is
            outside ``[0, 1]``.
    """

    per_token_nats: float
    moved_fraction: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.per_token_nats) and math.isfinite(self.moved_fraction)):
            raise ValueError(
                f"ForwardNoiseTolerance must be finite, got per_token_nats={self.per_token_nats!r}, "
                f"moved_fraction={self.moved_fraction!r}"
            )
        if self.per_token_nats < 0:
            raise ValueError(f"ForwardNoiseTolerance.per_token_nats must be >= 0, got {self.per_token_nats!r}")
        if not 0.0 <= self.moved_fraction <= 1.0:
            raise ValueError(f"ForwardNoiseTolerance.moved_fraction must be in [0, 1], got {self.moved_fraction!r}")


# Today's dense gate: kernel non-determinism moves bf16 logprobs by ~1e-3 while dropout at p=0.1
# moves them by O(0.1-1), so 0.05 nats on ANY token separates noise from a stochastic layer.
STRICT_FORWARD_NOISE_TOLERANCE = ForwardNoiseTolerance(per_token_nats=0.05, moved_fraction=0.0)


@dataclass(frozen=True)
class VLMPolicyOutput:
    """Causal-VLM output consumed by posttrain objectives.

    Keeping the logits in a named result prevents an objective from depending on a
    concrete Qwen/HuggingFace output dataclass while still preserving the full vocabulary.
    """

    logits: torch.Tensor  # [B,T,V]


@dataclass(frozen=True)
class SelectedSFTPolicyOutput:
    """Full-vocabulary logits for the requested supervised next-token positions.

    Rows follow ``hidden[:, :-1][valid]`` boolean-index order. The backbone still
    processes every input position and all multimodal conditioning; only the
    output-head projection is restricted. An empty local selection must retain
    its graph so a zero-weight validation lane can still participate in backward.
    """

    logits: torch.Tensor  # [N,V]


@runtime_checkable
class SelectedSFTPolicy(Protocol):
    """Optional SFT-only projection; the ordinary full-logit API stays intact."""

    sft_logits_mode: str

    def forward_sft_batch(self, batch: dict[str, Any], valid: torch.Tensor) -> SelectedSFTPolicyOutput:
        """Project the shifted valid mask [B,T-1] after forwarding the complete context."""
        ...


@dataclass(frozen=True)
class NFTVelocityPredictions:
    """Model-owned values at one NFT forward-process coordinate."""

    x_t: torch.Tensor  # [B,...]
    t_effective: torch.Tensor  # broadcastable to [B,...]
    v_policy: torch.Tensor  # [B,...]
    v_old: torch.Tensor  # [B,...]
    v_ref: torch.Tensor | None  # [B,...] | None


@dataclass(frozen=True)
class FlowMatchingLossOutput:
    """Supervised flow-matching result consumed by `algorithm/flow_bc.py`.

    A named result, like `VLMPolicyOutput`, keeps the objective off the model's raw output dict (Isaac's
    carries `action_loss`, `backbone_features`, ... that no objective should depend on).
    """

    loss: torch.Tensor  # []  masked mean over the supervised region
    action_mask: torch.Tensor  # [B, horizon, action_dim]  ones over the supervised region


@runtime_checkable
class VLMPolicy(Protocol):
    """Narrow causal-VLM capability used by post-training objectives.

    The adapter, rather than the objective, owns model-specific multimodal kwargs and device
    movement. Policy-to-rollout mapping stays on the independent model-owned ``weight_mapper`` seam.
    """

    net: torch.nn.Module

    def validate_policy_batch(self, batch: dict[str, Any]) -> None:
        """Validate rank-local model inputs before any FSDP rank enters forward."""
        ...

    def prepare_policy_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move/filter model inputs before the rank-converged pre-forward barrier."""
        ...

    def forward_policy_batch(self, batch: dict[str, Any]) -> VLMPolicyOutput:
        """Run one shaped policy batch and return logits aligned to ``input_ids``."""
        ...


@runtime_checkable
class ForwardReplayPolicy(Protocol):
    """Optional capability: the model dispatches a forward through routing recorded elsewhere.

    Owned by the model adapter because routing is a model detail (which blocks route, what an
    expert id means, how a record is laid out). The record travels in the policy batch under
    ``schema/routing.py::FORWARD_REPLAY_BATCH_KEY`` (the contract module the rollout, the packer
    and the adapter share); ``prepare_policy_batch`` is its only consumer and the only writer
    of the pin, so the ahead, probe and train forwards of one chunk dispatch identically and an
    activation-checkpoint recompute sees the same selection. SINGLE WRITER: the batch-driven pin is
    the only pin there is -- the rollout records the routing, the packer aligns it, the adapter pins
    from the batch it prepares; no trainer, objective or callback stages a pin, and every prepared
    batch rewrites it. A batch without the key runs natural; a record that does not describe the
    model or the batch fails loud in ``prepare_policy_batch``. The trainer never pins: it reads
    ``replay_pinned`` to hold the run to ``rl.routing_replay`` and to choose the determinism-probe
    tolerance, and ``replay_pinned_extent`` for telemetry.
    """

    def replay_pinned(self) -> bool:
        """Whether the most recently prepared policy batch pinned the dispatch."""
        ...

    def replay_pinned_extent(self) -> tuple[int, int]:
        """``(tokens, blocks)`` the most recently prepared policy batch pinned: the token rows that
        dispatch through the pin (every prompt + completion position, the padding excluded when the
        batch carries an ``attention_mask``) and the routed blocks per row; ``(0, 0)`` when it ran
        natural. Telemetry only."""
        ...

    def replay_identity(self) -> dict[str, Any]:
        """JSON-compatible identity of what a record pins (block count, top-k, ...), for telemetry
        and the exact-resume identity: pinned versus natural dispatch changes every gradient."""
        ...


@runtime_checkable
class ForwardNoisePolicy(Protocol):
    """Optional capability: the model declares its own eval-vs-train forward-noise tolerance.

    A model without it is held to ``STRICT_FORWARD_NOISE_TOLERANCE``. A sparse MoE relaxes the gate
    only while its routing is pinned: unpinned, top-k over near-tied router scores turns kernel
    jitter into different experts for a share of tokens, and that churn IS what the probe must
    catch, so the strict gate applies.
    """

    def forward_noise_tolerance(self, *, replay_pinned: bool) -> ForwardNoiseTolerance:
        """The tolerance for a probe run with routing pinned (``True``) or natural (``False``)."""
        ...


@runtime_checkable
class LoadBalancingPolicy(Protocol):
    """Optional capability: the model collects the routing statistics an auxiliary load-balancing loss needs.

    Owned by the model adapter because which blocks route and what they stash is a model detail. The SFT
    objective (``algorithm/sft.py``, ``moe_lbl_coeff > 0``) asks for the capability with ``isinstance`` and
    refuses a policy without it before any forward, so no objective imports a model module. The native
    Qwen3-VL-MoE adapter is the one implementation; dense and HF policies declare nothing.
    """

    def begin_lbl_collection(self) -> int:
        """Switch the per-forward stash of every adapted sparse block on; returns the number of blocks switched
        (0 for a network without one, which lets the objective refuse a coefficient the model cannot honour)."""
        ...

    def collect_lbl_metadata(self) -> list[Any]:
        """Pop the per-layer routing metadata of the last forward, in module order (popping releases the router
        graph the mean probabilities carry)."""
        ...


@runtime_checkable
class SupervisedFlowMatchingPolicy(Protocol):
    """A model that owns its supervised flow-matching loss (behaviour-cloning fine-tuning).

    Distinct from `FlowMatchingPolicy`, the staged denoiser contract: here the model computes the whole
    loss itself and the objective only normalizes it across data-parallel ranks.
    """

    net: torch.nn.Module

    def flow_matching_loss(self, inputs: dict[str, Any]) -> FlowMatchingLossOutput:
        """Run the model's own training forward on collated inputs."""
        ...


@runtime_checkable
class FlowMatchingPolicy(Protocol):
    """Capability contract for a rectified-flow / diffusion policy (DESIGN §1.4).

    The policy owns its network(s), private conditioning representation, stochastic
    inputs, and staged denoiser mechanics; algorithms only drive these primitives.
    """

    net: torch.nn.Module  # policy network (trainable)
    net_ref: torch.nn.Module | None  # frozen reference; always .eval() (None ⇒ no ref)

    def reference_semantics_identity(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible identity for non-tensor forward semantics.

        Exact-resume checkpoints already bind every reference parameter and buffer. Adapters must
        additionally bind construction/config values that affect the reference forward but are not
        tensors (for example normalization epsilon, activation choices, and attention flags).
        """
        ...

    def encode_and_condition(
        self,
        batch: dict,
        *,
        sample_encode_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, FlowCondition]:
        """Encode a batch to the clean latent `x0` + a `FlowCondition`. Owns modality
        routing (e.g. T=1 single-frame ⇒ image schedule vs video) so callers stay
        modality-agnostic. Optional per-sample encode seeds make stochastic tokenizers
        in held-out validation invariant to batch size, ordering, and data-parallel
        topology. ``None`` preserves the model's ordinary training RNG path."""
        ...

    def condition_mask(self, cond: FlowCondition) -> FlowConditionMask:
        """Return the per-sample mask whose conditioning positions contribute zero denoise error."""
        ...

    def draw_sigma_epsilon(
        self,
        x0_shape: torch.Size,
        cond: FlowCondition,
        sample_seeds: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a noise level `sigma` and noise `epsilon` for one encoded batch.

        Optional per-sample seeds make held-out objective validation invariant to
        batch size and data-parallel topology.
        """
        ...

    def prepare_denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> Any:
        """Complete all data-dependent preparation for one denoise forward."""
        ...

    def denoise_prediction(self, *, net: torch.nn.Module, prepared: Any) -> Any:
        """Run only the model forward on already-prepared inputs."""
        ...

    def finalize_denoise_error(self, *, prepared: Any, prediction: Any) -> torch.Tensor:
        """Reduce one completed prediction to per-sample fp32 errors, shape ``[B]``."""
        ...

    def denoise_error(
        self,
        *,
        net: torch.nn.Module,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> torch.Tensor:
        """Compatibility wrapper for one complete staged denoiser call."""
        ...


@runtime_checkable
class PreferenceFlowPolicy(Protocol):
    """Optional pair-validation capability required only by preference objectives.

    DPO encodes winner and loser independently, asks this capability to validate their
    opaque contexts, then derives one condition mask and one shared noise trajectory
    from the winner before driving policy/reference forwards on both sides.
    """

    def assert_same_condition_context(self, winner: FlowCondition, loser: FlowCondition) -> None:
        """Reject preference sides whose non-generated conditioning context differs."""

        ...


@runtime_checkable
class EMAFlowMatchingPolicy(Protocol):
    """Optional capability for a policy whose EMA participates in training state.

    The trainer keeps this separate from :class:`FlowMatchingPolicy` because most
    diffusion policies do not maintain EMA. An implementation owns the complete
    persisted EMA state, including any persistent buffers its parameter updater
    does not copy.
    """

    net_ema: torch.nn.Module | None

    def update_ema(self, iteration: int) -> None:
        """Advance EMA after the applied zero-based optimizer update ``iteration``."""

        ...


@runtime_checkable
class OnlineFlowPolicy(FlowMatchingPolicy, Protocol):
    """Additional capability required by final-sample online flow objectives.

    NFT owns the positive/negative direction and loss math. The model adapter owns
    only the model-specific packing, condition-aware interpolation, and policy/old/reference
    forwards needed to evaluate one forward-process point.

    ``encode_and_condition``, ``condition_mask``, and ``draw_nft_sigma_epsilon``
    are rank-local preparation methods and must not enter distributed collectives.
    The objective converges failures from those methods before any rank enters
    ``nft_velocity_predictions``, which is the first collective-bearing stage.
    An adapter that needs collective preparation must expose it as a separately
    converged stage instead of hiding it behind these methods.
    """

    net_old: torch.nn.Module | None  # periodically refreshed behavior model; always .eval() when present

    def draw_nft_sigma_epsilon(self, x0_shape: torch.Size, cond: FlowCondition) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw NFT stochastic inputs in the pinned reference recipe's RNG order."""
        ...

    def nft_velocity_predictions(
        self,
        *,
        x0: torch.Tensor,
        cond: FlowCondition,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        condition_mask: FlowConditionMask,
    ) -> NFTVelocityPredictions:
        """Return policy, moving-old, and optional frozen-reference predictions.

        ``t_effective`` is broadcastable to ``x_t`` and zero on conditioned
        positions. Old/reference forwards have no gradients; policy gradients
        remain enabled. ``v_ref`` may be absent only when the objective disables KL.
        """
        ...

    def refresh_old(self) -> None:
        """Refresh the moving NFT behavior model from the committed policy."""
        ...

    def sample_final_latents(
        self,
        model_batch: dict,
        *,
        sampler: Any,
        seeds: list[int],
        num_steps: int,
        shift: float,
        guidance: float,
    ) -> list[torch.Tensor]:
        """Generate one final flow latent per seed with an explicitly injected sampler."""
        ...

    def decode_final_latents(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Decode a uniform latent group in one model-owned batched VAE call."""
        ...

    def video_reward_decoder(self) -> VideoDecoder:
        """Return a callable that attests the decoder's output range for reward transport."""
        ...

    def update_ema(self, iteration: int) -> None:
        """Advance the model's EMA after applied update ``iteration``; no-op when disabled."""
        ...
