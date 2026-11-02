# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unified Sample / Trajectory data model (DESIGN §1.4).

This is the foundation everything operates on. The schema here is the *general*
one (shaped to the final form); v0 only fills the text+image subset.

Discipline (DESIGN §1.4):
- a Transition may carry BOTH `reward` (RL signal) and `targets` (supervised
  signal); at least one. v0 uses only `reward`.
- reward and targets channels stay isolated (combiner only touches reward;
  supervised loss only touches targets).
- flat single-turn is near-zero overhead; nesting (sub_trajectories) is v2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

# `schema/` is the ONE layer that must not import torch at runtime — the public api boundary
# asserts it (`tests/test_api_public.py::test_public_api_import_stays_free_of_training_dependencies`).
# Tensor-typed fields therefore use a TYPE_CHECKING import + string annotations, the idiom already in
# `algorithm/primitives/objective.py` and `utils/dist.py`. `from __future__ import annotations` above makes every
# annotation lazy, so the dataclass machinery never evaluates these names.
if TYPE_CHECKING:
    import torch

FINAL_REWARD_KEY = "final"
WFM_CONDITIONING_KEY = "wfm_conditioning"
"""Key into ``Transition.obs.representations`` holding the group's shared WFM conditioning."""
FINISH_REASON_KEY = "finish_reason"
"""``Transition.policy_info`` key: how the rollout engine ended this completion.

vLLM's ``CompletionOutput.finish_reason`` vocabulary, stamped by ``roles/rollout/vllm.py`` on every
training and validation completion: ``"stop"`` (a stop token or stop string ended the generation),
``"length"`` (the engine cut it at its token budget, ``rollout.max_tokens`` or the model length) or
another engine verdict such as ``"abort"``. ``None`` or an absent key means the producer recorded
none (the HF and local rollouts, batches from before the key existed); a reader must treat that as
"not known to be truncated", never as "stopped".
"""
FINISH_REASON_STOP = "stop"
FINISH_REASON_LENGTH = "length"
ROLLOUT_TOPK_IDS_KEY = "rollout_topk_ids"
"""``Transition.policy_info`` key: the SAMPLER's top-k token ids per generated token, ``[T, k]``.

Stamped by ``roles/rollout/vllm.py`` only under ``rl.score_centering.enabled`` (the engine is asked
for ``SamplingParams(logprobs=k)`` in ``processed_logprobs`` mode). Row ``t`` is the head of the
distribution completion token ``t`` was DRAWN from, ranked by the sampler's own probability
(rank 1 first). Integer tensor; the head is never renormalised and the sampled token is NOT forced
into it (``score_centering`` keeps its logprob in ``rollout_logprobs`` instead).
"""
ROLLOUT_TOPK_LOGPROBS_KEY = "rollout_topk_logprobs"
"""``Transition.policy_info`` key: the sampler's log-probabilities at ``ROLLOUT_TOPK_IDS_KEY``, ``[T, k]``.

Same rows and columns as the ids; float32 on the wire -- never float16: a rounded head makes q != p for an
identical sampler and trainer and injects a correction with no cause (k = 128 x 4 B per token stays bounded);
the consumer reads it as float32. Values are ``log q`` of the distribution the token was sampled from — the
engine's PROCESSED logprobs (after temperature), not its raw temperature-1 ones.
"""


def is_length_verdict(finish_reason: str | None) -> bool:
    """Whether one recorded ``finish_reason`` value is the exact ``"length"`` verdict.

    The single truncation comparison every reader delegates to (``is_length_truncated`` for a
    ``policy_info`` mapping, ``RewardCandidate.is_length_truncated`` for a prepared reward candidate).
    Only ``"length"`` answers True; ``"stop"``, another engine string and ``None`` (no verdict
    recorded) all read as "not known to be truncated".
    """
    return finish_reason == FINISH_REASON_LENGTH


def is_length_truncated(policy_info: Mapping[str, Any]) -> bool:
    """Whether the producer recorded that the engine cut this completion at its token budget."""
    return is_length_verdict(policy_info.get(FINISH_REASON_KEY))


class Status(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class ObsBundle:
    """Multi-representation observation (DESIGN §1.4).

    v0: store everything inline in `representations` (e.g. {"prompt": ..., "image": ...}).
    TODO(v1+, §3.4): large fields (video/image) become lazy *handles* (metadata +
    remote addressing) pulled on demand; small fields stay inline.

    `id` is a globally-unique per-prompt tag stamped by the data role (env `reset`; deterministic,
    shard-scoped — see `roles/env/dataset.py::_stamp_prompt_id`). It is the GRPO `group_id`
    namespace: the rollout keys `Trajectory.group_id` off `obs.id` so completions of distinct
    prompts never share a group when fan-in concatenates multiple rollout replicas' outputs (a
    batch-local index would collide across replicas). Empty when a prompt is built outside an env.
    """

    representations: dict[str, Any] = field(default_factory=dict)  # raw | latent | features
    modality_tags: tuple[str, ...] = ()
    id: str = ""  # globally-unique per-prompt id (data role stamps a deterministic id); GRPO group_id namespace


@dataclass(frozen=True)
class SampleSource:
    """Stable decision identity, separate from its episode and transport fragment.

    The transition owns the absolute decision index; its trajectory owns the session.
    These IDs survive projection into one-decision scoring or training views.
    """

    sample_id: str
    episode_id: str
    fragment_id: str

    def __post_init__(self) -> None:
        for name in ("sample_id", "episode_id", "fragment_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"sample source {name} must be a nonempty string")


@dataclass(frozen=True)
class EnvironmentFeedback:
    """Raw control-step rewards, endings and the actual environment execution.

    Recipe-transformed rewards live on ``Transition.reward``. The generic interaction
    layer records these raw facts and does not apply aggregation or discounting.
    Optional control endings belong to chunk sessions advertising that detail;
    fixed-duration episodes may continue executing after an individual ending.
    """

    control_rewards: tuple[float, ...]
    executed_steps: int
    control_terminated: tuple[bool, ...] | None = None
    control_truncated: tuple[bool, ...] | None = None
    # A fixed-duration session may keep executing after a task termination. This
    # marks its physical episode boundary, independently of the raw task facts.
    episode_complete: bool | None = None

    def __post_init__(self) -> None:
        if type(self.executed_steps) is not int or self.executed_steps <= 0:
            raise ValueError("environment feedback executed_steps must be a positive integer")
        if not isinstance(self.control_rewards, tuple) or len(self.control_rewards) != self.executed_steps:
            raise ValueError("environment feedback control_rewards must contain one reward per executed step")
        for index, reward in enumerate(self.control_rewards):
            if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(reward):
                raise ValueError(f"environment feedback control_rewards[{index}] must be a finite real")
        if (self.control_terminated is None) != (self.control_truncated is None):
            raise ValueError("environment feedback requires both control_terminated and control_truncated")
        for name in ("control_terminated", "control_truncated"):
            flags = getattr(self, name)
            if flags is not None:
                if not isinstance(flags, tuple) or len(flags) != self.executed_steps:
                    raise ValueError(f"environment feedback {name} must contain one flag per executed step")
                for index, flag in enumerate(flags):
                    if type(flag) is not bool:
                        raise ValueError(f"environment feedback {name}[{index}] must be a boolean")
        if self.episode_complete is not None and type(self.episode_complete) is not bool:
            raise ValueError("environment feedback episode_complete must be a boolean")


# ``SFTTarget.supervise``: which assistant turns of a conversation carry loss. The SFT packer reads it; the JSONL / factory
# loader (``data/datasets/sft.py::normalize_row``) refuses any other value in a source row.
SFT_SUPERVISE_ALL = "all_assistant"  # every assistant body and its turn-ending token (the default; today's behaviour)
SFT_SUPERVISE_LAST = "last_assistant"  # the LAST assistant turn only; earlier assistant turns are context at loss 0
SFT_SUPERVISE_VALUES = (SFT_SUPERVISE_ALL, SFT_SUPERVISE_LAST)
# ``SFTTarget.opener``: whose token the ``<think>`` opener of a supervised assistant turn is (row rule R4 of the lab's brain
# program). The exporter records its per-row choice as ``meta.opener``; the loader requires the row field to
# agree with it, the packer reads the field.
SFT_OPENER_SELF = (
    "self"  # the opener is the student's first supervised token (the default; the dsh serve prefills nothing)
)
SFT_OPENER_PREFILLED = (
    "prefilled"  # the serve prefills the opener token itself: no loss on it, in every supervised block
)
SFT_OPENER_VALUES = (SFT_OPENER_SELF, SFT_OPENER_PREFILLED)


@dataclass
class SFTTarget:
    """Full-conversation supervised signal carried in ``Transition.targets``.

    Messages retain HF text/image/video content, reasoning and tool-call metadata. The
    weight-free SFT packer owns rendering, tokenization and assistant-only loss masks.

    ``supervise`` selects the assistant turns that carry loss: ``"all_assistant"`` (the default) supervises every
    assistant body and its ``<|im_end|>``; ``"last_assistant"`` supervises the LAST assistant turn only (its body
    through ``<|im_end|>``) and leaves every earlier assistant turn in the sequence at loss 0, like the observations
    around it. A prefix row (prompt = the served prefix of one model call, completion = the student's own turn at that
    call) is a ``last_assistant`` row. Any other value is refused here and by the source loader.

    ``opener`` says whose token the ``<think>`` opener of a supervised assistant turn is: ``"self"`` (the default: the
    student's first supervised token, as the dsh serve prefills nothing) or ``"prefilled"`` (the serve prefills the opener
    token itself, ``<|im_start|>assistant\n<think>`` with no newline after it: the packer masks the first ``<think>`` of
    every supervised assistant block of the row, the rule the run-level ``mask_think_opener`` applies to every row, and the
    supervised target begins with the newline that opens the reasoning text). Per row because the exporter writes the
    prefilled slice into the same row files as the self rows (row rule R4). Any other value is refused.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    supervise: str = SFT_SUPERVISE_ALL
    opener: str = SFT_OPENER_SELF

    def __post_init__(self) -> None:
        if self.supervise not in SFT_SUPERVISE_VALUES:
            raise ValueError(f"SFTTarget.supervise must be one of {list(SFT_SUPERVISE_VALUES)}, got {self.supervise!r}")
        if self.opener not in SFT_OPENER_VALUES:
            raise ValueError(f"SFTTarget.opener must be one of {list(SFT_OPENER_VALUES)}, got {self.opener!r}")


@dataclass
class Transition:
    """One step's data. `advantage` is the algorithm annotation the policy-gradient surrogate reads.

    THE ADVANTAGE CHANNEL. `advantage` holds EITHER a scalar (one number for the whole transition,
    broadcast over its completion tokens) OR a per-token 1-D tensor of length == the transition's
    completion-token count — the count the objective selects with `logprob_masks[:, :-1]`.

    Per-token is the GENERAL type and the scalar is the degenerate case, not the other way round.
    GRPO's group-relative estimate happens to be one number per sample, which is why this field was
    born a `float`; the shared policy-gradient surrogate already consumes a per-token advantage
    tensor. OPD's
    advantage is genuinely per token (`-kl_penalty_coef · D_t` varies with the teacher's
    disagreement at each position), and PPO's GAE uses the same full-completion width `[T]`.

    A tensor advantage rides the metadata/bulk seam like any other tensor field: `comm/codec.py`
    externalizes it to the producer's `BulkStore` rather than shipping it inline through the
    control plane.
    """

    obs: ObsBundle
    action: Any = None  # token / denoise step / control action
    next_obs: Optional[ObsBundle] = None  # actual next condition, captured before environment reset
    policy_info: dict[str, Any] = field(default_factory=dict)  # logprob, dist_params?, weight_version
    # PPO also carries per-action process rewards in this mapping.
    reward: Optional[dict[str, Any]] = None  # named components (§1.5); filled by RewardProvider/Combiner
    targets: Any = None  # supervised signal: SFTTarget, Preference, TeacherSignal, ...
    # Filled by the algorithm (#2 hook) at REWARD time — see `algorithm/annotators/advantage.py`.
    # Scalar or per-token `[n_completion_tokens]`; see the class docstring.
    advantage: Optional[Union[float, "torch.Tensor"]] = None
    step_idx: int = 0
    done: bool = True
    mask: Any = None
    algo_extra: dict[str, Any] = field(default_factory=dict)  # reserved for algorithm (#2)
    # APPENDED, not inserted, and the grouping this costs is deliberate. `truncated` reads
    # next to `done` and `info` next to `algo_extra`; both were written there first. But a
    # dataclass field's POSITION is its positional constructor ABI, and `schema/` is an
    # integration boundary other repos build against: inserting `truncated` before `mask`
    # rebinds an existing `Transition(..., done, mask, algo_extra)` call so the old mask
    # lands in `truncated` and the old algorithm metadata in `mask`. Nothing raises. The
    # adjacency is documented here instead, which is the cheaper half to give up.

    # `done` records an environment ending here; `truncated` says WHY. A time-limit cut is not a
    # terminal state, so bootstrapping V(s') through it is correct while bootstrapping
    # through a real terminal is not -- conflating the two is a value-function bug that
    # trains quietly. Gym split its own `done` for this reason. Multi-step embodied Envs
    # (DESIGN 1.3) were the first consumer, and single-turn text RL sets it too: a PPO rollout that
    # hits the token cap is `done` (the episode ended) with `truncated` True. Fixed-duration
    # sessions additionally use feedback.episode_complete for their physical reset boundary.
    truncated: bool = False
    # The `info` of DESIGN 1.3's own Env contract, `step(action) -> (obs, reward, done,
    # info)`. Env-owned per-step facts that are neither observation nor reward (BDDL goal
    # counts, collision flags, sim diagnostics). Distinct from `algo_extra`, which is
    # reserved for the algorithm: an Env writing there would collide with it.
    info: dict[str, Any] = field(default_factory=dict)
    # Interaction capabilities populate these; offline samples need no environment facts.
    source: SampleSource | None = None
    feedback: EnvironmentFeedback | None = None
    collection_end: bool = False


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)
    id: str = ""
    lineage: tuple[str, ...] = ()  # correlation / provenance (observability §11.2)
    session: Optional[str] = None
    status: Status = Status.PENDING
    group_id: Optional[str] = None  # GRPO grouping
    # Ray/NIXL data-plane lineage. Each entry is a serializable page descriptor, retained even if a
    # downstream role transforms the corresponding field, so the driver can release every source
    # lease together at the synchronous step boundary. ``Any`` keeps this public schema Ray-free.
    pool_leases: tuple[Any, ...] = ()
    # TODO(v2, §2.8): parent / sub_trajectories for hierarchical (macro→micro) ops

    @property
    def final_reward(self) -> Optional[float]:
        """Convenience: terminal scalar reward if present (v0)."""
        if not self.transitions:
            return None
        r = self.transitions[-1].reward
        return None if r is None else r.get(FINAL_REWARD_KEY)


@dataclass
class Preference:
    """Typed supervised signal for offline DPO (DESIGN §1.4): the winner/loser pair a
    `Transition` carries in `targets`. Each side is the per-sample batch dict the DPO
    objective consumes (`data.packer.preference.unpack_preferences` → `dpo_objective`).

    A malformed pair — a missing or non-dict side — is rejected HERE, at construction, so a
    bad pair fails loud at the producer (`roles.env.preference._pair_trajectory`) instead of
    surfacing deep in the objective's four forwards."""

    winner: dict
    loser: dict

    def __post_init__(self) -> None:
        for name, side in (("winner", self.winner), ("loser", self.loser)):
            if not isinstance(side, dict):
                raise ValueError(
                    f"[Preference] {name} must be a dict (the per-sample batch), got "
                    f"{type(side).__name__} — an offline-DPO pair needs both a winner and a loser side."
                )


@dataclass(frozen=True)
class WFMConditioningInput:
    """One shared, replayable conditioning template for an online WFM group.

    ``model_batch`` preserves the model's batch-size-one input convention. The
    producer-stamped ``template_identity`` binds the opaque payload without making
    consumers hash or compare large tensors. Both identities survive the metadata /
    bulk codec, and consumers treat the nested values as read-only.
    """

    id: str
    template_identity: str
    model_batch: dict[str, Any]
    prompt: str
    modality: Literal["image", "video"] = "video"

    def __post_init__(self) -> None:
        if not isinstance(self.template_identity, str) or not self.template_identity:
            raise ValueError("WFM conditioning template_identity must be a non-empty string")


def _tensor_shape(value: Any) -> Optional[tuple]:
    """`value`'s shape as a tuple when it is really a tensor/ndarray, else None.

    None covers two cases, and the second is the load-bearing one: `value is None` (the field is
    absent), and `value` is a `comm.transfer.mover.BulkRef` — the handle a tensor field becomes
    between `pack_payload` and `materialize_payload`. `comm/codec.py::_walk` rebuilds dataclasses
    with `dataclasses.replace()`, which **re-runs `__post_init__` on every pack and unpack**, so on
    the packed side a validator sees handles, not tensors. A shape assertion that did not tolerate
    that would pass in-process and fail the first time the payload crossed a worker boundary.

    Kept duck-typed rather than `isinstance(value, torch.Tensor)` so this module stays torch-free.
    """
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(shape)


@dataclass
class TeacherSignal:
    """What a teacher role produced for ONE transition, and nothing else — carried in `targets`.

    The teacher's RAW OUTPUT. It deliberately does not carry an advantage: OPD's per-token
    advantage is an *algorithm annotation* and rides `Transition.advantage`, the same channel GRPO
    and PPO use (see `Transition`). Keeping a second advantage here would mean two advantage
    concepts in one codebase for no gain.

    Exactly one of two payloads, each all-or-nothing:

    - **top-k, the shared-vocab path (v1).** `topk_ids [T, k]` are ids in the ROLLOUT's own token
      space — the objective scores vLLM's `prompt_token_ids` rather than a re-tokenization, so the
      teacher's ids must align to that space, not merely to a tokenizer with the same vocab size.
      `topk_logprobs [T, k]` are the teacher's logprobs for those ids.
    - **chunk, the cross-tokenizer DPCA path.** `chunk_bounds [C, 2]` are full-completion
      student-token `[start, end)` spans and `chunk_teacher_logprob [C]` the teacher's
      log-likelihood per chunk.

    NOT frozen, matching `Preference`: device moves `setattr` tensor fields in place.
    """

    topk_ids: Optional["torch.Tensor"] = None
    topk_logprobs: Optional["torch.Tensor"] = None
    chunk_bounds: Optional["torch.Tensor"] = None
    chunk_teacher_logprob: Optional["torch.Tensor"] = None

    def __post_init__(self) -> None:
        fields = (self.topk_ids, self.topk_logprobs, self.chunk_bounds, self.chunk_teacher_logprob)
        if all(value is None for value in fields):
            raise ValueError(
                "[TeacherSignal] all four fields are None — an empty signal carries no teacher "
                "information, so the student would be distilled against nothing while the loss "
                "still looked healthy. Set the top-k pair (shared-vocab path) or the chunk pair "
                "(cross-tokenizer DPCA path)."
            )
        # CO-PRESENCE, not presence. A `BulkRef` is not None, so both-or-neither is exactly the
        # predicate that survives the pack/unpack round trip described in `_tensor_shape`.
        for a_name, a, b_name, b in (
            ("topk_ids", self.topk_ids, "topk_logprobs", self.topk_logprobs),
            ("chunk_bounds", self.chunk_bounds, "chunk_teacher_logprob", self.chunk_teacher_logprob),
        ):
            if (a is None) != (b is None):
                present, absent = (a_name, b_name) if a is not None else (b_name, a_name)
                raise ValueError(
                    f"[TeacherSignal] {present} is set but {absent} is None — they are one payload "
                    f"and are meaningless apart (ids without logprobs, or the reverse)."
                )
        has_topk = self.topk_ids is not None
        has_chunks = self.chunk_bounds is not None
        if has_topk and has_chunks:
            raise ValueError(
                "[TeacherSignal] top-k and chunk payloads are both set — one signal must carry "
                "exactly one teacher-signal mode, never both."
            )
        ids_shape, logprobs_shape = _tensor_shape(self.topk_ids), _tensor_shape(self.topk_logprobs)
        if ids_shape is not None and logprobs_shape is not None:
            if len(ids_shape) != 2 or ids_shape != logprobs_shape:
                raise ValueError(
                    f"[TeacherSignal] topk_ids and topk_logprobs must both be [T, k] and identically "
                    f"shaped, got {ids_shape} and {logprobs_shape} — entry (t, j) of one names the "
                    f"token entry (t, j) of the other scores."
                )
        bounds_shape = _tensor_shape(self.chunk_bounds)
        chunk_logprob_shape = _tensor_shape(self.chunk_teacher_logprob)
        if bounds_shape is not None and chunk_logprob_shape is not None:
            if len(bounds_shape) != 2 or bounds_shape[1] != 2:
                raise ValueError(
                    f"[TeacherSignal] chunk_bounds must be [C, 2] (student-token [start, end) per "
                    f"chunk), got {bounds_shape}."
                )
            if chunk_logprob_shape != (bounds_shape[0],):
                raise ValueError(
                    f"[TeacherSignal] chunk_teacher_logprob must be [C] with C == chunk_bounds' "
                    f"first dim ({bounds_shape[0]}), got {chunk_logprob_shape}."
                )
