# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""vLLM rollout body (DESIGN §1.2 body=(a) local compute; §8.2 #6 engine integration).

`VLLMRollout`: a real inference engine (vLLM) dropped in as a sibling to `HFRollout`,
producing the unified Trajectory/Transition data model (§1.4). Route C (TP via
`external_launcher` on a per-replica rendezvous) is specced in
`docs/superpowers/specs/2026-07-06-per-replica-rendezvous-and-wg-shape-dispatch-design.md`.
Contract:

  - MUST emit ``policy_info["rollout_logprobs"]`` (π_old, per completion token; the
    policy-gradient ratio needs it). Same field name as ``HFRollout`` so the clipped
    objective reads one key for either body.
  - Emits ``policy_info["finish_reason"]`` (``schema.FINISH_REASON_KEY``): the engine's own
    verdict on how the completion ended, ``"stop"`` | ``"length"`` (cut at the token budget) |
    another engine string, or None when the engine reports none. The clipped objective's
    truncation diagnostics and ``rl.mask_truncated_completions`` read it.
  - When built with ``record_routed_experts`` (``rl.routing_replay='rollout'``), emits
    ``policy_info["rollout_routed_experts"]``: the engine's per-token top-k expert ids for every
    input position of prompt + completion, the record the sparse-MoE policy replays
    (``data/packer/forward_routing.py`` aligns it; the codec carries it as bulk).
  - MUST carry a correct ``group_id`` with group integrity (GRPO advantage needs the
    whole group, §2.3). vLLM ``SamplingParams(n=group)`` returns the G completions for a
    prompt atomically, so a group is all-or-nothing by construction.

``apply_weights`` is REAL: the bound backend's mover receives FULL tensors over the weight
plane and vLLM's own ``load_weights`` slices each into this rank's TP shard (the recv layout
is injected at bind from the policy for the COLLECTIVE/NCCL path — §2.4/§4). No bound backend
⇒ version-bump no-op (static-weight fallback).
"""

from __future__ import annotations

import functools
import hashlib
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cosmos_framework.posttrain.comm.weight_digest import normalize_tensor_digest, tensor_digests_equal
from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec
from cosmos_framework.posttrain.comm.weight_validation import check_weight_names
from cosmos_framework.posttrain.roles.rollout import vllm_weights as weight_compat
from cosmos_framework.posttrain.roles.rollout.base import (
    ShardSpec,
    WeightReceiverBase,
    group_ids_for_prompt,
)
from cosmos_framework.posttrain.roles.rollout.sampler_head import SamplerHeadSpec, parse_ranked_rows
from cosmos_framework.posttrain.roles.rollout.validation import (
    ValidationRolloutContext,
    distributed_validation_rng,
    normalize_validation_semantics_identity,
    rollout_validation_semantics_identity,
    run_validation_stage,
    seed_validation_rng,
    validation_generation_seed,
    validation_prompt_seed,
)
from cosmos_framework.posttrain.roles.vllm_request import vllm_request_media_identity
from cosmos_framework.posttrain.schema import (
    FINISH_REASON_KEY,
    ROLLOUT_TOPK_IDS_KEY,
    ROLLOUT_TOPK_LOGPROBS_KEY,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
)
from cosmos_framework.posttrain.schema.routing import ROLLOUT_ROUTED_EXPERTS_KEY
from cosmos_framework.posttrain.utils.dist import world_size_or_1
from cosmos_framework.posttrain.utils.log import logger
from cosmos_framework.posttrain.utils.profile import ProfileRecord

if TYPE_CHECKING:
    from cosmos_framework.posttrain.data.packer.base import DataPacker


def _require_identical_rank_identity(identity: dict[str, Any], *, context: str) -> None:
    """Prove every vLLM TP rank holds one identical finite request/output identity.

    ``external_launcher`` is SPMD: each rank invokes the facade and then participates in the same
    engine collectives. Matching startup artifacts is insufficient when a custom packer can depend
    on host-local state, so compare the bounded semantic proof around each collective-bearing call.
    """
    from cosmos_framework.posttrain.utils.dist import rank_identical
    from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

    normalized = normalize_validation_semantics_identity(
        identity,
        context=context,
    )
    digest = canonical_json_sha256(normalized, context=context)
    rank_identical(digest, context)


def _is_replica_leader() -> bool:
    """Return true for rank zero, including the non-distributed TP1 path."""
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_world_size() == 1 or dist.get_rank() == 0


def validate_limit_mm_against_engine(
    model: Any, configured: Mapping[str, int], *, owner: str, registry: Any | None = None
) -> set[str]:
    """Refuse a ``rollout.limit_mm_per_prompt`` key the BUILT engine's model does not support.

    vLLM converts an unknown modality key into a generic dummy option and the model's ``allowed_mm_limits``
    iterates only its supported modalities, so a mistyped or unsupported key is silently ignored by the engine.
    The supported set is model-specific, so it is read from the built engine:
    ``model.llm_engine.model_config`` through ``MULTIMODAL_REGISTRY.create_processor(...).info
    .get_supported_mm_limits()``. This intentionally does not call ``supports_multimodal_inputs``,
    whose answer reflects the configured limits and becomes false when every supported modality is
    disabled with a zero budget. Returns the model's intrinsic supported modality set. ``registry``
    is injectable for tests.
    """
    engine = getattr(model, "llm_engine", None)
    model_config = getattr(engine, "model_config", None)
    if model_config is None:
        raise ValueError(
            f"{owner}: rollout.limit_mm_per_prompt={dict(configured)!r} is set but the built engine exposes no "
            "llm_engine.model_config to check the supported modalities against (not a vllm.LLM?)"
        )
    if registry is None:
        from vllm.multimodal import MULTIMODAL_REGISTRY as registry  # type: ignore[no-redef]
    try:
        processing_info = registry.create_processor(model_config).info
    except ValueError as exc:
        raise ValueError(
            f"{owner}: rollout.limit_mm_per_prompt={dict(configured)!r} is set but the engine's model "
            f"{getattr(model_config, 'model', '?')!r} has no registered multimodal processor; "
            "the limit would be ignored"
        ) from exc
    supported = set(processing_info.get_supported_mm_limits())
    if not supported:
        raise ValueError(
            f"{owner}: rollout.limit_mm_per_prompt={dict(configured)!r} is set but the engine's model "
            f"{getattr(model_config, 'model', '?')!r} supports no multimodal inputs; the limit would be ignored"
        )
    unknown = sorted(set(configured) - supported)
    if unknown:
        raise ValueError(
            f"{owner}: rollout.limit_mm_per_prompt key(s) {unknown} are not modalities of the engine's model "
            f"{getattr(model_config, 'model', '?')!r} (supported: {sorted(supported)}); vLLM would silently ignore them"
        )
    return supported


class VLLMRollout(WeightReceiverBase):
    """v0 rollout body backed by a vLLM engine. Mirrors the `HFRollout` contract so the
    runner and `bind_device_groups` are unchanged — only the body class differs.

    TP is a property of the injected `LLM` (built in setup with tensor_parallel_size=K +
    distributed_executor_backend="external_launcher"), NOT of this body: under SPMD every
    TP rank's generate() returns the same full output, so this class is unchanged for TP>1."""

    def __init__(
        self,
        model: Any,
        *,
        packer: "DataPacker",
        sampling: dict | None = None,
        model_identifier: str | None = None,
        engine_config: dict[str, Any] | None = None,
        resolved_model_config: dict[str, Any] | None = None,
        record_routed_experts: bool = False,
        sleep_level: int = 1,
        topk_logprobs: int = 0,
    ) -> None:
        """Bind an already-built `vllm.LLM` engine, its packer and its sampling overrides to this body.

        Everything after ``model`` is KEYWORD-ONLY. The 2nd positional slot has changed meaning across
        commits (``sampling`` before the packer seam, ``packer`` after it), so a legacy positional call
        like ``VLLMRollout(llm, {"max_tokens": 4})`` would silently bind a sampling dict as the packer
        and surface later as an unrelated AttributeError. Keyword-only turns that into an immediate,
        accurate TypeError.

        Args:
            model: a built ``vllm.LLM`` instance (constructed in the Ray actor's setup, where the GPU is).
            packer: REQUIRED (keyword-only) DataPacker — the single prompt-shaping path
                (``get_rollout_input``). It is required HERE, not at ``generate``, because it is
                decidable at construction: a rollout built without one would bind, spawn every Ray actor
                and warm the engine, then die on the first generate() with the whole cluster up.
                Requiring it here also makes the ``adds_special_tokens`` advisory check
                unconditional — on the old packer-less path that check never ran at all.
            sampling: overrides for the default `SamplingParams` fields (temperature/top_p/max_tokens/...);
                must NOT set the framework-owned reserved keys ``n`` or ``logprobs``.
            model_identifier: Immutable model artifact identifier. Config-driven construction passes
                the exact vLLM ``model=`` value; validation fails closed when it is absent.
            engine_config: The complete finite JSON kwargs used to construct ``vllm.LLM``. Required
                for validation identity because runtime engine internals are version-specific and may
                contain mutable caches that must not be checkpoint semantics.
            resolved_model_config: Live resolved HF model config used by the engine, captured before
                construction. This binds architecture, RoPE, and other non-weight forward semantics
                even when a local ``name_or_path`` is mutated in place between resume attempts.
            record_routed_experts: The engine was built with ``enable_return_routed_experts`` and
                every training completion must carry ``policy_info["rollout_routed_experts"]``; a
                completion the engine returns without routing is then an error, not a natural run.
            sleep_level: vLLM sleep level used by ``sleep``. Level 1 keeps a host copy of the
                engine weights and drops the KV cache; level 2 drops both and is only correct when
                the following weight sync rewrites every engine parameter.
            topk_logprobs: > 0 asks the engine for that many top-k logprobs per generated token
                (``SamplingParams(logprobs=k)``; the engine must have been built with
                ``max_logprobs >= k`` and, for the head to be the SAMPLING distribution,
                ``logprobs_mode="processed_logprobs"`` -- `config/build.py::vllm_engine_kwargs`) and
                stamps the sampler head on every training completion as
                ``policy_info["rollout_topk_ids"]`` (int32 ``[T, k]``) and
                ``policy_info["rollout_topk_logprobs"]`` (float32 ``[T, k]``), the record
                `rl.score_centering` centres on. 0 (default) requests only the sampled token's
                logprob, byte-identical to before the knob existed.

        Raises:
            ValueError: ``packer`` is None, ``sampling`` sets a reserved key, ``sleep_level``
                is not 1 or 2, or ``topk_logprobs`` is negative.
        """
        if sleep_level not in (1, 2):
            raise ValueError(f"sleep_level must be 1 or 2, got {sleep_level!r}")
        if isinstance(topk_logprobs, bool) or not isinstance(topk_logprobs, int) or topk_logprobs < 0:
            raise ValueError(f"topk_logprobs must be a non-negative integer, got {topk_logprobs!r}")
        self._topk_logprobs = int(topk_logprobs)
        if self._topk_logprobs > 0:
            # The head is the SAMPLING distribution only when the engine was built in processed-logprob
            # mode with a wide enough max_logprobs: verify that against the construction identity here,
            # so the likelihood convention is a checked contract and not a call-site convention.
            SamplerHeadSpec(self._topk_logprobs).validate_engine_config(
                engine_config, owner="VLLMRollout(topk_logprobs)"
            )
        # model: a `vllm.LLM` instance, built inside the Ray actor's setup() so the engine
        # lives where the GPU is (DESIGN §8.2 #6). `sampling` overrides the SamplingParams
        # fields below (temperature/top_p/max_tokens/...).
        if packer is None:  # explicit packer=None: a bare TypeError would not say what to do
            raise ValueError(
                "VLLMRollout requires a DataPacker at construction — build one with "
                "init_data_packer(name_or_path) and pass VLLMRollout(llm, packer=..., sampling=...) "
                "(or build_vllm_rollout(cfg, packer))"
            )
        self.model = model
        self.sampling = sampling or {"temperature": 1.0, "top_p": 1.0, "max_tokens": 16}
        reserved = {"n", "logprobs"} & set(self.sampling)
        if reserved:  # framework-owned: n = GRPO group size (per generate call), logprobs = π_old
            raise ValueError(f"sampling must not set reserved SamplingParams keys {sorted(reserved)}")
        self.packer = packer
        self._validation_model_identifier = model_identifier
        self._validation_engine_config = None if engine_config is None else dict(engine_config)
        if engine_config is not None and engine_config.get("limit_mm_per_prompt") is not None:
            # A configured per-request multimodal budget is checked against the BUILT engine's model-specific
            # supported modalities (config-time validation only knows vLLM's global vocabulary).
            validate_limit_mm_against_engine(model, engine_config["limit_mm_per_prompt"], owner="VLLMRollout")
        self._validation_resolved_model_config = None if resolved_model_config is None else dict(resolved_model_config)
        self._record_routed_experts = bool(record_routed_experts)
        self._sleep_level = int(sleep_level)
        # Engine memory regions currently released by `sleep`; vLLM refuses to remap a mapped
        # region, so wake-ups are filtered through this set.
        self._asleep_tags: set[str] = set()
        if packer.adds_special_tokens():
            # NOT fatal. This rollout records `req.prompt_token_ids` — vLLM's OWN tokenization — in
            # policy_info, and the objective prefers those ids (design 2026-07-28 §10), so the
            # prefix scored IS the prefix generated no matter what the tokenizer adds. Refusing
            # here would block `llama`, which the packer registry explicitly supports. The warning
            # remains because the residual risk is real but narrow: a consumer that re-tokenizes
            # `obs` instead of using the stored ids would score a shifted prefix.
            logger.warning(
                "packer tokenizer adds special tokens (add_special_tokens=True != False). This is "
                "SAFE for the GRPO objective, which scores policy_info['prompt_ids'] (vLLM's own "
                "req.prompt_token_ids). It is only a hazard for a consumer that re-tokenizes the "
                "prompt instead — such a path needs a no-BOS tokenizer (e.g. Qwen) or pre-tokenized "
                "prompts (spec §9.6)."
            )
        self._weight_version = 0
        # A load_format="dummy" engine skipped its checkpoint read and holds random weights until
        # the first committed publication; version 0 cannot mark that because the initial
        # publication commits version 0 too.
        self._load_format = "auto" if engine_config is None else str(engine_config.get("load_format", "auto"))
        self._publication_committed = False
        # Per-stream coverage accumulator: engine param names loaded by the current sync stream
        # (reset on bucket 0, verified after the last bucket). None ⇒ the engine's load_weights
        # does not report loaded names, so coverage is unverifiable — the check degrades to a no-op
        # unless a dummy-loaded engine refuses the stream (`_check_unverifiable_coverage`).
        self._loaded_names: "set | None" = set()
        # Engine param-name set for the coverage check, filled lazily ONCE (see _engine_param_names).
        self._engine_names: "set | None" = None
        # Routed-expert owner module names for the MoE layout, filled lazily ONCE on the first mapped
        # stream (see _routed_expert_owner_names).
        self._moe_owner_names: "frozenset[str] | None" = None

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Bind explicit vLLM construction plus the live tokenizer/packer generation path."""
        if self._validation_engine_config is None:
            raise TypeError(
                "vLLM validation identity requires engine_config containing the exact vllm.LLM "
                "construction kwargs; use build_vllm_rollout or pass it explicitly"
            )
        if self._validation_resolved_model_config is None:
            raise TypeError(
                "vLLM validation identity requires the resolved model config used by the engine; "
                "use build_vllm_rollout or pass resolved_model_config explicitly"
            )
        return rollout_validation_semantics_identity(
            backend="vllm",
            model=self.model,
            packer=self.packer,
            sampling=dict(self.sampling),
            backend_config={
                "engine": self._validation_engine_config,
                "resolved_model_config": self._validation_resolved_model_config,
            },
            model_identifier="" if self._validation_model_identifier is None else self._validation_model_identifier,
        )

    def weight_version(self) -> int:
        """The training iteration whose weights the live engine currently holds — 0 at init, then the
        producer's ``_iter`` after each ``weight_sync`` (the policy stamps it once at end-of-stream via
        rollout-wide prepare/commit publication). Symmetric to the trainer's ``iteration()``: the united read for
        staleness checks / loop logging.

        Returns:
            The weight version (training iteration) the live engine currently holds.
        """
        return self._weight_version

    def commit_weight_version(self, version: int) -> None:
        """Publish a prepared version; the first commit is what lets a dummy-loaded engine sample."""
        super().commit_weight_version(version)
        self._publication_committed = True

    def _require_published_weights(self) -> None:
        """Fail before sampling while a ``load_format="dummy"`` engine still holds its random weights."""
        if getattr(self, "_load_format", "auto") == "dummy" and not getattr(self, "_publication_committed", False):
            raise RuntimeError(
                "vLLM rollout was built with load_format='dummy' and holds random weights until weight_sync "
                "commits a policy publication; no weight version has been committed yet"
            )

    def broadcast_live_weights(self, version: int, *, verify_values: bool = False) -> str:
        """Broadcast the underlying vLLM worker's active tensors across rollout replicas.

        The supported runtime contract has exactly one live model participant behind each rollout
        cell: TP>1 uses external_launcher (one TP rank per cell), while TP1 disables vLLM engine
        multiprocessing. ``_apply_to_models`` retains V0/V1 compatibility without moving tensors
        through the controller.
        """
        spec = getattr(self, "_weight_broadcast_spec", None)
        if spec is None:
            raise RuntimeError("rollout replica broadcast is not bound")
        results = weight_compat._apply_to_models(
            self.model,
            functools.partial(
                weight_compat._vllm_broadcast_live_weights,
                spec=spec,
                verify_values=verify_values,
            ),
        )
        if len(results) != 1:
            raise RuntimeError(
                "replica broadcast requires exactly one live vLLM model participant per rollout cell, "
                f"but apply_model reached {len(results)} participants"
            )
        digest = results[0]
        if not isinstance(digest, str):
            raise TypeError(f"vLLM live-weight broadcast returned {type(digest).__name__}, expected str digest")
        self.mark_weight_version_ready(version)
        return digest

    def initialize_weight_broadcast(self) -> None:
        """Create the side process group inside the single underlying vLLM model participant."""
        spec = getattr(self, "_weight_broadcast_spec", None)
        if spec is None:
            raise RuntimeError("rollout replica broadcast is not bound")
        results = weight_compat._apply_to_models(
            self.model,
            functools.partial(weight_compat._vllm_initialize_weight_broadcast, spec=spec),
        )
        if len(results) != 1:
            raise RuntimeError(
                "replica broadcast requires exactly one live vLLM model participant per rollout cell, "
                f"but apply_model reached {len(results)} participants"
            )

    # --- GPU residency ---
    _SLEEP_TAGS = ("weights", "kv_cache")

    def sleep(self) -> None:
        """Release the engine's weights and KV cache so the colocated trainer can use the card.

        Idempotent when every region is already asleep. Pinned vLLM ignores ``sleep`` while ANY
        region remains asleep, so cleanup after a partial wake first wakes the remaining region,
        then issues one full sleep. Requires an engine built with ``enable_sleep_mode=True``;
        ``build_vllm_rollout`` does that unless a colocated run opted out of the hand-off cycle
        (``rl.colocated_offload=False``), in which case the engine is built without sleep mode and
        the loop never calls this.
        """
        if self._asleep_tags == set(self._SLEEP_TAGS):
            return
        if self._asleep_tags:
            self.wake_up()
        self.model.sleep(level=self._sleep_level)
        self._asleep_tags = set(self._SLEEP_TAGS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def wake_up(self, tags: tuple[str, ...] | None = None) -> None:
        """Restore released engine memory; ``tags`` restores only ``weights`` and/or ``kv_cache``.

        Weight sync only needs the weights resident, so the loop wakes ``("weights",)`` before the
        sync and the KV cache before the next generation. Already-awake regions are skipped.
        """
        wanted = set(self._SLEEP_TAGS) if tags is None else set(tags)
        unknown = wanted - set(self._SLEEP_TAGS)
        if unknown:
            raise ValueError(f"unknown vLLM wake_up tags {sorted(unknown)}; expected subset of {self._SLEEP_TAGS}")
        pending = [tag for tag in self._SLEEP_TAGS if tag in wanted and tag in self._asleep_tags]
        if not pending:
            return
        self.model.wake_up(tags=pending)
        self._asleep_tags -= set(pending)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _require_awake(self, *tags: str) -> None:
        """Fail before touching the engine when any of ``tags`` (default: all) is still released."""
        asleep = getattr(self, "_asleep_tags", set())  # getattr: manual rigs bypass __init__
        blocked = sorted(asleep & set(tags or self._SLEEP_TAGS))
        if blocked:
            raise RuntimeError(f"vLLM rollout has {blocked} released; call wake_up() first")

    # --- weight-transfer edge target (DESIGN §8.2 #4) ---
    def apply_weights(self, source: "Any | None" = None) -> None:
        """Import trainer weights into the RUNNING vLLM engine — PEER-ONLY surface.

        ``source`` is the policy PEER (what ``weight_sync(rollout, policy)`` sends — the grid
        injects a PeerClient per dp-aligned cell); the sync streams bucket-by-bucket via
        ``pull_stream_from_peer`` (plan pulled lazily, transport from the bound mover). The mover
        delivers FULL (unsharded) tensors and vLLM's own ``load_weights`` slices each into this
        rank's TP shard (design §2.4/§4). ``None`` keeps the degenerate contracts (no backend /
        SharedMover ⇒ stamp-only). The old WeightPayload batch form was REMOVED — for a manual
        whole-model sync drive ``WeightSync.apply/export`` directly (as the NCCL rigs do).
        A ``load_format="dummy"`` engine refuses the stamp-only contracts: they would advertise a
        version over its random weights."""
        from cosmos_framework.posttrain.comm.transfer.movers import SharedMover
        from cosmos_framework.posttrain.comm.weight_transfer import pull_stream_from_peer

        be = getattr(self, "_weight_backend", None)
        if getattr(self, "_load_format", "auto") == "dummy" and (be is None or isinstance(be.mover, SharedMover)):
            raise RuntimeError(
                "vLLM rollout built with load_format='dummy' needs a streaming weight edge; with no backend "
                "or a SharedMover, apply_weights would only stamp a version over random weights"
            )
        return pull_stream_from_peer(self, source)

    def _engine_sink(self, state: dict) -> set[str] | None:
        """Load one streamed bucket into the in-process vLLM engine.

        Posttrain forces ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` before engine construction, so
        ``apply_model`` directly invokes this callable in the hosting Ray actor. Tensor views stay
        zero-copy; vLLM maps each HF name to its possibly fused local TP parameter.

        Args:
            state: mapping of HF param name → full tensor to load into the running engine in place.
        """
        self._require_awake("weights")
        named = list(state.items())
        witness_spec = getattr(self, "_weight_witness_spec", None)
        if isinstance(witness_spec, WeightWitnessSpec) and witness_spec.wire_name in state:
            results = weight_compat._apply_to_models(
                self.model,
                functools.partial(
                    weight_compat._vllm_load_weights_with_witness,
                    named=named,
                    spec=witness_spec,
                ),
            )
            loaded = results[0][0]
            expected_by_worker = [result[1] for result in results]
            self._weight_witness_expected = tuple(normalize_tensor_digest(value) for value in expected_by_worker)
        else:
            loaded = weight_compat._apply_to_model(
                self.model,
                functools.partial(weight_compat._vllm_load_weights, named=named),
            )
        return None if loaded is None else set(loaded)

    def set_weight_destination_names(self, names: tuple[str, ...]) -> None:
        """Resolve mapper destinations onto this engine's live routed-expert layout, then install them.

        The producer addresses routed-expert parameters on the checkpoint's expert container; only
        the engine knows whether it owns them there or one module deeper (``VllmMoeLayout``). The
        layout is decided here, once per stream and before any coverage check or tensor movement,
        from the live owner modules. Dense engines and identity-mapped streams pass through.
        """
        be = getattr(self, "_weight_backend", None)
        if be is None:
            return
        if names is not None:
            layout = weight_compat.resolve_vllm_moe_layout(self._routed_expert_owner_names(), names)
            if layout is not None:
                names = layout.live_destination_names(names)
        be.set_destination_names(names)

    def check_plan_coverage(self) -> None:
        """Compare mapper-proven destinations with the exact live engine namespace before bucket 0.

        Ordinary identity-mapped models publish no destination proof because vLLM may pack their HF
        checkpoint names; those retain the existing post-load coverage behavior. A semantic mapper
        publishes the post-packing destination set, making both missing and unknown names decidable
        through a read-only engine query before any tensor is loaded.
        """
        be = getattr(self, "_weight_backend", None)
        if be is None:
            return
        destination_names_fn = getattr(be, "destination_names", None)
        destination_names = destination_names_fn() if callable(destination_names_fn) else None
        if destination_names is None:
            super().check_plan_coverage()
            return
        check_weight_names(
            self._engine_param_names(),
            destination_names,
            context="mapped weight destinations do not cover the live vLLM engine exactly",
        )

    def set_weight_witness_spec(self, spec: WeightWitnessSpec) -> None:
        """Install the q-to-packed-QKV witness and reset this lane before bucket 0."""
        if not isinstance(spec, WeightWitnessSpec):
            raise TypeError(f"weight witness spec must be WeightWitnessSpec, got {type(spec).__name__}")
        self._weight_witness_spec = spec
        self._weight_witness_expected = None

    def require_weight_witness_verified(self, spec: WeightWitnessSpec) -> None:
        """Fail before readiness unless this stream verified the installed q-to-QKV mapping."""
        installed = getattr(self, "_weight_witness_spec", None)
        if installed != spec:
            raise RuntimeError(
                f"weight witness spec changed before readiness: installed={installed!r}, expected={spec!r}"
            )
        expected_by_worker = getattr(self, "_weight_witness_expected", None)
        if not isinstance(expected_by_worker, tuple) or not expected_by_worker:
            raise RuntimeError(
                f"live vLLM packed-QKV witness {spec.wire_name!r} -> {spec.destination_name!r} was not verified"
            )
        expected_shape = tuple(expected_by_worker[0].get("shape", ()))
        if len(expected_shape) != 2 or expected_shape[0] < 1:
            raise RuntimeError(f"live vLLM packed-QKV witness recorded invalid expected shape {expected_shape!r}")
        actual_by_worker = weight_compat._apply_to_models(
            self.model,
            functools.partial(
                weight_compat._vllm_live_q_witness,
                destination_name=spec.destination_name,
                q_rows=int(expected_shape[0]),
            ),
        )
        expected_values = [dict(expected) for expected in expected_by_worker]
        if len(actual_by_worker) != len(expected_values) or not all(
            tensor_digests_equal(actual, expected)
            for actual, expected in zip(actual_by_worker, expected_values, strict=True)
        ):
            raise ValueError(
                f"final live vLLM q segment changed after inline verification for {spec.destination_name!r}: "
                f"expected {expected_values!r}, got {actual_by_worker!r}"
            )

    # --- streaming weight sync (per-bucket); set_bucket_plan + check_plan_coverage on WeightReceiverBase ---
    def apply_bucket(self, payload: Any, i: int) -> None:
        """Consumer side of a streaming round: fetch ONLY bucket i and load it into the running engine
        in place (peak ≈ one bucket). PURE LOAD, fully delegated: all receive mechanics (collective
        recv_plan narrow/restore, staged fetch, consumer_done) live in WeightSync.apply_bucket — this
        body only injects its engine sink — and the streaming version protocol is owned by
        pull_stream_from_peer. No backend ⇒ no-op.

        vLLM repacks routed MoE weights into backend-specific kernel layouts after startup. A mapped
        stream whose destinations land on the engine's routed-expert owner modules therefore
        brackets its buckets with vLLM's official layerwise checkpoint reload, scoped only to those
        ``RoutedExperts`` modules so dense/QKV loading and its live witness stay unchanged.

        Coverage check (fail-loud, once per stream): the generic strict plan/model check in
        pull_stream_from_peer is skipped for engine sinks (plan names are HF, engine names are
        fused), and vLLM's load_weights only raises on UNKNOWN names — a plan MISSING engine params
        would silently keep initial-checkpoint weights while the version stamp advertises fresh.
        So after the LAST bucket, every engine param must appear in the union of load_weights'
        returned name-sets, or we raise.

        Args:
            payload: the streaming round's transport payload for this bucket (backend-specific;
                passed through to `WeightSync.apply_bucket`, which does the receive mechanics).
            i: bucket index within the stream (i==0 resets coverage, last bucket triggers the check).
        """
        be = getattr(self, "_weight_backend", None)
        if be is None:
            return
        self._coverage_reset(i)
        if i == 0:
            destination_names_fn = getattr(be, "destination_names", None)
            destination_names = destination_names_fn() if callable(destination_names_fn) else None
            # Destinations are live names by now (set_weight_destination_names resolved the layout);
            # the stream carries routed experts iff one of them belongs to a live owner module.
            self._moe_reload_active = destination_names is not None and any(
                name.rpartition(".")[0] in self._routed_expert_owner_names() for name in destination_names
            )
            self._moe_reload_counts = None
            if self._moe_reload_active:
                engine = getattr(self.model, "llm_engine", None)
                model_config = getattr(engine, "model_config", None)
                model_dtype = getattr(model_config, "dtype", None)
                if not isinstance(model_dtype, torch.dtype):
                    raise TypeError(
                        f"mapped vLLM MoE reload requires llm.llm_engine.model_config.dtype, got {model_dtype!r}"
                    )
                self._moe_reload_dtype = model_dtype
                counts = tuple(weight_compat._apply_to_models(self.model, weight_compat._vllm_begin_moe_reload))
                if not counts or any(type(count) is not int or count < 1 for count in counts):
                    raise RuntimeError(f"vLLM MoE reload initialized no routed-expert modules: {counts!r}")
                self._moe_reload_counts = counts
        be.apply_bucket(self.model, i, payload, sink=self._load_engine_bucket)
        if i == be.bucket_count() - 1 and bool(getattr(self, "_moe_reload_active", False)):
            finish = functools.partial(
                weight_compat._vllm_finish_moe_reload,
                model_dtype=self._moe_reload_dtype,
            )
            counts = tuple(weight_compat._apply_to_models(self.model, finish))
            expected = getattr(self, "_moe_reload_counts", None)
            if counts != expected:
                raise RuntimeError(f"vLLM MoE reload module count changed from {expected!r} to {counts!r}")
            self._moe_reload_active = False
        self._coverage_check(be, i)

    def _check_unverifiable_coverage(self, be: Any) -> None:
        """Refuse an unverifiable stream into a dummy-loaded engine.

        A coverage gap keeps checkpoint weights under ``auto`` but random ones under ``dummy``. Mapped
        destinations proven before bucket 0 (``check_plan_coverage``) do not settle it: they show that
        the plan names every live parameter, not that the engine's ``load_weights`` wrote each one.

        Args:
            be: the weight backend that streamed the unverifiable plan.
        """
        del be
        if getattr(self, "_load_format", "auto") == "dummy":
            raise ValueError(
                "vLLM rollout built with load_format='dummy' cannot verify this weight stream: the engine's "
                "load_weights reported no loaded names, so random initial weights could survive the publication"
            )

    def _engine_param_names(self) -> set:
        """The engine's param-name set for the coverage check, fetched ONCE per body (lazy). The
        parameter set is fixed at engine construction (vLLM never registers params between syncs),
        so enumerating it per sync stream would be pure waste in a per-step GRPO loop. Assumes the
        engine instance is never swapped out under this body (true of the current lifecycle: the
        LLM is injected at construction).

        Returns:
            The set of the engine's param names (cached after the first fetch).
        """
        names = getattr(self, "_engine_names", None)  # getattr: manual rigs bypass __init__
        if names is None:
            names = {
                name
                for name, _shape, _ndim in weight_compat._apply_to_model(
                    self.model,
                    weight_compat._vllm_named_shapes,
                )
            }
            self._engine_names = names
        return names

    def _routed_expert_owner_names(self) -> frozenset[str]:
        """Qualified names of the engine's routed-expert owner modules, fetched ONCE per body (lazy).

        Like ``_engine_param_names``: the module tree is fixed at engine construction, and under
        vLLM V1 the fetch is an apply_model call into the EngineCore worker.
        """
        names = getattr(self, "_moe_owner_names", None)  # getattr: manual rigs bypass __init__
        if names is None:
            names = frozenset(weight_compat._apply_to_model(self.model, weight_compat._vllm_routed_expert_owner_names))
            self._moe_owner_names = names
        return names

    def prepare_recv(self) -> dict[str, ShardSpec]:
        """This TP rank's receive layout per param as a `ShardSpec`. v1 receives whole tensors and
        lets vLLM's ``load_weights`` slice, so report the full extent of the engine's local params;
        v2 reports the real TP slice so the planner can send only that slice (§2.4)."""
        shapes = weight_compat._apply_to_model(self.model, weight_compat._vllm_named_shapes)
        return {n: ShardSpec(0, (shape[0] if ndim else 0), 0) for n, shape, ndim in shapes}

    shard_map = prepare_recv

    # --- generation ---
    def generate(
        self,
        prompts: list[ObsBundle],
        group: int = 8,
        *,
        validation: ValidationRolloutContext | None = None,
        **knobs: Any,
    ) -> list[Trajectory]:
        """One batched vLLM call for training or a pinned held-out validation request.

        Returns ``len(prompts) * group`` trajectories, in ``len(prompts)`` complete groups.
        """
        del knobs  # forward-compatible common rollout surface; this backend has no other knobs
        self._require_awake()
        if validation is not None:
            return self._generate_validation(prompts, validation)
        if not prompts:
            return []

        def prepare() -> tuple[None, list[Any], Any, dict[str, Any]]:
            from vllm import SamplingParams

            sampling = dict(self.sampling)
            # No packer check here: __init__ makes it REQUIRED, so self.packer is always set.
            # logprobs=0 returns the sampled token likelihood needed by DPCA and clipped GRPO;
            # logprobs=k (score centering) additionally returns the engine's top-k per position.
            params: Any = SamplingParams(n=group, logprobs=self._topk_logprobs, **sampling)
            requests = [self.packer.get_rollout_input(obs) for obs in prompts]
            identity = self._training_request_identity(prompts, requests, sampling, group)
            return None, requests, params, identity

        def parse(_state: None, request_outputs: list[Any]) -> list[Trajectory]:
            return self._parse_training_requests(prompts, request_outputs, group)

        return self._execute_training_generation(
            prepare,
            parse=parse,
            output_identity=self._training_output_identity,
            context="vLLM training",
        )

    def _execute_training_generation(
        self,
        prepare: Callable[[], tuple[Any, list[Any], Any, dict[str, Any]]],
        *,
        parse: Callable[[Any, list[Any]], list[Trajectory]],
        output_identity: Callable[[list[Trajectory]], dict[str, Any]],
        context: str,
    ) -> list[Trajectory]:
        """Execute the shared rank-converged vLLM training request lifecycle."""
        from cosmos_framework.posttrain.utils.dist import run_rank_converged

        def preflight() -> tuple[Any, list[Any], Any, dict[str, Any]]:
            self._require_published_weights()
            return prepare()

        profile = ProfileRecord("rollout_generate")
        with profile.phase("preflight"):
            state, requests, params, request_identity = run_rank_converged(
                preflight,
                f"{context} batch preflight",
            )
            run_rank_converged(
                lambda: _require_identical_rank_identity(
                    request_identity,
                    context=f"{context} request identity",
                ),
                f"{context} request convergence",
            )
        with profile.phase("generation"):
            request_outputs = run_rank_converged(
                lambda: self.model.generate(requests, params),
                f"{context} model generation",
            )
        with profile.phase("parse"):
            trajectories = run_rank_converged(
                lambda: parse(state, request_outputs),
                f"{context} output parsing",
            )
        with profile.phase("output_sync"):
            identity = run_rank_converged(
                lambda: output_identity(trajectories),
                f"{context} output identity preparation",
            )
            run_rank_converged(
                lambda: _require_identical_rank_identity(
                    identity,
                    context=f"{context} output identity",
                ),
                f"{context} output convergence",
            )
        profile.emit(role="rollout", method="generate")
        # Grid routing consumes only rollout lane zero. Return no tensor-bearing payload from the
        # other SPMD lanes after the all-lane digest barrier, so their actor stores cannot leak one
        # unreferenced completion batch per step.
        return trajectories if _is_replica_leader() else []

    @staticmethod
    def _training_request_identity(
        prompts: list[ObsBundle],
        requests: list[Any],
        sampling: dict[str, Any],
        group: int,
    ) -> dict[str, Any]:
        """Build a bounded proof of request IDs, ordered media content, and sampling controls."""
        if len(requests) != len(prompts):
            raise RuntimeError(f"vLLM shaped {len(requests)} request(s) for {len(prompts)} prompt(s)")
        require_all_content_uuids = world_size_or_1() > 1
        proofs: list[dict[str, Any]] = []
        for prompt, request in zip(prompts, requests, strict=True):
            if isinstance(request, str):
                request_proof: dict[str, Any] = {
                    "prompt": request,
                    **vllm_request_media_identity({}, context="vLLM training request"),
                }
            elif isinstance(request, Mapping):
                prompt_value = request.get("prompt")
                raw_prompt_token_ids = request.get("prompt_token_ids")
                prompt_token_ids = (
                    None if raw_prompt_token_ids is None else [int(token_id) for token_id in raw_prompt_token_ids]
                )
                if (prompt_value is None) == (prompt_token_ids is None):
                    raise RuntimeError("vLLM training request must carry exactly one of prompt or prompt_token_ids")
                request_proof = {
                    "prompt": prompt_value,
                    "prompt_token_ids": prompt_token_ids,
                    **vllm_request_media_identity(
                        request,
                        context="vLLM training request",
                        require_all_content_uuids=require_all_content_uuids,
                    ),
                }
            else:
                raise TypeError(f"vLLM training request must be str or mapping, got {type(request).__name__}")
            proofs.append({"sample_id": prompt.id, **request_proof})
        return {"group": group, "sampling": sampling, "requests": proofs}

    def _parse_training_requests(
        self,
        prompts: list[ObsBundle],
        request_outputs: list[Any],
        group: int,
    ) -> list[Trajectory]:
        """Parse a complete training batch locally before publishing rank success."""
        if len(request_outputs) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(request_outputs)} training request(s) for {len(prompts)} prompt(s)")
        trajectories: list[Trajectory] = []
        for pi, (obs, req) in enumerate(zip(prompts, request_outputs, strict=True)):
            prompt_ids = torch.tensor(req.prompt_token_ids, dtype=torch.long)  # [P]
            # group_id = obs.id (per-prompt uuid), NOT pi: batch-local indices collide across replicas
            # at fan-in and merge distinct prompts' GRPO groups (roles/rollout/base.py::group_ids_for_prompt).
            group_id, traj_ids = group_ids_for_prompt(obs, group)
            # Group integrity: vLLM returns exactly `group` completions for this prompt. A hard raise
            # (not assert) so the check survives python -O.
            if len(req.outputs) != group:
                raise RuntimeError(f"vLLM returned {len(req.outputs)} != request group={group} for prompt {pi}")
            for ki, out in enumerate(req.outputs):
                completion_ids = [int(token_id) for token_id in out.token_ids]
                comp_ids = torch.tensor(completion_ids, dtype=torch.long)  # [T]
                if not isinstance(out.text, str):
                    raise RuntimeError(f"vLLM completion text must be a string, got {type(out.text).__name__}")
                policy_info: dict[str, Any] = {
                    "completion_text": out.text,
                    "completion_token_count": len(completion_ids),
                    # Exact realized ids stay inline even when `action` is externalized to a BulkRef.
                    # A text-only cross-token teacher can therefore align the completion without
                    # materializing the image/video fields carried by the same ObsBundle.
                    "completion_ids": completion_ids,
                    "completion_boundary": {
                        "prompt_token_count": len(req.prompt_token_ids),
                        "completion_token_count": len(completion_ids),
                        "full_sequence_token_count": len(req.prompt_token_ids) + len(completion_ids),
                    },
                    "prompt_ids": prompt_ids,
                    "weight_version": self._weight_version,
                    # π_old, per generated token (§2.4). Field name matches HFRollout so the
                    # clipped objective reads one key for both.
                    "rollout_logprobs": self._token_logprobs(out),
                    # The engine's verdict on how this completion ended: "stop" | "length" (cut at
                    # max_tokens / the model length) | another engine string; None if unreported.
                    FINISH_REASON_KEY: self._finish_reason(out),
                }
                if self._record_routed_experts:
                    policy_info[ROLLOUT_ROUTED_EXPERTS_KEY] = self._routed_experts(
                        out, prompt_token_count=len(req.prompt_token_ids)
                    )  # [P+N-1,L,K]
                if self._topk_logprobs:
                    # The sampler's head per generated token (score centering): the k ids the engine
                    # ranked highest and its log q at them, rank order, NOT renormalised; the sampled
                    # token stays in `rollout_logprobs` whether or not it is one of them.
                    head_ids, head_logprobs = self._topk_head(out, self._topk_logprobs)  # [T,k], [T,k]
                    policy_info[ROLLOUT_TOPK_IDS_KEY] = head_ids
                    policy_info[ROLLOUT_TOPK_LOGPROBS_KEY] = head_logprobs
                tr = Transition(
                    obs=obs,
                    action=comp_ids,
                    policy_info=policy_info,
                    done=True,
                )
                trajectories.append(
                    Trajectory(
                        transitions=[tr],
                        id=traj_ids[ki],
                        group_id=group_id,
                        status=Status.COMPLETED,
                    )
                )
        return trajectories

    @staticmethod
    def _training_output_identity(trajectories: list[Trajectory]) -> dict[str, Any]:
        """Project tensor-free training metadata so every TP lane must agree before publication."""
        rows: list[dict[str, Any]] = []
        for trajectory in trajectories:
            transition = trajectory.transitions[0]
            policy_info = transition.policy_info
            prompt_ids = policy_info["prompt_ids"]
            rows.append(
                {
                    "trajectory_id": trajectory.id,
                    "group_id": trajectory.group_id,
                    "prompt_ids": prompt_ids.tolist(),
                    "completion_ids": list(policy_info["completion_ids"]),
                    "completion_text": policy_info["completion_text"],
                    "completion_boundary": dict(policy_info["completion_boundary"]),
                    "rollout_logprobs_hex": [float(value).hex() for value in policy_info["rollout_logprobs"]],
                    "weight_version": policy_info["weight_version"],
                    "finish_reason": policy_info.get(FINISH_REASON_KEY),
                }
            )
            routing = policy_info.get(ROLLOUT_ROUTED_EXPERTS_KEY)  # [P+N-1,L,K] | None
            if routing is not None:
                rows[-1]["rollout_routed_experts_sha256"] = hashlib.sha256(routing.numpy().tobytes()).hexdigest()
            head_ids = policy_info.get(ROLLOUT_TOPK_IDS_KEY)  # [T,k] | None
            if head_ids is not None:
                head_logprobs = policy_info[ROLLOUT_TOPK_LOGPROBS_KEY]  # [T,k]
                rows[-1]["rollout_topk_sha256"] = hashlib.sha256(
                    head_ids.numpy().tobytes() + head_logprobs.numpy().tobytes()
                ).hexdigest()
        return {"trajectories": rows}

    def _generate_validation(
        self,
        prompts: list[ObsBundle],
        validation: ValidationRolloutContext,
    ) -> list[Trajectory]:
        """Run one held-out vLLM batch with rank-converged preparation and parsing.

        Under ``external_launcher`` every TP rank invokes the facade and generation uses that
        rollout replica's default process group. Prompt rendering, explicit tokenization, request
        construction, and RNG setup therefore finish on every rank before any facade enters
        ``model.generate``. Training generation stays on the original path above.
        """

        def _prepare_sampling() -> dict[str, Any]:
            if not isinstance(validation, ValidationRolloutContext):
                raise TypeError(f"validation must be ValidationRolloutContext or None, got {type(validation).__name__}")
            self._require_published_weights()
            self.require_validation_weight_version(validation)
            sampling = dict(self.sampling)
            sampling["repetition_penalty"] = validation.repetition_penalty
            sampling.pop("seed", None)  # validation owns one stable seed per sample/generation request
            if validation.temperature == 0.0:
                # vLLM's greedy spelling is temperature=0. top-p/top-k do not participate in
                # greedy decoding, so remove inherited training filters unless validation names
                # them explicitly below.
                sampling["temperature"] = 0.0
                sampling.pop("top_p", None)
                sampling.pop("top_k", None)
            else:
                sampling["temperature"] = validation.temperature
            if validation.top_p is not None:
                sampling["top_p"] = validation.top_p
            elif validation.temperature > 0.0:
                sampling["top_p"] = 1.0
            if validation.top_k is not None:
                sampling["top_k"] = validation.top_k
            elif validation.temperature > 0.0:
                sampling["top_k"] = -1
            if validation.max_tokens is not None:
                sampling["max_tokens"] = validation.max_tokens
            return sampling

        sampling = run_validation_stage(_prepare_sampling, "vLLM validation runtime preparation")
        if not prompts:
            return []

        with distributed_validation_rng("vLLM validation rollout"):

            def _prepare_batch() -> tuple[
                list[tuple[ObsBundle, int, dict[str, Any]]],
                list[dict[str, Any]],
                list[Any],
                dict[str, Any],
            ]:
                from vllm import SamplingParams

                rendered: list[tuple[ObsBundle, dict[str, Any]]] = []
                for obs in prompts:
                    seed_validation_rng(validation_prompt_seed(validation.seed, obs.id))
                    rendered.append((obs, self._prepare_validation_request(obs)))
                request_inputs = [
                    (obs, generation_index, dict(prompt))
                    for obs, prompt in rendered
                    for generation_index in range(validation.n_generation)
                ]
                params = [
                    SamplingParams(
                        n=1,
                        logprobs=None,
                        seed=validation_generation_seed(validation.seed, obs.id, generation_index),
                        **sampling,
                    )
                    for obs, generation_index, _prompt in request_inputs
                ]
                prompt_inputs = [prompt for _obs, _generation_index, prompt in request_inputs]
                request_identity = {
                    "sampling": sampling,
                    "requests": [
                        {
                            "sample_id": obs.id,
                            "generation_index": generation_index,
                            "prompt_token_ids": prompt["prompt_token_ids"],
                            "seed": validation_generation_seed(validation.seed, obs.id, generation_index),
                            **vllm_request_media_identity(
                                prompt,
                                context="vLLM validation request",
                                require_all_content_uuids=world_size_or_1() > 1,
                            ),
                        }
                        for obs, generation_index, prompt in request_inputs
                    ],
                }
                return request_inputs, prompt_inputs, params, request_identity

            request_inputs, prompt_inputs, params, request_identity = run_validation_stage(
                _prepare_batch,
                "vLLM validation batch preflight",
            )
            run_validation_stage(
                lambda: _require_identical_rank_identity(
                    request_identity,
                    context="vLLM validation request identity",
                ),
                "vLLM validation request convergence",
            )
            request_outputs = run_validation_stage(
                lambda: self.model.generate(prompt_inputs, params),
                "vLLM validation model generation",
            )

        return run_validation_stage(
            lambda: self._parse_validation_requests(
                request_inputs=request_inputs,
                request_outputs=request_outputs,
                validation=validation,
            ),
            "vLLM validation output parsing",
        )

    def _prepare_validation_request(self, obs: ObsBundle) -> dict[str, Any]:
        """Pretokenize one validation prompt while retaining model-owned request side inputs."""
        shaped = self.packer.get_rollout_input(obs)
        if isinstance(shaped, str):
            request: dict[str, Any] = {}
            prompt: str | None = shaped
            raw_prompt_token_ids: Any = None
        elif isinstance(shaped, Mapping):
            request = dict(shaped)
            prompt_value = request.get("prompt")
            raw_prompt_token_ids = request.get("prompt_token_ids")
            if (prompt_value is None) == (raw_prompt_token_ids is None):
                raise RuntimeError("vLLM validation request must carry exactly one of prompt or prompt_token_ids")
            if prompt_value is not None and not isinstance(prompt_value, str):
                raise TypeError(f"vLLM validation prompt must be a string, got {type(prompt_value).__name__}")
            prompt = prompt_value
        else:
            raise TypeError(f"vLLM validation request must be str or mapping, got {type(shaped).__name__}")

        request.pop("prompt", None)
        if prompt is not None:
            encoded = self.packer.tokenizer(prompt, add_special_tokens=True)
            raw_prompt_token_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if isinstance(raw_prompt_token_ids, (str, bytes)):
            raise TypeError("vLLM validation prompt_token_ids must be an integer sequence")
        try:
            prompt_token_ids = [int(token_id) for token_id in raw_prompt_token_ids]
        except (TypeError, ValueError) as error:
            raise TypeError("vLLM validation prompt_token_ids must be an integer sequence") from error
        if not prompt_token_ids:
            raise ValueError("vLLM validation prompt_token_ids must not be empty")
        request["prompt_token_ids"] = prompt_token_ids
        return request

    def _parse_validation_requests(
        self,
        *,
        request_inputs: list[tuple[ObsBundle, int, dict[str, Any]]],
        request_outputs: list[Any],
        validation: ValidationRolloutContext,
    ) -> list[Trajectory]:
        """Parse the complete validation batch locally before reporting rank success."""
        if len(request_outputs) != len(request_inputs):
            raise RuntimeError(
                f"vLLM returned {len(request_outputs)} validation request(s) != expected {len(request_inputs)}"
            )
        trajectories: list[Trajectory] = []
        for prompt_index, ((obs, generation_index, _prompt), request) in enumerate(
            zip(request_inputs, request_outputs)
        ):
            prompt_ids = torch.tensor(request.prompt_token_ids, dtype=torch.long)  # [P]
            group_id, trajectory_ids = group_ids_for_prompt(obs, validation.n_generation)
            if len(request.outputs) != 1:
                raise RuntimeError(
                    f"vLLM returned {len(request.outputs)} != request group=1 for validation prompt {prompt_index}"
                )
            output = request.outputs[0]
            inline_completion_ids = [int(token_id) for token_id in output.token_ids]
            completion_ids = torch.tensor(inline_completion_ids, dtype=torch.long)  # [T]
            policy_info: dict[str, Any] = {
                "completion_text": output.text,
                "completion_token_count": len(inline_completion_ids),
                "completion_ids": inline_completion_ids,
                "completion_boundary": {
                    "prompt_token_count": len(request.prompt_token_ids),
                    "completion_token_count": len(inline_completion_ids),
                    "full_sequence_token_count": len(request.prompt_token_ids) + len(inline_completion_ids),
                },
                "prompt_ids": prompt_ids,
                "weight_version": self._weight_version,
                "validation_step": validation.model_step,
                FINISH_REASON_KEY: self._finish_reason(output),
            }
            transition = Transition(
                obs=obs,
                action=completion_ids,
                policy_info=policy_info,
                done=True,
            )
            trajectories.append(
                Trajectory(
                    transitions=[transition],
                    id=trajectory_ids[generation_index],
                    group_id=group_id,
                    status=Status.COMPLETED,
                )
            )
        return trajectories

    @staticmethod
    def _routed_experts(out: Any, *, prompt_token_count: int) -> torch.Tensor:
        """The engine's top-k expert ids for every input position of prompt + completion, ``[P+N-1,L,K]``.

        vLLM's ``CompletionOutput.routed_experts`` (``[seq_len, layer, top_k]``) covers the prompt at
        prefill and each sampled token when it is fed back, so a completion of N tokens yields
        P + N - 1 rows: the last sampled token is never an input and has no row. The engine sizes the
        ids by its expert count (uint8 up to 256 experts, uint16 beyond); uint8 is kept and anything
        wider is narrowed to int16, the two dtypes the packer and the MoE adapter accept.

        Raises:
            RuntimeError: routing is missing (the engine was not built to return it), the completion
                is empty, or the row count does not match the sequence.
        """
        routed = out.routed_experts
        if routed is None:
            raise RuntimeError(
                "vLLM returned no routed experts; rl.routing_replay='rollout' needs an engine built with "
                "enable_return_routed_experts=True (build_vllm_rollout(return_routed_experts=True))"
            )
        completion_token_count = len(out.token_ids)
        if completion_token_count == 0:
            raise RuntimeError("vLLM returned an empty completion, which has no routed experts to replay")
        array = np.asarray(routed)  # [P+N-1,L,K]
        expected_rows = prompt_token_count + completion_token_count - 1
        if array.ndim != 3 or array.shape[0] != expected_rows:
            raise RuntimeError(
                f"vLLM returned routed experts of shape {tuple(array.shape)} for a {prompt_token_count}-token "
                f"prompt and a {completion_token_count}-token completion; expected [{expected_rows}, layers, top_k] "
                "(one row per input position, the last sampled token is never fed back)"
            )
        if array.dtype == np.uint8:
            return torch.from_numpy(array.copy())
        if not np.issubdtype(array.dtype, np.integer) or (array.size and (array.min() < 0 or array.max() >= 2**15)):
            raise RuntimeError(f"vLLM routed experts must be non-negative int16-range ids, got {array.dtype}")
        return torch.from_numpy(array.astype(np.int16))

    @staticmethod
    def _finish_reason(out: Any) -> str | None:
        """vLLM's ``CompletionOutput.finish_reason`` as a plain string, None when the engine set none.

        Recorded verbatim rather than validated against a vocabulary: the consumers key on the one
        value that matters (``"length"``, see ``schema.is_length_truncated``) and a future engine
        string must surface in the record, not abort the batch (`roles/rollout/ppo.py` refuses
        instead because PPO derives its termination flags from it).
        """
        reason = getattr(out, "finish_reason", None)
        return None if reason is None else str(reason)

    @staticmethod
    def _topk_head(out: Any, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:  # returns [T,k] int32, [T,k] fp32
        """The engine's top-k head per generated token, in rank order, from `out.logprobs`.

        The protocol (exactly the ranks 1..k once each per row, positive integer ranks, non-negative ids,
        finite-or-``-inf`` logprobs <= 1e-6) is `roles/rollout/sampler_head.py::parse_ranked_rows`, the
        same parser `roles/rollout/ppo.py` uses for its behavior head; a violation raises
        `SamplerHeadProtocolError` (a RuntimeError). Ids are stored as int32, logprobs as float32 --
        never float16: rounding the head on the wire made q != p for an IDENTICAL sampler and trainer (a
        128-token head of mass 0.99 with q = p gave rho 1.11878 and |correction| 0.578 under float16,
        exactly zero under float32). A logprob of -inf (a token the engine masked out) is kept: it is a
        zero sampler mass.
        """
        if out.logprobs is not None and len(out.logprobs) != len(out.token_ids):
            raise RuntimeError(
                f"vLLM returned {len(out.logprobs)} logprob row(s) for {len(out.token_ids)} sampled token(s)"
            )
        ids_rows, lp_rows, _ = parse_ranked_rows(out.logprobs, top_k)
        ids = torch.tensor(ids_rows, dtype=torch.int32).reshape(len(ids_rows), top_k)  # [T,k]
        lps = torch.tensor(lp_rows, dtype=torch.float32).reshape(len(lp_rows), top_k)  # [T,k]
        return ids, lps

    @staticmethod
    def _token_logprobs(out: Any) -> list[float]:
        """Per-token logprob of the *sampled* tokens (π_old). vLLM's `out.logprobs` is a
        list (one dict per generated position) keyed by token id → Logprob. Fail loud if
        logprobs are MISSING (None ⇒ SamplingParams(logprobs=...) not set) rather than silently
        emitting an empty π_old — but an empty LIST is a valid zero-token completion, not an error."""
        if out.logprobs is None:
            raise RuntimeError("vLLM returned no logprobs; SamplingParams(logprobs=...) must be set")
        if len(out.logprobs) != len(out.token_ids):
            raise RuntimeError(
                f"vLLM returned {len(out.logprobs)} logprob row(s) for {len(out.token_ids)} sampled token(s)"
            )
        lps: list[float] = []
        for tid, lp_dict in zip(out.token_ids, out.logprobs, strict=True):
            entry = lp_dict.get(tid)
            if entry is None:  # sampled token must always be in its own position's logprob dict
                raise RuntimeError(f"sampled token {tid} missing from vLLM logprob dict {list(lp_dict)}")
            logprob = float(entry.logprob)
            if not math.isfinite(logprob) or logprob > 1.0e-6:
                raise RuntimeError(f"sampled token {tid} has invalid vLLM logprob {logprob!r}")
            lps.append(logprob)
        return lps
