# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared contract for training-time rollout bodies.

Rollouts generate grouped ``Trajectory`` completions and passively receive producer-driven weight
updates. Training uses ``weight_sync(rollout, policy)``; ``apply_weights`` remains the manual
peer-pull surface. Stream readiness and rollout-wide version publication are separate phases.
Per-call ``group``, ``seed``, and ``validation`` knobs keep implementations
interchangeable. Outcome-only ``PolicyBody`` inference is a separate contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, WeightWitnessSpec, shared_model_identity
from cosmos_framework.posttrain.roles.rollout.validation import ValidationRolloutContext
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory


def group_ids_for_prompt(obs: ObsBundle, group: int) -> tuple[str, list[str]]:
    """Return the stable GRPO group and completion ids for one prompt.

    ``obs.id`` avoids replica-local index collisions; an empty id fails before groups can collapse.
    """
    if not obs.id:
        raise ValueError(
            "ObsBundle.id is empty; the data role must stamp a unique id per prompt (the GRPO "
            "group_id namespace — see roles/env/dataset.py). An empty id collapses every prompt "
            "into one group and silently corrupts the per-prompt advantage mean/std."
        )
    return obs.id, [f"{obs.id}_k{ki}" for ki in range(group)]


@dataclass(frozen=True)
class ShardSpec:
    """This rank's half-open parameter slice along ``dim``."""

    offset: int
    length: int
    dim: int


class WeightReceiverBase:
    """Shared passive receiver surface for rollout weight synchronization."""

    model: Any  # the sink the backend loads into, or the engine handle an engine sink stands in for
    _weight_version: int
    _pending_weight_version: int | None
    _prepared_weight_version: int | None
    _weight_broadcast_spec: Any
    _loaded_names: set[str] | None  # per-stream coverage accumulator; None once unverifiable

    def bind_weight_backend(self, backend: WeightSync) -> None:
        """Take ownership of a weight backend and initialize this receiver's engine-side resources."""
        current = getattr(self, "_weight_backend", None)
        if current is backend:
            self.initialize_weight_receiver()
            return
        if current is not None:
            backend.close()
            raise RuntimeError("a weight backend is already bound; replacing it online is unsupported")
        self._weight_backend = backend
        try:
            self.initialize_weight_receiver()
        except BaseException:
            self._weight_backend = None
            backend.close()
            raise

    def initialize_weight_receiver(self) -> None:
        """Initialize engine-side receiving resources at bind; ordinary in-process sinks need none."""

    def require_validation_weight_version(self, context: ValidationRolloutContext) -> None:
        """Fail before generation unless this rollout holds the requested committed weights."""
        actual = int(getattr(self, "_weight_version", -1))
        if actual != context.model_step:
            raise RuntimeError(
                f"validation pinned model_step={context.model_step}, but rollout holds weight version {actual}"
            )

    def sleep(self) -> None:
        """Give this body's GPU memory to the colocated trainer for the update phase.

        The default is a no-op: a rollout that shares the trainer's model instance (HF, WFM) owns
        no memory of its own to release. Engine-backed rollouts override it.
        """
        return None

    def wake_up(self, tags: tuple[str, ...] | None = None) -> None:
        """Restore GPU state released by ``sleep``; ``tags`` narrows to part of it. No-op by default."""
        del tags
        return None

    def discard_payload(self, _payload: list[Trajectory]) -> None:
        """Consume and drop a completed rollout payload at its owning rollout cell.

        The body intentionally does nothing: for a bulk-consuming rollout WorkerGroup, the actor
        transport materializes the argument before this method runs, which consumes every local
        ``BulkRef`` and releases its producer-store ownership without sending tensors through the
        controller. Held-out validation has no replay/trainer consumer, so this is its explicit
        terminal sink.
        """
        return None

    def shared_weight_identity(self) -> str | None:
        """Identity proof for byte-free shared/no-backend weight edges; real streams need none."""
        backend = getattr(self, "_weight_backend", None)
        if backend is not None:
            from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

            if not isinstance(backend.mover, SharedMover):
                return None
        return shared_model_identity(self.model)

    def prepare_weight_version(self, version: int, producer_identity: str | None) -> None:
        """Validate readiness without publishing a new rollout version.

        ``weight_sync`` calls this on every rollout cell only after every policy push has drained.
        No cell mutates ``_weight_version`` in this phase, so one missing/failed lane prevents the
        later commit everywhere rather than leaving a partially advertised mesh.
        """
        version = int(version)
        actual = int(getattr(self, "_weight_version", -1))
        backend = getattr(self, "_weight_backend", None)
        broadcast_spec = getattr(self, "_weight_broadcast_spec", None)
        from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

        has_stream = backend is not None and not isinstance(backend.mover, SharedMover)
        if has_stream or broadcast_spec is not None:
            pending = getattr(self, "_pending_weight_version", None)
            if pending != version:
                raise RuntimeError(
                    f"weight-sync barrier reached version {version}, but this rollout lane is ready for {pending!r}"
                )
        if not has_stream and broadcast_spec is None:
            consumer_identity = self.shared_weight_identity()
            if producer_identity is None or consumer_identity != producer_identity:
                raise RuntimeError(
                    "byte-free weight sync requires rollout and policy to share the exact model object; "
                    f"producer identity={producer_identity!r}, rollout identity={consumer_identity!r}"
                )
        if version < actual:
            raise RuntimeError(f"weight version confirmation regressed from {actual} to {version}")
        self._prepared_weight_version = version

    def commit_weight_version(self, version: int) -> None:
        """Publish a version after every rollout lane passed ``prepare_weight_version``."""
        version = int(version)
        prepared = getattr(self, "_prepared_weight_version", None)
        if prepared != version:
            raise RuntimeError(
                f"weight version {version} was not prepared on this rollout lane (prepared={prepared!r})"
            )
        self._weight_version = version
        self._pending_weight_version = None
        self._prepared_weight_version = None

    def set_bucket_plan(self, plan: Any) -> None:
        be = getattr(self, "_weight_backend", None)
        if be is not None:
            be.set_bucket_plan(plan)

    def release_weight_buffer(self) -> None:
        """Release transport staging state after every bucket has landed."""
        be = getattr(self, "_weight_backend", None)
        if be is not None:
            be.release_staged()

    def set_weight_destination_names(self, names: tuple[str, ...]) -> None:
        """Install producer-mapped live sink names before plan coverage is checked."""
        be = getattr(self, "_weight_backend", None)
        if be is not None:
            be.set_destination_names(names)

    def check_plan_coverage(self) -> None:
        """Producer-driven receiver guard: verify the pushed plan covers this rollout's model 1:1
        (strict), the symmetric check to `pull_stream_from_peer`'s pre-stream coverage. No backend ⇒
        no-op. A state_dict-less engine sink is exempt here because its checkpoint names can differ
        from live fused/TP-sharded names; ``VLLMRollout`` overrides this method when the producer
        supplies mapper-proven destinations."""
        from cosmos_framework.posttrain.comm.weight_transfer import _strict_plan_coverage

        be = getattr(self, "_weight_backend", None)
        if be is not None:
            _strict_plan_coverage(self, be)

    # --- streaming weight sync (per-bucket): the receive skeleton every engine-sink body shares ---
    def _engine_sink(self, state: dict) -> set[str] | None:
        """Load one fetched bucket (``{wire_name: tensor}``) into the live engine.

        Return the names actually loaded, or None if the engine cannot report them. The receiver owns
        coverage accumulation; an unverifiable bucket makes the entire stream unverifiable until reset.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _engine_sink()")

    def _load_engine_bucket(self, state: dict) -> None:
        """Accumulate the engine's reported result without teaching the transport about coverage."""
        loaded = self._engine_sink(state)
        current = getattr(self, "_loaded_names", None)
        if current is not None:
            self._loaded_names = None if loaded is None else current | loaded

    def _engine_param_names(self) -> set[str]:
        """The live engine's parameter names, in wire naming, for the post-stream coverage check."""
        raise NotImplementedError(f"{type(self).__name__} must implement _engine_param_names()")

    def apply_bucket(self, payload: Any, i: int) -> None:
        """Consumer side of one streaming round: fetch ONLY bucket ``i`` and load it through ``_engine_sink``.

        PURE LOAD, fully delegated: the receive mechanics (staged fetch, consumer_done) live in
        ``WeightSync.apply_bucket`` and the version protocol in ``push_stream_to_peer``. ``i == 0`` resets the
        coverage accumulator; the last bucket runs the coverage check. No backend ⇒ no-op. An engine that
        brackets the stream with its own reload protocol (vLLM's MoE layerwise reload) overrides this and
        keeps the same three steps around its bracket.

        Args:
            payload: the streaming round's transport payload for this bucket (backend-specific).
            i: bucket index within the stream.
        """
        be = getattr(self, "_weight_backend", None)
        if be is None:
            return
        self._coverage_reset(i)
        be.apply_bucket(self.model, i, payload, sink=self._load_engine_bucket)
        self._coverage_check(be, i)

    def _coverage_reset(self, i: int) -> None:
        """Reset the per-stream coverage accumulator at the start of a new stream.

        Args:
            i: bucket index; the accumulator is cleared only when ``i == 0`` (first bucket).
        """
        # Per-stream coverage accumulator resets on the FIRST bucket (names loaded by an earlier
        # stream must not mask a gap in the current one).
        if i == 0:
            self._loaded_names = set()

    def _coverage_check(self, be: Any, i: int) -> None:
        """After the last bucket, fail loud if any engine param was never loaded by the stream.

        Args:
            be: the weight backend, queried for ``bucket_count()`` to detect the last bucket.
            i: bucket index; the check runs only on the final bucket. A stream whose sink reported no
                loaded names is unverifiable and goes to ``_check_unverifiable_coverage`` instead.
        """
        if i != be.bucket_count() - 1:
            return
        if self._loaded_names is None:
            self._check_unverifiable_coverage(be)
            return
        missing = sorted(self._engine_param_names() - self._loaded_names)
        if missing:
            raise ValueError(
                f"weight-sync stream did not cover the engine: {len(missing)} param(s) never "
                f"loaded ({missing[:5]}{'...' if len(missing) > 5 else ''}) — the bucket plan is "
                f"missing params the engine has (resized/retied/truncated plan?); these would "
                f"silently keep their initial-checkpoint weights"
            )

    def _check_unverifiable_coverage(self, be: Any) -> None:
        """Decide a completed stream whose sink reported no loaded names; the default accepts it.

        Args:
            be: the weight backend that streamed the unverifiable plan.
        """
        del be

    def set_weight_witness_spec(self, spec: WeightWitnessSpec) -> None:
        """Install mapper witness metadata before bucket 0."""
        self._weight_witness_spec = spec

    def require_weight_witness_verified(self, spec: WeightWitnessSpec) -> None:
        """Reject semantic witnesses on sinks without an inline verification adapter."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot verify mapped live weight witness {spec.destination_name!r}"
        )

    def mark_weight_version_ready(self, version: int) -> None:
        """Record that this lane loaded a complete stream without publishing it yet."""
        if getattr(self, "_weight_backend", None) is None and getattr(self, "_weight_broadcast_spec", None) is None:
            raise ValueError(
                "mark_weight_version_ready on a rollout with no weight backend bound and no replica broadcast "
                "bound: refusing to advertise a fresh version over an unsynced model"
            )
        version = int(version)
        actual = int(getattr(self, "_weight_version", -1))
        if version < actual:
            raise RuntimeError(f"weight readiness regressed from published version {actual} to {version}")
        pending = getattr(self, "_pending_weight_version", None)
        if pending is not None and pending != version:
            raise RuntimeError(f"rollout lane already has pending weight version {pending}, cannot mark {version}")
        self._pending_weight_version = version

    def broadcast_live_weights(self, version: int, *, verify_values: bool = False) -> str:
        """Copy the source replica's live model tensors into this corresponding-rank lane."""
        from cosmos_framework.posttrain.comm.weight_broadcast import broadcast_live_model

        spec = getattr(self, "_weight_broadcast_spec", None)
        if spec is None:
            raise RuntimeError("rollout replica broadcast is not bound")
        digest = broadcast_live_model(self.model, spec, verify_values=verify_values)
        self.mark_weight_version_ready(version)
        return digest

    def initialize_weight_broadcast(self) -> None:
        """Build this lane's side process group in the process that owns its live model."""
        from cosmos_framework.posttrain.comm.weight_broadcast import initialize_replica_broadcast

        spec = getattr(self, "_weight_broadcast_spec", None)
        if spec is None:
            raise RuntimeError("rollout replica broadcast is not bound")
        initialize_replica_broadcast(self.model, spec)


@runtime_checkable
class Rollout(Protocol):
    """Structural contract every training-time rollout body satisfies (generate + apply_weights)."""

    def generate(self, prompts: list[ObsBundle], **knobs: Any) -> list[Trajectory]:
        """Sample completions for each prompt and wrap them as `Trajectory`s. Per-call knobs
        (`group`, `seed`, …) ride in `**knobs`; a body reads what it supports and ignores the rest."""
        ...

    def sleep(self) -> None:
        """Release this body's own GPU memory between generation phases (no-op for shared models)."""
        ...

    def wake_up(self, tags: tuple[str, ...] | None = None) -> None:
        """Restore memory released by ``sleep``; ``tags`` restores only the named parts."""
        ...

    def set_weight_witness_spec(self, spec: WeightWitnessSpec) -> None:
        """Install mapper witness metadata before bucket 0."""
        ...

    def require_weight_witness_verified(self, spec: WeightWitnessSpec) -> None:
        """Require inline live verification before publishing readiness."""
        ...

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Return stable finite JSON describing every non-weight generation semantic."""
        ...

    def apply_weights(self, source: "Any | None" = None) -> None:
        """Import updated trainer weights via the bound backend; no-op when trainer and rollout
        share a model instance (the SharedRef no-op, §8.2 #4). PEER-ONLY: ``source`` is the policy
        peer (WorkerGroup/PeerClient/body), or None for the degenerate stamp-only contracts. This is
        the KEPT consumer-PULL surface for manual/NCCL rigs, NOT the training-loop entry (the loop uses
        producer-driven ``weight_sync``/``push_weights`` — see module docstring). The WeightPayload
        batch form was removed (see module docstring)."""
        ...

    def mark_weight_version_ready(self, version: int) -> None:
        """Record a complete local stream without publishing it."""
        ...

    def prepare_weight_version(self, version: int, producer_identity: str | None) -> None:
        """Validate local readiness during the rollout-wide read-only phase."""
        ...

    def commit_weight_version(self, version: int) -> None:
        """Publish after every rollout lane completed the prepare phase."""
        ...

    def prepare_recv(self) -> dict[str, ShardSpec]:
        """Report THIS rank's receive layout per param as a `ShardSpec` — the backend's own shard,
        in the shared planner's terms (§2.4). v1 (no TP / gather-then-slice): the whole tensor
        `ShardSpec(0, dim0_len, 0)`. vLLM/SGLang report their real TP slice so v2 can send only that."""
        ...
