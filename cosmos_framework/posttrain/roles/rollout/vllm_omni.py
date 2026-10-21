# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rollout role -- the model-agnostic half of serving a policy on a vLLM-Omni engine.

Peer of ``roles/rollout/vllm.py`` for diffusion/multimodal policies. It owns everything that is the
same for any Omni-served model -- engine construction, the worker RPC, the ``infer``/``infer_batch``
contract that ``the external native policy rollout body`` also satisfies, and the receiving half of the
weight plane -- and leaves the model contract to subclass hooks.

Weight plane. The body is a ``WeightReceiverBase`` like ``VLLMRollout``; the difference is that the live
parameters sit in vLLM-Omni's diffusion worker, a separate process on the same GPU. At weight-backend binding the
worker exports a CUDA-IPC handle for every live parameter once per worker life (``roles/rollout/vllm_omni_worker.py``) and this
body maps them (``_import_live_parameters``). A pushed bucket arrives as owned tensors (the first hop, the
bound mover) and ``_engine_sink`` copies each one straight into the mapped parameter on this process's stream
and waits for the copies to land -- the second hop is a D2D copy, not a transfer, and no per-bucket RPC.
``verify_weight_values`` decides whether anything reads those bytes back (below). Per publication the worker
sees two RPCs: ``begin_weight_update`` before bucket 0 (the mapping is still the live one, the worker is idle)
and ``finish_weight_update`` before readiness (the same again; when verifying, an independent SHA256 re-read of
every live parameter against the digests this body took of the wire bytes, otherwise a bare epoch close).
vLLM-Omni answers a ``collective_rpc`` only after every rank's status was gathered, so each reply is an
"all workers done" barrier.

Inference is refused while a stream is open or after a failed one (``engine_invalidated``), for every
``weight_version`` including ``None``: sim callers pass no version, and a lane that already loaded new
weights under an uncommitted version would otherwise serve a mixture as the old policy. Recovery is
``rebuild_engine()`` followed by a full sync whose commit reopens inference.

What deliberately does NOT live here: the request payload. Omni requests are multimodal and their
schema is part of the model, not a knob, so ``_build_request`` builds the sampling-params object
itself. That also keeps ``OmniDiffusionSamplingParams`` out of this module, so importing it costs no
vLLM import (see ``tests/test_roles_rollout_vllm_omni.py``); ``Omni`` is imported inside ``__init__``
for the same reason.

Serving is outcome-only (one observation batch in, one action chunk out), which is a different
contract from ``roles/rollout/base.py::Rollout`` and its per-step ``Trajectory``; see that module's
docstring.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.posttrain.comm.transfer import ipc
from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata
from cosmos_framework.posttrain.comm.weight_validation import check_weight_names, copy_named_tensors
from cosmos_framework.posttrain.roles.rollout.base import WeightReceiverBase


@dataclass(frozen=True)
class BucketSinkTiming:
    """Wall clock of one bucket through ``OmniPolicyBody._engine_sink`` (``bucket_timings``).

    ``copy_s`` covers bucket validation and launching D2D copies into the mapped parameters; ``sync_s`` waits for them to land;
    ``compare_s`` reads the mapped memory back byte for byte and ``digest_s`` is the per-bucket SHA256 of the wire
    bytes -- both 0.0 with ``verify_weight_values`` off, where neither runs.
    """

    copy_s: float
    sync_s: float
    compare_s: float
    digest_s: float


@dataclass(frozen=True)
class VerifyTiming:
    """Wall clock of the end-of-stream ``finish_weight_update`` round trip (``verify_timing``): ``worker_s`` is the
    worker's own time inside it, ``digest_s`` the part spent on its SHA256 re-read (None when the body did not ask),
    ``verified`` the number of parameters that re-read covered (None with ``verify_weight_values`` off)."""

    rpc_s: float
    worker_s: float | None
    digest_s: float | None
    verified: int | None


class OmniPolicyBody(WeightReceiverBase):
    """A policy served by a vLLM-Omni engine, behind the sim-eval ``infer``/``infer_batch`` contract.

    Subclasses implement ``_engine_kwargs`` / ``_build_request`` / ``_parse_output`` and may add startup
    verification in ``_after_engine_start`` -- the engine, and therefore ``_worker_rpc``, is live when it runs.
    """

    #: Seconds a weight RPC may take before the body treats the worker as hung. The bracket RPCs do bookkeeping and,
    #: when verifying, one digest pass over the live head; a hung worker must become a failure, not a stuck loop.
    WEIGHT_RPC_TIMEOUT_S: float = 600.0

    def __init__(
        self,
        *,
        transfer_dtype: str | None = None,
        verify_weight_values: bool = False,
        **engine_kwargs: Any,
    ) -> None:
        """Start the engine and verify it runs on this body's GPU.

        Args:
            transfer_dtype: the wire dtype the producer declares (``Precision.transfer``). Read off the BODY by
                ``workers/ray.py::set_weight_backend`` / ``init_nixl`` when sizing staging, exactly as
                ``LLMTrainer.transfer_dtype`` is on the producer side.
            verify_weight_values: the single switch over every byte-level check of a publication -- the
                byte-wise compare of the mapped memory after each bucket, this body's per-bucket digests of the
                wire bytes, and the worker's independent SHA256 re-read of every live parameter at the end of the
                stream (one CPU digest pass on each side per sync -- the dominant sync cost). With it off a
                publication is backed only by the structural check, the mapping's provenance and the
                begin/finish bracket; `finish_weight_update` still runs, with no expectation to compare.
                Defaults to False. Callers opt in explicitly; the acceptance driver's verification policy
                decides which publications enable it.
            **engine_kwargs: ``Omni(...)`` kwargs; caller-supplied kwargs win over the model's defaults.
        """
        kwargs = self._engine_kwargs()
        kwargs.update(engine_kwargs)
        # Kept verbatim for `rebuild_engine`: the same construction after a failed stream.
        self._engine_kwargs_resolved = dict(kwargs)
        self.transfer_dtype = transfer_dtype
        self.verify_weight_values = bool(verify_weight_values)
        self._weight_version = 0
        self._pending_weight_version = None
        self._prepared_weight_version = None
        self._weight_broadcast_spec = None
        self._stream_open = False
        self.engine_invalidated = False
        self._loaded_names: set[str] | None = set()
        self._engine_names: set[str] | None = None
        self._expected_digests: dict[str, dict[str, Any]] = {}
        # Sync-cost records of the LAST stream, for the acceptance/bench drivers; reset with the digests.
        self.bucket_timings: list[BucketSinkTiming] = []
        self.verify_timing: VerifyTiming | None = None
        # The worker's live parameters mapped into this process over CUDA IPC (`_import_live_parameters`), the
        # export's description of them, and the bracket counter every publication opens on the worker. The
        # mapping aliases the worker's memory, so it is dropped BEFORE the engine on rebuild and close.
        self._live_params: dict[str, torch.Tensor] | None = None
        self._mapping: dict[str, Any] | None = None
        self._update_epoch = 0
        self._omni = None
        # `set_weight_backend` binds a backend only to bodies exposing `.model`, and `resolve_weight_mapper`
        # returns the identity mapper for it; no sink path touches it because `_engine_sink` is injected.
        self.model = None
        try:
            self._omni = self._start_engine()
            self.model = self._omni
            self.worker_device = self._verify_worker_device()
            self._after_engine_start()
        except BaseException:
            self.engine_invalidated = True
            self._close_engine()  # failed construction must release mappings before their owning worker
            raise

    def _start_engine(self) -> Any:
        """Construct the engine with this process's own torch.distributed rendezvous hidden from it.

        Inside a worker actor, ``workers/ray.py::_HostImpl.init_distributed`` leaves ``MASTER_ADDR`` /
        ``MASTER_PORT`` in the environment for the body's own process group, and every replica of a
        ``(W, 1)`` rollout gets the same port. vLLM-Omni's diffusion config gives ``MASTER_PORT`` precedence
        over every other source when it picks its worker's rendezvous port, so W engines on one node would
        all try to bind one port (``EADDRINUSE``, observed 2026-09-09). Hidden only while the engine is built:
        the config resolves its port in this process and the spawned worker inherits the environment at
        spawn time, while the actor's own group read these variables when it initialized.
        """
        from vllm_omni.entrypoints.omni import Omni

        hidden = {key: os.environ.pop(key) for key in ("MASTER_ADDR", "MASTER_PORT") if key in os.environ}
        try:
            return Omni(**self._engine_kwargs_resolved)
        finally:
            os.environ.update(hidden)

    # ── model hooks ──────────────────────────────────────────────────────────
    def _engine_kwargs(self) -> dict[str, Any]:
        """Model-specific ``Omni(...)`` kwargs. Called by ``__init__``, so it may read attributes the
        subclass set before delegating up."""
        raise NotImplementedError(f"{type(self).__name__} must implement _engine_kwargs()")

    def _build_request(self, obs_list: Sequence[dict], seed: int | None) -> tuple[list[str], Any]:
        """Observations -> ``(prompts, sampling_params)``.

        ``seed`` is handed over untouched: where a seed has to land is model-specific. vLLM-Omni's
        generic channel is ``OmniDiffusionSamplingParams.seed``, but a port whose pipeline ignores the
        generator derived from it needs the seed elsewhere, and ``OmniDiffusionRequest`` auto-fills
        ``seed`` when it is unset, so that field cannot express "unseeded" either.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _build_request()")

    def _parse_output(self, output: Any, n: int) -> list[dict]:
        """First ``OmniRequestOutput`` -> one ``{"action": ...}`` per input observation."""
        raise NotImplementedError(f"{type(self).__name__} must implement _parse_output()")

    def _engine_param_layout(self) -> dict[str, dict[str, Any]]:
        """``{wire_name: {"shape": [...], "dtype": str}}`` of the worker's live parameters, as the worker exported
        them (``_import_live_parameters``)."""
        mapping = getattr(self, "_mapping", None)  # getattr: manual rigs bypass __init__
        if mapping is None:
            raise RuntimeError("the worker's live parameters are not mapped; (re)build the engine first")
        return {name: dict(spec) for name, spec in mapping["layout"].items()}

    def _after_engine_start(self) -> None:
        """Model-specific startup verification; runs after construction AND after ``rebuild_engine``."""

    def _reset_engine_caches(self) -> None:
        """Forget model-specific facts read off a worker that no longer exists (``rebuild_engine``)."""

    # ── serving ──────────────────────────────────────────────────────────────
    def infer(self, obs: dict, *, seed: int | None = None, weight_version: int | None = None) -> dict:
        return self.infer_batch([obs], seed=seed, weight_version=weight_version)[0]

    def infer_batch(
        self,
        obs_list: Sequence[dict],
        *,
        seed: int | None = None,
        weight_version: int | None = None,
        sampling_mode: str = "default",
    ) -> list[dict]:
        """One engine call for all N observations; vectorized envs ride ONE batched request."""
        if not obs_list:
            return []
        self._require_serving_allowed()
        self._require_weight_version(weight_version)

        prompts, params = self._build_request(obs_list, seed)
        if sampling_mode != "default":
            self._configure_sampling_mode(params, sampling_mode)
        outputs = self._omni.generate(prompts, params)
        if not outputs:
            raise RuntimeError("the vLLM-Omni engine returned no output for a request")
        if sampling_mode != "default":
            self._validate_sampling_mode(outputs[0], sampling_mode)
        return self._parse_output(outputs[0], len(obs_list))

    def _configure_sampling_mode(self, params: Any, mode: str) -> None:
        """Optional model-owned sampling mode; unsupported requests fail before inference."""
        raise ValueError(f"{type(self).__name__} does not support sampling_mode={mode!r}")

    def _validate_sampling_mode(self, output: Any, mode: str) -> None:
        """A non-default mode must be attested by the model worker, not just requested."""
        raise ValueError(f"{type(self).__name__} cannot verify sampling_mode={mode!r}")

    def _require_serving_allowed(self) -> None:
        """Refuse to serve mid-stream or after a failed stream, whatever version the caller asks for."""
        # getattr: manual rigs build bodies with object.__new__ and set only the attributes a test reads.
        if getattr(self, "_stream_open", False) or getattr(self, "engine_invalidated", False):
            raise RuntimeError(
                "rollout refuses to serve: a weight stream is open or the engine was invalidated by a failed "
                "one, so the live weights are not a committed version (this applies to weight_version=None "
                "too). Every lane stays blocked until rebuild_engine() and a full sync commit."
            )

    def _require_weight_version(self, weight_version: int | None) -> None:
        """Refuse to generate against weights the caller did not ask for (stale = off-policy data)."""
        if weight_version is not None and int(weight_version) != self._weight_version:
            raise RuntimeError(
                f"rollout was asked for weight_version={int(weight_version)}, but holds {self._weight_version}"
            )

    def weight_version(self) -> int:
        return self._weight_version

    # ── worker RPC ───────────────────────────────────────────────────────────
    def _worker_rpc(self, method: str, *args: Any, timeout: float | None = None) -> dict[str, Any]:
        """Call a worker-extension method and return the one mapping the engine replies with.

        ``collective_rpc`` executes on every rank of every stage and nests the replies per stage, then per
        replying rank; this body drives ONE stage whose ONE reply is rank 0's (vLLM-Omni's inline stage
        client fixes ``unique_reply_rank=None``). A worker exception is re-raised by the executor, and the
        stage pool turns it into a ``{"supported": False, "error": ...}`` VALUE -- surfaced here as the
        failure it is rather than returned as a result.
        """
        result: Any = self._omni.engine.collective_rpc(method, timeout=timeout, args=tuple(args))
        depth = 0
        while isinstance(result, (list, tuple)):
            if not result:
                raise RuntimeError(f"{method} returned no worker results (nesting depth {depth})")
            if len(result) != 1:
                raise RuntimeError(
                    f"{method} returned {len(result)} results at nesting depth {depth}; this body drives one "
                    "stage with one replying rank"
                )
            result = result[0]
            depth += 1
        if not isinstance(result, Mapping):
            raise RuntimeError(
                f"{method} returned {type(result).__name__}, expected a mapping (unwrapped {depth} levels)"
            )
        if result.get("supported") is False and "error" in result:
            raise RuntimeError(f"{method} failed in the Omni worker: {result['error']}")
        return dict(result)

    def _verify_worker_device(self) -> dict[str, Any]:
        """Require the worker's CUDA device to be this body's, by index AND by UUID.

        A CUDA IPC handle names a device INDEX and the worker opens it under its own numbering: an equal
        index on a different physical card would map the wrong memory without an error. Run once at
        construction and again after ``rebuild_engine``; no weight RPC before it passed.
        """
        described = self._worker_rpc("describe_device")
        if not torch.cuda.is_available():
            return described
        index = torch.cuda.current_device()
        uuid = str(torch.cuda.get_device_properties(index).uuid)
        if described.get("index") != index or described.get("uuid") != uuid:
            raise RuntimeError(
                f"the vLLM-Omni worker runs on cuda:{described.get('index')} (uuid {described.get('uuid')}), but "
                f"this body holds cuda:{index} (uuid {uuid}); CUDA IPC needs the same physical GPU under the "
                "same device numbering on both sides"
            )
        return described

    # ── weight plane: receiving half ─────────────────────────────────────────
    def initialize_weight_receiver(self) -> None:
        """Export/map only for a bound weight receiver, once per worker life."""
        if self._live_params is not None:
            return
        try:
            self._import_live_parameters()
        except BaseException:
            self._stream_open = True
            self.engine_invalidated = True
            self._close_engine()
            raise

    def _import_live_parameters(self) -> None:
        """Map the worker's live parameters into this process: the second hop's destination for the worker's life.

        The worker exports one CUDA-IPC handle per parameter (once per worker life); this side rebuilds the views,
        checks each against the export's own description and this body's device, and keeps them until
        ``_drop_live_mapping``. Replies come from rank 0 only, so one worker per engine is all a body can write into.
        """
        reply = self._worker_rpc("export_live_parameters", timeout=self.WEIGHT_RPC_TIMEOUT_S)
        if reply.get("world_size") != 1:
            raise RuntimeError(
                f"direct parameter writes need one worker per engine; this engine runs {reply.get('world_size')!r}"
            )
        index: int | None = None
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            uuid = str(torch.cuda.get_device_properties(index).uuid)
            device = reply.get("device") or {}
            if device.get("index") != index or device.get("uuid") != uuid:
                raise RuntimeError(
                    f"the exported parameters live on cuda:{device.get('index')} (uuid {device.get('uuid')}), but this "
                    f"body holds cuda:{index} (uuid {uuid}); CUDA IPC needs the same physical GPU under the same numbering"
                )
        params = reply.get("params")
        if not isinstance(params, Mapping) or not params:
            raise RuntimeError(f"export_live_parameters returned {params!r}, expected the parameter handles")
        live: dict[str, torch.Tensor] = {}
        layout: dict[str, dict[str, Any]] = {}
        view = None
        try:
            for name, spec in params.items():
                view = ipc.tensor_from_ipc(spec["handle"])
                shape = [int(dim) for dim in spec["shape"]]
                dtype = str(spec["dtype"])
                if list(view.shape) != shape or str(view.dtype) != dtype or not view.is_contiguous():
                    raise RuntimeError(
                        f"mapped parameter {name!r} is {tuple(view.shape)} {view.dtype} "
                        f"(contiguous={view.is_contiguous()}), but the worker exported {tuple(shape)} {dtype}"
                    )
                if index is not None and view.device.index != index:
                    raise RuntimeError(f"mapped parameter {name!r} landed on {view.device}, expected cuda:{index}")
                live[name] = view
                layout[name] = {"shape": shape, "dtype": dtype}
        except BaseException:
            # The exception keeps this frame alive: release both the collected views and the pending one
            # before the caller shuts down the process that owns their CUDA storage.
            live.clear()
            view = None
            raise
        self._live_params = live
        self._mapping = {"pid": reply.get("pid"), "layout": layout}
        self._engine_names = set(layout)

    def _drop_live_mapping(self) -> None:
        """Release this process's views of the worker's parameters -- before the worker process goes away."""
        if self._live_params is not None:
            self._live_params.clear()
        self._live_params = None
        self._mapping = None

    def set_bucket_plan(self, plan: Any) -> None:
        """Open the stream: install the plan, block inference, open the worker's update bracket."""
        if getattr(self, "engine_invalidated", False):
            raise RuntimeError(
                "the Omni engine was invalidated by a failed weight stream; call rebuild_engine() before syncing"
            )
        if getattr(self, "_live_params", None) is None:
            raise RuntimeError("the worker's live parameters are not mapped; bind a weight backend before syncing")
        super().set_bucket_plan(plan)
        self._stream_open = True
        self._update_epoch = int(getattr(self, "_update_epoch", 0)) + 1
        try:
            reply = self._worker_rpc("begin_weight_update", self._update_epoch, timeout=self.WEIGHT_RPC_TIMEOUT_S)
            if reply.get("epoch") != self._update_epoch:
                raise RuntimeError(
                    f"begin_weight_update answered epoch {reply.get('epoch')!r}, expected {self._update_epoch}"
                )
        except BaseException:
            # The worker refused (a stale mapping, a protocol error) or hung: nothing may be written into it.
            self.engine_invalidated = True
            raise

    def check_plan_coverage(self) -> None:
        """Before bucket 0: the pushed plan must name the worker's live parameters exactly, shape and dtype too.

        The base check exempts a ``.model`` without ``state_dict`` (this body's engine handle) and the
        identity mapper publishes no destination names, so without this override no coverage check would
        run at all. Comparing dtypes here is what rejects a wrong ``transfer_dtype`` before any bytes move.
        """
        be = getattr(self, "_weight_backend", None)
        if be is None:
            return
        planned = {
            name: (tuple(int(d) for d in shape), str(dtype)) for bucket in be.plan() for (name, shape, dtype) in bucket
        }
        live = {
            name: (tuple(int(d) for d in spec["shape"]), str(spec["dtype"]))
            for name, spec in self._engine_param_layout().items()
        }
        check_weight_names(live, planned, context="bucket plan does not cover the Omni worker's live parameters 1:1")
        mismatched = sorted(
            f"{name}: plan {planned[name]} vs live {live[name]}"
            for name in set(live) & set(planned)
            if planned[name] != live[name]
        )
        if mismatched:
            raise ValueError(
                "bucket plan does not cover the Omni worker's live parameters 1:1: "
                f"shape_or_dtype={mismatched[:5]} -- the trainer's wire scope, the engine's head, or the "
                "declared transfer dtype disagree"
            )
        self._engine_names = set(live)

    def _engine_param_names(self) -> set[str]:
        names = getattr(self, "_engine_names", None)  # getattr: manual rigs bypass __init__
        if names is None:
            names = set(self._engine_param_layout())
            self._engine_names = names
        return names

    def apply_bucket(self, payload: Any, i: int) -> None:
        try:
            super().apply_bucket(payload, i)
        except BaseException:
            # The worker may hold a half-loaded bucket; nothing on this side can tell how much landed.
            self.engine_invalidated = True
            raise

    def _coverage_reset(self, i: int) -> None:
        super()._coverage_reset(i)
        if i == 0:
            self._expected_digests = {}
            self.bucket_timings = []
            self.verify_timing = None

    def _engine_sink(self, state: dict) -> set[str]:
        """Second hop: write the fetched bucket straight into the worker's mapped parameters.

        The whole bucket is checked against the mapping before any copy (a bad name or shape leaves the worker
        untouched), and the copies are asynchronous on this process's stream, so the synchronize is what makes them
        visible to the worker before it is told. Both are unconditional.

        `verify_weight_values` then decides whether anything reads the bytes back: with it on, the byte-wise compare
        reads the mapped memory and the wire digests are recorded for the worker's re-read at `finish`; with it off,
        neither runs (`compare_s` and `digest_s` are 0) and this publication has no byte-level evidence -- the
        structural check, the mapping's provenance and the update bracket are all that stand behind it.
        """
        live = getattr(self, "_live_params", None)
        if live is None:
            raise RuntimeError("the worker's live parameters are not mapped; bind a weight backend before syncing")
        try:
            started = time.perf_counter()
            try:
                names = copy_named_tensors(state.items(), live)
            except KeyError as error:
                raise ValueError(
                    f"the bucket carries {error.args[0]!r}, which is not a live parameter of the engine"
                ) from error
            copied_at = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            synced_at = time.perf_counter()
            verifying = bool(getattr(self, "verify_weight_values", False))
            compare_s = 0.0
            if verifying:
                # One launch per parameter, ONE host read for the bucket: `torch.equal` per name would sync each time.
                flags = torch.stack(
                    [
                        torch.ne(
                            live[name].reshape(-1).view(torch.uint8),
                            state[name].contiguous().reshape(-1).view(torch.uint8),
                        ).any()
                        for name in names
                    ]
                )
                differing = [name for name, flag in zip(names, flags.tolist(), strict=True) if flag]
                if differing:
                    raise RuntimeError(
                        f"{len(differing)} live parameter(s) do not hold the bytes just copied into them: {differing[:5]}"
                    )
                compare_s = time.perf_counter() - synced_at
            digest_s = 0.0
            if getattr(self, "verify_weight_values", False):
                # The expectation is the WIRE bytes -- what the producer cast to `transfer_dtype` and this
                # process fetched -- recorded per bucket and verified once, before readiness.
                digest_started = time.perf_counter()
                digests = getattr(self, "_expected_digests", None)
                if digests is None:
                    digests = self._expected_digests = {}
                digests.update({name: tensor_digest_metadata(tensor) for name, tensor in state.items()})
                digest_s = time.perf_counter() - digest_started
            timings = getattr(self, "bucket_timings", None)
            if timings is None:
                timings = self.bucket_timings = []
            timings.append(
                BucketSinkTiming(
                    copy_s=copied_at - started,
                    sync_s=synced_at - copied_at,
                    compare_s=compare_s,
                    digest_s=digest_s,
                )
            )
            return names
        finally:
            live = None

    def mark_weight_version_ready(self, version: int) -> None:
        """Close the worker's update bracket -- proving the mapping is still live and, when verifying, the bytes --
        then record readiness.

        A refused bracket invalidates the engine like a failed bucket does: the worker holds bytes nobody can vouch
        for, so the lane must be rebuilt and fully re-synced before it serves again.
        """
        try:
            self._finish_weight_update(version)
        except BaseException:
            self.engine_invalidated = True
            raise
        super().mark_weight_version_ready(version)

    def _finish_weight_update(self, version: int) -> None:
        expected: dict[str, dict[str, Any]] | None = None
        if getattr(self, "verify_weight_values", False):
            expected = dict(getattr(self, "_expected_digests", None) or {})
            names = self._engine_param_names()
            if set(expected) != names:
                raise RuntimeError(
                    f"cannot verify weight version {version}: digests were staged for {len(expected)} of "
                    f"{len(names)} live parameters"
                )
        epoch = self._update_epoch
        started = time.perf_counter()
        reply = self._worker_rpc("finish_weight_update", epoch, expected, timeout=self.WEIGHT_RPC_TIMEOUT_S)
        rpc_s = time.perf_counter() - started
        if reply.get("epoch") != epoch:
            raise RuntimeError(f"finish_weight_update answered epoch {reply.get('epoch')!r}, expected {epoch}")
        if expected is not None and reply.get("verified") != len(expected):
            raise RuntimeError(f"the worker verified {reply.get('verified')!r} parameters, expected {len(expected)}")
        self.verify_timing = VerifyTiming(
            rpc_s=rpc_s,
            worker_s=reply.get("seconds"),
            digest_s=reply.get("digest_s"),
            verified=reply.get("verified"),
        )

    def commit_weight_version(self, version: int) -> None:
        super().commit_weight_version(version)
        self._stream_open = False

    # ── engine lifecycle ─────────────────────────────────────────────────────
    def rebuild_engine(self) -> None:
        """Replace the worker and keep inference blocked until a full weight stream commits.

        Invalidation precedes teardown even outside a weight stream: a new worker loads the base checkpoint,
        while `_weight_version` still records the last committed policy. A failed rebuild stays invalidated;
        a successful bound receiver permits synchronization, with only `commit_weight_version` reopening inference.
        An inference-only body has no weight backend and resumes serving its base checkpoint after successful startup.
        """
        bound = getattr(self, "_weight_backend", None) is not None
        self._stream_open = True
        self.engine_invalidated = True
        self._pending_weight_version = None
        self._prepared_weight_version = None
        try:
            # The mapping goes first: its views alias storage owned by the worker about to be closed.
            self._close_engine()
            self._engine_names = None
            self._expected_digests = {}
            self._reset_engine_caches()
            self._omni = self._start_engine()
            self.model = self._omni
            self.worker_device = self._verify_worker_device()
            self._after_engine_start()
            if bound:
                self.initialize_weight_receiver()
        except BaseException:
            self._close_engine()
            raise
        self.engine_invalidated = False
        self._stream_open = bound

    def _close_engine(self) -> None:
        """Release imported views before their owner, clearing references even if engine shutdown fails."""
        self._drop_live_mapping()
        omni = getattr(self, "_omni", None)
        try:
            if omni is not None:
                omni.close()
        finally:
            self._omni = None
            self.model = None

    def close(self) -> None:
        """The mapping first (its views alias the worker's memory), then the engine, then the first-hop backend."""
        try:
            self._close_engine()
        finally:
            backend = getattr(self, "_weight_backend", None)
            if backend is not None:
                backend.close()
