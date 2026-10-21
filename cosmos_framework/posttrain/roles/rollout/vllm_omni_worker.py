# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The worker-process half of the vLLM-Omni weight plane: exported live parameters, update brackets, optional proof.

vLLM-Omni composes ``worker_extension_cls`` into its diffusion worker by dynamic inheritance, so inside these
methods ``self`` IS the ``DiffusionWorker`` and the live pipeline is ``self.model_runner.pipeline``. This is the
model-agnostic base; a model's extension subclasses it and names its live parameters (in wire naming) through
``_live_named_parameters``. Pure torch plus ``comm/transfer/ipc``: no vLLM import, so the protocol is testable on
CPU (``tests/test_roles_rollout_vllm_omni_weights.py``).

The second hop, reversed. Once per worker life, ``export_live_parameters`` hands the rollout body
(``roles/rollout/vllm_omni.py::OmniPolicyBody``) a CUDA-IPC handle for every live parameter the weight plane may
update; the body maps them once and from then on writes each received bucket straight into this worker's parameter
memory with a D2D ``copy_`` on its own stream, waits for the copies to land, and reads the memory back byte for
byte. The worker's part per publication is a bracket: ``begin_weight_update(epoch)`` proves the mapping is still
the live one (no parameter re-allocated) and that this worker is idle before the body writes;
``finish_weight_update(epoch, expected)`` proves the same again afterwards and, when the body asks
(``verify_weight_values``), re-reads every live parameter through an independent SHA256 and compares it with the
digests the body took of the wire bytes. Nothing here moves bytes: the export is a handle, the bracket is
bookkeeping.

The export is a one-shot per worker process: every exported storage holds one CUDA IPC event of torch's fixed
budget, so a body that needs a fresh mapping rebuilds the engine rather than exporting twice. Any failure raises
here; the engine surfaces it to the body with this rank's traceback.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from cosmos_framework.posttrain.comm.transfer import ipc
from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata, tensor_digests_equal

#: Per-worker weight-plane state, created lazily on the instance: the worker is built without this class's __init__.
_STATE_ATTR = "_omni_weight_plane"
#: ``OmniDiffusionConfig`` switches that move the live parameters out from under an exported handle.
_INCOMPATIBLE_CONFIG = ("enable_sleep_mode", "enable_cpu_offload", "enable_layerwise_offload")


class OmniIpcWorkerExtension:
    """Reporting, parameter export and update-bracket methods mixed into a vLLM-Omni diffusion worker."""

    # ── model hook ───────────────────────────────────────────────────────────
    def _live_named_parameters(self) -> Iterable[tuple[str, torch.Tensor]]:
        """``(wire_name, live_parameter)`` for every parameter the weight plane may update."""
        raise NotImplementedError(f"{type(self).__name__} must implement _live_named_parameters()")

    # ── reporting ────────────────────────────────────────────────────────────
    def describe_device(self) -> dict[str, Any]:
        """The CUDA device this worker runs on, named both ways a CUDA IPC handle depends on.

        The handle carries a raw device INDEX, so the body compares the index AND the UUID: equal indices on
        different physical cards would open the wrong memory without an error.
        """
        index = torch.cuda.current_device() if torch.cuda.is_available() else None
        uuid = str(torch.cuda.get_device_properties(index).uuid) if index is not None else None
        return {
            "index": index,
            "uuid": uuid,
            "pid": os.getpid(),
            # The determinism lever the parity gates require; read HERE so a report proves the worker saw it.
            "sdpa_mode": os.environ.get("GR00T_DIT_SDPA_MODE"),
        }

    def digest_live_parameters(self, names: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        """Exact-byte digests of the live parameters (all of them, or ``names``): the independent reader the
        acceptance driver and the L1 test compare against."""
        live = dict(self._live_named_parameters())
        wanted = list(live) if names is None else [str(name) for name in names]
        missing = [name for name in wanted if name not in live]
        if missing:
            raise KeyError(f"{len(missing)} requested parameters are not live in this worker: {missing[:5]}")
        return {name: tensor_digest_metadata(live[name]) for name in wanted}

    # ── the exported mapping ─────────────────────────────────────────────────
    def _weight_plane_state(self) -> dict[str, Any]:
        state = self.__dict__.get(_STATE_ATTR)
        if state is None:
            state = {"exported": None, "open_epoch": None, "last_epoch": 0}
            self.__dict__[_STATE_ATTR] = state
        return state

    def export_live_parameters(self) -> dict[str, Any]:
        """Export a CUDA-IPC handle for every live parameter, once per worker life.

        Returns ``{"pid", "device": {"index", "uuid"}, "world_size", "params": {name: {"handle", "shape",
        "dtype"}}}``. The body maps the handles once and writes into them for the rest of this worker's life, so
        the preconditions are exactly what would make such a mapping wrong or impossible: the parameters must live
        in torch's native caching allocator (no sleep-mode VMM pool, no ``cudaMallocAsync``, no expandable
        segments), must not be offloaded or re-created by a custom pipeline, and must be contiguous tensors on this
        worker's device. A second export is refused: each exported storage holds one CUDA IPC event of torch's
        fixed budget, and a body that needs a fresh mapping rebuilds the engine.
        """
        state = self._weight_plane_state()
        if state["exported"] is not None:
            raise RuntimeError(
                "the live parameters were already exported by this worker; rebuild the engine for a fresh mapping"
            )
        config = getattr(self, "od_config", None)
        for field in _INCOMPATIBLE_CONFIG:
            if getattr(config, field, False):
                raise RuntimeError(
                    f"cannot export the live parameters with {field}=True: they would not stay in place behind the handle"
                )
        if getattr(config, "custom_pipeline_args", None) is not None:
            raise RuntimeError(
                "cannot export the live parameters with custom_pipeline_args: re_init_pipeline re-allocates them"
            )
        cuda = torch.cuda.is_available()
        device = self.describe_device()
        if cuda and torch.cuda.get_allocator_backend() != "native":
            raise RuntimeError(
                f"cannot export the live parameters from the {torch.cuda.get_allocator_backend()!r} allocator; "
                "CUDA IPC needs torch's native caching allocator"
            )
        params: dict[str, dict[str, Any]] = {}
        exported: dict[str, int] = {}
        for name, param in self._live_named_parameters():
            tensor = param.detach()  # a Parameter would rebuild as a Parameter on the other side
            if cuda and (not tensor.is_cuda or tensor.device.index != device["index"]):
                raise RuntimeError(
                    f"live parameter {name!r} is on {tensor.device}, not this worker's cuda:{device['index']}"
                )
            if not tensor.is_contiguous():
                raise RuntimeError(
                    f"live parameter {name!r} is not contiguous; the body's byte-wise compare needs a flat view of it"
                )
            try:
                handle = ipc.tensor_to_ipc(tensor)
            except RuntimeError as error:
                raise RuntimeError(
                    f"cannot export live parameter {name!r} over CUDA IPC: {error}. Torch refuses to share storages "
                    "allocated with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless "
                    "TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC is set; the weight plane needs the default allocator."
                ) from error
            params[name] = {
                "handle": handle,
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype),
            }
            exported[name] = int(tensor.data_ptr())
        if not params:
            raise RuntimeError("this worker has no live parameters to export")
        state["exported"] = exported
        import torch.distributed as dist

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return {
            "pid": device["pid"],  # the same process `describe_device` named: the body pairs mapping and worker by it
            "device": {"index": device["index"], "uuid": device["uuid"]},
            "world_size": world_size,
            "params": params,
        }

    def _check_live_pointers(self) -> None:
        """The exported mapping must still describe the live parameters: same names, same storage addresses."""
        exported = self._weight_plane_state()["exported"]
        if exported is None:
            raise RuntimeError(
                "the live parameters were never exported; the body imports the mapping before any update"
            )
        live = {name: int(param.detach().data_ptr()) for name, param in self._live_named_parameters()}
        if set(live) != set(exported):
            raise RuntimeError(
                f"the live parameter names changed since the export ({sorted(set(live) ^ set(exported))[:5]}); "
                "the body's mapping is stale -- rebuild the engine"
            )
        moved = sorted(name for name, pointer in live.items() if pointer != exported[name])
        if moved:
            raise RuntimeError(
                f"{len(moved)} live parameter(s) were re-allocated since the export ({moved[:5]}); the body's mapping "
                "is stale -- rebuild the engine"
            )

    # ── the update bracket ───────────────────────────────────────────────────
    def begin_weight_update(self, epoch: int) -> dict[str, Any]:
        """Open an update bracket: the mapping is still the live one and this worker is idle, so the body may write.

        A newer epoch supersedes an open one: a stream that failed on ANOTHER lane leaves this lane's bracket open
        with nothing to close it (only the failed lane is rebuilt), and the recovery sync must be able to start.
        Epochs are the body's monotonically increasing counter; one that does not advance is a protocol error.
        Returns ``{"pid", "epoch", "superseded": the epoch left open, or None}``.
        """
        state = self._weight_plane_state()
        epoch = int(epoch)
        if epoch <= state["last_epoch"]:
            raise RuntimeError(f"begin_weight_update({epoch}) does not advance past epoch {state['last_epoch']}")
        self._check_live_pointers()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # explicit quiescence: nothing of this worker's is touching the parameters
        superseded = state["open_epoch"]
        state["open_epoch"] = state["last_epoch"] = epoch
        return {"pid": os.getpid(), "epoch": epoch, "superseded": superseded}

    def finish_weight_update(
        self, epoch: int, expected: Mapping[str, Mapping[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Close the bracket the body opened, after its writes completed; optionally prove the live bytes.

        Runs on every rank (the engine executes an RPC on all of them and gathers their statuses before rank 0
        replies), so a failure on any rank raises into the body's call. ``expected`` -- the body's digests of the
        wire bytes it copied, present only under ``verify_weight_values`` -- is compared with an independent re-read
        of EVERY live parameter, in both directions: a live parameter without an expectation is as much a failure
        as an unknown name or one differing byte. Returns ``{"pid", "epoch", "verified": n | None, "seconds",
        "digest_s": the re-read's own time | None}``.
        """
        started = time.perf_counter()
        state = self._weight_plane_state()
        epoch = int(epoch)
        if state["open_epoch"] is None:
            raise RuntimeError(f"finish_weight_update({epoch}): no update is open on this worker")
        if state["open_epoch"] != epoch:
            raise RuntimeError(f"finish_weight_update({epoch}): the open update is epoch {state['open_epoch']}")
        self._check_live_pointers()
        verified = None
        digest_s = None
        if expected is not None:
            digest_started = time.perf_counter()
            live = dict(self._live_named_parameters())
            missing = sorted(set(live) - set(expected))
            unknown = sorted(set(expected) - set(live))
            if missing or unknown:
                raise ValueError(
                    f"live-parameter verification names do not match: without expectation={missing[:5]}, "
                    f"unknown to this worker={unknown[:5]}"
                )
            mismatched = [
                name
                for name in sorted(expected)
                if not tensor_digests_equal(tensor_digest_metadata(live[name]), expected[name])
            ]
            if mismatched:
                raise ValueError(
                    f"{len(mismatched)} live parameter(s) do not hold the staged bytes: {mismatched[:5]}"
                    f"{'...' if len(mismatched) > 5 else ''}"
                )
            verified = len(expected)
            digest_s = time.perf_counter() - digest_started
        state["open_epoch"] = None
        return {
            "pid": os.getpid(),
            "epoch": epoch,
            "verified": verified,
            "seconds": time.perf_counter() - started,
            "digest_s": digest_s,
        }
