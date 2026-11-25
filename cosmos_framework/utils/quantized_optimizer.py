# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Ordinary CUDA FP32 parameters with pinned, optionally paged eight-bit moments.

Imported only by the selected factory branch. Quantization mathematics is upstream.
No DTensor views, implicit parameter conversion, or global override registration.

Under ``update_policy="fp32_master"`` the backend sees each rank's local FSDP2 shard.
Eligibility for eight-bit moments (``bnb_min_8bit_size``) is decided from the logical
parameter, so it is the same on every rank and at every world size. The blockwise
quantization boundaries (256-element blocks and their scales), however, are laid over
each LOCAL shard: exact eight-bit numerics are not world-size invariant. That matches
the same-topology exact-resume rule; changing the world size is a weights-only restart.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
from collections import defaultdict
from copy import deepcopy
from functools import cache
from types import MethodType, SimpleNamespace

import torch

from cosmos_framework.utils.quantized_optimizer_names import QUANTIZED_OPTIMIZER_NAMES as NAMES

VERSION = "0.49.2"


def _owned_state_buffer(optimizer, parameter, dtype=torch.float32):
    """Upstream managed layout, with a deleter owned by the exported buffer.

    Mirrors ``bitsandbytes.functional.get_paged`` (bitsandbytes 0.49.2), which creates a numpy
    view of cudaMallocManaged memory without a free callback. The ndarray here owns that
    callback; PyTorch frombuffer retains it until the LAST storage alias dies, including
    detach(). It calls the private native symbols ``lib.cget_managed_ptr`` and
    ``lib._lib.cudaFree``: the release is pinned exactly (``VERSION``), and any version bump must
    re-audit this function against upstream ``get_paged`` and ``GlobalPageManager``.
    """
    if not optimizer.is_paged or parameter.numel() < 100000:
        return torch.zeros_like(parameter, dtype=dtype, device=parameter.device)
    import ctypes
    import weakref

    import numpy as np
    from bitsandbytes import functional as functional
    from bitsandbytes.cextension import lib

    free = lib._lib.cudaFree  # pinned native ABI; fail rather than leak on missing symbol
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_int
    device = parameter.device
    size = parameter.numel() * dtype.itemsize
    with torch.cuda.device(device):
        pointer = lib.cget_managed_ptr(ctypes.c_size_t(size))
    if not pointer:
        raise MemoryError("bitsandbytes managed allocation returned null")
    array = np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(size,))

    def release():
        with torch.cuda.device(device):
            error = free(ctypes.c_void_p(pointer))
        if error:
            import warnings

            warnings.warn(f"cudaFree of managed optimizer state failed: {error}", ResourceWarning)

    owner = weakref.finalize(array, release)
    owner.atexit = False  # CUDA context teardown owns process-exit cleanup
    buffer = torch.frombuffer(array, dtype=dtype, count=parameter.numel()).view(parameter.shape)
    buffer.is_paged = True
    buffer.page_deviceid = parameter.device.index
    buffer._cosmos_managed_owner = weakref.ref(array)
    functional.fill(buffer, 0)
    optimizer.page_mng.paged_tensors.append(buffer)
    return buffer


class _ManagedCudaView:
    """``__cuda_array_interface__`` over a managed (paged) buffer, so torch can address it as CUDA memory."""

    def __init__(self, tensor):
        self.owner = tensor  # keeps the managed allocation alive for the view's lifetime
        self.__cuda_array_interface__ = {
            "shape": tuple(tensor.shape),
            "typestr": torch.empty((), dtype=tensor.dtype).numpy().dtype.str,
            "data": (tensor.data_ptr(), False),
            "version": 3,
            "strides": None,
        }


def device_view(tensor):
    """A zero-copy CUDA view of optimizer state, for device-side copies.

    Paged eight-bit moments are CPU tensors over ``cudaMallocManaged`` memory
    (``_owned_state_buffer``): ``clone()``/``copy_()`` on them run on the host and migrate
    every page to it. Their CUDA view keeps such copies on the device. Other tensors are
    returned unchanged. Host-side readers of optimizer state use ``copy_state_tensor``.
    """
    if not getattr(tensor, "is_paged", False) or tensor.is_cuda:
        return tensor
    if not tensor.is_contiguous():
        raise ValueError("paged optimizer state must be contiguous")
    return torch.as_tensor(_ManagedCudaView(tensor), device=torch.device("cuda", tensor.page_deviceid))


def stage_to_pinned_host(tensor):
    """Start an independent host copy of ``tensor`` in new pinned memory; returns it.

    CUDA tensors, and paged (managed-memory) optimizer state through its ``device_view``, are
    copied by an asynchronous device-side copy ordered on the current stream, so paged state
    never migrates its pages to the host. The copy is complete once that stream reaches it:
    a later stream-ordered use (e.g. a non-blocking copy back) needs no synchronization, but a
    host read must synchronize first. Other CPU tensors are copied synchronously.
    """
    source = device_view(tensor).detach()
    if not source.is_cuda:
        return source.to("cpu", copy=True)
    staged = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
    staged.copy_(source, non_blocking=True)
    return staged


def copy_state_tensor(value, *, ordinary, paged_devices=None):
    """A copy of one optimizer-state tensor that is safe to read outside the optimizer step.

    The single place that encodes how optimizer state is read: host-side readers (snapshots,
    rollback rings, digests) must go through this helper, or ``device_view`` for device-side
    work. Paged eight-bit moments (``is_paged``) are CPU tensors over CUDA managed memory, and
    any host read of them (``clone``, ``copy_``, ``numpy``, hashing) migrates every page to the
    host. They are always staged into new pinned host memory through their CUDA view
    (``stage_to_pinned_host``), asynchronously on the current stream; their CUDA device is added
    to ``paged_devices`` so the caller can synchronize once per device before a host read (a
    stream-ordered use needs no wait). ``ordinary`` decides every other tensor: ``"device"``
    clones it where it lives, ``"host"`` copies it into pageable host memory (pinning it would
    keep page-locked host memory the size of the state cached for the run), and ``None`` returns
    it uncopied, for a caller that only reads it on the host.
    """
    if getattr(value, "is_paged", False) and not value.is_cuda:
        if paged_devices is not None:
            paged_devices.add(torch.device("cuda", value.page_deviceid))
        return stage_to_pinned_host(value)
    if ordinary is None:
        return value
    if ordinary == "device":
        return value.detach().clone()
    if ordinary != "host":
        raise ValueError(f"unknown ordinary-state copy {ordinary!r}")
    out = torch.empty(value.shape, dtype=value.dtype, device="cpu")
    out.copy_(value.detach())
    return out


def state_log_summary(report):
    """The INFO-log form of a ``state_report``: its integer counters and a short identity digest.

    The identities a report carries (``backend``, ``master_identity``) list every parameter's
    name, shape and topology; the log keeps a digest of them, to compare runs by, not the lists.
    """
    summary = {key: value for key, value in report.items() if type(value) is int}
    identities = {key: report[key] for key in ("backend", "master_identity") if key in report}
    if identities:
        encoded = json.dumps(identities, sort_keys=True, default=str).encode()
        summary["identity_digest"] = hashlib.sha256(encoded).hexdigest()[:16]
    return summary


def require_installed():
    """Pure preflight: the pinned bitsandbytes release is installed. No import, no CUDA.

    The release segment must be exactly ``VERSION`` (its moment layout and quantization maps are
    the saved format); a local build tag such as ``0.49.2+cu130`` is the same release.
    """
    from packaging.version import InvalidVersion, Version

    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Install cosmos-framework[quantized-optimizers] for bitsandbytes optimizers") from exc
    try:
        parsed = Version(version)
    except InvalidVersion:
        parsed = None
    if parsed is None or parsed.release != Version(VERSION).release or parsed.pre or parsed.dev:
        raise RuntimeError(f"bitsandbytes=={VERSION} required; found {version}")


# (CUDA device index, paged) pairs whose kernel/allocation probe already passed in this process.
_PROBED: set[tuple[int, bool]] = set()


def require_backend(*, paged=False):
    """Import bitsandbytes and probe its CUDA kernels on the current device, once per process."""
    require_installed()
    if not torch.cuda.is_available():
        raise ValueError("bitsandbytes optimizers require CUDA")
    import bitsandbytes as bnb
    from bitsandbytes.cextension import lib

    if not getattr(lib, "compiled_with_cuda", False):
        raise RuntimeError("bitsandbytes CUDA kernels are unavailable")
    key = (torch.cuda.current_device(), bool(paged))
    if key in _PROBED:
        return bnb
    # Exercise the selected CUDA allocation/update path before model construction.
    # No RNG use and no model update: this scratch tensor is discarded immediately.
    probe = torch.nn.Parameter(torch.ones(131072 if paged else 4096, device="cuda"))
    probe.grad = torch.zeros_like(probe)
    optimizer = bnb.optim.AdamW8bit([probe], is_paged=paged)
    optimizer.mng = SimpleNamespace(index2config={}, module_weight_config_triple=[])
    optimizer.page_mng = SimpleNamespace(paged_tensors=[])
    optimizer.get_state_buffer = MethodType(_owned_state_buffer, optimizer)
    try:
        optimizer.step()
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError("bitsandbytes CUDA optimizer kernel/allocation preflight failed") from exc
    finally:
        del optimizer.get_state_buffer  # avoid a bound-method/self cycle retaining the probe
    _PROBED.add(key)
    return bnb


def _snapshot(value):
    """An independent CPU copy of a (nested) optimizer state, safe to read on the host on return."""
    paged_devices = set()
    result = _snapshot_into(value, paged_devices)
    for device in paged_devices:
        torch.cuda.synchronize(device)  # once per device: every staged copy of paged state is complete
    return result


def _snapshot_into(value, paged_devices):
    if isinstance(value, torch.Tensor):
        # Paged state is staged on its GPU (no page migration to the host on every save) and
        # ``_snapshot`` waits once per device; everything else gets an ordinary pageable copy.
        # Never deepcopy paged tensors: their Python attributes include managed pointers.
        return copy_state_tensor(value, ordinary="host", paged_devices=paged_devices)
    if isinstance(value, dict):
        return {k: _snapshot_into(v, paged_devices) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot_into(v, paged_devices) for v in value)
    return deepcopy(value)


def canonical_parameter_names(net):
    """``id(parameter) -> name`` for live trainables, in model state-dict naming.

    Activation-checkpoint (``_checkpoint_wrapped_module.``) and ``torch.compile``
    (``_orig_mod.``) segments are removed exactly as ``torch.distributed.checkpoint.state_dict``
    removes them, so optimizer names, ``bnb_fp32_state_names`` and checkpoint identities all
    match ``get_model_state_dict`` keys whether or not the model is wrapped.
    """
    trainables = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    if not isinstance(net, torch.nn.Module):
        return {id(p): n for n, p in trainables}  # already canonical (e.g. the master owner's view)
    from torch.distributed.checkpoint.state_dict import _get_fqns

    names = {}
    for name, parameter in trainables:
        fqns = _get_fqns(net, name)
        names[id(parameter)] = next(iter(fqns)) if len(fqns) == 1 else name
    return names


def named_parameter_groups(net, params, *, context):
    """Normalize optimizer groups over the final model's trainables, shared by both owners.

    Returns ``(groups, selected, names)``: each group gains ``param_names`` (the canonical
    trainable name of every member), ``selected`` is the flat member list and ``names`` maps
    ``id(parameter)`` to its name. Empty or duplicated (including tied-alias) members fail.
    """
    groups = params if params and isinstance(params[0], dict) else [{"params": params}]
    names = canonical_parameter_names(net)
    selected = [p for g in groups for p in g["params"]]
    if not selected or len({id(p) for p in selected}) != len(selected):
        raise ValueError(f"{context} requires nonempty, identity-unique trainables")
    if any(id(p) not in names for p in selected):
        raise ValueError(f"{context} parameter is not a trainable owned by the final model")
    groups = [dict(g, param_names=[names[id(p)] for p in g["params"]]) for g in groups]
    return groups, selected, names


def build_quantized_optimizer(net, params, config, *, _local_owner=None):
    """Build from the final trainable objects and existing stable LR/decay partition."""
    groups, selected, names = named_parameter_groups(net, params, context="quantized optimizer")
    for p in selected:
        if type(p) is not torch.nn.Parameter or not p.requires_grad:
            raise ValueError(
                "quantized optimizer requires ordinary trainable Parameters, not DTensor/quantized weights"
            )
        if (
            p.device.type != "cuda"
            or p.dtype != torch.float32
            or not p.is_contiguous()
            or (not p.numel() and _local_owner is None)
        ):
            raise ValueError(
                "quantized optimizer requires nonempty contiguous CUDA FP32 trainables; no implicit masters"
            )
    if len({p.device for p in selected}) != 1:
        raise ValueError("quantized optimizer requires one CUDA device")
    if _local_owner is None and torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("quantized optimizer distributed strategies are not supported")
    exceptions = set(config.bnb_fp32_state_names)
    missing = exceptions - {names[id(p)] for p in selected}
    if missing:
        raise ValueError(f"unmatched FP32-state parameter names (use canonical tied name): {sorted(missing)}")
    for g in groups:
        invalid = set(g) & {"fused", "foreach", "capturable", "amsgrad", "master_weights", "percentile_clipping"}
        if invalid:
            raise ValueError(f"unsupported bitsandbytes group options: {sorted(invalid)}")
    with torch.cuda.device(selected[0].device):
        bnb = require_backend(paged=config.name == NAMES[1])

    # The class may outlive an instance (Python cycles, tracing/type caches).
    # Keep every parameter reference on the instance, never in class closures.
    return _optimizer_type(bnb.optim.AdamW8bit)(groups, selected, names, config, exceptions, _local_owner)


@cache
def _optimizer_type(base):
    return type("CosmosAdamW8bit", (_QuantizedOptimizerMethods, base), {"__module__": __name__})


class _QuantizedOptimizerMethods:
    cosmos_quantized_optimizer = True
    get_state_buffer = _owned_state_buffer

    def __init__(self, groups, selected, names, config, exceptions, _local_owner):
        self._selected = tuple(selected)
        self._names = dict(names)
        self._exceptions = frozenset(exceptions)
        super().__init__(
            groups,
            lr=config.lr,
            betas=tuple(config.betas),
            eps=config.eps,
            weight_decay=config.weight_decay,
            min_8bit_size=config.bnb_min_8bit_size,
            percentile_clipping=100,
            block_wise=True,
            is_paged=config.name == NAMES[1],
        )
        # Upstream stores managers globally. Own both registries per optimizer so
        # another trial's index overrides cannot change this model's numerics.
        self.mng = SimpleNamespace(index2config={}, module_weight_config_triple=[])
        self.page_mng = SimpleNamespace(paged_tensors=[])
        logical = (
            {entry["name"]: math.prod(entry["shape"]) for entry in _local_owner["parameters"]}
            if _local_owner is not None
            else {}
        )
        for gi, group in enumerate(self.param_groups):
            for pi, name in enumerate(group["param_names"]):
                if name in exceptions:
                    self.mng.index2config[(gi, pi)] = {"optim_bits": 32}
                elif name in logical:
                    # bitsandbytes compares ``min_8bit_size`` with the tensor it sees, i.e. the
                    # local FSDP shard. Decide from the LOGICAL parameter instead, so every rank
                    # and every world size gives one parameter the same moment precision.
                    eligible = logical[name] >= config.bnb_min_8bit_size
                    self.mng.index2config[(gi, pi)] = {"min_8bit_size": 1} if eligible else {"optim_bits": 32}
        self.identity = {
            "format": "cosmos_bnb_v1",
            "backend": "bitsandbytes",
            "version": VERSION,
            "name": config.name,
            "storage": "float32",
            "topology": "single_cuda",
            "min_8bit_size": config.bnb_min_8bit_size,
            "warmup_steps": config.warmup_steps,
            "fp32_state_names": sorted(exceptions),
            "groups": [
                {
                    "names": g["param_names"],
                    "shapes": [list(p.shape) for p in g["params"]],
                    "name": g.get("name"),
                    "initial_lr": g["lr"],
                    "betas": tuple(g["betas"]),
                    "eps": g["eps"],
                    "weight_decay": g["weight_decay"],
                }
                for g in self.param_groups
            ],
        }
        # Log the state report on the first step, and again whenever ``init_state`` allocates
        # state for another parameter: a flag, not a per-step scan over every parameter's state.
        self._report_state = True
        # The local-master owner rejects nonfinite gradients across all ranks before it
        # stages them; a second full scan here would only repeat that host decision.
        self._owner_checks_finite = _local_owner is not None
        # Set by ``note_gradients_checked_finite``; consumed by the next step.
        self._gradients_checked = False
        # An FP32-master owner logs its own state report, which includes this backend's counters.
        self._logs_state_report = _local_owner is None
        if _local_owner is not None:
            self.identity["topology"] = deepcopy(_local_owner)
            if any(entry["shape"] != entry["local_shape"] for entry in _local_owner["parameters"]):
                # Sharded layouts decide 8-bit eligibility from logical sizes (see above). Checkpoints
                # written before that rule may hold FP32 moments for small shards of large tensors;
                # name the rule so such a checkpoint is refused by identity, never by a dtype error.
                self.identity["min_8bit_size_basis"] = "logical_parameter"

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("quantized optimizer closures are unsupported")
        # Precheck the WHOLE update, before moments or clocks can change.
        grads = [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
        if any(g.is_sparse for g in grads):
            raise ValueError("quantized optimizer does not support sparse gradients")
        prechecked, self._gradients_checked = self._gradients_checked, False
        # A single host decision and no full-size boolean intermediates.
        if (
            grads
            and not (self._owner_checks_finite or prechecked)
            and not torch.isfinite(torch.stack(torch._foreach_norm(grads, float("inf")))).all()
        ):
            raise FloatingPointError("nonfinite gradients: quantized update refused without mutation")
        self._update_parameters()
        if self._report_state:
            self._report_state = False
            if self._logs_state_report:
                logging.getLogger("posttrain").info(
                    "Quantized optimizer state: %s", state_log_summary(self.state_report())
                )
        return None

    def _update_parameters(self):
        """Upstream ``bitsandbytes.optim.optimizer.Optimizer8bit.step`` (0.49.2), minus its per-parameter sync.

        A copy of upstream's loop: the release is pinned exactly (``VERSION``), and any version
        bump must re-audit this method against upstream ``Optimizer8bit.step``.
        ``test_step_syncs_at_most_once_and_matches_upstream_numerics`` checks it bit for bit
        against upstream ``Optimizer8bit.step`` on the installed release.

        Upstream calls ``sync_gpu`` (a full ``torch.cuda.synchronize()``) after EVERY parameter's
        update. Nothing needs it: each parameter's prefetch and update kernels are ordered on the
        device, no update reads another parameter's state (the quantization maps are constant
        tables), and no host code reads state inside the loop. Prefetching managed pages is only
        a placement hint; the driver keeps them coherent either way. Paged runs keep upstream's
        single end-of-step synchronization, because paged state is also reachable from the host.
        The same kernels run in the same order on the same stream, so the numerics are unchanged.
        """
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True
        last = None
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if not self.state[p]:
                    self.init_state(group, p, gindex, pindex)
                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                last = p
        if self.is_paged and last is not None:
            torch.cuda.synchronize(last.device)

    def note_gradients_checked_finite(self) -> None:
        """The caller has collectively verified that the next step's gradients are finite.

        ``LLMTrainer`` runs ``all_ranks_gradients_finite`` (``force_finite_grads``, required for
        these names) before stepping; the next step then skips its own scan and host sync. A step
        that nobody noted still scans.
        """
        self._gradients_checked = True

    def init_state(self, group, p, gindex, pindex):
        """Upstream allocation (first step, or ``load_state_dict``); marks the state report as stale."""
        super().init_state(group, p, gindex, pindex)
        self._report_state = True

    def reset_state(self):
        """Drop every moment, including the managed buffers that ``page_mng`` would otherwise keep alive."""
        self.state, self.page_mng.paged_tensors = defaultdict(dict), []
        self._report_state = True

    def state_report(self):
        report = {
            "backend": self.identity,
            "trainable_elements": sum(p.numel() for p in self._selected),
            "quantized_elements": 0,
            "fp32_elements": 0,
            "moment_bytes": 0,
            "metadata_bytes": 0,
            "paged_moment_bytes": 0,
            "uninitialized_elements": 0,
            "exceptions": {},
        }
        seen = set()
        for p in self._selected:
            state = self.state.get(p, {})
            if not state:
                report["uninitialized_elements"] += p.numel()
                continue
            quantized = state["state1"].dtype == torch.uint8
            report["quantized_elements" if quantized else "fp32_elements"] += p.numel()
            if not quantized:
                report["exceptions"][self._names[id(p)]] = (
                    "explicit" if self._names[id(p)] in self._exceptions else "small_tensor"
                )
            for key, value in state.items():
                if not isinstance(value, torch.Tensor) or value.data_ptr() in seen:
                    continue
                seen.add(value.data_ptr())
                size = value.numel() * value.element_size()
                report["moment_bytes" if key in ("state1", "state2") else "metadata_bytes"] += size
                if key in ("state1", "state2") and getattr(value, "is_paged", False):
                    report["paged_moment_bytes"] += size
        return report

    def state_dict(self):
        torch.cuda.synchronize(self._selected[0].device)
        # Bypass upstream FSDP wrapping: this format explicitly refuses sharding.
        result = _snapshot(torch.optim.Optimizer.state_dict(self))
        result["cosmos_quantized_optimizer"] = deepcopy(self.identity)
        return result

    def _expected_state_layout(self, group, p, gindex, pindex):
        """``{key: (dtype, shape)}`` that bitsandbytes 0.49.2 ``init_state`` allocates for ``p``, without allocating.

        Construction fixes ``block_wise=True``, ``percentile_clipping=100`` and ``max_unorm=0``, so
        no clipping or max-norm buffers exist.
        """
        config = self.get_config(gindex, pindex, group)
        shape = tuple(p.shape)
        if config["optim_bits"] != 8 or p.numel() < config["min_8bit_size"]:
            return {"state1": (torch.float32, shape), "state2": (torch.float32, shape)}
        blocks = (p.numel() + 255) // 256
        return {
            "state1": (torch.uint8, shape),
            "state2": (torch.uint8, shape),
            "qmap1": (torch.float32, (256,)),
            "qmap2": (torch.float32, (256,)),
            "absmax1": (torch.float32, (blocks,)),
            "absmax2": (torch.float32, (blocks,)),
        }

    def validate_state_dict(self, saved):
        """Check a host-resident payload against this optimizer, without allocating or changing anything.

        Identity, group structure, parameter mapping and every saved state entry's keys, counter,
        dtypes, shapes and quantization maps. ``load_state_dict`` runs it first; an owner may also
        run it while staging, to decide before any commit.
        """
        if saved.get("cosmos_quantized_optimizer") != self.identity:
            raise ValueError("quantized optimizer exact-resume identity mismatch; use an explicit weights-only restart")
        live_groups = self.param_groups
        if len(saved["param_groups"]) != len(live_groups):
            raise ValueError("quantized optimizer group count mismatch")
        ids = [i for g in saved["param_groups"] for i in g["params"]]
        if len(ids) != len(self._selected) or len(set(ids)) != len(ids) or set(saved["state"]) - set(ids):
            raise ValueError("invalid quantized optimizer parameter mapping")
        for old, new in zip(saved["param_groups"], live_groups):
            if old["param_names"] != new["param_names"] or len(old["params"]) != len(new["params"]):
                raise ValueError("quantized optimizer parameter order mismatch")
            for key in ("betas", "eps", "weight_decay"):
                if old[key] != new[key]:
                    raise ValueError(f"quantized optimizer {key} mismatch")
        tables = None  # the constant quantization maps on the host, built once on first use
        for gi, (old, group) in enumerate(zip(saved["param_groups"], live_groups)):
            for pi, (idx, p) in enumerate(zip(old["params"], group["params"])):
                source = saved["state"].get(idx, {})
                if not source:
                    continue
                expected = self._expected_state_layout(group, p, gi, pi)
                if set(source) != {"step", *expected} or type(source["step"]) is not int or source["step"] < 0:
                    raise ValueError("invalid quantized optimizer state schema/counter")
                for key, (dtype, shape) in expected.items():
                    value = source[key]
                    if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
                        raise ValueError(f"invalid quantized optimizer state {key} shape/dtype")
                if "qmap1" in source:
                    if tables is None:
                        tables = self._host_quantization_maps()
                    if any(not torch.equal(source[key].cpu(), table) for key, table in tables.items()):
                        raise ValueError("quantized optimizer quantization map mismatch")

    def _host_quantization_maps(self):
        """``{"qmap1": ..., "qmap2": ...}``: host copies of the constant tables ``init_state`` assigns."""
        if "dynamic" not in self.name2qmap:
            self.fill_qmap()  # the constant tables init_state would build
        return {"qmap1": self.name2qmap["dynamic"].cpu(), "qmap2": self.name2qmap["udynamic"].cpu()}

    def load_state_dict(self, saved):
        # Validate everything before changing live state. Allocation is transactional:
        # failures restore the previous state and page ownership.
        self.validate_state_dict(saved)
        live_groups = self.param_groups
        # Stage group metadata before replacing any live state. A failed
        # host allocation/copy must not leave new moments with old groups.
        restored_groups = [
            dict(_snapshot(old), params=new["params"]) for old, new in zip(saved["param_groups"], live_groups)
        ]
        previous, pages = self.state, self.page_mng.paged_tensors
        self.state, self.page_mng.paged_tensors = defaultdict(dict), []
        try:
            for gi, (old, group) in enumerate(zip(saved["param_groups"], live_groups)):
                for pi, (idx, p) in enumerate(zip(old["params"], group["params"])):
                    source = saved["state"].get(idx, {})
                    if not source:
                        continue
                    self.init_state(group, p, gi, pi)  # allocation only, never step()
                    target = self.state[p]
                    if set(source) != set(target) or type(source["step"]) is not int or source["step"] < 0:
                        raise ValueError("invalid quantized optimizer state schema/counter")
                    for key, value in source.items():
                        if isinstance(target[key], torch.Tensor):
                            if (
                                not isinstance(value, torch.Tensor)
                                or value.shape != target[key].shape
                                or value.dtype != target[key].dtype
                            ):
                                raise ValueError(f"invalid quantized optimizer state {key} shape/dtype")
                            # qmaps are shared constant tables, never written: validate_state_dict above
                            # already compared the saved ones with them.
                            if key.startswith("qmap"):
                                continue
                            # Paged targets are written on their GPU, not through host page faults.
                            device_view(target[key]).copy_(value)
                        else:
                            target[key] = value
            torch.cuda.synchronize(self._selected[0].device)
        except Exception:
            self.state, self.page_mng.paged_tensors = previous, pages
            raise
        self.param_groups = restored_groups
        self.initialized = True
        self._report_state = True
