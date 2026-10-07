# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offload large activation-checkpoint saves to CPU memory.

Saved-tensor hooks surround checkpointed module calls, so other autograd
saves made during denoising stay on their original device. Eligible CUDA
tensors move to CPU memory during forward when live cgroup headroom can retain
the configured reserve, then return synchronously during backward recomputation.
When headroom is insufficient or unknown, they stay on GPU and a warning is logged.
The hooks remain outside the ``torch.compile``
boundary.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint

from cosmos_framework.utils import log

# Only tensors at least this large are offloaded (avoids wasting PCIe
# bandwidth on tiny metadata / scalar tensors).
_MIN_OFFLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_MIN_CGROUP_MEMORY_FREE_FRACTION = 0.10
_skip_warning_counts: dict[tuple[int, str], int] = {}
_successful_offload_totals: dict[int, tuple[int, int]] = {}
_logged_cgroup_setup_pids: set[int] = set()


@dataclass(frozen=True)
class _CgroupMemoryLimit:
    usage_path: Path
    limit_bytes: int


@dataclass(frozen=True)
class _StaticCgroupMemory:
    limits: tuple[_CgroupMemoryLimit, ...]
    error: str | None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii")
    except OSError:
        return None


def _parse_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        number = int(value.strip())
    except ValueError:
        return None
    return number if 0 <= number < 2**60 else None


def _read_bytes(path: Path) -> int | None:
    return _parse_bytes(_read_text(path))


def _unescape_mountinfo_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _find_namespaced_cgroup(mount_point: Path, pid: int) -> Path | None:
    """Find the visible cgroup containing this process when its name is hidden by a namespace."""
    try:
        directories = (mount_point, *mount_point.iterdir())
    except OSError:
        return None
    matches: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        procs = _read_text(directory / "cgroup.procs")
        if procs is not None and str(pid) in procs.splitlines():
            matches.append(directory)
    return matches[0] if len(matches) == 1 else None


def _find_cgroup_mount(mountinfo: str, cgroup_path: str, *, v2: bool) -> tuple[Path, Path] | None:
    process_path = PurePosixPath(cgroup_path)
    matches: list[tuple[int, Path, Path]] = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        mount_fields = before.split()
        fs_fields = after.split()
        if len(mount_fields) < 5 or len(fs_fields) < 3:
            continue
        fs_type = fs_fields[0]
        if v2 and fs_type != "cgroup2":
            continue
        if not v2 and (fs_type != "cgroup" or "memory" not in fs_fields[2].split(",")):
            continue
        mount_root = PurePosixPath(_unescape_mountinfo_path(mount_fields[3]))
        mount_point = Path(_unescape_mountinfo_path(mount_fields[4]))
        try:
            relative_path = process_path.relative_to(mount_root)
        except ValueError:
            if process_path != PurePosixPath("/") or mount_root != PurePosixPath("/.."):
                continue
            leaf = _find_namespaced_cgroup(mount_point, os.getpid())
            if leaf is None:
                continue
        else:
            leaf = mount_point / relative_path
        if leaf.is_dir():
            matches.append((len(mount_root.parts), mount_point, leaf))
    if not matches:
        return None
    _, mount_point, leaf = max(matches, key=lambda match: match[0])
    return mount_point, leaf


def _cgroup_limits(mount_point: Path, leaf: Path, *, v2: bool) -> tuple[tuple[_CgroupMemoryLimit, ...], str | None]:
    limit_name = "memory.max" if v2 else "memory.limit_in_bytes"
    usage_name = "memory.current" if v2 else "memory.usage_in_bytes"
    limits: list[_CgroupMemoryLimit] = []
    for directory in (leaf, *leaf.parents):
        limit_path = directory / limit_name
        limit_text = _read_text(limit_path)
        if limit_text is None and directory != mount_point:
            return (), f"cannot read {limit_path}"
        if limit_text is not None:
            token = limit_text.strip()
            limit_bytes = _parse_bytes(token)
            v1_unlimited = not v2 and token.isdecimal() and int(token) >= 2**60
            if token != "max" and limit_bytes is None and not v1_unlimited:
                return (), f"invalid memory limit in {limit_path}: {token!r}"
            if limit_bytes is not None:
                limits.append(_CgroupMemoryLimit(directory / usage_name, limit_bytes))
        if directory == mount_point:
            break
    return tuple(limits), None


@lru_cache(maxsize=1)
def _static_cgroup_memory(pid: int, proc_root: Path) -> _StaticCgroupMemory:
    """Cache fixed memory limits and their usage paths for one process."""
    del pid  # A forked process must get its own cache entry.
    cgroup_text = _read_text(proc_root / "self" / "cgroup")
    mountinfo = _read_text(proc_root / "self" / "mountinfo")
    if cgroup_text is None or mountinfo is None:
        missing = proc_root / "self" / ("cgroup" if cgroup_text is None else "mountinfo")
        return _StaticCgroupMemory((), f"cannot read {missing}")
    v2_path: str | None = None
    v1_path: str | None = None
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, path = fields
        if hierarchy == "0" and not controllers:
            v2_path = path
        elif "memory" in controllers.split(","):
            v1_path = path
    # A v1 memory entry owns the memory controller on hybrid cgroup systems.
    # The v2 hierarchy can be mounted there without exposing memory.max.
    cgroup_path, v2 = (v1_path, False) if v1_path is not None else (v2_path, True)
    if cgroup_path is not None:
        mount = _find_cgroup_mount(mountinfo, cgroup_path, v2=v2)
        if mount is not None:
            mount_point, leaf = mount
            limits, error = _cgroup_limits(mount_point, leaf, v2=v2)
            if error is not None:
                return _StaticCgroupMemory((), error)
            if limits:
                return _StaticCgroupMemory(limits, None)
    return _StaticCgroupMemory((), "no verified finite memory limit for this process's cgroup")


def _check_cgroup_memory_headroom(
    requested_bytes: int,
    min_free_fraction: float,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[str, str] | None:
    """Describe why a CPU copy should stay on GPU, if applicable.

    This is an adaptive check; other processes can allocate RAM before the copy.
    """
    static = _static_cgroup_memory(os.getpid(), proc_root)
    if static.error is not None:
        return "unavailable", f"{static.error}; requested={requested_bytes} bytes"
    for limit in static.limits:
        used = _read_bytes(limit.usage_path)
        if used is None:
            return "unreadable", f"cannot read {limit.usage_path}; requested={requested_bytes} bytes"
        reserve = math.ceil(limit.limit_bytes * min_free_fraction)
        available = max(0, limit.limit_bytes - used)
        if available - requested_bytes < reserve:
            return (
                str(limit.usage_path),
                f"measured cgroup headroom below reserve at {limit.usage_path}: "
                f"requested={requested_bytes} bytes, usage={used} bytes, available={available} bytes, "
                f"reserve={reserve} bytes, limit={limit.limit_bytes} bytes; usage may include reclaimable cache",
            )
    return None


def _warn_skipped_offload(reason_key: str, message: str) -> None:
    pid = os.getpid()
    key = (pid, reason_key)
    count = _skip_warning_counts.get(key, 0) + 1
    _skip_warning_counts[key] = count
    if count & (count - 1):
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else os.environ.get("RANK", "0")
    )
    log.warning(f"Activation CPU offload skipped on rank={rank} ({count} skips): {message}", rank0_only=False)


def _log_successful_offload(copied_bytes: int) -> None:
    pid = os.getpid()
    count, total_bytes = _successful_offload_totals.get(pid, (0, 0))
    count += 1
    total_bytes += copied_bytes
    _successful_offload_totals[pid] = (count, total_bytes)
    if count & (count - 1):
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else os.environ.get("RANK", "0")
    )
    log.info(
        f"Activation CPU offload succeeded on rank={rank} "
        f"({count} copies, {total_bytes} cumulative bytes copied, latest={copied_bytes} bytes)",
        rank0_only=False,
    )


def _log_cgroup_setup_once(proc_root: Path = Path("/proc")) -> None:
    pid = os.getpid()
    if pid in _logged_cgroup_setup_pids:
        return
    _logged_cgroup_setup_pids.add(pid)
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else os.environ.get("RANK", "0")
    )
    cgroup_text = _read_text(proc_root / "self" / "cgroup")
    mountinfo = _read_text(proc_root / "self" / "mountinfo")
    cgroup_mounts = [line for line in (mountinfo or "").splitlines() if " - cgroup " in line or " - cgroup2 " in line]
    candidate_limits: list[tuple[str, str | None]] = []
    for line in (cgroup_text or "").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, cgroup_path = fields
        v2 = hierarchy == "0" and not controllers
        if not v2 and "memory" not in controllers.split(","):
            continue
        mount = _find_cgroup_mount(mountinfo or "", cgroup_path, v2=v2)
        if mount is None:
            candidate_limits.append((cgroup_path, "no matching visible mount"))
            continue
        mount_point, leaf = mount
        limit_name = "memory.max" if v2 else "memory.limit_in_bytes"
        for directory in (leaf, *leaf.parents):
            limit_path = directory / limit_name
            candidate_limits.append((str(limit_path), _read_text(limit_path)))
            if directory == mount_point:
                break
    static = _static_cgroup_memory(pid, proc_root)
    limit_paths = [
        (
            str(
                limit.usage_path.with_name(
                    "memory.max" if limit.usage_path.name == "memory.current" else "memory.limit_in_bytes"
                )
            ),
            limit.limit_bytes,
            str(limit.usage_path),
        )
        for limit in static.limits
    ]
    log.info(
        f"Activation CPU offload cgroup setup on rank={rank} pid={pid}: "
        f"/proc/self/cgroup={cgroup_text!r}; cgroup mountinfo={cgroup_mounts!r}; "
        f"candidate limits (path, raw value)={candidate_limits!r}; "
        f"resolved limits (path, bytes, usage path)={limit_paths!r}; error={static.error!r}",
        rank0_only=False,
    )


class ActivationOffloadContext(saved_tensors_hooks):
    """Offload activation-checkpoint saved tensors to host memory.

    Each pack allocates a CPU tensor. Autograd releases it after
    the owning graph node runs backward.

    Args:
        min_offload_bytes: CUDA tensors smaller than this stay on GPU.
            Defaults to 10 MiB.
        min_cgroup_memory_free_fraction: Minimum free fraction of every finite
            cgroup memory limit after a copy.
    """

    def __init__(
        self,
        min_offload_bytes: int = _MIN_OFFLOAD_BYTES,
        *,
        min_cgroup_memory_free_fraction: float = _DEFAULT_MIN_CGROUP_MEMORY_FREE_FRACTION,
    ) -> None:
        if not 0 <= min_cgroup_memory_free_fraction < 1:
            raise ValueError("min_cgroup_memory_free_fraction must be in [0, 1)")
        self.min_offload_bytes: int = min_offload_bytes
        self.min_cgroup_memory_free_fraction: float = min_cgroup_memory_free_fraction

        super().__init__(self._pack, self._unpack)

    # ------------------------------------------------------------------
    # Pack (forward direction): synchronous copy to the host.
    # ------------------------------------------------------------------
    # activation: [*shape]; returns [*shape] or ([*shape], device).
    def _pack(self, activation: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.device]:
        num_bytes = activation.element_size() * activation.nelement()
        if (
            activation.device.type != "cuda"
            or num_bytes < self.min_offload_bytes
            or isinstance(activation, torch.nn.Parameter)
        ):
            # Small, CPU, and parameter tensors stay as-is.
            return activation

        shortfall = _check_cgroup_memory_headroom(num_bytes, self.min_cgroup_memory_free_fraction)
        if shortfall is not None:
            reason_key, message = shortfall
            _warn_skipped_offload(reason_key, message)
            return activation
        # Device→pageable copies are synchronous in the CUDA runtime, so
        # the data is fully materialised on the host when this returns
        # and autograd may free the GPU tensor.
        cpu_tensor = torch.empty_like(activation, device="cpu")  # [*shape]
        cpu_tensor.copy_(activation, non_blocking=False)  # [*shape]
        _log_successful_offload(num_bytes)
        return (cpu_tensor, activation.device)

    # ------------------------------------------------------------------
    # Unpack (backward direction): fetch back to GPU.
    # ------------------------------------------------------------------
    # packed tensor: [*shape]; returns [*shape].
    def _unpack(self, packed: torch.Tensor | tuple[torch.Tensor, torch.device]) -> torch.Tensor:
        if isinstance(packed, torch.Tensor):
            # Was not offloaded — return as-is.
            return packed

        cpu_tensor, device = packed
        # Pageable host memory requires a synchronous copy on the current stream.
        return cpu_tensor.to(device=device, non_blocking=False)  # [*shape]


def checkpoint_with_flattened_inputs(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
    determinism_check: str = "default",
    context_fn: Callable[[], tuple[Any, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Save nested tensor inputs through the checkpoint's tensor-save path.

    ``SequencePack`` is a dictionary, so an uncompiled non-reentrant checkpoint
    otherwise retains it as one Python argument without presenting its tensors
    to the surrounding saved-tensor hooks. The flattened leaves are direct
    checkpoint arguments in both eager and compiled execution.
    """
    flat_inputs, spec = tree_flatten((args, kwargs))

    def call_with_flat_inputs(*saved_inputs: Any) -> Any:
        restored_args, restored_kwargs = tree_unflatten(list(saved_inputs), spec)
        return function(*restored_args, **restored_kwargs)

    checkpoint_options: dict[str, Any] = {
        "use_reentrant": False,
        "preserve_rng_state": preserve_rng_state,
        "determinism_check": determinism_check,
    }
    if context_fn is not None:
        checkpoint_options["context_fn"] = context_fn
    return checkpoint(call_with_flat_inputs, *flat_inputs, **checkpoint_options)


def offload_checkpoint_inputs(
    module: torch.nn.Module,
    *,
    min_cgroup_memory_free_fraction: float = _DEFAULT_MIN_CGROUP_MEMORY_FREE_FRACTION,
) -> None:
    """Scope saved-tensor offloading to one checkpointed module call.

    Install after ``torch.compile`` and before FSDP, so the hook surrounds the
    compiled checkpoint boundary without becoming part of its traced graph.
    """
    _log_cgroup_setup_once()
    forward = module.forward

    @wraps(forward)
    def offloaded_forward(*args: Any, **kwargs: Any) -> Any:
        with ActivationOffloadContext(min_cgroup_memory_free_fraction=min_cgroup_memory_free_fraction):
            return forward(*args, **kwargs)

    module.forward = offloaded_forward
