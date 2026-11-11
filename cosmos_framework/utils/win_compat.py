# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compatibility shims for POSIX-only primitives, for the native-Windows port.

On Linux every helper here delegates to the stdlib primitive it replaces, so the
behaviour is byte-for-byte unchanged. On Windows it provides an equivalent built
on ``msvcrt`` / Win32.

Motivation: several modules on the reasoner/VLM SFT path import ``fcntl`` at
module scope, which makes the whole module unimportable on Windows even when the
locking code itself is never exercised.
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO, Union

WINDOWS: bool = sys.platform == "win32"

# fcntl.flock operation constants, mirrored so callers can use one spelling.
LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

_FileOrFd = Union[int, IO]


def _as_fd(f: _FileOrFd) -> int:
    return f if isinstance(f, int) else f.fileno()


if not WINDOWS:
    import fcntl as _fcntl

    def flock(f: _FileOrFd, operation: int) -> None:
        """Advisory whole-file lock. Thin passthrough to ``fcntl.flock``."""
        _fcntl.flock(_as_fd(f), operation)

else:
    import msvcrt

    # msvcrt.locking() locks a byte range starting at the current file offset.
    # A single byte at offset 0 is the conventional stand-in for a whole-file
    # advisory lock, and works even on a zero-length file.
    _LOCK_BYTES = 1
    _BLOCKING_TIMEOUT_S = 300.0
    _POLL_INTERVAL_S = 0.05

    def flock(f: _FileOrFd, operation: int) -> None:
        """Advisory whole-file lock implemented with ``msvcrt.locking``.

        Differences from ``fcntl.flock`` that do not matter for the call sites in
        this repo (both take LOCK_EX and hold it briefly around a download or a
        cache warmup):

        * Windows byte-range locks are *mandatory*, not advisory.
        * There is no shared-lock mode; ``LOCK_SH`` is upgraded to exclusive.
        """
        fd = _as_fd(f)
        # msvcrt.locking() is offset-relative, so normalise to 0 and restore.
        try:
            pos = os.lseek(fd, 0, os.SEEK_CUR)
        except OSError:
            pos = None
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if operation & LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, _LOCK_BYTES)
                return
            if operation & LOCK_NB:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCK_BYTES)
                return
            deadline = time.monotonic() + _BLOCKING_TIMEOUT_S
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCK_BYTES)
                    return
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(_POLL_INTERVAL_S)
        finally:
            if pos is not None:
                try:
                    os.lseek(fd, pos, os.SEEK_SET)
                except OSError:
                    pass


# torch.distributed collectives that the gloo shims stage through host memory, and the other
# tensor collectives, which are only guarded: the Windows gloo build faults on a CUDA tensor in
# any of them, and a clear error beats an access violation with no Python traceback.
_GLOO_STAGED_COLLECTIVES = ("all_reduce", "broadcast", "reduce", "all_gather")
_GLOO_GUARDED_COLLECTIVES = (
    "all_gather_coalesced",
    "all_gather_into_tensor",
    "all_reduce_coalesced",
    "all_to_all",
    "all_to_all_single",
    "gather",
    "irecv",
    "isend",
    "recv",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "scatter",
    "send",
)


def _shim_gloo_collectives(namespaces, needs_staging) -> list[str]:
    """Wrap the tensor collectives found on ``namespaces``; return the names wrapped.

    ``needs_staging(tensor)`` says whether a tensor cannot be handed to the backend as is.
    Split out of :func:`install_gloo_cuda_collective_shims` so the wrappers can be tested
    without Windows or CUDA.
    """
    import functools
    import inspect

    import torch

    def tensors_in(values):
        for value in values:
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (item for item in value if isinstance(item, torch.Tensor))

    def staged(name, fn):
        """all_reduce / broadcast / reduce: (tensor, ...); all_gather: (tensor_list, tensor, ...)."""
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                return fn(*args, **kwargs)  # let the collective report the bad call itself
            call = bound.arguments
            tensor, tensor_list = call.get("tensor"), call.get("tensor_list")
            if not isinstance(tensor, torch.Tensor) or not any(
                needs_staging(t) for t in tensors_in([tensor, tensor_list])
            ):
                return fn(*args, **kwargs)
            # async_op is resolved from the bound call, so a positional True is seen as well.
            if call.get("async_op", False):
                raise RuntimeError(
                    f"torch.distributed.{name}(async_op=True) on a CUDA tensor is not supported on "
                    "Windows: the gloo backend cannot take CUDA tensors and the host-memory staging "
                    "is synchronous. Use async_op=False or pass a CPU tensor."
                )
            # copy=True keeps the staged tensor distinct from the caller's on every device.
            host = tensor.detach().to("cpu", copy=True)
            call["tensor"] = host
            if tensor_list is None:
                fn(*bound.args, **bound.kwargs)
                tensor.copy_(host.to(tensor.device))
            else:
                host_list = [t.detach().to("cpu", copy=True) for t in tensor_list]
                call["tensor_list"] = host_list
                fn(*bound.args, **bound.kwargs)
                for dst, src in zip(tensor_list, host_list):
                    dst.copy_(src.to(dst.device))
            return None

        return inner

    def guarded(name, fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if any(needs_staging(t) for t in tensors_in([*args, *kwargs.values()])):
                raise RuntimeError(
                    f"torch.distributed.{name} was called with a CUDA tensor. The Windows gloo "
                    "backend cannot run collectives on CUDA tensors, and only "
                    f"{', '.join(_GLOO_STAGED_COLLECTIVES)} are staged through host memory. "
                    "Move the tensor to the CPU first."
                )
            return fn(*args, **kwargs)

        return inner

    wrappers = {}
    for name in (*_GLOO_STAGED_COLLECTIVES, *_GLOO_GUARDED_COLLECTIVES):
        fn = next((getattr(ns, name) for ns in namespaces if hasattr(ns, name)), None)
        if fn is None:
            continue
        stageable = name in _GLOO_STAGED_COLLECTIVES and "tensor" in inspect.signature(fn).parameters
        wrappers[name] = staged(name, fn) if stageable else guarded(name, fn)
    for ns in namespaces:
        for name, wrapper in wrappers.items():
            if hasattr(ns, name):
                setattr(ns, name, wrapper)
    return list(wrappers)


def install_gloo_cuda_collective_shims() -> bool:
    """Make torch.distributed collectives safe for CUDA tensors under gloo.

    NCCL has no Windows build, so gloo is the only available backend — but the
    Windows gloo build has no CUDA-aware collective path. Passing a CUDA tensor
    to ``all_reduce`` / ``broadcast`` / ``reduce`` / ``all_gather`` faults the
    process with an access violation (0xC0000005) and no Python traceback.

    These wrappers round-trip CUDA tensors through host memory and copy the
    result back in place, preserving the collectives' in-place contract. On a
    single rank the collectives are mathematically no-ops, so the only cost is
    a small D2H/H2D copy per call. ``async_op=True`` cannot be staged this way
    and raises for a CUDA tensor, as do the other tensor collectives
    (``_GLOO_GUARDED_COLLECTIVES``), which this repo does not call.

    The wrappers replace the attributes of ``torch.distributed`` and
    ``torch.distributed.distributed_c10d``, which is how this repo reaches the
    collectives (``dist.all_reduce(...)``). A name bound earlier with
    ``from torch.distributed import all_reduce`` keeps the unwrapped function.

    Returns True if the shims were installed, False if they were already present
    or the platform does not need them.
    """
    if not WINDOWS:
        return False

    import torch.distributed as dist
    from torch.distributed import distributed_c10d

    if getattr(dist, "_cosmos_win_gloo_shimmed", False):
        return False

    _shim_gloo_collectives((dist, distributed_c10d), needs_staging=lambda tensor: tensor.is_cuda)
    dist._cosmos_win_gloo_shimmed = True
    return True


def link_dir(target: Union[str, os.PathLike], link: Union[str, os.PathLike]) -> str:
    """Point ``link`` at directory ``target``; return the strategy actually used.

    Off Windows this is ``os.symlink`` and nothing else: a failure raises, as it
    did before this helper existed.

    Creating a symlink on Windows needs either Developer Mode or
    SeCreateSymbolicLinkPrivilege, neither of which is guaranteed. There, fall back
    to a directory junction (no privilege required), then to a plain text pointer
    file so a convenience alias never aborts a training run.

    Returns one of ``"symlink"``, ``"junction"``, ``"pointer-file"``.
    """
    target_s, link_s = os.fspath(target), os.fspath(link)
    if not WINDOWS:
        os.symlink(target_s, link_s, target_is_directory=True)
        return "symlink"

    try:
        os.symlink(target_s, link_s, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        pass

    try:
        import _winapi  # type: ignore[attr-defined]

        _winapi.CreateJunction(target_s, link_s)
        return "junction"
    except (OSError, ImportError, AttributeError):
        pass

    with open(link_s + ".txt", "w", encoding="utf-8") as handle:
        handle.write(target_s + "\n")
    return "pointer-file"


def remove_dir_link(link: Union[str, os.PathLike]) -> None:
    """Remove what :func:`link_dir` created at ``link``; never delete a directory's contents.

    Off Windows this is ``os.remove`` and nothing else, so a real directory at
    ``link`` raises as it did before this helper existed. On Windows a junction or
    a directory symlink may need ``os.rmdir``; that also removes only the link (or
    an empty directory) and raises on a directory that has contents.
    """
    link_s = os.fspath(link)
    if not WINDOWS:
        os.remove(link_s)
        return
    try:
        os.remove(link_s)
    except OSError:
        os.rmdir(link_s)


def attention_implementation(requested: str) -> str:
    """Return the attention implementation to use for ``requested`` on this platform.

    ``cosmos`` gives the Qwen3-VL vision tower a single varlen attention call per block, which
    only the FlashAttention backends serve (cuDNN has no varlen integration). FlashAttention has
    no Windows build, so on ``win32`` the first training step fails with "Could not find a
    compatible Attention backend". Fall back to ``sdpa`` there: the trainer forwards
    ``attention_mask`` under sdpa, so the fallback trains correctly. Other platforms and other
    implementations are returned unchanged.
    """
    if requested == "cosmos" and sys.platform == "win32":
        from cosmos_framework.utils import log

        log.warning(
            "attn_implementation='cosmos' has no attention backend for the Qwen3-VL vision tower on "
            "Windows (no FlashAttention build; cuDNN rejects varlen). Using 'sdpa' instead."
        )
        return "sdpa"
    return requested
