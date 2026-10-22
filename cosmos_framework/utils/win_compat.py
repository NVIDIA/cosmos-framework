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


def install_gloo_cuda_collective_shims() -> bool:
    """Make torch.distributed collectives safe for CUDA tensors under gloo.

    NCCL has no Windows build, so gloo is the only available backend — but the
    Windows gloo build has no CUDA-aware collective path. Passing a CUDA tensor
    to ``all_reduce`` / ``broadcast`` / ``reduce`` / ``all_gather`` faults the
    process with an access violation (0xC0000005) and no Python traceback.

    These wrappers round-trip CUDA tensors through host memory and copy the
    result back in place, preserving the collectives' in-place contract. On a
    single rank the collectives are mathematically no-ops, so the only cost is
    a small D2H/H2D copy per call.

    Returns True if the shims were installed, False if they were already present
    or the platform does not need them.
    """
    if not WINDOWS:
        return False

    import functools

    import torch
    import torch.distributed as dist

    if getattr(dist, "_cosmos_win_gloo_shimmed", False):
        return False

    def _tensor_first(fn):
        """For all_reduce / broadcast / reduce: signature (tensor, ...)."""

        @functools.wraps(fn)
        def inner(tensor, *args, **kwargs):
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda and not kwargs.get("async_op", False):
                staged = tensor.detach().cpu()
                fn(staged, *args, **kwargs)
                tensor.copy_(staged.to(tensor.device))
                return None
            return fn(tensor, *args, **kwargs)

        return inner

    def _all_gather(fn):
        """Signature (tensor_list, tensor, ...)."""

        @functools.wraps(fn)
        def inner(tensor_list, tensor, *args, **kwargs):
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda and not kwargs.get("async_op", False):
                staged_out = [t.detach().cpu() for t in tensor_list]
                fn(staged_out, tensor.detach().cpu(), *args, **kwargs)
                for dst, src in zip(tensor_list, staged_out):
                    dst.copy_(src.to(dst.device))
                return None
            return fn(tensor_list, tensor, *args, **kwargs)

        return inner

    dist.all_reduce = _tensor_first(dist.all_reduce)
    dist.broadcast = _tensor_first(dist.broadcast)
    dist.reduce = _tensor_first(dist.reduce)
    dist.all_gather = _all_gather(dist.all_gather)
    dist._cosmos_win_gloo_shimmed = True
    return True


def link_dir(target: Union[str, os.PathLike], link: Union[str, os.PathLike]) -> str:
    """Point ``link`` at directory ``target``; return the strategy actually used.

    Creating a symlink on Windows needs either Developer Mode or
    SeCreateSymbolicLinkPrivilege, neither of which is guaranteed. Fall back to a
    directory junction (no privilege required), then to a plain text pointer file
    so a convenience alias never aborts a training run.

    Returns one of ``"symlink"``, ``"junction"``, ``"pointer-file"``.
    """
    target_s, link_s = os.fspath(target), os.fspath(link)
    try:
        os.symlink(target_s, link_s, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass

    if WINDOWS:
        try:
            import _winapi  # type: ignore[attr-defined]

            _winapi.CreateJunction(target_s, link_s)
            return "junction"
        except Exception:
            pass

    with open(link_s + ".txt", "w", encoding="utf-8") as handle:
        handle.write(target_s + "\n")
    return "pointer-file"
