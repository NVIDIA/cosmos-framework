# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Durable checkpoint component, layout, and path IO."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from cosmos_framework.posttrain.utils.log import logger

LATEST_FILE = "latest_checkpoint.txt"


def release_page_cache(path: str, *, sync: bool = False) -> bool:
    """Ask the kernel to drop ``path``'s pages from the page cache; best effort, never raises.

    A checkpoint payload is written once, hashed, uploaded, and then only ever read again by a
    resume, so its cached pages are dead weight inside a job cgroup whose limit counts the page
    cache together with anonymous memory. ``sync=True`` flushes dirty pages first (``fsync``), which
    is what makes a just-written file droppable at all. Returns whether the advice was issued;
    ``False`` where ``posix_fadvise`` is unavailable or the file could not be opened or advised.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or flag is None:
        logger.debug(f"page cache release unavailable on this platform: {path}")
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        logger.debug(f"page cache release skipped ({error}): {path}")
        return False
    try:
        if sync:
            os.fsync(descriptor)
        fadvise(descriptor, 0, 0, flag)
    except OSError as error:
        logger.debug(f"page cache release failed ({error}): {path}")
        return False
    finally:
        os.close(descriptor)
    logger.debug(f"page cache released{' after fsync' if sync else ''}: {path}")
    return True


def fsync_directory(path: str) -> None:
    """Persist prior renames and directory-entry updates."""
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file through a same-directory temporary."""
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(os.path.dirname(path))


def atomic_write_bytes(path: str, payload: bytes) -> None:
    """Durably replace exact bytes through a same-directory temporary."""
    temporary = path + ".part"
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(os.path.dirname(path))


def update_latest(iter_dir: str) -> None:
    """Durably point a checkpoint root at one complete iteration directory."""
    root = os.path.dirname(os.path.abspath(iter_dir))
    pointer = os.path.join(root, LATEST_FILE)
    temporary = pointer + ".tmp"
    with open(temporary, "w") as handle:
        handle.write(os.path.basename(iter_dir) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, pointer)  # Readers never observe a torn pointer.
    fsync_directory(root)


def write_component_files(iter_dir: str, components: dict[str, dict | None]) -> bool:
    """Write non-empty consolidated components and report whether anything landed."""
    import torch

    wrote = False
    for name, state in components.items():
        if not state:
            continue
        os.makedirs(iter_dir, exist_ok=True)
        torch.save(state, os.path.join(iter_dir, f"{name}.pt"))
        wrote = True
    return wrote


def load_component(iter_dir: str, name: str) -> dict | None:
    """Load a consolidated component, or return ``None`` when it was not saved.

    ``mmap=True`` is load-bearing: every rank reads checkpoints that can exceed 100 GB, and mmap
    allows their tensors to share the filesystem page cache instead of materializing one full copy
    per rank.
    """
    import torch

    path = os.path.join(iter_dir, f"{name}.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def component_kind(iter_dir: str, name: str) -> str | None:
    """Return ``pt``, ``dcp``, or ``None`` for one checkpoint component."""
    if os.path.isfile(os.path.join(iter_dir, f"{name}.pt")):
        return "pt"
    if os.path.isdir(os.path.join(iter_dir, name)):
        return "dcp"
    return None


def save_component_dcp(iter_dir: str, name: str, state: dict) -> None:
    """Collectively save one sharded DCP component."""
    # Previous async archive/retention may still be pruning. Join it before the next DCP directory
    # appears, or that retention pass could mistake this in-progress save for a completed iter.
    from cosmos_framework.posttrain.checkpoint.worker import wait_for_pending

    wait_for_pending()
    import torch.distributed.checkpoint as dcp

    dcp.save(state, checkpoint_id=os.path.join(iter_dir, name))


def load_component_dcp(iter_dir: str, name: str, state: dict, *, allow_partial: bool = False) -> bool:
    """Load one DCP component in place, optionally allowing lazy optimizer entries."""
    import torch.distributed.checkpoint as dcp

    path = os.path.join(iter_dir, name)
    if not os.path.isdir(path):
        return False
    if allow_partial:
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

        dcp.load(state, checkpoint_id=path, planner=DefaultLoadPlanner(allow_partial_load=True))
    else:
        dcp.load(state, checkpoint_id=path)
    return True


def is_complete_iter_dir(iter_dir: str) -> bool:
    """Return whether a pointer-less ``iter_*`` directory is minimally resumable."""
    # DCP creates model/ before the final trainer.pt commit. Requiring both prevents pointer-loss
    # fallback from choosing a higher interrupted save over the preceding complete checkpoint.
    trainer_exists = os.path.isfile(os.path.join(iter_dir, "trainer.pt"))
    model_exists = os.path.isfile(os.path.join(iter_dir, "model.pt")) or os.path.isdir(os.path.join(iter_dir, "model"))
    return trainer_exists and model_exists


def checkpoint_trainer_iteration(path: str) -> int:
    """The optimizer iteration a resumable checkpoint committed, read from its trainer state.

    A checkpoint directory carries it in ``trainer.pt`` (the control component every complete ``iter_*`` commits
    last, in the consolidated and the DCP layout alike); a legacy single-file checkpoint carries it at the top
    level of the saved mapping. ``optimizer_step`` is the durable clock and ``iter`` its legacy spelling; the
    trainer writes them identical and its restore refuses a checkpoint where they differ, so the same holds here.
    Tensors are memory-mapped, never materialized: only the two small integers are read.

    Raises:
        ValueError: when ``path`` holds no readable trainer state or that state records no valid iteration. The
            caller must treat that as "unknown", never as zero.
    """
    import torch

    if path.startswith("s3://"):
        raise ValueError(f"trainer state at {path!r} is not readable locally")
    if os.path.isdir(path):
        source = "trainer.pt"
        state = load_component(path, "trainer")
        if state is None:
            raise ValueError(f"checkpoint directory {path!r} holds no trainer.pt")
    elif os.path.isfile(path):
        source = "the legacy checkpoint file"
        try:
            state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except Exception as error:  # noqa: BLE001 -- an unreadable checkpoint is one whose iteration is unknown
            raise ValueError(f"legacy checkpoint file {path!r} could not be read: {error}") from error
    else:
        raise ValueError(f"checkpoint {path!r} does not exist")
    if not isinstance(state, Mapping):
        raise ValueError(f"{source} at {path!r} is not a mapping of trainer state")
    clocks = {key: state[key] for key in ("optimizer_step", "iter") if key in state}
    if not clocks:
        raise ValueError(f"{source} at {path!r} records neither optimizer_step nor iter")
    for key, value in clocks.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{source} at {path!r} records an invalid {key}: {value!r}")
    if len(set(clocks.values())) != 1:
        raise ValueError(
            f"{source} at {path!r} records optimizer_step {clocks['optimizer_step']} but iter {clocks['iter']}"
        )
    return next(iter(clocks.values()))


def latest_dir(ckpt_root: str) -> str | None:
    """Resolve the latest complete iteration directory under a checkpoint root."""
    pointer = os.path.join(ckpt_root, LATEST_FILE)
    if os.path.isfile(pointer):
        with open(pointer) as handle:
            name = handle.read().strip()
        candidate = os.path.join(ckpt_root, name)
        if os.path.isdir(candidate) or os.path.isfile(candidate):
            return candidate
    if not os.path.isdir(ckpt_root):
        return None
    iterations = sorted(
        name
        for name in os.listdir(ckpt_root)
        if name.startswith("iter_") and is_complete_iter_dir(os.path.join(ckpt_root, name))
    )
    return os.path.join(ckpt_root, iterations[-1]) if iterations else None


_RUN_TS_RE = None


def resolve_run_ts(name_dir: str, resume: str, *, now: str | None = None) -> str:
    """Adopt or mint the timestamp identifying one run attempt."""
    global _RUN_TS_RE
    if _RUN_TS_RE is None:
        import re

        _RUN_TS_RE = re.compile(r"^\d{14}$")
    if os.path.isdir(os.path.join(name_dir, "checkpoints")):
        return ""
    fresh = not resume or resume == "off"
    if not fresh and resume != "auto":
        marker = os.path.abspath(resume)
        root = os.path.abspath(name_dir) + os.sep
        if marker.startswith(root):
            timestamp = marker[len(root) :].split(os.sep, 1)[0]
            if _RUN_TS_RE.match(timestamp):
                return timestamp
    if not fresh and os.path.isdir(name_dir):
        existing = sorted(name for name in os.listdir(name_dir) if _RUN_TS_RE.match(name))
        if existing:
            return existing[-1]
    if now is not None:
        return now
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M%S")


def resolve_resume(resume: str, ckpt_root: str) -> str | None:
    """Resolve ``off``, ``auto``, or an explicit checkpoint path."""
    if not resume or resume == "off":
        return None
    if resume == "auto":
        return latest_dir(ckpt_root)
    if os.path.isdir(resume) or os.path.isfile(resume):
        return resume
    raise FileNotFoundError(
        f"rl.resume={resume!r} does not exist (expected an iter_* checkpoint dir, a legacy "
        ".pt file, or 'auto'). Refusing to silently start from scratch."
    )


def rename_directory_noreplace(source: str, target: str) -> None:
    """Atomically rename a directory without replacing an existing target."""
    # A check followed by os.rename is unsafe: a competing process can create target between them.
    # Filesystems without RENAME_NOREPLACE fail closed and leave staging available for retry.
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory rename is unavailable on this host", target)
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory rename is unsupported by this kernel or filesystem",
                target,
            )
        raise OSError(error_number, os.strerror(error_number), target)
