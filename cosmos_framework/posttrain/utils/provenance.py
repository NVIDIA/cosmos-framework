# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run provenance: which framework code, interpreter, and launch produced one run attempt.

A run directory is reused across requeues and warm continuations, and the framework tree a job
executes is chosen by its batch script (``I4_ROOT``/``PYTHONPATH``), not by the checkout an operator
is reading. Git metadata alone does not identify that code either: several live trees can share one
HEAD (or none) while their working files differ. Each attempt therefore records a content hash of
the posttrain sources it actually imported next to the git view, the interpreter, and the launch
context, as ``<run_dir>/provenance.<attempt_ts>.json`` and as the top-level ``provenance`` mapping
of ``config_resolved.yaml``. When a recipe's ``build_config`` resolved a named variant it registers the
name here (:func:`record_recipe_variant`) and the record carries it as ``recipe_variant`` (review
The resolved config carries the variant's VALUES -- ``rl.mask_think_opener`` tells
``prefilled_opener`` from ``default`` today -- but not its NAME, so a readout could not name the variant
that produced them; the name is recorded here).

The git view is ``git_commit`` (HEAD) and ``git_dirty``: whether the tracked files under
``git_dirty_scope`` differ from HEAD. The scope is the posttrain package when it lives inside the
checkout (the tree ``tree_hash`` describes; ``git status`` limited to its pathspec stats hundreds of
files, not the checkout's twenty thousand on a networked filesystem, which is what made the
unscoped probe miss its time box and record ``null`` without a reason in the canaries),
else the whole checkout. ``(git_commit, git_dirty=False)`` therefore names the committed
posttrain tree; ``tree_hash`` names its bytes either way.

Failure semantics: provenance is best effort and must never fail or stall the run it describes.
Every probe is guarded individually (a failed probe records ``None`` and its reason under
``errors``), the source-tree walk runs in a subprocess with an outer timeout, and the git probes
are time-boxed. A git probe that cannot answer because git is absent or the directory is not a
checkout records ``None`` without an error (a tree exported with ``git archive`` has no git view);
one that times out or fails for another reason records ``None`` AND the reason, so a ``null`` is
never silent. All probes share ``budget_s``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from cosmos_framework.posttrain.utils.log import logger

POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
"""The posttrain package this module was imported from: the tree whose code runs."""

REPO_ROOT = POSTTRAIN_ROOT.parents[1]
"""The checkout containing it (``<root>/cosmos_framework/posttrain``)."""

GIT_TIMEOUT_S = 5.0
BUDGET_S = 8.0
ENV_KEYS = ("I4_ROOT", "PYTHONPATH")
TREE_HASH_CHUNK_BYTES = 1024 * 1024
TREE_HASH_COMMAND = "--hash-tree-json"


class TreeHash(NamedTuple):
    sha256: str
    files: int


class _RecipeVariant:
    """Process-wide holder of the recipe variant name a config builder resolved; ``None`` until one did (a GRPO run, or
    a config loaded from YAML, records ``recipe_variant: null``)."""

    name: str | None = None


def record_recipe_variant(name: str) -> None:
    """Register the RESOLVED recipe variant name for this process's provenance record.

    ``SFT_RECIPE_VARIANT`` is not in ``ENV_KEYS`` and the resolved config carries the variant's VALUES
    (``rl.mask_think_opener`` tells ``prefilled_opener`` from ``default`` today) but not its NAME, so a readout could not
    name the variant that produced them; the name is recorded here. A recipe's ``build_config`` calls this after the
    variant applied
    (an explicit ``build_config(variant=...)`` is recorded like one taken from the environment); ``collect_provenance``
    records it as ``recipe_variant`` beside ``git_commit``. An empty or non-string name is refused.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"recipe variant name must be a non-empty string, got {name!r}")
    _RecipeVariant.name = name


def recipe_variant() -> str | None:
    """The registered recipe variant name, or ``None`` when no recipe resolved one in this process."""
    return _RecipeVariant.name


def _file_sha256(path: Path, *, deadline: float | None) -> str:
    """Hash one file in chunks, honoring the direct caller's cooperative deadline."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"tree hash exceeded its deadline while reading {path}")
            chunk = handle.read(TREE_HASH_CHUNK_BYTES)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def hash_tree(root: str | os.PathLike[str], *, suffix: str = ".py", deadline: float | None = None) -> TreeHash:
    """Content identity of every ``suffix`` file under ``root``.

    sha256 over the sorted ``(relative posix path, sha256(file bytes))`` pairs, so the value depends
    only on source paths and contents: not on the root's location, traversal order, mtimes, or files
    without the suffix. Hidden and ``__pycache__`` directories are skipped; symlinked directories are
    not followed.

    Raises:
        FileNotFoundError: ``root`` is not a directory (a hash of nothing would look like a tree).
        TimeoutError: ``deadline`` (``time.monotonic`` seconds) passed between filesystem
            operations. Use :func:`hash_tree_with_timeout` when the caller must also bound a blocked
            walk or read syscall.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"tree hash root is not a directory: {root}")
    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith(".") and name != "__pycache__")
        for filename in filenames:
            if not filename.endswith(suffix):
                continue
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"tree hash of {root} exceeded its deadline after {len(entries)} files")
            path = Path(dirpath, filename)
            entries.append((path.relative_to(root).as_posix(), _file_sha256(path, deadline=deadline)))
    digest = hashlib.sha256()
    for relative, file_sha256 in sorted(entries):
        digest.update(f"{relative}\0{file_sha256}\n".encode())
    return TreeHash(digest.hexdigest(), len(entries))


def hash_tree_with_timeout(
    root: str | os.PathLike[str],
    *,
    suffix: str = ".py",
    timeout_s: float,
) -> TreeHash:
    """Hash a tree in an isolated process whose walk and reads are bounded by ``timeout_s``.

    The inner cooperative deadline cannot interrupt an ``os.walk`` or ``read`` blocked in the
    kernel. The outer subprocess timeout can terminate that worker while the training driver
    continues, so provenance never waits indefinitely on a shared filesystem.
    """
    if timeout_s <= 0:
        raise TimeoutError(f"tree hash of {os.fspath(root)} has no time remaining")
    pythonpath = os.pathsep.join(os.path.abspath(entry or os.curdir) for entry in sys.path)
    env = {**os.environ, "PYTHONPATH": pythonpath}
    command = [
        sys.executable,
        os.fspath(Path(__file__).resolve()),
        TREE_HASH_COMMAND,
        os.fspath(root),
        suffix,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"tree hash of {os.fspath(root)} exceeded {timeout_s:g}s") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
        raise RuntimeError(f"tree hash worker failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"tree hash worker returned invalid JSON: {completed.stdout!r}") from error
    if not payload.get("ok", False):
        error_type = str(payload.get("error_type", "RuntimeError"))
        message = str(payload.get("error", "tree hash worker failed"))
        if error_type == "FileNotFoundError":
            raise FileNotFoundError(message)
        if error_type == "TimeoutError":
            raise TimeoutError(message)
        raise RuntimeError(f"{error_type}: {message}")
    return TreeHash(str(payload["sha256"]), int(payload["files"]))


NO_GIT_VIEW_MARKERS = (
    "not a git repository",
    "cannot change to",  # `git -C <dir>` on a directory that does not exist
    "unknown revision",
    "ambiguous argument",
    "bad revision",
)
"""git stderr that means "there is nothing to answer" (not a checkout, a missing directory, an unborn HEAD): ``None``
without an error."""


class GitProbeError(RuntimeError):
    """A git probe had an answer to give and did not get it: it exceeded its time box or git failed for a reason
    other than "not a repository". The message is the reason the record keeps under ``errors``."""


def _git(repo_root: str | os.PathLike[str], args: list[str], timeout_s: float) -> str | None:
    """stdout of one git command; None when git is absent or ``repo_root`` is not inside a checkout.

    Raises ``GitProbeError`` when the command exceeds ``timeout_s`` or exits non-zero for any other reason, so the
    caller's probe records why the field is ``None`` (the canaries' ``git_dirty: null`` carried no
    reason).
    """
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise GitProbeError(f"git {args[0]} exceeded its {timeout_s:g}s time box") from error
    except FileNotFoundError:
        return None
    except (OSError, subprocess.SubprocessError) as error:
        raise GitProbeError(f"git {args[0]} did not run: {type(error).__name__}: {error}") from error
    if completed.returncode == 0:
        return completed.stdout
    reason = completed.stderr.strip().splitlines()[0] if completed.stderr.strip() else f"exit {completed.returncode}"
    if any(marker in reason for marker in NO_GIT_VIEW_MARKERS):
        return None
    raise GitProbeError(f"git {args[0]} exited {completed.returncode}: {reason[:200]}")


def git_commit(repo_root: str | os.PathLike[str], *, timeout_s: float = GIT_TIMEOUT_S) -> str | None:
    """``HEAD`` of the checkout at ``repo_root``; None when there is no git view (see ``_git`` for the raised cases)."""
    output = _git(repo_root, ["rev-parse", "HEAD"], timeout_s)
    return (output.strip() or None) if output else None


def git_dirty(
    repo_root: str | os.PathLike[str], *, paths: tuple[str, ...] = (), timeout_s: float = GIT_TIMEOUT_S
) -> bool | None:
    """Whether tracked files at ``repo_root`` (under ``paths``, pathspecs relative to it, when given) differ from
    ``HEAD``; None when there is no git view (see ``_git`` for the raised cases)."""
    args = ["status", "--porcelain", "--untracked-files=no"]
    if paths:
        args += ["--", *paths]
    output = _git(repo_root, args, timeout_s)
    return None if output is None else bool(output.strip())


def dirty_scope(repo_root: str | os.PathLike[str], posttrain_root: str | os.PathLike[str]) -> str | None:
    """The pathspec ``git_dirty`` is limited to: the posttrain package relative to the checkout when it lives inside
    it, else None (the whole checkout)."""
    try:
        relative = Path(posttrain_root).resolve().relative_to(Path(repo_root).resolve())
    except ValueError:
        return None
    scope = relative.as_posix()
    return scope if scope not in ("", ".") else None


def _torch_version() -> str | None:
    module = sys.modules.get("torch")
    if module is not None:
        return str(module.__version__)
    from importlib import metadata

    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


def collect_provenance(
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    posttrain_root: str | os.PathLike[str] = POSTTRAIN_ROOT,
    attempt_ts: str | None = None,
    budget_s: float = BUDGET_S,
) -> dict[str, Any]:
    """One JSON-compatible provenance record for the current process; never raises.

    The tree hash runs first in a subprocess: it is the field that identifies the code, and the
    outer timeout bounds filesystem traversal and reads that block inside the kernel. The git probes
    then share what is left of ``budget_s``, each capped at ``GIT_TIMEOUT_S``. A probe that fails
    leaves ``None`` in its field and the reason under ``errors``.
    """
    deadline = time.monotonic() + budget_s
    errors: dict[str, str] = {}

    def probe(name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 -- provenance must never fail the run it describes
            errors[name] = f"{type(error).__name__}: {error}"
            return None

    def remaining(cap: float) -> float:
        available = min(cap, deadline - time.monotonic())
        if available <= 0:
            raise TimeoutError(f"provenance collection exceeded its {budget_s:g}s budget")
        return available

    now = datetime.now().astimezone()
    tree = probe(
        "tree_hash",
        lambda: hash_tree_with_timeout(posttrain_root, timeout_s=remaining(BUDGET_S)),
    )
    scope = dirty_scope(repo_root, posttrain_root)
    record: dict[str, Any] = {
        "attempt_ts": attempt_ts or now.strftime("%Y%m%d%H%M%S"),
        "timestamp": now.isoformat(timespec="seconds"),
        "hostname": probe("hostname", socket.gethostname),
        # The torchrun launcher unsets SLURM_JOB_ID before the entry runs (it is the rendezvous id and must
        # not reach the workers); it re-exports the value as POSTTRAIN_SLURM_JOB_ID first.
        "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("POSTTRAIN_SLURM_JOB_ID") or None,
        "python_version": platform.python_version(),
        "torch_version": probe("torch_version", _torch_version),
        "argv": list(sys.argv),
        "cwd": probe("cwd", os.getcwd),
        "env": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
        "repo_root": os.fspath(repo_root),
        "posttrain_root": os.fspath(posttrain_root),
        "git_commit": probe("git_commit", lambda: git_commit(repo_root, timeout_s=remaining(GIT_TIMEOUT_S))),
        "git_dirty": probe(
            "git_dirty",
            lambda: git_dirty(repo_root, paths=(scope,) if scope else (), timeout_s=remaining(GIT_TIMEOUT_S)),
        ),
        "git_dirty_scope": scope or ".",
        # The recipe variant a config builder resolved in this process; None when none did.
        "recipe_variant": recipe_variant(),
        "tree_hash": tree.sha256 if tree is not None else None,
        "tree_files": tree.files if tree is not None else None,
    }
    if errors:
        record["errors"] = errors
    return record


def write_provenance(out_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> str:
    """Write ``record`` to ``<out_dir>/provenance.<attempt_ts>.json`` without overwriting an earlier attempt.

    Two attempts stamped in the same second get ``-1``, ``-2``, ... suffixes; the file is created
    exclusively so a concurrent writer can never clobber another attempt's record.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = f"provenance.{record['attempt_ts']}"
    for suffix in ("", *(f"-{index}" for index in range(1, 100))):
        path = os.path.join(out_dir, f"{stem}{suffix}.json")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path
    raise FileExistsError(f"{stem}*.json: every candidate name is taken under {os.fspath(out_dir)}")


def embed_provenance(resolved_config: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Set ``provenance`` as a top-level mapping of the YAML document at ``resolved_config``.

    Load-modify-dump keeps the document one mapping (a re-stamp replaces the key instead of appending
    a duplicate). The file is the plain-scalar YAML ``LazyConfig.save_yaml`` writes with the same
    dumper settings, so the round trip changes nothing but the new key. The new document is written
    to a sibling ``.tmp`` and renamed over the original, so the run's primary config artifact is at
    every instant either the old or the new complete file, never truncated by a kill mid-dump.
    """
    import yaml

    with open(resolved_config) as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"{os.fspath(resolved_config)}: expected a top-level mapping, found {type(document).__name__}")
    document["provenance"] = dict(record)
    staged = f"{os.fspath(resolved_config)}.tmp"
    try:
        with open(staged, "w") as handle:
            yaml.safe_dump(document, handle, default_flow_style=False, sort_keys=True)
        os.replace(staged, resolved_config)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def stamp_provenance(
    out_dir: str | os.PathLike[str],
    *,
    resolved_config: str | os.PathLike[str] | None = None,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Record this attempt's provenance beside the run artifacts; never raises.

    Writes ``provenance.<attempt_ts>.json`` under ``out_dir`` and, when ``resolved_config`` names a
    YAML mapping, embeds the same record as its top-level ``provenance`` key. ``record`` defaults to
    :func:`collect_provenance`. Returns the record, or None when not even the JSON could be written;
    either failure is a warning in the log, never an error for the run.
    """
    try:
        stamped = dict(record) if record is not None else collect_provenance()
        path = write_provenance(out_dir, stamped)
    except Exception as error:  # noqa: BLE001 -- provenance must never fail the run it describes
        logger.warning(
            f"provenance: not recorded under {os.fspath(out_dir)} ({type(error).__name__}: {error}); continuing"
        )
        return None
    if resolved_config is not None:
        try:
            embed_provenance(resolved_config, stamped)
        except Exception as error:  # noqa: BLE001 -- the JSON record already exists; the YAML copy is a convenience
            logger.warning(
                f"provenance: not embedded in {os.fspath(resolved_config)} ({type(error).__name__}: {error}); see {path}"
            )
    logger.info(
        f"provenance: git_commit={stamped.get('git_commit')} git_dirty={stamped.get('git_dirty')} "
        f"(scope {stamped.get('git_dirty_scope')}) recipe_variant={stamped.get('recipe_variant')} "
        f"tree_hash={stamped.get('tree_hash')} "
        f"tree_files={stamped.get('tree_files')} errors={stamped.get('errors')} -> {path}"
    )
    return stamped


def _main(argv: list[str]) -> int:
    """Internal subprocess protocol for :func:`hash_tree_with_timeout`."""
    if len(argv) != 3 or argv[0] != TREE_HASH_COMMAND:
        return 2
    try:
        tree = hash_tree(argv[1], suffix=argv[2])
        payload: dict[str, Any] = {"ok": True, "sha256": tree.sha256, "files": tree.files}
    except Exception as error:  # noqa: BLE001 -- the parent records this probe failure
        payload = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
