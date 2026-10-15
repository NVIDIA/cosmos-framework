# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Render a manifest: which worker takes which run, and how ranks stay in step.

Two hazards live here, and both cost a night's compute before being understood.

**Ranks must agree.** The model runs distributed, so every rank walks this loop. Any decision made
on rank 0 alone -- whether a run is already claimed, whether it failed -- makes rank 0 branch while
the others enter the model's collectives with nobody to meet, and the job dies on an NCCL abort
instead of moving to the next run. Six shards of eight died that way. Every branch below is
therefore agreed across ranks before it is taken.

**A claim needs an expiry.** Workers steal work from a shared directory by creating a lock file.
A worker that dies mid-run leaves that lock behind with no output, and nothing will ever pick the
run up again. Locks older than the timeout with nothing rendered are treated as abandoned.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .config import Profile


def run_manifest(
    profile: Profile,
    model,
    runs: list[dict],
    donors: dict[str, dict],
    out_root: Path,
    *,
    seed: int = 0,
    shard: int = 0,
    shards: int = 1,
    claim_timeout: float = 900.0,
    is_rank0: Callable[[], bool] = lambda: True,
    log: Callable[[str], None] = print,
) -> dict:
    """Render everything in ``runs`` that no other worker has taken, and report what happened."""
    from .rollout import render

    out_root.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        if is_rank0():  # every rank runs this loop; only one should narrate it
            log(message)

    # Start each shard on its own stride so concurrent workers begin apart, then fall through to
    # everything else and let the claims sort out the overlap. Work stealing beats a static split
    # because a preempted or slow worker would otherwise strand its share.
    stride = runs[shard :: max(1, shards)]
    order = stride + [r for r in runs if r not in stride]
    todo = [r for r in order if not (out_root / r["name"] / "synth.mp4").is_file()]
    say(f"{len(todo)} runs to attempt of {len(runs)} in the manifest")

    done, failed, skipped = 0, [], 0
    for run in todo:
        run_dir = out_root / run["name"]
        verdict = agree(decide(run_dir, claim_timeout) if is_rank0() else None)
        if verdict != "take":
            skipped += verdict == "claimed"
            continue

        say(f"[{done + len(failed) + 1}/{len(todo)}] {run['name']}")
        ok = True
        try:
            render(
                profile,
                model,
                run,
                run_dir,
                donors.get(run["donor"]) if run.get("donor") else None,
                seed=seed,
                is_rank0=is_rank0,
            )
        except (Exception, SystemExit) as exc:  # one bad run must not cost the batch
            # SystemExit included on purpose: the reference model calls sys.exit on some input
            # errors, which inherits from BaseException and would otherwise kill the worker.
            log(f"  failed: {type(exc).__name__}: {exc}")
            ok = False
        if not all_ok(ok):
            failed.append(run["name"])
            if is_rank0():
                (run_dir / ".claim").unlink(missing_ok=True)  # let another worker retry it
        else:
            done += 1

    say(f"done: {done} rendered, {len(failed)} failed, {skipped} claimed by another worker")
    if failed:
        say("failed runs: " + ", ".join(failed))
    return {"rendered": done, "failed": failed, "skipped": skipped, "attempted": len(todo)}


def decide(run_dir: Path, claim_timeout_s: float) -> str:
    """One run's verdict: ``take`` it, skip it as ``done``, or leave it to its ``claimed`` owner.

    Only rank 0 should call this; the others receive the answer through :func:`agree`.
    """
    if (run_dir / "synth.mp4").is_file():
        return "done"
    run_dir.mkdir(parents=True, exist_ok=True)
    claim = run_dir / ".claim"
    try:
        claim.touch(exist_ok=False)
        return "take"
    except FileExistsError:
        pass
    if time.time() - claim.stat().st_mtime > claim_timeout_s:
        claim.touch()  # steal it, and reset the clock for whoever comes next
        return "take"
    return "claimed"


def agree(value):
    """Rank 0's value, on every rank. A no-op when not running distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            box = [value]
            dist.broadcast_object_list(box, src=0)
            return box[0]
    except Exception:
        pass
    return value


def all_ok(ok: bool) -> bool:
    """True only if every rank succeeded, so a failure on one is a failure for all."""
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            flag = torch.tensor([1 if ok else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            return bool(flag.item())
    except Exception:
        pass
    return ok
