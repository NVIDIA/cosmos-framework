# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Preemption/timeout-aware termination watch for training loops.

Slurm signal timeline, which is what the design below follows from:

- Timeout (job hits `--time`): SIGUSR1 to the batch shell N seconds early (N from
  `--signal=B:SIGUSR1@N`), then SIGTERM to everything, then SIGKILL after KillWait.
- Preemption (higher-priority job wants the nodes) with a GraceTime: the job's TASKS are signalled
  at selection time and SIGCONT/SIGTERM/SIGKILL follow at the new end time (slurm.conf(5),
  GraceTime). WHICH signal arrives at selection time is the load-bearing detail: with
  `PreemptParameters=send_user_signal` (`SlurmctldParameters=preempt_send_user_signal` in older
  releases) AND a `--signal=` on the job, it is the job's own user signal — `B:SIGUSR1` to the batch
  shell, the same path as the timeout; WITHOUT a `--signal=` on the job (or without the parameter)
  it is SIGTERM to every task process at once, and nothing that dies on SIGTERM lives to save. A job
  submitted without `--signal=` therefore gets no warning at all on a preempting QOS: the first thing
  it hears is the SIGTERM that ends it (docs/reference/checkpoint.md, "Preemption / timeout").
- User cancel (`scancel`): SIGTERM only, no SIGUSR1. So SIGUSR1 is precisely the signal
  that distinguishes "you are about to lose the nodes" from "a human stopped this run".

The load-bearing facts:

- Slurm delivers SIGUSR1 to the BATCH SHELL, not into the pyxis/enroot container — the
  battle-tested detection mechanism is a SENTINEL FILE the batch script's trap writes on
  the shared filesystem (`$SLURM_LOG_DIR/SIGUSR1_RECEIVED`); the loop polls it each step.
- There are >= 300 s of grace (`--signal=B:SIGUSR1@600` for timeout, `GraceTime=300` for
  preemption) — plenty for one save at a step boundary.
- We ALSO trap SIGUSR1/SIGTERM directly (flag-only, async-signal-safe): for torchrun runs
  outside a container the signals do reach the process, and logging SIGTERM delivery is
  useful even when it's too late to act. On the Ray grid this handler is installed in the driver
  AFTER `ray.init` (setup precedes the loop), so it replaces the `sys.exit(signum)` SIGTERM handler
  Ray installs on a driver and the loop survives to reach its step boundary. It protects ONLY the
  driver: each Ray worker process hosting a role actor keeps Ray's own SIGTERM handler
  (`ray/_private/worker.py`, `main_loop`: exit 1) and the `ray start` node process kills every Ray
  child on SIGTERM (`ray/_private/node.py`). Actor code cannot replace those handlers — `_Host`
  actor methods run on Ray executor threads (`workers/ray.py`), and Python accepts `signal.signal`
  only on the main thread — so a SIGTERM delivered to the tasks ends the collective before the
  driver can run its save. The user-signal path above is what makes the save possible.

Correctness note for multi-rank loops (torchrun/local backend runs the loop ON EVERY
rank): ranks can observe the sentinel/flag at different step boundaries, and the
emergency save is a COLLECTIVE (FSDP gather / RNG all_gather) — an inconsistent decision
deadlocks. `should_stop()` therefore all-reduces (MAX) the local flag so every rank
decides identically at the same step. On the Ray grid only the driver runs the loop, and
the fan-out save is uniform by construction.
"""

from __future__ import annotations

import os
import signal


class TerminationWatch:
    """Poll-at-step-boundary termination detector: sentinel file (Slurm path) + direct
    signals (torchrun path). Install once before the loop; call `should_stop()` once per
    step — it is collective-consistent across ranks (see module docstring)."""

    def __init__(self, sentinel: str | None = None):
        slurm_log_dir = os.environ.get("SLURM_LOG_DIR", "")
        default = os.path.join(slurm_log_dir, "SIGUSR1_RECEIVED") if slurm_log_dir else ""
        self._sentinel = sentinel if sentinel is not None else default
        self._signaled: str | None = None
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)  # flag-only: async-signal-safe
            except (ValueError, OSError):
                pass  # not the main thread of the process (e.g. a Ray worker) — sentinel still works

    def _on_signal(self, signum: int, frame: object) -> None:
        self._signaled = signal.Signals(signum).name
        print(f"[termination] received {self._signaled}", flush=True)

    def _local_reason(self) -> str | None:
        if self._sentinel and os.path.exists(self._sentinel):
            return f"sentinel {self._sentinel}"
        return self._signaled and f"signal {self._signaled}"

    def should_stop(self) -> str | None:
        """Reason string if ANY rank saw a termination trigger, else None — identical on
        every rank of the process group (all-reduce MAX), so the caller's emergency save
        stays a well-formed collective."""
        from cosmos_framework.posttrain.utils.dist import rank_any

        reason = self._local_reason()
        if not rank_any(bool(reason)):
            return None
        # `_local_reason` returns `self._signaled and ...`, so a falsy `_signaled` makes it False or
        # 0 rather than None. Normalize here: the annotation says `str | None` and the callers test
        # identity against None.
        return reason if isinstance(reason, str) else "peer rank signaled"
