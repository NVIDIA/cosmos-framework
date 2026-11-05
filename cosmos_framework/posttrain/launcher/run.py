# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Atomic run identity reservation and recoverable scheduler hand-off for launchers.

The marker is both the pre-submission reservation and the accepted run's durable global-step record. A launcher that
fails before the scheduler accepts its training job releases only the exact reservation it created. When an accepted
dependency must outlive a rejected training submission, a separate launch-state record preserves its job identity so a
later invocation can reuse or reconcile it instead of either duplicating the dependency or permanently blocking the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes, fsync_directory

GLOBAL_START_MARKER = ".global_start"
LAUNCH_STATE_MARKER = ".posttrain-launch.json"
LAUNCH_STATE_PROTOCOL = "posttrain.run-launch.v1"
LAUNCH_LOCK_DIRECTORY = ".posttrain-launch-locks"


@dataclass(frozen=True)
class RunLaunchState:
    """Durable identity of a scheduler-accepted dependency or training job."""

    start: int
    phase: str
    artifact: str
    job_id: str | None
    dependency_job_id: str | None


@dataclass(frozen=True)
class RunReservation:
    """One run directory's exclusive global-step identity."""

    runs_root: str
    run: str
    start: int

    @property
    def directory(self) -> str:
        return os.path.join(self.runs_root, self.run)

    @property
    def marker(self) -> str:
        return os.path.join(self.directory, GLOBAL_START_MARKER)

    @property
    def state_path(self) -> str:
        return os.path.join(self.directory, LAUNCH_STATE_MARKER)

    @property
    def lock_path(self) -> str:
        return os.path.join(self.runs_root, LAUNCH_LOCK_DIRECTORY, f"{self.run}.lock")

    def reserve(self) -> str:
        """Create the marker without replacement; return its path."""
        os.makedirs(self.directory, exist_ok=True)
        if os.path.isfile(self.marker):
            current = _read(self.marker)
            raise FileExistsError(
                f"global-start already reserved or launched: {self.marker} records {current}, requested {self.start}"
            )
        temporary = f"{self.marker}.tmp.{os.getpid()}"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(f"{self.start}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, self.marker)
            fsync_directory(self.directory)
        except FileExistsError as error:
            current = _read(self.marker)
            raise FileExistsError(
                f"global-start reservation lost: {self.marker} records {current}, requested {self.start}"
            ) from error
        finally:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass
        return self.marker

    def release(self) -> None:
        """Remove only a marker that still records this reservation's start."""
        current = _read(self.marker)
        if current != str(self.start):
            raise RuntimeError(f"refusing to release {self.marker}: records {current}, owned {self.start}")
        if os.path.exists(self.state_path):
            raise RuntimeError(
                f"refusing to release {self.marker}: accepted scheduler state remains at {self.state_path}"
            )
        os.remove(self.marker)
        fsync_directory(self.directory)
        try:
            os.rmdir(self.directory)
        except OSError:
            pass

    def begin(self, artifact: str) -> RunLaunchState | None:
        """Reserve a new run or resume its recorded dependency.

        Callers must hold an exclusive lock on :attr:`lock_path` until the scheduler hand-off finishes. ``None`` means
        this invocation created a fresh reservation; a returned dependency is safe to reconcile and reuse.
        """
        artifact = os.path.abspath(artifact)
        if not os.path.exists(self.marker):
            if os.path.exists(self.state_path):
                raise RuntimeError(f"launch state exists without {self.marker}: {self.state_path}")
            self.reserve()
            return None
        current = _read(self.marker)
        if current != str(self.start):
            raise FileExistsError(
                f"global-start already reserved or launched: {self.marker} records {current}, requested {self.start}"
            )
        state = self.read_state()
        if state is None or state.phase != "dependency":
            raise FileExistsError(f"global-start already reserved or launched: {self.marker} records {current}")
        if state.artifact != artifact:
            raise RuntimeError(
                f"refusing to resume {self.run}: dependency artifact {state.artifact!r} != requested {artifact!r}"
            )
        return state

    def read_state(self) -> RunLaunchState | None:
        """Read and validate the optional scheduler hand-off record."""
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            raise RuntimeError(f"unreadable launch state {self.state_path}: {error}") from error
        if not isinstance(payload, dict) or payload.get("protocol") != LAUNCH_STATE_PROTOCOL:
            raise RuntimeError(f"unsupported launch state at {self.state_path}")
        try:
            state = RunLaunchState(
                start=int(payload["start"]),
                phase=str(payload["phase"]),
                artifact=str(payload["artifact"]),
                job_id=str(payload["job_id"]) if payload.get("job_id") is not None else None,
                dependency_job_id=(
                    str(payload["dependency_job_id"]) if payload.get("dependency_job_id") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid launch state at {self.state_path}: {error}") from error
        valid_job = state.job_id is not None and state.job_id.isdigit()
        valid_dependency = state.dependency_job_id is None or state.dependency_job_id.isdigit()
        valid_phase = (
            (state.phase == "dependency" and valid_job and state.dependency_job_id is None)
            or (state.phase == "submitting" and state.job_id is None and valid_dependency)
            or (state.phase == "launched" and valid_job and state.dependency_job_id is None)
        )
        if state.start != self.start or not valid_phase:
            raise RuntimeError(f"launch state at {self.state_path} does not match run start {self.start}")
        return state

    def record_dependency(self, job_id: str, artifact: str) -> None:
        """Record a scheduler-accepted dependency before attempting the training submission."""
        current = self.read_state()
        if current is not None and not (
            current.phase == "dependency"
            and current.job_id == str(job_id)
            and current.artifact == os.path.abspath(artifact)
        ):
            raise RuntimeError(f"refusing to replace {current.phase} state at {self.state_path} with a dependency")
        self._write_state("dependency", artifact, job_id=job_id)

    def prepare_submission(self, artifact: str) -> None:
        """Persist the fail-closed hand-off before asking the scheduler to accept training."""
        artifact = os.path.abspath(artifact)
        current = self.read_state()
        if current is not None and current.phase != "dependency":
            raise RuntimeError(f"refusing to prepare training from {current.phase} state at {self.state_path}")
        if current is not None and current.artifact != artifact:
            raise RuntimeError(f"refusing to prepare training for {artifact!r}; dependency owns {current.artifact!r}")
        self._write_state(
            "submitting",
            artifact,
            dependency_job_id=current.job_id if current is not None else None,
        )

    def abort_submission(self) -> None:
        """Restore the dependency, or the empty reservation, after scheduler rejection."""
        state = self.read_state()
        if state is None or state.phase != "submitting":
            raise RuntimeError(f"refusing to abort non-submitting state at {self.state_path}")
        if state.dependency_job_id is not None:
            self._write_state("dependency", state.artifact, job_id=state.dependency_job_id)
        else:
            os.remove(self.state_path)
            fsync_directory(self.directory)

    def record_launched(self, job_id: str, artifact: str) -> None:
        """Make scheduler acceptance of the training job durable."""
        state = self.read_state()
        if state is None or state.phase != "submitting" or state.artifact != os.path.abspath(artifact):
            raise RuntimeError(f"refusing to record training from non-matching state at {self.state_path}")
        self._write_state("launched", artifact, job_id=job_id)

    def clear_dependency(self, job_id: str) -> None:
        """Forget only the matching terminal dependency, leaving the run reservation intact."""
        state = self.read_state()
        if state is None or state.phase != "dependency" or state.job_id != str(job_id):
            raise RuntimeError(f"refusing to clear non-matching dependency {job_id!r} at {self.state_path}")
        os.remove(self.state_path)
        fsync_directory(self.directory)

    def _write_state(
        self,
        phase: str,
        artifact: str,
        *,
        job_id: str | None = None,
        dependency_job_id: str | None = None,
    ) -> None:
        current = _read(self.marker)
        if current != str(self.start):
            raise RuntimeError(
                f"refusing to record launch state: {self.marker} records {current}, expected {self.start}"
            )
        if job_id is not None and not str(job_id).isdigit():
            raise ValueError(f"scheduler job id must be numeric, got {job_id!r}")
        if dependency_job_id is not None and not str(dependency_job_id).isdigit():
            raise ValueError(f"dependency job id must be numeric, got {dependency_job_id!r}")
        payload = {
            "protocol": LAUNCH_STATE_PROTOCOL,
            "start": self.start,
            "phase": phase,
            "artifact": os.path.abspath(artifact),
            "job_id": str(job_id) if job_id is not None else None,
            "dependency_job_id": str(dependency_job_id) if dependency_job_id is not None else None,
        }
        atomic_write_bytes(self.state_path, (json.dumps(payload, sort_keys=True) + "\n").encode())


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return "unreadable"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reserve and reconcile one posttrain run launch identity.")
    parser.add_argument(
        "action",
        choices=(
            "reserve",
            "release",
            "begin",
            "record-dependency",
            "prepare-submission",
            "abort-submission",
            "record-launched",
            "clear-dependency",
            "lock-path",
        ),
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--artifact")
    parser.add_argument("--job-id")
    args = parser.parse_args(argv)
    reservation = RunReservation(args.runs_root, args.run, args.start)
    try:
        if args.action == "lock-path":
            print(reservation.lock_path)
        elif args.action == "reserve":
            reservation.reserve()
        elif args.action == "release":
            reservation.release()
        elif args.action == "begin":
            if not args.artifact:
                parser.error("begin requires --artifact")
            state = reservation.begin(args.artifact)
            print("NEW" if state is None else state.job_id)
        elif args.action == "prepare-submission":
            if not args.artifact:
                parser.error("prepare-submission requires --artifact")
            reservation.prepare_submission(args.artifact)
        elif args.action == "abort-submission":
            reservation.abort_submission()
        elif args.action in {"record-dependency", "record-launched"}:
            if not args.artifact or not args.job_id:
                parser.error(f"{args.action} requires --artifact and --job-id")
            record = (
                reservation.record_dependency if args.action == "record-dependency" else reservation.record_launched
            )
            record(args.job_id, args.artifact)
        else:
            if not args.job_id:
                parser.error("clear-dependency requires --job-id")
            reservation.clear_dependency(args.job_id)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"run reservation: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
