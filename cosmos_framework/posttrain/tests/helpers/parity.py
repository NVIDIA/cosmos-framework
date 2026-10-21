# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-agnostic parity harness: drive the decoupled trainer/env stack GPU-lessly and verify
that a training run is reproducible and that an interrupted run resumes bit-identically.

It takes zero-arg `build_trainer` / `build_env` callables (the toy CPU stack for L0, the real
8B stack later) so the same assertions run on any model. Three tiers:
- `run_trajectory` + `assert_loss_parity` — the per-step loss matches a pinned reference (a
  signal-preserving bit-match against a golden). The golden lives inline in the test that owns it;
  regenerate it with `tests/parity_cli.py --print-losses`.
- `assert_overfit` — the assembled loop actually learns (reference-free: loss drops below a floor
  on a tiny fixed pair set); catches signal-destroying bugs a bit-match reference can hide.
- `assert_resume_parity` — an interrupted run bit-equals the uninterrupted one through
  save -> rebuild -> restore, isolating the TRAINER checkpoint (net/opt/iter/RNG) round-trip by
  driving a cursor-free data source.
- `assert_combined_resume_parity` — the same continuation proof with a cursor-bearing data
  source, validating that one production checkpoint restores data and trainer state together.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch

from cosmos_framework.posttrain.checkpoint import load_component
from cosmos_framework.posttrain.control.checkpointing import save_policy_checkpoint
from cosmos_framework.posttrain.entrypoints.dpo import resolve, train_step
from cosmos_framework.posttrain.utils.rank_state import state_for_current_rank

TrainerBuilder = Callable[[], Any]  # () -> a trainer exposing train/save/load/iteration
EnvBuilder = Callable[[], Any]  # () -> an env exposing get_samples(n) -> list[Trajectory]
SampleKey = Callable[[Any], Any]  # Trajectory -> stable, exactly comparable sample identity


class _ObservedEnv:
    """Record emitted samples while preserving the wrapped env's checkpoint surface."""

    def __init__(self, inner: Any, sample_key: SampleKey) -> None:
        self._inner = inner
        self._sample_key = sample_key
        self.sample_keys: list[Any] = []

    def get_samples(self, n: int) -> list[Any]:
        samples = self._inner.get_samples(n)
        self.sample_keys.extend(self._sample_key(sample) for sample in samples)
        return samples

    def state_dict(self) -> dict[str, Any] | None:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        return bool(self._inner.load_state_dict(state))


def _assert_exact_state(actual: Any, expected: Any, *, path: str) -> None:
    """Recursively compare checkpoint state without numerical tolerance."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
            raise AssertionError(f"combined resume parity: checkpoint state differs at {path}")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
            raise AssertionError(f"combined resume parity: checkpoint state differs at {path}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"combined resume parity: checkpoint mapping differs at {path}")
        for key in expected:
            _assert_exact_state(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"combined resume parity: checkpoint sequence differs at {path}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_exact_state(actual_item, expected_item, path=f"{path}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"combined resume parity: checkpoint value differs at {path}")


def _seed_all(seed: int) -> None:
    # Mirror the checkpoint RNG contract (python / numpy / torch / cuda) this harness validates, so
    # capture and replay can't diverge through a global RNG the seeding missed (torch covers cuda).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_from(trainer: Any, base_ckpt: str | None) -> None:
    """Anchor net AND the frozen ref on the base weights (copy_mode) — the frozen ref is never
    saved in a checkpoint, so this is the only place it gets its anchor and it MUST run on every
    start, including resume (skipping it on resume was a silent objective-corruption bug)."""
    if base_ckpt is not None:
        resolve(trainer.load(base_ckpt, copy_mode=True))


def _run_steps(trainer: Any, env: Any, steps: int, *, pairs_per_step: int) -> list[float]:
    """Drive `steps` optimizer steps of the SHIPPING inner step (`entrypoints.dpo.train_step`), so the
    harness validates the real loop rather than a copy. The per-step micro-batch count is the TRAINER's
    own `grad_accum` — one knob, so the harness and the trainer can't drift. Returns the per-step
    canonical DPO training-loss list."""
    grad_accum = getattr(trainer, "grad_accum", 1)
    history: list[float] = []
    for _ in range(steps):
        metrics = train_step(env, trainer, pairs_per_step=pairs_per_step, grad_accum=grad_accum)
        history.append(float(metrics["train/loss_avg"]))
    return history


def run_trajectory(
    build_trainer: TrainerBuilder,
    build_env: EnvBuilder,
    steps: int,
    *,
    pairs_per_step: int = 1,
    seed: int = 0,
    base_ckpt: str | None = None,
) -> list[float]:
    """Build a fresh trainer+env and run `steps` optimizer steps of the finite loop path; return
    the per-step canonical DPO loss list. `seed` seeds the sample+noise streams for reproducibility."""
    _seed_all(seed)
    trainer = build_trainer()
    _init_from(trainer, base_ckpt)
    env = build_env()
    return _run_steps(trainer, env, steps, pairs_per_step=pairs_per_step)


def assert_loss_parity(
    losses: Sequence[float],
    reference: Sequence[float],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """The per-step losses match the pinned reference within tolerance (default near-bit-exact)."""
    if len(losses) != len(reference):
        raise AssertionError(f"loss parity: {len(losses)} steps vs {len(reference)} in reference")
    for i, (a, b) in enumerate(zip(losses, reference)):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"loss parity: step {i} diverged: {a!r} vs reference {b!r} (rtol={rtol} atol={atol})")


def parse_printed_losses(output: str, expected_len: int) -> list[float]:
    """Pull the paste-ready loss list the launcher prints on rank 0: the last line that parses as a
    numeric list of `expected_len` (ignoring the '#'-prefixed header and any torchrun log lines). Each
    element is parsed with `float()` (not `ast.literal_eval`) so a diverged run's bare `nan`/`inf` tokens
    still parse — a NaN run then fails as a loss divergence, not as a confusing "no loss list found"."""
    for line in reversed(output.splitlines()):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                vals = [float(t) for t in s[1:-1].split(",") if t.strip()]
            except ValueError:
                continue
            if len(vals) == expected_len:
                return vals
    raise AssertionError(f"no {expected_len}-length loss list in parity output:\n{output[-2000:]}")


def assert_overfit(
    build_trainer: TrainerBuilder,
    build_env: EnvBuilder,
    *,
    steps: int,
    floor: float,
    pairs_per_step: int = 1,
    seed: int = 0,
) -> list[float]:
    """Reference-free "does the assembled loop learn" check: train on a tiny fixed pair set and
    assert the final loss drops below `floor` (well under the ln 2 ~= 0.693 chance init). Catches
    signal-destroying bugs (a dropped prefactor, a bf16 cast) that a bit-match reference cannot."""
    losses = run_trajectory(build_trainer, build_env, steps, pairs_per_step=pairs_per_step, seed=seed)
    if not losses[-1] < floor:
        raise AssertionError(
            f"overfit canary: final loss {losses[-1]:.4f} did not drop below {floor} over {steps} steps "
            f"(init {losses[0]:.4f}) — the assembled loop is not learning"
        )
    return losses


def assert_resume_parity(
    build_trainer: TrainerBuilder,
    build_env: EnvBuilder,
    *,
    total_steps: int,
    break_at: int,
    tmp_dir: str,
    base_ckpt: str,
    pairs_per_step: int = 1,
    seed: int = 0,
    skip_init_from: bool = False,
) -> tuple[list[float], list[float]]:
    """Trainer-side resume parity: an interrupted run bit-equals the uninterrupted one through
    save -> rebuild(fresh init) -> init_from(base, copy_mode) -> resume -> continue.

    `build_env` MUST be cursor-free (same stream regardless of position, e.g. `FixedPairEnv`), so
    the data is identical across the interruption without a persisted cursor — this isolates the
    trainer checkpoint (net/opt/iter/RNG) round-trip.

    `skip_init_from=True` drives the NEGATIVE path: rebuilding WITHOUT the copy_mode anchor leaves
    the fresh net_ref off the base weights (a frozen ref is never checkpointed), so the tail must
    diverge — the check then raises, proving the restore order is load-bearing."""
    if not 0 < break_at < total_steps:
        raise ValueError(f"need 0 < break_at < total_steps, got break_at={break_at} total_steps={total_steps}")

    # (i) uninterrupted reference.
    ref = run_trajectory(
        build_trainer,
        build_env,
        total_steps,
        pairs_per_step=pairs_per_step,
        seed=seed,
        base_ckpt=base_ckpt,
    )

    # (ii) interrupted: run break_at steps on an identically-seeded run, then checkpoint.
    _seed_all(seed)
    trainer = build_trainer()
    _init_from(trainer, base_ckpt)
    env = build_env()
    _run_steps(trainer, env, break_at, pairs_per_step=pairs_per_step)
    iter_dir = os.path.join(tmp_dir, f"iter_{int(resolve(trainer.iteration())):06d}")
    resolve(trainer.save(iter_dir))

    # (iii) rebuild a FRESH trainer + restore: copy_mode init_from FIRST (invariant #1), then resume.
    resumed = build_trainer()
    if not skip_init_from:
        _init_from(resumed, base_ckpt)
    resolve(resumed.load(iter_dir))

    # (iv) continue on the RESTORED RNG (no re-seed) with a fresh cursor-free env (identical stream).
    env2 = build_env()
    tail = _run_steps(resumed, env2, total_steps - break_at, pairs_per_step=pairs_per_step)

    expected = ref[break_at:]
    for i, (a, b) in enumerate(zip(tail, expected)):
        if a != b:
            raise AssertionError(
                f"resume parity: post-break step {break_at + i} diverged: {a!r} vs uninterrupted {b!r} "
                f"(trainer checkpoint did not restore net/opt/iter/RNG bit-identically)"
            )
    return ref, tail


def assert_combined_resume_parity(
    build_trainer: TrainerBuilder,
    build_env: EnvBuilder,
    *,
    total_steps: int,
    break_at: int,
    tmp_dir: str,
    base_ckpt: str,
    sample_key: SampleKey,
    pairs_per_step: int = 1,
    seed: int = 0,
    skip_data_restore: bool = False,
) -> tuple[list[float], list[float]]:
    """Combined continuation parity for one production-shaped DPO checkpoint.

    The interrupted leg saves the cursor through `save_policy_checkpoint`, rebuilds
    both objects, restores the rank-local data component before the trainer component
    (the entrypoint's setup order), and continues without reseeding. The emitted
    samples, losses, iteration, and final model/optimizer/trainer/data components must
    exactly match an uninterrupted run.

    `skip_data_restore=True` is a negative control: it restores the trainer while
    restarting the data source, so the parity assertion must fail.
    """
    if not 0 < break_at < total_steps:
        raise ValueError(f"need 0 < break_at < total_steps, got break_at={break_at} total_steps={total_steps}")

    # (i) Uninterrupted reference, including its final production-shaped checkpoint.
    _seed_all(seed)
    reference_trainer = build_trainer()
    _init_from(reference_trainer, base_ckpt)
    reference_env = _ObservedEnv(build_env(), sample_key)
    reference_losses = _run_steps(
        reference_trainer,
        reference_env,
        total_steps,
        pairs_per_step=pairs_per_step,
    )
    reference_checkpoint = os.path.join(tmp_dir, "uninterrupted", "checkpoints", f"iter_{total_steps:06d}")
    save_policy_checkpoint(reference_trainer, reference_checkpoint, data=reference_env)

    # (ii) Identically seeded prefix committed through the same checkpoint boundary as the entrypoint.
    _seed_all(seed)
    interrupted_trainer = build_trainer()
    _init_from(interrupted_trainer, base_ckpt)
    interrupted_env = _ObservedEnv(build_env(), sample_key)
    prefix_losses = _run_steps(
        interrupted_trainer,
        interrupted_env,
        break_at,
        pairs_per_step=pairs_per_step,
    )
    interrupted_checkpoint = os.path.join(tmp_dir, "interrupted", "checkpoints", f"iter_{break_at:06d}")
    save_policy_checkpoint(interrupted_trainer, interrupted_checkpoint, data=interrupted_env)

    # (iii) Fresh data + trainer objects, restored in production setup order: data first, policy second.
    resumed_env = _ObservedEnv(build_env(), sample_key)
    data_component = load_component(interrupted_checkpoint, "data")
    data_state = state_for_current_rank(data_component)
    if data_state is None:
        raise AssertionError("combined resume parity: checkpoint has no compatible data cursor")
    if not skip_data_restore and not resumed_env.load_state_dict(data_state):
        raise AssertionError("combined resume parity: data cursor restore was not applied")

    resumed_trainer = build_trainer()
    _init_from(resumed_trainer, base_ckpt)
    resolve(resumed_trainer.load(interrupted_checkpoint))
    tail_losses = _run_steps(
        resumed_trainer,
        resumed_env,
        total_steps - break_at,
        pairs_per_step=pairs_per_step,
    )

    combined_samples = [*interrupted_env.sample_keys, *resumed_env.sample_keys]
    if combined_samples != reference_env.sample_keys:
        raise AssertionError("combined resume parity: emitted sample sequence differs after restore")
    if prefix_losses != reference_losses[:break_at]:
        raise AssertionError("combined resume parity: pre-checkpoint losses differ from uninterrupted run")
    if tail_losses != reference_losses[break_at:]:
        raise AssertionError("combined resume parity: post-checkpoint losses differ from uninterrupted run")
    if int(resolve(reference_trainer.iteration())) != total_steps:
        raise AssertionError("combined resume parity: uninterrupted trainer iteration is incorrect")
    if int(resolve(resumed_trainer.iteration())) != total_steps:
        raise AssertionError("combined resume parity: resumed trainer iteration is incorrect")

    # (iv) The final continuation state itself is identical, not only its observed losses.
    resumed_checkpoint = os.path.join(tmp_dir, "resumed", "checkpoints", f"iter_{total_steps:06d}")
    save_policy_checkpoint(resumed_trainer, resumed_checkpoint, data=resumed_env)
    for component_name in ("model", "optim", "trainer", "data"):
        expected_component = load_component(reference_checkpoint, component_name)
        actual_component = load_component(resumed_checkpoint, component_name)
        if expected_component is None or actual_component is None:
            raise AssertionError(f"combined resume parity: missing final {component_name} component")
        _assert_exact_state(
            actual_component,
            expected_component,
            path=component_name,
        )
    return reference_losses, tail_losses
