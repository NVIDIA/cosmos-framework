# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU tests for the parity harness: loss reproducibility + pinned-reference parity, the
overfit canary, trainer-side and combined data-plus-trainer continuation parity, the
restore-order negatives, the calls-vs-pairs loop semantics, and the launcher (toy checks +
reference capture)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.roles.env.preference import DatasetPreferenceEnv
from cosmos_framework.posttrain.schema import Preference, Trajectory
from cosmos_framework.posttrain.tests.helpers.parity import (
    assert_combined_resume_parity,
    assert_loss_parity,
    assert_overfit,
    assert_resume_parity,
    parse_printed_losses,
    run_trajectory,
)
from cosmos_framework.posttrain.tests.parity_cli import (
    _BASE_SEED,
    _toy_build_env,
    _toy_build_fixed_env,
    _toy_build_trainer,
)
from cosmos_framework.posttrain.tests.parity_cli import main as parity_main

# Pinned golden for the toy DPO stack (8 steps, seed 0). Regenerate after any toy-stack change with:
#   python -m cosmos_framework.posttrain.tests.parity_cli --model toy --steps 8 --print-losses
_TOY_REFERENCE = [
    0.6931471824645996,
    0.7068293690681458,
    0.824951171875,
    1.0287986993789673,
    0.5098291635513306,
    0.856062650680542,
    0.710994303226471,
    0.7755860090255737,
]


class _CursorPairDataset:
    """Map-style toy pairs with load-bearing item order and dataset-local RNG state."""

    def __init__(self, size: int = 7, dim: int = 8, seed: int = 17) -> None:
        self._size = size
        self._dim = dim
        self._seed = seed
        base_generator = torch.Generator().manual_seed(seed)
        self._bases = [torch.randn(dim, generator=base_generator) for _ in range(size)]  # size × [D]
        self._sample_generator = torch.Generator().manual_seed(seed + 1)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[str, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        base = self._bases[index]  # [D]
        winner = base + 0.1 * torch.randn(self._dim, generator=self._sample_generator)  # [D]
        loser = base + torch.randn(self._dim, generator=self._sample_generator)  # [D]
        return f"cursor-{index}", {"x": winner}, {"x": loser}

    def cursor_fingerprint(self) -> str:
        return f"cursor-pair-v1:{self._size}:{self._dim}:{self._seed}"

    def cursor_state_dict(self) -> dict[str, Any]:
        return {"sample_generator": self._sample_generator.get_state()}

    def load_cursor_state_dict(self, state: dict[str, Any]) -> None:
        generator_state = state.get("sample_generator")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("cursor-pair sample generator state must be a tensor")
        self._sample_generator.set_state(generator_state)


def _build_cursor_env() -> DatasetPreferenceEnv:
    return DatasetPreferenceEnv(
        _CursorPairDataset(),
        shuffle=True,
        seed=5,
        cursor_resume="strict",
    )


def _trajectory_key(trajectory: Trajectory) -> tuple[str, str, tuple[float, ...], tuple[float, ...]]:
    transition = trajectory.transitions[0]
    if not isinstance(transition.targets, Preference):
        raise TypeError("combined continuation test requires Preference targets")
    prompt = str(transition.obs.representations["prompt"])
    winner = tuple(float(value) for value in transition.targets.winner["x"].tolist())
    loser = tuple(float(value) for value in transition.targets.loser["x"].tolist())
    return trajectory.id, prompt, winner, loser


class _SpyEnv:
    """Wraps an env, recording each get_samples call's requested count."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[int] = []

    def get_samples(self, n: int):
        self.calls.append(n)
        return self.inner.get_samples(n)


@pytest.mark.L0
@pytest.mark.CPU
def test_loss_trajectory_is_reproducible():
    a = run_trajectory(_toy_build_trainer, _toy_build_env, 8, seed=0)
    b = run_trajectory(_toy_build_trainer, _toy_build_env, 8, seed=0)
    assert a == b  # identical seeds + fresh builds => bit-identical


@pytest.mark.L0
@pytest.mark.CPU
def test_loss_trajectory_matches_reference():
    losses = run_trajectory(_toy_build_trainer, _toy_build_env, 8, seed=0)
    # Cross-platform tolerance: CPU FP drifts ~1e-4 across arch (arm64 CI vs x86/macOS), so this is a
    # coarse regression check; the strong bit-exact guarantee is resume parity (same process).
    assert_loss_parity(losses, _TOY_REFERENCE, rtol=1e-3, atol=1e-5)


@pytest.mark.L0
@pytest.mark.CPU
def test_overfit_canary_learns():
    losses = assert_overfit(_toy_build_trainer, _toy_build_fixed_env, steps=60, floor=0.5, seed=0)
    assert losses[0] > losses[-1]  # it actually descended


@pytest.mark.L0
@pytest.mark.CPU
def test_calls_vs_pairs_semantics():
    # grad_accum is the TRAINER's knob (the harness reads it back): with grad_accum=2 over 3 steps the
    # harness makes 6 train() calls (= 6 get_samples), but the optimizer steps only 3 times (accumulate
    # 2 micro-batches, step once) — the "one optimizer step == grad_accum calls" contract, now tested.
    spies: list[_SpyEnv] = []
    opt_steps = {"n": 0}

    def build_trainer():
        tr = _toy_build_trainer(grad_accum=2)
        real_step = tr.opt.step

        def counting_step(*a, **k):
            opt_steps["n"] += 1
            return real_step(*a, **k)

        tr.opt.step = counting_step
        return tr

    def build_env():
        spy = _SpyEnv(_toy_build_fixed_env())
        spies.append(spy)
        return spy

    run_trajectory(build_trainer, build_env, steps=3, pairs_per_step=2, seed=0)
    spy = spies[-1]
    assert len(spy.calls) == 3 * 2  # train() calls per step == grad_accum, over 3 steps
    assert all(n == 2 for n in spy.calls)  # pairs per call == pairs_per_step
    assert opt_steps["n"] == 3  # exactly 3 OPTIMIZER updates (grad_accum=2 accumulates)


@pytest.mark.L0
@pytest.mark.CPU
def test_resume_parity_trainer_side(tmp_path):
    base = os.path.join(str(tmp_path), "base", "iter_000000")
    _toy_build_trainer(seed=_BASE_SEED).save(base)  # anchor from a DISTINCT init
    _, tail = assert_resume_parity(
        _toy_build_trainer,
        _toy_build_fixed_env,
        total_steps=8,
        break_at=3,
        tmp_dir=str(tmp_path),
        base_ckpt=base,
        seed=0,
    )
    assert len(tail) == 5  # total_steps - break_at, and it bit-equaled the uninterrupted tail


@pytest.mark.L0
@pytest.mark.CPU
def test_resume_negative_skip_init_from_diverges(tmp_path):
    base = os.path.join(str(tmp_path), "base", "iter_000000")
    _toy_build_trainer(seed=_BASE_SEED).save(base)
    with pytest.raises(RuntimeError, match="checkpoint reference identity"):
        assert_resume_parity(
            _toy_build_trainer,
            _toy_build_fixed_env,
            total_steps=8,
            break_at=3,
            tmp_dir=str(tmp_path),
            base_ckpt=base,
            seed=0,
            skip_init_from=True,  # the fresh net_ref lacks the base anchor and must fail closed
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_combined_data_and_trainer_resume_parity(tmp_path: Path) -> None:
    base = os.path.join(str(tmp_path), "base", "iter_000000")
    _toy_build_trainer(seed=_BASE_SEED).save(base)
    _, tail = assert_combined_resume_parity(
        _toy_build_trainer,
        _build_cursor_env,
        total_steps=8,
        break_at=3,
        tmp_dir=str(tmp_path),
        base_ckpt=base,
        sample_key=_trajectory_key,
        seed=0,
    )
    assert len(tail) == 5


@pytest.mark.L0
@pytest.mark.CPU
def test_combined_resume_without_data_cursor_diverges(tmp_path: Path) -> None:
    base = os.path.join(str(tmp_path), "base", "iter_000000")
    _toy_build_trainer(seed=_BASE_SEED).save(base)
    with pytest.raises(AssertionError, match="emitted sample sequence differs"):
        assert_combined_resume_parity(
            _toy_build_trainer,
            _build_cursor_env,
            total_steps=8,
            break_at=3,
            tmp_dir=str(tmp_path),
            base_ckpt=base,
            sample_key=_trajectory_key,
            seed=0,
            skip_data_restore=True,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_cli_toy_runs_and_exits_zero():
    # Thin launcher on the toy tier: default = trainer-side resume parity; --overfit = standalone canary.
    assert parity_main(["--model", "toy", "--steps", "6", "--break-at", "3"]) == 0
    assert parity_main(["--model", "toy", "--overfit", "--overfit-steps", "60", "--overfit-floor", "0.5"]) == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_cli_print_losses(capsys):
    import ast

    assert parity_main(["--model", "toy", "--steps", "5", "--print-losses"]) == 0
    out = capsys.readouterr().out
    printed = ast.literal_eval(next(ln for ln in out.splitlines() if ln.startswith("[")))
    assert isinstance(printed, list) and len(printed) == 5  # a paste-ready 5-step list


@pytest.mark.L0
@pytest.mark.CPU
def test_parse_printed_losses():
    # Ignores the '#' header + a wrong-length decoy list; floats each element so nan/inf lists still
    # parse — a diverged run must fail as a loss divergence, not a confusing "no loss list found".
    import math

    output = "# steps=3 seed=0 torch=x\n[1.0, 2.0]\n[0.1, nan, inf]\n"
    vals = parse_printed_losses(output, 3)
    assert len(vals) == 3 and vals[0] == 0.1 and math.isnan(vals[1]) and math.isinf(vals[2])
    with pytest.raises(AssertionError, match="no 4-length"):
        parse_printed_losses(output, 4)
