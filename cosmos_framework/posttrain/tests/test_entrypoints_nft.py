# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Online NFT control-loop tests: the termination-triggered emergency save follows `rl.emergency_checkpoint_components`.

Review of c2af017642d (thread 0069d0e0): the loop called `ensure_resumable` at a termination trigger, so the shared
`RunConfig` selector was silently ignored by NFT. It now goes through `CheckpointCoordinator.save_emergency` like the
GRPO and DPO loops; the step body (`online_nft_step`) is replaced by a fake that applies one optimizer step.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS
from cosmos_framework.posttrain.tests.helpers.entry_loop import loop_config as _config
from cosmos_framework.posttrain.tests.helpers.entry_loop import mock_loop_env as _mock_env

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _nft_preemption_plan_saves(
    monkeypatch: pytest.MonkeyPatch, **rl_overrides: object
) -> list[tuple[str, tuple[str, ...] | None]]:
    """Run the NFT loop to a termination trigger at the step-2 boundary (save_every=0) under a model-only cadence
    plan; return every policy save as (iter dir name, `components` override or None). None = the trainer's
    configured plan (the cadence call shape); a tuple = an explicit override."""
    from cosmos_framework.posttrain.entrypoints import nft

    _mock_env(monkeypatch, stops=[None, None, "signal"])  # fires after step 2
    n = {"train": 0}
    saves: list[tuple[str, tuple[str, ...] | None]] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
            del data_state
            saves.append((Path(path).name, kwargs.get("components")))

    def _step(units: dict[str, Any], rl: Any, *, step: int) -> dict[str, float]:
        del rl, step
        n["train"] += 1
        units["policy"]._iteration += 1
        return {"train/loss_avg": 0.5}

    monkeypatch.setattr(nft, "online_nft_step", _step)
    units = {"policy": _Policy(), "data": SimpleNamespace(state_dict=lambda: None)}
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):  # the model-only cadence plan
        summary = nft.loop(
            units,
            _config(max_steps=5, save_every=0, checkpoint_components=["model"], **rl_overrides),
            "/tmp/out",
        )
    assert n["train"] == 3 and summary["preempted"] == "signal"  # broke at step 2 of 5
    return saves


def test_nft_loop_preemption_emergency_save_honours_the_run_plan_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """`rl.emergency_checkpoint_components="run"` with a model-only cadence plan: the termination trigger writes exactly
    ONE save through the cadence path (no `components` override), not the forced full plan."""
    saves = _nft_preemption_plan_saves(monkeypatch, emergency_checkpoint_components="run")
    assert saves == [("iter_000003", None)], f"emergency save wrote {saves}; expected the run's own (model-only) plan"


def test_nft_loop_preemption_emergency_save_defaults_to_the_full_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without opting in, a model-only cadence plan still gets the forced full plan at a termination trigger."""
    saves = _nft_preemption_plan_saves(monkeypatch)
    assert saves == [("iter_000003", FULL_CHECKPOINT_COMPONENTS)], f"emergency save wrote {saves}"
