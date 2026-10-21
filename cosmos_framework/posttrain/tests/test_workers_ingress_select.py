# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Control-kwargs plumbing through `WorkerGroup`'s LOCAL call path (spec 2026-08-12 §3.4).

`WorkerGroup._submit`'s `self._backend is None` branch (`workers/worker_group.py:309`) invokes
the body directly — it never reaches `GridBackend._actor_args`/`_align_one`
(`workers/grid.py:413`), so a GridBackend-only implementation of the ingress selection would
raise `TypeError: train() got an unexpected keyword argument 'mini'` on the path every CPU
example and every L0 test actually runs. These tests drive that local path directly.

The registry entry (`roles/core/dispatch.py:60`) is keyed `("trainer", "train", "batch")` —
DPO's `policy` WG shares this SAME entry (`entrypoints/dpo.py: ROLE_SPECS = [..., ("policy",
"trainer")]`) but never passes the control kwargs, so selection must stay OPT-IN on their
presence (`test_a_call_without_control_kwargs_is_untouched` below is the DPO regression guard).

Note: the driver always passes all four of `select_chunk`'s required keyword-only axis
parameters (`mini`, `micro`, `n_mini`, `n_micro` — see `workers/dispatch.py:select_chunk`,
Task 6); the registry's `select_params` (`roles/core/dispatch.py:60`) declares exactly those
four, and this file's calls always supply all four together (never a subset) precisely because
`select_chunk` has no defaults for any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cosmos_framework.posttrain.workers.worker_group import WorkerGroup


@dataclass
class _Item:
    group_id: str
    id: str


def _items(n):
    """n fake trajectories with the two fields `select_chunk`'s canonical sort key reads."""
    return [_Item(group_id=f"g{i // 4:04d}", id=f"t{i:03d}") for i in range(n)]


class _RecordingTrainer:
    """A trainer body with today's exact `Trainer.train` signature — no `**kw` — that records
    whatever batch it was invoked with, so a test can assert what actually reached the body."""

    def __init__(self):
        self.last_batch = None

    def train(self, batch):
        self.last_batch = batch
        return {}


def _worker_group_over(body, kind: str) -> WorkerGroup:
    """A local (unbound) WorkerGroup wrapping `body`, whose role is `kind` — the `(kind, method,
    param)` registry key `_submit`'s local path looks up (`workers/worker_group.py:_local_select`)."""
    return WorkerGroup("w", kind, body=body)


@pytest.mark.L0
@pytest.mark.CPU
def test_control_kwargs_never_reach_the_body():
    """Dispatch CONSUMES mini/micro/n_mini/n_micro, so Trainer.train keeps its signature."""
    seen = {}

    class Body:
        def train(self, batch):  # exactly today's signature — no **kw
            seen["batch"], seen["kw"] = batch, None
            return {}

    wg = _worker_group_over(Body(), kind="trainer")
    wg.train(_items(64), mini=0, micro=1, n_mini=1, n_micro=4)
    assert len(seen["batch"]) == 16  # 64 / (1 mini * 4 micro * 1 dp)


@pytest.mark.L0
@pytest.mark.CPU
def test_local_backend_selects_too():
    """The local path bypasses GridBackend._actor_args entirely (worker_group.py:309), so a
    GridBackend-only implementation raises TypeError on the path every CPU example runs."""
    body = _RecordingTrainer()
    wg = WorkerGroup("w", "trainer", body=body)  # local sync-eager: _backend is None
    wg.train(_items(32), mini=1, micro=0, n_mini=2, n_micro=2)
    assert len(body.last_batch) == 8  # 32 / (2 mini * 2 micro * 1 dp)


@pytest.mark.L0
@pytest.mark.CPU
def test_a_call_without_control_kwargs_is_untouched():
    """DPO shares the ('trainer','train','batch') registry entry and never passes select
    control kwargs. An unconditional predicate would train it on 1/dp_world of its data."""
    body = _RecordingTrainer()
    wg = WorkerGroup("w", "trainer", body=body)
    wg.train(_items(32))
    assert len(body.last_batch) == 32  # full batch, unsliced
