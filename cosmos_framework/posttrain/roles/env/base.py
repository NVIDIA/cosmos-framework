# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Env / Source abstraction (DESIGN §1.3).

The dataflow source is a pluggable Env; rollout is always downstream. A unified
Gym-style interface covers everything:
    reset() -> obs
    step(action) -> (obs, reward, done, info)

- simulator        = full multi-step Env  (step-wise reset/step: future). The
                     sim-eval `robolab.SimBody` is the first sim Env variant, but
                     batch/outcome-only (`run_tasks`, no per-step Trajectory) — see
                     `docs/eval/sim-eval.md` §8.
- world model      = a *learned* Env       (TODO: future)
- single-turn text/VLM RL = a degenerate **1-step Env** (reset→prompt, step→reward+done)

Concrete sources (datasets / sims / world-models) live in sibling modules and are
declared via `roles.env(...)`.

**A SIMULATOR DOES NOT IMPLEMENT THIS -- see `sim_role.py::SimRole`.** The `Env` below is the
1-step dataset contract, which `DatasetEnv` and `ArithmeticEnv` really do satisfy. A simulator
holds `n_lanes` concurrent episodes, is stepped in a batch by the caller, and owns truncation;
it is `roles/env/sim_role.py` (framework seam) over `embodied/sim.py` (engine seam). The line
above about `robolab.SimBody` predates that split and describes the eval path, not this
Protocol.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from cosmos_framework.posttrain.schema import ObsBundle


class Env(Protocol):
    def reset(self) -> ObsBundle: ...
    def step(self, action: Any) -> tuple[Optional[ObsBundle], Optional[float], bool, dict]: ...
