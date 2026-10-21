# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic preference Env — a TEST FIXTURE (DESIGN §1.3).

Not a production source. Fabricates separable preference pairs so the offline DPO loop
runs on CPU with no data / dataset wiring — the counterpart to the toy model fixture
(`tests/helpers/toy_model.py`). It emits the SAME production-shaped `Trajectory` as the real
`DPOPreferenceEnv` by reusing that module's `_pair_trajectory` builder, so a test pair is
interchangeable with a real one through the trainer/objective. Lives under `tests/` so no
test-only code sits next to the production Envs (`roles/env/preference.py`).

Each pair shares a 1-D latent target; the winner sits closer to it than the loser, so the
DPO margin has signal and the optimizer has something to learn (mirrors the variance reward
in the GRPO e2e test). Samples are `{"x": Tensor[D]}`, consumable by `ToyFlowModel`.
"""

from __future__ import annotations

import random
from typing import Any

from cosmos_framework.posttrain.roles.env.preference import _pair_trajectory
from cosmos_framework.posttrain.schema import Trajectory


class SyntheticPreferenceEnv:
    """Test-only synthetic preference source (see module docstring)."""

    def __init__(self, dim: int = 8, seed: int = 0):
        import torch

        self._torch = torch
        self._rng = random.Random(seed)
        self._gen = torch.Generator().manual_seed(seed)
        self._dim = dim
        self._n = 0

    def reset(self) -> Trajectory:
        torch = self._torch
        base = torch.randn(self._dim, generator=self._gen)
        winner = base + 0.1 * torch.randn(self._dim, generator=self._gen)  # near the target
        loser = base + 1.0 * torch.randn(self._dim, generator=self._gen)  # farther
        traj = _pair_trajectory(self._n, f"prompt-{self._n}", {"x": winner}, {"x": loser})
        self._n += 1
        return traj

    def step(self, action: Any):
        return None, None, True, {}

    def get_samples(self, n: int) -> list[Trajectory]:
        return [self.reset() for _ in range(n)]


class FixedPairEnv:
    """Deterministic, cursor-free preference source: it yields the SAME fixed pair set on every
    `get_samples` call, so the data stream does not depend on any position/cursor. Used by the
    parity harness's overfit canary (a tiny fixed set to memorize) and its trainer-side resume
    check, where the stream must be identical across an interruption WITHOUT a persisted cursor —
    isolating the trainer checkpoint (net/opt/iter/RNG) round-trip from the env cursor."""

    def __init__(self, dim: int = 8, seed: int = 0, n_pairs: int = 1):
        import torch

        gen = torch.Generator().manual_seed(seed)
        base = torch.randn(dim, generator=gen)
        self._pairs: list[tuple[dict, dict]] = []
        for _ in range(n_pairs):
            winner = base + 0.1 * torch.randn(dim, generator=gen)  # near the shared target
            loser = base + 1.0 * torch.randn(dim, generator=gen)  # farther
            self._pairs.append(({"x": winner}, {"x": loser}))

    def get_samples(self, n: int) -> list[Trajectory]:
        out: list[Trajectory] = []
        for i in range(n):
            winner, loser = self._pairs[i % len(self._pairs)]
            # Clone per trajectory so a consumer never shares storage across calls/trajectories — the
            # resume proof depends on a stream that is identical AND independent between runs.
            winner = {k: v.clone() for k, v in winner.items()}
            loser = {k: v.clone() for k, v in loser.items()}
            out.append(_pair_trajectory(i, f"fixed-{i}", winner, loser))
        return out
