# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""DPO observability callbacks (DESIGN §2.6) — real, non-blocking observers.

`DPOMetricLogger` records the per-step DPO metrics and prints a rolling line; it is the
default training-visibility hook. `DPORewardTrendCallback` watches the implicit-reward
signal (margin / accuracy) over a window and flags when the policy is NOT separating
winners from losers (margin mean ≤ 0), the canonical "DPO is not learning / sign flip /
ref drift" symptom. Both are pure observers — they read the metrics dict the trainer
already produces and never touch the model.

NOTE (seam, not stubbed): a *sample/visualization* callback (decode policy/EMA/ref
generations for eyeballing) needs the model's inference/generate path + an artifact
sink; that is a separate integration (P3 T2I/eval) and is intentionally left to wire on
the same `Callback` interface rather than faked here.

TODO(dinghaoy): move these algorithm-specific callbacks behind a registration mechanism
(selectable by name from config, like `control/recipe.RECIPES`) instead of importing/
constructing them directly in `config/build.py`. Deferred to a follow-up MR (Greptile #11).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from cosmos_framework.posttrain.callbacks.base import Callback
from cosmos_framework.posttrain.utils.log import logger


class DPOMetricLogger(Callback):
    """Record every step's DPO metrics; print every `log_every` steps. Keeps the full
    `history` (list of dicts) and the latest values for downstream inspection/eval."""

    KEYS = (
        "train/loss_avg",
        "train/dpo_reward_margin",
        "train/dpo_accuracy",
        "train/dpo_reward_winner",
        "train/dpo_reward_loser",
    )

    def __init__(self, log_every: int = 1):
        self.log_every = max(1, int(log_every))
        self.history: list[dict[str, float]] = []

    def on_train_step(self, trainer: Any, metrics: dict) -> None:
        row = {k: float(metrics[k]) for k in self.KEYS if k in metrics}
        self.history.append(row)
        n = len(self.history)
        if n % self.log_every == 0:
            loss = row.get("train/loss_avg", float("nan"))
            margin = row.get("train/dpo_reward_margin", float("nan"))
            acc = row.get("train/dpo_accuracy", float("nan"))
            logger.debug(f"trainer step {n:5d}  loss {loss:.4f}  margin {margin:+.4f}  acc {acc:.3f}")


class DPORewardTrendCallback(Callback):
    """Track the reward margin + accuracy over a sliding `window` and flag stagnation.
    A healthy DPO run drives `margin` positive and `accuracy` → 1; a window-mean margin
    ≤ `min_margin` means the policy is not separating winners from losers (warn once per
    breach). Exposes `margin_mean()` / `accuracy_mean()` for eval/dashboards."""

    def __init__(self, window: int = 50, min_margin: float = 0.0, warn: bool = True):
        self.window = max(1, int(window))
        self.min_margin = float(min_margin)
        self.warn = warn
        self._margins: deque[float] = deque(maxlen=self.window)
        self._accs: deque[float] = deque(maxlen=self.window)
        self._steps = 0
        self._in_breach = False

    def on_train_step(self, trainer: Any, metrics: dict) -> None:
        if "train/dpo_reward_margin" in metrics:
            self._margins.append(float(metrics["train/dpo_reward_margin"]))
        if "train/dpo_accuracy" in metrics:
            self._accs.append(float(metrics["train/dpo_accuracy"]))
        self._steps += 1
        # Only judge once a full window has accumulated (a few noisy steps at init aren't
        # evidence — at init margin is amplified ref/policy kernel noise, not a signal).
        breached = len(self._margins) == self.window and self.margin_mean() <= self.min_margin
        # Edge-triggered: warn once on the False -> True transition, re-arm when the margin
        # recovers above the threshold. Avoids flooding the log every step while in breach.
        if self.warn and breached and not self._in_breach:
            logger.warning(
                f"margin mean over last {self.window} steps = {self.margin_mean():+.4f} "
                f"≤ {self.min_margin:+.4f} — policy not separating winners from losers "
                "(check beta / ref drift / sign)."
            )
        self._in_breach = breached

    def margin_mean(self) -> float:
        return sum(self._margins) / len(self._margins) if self._margins else float("nan")

    def accuracy_mean(self) -> float:
        return sum(self._accs) / len(self._accs) if self._accs else float("nan")
