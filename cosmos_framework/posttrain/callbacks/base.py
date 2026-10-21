# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Callback base (DESIGN §2.6) — cross-cutting observability hooks.

Callbacks hang on the control loop's event points and declare a *kind*: **observer**
(async-safe, off-hot-path, must not block / mutate training) or **gate** (sync,
on-path, rare). v0 implements the observer slice for the single-process loop: the
`DiffusionTrainer` invokes the registered callbacks synchronously at the event points
it owns (`on_init`, `on_train_step`, `on_checkpoint`). This is the §2.6 subscription
seam; the full async control-plane event bus (§11.2, designed-not-built) can later
drive these same hooks without changing callback code.

Distinct from the typed plugin points (reward / objective / dispatch / filter) — those
are domain extension seams; callbacks are observability/lifecycle hooks. Keep
observers cheap and side-effect-free w.r.t. the model (they may read, never mutate).
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.posttrain.utils.log import logger


class Callback:
    """No-op observer base — override only the event points you need."""

    def on_init(self, trainer: Any) -> None: ...

    def on_train_step(self, trainer: Any, metrics: dict) -> None: ...

    def on_checkpoint(self, trainer: Any, path: str) -> None: ...


class CallbackList:
    """Invokes a list of callbacks at each event point. Observer semantics: a failing
    callback must not crash training, so exceptions are swallowed with a printed warning
    (observability is best-effort; it never gates the loop)."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = list(callbacks or [])

    def _fan(self, event: str, *args: Any) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, event)(*args)
            except Exception as e:  # observers must never break the train loop (§2.6)
                logger.warning(f"callback {type(cb).__name__}.{event} raised (ignored): {e}")

    def on_init(self, trainer: Any) -> None:
        self._fan("on_init", trainer)

    def on_train_step(self, trainer: Any, metrics: dict) -> None:
        self._fan("on_train_step", trainer, metrics)

    def on_checkpoint(self, trainer: Any, path: str) -> None:
        self._fan("on_checkpoint", trainer, path)
