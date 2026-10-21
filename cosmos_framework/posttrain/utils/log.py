# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Package-wide logger for posttrain (replaces ad-hoc `print`).

Same shape as cosmos-rl's `utils/logging.py`: ONE module-level `logger`, configured at
import, imported directly — no factory ceremony:

    from cosmos_framework.posttrain.utils.log import logger
    logger.info("...")        # operator-facing events (mode banner, resume, saves)
    logger.debug("...")       # diagnostics (per-role builds, worker-side metric lines)

Additions over cosmos-rl's: a RANK tag injected at emit time (Ray workers and torchrun
ranks set RANK after this module is imported), so interleaved multi-process output stays
attributable. stdlib-only on purpose: the multinode launcher imports this on login nodes
that have no torch.

Level: POSTTRAIN_LOG_LEVEL env (default INFO). Format:
    [posttrain|I|r0] 2026-07-03 10:00:00 message
"""

from __future__ import annotations

import logging
import os


class _RankFilter(logging.Filter):
    """Inject the CURRENT rank at emit time (not import time)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = os.environ.get("RANK", "0")
        return True


logger = logging.getLogger("posttrain")

if not logger.handlers:
    _level = getattr(logging, os.environ.get("POSTTRAIN_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(_level)
    _handler = logging.StreamHandler()
    _handler.setLevel(_level)
    _handler.addFilter(_RankFilter())
    _handler.setFormatter(
        logging.Formatter("[posttrain|%(levelname).1s|r%(rank)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.propagate = False


def info0(msg: str) -> None:
    """INFO on rank 0, DEBUG on other ranks — for events with COLLECTIVE semantics
    (save, resume, DONE): under torchrun every rank runs the same code, so a plain
    `logger.info` prints the same line world_size times. rank 0's line IS the global
    truth for barriered operations; the other ranks stay visible when debugging via
    POSTTRAIN_LOG_LEVEL=DEBUG (per-rank lines matter exactly when a collective hangs)."""
    if os.environ.get("RANK", "0") == "0":
        logger.info(msg)
    else:
        logger.debug(msg)
