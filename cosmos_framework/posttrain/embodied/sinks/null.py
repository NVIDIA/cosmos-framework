# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Outcome-only sink -- the SHAPE evaluation has, though not the sink it builds.

Eval builds `sinks/evaljsonl.py`, which is on an archive branch and **not in this series** -- neither the
sink nor the `policy_eval.py` line that selects it is in this tree, so do not go looking for
them. This module is the minimal outcome-only sink and what the contract tests drive.

This is the whole of the difference between the eval path and the collection path. Eval
keeps the outcome and throws the trajectory away; collection keeps every frame. Making that
a sink choice rather than a code path is the point of the layer.

``already_landed`` always returns ``None``: with nothing persisted there is nothing to
resume from, and claiming otherwise would let an eval skip episodes it never ran.
"""

from __future__ import annotations

from typing import Any, Mapping

from cosmos_framework.posttrain.embodied.registry import register_sink
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, Outcome
from cosmos_framework.posttrain.schema.trajectory import Transition


class NullEpisodeWriter:
    """Counts frames and drops them.

    It counts rather than doing literally nothing so that ``steps`` from the driver and
    ``frames`` from the writer can be cross-checked in an eval run -- if those two ever
    disagree, the frame-alignment convention has been broken somewhere, and an eval is the
    cheapest place to notice.
    """

    __slots__ = ("frames", "_uri")

    def __init__(self, key: EpisodeKey) -> None:
        self.frames = 0
        self._uri = f"null://{key.slug()}"

    def append(self, frame: Transition) -> None:
        del frame  # the contract is that the frame is consumed before return; dropping is consuming
        self.frames += 1

    def commit(self, outcome: Outcome, metrics: Mapping[str, Any]) -> str:
        del outcome, metrics
        return self._uri

    def abort(self, reason: str) -> None:
        del reason
        self.frames = 0


class NullSink:
    """Records nothing. Keeps the last heartbeat so a caller can inspect liveness."""

    #: NO -- it records nothing at all, so there is nothing for extrinsics to be recorded
    #: INTO. A smoke run should not pay 2.4 ms/step to build an array it discards.
    wants_extrinsics = False

    def __init__(self, **kw: Any) -> None:
        del kw  # accepts and ignores sink_options so eval and collect share one config shape
        self.last_heartbeat: Mapping[str, Any] | None = None
        self.episodes = 0

    def open_episode(self, key: EpisodeKey, meta: Mapping[str, Any]) -> NullEpisodeWriter:
        del meta
        self.episodes += 1
        return NullEpisodeWriter(key)

    def already_landed(self, key: EpisodeKey) -> str | None:
        del key
        return None

    def heartbeat(self, snapshot: Mapping[str, Any]) -> None:
        self.last_heartbeat = snapshot

    def close(self) -> None:
        return None


register_sink("null", NullSink)
