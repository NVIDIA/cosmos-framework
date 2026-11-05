# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Initialize an EMA tree after a strict weight-only base-model load."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

import torch

from cosmos_framework.utils.callback import Callback


class InitializeEmaFromRegular(Callback):
    """Copy regular weights, including fresh action modules, into a new EMA.

    The copy applies only to a fresh Stage 0 warm start. Full-state resume
    restores and preserves the saved EMA.
    """

    def __init__(self, source_checkpoint: str, enabled: bool = True) -> None:
        if not source_checkpoint:
            raise ValueError("EMA initialization requires the declared source checkpoint.")
        self.source_checkpoint: str = source_checkpoint.rstrip("/")
        self.enabled: bool = enabled

    def _matches_source(self, checkpoint_path: str) -> bool:
        roots = {self.source_checkpoint}
        parsed = urlsplit(self.source_checkpoint)
        if parsed.scheme in {"s3", "gs"}:
            # The checkpointer may report its object-key expansion rather than
            # the caller's URI. Only the declared key is accepted.
            roots.add(f"s3://ckpt/{parsed.path.lstrip('/')}")
        return any(
            checkpoint_path == root
            or re.fullmatch(re.escape(root) + r"/+checkpoints/iter_\d{9}", checkpoint_path) is not None
            for root in roots
        )

    def _is_same_job_resume(self, checkpoint_path: str) -> bool:
        trainer = getattr(self, "trainer", None)
        checkpointer = getattr(trainer, "checkpointer", None)
        save_dirname = getattr(checkpointer, "save_dirname", None)
        if not isinstance(save_dirname, str) or not save_dirname:
            return False
        checkpoint_parent = checkpoint_path.rstrip("/").rsplit("/", 1)[0]
        return checkpoint_parent == save_dirname.rstrip("/")

    @torch.no_grad()
    def on_load_checkpoint_end(
        self,
        model: Any,
        iteration: int = 0,
        checkpoint_path: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        normalized_path = None if checkpoint_path is None else checkpoint_path.rstrip("/")
        if iteration != 0 or (normalized_path is not None and self._is_same_job_resume(normalized_path)):
            # A full-state resume has already restored net_ema. Reinitializing it
            # from regular weights would make restart behavior checkpoint-dependent.
            return
        if normalized_path is None or not self._matches_source(normalized_path):
            raise RuntimeError("Fresh Stage 0 did not load the declared base checkpoint.")
        if not model.config.ema.enabled:
            raise RuntimeError("Fresh Bimanual Stage 0 must keep EMA enabled.")
        model.net_ema_worker.copy_to(src_model=model.net, tgt_model=model.net_ema)
        model.sync_ema_buffers()
