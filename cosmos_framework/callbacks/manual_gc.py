# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import gc
from typing import Any

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.utils import log


class ManualGarbageCollection(EveryN):
    """
    Disable auto gc and manually trigger garbage collection every N iterations
    It is super useful for large scale training to reduce gpu sync time!
    Can reach 50% speedup.

    It is important to note that this callback only disables gc in main process and have auto gc enabled in subprocesses.

    With warm_up=0, GC is disabled at the first training step, after pre-warmed dataloader
    workers have been created. Positive warm_up values preserve the legacy behavior and
    disable GC after that many periodic callback invocations.
    """

    def __init__(self, *args: Any, warm_up: int = 5, gc_level: int = 1, **kwargs: Any) -> None:
        kwargs["barrier_after_run"] = False
        super().__init__(*args, **kwargs)

        self.counter: int = 0
        self.warm: int = warm_up
        self.gc_level: int = gc_level

    def on_training_step_start(
        self,
        model: Any,
        data: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        del model, data, iteration
        if self.warm == 0 and gc.isenabled():
            gc.collect(self.gc_level)
            gc.disable()
            log.critical("Garbage collection disabled")

    def every_n_impl(
        self,
        trainer: Any,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: Any,
        iteration: int,
    ) -> None:
        del trainer, model, data_batch, output_batch, loss, iteration
        self.counter += 1
        if self.counter < self.warm:
            return
        if self.counter == self.warm:
            gc.disable()
            log.critical("Garbage collection disabled")

        gc.collect(self.gc_level)
