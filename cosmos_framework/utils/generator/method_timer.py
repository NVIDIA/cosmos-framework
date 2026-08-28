# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any

import torch


class MethodTimer:
    """Splits one method's time out of a step's wall clock, without perturbing it.

    Monkeypatches ``getattr(model, method_name)`` to bracket each call with a pair of
    ``torch.cuda.Event``. Recording an event is a stream marker, not a synchronize -- it
    costs a small host-side enqueue and nothing on the GPU side, so it does not serialize
    work or change what overlaps with what. ``Event.elapsed_time`` itself blocks the CPU
    until both of its events have completed (the CUDA runtime's own behavior for
    ``cudaEventElapsedTime``, not something this class adds), so ``stop_and_read_ms`` never
    needs a ``torch.cuda.synchronize()`` call of its own beforehand.

    Off the timed path (``reset()`` not called since the last ``stop_and_read_ms()``), calls
    pass straight through with no event recording at all -- e.g. warm-up steps in the
    ``cost_model`` benchmark, or the ``hit_thres`` warm-up window in ``IterSpeed``.

    Generic over which method it brackets, so one class serves every monkeypatched timing
    site: the offline ``cost_model.benchmark`` calibration script times ``model.encode``
    with it, and live in production training ``callbacks.iter_speed.IterSpeed`` installs one
    per method it wants a per-step time series for (``encode``, ``_prepare_training_data``).
    """

    def __init__(self, model: Any, method_name: str) -> None:
        self._model = model
        self._method_name = method_name
        self._original = getattr(model, method_name)
        self._pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._active = False
        setattr(model, method_name, self._timed_call)

    def _timed_call(self, *args: Any, **kwargs: Any) -> Any:
        if not self._active:
            return self._original(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = self._original(*args, **kwargs)
        end.record()
        self._pairs.append((start, end))
        return result

    def reset(self) -> None:
        """Start a new timed window: drop any prior pairs and begin recording."""
        self._pairs = []
        self._active = True

    def stop_and_read_ms(self) -> float:
        """Stop recording and return total time (ms) across every call since ``reset()``.

        Blocks the CPU on the completion of every recorded pair -- see the class docstring --
        so the caller does not need to synchronize first.
        """
        self._active = False
        return sum(start.elapsed_time(end) for start, end in self._pairs)

    def restore(self) -> None:
        """Undo the monkeypatch, restoring the model's own method."""
        setattr(self._model, self._method_name, self._original)
