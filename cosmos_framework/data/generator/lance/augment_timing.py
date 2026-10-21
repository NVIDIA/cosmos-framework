# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dataloader timing instrumentation for the Lance providers.

The Lance image/video providers apply their augmentor chain in a plain Python
loop, so — unlike ``cosmos_framework/datasets/webdataset/webdataset.py`` — they never
stamp the per-sample timing fields (``_sample_time``, ``_aug_time``,
``_aug_step_times``) that ``joint_dataloader._aggregate_worker_timing`` folds
into ``_worker_*`` batch fields. Without those fields the ``dataloading_monitor``
callback emits no ``dl_worker_batch_time`` / ``dl_worker_balance_per_dataset`` /
``dl_worker_augmentation`` metrics for Lance runs.

:func:`iter_augmented_with_timing` reproduces the WebDataset instrumentation so
those W&B metrics appear for Lance-backed training too. The augmentor step name
matches WebDataset's ``augmentor_fn`` (``__name__`` if present, else the class
name) so the ``aug_steps/<Step>`` series line up across data backends.
"""

import time
from contextlib import nullcontext
from typing import Any, Callable, Iterable, Iterator

from cosmos_framework.data.generator.lance.progress_logging import RejectionProgress, watch_operation
from cosmos_framework.data.generator.lance.sample_rng import (
    LANCE_SAMPLE_RESUME_CURSOR_KEY,
    LANCE_SAMPLE_RNG_SEED_KEY,
    isolated_sample_rng,
)


def _augmentor_step_name(aug: Any) -> str:
    """Match the step name used by WebDataset ``Dataset.augmentor_fn``."""
    return getattr(aug, "__name__", None) or type(aug).__name__


class _AugmentedWithTimingIterator:
    """Apply one timed augmentation step without retaining its returned sample.

    A generator frame keeps its local variables while it is suspended at
    ``yield``. Multi-source loaders keep one suspended iterator per source, so
    retaining ``sample`` or ``result`` there can pin every source's most recent
    decoded media tensor. ``__next__`` returns from a normal call frame instead;
    its per-sample locals are released as soon as the caller receives the item.
    """

    def __init__(
        self,
        source: Iterable[dict[str, Any] | None],
        augmentors: list[Any],
        prepare: Callable[[dict[str, Any]], dict[str, Any] | None] | None,
        max_consecutive_drops: int | None,
        drop_context: str,
        drop_error_hint: str | None,
        yield_drops: bool,
    ) -> None:
        self._source: Iterator[dict[str, Any] | None] = iter(source)
        self._augmentors: list[Any] = augmentors
        self._prepare: Callable[[dict[str, Any]], dict[str, Any] | None] | None = prepare
        self._max_consecutive_drops: int | None = max_consecutive_drops
        self._drop_context: str = drop_context
        self._drop_error_hint: str | None = drop_error_hint
        self._yield_drops: bool = yield_drops
        self._consecutive_drops: int = 0
        self._last_resume_cursor: int | None = None
        self._progress_source: str = str(getattr(source, "source_name", drop_context))
        self._rejection_progress: RejectionProgress = RejectionProgress(self._progress_source)

    def __iter__(self) -> "_AugmentedWithTimingIterator":
        return self

    def close(self) -> None:
        """Forward teardown to the source generator.

        Every call site does ``yield from iter_augmented_with_timing(...)`` inside a
        generator, and ``yield from`` delegates ``close`` to the sub-iterator. Back
        when this class was itself a generator, closing the outer generator threw
        ``GeneratorExit`` into it and closed the Lance source. A plain iterator class
        has no ``close``, so that delegation silently stopped and the source's
        ``pa.memory_map`` handle was only released when GC happened to collect it --
        with one suspended iterator per stream, that defers teardown unpredictably.
        """
        close = getattr(self._source, "close", None)
        if close is not None:
            close()

    @property
    def last_resume_cursor(self) -> int | None:
        """Return the source cursor after the most recent raw sample attempt."""
        return self._last_resume_cursor

    def _record_drop(self) -> None:
        self._consecutive_drops += 1
        if self._max_consecutive_drops is not None and self._consecutive_drops >= self._max_consecutive_drops:
            message = f"{self._drop_context} dropped {self._consecutive_drops} consecutive samples."
            if self._drop_error_hint is not None:
                message = f"{message} {self._drop_error_hint}"
            raise RuntimeError(message)

    def __next__(self) -> dict[str, Any] | None:
        while True:
            t_start = time.monotonic()
            with watch_operation("fetch_sample", source=self._progress_source):
                sample = next(self._source)
            if sample is None:
                self._last_resume_cursor = None
                self._rejection_progress.record("source", time.monotonic() - t_start, "")
                self._record_drop()
                if self._yield_drops:
                    return None
                continue
            resume_cursor = sample.get(LANCE_SAMPLE_RESUME_CURSOR_KEY)
            if resume_cursor is not None and not isinstance(resume_cursor, int):
                raise TypeError(f"{LANCE_SAMPLE_RESUME_CURSOR_KEY} must be an int, got {type(resume_cursor).__name__}.")
            self._last_resume_cursor = resume_cursor

            sample_id = str(sample.get("__key__", ""))
            sample_url = str(sample.get("__url__", ""))
            with watch_operation("prepare", source=self._progress_source, sample_id=sample_id, url=sample_url):
                result = self._prepare(sample) if self._prepare is not None else sample
            if result is None:
                self._rejection_progress.record("prepare", time.monotonic() - t_start, sample_id)
                self._record_drop()
                if self._yield_drops:
                    return None
                continue

            sample_seed = result.pop(LANCE_SAMPLE_RNG_SEED_KEY, None)
            result.pop(LANCE_SAMPLE_RESUME_CURSOR_KEY, None)
            if sample_seed is not None and not isinstance(sample_seed, int):
                raise TypeError(f"{LANCE_SAMPLE_RNG_SEED_KEY} must be an int, got {type(sample_seed).__name__}.")

            aug_start = time.monotonic()
            aug_step_times: dict[str, float] = {}
            rng_context = isolated_sample_rng(sample_seed) if sample_seed is not None else nullcontext()
            with rng_context:
                for aug in self._augmentors:
                    if result is None:
                        break
                    step_start = time.monotonic()
                    step_name = _augmentor_step_name(aug)
                    with watch_operation(step_name, source=self._progress_source, sample_id=sample_id, url=sample_url):
                        result = aug(result)
                    aug_step_times[step_name] = time.monotonic() - step_start
                    if result is None:
                        self._rejection_progress.record(step_name, time.monotonic() - t_start, sample_id)
            if result is None:
                self._record_drop()
                if self._yield_drops:
                    return None
                continue

            self._consecutive_drops = 0
            self._rejection_progress.record(None, 0.0, sample_id)
            result["_aug_step_times"] = aug_step_times
            result["_aug_time"] = time.monotonic() - aug_start
            if "__url__" in result:
                result["__url__"] = str(result["__url__"])
            result["_sample_time"] = time.monotonic() - t_start
            return result


def iter_augmented_with_timing(
    source: Iterable[dict[str, Any] | None],
    augmentors: list[Any],
    prepare: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    max_consecutive_drops: int | None = None,
    drop_context: str = "Lance augmentor pipeline",
    drop_error_hint: str | None = None,
    yield_drops: bool = False,
) -> Iterator[dict[str, Any] | None]:
    """Apply ``augmentors`` to each ``source`` sample, stamping timing fields.

    Mirrors the instrumentation in ``cosmos_framework/datasets/webdataset/webdataset.py``
    (``_sample_timer`` + ``augmentor_fn``):

    - ``_sample_time``: wall-clock to produce one sample (source fetch + prepare
      + full augmentation), matching the WebDataset pipeline's last-stage timer.
    - ``_aug_time``: total time spent inside the augmentor chain.
    - ``_aug_step_times``: ``{step_name: seconds}`` per augmentor.

    Args:
        source: Upstream iterable yielding raw sample dicts (Lance I/O + decode),
            or None for a dropped source occurrence.
        augmentors: Instantiated augmentor callables, applied in order. Any one
            returning ``None`` drops the sample (no value is yielded).
        prepare: Optional per-sample hook run before augmentation (e.g. caption
            injection or video decode). Returning ``None`` drops the sample.
        max_consecutive_drops: Raise after this many consecutive prepare or
            augmentation drops. ``None`` preserves the unbounded default used
            by existing image/video providers.
        drop_context: Source label included in the bounded-drop error.
        drop_error_hint: Optional caller-specific troubleshooting guidance.
        yield_drops: Yield ``None`` for a dropped sample instead of retrying the
            same source internally. The VLM mixer uses this to redraw a source.

    Yields:
        Augmented sample dicts with the timing fields above attached.
    """
    if max_consecutive_drops is not None and max_consecutive_drops < 1:
        raise ValueError(f"max_consecutive_drops must be positive, got {max_consecutive_drops}.")
    return _AugmentedWithTimingIterator(
        source=source,
        augmentors=augmentors,
        prepare=prepare,
        max_consecutive_drops=max_consecutive_drops,
        drop_context=drop_context,
        drop_error_hint=drop_error_hint,
        yield_drops=yield_drops,
    )
