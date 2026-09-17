# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in process-local cache for dataloader processor LazyCall configs."""

import os
import threading
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from cosmos_framework.utils.lazy_config import instantiate


@dataclass(frozen=True)
class _ProcessorCacheEntry:
    processor: Any
    config_fingerprint: Hashable


_PROCESSOR_CACHE: dict[tuple[int, str], _ProcessorCacheEntry] = {}
_PROCESSOR_CACHE_LOCK = threading.Lock()


def _config_fingerprint(value: Any) -> Hashable:
    """Return a stable, hashable representation of one processor config."""
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=False)
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((str(key), _config_fingerprint(item)) for key, item in value.items())),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_config_fingerprint(item) for item in value))
    if isinstance(value, set):
        return ("set", tuple(sorted((_config_fingerprint(item) for item in value), key=repr)))
    if isinstance(value, Hashable):
        return (type(value).__module__, type(value).__qualname__, value)
    return (type(value).__module__, type(value).__qualname__, repr(value))


def get_cached_processor(*, cache_key: str, processor_config: Any) -> Any:
    """Instantiate ``processor_config`` once for one explicit process-local key.

    This helper is intentionally opt-in: callers must wrap a processor LazyCall
    and provide a cache key. Direct processor instantiation, including model
    processor construction, remains independent of this cache.
    """
    if not cache_key:
        raise ValueError("cache_key must not be empty.")
    process_cache_key = (os.getpid(), cache_key)
    config_fingerprint = _config_fingerprint(processor_config)
    with _PROCESSOR_CACHE_LOCK:
        entry = _PROCESSOR_CACHE.get(process_cache_key)
        if entry is not None:
            if entry.config_fingerprint != config_fingerprint:
                raise ValueError(
                    f"cache_key {cache_key!r} is already bound to a different processor_config "
                    f"in process {process_cache_key[0]}."
                )
            return entry.processor

        processor = instantiate(processor_config)
        _PROCESSOR_CACHE[process_cache_key] = _ProcessorCacheEntry(
            processor=processor,
            config_fingerprint=config_fingerprint,
        )
        return processor


def clear_processor_cache() -> None:
    """Clear all process-keyed entries, primarily for isolated tests."""
    with _PROCESSOR_CACHE_LOCK:
        _PROCESSOR_CACHE.clear()
