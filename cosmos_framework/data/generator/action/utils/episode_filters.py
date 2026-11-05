# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Storage-independent episode keys for optional dataset filter metadata."""

from typing import Any


def normalize_episode_filter_keys(filters: dict[str, Any], *, source_prefix: str = "") -> dict[str, Any]:
    """Preserve complete relative episode IDs; URI keys require an explicit dataset prefix."""
    normalized: dict[str, Any] = {}
    for key, ranges in filters.items():
        recording = key.split("--", 1)[0].removesuffix("/recordings/MP4").rstrip("/")
        episode = recording
        if "://" in recording:
            if not source_prefix:
                raise ValueError("source_prefix is required for URI-based episode filter keys")
            prefix = source_prefix.rstrip("/") + "/"
            if not recording.startswith(prefix):
                raise ValueError("Episode filter URI does not match source_prefix")
            episode = recording.removeprefix(prefix)
        if not episode:
            raise ValueError("Filter entries must identify an episode")
        if episode in normalized and normalized[episode] != ranges:
            raise ValueError(f"Conflicting filter ranges for episode {episode!r}")
        normalized[episode] = ranges
    return normalized
