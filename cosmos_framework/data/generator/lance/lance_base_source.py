# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from dataclasses import dataclass, field


def combine_lance_filters(lance_filters: tuple[str, ...]) -> str:
    """Return predicates as one stable parenthesized SQL expression."""
    if not lance_filters:
        return ""
    combined_filter = " OR ".join(f"({lance_filter})" for lance_filter in lance_filters)
    if len(lance_filters) == 1:
        return combined_filter
    return f"({combined_filter})"


@dataclass
class LanceBaseDatasetSpec:
    """Storage identity shared by video, image, and Action Lance sources.

    ``lance_uri`` is always the training-facing S3-compatible URI. The offline
    cache builder continues to read that URI through the training source, but
    writes the cache through its native ``gs://`` alias by default. Set
    ``use_native_gcs_for_cache_write=False`` only for non-GCS storage.
    """

    name: str
    lance_uri: str
    use_native_gcs_for_cache_write: bool = field(default=True, kw_only=True)
