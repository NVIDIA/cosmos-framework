# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compare portable tokenizer semantics across immutable snapshot locations."""

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path


def relocated_warm_start_media(tokenizer, media: dict, recorded: dict) -> dict:
    """Return a comparison identity without changing the live packer or run identity.

    Older portable records include absolute tokenizer source/file names in their
    media identity. Reconstruct those location labels on a detached tokenizer;
    every non-location setting, vocabulary, backend and media field must still
    match. The model builder separately verifies the immutable base/processor
    file hashes and all adapter settings before loading any trainable state.
    Exact resume deliberately does not use this warm-start comparison.
    """
    from cosmos_framework.posttrain.roles.rollout.validation import _tokenizer_semantics_identity

    if recorded == media:
        return media
    if _tokenizer_semantics_identity(tokenizer) != media.get("packer", {}).get("tokenizer"):
        raise ValueError("warm-start tokenizer differs from the live adapter media identity")
    saved = recorded.get("packer", {}).get("tokenizer", {}).get("settings", {}).get("name_or_path")
    current = getattr(tokenizer, "name_or_path", None)
    if not isinstance(saved, str) or not isinstance(current, str):
        raise ValueError("adapter media compatibility mismatch; tokenizer relocation cannot explain these settings")
    source, destination = Path(current), Path(saved)
    if not source.is_absolute() or not source.is_dir() or not destination.is_absolute():
        raise ValueError("adapter tokenizer relocation requires immutable local snapshot paths")
    rebound = copy(tokenizer)
    rebound.init_kwargs = deepcopy(tokenizer.init_kwargs)
    rebound.name_or_path = saved
    if "name_or_path" in rebound.init_kwargs:
        if rebound.init_kwargs["name_or_path"] != current:
            raise ValueError("tokenizer initialization source differs from its current snapshot")
        rebound.init_kwargs["name_or_path"] = saved
    file_fields = set(tokenizer.vocab_files_names)
    for name, value in rebound.init_kwargs.items():
        if name.endswith("_file") and value is not None:
            if name not in file_fields or not isinstance(value, str):
                raise ValueError(f"unrecognized tokenizer source field for relocation: {name}")
            path = Path(value)
            if not path.is_absolute() or not path.is_relative_to(source) or ".." in path.parts:
                raise ValueError(f"tokenizer source file is outside the immutable snapshot: {name}")
            rebound.init_kwargs[name] = str(destination / path.relative_to(source))
    comparison = deepcopy(media)
    comparison["packer"]["tokenizer"] = _tokenizer_semantics_identity(rebound)
    if comparison != recorded:
        raise ValueError("adapter media compatibility mismatch beyond tokenizer snapshot location")
    return comparison
