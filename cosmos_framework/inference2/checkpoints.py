# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Minimal checkpoint resolver for inference2: name -> local directory.

Replaces the old ``CheckpointOverrides.build_checkpoint`` + ``download_checkpoint``
(Overrides/Args) machinery with a small name->HF-repo table. A friendly name or a bare
HF repo id downloads via ``CheckpointDirHf``; a local path is returned as-is.
"""

from pathlib import Path

from cosmos_framework.utils.checkpoint_db import CheckpointDirHf

# Friendly name -> (HF repository, revision). A pure "name -> download location" table; it
# holds NO model config (that is read from the checkpoint's config.json["model"]).
DOWNLOAD_REGISTRY: dict[str, tuple[str, str]] = {
    "Cosmos3-Nano": ("nvidia/Cosmos3-Nano", "main"),
    "Cosmos3-Super": ("nvidia/Cosmos3-Super", "main"),
    "Cosmos3-Super-Image2Video": ("nvidia/Cosmos3-Super-Image2Video", "main"),
    "Cosmos3-Super-Text2Image": ("nvidia/Cosmos3-Super-Text2Image", "main"),
}


def resolve_checkpoint(checkpoint: str) -> Path:
    """Return a local directory for ``checkpoint``:

    * an existing local directory path -> returned as-is;
    * a friendly registry name (e.g. ``Cosmos3-Nano``) -> downloaded via HF;
    * a bare HF repo id (contains ``/``) -> downloaded via HF at revision ``main``.
    """
    p = Path(checkpoint).expanduser()
    if p.is_dir():
        return p
    if checkpoint in DOWNLOAD_REGISTRY:
        repository, revision = DOWNLOAD_REGISTRY[checkpoint]
    elif "/" in checkpoint:
        repository, revision = checkpoint, "main"
    else:
        raise ValueError(
            f"checkpoint {checkpoint!r} is neither a local dir, a known name "
            f"({sorted(DOWNLOAD_REGISTRY)}), nor an HF repo id"
        )
    return Path(CheckpointDirHf(repository=repository, revision=revision).download())
