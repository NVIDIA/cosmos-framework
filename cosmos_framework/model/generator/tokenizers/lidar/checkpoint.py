# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint helpers for the LiDAR TransformerVAE."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from cosmos_framework.utils.easy_io import easy_io

# =============================================================================
# Key remapping
# =============================================================================


def remap_checkpoint_state(
    state: Mapping[str, Any],
    model_keys: set[str],
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Select network tensors and strip known training-wrapper prefixes.

    Training checkpoints often nest weights under ``model.network.*`` /
    ``network.*``. Anything that is not a tensor (optimizer state, iteration
    counters, EMA copies, unused training-only params) is ignored.
    """
    remapped: dict[str, torch.Tensor] = {}
    ignored: set[str] = set()
    prefixes = ("model.network.", "network.", "model.vae.", "vae.")

    for original_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            ignored.add(original_key)
            continue

        # Strip the first matching training-wrapper prefix, if any.
        key = original_key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break

        if key in model_keys:
            remapped[key] = value
        else:
            ignored.add(original_key)

    return remapped, ignored


# =============================================================================
# Artifact I/O
# =============================================================================


_SAFETENSORS_SUFFIX = ".safetensors"


def resolve_artifact_path(path: str, bucket_name: str = "") -> str:
    """Join ``bucket_name`` for relative paths; leave URIs unchanged."""
    if "://" in path or not bucket_name:
        return path
    return f"s3://{bucket_name}/{path.lstrip('/')}"


def load_artifact(path: str, *, backend_args: dict[str, Any] | None = None) -> Any:
    """Load a local or remote payload (checkpoint / latent stats), torch or safetensors.

    A published tokenizer ships as ``.safetensors``: a flat tensor mapping carrying the weights
    alongside ``latent_mean`` / ``latent_std``, which the readers below already accept.
    """
    path = _resolve_published_artifact(path)
    if _artifact_suffix(path) == _SAFETENSORS_SUFFIX:
        return _load_safetensors_artifact(path, backend_args=backend_args)
    if "://" in path:
        # Checkpoints embed OmegaConf containers; require full unpickling.
        return easy_io.load(path, backend_args=backend_args, map_location="cpu", weights_only=False)

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"LiDAR tokenizer artifact not found: {resolved}")
    return torch.load(resolved, map_location="cpu", weights_only=False)


def _artifact_suffix(path: str) -> str:
    """Suffix of the path component; a presigned URL carries its signature in the query."""
    return PurePosixPath(path.partition("#")[0].partition("?")[0]).suffix


def _resolve_published_artifact(path: str) -> str:
    """Resolve a published-artifact registry key to a real path; other URIs pass through.

    A public export's ``vae_path`` becomes ``s3://bucket/...`` -- a registry key, not an address.
    Failures are not caught: with ``check_exists=False`` the only one is a registered artifact
    that would not download, and that cause is what the caller needs.
    """
    from cosmos_framework.utils.checkpoint_db import download_checkpoint_v2

    return download_checkpoint_v2(path, check_exists=False)


def _load_safetensors_artifact(path: str, *, backend_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a ``.safetensors`` tokenizer into a flat ``{name: tensor}`` mapping."""
    from safetensors.torch import load as load_safetensors_bytes
    from safetensors.torch import load_file as load_safetensors_file

    if "://" in path:
        return load_safetensors_bytes(easy_io.get(path, backend_args=backend_args))

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"LiDAR tokenizer artifact not found: {resolved}")
    return load_safetensors_file(resolved, device="cpu")


def parse_latent_stats(stats: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse ``(mean, std)`` or ``{"mean": ..., "std": ...}`` latent statistics.

    Returns:
        mean, std: each shaped ``[C]`` (one value per latent channel).
    """
    if isinstance(stats, (tuple, list)) and len(stats) == 2:
        mean, std = stats
    elif isinstance(stats, Mapping) and "mean" in stats and "std" in stats:
        mean, std = stats["mean"], stats["std"]
    else:
        raise ValueError("Latent stats must be a (mean, std) pair or mapping with mean/std keys")

    return torch.as_tensor(mean), torch.as_tensor(std)  # [C], [C]


def parse_lidar_stats(
    stats: Any,
) -> tuple[torch.Tensor, torch.Tensor, float | None, float | None]:
    """Parse latent statistics plus optional model-owned metric range bounds."""
    mean, std = parse_latent_stats(stats)
    if not isinstance(stats, Mapping):
        return mean, std, None, None
    min_range = float(stats["min_range"]) if "min_range" in stats else None
    max_range = float(stats["max_range"]) if "max_range" in stats else None
    if (min_range is None) != (max_range is None):
        raise ValueError("LiDAR stats must contain both min_range and max_range")
    if min_range is not None and max_range is not None and max_range <= min_range:
        raise ValueError(f"max_range must exceed min_range, got {min_range=} and {max_range=}")
    return mean, std, min_range, max_range


def parse_lidar_checkpoint_stats(
    checkpoint: Any,
) -> tuple[torch.Tensor, torch.Tensor, float | None, float | None]:
    """Read latent statistics and range bounds embedded by the training wrapper."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint model state must be a mapping, got {type(state).__name__}")

    def _get(name: str, *, required: bool = True) -> Any:
        for key in (name, f"model.{name}"):
            if key in state:
                return state[key]
        if required:
            raise ValueError(f"LiDAR tokenizer checkpoint does not contain {name!r}")
        return None

    stats = {
        "mean": _get("latent_mean"),
        "std": _get("latent_std"),
    }
    min_range = _get("min_range", required=False)
    max_range = _get("max_range", required=False)
    if (min_range is None) != (max_range is None):
        raise ValueError("LiDAR tokenizer checkpoint must contain both min_range and max_range")
    if min_range is not None:
        stats.update(min_range=min_range, max_range=max_range)
    return parse_lidar_stats(stats)


# =============================================================================
# Warm start
# =============================================================================


def resize_azimuth_embedding(embedding: torch.Tensor, width: int) -> torch.Tensor:
    """Resample a learned range-view embedding ``[1,H,W,C]`` onto ``width`` azimuth bins.

    Azimuth wraps, so the last column neighbours the first and source columns
    are read modulo the source width. Sample positions sit at bin centers, which
    keeps both grids spanning the same 360 degrees at any bin count.
    """
    source_width = int(embedding.shape[2])
    if source_width < 1:
        raise ValueError(f"embedding needs a non-empty azimuth axis, got shape {tuple(embedding.shape)}")
    if width < 1:
        raise ValueError(f"target width must be positive, got {width}")
    if source_width == width:
        return embedding

    # float64 so that bin centers stay exact for large widths.
    centers = (torch.arange(width, dtype=torch.float64) + 0.5) * source_width / width - 0.5  # [width]
    lower = torch.floor(centers)
    weight = (centers - lower).to(embedding.dtype)[None, None, :, None]  # [1,1,width,1]
    left = lower.to(torch.long) % source_width  # [width]
    right = (lower.to(torch.long) + 1) % source_width  # [width]
    return embedding[:, :, left, :] * (1.0 - weight) + embedding[:, :, right, :] * weight


def _differs_only_along_azimuth(source: torch.Tensor, target: torch.Tensor) -> bool:
    """True when two ``[1,H,W,C]`` range-view embeddings differ only in width."""
    if source.ndim != 4 or target.ndim != 4:
        return False
    return (
        source.shape[0] == target.shape[0]
        and source.shape[1] == target.shape[1]
        and source.shape[3] == target.shape[3]
        and source.shape[2] != target.shape[2]
    )


def warm_start_network(
    network: torch.nn.Module,
    checkpoint_path: str,
    *,
    backend_args: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Initialize ``network`` from a checkpoint that need not match it exactly.

    For starting a recipe from a published tokenizer whose range view or channel
    layout differs. Tensors that already agree are copied, range-view positional
    embeddings are resampled across azimuth, and anything still incompatible is
    left at its freshly initialized value instead of raising -- unlike
    :func:`load_model_checkpoint`, which serves published inference weights and
    must reject any drift.

    Returns the per-category key lists so callers can log what actually landed.
    """
    payload = load_artifact(checkpoint_path, backend_args=backend_args)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(payload).__name__}")
    raw_state = payload.get("model", payload)
    if not isinstance(raw_state, Mapping):
        raise TypeError(f"Checkpoint model state must be a mapping, got {type(raw_state).__name__}")

    own_state = network.state_dict()
    remapped, _ = remap_checkpoint_state(raw_state, set(own_state))

    usable: dict[str, torch.Tensor] = {}
    resized: list[str] = []
    skipped: list[str] = []
    for key, tensor in remapped.items():
        target = own_state[key]
        if tensor.shape == target.shape:
            usable[key] = tensor
        elif key.endswith("spatial_pe.embedding") and _differs_only_along_azimuth(tensor, target):
            usable[key] = resize_azimuth_embedding(tensor, int(target.shape[2]))
            resized.append(key)
        else:
            skipped.append(key)

    # ``strict=False`` tolerates absent keys but still raises on a shape
    # difference, so incompatible tensors have to be dropped above.
    load_info = network.load_state_dict(usable, strict=False)
    return {
        "copied": sorted(set(usable) - set(resized)),
        "resized": sorted(resized),
        "skipped": sorted(skipped),
        "missing": sorted(load_info.missing_keys),
    }


# =============================================================================
# Model loading
# =============================================================================


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    backend_args: dict[str, Any] | None = None,
    error_prefix: str = "Incompatible LiDAR tokenizer checkpoint",
) -> Mapping[str, Any]:
    """Remap and strictly load a published LiDAR tokenizer checkpoint.

    Unknown training-only keys are filtered by ``remap_checkpoint_state``.
    Missing or shape-mismatched keys required by ``model`` raise.
    """
    payload = load_artifact(checkpoint_path, backend_args=backend_args)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(payload).__name__}")

    # Prefer nested ``payload["model"]`` when present (training wrapper layout).
    raw_state = payload.get("model", payload)
    if not isinstance(raw_state, Mapping):
        raise TypeError(f"Checkpoint model state must be a mapping, got {type(raw_state).__name__}")

    own_state = model.state_dict()
    remapped, _ = remap_checkpoint_state(raw_state, set(own_state))

    missing = set(own_state) - set(remapped)
    mismatched = {
        key: (tuple(remapped[key].shape), tuple(own_state[key].shape))
        for key in remapped.keys() & own_state.keys()
        if remapped[key].shape != own_state[key].shape
    }
    if missing or mismatched:
        missing_preview = sorted(missing)[:20]
        raise RuntimeError(
            f"{error_prefix}: missing={missing_preview}"
            f"{'...' if len(missing) > len(missing_preview) else ''}, mismatched={mismatched}"
        )

    model.load_state_dict(remapped, strict=True)
    return payload
