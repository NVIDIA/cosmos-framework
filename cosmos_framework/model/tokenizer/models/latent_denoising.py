# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Latent-corruption utilities for denoising-tokenizer training."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.model.tokenizer.checkpoint_identity import (
    extract_checkpoint_provenance,
    resolve_checkpoint_identity,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _checkpoint_identities_match(
    recorded_identity: Mapping[str, object],
    current_identity: Mapping[str, object],
) -> bool:
    """Compare local checkpoints by content while keeping remote locations identity-bearing."""
    recorded_kind = recorded_identity.get("kind")
    current_kind = current_identity.get("kind")
    if recorded_kind != current_kind:
        return False
    if recorded_kind == "local_file":
        content_keys = ("kind", "size", "sha256")
        return all(recorded_identity.get(key) == current_identity.get(key) for key in content_keys)
    if recorded_kind == "local_directory":
        content_keys = ("kind", "files")
        return all(recorded_identity.get(key) == current_identity.get(key) for key in content_keys)
    return dict(recorded_identity) == dict(current_identity)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash one local normalization sidecar without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_latent_normalization(
    path: str,
    *,
    latent_channels: int,
    backend_args: Mapping[str, Any] | None = None,
    expected_sidecar_sha256: str = "",
    expected_checkpoint_path: str | None = None,
    require_checkpoint_identity: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:  # returns mean: [C], std: [C]
    """Load and validate per-channel deployment latent statistics."""
    if not path:
        raise ValueError("A latent-normalization sidecar path is required.")
    if isinstance(backend_args, DictConfig):
        plain_backend_args = OmegaConf.to_container(backend_args, resolve=True)
        if not isinstance(plain_backend_args, dict):
            raise TypeError("Latent-normalization backend_args must contain a mapping.")
        backend_args = plain_backend_args
    elif backend_args is not None:
        backend_args = dict(backend_args)

    if expected_sidecar_sha256:
        if _SHA256_PATTERN.fullmatch(expected_sidecar_sha256) is None:
            raise ValueError(
                "expected_sidecar_sha256 must contain exactly 64 lower-case hexadecimal characters; "
                f"got {expected_sidecar_sha256!r}."
            )
        sidecar_path = Path(path)
        if not sidecar_path.is_file():
            raise ValueError(
                f"SHA-256-pinned latent normalization currently requires a local immutable sidecar file; got {path!r}."
            )
        actual_sidecar_sha256 = _sha256_file(sidecar_path)
        if actual_sidecar_sha256 != expected_sidecar_sha256:
            raise ValueError(
                f"Latent-normalization sidecar SHA-256 mismatch for {path}: expected "
                f"{expected_sidecar_sha256}, got {actual_sidecar_sha256}."
            )

    if path.lower().endswith(".json"):
        payload = easy_io.load(path, backend_args=backend_args)
    else:
        payload = easy_io.load(path, backend_args=backend_args, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Latent-normalization sidecar must contain a mapping, got {type(payload).__name__}.")
    if "mean" not in payload or "std" not in payload:
        raise ValueError("Latent-normalization sidecar must contain mean and std entries.")
    sidecar_channels = payload.get("z_dim")
    if sidecar_channels is not None and (isinstance(sidecar_channels, bool) or not isinstance(sidecar_channels, int)):
        raise ValueError(f"Latent-normalization z_dim must be an integer, got {sidecar_channels!r}.")
    if isinstance(sidecar_channels, int) and sidecar_channels != latent_channels:
        raise ValueError(f"Latent-normalization sidecar has z_dim={sidecar_channels}, expected {latent_channels}.")
    source_checkpoint, source_checkpoint_identity = extract_checkpoint_provenance(payload, source=path)
    if source_checkpoint_identity is not None or require_checkpoint_identity:
        if not expected_checkpoint_path:
            raise ValueError(
                "An expected checkpoint path is required when latent-normalization checkpoint/statistics "
                "matching is enabled."
            )
        if not source_checkpoint_identity:
            raise ValueError(
                f"Latent-normalization sidecar {path} must contain source_checkpoint_identity "
                "when checkpoint/statistics matching is required."
            )
        credentials_path = ""
        if backend_args is not None:
            credentials_value = backend_args.get("s3_credential_path", "")
            if not isinstance(credentials_value, str):
                raise TypeError("Latent-normalization s3_credential_path must be a string.")
            credentials_path = credentials_value
        current_checkpoint_identity = resolve_checkpoint_identity(expected_checkpoint_path, credentials_path)
        recorded_kind = source_checkpoint_identity.get("kind")
        current_kind = current_checkpoint_identity.get("kind")
        both_local = recorded_kind in {"local_file", "local_directory"} and recorded_kind == current_kind
        if not both_local and source_checkpoint != expected_checkpoint_path:
            raise ValueError(
                f"Latent-normalization sidecar {path} was computed for checkpoint "
                f"{source_checkpoint!r}, expected {expected_checkpoint_path!r}."
            )
        if not _checkpoint_identities_match(source_checkpoint_identity, current_checkpoint_identity):
            raise ValueError(
                f"Latent-normalization sidecar {path} checkpoint identity does not match the current checkpoint "
                f"at {expected_checkpoint_path}; regenerate statistics from the exact checkpoint used for training."
            )

    def _as_vector(values: object, name: str) -> torch.Tensor:  # returns [C]
        if isinstance(values, torch.Tensor):
            if values.dtype == torch.bool:
                raise ValueError(f"Latent-normalization {name} must contain real numbers, not booleans.")
            if values.is_complex():
                raise ValueError(f"Latent-normalization {name} must contain real numbers, not complex values.")
            vector = values.detach().to(device="cpu", dtype=torch.float32)  # [C]
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise TypeError(f"Latent-normalization {name} must contain only real numbers.")
            vector = torch.tensor(list(values), dtype=torch.float32)  # [C]
        else:
            raise TypeError(f"Latent-normalization {name} must be a tensor or numeric sequence.")
        if vector.shape != (latent_channels,):
            raise ValueError(
                f"Latent-normalization {name} must have shape ({latent_channels},), got {tuple(vector.shape)}."
            )
        finite_mask = torch.isfinite(vector)  # [C]
        if not bool(finite_mask.all()):
            raise ValueError(f"Latent-normalization {name} contains non-finite values.")
        return vector  # [C]

    mean = _as_vector(payload["mean"], "mean")  # [C]
    std = _as_vector(payload["std"], "std")  # [C]
    positive_mask = std > 0  # [C]
    if not bool(positive_mask.all()):
        raise ValueError("Latent-normalization std must be strictly positive.")
    return mean, std


def interpolate_latent_noise(
    clean_latent: torch.Tensor,  # [N,C]
    noise: torch.Tensor,  # [N,C]
    corruption_level: torch.Tensor,  # [B]
    batch_indices: torch.Tensor,  # [N]
) -> torch.Tensor:  # returns [N,C]
    """Interpolate normalized clean latents toward Gaussian noise."""
    if clean_latent.shape != noise.shape:
        raise ValueError(
            f"clean_latent and noise must have identical shapes, got {clean_latent.shape} and {noise.shape}."
        )
    if clean_latent.ndim != 2:
        raise ValueError(f"clean_latent must have shape [N,C], got {tuple(clean_latent.shape)}.")
    if corruption_level.ndim != 1:
        raise ValueError(f"corruption_level must have shape [B], got {tuple(corruption_level.shape)}.")
    if batch_indices.ndim != 1 or batch_indices.shape[0] != clean_latent.shape[0]:
        raise ValueError(f"batch_indices must have shape ({clean_latent.shape[0]},), got {tuple(batch_indices.shape)}.")
    if batch_indices.numel() > 0:
        min_batch_index = int(batch_indices.min())
        max_batch_index = int(batch_indices.max())
        if min_batch_index < 0 or max_batch_index >= corruption_level.shape[0]:
            raise ValueError("batch_indices contains a sample index outside the corruption-level batch.")

    token_level = corruption_level.index_select(0, batch_indices).to(dtype=clean_latent.dtype)[:, None]  # [N,1]
    return (1.0 - token_level) * clean_latent + token_level * noise  # [N,C]
