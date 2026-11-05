# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""General pretrained-weight loader: safetensors -> (FSDP-sharded) model.

Model-agnostic port of the cosmos3 ``safetensors_loader`` (``load_language_model`` +
``_shard_tensor_on_fsdp_mesh`` + ``_get_dp_shard_mesh``), parameterized by a ``LoadPlan``
(``rename`` + ``skip_patterns``) instead of the per-family key remap / fixed skip list.

Each rank fills only its ``dp_shard`` slice: for a sharded (DTensor) target the full
checkpoint tensor is sliced along dim 0 by ``(dp_shard rank, size)`` and copied into the
target's local shard; for a plain tensor the full tensor is copied. ``skip_patterns``
carry BOTH meanings (matching cosmos3's Phase-5 skip + Phase-6 tolerance): a resolved
model key matching a pattern is (a) never copied AND (b) tolerated when absent from the
checkpoint (exempt from the completeness check).

Distinct from the DCP training-checkpoint path (that's ``LLMTrainer.save/load``). Design
spec section 4.3.
"""

from __future__ import annotations

import os
import re

import torch

from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
from cosmos_framework.posttrain.model.specs import LoadPlan

# Wrapper prefixes that torch.compile / activation-checkpointing inject into
# parameter names; stripped so checkpoint keys line up with the raw model keys.
_WRAPPER_PREFIXES = ("_orig_mod.", "_checkpoint_wrapped_module.")


def _canonical_key(name: str) -> str:
    """Strip torch.compile / checkpoint-wrapper prefixes from a state-dict key."""
    for pfx in _WRAPPER_PREFIXES:
        name = name.replace(pfx, "")
    return name


def _resolve_safetensors(checkpoint_path: str, credential_path: str | None) -> list[str]:
    """Resolve ``checkpoint_path`` to a list of local ``.safetensors`` file paths.

    LOCAL resolution (a directory of ``*.safetensors`` or a single ``*.safetensors``
    file) is fully implemented here and requires no heavy dependencies. Hugging Face
    Hub repo ids / ``hf://`` URIs are resolved by lazily delegating to the cosmos3
    ``safetensors_loader`` (which pulls ``huggingface_hub`` / ``easy_io``) and then
    listing the downloaded snapshot locally. S3 (``s3://``) resolution is a clear TODO
    branch — the cosmos3 ``MultiRankCheckpointLoader`` streams S3 bytes via ``easy_io``,
    which does not fit this ``safe_open``-on-path reader without extra plumbing.
    """
    expanded = os.path.expanduser(checkpoint_path)

    # --- Local directory of safetensors shards. ---
    if os.path.isdir(expanded):
        files = sorted(f for f in os.listdir(expanded) if f.endswith(".safetensors"))
        if files:
            return [os.path.join(expanded, f) for f in files]

    # --- Local single safetensors file. ---
    if expanded.endswith(".safetensors") and os.path.isfile(expanded):
        return [expanded]

    # --- Remote resolution. ---
    if checkpoint_path.startswith("s3://"):
        # TODO: port the cosmos3 MultiRankCheckpointLoader easy_io/S3 streaming path.
        raise NotImplementedError(
            f"S3 checkpoint resolution is not implemented in weight_loader yet: {checkpoint_path}. "
            "Download the safetensors locally, or use the cosmos3 safetensors_loader "
            "(MultiRankCheckpointLoader) which streams S3 via easy_io."
        )

    # Hugging Face Hub (explicit hf:// URI or a bare org/repo id). Delegate the
    # download to cosmos3's resolver, then list the local snapshot.
    from cosmos_framework.model.generator.utils.safetensors_loader import (
        _download_hf_checkpoint,
        _is_hf_checkpoint_candidate,
    )

    if _is_hf_checkpoint_candidate(checkpoint_path):
        local_dir = _download_hf_checkpoint(checkpoint_path)
        files = sorted(f for f in os.listdir(local_dir) if f.endswith(".safetensors"))
        if files:
            return [os.path.join(local_dir, f) for f in files]

    raise FileNotFoundError(f"No .safetensors files found for checkpoint path: {checkpoint_path}")


def _local_shard(full: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the local-shard view of ``full`` matching ``target``'s layout.

    Single device (target is a plain tensor) -> the whole tensor. FSDP2 param (target is
    a DTensor) -> shard ``full`` EXACTLY like ``target`` via ``distribute_tensor`` on the
    target's own mesh + placements. This handles uneven sharding (dim 0 not divisible by
    the shard size, e.g. a 50257-row vocab embedding across 2 ranks) and any padding FSDP2
    applies — the earlier even-``chunk`` port wrongly required divisibility.
    """
    if not hasattr(target, "to_local"):
        return full.contiguous()

    from torch.distributed.tensor import distribute_tensor

    local = target.to_local()
    src = full.to(local.dtype).to(local.device)
    # distribute_tensor produces a DTensor with target's exact placement/mesh, so its
    # local shape matches target.to_local() row-for-row (including any padding).
    # src_data_rank=None -> purely local slice (no scatter): every rank already loaded the
    # SAME `full` from the same checkpoint, so the default src_data_rank=0 broadcast is wasted.
    return distribute_tensor(src, target.device_mesh, target.placements, src_data_rank=None).to_local().contiguous()


def load_weights(
    model,
    checkpoint_path: str,
    plan: LoadPlan,
    pdims: ParallelDims | None = None,
    credential_path: str | None = None,
) -> set[str]:
    """Load safetensors weights from ``checkpoint_path`` into ``model``.

    Args:
        model: The (possibly FSDP2-sharded) module to fill. Parameters may be plain
            tensors (single device) or DTensors (each rank owns a dim-0 shard).
        checkpoint_path: Local dir / file of ``*.safetensors`` (fully supported), an
            ``hf://org/repo`` or bare ``org/repo`` Hub id (delegated to cosmos3), or an
            ``s3://`` URI (currently a clear TODO — raises ``NotImplementedError``).
        plan: ``LoadPlan`` — ``rename`` maps checkpoint keys to model keys (``None`` =
            identity); ``skip_patterns`` are regexes on the resolved model key that both
            skip-copy the key and tolerate its absence from the checkpoint.
        pdims: ``ParallelDims`` for dp_shard striping, or ``None`` for single-rank load.
        credential_path: Optional credentials path for remote resolution.

    Returns:
        The set of model state-dict keys populated from the checkpoint.

    Raises:
        ValueError: when a required (non-skipped) model parameter is missing from the
            checkpoint.
    """
    from safetensors import safe_open

    # Canonical model state dict (compile/checkpoint wrapper prefixes stripped).
    model_sd = {_canonical_key(k): v for k, v in model.state_dict().items()}
    skip = [re.compile(p) for p in plan.skip_patterns]

    def _skipped(key: str) -> bool:
        return any(r.search(key) for r in skip)

    loaded: set[str] = set()
    for path in _resolve_safetensors(checkpoint_path, credential_path):
        with safe_open(path, framework="pt", device="cpu") as f:
            for ckpt_key in f.keys():
                key = plan.rename(ckpt_key) if plan.rename else ckpt_key
                if _skipped(key) or key not in model_sd:
                    continue  # skip-copy matched keys; ignore extra checkpoint keys
                target = model_sd[key]
                is_dtensor = hasattr(target, "to_local")
                local_view = target.to_local() if is_dtensor else target
                full = f.get_tensor(ckpt_key).to(local_view.dtype)
                shard = _local_shard(full, target)
                if shard.device != local_view.device:
                    shard = shard.to(local_view.device)
                if shard.shape != local_view.shape:
                    raise ValueError(
                        f"Shape mismatch for '{key}': local_view={tuple(local_view.shape)}, "
                        f"shard={tuple(shard.shape)} (checkpoint dim-0={full.shape[0]})"
                    )
                with torch.no_grad():
                    local_view.copy_(shard)
                loaded.add(key)

    # Completeness check: every model key must be loaded UNLESS it matches a skip
    # pattern (tolerate-absent). Mirrors cosmos3's Phase-6 tolerance.
    missing = {k for k in model_sd if k not in loaded and not _skipped(k)}
    if missing:
        raise ValueError(
            f"load_weights: {len(missing)} required model parameter(s) not found in "
            f"checkpoint '{checkpoint_path}'. First up to 10: {sorted(missing)[:10]}"
        )

    return loaded
