# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prompt-source construction, content identity, and held-out materialization."""

from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import TYPE_CHECKING, Any, Literal

from cosmos_framework.posttrain.roles.env.dataset import (
    ArithmeticEnv,
    DatasetEnv,
    TrainValidationEnv,
    hf_items,
    jsonl_prompts,
    map_prompt_items,
    require_readable_jsonl,
)
from cosmos_framework.posttrain.roles.env.identity import dataset_fingerprint, stable_value_digest

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.rl import GRPOConfig


def prompt_source(rl: GRPOConfig) -> tuple[Literal["arithmetic", "jsonl", "hf", "custom"], bool]:
    """Return the configured prompt-source kind and whether it carries references."""
    if (rl.data.name or "").strip():
        return "hf", bool(rl.data.response_column)
    if (rl.data.factory or "").strip():
        return "custom", bool(rl.data.response_column)
    if rl.data.path:
        return "jsonl", bool(rl.data.response_column)
    return "arithmetic", True


@dataclasses.dataclass(frozen=True)
class DatasetSourceIdentity:
    """Content identity used to reject train/validation source overlap."""

    fingerprint: str
    prompt_membership: frozenset[bytes] | None

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError("dataset source fingerprint must be a non-empty string")
        if self.prompt_membership is not None and not isinstance(self.prompt_membership, frozenset):
            raise TypeError("prompt membership must be a frozenset or None")


def validate_dataset_source(data_config: Any) -> None:
    """Fail before role placement when a configured dataset cannot be loaded."""
    if data_config.path:
        require_readable_jsonl(data_config.path)
    if data_config.factory:
        from cosmos_framework.posttrain.data.datasets.factory import validate_dataset_factory

        validate_dataset_factory(data_config.factory)
    if data_config.items_fn:
        from cosmos_framework.utils.serialization import load_callable

        try:
            load_callable(data_config.items_fn)
        except Exception as error:  # noqa: BLE001 -- convert arbitrary import failure into config context
            raise ValueError(
                f"items_fn={data_config.items_fn!r} is not importable: {error}. It must be 'pkg.mod.fn' on PYTHONPATH."
            ) from error


def dataset_items(data_config: Any) -> Any:
    """Load one explicit prompt dataset into the canonical prompt/reference view."""
    row_map = None
    if data_config.items_fn:
        from cosmos_framework.utils.serialization import load_callable

        row_map = load_callable(data_config.items_fn)
    if data_config.name:
        return hf_items(data_config, row_map=row_map)
    if data_config.factory:
        from cosmos_framework.posttrain.data.datasets.factory import build_custom_dataset

        dataset = build_custom_dataset(data_config.factory, data_config.factory_kwargs)
        identity = stable_value_digest(
            {
                "factory": data_config.factory,
                "factory_kwargs": dict(data_config.factory_kwargs),
            }
        )
        return map_prompt_items(
            dataset,
            data_config.prompt_column,
            data_config.response_column,
            row_map,
            source_identity=f"custom-prompt-sha256:{hashlib.sha256(identity).hexdigest()}",
        )
    if data_config.path:
        return jsonl_prompts(
            data_config.path,
            data_config.prompt_column,
            data_config.response_column,
        )
    raise ValueError("an explicit GRPO dataset requires name, path, or factory")


def data_shard(config: Any) -> dict[str, int]:
    """Return shard coordinates along the data role's rollout-derived replica axis."""
    from cosmos_framework.posttrain.config.build import replica_coords

    grid = getattr(config, "grid", None)
    rollout_shape = getattr(grid, "rollout_shape", None)
    count = int(rollout_shape()[0] if callable(rollout_shape) else (getattr(grid, "num_replicas", 1) or 1))
    if count <= 1:
        return {}
    index = replica_coords()[0]
    if not 0 <= index < count:
        raise ValueError(
            f"data shard index {index} out of range for data-role replica count {count} (R_rollout): "
            "the placement coordinates and the declared replica count disagree"
        )
    return {"shard_index": index, "shard_count": count}


def dataset_source_identity(data_config: Any, items: Any) -> DatasetSourceIdentity:
    """Resolve stable source and prompt-membership identities without media decoding."""
    if data_config.path:
        row_digests = sorted(stable_value_digest(items[index]) for index in range(len(items)))
        digest = hashlib.sha256()
        digest.update(len(row_digests).to_bytes(8, byteorder="big"))
        for row_digest in row_digests:
            digest.update(len(row_digest).to_bytes(8, byteorder="big"))
            digest.update(row_digest)
        prompt_membership = frozenset(stable_value_digest(items[index]["prompt"]) for index in range(len(items)))
        return DatasetSourceIdentity(
            fingerprint=f"jsonl-membership-sha256={digest.hexdigest()}|len={len(row_digests)}",
            prompt_membership=prompt_membership,
        )
    source_fingerprint = getattr(items, "raw_source_fingerprint", None)
    fingerprint = str(source_fingerprint()) if callable(source_fingerprint) else dataset_fingerprint(items)
    return DatasetSourceIdentity(fingerprint=fingerprint, prompt_membership=None)


def reject_dataset_overlap(training: DatasetSourceIdentity, validation: DatasetSourceIdentity) -> None:
    """Reject identical sources and any provable shared prompt membership."""
    if training.fingerprint == validation.fingerprint:
        raise ValueError("training and validation datasets resolve to the same content fingerprint")
    if training.prompt_membership is None or validation.prompt_membership is None:
        return
    overlap_count = len(training.prompt_membership.intersection(validation.prompt_membership))
    if overlap_count:
        raise ValueError(
            "training and validation JSONL datasets overlap semantically: "
            f"{overlap_count} canonical prompt identity/identities are shared"
        )


def materialize_validation_items(
    items: Any,
    *,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Freeze held-out rows under isolated deterministic per-index RNG streams."""
    limit = len(items) if max_samples is None else min(int(max_samples), len(items))
    python_state = random.getstate()
    numpy_module: Any | None = None
    numpy_state: Any | None = None
    torch_module: Any | None = None
    torch_state: Any | None = None
    cuda_states: Any | None = None
    cuda_was_initialized = False
    try:
        try:
            import numpy as np

            numpy_module = np
            numpy_state = np.random.get_state()
        except ImportError:
            pass
        try:
            import torch

            torch_module = torch
            torch_state = torch.random.get_rng_state()  # [RNG]
            cuda_was_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
            if cuda_was_initialized:
                cuda_states = torch.cuda.get_rng_state_all()  # one [RNG] state per initialized CUDA device
        except ImportError:
            pass

        materialized: list[dict[str, Any]] = []
        for index in range(limit):
            digest = hashlib.blake2b(
                f"{int(seed)}\0{index}".encode(),
                digest_size=8,
                person=b"pt-val-map",
            ).digest()
            item_seed = int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)
            random.seed(item_seed)
            if numpy_module is not None:
                numpy_module.random.seed(item_seed % (2**32))
            if torch_module is not None:
                torch_module.random.default_generator.manual_seed(item_seed)
                if cuda_was_initialized:
                    torch_module.cuda.manual_seed_all(item_seed)
            torch_seed_functions: tuple[Any, Any, Any] | None = None
            if torch_module is not None:
                torch_seed_functions = (
                    torch_module.manual_seed,
                    torch_module.cuda.manual_seed,
                    torch_module.cuda.manual_seed_all,
                )

                def _forbid_torch_seed(*_args: Any, **_kwargs: Any) -> None:
                    raise RuntimeError(
                        "validation items_fn must not reseed PyTorch; use the deterministic RNG "
                        "stream supplied by validation materialization"
                    )

                torch_module.manual_seed = _forbid_torch_seed
                torch_module.cuda.manual_seed = _forbid_torch_seed
                torch_module.cuda.manual_seed_all = _forbid_torch_seed
            try:
                row = items[index]
            finally:
                if torch_module is not None and torch_seed_functions is not None:
                    (
                        torch_module.manual_seed,
                        torch_module.cuda.manual_seed,
                        torch_module.cuda.manual_seed_all,
                    ) = torch_seed_functions
            if (
                torch_module is not None
                and not cuda_was_initialized
                and torch_module.cuda.is_available()
                and torch_module.cuda.is_initialized()
            ):
                raise RuntimeError(
                    "validation items_fn initialized CUDA while materializing data; items_fn must be "
                    "a deterministic, side-effect-free CPU row transform"
                )
            if not isinstance(row, dict):
                raise TypeError(
                    f"validation row {index} must resolve to a dict after items_fn/column mapping, "
                    f"got {type(row).__name__}"
                )
            materialized.append(dict(row))
        return materialized
    finally:
        random.setstate(python_state)
        if numpy_module is not None and numpy_state is not None:
            numpy_module.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.random.set_rng_state(torch_state)  # [RNG]
            if cuda_states is not None:
                torch_module.cuda.set_rng_state_all(cuda_states)  # one [RNG] state per initialized CUDA device


def build_prompt_env(rl: Any, config: Any = None) -> Any:
    """Build the GRPO training source and optional finite held-out view."""
    from cosmos_framework.posttrain.config.rl import resolve_data_seed

    shard = data_shard(config)
    # One resolved prompt-order seed (`rl.seed`, or top-level `rollout.seed` under
    # `rl.data_seed_follows_rollout_seed`) — the same value preflight resolved.
    data_seed = resolve_data_seed(rl, getattr(config, "rollout", None))
    # Exact continuation of a model-only artifact: the draws each shard skips, and the shard count
    # they were computed for (`DatasetConfig.cursor_offset` / `cursor_shards`; 0 = none / unchecked).
    cursor_offset = int(getattr(rl.data, "cursor_offset", 0) or 0)
    cursor_shards = int(getattr(rl.data, "cursor_shards", 0) or 0)
    training_items: Any | None = None
    if not rl.data.name and not rl.data.path and not rl.data.factory:
        if cursor_offset or cursor_shards:
            raise ValueError(
                "rl.data.cursor_offset/cursor_shards need an explicit prompt source: the arithmetic source "
                "synthesizes prompts from its seed and has no cursor to continue"
            )
        training_env: Any = ArithmeticEnv(seed=data_seed, max_operand=rl.max_operand, **shard)
    else:
        training_items = dataset_items(rl.data)
        training_env = DatasetEnv(
            training_items,
            shuffle=bool(rl.data.shuffle),
            seed=data_seed,
            repeat=True,
            cursor_offset=cursor_offset,
            cursor_shards=cursor_shards,
            **shard,
        )
    validation = getattr(rl, "validation", None)
    if validation is None or not bool(getattr(validation, "enabled", False)):
        return training_env

    validation_items = dataset_items(validation.data)
    validation_identity = dataset_source_identity(validation.data, validation_items)
    if training_items is not None:
        reject_dataset_overlap(dataset_source_identity(rl.data, training_items), validation_identity)
    materialized = materialize_validation_items(
        validation_items,
        max_samples=validation.max_samples,
        seed=int(validation.seed),
    )
    return TrainValidationEnv(
        training_env,
        materialized,
        seed=int(validation.seed),
        max_samples=None,
        stream_fingerprint=dataset_fingerprint(materialized),
        **shard,
    )
