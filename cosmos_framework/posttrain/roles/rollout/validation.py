# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Resolved held-out sampling and rollout-semantics identity."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any, TypeVar

from cosmos_framework.posttrain.data.packer.base import PackerSemanticsIdentity
from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.posttrain.utils.entry import resolve_all
from cosmos_framework.posttrain.utils.identity import (
    canonical_json_sha256 as _canonical_json_sha256,
)
from cosmos_framework.posttrain.utils.identity import json_identity_equal, project_typed_json_value
from cosmos_framework.posttrain.utils.identity import (
    normalize_json_dict as normalize_validation_semantics_identity,
)
from cosmos_framework.posttrain.utils.identity import qualified_type as _qualified_type
from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

_StageResult = TypeVar("_StageResult")


def validation_semantics_equal(first: Any, second: Any) -> bool:
    """Compare identities without collapsing JSON scalar types."""
    return json_identity_equal(first, second, context="validation semantics identity")


def _declared_semantics(value: Any, *, context: str) -> dict[str, Any] | None:
    capability = getattr(value, "validation_semantics_identity", None)
    if not callable(capability):
        return None
    return normalize_validation_semantics_identity(capability(), context=f"{context}.validation_semantics_identity()")


def _tokenizer_semantics_identity(tokenizer: Any) -> dict[str, Any]:
    """Bind the tokenizer artifact and exposed tokenization settings."""
    if tokenizer is None:
        raise TypeError("validation rollout requires a live tokenizer")
    declared = _declared_semantics(tokenizer, context="tokenizer")

    vocab_digest: str | None = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if not isinstance(vocab, dict) or any(
            not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int)
            for token, token_id in vocab.items()
        ):
            raise TypeError("tokenizer.get_vocab() must return dict[str, int] for validation identity")
        vocab_digest = _canonical_json_sha256(vocab, context="tokenizer vocabulary")

    backend_digest: str | None = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    if callable(backend_to_str):
        backend_json = backend_to_str()
        if not isinstance(backend_json, str) or not backend_json:
            raise TypeError("tokenizer.backend_tokenizer.to_str() must return non-empty JSON")
        try:
            json.loads(backend_json)
        except (TypeError, ValueError) as error:
            raise TypeError("tokenizer backend artifact is not valid JSON") from error
        backend_digest = hashlib.sha256(backend_json.encode()).hexdigest()

    if vocab_digest is None and backend_digest is None and declared is None:
        raise TypeError(
            f"tokenizer {_qualified_type(tokenizer)} cannot prove its artifact; expose get_vocab(), "
            "backend_tokenizer.to_str(), or validation_semantics_identity()"
        )

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    normalized_init = (
        None if init_kwargs is None else project_typed_json_value(init_kwargs, context="tokenizer.init_kwargs")
    )
    settings: dict[str, Any] = {}
    for field in (
        "name_or_path",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "split_special_tokens",
        "add_bos_token",
        "add_eos_token",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
        "additional_special_tokens_ids",
        "all_special_ids",
        "special_tokens_map",
        "chat_template",
    ):
        if hasattr(tokenizer, field):
            settings[field] = project_typed_json_value(getattr(tokenizer, field), context=f"tokenizer.{field}")
    return normalize_validation_semantics_identity(
        {
            "class": _qualified_type(tokenizer),
            "vocab_sha256": vocab_digest,
            "backend_json_sha256": backend_digest,
            "init_kwargs_sha256": (
                None
                if normalized_init is None
                else _canonical_json_sha256(normalized_init, context="tokenizer.init_kwargs")
            ),
            "settings": settings,
            "declared": declared,
        },
        context="tokenizer validation semantics identity",
    )


def _packer_semantics_identity(packer: Any) -> dict[str, Any]:
    """Bind packer code and require declarations for configured instances."""
    if packer is None:
        raise TypeError("validation rollout requires a data packer")
    declared = (
        normalize_validation_semantics_identity(
            packer.validation_semantics_identity(), context="packer.validation_semantics_identity()"
        )
        if isinstance(packer, PackerSemanticsIdentity)
        else None
    )
    try:
        source = inspect.getsource(type(packer)).replace("\r\n", "\n")
    except (OSError, TypeError) as error:
        if declared is None:
            raise TypeError(
                f"packer {_qualified_type(packer)} has no inspectable code; implement "
                "PackerSemanticsIdentity.validation_semantics_identity()"
            ) from error
        source = ""
    instance_fields = sorted(
        name for name in vars(packer) if name != "tokenizer" and not name.startswith("_posttrain_")
    )
    if instance_fields and declared is None:
        raise TypeError(
            f"packer {_qualified_type(packer)} has configured fields {instance_fields}; implement "
            "PackerSemanticsIdentity.validation_semantics_identity() to declare their generation semantics"
        )
    return {
        "class": _qualified_type(packer),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        "declared": declared,
    }


def rollout_validation_semantics_identity(
    *,
    backend: str,
    model: Any,
    packer: Any,
    sampling: dict[str, Any],
    backend_config: dict[str, Any],
    model_identifier: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact-resume proof for one rollout backend."""
    if not isinstance(model_identifier, str) or not model_identifier.strip():
        raise TypeError(f"{backend} validation rollout requires a stable non-empty model identifier")
    normalized_backend = normalize_validation_semantics_identity(backend_config, context=f"{backend} backend config")
    normalized_sampling = normalize_validation_semantics_identity(sampling, context=f"{backend} rollout sampling")
    return normalize_validation_semantics_identity(
        {
            "version": 1,
            "backend": backend,
            "backend_class": _qualified_type(model),
            "model_identifier": model_identifier,
            "backend_config_sha256": _canonical_json_sha256(
                normalized_backend,
                context=f"{backend} backend config",
            ),
            "sampling": normalized_sampling,
            "packer": _packer_semantics_identity(packer),
            "tokenizer": _tokenizer_semantics_identity(getattr(packer, "tokenizer", None)),
            "extra": {}
            if extra is None
            else normalize_validation_semantics_identity(extra, context=f"{backend} extra"),
        },
        context=f"{backend} rollout validation semantics identity",
    )


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


def run_validation_stage(stage: Callable[[], _StageResult], context: str) -> _StageResult:
    """Converge rank-local setup before the next validation collective."""
    return run_rank_converged(stage, context)


@contextmanager
def distributed_validation_rng(context: str) -> Iterator[None]:
    """Converge validation RNG snapshot and restoration across rollout ranks."""
    manager = isolated_validation_rng()
    entered = False

    def enter() -> None:
        nonlocal entered
        manager.__enter__()
        entered = True

    entry_error: BaseException | None = None
    try:
        run_validation_stage(enter, f"{context} RNG snapshot")
    except BaseException as error:
        entry_error = error
    if entry_error is not None:
        cleanup_error: BaseException | None = None
        try:
            run_validation_stage(
                lambda: manager.__exit__(None, None, None) if entered else None,
                f"{context} RNG restoration",
            )
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            _add_exception_note(entry_error, f"RNG cleanup after snapshot failure also failed: {cleanup_error}")
        raise entry_error

    body_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        body_error = error

    cleanup_error = None
    try:
        run_validation_stage(
            lambda: manager.__exit__(
                type(body_error) if body_error is not None else None,
                body_error,
                body_error.__traceback__ if body_error is not None else None,
            ),
            f"{context} RNG restoration",
        )
    except BaseException as error:
        cleanup_error = error
    if body_error is not None:
        if cleanup_error is not None:
            _add_exception_note(body_error, f"RNG restoration also failed: {cleanup_error}")
        raise body_error
    if cleanup_error is not None:
        raise cleanup_error


def validation_generation_seed(base_seed: int, sample_id: str, generation_index: int) -> int:
    """Derive a per-completion seed independent of batching and topology."""
    if not sample_id:
        raise ValueError("validation sample_id must be non-empty")
    if generation_index < 0:
        raise ValueError(f"validation generation_index must be >= 0, got {generation_index}")
    payload = f"{int(base_seed)}\0{sample_id}\0{generation_index}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"pt-val-rng").digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def validation_prompt_seed(base_seed: int, sample_id: str) -> int:
    """Derive a stable validation prompt-rendering seed."""
    if not sample_id:
        raise ValueError("validation sample_id must be non-empty")
    payload = f"{int(base_seed)}\0{sample_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"pt-val-prm").digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def seed_validation_rng(seed: int) -> None:
    """Seed validation-owned RNGs without queuing a lazy CUDA seed."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_initialized():
        # Isolation snapshots every initialized CUDA generator.
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_validation_rng(*, enabled: bool = True) -> Iterator[None]:
    """Restore Python, NumPy, CPU torch, and initialized CUDA RNGs."""
    if not enabled:
        yield
        return

    import random

    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()  # [RNG]
    cuda_was_initialized = bool(torch.cuda.is_initialized())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_was_initialized else None
    body_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        body_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        for restore in (
            lambda: random.setstate(python_state),
            lambda: np.random.set_state(numpy_state),
            lambda: torch.random.set_rng_state(torch_state),
        ):
            try:
                restore()
            except BaseException as error:
                cleanup_errors.append(error)
        if cuda_was_initialized:
            try:
                if cuda_states is None:
                    raise AssertionError("initialized CUDA validation RNG snapshot is missing")
                torch.cuda.set_rng_state_all(cuda_states)
            except BaseException as error:
                cleanup_errors.append(error)
        elif torch.cuda.is_initialized():
            cleanup_errors.append(
                RuntimeError("validation prompt rendering initialized CUDA and cannot restore its prior state")
            )
        if cleanup_errors:
            message = "validation rollout RNG restoration failed: " + "; ".join(
                f"{type(error).__name__}: {error}" for error in cleanup_errors
            )
            if body_error is not None:
                _add_exception_note(body_error, message)
            else:
                raise RuntimeError(message) from cleanup_errors[0]


@dataclass(frozen=True)
class ValidationRolloutContext:
    """Held-out sampling configuration pinned to one model step."""

    model_step: int
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = 1
    repetition_penalty: float = 1.0
    n_generation: int = 1
    max_tokens: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.model_step, bool) or not isinstance(self.model_step, int) or self.model_step < 0:
            raise ValueError(f"validation model_step must be >= 0, got {self.model_step}")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0.0
        ):
            raise ValueError(f"validation temperature must be >= 0, got {self.temperature}")
        if self.top_p is not None and (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, Real)
            or not math.isfinite(float(self.top_p))
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError(f"validation top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1
        ):
            raise ValueError(f"validation top_k must be >= 1 when set, got {self.top_k!r}")
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, Real)
            or not math.isfinite(float(self.repetition_penalty))
            or self.repetition_penalty <= 0.0
        ):
            raise ValueError(f"validation repetition_penalty must be > 0, got {self.repetition_penalty}")
        if isinstance(self.n_generation, bool) or not isinstance(self.n_generation, int) or self.n_generation < 1:
            raise ValueError(f"validation n_generation must be >= 1, got {self.n_generation}")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1
        ):
            raise ValueError(f"validation max_tokens must be >= 1 when set, got {self.max_tokens}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"validation seed must be an integer, got {self.seed!r}")

    def sampling_identity(self) -> dict[str, int | float | None]:
        """Return sampling semantics without the per-run model step."""
        return {
            "temperature": float(self.temperature),
            "top_p": None if self.top_p is None else float(self.top_p),
            "top_k": None if self.top_k is None else int(self.top_k),
            "repetition_penalty": float(self.repetition_penalty),
            "n_generation": int(self.n_generation),
            "max_tokens": None if self.max_tokens is None else int(self.max_tokens),
            "seed": int(self.seed),
        }


def resolved_validation_sampling(
    validation: Any,
    *,
    effective_max_tokens: int,
) -> dict[str, int | float | None]:
    """Resolve backend-independent validation sampling values."""
    sampling = validation.sampling
    return {
        "temperature": float(sampling.temperature),
        "top_p": None if sampling.top_p is None else float(sampling.top_p),
        "top_k": None if sampling.top_k is None else int(sampling.top_k),
        "repetition_penalty": float(sampling.repetition_penalty),
        "n_generation": int(sampling.n_generation),
        "max_tokens": int(effective_max_tokens),
    }


def build_validation_context(
    validation: Any,
    model_step: int,
    *,
    effective_max_tokens: int,
) -> ValidationRolloutContext:
    """Map resolved validation config onto the rollout request contract."""
    sampling = resolved_validation_sampling(validation, effective_max_tokens=effective_max_tokens)
    return ValidationRolloutContext(
        model_step=model_step,
        temperature=float(sampling["temperature"]),
        top_p=None if sampling["top_p"] is None else float(sampling["top_p"]),
        top_k=None if sampling["top_k"] is None else int(sampling["top_k"]),
        repetition_penalty=float(sampling["repetition_penalty"]),
        n_generation=int(sampling["n_generation"]),
        max_tokens=int(sampling["max_tokens"]),
        seed=int(validation.seed),
    )


def resolve_rollout_validation_semantics(rollout: Any) -> dict[str, Any]:
    """Resolve and exact-compare validation semantics across every rollout cell."""
    capability = getattr(rollout, "validation_semantics_identity", None)
    if not callable(capability):
        raise TypeError("validation rollout must implement validation_semantics_identity()")

    def prepare() -> list[dict[str, Any]]:
        identities = [
            normalize_validation_semantics_identity(
                identity,
                context=f"rollout cell {index} validation semantics identity",
            )
            for index, identity in enumerate(resolve_all(capability()))
        ]
        if not identities:
            raise RuntimeError("rollout validation semantics returned no cells")
        if any(not validation_semantics_equal(identity, identities[0]) for identity in identities[1:]):
            raise RuntimeError(f"rollout validation semantics differ across cells: {identities!r}")
        return identities

    identities = run_rank_converged(prepare, "GRPO rollout validation identity preparation")
    by_rank = gather_rank_states(
        identities[0],
        context="GRPO rollout validation semantics preparation",
    )
    if any(not validation_semantics_equal(identity, identities[0]) for identity in by_rank.values()):
        raise RuntimeError(f"rollout validation semantics differ across local ranks: {by_rank!r}")
    return identities[0]
