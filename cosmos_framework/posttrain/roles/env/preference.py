# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""One-step preference-pair environments for offline DPO.

Storage and preprocessing stay in ``data/datasets``; this module converts each
``(prompt, winner, loser)`` item into a typed preference ``Trajectory``.
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

from cosmos_framework.posttrain.config.rl import (
    CURSOR_RESUME_AUTO,
    CURSOR_RESUME_MODES,
    CURSOR_RESUME_OFF,
)

# Re-export the shared cursor error for compatibility with preference checkpoint callers.
from cosmos_framework.posttrain.roles.env.cursor import CursorStateUnavailable, IndexedCursor
from cosmos_framework.posttrain.schema import ObsBundle, Preference, Status, Trajectory, Transition
from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed
from cosmos_framework.posttrain.utils.log import logger

__all__ = [
    "CursorStateUnavailable",
    "DPOPreferenceEnv",
    "DatasetPreferenceEnv",
    "PreferenceSourceIdentity",
    "PreferenceValidationBatch",
    "build_preference_env",
]


@dataclass(frozen=True)
class PreferenceValidationBatch:
    """One physical FSDP validation batch plus its real logical pair count."""

    trajectories: list[Trajectory]
    logical_count: int
    batch_index: int


@dataclass(frozen=True)
class PreferenceSourceIdentity:
    """Portable preference-source identity used for train/validation leakage checks."""

    raw_fingerprint: str
    semantic_fingerprint: str
    membership_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.raw_fingerprint, str) or not isinstance(self.semantic_fingerprint, str):
            raise TypeError("preference source fingerprints must be strings")
        if not self.raw_fingerprint or not self.semantic_fingerprint:
            raise ValueError("preference source fingerprints must be non-empty")
        if not isinstance(self.membership_ids, tuple):
            raise TypeError("preference membership identities must be a tuple")
        if any(not isinstance(value, str) or not value for value in self.membership_ids):
            raise TypeError("preference membership identities must be non-empty strings")


class _CustomPreferenceDataset:
    """Add framework-owned cursor and membership identities to canonical custom triples."""

    def __init__(self, dataset: Any, source_identity: str) -> None:
        self._dataset: Any = dataset
        self._source_identity: str = source_identity

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[index]

    def __getattr__(self, name: str) -> Any:
        """Forward only the optional paired dataset cursor-state protocol."""
        if name in {"cursor_state_dict", "load_cursor_state_dict"}:
            return getattr(self._dataset, name)
        raise AttributeError(name)

    def cursor_fingerprint(self) -> str:
        """Return deterministic factory identity, refined by an optional dataset hook."""
        fingerprint = getattr(self._dataset, "cursor_fingerprint", None)
        from cosmos_framework.posttrain.roles.env.identity import stable_value_digest

        refinement = str(fingerprint()) if callable(fingerprint) else None
        identity = stable_value_digest((self._source_identity, len(self._dataset), refinement))
        return f"custom-preference-cursor-sha256:{hashlib.sha256(identity).hexdigest()}"

    def membership_ids(self) -> tuple[str, ...]:
        """Return dataset-provided metadata-only aliases for leakage rejection."""
        membership = getattr(self._dataset, "membership_ids", None)
        if not callable(membership):
            raise TypeError(
                f"custom preference dataset {type(self._dataset).__qualname__} must expose membership_ids() "
                "for train/validation leakage checks; the hook must return metadata-only aliases "
                "without reading or decoding media"
            )
        return tuple(membership())


def _pair_trajectory(
    idx: int,
    prompt: Any,
    winner: dict,
    loser: dict,
    *,
    validation_weight: float | None = None,
) -> Trajectory:
    """Build a one-transition trajectory with a validated preference target."""
    tr = Transition(
        obs=ObsBundle(representations={"prompt": prompt}, modality_tags=("text",)),
        targets=Preference(winner=winner, loser=loser),
        done=True,
    )
    if validation_weight is not None:
        tr.algo_extra["validation_weight"] = float(validation_weight)
    return Trajectory(transitions=[tr], id=f"pref-{idx}", status=Status.COMPLETED)


class DatasetPreferenceEnv:
    """1-step Env over a preference dataset whose items are
    `(prompt, winner_dict, loser_dict)`. `step()` ends the episode (no env-native
    reward — the DPO reward is implicit in the objective).

    **Rank-aware data sharding (aligns with cosmos-rl).** Under FSDP-as-data-parallel
    each rank must train on a DIFFERENT, non-overlapping slice of the preference pairs
    (cosmos-rl uses `torch.utils.data.distributed.DistributedSampler`; FSDP then
    reduce-scatters the gradients). For a map-style dataset (`__len__`+`__getitem__`,
    e.g. `DPOPreferenceDataset`) this Env reproduces that: per epoch it shuffles the
    indices (seeded, identical across ranks) and takes the strided shard
    `indices[rank::world]` — exactly DistributedSampler's split — reshuffling each epoch.
    Per-rank noise stays independent (drawn in the objective), matching cosmos-rl.

    The traversal itself is `roles/env/cursor.py::IndexedCursor`, shared with `DatasetEnv`; this
    class only turns an item into a preference `Trajectory`. A non-indexable (iterable-only)
    dataset is still ACCEPTED at construction — `cursor_resume` governs what `state_dict` does with
    it — but drawing from one now RAISES. It used to be cycled without sharding, so every rank saw
    identical pairs and data parallelism silently duplicated the batch."""

    def __init__(
        self,
        dataset: Any,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        cursor_resume: str = CURSOR_RESUME_AUTO,
        source_fingerprint: str = "",
        raw_source_fingerprint: str = "",
    ) -> None:
        import torch.distributed as dist

        if cursor_resume not in CURSOR_RESUME_MODES:
            raise ValueError(f"cursor_resume must be one of {CURSOR_RESUME_MODES}, got {cursor_resume!r}")
        self._dataset = dataset
        self._rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self._world = world_size if world_size is not None else (dist.get_world_size() if dist.is_initialized() else 1)
        self._cursor_resume = cursor_resume
        self._indexable = hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
        fingerprint = getattr(dataset, "cursor_fingerprint", None)
        fallback_fingerprint = (
            str(fingerprint())
            if callable(fingerprint)
            else f"{type(dataset).__module__}.{type(dataset).__qualname__}:"
            f"{len(dataset) if self._indexable else 'iterable'}"
        )
        self._source_fingerprint = source_fingerprint or fallback_fingerprint
        self._raw_source_fingerprint = raw_source_fingerprint or self._source_fingerprint
        self._cursor_warnings: set[str] = set()
        self._loaded_checkpoint_provenance: dict[str, int | str] | None = None
        # RANK, not a replica index: DPO is offline, so the FSDP rank axis is its only
        # data-parallel axis and nothing downstream divides it. (GRPO passes a replica index
        # instead — see `roles/env/cursor.py`.)
        self._cursor = (
            IndexedCursor(
                dataset,
                shard_index=self._rank,
                shard_count=self._world,
                shuffle=shuffle,
                seed=seed,
                cursor_resume=cursor_resume,
                label="DatasetPreferenceEnv",  # preserves this class's existing error strings
                coord_names=("rank", "world_size"),  # DPO's axis, and what its messages/tests say
            )
            if self._indexable
            else None
        )

    def _warn_cursor_once(self, message: str) -> None:
        if message in self._cursor_warnings:
            return
        self._cursor_warnings.add(message)
        logger.warning(message)

    def set_loaded_checkpoint_provenance(self, provenance: dict[str, int | str]) -> None:
        """Bind the exact checkpoint generation from which this env restored its cursor."""
        self._loaded_checkpoint_provenance = dict(provenance)

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        """Return the exact directory generation observed during cursor setup, if any."""
        return None if self._loaded_checkpoint_provenance is None else dict(self._loaded_checkpoint_provenance)

    def membership_ids(self) -> tuple[str, ...]:
        """Flattened semantic preference-pair aliases for leakage rejection.

        One selected row may expose multiple identities (for example its canonical media locator
        and a manifest content digest). The dataset owns row coverage; this boundary validates only
        the flattened alias representation rather than assuming one identity per row.
        """
        membership = getattr(self._dataset, "membership_ids", None)
        if not callable(membership):
            raise TypeError(
                f"preference dataset {type(self._dataset).__qualname__} does not expose semantic membership identity"
            )
        values = tuple(membership())
        if len(self._dataset) > 0 and not values:
            raise RuntimeError("preference membership identity aliases are empty for a non-empty selected dataset")
        if any(not isinstance(value, str) or not value for value in values):
            raise TypeError("preference membership identities must be non-empty strings")
        return values

    def preference_source_identity(self) -> PreferenceSourceIdentity:
        """Return the source identity without reading or decoding media payloads."""
        return PreferenceSourceIdentity(
            raw_fingerprint=self._raw_source_fingerprint,
            semantic_fingerprint=self._source_fingerprint,
            membership_ids=self.membership_ids(),
        )

    def reject_preference_source_overlap(self, held_out: PreferenceSourceIdentity) -> None:
        """Reject a held-out identity that overlaps this training source."""
        if not isinstance(held_out, PreferenceSourceIdentity):
            raise TypeError(f"held-out preference identity must be PreferenceSourceIdentity, got {held_out!r}")
        overlap = set(self.membership_ids()).intersection(held_out.membership_ids)
        reasons: list[str] = []
        if self._raw_source_fingerprint == held_out.raw_fingerprint:
            reasons.append("raw source fingerprint matches")
        if self._source_fingerprint == held_out.semantic_fingerprint:
            reasons.append("semantic source fingerprint matches")
        if overlap:
            reasons.append(f"{len(overlap)} held-out pair identity/identities are present in training")
        if reasons:
            raise ValueError("training and validation preference sources overlap semantically: " + "; ".join(reasons))

    def _cursor_capability_available(
        self,
        unavailable: CursorStateUnavailable | None,
        *,
        context: str,
        fallback: str,
    ) -> bool:
        """Converge optional cursor support into one all-rank policy decision."""
        from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

        unavailable_by_rank = gather_rank_states(
            None if unavailable is None else str(unavailable),
            context=f"{context} preparation",
        )
        messages = {rank: message for rank, message in unavailable_by_rank.items() if message is not None}
        if not messages:
            return True
        if len(messages) != len(unavailable_by_rank):
            raise RuntimeError(f"{context} differs across ranks: unavailable on ranks {sorted(messages)}")
        message = str(next(iter(messages.values())))
        if self._cursor_resume == CURSOR_RESUME_AUTO:
            self._warn_cursor_once(f"{message}; {fallback}")
            return False
        raise CursorStateUnavailable(message)

    def state_dict(self) -> dict[str, Any] | None:
        """Return an exact local cursor according to the configured resume policy.

        ``auto`` degrades unsupported sources to a checkpoint without a data component;
        ``strict`` fails; ``off`` omits the component explicitly. Other snapshot errors
        remain failures in every mode.
        """
        if self._cursor_resume == CURSOR_RESUME_OFF:
            return None
        state: dict[str, Any] | None = None
        unavailable: CursorStateUnavailable | None = None
        snapshot_error: BaseException | None = None
        try:
            state = self._local_state_dict()
        except CursorStateUnavailable as error:
            unavailable = error
        except BaseException as error:  # noqa: BLE001 — converted to a rank-consistent checkpoint outcome
            snapshot_error = error
        raise_if_any_rank_failed(snapshot_error, "preference source snapshot")
        if not self._cursor_capability_available(
            unavailable,
            context="preference source cursor capability",
            fallback="checkpoint will omit the data cursor and resume non-exactly",
        ):
            return None
        if state is None:
            raise AssertionError("preference source snapshot converged without state")
        return state

    def _local_state_dict(self) -> dict[str, Any]:
        """Snapshot the exact next-item position for a map-style preference source.

        Delegates to `IndexedCursor`. Iterable-only sources have no stable item identity, so
        they raise here and the policy wrapper above turns that into the configured outcome
        (`auto`/`off` omit the cursor, `strict` fails).
        """
        if self._cursor is None:
            raise CursorStateUnavailable("DatasetPreferenceEnv cursor persistence requires a map-style dataset")
        state = self._cursor.state_dict()
        if state is None:
            # RuntimeError, not AssertionError: this is an invariant breach, not a failed test
            # assertion, and the caller converts anything raised here into a snapshot_error that
            # `raise_if_any_rank_failed` propagates. Unreachable today -- the cursor is built with
            # this same `cursor_resume`, so it declines only when 'off', which `state_dict` already
            # returned for -- but it guards the pairing, so it should say what it means.
            raise RuntimeError("cursor declined a snapshot outside cursor_resume='off'")
        return state

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        """Restore locally, then converge incompatibilities before policy setup."""
        if self._cursor_resume == CURSOR_RESUME_OFF:
            self._warn_cursor_once(
                "data cursor restore is disabled by cursor_resume='off'; source restarts non-exactly"
            )
            return False
        unavailable: CursorStateUnavailable | None = None
        restore_error: BaseException | None = None
        try:
            self._load_state_dict_local(state)
        except CursorStateUnavailable as error:
            unavailable = error
        except BaseException as error:  # noqa: BLE001 — converted to a rank-consistent restore outcome
            restore_error = error
        raise_if_any_rank_failed(restore_error, "preference source restore")
        if not self._cursor_capability_available(
            unavailable,
            context="preference source cursor capability",
            fallback="source restarts non-exactly instead of restoring its cursor",
        ):
            return False
        return True

    def _load_state_dict_local(self, state: dict[str, Any]) -> None:
        """Restore a cursor snapshot after validating source and sharding identity."""
        if self._cursor is None:
            raise CursorStateUnavailable("DatasetPreferenceEnv cursor persistence requires a map-style dataset")
        self._cursor.load_state_dict(state)

    def _next_triple(self) -> tuple[Any, Any, Any]:
        """The next `(prompt, winner, loser)` for this rank.

        Raises:
            CursorStateUnavailable: the source is iterable-only. It used to be cycled here
                WITHOUT sharding, so every rank saw identical pairs and data-parallel training
                silently duplicated its batch. That is refused rather than repeated; no shipped
                dataset takes this path (`SyntheticPreferenceDataset` subclasses
                `DPOPreferenceDataset` and inherits `__getitem__`).
        """
        if self._cursor is None:
            raise CursorStateUnavailable(
                "DatasetPreferenceEnv requires a map-style dataset (__len__ + __getitem__): an "
                "iterable-only source cannot be rank-sharded, and cycling it unsharded would give "
                "every rank the same pairs."
            )
        return self._dataset[self._cursor.next_index()]

    def reset(self) -> Trajectory:
        prompt, winner, loser = self._next_triple()
        # consumed-1: the draw above already advanced the cursor, and the index identifies the
        # pair just drawn. Sourcing it from the cursor is what makes it survive a resume.
        return _pair_trajectory(self._cursor.consumed - 1, prompt, winner, loser)

    def step(self, action: Any) -> tuple[None, None, bool, dict[str, Any]]:
        return None, None, True, {}

    def get_samples(self, n: int) -> list[Trajectory]:
        return [self.reset() for _ in range(n)]

    @property
    def source_fingerprint(self) -> str:
        """Stable held-out source identity supplied by setup, with a local fallback."""
        return self._source_fingerprint

    def validation_batches(
        self,
        batch_size: int,
        max_samples: int | None = None,
    ) -> Iterator[PreferenceValidationBatch]:
        """Build a finite, deterministic, masked-padded shard without touching the train cursor.

        Every rank receives the same physical length and batch schedule. Global padding slots repeat
        a real row so decoding/model structure remains valid, but carry ``validation_weight=0`` and
        therefore contribute neither metrics nor ``logical_count``.
        """
        self.validation_batch_count(batch_size, max_samples)
        logical_size = self._validation_logical_size(max_samples)
        local_size = (logical_size + self._world - 1) // self._world
        padded_size = local_size * self._world
        local_slots = list(range(self._rank, padded_size, self._world))
        save = getattr(self._dataset, "cursor_state_dict", None)
        load = getattr(self._dataset, "load_cursor_state_dict", None)
        if callable(save) != callable(load):
            raise ValueError("finite validation requires both dataset cursor save and restore hooks, or neither")
        dataset_state = copy.deepcopy(save()) if callable(save) else None
        try:
            for batch_index, start in enumerate(range(0, local_size, batch_size)):
                trajectories: list[Trajectory] = []
                logical_count = 0
                for slot in local_slots[start : start + batch_size]:
                    is_real = slot < logical_size
                    dataset_index = slot if is_real else slot % logical_size
                    prompt, winner, loser = self._dataset[dataset_index]
                    trajectories.append(
                        _pair_trajectory(
                            dataset_index,
                            prompt,
                            winner,
                            loser,
                            validation_weight=1.0 if is_real else 0.0,
                        )
                    )
                    logical_count += int(is_real)
                yield PreferenceValidationBatch(
                    trajectories=trajectories,
                    logical_count=logical_count,
                    batch_index=batch_index,
                )
        finally:
            if callable(load):
                load(copy.deepcopy(dataset_state))

    def validation_batch_count(self, batch_size: int, max_samples: int | None = None) -> int:
        """Physical batch count on every rank for one finite validation pass."""
        if batch_size < 1:
            raise ValueError(f"validation batch_size must be >= 1, got {batch_size}")
        logical_size = self._validation_logical_size(max_samples)
        local_size = (logical_size + self._world - 1) // self._world
        return (local_size + batch_size - 1) // batch_size

    def _validation_logical_size(self, max_samples: int | None) -> int:
        """Validate and return the globally real held-out pair inventory."""
        if self._cursor is None:
            raise CursorStateUnavailable("finite validation requires a map-style preference dataset")
        dataset_size = len(self._dataset)
        logical_size = dataset_size if max_samples is None else min(dataset_size, max_samples)
        if logical_size < 1:
            raise ValueError("preference validation source is empty")
        return logical_size


class DPOPreferenceEnv(DatasetPreferenceEnv):
    """`DatasetPreferenceEnv` over the real `DPOPreferenceDataset` (JSONL video pairs).
    Convenience constructor so experiments name a JSONL + resolution instead of wiring
    the dataset by hand."""

    def __init__(
        self,
        jsonl_path: str,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        cursor_resume: str = CURSOR_RESUME_AUTO,
        source_fingerprint: str = "",
        raw_source_fingerprint: str = "",
        **dataset_kwargs: Any,
    ) -> None:
        from cosmos_framework.posttrain.data.datasets.dpo_preference import DPOPreferenceDataset

        # Env-level args (rank/world/shuffle/seed) drive DistributedSampler-style sharding;
        # the rest (resolution, media, num_frames, cfg_dropout_rate, …) configure the dataset.
        super().__init__(
            DPOPreferenceDataset(jsonl_path, **dataset_kwargs),
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=seed,
            cursor_resume=cursor_resume,
            source_fingerprint=source_fingerprint,
            raw_source_fingerprint=raw_source_fingerprint,
        )


def _preference_source_fingerprint(data: Any, seed: int) -> str:
    """Return the raw source identity used by train/validation separation."""
    factory = str(getattr(data, "factory", ""))
    if factory:
        from cosmos_framework.posttrain.roles.env.identity import stable_value_digest

        identity = stable_value_digest(
            {
                "factory": factory,
                "factory_kwargs": dict(getattr(data, "factory_kwargs", {})),
            }
        )
        return f"custom-preference-sha256:{hashlib.sha256(identity).hexdigest()}"
    if int(data.synthetic_pairs) > 0:
        return f"synthetic-preference-seed:{seed}"
    if not data.jsonl_path:
        return "unconfigured-preference-source"
    digest = hashlib.sha256()
    path = os.path.realpath(os.path.expanduser(str(data.jsonl_path)))
    with open(path, "rb") as manifest:
        for chunk in iter(lambda: manifest.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"preference-jsonl-sha256:{digest.hexdigest()}"


def build_preference_env(
    data: Any,
    seed: int,
    *,
    include_source_identity: bool = False,
) -> DatasetPreferenceEnv:
    """Build one rank-sharded preference environment from its typed data config."""
    cursor_resume = data.cursor_resume
    factory = str(getattr(data, "factory", ""))
    if not factory and not data.jsonl_path and not data.tar_path and int(data.synthetic_pairs) <= 0:
        raise ValueError(
            "no preference data configured: set rl.data.jsonl_path (real pairs), "
            "rl.data.tar_path (tar offsets), rl.data.synthetic_pairs>0 (toy data), or "
            "rl.data.factory (custom map-style source)."
        )
    raw_fingerprint = _preference_source_fingerprint(data, seed) if include_source_identity else ""
    if factory:
        from cosmos_framework.posttrain.data.datasets.factory import build_custom_dataset

        custom_fingerprint = raw_fingerprint or _preference_source_fingerprint(data, seed)
        dataset = _CustomPreferenceDataset(
            build_custom_dataset(factory, getattr(data, "factory_kwargs", {})),
            custom_fingerprint,
        )
    elif int(data.synthetic_pairs) > 0:
        from cosmos_framework.posttrain.data.datasets.synthetic_preference import SyntheticPreferenceDataset

        dataset = SyntheticPreferenceDataset(
            n_pairs=int(data.synthetic_pairs),
            resolution=data.resolution,
            num_frames=data.num_frames,
            media=data.media,
            cfg_dropout_rate=data.cfg_dropout_rate,
            seed=seed,
        )
    elif data.tar_path:
        from cosmos_framework.posttrain.data.datasets.tar_dpo_preference import TarDPOPreferenceDataset

        dataset = TarDPOPreferenceDataset(
            data.jsonl_path,
            tar_path=data.tar_path,
            resolution=data.resolution,
            num_frames=data.num_frames,
            media=data.media,
            caption_key=data.caption_key,
            min_score_gap=data.min_score_gap,
            min_wm_gap=data.min_wm_gap,
            min_align_gap=data.min_align_gap,
            cfg_dropout_rate=data.cfg_dropout_rate,
            seed=seed,
        )
    else:
        return DPOPreferenceEnv(
            data.jsonl_path,
            resolution=data.resolution,
            num_frames=data.num_frames,
            media=data.media,
            cfg_dropout_rate=data.cfg_dropout_rate,
            shuffle=data.shuffle,
            seed=seed,
            cursor_resume=cursor_resume,
            raw_source_fingerprint=raw_fingerprint,
        )
    return DatasetPreferenceEnv(
        dataset,
        shuffle=data.shuffle,
        seed=seed,
        cursor_resume=cursor_resume,
        raw_source_fingerprint=raw_fingerprint,
    )


def validate_preference_sources(data: Any, policy: Any) -> None:
    """Reject semantic overlap between bound training and held-out sources."""
    from cosmos_framework.posttrain.utils.dist import run_rank_converged
    from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical

    def validate() -> None:
        held_out = resolve_identical(
            policy.validation_source_identity(),
            operation="policy.validation_source_identity",
        )
        resolve_all(data.reject_preference_source_overlap(held_out))

    run_rank_converged(validate, "preference train/validation source separation")
