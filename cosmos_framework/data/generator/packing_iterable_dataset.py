# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Abstract base class for pool-based token-budget bin-packing over multiple datasets.

Extracted from ``cosmos_framework.data.generator.reasoner.joint_dataset_dynamic_batch_webloader``
so that both the VLM and VFM internal dataloaders can share a single packing implementation.

Usage
-----
Subclass and implement ``compute_sample_tokens(sample) -> int``.
Optionally override ``collate_batch(samples) -> Any`` for custom collation.

    class MyPacker(PackingIterableDataset):
        def compute_sample_tokens(self, sample):
            return len(sample["input_ids"])
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from enum import Enum
from typing import Any

import torch

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils import log

# The collate (datasets/reasoner/collate_fn.py) right-pads every row up to a multiple of this for FP8.
# The packer rounds its padded-cost / over_budget accounting to the SAME multiple so the two cannot
# diverge: a mismatch would silently make the budget gate and the over_budget telemetry disagree with
# the real batch shape. There is no per-run knob -- this is fixed by the FP8 kernel requirement.
FP8_PAD_MULTIPLE = 16


def round_up_to_multiple(value: int, multiple: int = FP8_PAD_MULTIPLE) -> int:
    """Round ``value`` up to a positive alignment ``multiple``."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return (value + multiple - 1) // multiple * multiple


class Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class SingletonCause(Enum):
    """Why a packed batch ended up as a single sample.

    Surfaced as sync-free packing telemetry (singleton
    root-cause counters) so a high ``singleton_rate`` can be attributed to the right lever: budget vs
    ``pool_size`` vs interleaving vs config cap.
    """

    NONE = 0  # not a singleton (k > 1)
    LONG_THRESHOLD = 1  # seed length >= long_threshold -> forced singleton
    BUDGET_CLIFF = 2  # seed alone >= per-batch token budget (_max_tokens)
    FLOP_BUDGET = 3  # seed alone >= FLOP budget (FLOP path only)
    MAX_BATCH_CAP = 4  # max_batch_size == 1 -> singleton by configuration
    NO_CANDIDATE = 5  # no same-modality candidate fit the budget


class PackingIterableDataset(torch.utils.data.IterableDataset, ABC):
    """Pool-based greedy bin-packing IterableDataset.

    Maintains a pool of ``pool_size`` samples and assembles batches by
    greedily selecting candidates that fit within the token budget
    ``max_tokens``.  Subclasses supply two hooks:

    * ``compute_sample_tokens(sample)`` — token cost of one sample (abstract).
    * ``collate_batch(samples)`` — assemble a packed list into a batch
      (default: identity, returns the list unchanged).

    Parameters
    ----------
    datasets_cfg:
        Mapping ``{name: {"dataset": <iterable>, "ratio": <float>}}``.
        The *dataset* value may be a Hydra lazy config, an already-constructed
        ``IterableDataset``, or a plain ``DataLoader`` (its ``.dataset`` is
        unwrapped automatically).
    max_tokens:
        Token budget per batch (padded cost = ``cur_max_len * batch_size``).
    pool_size:
        Number of samples to buffer before selecting a batch.
    max_batch_size:
        Hard cap on items per batch (0 or None = no cap).
    long_threshold:
        Samples with token count ``>= long_threshold`` are emitted as
        singletons regardless of budget.
    batching_strategy:
        ``"prefer_closest"`` (default) or ``"prefer_first"``.
    """

    def __init__(
        self,
        datasets_cfg: dict[str, dict[str, object]],
        max_tokens: int,
        pool_size: int,
        max_batch_size: int,
        long_threshold: int,
        batching_strategy: str,
        seed: int | None = None,
        legacy_budget_gate: bool | None = None,
        flat_budget: bool | None = None,
        true_packing: bool | None = None,
    ) -> None:
        super().__init__()

        assert batching_strategy in ("prefer_first", "prefer_closest"), (
            f"batching_strategy must be 'prefer_first' or 'prefer_closest', got {batching_strategy!r}"
        )

        self.max_tokens = max_tokens
        self.pool_size = pool_size
        self.long_threshold = long_threshold
        self.max_batch_size = max_batch_size
        self.batching_strategy = batching_strategy

        # --- Packing-optimization knobs ---
        # Each resolves from its constructor arg when given, else an env var (same A/B-by-env
        # pattern as COSMOS_VLM_TELEMETRY), else a safe default. This lets the validation harness
        # flip behaviour per-arm via extra_env_vars without editing the experiment config.
        # Scope note: self.seed seeds ONLY the packer's source-interleaving RNG (see
        # _ensure_worker_init / _get_next_sample -- i.e. WHICH sub-dataset each next sample is drawn
        # from). It does NOT seed the shard distributor; that is seeded independently by
        # data_setting.distributor_seed (also 1993 by default and likewise deterministic per
        # (seed, rank, worker, epoch)). COSMOS_PACK_SEED does not reach the distributor unless that
        # config value is itself wired to it.
        self.seed: int = int(os.environ.get("COSMOS_PACK_SEED", "1993")) if seed is None else int(seed)
        # Budget gate: the correct budget gate evaluates the budget at the POST-admit max length.
        # The legacy (buggy) gate evaluated it at the pre-admit max, admitting candidates that push
        # the batch across the half-budget cliff (up to ~2x over budget). Default = fixed; the
        # legacy toggle reproduces the bug so the over_budget counter can prove the fix.
        self.legacy_budget_gate: bool = (
            os.environ.get("COSMOS_PACK_LEGACY_BUDGET_GATE", "0") == "1"
            if legacy_budget_gate is None
            else bool(legacy_budget_gate)
        )
        # Flat budget: replaces the discontinuous half-budget step. The legacy budget gives the
        # full max_tokens below l_max=1000 and halves it (max_tokens//2) at/above -- a step that
        # straddling batches can exceed or underfill. The half-budget was an OOM guard, but an
        # on-cluster A/B showed it guards a non-problem:
        # peak memory is set by the longest forced singleton and is invariant across budget policies,
        # so halving the budget only throttles packing density and costs throughput. The flat budget
        # B(l)=max_tokens removes the cliff and measured +8.6% useful_tps at an identical peak_mem.
        # Off by default (the step stays the default so the correctness gates are untouched); the
        # validation harness flips it per-arm via COSMOS_PACK_FLAT_BUDGET.
        self.flat_budget: bool = (
            os.environ.get("COSMOS_PACK_FLAT_BUDGET", "0") == "1" if flat_budget is None else bool(flat_budget)
        )
        if self.flat_budget and self.legacy_budget_gate:
            raise ValueError(
                "flat_budget and legacy_budget_gate cannot both be enabled: the combination is neither the "
                "historical unrounded gate nor the fixed flat-budget policy"
            )
        # True packing: when on, the collate concatenates the pack
        # into ONE B=1 row of sum(L_i) tokens with block-diagonal varlen attention -- there is NO
        # per-sample row padding, so the batch cost is round16(sum(L_i)) instead of the padded
        # k*round16(max L_i). The budget gate, candidate selection, and over_budget telemetry all
        # charge this packed cost so the packer fills sum(L_i) up to max_tokens (the throughput win).
        # Off by default (= shipped padded batching); the harness flips it per-arm via
        # COSMOS_PACK_TRUE_PACKING. Intended to run with flat_budget (the harness sets both).
        self.true_packing: bool = (
            os.environ.get("COSMOS_PACK_TRUE_PACKING", "0") == "1" if true_packing is None else bool(true_packing)
        )
        if self.true_packing and self.legacy_budget_gate:
            raise ValueError(
                "true_packing and legacy_budget_gate cannot both be enabled: the historical gate "
                "models padded rows and is not a valid true-packing policy"
            )

        self._pool: deque[dict[str, Any]] = deque()
        self._dataset_names: list[str] = []
        self._ratios: list[float] = []
        self._datasets: list[torch.utils.data.IterableDataset] = []

        for name, cfg in datasets_cfg.items():
            assert {"ratio", "dataset"} <= cfg.keys(), (
                f"Each entry must have 'dataset' and 'ratio' keys: {name} -> {cfg.keys()}"
            )
            ratio = cfg["ratio"]
            if ratio == 0:
                log.info(f"Skipping dataset {name} with ratio {ratio}")
                continue
            dataset_cfg = cfg["dataset"]

            ds = (
                instantiate(dataset_cfg)
                if not isinstance(dataset_cfg, (torch.utils.data.IterableDataset, torch.utils.data.DataLoader))
                else dataset_cfg
            )
            if isinstance(ds, torch.utils.data.DataLoader):
                ds = ds.dataset
            if hasattr(ds, "build_dataset") and callable(getattr(ds, "build_dataset")):
                ds = ds.build_dataset()

            assert isinstance(ds, torch.utils.data.IterableDataset), (
                f"Expected an IterableDataset, got {type(ds)} for {name}"
            )

            self._dataset_names.append(name)
            self._ratios.append(float(ratio))
            self._datasets.append(ds)
            log.info(f"Added dataset {name} with ratio {ratio}")

        log.info(f"added data: {list(datasets_cfg.keys())}")
        assert len(self._datasets) > 0, "No datasets added"
        self._data_len: int = sum(int(getattr(ds, "total_images", 0)) for ds in self._datasets)
        if self._data_len == 0:
            self._data_len = 10**12
        # Determinism: build per-worker iterators + RNG lazily in __iter__ (NOT here in __init__, which runs
        # in the MAIN process before the DataLoader forks workers). Eager construction here risks
        # all forked workers sharing iterator / RNG state. None until _ensure_worker_init().
        self.iterators: list[Iterator[Any]] | None = None
        self._rng: random.Random | None = None
        self._worker_epoch: int | None = None
        # Stats cache: per-batch caches keyed by id(sample) so compute_sample_tokens / FLOPs are not
        # recomputed for every candidate on every admit. Cleared at the start of each batch. The
        # FLOP cache stores the subclass's per-sample (total_tokens, visual_tokens, num_patches)
        # tuple (see JointDatasetDynamicBatchingWebLoader._flop_stats).
        self._tok_cache: dict[int, int] = {}
        self._flop_cache: dict[int, tuple[int, int, tuple[int, ...]]] = {}

        # Startup visibility: log the RESOLVED packing knobs once (this __init__ runs in the main
        # process before the DataLoader forks workers, so it logs exactly once per run). Warn loudly
        # when a NON-default behaviour is active so a flag left enabled via env in a real run is
        # obvious in the logs rather than silently changing packing density / admission.
        log.info(
            f"PackingIterableDataset knobs: seed={self.seed} legacy_budget_gate={self.legacy_budget_gate} "
            f"flat_budget={self.flat_budget} true_packing={self.true_packing} max_tokens={self.max_tokens} "
            f"max_batch_size={self.max_batch_size} "
            f"pool_size={self.pool_size} long_threshold={self.long_threshold} "
            f"batching_strategy={self.batching_strategy!r}"
        )
        if self.true_packing:
            log.warning(
                "COSMOS_PACK_TRUE_PACKING is ON: each pack is concatenated into ONE B=1 varlen row "
                "(block-diagonal attention, no per-sample row padding) instead of padded rows. This is a "
                "non-default layout -- enable intentionally."
            )
        if self.legacy_budget_gate:
            log.warning(
                "COSMOS_PACK_LEGACY_BUDGET_GATE is ON: reproducing the PRE-FIX packer (budget gate at "
                "the pre-admit max + unrounded cost). This admits over-budget batches and is intended "
                "for A/B reproduction only, NOT production."
            )
        if self.flat_budget:
            log.warning(
                "COSMOS_PACK_FLAT_BUDGET is ON: flat budget B(l)=max_tokens replacing the default "
                "half-budget step. This is non-default packing behaviour -- enable intentionally."
            )
        if self.true_packing:
            log.warning(
                "COSMOS_PACK_TRUE_PACKING is ON: samples are concatenated into one block-diagonal "
                "attention row. This is non-default packing behaviour -- enable intentionally."
            )

    # ------------------------------------------------------------------
    # Abstract / overridable hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_sample_tokens(self, sample: dict[str, Any]) -> int:
        """Return the token cost of one sample for packing budget accounting."""

    def collate_batch(self, samples: list[dict[str, Any]]) -> Any:
        """Assemble a packed list of samples into one batch.

        Default implementation returns the list unchanged (identity).
        Override to pad, stack, or transform samples into tensors.
        """
        return samples

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._data_len

    def __iter__(self) -> Iterator[Any]:
        self._ensure_worker_init()
        while True:
            batch = self._best_fit_batch()
            yield self.collate_batch(batch)

    # ------------------------------------------------------------------
    # Internal packing helpers (moved verbatim from _JointIterableDataset)
    # ------------------------------------------------------------------

    def _ensure_worker_init(self) -> None:
        """Build per-worker iterators + a decorrelated, reproducible RNG on first use.

        Runs inside the DataLoader worker (not ``__init__``) so each ``(rank, worker)`` gets its
        own underlying webdataset iterators and its own packer RNG, seeded from
        ``(seed, rank, worker, epoch)``. Fresh runs with the same inputs are reproducible and
        different workers are decorrelated. Exact mid-epoch resume is not provided because the
        packer pool, iterator cursors, and RNG state are not checkpointed. ``hash()`` of an all-int
        tuple is stable across processes (``PYTHONHASHSEED`` only randomizes ``str``/``bytes`` hashing).
        """
        winfo = torch.utils.data.get_worker_info()
        worker_id = int(winfo.id) if winfo is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        epoch = int(os.environ.get("WDS_EPOCH_NUM", "0"))
        if self._worker_epoch == epoch and self._rng is not None and self.iterators is not None:
            return
        # A new epoch must not consume samples buffered under the previous epoch's RNG/iterators.
        self._pool.clear()
        self._tok_cache.clear()
        self._flop_cache.clear()
        self._rng = random.Random(hash((self.seed, rank, worker_id, epoch)))
        self.iterators = [iter(ds) for ds in self._datasets]
        self._worker_epoch = epoch

    def _max_tokens(self, cur_max: int) -> int:
        # Flat budget: B(l)=max_tokens replacing the discontinuous half-budget step (full max_tokens
        # below l_max=1000, max_tokens//2 at/above). The step is a cliff a straddling batch can exceed
        # or underfill; the flat budget is constant so neither can happen. The half-budget
        # guarded a memory limit that the longest forced singleton already sets, so dropping it is
        # memory-neutral and packs denser (+8.6% useful_tps on-cluster). Off by default (= step).
        if self.flat_budget:
            return self.max_tokens
        if cur_max < 1000:
            return self.max_tokens
        return self.max_tokens // 2

    def _round_up(self, n: int) -> int:
        # Round up to the FP8 pad multiple the collate actually pads to, so the packer's budget
        # accounting matches the realized batch shape (see FP8_PAD_MULTIPLE).
        return round_up_to_multiple(n)

    def _get_next_sample(self) -> dict[str, Any]:
        self._ensure_worker_init()
        index_id = self._rng.choices(range(len(self.iterators)), weights=self._ratios, k=1)[0]
        curr_dataset = self.iterators[index_id]
        try:
            output = next(curr_dataset)
        except StopIteration:
            log.critical(f"dataset {self._dataset_names[index_id]} exhausted")
            self.iterators[index_id] = iter(self._datasets[index_id])
            output = next(self.iterators[index_id])
        return output

    def _fill_pool(self) -> None:
        while len(self._pool) < self.pool_size:
            self._pool.append(self._get_next_sample())

    def _padded_cost(self, cur_max: int, k: int) -> int:
        # Padded-cost rounding: the collate pads every row to the FP8_PAD_MULTIPLE-rounded batch
        # max (for FP8), so the budget must be charged against the SAME rounded length or it
        # under-counts and over-fills. legacy_budget_gate keeps the original unrounded cost for a faithful A/B.
        if self.legacy_budget_gate:
            return cur_max * k
        return self._round_up(cur_max) * k

    def _realized_padded_cost(self, cur_max: int, k: int) -> int:
        # The cost the collate will ACTUALLY pay (always rounded) -- used by the over_budget
        # telemetry independently of legacy_budget_gate so the A/B measures the true violation.
        return self._round_up(cur_max) * k

    def _batch_cost(self, cur_max: int, cur_sum: int, k: int) -> int:
        # True packing: the collate concatenates the pack into ONE row, so the realized token
        # count is round16(sum(L_i)) -- no per-sample padding. The padded path keeps charging
        # k*round16(max L_i). Both the candidate gate and the over_budget telemetry route through
        # here so the budget the packer enforces matches the row the collate actually builds.
        if self.true_packing:
            return self._round_up(cur_sum)
        return self._padded_cost(cur_max, k)

    def _budget_arg(self, cur_max: int, new_max: int) -> int:
        # Budget gate: evaluate the per-batch token budget at the POST-admit max (new_max). The
        # legacy (buggy) path evaluated it at the pre-admit max (cur_max), admitting candidates that push
        # the batch across the half-budget cliff (up to ~2x over budget). Flat budget: constant
        # (no cliff to straddle) so it is always sized at new_max regardless of the legacy toggle.
        if self.flat_budget:
            return new_max
        return cur_max if self.legacy_budget_gate else new_max

    def _tokens(self, sample: dict[str, Any]) -> int:
        # Stats cache: cache compute_sample_tokens per pooled sample (id-keyed; the pool holds the
        # refs so ids are stable for the batch's lifetime). Avoids the O(k*pool) recompute in the loops.
        key = id(sample)
        val = self._tok_cache.get(key)
        if val is None:
            val = self.compute_sample_tokens(sample)
            self._tok_cache[key] = val
        return val

    def _finalize_batch(
        self, chosen: list[dict[str, Any]], cur_max: int, cause: SingletonCause
    ) -> list[dict[str, Any]]:
        """Stash sync-free packing telemetry on the seed sample.

        ``custom_collate`` emits ``singleton_cause`` / ``over_budget`` as batch-level CPU ints
        (stripped before the HF forward). ``over_budget`` flags a MULTI-sample batch whose realized
        padded cost exceeds the (correct) budget -- the budget-gate symptom. Long singletons are exempt
        (they are allowed to exceed the budget by design).
        """
        k = len(chosen)
        over_budget = 0
        if k > 1:
            # True packing charges round16(sum L_i) (no per-sample padding); padded charges
            # k*round16(max L_i). cur_sum is cheap off the per-sample token cache.
            cur_sum = sum(self._tokens(s) for s in chosen)
            realized = self._round_up(cur_sum) if self.true_packing else self._realized_padded_cost(cur_max, k)
            if realized > self._max_tokens(cur_max):
                over_budget = 1
        chosen[0]["singleton_cause"] = int(cause.value if k == 1 else SingletonCause.NONE.value)
        chosen[0]["over_budget"] = over_budget
        # Authoritative true-packing marker for the collate: the collate concatenates instead
        # of stacking when this is set, so it never re-reads the env. Stamped on the seed (like the
        # other sync-free telemetry) and stripped before the HF forward.
        chosen[0]["true_packing"] = bool(self.true_packing)
        return chosen

    def _get_modality(self, sample: dict[str, Any]) -> Modality:
        if "pixel_values" in sample:
            return Modality.IMAGE
        elif "pixel_values_videos" in sample:
            return Modality.VIDEO
        return Modality.TEXT

    def _best_fit_batch(self) -> list[dict[str, Any]]:
        """Build one batch using the configured token-budget strategy."""
        self._tok_cache.clear()
        self._flop_cache.clear()
        self._fill_pool()
        seed = self._pool.popleft()
        seed_modality = self._get_modality(seed)
        L0 = self._tokens(seed)

        if L0 >= self.long_threshold:
            return self._finalize_batch([seed], L0, SingletonCause.LONG_THRESHOLD)
        if L0 >= self._max_tokens(L0):
            return self._finalize_batch([seed], L0, SingletonCause.BUDGET_CLIFF)
        if self.max_batch_size == 1:
            return self._finalize_batch([seed], L0, SingletonCause.MAX_BATCH_CAP)

        chosen = [seed]
        cur_max = L0
        cur_sum = L0  # running sum of lengths for the true-packing cost; ignored when padded.

        while self._pool:
            if self.max_batch_size and len(chosen) >= self.max_batch_size:
                break
            best_idx = self._find_best_candidate(cur_max, cur_sum, len(chosen), seed_modality)
            if best_idx is None:
                break
            cand = self._remove_from_pool(best_idx)
            chosen.append(cand)
            cand_len = self._tokens(cand)
            cur_max = max(cur_max, cand_len)
            cur_sum += cand_len

        cause = SingletonCause.NO_CANDIDATE if len(chosen) == 1 else SingletonCause.NONE
        return self._finalize_batch(chosen, cur_max, cause)

    def _find_best_candidate(self, cur_max: int, cur_sum: int, num_chosen: int, seed_modality: Modality) -> int | None:
        if self.batching_strategy == "prefer_first":
            return self._find_best_candidate_prefer_first(cur_max, cur_sum, num_chosen, seed_modality)
        return self._find_best_candidate_prefer_closest(cur_max, cur_sum, num_chosen, seed_modality)

    def _find_best_candidate_prefer_first(
        self, cur_max: int, cur_sum: int, num_chosen: int, seed_modality: Modality
    ) -> int | None:
        best_idx = None
        best_new_tokens = None
        for idx, cand in enumerate(self._pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self._tokens(cand)
            new_max = max(cur_max, L)
            new_tokens = self._batch_cost(new_max, cur_sum + L, num_chosen + 1)
            if new_tokens <= self._max_tokens(self._budget_arg(cur_max, new_max)):
                # True packing gives ``prefer_first`` its literal first-feasible semantics. Preserve
                # the pre-feature padded behavior exactly: scan every feasible candidate and choose
                # the one with the smallest physical padded cost (first candidate wins cost ties).
                if self.true_packing:
                    return idx
                if best_new_tokens is None or new_tokens < best_new_tokens:
                    best_new_tokens = new_tokens
                    best_idx = idx
        return best_idx

    def _find_best_candidate_prefer_closest(
        self, cur_max: int, cur_sum: int, num_chosen: int, seed_modality: Modality
    ) -> int | None:
        best_idx = None
        best_new_tokens = None
        smallest_length_diff = None
        for idx, cand in enumerate(self._pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self._tokens(cand)
            new_max = max(cur_max, L)
            new_tokens = self._batch_cost(new_max, cur_sum + L, num_chosen + 1)
            if new_tokens <= self._max_tokens(self._budget_arg(cur_max, new_max)):
                length_diff = abs(L - cur_max)
                # In padded mode the smallest admitted physical batch cost is the best-fit
                # criterion. Under true packing that cost is nearly just sum(lengths), which
                # degenerates into "pick the shortest" and defeats the documented closest-fit
                # strategy. Preserve padded behavior, but prioritize length proximity for a
                # true-packed row (then use cost as the deterministic tie-breaker).
                closer_true_pack = self.true_packing and (
                    smallest_length_diff is None
                    or length_diff < smallest_length_diff
                    or (length_diff == smallest_length_diff and new_tokens < best_new_tokens)
                )
                cheaper_padded = not self.true_packing and (
                    best_new_tokens is None
                    or new_tokens < best_new_tokens
                    or (new_tokens == best_new_tokens and length_diff < smallest_length_diff)
                )
                if closer_true_pack or cheaper_padded:
                    best_new_tokens = new_tokens
                    best_idx = idx
                    smallest_length_diff = length_diff
        return best_idx

    def _remove_from_pool(self, idx: int) -> dict[str, Any]:
        if idx == 0:
            return self._pool.popleft()
        elif idx == len(self._pool) - 1:
            return self._pool.pop()
        else:
            self._pool.rotate(-idx)
            item = self._pool.popleft()
            self._pool.rotate(idx)
            return item
