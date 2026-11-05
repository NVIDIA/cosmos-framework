# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact data continuation plans for starting a new run from a model-only artifact.

Full resumable checkpoints restore the source cursor through training_state. A model-only artifact intentionally
does not, so a new run needs an explicit plan binding the source pool, cursor order and exact prompt-draw count.

Two ways to continue, both from the source run's `config_resolved.yaml`:

- `ContinuationPlan.overrides()` -- EXACT: the new run reads the SAME pool with the same shuffle flag, the same
  resolved data seed and the same data-replica count, and every data shard advances its `IndexedCursor` by the draws
  that shard consumed (`rl.data.cursor_offset`, asserted against `rl.data.cursor_shards`). Its first batch is the
  source run's NEXT batch, epoch wraps included. This is the remedy for a warm restart that re-served the source
  run's first prompts to a later policy (identical step-1 batches on every restart).
- `ContinuationPlan.materialize()` -- a remaining-pool VIEW: the unseen rows written in file order to a new JSONL;
  the new run reshuffles them with its own seed. Membership is exact, order is not. Kept for the campaign CLIs.

Two bindings keep the plan exact. THE STEP AXIS: the boundary is the artifact's iteration (`iter_*` counts OPTIMIZER
steps; one rollout batch advances `rl.n_mini` of them), converted to rollout-batch steps with the `n_mini` of the
source's resolved config; an iteration that is not a batch boundary is refused. THE CHAIN: a source run that was itself
an exact continuation carries `rl.data.cursor_offset` / `cursor_shards` in its resolved config, and the plan carries
that base offset forward -- an A -> B -> C chain emits A's offset plus B's per-shard draws, never B's draws alone.
Counts are exact non-bool integers; nothing is coerced (the cursor and the config refuse coercion, so does the plan).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.utils.object_store import file_sha256

MANIFEST_VERSION = 1
RESOLVED_CONFIG = "config_resolved.yaml"


class _Rows:
    """Map-style cursor source that returns each pool index unchanged."""

    def __init__(self, n_rows: int) -> None:
        self._n_rows = n_rows

    def __len__(self) -> int:
        return self._n_rows

    def __getitem__(self, index: int) -> int:
        return index


@dataclass(frozen=True)
class ContinuationManifest:
    """Identity of one materialized remaining-pool view."""

    pool: str
    pool_rows: int
    pool_sha256: str
    consumed: int
    shuffle: bool
    seed: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        """Return the existing v1 JSON payload so current consumers remain byte-compatible after formatting."""
        return {
            "version": MANIFEST_VERSION,
            "pool": self.pool,
            "pool_rows": self.pool_rows,
            "pool_sha256": self.pool_sha256,
            "consumed": self.consumed,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "rows": self.rows,
            "order": "framework IndexedCursor(shuffle, seed) draw order; unseen rows kept in file order",
        }


@dataclass(frozen=True)
class ContinuationPlan:
    """Resolved source identity and exact draw boundary of a new-run continuation."""

    run_dir: str
    pool: str
    shuffle: bool
    seed: int
    batch: int
    group: int
    # The boundary as the artifact names it (`iter_*` = OPTIMIZER steps) and as rollout batches (`model_step // n_mini`).
    model_step: int
    batch_steps: int
    nominal_consumed: int
    consumed: int
    dynamic_sampling: bool
    # Data shards of the source run (= its rollout replica count, `roles/env/prompt.py::data_shard`). Every shard
    # answers every `get_samples` call with the same count, so `consumed` is `data_replicas` x the per-shard draws.
    data_replicas: int = 1
    # Optimizer steps per rollout batch of the source run (`RunConfig.n_mini`): 1 unless it mini-batched.
    n_mini: int = 1
    # Per-shard draws the source run itself skipped (`rl.data.cursor_offset` of its resolved config): non-zero when the
    # source was already an exact continuation. The emitted offset is this base plus the source's own per-shard draws.
    base_offset: int = 0

    @classmethod
    def from_run(cls, run_dir: str, model_step: int, consumed: int | None = None) -> ContinuationPlan:
        """Resolve one continuation from the artifact's iteration, requiring an exact count under dynamic sampling.

        Args:
            run_dir: the source run directory (its `config_resolved.yaml` is read).
            model_step: the artifact's iteration (`checkpoint_artifacts/iter_<model_step>`): OPTIMIZER steps, a multiple
                of the source's `rl.n_mini`; converted to rollout-batch steps here.
            consumed: the exact prompt-draw count of the source run (all shards together); required when dynamic
                sampling was on, otherwise defaults to the nominal `batch_steps x prompts per step`.

        Raises:
            TypeError: `model_step` or `consumed` is not an exact `int` (a `bool`, a float or a string is refused).
            ValueError: a negative count, an iteration off the batch grid, a count below the nominal or not divisible
                by the shard count, a refill mode with unequal shard draws, or a source base offset bound to another
                shard count.
        """
        model_step = _exact_count(model_step, "model_step")
        if consumed is not None:
            consumed = _exact_count(consumed, "consumed")
        config = run_data_config(run_dir)
        data_replicas = run_data_replicas(run_dir)
        n_mini = config["n_mini"]
        if model_step % n_mini:
            raise ValueError(
                f"{run_dir}: model_step {model_step} is not a multiple of rl.n_mini={n_mini}: the artifact's iter_* "
                "counts optimizer steps and one rollout batch advances n_mini of them, so this run's model-only "
                "artifacts sit only at multiples of n_mini; pass the artifact's iteration"
            )
        batch_steps = model_step // n_mini
        nominal = consumed_prompts(batch_steps, config["batch"], config["group"])
        fraction = config["min_informative_groups_frac"]
        dynamic = config["min_informative_groups"] > 0 or (fraction is not None and float(fraction) > 0)
        refill_mode = config["dynamic_sampling_refill_mode"]
        if dynamic and refill_mode != "all":
            # Every shard answers every refill draw only under the default refill mode. Under `short_cells` the
            # cells that still owe informative groups refill alone and the idle cells draw 0 (`entrypoints/grpo.py`,
            # `rl.dynamic_sampling_refill_mode`; `train/grpo_dynamic_sampling_avoided_prompts` counts the skipped
            # draws), so the shards' draw counts differ and one total split evenly over them is no shard's count:
            # the continuation would re-serve trained prompts on the shards that drew more and skip unseen prompts on
            # the shards that drew less (e.g. per-shard [4, 2] -> offset 3 on both).
            raise ValueError(
                f"{run_dir}: rl.dynamic_sampling_refill_mode={refill_mode!r} with dynamic sampling enabled: the data "
                "shards drew UNEQUAL counts (short cells refill alone, idle cells draw 0), so `--consumed` split evenly "
                f"over {data_replicas} shard(s) is not any shard's draw count and the continuation would re-serve "
                "trained prompts on one shard and skip unseen prompts on another. An exact continuation of such a run "
                "needs the per-shard draw counts, which a model-only artifact does not carry (follow-up: a `data_cursor` "
                "component in every partial artifact); refused"
            )
        if consumed is None and dynamic:
            raise ValueError(
                f"{run_dir}: dynamic sampling was enabled (min_informative_groups={config['min_informative_groups']}, "
                f"min_informative_groups_frac={fraction!r}), so the nominal {nominal} prompts is only a lower bound; "
                "pass --consumed with the exact prompt-draw count"
            )
        exact = nominal if consumed is None else consumed
        if exact < nominal:
            raise ValueError(
                f"--consumed {exact} is below the nominal {nominal} = {batch_steps} rollout-batch steps (model step "
                f"{model_step} / n_mini {n_mini}) x prompts per step"
            )
        if exact % data_replicas:
            raise ValueError(
                f"--consumed {exact} is not a multiple of the run's {data_replicas} data shard(s): every shard draws "
                "the same count per step, so the total must divide evenly"
            )
        base_offset, base_shards = config["cursor_offset"], config["cursor_shards"]
        if base_offset and base_shards != data_replicas:
            # The source was itself a continuation. Its offset is per shard, exact only for the shard count it was
            # computed for; a base bound to another count (or to none, `cursor_shards=0`) cannot be carried forward.
            bound = (
                f"for {base_shards} data shard(s)" if base_shards else "without a shard count (rl.data.cursor_shards=0)"
            )
            raise ValueError(
                f"{run_dir}: its rl.data.cursor_offset={base_offset} was computed {bound} but the run had "
                f"{data_replicas} data shard(s): the per-shard stride differs, so the source run's own continuation "
                "was not exact and this plan cannot carry its offset forward; refused"
            )
        pool = str(config["pool"])
        if not os.path.isfile(pool):
            raise FileNotFoundError(f"{run_dir}: its pool {pool} is not on disk")
        return cls(
            run_dir=os.path.abspath(run_dir),
            pool=os.path.abspath(pool),
            shuffle=bool(config["shuffle"]),
            seed=int(config["seed"]),
            batch=int(config["batch"]),
            group=int(config["group"]),
            model_step=model_step,
            batch_steps=batch_steps,
            nominal_consumed=nominal,
            consumed=exact,
            dynamic_sampling=dynamic,
            data_replicas=data_replicas,
            n_mini=n_mini,
            base_offset=base_offset,
        )

    @property
    def consumed_per_shard(self) -> int:
        """Draws each of the source run's data shards consumed in THAT run (`consumed` split over its shards)."""
        return self.consumed // self.data_replicas

    @property
    def cursor_offset(self) -> int:
        """The `rl.data.cursor_offset` the continuation emits: the source run's own base offset (non-zero when it was
        itself a continuation) plus the draws each of its shards consumed. Draw `cursor_offset + 1` of the seed's
        sequence is the continuation's first prompt on every shard."""
        return self.base_offset + self.consumed_per_shard

    @property
    def total_consumed(self) -> int:
        """Pool draws the remaining-pool view excludes: the accumulated per-shard offset over every shard."""
        return self.cursor_offset * self.data_replicas

    def overrides(self) -> list[str]:
        """Dotted overrides that make a new run serve the source run's NEXT prompts.

        The continuation keeps the SAME pool and shuffle flag, pins the prompt-order seed to the value the source
        run's cursor actually permuted with (`rl.seed=<resolved>` with `rl.data_seed_follows_rollout_seed=false`, so a
        continuation that varies `rollout.seed` for sampling keeps the source order), and advances every data shard
        by its consumed draws. The data role refuses the run if its data-replica count differs from the source's.

        Returns:
            One `key=value` string per override, in the order they should be appended (later keys win).
        """
        return [
            f"rl.data.path={self.pool}",
            f"rl.data.shuffle={'true' if self.shuffle else 'false'}",
            f"rl.seed={self.seed}",
            "rl.data_seed_follows_rollout_seed=false",
            f"rl.data.cursor_offset={self.cursor_offset}",
            f"rl.data.cursor_shards={self.data_replicas}",
        ]

    def materialize(self, out_dir: str | None = None) -> tuple[str, dict[str, Any]]:
        """Write or reuse this plan's remaining-pool view."""
        return slice_pool(
            self.pool,
            self.total_consumed,
            shuffle=self.shuffle,
            seed=self.seed,
            out_dir=out_dir,
        )


def _exact_count(value: Any, name: str) -> int:
    """An exact non-negative `int`. A `bool`, a float or a numeric string is refused, never coerced: `int("440")` or
    `int(440.9)` would move an exact continuation to another prompt boundary without a word."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def prompts_per_step(batch: int, group: int) -> int:
    """Prompts consumed per step: rollout batch divided by rollouts per prompt."""
    if batch <= 0 or group <= 0 or batch % group:
        raise ValueError(f"batch_per_replica={batch} must be a positive multiple of group_size={group}")
    return batch // group


def consumed_prompts(steps: int, batch: int, group: int) -> int:
    """Nominal prompt draws for completed steps; dynamic sampling may make this a lower bound."""
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    return steps * prompts_per_step(batch, group)


def seen_indices(n_rows: int, consumed: int, *, shuffle: bool, seed: int) -> list[int]:
    """Pool indices drawn by the existing framework cursor before the continuation boundary."""
    if n_rows <= 0:
        raise ValueError("the pool is empty")
    if consumed < 0:
        raise ValueError(f"consumed must be >= 0, got {consumed}")
    if consumed >= n_rows:
        raise ValueError(
            f"the run consumed {consumed} prompts from a {n_rows}-row pool: at least one full pass, nothing unseen remains"
        )
    cursor = IndexedCursor(_Rows(n_rows), shard_index=0, shard_count=1, shuffle=shuffle, seed=seed, label="pool_offset")
    return [cursor.next_index() for _ in range(consumed)]


def read_rows(pool: str) -> list[bytes]:
    """Read JSONL rows byte-for-byte without their newline."""
    with open(pool, "rb") as handle:
        rows = handle.read().split(b"\n")
    if rows and rows[-1] == b"":
        rows.pop()
    return rows


def slice_path(pool: str, consumed: int, out_dir: str | None = None) -> str:
    stem, extension = os.path.splitext(os.path.basename(pool))
    return os.path.join(
        out_dir or os.path.dirname(os.path.abspath(pool)),
        f"{stem}_from{consumed}{extension or '.jsonl'}",
    )


def manifest_path(slice_file: str) -> str:
    stem, _extension = os.path.splitext(slice_file)
    return f"{stem}_manifest.json"


def read_manifest(slice_file: str) -> dict[str, Any] | None:
    try:
        with open(manifest_path(slice_file)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def slice_pool(
    pool: str,
    consumed: int,
    *,
    shuffle: bool,
    seed: int,
    out_dir: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Write or reuse a pool view excluding exactly the cursor draws in the plan."""
    pool = os.path.abspath(pool)
    rows = read_rows(pool)
    seen = set(seen_indices(len(rows), consumed, shuffle=shuffle, seed=seed))
    keep = [index for index in range(len(rows)) if index not in seen]
    manifest = ContinuationManifest(
        pool=pool,
        pool_rows=len(rows),
        pool_sha256=file_sha256(pool),
        consumed=consumed,
        shuffle=bool(shuffle),
        seed=int(seed),
        rows=len(keep),
    ).to_dict()
    out = slice_path(pool, consumed, out_dir)
    if os.path.exists(out):
        existing = read_manifest(out)
        if existing == manifest:
            return out, manifest
        raise FileExistsError(
            f"{out} exists with a different identity ({existing}); refusing to rewrite a slice a job may be reading "
            f"(pass --out-dir for a fresh one)"
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    temporary = out + ".part"
    with open(temporary, "wb") as handle:
        for index in keep:
            handle.write(rows[index] + b"\n")
    os.replace(temporary, out)
    encoded = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    atomic_write_bytes(manifest_path(out), encoded)
    return out, manifest


def run_data_config(run_dir: str) -> dict[str, Any]:
    """Read the data facts needed for continuation from config_resolved.yaml.

    ``seed`` is the seed the run's prompt cursor actually permuted with: `rl.seed`, or `rollout.seed` when
    `rl.data_seed_follows_rollout_seed` is set, resolved by the same `config.rl.resolve_data_seed` the data
    role uses. Reading `rl.seed` alone sliced the wrong rows for every run whose order followed `rollout.seed`.
    ``n_mini`` mirrors `RunConfig.n_mini` (`batch_per_replica // mini_batch_per_replica`, 1 when unset): the
    optimizer steps one rollout batch advances, which is the clock `iter_*` counts. ``cursor_offset`` /
    ``cursor_shards`` are the run's OWN continuation (non-zero when it was started from a model-only artifact
    itself); a plan built from it carries that base offset forward.
    """
    from types import SimpleNamespace

    import yaml

    from cosmos_framework.posttrain.config.rl import resolve_data_seed

    path = os.path.join(run_dir, RESOLVED_CONFIG)
    with open(path) as handle:
        resolved = yaml.safe_load(handle) or {}
    rl = resolved.get("rl") or {}
    rollout = resolved.get("rollout")
    data = rl.get("data") or {}
    try:
        seed = resolve_data_seed(
            SimpleNamespace(
                seed=rl["seed"],
                data_seed_follows_rollout_seed=bool(rl.get("data_seed_follows_rollout_seed", False)),
            ),
            None if not isinstance(rollout, dict) else SimpleNamespace(seed=rollout.get("seed")),
        )
        batch = int(rl["batch_per_replica"])
        mini = rl.get("mini_batch_per_replica", 0) or 0
        return {
            "pool": str(data["path"]),
            "shuffle": bool(data["shuffle"]),
            "seed": seed,
            "batch": batch,
            "group": int(rl["group_size"]),
            "n_mini": batch // int(mini) if mini else 1,
            "cursor_offset": _exact_count(data.get("cursor_offset", 0) or 0, "rl.data.cursor_offset"),
            "cursor_shards": _exact_count(data.get("cursor_shards", 0) or 0, "rl.data.cursor_shards"),
            "min_informative_groups": int(rl.get("min_informative_groups", 0)),
            "min_informative_groups_frac": rl.get("min_informative_groups_frac"),
            # Absent on trees without the knob (= the default "all": every shard answers every refill draw).
            "dynamic_sampling_refill_mode": str(rl.get("dynamic_sampling_refill_mode", "all")),
        }
    except KeyError as error:
        raise KeyError(
            f"{path}: rl.{error.args[0]} missing (rl.data.path/shuffle, rl.seed, rl.batch_per_replica, rl.group_size; "
            "rollout.seed under rl.data_seed_follows_rollout_seed)"
        ) from error


def run_data_replicas(run_dir: str) -> int:
    """The source run's data shard count from config_resolved.yaml: its rollout replica count, at least 1.

    Mirrors `roles/env/prompt.py::data_shard` over `GridConfig.rollout_shape()`: `grid.shapes.rollout[0]` when
    `grid.shapes` is set, else `grid.num_replicas`; a count of one (or no grid section) means one unsharded cursor.
    """
    import yaml

    path = os.path.join(run_dir, RESOLVED_CONFIG)
    with open(path) as handle:
        grid = (yaml.safe_load(handle) or {}).get("grid") or {}
    shapes = grid.get("shapes") or {}
    if shapes:
        try:
            replicas = shapes["rollout"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f"{path}: grid.shapes.rollout must be [replicas, ranks], got {shapes.get('rollout')!r}"
            ) from error
    else:
        replicas = grid.get("num_replicas", 1)
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError(f"{path}: the rollout replica count must be a positive integer, got {replicas!r}")
    return int(replicas)


def slice_for_run(
    run_dir: str,
    model_step: int,
    *,
    consumed: int | None = None,
    out_dir: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compatibility function used by campaign CLIs; delegates to ContinuationPlan (`model_step` = the artifact's iteration)."""
    return ContinuationPlan.from_run(run_dir, model_step, consumed).materialize(out_dir)


def continuation_overrides(run_dir: str, model_step: int, *, consumed: int | None = None) -> list[str]:
    """Compatibility function used by campaign CLIs; the exact-continuation overrides of `ContinuationPlan`."""
    return ContinuationPlan.from_run(run_dir, model_step, consumed).overrides()
