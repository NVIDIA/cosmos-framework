# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure routing and ingress selection for asymmetric worker meshes.

Ray-free module: routing is a pure function of producer and consumer shapes,
computed at submit time, never at bind. No imports of torch or ray at module level.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def select_model_dp(items: list[Any], *, dp_rank: int, dp_world: int) -> list[Any]:
    """Select one contiguous model-data-parallel shard from a complete ordered batch."""
    if type(dp_rank) is not int or type(dp_world) is not int or dp_world < 1 or not 0 <= dp_rank < dp_world:
        raise ValueError(f"invalid model-DP coordinate {dp_rank}/{dp_world}")
    if len(items) % dp_world:
        raise ValueError(f"batch size {len(items)} is not divisible by model-DP world {dp_world}")
    shard_size = len(items) // dp_world
    return items[dp_rank * shard_size : (dp_rank + 1) * shard_size]


def select_chunk(
    items: list[Any],
    *,
    mini: int,
    micro: int,
    n_mini: int,
    n_micro: int,
    dp_rank: int,
    dp_world: int,
    ahead_pass: bool = False,
    shuffle: bool = False,
) -> list[Any]:
    """Select one consumer rank's mini-, micro-, and DP-sharded batch slice.

    With ``shuffle=False`` (the default), this executes the legacy selector exactly: sort, chunk
    along mini/micro, then interleave the resulting chunk across DP ranks. With ``shuffle=True``,
    the AlpaGym path instead splits the DP axis by episode, permutes this rank's share, and only
    then chunks it along mini/micro. Keeping the complete legacy path behind the default branch is
    deliberate: episode metadata did not participate in selection before this option existed.

    `ahead_pass` is ACCEPTED AND IGNORED here. It rides the same control-kwarg channel because it is
    a per-call ROUTING fact -- the batch is materialized twice this loop step, so `workers/grid.py`
    must size the bulk refcount for two readers -- not a selection input. Selection must stay a pure
    function of (mini, micro, dp), or the two passes would not produce identical chunks and the
    order-keyed π_old cache would mis-pair. Declared explicitly rather than swallowed by **_ so a
    typo in the kwarg name fails loudly instead of silently reverting the refcount.

    Args:
        items: The sole policy execution replica's trajectories, in any order. When ``shuffle`` is
            enabled, items sharing a ``session`` are one episode and stay on one rank; items without
            one are their own.
        mini: Mini-batch index in ``[0, n_mini)``.
        micro: Micro-batch index in ``[0, n_micro)``.
        n_mini: Mini-batches per loop step.
        n_micro: Micro-batches per optimizer step.
        dp_rank: This rank's flattened model-DP coordinate.
        dp_world: The full ``dp_replicate * dp_shard`` extent. Values at most one skip the interleave.
        ahead_pass: Accepted and ignored; a routing fact for the bulk refcount, never a selection input.
        shuffle: enable the episode-first split and permute this rank's share before chunking. Off
            by default -- see the body.

    Returns:
        This rank's items. The default path preserves canonical sorted order; the enabled path is a
        deterministic seeded permutation.

    Raises:
        ValueError: The batch cannot be divided exactly across the mini/micro axes.
    """
    # DEFAULT MEANS LEGACY, literally rather than algebraically. The old order was sort -> mini ->
    # micro -> DP interleave, and it ignored `session` entirely. Keeping it as its own branch makes
    # the opt-in contract testable for every previously-valid item shape, including duck-typed
    # items with no `session` attribute.
    n_chunks = n_mini * n_micro
    if not shuffle:
        if len(items) % n_chunks:
            raise ValueError(
                f"select_chunk got {len(items)} items, not a multiple of n_mini*n_micro="
                f"{n_mini}*{n_micro}={n_chunks}: the mini/micro floor divisions would silently drop the "
                f"{len(items) % n_chunks}-item tail. The collected batch must be exact; a short batch "
                f"means an upstream filter or abort changed the count."
            )
        ordered = sorted(items, key=lambda item: (str(item.group_id), str(item.id)))
        mini_len = len(ordered) // n_mini
        chunk = ordered[mini * mini_len : (mini + 1) * mini_len]
        micro_len = mini_len // n_micro
        chunk = chunk[micro * micro_len : (micro + 1) * micro_len]
        if dp_world <= 1:
            return chunk
        return [item for index, item in enumerate(chunk) if index % dp_world == dp_rank]

    ordered = sorted(items, key=lambda item: (str(item.group_id), str(item.id)))

    # DP FIRST, and by EPISODE -- the order the cosmos-rl baseline splits in. Its dispatcher hands
    # each policy rank whole rollouts (`received_rollouts=3` of a 6-rollout group at dp_shard=2);
    # the rank then flattens them, shuffles its own pool, and walks it one sample per optimizer
    # step. Splitting per ITEM instead would hand a rank steps from every episode, which covers the
    # same batch but pairs different steps at each update.
    #
    # An episode is `session`; items that do not carry one (one trajectory per item, as text GRPO
    # produces) fall back to `id`, which is unique, so every item is its own unit and this
    # degenerates to the per-item split exactly.
    units: list[str] = []
    members: dict[str, list] = {}
    for item in ordered:
        key = str(getattr(item, "session", None) or item.id)
        if key not in members:
            members[key] = []
            units.append(key)
        members[key].append(item)
    sizes = {len(members[key]) for key in units}
    if dp_world > 1 and (len(sizes) != 1 or len(units) % dp_world):
        # Both conditions are what make every rank's share the same LENGTH, and an unequal share is
        # not a wrong answer but a hang: ranks would run different numbers of forwards and desync
        # on the first collective inside one.
        raise ValueError(
            f"select_chunk cannot split {len(units)} units of sizes {sorted(sizes)} across "
            f"dp_world={dp_world}: the units must be equal-sized and divide the rank count, or the "
            f"ranks receive different numbers of micro-batches and deadlock rather than fail."
        )
    mine = (
        ordered
        if dp_world <= 1
        else [item for index, key in enumerate(units) if index % dp_world == dp_rank for item in members[key]]
    )

    if len(mine) % n_chunks:
        raise ValueError(
            f"select_chunk got {len(mine)} items for this rank, not a multiple of n_mini*n_micro="
            f"{n_mini}*{n_micro}={n_chunks}: the mini/micro floor divisions would silently drop the "
            f"{len(mine) % n_chunks}-item tail. The collected batch must be exact; a short batch "
            f"means an upstream filter or abort changed the count."
        )

    # Permute, because the canonical sort groups an episode's steps together and an advantage
    # broadcast across those steps then reaches the optimizer as a run: every step of episode k,
    # then every step of k+1, each run pushing one advantage. cosmos-rl draws `torch.randperm` over
    # its own pool before each pass (`alpagym_runtime/cosmos/trainer.py::_run_training_loop`); this
    # is that draw, and like it the permutation is per-rank, over the share the rank actually holds.
    #
    # OPT-IN, because it is a real change of training order and only one caller needs it. An entry
    # whose items each carry their own advantage -- one trajectory per item, as text GRPO produces
    # -- has no run to break up, and leaving it off keeps that path byte-identical to what it was
    # validated at.
    #
    # Seeded from the share's own canonical ids, NOT a global RNG: selection has to be a pure
    # function of its inputs so the pi_old ahead pass re-derives the same chunk for the same (mini,
    # micro), and reproducible across runs (prompt ids are monotonic -- `DatasetEnv.reset`).
    digest = hashlib.blake2b(
        "\n".join(f"{item.group_id}\x00{item.id}" for item in mine).encode(), digest_size=8
    ).digest()
    random.Random(int.from_bytes(digest, "big")).shuffle(mine)

    mini_len = len(mine) // n_mini
    chunk = mine[mini * mini_len : (mini + 1) * mini_len]
    micro_len = mini_len // n_micro
    return chunk[micro * micro_len : (micro + 1) * micro_len]


def route_sharded(P: int, R_c: int) -> tuple[str, int]:
    """Resolve data redistribution routing based on producer and consumer unit counts.

    Args:
        P: producer unit count (from declared sub-shape)
        R_c: consumer replica count

    Returns:
        tuple of (op, n) where op is "one", "reduce", or "split":
        - ("one", 1): P == R_c — 1:1 distinct units to replicas
        - ("reduce", n): P == n*R_c — n contiguous units per replica, reduced via callable
        - ("split", s): R_c == s*P — each unit shared by s replicas, split via callable

    Raises:
        ValueError: if either count is < 1, or if the counts are not evenly divisible (names P,
            R_c, and the suggested fix).
    """
    # Degenerate counts fail loud like everything else in this layer, rather than surfacing as a
    # ZeroDivisionError (R_c == 0, three frames down in `P % R_c`) or, worse, silently: P == 0 would
    # satisfy `0 % R_c == 0` and route ("reduce", 0), which `units_for_replica` turns into an empty
    # ref list — the consumer would reduce over NOTHING (a silent empty batch, no raise).
    if P < 1 or R_c < 1:
        raise ValueError(
            f"Sharded: producer units P={P} and consumer replicas R_c={R_c} must both be >= 1 — a "
            f"Sharded edge needs at least one producer unit and one consumer replica (P==0 routes an "
            f"empty batch through reduce; R_c==0 divides by zero). If P looks like 0, a producer WG "
            f"may be mis-declared; declare SPMD producers at ranks=1 (spec 2026-07-22 §2.2)"
        )

    if P == R_c:
        return ("one", 1)

    if P % R_c == 0:
        n = P // R_c
        return ("reduce", n)

    if R_c % P == 0:
        s = R_c // P
        return ("split", s)

    # Not evenly divisible
    raise ValueError(
        f"Sharded: producer units P={P} and consumer replicas R_c={R_c} are not evenly divisible "
        f"— declare a mesh with matching counts, or (P<R_c) declare split= / (P>R_c) declare reduce=. "
        f"If the producer is a multi-rank SPMD helper, its mirrored rank lanes inflate P — declare it "
        f"at ranks=1 so its output counts as one unit per replica (spec 2026-07-22 §2.2)"
    )


def readers_for(Kc: int, n: int, op: str, *, select: bool, dp_world: int | None = None) -> int:
    """How many consumer ranks will each independently pull a given kept ref (spec §9.5).

    This is the producer's `BulkStore` refcount: the entry pops only once every reader has pulled,
    so too low faults a later puller (`KeyError`) and too high leaks the entry.

    Without `select`, every rank of the consumer cell materializes the FULL kept slice, so all of
    them read each ref: `readers = Kc` (or `Kc·n` for a `split` fan-out, where each unit is also
    shared across the `n` producer units feeding a replica).

    With `select` (the dp interleave of `select_chunk`), a kept item belongs to ONE dp coordinate,
    so only the ranks SHARING that dp_rank pull it — and the ranks sharing a dp_rank are exactly the
    non-DP group (TP/PP), of size `Kc / dp_world`. That is the whole point: a rank in a different
    TP/PP position but the SAME dp_rank reads the SAME sample, so it must be counted. `dp_world`
    defaults to `Kc` — the current pure-DP case where every policy rank owns one DP coordinate and
    `readers == 1`. A future policy world with TP/PP has `dp_world < Kc`, so
    `readers == Kc / dp_world > 1`.

    Args:
        Kc: the consumer cell's rank width (its `(R, K)` sub-shape's K).
        n: the routing scale — `1` for `one`, the reduce/split count otherwise.
        op: the routing op, `"one"` | `"reduce"` | `"split"`.
        select: whether `select_chunk`'s dp interleave runs on this edge (the opt-in per-rank pull).
        dp_world: complete model-DP extent; `None` ⇒ `Kc` (no TP/PP inside the rank world).

    Returns:
        The reader count for one kept ref.

    Raises:
        ValueError: `select` and `dp_world` does not divide `Kc` — the shape and the parallelism
            factorization disagree, which would give a fractional (and therefore wrong) refcount.
    """
    total = Kc * n if op == "split" else Kc
    if not select:
        return total
    world = Kc if dp_world is None else dp_world
    if world < 1 or Kc % world:
        raise ValueError(
            f"dp_world={world} must be a positive divisor of Kc={Kc}: the consumer's rank width must "
            f"be a whole number of DP coordinates, else the per-ref reader count (Kc/dp_world, the non-DP "
            f"group size) is fractional — a leaked or early-popped BulkStore entry."
        )
    return total // world


def units_for_replica(op: str, n: int, P: int, r: int) -> list[int]:
    """Contiguous unit indices assigned to consumer replica r.

    Args:
        op: dispatch operation ("one", "reduce", or "split")
        n: parameter from route_sharded (count scaling)
        P: producer unit count
        r: consumer replica index

    Returns:
        list of producer unit indices for replica r.
    """
    if op == "one":
        return [r]
    elif op == "reduce":
        # n units per replica, replica r gets units [n*r, n*r+n-1]
        return list(range(n * r, n * r + n))
    elif op == "split":
        # Each unit shared across multiple replicas; replica r uses unit r // n
        return [r // n]
    else:
        raise ValueError(f"unknown routing op {op!r} (expected one|reduce|split)")


def split_index(op: str, n: int, r: int) -> int:
    """Determine which shard index this replica keeps under split operation.

    Args:
        op: dispatch operation ("one", "reduce", or "split")
        n: parameter from route_sharded (shard count for split)
        r: consumer replica index

    Returns:
        shard index for split, 0 for other operations.
    """
    if op == "split":
        return r % n
    return 0


def flat_units(shape: tuple) -> list[tuple]:
    """Row-major flattening of producer sub-shape into (r, k) cell indices.

    Produces the unit definition respecting declared sub-shape: units are producer cells,
    ordered row-major (replica-first, then rank).

    Args:
        shape: (R, K) tuple of producer replicas and ranks

    Returns:
        list of (r, k) tuples in row-major order.
    """
    if not shape or len(shape) != 2:
        raise ValueError(f"Expected 2-tuple shape, got {shape}")

    R, K = shape
    result = []
    for r in range(R):
        for k in range(K):
            result.append((r, k))
    return result
