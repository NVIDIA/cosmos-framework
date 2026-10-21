# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Per-(kind, method, param) dispatch registry (spec 2026-07-22 §2.1).

Routing semantics are a property of the METHOD's contract, not of a call site or a
topology — "`train` consumes the producer's units grouped per my replica" holds at every
topology. So the declaration lives here, keyed `(role_kind, method, param)`, in a module
every entry file already imports driver-side. `roles/core/*` must stay Ray-free (and
torch-free) at import — this module is consulted driver-side by `GridBackend.submit`
before any actor exists.

`DISPATCH` maps `(kind, method, param)` to a `Sharded`/`Mirror` spec (Task A2 markers,
constructed with `arg=None` — the registry declares SHAPE, not a wrapped value). `reduce`/
`split` on those specs are STRING references (`"module:attr"`), resolved lazily actor-side
via `resolve_fn` so the driver never imports the (possibly torch-heavy) role body module
that defines them.

`PROFILE_DISPATCH` adds algorithm-scoped overrides without changing generic role contracts.
Its `RankAligned` entries require identical producer/consumer meshes and forward the exact
`(replica, rank)` reference without a `Dispatched` materialization wrapper.

Precedence (spec 2026-07-22 §2.1), all fail-loud, enforced by callers of this module (not here):
1. Call-site marker (explicit `RankAligned(...)`, `Sharded(...)`, or `Mirror(...)` wrapper on the arg).
2. Registry default (`lookup` below) — the normal path; loops stay marker-free.
3. Bare fallback — no entry, no marker: today's `dp_aligned` iff replica counts match,
   else RAISE naming the missing `(kind, method, param)`.

**Param-name binding (`bind_param_names`).** DISPATCH is keyed by parameter name, but call
sites hand `submit` positional args / kwargs, not names — `bind_param_names` maps a call's
`(args, kw)` onto the names `lookup` expects.

**Why `_SIGNATURES` is a hand-maintained table, not protocol reflection.** `bind_param_names`
could in principle read the live `RolloutProtocol`/`TrainerProtocol` signatures instead of a literal
table — but not torch-free, which this driver-side module must be. `roles/trainer/base.py` now also
owns the torch-backed `BaseTrainer` implementation, and importing either role protocol by dotted
path first runs the corresponding package initializer and its concrete engine imports. So any
dotted-path import of the protocol modules pulls torch into the driver. Hence the small literal
`_SIGNATURES` table below.
"""

from __future__ import annotations

import importlib

from cosmos_framework.posttrain.workers.worker_group import Mirror, RankAligned, Sharded

DispatchSpec = RankAligned | Sharded | Mirror

# Literal (kind, method) -> ordered positional parameter names. Hand-maintained rather than
# reflected off the Rollout/Trainer Protocols: their dotted-path import pulls torch onto the
# driver (see module docstring), which this module must stay free of.
_SIGNATURES: dict[tuple[str, str], tuple[str, ...]] = {
    ("rollout", "generate"): ("prompts",),
    ("trainer", "train"): ("batch",),
    # The π_old ahead pass. A DISPATCH entry alone is NOT enough: without a name for the positional
    # arg, `bind_param_names` yields None, no spec matches, `take_control_kwargs` strips nothing, and
    # the control kwargs land on the body as `compute_old_logprobs(batch, mini=...) -> TypeError`.
    # This table and DISPATCH must be edited together for any new selected method.
    ("trainer", "compute_old_logprobs"): ("batch",),
    ("reward", "score_with_provider_report"): ("trajectories",),
    ("buffer", "put_samples"): ("trajectories",),
    ("reference", "score"): ("trajectories",),
    ("critic", "predict"): ("trajectories",),
    ("trainer", "prepare_batch"): ("trajectories",),
    ("trainer", "prepare_ppo_targets"): ("trajectories",),
    ("critic", "install_targets"): ("targets",),
}

# Topology-independent, placement-independent routing defaults (spec 2026-07-22 §2.1). Both entries hold
# at EVERY topology and degenerate to op="one" (a single forwarded ref, byte-identical to the
# pre-dispatch path) whenever producer units == consumer replicas — so a symmetric or single-cell
# loop never pays for them. The string refs are resolved actor-side by `resolve_fn`:
#   - train/batch  reduce (FanIn): P > R_c routes n contiguous producer units per consumer replica,
#                  concat_batches merges the trajectory lists (roles/trainer/llm.py).
#   - generate/prompts split (FanOut): R_c > P shares one producer ref across s replicas, each
#                  keeping its 1/s slice via split_prompts (roles/env/dataset.py) — the asymmetric
#                  data-parallel rollout edge (e.g. data (R,1) -> rollout (s·R, K)).
DISPATCH: dict[tuple[str, str, str], DispatchSpec] = {
    ("trainer", "train", "batch"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches",
        # Selection is OPT-IN on the presence of these sibling kwargs (spec 2026-08-12 §3.4).
        # GRPO always passes them; DPO never does, and absence means today's behaviour exactly.
        # This edge is keyed on the role KIND, so DPO shares it — see `take_control_kwargs`.
        select="cosmos_framework.posttrain.workers.dispatch:select_chunk",
        select_params=("mini", "micro", "n_mini", "n_micro", "ahead_pass", "shuffle"),
    ),
    # π_old ahead pass. MUST mirror the ("trainer","train","batch") entry exactly: the frozen π_old
    # is keyed by CALL ORDER, so chunk N of this pass has to be the same trajectories as chunk N of
    # training. A different reduce/select here would silently mis-pair them and the clip would bind
    # against the wrong tokens — worse than not freezing at all.
    ("trainer", "compute_old_logprobs", "batch"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches",
        select="cosmos_framework.posttrain.workers.dispatch:select_chunk",
        select_params=("mini", "micro", "n_mini", "n_micro", "ahead_pass", "shuffle"),
    ),
    ("rollout", "generate", "prompts"): Sharded(split="cosmos_framework.posttrain.roles.env.dataset:split_prompts"),
    ("buffer", "put_samples", "trajectories"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches"
    ),
    ("reference", "score", "trajectories"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches",
        select="cosmos_framework.posttrain.workers.dispatch:select_model_dp",
        rank_partitioned=True,
    ),
    ("critic", "predict", "trajectories"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches",
        select="cosmos_framework.posttrain.workers.dispatch:select_model_dp",
        rank_partitioned=True,
    ),
    ("trainer", "prepare_batch", "trajectories"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches",
        select="cosmos_framework.posttrain.workers.dispatch:select_model_dp",
        rank_partitioned=True,
    ),
    ("trainer", "prepare_ppo_targets", "trajectories"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches",
        select="cosmos_framework.posttrain.workers.dispatch:select_model_dp",
        rank_partitioned=True,
    ),
    ("critic", "install_targets", "targets"): Sharded(
        reduce="cosmos_framework.posttrain.roles.trainer.llm:concat_batches",
        select="cosmos_framework.posttrain.workers.dispatch:select_model_dp",
        rank_partitioned=True,
    ),
}

# Algorithm-specific method contracts override the generic role defaults without putting routing
# markers in the control loop. Online WFM currently produces one distinct sample on every model
# rank, so every data edge is exact-rank SPMD. A future disaggregated NFT layout changes this table
# and placement declarations; ``online_nft_step`` remains untouched.
PROFILE_DISPATCH: dict[str, dict[tuple[str, str, str], DispatchSpec]] = {
    "nft": {
        ("rollout", "generate", "prompts"): RankAligned(),
        ("reward", "score_with_provider_report", "trajectories"): RankAligned(),
        ("buffer", "put_samples", "trajectories"): RankAligned(),
        ("trainer", "train", "batch"): RankAligned(),
    }
}


def lookup(
    kind: str,
    method: str,
    param: str,
    *,
    dispatch_profile: str | None = None,
) -> DispatchSpec | None:
    """Registry default for a call-site arg (spec 2026-07-22 §2.1 precedence tier 2). None ⇒ no
    registry entry; caller falls through to a call-site marker check then the bare path.

    Args:
        kind: The consumer role kind (e.g. ``"trainer"``, ``"rollout"``).
        method: The role method being dispatched (e.g. ``"train"``, ``"generate"``).
        param: The parameter name of the arg being routed.

    Returns:
        The registered `RankAligned`/`Sharded`/`Mirror` routing spec for
        ``(kind, method, param)``, or ``None`` if none.
    """
    key = (kind, method, param)
    if dispatch_profile is not None:
        profile = PROFILE_DISPATCH.get(dispatch_profile)
        if profile is None:
            raise ValueError(
                f"unknown dispatch profile {dispatch_profile!r}; supported profiles are {sorted(PROFILE_DISPATCH)}"
            )
        scoped = profile.get(key)
        if scoped is not None:
            return scoped
    return DISPATCH.get(key)


def resolve_fn(ref: str):
    """Import a `"module:attr"` string reference (actor-side use — the actor already
    imports the role module that defines `attr`, so this stays torch-free on the driver
    even though the resolved callable may not be).

    Args:
        ref: A ``"module:attr"`` reference; malformed (no ``":"``) raises `ValueError`.

    Returns:
        The `attr` object imported from `module` (the resolved `reduce`/`split` callable).
    """
    if ":" not in ref:
        raise ValueError(f"malformed function reference (expected 'module:attr'): {ref!r}")
    module_name, _, attr_name = ref.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def bind_param_names(kind: str, method: str, args: tuple, kw: dict) -> list[str | None]:
    """Map a call's positional `args` / keyword `kw` onto the parameter names `lookup`
    expects. Positional args bind via `_SIGNATURES[(kind, method)]` by position — `None`
    for positions beyond the known signature (or an unknown `(kind, method)` entirely, so
    every position is unbindable). Keyword args already carry their own name.

    Args:
        kind: The role kind used to key `_SIGNATURES`.
        method: The role method used to key `_SIGNATURES`.
        args: The call's positional args, bound by position to the signature's parameter names.
        kw: The call's keyword args, each already carrying its own name.

    Returns:
        One name per arg in order — positional names (or ``None`` past the known signature), then the
        keyword keys.
    """
    names = _SIGNATURES.get((kind, method), ())
    bound: list[str | None] = [names[i] if i < len(names) else None for i in range(len(args))]
    bound.extend(kw.keys())
    return bound
