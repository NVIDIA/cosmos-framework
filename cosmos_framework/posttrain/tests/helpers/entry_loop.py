# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared @L0 @CPU scaffolding for the entry loop-invariant tests (dpo/grpo).

Only the mechanical, entry-agnostic plumbing lives here: the config/reporting/watch mocks and the
SimpleNamespace config wrapper the loops read. The per-entry pieces — the role `units` (grpo has a
rollout + weight_sync; dpo is offline) and the fake policy's loss key — diverge and stay inline in each
entry's test file, matching the inline-loop doctrine.
"""

from __future__ import annotations

import errno
import os
from types import SimpleNamespace

import pytest


def emulate_supported_atomic_noreplace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for a filesystem supporting atomic ``RENAME_NOREPLACE``.

    The production helper deliberately fails closed on Lustre versions that reject that primitive.
    Entry-loop tests exercise the post-promotion transaction, so they install this synchronous
    single-process stand-in while manager tests separately cover unsupported filesystems and races.
    """
    from cosmos_framework.posttrain.checkpoint import external_copy

    def _rename_noreplace(source: str, target: str) -> None:
        if os.path.lexists(target):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), target)
        os.rename(source, target)

    monkeypatch.setattr(external_copy, "_rename_directory_noreplace", _rename_noreplace)


def mock_loop_env(monkeypatch: pytest.MonkeyPatch, stops: list[str | None] | None = None) -> None:
    """Mock the loop's config/reporting/watch plumbing. `stops` = the values the fake `should_stop()`
    yields in order (None = keep going); default is always-None."""
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda x: x)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *a, **k: [])
    seq = list(stops or [])
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: seq.pop(0) if seq else None),
    )


def loop_config(**rl: object) -> SimpleNamespace:
    """A config carrying only what the entry loops read: `grid.backend` + the `rl.*` knobs passed as
    kwargs (max_steps, save_every, and the per-entry batch-size field).

    GRPO's loop reads `batch_per_replica` (the policy draw) and the DERIVED `prompts_per_data_replica`
    (the data draw). This is a raw `SimpleNamespace`, not a `GRPOConfig`, so it does not run the real
    property — set both here. Convenience: a test states how many prompts each data replica draws
    (`prompts_per_step`) and the group size; in the single-replica test topology `N_data == N_policy
    == 1`, so `batch_per_replica == prompts_per_step * group_size` and `prompts_per_data_replica ==
    prompts_per_step`. `n_mini == n_micro == 1` unless the test overrides them.

    Args:
        **rl: the `rl.*` fields the loop under test reads.

    Returns:
        A `SimpleNamespace` config with `grid.backend="local"` and the assembled `rl`.
    """
    if "prompts_per_step" in rl and "group_size" in rl:
        pps = int(rl.pop("prompts_per_step"))  # per-data-replica draw; the loop no longer has this field
        rl.setdefault("batch_per_replica", pps * int(rl["group_size"]))  # type: ignore[arg-type]
        rl.setdefault("prompts_per_data_replica", pps)
        rl.setdefault("n_mini", 1)
        rl.setdefault("n_micro", 1)
    # `needs_cached_old_logprobs` is a real GRPOConfig PROPERTY (n_mini > 1 and not
    # use_rollout_logprobs_for_loss), and this namespace does not run properties — so mirror its
    # value here, exactly as `prompts_per_data_replica` above is mirrored. A test that overrides
    # n_mini gets the ahead pass driven, which is what makes the driver's call order testable.
    # Dynamic-sampling knobs are plain GRPOConfig fields the loop reads directly; mirror their defaults.
    rl.setdefault("min_informative_groups", 0)
    rl.setdefault("min_informative_groups_frac", None)
    # `effective_min_informative_groups` is the real GRPOConfig PROPERTY the loop reads for its quota (the
    # literal count, or the fraction's ceil over groups_per_step); mirror it like `prompts_per_data_replica`.
    rl.setdefault("effective_min_informative_groups", int(rl["min_informative_groups"]))
    rl.setdefault("dynamic_sampling_max_extra_rollouts", 4)
    rl.setdefault("dynamic_sampling_refill_mode", "all")
    rl.setdefault("dynamic_sampling_quota_scope", "cell")
    rl.setdefault("dynamic_sampling_refill_dispatch", "per_round")
    rl.setdefault("dynamic_sampling_on_budget_exhausted", "raise")
    # `emergency_checkpoint_components` is a plain RunConfig field the loops read at a termination
    # trigger; mirror its default ("full" = upgrade to the exact-resume plan).
    rl.setdefault("emergency_checkpoint_components", "full")
    rl.setdefault(
        "needs_cached_old_logprobs",
        int(rl.get("n_mini", 1) or 1) > 1 and not bool(rl.get("use_rollout_logprobs_for_loss", False)),
    )
    return SimpleNamespace(grid=SimpleNamespace(backend="local"), rl=SimpleNamespace(**rl))
