# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Per-(kind, method, param) dispatch registry (Task A3, DESIGN §2.1)."""

from __future__ import annotations

import operator
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.posttrain.roles.core.dispatch import (
    DISPATCH,
    PROFILE_DISPATCH,
    bind_param_names,
    lookup,
    resolve_fn,
)
from cosmos_framework.posttrain.workers.dispatch import select_model_dp
from cosmos_framework.posttrain.workers.worker_group import RankAligned, Sharded, take_control_kwargs

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_dispatch_import_stays_torch_free() -> None:
    """Driver routing must not import the torch-backed trainer implementation."""
    repo_root = Path(__file__).resolve().parents[4]
    code = (
        "import sys;"
        "import cosmos_framework.posttrain.roles.core.dispatch;"
        "print('TORCH' if 'torch' in sys.modules else 'notorch')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "notorch"


def test_lookup_hit_returns_registered_sharded_spec():
    spec = lookup("trainer", "train", "batch")
    assert isinstance(spec, Sharded)
    assert spec.reduce == "cosmos_framework.posttrain.roles.trainer.llm:concat_batches"
    assert DISPATCH[("trainer", "train", "batch")] is spec


def test_lookup_hit_returns_registered_generate_split_spec():
    # Task C2: generate/prompts is now registered (was None pre-C2, when split_prompts didn't exist).
    spec = lookup("rollout", "generate", "prompts")
    assert isinstance(spec, Sharded)
    assert spec.split == "cosmos_framework.posttrain.roles.env.dataset:split_prompts"
    assert spec.reduce is None  # split-only: FanOut edge, never a reduce
    assert DISPATCH[("rollout", "generate", "prompts")] is spec


def test_nft_profile_overrides_generic_sharded_edges_with_exact_rank_contracts() -> None:
    expected = {
        ("rollout", "generate", "prompts"),
        ("reward", "score_with_provider_report", "trajectories"),
        ("buffer", "put_samples", "trajectories"),
        ("trainer", "train", "batch"),
    }
    assert set(PROFILE_DISPATCH["nft"]) == expected
    for kind, method, param in expected:
        assert isinstance(lookup(kind, method, param, dispatch_profile="nft"), RankAligned)
    assert isinstance(lookup("trainer", "train", "batch"), Sharded), "GRPO default must stay unchanged"


def test_unknown_dispatch_profile_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown dispatch profile"):
        lookup("trainer", "train", "batch", dispatch_profile="typo")


def test_lookup_miss_returns_none():
    assert lookup("nope", "nope", "nope") is None
    # The reward edge must stay OUT of the registry: its producer is the rollout, whose `ranks` is
    # the vLLM tensor-parallel size in every entry but AlpaGym's, and `_dispatch_from_spec` rejects
    # a Sharded edge with `K_p > 1` before routing. Registering it once broke every TP>1 recipe.
    # Spell the param the way `score` actually binds it -- an earlier version of this line said
    # "trajs", which no registration could ever match, so the guard passed while the edge WAS
    # registered under "trajectories".
    assert lookup("reward", "score", "trajectories") is None
    buffer_spec = lookup("buffer", "put_samples", "trajectories")
    assert isinstance(buffer_spec, Sharded)
    assert buffer_spec.reduce == "cosmos_framework.posttrain.roles.trainer.llm:concat_batches"


@pytest.mark.parametrize(
    ("kind", "method", "param"),
    (
        ("reference", "score", "trajectories"),
        ("critic", "predict", "trajectories"),
        ("trainer", "prepare_batch", "trajectories"),
        ("critic", "install_targets", "targets"),
    ),
)
def test_ppo_model_ingress_uses_the_shared_dispatch_registry(kind: str, method: str, param: str) -> None:
    spec = lookup(kind, method, param)

    assert isinstance(spec, Sharded)
    assert spec.rank_partitioned is True
    assert resolve_fn(spec.select) is select_model_dp
    assert take_control_kwargs(spec, {}) == ({}, {})


def test_select_model_dp_partitions_an_ordered_batch_exactly_once() -> None:
    batch = list(range(8))

    shards = [select_model_dp(batch, dp_rank=rank, dp_world=4) for rank in range(4)]

    assert shards == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert [item for shard in shards for item in shard] == batch


def test_resolve_fn_imports_module_attr_ref():
    assert resolve_fn("operator:add") is operator.add


def test_resolve_fn_round_trips_registered_reduce_and_split():
    """The registry ships STRING refs (`"module:attr"`) so the driver stays torch-free; the actor
    resolves them via `resolve_fn`. Prove both registered strings actually import to the real
    Task-C2 helpers — a typo or a moved helper would fail loud here, not silently at first sync."""
    from cosmos_framework.posttrain.roles.env.dataset import split_prompts
    from cosmos_framework.posttrain.roles.trainer.llm import concat_batches

    assert resolve_fn(DISPATCH[("trainer", "train", "batch")].reduce) is concat_batches
    assert resolve_fn(DISPATCH[("rollout", "generate", "prompts")].split) is split_prompts


def test_resolve_fn_rejects_malformed_ref():
    with pytest.raises(ValueError):
        resolve_fn("noclon")


def test_bind_param_names_trainer_train_positional():
    assert bind_param_names("trainer", "train", (object(),), {}) == ["batch"]


def test_bind_param_names_kwargs_bind_by_own_name():
    assert bind_param_names("rollout", "generate", (), {"prompts": 1}) == ["prompts"]


def test_bind_param_names_unknown_method_positional_unbindable():
    assert bind_param_names("nope", "nope", (object(), object()), {}) == [None, None]


def test_every_dispatch_key_is_covered_by_signatures():
    """Guard (positional-call tier-2 misses): every ``DISPATCH`` key ``(kind, method, param)`` must
    have a ``_SIGNATURES[(kind, method)]`` tuple that INCLUDES ``param``. Otherwise a positional call
    site's arg can never bind to that param name (``bind_param_names`` returns None for it), so the
    registry default silently never fires and the edge falls through to the bare path."""
    from cosmos_framework.posttrain.roles.core.dispatch import _SIGNATURES

    keys = list(DISPATCH)
    keys.extend(key for profile in PROFILE_DISPATCH.values() for key in profile)
    for kind, method, param in keys:
        sig = _SIGNATURES.get((kind, method))
        assert sig is not None, (
            f"DISPATCH has ({kind!r},{method!r},{param!r}) but _SIGNATURES lacks ({kind!r},{method!r})"
        )
        assert param in sig, f"_SIGNATURES[({kind!r},{method!r})]={sig} does not cover param {param!r}"
