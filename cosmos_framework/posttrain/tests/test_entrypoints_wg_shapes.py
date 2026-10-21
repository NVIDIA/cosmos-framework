# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Entry-point WorkerGroup shape wiring (spec 2026-07-22 §2.2).

The GRPO / DPO entries must declare their SPMD helper roles (env / reward / buffer) at ranks=1. NFT's
collective-coupled WFM roles instead share one exact-rank mesh behind an algorithm dispatch profile. In the
fallback colocated topology, rollout and trainer share one K; explicit asymmetric shapes may give
them independent rank widths. A helper behind the registered Sharded generate/train edge is rejected
at dispatch when K_p>1, so a helper mistakenly left at model width crashes the default CLI run on its
first edge. These are cheap, Ray-free assertions over the factored `declare_wgs` helpers — no cluster,
no torch, no bind.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.entrypoints.dpo import declare_wgs as declare_dpo_wgs
from cosmos_framework.posttrain.entrypoints.grpo import declare_wgs as declare_grpo_wgs
from cosmos_framework.posttrain.entrypoints.nft import declare_wgs as declare_nft_wgs


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_declares_uniform_collective_mesh_and_scoped_dispatch() -> None:
    shapes = {name: (1, 8) for name in ("data", "rollout", "rm", "replay", "policy")}
    wgs = declare_nft_wgs(8, shapes=shapes, policy_parallelism=(2, 4))
    assert {(worker.replicas, worker.ranks) for worker in wgs.values()} == {(1, 8)}
    assert {worker.dispatch_profile for worker in wgs.values()} == {"nft"}
    assert wgs["policy"].model_parallelism == (2, 4)
    assert wgs["rm"].needs_bulk is True


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_helpers_are_ranks1_and_model_wgs_carry_tp():
    """GRPO: data / rm / replay are ranks=1 SPMD helpers; rollout + policy carry ranks=TP."""
    tp = 4
    wg = declare_grpo_wgs(tp)
    assert {wg["data"].ranks, wg["rm"].ranks, wg["replay"].ranks} == {1}  # SPMD helpers
    assert wg["rollout"].ranks == tp and wg["policy"].ranks == tp  # model-bearing WGs span the mesh
    # kinds are what the guard / weight-sync bind key off — assert the role tags too.
    assert wg["rollout"].role == "rollout" and wg["policy"].role == "trainer"
    assert wg["data"].role == "env" and wg["rm"].role == "reward" and wg["replay"].role == "buffer"


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_tp1_degenerates_to_uniform_ranks1():
    """At TP=1 every WG is ranks=1 (a single-cell loop): the ranks=1 helpers and the model WGs coincide,
    so the shape wiring stays valid with no Sharded K_p>1 edge to reject."""
    wg = declare_grpo_wgs(1)
    assert all(w.ranks == 1 for w in wg.values())


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_env_helper_is_ranks1_while_trainer_carries_ranks():
    """DPO: the `data` env helper is ranks=1 (its mirrored rank lanes must not double-count behind the
    Sharded train edge); the `policy` trainer FSDP-shards across `ranks`."""
    ranks = 4
    wg = declare_dpo_wgs(ranks)
    assert wg["data"].ranks == 1 and wg["data"].role == "env"
    assert wg["policy"].ranks == ranks and wg["policy"].role == "trainer"


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_local_ranks1_is_unchanged():
    """The local backend forces ranks=1 for every role (torchrun carries parallelism); declare_wgs(1)
    must therefore leave both roles at ranks=1 — byte-identical to the pre-migration default path."""
    wg = declare_dpo_wgs(1)
    assert wg["data"].ranks == 1 and wg["policy"].ranks == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_wg_declares_needs_bulk_exactly_when_prefix_filtering_is_on():
    """The reward role must declare `needs_bulk` iff it will actually compute on a bulk field.

    Found by RUNNING the example, not by this suite: the CPU tests never cross a Ray edge, so they
    exercise the annotator on materialized tensors and pass either way. Under Ray the codec leaves
    `transition.action` as a lazy `BulkRef` at any ingress whose WG declared `needs_bulk=False`, and
    shared-prefix filtering — which moved to the reward stage because it is group-relative — reads
    exactly that field. The R=2 vLLM run died on its first batch with
    `TypeError: 'BulkRef' object is not iterable`.

    BOTH directions are asserted. `needs_bulk=True` materializes EVERY bulk field, obs
    representations included, so pinning it on unconditionally would drag raw video/images into a
    reward worker that only wanted token ids — the traffic DESIGN §3.4 exists to prevent. And the
    predicate must be the one `build_reward_annotators` uses, or the shape and the body disagree:
    `min_filter_prefix_tokens=-1` is truthy but DISABLED, so a truthiness test would pull bulk for a
    filter that does nothing.
    """
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import build_reward_annotators, prefix_filter_enabled
    from cosmos_framework.posttrain.config.rl import GRPOConfig
    from cosmos_framework.posttrain.entrypoints.grpo import declare_wgs

    def _FakePacker():  # noqa: N802 - a stand-in for the tokenizer owner; no download in an L0
        return SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=7))

    assert declare_wgs(1)["rm"].needs_bulk is False, "the default reward role must not pull bulk"
    assert declare_wgs(1, rm_needs_bulk=True)["rm"].needs_bulk is True

    # The predicate the entry passes must agree with the one that decides whether to build the filter.
    for threshold, expected in ((None, False), (0, False), (-1, False), (1, True), (4, True)):
        rl = GRPOConfig(group_size=2)
        rl.min_filter_prefix_tokens = threshold
        assert prefix_filter_enabled(rl) is expected, f"min_filter_prefix_tokens={threshold!r}"
        annotators = build_reward_annotators(rl, "gpt2", packer=_FakePacker())
        assert (len(annotators) == 2) is expected, (
            f"min_filter_prefix_tokens={threshold!r}: prefix_filter_enabled says {expected} but "
            f"{len(annotators)} annotator(s) were built — the WG shape and the reward body disagree"
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_annotators_resolve_the_same_model_id_the_trainer_loads():
    """`model_hf.name_or_path` defaults to "" — the reward stage must use the SAME fallback.

    `build_trainer` falls back to `config.rollout.name_or_path` because the trainer and the inference
    engine have to load the same weights or the weight sync is meaningless. The shared-prefix
    annotator needs that model's tokenizer for its EOS, and reading `config.model_hf.name_or_path`
    raw handed `init_data_packer("")` to it on the default config: a confusing failure a long way
    from its cause, and only when prefix filtering is enabled. Worse than failing would be a
    DIFFERENT model resolving here than the trainer loads — the filter would then trim by another
    tokenizer's EOS.
    """
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import effective_model_id

    default_root = SimpleNamespace(
        model_hf=SimpleNamespace(name_or_path=""),  # the DEFAULT — not a contrived value
        rollout=SimpleNamespace(name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    )
    assert effective_model_id(default_root) == "Qwen/Qwen2.5-0.5B-Instruct"

    # An explicit trainer model wins; the two are allowed to differ and the trainer's is the truth.
    explicit = SimpleNamespace(
        model_hf=SimpleNamespace(name_or_path="my/trainer-model"),
        rollout=SimpleNamespace(name_or_path="my/engine-model"),
    )
    assert effective_model_id(explicit) == "my/trainer-model"

    neither = SimpleNamespace(model_hf=SimpleNamespace(name_or_path=""), rollout=SimpleNamespace(name_or_path=""))
    assert effective_model_id(neither) == "", "empty is reported as empty, not guessed at"


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_per_role_shapes_from_grid_shapes():
    """§8.2: with `grid.shapes` set, policy and rollout carry their OWN (R, K); the experience roles
    (data/rm/replay) derive at R = R_rollout, K = 1 — that keeps `data → rollout` at op="one" and
    leaves `replay → policy` the single asymmetric edge."""
    from cosmos_framework.posttrain.config.rl import GridConfig
    from cosmos_framework.posttrain.entrypoints.grpo import declare_wgs

    shapes = GridConfig(shapes={"policy": [1, 8], "rollout": [4, 2]}).shapes  # validated -> tuples
    wg = declare_wgs(ranks=2, replicas=4, shapes=shapes, policy_parallelism=(2, 4))
    assert (wg["policy"].replicas, wg["policy"].ranks) == (1, 8)
    assert wg["policy"].model_parallelism == (2, 4)
    assert (wg["rollout"].replicas, wg["rollout"].ranks) == (4, 2)
    for helper in ("data", "rm", "replay"):
        assert (wg[helper].replicas, wg[helper].ranks) == (4, 1), f"{helper} derives at (R_rollout, 1)"
