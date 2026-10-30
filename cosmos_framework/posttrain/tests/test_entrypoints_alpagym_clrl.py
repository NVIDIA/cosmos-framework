# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the AlpaGym closed-loop entry's WorkerGroup shapes.

AlpaGym's rollout is an HF model behind an InferenceEngine and does not tensor-parallel shard,
while its trainer needs FSDP. The two meshes are therefore TRANSPOSED — policy (1,4), rollout
(4,1). `entrypoints/grpo.py::declare_wgs` can now express those model shapes, but it derives every
experience role at rollout shape; AlpaGym's smaller scene source cannot be sharded that way, so this
entry keeps a single-cell data WorkerGroup and uses the registered prompt split.

The transposed shape is not a workaround. `comm/weight_pairing.py::full_tensor_actor_targets` pairs
cells by flat bundle index, so policy cell (0,k) pushes to rollout cell (k,0) on the same GPU, and
its docstring names transposed meshes as the case it exists for.

FSDP on the policy side is a REQUIREMENT here, not a tuning choice: as measured, an
unsharded trainer is 54.62 GiB and cannot sit beside the rollout's 20.64 GiB on one 80 GiB card.
The symmetric alternative — policy (4,1), one full replica per GPU — is arithmetically impossible,
which is what `test_policy_is_sharded_not_replicated` guards.
"""

from __future__ import annotations

from typing import Any

import pytest

from cosmos_framework.posttrain.comm.weight_pairing import full_tensor_actor_targets
from cosmos_framework.posttrain.entrypoints.alpagym_clrl import (
    ROLE_SPECS,
    AlpagymSection,
    EntryConfig,
    EpisodeScorer,
    _updates_per_step,
    declare_wgs,
)
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, ObsBundle, Status, Trajectory, Transition


def _entry_config(*, prompts: int, group_size: int, ticks: int, gpus: int, placement: str) -> EntryConfig:
    return EntryConfig(
        alpagym=AlpagymSection(resolved_config_path="unused", project_root="unused"),
        gpus=gpus,
        steps=1,
        prompts_per_step=prompts,
        group_size=group_size,
        ticks_per_episode=ticks,
        placement=placement,
    )


def _episode(trajectory_id: str, reward: float) -> Trajectory:
    transitions = [
        Transition(
            obs=ObsBundle(id="scene"),
            reward={"term/alpasim": reward, FINAL_REWARD_KEY: reward},
            done=index == 1,
        )
        for index in range(2)
    ]
    return Trajectory(
        transitions=transitions,
        id=trajectory_id,
        group_id="scene",
        status=Status.COMPLETED,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_episode_scorer_preserves_env_rewards_and_splits_after_group_annotation() -> None:
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    episodes = [_episode("low", 0.0), _episode("high", 1.0)]
    original_rewards = [transition.reward for episode in episodes for transition in episode.transitions]
    scorer = EpisodeScorer(annotators=(group_relative(group_size=2),))

    ticks = scorer.score(episodes)

    assert [tick.id for tick in ticks] == ["low_t0", "low_t1", "high_t0", "high_t1"]
    assert [tick.transitions[0].advantage for tick in ticks] == pytest.approx(
        [-1.0, -1.0, 1.0, 1.0],
        abs=3e-6,
    )
    assert [transition.reward for episode in episodes for transition in episode.transitions] == original_rewards
    assert all(
        transition.reward == {"term/alpasim": reward, FINAL_REWARD_KEY: reward}
        for episode, reward in zip(episodes, (0.0, 1.0), strict=True)
        for transition in episode.transitions
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_episode_scorer_rolls_back_partial_annotation_failure() -> None:
    episodes = [_episode("low", 0.0), _episode("high", 1.0)]

    def _fail_after_write(batch: list[Trajectory]) -> None:
        transition = batch[0].transitions[0]
        transition.advantage = 99.0
        transition.algo_extra["partial"] = True
        raise RuntimeError("annotation failed")

    scorer = EpisodeScorer(annotators=(_fail_after_write,))

    with pytest.raises(RuntimeError, match="annotation failed"):
        scorer.score(episodes)

    assert all(
        transition.advantage is None and transition.algo_extra == {}
        for episode in episodes
        for transition in episode.transitions
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_updates_per_step_rejects_fractional_policy_batches() -> None:
    config = _entry_config(prompts=1, group_size=3, ticks=1, gpus=2, placement="colocated")

    with pytest.raises(ValueError, match=r"batch_size=3.*policy width=2"):
        _updates_per_step(config)


@pytest.mark.L0
@pytest.mark.CPU
def test_updates_per_step_uses_the_full_tick_batch() -> None:
    config = _entry_config(prompts=1, group_size=2, ticks=4, gpus=4, placement="colocated")

    assert _updates_per_step(config) == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_transposed_mesh_shapes():
    wgs = declare_wgs(gpus=4)
    assert (wgs["policy"].replicas, wgs["policy"].ranks) == (1, 4)
    assert (wgs["rollout"].replicas, wgs["rollout"].ranks) == (4, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_is_sharded_not_replicated():
    """One FSDP replica across all GPUs — never one full replica per GPU (see module docstring)."""
    wgs = declare_wgs(gpus=4)
    assert wgs["policy"].replicas == 1
    assert wgs["policy"].ranks == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_helper_shapes_keep_the_loop_free_of_dispatch_markers():
    """`rm`/`replay` MATCH the rollout so no edge needs a call-site `Sharded`.

    This is what keeps the orchestration loop at the altitude DESIGN 2.1 requires -- mesh shapes
    must not appear in it. `rm`/`replay` at (1,1) would leave `rollout -> rm` with mismatched
    replica counts, which the bare-handle path rejects, forcing a marker into the loop. `data` is
    the exception: the scene source is smaller than `gpus`, and its edge is covered by the
    registered `("rollout","generate","prompts")` split. Ranks stay 1 everywhere -- a multi-rank
    producer behind a Sharded edge is rejected, its mirrored lanes double-counting as units.
    """
    wgs = declare_wgs(gpus=4)
    assert (wgs["data"].replicas, wgs["data"].ranks) == (1, 1)
    for helper in ("rm", "replay"):
        assert (wgs[helper].replicas, wgs[helper].ranks) == (wgs["rollout"].replicas, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_each_policy_cell_pairs_with_the_rollout_cell_on_its_own_gpu():
    wgs = declare_wgs(gpus=4)
    targets = full_tensor_actor_targets(
        (wgs["policy"].replicas, wgs["policy"].ranks),
        (wgs["rollout"].replicas, wgs["rollout"].ranks),
    )
    # Targets are TUPLES: the route is one-to-many since upstream widened it to fan a policy
    # actor out over several rollout actors when the pools differ in size. Equal products,
    # as here, still give exactly one rollout cell per policy cell.
    assert targets == {(0, 0): ((0, 0),), (0, 1): ((1, 0),), (0, 2): ((2, 0),), (0, 3): ((3, 0),)}


@pytest.mark.L0
@pytest.mark.CPU
def test_products_match_so_the_bind_guard_passes():
    """`full_tensor_actor_targets` requires R_p*K_p == R_r*K_r; it is a bijection over the GPUs."""
    wgs = declare_wgs(gpus=4)
    assert wgs["policy"].replicas * wgs["policy"].ranks == wgs["rollout"].replicas * wgs["rollout"].ranks


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("gpus", [1, 2, 8])
def test_shape_scales_with_the_gpu_count(gpus):
    """The pairing bijection must hold at every width, not just the 4-GPU case we run today."""
    wgs = declare_wgs(gpus=gpus)
    assert (wgs["policy"].replicas, wgs["policy"].ranks) == (1, gpus)
    assert (wgs["rollout"].replicas, wgs["rollout"].ranks) == (gpus, 1)
    targets = full_tensor_actor_targets((1, gpus), (gpus, 1))
    assert len(targets) == gpus
    assert set(targets.values()) == {((k, 0),) for k in range(gpus)}


@pytest.mark.L0
@pytest.mark.CPU
def test_role_kinds_match_the_declared_specs():
    """Every declared role is built, with the kind ROLE_SPECS names — the shapes and the setup
    closure come from one list, so the roles built cannot drift from the roles declared. (This
    entry has no manifest; see the TODO beside ROLE_SPECS.)"""
    wgs = declare_wgs(gpus=4)
    assert set(wgs) == {name for name, _ in ROLE_SPECS}
    for name, kind in ROLE_SPECS:
        assert wgs[name].role == kind


@pytest.mark.L0
@pytest.mark.CPU
def test_disaggregated_splits_the_gpus_and_keeps_the_bijection():
    """Trainer and rollout take half the GPUs each -- the layout cosmos-rl ran.

    The mesh stays TRANSPOSED, so `full_tensor_actor_targets`' `R_p x K_p == R_r x K_r` still holds and
    each policy cell keeps exactly one rollout peer; what changes is that the peer is on ANOTHER
    GPU, which is why `run` asks for the NIXL transport there. An odd GPU count cannot split evenly
    and is rejected rather than silently starving one side.
    """
    wgs = declare_wgs(gpus=4, placement="disaggregated")
    assert (wgs["policy"].replicas, wgs["policy"].ranks) == (1, 2)
    assert (wgs["rollout"].replicas, wgs["rollout"].ranks) == (2, 1)
    for helper in ("rm", "replay"):
        assert (wgs[helper].replicas, wgs[helper].ranks) == (2, 1)
    assert (wgs["data"].replicas, wgs["data"].ranks) == (1, 1)
    assert wgs["policy"].replicas * wgs["policy"].ranks == wgs["rollout"].replicas * wgs["rollout"].ranks

    with pytest.raises(ValueError, match="odd"):
        declare_wgs(gpus=3, placement="disaggregated")
    with pytest.raises(ValueError, match="colocated"):
        declare_wgs(gpus=4, placement="nonsense")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("gpu_placement", "expected_device_groups", "expected_transport"),
    [
        ("colocated", [["alpagym_experience", "alpagym_learner"]], None),
        ("disaggregated", [["alpagym_experience"], ["alpagym_learner"]], "nixl"),
    ],
)
def test_run_lowers_placement_to_explicit_device_groups(
    monkeypatch: pytest.MonkeyPatch,
    gpu_placement: str,
    expected_device_groups: list[list[str]],
    expected_transport: str | None,
) -> None:
    from cosmos_framework.posttrain.control import placement
    from cosmos_framework.posttrain.entrypoints import alpagym_clrl

    captured: dict[str, Any] = {}

    def _capture(device_groups: list[Any], **kwargs: Any) -> None:
        captured["device_groups"] = device_groups
        captured["kwargs"] = kwargs

    monkeypatch.setattr(placement, "bind_device_groups", _capture)
    monkeypatch.setattr(alpagym_clrl, "make_setup", lambda *_args, **_kwargs: lambda: {})
    monkeypatch.setattr(alpagym_clrl, "loop", lambda _wgs, _config: {"ok": True})
    config = _entry_config(prompts=1, group_size=1, ticks=4, gpus=4, placement=gpu_placement)

    assert alpagym_clrl.run(config) == {"ok": True}
    assert [
        [actor_group.name for actor_group in device_group.actor_groups] for device_group in captured["device_groups"]
    ] == expected_device_groups
    assert all(device_group.device == "gpu" for device_group in captured["device_groups"])
    assert captured["kwargs"]["weight_sync"] is True
    assert captured["kwargs"]["weight_sync_transport"] == expected_transport
    assert "num_gpus_each" not in captured["kwargs"]
