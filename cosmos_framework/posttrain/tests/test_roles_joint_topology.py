# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real Joint PPO collection, training and evaluation over independently bound roles."""

from __future__ import annotations

import os
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.control.loop.decision_window import collect_decision_window, evaluate_decision_cases
from cosmos_framework.posttrain.control.placement import (
    ActorGroup,
    DeviceGroup,
    bind_device_groups,
    validate_peer_chain,
)
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


@pytest.fixture
def bind_joint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ray

    from cosmos_framework.posttrain.checkpoint import collect_rng_state, restore_rng_state
    from cosmos_framework.posttrain.embodied import embodiment, registry
    from cosmos_framework.posttrain.embodied.sinks.null import NullSink

    registry.register_sink("null", NullSink)
    rng_state = collect_rng_state()
    monkeypatch.setattr(embodiment, "MAPS", dict(embodiment.MAPS))
    monkeypatch.setattr(registry, "_SIMS", dict(registry._SIMS))
    monkeypatch.setattr(registry, "_SINKS", dict(registry._SINKS))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    cleanup = ExitStack()
    cleanup.callback(restore_rng_state, rng_state)
    cleanup.callback(ray.shutdown)

    def bind(topology: str, *, failure_once: bool = False, permute_peer_routes: bool = False):
        from cosmos_framework.posttrain.tests.helpers.joint_topology import setup_roles

        replicas = 1 if topology == "local" else 2
        recipe = JointPPOConfig(
            num_envs=2 * replicas, window_decisions=2, global_minibatch=4, microbatch=1, max_steps=2, save_every=0
        )
        units = {
            "sim": roles.env("sim", replicas=replicas),
            "collector": roles.env("collector", replicas=replicas, needs_bulk=True),
            "rollout": roles.rollout("rollout", replicas=replicas),
            "reward": roles.reward("reward", replicas=replicas),
            "replay": roles.buffer("replay", replicas=replicas),
            "policy": roles.trainer("policy"),
        }
        if topology == "separate":
            groups = (("sim",), ("collector", "reward", "replay"), ("rollout",), ("policy",))
        elif topology == "sim_separate":
            groups = (("sim",), ("collector", "rollout", "reward", "replay"), ("policy",))
        elif topology == "reentrant":
            groups = (("sim", "rollout"), ("collector", "reward", "replay"), ("policy",))
        else:
            groups = (("sim", "collector", "rollout", "reward", "replay"), ("policy",))
        actors = tuple(
            ActorGroup(
                tuple(units[name] for name in names),
                partial(setup_roles, names, recipe, str(tmp_path), failure_once),
            )
            for names in groups
        )
        bind_device_groups(
            [DeviceGroup(actors, device="cpu")],
            backend="local" if topology == "local" else "ray",
            temp_dir=f"/proc/{os.getpid()}/cwd/ray",
            threshold=4,
        )
        cleanup.callback(lambda: resolve_all(units["policy"].finalize_checkpoints()))
        for name in ("rollout", "sim", "collector"):
            cleanup.callback(lambda role: resolve_all(role.close()), units[name])
        if permute_peer_routes:
            from cosmos_framework.posttrain.workers.grid import dp_aligned

            def permuted(producer_shape, consumer_shape, meta):
                return {
                    cell: [((replica + 1) % producer_shape[0], rank) for replica, rank in targets]
                    for cell, targets in dp_aligned(producer_shape, consumer_shape, meta).items()
                }

            monkeypatch.setattr(units["collector"]._backend, "align", permuted)
        validate_peer_chain((units["sim"], units["collector"], units["rollout"]))
        resolve_all(units["collector"].bind_rollout(units["rollout"]))
        return recipe, units

    try:
        yield bind
    finally:
        cleanup.close()


@pytest.mark.parametrize("topology", ["local", "colocated", "sim_separate", "separate"])
def test_joint_roles_collect_update_and_evaluate_through_public_peers(bind_joint, topology: str) -> None:
    recipe, units = bind_joint(topology)
    collector, simulator, rollout, policy = (units[name] for name in ("collector", "sim", "rollout", "policy"))
    before = resolve_all(collector.state_dict())
    learner_before = resolve_all(policy.evaluation_state())
    rollout_rng = resolve_all(rollout.rng_state())
    evaluated = resolve_all(
        evaluate_decision_cases(collector, simulator, version=0, cases=[{"case_id": 0}], seed=17, decisions=2)
    )
    assert all(row["training_session_unchanged"] and len(row["episodes"]) == 1 for row in evaluated)
    assert resolve_all(collector.state_dict()) == before
    assert resolve_all(rollout.rng_state()) == rollout_rng
    assert resolve_all(policy.evaluation_state()) == learner_before
    assert all(row["evaluation_sessions_closed"] and row["turn"] == 0 for row in resolve_all(simulator.state()))

    samples = collect_decision_window(collector, simulator, version=0, decisions=recipe.window_decisions)
    resolve_all(units["replay"].put_samples(units["reward"].score(samples)))
    receipt = resolve_identical(
        policy.prepare_ppo_targets(units["replay"].get_exact_samples()), operation="test targets"
    )
    if topology == "local":
        parameters_before = {name: value.detach().clone() for name, value in policy.body.model.net.named_parameters()}
    updated = resolve_identical(policy.train_batch(receipt["batch_id"]), operation="test joint update")
    resolve_all(policy.clear_batch(receipt["batch_id"]))
    assert receipt["decisions"] == recipe.decisions
    assert updated["optimizer_step"] == recipe.decisions // recipe.global_minibatch
    assert all(delta > 0 for delta in updated["parameter_update_l2"].values())
    if topology == "local":
        for prefix in ("backbone.", "actor.", "value."):
            assert any(
                not torch.equal(parameters_before[name], value)
                for name, value in policy.body.model.net.named_parameters()
                if name.startswith(prefix)
            )
    assert all(row["next_window"] == 1 for row in resolve_all(collector.state_dict()))
    assert {row["replica"] for row in resolve_all(collector.state_dict())} == set(range(recipe.num_envs // 2))
    resolve_all(collector.close())
    assert all(not row["closed"] for row in resolve_all(simulator.state()))
    resolve_all(simulator.close())
    assert all(row["closed"] for row in resolve_all(simulator.state()))


@pytest.mark.parametrize("permute_peer_routes", [False, True])
def test_joint_peer_chain_rejects_actor_reentry_before_collection(bind_joint, permute_peer_routes: bool) -> None:
    expected = "actor wait cycle across replicas" if permute_peer_routes else "re-enters a blocked actor"
    with pytest.raises(ValueError, match=expected):
        bind_joint("reentrant", permute_peer_routes=permute_peer_routes)


@pytest.mark.parametrize("topology", ["local", "separate"])
def test_joint_window_failure_cleans_every_replica_and_can_collect_again(bind_joint, topology: str) -> None:
    recipe, units = bind_joint(topology, failure_once=True)
    collector, simulator = units["collector"], units["sim"]
    # Keep the original exception alive while the public flow recovers.
    with pytest.raises(Exception, match="injected topology simulator failure") as _retained:
        collect_decision_window(collector, simulator, version=0, decisions=recipe.window_decisions)
    assert all(row["next_window"] == 0 for row in resolve_all(collector.state_dict()))
    samples = collect_decision_window(collector, simulator, version=0, decisions=recipe.window_decisions)
    resolve_all(units["replay"].put_samples(units["reward"].score(samples)))
    receipt = resolve_identical(
        units["policy"].prepare_ppo_targets(units["replay"].get_exact_samples()), operation="recovered targets"
    )
    resolve_all(units["policy"].clear_batch(receipt["batch_id"]))
    assert all(row["next_window"] == 1 for row in resolve_all(collector.state_dict()))
    resolve_all(simulator.close())
    assert all(row["closed"] for row in resolve_all(simulator.state()))


@pytest.mark.parametrize(("evaluation_enabled", "transport"), [(True, "direct"), (True, "gateway"), (False, "direct")])
def test_sim_role_window_surface_and_capability_are_interchangeable(
    bind_joint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str, evaluation_enabled: bool
) -> None:
    import sys

    from cosmos_framework.posttrain.embodied import wire
    from cosmos_framework.posttrain.embodied.sim import SimProtocolError
    from cosmos_framework.posttrain.roles.env.collector import CollectorBody
    from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
    from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody
    from cosmos_framework.posttrain.roles.env.sim_role import require_sim_capability
    from cosmos_framework.posttrain.tests.helpers.joint_topology import (
        SIM_NAME,
        TopologyEnvironment,
        TopologyPolicy,
        register_test_simulator,
    )

    register_test_simulator()
    monkeypatch.setattr(wire, "SHM_DIR", str(tmp_path))
    monkeypatch.setenv("B1K_SHM_DIR", str(tmp_path))
    extras = ("decision_chunks", "fixed_case_evaluation") if evaluation_enabled else ("decision_chunks",)
    options = dict(
        sim=SIM_NAME, sink="null", embodiment=SIM_NAME, sim_options={"output": str(tmp_path), "extras": extras}
    )
    body = (
        CollectorBody(**options)
        if transport == "direct"
        else SimGatewayBody(
            **options,
            sim_python=sys.executable,
            sim_gpu="0",
            sim_ipc_dir=f"/proc/{os.getpid()}/cwd",
            preload=("cosmos_framework.posttrain.tests.helpers.joint_topology",),
        )
    )
    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=4, microbatch=1)
    collector, simulator, rollout = roles.env("collector"), roles.env("sim"), roles.rollout("rollout")
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    (
                        ActorGroup(
                            (collector, simulator, rollout),
                            lambda: {
                                "collector": PPOWindowCollector(recipe, 0, 2, environment=TopologyEnvironment()),
                                "sim": body,
                                "rollout": TopologyPolicy(),
                            },
                        ),
                    ),
                    device="cpu",
                )
            ],
            backend="local",
        )
        resolve_all(collector.bind_rollout(rollout))
        require_sim_capability(body, "decision_chunks")
        with pytest.raises(ValueError, match="episode work units"):
            simulator.run_window([{"invalid": "episode unit"}], policy=collector, version=0, decisions=2)
        samples = collect_decision_window(collector, simulator, version=0, decisions=2).result()
        assert len(samples) == recipe.decisions
        assert all(sample.transitions[0].obs.representations["raw"]["image"].shape == (64, 64, 3) for sample in samples)
        if evaluation_enabled:
            require_sim_capability(body, "fixed_case_evaluation")
            result = evaluate_decision_cases(
                collector, simulator, version=0, cases=[{"case_id": 0}], seed=17, decisions=2
            ).result()
            assert result["training_session_unchanged"] and len(result["episodes"]) == 1
        else:
            with pytest.raises(SimProtocolError, match="fixed_case_evaluation"):
                require_sim_capability(body, "fixed_case_evaluation")
    finally:
        body.close()
        if transport == "gateway":
            assert body._proc.poll() is not None
        if collector.body is not None:
            resolve_all(collector.close())
            resolve_all(rollout.close())
