# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Recipe → Manifest builders. Each derives roles from the entrypoint's own `ROLE_SPECS` and recipe
choices from the same typed config fields worker construction reloads."""

from __future__ import annotations

from cosmos_framework.posttrain.algorithm.reward import RewardProviderSpec
from cosmos_framework.posttrain.control import capabilities_builtin  # registers descriptors; also owns the impl keys
from cosmos_framework.posttrain.control.manifest import Edge, Manifest, Plane, RoleDecl
from cosmos_framework.posttrain.control.placement.gpu_pair import WeightRelationship


def dpo_manifest(role_specs: list[tuple[str, str]]) -> Manifest:
    """The DPO recipe topology, derived from the entry's `ROLE_SPECS` — the same list `run()` builds its
    WorkerGroups from, so adding or renaming a role changes what gets checked. Each `(name, kind)` maps
    to the `dpo_<kind>` capability; an unregistered one is a wiring error, which is how drift surfaces.

    The DATA edge stays declared here: the loop's data path is imperative (`data.get_samples` →
    `policy.train`), not derivable from the role list."""
    return Manifest(
        roles=tuple(RoleDecl(name=name, impl_key=f"dpo_{kind}") for name, kind in role_specs),
        # data→policy rides the DATA plane in-process (offline DPO): a Trajectory ('object') payload.
        # Tagged so the real recipe exercises the transport check (data-plane edges are always movable).
        edges=(
            Edge(
                src="data",
                dst="policy",
                plane=Plane.DATA,
                relationship="same_process",
                transport_capability="object",
            ),
        ),
    )


def sft_manifest(role_specs: list[tuple[str, str]]) -> Manifest:
    """Offline supervised data → policy; local/torchrun has no weight or rollout edge."""
    return Manifest(
        roles=tuple(RoleDecl(name=name, impl_key=f"sft_{kind}") for name, kind in role_specs),
        edges=tuple(
            Edge(name, "policy", Plane.DATA, relationship="same_process", transport_capability="object")
            for name, kind in role_specs
            if kind == "env"
        ),
    )


def grpo_manifest(
    role_specs: list[tuple[str, str]],
    source: str,
    ground_truth: bool,
    provider_spec: RewardProviderSpec,
    *,
    weight_relationship: WeightRelationship,
) -> Manifest:
    """The GRPO recipe topology, from the entry's `ROLE_SPECS` plus the config choices that change
    what the roles can do: which prompt source (`source`, and whether it emits the `reference`
    `ground_truth`) and which canonical fields the selected reward provider needs. Provider names
    remain run-local and never become process-global capability registrations."""
    env_key = f"grpo_env_{source}"
    if source != "arithmetic" and ground_truth:
        env_key += "_with_answers"
    rollout_key = "grpo_rollout_with_answers" if ground_truth else "grpo_rollout"
    impl_keys = {
        "env": env_key,
        "rollout": rollout_key,
        "reward": capabilities_builtin.grpo_reward_capability_key(provider_spec.needs),
    }
    return Manifest(
        roles=tuple(RoleDecl(name=name, impl_key=impl_keys.get(kind, f"grpo_{kind}")) for name, kind in role_specs),
        edges=(
            # The loop's actual experience flow. The rollout descriptor carries source fields forward
            # with the completion, so source/reward compatibility is checked without inventing a
            # direct data->rm edge that runtime never executes.
            # DATA transport is resolved by the WorkerGroup dispatch path. Do not claim one static
            # placement relationship here: replay and policy are in different colocations, while
            # experience roles may share a host under the current recipe.
            Edge("data", "rollout", Plane.DATA),
            Edge("rollout", "rm", Plane.DATA),
            Edge("rm", "replay", Plane.DATA),
            Edge("replay", "policy", Plane.DATA),
            # The entrypoint resolves this from the same GPU-placement config used to construct
            # DeviceGroups, so AOT validation and the live binder describe one weight edge.
            Edge(
                "policy",
                "rollout",
                Plane.WEIGHT,
                relationship=weight_relationship,
                transport_capability="tensor",
            ),
        ),
    )


def sim_manifest(
    *,
    sim_impl: str = capabilities_builtin.SIM_IMPL_KEY,
    policy_impl: str = capabilities_builtin.POLICY_IMPL_KEY,
) -> Manifest:
    """The sim-eval / data-collection topology.

    **ONE IMPL KEY, NOT THREE STRINGS THAT MATCH TODAY.** The defaults come from
    ``capabilities_builtin`` rather than being spelled here: one ``impl_key`` has to resolve the
    ``CapabilityDescriptor``, this ``RoleDecl``, and the body builder in
    ``roles/core/registry.py``. A fourth literal is how those drift.

    **ONE edge, not two.** The observation goes env->policy and the action chunk comes back as the
    RETURN VALUE of that same invocation, so it shares this edge's transport by construction. A
    second ``policy->env`` edge would declare a transport selection that nothing could ever act on
    -- there is no separate call to route -- and `bind_check` would then dutifully validate a
    fiction.

    ``relationship="host"``: the two roles are colocated per pod under ``host_affinity`` with
    ``host_single_thread=True`` (Isaac must build SimulationApp on the process main thread), so they
    share a machine and not a process. ``transport_capability="object"``: the payload is an
    ``ObsBundle`` of numpy arrays, never a live tensor.

    **Both tags are PROVENANCE here, not a check.** ``bind_check`` routes them through
    ``transport_capability_ok``, and on a DATA edge every combination passes -- including
    ``transport_capability="tensor"``, which would be a lie about this payload. They are declared
    because they are true and because the runtime mover selector reads the same pair. A green
    ``bind_check`` has NOT ruled out a transport problem on this edge, and nothing pins that.

    DECLARE-ONLY: outside its test, nothing calls ``bind_check`` on this manifest.
    """
    return Manifest(
        roles=(RoleDecl(name="env", impl_key=sim_impl), RoleDecl(name="policy", impl_key=policy_impl)),
        edges=(
            Edge(
                src="env",
                dst="policy",
                plane=Plane.DATA,
                relationship="host",
                transport_capability="object",
            ),
        ),
    )


def nft_manifest(role_specs: list[tuple[str, str]]) -> Manifest:
    """Synchronous colocated online-WFM topology.

    Rollout and trainer are separate bodies over one model instance. The weight
    edge is byte-free but load-bearing: its two-phase publication proves shared
    object identity and advances the rollout's committed model version.
    """
    return Manifest(
        roles=tuple(RoleDecl(name=name, impl_key=f"nft_{kind}") for name, kind in role_specs),
        edges=(
            Edge("data", "rollout", Plane.DATA, relationship="same_process", transport_capability="object"),
            Edge("rollout", "rm", Plane.DATA, relationship="same_process", transport_capability="object"),
            Edge("rm", "replay", Plane.DATA, relationship="same_process", transport_capability="object"),
            Edge("replay", "policy", Plane.DATA, relationship="same_process", transport_capability="object"),
            Edge("policy", "rollout", Plane.WEIGHT, relationship="same_process", transport_capability="tensor"),
        ),
    )


def ppo_manifest(
    role_specs: list[tuple[str, str]],
    *,
    ground_truth: bool,
    reward_needs_ground_truth: bool,
    weight_relationship: WeightRelationship,
) -> Manifest:
    """Synchronous PPO role graph over the shared data and weight planes."""
    impl_keys = {
        "env": "ppo_env_with_answers" if ground_truth else "ppo_env",
        "rollout": "ppo_rollout_with_answers" if ground_truth else "ppo_rollout",
        "reward": "ppo_reward_with_answers" if reward_needs_ground_truth else "ppo_reward",
        "buffer": "ppo_buffer",
        "reference": "ppo_reference",
        "critic": "ppo_critic",
        "trainer": "ppo_trainer",
    }
    return Manifest(
        roles=tuple(RoleDecl(name=name, impl_key=impl_keys[kind]) for name, kind in role_specs),
        edges=(
            Edge("data", "rollout", Plane.DATA),
            Edge("rollout", "rm", Plane.DATA),
            Edge("rm", "replay", Plane.DATA),
            Edge("replay", "reference", Plane.DATA),
            Edge("reference", "critic", Plane.DATA),
            Edge("critic", "policy", Plane.DATA),
            Edge("policy", "critic", Plane.DATA),
            Edge(
                "policy",
                "rollout",
                Plane.WEIGHT,
                relationship=weight_relationship,
                transport_capability="tensor",
            ),
        ),
    )


def joint_ppo_manifest(role_specs: list[tuple[str, str]], *, weight_relationship: WeightRelationship) -> Manifest:
    """The collector owns feedback association; its sampled decisions enter reward and replay."""
    return Manifest(
        roles=tuple(RoleDecl(name=name, impl_key=f"joint_ppo_{kind}") for name, kind in role_specs),
        edges=(
            *(Edge(name, "collector", Plane.DATA) for name, kind in role_specs if kind == "sim"),
            Edge("collector", "rollout", Plane.DATA),
            Edge("collector", "rm", Plane.DATA),
            Edge("rm", "replay", Plane.DATA),
            Edge("replay", "policy", Plane.DATA),
            Edge("policy", "rollout", Plane.WEIGHT, relationship=weight_relationship, transport_capability="tensor"),
        ),
    )
