# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""End-to-end: the B.1 GRPO loop over real components, tiny model, local backend.

Synthetic variance reward (by completion length parity) guarantees nonzero advantage
so we exercise the full loop + a real optimizer step. This is a flow test, NOT a
convergence test (DESIGN §12)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel

from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
from cosmos_framework.posttrain.algorithm.grpo import make_objective
from cosmos_framework.posttrain.algorithm.reward import (
    RewardCandidate,
    RewardCombiner,
    RewardContext,
    RewardProviderResult,
    bind_candidate_provider,
)
from cosmos_framework.posttrain.control import ActorGroup, DeviceGroup, ExperienceBuffer, bind_device_groups
from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
from cosmos_framework.posttrain.data.packer import init_data_packer
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.env import ArithmeticEnv
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.roles.rollout import HFRollout
from cosmos_framework.posttrain.roles.trainer import LLMTrainer


def _variance_reward(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    # deterministic but variable across completions -> nonzero group advantage
    del context
    return RewardProviderResult(terms={"task": float(len(candidate.completion) % 2)})


# G is the completions-per-prompt these flows roll out. It is a module constant so the reward
# stage's advantage estimator can be built with the same number the tests generate with -- the
# estimator checks group completeness, and hard-coding a second copy would let the two drift.
G = 3


def _setup():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    packer = init_data_packer("gpt2")  # rollout + objective shaping seam; owns the tokenizer (design 2026-07-28)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    return {
        "data": ArithmeticEnv(seed=0, max_operand=9),
        "rollout": HFRollout(model, packer=packer, sampling={"do_sample": True, "max_new_tokens": 4}),
        "rm": RewardWorker(
            [bind_candidate_provider("variance", _variance_reward)],
            RewardCombiner(),
            annotators=(group_relative(group_size=G),),
            training_group_size=G,
        ),
        "replay": ExperienceBuffer(),
        "policy": LLMTrainer(model, make_objective(packer), opt),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_b1_grpo_loop_runs_two_steps_local():
    data = roles.env("data")
    rollout = roles.rollout("rollout")
    rm = roles.reward("rm")
    replay = roles.buffer("replay")
    policy = roles.trainer("policy")
    bind_device_groups(
        [
            DeviceGroup(
                actor_groups=(ActorGroup(members=(data, rollout, rm, replay, policy), setup=_setup),),
                device="cpu",
            )
        ],
        backend="local",
    )

    P = 2
    before = next(policy.body.model.parameters()).detach().clone()
    losses = []
    for _ in range(2):
        prompts = data.get_samples(P).result()
        trajs = rollout.generate(prompts, group=G).result()
        scored = rm.score(trajs).result()
        replay.put_samples(scored)
        batch = replay.get_samples(P * G).result()
        loss = policy.train(batch).result()["train/loss_avg"]
        policy.push_weights()
        assert isinstance(loss, float) and loss == loss  # finite
        losses.append(loss)
    after = next(policy.body.model.parameters()).detach()
    assert len(losses) == 2
    assert not torch.allclose(before, after)  # optimizer stepped over the run


@pytest.mark.L1
@pytest.mark.CPU
def test_grid_form_a_runs_two_steps_cpu():
    """Form A: all five WGs share one `(1,2)` ActorGroup on CPU."""
    ray = pytest.importorskip("ray")

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups

    data = roles.env("data")
    rollout = roles.rollout("rollout", replicas=1, ranks=2)
    rm = roles.reward("rm")
    replay = roles.buffer("replay")
    policy = roles.trainer("policy", replicas=1, ranks=2)
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(data, rollout, rm, replay, policy), setup=_setup),),
                    device="cpu",
                )
            ],
            backend="ray",
            threshold=0,
        )
        P = 2
        losses = []
        for _ in range(2):
            prompts = data.get_samples(P)
            trajs = rollout.generate(prompts, group=G)
            scored = rm.score(trajs)
            replay.put_samples(scored)
            batch = replay.get_samples(P * G)
            loss = policy.train(batch).result()["train/loss_avg"]
            policy.push_weights()
            rollout.apply_weights()
            assert isinstance(loss, float) and loss == loss  # finite
            losses.append(loss)
        assert len(losses) == 2
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Form B helpers: rollout `(2,2)` ActorGroup + policy `(1,4)` ActorGroup with a `(2,2)` model mesh.
# These mirror the two-ActorGroup rollout/trainer split the entrypoints build, but use a
# tiny GPT-2 model so the CPU test stays fast.
# ---------------------------------------------------------------------------


class _WeightProbe:
    """Test-only introspection hooks for the weight-sync value tests — kept OUT of the
    production bodies (LLMTrainer/HFRollout). Built by the setup builders below, so a
    ``_Probe*`` body ships into the Ray actor like any other body (monkeypatching the
    driver-side class would not reach the actor process). Touches only ``self.model`` /
    ``self._weight_backend``, so one mixin serves both trainer and rollout."""

    def set_params(self, seed: int) -> None:
        # Deterministically overwrite every parameter from `seed` (device-independent, so trainer and
        # rollout cells produce identical values for the same seed).
        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for p in self.model.parameters():
                p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(generator=gen).to(p.dtype).to(p.device))

    def state_cpu(self) -> dict:
        # Full state_dict on CPU for exact per-tensor comparison across processes.
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def weight_backend_kind(self) -> str:
        # Class name of the bound mover (e.g. "IpcMover"), or "NoneType".
        be = getattr(self, "_weight_backend", None)
        return type(getattr(be, "mover", be)).__name__


class _ProbeTrainer(_WeightProbe, LLMTrainer):
    """LLMTrainer + test-only weight-sync probes."""


class _ProbeRollout(_WeightProbe, HFRollout):
    """HFRollout + test-only weight-sync probes."""


def _setup_rollout():
    """Build the rollout policy body: data env, rollout, reward model, replay buffer."""
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    packer = init_data_packer("gpt2")  # rollout-side shaping seam; owns the tokenizer (design 2026-07-28)
    return {
        "data": ArithmeticEnv(seed=0, max_operand=9),
        "rollout": _ProbeRollout(model, packer=packer, sampling={"do_sample": True, "max_new_tokens": 4}),
        "rm": RewardWorker(
            [bind_candidate_provider("variance", _variance_reward)],
            RewardCombiner(),
            annotators=(group_relative(group_size=G),),
            training_group_size=G,
        ),
        "replay": ExperienceBuffer(),
    }


def _setup_trainer():
    """Build the trainer policy body: trainable model + optimizer."""
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    packer = init_data_packer("gpt2")  # objective-side shaping seam (design 2026-07-28)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    return {"policy": _ProbeTrainer(model, make_objective(packer), opt)}


@pytest.mark.L1
@pytest.mark.CPU
def test_grid_form_b_two_groups_runs_two_steps_cpu():
    """Form B: rollout `(2,2)` plus one policy `(1,4)` ActorGroup on CPU.

    Exercises the cross-group batch handoff: replay.get_samples() produces the batch on the rollout
    mesh; policy.train() consumes it on the trainer mesh. The registered train edge reduces both
    replay units into the sole policy execution replica before its ranks consume the batch.
    """
    ray = pytest.importorskip("ray")

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups

    # Rollout group: data / rollout / rm / replay — rollout 2x2, helpers ranks=1
    data = roles.env("data", replicas=2)
    rollout = roles.rollout("rollout", replicas=2, ranks=2)
    rm = roles.reward("rm", replicas=2)
    replay = roles.buffer("replay", replicas=2)
    # Trainer group: one execution replica over the same total actor count.
    policy = roles.trainer("policy", replicas=1, ranks=4)
    try:
        # Two distinct ActorGroups: cross-group alignment is the key difference from form A.
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(members=(data, rollout, rm, replay), setup=_setup_rollout),
                        ActorGroup(members=(policy,), setup=_setup_trainer),
                    ),
                    device="cpu",
                )
            ],
            backend="ray",
            threshold=0,
        )
        P = 2  # small batch for a fast CPU flow check
        for _ in range(2):
            prompts = data.get_samples(P)
            trajs = rollout.generate(prompts, group=G)
            scored = rm.score(trajs)
            replay.put_samples(scored)
            # Batch produced on the rollout mesh; consumed by the trainer mesh.
            batch = replay.get_samples(P * G)
            # Cross-group: policy.train pulls the batch via peer registry (one-sided).
            loss = policy.train(batch).result()["train/loss_avg"]
            policy.push_weights()
            rollout.apply_weights()
            assert isinstance(loss, float) and loss == loss  # finite
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Weight-sync exact-value proofs (opt-in weight_sync=True). Only the trainer is
# seeded; the pre-sync inequality guards a vacuous pass. See comm/weight_transfer.py
# and operator-provided multinode documentation.
# ---------------------------------------------------------------------------


def _setup_rollout_cuda():
    """Rollout bodies with the model on CUDA (IpcMover needs GPU tensors for the IPC handle)."""
    d = _setup_rollout()
    d["rollout"].model = d["rollout"].model.cuda()
    return d


def _setup_trainer_cuda():
    """Trainer body with the model on CUDA. (No train() in the IPC test, so the CPU-built optimizer
    is never used — set_params/push/apply operate on the model directly.)"""
    d = _setup_trainer()
    d["policy"].model = d["policy"].model.cuda()
    return d


# NOTE: role naming follows this package's convention — the ROLLOUT WG is "rollout" and the
# TRAINER/learner WG is "policy" (matches _setup_rollout / _setup_trainer keys above).


@pytest.mark.L1
@pytest.mark.CPU
def test_weight_sync_copies_exact_values():
    """Gold-standard weight-sync check: set the trainer's weights to fixed known values, run the
    real sync, and assert EVERY rollout tensor is bit-exactly equal to the trainer's — a per-tensor
    proof (not a scalar sum) that push→apply transfers weights faithfully (CopyMover, explicit
    equal-product flat actor route). Only the trainer is seeded, so the pre-sync inequality guards
    against a vacuous pass."""
    ray = pytest.importorskip("ray")

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups

    data = roles.env("data", replicas=2, ranks=2)
    rollout = roles.rollout("rollout", replicas=2, ranks=2)
    rm = roles.reward("rm", replicas=2, ranks=2)
    replay = roles.buffer("replay", replicas=2, ranks=2)
    policy = roles.trainer(
        "policy",
        replicas=1,
        ranks=4,
    )  # "policy" = trainer/learner
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(members=(data, rollout, rm, replay), setup=_setup_rollout),
                        ActorGroup(members=(policy,), setup=_setup_trainer),
                    ),
                    device="cpu",
                )
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
        )
        policy.set_params(20260630).all()  # .all() forces execution on every trainer cell

        t0 = policy.state_cpu().all()  # {(r,k): {name: cpu tensor}} — trainer
        r0 = rollout.state_cpu().all()
        policy_cells = sorted(t0)
        rollout_cells = sorted(r0)
        # Pre-sync: at least one flat-aligned tensor differs (rollout has an independent random init).
        assert any(
            not torch.equal(t0[p_cell][name], r0[r_cell][name])
            for p_cell, r_cell in zip(policy_cells, rollout_cells, strict=True)
            for name in t0[p_cell]
        )

        weight_sync(rollout, policy)  # the UNITED entry point — real trainer→rollout sync (Copy)

        t1 = policy.state_cpu().all()
        r1 = rollout.state_cpu().all()
        for p_cell, r_cell in zip(sorted(t1), sorted(r1), strict=True):
            assert set(t1[p_cell]) == set(r1[r_cell]), f"cells {p_cell}->{r_cell}: param-name mismatch"
            for name in t1[p_cell]:
                assert torch.equal(r1[r_cell][name], t1[p_cell][name]), (
                    f"cells {p_cell}->{r_cell} param {name}: rollout != trainer after sync"
                )
    finally:
        ray.shutdown()


@pytest.mark.L1
@pytest.mark.GPU
def test_weight_sync_copies_exact_values_ipc():
    """Same exact-value proof as the CopyMover test, but through the REAL IPC bind path:
    two ActorGroups in one GPU DeviceGroup imply same-GPU CUDA IPC. Confirms the resolved mover is
    IpcMover, then asserts bit-exact per-tensor sync. GPU only (2 actors on one GPU)."""
    ray = pytest.importorskip("ray")
    if not torch.cuda.is_available():
        pytest.skip("GPU required for IPC backend")

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups

    data = roles.env("data")
    rollout = roles.rollout("rollout")
    rm = roles.reward("rm")
    replay = roles.buffer("replay")
    policy = roles.trainer("policy")  # "policy" = trainer/learner
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(members=(data, rollout, rm, replay), setup=_setup_rollout_cuda),
                        ActorGroup(members=(policy,), setup=_setup_trainer_cuda),
                    ),
                    device="gpu",
                )
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
        )
        assert policy.weight_backend_kind().result() == "IpcMover"
        assert rollout.weight_backend_kind().result() == "IpcMover"

        policy.set_params(20260630).result()
        t0 = policy.state_cpu().result()  # 1x1 mesh ⇒ .result() is the single cell's state
        r0 = rollout.state_cpu().result()
        assert any(not torch.equal(t0[n], r0[n]) for n in t0)  # pre-sync differ

        weight_sync(rollout, policy)  # the UNITED entry point — real CUDA-IPC sync

        t1 = policy.state_cpu().result()
        r1 = rollout.state_cpu().result()
        assert set(t1) == set(r1)
        for n in t1:
            assert torch.equal(r1[n], t1[n]), f"param {n}: rollout != trainer after IPC sync"
    finally:
        ray.shutdown()


@pytest.mark.L1
@pytest.mark.GPU
def test_weight_sync_exact_values_multinode():
    """Multinode exact-value proof: on a >=2-node Ray cluster (RAY_ADDRESS set by
    launch_multinode.sh), a mesh sized to span nodes syncs weights bit-exactly per cell. Only the
    trainer is seeded; the pre-sync inequality guards a vacuous pass. Skipped unless RAY_ADDRESS
    points at a real multinode cluster (never in CI)."""
    import os

    ray = pytest.importorskip("ray")
    if not os.environ.get("RAY_ADDRESS"):
        pytest.skip("needs a multinode Ray cluster (run under launch_multinode.sh: RAY_ADDRESS set)")

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.utils.mesh import detect_mesh

    R, K = detect_mesh()  # (nodes, GPUs-per-node); run with NNODES>=2 so the mesh tiles across nodes
    assert R >= 2, "set NNODES>=2 (run under launch_multinode.sh context) for the multinode test"
    data = roles.env("data", replicas=R, ranks=K)
    rollout = roles.rollout("rollout", replicas=R, ranks=K)
    rm = roles.reward("rm", replicas=R, ranks=K)
    replay = roles.buffer("replay", replicas=R, ranks=K)
    policy = roles.trainer(
        "policy",
        replicas=1,
        ranks=R * K,
    )  # "policy" = trainer/learner
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(members=(data, rollout, rm, replay), setup=_setup_rollout_cuda),
                        ActorGroup(members=(policy,), setup=_setup_trainer_cuda),
                    ),
                    device="gpu",
                )
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
        )
        gpu_nodes = [n for n in ray.nodes() if n.get("Alive") and n["Resources"].get("GPU", 0)]
        assert len(gpu_nodes) >= 2, "expected the cluster to have >=2 GPU nodes"

        # Prove the mesh spans nodes and equal-product flat actor pairs share a node/GPU.
        tnode, rnode = policy.node_id().all(), rollout.node_id().all()
        assert len(set(tnode.values()) | set(rnode.values())) >= 2, "mesh did not span >=2 nodes"
        for p_cell, r_cell in zip(sorted(tnode), sorted(rnode), strict=True):
            assert tnode[p_cell] == rnode[r_cell], (
                f"flat cells {p_cell}->{r_cell}: trainer/rollout not on the same node (IPC requirement)"
            )
        # Confirm the real IPC path was selected (not a Copy fallback that would also pass equality).
        assert set(policy.weight_backend_kind().all().values()) == {"IpcMover"}
        assert set(rollout.weight_backend_kind().all().values()) == {"IpcMover"}

        policy.set_params(20260701).all()  # force execution on every trainer cell
        t0, r0 = policy.state_cpu().all(), rollout.state_cpu().all()
        assert any(
            not torch.equal(t0[p_cell][name], r0[r_cell][name])
            for p_cell, r_cell in zip(sorted(t0), sorted(r0), strict=True)
            for name in t0[p_cell]
        )

        weight_sync(rollout, policy)  # the UNITED entry point — per-bundle IPC across nodes

        t1, r1 = policy.state_cpu().all(), rollout.state_cpu().all()
        for p_cell, r_cell in zip(sorted(t1), sorted(r1), strict=True):
            assert set(t1[p_cell]) == set(r1[r_cell]), f"flat cells {p_cell}->{r_cell}: param-name mismatch"
            for name in t1[p_cell]:
                assert torch.equal(r1[r_cell][name], t1[p_cell][name]), (
                    f"flat cells {p_cell}->{r_cell} param {name}: rollout != trainer after sync"
                )
    finally:
        ray.shutdown()
