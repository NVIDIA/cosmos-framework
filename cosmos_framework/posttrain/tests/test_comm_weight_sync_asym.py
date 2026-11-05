# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU value-correctness for full-tensor flat-actor weight pairing.

Equal-product tests use same-GPU IPC and the flat-index bijection. Disaggregated tests use independent
NIXL pools with unequal products: policy-wider activates only the required source actors, while
rollout-wider assigns every rollout actor a direct full-tensor source. Every case checks a live vLLM
parameter on every rollout lane plus the controller-published producer iteration. Exact 64↔4
topology arithmetic is covered by the CPU pairing tests; live witnesses scale it to 4→2 and 2→4.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import pytest

QWEN = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
QWEN3 = "Qwen/Qwen3Guard-Gen-0.6B"  # 16 heads ⇒ TP=4 valid (Qwen2.5-0.5B's 14 heads are not divisible by 4)
# A REPLICATED (not TP-sharded), non-fused RMSNorm param present under the SAME name in HF and
# vLLM: every TP lane — lead OR fed — holds the full tensor, so reading it on any (r,k) actor
# witnesses that that exact lane received the synced bytes. Same key in both Qwen2.5 and Qwen3.
IKEY = "model.layers.0.input_layernorm.weight"
STAMP_ITER = 7  # a distinctive producer training iteration to prove version propagation (init is 0)


# --- role bodies, built INSIDE each actor (mirrors the sync-correctness harness) ------------------
def _setup_tp_rollout(model: str = QWEN) -> dict:
    import torch
    from vllm import LLM

    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout

    ws = torch.distributed.get_world_size()  # per-replica rendezvous ran ⇒ TP group of size ws
    llm = LLM(
        model=model,
        tensor_parallel_size=ws,
        distributed_executor_backend="external_launcher",
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="float32",
        enforce_eager=True,
    )
    # Packer is REQUIRED at construction now (fail fast, not on the first generate); this rig is
    # weight-only, so the packer is never exercised beyond its special-token guard.
    return {"rollout": VLLMRollout(llm, packer=init_data_packer(model))}


def _setup_fsdp_trainer(model: str = QWEN) -> dict:
    import torch

    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer
    from cosmos_framework.posttrain.workers.ray import current_pdims

    hf = build_hf_model(
        HFModelConfig(name_or_path=model, dtype="float32", attn_impl="eager", auto_class="AutoModelForCausalLM"),
        pdims=current_pdims(),  # injected FSDP mesh (world = policy rank width)
    )
    return {
        "policy": LLMTrainer(
            hf,
            make_objective(init_data_packer("gpt2")),
            torch.optim.AdamW(hf.net.parameters(), lr=1e-6),
        )
    }


# --- actor-side probes (ikey bound per-model via functools.partial in the driver) -----------------
def _read_rollout_vllm(host, ikey: str):
    # INSIDE a rollout actor: read the engine's live witness value (full tensor; replicated across TP).
    from cosmos_framework.posttrain.roles.rollout import vllm_weights

    r = host._bodies["rollout"]
    return vllm_weights._apply_to_model(
        r.model,
        lambda mm: dict(mm.named_parameters())[ikey].detach().float().cpu(),
    )


def _read_rollout_version(host):
    return host._bodies["rollout"].weight_version()


def _modify_and_stamp_trainer(host, ikey: str, delta: float, iteration: int) -> None:
    # INSIDE each FSDP policy rank: bump the witness's LOCAL shard (and therefore full tensor)
    # and stamp this lane's training iteration (the version the sync must propagate to every rollout
    # lane). Every actor belongs to one policy execution replica and carries the same committed
    # iteration, so whichever actor serves a rollout target stamps the same version.
    import torch

    body = host._bodies["policy"]
    p = dict(body._net().named_parameters())[ikey]
    with torch.no_grad():
        p.add_(delta)
    body._iter = iteration


def _run_asym_case(
    policy_shape: tuple[int, int],
    rollout_shape: tuple[int, int],
    model: str,
    ikey: str,
    *,
    disaggregated: bool = False,
    policy_parallelism: tuple[int, int] | None = None,
    sync_deltas: tuple[float, ...] = (5.0,),
) -> tuple[dict, dict, dict, Any]:
    """Bind an IPC or NIXL mesh and return each rollout lane's before/after/version witness."""
    ray = pytest.importorskip("ray")
    import torch

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    R_p, K_p = policy_shape
    R_r, K_r = rollout_shape
    read = functools.partial(_read_rollout_vllm, ikey=ikey)
    if not sync_deltas:
        raise ValueError("sync_deltas must contain at least one update")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        rollout = roles.rollout("rollout", replicas=R_r, ranks=K_r)
        policy = roles.trainer(
            "policy",
            replicas=R_p,
            ranks=K_p,
            model_parallelism=policy_parallelism or (1, K_p),
        )
        rollout_actor_group = ActorGroup(
            members=(rollout,),
            setup=functools.partial(_setup_tp_rollout, model=model),
        )
        policy_actor_group = ActorGroup(
            members=(policy,),
            setup=functools.partial(_setup_fsdp_trainer, model=model),
        )
        device_groups = (
            [
                DeviceGroup(actor_groups=(rollout_actor_group,), device="gpu"),
                DeviceGroup(actor_groups=(policy_actor_group,), device="gpu"),
            ]
            if disaggregated
            else [DeviceGroup(actor_groups=(rollout_actor_group, policy_actor_group), device="gpu")]
        )
        bind_device_groups(
            device_groups,
            backend="ray",
            threshold=0,
            weight_sync=True,
            weight_sync_transport="nixl" if disaggregated else None,
            weight_sync_probe_bytes=64 * 1024 * 1024,
            weight_sync_min_gbps=0.0,
        )
        rollout_cells = [(r, k) for r in range(R_r) for k in range(K_r)]
        before = {c: ray.get(rollout._backend.actors[c].run.remote(read)) for c in rollout_cells}
        for sync_index, delta in enumerate(sync_deltas):
            modify = functools.partial(
                _modify_and_stamp_trainer,
                ikey=ikey,
                delta=delta,
                iteration=STAMP_ITER + sync_index,
            )
            ray.get([policy._backend.actors[(r, k)].run.remote(modify) for r in range(R_p) for k in range(K_p)])
            weight_sync(rollout, policy)  # routed push + rollout-wide prepare/commit publication

        after = {c: ray.get(rollout._backend.actors[c].run.remote(read)) for c in rollout_cells}
        versions = {c: ray.get(rollout._backend.actors[c].run.remote(_read_rollout_version)) for c in rollout_cells}
        return before, after, versions, torch
    finally:
        ray.shutdown()


def _assert_lands(
    before,
    after,
    versions,
    torch,
    cells,
    *,
    expected_delta: float = 5.0,
    expected_iteration: int = STAMP_ITER,
) -> None:
    for c in cells:
        assert not torch.allclose(after[c], before[c]), f"rollout {c}: sync changed nothing (no-op would pass)"
        assert torch.allclose(after[c], before[c] + expected_delta, atol=1e-4, rtol=0.0), (
            f"rollout {c}: vLLM weight delta mismatch "
            f"(max abs diff {(after[c] - before[c] - expected_delta).abs().max().item()})"
        )
        assert versions[c] == expected_iteration, (
            f"rollout {c}: weight_version {versions[c]} != producer _iter {expected_iteration}"
        )


def _enough_gpus(n: int) -> bool:
    if os.environ.get("POSTTRAIN_NIXL_CROSS_NODE") == "1":
        import ray

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)
        available = int(ray.cluster_resources().get("GPU", 0))
        if available < n:
            pytest.fail(f"asymmetric NIXL qualification requires {n} cluster GPUs, found {available}")
        return True
    import torch

    return torch.cuda.device_count() >= n


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_asym_fanout_policy1x4_hsdp2x2_rollout2x2() -> None:
    # FAN-OUT: policy (1,4) → rollout (2,2). Policy replica 0's 4 cells (flat g=0..3) push to the 4
    # same-GPU rollout cells across both rollout replicas — every lane has a same-GPU policy source,
    # no broadcast. Values must land +5 on EVERY rollout lane (both replicas), version stamped everywhere.
    pytest.importorskip("vllm")
    if not _enough_gpus(4):
        pytest.skip("needs 4 GPUs (policy 1x4 + rollout 2x2 at 0.5 GPU each)")
    before, after, versions, torch = _run_asym_case(
        (1, 4),
        (2, 2),
        QWEN,
        IKEY,
        policy_parallelism=(2, 2),
    )
    _assert_lands(before, after, versions, torch, [(0, 0), (0, 1), (1, 0), (1, 1)])  # both engines


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_asym_showcase_policy1x8_rollout4x2() -> None:
    # The flagship equal-product shape: one 8-rank policy world ⇐ four rollout TP2 engines.
    pytest.importorskip("vllm")
    if not _enough_gpus(8):
        pytest.skip("needs 8 GPUs (policy 1x8 + rollout 4x2 at 0.5 GPU each)")
    before, after, versions, torch = _run_asym_case(
        (1, 8),
        (4, 2),
        QWEN,
        IKEY,
        policy_parallelism=(2, 4),
    )
    _assert_lands(before, after, versions, torch, [(r, k) for r in range(4) for k in range(2)])  # all 8 lanes


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_nixl_hsdp2x2_policy4_to_rollout2_selected_actors() -> None:
    pytest.importorskip("vllm")
    from cosmos_framework.posttrain.workers.nixl import nixl_available

    ok, why = nixl_available()
    if not ok:
        pytest.skip(f"NIXL unavailable: {why}")
    if not _enough_gpus(6):
        pytest.skip("needs 6 GPUs: HSDP(2,2) policy pool + TP2 rollout pool")
    before, after, versions, torch = _run_asym_case(
        (1, 4),
        (1, 2),
        QWEN,
        IKEY,
        disaggregated=True,
        policy_parallelism=(2, 2),
    )
    _assert_lands(before, after, versions, torch, [(0, 0), (0, 1)])


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_nixl_policy2_to_rollout4_direct_actor_fanout() -> None:
    pytest.importorskip("vllm")
    from cosmos_framework.posttrain.workers.nixl import nixl_available

    ok, why = nixl_available()
    if not ok:
        pytest.skip(f"NIXL unavailable: {why}")
    if not _enough_gpus(6):
        pytest.skip("needs 6 GPUs: FSDP2 policy pool + TP4 rollout pool")
    before, after, versions, torch = _run_asym_case(
        (1, 2),
        (1, 4),
        QWEN3,
        IKEY,
        disaggregated=True,
        sync_deltas=(5.0, 7.0),
    )
    _assert_lands(
        before,
        after,
        versions,
        torch,
        [(0, lane) for lane in range(4)],
        expected_delta=12.0,
        expected_iteration=STAMP_ITER + 1,
    )


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_nixl_multi_engine_direct_actor_fanout() -> None:
    pytest.importorskip("vllm")
    from cosmos_framework.posttrain.workers.nixl import nixl_available

    ok, why = nixl_available()
    if not ok:
        pytest.skip(f"NIXL unavailable: {why}")
    if not _enough_gpus(8):
        pytest.skip("needs 8 GPUs: policy (1,2) pool + rollout (3,2) pool")
    before, after, versions, torch = _run_asym_case(
        (1, 2),
        (3, 2),
        QWEN,
        IKEY,
        disaggregated=True,
    )
    _assert_lands(before, after, versions, torch, [(replica, lane) for replica in range(3) for lane in range(2)])
