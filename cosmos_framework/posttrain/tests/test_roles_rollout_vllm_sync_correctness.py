# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""VALUE-correctness of the vLLM weight sink (the check the e2e RUN did not prove): a MODIFIED
trainer param, pushed through the real ``VLLMRollout.apply_weights`` path, must land VALUE-for-value
in the live vLLM engine. Asserts the target param (a) DIFFERS from the trainer's modified value
BEFORE sync (kills a silent no-op) and (b) EQUALS it AFTER. Single process (CopyMover, by-value) —
so no Ray/IPC; the cross-process IPC transport is the shared-GPU example. Requires vLLM + GPU.
"""

from __future__ import annotations

import os

import pytest

from cosmos_framework.posttrain.tests.helpers.vllm_engine import release_in_process_engines

MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# vLLM freezes its environment reads once the first in-process EngineCore initialises, so the TP=2
# test cannot pick this per test. The pytest process already holds a CUDA context by then, and a
# forked worker cannot re-initialise CUDA.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture(autouse=True)
def _release_engines_after_each_test():
    yield
    release_in_process_engines()


# Non-fused params present under the SAME name in both HF and vLLM (vLLM fuses q/k/v and gate/up,
# so those keys would not compare 1:1 — layernorm/down_proj are safe, distinctive witnesses).
KEYS = ["model.layers.0.input_layernorm.weight", "model.layers.0.mlp.down_proj.weight"]


def _read_vllm(rollout, keys):
    # read named params from inside the engine worker (V1) / in-process (V0)
    from cosmos_framework.posttrain.roles.rollout import vllm_weights

    return vllm_weights._apply_to_model(
        rollout.model,
        lambda m: {k: dict(m.named_parameters())[k].detach().float().cpu() for k in keys},
    )


# (the batch-form value test was removed with the WeightPayload apply_weights arm;
# its streaming twin test_vllm_streaming_lands_trainer_values below is the value gate)


# --- cross-process IPC variant: two ActorGroups in one GPU DeviceGroup ---------------------------

IKEY = "model.layers.0.input_layernorm.weight"


def _setup_trainer_min():
    import torch
    from transformers import AutoModelForCausalLM

    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    return {
        "policy": LLMTrainer(m, make_objective(init_data_packer("gpt2")), torch.optim.AdamW(m.parameters(), lr=1e-6))
    }


def _setup_rollout_min():
    from vllm import LLM

    from cosmos_framework.posttrain.config.build import _configure_vllm_runtime
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout

    _configure_vllm_runtime()  # in-process EngineCore, as in production: apply_model(fn) is a direct call
    llm = LLM(model=MODEL, gpu_memory_utilization=0.3, max_model_len=512, enforce_eager=True)
    # Packer is REQUIRED at construction (fail fast, before actors spin up); weight-only rig.
    return {"rollout": VLLMRollout(llm, packer=init_data_packer(MODEL))}


def _modify_trainer(host):
    # runs INSIDE the trainer actor: bump the witness param and return its new value
    import torch

    net = host._bodies["policy"]._net()
    p = dict(net.named_parameters())[IKEY]
    with torch.no_grad():
        p.add_(5.0)
    return p.detach().float().cpu()


def _read_rollout_vllm(host):
    # runs INSIDE the rollout actor: read the engine's live param via apply_model
    from cosmos_framework.posttrain.roles.rollout import vllm_weights

    r = host._bodies["rollout"]
    return vllm_weights._apply_to_model(
        r.model,
        lambda mm: dict(mm.named_parameters())[IKEY].detach().float().cpu(),
    )


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_cross_process_ipc():
    # The REAL same-GPU path: trainer + vLLM are separate ActorGroups in one GPU DeviceGroup, so the
    # binder resolves same_gpu ⇒ IpcMover. A modified trainer param must cross the process boundary
    # via CUDA-IPC and land VALUE-for-value in the live vLLM engine.
    pytest.importorskip("vllm")
    ray = pytest.importorskip("ray")
    import torch

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        policy = roles.trainer("policy")
        rollout = roles.rollout("rollout")
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(members=(policy,), setup=_setup_trainer_min),
                        ActorGroup(members=(rollout,), setup=_setup_rollout_min),
                    ),
                    device="gpu",
                )
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
        )
        pa = policy._backend.actors[(0, 0)]
        ra = rollout._backend.actors[(0, 0)]
        want = ray.get(pa.run.remote(_modify_trainer))  # modify trainer + capture target
        before = ray.get(ra.run.remote(_read_rollout_vllm))
        assert not torch.allclose(before, want), "vLLM already == target before sync (a no-op would pass)"

        weight_sync(rollout, policy)  # REAL cross-process same-GPU IPC sync (peer-pull streaming)

        after = ray.get(ra.run.remote(_read_rollout_vllm))
        assert torch.allclose(after, want, atol=0.0, rtol=0.0), (
            f"cross-process IPC: vLLM weight != trainer after sync (max abs diff {(after - want).abs().max().item()})"
        )
    finally:
        ray.shutdown()


# --- TP slicing + in-place: does a FULL tensor get correctly SLICED into each TP rank's live buffer? ---

# down_proj is RowParallelLinear ⇒ TP-SHARDED (unlike layernorm, which is replicated). Same name in
# HF and vLLM (not fused, unlike qkv/gate_up), so it maps 1:1.
SHARDED_KEY = "model.layers.0.mlp.down_proj.weight"


def _read_sharded_witness(m):
    # MODULE-LEVEL (picklable to the worker) — return PRIMITIVES only: a tensor nested in a tuple is
    # msgpack-serialized to a plain list by apply_model, so return the first row as a list[float] and
    # rebuild a tensor on the driver. Per TP rank: (tp_rank, shape, first-row values, GPU data_ptr).
    from vllm.distributed import get_tensor_model_parallel_rank

    p = dict(m.named_parameters())[SHARDED_KEY]
    return (get_tensor_model_parallel_rank(), list(p.shape), p[0].detach().float().cpu().tolist(), p.data_ptr())


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_tp2_sync_slices_inplace():
    # Confirms the three things "well sliced in-place" means: (a) the param is really TP-SHARDED (the
    # two ranks hold DIFFERENT slices, not a replicated copy); (b) after syncing a full tensor, EACH
    # rank's OWN slice is updated (so load_weights sliced correctly per rank); (c) the write is
    # IN-PLACE (the param's GPU buffer address is unchanged — load_weights copied into existing memory,
    # not a fresh tensor the engine ignores). Standalone vLLM TP=2 (its own workers) + CopyMover, 2 GPUs.
    vllm = pytest.importorskip("vllm")
    import torch
    from transformers import AutoModelForCausalLM

    from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync
    from cosmos_framework.posttrain.config.build import _configure_vllm_runtime
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs for TP=2")

    _configure_vllm_runtime()  # in-process EngineCore; TP workers still receive apply_model via cloudpickle
    llm = vllm.LLM(
        model=MODEL,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="float32",
        enforce_eager=True,
    )
    rollout = VLLMRollout(llm, packer=init_data_packer(MODEL))  # packer REQUIRED at construction; weight-only rig
    rollout._weight_backend = WeightSync(CopyMover())  # by-value full tensors; vLLM slices per rank

    def read_all():
        # apply_model returns ONE result per TP worker: (tp_rank, shape, cpu slice, gpu data_ptr)
        rows = llm.apply_model(_read_sharded_witness)
        return {r[0]: r[1:] for r in rows}  # keyed by tp_rank so we compare like-for-like

    before = read_all()  # {rank: (shape, first-row list, data_ptr)}
    print(f"[witness] {SHARDED_KEY}: per-rank shapes = {dict((r, b[0]) for r, b in before.items())}")
    assert set(before) == {0, 1}, f"expected TP ranks {{0,1}}, got {set(before)}"
    ptr0, ptr1 = before[0][2], before[1][2]

    # DEFINITIVE slicing proof: set the trainer's FULL down_proj so column j holds the value j. A
    # correct TP sync must give each rank a DISTINCT contiguous column-range whose row-0 segments
    # reconstruct arange(ncols) — impossible if the full tensor were replicated or mis-sliced.
    net = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda()
    with torch.no_grad():
        full = dict(net.named_parameters())[SHARDED_KEY]  # [out, ncols]
        ncols = full.shape[1]
        full.copy_(torch.arange(ncols, dtype=full.dtype, device=full.device).expand_as(full))
    # PEER form, in-process: the trainer body itself is the peer (has get_bucket_plan/push_bucket);
    # the sync streams per bucket and vLLM's load_weights slices each full tensor per TP rank.
    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    trainer = LLMTrainer(net, make_objective(init_data_packer("gpt2")), torch.optim.AdamW(net.parameters(), lr=1e-6))
    trainer._weight_backend = WeightSync(CopyMover())
    rollout.apply_weights(trainer)  # peer-pull → full tensors → load_weights slices

    after = read_all()
    seg0 = torch.tensor(after[0][1])  # rank-0 first row = the column-index values it received
    seg1 = torch.tensor(after[1][1])  # rank-1 first row
    # (a) the two ranks got DIFFERENT slices (real per-rank sharding, not a replicated copy)
    assert not torch.equal(seg0, seg1), "both TP ranks received the SAME slice — sync not sliced per rank"
    # (b) correct slicing: the two contiguous segments together reconstruct the full arange(ncols),
    #     i.e. each rank got exactly its own column-range of the full tensor.
    recon = torch.cat(sorted([seg0, seg1], key=lambda s: float(s[0])))
    assert torch.allclose(recon, torch.arange(ncols, dtype=torch.float32), atol=1e-3), (
        f"reconstructed columns != arange(0..{ncols}) — wrong TP slicing"
    )
    # (c) IN-PLACE: the live GPU weight buffers were overwritten, not replaced (addresses unchanged)
    assert after[0][2] == ptr0 and after[1][2] == ptr1, "load_weights was NOT in-place (data_ptr changed)"


# --- FSDP-2 trainer + TP-2 vLLM rollout: sharded↔sharded in one GPU DeviceGroup (needs 2 GPUs) ---


def _setup_fsdp_trainer():
    # FSDP2-sharded HFModel (shard degree = injected mesh world). float32 for exact comparison.
    import torch

    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer
    from cosmos_framework.posttrain.workers.ray import current_pdims

    hf = build_hf_model(
        HFModelConfig(name_or_path=MODEL, dtype="float32", attn_impl="eager", auto_class="AutoModelForCausalLM"),
        pdims=current_pdims(),  # injected FSDP mesh (world=2, dp_shard=2)
    )
    return {
        "policy": LLMTrainer(
            hf,
            make_objective(init_data_packer("gpt2")),
            torch.optim.AdamW(hf.net.parameters(), lr=1e-6),
        )
    }


def _setup_tp_rollout():
    import torch
    from vllm import LLM

    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout

    ws = torch.distributed.get_world_size()  # per-replica rendezvous ran ⇒ TP group of size ws
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=ws,
        distributed_executor_backend="external_launcher",
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="float32",
        enforce_eager=True,
    )
    return {"rollout": VLLMRollout(llm, packer=init_data_packer(MODEL))}  # packer REQUIRED at construction


def _modify_trainer_fsdp(host):
    # runs INSIDE each FSDP rank actor: bump the witness param's LOCAL shard by +5 ⇒ full tensor +5
    import torch

    p = dict(host._bodies["policy"]._net().named_parameters())[IKEY]
    with torch.no_grad():
        p.add_(5.0)


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_fsdp2_tp2():
    # The sharded↔sharded same-GPU path: an FSDP-2 HFModel trainer and a TP-2 vLLM rollout,
    # bin-packed rank-for-rank (trainer FSDP rank k shares GPU k with vLLM TP rank k). A modified
    # trainer param must be GATHERED across FSDP (full_tensor), cross the boundary via CUDA-IPC, and be
    # SLICED into each vLLM TP shard by load_weights — landing VALUE-for-value in the live engine.
    pytest.importorskip("vllm")
    ray = pytest.importorskip("ray")
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs (FSDP shard 2 + vLLM TP 2)")

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        rollout = roles.rollout("rollout", ranks=2)
        policy = roles.trainer("policy", ranks=2)
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(members=(rollout,), setup=_setup_tp_rollout),
                        ActorGroup(members=(policy,), setup=_setup_fsdp_trainer),
                    ),
                    device="gpu",
                )
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
        )
        ra0 = rollout._backend.actors[(0, 0)]
        before = ray.get(ra0.run.remote(_read_rollout_vllm))  # vLLM witness (== checkpoint, both TP ranks)
        # modify BOTH FSDP ranks' local shards (+5) → the gathered full tensor is +5 everywhere
        ray.get([policy._backend.actors[(0, k)].run.remote(_modify_trainer_fsdp) for k in range(2)])

        weight_sync(rollout, policy)  # full_tensor() gather → IPC → vLLM TP load_weights (streamed)

        after = ray.get(ra0.run.remote(_read_rollout_vllm))
        assert not torch.allclose(after, before), "sync did not change vLLM weights (a no-op would pass)"
        assert torch.allclose(after, before + 5.0, atol=1e-4, rtol=0.0), (
            f"FSDP2→TP2 sync: vLLM weight != trainer+5 (max abs diff {(after - before - 5.0).abs().max().item()})"
        )
    finally:
        ray.shutdown()


# --- STREAMING value correctness: the same guarantees, synced bucket-by-bucket (peak ≈ one bucket) ---


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_streaming_lands_trainer_values():
    # Single-process CopyMover, but synced ONE BUCKET AT A TIME (small bucket_bytes → many buckets).
    # A modified trainer must still land value-for-value in the vLLM engine via the streaming path.
    vllm = pytest.importorskip("vllm")
    import torch
    from transformers import AutoModelForCausalLM

    from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, bucket_plan, model_manifest
    from cosmos_framework.posttrain.config.build import _configure_vllm_runtime
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    _configure_vllm_runtime()  # in-process EngineCore, as in production
    llm = vllm.LLM(model=MODEL, gpu_memory_utilization=0.35, max_model_len=512, enforce_eager=True)
    rollout = VLLMRollout(llm, packer=init_data_packer(MODEL))  # packer REQUIRED at construction; weight-only rig

    trainer = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for k in KEYS:
            dict(trainer.named_parameters())[k].add_(5.0)
    want = {k: dict(trainer.named_parameters())[k].detach().float().cpu() for k in KEYS}

    plan = bucket_plan(model_manifest(trainer), bucket_bytes=8 * 1024 * 1024)  # ~8 MiB → many buckets
    assert len(plan) >= 3
    prod = WeightSync(CopyMover())
    prod.set_bucket_plan(plan)
    rollout._weight_backend = WeightSync(CopyMover())
    rollout.set_bucket_plan(plan)

    before = _read_vllm(rollout, KEYS)
    for k in KEYS:
        assert not torch.allclose(before[k], want[k]), f"{k} already == target before streaming sync"
    for i in range(len(plan)):
        rollout.apply_bucket(prod.stage_bucket(trainer, i), i)  # stream bucket i
    after = _read_vllm(rollout, KEYS)
    for k in KEYS:
        assert torch.allclose(after[k], want[k], atol=1e-2), (
            f"{k}: streamed vLLM weight != trainer (max diff {(after[k] - want[k]).abs().max().item()})"
        )


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_streaming_fsdp2_tp2():
    # The sharded↔sharded case, STREAMED through the WEIGHT-CHANNEL PEER FORM: FSDP-2 trainer + TP-2
    # vLLM, synced with ONE role-edge call — rollout.apply_weights(policy) — through the real
    # Grid/PeerClient path (WorkerGroup-as-arg → PeerClient per dp-aligned cell → lazy plan pull →
    # per-bucket IPC pulls). Confirms gather+slice per bucket lands value-for-value with no driver
    # Phase-0 and no driver stream loop.
    pytest.importorskip("vllm")
    ray = pytest.importorskip("ray")
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        rollout = roles.rollout("rollout", ranks=2)
        policy = roles.trainer("policy", ranks=2)
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(members=(rollout,), setup=_setup_tp_rollout),
                        ActorGroup(members=(policy,), setup=_setup_fsdp_trainer),
                    ),
                    device="gpu",
                )
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
        )

        ra0 = rollout._backend.actors[(0, 0)]
        before = ray.get(ra0.run.remote(_read_rollout_vllm))
        ray.get([policy._backend.actors[(0, k)].run.remote(_modify_trainer_fsdp) for k in range(2)])

        weight_sync(rollout, policy)  # the UNITED entry point (drains internally); asserts read post-sync

        after = ray.get(ra0.run.remote(_read_rollout_vllm))
        assert not torch.allclose(after, before), "streaming sync did not change vLLM weights"
        assert torch.allclose(after, before + 5.0, atol=1e-4), (
            f"streamed FSDP2→TP2: vLLM weight != trainer+5 (max diff {(after - before - 5.0).abs().max().item()})"
        )
    finally:
        ray.shutdown()


@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_sync_fsdp2_tp2_nixl():
    """The same VALUE-for-value guarantee as test_vllm_sync_fsdp2_tp2, but DISAGGREGATED over
    NIXL across disjoint GPU DeviceGroups instead of one shared GPU DeviceGroup using CUDA-IPC.

    Worth its own test because the NIXL mover changes the wire format, not just the transport:
    a bucket travels as ONE flat descriptor carved on the consumer via handle.meta["plan"],
    rather than one tensor per parameter. A carve that returned correct-looking tensors in the
    wrong ORDER -- or a plan whose sorted-name ordering disagreed with what load_weights
    expects -- would produce a silently wrong model that every other NIXL test still passes,
    because they only check round-trip identity, never the vLLM sink.

    The `not allclose(after, before)` assertion is load-bearing: a sync that silently did
    nothing would otherwise pass, and a frozen rollout still yields plausible GRPO losses.
    """
    pytest.importorskip("vllm")
    ray = pytest.importorskip("ray")
    import torch

    from cosmos_framework.posttrain.workers.nixl import nixl_available

    ok, why = nixl_available()
    if not ok:
        pytest.skip(f"NIXL unavailable: {why}")
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs (two disjoint 2-GPU DeviceGroups)")

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        rollout = roles.rollout("rollout", ranks=2)
        policy = roles.trainer("policy", ranks=2)
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(rollout,), setup=_setup_tp_rollout),),
                    device="gpu",
                ),
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(policy,), setup=_setup_fsdp_trainer),),
                    device="gpu",
                ),
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
            weight_sync_transport="nixl",
        )
        ra0 = rollout._backend.actors[(0, 0)]
        before = ray.get(ra0.run.remote(_read_rollout_vllm))
        ray.get([policy._backend.actors[(0, k)].run.remote(_modify_trainer_fsdp) for k in range(2)])

        weight_sync(rollout, policy)  # full_tensor() gather → flat pack → NIXL → vLLM TP load_weights

        after = ray.get(ra0.run.remote(_read_rollout_vllm))
        assert not torch.allclose(after, before), "sync did not change vLLM weights (a no-op would pass)"
        assert torch.allclose(after, before + 5.0, atol=1e-4, rtol=0.0), (
            f"FSDP2→TP2 NIXL sync: vLLM weight != trainer+5 (max abs diff {(after - before - 5.0).abs().max().item()})"
        )
    finally:
        ray.shutdown()
