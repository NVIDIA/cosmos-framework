# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""vLLM TP=2 via external_launcher SPMD (Task #3 route C): end-to-end through the grid mesh.

Needs 2 GPUs + the vLLM venv, so it is RunIf/importorskip-guarded and L1/GPU (TP engine init
is slow). Verifies the TP>1 generate slice still yields group integrity + per-token logprob.
The rollout setup lives HERE (test fixture) rather than being imported from a demo script:
`entrypoints/grpo.py` is the supported way to run a real trainer + weight sync on the same
external_launcher TP engine."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: F841
vllm = pytest.importorskip("vllm")  # noqa: F841

from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.posttrain.schema import Status

TP = 2
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 14 attention heads → divisible by TP=2 (vLLM requires it)
GROUP_SIZE = 3


def _setup_tp_rollout() -> dict:
    """Bodies for each TP-rank actor — built INSIDE the Ray actor where its GPU is. The replica's
    per-rank torch.distributed group is already up (spawn_grid Pass 2 ran init_distributed before
    this setup); vLLM external_launcher REUSES it as the TP group of size WORLD_SIZE — no nested
    Ray/mp, no manual rendezvous here."""
    import torch
    from vllm import LLM

    from cosmos_framework.posttrain.algorithm.reward import DEFAULT_REWARD_PROVIDER_CATALOG, RewardCombiner
    from cosmos_framework.posttrain.control import ExperienceBuffer
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.env import ArithmeticEnv
    from cosmos_framework.posttrain.roles.reward import RewardWorker
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout

    assert torch.distributed.is_initialized(), "per-replica rendezvous not run (need ranks>1 on GPU)"
    ws = torch.distributed.get_world_size()
    assert ws == TP, f"TP world size {ws} != TP={TP}"
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=ws,
        distributed_executor_backend="external_launcher",
        # Per-actor GPU isolation (CUDA_VISIBLE_DEVICES) breaks vLLM's CustomAllreduce P2P check →
        # disable it and fall back to NCCL allreduce across the isolated per-actor processes.
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.6,
        max_model_len=1024,
        enforce_eager=True,
    )
    return {
        "data": ArithmeticEnv(seed=0, max_operand=50),
        "policy": VLLMRollout(
            llm, sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 16}, packer=init_data_packer(MODEL)
        ),
        "rm": RewardWorker(
            [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
            RewardCombiner(),
            training_group_size=GROUP_SIZE,
        ),
        "replay": ExperienceBuffer(),
    }


@RunIf(min_gpus=2)
@pytest.mark.L1
@pytest.mark.GPU
def test_vllm_tp2_generate_group_integrity_and_logprob():
    import ray

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    P, G = 2, GROUP_SIZE
    policy = roles.rollout("policy", ranks=2)
    data = roles.env("data")
    rm = roles.reward("rm")
    replay = roles.buffer("replay")
    bind_device_groups(
        [
            DeviceGroup(
                actor_groups=(ActorGroup(members=(policy, data, rm, replay), setup=_setup_tp_rollout),),
                device="gpu",
            )
        ],
        backend="ray",
        threshold=0,
    )
    try:
        prompts = data.get_samples(P)
        trajs = policy.generate(prompts, group=G)
        replay.put_samples(rm.score(trajs))
        batch = replay.get_samples(P * G).result()

        assert len(batch) == P * G  # group integrity (§2.3)
        assert len({t.group_id for t in batch}) == P  # P complete groups
        for t in batch:
            assert t.status == Status.COMPLETED
            lp = t.transitions[0].policy_info["rollout_logprobs"]  # π_old per token (§2.4)
            assert isinstance(lp, list) and len(lp) > 0
    finally:
        ray.shutdown()
