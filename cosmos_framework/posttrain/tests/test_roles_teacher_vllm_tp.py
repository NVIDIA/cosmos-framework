# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L1 @GPU: ranked vLLM teacher TP=2 through the real Ray grid."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from cosmos_framework.utils.helper_test import RunIf
from cosmos_framework.posttrain.schema import ObsBundle, Status, TeacherSignal, Trajectory, Transition

TP = 2
TOP_K = 4
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class _SignalSink:
    """Materialize the leader payload inside one bulk-consuming actor."""

    def inspect(self, trajectories: list[Trajectory]) -> dict[str, object]:
        if len(trajectories) != 1:
            raise ValueError(f"signal sink expected one trajectory, got {len(trajectories)}")
        signal = trajectories[0].transitions[0].targets
        if not isinstance(signal, TeacherSignal):
            raise TypeError(f"signal sink expected TeacherSignal, got {type(signal).__name__}")
        if not isinstance(signal.topk_ids, torch.Tensor) or not isinstance(signal.topk_logprobs, torch.Tensor):
            raise TypeError("signal sink received an unmaterialized top-k payload")
        return {
            "shape": tuple(signal.topk_ids.shape),
            "completion_tokens": len(trajectories[0].transitions[0].action),
            "finite": bool(torch.isfinite(signal.topk_logprobs).all()),
        }


def _setup_tp_teacher() -> dict[str, object]:
    """Build each teacher lane after the per-replica TP rendezvous exists."""
    from vllm import LLM

    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.teacher import VLLMTeacher

    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    assert world_size == TP
    model = LLM(
        model=MODEL,
        tensor_parallel_size=world_size,
        distributed_executor_backend="external_launcher",
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.6,
        max_model_len=512,
        max_logprobs=TOP_K,
        logprobs_mode="processed_logprobs",
        enforce_eager=True,
    )
    return {
        "teacher": VLLMTeacher(
            model,
            packer=init_data_packer(MODEL),
            top_k=TOP_K,
            max_attempts=1,
        ),
        "sink": _SignalSink(),
    }


def _trajectory() -> Trajectory:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tokenizer("1 + 1 =", add_special_tokens=False).input_ids
    completion_ids = tokenizer(" 2", add_special_tokens=False).input_ids
    transition = Transition(
        obs=ObsBundle(representations={"prompt": "1 + 1 ="}, id="ranked-teacher-p0"),
        action=torch.tensor(completion_ids, dtype=torch.long),  # [T]
        policy_info={
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),  # [P]
            "completion_ids": list(completion_ids),
            "completion_text": " 2",
            "completion_boundary": {
                "prompt_token_count": len(prompt_ids),
                "completion_token_count": len(completion_ids),
                "full_sequence_token_count": len(prompt_ids) + len(completion_ids),
            },
        },
        done=True,
    )
    return Trajectory(
        transitions=[transition],
        id="ranked-teacher-p0_k0",
        group_id="ranked-teacher-p0",
        status=Status.COMPLETED,
    )


@RunIf(min_gpus=2)
@pytest.mark.L1
@pytest.mark.GPU
def test_ranked_teacher_tp2_converges_and_publishes_only_leader_payload() -> None:
    import ray

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles
    from cosmos_framework.posttrain.utils.entry import resolve_identical

    teacher = roles.teacher("teacher", ranks=TP)
    sink = roles.reward("sink", needs_bulk=True)
    bind_device_groups(
        [
            DeviceGroup(
                actor_groups=(ActorGroup(members=(teacher, sink), setup=_setup_tp_teacher),),
                device="gpu",
            )
        ],
        backend="ray",
        threshold=0,
    )
    try:
        scored = teacher.score([_trajectory()])
        stats = resolve_identical(teacher.batch_stats(), operation="ranked teacher TP2 stats")
        inspection = sink.inspect(scored).result()

        assert stats["train/opd_teacher_score_request_count"] == 1
        assert stats["train/opd_teacher_score_success_count"] == 1
        assert inspection["shape"] == (inspection["completion_tokens"], TOP_K)
        assert inspection["finite"] is True
    finally:
        ray.shutdown()
