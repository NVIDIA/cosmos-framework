# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: exact DPCA alignment, signal construction, credit, and additive composition."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.dpca import (
    AlignmentChunk,
    DPCATeacherCreditSource,
    align_completion_chunks,
    build_dpca_signal,
)
from cosmos_framework.posttrain.algorithm.opd import TopKSliceDivergence, make_advantage_objective
from cosmos_framework.posttrain.schema import ObsBundle, Status, TeacherSignal, Trajectory, Transition


def _decoder(vocabulary: dict[int, str]) -> Callable[[list[int]], str]:
    def decode(token_ids: list[int]) -> str:
        return "".join(vocabulary[token_id] for token_id in token_ids)

    return decode


def _trajectory(
    rollout_logprobs: list[float],
    *,
    n_ignore_prefix_tokens: int = 0,
    advantage: float = 0.0,
) -> Trajectory:
    completion_ids = torch.arange(1, len(rollout_logprobs) + 1, dtype=torch.long)  # [T]
    transition = Transition(
        obs=ObsBundle(representations={"prompt": "unused"}, id="p0"),
        action=completion_ids,
        policy_info={
            "prompt_ids": torch.tensor([0], dtype=torch.long),  # [P]
            "completion_ids": completion_ids.tolist(),
            "rollout_logprobs": rollout_logprobs,
        },
        advantage=advantage,
        algo_extra={"n_ignore_prefix_tokens": n_ignore_prefix_tokens},
        done=True,
    )
    return Trajectory(transitions=[transition], id="g0_k0", group_id="g0", status=Status.COMPLETED)


def _chunk_signal(bounds: list[list[int]], teacher_logprobs: list[float]) -> TeacherSignal:
    return TeacherSignal(
        chunk_bounds=torch.tensor(bounds, dtype=torch.long),  # [C,2]
        chunk_teacher_logprob=torch.tensor(teacher_logprobs, dtype=torch.float32),  # [C]
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_alignment_and_signal_use_exact_full_cover_chunks() -> None:
    student_decode = _decoder({1: "a", 2: "bc", 3: "d"})
    teacher_decode = _decoder({10: "ab", 11: "c", 12: "d"})
    chunks = align_completion_chunks(
        [1, 2, 3],
        [10, 11, 12],
        student_decode,
        teacher_decode,
    )

    assert chunks == [
        AlignmentChunk(student_start=0, student_end=2, teacher_start=0, teacher_end=2),
        AlignmentChunk(student_start=2, student_end=3, teacher_start=2, teacher_end=3),
    ]
    signal = build_dpca_signal(
        [1, 2, 3],
        [10, 11, 12],
        [-0.2, -0.3, -0.4],
        student_decode,
        teacher_decode,
    )
    expected_bounds = torch.tensor([[0, 2], [2, 3]], dtype=torch.long)  # [C,2]
    expected_logprobs = torch.tensor([-0.5, -0.4])  # [C]
    assert torch.equal(signal.chunk_bounds, expected_bounds)
    assert torch.allclose(signal.chunk_teacher_logprob, expected_logprobs)
    assert signal.topk_ids is None
    with pytest.raises(ValueError, match="exceeded max_chunk_tokens"):
        align_completion_chunks(
            [1, 2],
            [10, 11],
            _decoder({1: "a", 2: "bc"}),
            _decoder({10: "ab", 11: "c"}),
            max_chunk_tokens=1,
        )
    with pytest.raises(ValueError, match="completion texts do not match"):
        align_completion_chunks([1], [10], _decoder({1: "a"}), _decoder({10: "b"}))


@pytest.mark.L0
@pytest.mark.CPU
def test_dpca_credit_math_and_ignore_boundary() -> None:
    source = DPCATeacherCreditSource()
    trajectory = _trajectory([-1.0, -1.0])
    teacher_better = source.per_token_credit(  # [T]
        [trajectory],
        [_chunk_signal([[0, 2]], [-1.0])],
        torch.device("cpu"),
    )
    teacher_worse = source.per_token_credit(  # [T]
        [trajectory],
        [_chunk_signal([[0, 2]], [-4.0])],
        torch.device("cpu"),
    )
    exact_zero = source.per_token_credit(  # [T]
        [_trajectory([0.0, -1.0])],
        [_chunk_signal([[0, 1], [1, 2]], [-1.0e-6, -0.5])],
        torch.device("cpu"),
    )
    ignored = source.per_token_credit(  # [T-n_ignore]
        [_trajectory([-1.0, -1.0, -1.0], n_ignore_prefix_tokens=1)],
        [_chunk_signal([[0, 3]], [-1.5])],
        torch.device("cpu"),
    )

    assert torch.allclose(teacher_better, torch.tensor([0.5, 0.5]))
    assert torch.allclose(teacher_worse, torch.tensor([-1.0, -1.0]))
    assert torch.equal(exact_zero, torch.tensor([0.0, 0.5]))
    assert torch.allclose(ignored, torch.tensor([0.5, 0.5]))
    assert not teacher_better.requires_grad


@pytest.mark.L0
@pytest.mark.CPU
def test_dpca_credit_rejects_malformed_inputs() -> None:
    mismatched_action = _trajectory([-1.0, -1.0])
    mismatched_action.transitions[0].action = torch.tensor([1, 3], dtype=torch.long)  # [T]
    cases = [
        (_trajectory([-1.0, -1.0, -1.0]), _chunk_signal([[0, 1], [2, 3]], [-1.0, -1.0]), "contiguous"),
        (_trajectory([float("nan")]), _chunk_signal([[0, 1]], [-1.0]), "rollout logprobs contains non-finite"),
        (_trajectory([-1.0]), _chunk_signal([[0, 1]], [float("inf")]), "teacher chunk logprobs contains non-finite"),
        (_trajectory([0.1]), _chunk_signal([[0, 1]], [-1.0]), "rollout logprobs contains a positive value"),
        (_trajectory([-1.0]), _chunk_signal([[0, 1]], [0.1]), "teacher chunk logprobs contains a positive value"),
        (_trajectory([-1.0e-8]), _chunk_signal([[0, 1]], [-1.0]), "near-zero L_S"),
        (mismatched_action, _chunk_signal([[0, 2]], [-1.0]), "action differs from inline completion_ids"),
    ]
    for trajectory, signal, message in cases:
        with pytest.raises(ValueError, match=message):
            DPCATeacherCreditSource().per_token_credit(
                [trajectory],
                [signal],
                torch.device("cpu"),
            )


class _Packer:
    def get_policy_input(
        self,
        _sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> SimpleNamespace:
        completion_ids = rollout_output.tolist() if hasattr(rollout_output, "tolist") else list(rollout_output)
        prefix = list(prompt_ids or [0])
        return SimpleNamespace(
            input_ids=prefix + completion_ids,
            logprob_masks=[0] * (len(prefix) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0],
        )

    def policy_compute_max_len(self, processed_samples: list[Any]) -> int:
        return max(len(sample.input_ids) for sample in processed_samples)

    def policy_collate_fn(self, processed_samples: list[Any], computed_max_len: int) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(  # [B,L]
            [sample.input_ids + [0] * (computed_max_len - len(sample.input_ids)) for sample in processed_samples],
            dtype=torch.long,
        )
        masks = torch.tensor(  # [B,L]
            [
                sample.logprob_masks + [0] * (computed_max_len - len(sample.logprob_masks))
                for sample in processed_samples
            ],
            dtype=torch.bool,
        )
        return {"input_ids": input_ids, "logprob_masks": masks}


class _Net(torch.nn.Module):
    table: torch.nn.Parameter

    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros((4, 4), dtype=torch.float32))  # [V,V]

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:  # input_ids: [B,L], returns logits [B,L,V]
        logits = self.table[input_ids]  # [B,L,V]
        return SimpleNamespace(logits=logits)


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_objective_adds_dpca_credit_instead_of_subtracting_it() -> None:
    model = _Net()
    trajectory = _trajectory([-1.0, -1.0], advantage=0.25)
    trajectory.transitions[0].targets = _chunk_signal([[0, 2]], [-1.0])
    objective = make_advantage_objective(
        _Packer(),
        jsd_beta=None,
        teacher_credit=DPCATeacherCreditSource(),
        kl_penalty_coef=2.0,
        epsilon_low=10.0,
        epsilon_high=10.0,
    )

    loss, metrics = objective(model, [trajectory])  # [], metrics
    loss.backward()

    assert metrics["opd_advantage"] == pytest.approx(1.25)
    assert metrics["opd_teacher_credit"] == pytest.approx(0.5)
    assert model.table.grad is not None
    assert float(model.table.grad.abs().sum()) > 0.0
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_advantage_objective(
            _Packer(),
            divergence=TopKSliceDivergence(beta=1.0),
            teacher_credit=DPCATeacherCreditSource(),
        )
