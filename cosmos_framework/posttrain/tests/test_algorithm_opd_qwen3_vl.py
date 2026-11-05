# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: OPD forwards complete VLM batches through the VLMPolicy capability."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.opd import make_direct_objective
from cosmos_framework.posttrain.model.protocol import VLMPolicyOutput
from cosmos_framework.posttrain.schema import ObsBundle, Status, TeacherSignal, Trajectory, Transition


class _VisionPacker:
    def __init__(self) -> None:
        self.policy_input_calls: int = 0

    def get_policy_input(
        self,
        _sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> SimpleNamespace:
        self.policy_input_calls += 1
        completion_ids = rollout_output.tolist() if hasattr(rollout_output, "tolist") else list(rollout_output)
        prefix = list(prompt_ids or [1, 9])
        return SimpleNamespace(
            input_ids=prefix + completion_ids,
            logprob_masks=[0] * (len(prefix) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0],
            pixel_values=torch.ones((4, 3), dtype=torch.float32),  # [N_patch,D]
            image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),  # [N_image,3]
        )

    def policy_compute_max_len(self, processed_samples: list[SimpleNamespace]) -> int:
        return max(len(sample.input_ids) for sample in processed_samples)

    def policy_collate_fn(
        self,
        processed_samples: list[SimpleNamespace],
        computed_max_len: int,
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(  # [B,L]
            [sample.input_ids + [0] * (computed_max_len - len(sample.input_ids)) for sample in processed_samples],
            dtype=torch.long,
        )
        logprob_masks = torch.tensor(  # [B,L]
            [
                sample.logprob_masks + [0] * (computed_max_len - len(sample.logprob_masks))
                for sample in processed_samples
            ],
            dtype=torch.bool,
        )
        pixel_values = torch.cat([sample.pixel_values for sample in processed_samples], dim=0)  # [N_patch,D]
        image_grid_thw = torch.cat(  # [N_image,3]
            [sample.image_grid_thw for sample in processed_samples],
            dim=0,
        )
        return {
            "input_ids": input_ids,
            "logprob_masks": logprob_masks,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


class _Net(torch.nn.Module):
    table: torch.nn.Parameter

    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros((16, 16), dtype=torch.float32))  # [V,V]

    def forward(self, _input_ids: torch.Tensor) -> None:
        raise AssertionError("OPD bypassed VLMPolicy.forward_policy_batch")


class _VLMPolicy:
    def __init__(self) -> None:
        self.net = _Net()
        self.seen_batch: dict[str, Any] | None = None

    def forward_policy_batch(self, batch: dict[str, Any]) -> VLMPolicyOutput:
        self.seen_batch = batch
        input_ids = batch["input_ids"]  # [B,L]
        logits = self.net.table[input_ids]  # [B,L,V]
        return VLMPolicyOutput(logits=logits)


def _trajectory(completion_ids: list[int]) -> Trajectory:
    transition = Transition(
        obs=ObsBundle(representations={"messages": []}, modality_tags=("text", "image"), id="p0"),
        action=torch.tensor(completion_ids, dtype=torch.long),  # [T]
        policy_info={"prompt_ids": torch.tensor([1, 9], dtype=torch.long)},  # [P]
        done=True,
    )
    if completion_ids:
        transition.targets = TeacherSignal(
            topk_ids=torch.tensor([[0]], dtype=torch.int32),  # [T,K]
            topk_logprobs=torch.tensor([[-0.4]], dtype=torch.float32),  # [T,K]
        )
    return Trajectory(transitions=[transition], id="g0_k0", group_id="g0", status=Status.COMPLETED)


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_opd_uses_vlm_policy_capability_with_complete_media_batch() -> None:
    policy = _VLMPolicy()
    objective = make_direct_objective(_VisionPacker(), jsd_beta=1.0)

    loss, _metrics = objective(policy, [_trajectory([3])])  # [], metrics
    loss.backward()

    assert policy.seen_batch is not None
    assert set(policy.seen_batch) >= {"input_ids", "logprob_masks", "pixel_values", "image_grid_thw"}
    assert policy.net.table.grad is not None


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_opd_vlm_probe_reuses_the_first_shaped_media_batch() -> None:
    packer = _VisionPacker()
    policy = _VLMPolicy()
    objective = make_direct_objective(packer, jsd_beta=1.0)

    loss, metrics = objective(policy, [_trajectory([])])  # [], metrics
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert metrics["opd_valid_tokens"] == 0
    assert packer.policy_input_calls == 1
    assert policy.seen_batch is not None
    assert set(policy.seen_batch) >= {"input_ids", "logprob_masks", "pixel_values", "image_grid_thw"}
    assert policy.net.table.grad is not None
