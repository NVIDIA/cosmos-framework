# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective, make_objective  # noqa: E402
from cosmos_framework.posttrain.config.rl import GRPOConfig  # noqa: E402
from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput  # noqa: E402
from cosmos_framework.posttrain.model.protocol import VLMPolicyOutput  # noqa: E402
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory, Transition  # noqa: E402


@dataclass
class _PolicyInput:
    input_ids: list[int]
    attention_mask: list[bool]
    logprob_masks: list[int]
    pixel_values: torch.Tensor  # [3]


class _MediaPacker:
    def get_policy_input(
        self,
        sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> _PolicyInput:
        del prompt_ids
        completion_ids = [int(token_id) for token_id in rollout_output]
        prompt = [1, int(sample.representations["prompt_token"])]
        input_ids = prompt + completion_ids
        pixel_values = torch.tensor(sample.representations["pixel_values"], dtype=torch.float32)  # [3]
        return _PolicyInput(
            input_ids=input_ids,
            attention_mask=[True] * len(input_ids),
            logprob_masks=[0] * (len(prompt) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0],
            pixel_values=pixel_values,
        )

    def policy_compute_max_len(self, processed_samples: list[_PolicyInput]) -> int:
        return max(len(sample.input_ids) for sample in processed_samples)

    def policy_collate_fn(
        self,
        processed_samples: list[_PolicyInput],
        computed_max_len: int,
    ) -> dict[str, Any]:
        input_ids = torch.tensor(  # [B,L]
            [sample.input_ids + [0] * (computed_max_len - len(sample.input_ids)) for sample in processed_samples],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(  # [B,L]
            [
                sample.attention_mask + [False] * (computed_max_len - len(sample.attention_mask))
                for sample in processed_samples
            ],
            dtype=torch.bool,
        )
        logprob_masks = torch.tensor(  # [B,L]
            [
                sample.logprob_masks + [0] * (computed_max_len - len(sample.logprob_masks))
                for sample in processed_samples
            ],
            dtype=torch.bool,
        )
        pixel_values = torch.stack([sample.pixel_values for sample in processed_samples])  # [B,3]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logprob_masks": logprob_masks,
            "pixel_values": pixel_values,
        }


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(16, 8)
        self.output: nn.Linear = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Any:  # input_ids: [B,L], returns logits [B,L,V]
        hidden = self.embedding(input_ids)  # [B,L,D]
        logits = self.output(hidden)  # [B,L,V]
        return SimpleNamespace(logits=logits)


class _VLMPolicy:
    def __init__(self) -> None:
        self.net: _TinyNet = _TinyNet()
        self.seen: dict[str, Any] = {}
        self.prepared: bool = False

    def validate_policy_batch(self, batch: dict[str, Any]) -> None:
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["pixel_values"], torch.Tensor)

    def prepare_policy_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.validate_policy_batch(batch)
        self.prepared = True
        return batch

    def forward_policy_batch(self, batch: dict[str, Any]) -> VLMPolicyOutput:
        self.seen = dict(batch)
        output = self.net(batch["input_ids"])  # output.logits: [B,L,V]
        return VLMPolicyOutput(logits=output.logits)


def _trajectories() -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for index, advantage in enumerate((1.0, -1.0)):
        transition = Transition(
            obs=ObsBundle(
                representations={
                    "prompt_token": index + 2,
                    "pixel_values": [float(index), 0.5, 1.0],
                },
                modality_tags=("text", "image"),
                id=f"sample-{index}",
            ),
            action=[index + 4],
            policy_info={},
            done=True,
        )
        transition.advantage = advantage
        trajectories.append(Trajectory(transitions=[transition], id=f"trajectory-{index}", group_id="group"))
    return trajectories


def _empty_completion_trajectory() -> Trajectory:
    transition = Transition(
        obs=ObsBundle(
            representations={
                "prompt_token": 2,
                "pixel_values": [0.25, 0.5, 1.0],
            },
            modality_tags=("text", "image"),
            id="empty-completion-sample",
        ),
        action=[],
        policy_info={},
        done=True,
    )
    transition.advantage = 0.0
    return Trajectory(transitions=[transition], id="empty-completion-trajectory", group_id="group")


def _config() -> GRPOConfig:
    return GRPOConfig(
        group_size=2,
        epsilon_low=10.0,
        epsilon_high=10.0,
        kl_beta=0.0,
        temperature=1.0,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_clipped_grpo_forwards_the_complete_collated_batch_to_a_vlm_policy() -> None:
    policy = _VLMPolicy()

    loss, _ = make_clipped_objective(_MediaPacker(), _config())(policy, _trajectories())  # loss: []

    assert torch.isfinite(loss)
    assert policy.prepared is True
    assert set(policy.seen) >= {"input_ids", "attention_mask", "pixel_values"}
    assert tuple(policy.seen["pixel_values"].shape) == (2, 3)


@pytest.mark.L0
@pytest.mark.CPU
def test_clipped_grpo_refuses_model_inputs_without_a_vlm_policy_capability() -> None:
    with pytest.raises(TypeError, match="pixel_values"):
        make_clipped_objective(_MediaPacker(), _config())(_TinyNet(), _trajectories())


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_grpo_rejects_structurally_multimodal_policy_input_before_forward() -> None:
    """The legacy objective must inspect shaped data instead of trusting a packer type flag."""

    class _Packer:
        def get_policy_input(self, *_args: Any, **_kwargs: Any) -> RLPolicyInput:
            return RLPolicyInput(
                input_ids=[1, 2, 4],
                logprob_masks=[0, 1, 0],
                mm_kwargs={"pixel_values": torch.ones((1, 3))},  # [N,C]
            )

    with pytest.raises(ValueError, match=r"legacy GRPO.*multimodal"):
        make_objective(_Packer())(_TinyNet(), _trajectories())


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_completion_probe_preserves_media_for_the_vlm_policy() -> None:
    policy = _VLMPolicy()

    loss, metrics = make_clipped_objective(_MediaPacker(), _config())(  # loss: []
        policy,
        [_empty_completion_trajectory()],
    )

    assert float(loss.detach()) == 0.0
    assert metrics["grpo_valid_tokens"] == 0
    assert policy.prepared is True
    assert "pixel_values" in policy.seen
    assert tuple(policy.seen["pixel_values"].shape) == (1, 3)
