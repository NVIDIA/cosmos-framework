# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `VLLMRollout` records the engine's routed experts as `policy_info["rollout_routed_experts"]`.

Drives `_parse_training_requests` with fake vLLM request outputs shaped like `CompletionOutput`
(`routed_experts`: ``[seq_len, layer, top_k]``, prompt-inclusive, one row per input position).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore
from cosmos_framework.posttrain.comm.codec import materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.roles.rollout import VLLMRollout
from cosmos_framework.posttrain.schema import ObsBundle
from cosmos_framework.posttrain.schema.routing import ROLLOUT_ROUTED_EXPERTS_KEY

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_LAYERS, _TOP_K = 3, 2


class _Packer:
    tokenizer = SimpleNamespace(pad_token_id=0)

    @staticmethod
    def adds_special_tokens() -> bool:
        return False


def _output(token_ids: list[int], routed: np.ndarray | None) -> SimpleNamespace:
    logprobs = [{token: SimpleNamespace(logprob=-0.5)} for token in token_ids]
    return SimpleNamespace(token_ids=token_ids, text="ok", logprobs=logprobs, routed_experts=routed)


def _request(prompt_ids: list[int], outputs: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(prompt_token_ids=prompt_ids, outputs=outputs)


def _parse(record: bool, request: SimpleNamespace) -> list:
    rollout = VLLMRollout(object(), packer=_Packer(), record_routed_experts=record)
    return rollout._parse_training_requests([ObsBundle(representations={"prompt": "q"}, id="p")], [request], 1)


def test_routed_experts_are_recorded_per_completion_and_ride_as_bulk() -> None:
    """Each completion keeps the engine's [P+N-1, L, K] record next to rollout_logprobs, as bulk on the wire; absent when not recording."""
    prompt, completion = [11, 12, 13], [21, 22]
    routed = np.arange((len(prompt) + len(completion) - 1) * _LAYERS * _TOP_K, dtype=np.uint8).reshape(
        -1, _LAYERS, _TOP_K
    )
    (trajectory,) = _parse(True, _request(prompt, [_output(completion, routed)]))
    info = trajectory.transitions[0].policy_info
    record = info[ROLLOUT_ROUTED_EXPERTS_KEY]
    assert record.dtype == torch.uint8 and record.shape == (4, _LAYERS, _TOP_K)
    assert np.array_equal(record.numpy(), routed) and len(info["rollout_logprobs"]) == 2
    assert "rollout_routed_experts_sha256" in VLLMRollout._training_output_identity([trajectory])["trajectories"][0]

    packed = pack_payload(trajectory, BulkStore(), threshold=8)
    assert isinstance(packed.transitions[0].policy_info[ROLLOUT_ROUTED_EXPERTS_KEY], BulkRef)
    restored = materialize_payload(packed, LocalStoreMover(cross_process=True))
    assert torch.equal(restored.transitions[0].policy_info[ROLLOUT_ROUTED_EXPERTS_KEY], record)

    with pytest.raises(RuntimeError, match="enable_return_routed_experts"):
        _parse(True, _request(prompt, [_output(completion, None)]))
    (natural,) = _parse(False, _request(prompt, [_output(completion, None)]))
    assert ROLLOUT_ROUTED_EXPERTS_KEY not in natural.transitions[0].policy_info
