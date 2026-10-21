# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: recorded expert routing on the policy path (`data/packer/forward_routing.py`)."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.data.packer.decoder_only_llm import DecoderOnlyLLMDataPacker, shape_policy_input
from cosmos_framework.posttrain.schema import ObsBundle, Transition
from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY, ROLLOUT_ROUTED_EXPERTS_KEY

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_LAYERS, _TOP_K = 2, 3
_PROMPT = [1, 2, 3]


def _transition(completion: list[int], routing: torch.Tensor | None) -> Transition:
    info: dict = {"prompt_ids": torch.tensor(_PROMPT)}
    if routing is not None:
        info[ROLLOUT_ROUTED_EXPERTS_KEY] = routing
    return Transition(
        obs=ObsBundle(representations={"prompt": "p"}, id="s"), action=torch.tensor(completion), policy_info=info
    )


def _record(sequence_length: int) -> torch.Tensor:
    return torch.randint(0, 8, (sequence_length - 1, _LAYERS, _TOP_K), dtype=torch.uint8)  # [T-1,L,K]


def test_recorded_routing_is_carried_and_collated_token_aligned() -> None:
    """Collation lays the record under the batch key as [B, T_max, L, K]: row t is input token t, the rest is the fill row."""
    packer = DecoderOnlyLLMDataPacker()
    packer.tokenizer = type("_Tok", (), {"pad_token_id": 0})()
    long_record, short_record = _record(5), _record(4)
    processed = [
        shape_policy_input(_transition([7, 8], long_record), packer),
        shape_policy_input(_transition([7], short_record), packer),
    ]
    assert torch.equal(processed[0].forward_routing, long_record)

    batch = packer.policy_collate_fn(processed, packer.policy_compute_max_len(processed))
    routing = batch[FORWARD_REPLAY_BATCH_KEY]
    assert routing.shape == (2, 5, _LAYERS, _TOP_K) and routing.dtype == torch.uint8
    filler = torch.arange(_TOP_K, dtype=torch.uint8).expand(_LAYERS, -1)
    assert torch.equal(routing[0, :4], long_record) and torch.equal(routing[0, 4], filler)
    assert torch.equal(routing[1, :3], short_record) and torch.equal(routing[1, 3:], filler.expand(2, -1, -1))

    natural = shape_policy_input(_transition([7], None), packer)
    assert FORWARD_REPLAY_BATCH_KEY not in packer.policy_collate_fn([natural], 4)
    with pytest.raises(ValueError, match="every sequence or for none"):
        packer.policy_collate_fn([processed[1], natural], 4)
