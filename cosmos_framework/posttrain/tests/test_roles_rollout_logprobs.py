# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""π_old per-token logprob capture in HFRollout (Task 4).

Asserts `HFRollout.generate` emits `policy_info["rollout_logprobs"]` as a list of floats with one
entry per completion token — the field the clipped GRPO objective (algorithm/grpo.py) reads as
π_old. Tiny random GPT2 on CPU (the vLLM path needs a GPU and is validated in the smoke test)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel

from cosmos_framework.posttrain.data.packer import init_data_packer
from cosmos_framework.posttrain.roles.rollout import HFRollout
from cosmos_framework.posttrain.schema import ObsBundle


def _tiny():
    # Cannot import the fixture from test_roles_rollout_hf.py (tests never import each other —
    # posttrain/.claude/CLAUDE.md), so replicate the tiny random GPT2 + gpt2 packer here.
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    model = GPT2LMHeadModel(cfg).eval()
    packer = init_data_packer("gpt2")  # rollout-side shaping seam; owns the tokenizer
    return model, packer


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_rollout_emits_per_token_rollout_logprobs():
    model, packer = _tiny()
    rollout = HFRollout(
        model,
        packer=packer,
        sampling={"do_sample": True, "max_new_tokens": 4, "temperature": 1.0},
        capture_logprobs=True,  # OFF by default: output_scores costs ~10GB/prompt on a real model
    )
    obs = ObsBundle(representations={"prompt": "1 + 1 =", "reference": 2}, id="p0")
    trajs = rollout.generate([obs], group=2)

    assert len(trajs) == 2
    for t in trajs:
        tr = t.transitions[0]
        assert "rollout_logprobs" in tr.policy_info
        lps = tr.policy_info["rollout_logprobs"]
        # A Python list of plain floats, matching behavior-logprob alignment input.
        assert isinstance(lps, list)
        assert all(isinstance(x, float) for x in lps)
        # One logprob per sampled completion token; alignment enforces this length parity.
        assert len(lps) == len(tr.action)
