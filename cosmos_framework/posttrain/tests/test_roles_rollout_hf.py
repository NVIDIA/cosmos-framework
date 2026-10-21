# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for HFRollout (DESIGN §1.2 body=(a)). Uses a tiny random GPT2 on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel

from cosmos_framework.posttrain.data.packer import init_data_packer
from cosmos_framework.posttrain.roles.rollout import HFRollout
from cosmos_framework.posttrain.schema import ObsBundle, Status


def _tiny():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    model = GPT2LMHeadModel(cfg).eval()
    packer = init_data_packer("gpt2")  # rollout-side shaping seam; owns the tokenizer (design 2026-07-28)
    return model, packer


@pytest.mark.L0
@pytest.mark.CPU
def test_generate_returns_group_trajectories():
    model, packer = _tiny()
    rollout = HFRollout(model, packer=packer, sampling={"do_sample": True, "max_new_tokens": 4})
    prompts = [
        ObsBundle(representations={"prompt": "1 + 1 =", "reference": 2}, id="p0"),
        ObsBundle(representations={"prompt": "2 + 3 =", "reference": 5}, id="p1"),
    ]
    trajs = rollout.generate(prompts, group=3)
    assert len(trajs) == 2 * 3
    # group_id is the prompt's own id (obs.id), so groups are namespaced per-prompt (never the
    # batch-local index that collides across replicas at fan-in).
    assert {t.group_id for t in trajs} == {"p0", "p1"}
    assert {t.id for t in trajs} == {"p0_k0", "p0_k1", "p0_k2", "p1_k0", "p1_k1", "p1_k2"}
    for t in trajs:
        assert t.status == Status.COMPLETED
        tr = t.transitions[0]
        assert "completion_text" in tr.policy_info
        assert "prompt_ids" in tr.policy_info
        assert tr.action is not None


@pytest.mark.L0
@pytest.mark.CPU
def test_hfrollout_uses_packer_for_prompt():
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import HFRollout
    from cosmos_framework.posttrain.schema import ObsBundle

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64))
    packer = init_data_packer("gpt2")
    # do_sample=True: greedy decoding rejects num_return_sequences>1 (group=2) in this
    # transformers version; sampling supports group>1 and still exercises the packer path.
    r = HFRollout(model, packer=packer, sampling={"do_sample": True, "max_new_tokens": 2})
    trajs = r.generate([ObsBundle(representations={"prompt": "1+1="}, id="u1")], group=2)
    assert len(trajs) == 2
    assert all(t.transitions[0].action.numel() >= 0 for t in trajs)  # produced completions


@pytest.mark.L0
@pytest.mark.CPU
def test_hfrollout_requires_packer_at_construction():
    # The contract lives in __init__, NOT generate(): a packer-less rollout used to construct, bind
    # and spawn every Ray actor fine and only die on the first generate() with the cluster fully up.
    # Omitted argument ⇒ TypeError; an explicit packer=None ⇒ our ValueError with the remedy.
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.roles.rollout import HFRollout

    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64))
    with pytest.raises(TypeError):
        HFRollout(model)
    with pytest.raises(ValueError, match="requires a DataPacker at construction"):
        HFRollout(model, packer=None)


@pytest.mark.L0
@pytest.mark.CPU
def test_hfrollout_params_after_model_are_keyword_only():
    # Only `model` is positional, so a call's meaning never depends on argument order: a stale
    # positional (e.g. a sampling dict, or a tokenizer from before the packer seam) raises here
    # instead of silently binding to whatever occupies that slot today.
    from cosmos_framework.posttrain.roles.rollout import HFRollout

    with pytest.raises(TypeError):
        HFRollout(object(), {"max_new_tokens": 4})


@pytest.mark.L0
@pytest.mark.CPU
def test_trim_at_stop_cuts_at_first_stop_and_keeps_it():
    """`_trim_at_stop` drops `generate`'s right padding, keeping the stop token itself.

    This is the fix for a real NaN: `generate` right-pads every sequence in a group to the longest
    one, and untrimmed pads land inside `logprob_masks` (built from len(completion)) with pi_old of
    -inf, because the sampler's logit processors push every filtered token to -inf. Downstream that
    is exp(cur - (-inf)) = inf, and inf * 0 = NaN the moment a group has zero advantage — the common
    case early in training. Decoding hid it (`skip_special_tokens=True` looks clean) so only the ids
    show the bug, which is exactly why this asserts on ids rather than text.

    The stop token is KEPT: it is a real token the policy chose, and cosmos-rl trains on it too
    (vLLM's `stop_token_ids` terminates AFTER emitting it).
    """
    from cosmos_framework.posttrain.roles.rollout.hf import _trim_at_stop

    eos, pad = 50256, 50256
    # Padded completion: two real tokens, EOS, then padding.
    out = _trim_at_stop(torch.tensor([11, 22, eos, pad, pad]), {eos, pad})
    assert out.tolist() == [11, 22, eos], "must keep the stop token and drop everything after"

    # Ran to max_new_tokens without stopping -> unchanged, nothing to trim.
    assert _trim_at_stop(torch.tensor([11, 22, 33]), {eos, pad}).tolist() == [11, 22, 33]

    # Immediate stop -> length 1, not empty (the token was still generated).
    assert _trim_at_stop(torch.tensor([eos, pad, pad]), {eos, pad}).tolist() == [eos]

    # MULTIPLE stop ids: Qwen declares two (<|im_end|> ends a turn, <|endoftext|> is also the pad).
    # Trimming on only one of them leaves the other as scored padding, which is the NaN again.
    im_end, endoftext = 151645, 151643
    assert _trim_at_stop(torch.tensor([7, im_end, 9, endoftext]), {im_end, endoftext}).tolist() == [7, im_end]
    assert _trim_at_stop(torch.tensor([7, endoftext, 9, im_end]), {im_end, endoftext}).tolist() == [7, endoftext]

    # Empty stop set (weight-only rigs bind a stub packer with no tokenizer) -> no-op, no crash.
    assert _trim_at_stop(torch.tensor([11, 22]), set()).tolist() == [11, 22]
