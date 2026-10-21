# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cosmos_framework.posttrain.data.packer.decoder_only_llm import DecoderOnlyLLMDataPacker
from cosmos_framework.posttrain.schema import ObsBundle


def _packer():
    p = DecoderOnlyLLMDataPacker()
    p.setup("gpt2")
    return p


@pytest.mark.L0
@pytest.mark.CPU
def test_get_rollout_input_returns_prompt_text():
    p = _packer()
    obs = ObsBundle(representations={"prompt": "3+4="}, id="x")
    assert p.get_rollout_input(obs) == "3+4="


@pytest.mark.L0
@pytest.mark.CPU
def test_get_rollout_input_non_str_raises():
    p = _packer()
    with pytest.raises(ValueError, match="representations\\['prompt'\\]"):
        p.get_rollout_input(ObsBundle(representations={"prompt": 123}, id="x"))


@pytest.mark.L0
@pytest.mark.CPU
def test_get_policy_input_ids_and_mask_align():
    p = _packer()
    obs = ObsBundle(representations={"prompt": "3+4="}, id="x")
    prompt_ids = p.tokenizer("3+4=", add_special_tokens=False).input_ids
    completion_ids = p.tokenizer(" 7", add_special_tokens=False).input_ids
    proc = p.get_policy_input(obs, " 7")
    # input_ids = prompt + completion
    assert proc.input_ids == prompt_ids + completion_ids
    # mask length == input_ids length; exactly len(completion) ones
    assert len(proc.logprob_masks) == len(proc.input_ids)
    assert sum(proc.logprob_masks) == len(completion_ids)
    # ones sit at [len(prompt)-1 : len(prompt)-1+len(completion)]
    P, C = len(prompt_ids), len(completion_ids)
    assert proc.logprob_masks[P - 1 : P - 1 + C] == [1] * C
    assert proc.logprob_masks[-1] == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_get_policy_input_accepts_token_ids():
    p = _packer()
    obs = ObsBundle(representations={"prompt": "3+4="}, id="x")
    completion_ids = p.tokenizer(" 7", add_special_tokens=False).input_ids
    proc = p.get_policy_input(obs, completion_ids)  # list[int] path
    assert proc.input_ids[-len(completion_ids) :] == completion_ids


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_collate_pads_and_masks():
    p = _packer()
    o = ObsBundle(representations={"prompt": "3+4="}, id="x")
    a = p.get_policy_input(o, " 7")
    b = p.get_policy_input(o, " 11 12 13")  # longer completion
    m = p.policy_compute_max_len([a, b])
    assert m == max(len(a.input_ids), len(b.input_ids))
    out = p.policy_collate_fn([a, b], m)
    assert out["input_ids"].shape == (2, m) and out["input_ids"].dtype == torch.long
    assert out["logprob_masks"].shape == (2, m) and out["logprob_masks"].dtype == torch.bool
    assert not out["input_ids"].is_cuda  # CPU-side packer
    # short row padded with pad_token_id, mask False in the pad region
    assert out["input_ids"][0, len(a.input_ids) :].eq(p.pad_token_id).all()
    assert not out["logprob_masks"][0, len(a.input_ids) :].any()


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_collate_empty_raises():
    p = _packer()
    with pytest.raises(ValueError, match="0 samples"):
        p.policy_collate_fn([], 4)


@pytest.mark.L0
@pytest.mark.CPU
def test_sft_process_sample_future():
    with pytest.raises(NotImplementedError, match="future work"):
        _packer().sft_process_sample({})


@pytest.mark.L0
@pytest.mark.CPU
def test_get_rollout_input_passes_raw_string_through():
    p = _packer()  # gpt2-backed helper already in this file
    obs = ObsBundle(representations={"prompt": "3+4="}, id="x")
    assert p.get_rollout_input(obs) == "3+4="  # verbatim, no template


@pytest.mark.L0
@pytest.mark.CPU
def test_get_rollout_input_applies_chat_template_for_conversation():
    p = DecoderOnlyLLMDataPacker()
    p.setup("Qwen/Qwen2.5-0.5B-Instruct")  # has a chat template
    convo = [{"role": "user", "content": "3+4="}]
    obs = ObsBundle(representations={"prompt": convo}, id="x")
    out = p.get_rollout_input(obs)
    assert isinstance(out, str) and "<|im_start|>" in out and out.rstrip().endswith("assistant")


@pytest.mark.L0
@pytest.mark.CPU
def test_assert_no_special_tokens_passes_for_qwen_and_gpt2():
    for m in ("gpt2", "Qwen/Qwen2.5-0.5B-Instruct"):
        p = DecoderOnlyLLMDataPacker()
        p.setup(m)
        p.assert_no_special_tokens()  # no raise — neither adds special tokens


@pytest.mark.L0
@pytest.mark.CPU
def test_assert_no_special_tokens_raises_on_bos_adding_tokenizer():
    p = _packer()  # gpt2

    class _BosTok:  # synthetic tokenizer that adds a BOS when add_special_tokens=True
        def __call__(self, s, add_special_tokens=True, **kw):
            class _R:
                pass

            r = _R()
            r.input_ids = ([1] if add_special_tokens else []) + [ord(c) for c in s]
            return r

    p.tokenizer = _BosTok()
    with pytest.raises(ValueError, match="adds special tokens"):
        p.assert_no_special_tokens()


@pytest.mark.L0
@pytest.mark.CPU
def test_adds_special_tokens_predicate_matches_assert():
    """`adds_special_tokens` is the non-raising predicate behind `assert_no_special_tokens`; the
    two must agree. VLLMRollout uses the predicate (and warns) rather than refusing, since a
    BOS-adding tokenizer is no longer a correctness failure once the objective scores the stored
    `prompt_ids` — and `llama` is a REGISTERED packer that the old hard guard blocked."""
    pytest.importorskip("transformers")
    from cosmos_framework.posttrain.data.packer import init_data_packer

    for name in ("gpt2", "Qwen/Qwen2.5-0.5B-Instruct"):
        p = init_data_packer(name)
        assert p.adds_special_tokens() is False
        p.assert_no_special_tokens()  # agrees: no raise
