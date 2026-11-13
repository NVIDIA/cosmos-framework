# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Text-only public preparation preserves the tokenizer's native chat template."""

import copy

import pytest
import torch

from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime


def text_runtime(model_type):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["<unk>", "<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "short", "long", "question", "answer"]
    backend = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    # Gemma's ordinary tokenizer adds BOS; its native chat template already owns
    # that token. Smol's template owns role tokens without a default BOS insertion.
    if model_type == "gemma":
        backend.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    template = "{{ bos_token }} " if model_type == "gemma" else ""
    template += (
        "{% for message in messages %}<{{ message['role'] }}> {{ message['content'] }} <eos> {% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=["<user>", "<assistant>"],
        chat_template=template,
        padding_side="left",
    )
    runtime = object.__new__(CosmosFrameworkRuntime)
    runtime.model_type = model_type
    runtime.processor = tokenizer
    return runtime


@pytest.mark.parametrize("model_type", ["gemma", "llama"])
@pytest.mark.parametrize("batched", [False, True])
def test_public_text_ids_equal_native_template_tokenization(model_type, batched):
    runtime = text_runtime(model_type)
    conversations = [[{"role": "user", "content": "short question"}]]
    if batched:
        conversations.append(
            [
                {"role": "user", "content": "long long question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "question"},
            ]
        )
    before = copy.deepcopy(conversations)
    expected = runtime.processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True
    )
    if batched:
        actual = runtime._prepare_transformers_batch(conversations)
    else:
        actual = runtime._prepare_transformers_inputs(conversations[0])
    assert torch.equal(actual["input_ids"], expected["input_ids"])
    assert torch.equal(actual["attention_mask"], expected["attention_mask"])
    assert conversations == before
    for ids, mask in zip(actual["input_ids"], actual["attention_mask"]):
        unpadded = ids[mask.bool()]
        assert unpadded.eq(runtime.processor.bos_token_id).sum().item() == int(model_type == "gemma")
        assert unpadded[-1].item() == runtime.processor.convert_tokens_to_ids("<assistant>")
    if batched:
        assert not actual["attention_mask"][0, 0]  # Unequal lengths really exercise left padding.


@pytest.mark.parametrize("model_type", ["gemma", "llama"])
def test_text_fragments_normalize_before_native_template_without_mutation(model_type):
    runtime = text_runtime(model_type)
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "short"}, {"type": "text", "text": "question"}]}
    ]
    before = copy.deepcopy(conversation)
    actual = runtime._prepare_transformers_inputs(conversation)
    expected = runtime.processor.apply_chat_template(
        [{"role": "user", "content": "short\nquestion"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    assert torch.equal(actual["input_ids"], expected["input_ids"])
    assert torch.equal(actual["attention_mask"], expected["attention_mask"])
    assert conversation == before


def test_fixture_exercises_duplicate_special_token_regression():
    runtime = text_runtime("gemma")
    conversation = [{"role": "user", "content": "short question"}]
    expected = runtime.processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    rendered = runtime.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    assert runtime.processor(rendered)["input_ids"] == [runtime.processor.bos_token_id, *expected]
