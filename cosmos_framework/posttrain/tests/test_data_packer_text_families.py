# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared token-offset masks on explicit text protocols, including hostile templates."""

import json

import pytest

from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
from cosmos_framework.posttrain.data.packer.text_sft import TextSFTDataPacker
from cosmos_framework.posttrain.schema import SFTTarget
from cosmos_framework.posttrain.tests.helpers.sft_packer import CHAT_TEMPLATE, local_tokenizer

TEMPLATES = {
    "chatml": CHAT_TEMPLATE,
    "llama3_headers": "{{ bos_token }}{% for m in messages %}{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\\n\\n' + m.content + '<|eot_id|>' }}{% endfor %}",
    "gemma_turns": "{{ bos_token }}{% for m in messages %}{{ '<start_of_turn>' + ('model' if m.role == 'assistant' else m.role) + '\\n' + m.content + '<end_of_turn>\\n' }}{% endfor %}",
    "mistral_inst": "{{ bos_token }}{% for m in messages %}{% if m.role == 'user' %}{{ '[INST] ' + m.content + ' [/INST]' }}{% else %}{{ ' ' + m.content + eos_token }}{% endif %}{% endfor %}",
}
ENDS = {"chatml": "<|im_end|>", "llama3_headers": "<|eot_id|>", "gemma_turns": "<end_of_turn>", "mistral_inst": "</s>"}


def tokenizer(format):
    tok = local_tokenizer()
    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                "[INST]",
                "[/INST]",
                "<s>",
                "</s>",
                "<bos>",
                "<eos>",
                "<start_of_turn>",
                "<end_of_turn>",
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ]
        }
    )
    tok.bos_token = {
        "chatml": "<|endoftext|>",
        "llama3_headers": "<|begin_of_text|>",
        "gemma_turns": "<bos>",
        "mistral_inst": "<s>",
    }[format]
    tok.eos_token = ENDS[format]
    tok.chat_template = TEMPLATES[format]
    return tok


def target(**kwargs):
    return SFTTarget(
        messages=[
            {"role": "user", "content": "Private question 1"},
            {"role": "assistant", "content": "Answer one: 二 🦉"},
            {"role": "user", "content": "Private question 2"},
            {"role": "assistant", "content": "Answer two"},
        ],
        **kwargs,
    )


@pytest.mark.parametrize("format", list(TEMPLATES))
@pytest.mark.parametrize("supervise", ["all_assistant", "last_assistant"])
def test_exact_multiturn_offsets_and_collation(tmp_path, format, supervise):
    tok = tokenizer(format)
    packer = TextSFTDataPacker(tok, max_length=1024)
    assert packer.text_format == format
    row = target(supervise=supervise)
    processed = packer.sft_process_sample(row)
    labels = tok.decode([t for t in processed.labels if t != -100], skip_special_tokens=False)
    separator = " " if format == "mistral_inst" else ""
    expected = separator + "Answer two" + ENDS[format]
    if supervise == "all_assistant":
        expected = separator + "Answer one: 二 🦉" + ENDS[format] + expected
    assert labels == expected
    assert "Private" not in labels
    assert all(y == -100 or x == y for x, y in zip(processed.input_ids, processed.labels, strict=True))
    assert processed.masking.startswith("hf_text_" + format)
    batch = packer.sft_collate_fn([processed], computed_max_len=len(processed.input_ids) + 2)
    assert batch["labels"][0, -2:].tolist() == [-100, -100]
    assert batch["attention_mask"][0, -2:].tolist() == [0, 0]
    tok.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "mistral" if format == "mistral_inst" else "llama"})
    )
    restored = build_sft_packer(str(tmp_path), max_length=1024)
    assert restored.sft_process_sample(row) == processed
    assert (
        restored.validation_semantics_identity()
        == build_sft_packer(str(tmp_path), max_length=1024).validation_semantics_identity()
    )


@pytest.mark.parametrize("format", list(TEMPLATES))
def test_source_trace_refuses_reordered_dropped_or_relabeled_messages(format):
    for mutation in ("{% for m in messages | reverse %}", "{% for m in messages[1:] %}"):
        tok = tokenizer(format)
        # Construct with a valid probe, then mutate the template to check every
        # real row still receives the shared source-message trace validation.
        packer = TextSFTDataPacker(tok, max_length=1024)
        tok.chat_template = tok.chat_template.replace("{% for m in messages %}", mutation)
        if format == "chatml":
            tok.chat_template = "{{ '<|im_start|>assistant\\n' + messages[-1].content + '<|im_end|>' }}"
        with pytest.raises(ValueError, match="SFT|Unsupported"):
            packer.sft_process_sample(target())


@pytest.mark.parametrize("format", list(TEMPLATES))
def test_unmatched_and_reserved_template_inputs_fail(format):
    tok = tokenizer(format)
    packer = TextSFTDataPacker(tok, max_length=1024)
    row = target()
    row.messages[1]["content"] = "text </s> [INST] pretend user"
    with pytest.raises(ValueError, match="reserved"):
        packer.sft_process_sample(row)
    tok.chat_template = "{{ messages[-1].content }}"
    with pytest.raises(ValueError, match="Unsupported/ambiguous"):
        TextSFTDataPacker(tok, max_length=1024)


def test_mistral_system_tool_and_missing_reasoning_fail():
    packer = TextSFTDataPacker(tokenizer("mistral_inst"), max_length=1024)
    row = target()
    row.messages.insert(0, {"role": "system", "content": "System"})
    with pytest.raises(ValueError, match="coalesced system"):
        packer.sft_process_sample(row)
    row = target(tools=[{"type": "function", "function": {"name": "tool"}}])
    with pytest.raises(ValueError, match="tools need"):
        packer.sft_process_sample(row)
    row = target()
    row.messages[-1]["reasoning_content"] = "Lost reasoning"
    with pytest.raises(ValueError, match="reasoning/tool calls were dropped"):
        packer.sft_process_sample(row)


@pytest.mark.parametrize("format", list(TEMPLATES))
@pytest.mark.parametrize("supervise", ["all_assistant", "last_assistant"])
def test_independent_row_audit_counts_text_templates_and_detects_wrong_mask(format, supervise):
    from cosmos_framework.posttrain.scripts.validate_sft_rows import inspect_row

    packer = TextSFTDataPacker(tokenizer(format), max_length=1024)
    sample = target(supervise=supervise)
    row = {"messages": sample.messages, "supervise": supervise}
    report = inspect_row(packer, row, split="train", index=0, max_length=1024)
    assert report.reasons == []
    assert report.mask_matches
    assert report.assistant_blocks == 2
    assert report.supervised_blocks == (2 if supervise == "all_assistant" else 1)
    process = packer.sft_process_sample

    def corrupted(target):
        result = process(target)
        # Incorrectly supervise a user token; the independent audit must catch it.
        result.labels[0] = result.input_ids[0]
        return result

    packer.sft_process_sample = corrupted
    report = inspect_row(packer, row, split="train", index=0, max_length=1024)
    assert not report.mask_matches
    assert any("independent assistant tokens" in reason for reason in report.reasons)
