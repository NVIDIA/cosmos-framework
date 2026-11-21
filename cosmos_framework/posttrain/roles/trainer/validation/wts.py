# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact WTS teacher-forced scoring shared by native and fresh-process evaluation.

This is restricted-choice and first-token scoring, not free generation. The
native SFT objective owns causal shifts, masks and supervised-token reductions.
No data source, training cursor, optimizer or publication state is owned here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

LETTERS = ("A", "B", "C", "D")

OPTION = re.compile(r"(?m)^[ \t]*([A-Z])([:.])[ \t]+(?=\S)")

OPTION_MARKER = re.compile(r"(?m)^[ \t]*([A-Za-z])([:.])")


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_choices(choices: tuple[str, ...]) -> None:
    if not 2 <= len(choices) <= len(LETTERS) or choices != LETTERS[: len(choices)]:
        raise ValueError(
            "WTS requires two to four ordered contiguous choices starting at A; no gaps, extra, or duplicate letters"
        )


def parse_options(question: str) -> tuple[str, ...]:
    # Inspect every marker, including empty/malformed options. Otherwise an empty
    # final D option could silently turn an invalid four-choice row into A/B/C.
    matches = list(OPTION_MARKER.finditer(question))
    choices = tuple(match[1] for match in matches)
    validate_choices(choices)
    if any(OPTION.match(question, match.start()) is None for match in matches):
        raise ValueError("each WTS option marker requires whitespace and nonempty text on the same line")
    if len({match[2] for match in matches}) != 1:
        raise ValueError("WTS option lines must consistently use ':' or '.'")
    if not question[: matches[0].start()].strip():
        raise ValueError("WTS question before the options must not be empty")
    return choices


def validate_wts(row: Any) -> tuple[str, str, tuple[str, ...], str]:
    if not isinstance(row, dict) or isinstance(row.get("id"), bool) or not isinstance(row.get("id"), (str, int)):
        raise TypeError("each WTS row requires a string/integer id")
    identifier = str(row["id"])
    if not identifier:
        raise ValueError("WTS row id must not be empty")
    messages = row.get("messages")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("WTS requires a messages list")
    roles = [message.get("role") for message in messages]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        raise ValueError("WTS scorer supports one user/assistant exchange and an optional leading system message")
    if any(message.get("tool_calls") or message.get("reasoning_content") for message in messages) or row.get("tools"):
        raise ValueError("WTS scorer does not accept tool calls or reasoning-content answers")
    answer = messages[-1].get("content")
    if not isinstance(answer, str) or answer not in LETTERS:
        raise ValueError("WTS gold assistant answer must be exactly one uppercase letter A/B/C/D")
    content = messages[-2].get("content")
    if not isinstance(content, list) or any(not isinstance(part, dict) for part in content):
        raise ValueError("WTS user content requires native structured text and video")
    videos = [part.get("video") for part in content if part.get("type") == "video"]
    if len(videos) != 1 or not isinstance(videos[0], str) or not videos[0]:
        raise ValueError("each WTS question must condition on exactly one local video")
    if any(part.get("type") not in {"text", "video"} for part in content):
        raise ValueError("WTS scorer supports only text and video content parts")
    texts = [part.get("text") for part in content if part.get("type") == "text"]
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("WTS question requires nonempty text content")
    choices = parse_options("\n".join(texts))
    if answer not in choices:
        raise ValueError("WTS gold answer must belong to the actual question choices")
    if row.get("opener", "self") != "self":
        raise ValueError("single-letter WTS answers require opener=self")
    return identifier, answer, choices, videos[0]


def answer_alignment(
    tokenizer: Any,
    input_ids: list[int],
    labels: list[int],
    answer: str,
    *,
    choices: tuple[str, ...] = LETTERS,
) -> tuple[int, dict[str, int]]:
    validate_choices(choices)
    if answer not in choices:
        raise ValueError("gold answer must belong to the actual question choices")
    if len(input_ids) != len(labels):
        raise ValueError("input and label lengths disagree")
    supervised = [position for position, label in enumerate(labels) if label != -100]
    if not supervised or supervised[0] < 1:
        raise ValueError("answer must have a supervised token and a preceding prediction position")
    position = supervised[0]
    prefix_ids = input_ids[:position]
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not prefix.endswith("<|im_start|>assistant\n"):
        raise ValueError("first supervised label must immediately follow the native assistant header")
    if tokenizer(prefix, add_special_tokens=False)["input_ids"] != prefix_ids:
        raise ValueError("assistant prefix does not round-trip to the exact expanded input tokens")
    candidates = {}
    for letter in choices:
        encoded = tokenizer(prefix + letter, add_special_tokens=False)["input_ids"]
        if len(encoded) != position + 1 or encoded[:position] != prefix_ids:
            raise ValueError(f"candidate {letter} is not exactly one context-stable next token")
        token = encoded[-1]
        if tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) != letter:
            raise ValueError(f"candidate {letter} token does not decode to the exact letter")
        candidates[letter] = token
    if len(set(candidates.values())) != len(choices):
        raise ValueError("answer letters must have distinct token IDs")
    if answer not in candidates or labels[position] != candidates[answer] or input_ids[position] != candidates[answer]:
        raise ValueError("first supervised label is not the exact gold answer token")
    return position, candidates


def score_answer(logits: Any, answer_position: int, candidates: dict[str, int], answer: str) -> dict[str, Any]:
    if logits.ndim != 3 or logits.shape[0] != 1 or not 0 < answer_position < logits.shape[1]:
        raise ValueError("expected [1, sequence, vocabulary] logits and a valid answer position")
    # Labels are unshifted: the token at p is predicted by logits at p-1.
    return score_prediction(logits[0, answer_position - 1], answer_position, candidates, answer)


def score_prediction(scores: Any, answer_position: int, candidates: dict[str, int], answer: str) -> dict[str, Any]:
    """Score one full-vocabulary prediction; shared by dense and selected-head layouts."""
    import torch

    if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or answer_position < 1:
        raise ValueError("expected full-vocabulary prediction scores and a positive answer position")
    choices = tuple(candidates)
    validate_choices(choices)
    if len(set(candidates.values())) != len(choices) or answer not in candidates:
        raise ValueError("expected distinct ordered answer token IDs and gold among actual choices")
    scores = scores.detach().float()
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite answer-position logits")
    if any(not 0 <= token < scores.shape[0] for token in candidates.values()):
        raise ValueError("answer token is outside model vocabulary")
    ids = torch.tensor(list(candidates.values()), device=scores.device)
    choice_scores = scores[ids]
    best_choice_index = int(choice_scores.argmax())
    choice_prediction = choices[best_choice_index]
    first_token = int(scores.argmax())
    reverse = {token: letter for letter, token in candidates.items()}
    gold_id = candidates[answer]
    return {
        "gold": answer,
        "candidate_token_ids": candidates,
        "answer_token_position": answer_position,
        "prediction_logit_position": answer_position - 1,
        "restricted_choice_prediction": choice_prediction,
        "restricted_choice_correct": choice_prediction == answer,
        "strict_first_token_id": first_token,
        "strict_first_token_letter": reverse.get(first_token),
        "strict_first_token_correct": first_token == gold_id,
        "valid_first_letter": first_token in reverse,
        "answer_nll": float(torch.logsumexp(scores, dim=0) - scores[gold_id]),
        "restricted_choice_nll": float(torch.logsumexp(choice_scores, dim=0) - scores[gold_id]),
        "candidate_logits": dict(zip(choices, choice_scores.cpu().tolist(), strict=True)),
        "restricted_choice_tied_max": int((choice_scores == choice_scores.max()).sum()) > 1,
        "strict_first_token_tied_max": int((scores == scores.max()).sum()) > 1,
    }


def evaluate_row(model: Any, packer: Any, objective: Any, row: dict[str, Any]) -> dict[str, Any]:
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
    from cosmos_framework.posttrain.roles.trainer.validation.sft import (
        evaluate_sft_batch,
    )
    from cosmos_framework.posttrain.schema import (
        ObsBundle,
        SFTTarget,
        Trajectory,
        Transition,
    )

    _, answer, choices, _ = validate_wts(row)
    target = SFTTarget(
        messages=row["messages"],
        tools=row.get("tools"),
        supervise=row.get("supervise", "all_assistant"),
        opener=row.get("opener", "self"),
    )
    captured: dict[str, Any] = {}
    collate = packer.sft_collate_fn

    def observe_batch(samples, *args, **kwargs):
        import torch

        if len(samples) != 1 or "inputs" in captured:
            raise RuntimeError("heldout scorer requires exactly one native collation per row")
        batch = collate(samples, *args, **kwargs)
        position, candidates = answer_alignment(
            packer.tokenizer,
            samples[0].input_ids,
            samples[0].labels,
            answer,
            choices=choices,
        )
        valid = (
            (batch["labels"][:, 1:] != -100)
            & batch["attention_mask"][:, 1:].bool()
            & batch["attention_mask"][:, :-1].bool()
        )
        selected_positions = torch.nonzero(valid[0]).flatten()
        answer_rows = torch.nonzero(selected_positions == position - 1).flatten()
        if answer_rows.numel() != 1:
            raise ValueError("gold answer prediction must occur exactly once in the native supervised selection")
        captured.update(
            position=position,
            candidates=candidates,
            inputs={name: tensor_digest(value.contiguous()) for name, value in sorted(batch.items())},
            sequence_length=len(samples[0].input_ids),
            supervised_tokens=int(valid.sum()),
            answer_selected_row=int(answer_rows.item()),
        )
        if "pixel_values_videos" not in batch or not batch["pixel_values_videos"].numel():
            raise RuntimeError("native heldout row lost actual video conditioning")
        return batch

    def observe_inputs(_module, _args, kwargs):
        if "forward_seen" in captured or "inputs" not in captured:
            raise RuntimeError("heldout scorer requires exactly one native forward after collation")
        for name, expected in captured["inputs"].items():
            if name != "labels" and (name not in kwargs or tensor_digest(kwargs[name].contiguous()) != expected):
                raise RuntimeError(f"native forward changed the observed {name} input")
        captured["forward_seen"] = True

    def observe_forward(_module, _args, kwargs, output):
        observe_inputs(_module, _args, kwargs)
        import torch

        logits = output if isinstance(output, torch.Tensor) else output.logits
        captured["scores"] = score_answer(logits, captured["position"], captured["candidates"], answer)

    def observe_selected_head(_module, _args, output):
        import torch

        if not captured.get("forward_seen") or "scores" in captured:
            raise RuntimeError("selected head must execute once after the observed native backbone")
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != captured["supervised_tokens"]:
            raise ValueError("selected head must return [supervised tokens, full vocabulary] logits")
        captured["scores"] = score_prediction(
            output[captured["answer_selected_row"]], captured["position"], captured["candidates"], answer
        )

    packer.sft_collate_fn = observe_batch
    handles = []
    try:
        if getattr(model, "sft_logits_mode", "full") == "supervised":
            handles.append(model.net.model.register_forward_pre_hook(observe_inputs, with_kwargs=True))
            handles.append(model.net.lm_head.register_forward_hook(observe_selected_head))
        else:
            handles.append(model.net.register_forward_hook(observe_forward, with_kwargs=True))
        metrics = evaluate_sft_batch(
            model,
            objective,
            [Trajectory(transitions=[Transition(obs=ObsBundle(), targets=target)])],
        )
    finally:
        for handle in handles:
            handle.remove()
        packer.sft_collate_fn = collate
    loss, count = float(metrics["sft_loss"]), int(metrics["sft_valid_tokens"])
    if not math.isfinite(loss) or count < 1 or count != captured["supervised_tokens"]:
        raise RuntimeError("native SFT loss/token count is invalid or differs from observed supervision")
    return {
        **captured["scores"],
        "native_sft_ce": loss,
        "native_supervised_tokens": count,
        "sequence_length": captured["sequence_length"],
        "inputs": captured["inputs"],
        "input_sha256": json_digest(captured["inputs"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate zero rows")
    count = len(rows)
    tokens = sum(row["native_supervised_tokens"] for row in rows)
    return {
        "rows": count,
        "native_supervised_tokens": tokens,
        "restricted_choice_accuracy": sum(row["restricted_choice_correct"] for row in rows) / count,
        "strict_first_token_accuracy": sum(row["strict_first_token_correct"] for row in rows) / count,
        "valid_letter_rate": sum(row["valid_first_letter"] for row in rows) / count,
        "token_weighted_ce": math.fsum(row["native_sft_ce"] * row["native_supervised_tokens"] for row in rows) / tokens,
        "answer_nll": math.fsum(row["answer_nll"] for row in rows) / count,
        "restricted_choice_nll": math.fsum(row["restricted_choice_nll"] for row in rows) / count,
        "restricted_choice_tied_rows": sum(row["restricted_choice_tied_max"] for row in rows),
        "strict_first_token_tied_rows": sum(row["strict_first_token_tied_max"] for row in rows),
        "unique_exact_video_content_clusters": len({row["media_sha256"] for row in rows}),
    }
