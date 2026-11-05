# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The vLLM sampler-head capture contract, shared by GRPO score centering and PPO behavior scores.

Two consumers need the SAME thing from the engine: for every generated token, the ``k`` token ids the
sampler ranked highest and its log-probability at each of them, under the distribution the token was
DRAWN from. That has two halves, and both are rollout infrastructure rather than algorithm logic:

* **Engine construction.** ``SamplingParams(logprobs=k)`` only returns a head when the engine was built
  with ``max_logprobs >= k``; and the head is the sampling distribution only in
  ``logprobs_mode="processed_logprobs"`` (the log-softmax AFTER temperature and any sampler mask), never
  the raw temperature-1 mode. :meth:`SamplerHeadSpec.engine_kwargs` is the one place that spells the two
  keys; :meth:`SamplerHeadSpec.validate_engine_config` refuses a rollout bound to an engine whose
  construction identity does not carry them (a shape-compatible head from a raw-logprob engine would
  produce a silently wrong correction).
* **Parsing.** vLLM returns one ``dict[token_id -> Logprob(logprob, rank, ...)]`` per generated token: the
  entries ranked ``1..k`` plus, when the sampled token fell outside the head, one more entry with
  ``rank > k``. :func:`parse_ranked_rows` enforces exactly the ranks ``1..k`` once each, non-negative
  integer ids, positive integer ranks, finite-or-``-inf`` logprobs no larger than ``1e-6``, and (for the
  PPO caller) the presence of the sampled token's own entry. Anything else is a producer fault and
  raises :class:`SamplerHeadProtocolError`.

The two callers keep their own output schemas (GRPO: ``[T, k]`` int32/float32 tensors in ``policy_info``;
PPO: tuples in its frozen records); only the protocol lives here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PROCESSED_LOGPROBS = "processed_logprobs"
MAX_LOGPROBS_KEY = "max_logprobs"
LOGPROBS_MODE_KEY = "logprobs_mode"
LOGPROB_UPPER_BOUND = 1.0e-6


class SamplerHeadProtocolError(ValueError, RuntimeError):
    """A vLLM logprob record that violates the sampler-head protocol (a producer/engine fault).

    Both ``ValueError`` (PPO's historical type) and ``RuntimeError`` (GRPO's) so either caller's
    ``except`` and tests keep matching.
    """


@dataclass(frozen=True)
class SamplerHeadSpec:
    """How many ranked entries the engine must return per generated token, and how it must be built."""

    top_k: int

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"sampler head top_k must be a positive integer, got {self.top_k!r}")

    def engine_kwargs(self) -> dict[str, Any]:
        """The two ``vllm.LLM`` kwargs the head needs (added on top of the ordinary engine kwargs)."""
        return {MAX_LOGPROBS_KEY: self.top_k, LOGPROBS_MODE_KEY: PROCESSED_LOGPROBS}

    def validate_engine_config(self, engine_config: Mapping[str, Any] | None, *, owner: str) -> None:
        """Refuse an engine whose construction identity cannot have produced this head.

        ``engine_config`` is the exact ``vllm.LLM`` kwargs the engine was built with (the rollout's
        validation identity). ``None`` is refused: without it the likelihood convention is a call-site
        convention, which is exactly what this check exists to remove.
        """
        if engine_config is None:
            raise ValueError(
                f"{owner} captures a sampler head of {self.top_k} but was given no engine_config; pass the exact "
                f"vllm.LLM kwargs so {MAX_LOGPROBS_KEY!r} >= {self.top_k} and {LOGPROBS_MODE_KEY!r} == "
                f"{PROCESSED_LOGPROBS!r} can be verified at construction"
            )
        if not isinstance(engine_config, Mapping):
            raise TypeError(
                f"{owner}: engine_config must be a mapping of vllm.LLM kwargs, got {type(engine_config).__name__}"
            )
        mode = engine_config.get(LOGPROBS_MODE_KEY)
        if mode != PROCESSED_LOGPROBS:
            raise ValueError(
                f"{owner}: the sampler head must come from {LOGPROBS_MODE_KEY}={PROCESSED_LOGPROBS!r} (the distribution the "
                f"token was sampled from); the engine was built with {mode!r}"
            )
        max_logprobs = engine_config.get(MAX_LOGPROBS_KEY)
        if isinstance(max_logprobs, bool) or not isinstance(max_logprobs, int) or max_logprobs < self.top_k:
            raise ValueError(
                f"{owner}: the engine must be built with {MAX_LOGPROBS_KEY} >= {self.top_k} to return the head; got {max_logprobs!r}"
            )


def _check_logprob(value: Any, *, what: str) -> float:
    try:
        logprob = float(value)
    except (TypeError, ValueError) as error:
        raise SamplerHeadProtocolError(f"{what} is not a number: {value!r}") from error
    if math.isnan(logprob) or logprob == math.inf or logprob > LOGPROB_UPPER_BOUND:
        raise SamplerHeadProtocolError(f"{what} has invalid vLLM logprob {logprob!r}")
    return logprob


def parse_ranked_rows(
    rows: Any,
    top_k: int,
    *,
    sampled_token_ids: Sequence[int] | None = None,
) -> tuple[list[list[int]], list[list[float]], list[float] | None]:
    """Parse vLLM's per-token logprob dictionaries into rank-ordered ``[T][k]`` id and logprob rows.

    Args:
        rows: ``CompletionOutput.logprobs`` — one ``Mapping[token_id, Logprob]`` per generated token.
        top_k: the head width; exactly the ranks ``1..top_k`` must be present once in every row.
        sampled_token_ids: when given, ``len(rows)`` must equal its length and every row must carry the
            sampled token's own entry (PPO's behavior logprob); its finite logprobs are returned third.

    Returns:
        ``(ids_rows, logprob_rows, sampled_logprobs)`` with ``sampled_logprobs`` ``None`` when
        ``sampled_token_ids`` is ``None``. A ``-inf`` head logprob is kept (a token the engine masked out
        is a zero sampler mass), a ``-inf`` SAMPLED logprob is refused (the drawn token cannot have zero mass).
    """
    spec = SamplerHeadSpec(top_k)
    if rows is None:
        raise SamplerHeadProtocolError("vLLM returned no logprobs; SamplingParams(logprobs=top_k) must be set")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SamplerHeadProtocolError(
            f"vLLM logprobs must be a sequence of per-token mappings, got {type(rows).__name__}"
        )
    if sampled_token_ids is not None and len(rows) != len(sampled_token_ids):
        raise SamplerHeadProtocolError(
            f"vLLM returned {len(rows)} logprob row(s) for {len(sampled_token_ids)} sampled token(s)"
        )
    expected_ranks = list(range(1, spec.top_k + 1))
    ids_rows: list[list[int]] = []
    lp_rows: list[list[float]] = []
    sampled: list[float] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"vLLM logprob row {position} must be a token-id mapping, got {type(row).__name__}")
        ranked_ids: dict[int, int] = {}
        ranked_lps: dict[int, float] = {}
        for token_id, entry in row.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise SamplerHeadProtocolError(
                    f"vLLM logprob row {position} has a non-integer or negative token id {token_id!r}"
                )
            rank = getattr(entry, "rank", None)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise SamplerHeadProtocolError(
                    f"vLLM logprob entry for token {token_id} at position {position} has no rank (a positive integer is required), got {rank!r}"
                )
            if rank > spec.top_k:
                continue  # the sampled token outside the head; its logprob is read separately
            if rank in ranked_ids:
                raise SamplerHeadProtocolError(f"duplicate top-k rank {rank} at position {position}")
            ranked_ids[rank] = int(token_id)
            ranked_lps[rank] = _check_logprob(
                getattr(entry, "logprob", None), what=f"token {token_id} at position {position}"
            )
        if sorted(ranked_ids) != expected_ranks:
            raise SamplerHeadProtocolError(
                f"vLLM top-k head at position {position} has ranks {sorted(ranked_ids)}, expected 1..{spec.top_k} "
                "(engine built with max_logprobs < top_k, or a vocabulary smaller than top_k?)"
            )
        ids_rows.append([ranked_ids[rank] for rank in expected_ranks])
        lp_rows.append([ranked_lps[rank] for rank in expected_ranks])
        if sampled_token_ids is not None:
            entry = row.get(sampled_token_ids[position])
            if entry is None:
                raise SamplerHeadProtocolError(
                    f"sampled token {sampled_token_ids[position]} is missing from vLLM logprob row {position}"
                )
            value = _check_logprob(
                getattr(entry, "logprob", None), what=f"sampled-action logprob at position {position}"
            )
            if not math.isfinite(value):
                raise SamplerHeadProtocolError(
                    f"sampled-action logprob at position {position} must be finite, got {value!r}"
                )
            sampled.append(value)
    return ids_rows, lp_rows, (sampled if sampled_token_ids is not None else None)
