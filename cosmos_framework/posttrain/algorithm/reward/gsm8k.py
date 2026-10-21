# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GSM8K verifiable reward -- cosmos-rl's `gsm8k_reward_fn`, ported VERBATIM.

DATASET-SPECIFIC, which is why it is here and not in `reward.py`. The regex, the normalization and
the `except: return 0.0` guard are parity surface: they must match the reference exactly, so they are
kept where a divergence is visible rather than mixed into the generic machinery. Sitting between a
Protocol and a dataclass in a file called `reward.py`, this reads as general code and invites being
"improved" -- and any such edit silently breaks parity.

Its paired half is `configs/grpo_qwen3b_gsm8k.py::GSM8K_SYSTEM_PROMPT`, which exists to make a model emit the
`#### <number>` form this parses. The two are one contract split across two roles. That coupling is
why the prompt wording matters so much here: the verifier is EXACT-MATCH, so `#### 72.` scores 0
against gold `72`, and a laxer instruction costs ~0.20 mean reward (docs/design/grpo-cosmos-rl-parity.md).

NOTE: cosmos-rl's `gsm8k` reward does NOT use `math_verify` -- that library backs only its
`boxed_math_reward_fn`. Parity means porting the regex and taking no such dependency.
"""

from __future__ import annotations

import re

from cosmos_framework.posttrain.algorithm.reward.group import RewardCandidate, RewardContext
from cosmos_framework.posttrain.algorithm.reward.result import RewardProviderResult


def extract_gsm8k_answer(solution_str: str) -> str | None:
    """Strict `#### (number)` extraction (cosmos-rl gsm8k parity).

    Args:
        solution_str: model completion or reference answer text.

    Returns:
        The normalized number string after ``#### `` (commas/`$` stripped), or None if absent.
    """
    m = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str or "")
    if m is None:
        return None
    return m.group(0).split("#### ")[1].replace(",", "").replace("$", "")


def gsm8k_reward_fn(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """GSM8K verifiable reward (cosmos-rl `gsm8k_reward_fn`, VERBATIM regex; no math_verify).

    Compares the strict `#### N` answer parsed from the completion against the reference.

    Args:
        candidate: The immutable candidate completion.
        context: The common group context carrying the gold reference.

    Returns:
        A result whose ``task`` term is 1.0 when the extracted answers match and both are non-None;
        otherwise 0.0.
    """
    completion = candidate.completion
    reference = context.reference
    try:
        pred = extract_gsm8k_answer(completion)
        gold = extract_gsm8k_answer(reference)  # type: ignore[arg-type]
        return RewardProviderResult(terms={"task": 1.0 if (pred is not None and pred == gold) else 0.0})
    except Exception:  # noqa: BLE001 — parity with cosmos-rl's guard
        return RewardProviderResult(terms={"task": 0.0})
