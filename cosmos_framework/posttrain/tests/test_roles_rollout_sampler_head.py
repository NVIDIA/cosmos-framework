# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The shared vLLM sampler-head capture contract (`roles/rollout/sampler_head.py`), used by GRPO score
centering and PPO behavior scores: engine kwargs, construction-time validation, and the ranked-row parser."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.roles.rollout.sampler_head import (
    SamplerHeadProtocolError,
    SamplerHeadSpec,
    parse_ranked_rows,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _row(entries: dict[int, tuple[float, int]]) -> dict:
    return {tok: SimpleNamespace(logprob=lp, rank=rank) for tok, (lp, rank) in entries.items()}


def test_spec_engine_kwargs_and_validation_are_the_one_source_of_the_likelihood_convention():
    spec = SamplerHeadSpec(128)
    assert spec.engine_kwargs() == {"max_logprobs": 128, "logprobs_mode": "processed_logprobs"}
    spec.validate_engine_config({"max_logprobs": 128, "logprobs_mode": "processed_logprobs", "model": "m"}, owner="t")
    spec.validate_engine_config({"max_logprobs": 200, "logprobs_mode": "processed_logprobs"}, owner="t")
    with pytest.raises(ValueError, match="engine_config"):
        spec.validate_engine_config(None, owner="t")
    with pytest.raises(ValueError, match="processed_logprobs"):
        spec.validate_engine_config({"max_logprobs": 128, "logprobs_mode": "raw_logprobs"}, owner="t")
    with pytest.raises(ValueError, match="processed_logprobs"):
        spec.validate_engine_config({"max_logprobs": 128}, owner="t")  # mode absent: vLLM's default is raw
    with pytest.raises(ValueError, match="max_logprobs"):
        spec.validate_engine_config({"max_logprobs": 20, "logprobs_mode": "processed_logprobs"}, owner="t")
    with pytest.raises(ValueError, match="max_logprobs"):
        spec.validate_engine_config({"logprobs_mode": "processed_logprobs"}, owner="t")
    with pytest.raises(TypeError):
        spec.validate_engine_config(["max_logprobs"], owner="t")  # type: ignore[arg-type]
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            SamplerHeadSpec(bad)  # type: ignore[arg-type]


def test_parser_returns_rank_ordered_rows_and_the_sampled_logprob_for_the_ppo_caller():
    rows = [
        _row({7: (-0.1, 1), 3: (-1.5, 2), 9: (-9.0, 3)}),  # sampled token 9 outside the k=2 head
        _row({3: (-0.2, 1), 7: (-2.0, 2)}),  # sampled token 7 inside the head
    ]
    ids, lps, sampled = parse_ranked_rows(rows, 2, sampled_token_ids=(9, 7))
    assert ids == [[7, 3], [3, 7]] and lps == [[-0.1, -1.5], [-0.2, -2.0]] and sampled == [-9.0, -2.0]
    ids2, lps2, none = parse_ranked_rows(rows, 2)  # the GRPO caller: no sampled ids -> None third
    assert (ids2, lps2, none) == (ids, lps, None)
    minus_inf_head = [_row({1: (-0.5, 1), 2: (-math.inf, 2), 5: (-0.9, 3)})]
    _, lps3, _ = parse_ranked_rows(minus_inf_head, 2)
    assert lps3 == [[-0.5, -math.inf]]  # a masked-out head token is a zero sampler mass and is kept


def test_parser_refuses_every_protocol_violation_with_one_error_type_both_callers_catch():
    ok = _row({7: (-0.1, 1), 3: (-1.5, 2)})
    assert issubclass(SamplerHeadProtocolError, ValueError) and issubclass(SamplerHeadProtocolError, RuntimeError)
    with pytest.raises(SamplerHeadProtocolError, match="no logprobs"):
        parse_ranked_rows(None, 2)
    with pytest.raises(SamplerHeadProtocolError, match="row\\(s\\) for 2 sampled"):
        parse_ranked_rows([ok], 2, sampled_token_ids=(7, 3))
    with pytest.raises(TypeError, match="token-id mapping"):
        parse_ranked_rows([[1, 2]], 2)
    with pytest.raises(SamplerHeadProtocolError, match="negative token id"):
        parse_ranked_rows([_row({-1: (-0.1, 1), 3: (-1.5, 2)})], 2)
    with pytest.raises(SamplerHeadProtocolError, match="has no rank"):
        parse_ranked_rows([{7: SimpleNamespace(logprob=-0.1, rank=None), 3: SimpleNamespace(logprob=-1.5, rank=2)}], 2)
    with pytest.raises(SamplerHeadProtocolError, match="duplicate top-k rank 1"):
        parse_ranked_rows([_row({7: (-0.1, 1), 3: (-1.5, 1)})], 2)
    with pytest.raises(SamplerHeadProtocolError, match="expected 1..2"):
        parse_ranked_rows([_row({7: (-0.1, 1)})], 2)  # engine built with max_logprobs < k
    with pytest.raises(SamplerHeadProtocolError, match="invalid vLLM logprob"):
        parse_ranked_rows([_row({7: (0.5, 1), 3: (-1.5, 2)})], 2)  # positive logprob
    with pytest.raises(SamplerHeadProtocolError, match="missing from vLLM logprob row 0"):
        parse_ranked_rows([ok], 2, sampled_token_ids=(42,))
    with pytest.raises(SamplerHeadProtocolError, match="must be finite"):
        parse_ranked_rows([_row({7: (-0.1, 1), 3: (-1.5, 2), 8: (-math.inf, 3)})], 2, sampled_token_ids=(8,))


def test_grpo_and_ppo_rollouts_share_the_parser_and_the_construction_check():
    """FAIL-ON-OLD: before this module each rollout carried its own copy of the protocol and validated only the
    integer k at construction; now both import the same functions and refuse a raw-logprob engine."""
    import inspect

    from cosmos_framework.posttrain.config.build import ppo_vllm_engine_kwargs, vllm_engine_kwargs
    from cosmos_framework.posttrain.config.rl import RolloutConfig
    from cosmos_framework.posttrain.roles.rollout import ppo as ppo_mod
    from cosmos_framework.posttrain.roles.rollout import vllm as vllm_mod

    assert "parse_ranked_rows(" in inspect.getsource(vllm_mod.VLLMRollout._topk_head)
    assert "parse_ranked_rows(" in inspect.getsource(ppo_mod.PPOVLLMRollout._behavior_scores)
    cfg = RolloutConfig(name_or_path="m")
    grpo_kw = vllm_engine_kwargs(cfg, topk_logprobs=64)
    ppo_kw = ppo_vllm_engine_kwargs(cfg, behavior_top_k=64)
    assert grpo_kw["max_logprobs"] == ppo_kw["max_logprobs"] == 64
    assert grpo_kw["logprobs_mode"] == ppo_kw["logprobs_mode"] == "processed_logprobs"
    fake_packer = SimpleNamespace(adds_special_tokens=lambda: False)
    with pytest.raises(ValueError, match="engine_config"):
        ppo_mod.PPOVLLMRollout(
            object(), packer=fake_packer, sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4}, behavior_top_k=8
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="processed_logprobs"):
        ppo_mod.PPOVLLMRollout(
            object(),  # type: ignore[arg-type]
            packer=fake_packer,
            sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4},
            behavior_top_k=8,
            engine_config={"max_logprobs": 8, "logprobs_mode": "raw_logprobs"},
        )
