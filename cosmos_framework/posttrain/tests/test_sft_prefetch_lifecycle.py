# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prefetch consumes exactly one update and preserves native optimizer/cursor behavior."""

import copy

import pytest
import torch

from cosmos_framework.posttrain.config.sft import SFTConfig
from cosmos_framework.posttrain.entrypoints.sft import train_step
from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer
from cosmos_framework.posttrain.tests.test_entrypoints_sft import _runtime


def test_prefetch_preserves_updates_and_source_cursor(tmp_path):
    plain_config, plain_units, plain = _runtime(tmp_path, count=12)
    pref_config, pref_units, pref = _runtime(tmp_path, count=12, preprocessing_prefetch=True)
    observed = []
    finished = []
    pref.objective.prepare_window = lambda batches: observed.append(copy.deepcopy(batches))
    pref.objective.finish_window = lambda: finished.append(pref.iteration())
    for _ in range(3):
        expected = train_step(plain_units["data"], plain_units["policy"], plain_config.rl)
        actual = train_step(pref_units["data"], pref_units["policy"], pref_config.rl)
        assert actual == expected
    assert len(observed) == 3
    assert all(len(window) == 2 and all(len(batch) == 1 for batch in window) for window in observed)
    assert finished == [1, 2, 3]
    for expected, actual in zip(plain.model.parameters(), pref.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    from cosmos_framework.posttrain.utils.entry import resolve

    assert resolve(plain_units["data"].state_dict()) == resolve(pref_units["data"].state_dict())
    for key, state in plain.opt.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(pref.opt.state_dict()["state"][key][name], value, atol=0, rtol=0)


def test_input_pipeline_refuses_partial_accumulation_and_closes():
    trainer = _trainer(grad_accum=2)
    with pytest.raises(ValueError, match="does not support input prefetch"):
        trainer.prepare_input_window([[None], [None]])
    trainer.objective.prepare_window = lambda batches: None
    trainer.prepare_input_window([[None], [None]])
    trainer.train([None])
    with pytest.raises(ValueError, match="complete optimizer-update window"):
        trainer.prepare_input_window([[None], [None]])
    with pytest.raises(RuntimeError, match="inside gradient accumulation"):
        trainer.finish_input_window()
    trainer.train([None])
    trainer.finish_input_window()
    closed = []
    trainer.objective.close = lambda: closed.append(True)
    trainer.close_input_pipeline()
    assert closed == [True]


def test_prepare_failure_never_applies_an_optimizer_update(tmp_path):
    config, units, trainer = _runtime(tmp_path, preprocessing_prefetch=True)
    before = [value.detach().clone() for value in trainer.model.parameters()]

    def fail(batches):
        raise ValueError("worker rejected stochastic preprocessing")

    trainer.objective.prepare_window = fail
    with pytest.raises(ValueError, match="stochastic preprocessing"):
        train_step(units["data"], units["policy"], config.rl)
    assert trainer.iteration() == 0
    for expected, actual in zip(before, trainer.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("prefetch", [False, True])
def test_preprocessing_counters_cross_native_reporter_boundary(tmp_path, prefetch):
    from cosmos_framework.posttrain.utils.report import reportable_metrics, validate_metric_frame

    config, units, trainer = _runtime(tmp_path, preprocessing_cache_bytes=1024, preprocessing_prefetch=prefetch)
    trainer.objective.prepare_window = lambda batches: None
    trainer.objective.finish_window = lambda: None
    trainer.objective.pipeline_stats = lambda: {
        "samples": {"bytes": 64, "limit": 256, "hits": 2, "misses": 1},
        "videos": {"bytes": 128, "limit": 768, "hits": 1, "misses": 2},
    }
    frame = reportable_metrics(train_step(units["data"], units["policy"], config.rl))
    # The public reporter rejects undeclared namespaces before checkpointing.
    validate_metric_frame(frame)
    assert frame["train/preprocessing_samples_bytes"] == 64
    assert frame["train/preprocessing_videos_hits"] == 1
    assert trainer.iteration() == 1


@pytest.mark.parametrize("value", [-1, True, 1.5, "1024"])
def test_cache_limit_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="preprocessing_cache_bytes"):
        SFTConfig(preprocessing_cache_bytes=value)


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_prefetch_requires_boolean(value):
    with pytest.raises(TypeError, match="preprocessing_prefetch"):
        SFTConfig(preprocessing_prefetch=value)
