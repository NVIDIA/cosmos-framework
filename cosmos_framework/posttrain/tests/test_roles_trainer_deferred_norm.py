# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deferred token-mean normalisation across micro-batches (spec §3.5) and the
poisoned-window countdown that keeps a skipped chunk from contaminating the next
optimizer step (spec §5.2)."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.tests.helpers.toy_model import make_toy_trainer, toy_item

# Deliberately UNEQUAL lengths: a mean-of-means implementation (dividing each chunk by its own
# token count instead of the window's total) gives every chunk EQUAL weight regardless of how
# many tokens it holds, which only diverges from the correct Σ/Σ answer when chunks differ in
# size. Equal-length chunks would pass a buggy mean-of-means implementation by coincidence.
_LENGTHS = (3, 11, 5, 7, 2, 13, 4, 9)


def _unequal_length_batch(n: int = 8) -> list[dict]:
    if n > len(_LENGTHS):
        raise ValueError(f"only {len(_LENGTHS)} canonical unequal lengths are defined, got n={n}")
    return [toy_item(i, _LENGTHS[i]) for i in range(n)]


def _ok_chunk() -> list[dict]:
    return [toy_item(900, 3), toy_item(901, 2)]


def _nonfinite_chunk() -> list[dict]:
    return [toy_item(902, 3, poison=True)]


@pytest.mark.L0
@pytest.mark.CPU
def test_chunked_gradient_equals_unchunked():
    """Sequences of UNEQUAL length: a mean-of-means implementation fails this."""
    batch = _unequal_length_batch(n=8)  # lengths [3, 11, 5, 7, 2, 13, 4, 9]

    whole = make_toy_trainer(grad_accum=1)
    whole.train(batch)
    expected = {n: p.grad.clone() for n, p in whole.named_parameters()}

    chunked = make_toy_trainer(grad_accum=4)
    for j in range(4):
        chunked.train(batch[2 * j : 2 * j + 2])
    for name, p in chunked.named_parameters():
        torch.testing.assert_close(p.grad, expected[name], rtol=1e-5, atol=1e-6)


@pytest.mark.L0
@pytest.mark.CPU
def test_all_empty_mini_batch_yields_zero_grads_not_nan():
    """Σ denom == 0 must not become 1/0. See grpo.py:506 — an all-empty batch is reachable."""
    trainer = make_toy_trainer(grad_accum=2)
    for _ in range(2):
        trainer.train([])
    for _, p in trainer.named_parameters():
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_nonfinite_chunk_abandons_the_whole_mini_batch():
    """Chunk 1 of 4 goes non-finite: chunks 2 and 3 are discarded, no step fires for this
    window, and the NEXT window steps on its own four chunks with its own divisor."""
    trainer = make_toy_trainer(grad_accum=4)
    steps_before = trainer.optimizer_step_count

    trainer.train(_ok_chunk())
    trainer.train(_nonfinite_chunk())  # j == 1
    with pytest.raises(RuntimeError, match="poisoned mini-batch"):
        trainer._require_checkpoint_boundary()
    trainer.train(_ok_chunk())  # j == 2 — must be DISCARDED
    trainer.train(_ok_chunk())  # j == 3 — must be DISCARDED
    trainer._require_checkpoint_boundary()
    assert trainer.optimizer_step_count == steps_before  # no step for the poisoned window

    for _ in range(4):
        trainer.train(_ok_chunk())
    assert trainer.optimizer_step_count == steps_before + 1  # exactly one, on clean data
