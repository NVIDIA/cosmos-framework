# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Who takes which run. Getting this wrong cost a night of cluster time twice over.

Both failures were silent in the sense that matters: nothing raised where the mistake was made.
A rank-local skip surfaced later as an NCCL abort in a different part of the stack, and a stranded
claim surfaced as a batch that simply stopped finishing.
"""

from __future__ import annotations

import os
import time

from failuregen.runner import agree, all_ok, decide


def test_a_free_run_is_taken_and_claimed(tmp_path):
    run = tmp_path / "run"
    assert decide(run, claim_timeout_s=900) == "take"
    assert (run / ".claim").is_file()


def test_a_finished_run_is_skipped_as_done(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "synth.mp4").write_bytes(b"")
    assert decide(run, claim_timeout_s=900) == "done"
    assert not (run / ".claim").exists()  # no need to claim what is already rendered


def test_a_live_claim_is_left_to_its_owner(tmp_path):
    run = tmp_path / "run"
    assert decide(run, claim_timeout_s=900) == "take"
    assert decide(run, claim_timeout_s=900) == "claimed"


def test_a_claim_from_a_dead_worker_is_stolen(tmp_path):
    """A worker that dies mid-run leaves a claim with no output. Without an expiry that run is
    stranded for the rest of the batch, and the batch never finishes."""
    run = tmp_path / "run"
    assert decide(run, claim_timeout_s=900) == "take"
    claim = run / ".claim"
    stale = time.time() - 3600
    os.utime(claim, (stale, stale))

    assert decide(run, claim_timeout_s=900) == "take"
    assert claim.stat().st_mtime > stale  # stealing resets the clock for the next worker
    assert decide(run, claim_timeout_s=900) == "claimed"


def test_a_stale_claim_on_a_finished_run_is_still_done(tmp_path):
    run = tmp_path / "run"
    decide(run, claim_timeout_s=900)
    (run / "synth.mp4").write_bytes(b"")
    stale = time.time() - 3600
    os.utime(run / ".claim", (stale, stale))
    assert decide(run, claim_timeout_s=900) == "done"  # output wins over an expired claim


def test_the_agreement_helpers_pass_through_without_a_process_group():
    """Single-process runs must behave exactly as if every rank agreed, so the same code path
    works on a laptop and on four ranks."""
    assert agree("take") == "take"
    assert agree(None) is None
    assert all_ok(True) is True
    assert all_ok(False) is False
