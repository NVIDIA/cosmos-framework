# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""A read-ahead source cursor is never checkpointable before its window finishes."""

import pytest

from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer


def _prepared_trainer():
    trainer = _trainer(grad_accum=2)
    trainer.objective.prepare_window = lambda batches: None
    trainer.objective.finish_window = lambda: None
    trainer.prepare_input_window([[None], [None]])
    return trainer


@pytest.mark.parametrize("applied", [False, True], ids=["before_first_microbatch", "before_finish_after_update"])
def test_save_and_load_refuse_unfinished_prefetch_window(tmp_path, applied):
    trainer = _prepared_trainer()
    if applied:
        trainer.train([None])
        trainer.train([None])
    assert trainer._accum == 0
    with pytest.raises(RuntimeError, match="input.*window|prefetch"):
        trainer.save(str(tmp_path / "checkpoint"), data_state={"cursor": 2})
    assert not (tmp_path / "checkpoint").exists()
    with pytest.raises(RuntimeError, match="input.*window|prefetch"):
        trainer.load(str(tmp_path / "absent_checkpoint"))
    if applied:
        trainer.finish_input_window()
        trainer._require_checkpoint_boundary()


def test_failed_prepare_and_worker_cleanup_do_not_reenable_checkpoint_publication(tmp_path):
    trainer = _trainer(grad_accum=2)

    def fail(_batches):
        raise ValueError("worker failed after the source cursor advanced")

    trainer.objective.prepare_window = fail
    with pytest.raises(ValueError, match="worker failed"):
        trainer.prepare_input_window([[None], [None]])
    trainer.close_input_pipeline()
    with pytest.raises(RuntimeError, match="input.*window|prefetch"):
        trainer.save(str(tmp_path / "checkpoint"), data_state={"cursor": 2})
    assert not (tmp_path / "checkpoint").exists()
