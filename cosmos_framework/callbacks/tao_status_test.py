# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from cosmos_framework.callbacks.tao_status import TAOStatusCallback


def _callback(tmp_path) -> TAOStatusCallback:
    callback = TAOStatusCallback(
        enabled=True,
        status_file_path=str(tmp_path / "status.json"),
        experiment_name="test",
    )
    callback.config = SimpleNamespace(
        job=SimpleNamespace(name="job", path_local=str(tmp_path)),
        trainer=SimpleNamespace(max_iter=6, max_val_iter=2, logging_iter=1),
    )
    callback.trainer = SimpleNamespace(max_iterations=6, num_epochs=2, steps_per_epoch=3)
    return callback


def _records(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "status.json").read_text(encoding="utf-8").splitlines()]


def test_tao_status_callback_writes_training_validation_and_success(tmp_path) -> None:
    callback = _callback(tmp_path)
    callback.on_train_start(model=None, iteration=0)
    callback.on_training_step_start(model=None, data={}, iteration=0)
    callback.on_training_step_batch_end(
        model=None,
        data_batch={"input_ids": torch.zeros(2, 3)},
        output_batch={"loss_numerator": torch.tensor(6.0), "loss_denominator": torch.tensor(12)},
        loss=torch.tensor(0.5),
        iteration=3,
    )
    callback.on_training_step_end(
        model=None,
        data_batch={"input_ids": torch.zeros(2, 3)},
        output_batch={"loss_numerator": torch.tensor(6.0), "loss_denominator": torch.tensor(12)},
        loss=torch.tensor(0.5),
        iteration=3,
    )
    callback.on_validation_start(model=None, dataloader_val=None, iteration=3)
    callback.on_validation_step_end(
        model=None,
        data_batch={"input_ids": torch.zeros(2, 3)},
        output_batch={"loss_numerator": torch.tensor(2.0), "loss_denominator": torch.tensor(8)},
        loss=torch.tensor(0.25),
        iteration=3,
    )
    callback.on_validation_end(model=None, iteration=3)
    callback.on_train_end(model=None, iteration=6)
    callback.on_app_end()

    records = _records(tmp_path)
    assert [record["status"] for record in records] == [
        "STARTED",
        "RUNNING",
        "RUNNING",
        "RUNNING",
        "RUNNING",
        "RUNNING",
        "SUCCESS",
    ]
    assert records[1]["kpi"]["train/step_loss"] == 0.5
    assert records[1]["epoch"] == 1
    assert records[1]["max_epoch"] == 2
    assert records[1]["step_in_epoch"] == 3
    assert records[1]["steps_per_epoch"] == 3
    assert records[1]["max_step"] == 6
    assert records[2]["phase"] == "validation_starting"
    assert records[3]["max_validation_batches"] == 2
    assert "epoch 1/2" in records[3]["message"]
    assert records[4]["kpi"]["val/avg_loss"] == 0.25
    assert records[4]["kpi"]["val/loss_numerator"] == 2.0
    assert records[4]["kpi"]["val/valid_label_count"] == 8
    assert "epoch 1/2" in records[4]["message"]
    assert records[5]["phase"] == "training_complete"
    assert records[5]["kpi"]["train/avg_loss"] == 0.5
    assert records[5]["train_loss_numerator"] == 6.0
    assert records[5]["train_valid_label_count"] == 12
    assert records[5]["kpi"]["train/valid_label_count"] == 12
    assert records[-1]["kpi"]["val/loss"] == 0.25
    assert records[-1]["epoch"] == 2
    assert records[-1]["completed_epochs"] == 2


def test_tao_status_callback_reports_checkpoint_event(tmp_path) -> None:
    callback = _callback(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "epoch_1"
    checkpoint.mkdir(parents=True)
    (tmp_path / "checkpoints" / "latest_checkpoint.txt").write_text("epoch_1\n")
    callback.on_save_checkpoint_success(iteration=3, elapsed_time=1.25)
    record = _records(tmp_path)[-1]
    assert record["phase"] == "checkpoint_complete"
    assert record["checkpoint_iteration"] == 3
    assert record["checkpoint_elapsed_seconds"] == 1.25
    assert record["checkpoint_path"] == str(checkpoint.resolve())


def test_tao_status_callback_ignores_dcp_marker_nul_padding(tmp_path) -> None:
    callback = _callback(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "epoch_1"
    checkpoint.mkdir(parents=True)
    (tmp_path / "checkpoints" / "latest_checkpoint.txt").write_bytes(b"epoch_1\x00\x00\x00")

    callback.on_save_checkpoint_success(iteration=3, elapsed_time=1.25)

    record = _records(tmp_path)[-1]
    assert record["phase"] == "checkpoint_complete"
    assert record["checkpoint_path"] == str(checkpoint.resolve())


def test_tao_status_callback_writes_failure(tmp_path) -> None:
    callback = _callback(tmp_path)
    callback.on_train_start(model=None, iteration=0)
    callback.on_exception(RuntimeError("boom"))

    records = _records(tmp_path)
    assert records[-1]["status"] == "FAILURE"
    assert records[-1]["error_type"] == "RuntimeError"
