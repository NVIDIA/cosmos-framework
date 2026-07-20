# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

if "megatron.core" not in sys.modules:
    megatron_module = ModuleType("megatron")
    megatron_core_module = ModuleType("megatron.core")
    megatron_core_module.parallel_state = SimpleNamespace()  # type: ignore[attr-defined]
    megatron_module.core = megatron_core_module  # type: ignore[attr-defined]
    sys.modules["megatron"] = megatron_module
    sys.modules["megatron.core"] = megatron_core_module

import cosmos_framework.callbacks.dmd2_metrics as ledger_module
from cosmos_framework.callbacks.dmd2_metrics import DMD2ParityLedger
from cosmos_framework.callbacks.grad_clip_distillation import GradClip


class _PhaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(grad_clip=True)
        self._distillation_parity_grad_metrics: dict[str, float] = {}

    def get_phase(self, iteration: int) -> str:
        return "student" if iteration % 5 == 0 else "critic"

    def get_optimizer_key(self, iteration: int) -> str:
        return "net" if self.get_phase(iteration) == "student" else "fake_score"

    def is_image_batch(self, data_batch: dict[str, Any]) -> bool:
        return bool(data_batch.get("is_image", False))


class _PhaseOptimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter

    def parameters_for_key(self, key: str) -> list[torch.nn.Parameter]:
        assert key == "net"
        return [self.parameter]


@pytest.mark.L0
@pytest.mark.CPU
def test_disabled_ledger_does_not_write_or_run_collectives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "disabled.jsonl"
    callback = DMD2ParityLedger(output_path="")
    model = _PhaseModel()

    def _unexpected_collective(payload: object) -> list[object]:
        raise AssertionError(f"disabled ledger gathered {payload!r}")

    monkeypatch.setattr(ledger_module.distributed, "all_gather_object", _unexpected_collective)
    callback.on_training_step_end(
        model,
        {"sample_id": ["sample-a"]},
        {"fake_score_loss": torch.tensor(1.0)},  # []
        torch.tensor(1.0),  # []
        iteration=1,
    )

    assert not output_path.exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_ledger_writes_sorted_sample_keys_phase_rng_and_present_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "parity.jsonl"
    callback = DMD2ParityLedger(output_path=str(output_path))
    model = _PhaseModel()
    model._distillation_parity_grad_metrics = {
        "clip_grad_norm/net_selected_preclip": 3.0,
        "clip_grad_norm/net_selected_clip_scale": 0.5,
        "clip_grad_norm/net_selected_clip_norm": 1.5,
    }
    monkeypatch.setattr(ledger_module, "_local_rng_checksum", lambda: "rank-0-rng")
    reduced_metrics: list[torch.Tensor] = []

    def reduce_metric(metric: torch.Tensor, op: object) -> None:  # metric: []
        assert op == torch.distributed.ReduceOp.AVG
        reduced_metrics.append(metric.clone())  # []

    monkeypatch.setattr(ledger_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(ledger_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ledger_module.dist, "all_reduce", reduce_metric)
    monkeypatch.setattr(ledger_module.distributed, "all_gather_object", lambda payload: [payload])
    monkeypatch.setattr(ledger_module.distributed, "is_rank0", lambda: True)

    callback.on_training_step_end(
        model,
        {
            "sample_id": ["sample-b", "sample-a"],
            "images": torch.tensor([[1.0], [2.0]]),  # [B,C]
        },
        {
            "vsd_loss": torch.tensor(2.0),  # []
            "total_generator_loss": torch.tensor(4.0),  # []
            "ignored_metric": torch.tensor(9.0),  # []
        },
        torch.tensor(4.0),  # []
        iteration=5,
    )

    record = json.loads(output_path.read_text().strip())
    assert record["iteration"] == 5
    assert record["phase"] == "critic"
    assert record["sample_keys"] == ["sample-a", "sample-b"]
    assert len(record["input_digest"]) == 64
    assert len(record["rng_checksum"]) == 64
    assert record["vsd_loss"] == 2.0
    assert record["total_generator_loss"] == 4.0
    assert record["clip_grad_norm/net_selected_preclip"] == 3.0
    assert record["clip_grad_norm/net_selected_clip_scale"] == 0.5
    assert record["clip_grad_norm/net_selected_clip_norm"] == 1.5
    assert "ignored_metric" not in record
    assert len(reduced_metrics) == 5


@pytest.mark.L0
@pytest.mark.CPU
def test_input_digest_is_independent_of_mapping_order(monkeypatch: pytest.MonkeyPatch) -> None:
    first_batch = {
        "sample_id": ["sample-a"],
        "images": torch.tensor([[1.0, 2.0]]),  # [B,C]
        "metadata": {"height": 1, "width": 2},
    }
    reordered_batch = {
        "metadata": {"width": 2, "height": 1},
        "images": torch.tensor([[1.0, 2.0]]),  # [B,C]
        "sample_id": ["sample-a"],
    }
    monkeypatch.setattr(ledger_module.distributed, "all_gather_object", lambda payload: [payload])

    assert ledger_module._input_digest(first_batch) == ledger_module._input_digest(reordered_batch)


@pytest.mark.L0
@pytest.mark.CPU
def test_input_digest_ignores_worker_timing_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    first_batch = {
        "sample_id": ["sample-a"],
        "images": torch.tensor([[1.0, 2.0]]),  # [B,C]
        "_worker_io_time": torch.tensor([0.1]),  # [B]
        "_worker_aug_step_times": {"resize": 0.2},
    }
    later_batch = {
        "sample_id": ["sample-a"],
        "images": torch.tensor([[1.0, 2.0]]),  # [B,C]
        "_worker_io_time": torch.tensor([9.9]),  # [B]
        "_worker_aug_step_times": {"resize": 8.8},
    }
    monkeypatch.setattr(ledger_module.distributed, "all_gather_object", lambda payload: [payload])

    assert ledger_module._input_digest(first_batch) == ledger_module._input_digest(later_batch)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("iteration", "expected_phase"),
    [
        (1, "student"),
        (5, "critic"),
        (6, "student"),
    ],
)
def test_ledger_phase_describes_completed_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    iteration: int,
    expected_phase: str,
) -> None:
    output_path = tmp_path / "parity.jsonl"
    callback = DMD2ParityLedger(output_path=str(output_path))
    model = _PhaseModel()
    monkeypatch.setattr(ledger_module, "_local_rng_checksum", lambda: "rank-0-rng")
    monkeypatch.setattr(ledger_module.distributed, "all_gather_object", lambda payload: [payload])
    monkeypatch.setattr(ledger_module.distributed, "is_rank0", lambda: True)

    callback.on_training_step_end(
        model,
        {"sample_id": ["sample-a"]},
        {},
        torch.tensor(0.0),  # []
        iteration=iteration,
    )

    record = json.loads(output_path.read_text().strip())
    assert record["iteration"] == iteration
    assert record["phase"] == expected_phase


@pytest.mark.L0
@pytest.mark.CPU
def test_grad_clip_exposes_computed_phase_metrics() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))  # [2]
    parameter.grad = torch.tensor([3.0, 4.0])  # [2]
    model = _PhaseModel()
    optimizer = _PhaseOptimizer(parameter)
    callback = GradClip(clip_norm=10.0)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=10))
    callback.on_training_step_start(model, {"is_image": True}, iteration=5)

    callback.on_before_optimizer_step(model, optimizer, None, None, iteration=5)  # type: ignore[arg-type]

    assert model._distillation_parity_grad_metrics == {
        "clip_grad_norm/net_selected_preclip": 5.0,
        "clip_grad_norm/net_selected_clip_scale": 1.0,
        "clip_grad_norm/net_selected_clip_norm": 10.0,
    }




@pytest.mark.L0
@pytest.mark.CPU
def test_public_metrics_name_preserves_internal_ledger_alias() -> None:
    public_metrics = getattr(ledger_module, "DMD2Metrics", None)

    assert public_metrics is not None
    assert issubclass(public_metrics, ledger_module.DMD2ParityLedger)
    assert ledger_module.__all__ == ("DMD2Metrics", "DMD2ParityLedger", "PARITY_KEYS")
