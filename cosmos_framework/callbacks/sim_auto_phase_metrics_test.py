# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.sim_auto_phase_metrics import PhaseMetrics

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.mark.parametrize("iteration,phase,key", [(1, "student", "vsd_loss"), (2, "critic", "loss_fake_score")])
def test_phase_uses_completed_step_and_preserves_native_loss(tmp_path, iteration, phase, key):
    output_path = tmp_path / "new" / "phase_metrics.jsonl"
    model = SimpleNamespace(
        get_phase=lambda step: "student" if step % 6 == 0 else "critic",
        _distillation_parity_grad_metrics={"grad_norm": torch.tensor(3.0)},
    )
    callback = PhaseMetrics(str(output_path))
    callback.on_training_step_end(model, {}, {key: torch.tensor(2.0)}, torch.tensor(2.0), iteration)
    record = json.loads(output_path.read_text())
    assert record == {"iteration": iteration, "phase": phase, key: 2.0, "grad_norm": 3.0}
