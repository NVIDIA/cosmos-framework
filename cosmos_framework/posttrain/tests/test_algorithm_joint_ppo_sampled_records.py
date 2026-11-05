# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model records own raw inputs and preserve complete invocation aliases."""

import pytest
import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY, JointPPOStats
from cosmos_framework.posttrain.model.gr00t import ppo as gr00t
from cosmos_framework.posttrain.model.pi05 import records as pi05
from cosmos_framework.posttrain.schema import SampleSource, Status

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(params=["gr00t", "pi05"])
def invocation(request):
    indices = torch.zeros(8, dtype=torch.long)
    if request.param == "gr00t":
        api, key, field = gr00t, gr00t.CONDITION_KEY, "condition"
        mask = torch.ones(8, 3, dtype=torch.bool)
        payload = gr00t.Gr00tPPOCondition(torch.zeros(8, 3, 4), mask, mask, torch.zeros(8, 1, 132), indices)
        trace = gr00t.Gr00tPPOTrace(torch.zeros(8, 40, 132), torch.ones(8, 40, 132), indices, torch.zeros(8))
    else:
        api, key, field = pi05, pi05.INPUTS_KEY, "inputs"
        payload = pi05.Pi05ReplayInputs(
            tuple(torch.zeros(8, 224, 224, 3, dtype=torch.uint8) for _ in range(2)),
            torch.tensor([[True, True, False]]).expand(8, -1),
            torch.zeros(8, 200, dtype=torch.long),
            torch.ones(8, 200, dtype=torch.bool),
            torch.zeros(8, 32, dtype=torch.float64),
        )
        trace = pi05.Pi05PPOTrace(torch.zeros(8, 10, 32), torch.ones(8, 10, 32), indices)
    observations = [{"image": torch.arange(12).reshape(2, 2, 3), "state": {"values": [1.5]}}] * 8
    results = [
        dict(
            action=torch.full((trace.score_horizon, 7), float(row)),
            sample_index=row,
            old_logprob=-row - 0.125,
            old_value=row / 8,
            trace=trace,
            **{field: payload},
        )
        for row in range(8)
    ]
    sources = [
        dict(
            sample_id=f"sample-{row}",
            episode_id=f"episode-{row}",
            fragment_id=f"fragment-{row}",
            session=f"session-{row}",
            step_idx=row + 10,
            task_id=row,
        )
        for row in range(8)
    ]
    return api, key, payload, trace, observations, results, sources


def test_sampled_records_preserve_shared_invocation_and_own_raw_inputs(invocation):
    api, key, payload, trace, observations, results, sources = invocation
    records = api.sampled_decisions(observations, results, sources, seed=37, version=4)
    assert len(records) == 8
    for row, (record, result, source) in enumerate(zip(records, results, sources, strict=True)):
        assert len(record.transitions) == 1 and record.status == Status.COMPLETED
        step = record.transitions[0]
        assert record.id == record.group_id == step.obs.id == source["sample_id"]
        assert (record.session, step.step_idx) == (source["session"], source["step_idx"])
        assert step.source == SampleSource(source["sample_id"], source["episode_id"], source["fragment_id"])
        assert (step.done, step.next_obs, step.feedback) == (False, None, None)
        assert step.info == source and step.info is not source and step.action is result["action"]
        assert step.algo_extra == {STATS_KEY: JointPPOStats(result["old_logprob"], result["old_value"])}
        assert step.policy_info[api.TRACE_KEY] is trace and step.obs.representations[key] is payload
        assert step.policy_info == {api.TRACE_KEY: trace, "weight_version": 4, "sampling_seed": 37, "sample_index": row}
        assert (
            set(step.obs.representations) == {"raw", key} and step.obs.representations["raw"] is not observations[row]
        )
        torch.testing.assert_close(step.obs.representations["raw"]["image"], observations[row]["image"], rtol=0, atol=0)
    observations[0]["image"].fill_(99)
    observations[0]["state"]["values"][0] = 99
    sources[0]["task_id"] = 99
    for record in records:
        raw = record.transitions[0].obs.representations["raw"]
        assert raw["image"][0, 0, 0].item() == 0 and raw["state"]["values"] == [1.5]
    assert records[0].transitions[0].info["task_id"] == 0
    records[0].transitions[0].obs.representations["raw"]["image"].fill_(77)
    assert records[1].transitions[0].obs.representations["raw"]["image"][0, 0, 0].item() == 0


def test_sampled_records_reject_mismatched_invocation_lengths(invocation):
    api, _, _, _, observations, results, sources = invocation
    with pytest.raises(ValueError, match="zip"):
        api.sampled_decisions(observations, results[:-1], sources, seed=37, version=4)
