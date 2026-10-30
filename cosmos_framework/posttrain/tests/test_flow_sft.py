# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Physical demonstrations use the real SFT loop, cursor and validation boundary."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.flow_bc import make_objective
from cosmos_framework.posttrain.config.sft import SFTConfig, SFTDatasetConfig, SFTValidationConfig
from cosmos_framework.posttrain.control.validation.lifecycle import ValidationTrigger
from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset, flow_source_overlap
from cosmos_framework.posttrain.entrypoints.sft import evaluate, loop, train_step
from cosmos_framework.posttrain.model.protocol import FlowMatchingLossOutput
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.env.flow_sft import build_flow_sft_env
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState


def dataset(path, episode, count):
    path.mkdir(parents=True)
    rows = [
        {
            "id": f"{episode}-{i}",
            "episode_id": episode,
            "observation": {"value": i + 1},
            "actions": [[float(i)], [float(i + 1)]],
            "media": {},
        }
        for i in range(count)
    ]
    file = path / "data.jsonl"
    file.write_text("\n".join(json.dumps(r) for r in rows))
    return SFTDatasetConfig(path=str(file), shuffle=False)


def pack(trajectories):
    rows = [t for trajectory in trajectories for t in trajectory.transitions]
    return {
        "x": torch.tensor([t.obs.representations["flow_sample"]["observation"]["value"] for t in rows])
        .float()
        .reshape(-1, 1),
        "y": torch.tensor([t.action for t in rows]),
        "mask": torch.tensor([t.algo_extra.get("validation_weight", 1) for t in rows])[:, None, None].expand(-1, 2, 1),
    }


class Policy:
    def __init__(self):
        torch.manual_seed(17)
        self.net = torch.nn.Linear(1, 2)

    def flow_matching_loss(self, inputs):
        pred = self.net(inputs["x"]).unsqueeze(-1)
        mask = inputs["mask"]
        return FlowMatchingLossOutput(
            ((pred - inputs["y"]).square() * mask).sum() / mask.sum().clamp_min(1), mask, pred
        )


def trainer():
    model = Policy()
    result = LLMTrainer(
        model,
        make_objective(pack, training_prefix=2, action_dim=1, sft_metrics=True),
        torch.optim.AdamW(model.net.parameters(), lr=0.01),
        grad_accum=1,
        save_async=False,
        validation_state=ValidationState(),
        objective_evaluator=evaluate_sft_batch,
    )
    result.configure_validation({"enabled": False})
    return result


def test_flow_validation_padding_counts_and_exact_cursor_continuation(tmp_path):
    config = SFTConfig(
        task="flow_bc",
        flow_training_prefix=2,
        flow_action_dim=1,
        data=dataset(tmp_path / "train", "train-episode", 4),
        validation=SFTValidationConfig(enabled=True, batch_size=2, data=dataset(tmp_path / "val", "val-episode", 3)),
    )
    data = build_flow_sft_env(config.data, 7)
    val = build_flow_sft_env(config.validation.data, 8, training=False)
    flow_source_overlap(data, val)
    policy = trainer()
    result = evaluate({"validation": val, "policy": policy}, config, 0, frozenset({ValidationTrigger.BEFORE_TRAINING}))
    assert result.metrics["val/valid_action_count"] == 6
    assert result.sample_count == 3
    assert "val/valid_token_count" not in result.metrics
    for _ in range(3):
        report = train_step(data, policy, config)
        assert report["train/valid_action_count"] == 2
    checkpoint = str(tmp_path / "checkpoint")
    policy.save(checkpoint, data_state=data.state_dict())
    policy.finalize_checkpoints()
    expected_report = train_step(data, policy, config)
    expected = copy.deepcopy(policy.model.net.state_dict())
    cursor = data.state_dict()
    loaded = trainer()
    loaded.load(checkpoint)
    data.load_state_dict(loaded.checkpoint_manager.load_component(checkpoint, "data"))
    actual = train_step(data, loaded, config)
    assert actual["train/loss_avg"] == expected_report["train/loss_avg"]
    assert data.state_dict() == cursor
    for name, value in loaded.model.net.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_flow_runs_the_existing_sft_update_loop(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.tests.test_entrypoints_sft import _quiet_loop

    _quiet_loop(monkeypatch)
    rl = SFTConfig(
        task="flow_bc",
        flow_training_prefix=2,
        flow_action_dim=1,
        max_steps=3,
        data=dataset(tmp_path / "data", "episode", 4),
    )
    config = SimpleNamespace(
        rl=rl, job=SimpleNamespace(wandb_mode="disabled", project="flow", group="unit", name="unit")
    )
    policy = trainer()
    units = {
        "data": roles.env("data", body=build_flow_sft_env(rl.data, 7)),
        "policy": roles.trainer("policy", body=policy),
    }
    assert loop(units, config, str(tmp_path / "run"))["final_step"] == 3


def test_flow_dataset_refuses_split_overlap_and_changed_media_identity(tmp_path):
    train = dataset(tmp_path / "a", "same-episode", 2)
    val = dataset(tmp_path / "b", "same-episode", 2)
    with pytest.raises(ValueError, match="overlap"):
        flow_source_overlap(build_flow_sft_env(train, 7), build_flow_sft_env(val, 7, training=False))
    path = tmp_path / "a/data.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    media = tmp_path / "a/image.bin"
    media.write_bytes(b"first-content")
    rows[0]["media"] = {"camera": "image.bin"}
    path.write_text("\n".join(json.dumps(r) for r in rows))
    before = FlowSFTDataset(str(path)).cursor_fingerprint()
    media.write_bytes(b"other-content")
    assert FlowSFTDataset(str(path)).cursor_fingerprint() != before
