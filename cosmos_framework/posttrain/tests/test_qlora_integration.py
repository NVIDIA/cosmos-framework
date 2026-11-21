# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks for comparison/measurement tools; these do not qualify NF4 kernels."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.tests.qlora_integration import (
    _assert_close_tree,
    _interruption_state,
    _load_flat_dcp,
    _portable_trainer_controls,
    _probe_rows,
    _restore_probe_execution,
    inspect_checkpoint,
    interrupt_native,
    validate_checkpoint_controls,
)
from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase


def test_checkpoint_inspection_reads_real_dcp_controls_without_constructing_base(tmp_path):
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _save, _trainer

    trainer = _trainer()
    trainer.configure_validation({"enabled": False})
    trainer.train([None])
    path = _save(trainer, tmp_path)
    result = inspect_checkpoint(str(path), expected_step=1)
    assert result["healthy_completed_boundary"]
    assert len(result["parameters"]) == 2  # No fixed rank16 or full-model tensor assumption.
    assert result["optimizer_step"] == 1
    assert set(result["control_file_sha256"]) == {"trainer", "data", "model_metadata"}
    with pytest.raises(ValueError, match="requested boundary"):
        inspect_checkpoint(str(path), expected_step=2)
    state = torch.load(path / "trainer.pt", weights_only=False)
    state["accum"] = 1
    torch.save(state, path / "trainer.pt")
    with pytest.raises(ValueError, match="healthy completed"):
        inspect_checkpoint(str(path), expected_step=1)


def _native_action_checkpoint(tmp_path, *, warmup_steps=0, schedule_kind="constant"):
    from dataclasses import asdict

    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.config.build import build_sft_lr_scheduler
    from cosmos_framework.posttrain.config.rl import OptimizerConfig
    from cosmos_framework.posttrain.config.sft import SFTLRScheduleConfig
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests.test_family_backbone import TaskPolicy, batch, install
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state

    net = TaskPolicy()
    family = install(net)
    opt_config = OptimizerConfig(warmup_steps=warmup_steps)
    schedule = SFTLRScheduleConfig(kind=schedule_kind)
    family.codec.bind_training_identity(
        {
            "format": "cosmos-native-action-sft-v1",
            "seed": 17,
            "max_steps": 10,
            "optimizer": asdict(opt_config),
            "lr_schedule": asdict(schedule),
        }
    )
    optimizer = torch.optim.AdamW(family.codec.parameters.values(), lr=0.01)

    def objective(policy, unused):
        images, actions = batch()
        return (policy.net(images) - actions).square().mean(), {}

    trainer = LLMTrainer(
        SimpleNamespace(net=net),
        Objective(objective, reduction=Reduction.MEAN),
        optimizer,
        lr_scheduler=build_sft_lr_scheduler(optimizer, opt_config, schedule, 10),
        checkpoint_format="dcp",
    )
    trainer.train([None])
    path = tmp_path / "checkpoints/iter_000001"
    trainer.save(
        str(path),
        data_state=collect_rank_state(
            {
                "format": "action-sft-state-v1",
                "cursor": {"consumed": 1},
                "evaluation": [{"step": 0, "metrics": {"loss": 0.5}}, {"step": 1, "metrics": {"loss": 0.4}}],
            }
        ),
    )
    return path


@pytest.mark.parametrize("kind,warmup", [("constant", 0), ("constant", 2), ("cosine", 0)])
def test_native_action_inspection_reads_real_family_dcp_and_its_own_evaluation_state(tmp_path, kind, warmup):
    path = _native_action_checkpoint(tmp_path, schedule_kind=kind, warmup_steps=warmup)
    trainer = torch.load(path / "trainer.pt", weights_only=False)
    assert "validation" not in trainer
    assert ("lr_scheduler" in trainer) == (kind != "constant" or warmup > 0)
    result = inspect_checkpoint(str(path), expected_step=1)
    assert result["healthy_completed_boundary"]
    assert result["model_identity"]["format"] == "cosmos.family-backbone.v1"
    assert "head.weight" in result["parameters"]


@pytest.mark.parametrize("failure", ["missing", "class", "clock", "unexpected", "contract"])
def test_native_action_control_validation_binds_scheduler_presence_to_recorded_factory(tmp_path, failure):
    path = _native_action_checkpoint(tmp_path, warmup_steps=0 if failure == "unexpected" else 2)
    metadata = torch.load(path / "model_metadata.pt", weights_only=True)
    state = torch.load(path / "trainer.pt", weights_only=False)
    data = torch.load(path / "data.pt", weights_only=False)
    if failure == "missing":
        del state["lr_scheduler"]
    elif failure == "class":
        state["adapter_training"]["scheduler_class"] = None
    elif failure == "clock":
        state["lr_scheduler_clock"] = "optimizer_step"
    elif failure == "unexpected":
        state["lr_scheduler"] = {"last_epoch": 1}
    else:
        del metadata["identity"]["training_contract"]["optimizer"]["warmup_steps"]
    with pytest.raises(ValueError, match="scheduler"):
        validate_checkpoint_controls(metadata, state, data, expected_step=1)


@pytest.mark.parametrize("failure", ["missing", "future", "duplicate", "nonfinite", "cursor"])
def test_native_action_inspection_rejects_incomplete_or_invalid_task_evaluation(tmp_path, failure):
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state, state_for_current_rank

    path = _native_action_checkpoint(tmp_path)
    data = state_for_current_rank(torch.load(path / "data.pt", weights_only=False))
    if failure == "missing":
        del data["evaluation"]
    elif failure == "future":
        data["evaluation"][-1]["step"] = 2
    elif failure == "duplicate":
        data["evaluation"].append(copy.deepcopy(data["evaluation"][-1]))
    elif failure == "nonfinite":
        data["evaluation"][-1]["metrics"]["loss"] = float("nan")
    else:
        del data["cursor"]
    torch.save(collect_rank_state(data), path / "data.pt")
    with pytest.raises(ValueError, match="native action"):
        inspect_checkpoint(str(path), expected_step=1)


def test_existing_nonfamily_inspection_still_requires_its_resident_validation_controller(tmp_path):
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _save, _trainer

    trainer = _trainer()
    trainer.train([None])
    path = _save(trainer, tmp_path)
    with pytest.raises(ValueError, match="completed validation and publication"):
        inspect_checkpoint(str(path), expected_step=1)


def test_real_accumulation_interruption_proves_checkpoint_rejection_and_membership():
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer

    trainer = _trainer(grad_accum=2)
    checks = []
    trainer._net()._cosmos_validate_frozen_base = lambda: checks.append(True)
    for _ in range(4):
        trainer.train([None])
    boundary = _interruption_state(trainer, "boundary", 2)
    assert boundary["accumulation_step"] == 0
    assert not boundary["partial_checkpoint_rejected"]
    trainer.train([None])
    partial = _interruption_state(trainer, "accumulation", 2)
    assert partial["accumulation_step"] == 1 and partial["attempt_id"] == 3
    assert partial["partial_checkpoint_rejected"] and len(checks) == 2
    trainer.opt.param_groups[0]["params"].append(trainer._net().base.weight)
    with pytest.raises(ValueError, match="membership"):
        _interruption_state(trainer, "accumulation", 2)


def test_interruption_cannot_claim_frozen_proof_without_validator():
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer

    trainer = _trainer()
    trainer.train([None])
    with pytest.raises(ValueError, match="frozen-state validator"):
        _interruption_state(trainer, "boundary", 1)


@pytest.mark.parametrize("mode,expected_calls", [("boundary", 4), ("accumulation", 5)])
def test_interruption_wrapper_delegates_native_main_and_restores_hooks(tmp_path, monkeypatch, mode, expected_calls):
    from cosmos_framework.posttrain.entrypoints import sft
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests import qlora_integration as integration
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer
    from cosmos_framework.posttrain.utils import entry, qlora_metrics
    from cosmos_framework.utils import config

    class InjectedExit(Exception):
        pass

    trainer = _trainer(grad_accum=2)
    trainer._net()._cosmos_validate_frozen_base = lambda: None
    original_train, original_phase = LLMTrainer.train, qlora_metrics.record_qlora_phase
    calls = []

    def native_main():
        for _ in range(8):
            calls.append(True)
            trainer.train([None])
            if trainer._accum == 0:
                qlora_metrics.record_qlora_phase(f"validation_checkpoint_boundary_{trainer._iter}", reset_peak=True)

    def exit_process(code):
        assert code == integration.INTERRUPTION_EXIT_CODE
        raise InjectedExit

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    monkeypatch.setattr(config, "load_config", lambda *_: None)
    monkeypatch.setattr(entry, "out_dir", lambda _: str(tmp_path / "native"))
    monkeypatch.setattr(sft, "main", native_main)
    monkeypatch.setattr(integration, "inspect_checkpoint", lambda path, **_: {"path": path, "optimizer_step": 2})
    monkeypatch.setattr(integration.os, "_exit", exit_process)
    args = SimpleNamespace(
        config="fixture",
        override=[],
        checkpoint_root=str(tmp_path / "native/checkpoints"),
        receipt=str(tmp_path / "interruption.json"),
        after_step=2,
        mode=mode,
    )
    with pytest.raises(InjectedExit):
        interrupt_native(args)
    record = json.loads((tmp_path / "interruption.json").read_text())
    assert record["mode"] == mode and record["live_state"]["optimizer_step"] == 2
    assert len(calls) == expected_calls
    assert LLMTrainer.train is original_train and qlora_metrics.record_qlora_phase is original_phase


def test_flat_checkpoint_comparison_preserves_tensor_and_control_state(tmp_path):
    import torch.distributed.checkpoint as dcp

    path = tmp_path / "optim"
    state = {"state": {"factor": {"exp_avg": torch.arange(4).float(), "step": torch.tensor(3.0)}}, "groups": [1, 2]}
    dcp.save(state, checkpoint_id=path)
    restored = _load_flat_dcp(path)
    assert restored["groups"] == [1, 2]
    assert torch.equal(restored["state.factor.exp_avg"], torch.arange(4).float())
    _assert_close_tree(restored, copy.deepcopy(restored), atol=0, rtol=0, path="optim")
    changed = copy.deepcopy(restored)
    changed["state.factor.exp_avg"][0] += 0.1
    with pytest.raises(AssertionError):
        _assert_close_tree(restored, changed, atol=0, rtol=0, path="optim")
    _assert_close_tree(restored, changed, atol=0.11, rtol=0, path="optim")
    changed["groups"] = [1, 3]
    with pytest.raises(AssertionError, match="scalar/control"):
        _assert_close_tree(restored, changed, atol=0.11, rtol=0, path="optim")


def test_resume_controls_compare_rng_exactly_and_only_exclude_storage_identity():
    original = {
        "optimizer_step": 3,
        "rng": {"numpy": np.arange(5), "torch": torch.tensor([1, 2], dtype=torch.uint8)},
        "validation": {
            "progress": {"pending_step": None, "pending_metrics": {}},
            "publication": {
                "step": 3,
                "archive_required": False,
                "id": "first",
                "destination": {"root": "/a"},
                "manifest": "first.json",
            },
        },
    }
    other = copy.deepcopy(original)
    other["validation"]["publication"].update(id="second", destination={"root": "/b"}, manifest="second.json")
    _assert_close_tree(
        _portable_trainer_controls(original), _portable_trainer_controls(other), atol=0, rtol=0, path="trainer"
    )
    other["rng"]["numpy"][0] = 7
    with pytest.raises(AssertionError, match="array state"):
        _assert_close_tree(
            _portable_trainer_controls(original), _portable_trainer_controls(other), atol=0, rtol=0, path="trainer"
        )
    other["validation"]["progress"]["pending_step"] = 3
    with pytest.raises(ValueError, match="completed validation"):
        _portable_trainer_controls(other)
    assert original["validation"]["publication"]["id"] == "first"


def test_probe_rows_use_actual_video_and_existing_image_without_transcoding(tmp_path):
    from cosmos_framework.posttrain.data.datasets.sft import row_identity

    video, image = tmp_path / "clip.mp4", tmp_path / "frame.png"
    video.write_bytes(b"fixture file identity, decoding tested on GPU integration")
    image.write_bytes(b"fixed image bytes")
    row = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video", "video": "clip.mp4"}, {"type": "text", "text": "Describe the action."}],
            },
            {"role": "assistant", "content": "A person walks."},
        ]
    }
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(row) + "\n")
    rows = _probe_rows(str(path), 0, str(image), tmp_path)
    assert set(rows) == {"text", "image", "video"}
    assert rows["video"]["messages"][0]["content"][0]["video"] == str(video)
    copied = tmp_path / "copy.png"
    copied.write_bytes(image.read_bytes())
    second = _probe_rows(str(path), 0, str(copied), tmp_path)
    assert row_identity(rows["image"]) == row_identity(second["image"])
    with pytest.raises(ValueError, match="actual local video"):
        _probe_rows(str(path), 1, str(image), tmp_path)


def test_memory_telemetry_is_opt_in_and_does_not_initialize_cuda(tmp_path, monkeypatch):
    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    assert record_qlora_phase("disabled", reset_peak=True) is None
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("COSMOS_QLORA_METRICS_FILE", str(path))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    record = record_qlora_phase("cpu_loading", reset_peak=True)
    assert record["host_status"]["VmRSS"] > 0
    assert record["host_peak_rss_bytes"] > 0
    assert "cuda" not in record
    assert json.loads(path.read_text()) == record


def test_probe_reconstructs_execution_options_and_rejects_explicit_conflicts():
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_lora import build_config
    from cosmos_framework.posttrain.model.quantization.loader import resolved_execution_policy

    reference = build_config()
    reference.model.model_config.adapter_compute_dtype = "bfloat16"
    reference.model.model_config.sft_logits_mode = "supervised"
    identity = {
        "format": "cosmos.adapter-model.v2",
        "execution_policy": resolved_execution_policy(reference.model.model_config),
    }
    restored = build_config()
    _restore_probe_execution(restored, identity, [])
    assert restored.model.model_config.adapter_compute_dtype == "bfloat16"
    assert restored.model.model_config.sft_logits_mode == "supervised"
    with pytest.raises(ValueError, match="explicit adapter reload override"):
        _restore_probe_execution(build_config(), identity, ["model.model_config.adapter_compute_dtype=float32"])


@pytest.mark.parametrize("selected", [False, True])
def test_controlled_probe_observes_real_objective_full_and_selected_forward(selected):
    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker
    from cosmos_framework.posttrain.tests.qlora_integration import _evaluate_row
    from cosmos_framework.posttrain.tests.test_sft_selected_logits import _policies

    full, compact = _policies()
    policy = compact if selected else full
    packer = MemoryPacker()
    sample = packer.add([1, 2, 3, 4], [-100, -100, 3, 4])
    row = {"messages": sample.transitions[0].targets.messages}
    report, logits = _evaluate_row(policy, packer, make_objective(packer), row)
    assert report["supervised_tokens"] == 2
    assert report["logit_positions"] == [1, 2]
    assert logits.shape == (2, 16)
    assert torch.isfinite(logits).all()
    assert not policy.net._forward_hooks
    assert not policy.net.model._forward_pre_hooks
    assert not policy.net.lm_head._forward_hooks
