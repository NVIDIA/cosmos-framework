# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU DCP contract tests; CUDA kernel/memory qualification is a separate suite."""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.checkpoint.adapter import (
    load_adapter_weights,
    publish_adapter_artifact,
    read_adapter_metadata,
)
from cosmos_framework.posttrain.checkpoint.artifact import read_local_artifact_manifest
from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.utils.rank_state import collect_rank_state


class FrozenBase(nn.Linear):
    def state_dict(self, *args, **kwargs):
        raise AssertionError("full frozen-base state extraction is forbidden")


class TinyAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = FrozenBase(4, 4, bias=False).requires_grad_(False)
        self.base.register_buffer("packed_state", torch.arange(8, dtype=torch.uint8))
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self._cosmos_adapter_parameters = {
            "language.q_proj.lora_A.weight": self.lora_A.weight,
            "language.q_proj.lora_B.weight": self.lora_B.weight,
        }
        self._cosmos_adapter_checkpoint_codec = build_adapter_checkpoint_codec(
            self,
            identity={"base": {"repo": "test/base", "revision": "a" * 40}, "quantizer": "test-only"},
        )
        self._cosmos_adapter_checkpoint_codec.bind_training_identity({"schedule": "linear", "seed": 17})
        self._cosmos_adapter_checkpoint_codec.bind_processor_identity({"template": "controlled", "media": "native"})

    def _apply(self, *args, **kwargs):
        raise AssertionError("blanket movement/conversion is forbidden")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x))


def _objective(model, batch):
    x = torch.randn(2, 4)
    loss = model(x).square().mean()
    return loss, {"loss": float(loss.detach())}


def _trainer(seed: int = 17, **kwargs) -> LLMTrainer:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyAdapterModel()
    optimizer = torch.optim.AdamW([model.lora_A.weight, model.lora_B.weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)
    return LLMTrainer(
        model,
        Objective(_objective, reduction=Reduction.MEAN),
        optimizer,
        checkpoint_format="dcp",
        lr_scheduler=scheduler,
        **kwargs,
    )


def _save(trainer: LLMTrainer, tmp_path: Path, *, step: int | None = None) -> Path:
    path = tmp_path / "checkpoints" / f"iter_{trainer._iter if step is None else step:06d}"
    trainer.save(str(path), data_state=collect_rank_state({"cursor": trainer._iter}))
    return path


def _parameters(trainer: LLMTrainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer._adapter_checkpoint_codec().parameters.items()}


def test_adapter_dcp_resume_matches_uninterrupted_run_and_preserves_base(tmp_path: Path) -> None:
    source = _trainer()
    base = source._net().base.weight.detach().clone()
    packed = source._net().base.packed_state.clone()
    source.train([None])
    # Zero-initialized B makes the initial A gradient zero; B must nevertheless update.
    assert torch.count_nonzero(source._net().lora_B.weight) > 0
    source.train([None])
    checkpoint = _save(source, tmp_path)
    saved_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    source.train([None])
    expected = _parameters(source)
    expected_optimizer = copy.deepcopy(source.opt.state_dict())
    expected_scheduler = copy.deepcopy(source.lr_scheduler.state_dict())

    restored = _trainer()
    # Different initial scheduler LR is expected at resume; it is not a static identity field.
    assert restored.opt.param_groups[0]["lr"] != source.opt.param_groups[0]["lr"]
    assert restored.load(str(checkpoint)) == 2
    assert (random.random(), float(np.random.rand())) == saved_rng[:2]
    torch.testing.assert_close(torch.rand(3), saved_rng[2], rtol=0, atol=0)
    restored.train([None])
    for name, parameter in _parameters(restored).items():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    for key, state in restored.opt.state_dict()["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, expected_optimizer["state"][key][name], rtol=0, atol=0)
    assert restored.lr_scheduler.state_dict() == expected_scheduler
    torch.testing.assert_close(source._net().base.weight, base, rtol=0, atol=0)
    torch.testing.assert_close(source._net().base.packed_state, packed, rtol=0, atol=0)
    metadata = read_adapter_metadata(str(checkpoint), require_resume=True)
    assert set(metadata["parameters"]) == set(expected)
    assert source.onload() == 0.0


def test_model_only_artifact_carries_metadata_and_reloads(tmp_path: Path) -> None:
    trainer = _trainer()
    trainer.train([None])
    hint = tmp_path / "checkpoints" / "iter_000001"
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        trainer.save(str(hint), components=["model"])
    artifact = tmp_path / "checkpoint_artifacts" / "iter_000001"
    manifest = read_local_artifact_manifest(str(artifact))
    assert set(manifest["components"]) == {"model", "model_metadata"}
    assert manifest["checkpoint_format"] == "consolidated"
    assert all("sha256" in descriptor for descriptor in manifest["components"].values())
    assert not (tmp_path / "checkpoints" / "latest_checkpoint.txt").exists()
    restored = _trainer()
    load_adapter_weights(str(artifact), restored._adapter_checkpoint_codec(), manager=restored.checkpoint_manager)
    x = torch.ones(2, 4)
    torch.testing.assert_close(restored._net()(x), trainer._net()(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot resume"):
        restored.load(str(artifact))


def test_pristine_save_does_not_initialize_optimizer_and_restores_pristine(tmp_path: Path) -> None:
    trainer = _trainer()
    before = _parameters(trainer)
    checkpoint = _save(trainer, tmp_path)
    assert trainer.optimizer_is_pristine()
    assert all(torch.equal(before[name], value) for name, value in _parameters(trainer).items())
    restored = _trainer()
    restored.train([None])
    assert restored.load(str(checkpoint)) == 0
    assert restored.optimizer_is_pristine()


@pytest.mark.parametrize(
    "defect", ["missing_metadata", "missing_data", "missing_optimizer", "revision", "shape", "dtype", "extra"]
)
def test_adapter_reload_rejects_incomplete_or_incompatible_checkpoint(tmp_path: Path, defect: str) -> None:
    trainer = _trainer()
    trainer.train([None])
    checkpoint = _save(trainer, tmp_path)
    if defect.startswith("missing_"):
        target = {
            "missing_metadata": "model_metadata.pt",
            "missing_data": "data.pt",
            "missing_optimizer": "optim/.metadata",
        }[defect]
        (checkpoint / target).unlink()
    elif defect == "revision":
        trainer._adapter_checkpoint_codec().identity["base"]["revision"] = "b" * 40
    else:
        import torch.distributed.checkpoint as dcp

        state = {name: parameter.clone() for name, parameter in _parameters(trainer).items()}
        name = next(iter(state))
        if defect == "shape":
            state[name] = state[name][:-1]
        elif defect == "dtype":
            state[name] = state[name].to(torch.float16)
        else:
            state["frozen_base.weight"] = torch.ones(4, 4)
        dcp.save(state, checkpoint_id=str(checkpoint / "model"))
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        trainer.load(str(checkpoint))


def test_codec_rejects_optimizer_reordering_and_changed_static_options(tmp_path: Path) -> None:
    trainer = _trainer()
    checkpoint = _save(trainer, tmp_path)
    trainer.opt.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="optimizer identity"):
        trainer.load(str(checkpoint))
    trainer.opt.param_groups[0]["params"].reverse()
    trainer.opt.param_groups[0]["weight_decay"] = 0.5
    with pytest.raises(ValueError, match="optimizer identity"):
        trainer.load(str(checkpoint))


def test_codec_membership_references_and_unsupported_operations() -> None:
    trainer = _trainer()
    codec = trainer._adapter_checkpoint_codec()
    assert dict(codec.view.named_parameters()) == codec.parameters
    with pytest.raises(ValueError, match="offload"):
        trainer.offload()
    with pytest.raises(ValueError, match="weight sync"):
        trainer.prepare_weight_sync()
    with pytest.raises(ValueError, match="publication"):
        trainer.push_weights()
    trainer.opt.param_groups[0]["params"].append(trainer._net().base.weight)
    with pytest.raises(ValueError, match="membership"):
        codec.assert_live(optimizer=trainer.opt)
    trainer.opt.param_groups[0]["params"].pop()
    trainer._net().base.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable model parameters"):
        codec.assert_live()


def test_checkpoint_rejects_open_accumulation_and_missing_cursor(tmp_path: Path) -> None:
    trainer = _trainer(grad_accum=2)
    trainer.train([None])
    with pytest.raises(RuntimeError, match="completed update boundary"):
        _save(trainer, tmp_path)
    trainer.train([None])
    with pytest.raises(ValueError, match="data cursor"):
        trainer.save(str(tmp_path / "checkpoints" / "iter_000001"))


def test_metadata_preflight_compares_explicit_subset_before_model_build(tmp_path: Path) -> None:
    checkpoint = _save(_trainer(), tmp_path)
    metadata = read_adapter_metadata(str(checkpoint), expected_identity={"base": {"revision": "a" * 40}})
    assert metadata["identity"]["base"]["repo"] == "test/base"
    with pytest.raises(ValueError, match="revision differs"):
        read_adapter_metadata(str(checkpoint), expected_identity={"base": {"revision": "b" * 40}})


def test_fresh_torchrun_process_reload_preserves_controlled_logits(tmp_path: Path) -> None:
    trainer = _trainer()
    trainer.train([None])
    trainer.train([None])
    checkpoint = _save(trainer, tmp_path)
    expected = trainer._net()(torch.ones(2, 4)).detach()
    script = tmp_path / "reload.py"
    script.write_text(
        "import sys, torch\n"
        "import torch.distributed as dist\n"
        "from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer\n"
        "dist.init_process_group('gloo')\n"
        "try:\n"
        "    trainer = _trainer()\n"
        "    assert trainer.load(sys.argv[1]) == 2\n"
        "    torch.save(trainer._net()(torch.ones(2, 4)).detach(), sys.argv[2])\n"
        "finally:\n"
        "    dist.destroy_process_group()\n"
    )
    output = tmp_path / "logits.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            str(script),
            str(checkpoint),
            str(output),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
        capture_output=True,
        text=True,
        timeout=60,
    )
    torch.testing.assert_close(torch.load(output, weights_only=True), expected, rtol=0, atol=0)


def test_adapter_control_tail_is_synchronous_even_when_cpu_snapshot_is_immutable(tmp_path: Path, monkeypatch) -> None:
    from cosmos_framework.posttrain.checkpoint import manager

    def unexpected_async(*args, **kwargs):
        raise AssertionError("adapter checkpoints must finish control writes synchronously")

    monkeypatch.setattr(manager, "start_background_write", unexpected_async)
    checkpoint = _save(_trainer(), tmp_path)
    assert (checkpoint / "model_metadata.pt").is_file()
    assert (checkpoint / "trainer.pt").is_file()


def test_metadata_rejects_topology_and_missing_rng(tmp_path: Path) -> None:
    trainer = _trainer()
    checkpoint = _save(trainer, tmp_path)
    trainer_state = torch.load(checkpoint / "trainer.pt", weights_only=False)
    trainer_state.pop("rng")
    torch.save(trainer_state, checkpoint / "trainer.pt")
    with pytest.raises(RuntimeError, match="incomplete"):
        trainer.load(str(checkpoint))
    metadata = torch.load(checkpoint / "model_metadata.pt", weights_only=True)
    metadata["topology"]["world_size"] = 2
    torch.save(metadata, checkpoint / "model_metadata.pt")
    with pytest.raises(ValueError, match="topology"):
        read_adapter_metadata(str(checkpoint))


def test_model_replacement_or_dtype_change_invalidates_codec() -> None:
    trainer = _trainer()
    trainer._net().lora_A.weight.data = trainer._net().lora_A.weight.data.to(torch.float16)
    with pytest.raises(RuntimeError, match="dtype"):
        trainer.onload()
    trainer = _trainer()
    trainer._net().lora_A.weight = nn.Parameter(trainer._net().lora_A.weight.detach().clone())
    with pytest.raises(RuntimeError, match="identity"):
        trainer.onload()


def test_frozen_state_proof_runs_at_checkpoint_boundaries_and_control_refresh_preserves_identity(
    tmp_path: Path,
) -> None:
    trainer = _trainer()
    calls = []
    trainer._net()._cosmos_validate_frozen_base = lambda: calls.append("validate")
    trainer.onload()
    assert not calls
    checkpoint = _save(trainer, tmp_path)
    assert calls == ["validate"]
    # Validation publication refreshes trainer.pt from this same method.
    trainer.checkpoint_manager.replace_component(str(checkpoint), "trainer", trainer._trainer_state())
    trainer.load(str(checkpoint))
    assert calls == ["validate", "validate"]


def test_publish_portable_artifact_from_dcp_needs_no_base_and_detects_changed_bytes(tmp_path: Path) -> None:
    trainer = _trainer()
    trainer.train([None])
    checkpoint = _save(trainer, tmp_path / "source")
    hint = tmp_path / "portable" / "checkpoints" / "iter_000001"
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        artifact = Path(publish_adapter_artifact(str(checkpoint), str(hint)))
    metadata = read_adapter_metadata(str(artifact))
    assert metadata["optimizer"] is None
    state = torch.load(artifact / "model.pt", weights_only=True)
    assert set(state["net"]) == set(trainer._adapter_checkpoint_codec().parameters)
    for name, parameter in _parameters(trainer).items():
        torch.testing.assert_close(state["net"][name], parameter, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot resume"):
        read_adapter_metadata(str(artifact), require_resume=True)
    state["net"][next(iter(state["net"]))].add_(1)
    torch.save(state, artifact / "model.pt")
    with pytest.raises(ValueError, match="SHA256"):
        read_adapter_metadata(str(artifact))


def test_weight_reload_does_not_require_training_contract_but_resume_does(tmp_path: Path) -> None:
    source = _trainer()
    source.train([None])
    checkpoint = _save(source, tmp_path)
    restored = _trainer()
    # A serving/evaluation builder has the complete numerical model identity, without training data
    # or scheduler policy. Rebuild the codec as that factory would, without binding a training contract.
    identity = dict(restored._adapter_checkpoint_codec().identity)
    identity.pop("training_contract")
    codec = build_adapter_checkpoint_codec(restored._net(), identity=identity)
    restored._net()._cosmos_adapter_checkpoint_codec = codec
    load_adapter_weights(str(checkpoint), codec, manager=restored.checkpoint_manager)
    for name, value in _parameters(source).items():
        torch.testing.assert_close(codec.parameters[name], value, rtol=0, atol=0)
    codec.bind_training_identity({"schedule": "different"})
    with pytest.raises(ValueError, match="identity differs"):
        restored.load(str(checkpoint))
    processor = codec.identity.pop("processor_contract")
    with pytest.raises(ValueError, match="identity differs"):
        load_adapter_weights(str(checkpoint), codec, manager=restored.checkpoint_manager)
    codec.bind_processor_identity(processor)
    codec.bind_processor_identity(processor)  # Idempotent binding is safe.
    with pytest.raises(ValueError, match="another processor"):
        codec.bind_processor_identity({"template": "changed"})
    # Model-critical identity never receives the inference exemption.
    codec.identity["base"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="identity differs"):
        load_adapter_weights(str(checkpoint), codec, manager=restored.checkpoint_manager)
