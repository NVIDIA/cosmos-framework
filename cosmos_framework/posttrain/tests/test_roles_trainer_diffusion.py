# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for `DiffusionTrainer` save/load + the host→device data-fetch.

Uses the toy flow model so the shared component checkpoint layout
model/optim/trainer .pt, ref.pt only for a moving reference) runs without a GPU.
Covers resume (net/opt/iter restored; a frozen ref is NOT saved and NOT touched by
load — setup owns rebuilding it from base), init-from-base copy_mode (ref ≡ base,
optimizer/iter NOT restored), the moving-ref save gate, and the data mover.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.posttrain.algorithm.dpo import make_objective
from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.callbacks.base import Callback
from cosmos_framework.posttrain.checkpoint import CheckpointManager, load_component, save_component_dcp
from cosmos_framework.posttrain.checkpoint.reference import materialized_reference_sha256
from cosmos_framework.posttrain.config.build import build_wfm_lr_scheduler
from cosmos_framework.posttrain.config.rl import WFMLRSchedulerConfig
from cosmos_framework.posttrain.control.validation import ValidationTrigger
from cosmos_framework.posttrain.model.protocol import EMAFlowMatchingPolicy
from cosmos_framework.posttrain.roles.env.preference import DatasetPreferenceEnv
from cosmos_framework.posttrain.roles.trainer.diffusion import DiffusionTrainer
from cosmos_framework.posttrain.roles.trainer.validation import preference as validation_module
from cosmos_framework.posttrain.roles.trainer.validation.preference import (
    ObjectiveValidationRunner,
    _ValidationRuntime,
    move_to_device,
)
from cosmos_framework.posttrain.roles.trainer.validation.runtime import reshard_fsdp_modules
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.tests.helpers.preference_env import SyntheticPreferenceEnv
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel


class _EMAToyFlowModel(ToyFlowModel):
    def __init__(
        self,
        dim: int,
        seed: int,
        *,
        policy_dtype: torch.dtype = torch.float32,
        ema_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self.net.to(dtype=policy_dtype)
        self.net_ref.to(dtype=policy_dtype)
        self.net_ema: torch.nn.Module = deepcopy(self.net).to(dtype=ema_dtype or policy_dtype)
        self.ema_iterations: list[int] = []

    def update_ema(self, iteration: int) -> None:
        self.ema_iterations.append(iteration)
        self.net_ema.load_state_dict(self.net.state_dict())


class _EMANetWithoutUpdater(ToyFlowModel):
    update_ema: None = None

    def __init__(self, dim: int, seed: int) -> None:
        super().__init__(dim=dim, seed=seed)
        self.net_ema: torch.nn.Module = deepcopy(self.net)


class _EmptyEMAToyFlowModel(ToyFlowModel):
    net_ema: torch.nn.Module | None

    def __init__(self, dim: int, seed: int) -> None:
        super().__init__(dim=dim, seed=seed)
        self.net_ema = None

    def update_ema(self, iteration: int) -> None:
        del iteration


@dataclass(frozen=True)
class _FrozenPayload:
    tensor: torch.Tensor


class _FakeFSDPModule(torch.nn.Module):
    """CPU stand-in for the FSDP2 lifecycle surface used by validation cleanup."""

    fail: bool
    reshard_calls: int

    def __init__(self, *, fail: bool = False, child: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.fail = fail
        self.reshard_calls = 0
        if child is not None:
            self.child = child

    def reshard(self) -> None:
        self.reshard_calls += 1
        if self.fail:
            raise RuntimeError("synthetic reshard failure")


class _HeldOutTensorPairs:
    def __init__(self, size: int, dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._pairs: list[tuple[str, dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
        for index in range(size):
            base = torch.randn(dim, generator=generator)  # [D]
            winner = base + 0.1 * torch.randn(dim, generator=generator)  # [D]
            loser = base + torch.randn(dim, generator=generator)  # [D]
            self._pairs.append((f"held-out-{index}", {"x": winner}, {"x": loser}))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[str, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prompt, winner, loser = self._pairs[index]
        return prompt, {"x": winner["x"].clone()}, {"x": loser["x"].clone()}  # [D], [D]

    def cursor_fingerprint(self) -> str:
        return "held-out-tensor-pairs-v1"


def _trainer(seed: int) -> DiffusionTrainer:
    model = ToyFlowModel(dim=8, seed=seed)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    return DiffusionTrainer(model, make_objective(beta=100.0), opt)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("update_ema_after_iter", [False, True])
def test_offload_projects_every_non_policy_state_descriptor(update_ema_after_iter: bool) -> None:
    model = _EMAToyFlowModel(dim=8, seed=0)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        update_ema_after_iter=update_ema_after_iter,
    )
    offload = Mock()
    offload.offload.return_value = 17.0
    cast(Any, trainer)._host_offload = offload

    assert trainer.offload() == 17.0
    snapshots = offload.offload.call_args.args[0]
    assert snapshots == [model.net_old, model.net_ref, model.net_ema]
    assert offload.offload.call_args.kwargs["optimizer"] is trainer.opt
    assert list(offload.offload.call_args.kwargs["grads_of"]) == list(trainer.model.net.parameters())


def _wfm_trainer(scheduler_config: WFMLRSchedulerConfig, *, seed: int = 0) -> DiffusionTrainer:
    model = ToyFlowModel(dim=8, seed=seed)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    scheduler = build_wfm_lr_scheduler(optimizer, scheduler_config)
    return DiffusionTrainer(model, make_objective(beta=100.0), optimizer, scheduler=scheduler)


def _trainer_from_base(
    base_checkpoint: Path,
    *,
    seed: int,
    checkpoint_format: str = "consolidated",
) -> DiffusionTrainer:
    """Build a trainer and anchor its policy/reference from one identified base."""
    model = ToyFlowModel(dim=8, seed=seed)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        checkpoint_format=checkpoint_format,
        defer_frozen_reference_identity=True,
    )
    trainer.load(str(base_checkpoint), copy_mode=True)
    return trainer


def _trainer_with_validation(
    seed: int,
    *,
    held_out_size: int = 5,
    validation_batch_size: int = 2,
    rank: int = 0,
    world_size: int = 1,
    callbacks: list[Callback] | None = None,
    upload: dict[str, str] | None = None,
) -> DiffusionTrainer:
    model = ToyFlowModel(dim=8, seed=seed)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    source = DatasetPreferenceEnv(
        _HeldOutTensorPairs(held_out_size, dim=8, seed=100),
        rank=rank,
        world_size=world_size,
        shuffle=False,
        cursor_resume="off",
        source_fingerprint="held-out-tensor-pairs-v1",
    )
    return DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        optimizer,
        validation_source=source,
        validation_batch_size=validation_batch_size,
        validation_seed=17,
        validation_before_training=True,
        validation_every_steps=2,
        validation_train_total_steps=10,
        callbacks=callbacks,
        upload=upload,
    )


def _sd_close(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.allclose(a[k], b[k]) for k in a)


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_reshard_deduplicates_roots_and_preserves_primary_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.fsdp as fsdp

    child = _FakeFSDPModule()
    root = _FakeFSDPModule(child=child)
    monkeypatch.setattr(fsdp, "FSDPModule", _FakeFSDPModule)

    reshard_fsdp_modules({"net": root, "net_ref": root, "net_ema": child})

    assert root.reshard_calls == 1
    assert child.reshard_calls == 1

    healthy = _FakeFSDPModule()
    failing = _FakeFSDPModule(fail=True, child=healthy)
    runtime = _ValidationRuntime(roots={"net": failing})

    with pytest.raises(ValueError, match="primary validation failure") as error_info:
        ObjectiveValidationRunner._restore_state(runtime, ValueError("primary validation failure"))

    assert failing.reshard_calls == 1
    assert healthy.reshard_calls == 1
    assert isinstance(error_info.value.__cause__, RuntimeError)
    assert "synthetic reshard failure" in str(error_info.value.__cause__)


class _CountingToyFlowModel(ToyFlowModel):
    def __init__(self, dim: int = 8, seed: int = 0) -> None:
        super().__init__(dim=dim, seed=seed)
        self.refresh_calls = 0

    def refresh_ref(self) -> None:
        super().refresh_ref()
        self.refresh_calls += 1


class _ReferenceIdentityModule(torch.nn.Module):
    """Small module whose exact global state is comparable before/after DTensor sharding."""

    def __init__(self, weight: torch.Tensor) -> None:  # weight: [4,4]
        super().__init__()
        self.weight = torch.nn.Parameter(weight)  # [4,4]
        self.register_buffer("scale", torch.arange(4, dtype=torch.float32))  # [4]


def _test_reference_semantics_identity() -> dict[str, int | str]:
    """Stable semantics declaration for the minimal sharded-reference policy fixture."""
    return {"adapter": "sharded-reference-test", "version": 1}


class _ScalarSemanticReference(torch.nn.Linear):
    """Reference whose forward depends on one non-tensor construction value."""

    def __init__(self, dim: int, scale: float) -> None:
        super().__init__(dim, dim)
        self.scale = float(scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs) * self.scale  # [B,D]


class _ScalarSemanticToyFlowModel(ToyFlowModel):
    def __init__(self, dim: int, seed: int, reference_scale: float) -> None:
        super().__init__(dim=dim, seed=seed)
        reference = _ScalarSemanticReference(dim, reference_scale)
        reference.load_state_dict(self.net_ref.state_dict())
        reference.requires_grad_(False)
        reference.eval()
        self.net_ref = reference
        self._reference_scale = float(reference_scale)

    def reference_semantics_identity(self) -> dict[str, float | int | str]:
        identity: dict[str, float | int | str] = dict(super().reference_semantics_identity())
        identity["reference_scale"] = self._reference_scale
        return identity


def _sharded_reference_identity_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        mesh = dist.device_mesh.init_device_mesh("cpu", (world_size,))
        full_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)  # [4,4]
        sharded_weight = distribute_tensor(full_weight, mesh, [Shard(0)])  # [4,4] global
        sharded = _ReferenceIdentityModule(sharded_weight)
        sharded_digest = materialized_reference_sha256(sharded)

        replicated_weight = full_weight.clone()  # [4,4]
        replicated = _ReferenceIdentityModule(replicated_weight)
        replicated_digest = materialized_reference_sha256(replicated)
        assert sharded_digest == replicated_digest

        policy = SimpleNamespace(
            net=torch.nn.Linear(1, 1),
            net_ref=sharded,
            reference_semantics_identity=_test_reference_semantics_identity,
        )
        trainer = DiffusionTrainer(
            policy,
            Objective(lambda *_args, **_kwargs: None, Reduction.MEAN),
            torch.optim.SGD(policy.net.parameters(), lr=0.0),
        )
        assert trainer.reference_identity.current is not None
        assert trainer.reference_identity.current["source"]["sha256"] == replicated_digest
        gathered: list[str | None] = [None] * world_size
        dist.all_gather_object(gathered, sharded_digest)
        assert len(set(gathered)) == 1
    finally:
        dist.destroy_process_group()


def _reference_identity_convergence_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    failure_mode: str,
) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        torch.manual_seed(41)
        net = torch.nn.Linear(2, 2, bias=False)
        net_ref = torch.nn.Linear(2, 2, bias=False)
        if failure_mode == "tensor-mismatch" and rank == 1:
            with torch.no_grad():
                net_ref.weight.add_(1.0)  # [2,2]

        def _semantics() -> dict[str, bool | float | int | str]:
            if failure_mode == "capability-error" and rank == 1:
                raise ValueError("rank-local semantics failure")
            if failure_mode == "semantics-type-mismatch":
                scale: bool | float = True if rank == 0 else 1.0
            else:
                scale = 2.0 if failure_mode == "semantics-mismatch" and rank == 1 else 1.0
            return {"adapter": "reference-convergence-test", "scale": scale, "version": 1}

        policy = SimpleNamespace(
            net=net,
            net_ref=net_ref,
            reference_semantics_identity=_semantics,
        )
        error_text: str | None = None
        try:
            DiffusionTrainer(
                policy,
                Objective(lambda *_args, **_kwargs: None, Reduction.MEAN),
                torch.optim.SGD(net.parameters(), lr=0.0),
            )
        except RuntimeError as error:
            error_text = str(error)
        errors: list[str | None] = [None] * world_size
        dist.all_gather_object(errors, error_text)
        assert all(error is not None for error in errors)
        expected = "reference semantics identity" if failure_mode != "tensor-mismatch" else "frozen reference identity"
        assert all(expected in str(error) for error in errors)
    finally:
        dist.destroy_process_group()


def _validation_publication_io_failure_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    checkpoint_path: str,
    failure_stage: str,
) -> None:
    """Inject one rank-0 publication IO failure and prove every rank exits the same stage."""
    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint import load_component
    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager
    from cosmos_framework.posttrain.utils import object_store

    init_gloo_process_group(rank, world_size, rendezvous_path)
    original_mkstemp = checkpoint_manager.tempfile.mkstemp
    original_remove = checkpoint_manager.os.remove
    original_destination_identity = object_store.upload_destination_identity
    original_upload = checkpoint_manager.archive_checkpoint
    upload_calls = 0
    try:
        trainer = _trainer_with_validation(
            seed=5,
            upload={"bucket": "archive", "prefix": "posttrain/checkpoints"},
        )
        trainer.mark_validation_pending(0, frozenset({ValidationTrigger.FINAL}))
        trainer.save(checkpoint_path, publish=False)
        trainer.complete_validation(0)
        object_store.upload_destination_identity = lambda upload, iter_name: {
            "kind": "object_store",
            "endpoint_authority": "storage.example.test",
            "bucket": str(upload["bucket"]),
            "prefix": f"{str(upload['prefix']).rstrip('/')}/{iter_name}",
        }

        def _record_upload(
            iter_dir: str,
            upload: dict[str, str],
            *,
            overrides: dict[str, str] | None = None,
            commit: dict[str, Any] | None = None,
        ) -> None:
            nonlocal upload_calls
            del upload
            assert overrides is not None and commit is not None
            upload_calls += 1
            (Path(iter_dir) / ".uploaded").write_text("archive\n", encoding="utf-8")

        checkpoint_manager.archive_checkpoint = _record_upload
        if failure_stage == "preparation" and rank == 0:

            def _fail_mkstemp(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
                raise OSError("injected completed-trainer preparation failure")

            checkpoint_manager.tempfile.mkstemp = _fail_mkstemp
        elif failure_stage == "cleanup" and rank == 0:

            def _fail_completed_remove(path: str) -> None:
                if ".completed-trainer-" in os.path.basename(path):
                    raise PermissionError("injected completed-trainer cleanup failure")
                original_remove(path)

            checkpoint_manager.os.remove = _fail_completed_remove

        error_text: str | None = None
        try:
            trainer.save_validation_progress(checkpoint_path)
        except RuntimeError as error:
            error_text = str(error)
        errors: list[str | None] = [None] * world_size
        dist.all_gather_object(errors, error_text)
        assert all(error is not None and f"completed-trainer {failure_stage}" in error for error in errors)
        upload_counts: list[int | None] = [None] * world_size
        dist.all_gather_object(upload_counts, upload_calls)
        assert sum(int(count or 0) for count in upload_counts) == (0 if failure_stage == "preparation" else 1)
        validation_state = load_component(checkpoint_path, "trainer")["validation"]
        assert validation_state["publication"] is not None
    finally:
        checkpoint_manager.tempfile.mkstemp = original_mkstemp
        checkpoint_manager.os.remove = original_remove
        object_store.upload_destination_identity = original_destination_identity
        checkpoint_manager.archive_checkpoint = original_upload
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_save_load_resume(tmp_path):
    torch.manual_seed(0)
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    tr = _trainer(seed=0)
    for _ in range(3):
        tr.train(env.get_samples(4))
    ckpt = str(tmp_path / "checkpoints" / "iter_000003")
    tr.save(ckpt)
    saved_net = {k: v.clone() for k, v in tr.model.net.state_dict().items()}
    saved_iter = tr._iter
    assert saved_iter == 3
    # Component layout: model/optim/trainer present; frozen ref deliberately ABSENT
    # (setup rebuilds the anchor from base — saving it would be pure redundancy).
    for name in ("model", "optim", "trainer"):
        assert (tmp_path / "checkpoints" / "iter_000003" / f"{name}.pt").is_file()
    assert not (tmp_path / "checkpoints" / "iter_000003" / "ref.pt").exists()
    assert (tmp_path / "checkpoints" / "latest_checkpoint.txt").read_text().strip() == "iter_000003"

    # Fresh trainer rebuilt from the SAME base anchor → resume overwrites the trained net...
    tr2 = _trainer(seed=0)
    ref_before = {k: v.clone() for k, v in tr2.model.net_ref.state_dict().items()}
    assert not _sd_close(tr2.model.net.state_dict(), saved_net)
    restored = tr2.load(ckpt)  # resume; returns the restored iteration
    assert restored == saved_iter
    assert _sd_close(tr2.model.net.state_dict(), saved_net)
    assert tr2.iteration() == saved_iter
    # ...but NOT the frozen ref: load leaves the setup-built anchor untouched.
    assert _sd_close(tr2.model.net_ref.state_dict(), ref_before)
    # Optimizer state restored (AdamW step count present).
    assert tr2.opt.state_dict()["state"], "optimizer state not restored"
    # Training continues from the restored iter.
    tr2.train(env.get_samples(4))
    assert tr2._iter == saved_iter + 1


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_scheduler_identity_and_state_round_trip_exactly(tmp_path: Path) -> None:
    scheduler_config = WFMLRSchedulerConfig(
        warm_up_steps=2,
        cycle_length=6,
        f_start=0.1,
        f_max=1.0,
        f_min=0.25,
    )
    trainer = _wfm_trainer(scheduler_config)
    trainer.train(SyntheticPreferenceEnv(dim=8, seed=1).get_samples(2))
    checkpoint = tmp_path / "checkpoints" / "iter_000001"
    trainer.save(str(checkpoint))

    restored = _wfm_trainer(scheduler_config)
    assert restored.load(str(checkpoint)) == 1
    assert restored.sched.state_dict() == trainer.sched.state_dict()
    assert restored.opt.param_groups[0]["lr"] == pytest.approx(trainer.opt.param_groups[0]["lr"])

    trainer.opt.step()
    trainer.sched.step()
    restored.opt.step()
    restored.sched.step()
    assert restored.opt.param_groups[0]["lr"] == pytest.approx(trainer.opt.param_groups[0]["lr"])


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("failure", ["changed_config", "missing_identity", "unknown_version"])
def test_wfm_scheduler_resume_contract_fails_before_state_mutation(
    tmp_path: Path,
    failure: str,
) -> None:
    saved_config = WFMLRSchedulerConfig(warm_up_steps=2, cycle_length=6, f_start=0.1, f_max=1.0, f_min=0.25)
    saved = _wfm_trainer(saved_config)
    saved.train(SyntheticPreferenceEnv(dim=8, seed=1).get_samples(2))
    checkpoint = tmp_path / failure / "iter_000001"
    saved.save(str(checkpoint))

    configured = saved_config
    if failure == "changed_config":
        configured = WFMLRSchedulerConfig(
            warm_up_steps=2,
            cycle_length=7,
            f_start=0.1,
            f_max=1.0,
            f_min=0.25,
        )
    restored = _wfm_trainer(configured)
    if failure != "changed_config":
        trainer_path = checkpoint / "trainer.pt"
        trainer_state = torch.load(trainer_path, map_location="cpu", weights_only=False)
        if failure == "missing_identity":
            trainer_state.pop("scheduler_identity")
        else:
            trainer_state["scheduler_identity"]["version"] = 2
        torch.save(trainer_state, trainer_path)

    scheduler_before = copy.deepcopy(restored.sched.state_dict())
    optimizer_before = copy.deepcopy(restored.opt.state_dict())
    net_before = {key: value.clone() for key, value in restored.model.net.state_dict().items()}
    error = "predates the WFM scheduler" if failure == "missing_identity" else "scheduler exact-resume identity differs"
    with pytest.raises(RuntimeError, match=error):
        restored.load(str(checkpoint))

    assert restored.sched.state_dict() == scheduler_before
    assert restored.opt.state_dict() == optimizer_before
    assert _sd_close(restored.model.net.state_dict(), net_before)


@pytest.mark.L0
@pytest.mark.CPU
def test_resume_from_pruned_optimizer_names_init_from(tmp_path: Path) -> None:
    trainer = _trainer(seed=0)
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.save(str(checkpoint))
    (checkpoint / "optim.pt").unlink()

    resumed = _trainer(seed=1)
    with pytest.raises(RuntimeError, match="set rl.init_from"):
        resumed.load(str(checkpoint))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_diffusion_model_only_plan_skips_optimizer_and_training_state_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_format: str,
) -> None:
    """Model selection extracts policy/EMA state without touching continuation state."""
    import torch.distributed.checkpoint.state_dict as state_dict_module

    from cosmos_framework.posttrain.utils import rank_state

    def _forbid_extraction(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("excluded checkpoint state was extracted")

    model = _EMAToyFlowModel(dim=8, seed=0)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        optimizer,
        checkpoint_format=checkpoint_format,
        update_ema_after_iter=True,
        update_old_every_iter=1,
        update_ref_every_iter=1,
    )
    monkeypatch.setattr(optimizer, "state_dict", _forbid_extraction)
    monkeypatch.setattr(trainer, "_trainer_state", _forbid_extraction)
    monkeypatch.setattr(trainer, "_old_policy_checkpoint_record", _forbid_extraction)
    monkeypatch.setattr(model.net_old, "state_dict", _forbid_extraction)
    monkeypatch.setattr(model.net_ref, "state_dict", _forbid_extraction)
    monkeypatch.setattr(rank_state, "collect_rank_state", _forbid_extraction)
    monkeypatch.setattr(state_dict_module, "get_optimizer_state_dict", _forbid_extraction)

    save = Mock(return_value="unused")
    monkeypatch.setattr(trainer.checkpoint_manager, f"save_{checkpoint_format}", save)

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        trainer.save(
            str(tmp_path / "checkpoints" / "iter_000001"),
            data_state={"cursor": 1},
            update_latest=False,
            components=("model",),
        )

    if checkpoint_format == "dcp":
        captured, control = save.call_args.args[1:3]
        assert {key.split(".", 1)[0] for key in captured["model"]} == {"net", "net_ema"}
        assert control == {}
    else:
        captured = save.call_args.args[1]
        assert set(captured["model"]) == {"net", "net_ema"}
    assert {name for name, state in captured.items() if state is not None} == {"model"}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("moving", [False, True])
def test_diffusion_training_state_plan_extracts_only_continuation_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    moving: bool,
) -> None:
    """Training state includes moving networks but never policy/EMA or optimizer state."""

    def _forbid_extraction(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("excluded checkpoint state was extracted")

    model = _EMAToyFlowModel(dim=8, seed=0)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        optimizer,
        update_ema_after_iter=True,
        update_old_every_iter=int(moving),
        update_ref_every_iter=int(moving),
    )
    monkeypatch.setattr(model.net, "state_dict", _forbid_extraction)
    monkeypatch.setattr(model.net_ema, "state_dict", _forbid_extraction)
    monkeypatch.setattr(optimizer, "state_dict", _forbid_extraction)
    if not moving:
        monkeypatch.setattr(trainer, "_checkpoint_nets", _forbid_extraction)

    save = Mock(return_value="unused")
    monkeypatch.setattr(trainer.checkpoint_manager, "save_consolidated", save)

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        trainer.save(
            str(tmp_path / "checkpoints" / "iter_000001"),
            data_state={"cursor": 1},
            update_latest=False,
            components=("training_state",),
        )

    captured = save.call_args.args[1]
    expected = {"old", "ref", "trainer", "data"} if moving else {"trainer", "data"}
    assert {name for name, state in captured.items() if state is not None} == expected
    if moving:
        assert captured["trainer"]["old_policy_checkpoint"]["mode"] == "materialized"


@pytest.mark.L0
@pytest.mark.CPU
def test_moving_ref_is_saved(tmp_path):
    """update_ref_every_iter > 0 ⇒ the reference is training state and MUST be saved."""
    model = ToyFlowModel(dim=8, seed=0)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    tr = DiffusionTrainer(model, make_objective(beta=100.0), opt, update_ref_every_iter=5)
    ckpt = str(tmp_path / "iter_000000")
    tr.save(ckpt)
    assert (tmp_path / "iter_000000" / "ref.pt").is_file()
    # And it round-trips: a fresh trainer's ref gets overwritten on resume.
    model2 = ToyFlowModel(dim=8, seed=7)
    tr2 = DiffusionTrainer(
        model2, make_objective(beta=100.0), torch.optim.AdamW(model2.net.parameters(), lr=1e-2), update_ref_every_iter=5
    )
    assert not _sd_close(tr2.model.net_ref.state_dict(), tr.model.net_ref.state_dict())
    tr2.load(ckpt)
    assert _sd_close(tr2.model.net_ref.state_dict(), tr.model.net_ref.state_dict())


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_moving_old_policy_resume_is_exact_and_fails_before_mutation(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    source_model = ToyFlowModel(dim=8, seed=0)
    source = DiffusionTrainer(
        source_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(source_model.net.parameters(), lr=1e-2),
        update_old_every_iter=2,
        checkpoint_format=checkpoint_format,
    )
    for _ in range(3):
        source.train(env.get_samples(4))
    assert source.iteration() == 3
    assert not _sd_close(source_model.net.state_dict(), source_model.net_old.state_dict())
    checkpoint = tmp_path / checkpoint_format / "iter_000003"
    source.save(str(checkpoint))
    source.finalize_checkpoints()
    old_component = checkpoint / ("old" if checkpoint_format == "dcp" else "old.pt")
    assert old_component.exists()
    trainer_state = load_component(str(checkpoint), "trainer")
    assert isinstance(trainer_state, dict)
    assert trainer_state["old_policy_checkpoint"] == {
        "version": 1,
        "mode": "materialized",
        "optimizer_step": 3,
    }

    restored_model = ToyFlowModel(dim=8, seed=0)
    restored = DiffusionTrainer(
        restored_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(restored_model.net.parameters(), lr=1e-2),
        update_old_every_iter=2,
        checkpoint_format=checkpoint_format,
    )
    assert restored.load(str(checkpoint)) == 3
    assert _sd_close(restored_model.net.state_dict(), source_model.net.state_dict())
    assert _sd_close(restored_model.net_old.state_dict(), source_model.net_old.state_dict())

    cadence_model = ToyFlowModel(dim=8, seed=0)
    cadence_mismatch = DiffusionTrainer(
        cadence_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(cadence_model.net.parameters(), lr=1e-2),
        update_old_every_iter=3,
        checkpoint_format=checkpoint_format,
    )
    policy_before = deepcopy(cadence_model.net.state_dict())
    old_before = deepcopy(cadence_model.net_old.state_dict())
    with pytest.raises(RuntimeError, match="old-policy refresh cadence 2 differs from configured cadence 3"):
        cadence_mismatch.load(str(checkpoint))
    assert _sd_close(cadence_model.net.state_dict(), policy_before)
    assert _sd_close(cadence_model.net_old.state_dict(), old_before)

    incomplete = tmp_path / checkpoint_format / "missing-old"
    shutil.copytree(checkpoint, incomplete)
    if checkpoint_format == "dcp":
        shutil.rmtree(incomplete / "old")
    else:
        (incomplete / "old.pt").unlink()
    missing_model = ToyFlowModel(dim=8, seed=0)
    missing = DiffusionTrainer(
        missing_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(missing_model.net.parameters(), lr=1e-2),
        update_old_every_iter=2,
        checkpoint_format=checkpoint_format,
    )
    policy_before = deepcopy(missing_model.net.state_dict())
    old_before = deepcopy(missing_model.net_old.state_dict())
    with pytest.raises(RuntimeError, match="moving-old-policy resume requires an old component"):
        missing.load(str(incomplete))
    assert _sd_close(missing_model.net.state_dict(), policy_before)
    assert _sd_close(missing_model.net_old.state_dict(), old_before)

    if checkpoint_format == "dcp":
        unexpected = tmp_path / checkpoint_format / "unexpected-model-key"
        shutil.copytree(checkpoint, unexpected)
        shutil.rmtree(unexpected / "model")
        saved_policy = {f"net.{key}": value.detach().clone() for key, value in source_model.net.state_dict().items()}
        saved_policy["net.unexpected"] = torch.ones(1)
        save_component_dcp(str(unexpected), "model", saved_policy)
        strict_model = ToyFlowModel(dim=8, seed=0)
        strict = DiffusionTrainer(
            strict_model,
            make_objective(beta=100.0),
            torch.optim.AdamW(strict_model.net.parameters(), lr=1e-2),
            update_old_every_iter=2,
            checkpoint_format=checkpoint_format,
        )
        policy_before = deepcopy(strict_model.net.state_dict())
        old_before = deepcopy(strict_model.net_old.state_dict())
        with pytest.raises(RuntimeError, match="incompatible net tensor inventory.*unexpected"):
            strict.load(str(unexpected))
        assert _sd_close(strict_model.net.state_dict(), policy_before)
        assert _sd_close(strict_model.net_old.state_dict(), old_before)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_refresh_aligned_moving_old_checkpoint_is_derived_from_policy(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    source_model = ToyFlowModel(dim=8, seed=0)
    source = DiffusionTrainer(
        source_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(source_model.net.parameters(), lr=1e-2),
        update_old_every_iter=2,
        checkpoint_format=checkpoint_format,
    )
    for _ in range(2):
        source.train(env.get_samples(4))
    assert source.iteration() == 2
    assert _sd_close(source_model.net.state_dict(), source_model.net_old.state_dict())

    checkpoint = tmp_path / checkpoint_format / "iter_000002"
    source.save(str(checkpoint))
    source.finalize_checkpoints()

    old_component = checkpoint / ("old" if checkpoint_format == "dcp" else "old.pt")
    assert not old_component.exists()
    trainer_state = load_component(str(checkpoint), "trainer")
    assert isinstance(trainer_state, dict)
    assert trainer_state["old_policy_checkpoint"] == {
        "version": 1,
        "mode": "derived_from_policy",
        "optimizer_step": 2,
    }

    restored_model = ToyFlowModel(dim=8, seed=0)
    with torch.no_grad():
        for parameter in restored_model.net_old.parameters():
            parameter.add_(100.0)
    restored = DiffusionTrainer(
        restored_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(restored_model.net.parameters(), lr=1e-2),
        update_old_every_iter=2,
        checkpoint_format=checkpoint_format,
    )

    assert restored.load(str(checkpoint)) == 2
    assert _sd_close(restored_model.net.state_dict(), source_model.net.state_dict())
    assert _sd_close(restored_model.net_old.state_dict(), source_model.net.state_dict())


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_derived_old_marker_fails_before_resume_mutation(tmp_path: Path) -> None:
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    source_model = ToyFlowModel(dim=8, seed=0)
    source = DiffusionTrainer(
        source_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(source_model.net.parameters(), lr=1e-2),
        update_old_every_iter=2,
    )
    for _ in range(2):
        source.train(env.get_samples(4))
    checkpoint = tmp_path / "iter_000002"
    source.save(str(checkpoint))
    source.finalize_checkpoints()
    trainer_state = load_component(str(checkpoint), "trainer")
    assert isinstance(trainer_state, dict)
    trainer_state["old_policy_checkpoint"]["optimizer_step"] = 1
    replacement = checkpoint / "trainer-replacement.pt"
    torch.save(trainer_state, replacement)
    os.replace(replacement, checkpoint / "trainer.pt")

    restored_model = ToyFlowModel(dim=8, seed=0)
    restored = DiffusionTrainer(
        restored_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(restored_model.net.parameters(), lr=1e-2),
        update_old_every_iter=2,
    )
    policy_before = deepcopy(restored_model.net.state_dict())
    old_before = deepcopy(restored_model.net_old.state_dict())

    with pytest.raises(RuntimeError, match="storage marker step differs"):
        restored.load(str(checkpoint))
    assert _sd_close(restored_model.net.state_dict(), policy_before)
    assert _sd_close(restored_model.net_old.state_dict(), old_before)


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_single_file_checkpoint_loads(tmp_path):
    """A legacy frozen-DPO checkpoint cannot prove which unsaved base anchored its ref."""
    tr = _trainer(seed=0)
    payload = {
        "iter": 7,
        "accum": 0,
        "opt": tr.opt.state_dict(),
        "nets": {"net": tr.model.net.state_dict()},
    }
    legacy = tmp_path / "state.pt"
    torch.save(payload, str(legacy))
    tr2 = _trainer(seed=0)
    with pytest.raises(RuntimeError, match="predates the frozen-reference identity journal"):
        tr2.load(str(legacy))


@pytest.mark.L0
@pytest.mark.CPU
def test_load_copy_mode_init_from_base(tmp_path):
    torch.manual_seed(0)
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    tr = _trainer(seed=0)
    for _ in range(2):
        tr.train(env.get_samples(4))  # net diverges from net_ref (frozen)
    base_net = {k: v.clone() for k, v in tr.model.net.state_dict().items()}
    assert not _sd_close(tr.model.net.state_dict(), tr.model.net_ref.state_dict())
    ckpt = str(tmp_path / "iter_000002")
    tr.save(ckpt)

    tr2 = _trainer(seed=999)
    assert tr2.load(ckpt, copy_mode=True) == 0  # init-from-base
    # Policy, moving-old, and frozen-reference copies become the base; optimizer/iter do not.
    assert _sd_close(tr2.model.net.state_dict(), base_net)
    assert _sd_close(tr2.model.net_old.state_dict(), base_net)
    assert _sd_close(tr2.model.net_ref.state_dict(), base_net)
    assert tr2._iter == 0
    assert not tr2.opt.state_dict()["state"], "copy_mode must not restore optimizer state"

    with pytest.raises(ValueError, match="copy_mode requires strict=True"):
        _trainer(seed=999).load(ckpt, copy_mode=True, strict=False)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("source_format", ["consolidated", "dcp"])
def test_direct_primary_keeps_localized_init_from_on_the_local_reader(
    tmp_path: Path,
    source_format: str,
) -> None:
    """Selecting a remote primary must not redirect the separately localized base checkpoint."""
    source = _trainer(seed=0)
    source.checkpoint_format = source_format
    checkpoint = tmp_path / source_format / "iter_000000"
    source.save(str(checkpoint))
    source.finalize_checkpoints()
    expected = {name: value.detach().clone() for name, value in source.model.net.state_dict().items()}  # [*param]

    model = ToyFlowModel(dim=8, seed=999)
    manager = CheckpointManager(
        checkpoint_format="dcp",
        checkpoint_store={
            "backend": "object_store",
            "bucket": "primary",
            "prefix": "run/checkpoints",
            "credentials": "unused-remote-credentials.json",
        },
    )
    target = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        checkpoint_manager=manager,
        defer_frozen_reference_identity=True,
    )

    assert target.load(str(checkpoint), copy_mode=True) == 0
    assert _sd_close(target.model.net.state_dict(), expected)
    assert _sd_close(target.model.net_ref.state_dict(), expected)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_load_copy_mode_initializes_ema_when_present(tmp_path: Path, checkpoint_format: str) -> None:
    source_model = ToyFlowModel(dim=8, seed=0)
    source_model.net.to(dtype=torch.bfloat16)
    source_model.net_ref.to(dtype=torch.bfloat16)
    source = DiffusionTrainer(
        source_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(source_model.net.parameters(), lr=1e-2),
    )
    expected = {name: value.detach().clone() for name, value in source.model.net.state_dict().items()}  # [*param]
    checkpoint = tmp_path / checkpoint_format / "base"
    source.checkpoint_format = checkpoint_format
    source.save(str(checkpoint))
    source.finalize_checkpoints()

    model = _EMAToyFlowModel(
        dim=8,
        seed=999,
        policy_dtype=torch.bfloat16,
        ema_dtype=torch.float32,
    )
    target = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        update_ema_after_iter=True,
        checkpoint_format=checkpoint_format,
        defer_frozen_reference_identity=True,
    )
    assert target.load(str(checkpoint), copy_mode=True) == 0
    for actual in (model.net.state_dict(), model.net_ref.state_dict()):
        assert all(value.dtype is torch.bfloat16 for value in actual.values())
        assert all(torch.equal(value, expected[name]) for name, value in actual.items())
    old_state = model.net_old.state_dict()
    assert all(value.dtype is torch.float32 for value in old_state.values())
    assert all(torch.equal(value, expected[name].float()) for name, value in old_state.items())
    ema_state = model.net_ema.state_dict()
    assert all(value.dtype is torch.float32 for value in ema_state.values())
    assert all(torch.equal(value, expected[name].float()) for name, value in ema_state.items())


@pytest.mark.L0
@pytest.mark.CPU
def test_diffusion_trainer_requires_explicit_ema_capability() -> None:
    missing_updater = _EMANetWithoutUpdater(dim=8, seed=0)
    assert not isinstance(missing_updater, EMAFlowMatchingPolicy)
    with pytest.raises(ValueError, match="EMAFlowMatchingPolicy"):
        DiffusionTrainer(
            missing_updater,
            make_objective(beta=100.0),
            torch.optim.AdamW(missing_updater.net.parameters(), lr=1e-2),
            update_ema_after_iter=True,
        )

    missing_network = _EmptyEMAToyFlowModel(dim=8, seed=0)
    assert isinstance(missing_network, EMAFlowMatchingPolicy)
    with pytest.raises(ValueError, match="net_ema to be present"):
        DiffusionTrainer(
            missing_network,
            make_objective(beta=100.0),
            torch.optim.AdamW(missing_network.net.parameters(), lr=1e-2),
            update_ema_after_iter=True,
        )

    complete = _EMAToyFlowModel(dim=8, seed=0)
    assert isinstance(complete, EMAFlowMatchingPolicy)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_diffusion_trainer_excludes_unmaintained_ema_from_checkpoint(
    tmp_path: Path,
    checkpoint_format: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.utils.log import logger

    warnings: list[str] = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(logger, "warning", lambda message: warnings.append(str(message)))
    model = _EMAToyFlowModel(dim=8, seed=0)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        checkpoint_format=checkpoint_format,
    )
    assert warnings == [
        "policy exposes net_ema but EMA updates are disabled; the inactive EMA is not "
        "initialized from rl.init_from, checkpointed, or restored"
    ]

    checkpoint = tmp_path / checkpoint_format / "inactive-ema"
    trainer.save(str(checkpoint))
    trainer.finalize_checkpoints()
    if checkpoint_format == "consolidated":
        from cosmos_framework.posttrain.checkpoint import load_component

        keys = set((load_component(str(checkpoint), "model") or {}).keys())
    else:
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(str(checkpoint / "model")).read_metadata()
        keys = set(metadata.state_dict_metadata)
    assert any(key == "net" or key.startswith("net.") for key in keys)
    assert not any(key == "net_ema" or key.startswith("net_ema.") for key in keys)

    target_model = _EMAToyFlowModel(dim=8, seed=999)
    ema_before = {name: value.detach().clone() for name, value in target_model.net_ema.state_dict().items()}  # [*param]
    target = DiffusionTrainer(
        target_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(target_model.net.parameters(), lr=1e-2),
        checkpoint_format=checkpoint_format,
        defer_frozen_reference_identity=True,
    )
    assert target.load(str(checkpoint), copy_mode=True) == 0
    assert _sd_close(target_model.net.state_dict(), model.net.state_dict())
    assert _sd_close(target_model.net_ref.state_dict(), model.net.state_dict())
    assert _sd_close(target_model.net_ema.state_dict(), ema_before)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_ema_resume_requires_matching_enabled_contract(tmp_path: Path, checkpoint_format: str) -> None:
    source_model = _EMAToyFlowModel(dim=8, seed=0)
    source = DiffusionTrainer(
        source_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(source_model.net.parameters(), lr=1e-2),
        update_ema_after_iter=True,
        checkpoint_format=checkpoint_format,
    )
    source.train(SyntheticPreferenceEnv(dim=8, seed=1).get_samples(4))
    checkpoint = tmp_path / checkpoint_format / "enabled-ema"
    source.save(str(checkpoint))
    source.finalize_checkpoints()

    resumed_model = _EMAToyFlowModel(dim=8, seed=0)
    resumed = DiffusionTrainer(
        resumed_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(resumed_model.net.parameters(), lr=1e-2),
        update_ema_after_iter=True,
        checkpoint_format=checkpoint_format,
    )
    assert resumed.load(str(checkpoint)) == 1
    assert _sd_close(resumed_model.net_ema.state_dict(), source_model.net_ema.state_dict())

    disabled_model = _EMAToyFlowModel(dim=8, seed=0)
    disabled = DiffusionTrainer(
        disabled_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(disabled_model.net.parameters(), lr=1e-2),
        checkpoint_format=checkpoint_format,
    )
    with pytest.raises(RuntimeError, match="EMA-update contract differs"):
        disabled.load(str(checkpoint))

    legacy = tmp_path / checkpoint_format / "legacy-enabled-ema"
    shutil.copytree(checkpoint, legacy)
    trainer_state = torch.load(legacy / "trainer.pt", map_location="cpu", weights_only=False)
    trainer_state.pop("ema_updates_enabled")
    torch.save(trainer_state, legacy / "trainer.pt")
    legacy_model = _EMAToyFlowModel(dim=8, seed=0)
    legacy_target = DiffusionTrainer(
        legacy_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(legacy_model.net.parameters(), lr=1e-2),
        update_ema_after_iter=True,
        checkpoint_format=checkpoint_format,
    )
    with pytest.raises(RuntimeError, match="predates the EMA-update contract"):
        legacy_target.load(str(legacy))

    malformed = tmp_path / checkpoint_format / "malformed-ema-contract"
    shutil.copytree(checkpoint, malformed)
    malformed_state = torch.load(malformed / "trainer.pt", map_location="cpu", weights_only=False)
    malformed_state["ema_updates_enabled"] = None
    torch.save(malformed_state, malformed / "trainer.pt")
    malformed_model = _EMAToyFlowModel(dim=8, seed=0)
    malformed_target = DiffusionTrainer(
        malformed_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(malformed_model.net.parameters(), lr=1e-2),
        update_ema_after_iter=True,
        checkpoint_format=checkpoint_format,
    )
    with pytest.raises(RuntimeError, match="EMA-update contract must be a boolean"):
        malformed_target.load(str(malformed))

    missing_model = _EMAToyFlowModel(dim=8, seed=777)
    missing_source = DiffusionTrainer(
        missing_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(missing_model.net.parameters(), lr=1e-2),
        checkpoint_format=checkpoint_format,
    )
    missing_ema = tmp_path / checkpoint_format / "missing-enabled-ema"
    missing_source.save(str(missing_ema))
    missing_source.finalize_checkpoints()
    shutil.copyfile(checkpoint / "trainer.pt", missing_ema / "trainer.pt")

    strict_model = _EMAToyFlowModel(dim=8, seed=0)
    strict_target = DiffusionTrainer(
        strict_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(strict_model.net.parameters(), lr=1e-2),
        update_ema_after_iter=True,
        checkpoint_format=checkpoint_format,
    )
    net_before = {name: value.detach().clone() for name, value in strict_model.net.state_dict().items()}  # [*param]
    ema_before = {name: value.detach().clone() for name, value in strict_model.net_ema.state_dict().items()}  # [*param]
    with pytest.raises(RuntimeError, match="net_ema"):
        strict_target.load(str(missing_ema))
    assert _sd_close(strict_model.net.state_dict(), net_before)
    assert _sd_close(strict_model.net_ema.state_dict(), ema_before)


@pytest.mark.L0
@pytest.mark.CPU
def test_move_to_device_rebuilds_structure():
    dev = torch.device("cpu")
    traj = Trajectory(transitions=[])
    from cosmos_framework.posttrain.schema import Preference, Transition

    v = torch.randn(3, 4)
    # The production targets payload is a typed Preference (a dataclass) — move_to_device must
    # recurse into it (dataclass → dict → list → tensor) and rebuild it, leaving non-tensors alone.
    traj.transitions.append(
        Transition(obs=None, targets=Preference(winner={"video": [v], "cap": "hi"}, loser={"video": [v]}))
    )
    out = move_to_device([traj], dev)
    assert isinstance(out, list) and isinstance(out[0], Trajectory)
    tgt = out[0].transitions[0].targets
    assert isinstance(tgt, Preference)
    assert isinstance(tgt.winner["video"][0], torch.Tensor)
    assert tgt.winner["video"][0].device == dev
    assert tgt.winner["cap"] == "hi"  # non-tensors untouched
    # Source is not mutated (a moved copy is returned).
    assert traj.transitions[0].targets.winner["video"][0] is v

    frozen = _FrozenPayload(torch.ones(2))  # [2]
    frozen_out = move_to_device(frozen, dev)
    assert isinstance(frozen_out, _FrozenPayload)
    assert frozen_out is not frozen
    assert frozen_out.tensor.device == dev


@pytest.mark.L0
@pytest.mark.CPU
def test_held_out_objective_validation_is_finite_exact_and_side_effect_free() -> None:
    trainer = _trainer_with_validation(seed=3, held_out_size=5)
    net_probe = torch.nn.Dropout()
    trainer.model.net.add_module("validation_mode_probe", net_probe)
    auxiliary = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Identity())
    trainer.model.validation_auxiliary = auxiliary
    trainer.model.net.train()
    net_probe.eval()
    trainer.model.net_ref.eval()
    auxiliary.train()
    auxiliary[0].eval()
    roots = (trainer.model.net, trainer.model.net_ref, auxiliary)
    modes_before = {id(module): module.training for root in roots for module in root.modules()}
    objective = trainer.objective

    def _assert_eval_then_run(
        model: Any,
        trajectories: list[Trajectory],
    ) -> tuple[torch.Tensor, dict[str, Any]]:  # loss: [], scalar metrics
        assert all(not module.training for root in roots for module in root.modules())
        return objective(model, trajectories)

    trainer.objective = Objective(
        _assert_eval_then_run,
        reduction=objective.reduction,
        metric_adapter=objective.metric_adapter,
        identity=dict(objective.identity),
    )
    weights_before = {key: value.clone() for key, value in trainer.model.net.state_dict().items()}
    rng_before = torch.get_rng_state().clone()  # [RNG]
    reasons = frozenset({ValidationTrigger.BEFORE_TRAINING})

    trainer.mark_validation_pending(0, reasons)
    first = trainer.validate(model_step=0, trigger_reasons=reasons)
    comparable_first = {key: value for key, value in first.metrics.items() if key != "timer/validation_s"}
    assert first.model_step == 0
    assert first.sample_count == 5
    assert first.metrics["val/pair_count"] == 5
    assert first.metrics["val/batch_count"] == 3
    assert math.isfinite(float(first.metrics["val/loss_avg"]))
    assert trainer.pending_validation_step() == 0
    assert trainer.iteration() == 0
    assert trainer.model.net.training
    assert all(parameter.grad is None for parameter in trainer.model.net.parameters())
    assert _sd_close(trainer.model.net.state_dict(), weights_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    modes_after = {id(module): module.training for root in roots for module in root.modules()}
    assert modes_after == modes_before
    trainer.complete_ephemeral_validation(0)
    assert trainer.pending_validation_step() is None
    assert trainer.pending_validation_publication_step() is None

    second_trainer = _trainer_with_validation(seed=3, held_out_size=5)
    second_reasons = frozenset({ValidationTrigger.PERIODIC})
    second_trainer.mark_validation_pending(0, second_reasons)
    second = second_trainer.validate(model_step=0, trigger_reasons=second_reasons)
    comparable_second = {key: value for key, value in second.metrics.items() if key != "timer/validation_s"}
    assert comparable_second == comparable_first
    second_trainer.complete_validation(0)


@pytest.mark.L0
@pytest.mark.CPU
def test_diffusion_validation_requires_transaction_owned_pending_state() -> None:
    """The evaluator cannot silently create lifecycle state outside ValidationTransaction."""
    trainer = _trainer_with_validation(seed=3, held_out_size=1)

    with pytest.raises(RuntimeError, match="must mark model step 0 pending"):
        trainer.validate(0, frozenset({ValidationTrigger.BEFORE_TRAINING}))

    assert trainer.pending_validation_step() is None
    assert trainer.pending_validation_publication_step() is None


def _partitioned_validation_result(
    *,
    world_size: int,
    batch_size: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """Run simulated rank shards independently and recombine their weighted objective metrics."""
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective

    seeds_by_id: dict[str, int] = {}
    reports: list[Any] = []
    for rank in range(world_size):
        trainer = _trainer_with_validation(
            seed=3,
            held_out_size=7,
            validation_batch_size=batch_size,
            rank=rank,
            world_size=world_size,
        )
        objective = trainer.objective

        def _record_seeds(
            model: Any,
            trajectories: list[Trajectory],
            *,
            _objective: Objective = objective,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            for trajectory in trajectories:
                raw_seed = trajectory.transitions[0].algo_extra["validation_noise_seed"]
                seed = int(raw_seed)
                previous = seeds_by_id.setdefault(trajectory.id, seed)
                assert previous == seed
            return _objective(model, trajectories)

        trainer.objective = Objective(
            _record_seeds,
            reduction=objective.reduction,
            metric_adapter=objective.metric_adapter,
        )
        reasons = frozenset({ValidationTrigger.BEFORE_TRAINING})
        trainer.mark_validation_pending(0, reasons)
        report = trainer.validate(0, reasons)
        reports.append(report)
        trainer.complete_validation(0)

    total_pairs = sum(report.sample_count for report in reports)
    objective_keys = {
        key for key in reports[0].metrics if key.startswith("val/") and key not in {"val/pair_count", "val/batch_count"}
    }
    metrics = {
        key: sum(float(report.metrics[key]) * report.sample_count for report in reports) / total_pairs
        for key in objective_keys
    }
    return seeds_by_id, metrics


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_validation_per_sample_seed_is_batch_and_topology_invariant() -> None:
    """The same held-out pair receives the same noise and objective value after repartitioning."""
    baseline_seeds, baseline_metrics = _partitioned_validation_result(world_size=1, batch_size=1)
    expected_seeds = {
        f"pref-{index}": int.from_bytes(
            hashlib.sha256(f"17|held-out-tensor-pairs-v1|pref-{index}".encode()).digest()[:8],
            "big",
        )
        for index in range(7)
    }
    assert baseline_seeds == expected_seeds

    for world_size, batch_size in ((1, 3), (2, 2), (4, 2)):
        seeds, metrics = _partitioned_validation_result(
            world_size=world_size,
            batch_size=batch_size,
        )
        assert seeds == baseline_seeds
        assert metrics.keys() == baseline_metrics.keys()
        for key, expected in baseline_metrics.items():
            assert metrics[key] == pytest.approx(expected, rel=1e-6, abs=1e-7), key


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_seed_annotation_failure_converges_before_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer_with_validation(seed=3, held_out_size=1)
    source = trainer.validation_source
    assert source is not None
    invalid_trajectory = Trajectory(id="missing-transition", transitions=[])

    def _validation_batch_count(_batch_size: int, _max_samples: int | None) -> int:
        return 1

    def _validation_batches(
        _batch_size: int,
        _max_samples: int | None,
    ) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(batch_index=0, trajectories=[invalid_trajectory], logical_count=1)

    monkeypatch.setattr(source, "validation_batch_count", _validation_batch_count)
    monkeypatch.setattr(source, "validation_batches", _validation_batches)

    objective_called = False

    def _unexpected_objective(
        _model: Any,
        _trajectories: list[Trajectory],
    ) -> tuple[torch.Tensor, dict]:  # returns loss: [], metrics
        nonlocal objective_called
        objective_called = True
        raise AssertionError("objective must not run after validation seed preparation fails")

    trainer.objective = Objective(_unexpected_objective, Reduction.MEAN)
    converged: list[tuple[Exception | str | None, str]] = []

    def _converge(local_error: Exception | str | None, context: str) -> None:
        converged.append((local_error, context))
        if context == "validation sample seed preparation":
            assert isinstance(local_error, IndexError)
            raise RuntimeError("synthetic rank-converged seed failure")
        assert local_error is None

    monkeypatch.setattr(validation_module, "raise_if_any_rank_failed", _converge)
    reasons = frozenset({ValidationTrigger.BEFORE_TRAINING})
    trainer.mark_validation_pending(0, reasons)
    with pytest.raises(RuntimeError, match="synthetic rank-converged seed failure"):
        trainer.validate(0, reasons)

    assert not objective_called
    assert [context for _error, context in converged[-2:]] == [
        "validation sample seed preparation",
        "preference validation state restoration",
    ]
    assert converged[-1][0] is None


@pytest.mark.L0
@pytest.mark.CPU
def test_pending_validation_round_trips_in_trainer_checkpoint(tmp_path: Path) -> None:
    trainer = _trainer_with_validation(seed=5)
    reasons = frozenset({ValidationTrigger.FINAL})
    trainer.mark_validation_pending(0, reasons)
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.save(str(checkpoint))

    restored = _trainer_with_validation(seed=5)
    assert restored.load(str(checkpoint)) == 0
    assert restored.pending_validation_step() == 0
    report = restored.validate(0, frozenset({ValidationTrigger.PENDING}))
    assert report.trigger_reasons == frozenset({ValidationTrigger.FINAL, ValidationTrigger.PENDING})
    restored.complete_validation(0)
    assert restored.pending_validation_step() is None


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_completion_upload_failure_precedes_callback_and_leaves_no_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive failure stays recoverable and resume republishes without an async worker."""
    from cosmos_framework.posttrain.checkpoint import finalize, load_component, worker
    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager

    published: list[str] = []

    class _PublishSpy(Callback):
        def on_checkpoint(self, _trainer: Any, path: str) -> None:
            published.append(path)

    trainer = _trainer_with_validation(
        seed=5,
        callbacks=[_PublishSpy()],
        upload={"bucket": "archive", "prefix": "posttrain/checkpoints"},
    )
    trainer.mark_validation_pending(0, frozenset({ValidationTrigger.FINAL}))
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.save(str(checkpoint), publish=False)
    trainer.complete_validation(0)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.object_store.upload_destination_identity",
        lambda upload, iter_name: {
            "kind": "object_store",
            "endpoint_authority": "storage.example.test",
            "bucket": str(upload["bucket"]),
            "prefix": f"{str(upload['prefix']).rstrip('/')}/{iter_name}",
        },
    )

    def _fail_upload(
        iter_dir: str,
        upload: dict[str, str],
        *,
        overrides: dict[str, str] | None = None,
        commit: dict[str, Any] | None = None,
    ) -> None:
        del iter_dir, upload
        assert overrides is not None and commit is not None
        raise OSError("archive unavailable")

    monkeypatch.setattr(checkpoint_manager, "archive_checkpoint", _fail_upload)
    with pytest.raises(OSError, match="archive unavailable"):
        trainer.save_validation_progress(str(checkpoint))

    assert published == []
    assert worker._write_thread is None
    failed_state = load_component(str(checkpoint), "trainer")["validation"]
    assert failed_state["progress"]["pending_step"] is None
    assert failed_state["publication"]["step"] == 0

    trainer.upload["prefix"] = "different/checkpoints"
    with pytest.raises(RuntimeError, match="destination changed across resume"):
        trainer.save_validation_progress(str(checkpoint))
    trainer.upload["prefix"] = "posttrain/checkpoints"
    trainer.upload["credentials"] = "rotated-secret"
    with pytest.raises(OSError, match="archive unavailable"):
        trainer.save_validation_progress(str(checkpoint))

    resumed_published: list[str] = []

    class _ResumedPublishSpy(Callback):
        def on_checkpoint(self, _trainer: Any, path: str) -> None:
            resumed_published.append(path)

    restored = _trainer_with_validation(
        seed=5,
        callbacks=[_ResumedPublishSpy()],
        upload={"bucket": "archive", "prefix": "posttrain/checkpoints"},
    )
    assert restored.load(str(checkpoint)) == 0
    assert restored.pending_validation_step() is None
    assert restored.pending_validation_publication_step() == 0

    archived_states: list[dict[str, Any]] = []

    def _complete_upload(
        iter_dir: str,
        upload: dict[str, str],
        *,
        overrides: dict[str, str] | None = None,
        commit: dict[str, Any] | None = None,
    ) -> None:
        del upload
        assert overrides is not None and commit is not None
        archived_states.append(torch.load(overrides["trainer.pt"], map_location="cpu", weights_only=False))
        (Path(iter_dir) / ".uploaded").write_text("archive\n", encoding="utf-8")

    monkeypatch.setattr(checkpoint_manager, "archive_checkpoint", _complete_upload)
    original_replace = restored.checkpoint_manager.replace_component
    replace_calls = 0

    def _fail_final_clear(iter_dir: str, name: str, state: dict[str, Any]) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise OSError("crash before local clear")
        original_replace(iter_dir, name, state)

    monkeypatch.setattr(restored.checkpoint_manager, "replace_component", _fail_final_clear)
    with pytest.raises(OSError, match="crash before local clear"):
        restored.save_validation_progress(str(checkpoint))

    assert restored.pending_validation_publication_step() == 0
    assert load_component(str(checkpoint), "trainer")["validation"]["publication"] is not None
    assert resumed_published == []

    retry = _trainer_with_validation(
        seed=5,
        callbacks=[_ResumedPublishSpy()],
        upload={"bucket": "archive", "prefix": "posttrain/checkpoints"},
    )
    assert retry.load(str(checkpoint)) == 0
    assert retry.pending_validation_publication_step() == 0
    retry.save_validation_progress(str(checkpoint))

    assert archived_states[0]["validation"]["publication"] is None
    completed_state = load_component(str(checkpoint), "trainer")["validation"]
    assert completed_state["publication"] is None
    assert len(archived_states) == 2
    assert resumed_published == [str(checkpoint)]
    assert worker._write_thread is None
    finalize()


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_checkpoint_identity_includes_objective_identity(tmp_path: Path) -> None:
    trainer = _trainer_with_validation(seed=5)
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.save(str(checkpoint))

    restored = _trainer_with_validation(seed=5)
    restored.objective.identity["beta"] = 200.0
    with pytest.raises(RuntimeError, match="validation identity"):
        restored.load(str(checkpoint))


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_checkpoint_cannot_enable_diffusion_validation_on_exact_resume() -> None:
    trainer = _trainer_with_validation(seed=5)

    with pytest.raises(RuntimeError, match="predates the validation identity journal"):
        trainer._restore_trainer_state({"iter": 0, "accum": 0})


@pytest.mark.L0
@pytest.mark.CPU
def test_disabled_diffusion_validation_identity_has_no_inert_fields() -> None:
    trainer = _trainer(seed=5)
    trainer_state = trainer._trainer_state()
    validation_state = trainer_state["validation"]

    assert validation_state["identity"] == {"enabled": False}
    assert validation_state["progress"] == {
        "pending_step": None,
        "pending_reasons": [],
        "pending_metrics": {},
    }
    assert validation_state["publication"] is None
    assert trainer_state["reference_identity"]["mode"] == "frozen"
    assert trainer_state["reference_identity"]["source"]["kind"] == "materialized_reference"


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_journal_v1_loads_v2_rejects_and_disabled_legacy_resets() -> None:
    enabled = _trainer_with_validation(seed=5)
    v1_state = copy.deepcopy(enabled._trainer_state())
    v1_state["validation"]["version"] = 1
    v1_state["validation"].pop("publication")
    restored = _trainer_with_validation(seed=5)
    assert restored._restore_trainer_state(v1_state) == 0
    assert restored.pending_validation_step() is None
    assert restored.pending_validation_publication_step() is None

    v2_state = copy.deepcopy(v1_state)
    v2_state["validation"]["version"] = 2
    with pytest.raises(RuntimeError, match="unsupported schema"):
        restored._restore_trainer_state(v2_state)

    disabled = _trainer(seed=5)
    disabled.mark_validation_pending(0, frozenset({ValidationTrigger.FINAL}))
    disabled._restore_trainer_state(
        {
            "iter": 0,
            "accum": 0,
            "reference_identity": disabled.reference_identity.state_dict(),
        }
    )
    assert disabled.pending_validation_step() is None
    assert disabled.pending_validation_publication_step() is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_frozen_reference_identity_round_trips_and_rejects_same_path_byte_mutation(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    """Exact resume binds loaded net bytes, not the path or unrelated checkpoint components."""
    base_model = ToyFlowModel(dim=8, seed=11)
    base = DiffusionTrainer(
        base_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(base_model.net.parameters(), lr=1e-2),
        checkpoint_format=checkpoint_format,
    )
    base_checkpoint = tmp_path / checkpoint_format / "base" / "iter_000000"
    base.save(str(base_checkpoint))
    base.finalize_checkpoints()

    trainer = _trainer_from_base(
        base_checkpoint,
        seed=99,
        checkpoint_format=checkpoint_format,
    )
    trainer.train(SyntheticPreferenceEnv(dim=8, seed=3).get_samples(2))
    resume_checkpoint = tmp_path / checkpoint_format / "run" / "iter_000001"
    trainer.save(str(resume_checkpoint))
    trainer.finalize_checkpoints()

    (base_checkpoint / ".uploaded").write_text("publication side effect\n")
    restored = _trainer_from_base(
        base_checkpoint,
        seed=101,
        checkpoint_format=checkpoint_format,
    )
    assert restored.load(str(resume_checkpoint)) == 1

    # Replace only the loader-selected model component at the SAME base path. Unrelated optimizer,
    # trainer, publication, and data components remain intentionally outside the reference identity.
    alternate_model = ToyFlowModel(dim=8, seed=12)
    alternate = DiffusionTrainer(
        alternate_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(alternate_model.net.parameters(), lr=1e-2),
        checkpoint_format=checkpoint_format,
    )
    alternate_checkpoint = tmp_path / checkpoint_format / "alternate" / "iter_000000"
    alternate.save(str(alternate_checkpoint))
    alternate.finalize_checkpoints()
    if checkpoint_format == "dcp":
        shutil.rmtree(base_checkpoint / "model")
        shutil.copytree(alternate_checkpoint / "model", base_checkpoint / "model")
    else:
        shutil.copyfile(alternate_checkpoint / "model.pt", base_checkpoint / "model.pt")
    incompatible = _trainer_from_base(
        base_checkpoint,
        seed=101,
        checkpoint_format=checkpoint_format,
    )
    with pytest.raises(RuntimeError, match="checkpoint reference identity"):
        incompatible.load(str(resume_checkpoint))


@pytest.mark.L0
@pytest.mark.CPU
def test_frozen_reference_identity_rejects_changed_non_tensor_forward_semantics(tmp_path: Path) -> None:
    """Equal module schema/tensor bytes are insufficient when a scalar changes the forward."""
    original_model = _ScalarSemanticToyFlowModel(dim=8, seed=31, reference_scale=1.0)
    original = DiffusionTrainer(
        original_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(original_model.net.parameters(), lr=1e-2),
    )
    checkpoint = tmp_path / "semantic-run" / "iter_000000"
    original.save(str(checkpoint))

    changed_model = _ScalarSemanticToyFlowModel(dim=8, seed=31, reference_scale=2.0)
    changed = DiffusionTrainer(
        changed_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(changed_model.net.parameters(), lr=1e-2),
    )
    with pytest.raises(RuntimeError, match="checkpoint reference identity"):
        changed.load(str(checkpoint))


@pytest.mark.L0
@pytest.mark.CPU
def test_reference_identity_requires_adapter_declared_non_tensor_semantics() -> None:
    policy = SimpleNamespace(net=torch.nn.Linear(2, 2), net_ref=torch.nn.Linear(2, 2))
    with pytest.raises(TypeError, match="reference_semantics_identity"):
        DiffusionTrainer(
            policy,
            Objective(lambda *_args, **_kwargs: None, Reduction.MEAN),
            torch.optim.SGD(policy.net.parameters(), lr=0.0),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_sharded_frozen_reference_without_init_from_has_topology_independent_identity(
    tmp_path: Path,
) -> None:
    """The public quickstart has no init_from, so its FSDP ref is identified from global bytes."""
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "frozen-reference-identity-rendezvous")
    mp.spawn(
        _sharded_reference_identity_worker,
        args=(world_size, rendezvous_path),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "failure_mode",
    ["capability-error", "semantics-mismatch", "semantics-type-mismatch", "tensor-mismatch"],
)
def test_reference_identity_failures_converge_before_exact_resume(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / f"reference-identity-{failure_mode}-rendezvous")
    mp.spawn(
        _reference_identity_convergence_worker,
        args=(2, rendezvous_path, failure_mode),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("failure_stage", ["preparation", "cleanup"])
def test_validation_publication_rank0_temp_failures_converge_before_next_collective(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / f"validation-publication-{failure_stage}-rendezvous")
    checkpoint_path = str(tmp_path / failure_stage / "checkpoints" / "iter_000000")
    mp.spawn(
        _validation_publication_io_failure_worker,
        args=(2, rendezvous_path, checkpoint_path, failure_stage),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_unsharded_async_save_is_forced_sync_to_snapshot_live_tensors(tmp_path: Path) -> None:
    trainer = _trainer(seed=5)
    trainer.save_async = True
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    expected = {key: value.detach().clone() for key, value in trainer.model.net.state_dict().items()}  # [*param]

    trainer.save(str(checkpoint))
    with torch.no_grad():
        for parameter in trainer.model.net.parameters():
            parameter.add_(10.0)  # [*param]

    from cosmos_framework.posttrain.checkpoint import load_component

    saved = load_component(str(checkpoint), "model")["net"]
    assert _sd_close(saved, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_format_save_load_resume(tmp_path):
    """checkpoint_format="dcp": sharded dirs on disk (model/ optim/ as DCP, trainer.pt
    tiny), frozen ref still skipped, and resume round-trips net/opt/iter — load
    auto-detects the format from disk."""
    torch.manual_seed(0)
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    model = ToyFlowModel(dim=8, seed=0)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    tr = DiffusionTrainer(model, make_objective(beta=100.0), opt, checkpoint_format="dcp")
    for _ in range(2):
        tr.train(env.get_samples(4))
    ckpt = tmp_path / "checkpoints" / "iter_000002"
    tr.save(str(ckpt))
    # The DCP finalization (trainer.pt + pointer + retention/upload) runs on the async
    # write thread — join it before asserting, exactly as the entry does before exit.
    tr.finalize_checkpoints()
    assert (ckpt / "model").is_dir() and (ckpt / "optim").is_dir()  # DCP dirs
    assert (ckpt / "trainer.pt").is_file()  # tiny state stays a plain file
    assert not (ckpt / "ref").exists() and not (ckpt / "ref.pt").exists()  # frozen ref skipped
    assert (tmp_path / "checkpoints" / "latest_checkpoint.txt").read_text().strip() == "iter_000002"

    model2 = ToyFlowModel(dim=8, seed=0)
    tr2 = DiffusionTrainer(model2, make_objective(beta=100.0), torch.optim.AdamW(model2.net.parameters(), lr=1e-2))
    ref_before = {k: v.clone() for k, v in tr2.model.net_ref.state_dict().items()}
    assert tr2.load(str(ckpt)) == 2  # format auto-detected
    assert _sd_close(tr2.model.net.state_dict(), tr.model.net.state_dict())
    assert _sd_close(tr2.model.net_ref.state_dict(), ref_before)  # anchor untouched
    assert tr2.opt.state_dict()["state"], "optimizer state not restored via DCP"
    tr2.train(env.get_samples(4))
    assert tr2.iteration() == 3


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("inventory_delta", "error"),
    [
        ("missing", "missing 1 required tensors: net.new_state"),
        ("unexpected", "found 1 unexpected tensors: net.retired_state"),
    ],
)
def test_dcp_exact_resume_rejects_network_inventory_drift(
    tmp_path: Path,
    inventory_delta: str,
    error: str,
) -> None:
    source = _trainer(seed=0)
    source.checkpoint_format = "dcp"
    if inventory_delta == "unexpected":
        source.model.net.register_buffer("retired_state", torch.ones(1))  # [1]
    checkpoint = tmp_path / inventory_delta / "iter_000000"
    source.save(str(checkpoint))
    source.finalize_checkpoints()

    restored = _trainer(seed=0)
    if inventory_delta == "missing":
        restored.model.net.register_buffer("new_state", torch.ones(1))  # [1]
    before = {name: value.detach().clone() for name, value in restored.model.net.state_dict().items()}  # [*state]

    with pytest.raises(RuntimeError, match=error):
        restored.load(str(checkpoint))

    assert _sd_close(restored.model.net.state_dict(), before)


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_dcp_resume_rejects_unexpected_network_key_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ToyFlowModel(dim=8, seed=0)
    manager = CheckpointManager(
        checkpoint_format="dcp",
        checkpoint_store={
            "backend": "object_store",
            "bucket": "checkpoints",
            "prefix": "run/checkpoints",
            "credentials": "unused-remote-credentials.json",
        },
    )
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        checkpoint_manager=manager,
    )
    trainer_state = trainer._trainer_state()
    inventory = {
        f"net.{name}": tuple(value.shape)
        for name, value in trainer.model.net.state_dict().items()  # [*state]
    }
    inventory["net.retired_state"] = (1,)
    loaded: list[str] = []
    manager = trainer.checkpoint_manager
    monkeypatch.setattr(manager, "load_component", lambda _path, name: trainer_state if name == "trainer" else None)
    monkeypatch.setattr(manager, "has_dcp_component", lambda _path, name: name in {"model", "optim"})
    monkeypatch.setattr(manager, "dcp_tensor_inventory", lambda _path, name: inventory if name == "model" else {})
    monkeypatch.setattr(manager, "load_dcp_component", lambda _path, name, _state: loaded.append(name) or True)
    before = {name: value.detach().clone() for name, value in trainer.model.net.state_dict().items()}  # [*state]

    with pytest.raises(RuntimeError, match="found 1 unexpected tensors: net.retired_state"):
        trainer.load("s3://checkpoints/run/checkpoints/iter_000000")

    assert loaded == []
    assert _sd_close(trainer.model.net.state_dict(), before)


@pytest.mark.L0
@pytest.mark.CPU
def test_grad_clip_and_force_finite() -> None:
    """A finite global norm is clipped and exported as ``train/grad_norm``."""
    model = ToyFlowModel(dim=8, seed=0)
    opt = torch.optim.SGD(model.net.parameters(), lr=0.0)  # lr 0: isolate grad handling
    tr = DiffusionTrainer(model, make_objective(beta=100.0), opt, grad_clip_norm=1.0, force_finite_grads=True)
    env = SyntheticPreferenceEnv(dim=8, seed=1)

    metrics = tr.train(env.get_samples(4))
    assert "train/grad_norm" in metrics
    # After clipping, the realized global grad norm is <= clip_norm (+ tolerance).
    total = torch.sqrt(sum((p.grad**2).sum() for p in model.net.parameters() if p.grad is not None))
    assert float(total) <= 1.0 + 1e-4

    token_sum = Objective(lambda *_args: (torch.tensor(0.0), {}), Reduction.TOKEN_SUM)
    with pytest.raises(ValueError, match="token_sum reduction requires grad_accum=1"):
        DiffusionTrainer(model, token_sum, opt, grad_accum=2)


@pytest.mark.L0
@pytest.mark.CPU
def test_diffusion_trainer_updates_ema_after_applied_step() -> None:
    model = _EMAToyFlowModel(dim=8, seed=0)
    optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-2)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        optimizer,
        update_ema_after_iter=True,
    )

    source = SyntheticPreferenceEnv(dim=8, seed=1)
    trainer.train(source.get_samples(4))
    trainer.train(source.get_samples(4))

    assert model.ema_iterations == [0, 1]
    assert trainer.iteration() == 2
    assert _sd_close(model.net_ema.state_dict(), model.net.state_dict())


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"optimizer_step": 2.0}, "optimizer_step must be a non-negative integer"),
        ({"attempt_id": "3"}, "attempt_id must be a non-negative integer"),
        ({"accum": True}, "accum must be a non-negative integer"),
        ({"nonfinite_skips_total": -1}, "nonfinite_skips_total must be a non-negative integer"),
        ({"zero_grad_skips_total": -1}, "zero_grad_skips_total must be a non-negative integer"),
        ({"iter": 1}, "differs from optimizer_step"),
        ({"accum": 1}, "quiescent accumulation boundary"),
        ({"attempt_id": 4}, "attempt clock is inconsistent"),
        ({"nonfinite_skips_consecutive": 2}, "skip counters are invalid"),
    ],
)
def test_restore_rejects_malformed_optimizer_control_state(
    override: dict[str, Any],
    error: str,
) -> None:
    trainer = _trainer(seed=0)
    state: dict[str, Any] = {
        "optimizer_step": 2,
        "iter": 2,
        "attempt_id": 3,
        "accum": 0,
        "nonfinite_skips_total": 1,
        "nonfinite_skips_consecutive": 1,
    }
    state.update(override)

    with pytest.raises(RuntimeError, match=error):
        trainer._restore_trainer_state(state)


@pytest.mark.L0
@pytest.mark.CPU
def test_moving_reference_legacy_timing_is_converted_and_unknown_timing_fails() -> None:
    model = _CountingToyFlowModel(dim=8, seed=0)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        update_ref_every_iter=2,
    )
    assert model.refresh_calls == 1
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.add_(1.0)
    assert not _sd_close(model.net.state_dict(), model.net_ref.state_dict())

    assert trainer._restore_trainer_state({"iter": 2, "accum": 0}) == 2
    assert model.refresh_calls == 2
    assert _sd_close(model.net.state_dict(), model.net_ref.state_dict())

    with pytest.raises(RuntimeError, match="unsupported moving-reference refresh timing"):
        trainer._restore_trainer_state(
            {
                "optimizer_step": 2,
                "iter": 2,
                "attempt_id": 2,
                "accum": 0,
                "nonfinite_skips_total": 0,
                "nonfinite_skips_consecutive": 0,
                "ref_refresh_timing": "before_next_update_v0",
            }
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("missing_component", "error_type", "error"),
    [
        ("model", FileNotFoundError, "requires policy net state"),
        ("ref", RuntimeError, "moving-reference resume requires a ref component"),
        ("optim", RuntimeError, "use rl.init_from"),
    ],
)
def test_dcp_resume_requires_training_components(
    tmp_path: Path,
    missing_component: str,
    error_type: type[Exception],
    error: str,
) -> None:
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    model = ToyFlowModel(dim=8, seed=0)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        update_ref_every_iter=5,
        checkpoint_format="dcp",
    )
    trainer.train(env.get_samples(2))
    checkpoint = tmp_path / "complete" / "iter_000001"
    trainer.save(str(checkpoint))
    trainer.finalize_checkpoints()

    incomplete = tmp_path / f"missing-{missing_component}"
    shutil.copytree(checkpoint, incomplete)
    shutil.rmtree(incomplete / missing_component)
    restored_model = ToyFlowModel(dim=8, seed=7)
    restored = DiffusionTrainer(
        restored_model,
        make_objective(beta=100.0),
        torch.optim.AdamW(restored_model.net.parameters(), lr=1e-2),
        update_ref_every_iter=5,
    )
    with pytest.raises(error_type, match=error):
        restored._load_dcp(str(incomplete))


@pytest.mark.L0
@pytest.mark.CPU
def test_reduce_metrics_sorts_non_scalar_collectives(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    trainer = _trainer(seed=0)
    collective_payloads: list[torch.Tensor] = []

    def _record_all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
        del op
        payload = tensor.detach().clone()  # [*collective payload]
        collective_payloads.append(payload)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(dist, "all_reduce", _record_all_reduce)
    metrics = {
        "z_vector": torch.tensor([30.0, 31.0]),  # [2]
        "scalar": torch.tensor(20.0),  # []
        "a_vector": torch.tensor([10.0, 11.0]),  # [2]
    }

    reduced = trainer._reduce_metrics(metrics)
    expected_scalar_payload = torch.tensor([20.0])  # [1]

    assert len(collective_payloads) == 3
    assert torch.equal(collective_payloads[0], expected_scalar_payload)  # packed scalars first
    assert torch.equal(collective_payloads[1], metrics["a_vector"])
    assert torch.equal(collective_payloads[2], metrics["z_vector"])
    assert set(reduced) == set(metrics)
