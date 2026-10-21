# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Vertical CPU contracts for DPO trainer assembly and source-cursor checkpoint handoff."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.algorithm.dpo import make_objective
from cosmos_framework.posttrain.checkpoint import load_component
from cosmos_framework.posttrain.control.checkpointing import save_policy_checkpoint
from cosmos_framework.posttrain.entrypoints import dpo as dpo_entrypoint
from cosmos_framework.posttrain.entrypoints.dpo import make_setup
from cosmos_framework.posttrain.roles.env.preference import CursorStateUnavailable, DatasetPreferenceEnv
from cosmos_framework.posttrain.roles.trainer.diffusion import DiffusionTrainer
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel
from cosmos_framework.posttrain.utils.rank_state import state_for_current_rank
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup


class _CursorDataset:
    def __init__(self, size: int, seed: int) -> None:
        self._items = [f"pair-{index}" for index in range(size)]
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, dict[str, str], dict[str, str]]:
        suffix = self._rng.random()
        item = self._items[index]
        return f"{item}:{suffix}", {"item": item}, {"item": item}

    def cursor_fingerprint(self) -> str:
        return "|".join(self._items)

    def cursor_state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_cursor_state_dict(self, state: dict[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class _UnidentifiedDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[str, dict[str, str], dict[str, str]]:
        return str(index), {"item": "winner"}, {"item": "loser"}


def _entry_config() -> SimpleNamespace:
    return SimpleNamespace(
        job=SimpleNamespace(project="project", group="group", name="name"),
        rl=SimpleNamespace(run_ts="20260902000000"),
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_public_dpo_quickstart_keeps_frozen_reference_without_init_from() -> None:
    """Guard the supported random-backbone FSDP path that must derive identity from net_ref."""
    from cosmos_framework.posttrain.config.experiment.dpo.cosmos3_ga_16bm8b_dpo_v1 import (
        cosmos3_ga_16bm8b_dpo_v1,
    )

    rl = cosmos3_ga_16bm8b_dpo_v1["rl"]
    assert rl.init_from == ""
    assert bool(rl.ref.build_ref)
    assert int(rl.ref.update_every) == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_refreshes_reference_after_backbone_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.config import build as build_module
    from cosmos_framework.posttrain.model import cosmos3 as cosmos3_model_module

    events: list[str] = []
    training_data = SimpleNamespace(name="training")
    held_out_data = SimpleNamespace(name="held-out")

    class _SetupBackbone(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 1, bias=False)
            self.setup_complete: bool = True

    class _SetupAwarePolicy:
        def __init__(self, model: _SetupBackbone, build_ref: bool) -> None:
            assert model.setup_complete
            assert build_ref
            self.net: _SetupBackbone = model
            self.net_ref: torch.nn.Module = torch.nn.Linear(2, 1, bias=False)
            events.append("policy_constructed")

        def refresh_ref(self) -> None:
            assert self.net.setup_complete
            events.append("reference_refreshed")

        def reference_semantics_identity(self) -> dict[str, int | str]:
            return {"adapter": "setup-aware-policy-test", "version": 1}

    def _build_backbone(config: Any, world_size: int) -> _SetupBackbone:
        del config
        assert world_size == 2
        model = _SetupBackbone()
        events.append("backbone_setup_complete")
        return model

    def _build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
        del config
        events.append("optimizer_constructed")
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def _build_preference_env(
        data: Any,
        seed: int,
        *,
        include_source_identity: bool = False,
    ) -> DatasetPreferenceEnv:
        assert data is held_out_data
        assert seed == 23
        assert include_source_identity
        events.append("held_out_source_constructed")
        return DatasetPreferenceEnv(_CursorDataset(1, seed=seed), shuffle=False, seed=seed)

    monkeypatch.setattr(build_module, "build_backbone", _build_backbone)
    monkeypatch.setattr(build_module, "build_optimizer", _build_optimizer)
    monkeypatch.setattr(cosmos3_model_module, "Cosmos3FlowPolicy", _SetupAwarePolicy)
    monkeypatch.setattr("cosmos_framework.posttrain.roles.env.build_preference_env", _build_preference_env)
    rl = SimpleNamespace(
        seed=17,
        data=training_data,
        validation=SimpleNamespace(
            enabled=True,
            data=held_out_data,
            seed=23,
            batch_size=1,
            max_samples=None,
            before_training=True,
            every_steps=1,
        ),
        ref=SimpleNamespace(build_ref=True, update_every=1),
        optimizer=SimpleNamespace(grad_clip_norm=0.0, force_finite_grads=True),
        beta=1.0,
        grad_accum=1,
        max_steps=1,
        checkpoint_format="consolidated",
        save_async=False,
        keep_optim_last=0,
        keep_model_last=0,
        upload=SimpleNamespace(enabled=False),
        init_from="",
        resume="off",
    )

    trainer = dpo_entrypoint.build_trainer(_entry_config(), rl, world_size=2, resume_checkpoint="")

    assert isinstance(trainer, DiffusionTrainer)
    assert events == [
        "held_out_source_constructed",
        "backbone_setup_complete",
        "policy_constructed",
        "optimizer_constructed",
        "reference_refreshed",
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_defers_frozen_reference_identity_until_init_from_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entry avoids hashing a temporary backbone and binds what copy_mode loaded."""
    from cosmos_framework.posttrain.config import build as build_module
    from cosmos_framework.posttrain.model import cosmos3 as cosmos3_model_module
    from cosmos_framework.posttrain.roles.trainer import diffusion as diffusion_module

    base_checkpoint = tmp_path / "base-checkpoint.bin"
    base_checkpoint.write_bytes(b"base-placeholder")
    load_calls = 0

    class _Policy:
        def __init__(self, model: torch.nn.Module, build_ref: bool) -> None:
            assert build_ref
            self.net: torch.nn.Module = model
            self.net_ref: torch.nn.Module = torch.nn.Linear(2, 1, bias=False)

    class _DeferredTrainer:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            assert kwargs["defer_frozen_reference_identity"] is True
            assert "checkpoint_manager" in kwargs
            assert "checkpoint_store" not in kwargs
            assert "upload" not in kwargs

        def load(
            self,
            path: str,
            *,
            copy_mode: bool = False,
            copy_inventory_policy: Any = None,
        ) -> int:
            nonlocal load_calls
            assert copy_mode
            assert path == str(base_checkpoint)
            assert copy_inventory_policy.source_only_keys == ("llm2sound.weight",)
            load_calls += 1
            return 0

    monkeypatch.setattr(build_module, "build_backbone", lambda _config, _world: torch.nn.Linear(2, 1))
    monkeypatch.setattr(
        build_module,
        "build_optimizer",
        lambda model, _config: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(cosmos3_model_module, "Cosmos3FlowPolicy", _Policy)
    monkeypatch.setattr(diffusion_module, "DiffusionTrainer", _DeferredTrainer)
    rl = SimpleNamespace(
        seed=17,
        ref=SimpleNamespace(build_ref=True, update_every=0),
        optimizer=SimpleNamespace(grad_clip_norm=0.0, force_finite_grads=True),
        beta=1.0,
        grad_accum=1,
        max_steps=1,
        checkpoint_format="consolidated",
        save_async=False,
        keep_optim_last=0,
        keep_model_last=0,
        upload=SimpleNamespace(enabled=False),
        init_from=str(base_checkpoint),
        init_keys_to_skip_loading=["net.llm2sound.weight"],
        resume="off",
    )

    dpo_entrypoint.build_trainer(_entry_config(), rl, world_size=1, resume_checkpoint="")
    assert load_calls == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_seeds_random_init_rank_invariant_then_training_by_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen-ref construction is rank invariant, then each training shard owns its RNG stream."""
    from cosmos_framework.utils import misc
    from cosmos_framework.posttrain.config import build as build_module
    from cosmos_framework.posttrain.model import cosmos3 as cosmos3_model_module

    events: list[str] = []
    draws: list[tuple[float, float, torch.Tensor]] = []
    seed_calls: list[tuple[int, bool]] = []
    set_random_seed = misc.set_random_seed

    class _FrozenPolicy:
        def __init__(self, model: torch.nn.Module, build_ref: bool) -> None:
            assert build_ref
            self.net: torch.nn.Module = model
            self.net_ref: torch.nn.Module = copy.deepcopy(model)

        def reference_semantics_identity(self) -> dict[str, int | str]:
            return {"adapter": "seeded-frozen-policy-test", "version": 1}

    def _record_seed(seed: int, by_rank: bool = False) -> None:
        events.append(f"seed:{seed}:{by_rank}")
        seed_calls.append((seed, by_rank))
        set_random_seed(seed=seed, by_rank=by_rank)

    def _build_backbone(_config: Any, world_size: int) -> torch.nn.Module:
        events.append(f"backbone:{world_size}")
        draws.append((random.random(), float(np.random.random()), torch.rand(3)))
        return torch.nn.Linear(3, 2, bias=False)

    def _build_optimizer(model: torch.nn.Module, _config: Any) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=0.1)

    monkeypatch.setattr(misc, "set_random_seed", _record_seed)
    monkeypatch.setattr(build_module, "build_backbone", _build_backbone)
    monkeypatch.setattr(build_module, "build_optimizer", _build_optimizer)
    monkeypatch.setattr(cosmos3_model_module, "Cosmos3FlowPolicy", _FrozenPolicy)

    def _build(
        seed: int,
        world_size: int,
        ambient_seed: int,
        rank: int,
    ) -> tuple[str, tuple[float, float, torch.Tensor]]:
        random.seed(ambient_seed)
        np.random.seed(ambient_seed)
        torch.manual_seed(ambient_seed)
        monkeypatch.setattr(misc.distributed, "get_rank", lambda: rank)
        rl = SimpleNamespace(
            seed=seed,
            ref=SimpleNamespace(build_ref=True, update_every=0),
            optimizer=SimpleNamespace(grad_clip_norm=0.0, force_finite_grads=True),
            beta=1.0,
            grad_accum=1,
            max_steps=1,
            checkpoint_format="consolidated",
            save_async=False,
            keep_optim_last=0,
            keep_model_last=0,
            upload=SimpleNamespace(enabled=False),
            init_from="",
            resume="off",
        )
        trainer = dpo_entrypoint.build_trainer(
            _entry_config(),
            rl,
            world_size=world_size,
            resume_checkpoint="",
        )
        identity = str(trainer._trainer_state()["reference_identity"]["source"]["sha256"])
        continuation = (random.random(), float(np.random.random()), torch.rand(3))
        return identity, continuation

    first_identity, first_continuation = _build(seed=19, world_size=1, ambient_seed=100, rank=0)
    second_identity, second_continuation = _build(seed=19, world_size=4, ambient_seed=200, rank=3)
    different_identity, _different_continuation = _build(seed=20, world_size=4, ambient_seed=100, rank=3)

    assert seed_calls == [
        (19, False),
        (19, True),
        (19, False),
        (19, True),
        (20, False),
        (20, True),
    ]
    assert events == [
        "seed:19:False",
        "backbone:1",
        "seed:19:True",
        "seed:19:False",
        "backbone:4",
        "seed:19:True",
        "seed:20:False",
        "backbone:4",
        "seed:20:True",
    ]
    assert draws[0][0] == draws[1][0]
    assert draws[0][1] == draws[1][1]
    assert torch.equal(draws[0][2], draws[1][2])
    assert first_identity == second_identity
    assert draws[0][0] != draws[2][0]
    assert draws[0][1] != draws[2][1]
    assert not torch.equal(draws[0][2], draws[2][2])
    assert first_identity != different_identity
    assert first_continuation[0] != second_continuation[0]
    assert first_continuation[1] != second_continuation[1]
    assert not torch.equal(first_continuation[2], second_continuation[2])


@pytest.mark.L0
@pytest.mark.CPU
def test_entrypoint_handoff_commits_data_cursor_and_restores_next_item(tmp_path: Path) -> None:
    source = DatasetPreferenceEnv(_CursorDataset(7, seed=11), shuffle=True, seed=5)
    source.get_samples(3)
    data = WorkerGroup("data", role="env", body=source)

    model = ToyFlowModel(dim=8, seed=3)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
    )
    policy = WorkerGroup("policy", role="trainer", body=trainer)
    checkpoint = tmp_path / "checkpoints" / "iter_000000"

    save_policy_checkpoint(policy, str(checkpoint), data=data)
    saved = state_for_current_rank(load_component(str(checkpoint), "data"))
    assert saved is not None
    assert saved == source.state_dict()
    assert {"order", "next_index", "samples_emitted"}.isdisjoint(saved)

    expected = source.get_samples(1)[0].transitions[0].obs.representations["prompt"]
    restored = DatasetPreferenceEnv(_CursorDataset(7, seed=999), shuffle=True, seed=5)
    assert restored.load_state_dict(saved)
    actual = restored.get_samples(1)[0].transitions[0].obs.representations["prompt"]
    assert actual == expected


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("cursor_resume", ["auto", "off"])
def test_non_strict_checkpoint_omits_unsupported_data_cursor(
    tmp_path: Path,
    cursor_resume: str,
) -> None:
    data = WorkerGroup(
        "data",
        role="env",
        body=DatasetPreferenceEnv(_UnidentifiedDataset(), cursor_resume=cursor_resume),
    )
    model = ToyFlowModel(dim=8, seed=3)
    policy = WorkerGroup(
        "policy",
        role="trainer",
        body=DiffusionTrainer(
            model,
            make_objective(beta=100.0),
            torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        ),
    )
    checkpoint = tmp_path / "checkpoints" / "iter_000000"

    save_policy_checkpoint(policy, str(checkpoint), data=data)

    assert load_component(str(checkpoint), "data") is None
    assert load_component(str(checkpoint), "trainer") is not None


@pytest.mark.L0
@pytest.mark.CPU
def test_strict_checkpoint_rejects_unsupported_data_cursor(tmp_path: Path) -> None:
    data = WorkerGroup(
        "data",
        role="env",
        body=DatasetPreferenceEnv(_UnidentifiedDataset(), cursor_resume="strict"),
    )
    model = ToyFlowModel(dim=8, seed=3)
    policy = WorkerGroup(
        "policy",
        role="trainer",
        body=DiffusionTrainer(
            model,
            make_objective(beta=100.0),
            torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        ),
    )
    checkpoint = tmp_path / "checkpoints" / "iter_000000"

    with pytest.raises(CursorStateUnavailable, match="cursor_fingerprint"):
        save_policy_checkpoint(policy, str(checkpoint), data=data)

    assert not checkpoint.exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_pending_dpo_checkpoint_is_local_until_atomic_completion_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending save skips archive; archive receives the clear trainer before local commit."""
    from cosmos_framework.posttrain.control.validation import ValidationTrigger

    uploaded_trainer_states: list[dict[str, Any]] = []

    def _capture_upload(
        source: str,
        upload: dict[str, str],
        *,
        overrides: dict[str, str] | None = None,
        commit: dict[str, Any] | None = None,
    ) -> None:
        del upload
        assert overrides is not None and commit is not None
        uploaded_trainer_states.append(torch.load(overrides["trainer.pt"], map_location="cpu", weights_only=False))
        (Path(source) / ".uploaded").write_text("archive\n", encoding="utf-8")

    monkeypatch.setattr(
        "cosmos_framework.posttrain.checkpoint.manager.archive_checkpoint",
        _capture_upload,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.object_store.upload_destination_identity",
        lambda _upload, iter_name: {
            "kind": "object_store",
            "endpoint_authority": "storage.example.test",
            "bucket": "archive",
            "prefix": f"posttrain/checkpoints/{iter_name}",
        },
    )
    data = WorkerGroup(
        "data",
        role="env",
        body=DatasetPreferenceEnv(_CursorDataset(3, seed=11), cursor_resume="off"),
    )
    model = ToyFlowModel(dim=8, seed=3)
    trainer = DiffusionTrainer(
        model,
        make_objective(beta=100.0),
        torch.optim.AdamW(model.net.parameters(), lr=1e-2),
        upload={"bucket": "archive", "prefix": "posttrain/checkpoints", "credentials": "secret"},
        validation_source=DatasetPreferenceEnv(
            _CursorDataset(2, seed=17),
            shuffle=False,
            cursor_resume="off",
        ),
    )
    policy = WorkerGroup("policy", role="trainer", body=trainer)
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.mark_validation_pending(0, frozenset({ValidationTrigger.BEFORE_TRAINING}))

    save_policy_checkpoint(policy, str(checkpoint), data=data, publish=False)

    pending_state = load_component(str(checkpoint), "trainer")
    assert pending_state["validation"]["progress"]["pending_step"] == 0
    assert uploaded_trainer_states == []
    assert not (checkpoint / ".uploaded").exists()

    trainer.complete_validation(0)
    trainer.save_validation_progress(str(checkpoint))
    trainer.finalize_checkpoints()

    assert len(uploaded_trainer_states) == 1
    uploaded_progress = uploaded_trainer_states[0]["validation"]["progress"]
    assert uploaded_progress["pending_step"] is None
    assert uploaded_progress["pending_reasons"] == []
    assert load_component(str(checkpoint), "trainer")["validation"]["progress"] == uploaded_progress
    assert (checkpoint / ".uploaded").is_file()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "mismatch,cursor_resume,expected_error,expected_trainer_builds",
    [
        ("source", "auto", "dataset_len", 0),
        ("world_size", "auto", None, 1),
        ("world_size", "strict", "world_size", 0),
        ("missing", "auto", None, 1),
        ("missing", "strict", "requires a data checkpoint component", 0),
        ("empty", "auto", "world size must be positive", 0),
        ("empty", "strict", "world size must be positive", 0),
        ("malformed", "off", None, 1),
    ],
)
def test_setup_applies_cursor_resume_policy_before_building_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    cursor_resume: str,
    expected_error: str | None,
    expected_trainer_builds: int,
) -> None:
    saved_env = DatasetPreferenceEnv(_CursorDataset(7, seed=11), shuffle=True, seed=5)
    saved_env.get_samples(3)
    saved_state = saved_env.state_dict()
    assert saved_state is not None
    checkpoint = tmp_path / "checkpoints" / "iter_000002"
    checkpoint.mkdir(parents=True)
    # Provenance capture treats this as a real training-resume directory, so preserve the cursor
    # test's focus while providing the required non-data components.
    for name in ("model.pt", "optim.pt", "trainer.pt"):
        (checkpoint / name).write_bytes(b"checkpoint component\n")

    if mismatch == "source":
        component = {"version": 1, "world_size": 1, "states": {0: saved_state}}
        restored_env = DatasetPreferenceEnv(
            _CursorDataset(8, seed=11),
            shuffle=True,
            seed=5,
            cursor_resume=cursor_resume,
        )
    elif mismatch == "world_size":
        component = {"version": 1, "world_size": 2, "states": {0: saved_state, 1: saved_state}}
        restored_env = DatasetPreferenceEnv(
            _CursorDataset(7, seed=11),
            shuffle=True,
            seed=5,
            cursor_resume=cursor_resume,
        )
    elif mismatch == "malformed":
        component = {"broken": True}
        restored_env = DatasetPreferenceEnv(
            _CursorDataset(7, seed=11),
            cursor_resume=cursor_resume,
        )
    elif mismatch == "empty":
        component = {"version": 1, "world_size": 0, "states": {}}
        restored_env = DatasetPreferenceEnv(
            _CursorDataset(7, seed=11),
            cursor_resume=cursor_resume,
        )
    else:
        component = None
        restored_env = DatasetPreferenceEnv(
            _CursorDataset(7, seed=11),
            cursor_resume=cursor_resume,
        )
    if component is not None:
        torch.save(component, checkpoint / "data.pt")

    config = SimpleNamespace(
        job=SimpleNamespace(project="cosmos3_vfm", group="posttrain_dpo", name="cursor_setup_test"),
        rl=SimpleNamespace(
            resume=str(checkpoint),
            deterministic=True,
            data=SimpleNamespace(
                jsonl_path="unused",
                cursor_resume=cursor_resume,
            ),
        ),
    )
    trainer_builds = 0
    determinism_calls = 0

    def _load_config(config_path: str, opts: list[str]) -> Any:
        del config_path, opts
        return config

    def _instantiate(value: Any) -> Any:
        return value

    def _build_env(rl: Any) -> DatasetPreferenceEnv:
        assert determinism_calls == 1
        del rl
        return restored_env

    def _build_trainer(*args: Any, **kwargs: Any) -> None:
        nonlocal trainer_builds
        assert determinism_calls == 1
        del args, kwargs
        trainer_builds += 1

    def _enable_determinism() -> None:
        nonlocal determinism_calls
        determinism_calls += 1

    config_module = ModuleType("cosmos_framework.utils.config")
    lazy_config_module = ModuleType("cosmos_framework.utils.lazy_config")
    config_module.load_config = _load_config
    lazy_config_module.instantiate = _instantiate
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.lazy_config", lazy_config_module)
    monkeypatch.setattr(dpo_entrypoint, "build_env", _build_env)
    monkeypatch.setattr(dpo_entrypoint, "build_trainer", _build_trainer)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.determinism.enable_determinism",
        _enable_determinism,
    )
    setup = make_setup("unused.py", [], ["data", "policy"])
    if expected_error is None:
        setup()
    else:
        with pytest.raises(ValueError, match=expected_error):
            setup()
    assert trainer_builds == expected_trainer_builds
    assert determinism_calls == 1
