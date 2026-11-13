# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real public SFT loop/checkpoint continuation, with an independent whole-pass control."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
)
from cosmos_framework.posttrain.checkpoint import CheckpointManager
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTConfig, SFTDatasetConfig, SFTValidationConfig
from cosmos_framework.posttrain.control.validation.continuation import (
    PROTOCOL,
    continuation_stop_reason,
    expected_batch,
    preflight_source_progress,
    topology_identity,
    validate_checkpoint_progress,
    validate_progress,
)
from cosmos_framework.posttrain.entrypoints.sft import loop
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv, build_sft_env
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
from cosmos_framework.posttrain.tests.test_roles_trainer_sft_validation import _assert_equal
from cosmos_framework.posttrain.utils.entry import resolve

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _objective(model: Any, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    x = torch.tensor([[float(row.transitions[0].targets.messages[0]["content"])] for row in batch])
    weights = torch.tensor([row.transitions[0].algo_extra.get("validation_weight", 1.0) for row in batch])
    # Uneven supervised counts make unweighted local-mean averaging detectably wrong.
    counts = (x.flatten().long() % 3 + 1).float() * weights
    noise = torch.rand_like(x) + random.random() + float(np.random.random())
    loss = (((model(x) - (2 * x + noise)).square().flatten()) * counts).sum() / counts.sum().clamp_min(1)
    return loss, {
        "sft_loss": loss.detach().item(),
        "sft_tokens": int(counts.sum().item()),
        "sft_sequences": int(weights.sum().item()),
        OBJECTIVE_WEIGHT_SUM: counts.sum().item(),
    }


class _StatefulValidationRows:
    """Map-style preprocessing with checkpointed local state and process RNG use."""

    def __init__(self):
        self.draws = 0

    def __len__(self):
        return 5

    def cursor_fingerprint(self):
        return "stateful-validation-correctness-fixture-v1"

    def cursor_state_dict(self):
        return {"draws": self.draws}

    def load_cursor_state_dict(self, state):
        if set(state) != {"draws"} or type(state["draws"]) is not int or state["draws"] < 0:
            raise ValueError("invalid test dataset state")
        self.draws = state["draws"]

    def __getitem__(self, index):
        self.draws += 1
        value = 11 + index + 0.01 * (self.draws + random.random() + float(np.random.random()) + torch.rand(()).item())
        return {"messages": [{"role": "user", "content": str(value)}, {"role": "assistant", "content": "0"}]}


def _runtime(
    data_dir: Path,
    output: Path,
    chunk: int | None,
    checkpoint_format: str = "consolidated",
    *,
    stateful_validation: bool = False,
):
    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state

    data_dir.mkdir(exist_ok=True, parents=True)
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    for name, values in (("train", range(1, 5)), ("val", range(11, 16))):
        content = (
            "\n".join(
                json.dumps(
                    {"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": str(i * 2)}]}
                )
                for i in values
            )
            + "\n"
        )
        path = data_dir / f"{name}.jsonl"
        if rank == 0 and not path.exists():
            path.write_text(content)
        if dist.is_initialized():
            dist.barrier()
        if path.exists():
            assert path.read_text() == content
        else:
            raise AssertionError("rank-zero input creation did not complete")
    validation = SFTValidationConfig(
        enabled=True,
        before_training=True,
        every_steps=0,
        save_checkpoint=True,
        seed=42,
        continuation_batches=chunk,
        data=SFTDatasetConfig(path=str(data_dir / "val.jsonl"), shuffle=False),
    )
    rl = SFTConfig(
        data=SFTDatasetConfig(path=str(data_dir / "train.jsonl")),
        validation=validation,
        max_steps=2,
        grad_accum=1,
        save_every=1,
        save_async=False,
        checkpoint_format=checkpoint_format,
        optimizer=OptimizerConfig(lr=0.01, grad_clip_norm=0.0),
    )
    config = SimpleNamespace(
        rl=rl, job=SimpleNamespace(wandb_mode="disabled", project="continuation_test", group="unit", name="unit")
    )
    data = build_sft_env(rl.data, 7, rank=rank, world_size=world)
    held = build_sft_env(validation.data, validation.seed, training=False, rank=rank, world_size=world)
    if stateful_validation:
        held = SFTDatasetEnv(
            _StatefulValidationRows(), seed=validation.seed, shuffle=False, training=False, rank=rank, world_size=world
        )
    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(17)
    net = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1))
    net[1].eval()  # Mixed child mode must survive every validation chunk.
    if dist.is_initialized():
        import os

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        shape = (2, 2) if os.environ["CONTINUATION_TEST_LAYOUT"] == "hsdp" else (4,)
        mesh = init_device_mesh("cpu", shape, mesh_dim_names=("replicate", "shard") if len(shape) == 2 else ("shard",))
        fully_shard(net, mesh=mesh)
    objective = Objective(
        _objective,
        Reduction.TOKEN_SUM,
        metric_adapter=MetricAdapter(
            internal_prefix="sft_",
            phases=frozenset({"train", "val"}),
            names={"sft_loss": "loss_avg", "sft_tokens": "valid_token_count", "sft_sequences": "valid_sequence_count"},
            ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
            weighted_by={"sft_loss": "sft_tokens"},
        ),
    )
    manager = CheckpointManager(checkpoint_format=checkpoint_format)

    def checked_evaluator(model, objective, batch):
        modes = [module.training for module in model.modules()]
        value = evaluate_sft_batch(model, objective, batch)
        assert [module.training for module in model.modules()] == modes
        return value

    trainer = LLMTrainer(
        net,
        objective,
        torch.optim.AdamW(net.parameters(), lr=0.01),
        grad_accum=1,
        checkpoint_manager=manager,
        validation_state=ValidationState(),
        objective_evaluator=checked_evaluator,
    )
    checkpoint = manager.resolve_resume("auto", str(output / "checkpoints"))
    if checkpoint:
        trainer.load(checkpoint)
        data.load_state_dict(manager.load_component(checkpoint, "data"))
    identity = {
        "enabled": True,
        "source_fingerprint": held.source_fingerprint,
        "objective_identity": dict(objective.identity),
        "batch_size": 1,
        "max_samples": None,
        "seed": 42,
        "schedule": {"before_training": True, "every_steps": 0, "train_total_steps": 2},
    }
    if chunk is not None:
        identity["continuation"] = {"protocol": PROTOCOL, "topology": topology_identity(net)}
    trainer.configure_validation(identity)
    consumed = []
    draw = held.get_validation_batch

    def observed(index, batch_size, max_samples):
        result = draw(index, batch_size, max_samples)
        consumed.append((trainer.iteration(), index, [row.id for row in result]))
        return result

    held.get_validation_batch = observed
    units = {
        "data": roles.env("data", body=data),
        "validation": roles.env("validation", body=held),
        "policy": roles.trainer("policy", body=trainer),
    }
    return config, units, trainer, consumed, collect_rng_state()


def _one_allocation(data_dir: str, output: str, chunk: int | None, checkpoint_format: str, invocation: int) -> None:
    """Spawn-compatible fixture: each invocation constructs a new trainer and loads public checkpoints."""
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    reports = []
    reporter = SimpleNamespace(
        start=lambda: None,
        finish=lambda: None,
        report=lambda metrics, *, step: reports.append((step, copy.deepcopy(metrics))),
    )
    reporting.make_reporters = lambda *a, **kw: [reporter]
    termination.TerminationWatch = lambda: SimpleNamespace(should_stop=lambda: None)
    config, units, trainer, consumed, _ = _runtime(Path(data_dir), Path(output), chunk, checkpoint_format)
    result = loop(units, config, output)
    trainer.finalize_checkpoints()
    Path(output).mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "result": result,
            "consumed": consumed,
            "reports": reports,
            "trainer": trainer._trainer_state(),
            "model": copy.deepcopy(trainer.model.state_dict()),
            "optimizer": copy.deepcopy(trainer.opt.state_dict()),
            "data": resolve(units["data"].state_dict()),
        },
        Path(output) / f"invocation-{invocation}.pt",
    )


@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_fresh_process_validation_resume_matches_whole_pass_and_next_updates(
    tmp_path: Path, checkpoint_format: str
) -> None:
    import multiprocessing

    def launch(output: Path, chunk: int | None, invocation: int):
        process = multiprocessing.get_context("spawn").Process(
            target=_one_allocation, args=(str(tmp_path / "data"), str(output), chunk, checkpoint_format, invocation)
        )
        process.start()
        process.join(60)
        assert not process.is_alive(), "bounded CPU validation fixture hung"
        assert process.exitcode == 0
        return torch.load(output / f"invocation-{invocation}.pt", map_location="cpu", weights_only=False)

    expected = launch(tmp_path / "whole", None, 0)
    records = []
    for invocation in range(5):
        current = launch(tmp_path / "chunks", 2, invocation)
        records.append(current)
        if "preempted" not in current["result"]:
            break
    assert len(records) == 5 and "preempted" not in current["result"]
    assert [row for record in records for row in record["consumed"]] == expected["consumed"]
    assert len(expected["consumed"]) == 10  # Five initial + five final, no replay of committed rows.
    frames = [row for record in records for row in record["reports"]]

    def scientific(rows):
        return [
            (step, {key: value for key, value in metrics.items() if not key.startswith("timer/")})
            for step, metrics in rows
        ]

    assert scientific(frames) == scientific(expected["reports"])
    assert records[0]["reports"] == records[1]["reports"] == []  # No partial result is a task metric.
    for key in ("model", "optimizer", "data"):
        _assert_equal(current[key], expected[key])
    _assert_equal(current["trainer"]["rng"], expected["trainer"]["rng"])
    assert current["trainer"]["validation"]["version"] == 3
    assert expected["trainer"]["validation"]["version"] == 2
    assert current["trainer"]["validation"]["progress"]["pending_step"] is None
    assert current["trainer"]["validation"]["publication"] is None
    cursor = current["trainer"]["validation"]["continuation"]
    assert cursor["next_batch"] == cursor["layout"]["batch_count"] == 5


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_continuation_batch_limit_is_strict(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SFTValidationConfig(enabled=True, save_checkpoint=True, continuation_batches=value)


@pytest.mark.parametrize(
    "changes", [{"enabled": False}, {"save_checkpoint": False}, {"checkpoint_components": ["model"]}]
)
def test_continuation_requires_durable_full_checkpoint(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="continuation requires"):
        SFTValidationConfig(**{"enabled": True, "save_checkpoint": True, "continuation_batches": 2, **changes})


def test_partial_checkpoint_records_exact_receipts_and_rejects_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(reporting, "make_reporters", lambda *a, **kw: [])
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    config, units, trainer, consumed, before = _runtime(tmp_path / "data", tmp_path / "run", 2)
    result = loop(units, config, str(tmp_path / "run"))
    assert result["final_step"] == 0 and "preempted" in result
    assert len(consumed) == 2 and not trainer.opt.state
    saved = trainer.checkpoint_manager.load_component(str(tmp_path / "run/checkpoints/iter_000000"), "trainer")
    _assert_equal(saved["rng"], before)
    progress = saved["validation"]["continuation"]
    identity = trainer.validation_state.identity
    validate_progress(progress, identity, iteration=0, pending_step=0)
    for change in [
        lambda v: v.update(next_batch=3),
        lambda v: v["ranks"][0].update(numerator=1.0),
        lambda v: v["ranks"][0]["batches"][1][0].update(id="wrong"),
        lambda v: v["ranks"][0]["batches"][1][0].update(weight=True),
        lambda v: v.update(identity_sha256="0" * 64),
        lambda v: v.update(model_step=1),
        lambda v: v.update(rng={}),
        lambda v: v["ranks"][0]["batch_metrics"].pop(),
        lambda v: v.update(unknown=1),
    ]:
        bad = copy.deepcopy(progress)
        change(bad)
        with pytest.raises((ValueError, RuntimeError)):
            validate_progress(bad, identity, iteration=0, pending_step=0)


def test_checkpoint_preflight_rejects_changed_contract_before_tensor_or_rng_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(reporting, "make_reporters", lambda *a, **kw: [])
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    output = tmp_path / "run"
    config, units, trainer, _, _ = _runtime(tmp_path / "data", output, 2)
    loop(units, config, str(output))
    checkpoint = str(output / "checkpoints/iter_000000")
    manager = trainer.checkpoint_manager
    saved = manager.load_component(checkpoint, "trainer")
    identity = copy.deepcopy(trainer.validation_state.identity)
    lifecycle = CheckpointLifecycle(manager, str(output / "checkpoints"), checkpoint)
    _, _, target, _, _ = _runtime(tmp_path / "data", tmp_path / "new_model", 2)
    target.checkpoint_manager = manager
    # A failed control preflight must precede both checkpoint tensor reads and load().
    with torch.no_grad():
        for parameter in target.model.parameters():
            parameter.fill_(-71)
    snapshot = copy.deepcopy(target.model.state_dict())
    optimizer = copy.deepcopy(target.opt.state_dict())
    before_rng = collect_rng_state()
    original_load = manager.load_component
    record = saved
    tensor_reads = []

    def component(path, name):
        if name == "trainer":
            return copy.deepcopy(record)
        if name in {"model", "optim"}:
            tensor_reads.append(name)
        return original_load(path, name)

    monkeypatch.setattr(manager, "load_component", component)
    changes = [
        lambda value, expected: value["validation"].update(version=2),
        lambda value, expected: value["validation"].update(version=3.0),
        lambda value, expected: value["validation"].pop("continuation"),
        lambda value, expected: value["validation"].update(publication={}),
        lambda value, expected: value["validation"]["continuation"].update(rng={}),
        lambda value, expected: value["validation"]["continuation"]["rng"][0].update(torch=torch.zeros(1)),
        lambda value, expected: value["validation"]["continuation"].update(ranks={}),
        lambda value, expected: expected.update(source_fingerprint="foreign-source"),
        lambda value, expected: expected.update(seed=99),
        lambda value, expected: expected.update(batch_size=2),
        lambda value, expected: expected.update(max_samples=3),
        lambda value, expected: expected.update(objective_identity={"wrong_objective": True}),
        lambda value, expected: expected["continuation"]["topology"].update(world_size=2),
        lambda value, expected: expected["continuation"]["topology"].update(parameter_devices=["cuda"]),
        lambda value, expected: expected.pop("continuation"),
    ]
    for change in changes:
        record, expected = copy.deepcopy(saved), copy.deepcopy(identity)
        change(record, expected)

        def before_load(state, data_state):
            assert data_state is not None
            validate_checkpoint_progress(state, expected, target._parse_update_state(state).optimizer_step)

        with pytest.raises((ValueError, RuntimeError)):
            lifecycle.load_trainer_and_component(target, "data", before_load=before_load)
        assert tensor_reads == []
        _assert_equal(target.model.state_dict(), snapshot)
        _assert_equal(target.opt.state_dict(), optimizer)
        _assert_equal(collect_rng_state(), before_rng)
    record = saved
    assert (
        lifecycle.load_trainer_and_component(
            target,
            "data",
            before_load=lambda state, _: validate_checkpoint_progress(state, identity, 0),
        )[0]
        == 0
    )
    assert tensor_reads == ["model", "optim"]
    _assert_equal(target.model.state_dict(), trainer.model.state_dict())


def test_source_preflight_rejects_cursor_corruption_and_restores_live_source_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(reporting, "make_reporters", lambda *a, **kw: [])
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    config, units, trainer, _, _ = _runtime(tmp_path / "data", tmp_path / "run", 2)
    loop(units, config, str(tmp_path / "run"))
    source = units["validation"]
    progress = trainer.validation_continuation
    snapshot = resolve(source.state_dict())
    rng = collect_rng_state()
    preflight_source_progress(progress, source, batch_size=1, max_samples=None)
    for change in [
        lambda value: value["layout"].update(rows=8),
        lambda value: value["ranks"][0].update(source_state={}),
        lambda value: value["ranks"][0]["source_state"].update(world_size=2),
        lambda value: value["ranks"][0]["source_state"]["states"][0].update(seed=100),
        lambda value: value["ranks"][0]["source_state"]["states"][0].update(dataset_fingerprint="foreign"),
        lambda value: value["ranks"][0]["source_state"]["states"][0].update(position=2),
        lambda value: value["ranks"][0]["source_state"]["states"][0].update(dataset_state={"unexpected": True}),
    ]:
        bad = copy.deepcopy(progress)
        change(bad)
        with pytest.raises(ValueError):
            preflight_source_progress(bad, source, batch_size=1, max_samples=None)
        _assert_equal(resolve(source.state_dict()), snapshot)
        _assert_equal(collect_rng_state(), rng)


def test_appending_validation_progress_cannot_rewrite_committed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(reporting, "make_reporters", lambda *a, **kw: [])
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    config, units, trainer, _, _ = _runtime(tmp_path / "data", tmp_path / "run", 2)
    loop(units, config, str(tmp_path / "run"))
    original = copy.deepcopy(trainer.validation_continuation)
    following = copy.deepcopy(original)
    following["next_batch"] += 1
    row = following["ranks"][0]
    row["batches"].append(expected_batch(following["layout"], 0, 2))
    row["batch_metrics"].append({"numerator": 0.0, "tokens": 0.0, "sequences": 0.0})
    # A forged total that is internally self-consistent must still not replace an
    # already committed contribution. The ordinary structural validator accepts it.
    forged = copy.deepcopy(following)
    forged["ranks"][0]["batch_metrics"][0]["numerator"] += 1.0
    forged["ranks"][0]["numerator"] = sum(item["numerator"] for item in forged["ranks"][0]["batch_metrics"])
    validate_progress(forged, trainer.validation_state.identity, iteration=0, pending_step=0)
    with pytest.raises(RuntimeError, match="rewrite"):
        trainer.record_validation_continuation(forged)
    _assert_equal(trainer.validation_continuation, original)
    with pytest.raises(RuntimeError, match="advance"):
        trainer.record_validation_continuation(original)
    trainer.record_validation_continuation(following)
    assert trainer.validation_continuation["next_batch"] == 3


def test_termination_notice_commits_one_complete_batch_and_allows_new_chunk_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(reporting, "make_reporters", lambda *a, **kw: [])
    # The first poll is the pre-validation boundary; the signal arrives after batch zero.
    reasons = iter([None, "signal SIGUSR1"])
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: next(reasons)))
    output = tmp_path / "run"
    config, units, trainer, consumed, before = _runtime(tmp_path / "data", output, 100)
    result = loop(units, config, str(output))
    assert result["preempted"] == "signal SIGUSR1" and result["final_step"] == 0
    assert [row[1] for row in consumed] == [0]
    assert trainer.validation_continuation["next_batch"] == 1
    _assert_equal(trainer._trainer_state()["rng"], before)
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    config, units, restored, consumed, _ = _runtime(tmp_path / "data", output, 3)
    loop(units, config, str(output))
    assert [row[1] for row in consumed] == [1, 2, 3]
    assert restored.validation_continuation["next_batch"] == 4


@pytest.mark.parametrize("value", [False, 0, 1, "", [], {}])
def test_invalid_termination_reason_fails(value: Any) -> None:
    with pytest.raises(ValueError, match="stop reason"):
        continuation_stop_reason(lambda: value)


def test_topology_identity_binds_device_kind_without_requiring_cuda() -> None:
    cpu = topology_identity(torch.nn.Linear(1, 1))
    # This is metadata-only synthetic admission, not a real GPU execution result.
    declared_cuda = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]))
    gpu = topology_identity(declared_cuda)
    assert cpu["world_size"] == gpu["world_size"] == 1
    assert cpu["layouts"] == gpu["layouts"] == []
    assert cpu["parameter_devices"] == ["cpu"]
    assert gpu["parameter_devices"] == ["cuda"]
    assert cpu != gpu


@pytest.mark.parametrize("commit_first", [False, True])
def test_control_commit_failure_resumes_only_durable_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commit_first: bool
) -> None:
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(reporting, "make_reporters", lambda *a, **kw: [])
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    output = tmp_path / "run"
    config, units, trainer, consumed, before = _runtime(tmp_path / "data", output, 2)
    replace = trainer.checkpoint_manager.replace_component

    def fail(path, name, state):
        if commit_first:
            replace(path, name, state)
        raise OSError("injected control commit interruption")

    monkeypatch.setattr(trainer.checkpoint_manager, "replace_component", fail)
    with pytest.raises(OSError, match="injected"):
        loop(units, config, str(output))
    assert [row[1] for row in consumed] == [0, 1]
    _assert_equal(trainer._trainer_state()["rng"], before)
    config2, units2, restored, consumed2, _ = _runtime(tmp_path / "data", output, 2)
    assert restored.iteration() == 0 and not restored.opt.state
    loop(units2, config2, str(output))
    assert [row[1] for row in consumed2] == ([2, 3] if commit_first else [0, 1])
    progress = restored.validation_continuation
    assert progress["next_batch"] == (4 if commit_first else 2)
    assert len(progress["ranks"][0]["batches"]) == progress["next_batch"]


def test_partial_progress_does_not_rewrite_model_optimizer_or_training_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(reporting, "make_reporters", lambda *a, **kw: [])
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    output = tmp_path / "run"
    config, units, _, _, _ = _runtime(tmp_path / "data", output, 2, "dcp")
    loop(units, config, str(output))
    checkpoint = output / "checkpoints/iter_000000"

    def digest():
        return {
            str(path.relative_to(checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in checkpoint.rglob("*")
            if path.is_file() and path.name != "trainer.pt"
        }

    before = digest()
    config, units, trainer, _, _ = _runtime(tmp_path / "data", output, 2, "dcp")
    loop(units, config, str(output))
    assert digest() == before and trainer.iteration() == 0 and not trainer.opt.state


def test_continuation_restores_stateful_preprocessing_and_rng_without_replaying_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    records = {}
    for label, chunk in [("whole", None), ("chunks", 2)]:
        consumed_all, reports_all = [], []
        for _ in range(5):
            reports = []
            monkeypatch.setattr(
                reporting,
                "make_reporters",
                lambda *a, **kw: [
                    SimpleNamespace(
                        start=lambda: None,
                        finish=lambda: None,
                        report=lambda metrics, *, step: reports.append((step, copy.deepcopy(metrics))),
                    )
                ],
            )
            config, units, trainer, consumed, _ = _runtime(
                tmp_path / "data", tmp_path / label, chunk, stateful_validation=True
            )
            result = loop(units, config, str(tmp_path / label))
            consumed_all.extend(consumed)
            reports_all.extend(reports)
            source_state = resolve(units["validation"].state_dict())
            assert source_state["states"][0]["dataset_state"] == {"draws": 0}
            if "preempted" not in result:
                break
            progress = trainer.validation_continuation
            saved = progress["ranks"][0]["source_state"]["states"][0]["dataset_state"]
            assert saved == {"draws": progress["next_batch"]}
        assert "preempted" not in result
        records[label] = {
            "model": copy.deepcopy(trainer.model.state_dict()),
            "optimizer": copy.deepcopy(trainer.opt.state_dict()),
            "rng": trainer._trainer_state()["rng"],
            "consumed": consumed_all,
            "reports": [
                (step, {key: value for key, value in metrics.items() if not key.startswith("timer/")})
                for step, metrics in reports_all
            ],
        }
    _assert_equal(records["chunks"], records["whole"])


def _distributed_worker(rank: int, rendezvous: str, directory: str, layout: str) -> None:
    import os

    import torch.distributed as dist

    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    os.environ["CONTINUATION_TEST_LAYOUT"] = layout
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        # TerminationWatch returns a local signal description on the signaled rank
        # and 'peer rank signaled' elsewhere. Both forms must converge, not fail.
        assert continuation_stop_reason(lambda: "signal SIGUSR1" if rank == 2 else None) == "signal SIGUSR1"
        assert (
            continuation_stop_reason(lambda: "signal SIGUSR1" if rank == 2 else "peer rank signaled")
            == "peer rank signaled"
        )
        root = Path(directory)
        termination.TerminationWatch = lambda: SimpleNamespace(should_stop=lambda: None)
        records = {}

        def local(value):
            if isinstance(value, torch.Tensor):
                return (
                    value.detach().to_local().cpu().clone()
                    if hasattr(value, "to_local")
                    else value.detach().cpu().clone()
                )
            if isinstance(value, dict):
                return {key: local(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(local(item) for item in value)
            return copy.deepcopy(value)

        for label, chunk in [("whole", None), ("chunks", 1)]:
            consumed_all, reports_all = [], []
            for invocation in range(3):
                reports = []
                reporting.make_reporters = lambda *a, **kw: [
                    SimpleNamespace(
                        start=lambda: None,
                        finish=lambda: None,
                        report=lambda metrics, *, step: reports.append((step, copy.deepcopy(metrics))),
                    )
                ]
                config, units, trainer, consumed, _ = _runtime(root / "data", root / label, chunk, "dcp")
                result = loop(units, config, str(root / label))
                trainer.finalize_checkpoints()
                consumed_all.extend(consumed)
                reports_all.extend(reports)
                if "preempted" not in result:
                    break
            assert "preempted" not in result
            records[label] = {
                "model": local(trainer.model.state_dict()),
                "optimizer": local(trainer.opt.state_dict()),
                "trainer": trainer._trainer_state(),
                "data": resolve(units["data"].state_dict()),
                "consumed": consumed_all,
                "reports": reports_all,
            }
        for key in ("model", "optimizer", "data", "consumed"):
            _assert_equal(records["chunks"][key], records["whole"][key])
        _assert_equal(records["chunks"]["trainer"]["rng"], records["whole"]["trainer"]["rng"])

        def scientific(rows):
            return [
                (step, {key: value for key, value in metrics.items() if not key.startswith("timer/")})
                for step, metrics in rows
            ]

        assert scientific(records["chunks"]["reports"]) == scientific(records["whole"]["reports"])
        progress = records["chunks"]["trainer"]["validation"]["continuation"]
        assert progress["next_batch"] == 2 and len(progress["ranks"]) == 4
        assert sum(row["sequences"] for row in progress["ranks"].values()) == 5
        assert (
            sum(item["weight"] == 0 for row in progress["ranks"].values() for batch in row["batches"] for item in batch)
            == 3
        )
        (root / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "status": "PASS_LOCAL_CPU_FOUR_RANK_CONTROL",
                    "layout": layout,
                    "rank": rank,
                    "validation_rows": 5,
                    "zero_weight_padding": 3,
                    "real_two_host_qualification": False,
                }
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["fsdp2", "hsdp"])
def test_local_four_rank_dcp_continuation_uneven_tail(tmp_path: Path, layout: str) -> None:
    torch.multiprocessing.spawn(
        _distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), layout), nprocs=4, join=True
    )
    rows = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(4)]
    assert {row["rank"] for row in rows} == {0, 1, 2, 3}
