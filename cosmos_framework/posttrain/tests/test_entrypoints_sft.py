# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline SFT control-loop tests with a real optimizer and filesystem checkpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
)
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTConfig, SFTDatasetConfig, SFTValidationConfig
from cosmos_framework.posttrain.entrypoints.sft import evaluate, loop
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.env.sft import build_sft_env
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
from cosmos_framework.posttrain.schema import Trajectory

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _objective(model: Any, batch: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
    x = torch.tensor([[float(t.transitions[0].targets.messages[0]["content"])] for t in batch])  # [B,1]
    y = x * 2  # [B,1]
    weights = torch.tensor([t.transitions[0].algo_extra.get("validation_weight", 1) for t in batch])  # [B]
    count = weights.sum()  # []
    loss = (((model(x) - y).square().flatten()) * weights).sum() / count.clamp_min(1)  # []
    return loss, {
        "sft_loss": loss.detach().item(),
        "sft_tokens": count.item(),
        "sft_sequences": count.item(),
        OBJECTIVE_WEIGHT_SUM: count.item(),
    }


def _runtime(
    path: Path, *, count: int = 4, grad_clip_norm: float = 0.0, **rl: Any
) -> tuple[Any, dict[str, Any], LLMTrainer]:
    """A real SFT runtime; ``rl`` overrides the `SFTConfig` fields (a partial `checkpoint_components` plan is also
    handed to the trainer's checkpoint manager, as `entrypoints.sft.build_trainer` does with the lifecycle's);
    ``grad_clip_norm`` reaches both `OptimizerConfig.grad_clip_norm` and the trainer's `max_grad_norm`, as the
    entrypoint wires them."""
    from cosmos_framework.posttrain.checkpoint import CheckpointManager

    path.mkdir(parents=True, exist_ok=True)
    source = path / "data.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": str(i + 1)},
                        {"role": "assistant", "content": str((i + 1) * 2)},
                    ]
                }
            )
            for i in range(count)
        )
    )
    rl = {
        "max_steps": 4,
        "grad_accum": 2,
        "save_every": 2,
        "save_async": False,
        "optimizer": OptimizerConfig(lr=0.01, grad_clip_norm=grad_clip_norm),
        **rl,
    }
    config = SimpleNamespace(
        rl=SFTConfig(data=SFTDatasetConfig(path=str(source)), **rl),
        job=SimpleNamespace(wandb_mode="disabled", project="cosmos3_reasoner", group="sft_unit", name="unit"),
    )
    manager = (
        CheckpointManager(checkpoint_components=rl["checkpoint_components"]) if "checkpoint_components" in rl else None
    )
    data = build_sft_env(config.rl.data, 7)
    torch.manual_seed(17)
    model = torch.nn.Linear(1, 1)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = LLMTrainer(
        model,
        objective,
        optimizer,
        grad_accum=2,
        max_grad_norm=grad_clip_norm,
        save_async=False,
        checkpoint_manager=manager,
        validation_state=ValidationState(),
        objective_evaluator=evaluate_sft_batch,
    )
    trainer.configure_validation({"enabled": False})
    return config, {"data": roles.env("data", body=data), "policy": roles.trainer("policy", body=trainer)}, trainer


def _quiet_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *a, **kw: [])
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: None),
    )


class _Lines(logging.Handler):
    """The posttrain logger does not propagate to the root logger, so caplog misses it; collect its records."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_sft_console_headline_shows_the_step_size_regime() -> None:
    """The SFT console line must carry the gradient norm, the learning rate, the supervised-token
    count and the step time, and a learning rate of 1e-5 must be legible. The default GRPO/DPO headline keys
    (`utils/report.py::_CONSOLE_KEYS`) drop all four, which is what the parent passed."""
    import inspect

    from cosmos_framework.posttrain.entrypoints import sft as sft_entrypoint
    from cosmos_framework.posttrain.utils.log import logger as posttrain_logger
    from cosmos_framework.posttrain.utils.report import _CONSOLE_KEYS, ConsoleReporter

    keys = getattr(sft_entrypoint, "SFT_CONSOLE_KEYS", _CONSOLE_KEYS)  # without the SFT set, the default set
    frame = {
        "train/loss_avg": 1.2896,
        "train/grad_norm": 7.26,
        "train/learning_rate": 1e-5,
        "train/valid_token_count": 176703.0,
        "timer/e2e_step_s": 7.31,
        "val/loss_avg": 0.7271,
    }
    handler = _Lines()
    posttrain_logger.addHandler(handler)
    try:
        ConsoleReporter(total_steps=1401, headline_keys=keys).report(frame, step=1)
    finally:
        posttrain_logger.removeHandler(handler)
    line = handler.messages[-1]
    for column in (
        "train/loss_avg +1.2896",
        "train/grad_norm +7.2600",
        "train/learning_rate +1.000e-05",
        "train/valid_token_count +176703.0000",
        "timer/e2e_step_s +7.3100",
        "val/loss_avg +0.7271",
    ):
        assert column in line, f"console line lacks {column!r}: {line}"
    assert "headline_keys=SFT_CONSOLE_KEYS" in inspect.getsource(loop)


def test_loop_checkpoint_cursor_resume_matches_uninterrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet_loop(monkeypatch)
    config, units, full = _runtime(tmp_path / "source")
    complete = loop(units, config, str(tmp_path / "complete"))
    assert complete["final_step"] == 4
    expected = {key: value.clone() for key, value in full.model.state_dict().items()}
    expected_samples = units["data"].get_samples(4).result()

    config, units, partial = _runtime(tmp_path / "source")
    config.rl.max_steps = 2
    loop(units, config, str(tmp_path / "partial"))
    checkpoint = str(tmp_path / "partial" / "checkpoints" / "iter_000002")
    partial.finalize_checkpoints()
    config, units, restored = _runtime(tmp_path / "source")
    restored.load(checkpoint)
    from cosmos_framework.posttrain.checkpoint import CheckpointManager

    state = CheckpointManager().load_component(checkpoint, "data")
    assert state is not None
    units["data"].load_state_dict(state).result()
    config.rl.resume = checkpoint
    summary = loop(units, config, str(tmp_path / "partial"))
    assert summary["start"] == 2
    for key, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert units["data"].get_samples(4).result() == expected_samples


def _stop_after_two_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watch answers None, None, then "signal" at every later call (the loop asks once per step boundary)."""
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *a, **kw: [])
    calls = iter([None, None])
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: next(calls, "signal")),
    )


def test_loop_termination_emergency_save_honours_the_run_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SFT loop called `ensure_resumable` at a termination trigger, so
    `rl.emergency_checkpoint_components="run"` was silently ignored. Through `save_emergency`, "run" with a model-only
    cadence plan writes the model-only artifact (`checkpoint_artifacts/iter_*`) and no full checkpoint; the default
    "full" still writes the exact-resume checkpoint (`checkpoints/iter_*`) and no artifact."""
    _stop_after_two_calls(monkeypatch)
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):  # the model-only cadence plan, twice built
        config, units, _trainer = _runtime(
            tmp_path / "source", save_every=0, checkpoint_components=["model"], emergency_checkpoint_components="run"
        )
        summary = loop(units, config, str(tmp_path / "run"))
    step = summary["final_step"]
    assert summary["preempted"] == "signal" and 1 <= step < 4
    assert (tmp_path / "run" / "checkpoint_artifacts" / f"iter_{step:06d}").is_dir()
    assert not (tmp_path / "run" / "checkpoints").exists(), "the run plan must not escalate to a full checkpoint"

    _stop_after_two_calls(monkeypatch)
    config, units, _trainer = _runtime(tmp_path / "source_full", save_every=0)
    summary = loop(units, config, str(tmp_path / "full"))
    step = summary["final_step"]
    assert summary["preempted"] == "signal"
    assert (tmp_path / "full" / "checkpoints" / f"iter_{step:06d}").is_dir()
    assert not (tmp_path / "full" / "checkpoint_artifacts").exists()


def test_finite_evaluation_does_not_advance_source_or_rng(tmp_path: Path) -> None:
    from cosmos_framework.posttrain.control.validation.lifecycle import ValidationTrigger

    config, units, trainer = _runtime(tmp_path / "source", count=3)
    config.rl.validation = SFTValidationConfig(batch_size=2)
    validation = build_sft_env(config.rl.data, 7, training=False)
    units["validation"] = roles.env("validation", body=validation)
    state, rng = validation.state_dict(), torch.get_rng_state().clone()
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    report = evaluate(units, config.rl, 0, frozenset({ValidationTrigger.BEFORE_TRAINING}))
    assert report.sample_count == 3
    assert report.metrics["val/valid_token_count"] == 3
    assert validation.state_dict() == state
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[key])
    assert trainer.iteration() == 0


def test_real_qwen_multimodal_recipe_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise loading, actual processor/CE, local role binding, validation and DCP on a tiny model."""
    pytest.importorskip("transformers", minversion="5.8.1")
    from transformers import AutoModelForImageTextToText, Qwen3_5Config

    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.entrypoints.sft import run
    from cosmos_framework.posttrain.scripts.prepare_sft_example import prepare_example
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.init_local_process_group", lambda: None)
    _quiet_loop(monkeypatch)
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    model_path, data_path = tmp_path / "model", tmp_path / "data"
    tokenizer = local_tokenizer()
    processor = local_processor(tokenizer)
    model_config = Qwen3_5Config(
        text_config=dict(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        ),
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    AutoModelForImageTextToText.from_config(model_config).save_pretrained(model_path)
    processor.save_pretrained(model_path)
    prepare_example(data_path, train_samples=4, validation_samples=4)
    monkeypatch.setenv("SFT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SFT_DATA_ROOT", str(data_path))
    recipe = "cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py"
    opts = [
        "rl.max_steps=2",
        "rl.micro_batch_size=4",
        "rl.grad_accum=1",
        "rl.save_every=1",
        "rl.keep_optim_last=0",
        "rl.resume=off",
        "rl.master_dtype=float32",
        "rl.param_dtype=float32",
        "rl.validation.every_steps=1",
        "rl.validation.batch_size=4",
        "rl.processor_kwargs.images_kwargs.size.shortest_edge=16",
        "rl.processor_kwargs.images_kwargs.size.longest_edge=64",
        "rl.processor_kwargs.videos_kwargs.size.shortest_edge=16",
        "rl.processor_kwargs.videos_kwargs.size.longest_edge=64",
        "rl.processor_kwargs.videos_kwargs.num_frames=2",
    ]
    config = load_config(recipe, ["--", *opts])
    from cosmos_framework.posttrain.utils.entry import out_dir

    summary = run(config, recipe, opts, out_dir(config))
    assert summary["final_step"] == 2
    assert summary["final_loss"] is not None and summary["final_loss"] > 0
    assert list((tmp_path / "outputs").rglob("latest_checkpoint.txt"))
    checkpoint = next((tmp_path / "outputs").rglob("iter_000001"))
    resume_opts = [item for item in opts if not item.startswith("rl.resume=")]
    resume_opts += [f"rl.resume={checkpoint}", "job.name=resumed_example"]
    resumed_config = load_config(recipe, ["--", *resume_opts])
    resumed = run(resumed_config, recipe, resume_opts, out_dir(resumed_config))
    assert resumed["start"] == 1 and resumed["final_step"] == 2


def test_loop_writes_the_metrics_file_with_norm_lr_tokens_time_and_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every committed step lands as one JSON line in ``<run dir>/metrics.jsonl`` carrying the
    pre-clip gradient norm, the learning rate, the supervised-token count, the step time and the clip record
    (``train/grad_clipped``, ``train/grad_clip_coef``, ``train/grad_clip_fraction``). The parent wrote no file. The clip
    record follows ONE rule: ``train/grad_clip_coef`` is the factor
    ``clip_grad_norm_`` applied, ``min(1, grad_clip_norm / (norm + 1e-6))``, and ``train/grad_clipped`` is 1.0 exactly
    when that factor is below 1.0; every row below is checked against the rule, not against ``norm > grad_clip_norm``."""
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: None),
    )
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    config, units, _ = _runtime(tmp_path / "source", grad_clip_norm=1e-6)  # a tiny clip: every step is clipped
    run_dir = tmp_path / "run"
    summary = loop(units, config, str(run_dir))
    assert summary["final_step"] == 4
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    for row in rows:
        for key in (
            "time",
            "train/loss_avg",
            "train/grad_norm",
            "train/learning_rate",
            "train/valid_token_count",
            "timer/e2e_step_s",
            "train/grad_clipped",
            "train/grad_clip_coef",
            "train/grad_clip_fraction",
        ):
            assert key in row, (key, row)
        assert row["train/grad_clipped"] == 1.0 and row["train/grad_clip_fraction"] == 1.0
        assert 0 < row["train/grad_clip_coef"] <= 1e-6 / row["train/grad_norm"] + 1e-12
        assert row["train/grad_clip_coef"] == pytest.approx(min(1.0, 1e-6 / (row["train/grad_norm"] + 1e-6)))
        assert row["train/grad_clipped"] == float(row["train/grad_clip_coef"] < 1.0), "one rule: clipped <=> coef < 1"
        assert row["train/learning_rate"] == 0.01
    # The clip record is the TRAINER's, in the raw metrics of the applying microbatch, computed
    # from the norm the clip measured (on the parent the entrypoint re-derived it from the logged float after merging and
    # the trainer emitted no clip key); an accumulating microbatch and a trainer without a clip carry none.
    from cosmos_framework.posttrain.entrypoints import sft as entry
    from cosmos_framework.posttrain.utils.entry import resolve

    _, clip_units, _ = _runtime(tmp_path / "source_clip", grad_clip_norm=1e-6)
    accumulating = resolve(clip_units["policy"].train(clip_units["data"].get_samples(1)))
    assert accumulating["optim/update_outcome"] == "accumulating" and "train/grad_clipped" not in accumulating
    applied = resolve(clip_units["policy"].train(clip_units["data"].get_samples(1)))
    assert applied["optim/update_outcome"] == "applied"
    norm = applied["train/grad_norm"]
    assert applied["train/grad_clipped"] == 1.0
    assert applied["train/grad_clip_coef"] == pytest.approx(min(1.0, 1e-6 / (norm + 1e-6)))
    assert applied["train/grad_clipped"] == float(applied["train/grad_clip_coef"] < 1.0)
    assert not hasattr(entry, "add_clip_metrics")
    _, free_units, _ = _runtime(tmp_path / "source_noclip", grad_clip_norm=0.0)
    for _ in range(2):
        free = resolve(free_units["policy"].train(free_units["data"].get_samples(1)))
    assert free["optim/update_outcome"] == "applied" and "train/grad_clipped" not in free
    assert "train/grad_clip_coef" not in free


class _CountingDataset:
    """A map-style SFT source that counts row reads (``source_overlap`` indexes a dataset without a ``rows`` list)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.reads = 0

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.reads += 1
        return self._rows[index]


def _require_disjoint(training: Any, validation: Any) -> dict[str, Any]:
    """The one-process held-out contract over two SFT envs: ``data/datasets/sft.py::require_disjoint_datasets`` on their
    datasets (the census, then the two refusal rules). The setup's ``held_out_census`` runs the same two functions with
    the census on rank 0 only; the entrypoint keeps no one-process wrapper of its own."""
    from cosmos_framework.posttrain.data.datasets.sft import require_disjoint_datasets

    return require_disjoint_datasets(training.dataset, validation.dataset)


def _census_row(question: str, item: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": question[::-1]}],
        "meta": {"item": item, "pool": "TRAIN"},
    }


def _census_worker(rank: int, rendezvous: str, out_dir: str) -> None:
    """Two CPU ranks run the setup's held-out guard; each reports its row reads, its census and its refusal."""
    import torch
    import torch.distributed as dist

    from cosmos_framework.posttrain.entrypoints.sft import held_out_census

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.set_num_threads(1)
        training = _CountingDataset([_census_row(f"q{i}", f"item/{i}") for i in range(4)])
        held_out = _CountingDataset([_census_row(f"h{i}", f"held/{i}") for i in range(2)])
        census = held_out_census(SimpleNamespace(dataset=training), SimpleNamespace(dataset=held_out))
        reads: list[Any] = [None, None]
        dist.all_gather_object(reads, (training.reads + held_out.reads, census))
        overlapping = _CountingDataset([_census_row("q1", "item/1")])
        refusal: str | None = None
        try:
            held_out_census(SimpleNamespace(dataset=training), SimpleNamespace(dataset=overlapping))
        except (ValueError, RuntimeError) as error:
            refusal = f"{type(error).__name__}: {error}"
        refusals: list[Any] = [None, None]
        dist.all_gather_object(refusals, refusal)
        if rank == 0:
            Path(out_dir, "census.json").write_text(json.dumps({"reads": reads, "refusals": refusals}))
    finally:
        dist.destroy_process_group()


def test_held_out_census_runs_once_on_rank_zero_and_refuses_on_every_rank(tmp_path: Path) -> None:
    """FIXTURE: on the parent ``make_setup`` ran ``require_disjoint_sources`` through ``run_rank_converged`` on EVERY
    rank, so each rank walked both corpora: (6, 6) row reads per rank on the parent of the fix, MEASURED
    by a two-rank probe, not by this test, whose run on that parent stops at the ImportError of ``held_out_census``.
    ``held_out_census`` computes the census on rank 0 only and broadcasts it: rank 0 reads the 4 + 2 rows, rank 1 reads
    none, both hold the same census; an overlapping pair is refused on both ranks with the setup's message, collectively
    (``run_rank_converged`` turns the shared ``ValueError`` into one ``RuntimeError`` naming both ranks)."""
    import torch

    torch.multiprocessing.spawn(
        _census_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True
    )
    record = json.loads((tmp_path / "census.json").read_text())
    (reads0, census0), (reads1, census1) = record["reads"]
    assert (reads0, reads1) == (6, 0), "the census walks the corpora on rank 0 only"
    assert census0 == census1 and census0["shared_rows"] == 0 and census0["shared_items"] == 0
    assert census0["training_rows"] == 4 and census0["validation_rows"] == 2 and census0["validation_items"] == 2
    for refusal in record["refusals"]:
        assert refusal is not None and "1 shared row" in refusal and "1 shared item" in refusal, refusal
        assert "rank 0" in refusal and "rank 1" in refusal, refusal


def test_clip_record_counts_the_band_clip_grad_norm_scales(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``clip_grad_norm_`` scales every gradient by ``min(1, max_norm /
    (norm + 1e-6))``, so a norm inside the 1e-6 band below ``max_norm`` IS scaled (factor < 1) while the old record
    (``norm > max_grad_norm``) read 0.0 there. The record follows one rule: ``train/grad_clipped`` = 1.0 exactly when
    ``train/grad_clip_coef`` < 1.0. A norm 5e-7 below the cap is forced through the clip call; the applying microbatch
    records coef = 1 / (1 - 5e-7 + 1e-6) < 1 and clipped = 1.0 (the parent recorded 0.0)."""
    from cosmos_framework.posttrain.utils.entry import resolve

    _, units, _ = _runtime(tmp_path / "source", grad_clip_norm=1.0)
    forced = 1.0 - 5e-7

    def clip_grad_norm_(params: Any, max_norm: float, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert max_norm == 1.0
        return torch.tensor(forced)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_grad_norm_)
    for _ in range(2):
        metrics = resolve(units["policy"].train(units["data"].get_samples(1)))
    assert metrics["optim/update_outcome"] == "applied"
    assert metrics["train/grad_norm"] == pytest.approx(forced)
    coef = metrics["train/grad_clip_coef"]
    assert coef == pytest.approx(min(1.0, 1.0 / (forced + 1e-6))) and coef < 1.0
    assert metrics["train/grad_clipped"] == 1.0, "a norm the clip scaled is recorded as clipped (one rule: coef < 1)"


def test_setup_refuses_a_validation_source_overlapping_the_training_rows(tmp_path: Path) -> None:
    """The whole-source fingerprint guard passed a validation file that was a strict subset of the
    training file; the row-level guard refuses a shared content hash or a shared item id and accepts a disjoint set."""

    def write(path: Path, rows: list[dict[str, Any]]) -> SFTDatasetConfig:
        path.write_text("\n".join(json.dumps(row) for row in rows))
        return SFTDatasetConfig(path=str(path), shuffle=False)

    def row(question: str, item: str | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": question[::-1]}]
        }
        if item is not None:
            record["meta"] = {"item": item, "pool": "TRAIN"}
        return record

    training = build_sft_env(write(tmp_path / "train.jsonl", [row(f"q{i}", f"item/{i}") for i in range(8)]), 1)
    subset = build_sft_env(
        write(tmp_path / "subset.jsonl", [row("q0", "item/0"), row("q1", "item/1")]), 1, training=False
    )
    assert training.source_fingerprint != subset.source_fingerprint, "the whole-source check is what the parent had"
    with pytest.raises(ValueError, match="2 shared row"):
        _require_disjoint(training, subset)
    same_item = build_sft_env(write(tmp_path / "item.jsonl", [row("other words", "item/3")]), 1, training=False)
    with pytest.raises(ValueError, match="1 shared item"):
        _require_disjoint(training, same_item)
    same_content_other_meta = build_sft_env(
        write(tmp_path / "meta.jsonl", [{**row("q2"), "meta": {"item": "elsewhere/9", "pool": "UNUSED"}}]),
        1,
        training=False,
    )
    with pytest.raises(ValueError, match="1 shared row"):
        _require_disjoint(training, same_content_other_meta)
    held_out = build_sft_env(
        write(tmp_path / "heldout.jsonl", [row(f"h{i}", f"held/{i}") for i in range(3)]), 1, training=False
    )
    census = _require_disjoint(training, held_out)
    assert census["shared_rows"] == 0 and census["shared_items"] == 0
    assert census["training_rows"] == 8 and census["validation_rows"] == 3 and census["validation_items"] == 3


def test_disjointness_guard_requires_item_ids_and_hashes_media_bytes(tmp_path: Path) -> None:
    """The two bypasses of the row-level guard on the parent (cases 4
    and 5): a validation row WITHOUT ``meta.item`` that differed from a training row by one trailing space was ACCEPTED
    (its only identity was the content hash), and the same image row under a media root holding a COPY of the training
    image was ACCEPTED (``normalize_row`` rewrites the path to the resolved absolute path, so the path-based hash
    differed; a symlink resolved and was caught). Now validation rows must carry an item id when the training rows do,
    and media enter the row identity by file bytes. The seven refused shapes of that probe stay refused."""
    import os
    import shutil

    from PIL import Image

    def env(name: str, rows: list[Any], media_root: Path | None = None, training: bool = True) -> Any:
        path = tmp_path / name
        path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n")
        config = SFTDatasetConfig(path=str(path), media_root=str(media_root) if media_root else "", shuffle=False)
        return build_sft_env(config, 1, training=training)

    def refused(training: Any, validation: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _require_disjoint(training, validation)

    base_row = {
        "messages": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}],
        "meta": {"item": "pool/0", "pool": "TRAIN"},
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    other = {
        "messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
        "meta": {"item": "pool/1"},
    }
    train = env("train.jsonl", [base_row, other])
    reordered = {
        "tools": base_row["tools"],
        "meta": {"pool": "X", "item": "pool/0"},
        "messages": [{"content": "q0", "role": "user"}, {"content": "a0", "role": "assistant"}],
    }
    refused(train, env("v1.jsonl", [json.dumps(reordered, indent=2).replace("\n", " ")]), "1 shared row")
    refused(train, env("v2.jsonl", [{**base_row, "meta": {"item": "elsewhere/9"}}]), "1 shared row")
    refused(
        train,
        env(
            "v3.jsonl",
            [{**other, "messages": [{"role": "user", "content": "zzz"}, {"role": "assistant", "content": "yyy"}]}],
        ),
        "1 shared item",
    )
    # Bypass 4 (pre-fix): same content, no meta, one trailing space -> ACCEPTED there. Refused for the missing item.
    trailing = {
        "messages": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0 "}],
        "tools": base_row["tools"],
    }
    refused(
        train,
        env("v4.jsonl", [trailing]),
        "1 of 1 validation row\\(s\\) carry no meta.item while 2 of 2 training rows do",
    )
    refused(train, env("v4b.jsonl", [{"messages": base_row["messages"], "tools": base_row["tools"]}]), "1 shared row")

    for root in ("mrA", "mrB"):
        (tmp_path / root / "media" / "it").mkdir(parents=True)
    Image.new("RGB", (32, 32), (10, 20, 30)).save(tmp_path / "mrA" / "media" / "it" / "img.png")
    shutil.copyfile(tmp_path / "mrA" / "media" / "it" / "img.png", tmp_path / "mrB" / "media" / "it" / "img.png")
    (tmp_path / "mrC" / "media").mkdir(parents=True)
    os.symlink(tmp_path / "mrA" / "media" / "it", tmp_path / "mrC" / "media" / "it")
    img_row = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": "media/it/img.png"}, {"type": "text", "text": "what"}],
            },
            {"role": "assistant", "content": "a"},
        ]
    }
    # Bypass 5 (pre-fix): the same image row, media COPIED under another root, no meta on either side (so the item
    # rule is vacuous and the bytes rule alone must catch it) -> ACCEPTED there.
    train_image = env("trainA.jsonl", [img_row], media_root=tmp_path / "mrA")
    refused(train_image, env("v5.jsonl", [img_row], media_root=tmp_path / "mrB", training=False), "1 shared row")
    refused(train_image, env("v5b.jsonl", [img_row], media_root=tmp_path / "mrA", training=False), "1 shared row")
    refused(train_image, env("v7.jsonl", [img_row], media_root=tmp_path / "mrC", training=False), "1 shared row")
    train_image_item = env(
        "trainA_item.jsonl", [{**img_row, "meta": {"item": "pool/img"}}], media_root=tmp_path / "mrA"
    )
    refused(
        train_image_item,
        env("v6.jsonl", [{**img_row, "meta": {"item": "pool/img"}}], media_root=tmp_path / "mrB", training=False),
        "1 shared item",
    )
    # A different image under the copied root is a different row; without items on the training side no item is required.
    Image.new("RGB", (32, 32), (99, 20, 30)).save(tmp_path / "mrB" / "media" / "it" / "other.png")
    other_image = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": "media/it/other.png"}, {"type": "text", "text": "what"}],
            },
            {"role": "assistant", "content": "a"},
        ]
    }
    census = _require_disjoint(train_image, env("v8.jsonl", [other_image], media_root=tmp_path / "mrB", training=False))
    assert (
        census["shared_rows"] == 0
        and census["training_rows_with_item"] == 0
        and census["validation_rows_without_item"] == 1
    )
    held_out = env(
        "heldout.jsonl",
        [
            {
                **other,
                "messages": [{"role": "user", "content": "h"}, {"role": "assistant", "content": "o"}],
                "meta": {"item": "held/0"},
            }
        ],
    )
    census = _require_disjoint(train, held_out)
    assert census["shared_rows"] == 0 and census["shared_items"] == 0
    assert census["training_rows_with_item"] == 2 and census["validation_rows_without_item"] == 0


def test_metrics_file_survives_a_relaunch_past_the_last_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run 1 trains steps 1-2 and saves at 2; run 2 resumes from
    ``iter_000002``, commits step 3 and ends before a save (``save_every=0``: the crash-past-the-last-save state); run 3
    relaunches from ``iter_000002`` again. The pre-fix append-only file then held steps ``[1, 2, 3, 3, 4]`` and
    ``train/grad_clip_fraction`` restarted at the relaunch. Now the resumed attempt drops the rows it re-emits
    (``[1, 2, 3, 4]``) and the clip counters continue from the kept rows: with steps 1-2 clipped and steps 3-4 not, the
    fraction reads 1.0, 1.0, 2/3, 2/4 instead of restarting at 0."""
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: None),
    )
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    from cosmos_framework.posttrain.checkpoint import CheckpointManager

    run = tmp_path / "run"

    def steps() -> list[int]:
        return [json.loads(line)["step"] for line in (run / "metrics.jsonl").read_text().splitlines()]

    def resumed(
        checkpoint: str, *, max_steps: int, save_every: int, grad_clip_norm: float
    ) -> tuple[Any, dict[str, Any], LLMTrainer]:
        config, units, trainer = _runtime(tmp_path / "source", grad_clip_norm=grad_clip_norm)
        trainer.load(checkpoint)
        state = CheckpointManager().load_component(checkpoint, "data")
        assert state is not None
        units["data"].load_state_dict(state).result()
        config.rl.resume = checkpoint
        config.rl.max_steps, config.rl.save_every = max_steps, save_every
        return config, units, trainer

    config, units, trainer = _runtime(tmp_path / "source", grad_clip_norm=1e-6)  # a tiny clip: steps 1-2 are clipped
    config.rl.max_steps, config.rl.save_every = 2, 2
    assert loop(units, config, str(run))["final_step"] == 2
    trainer.finalize_checkpoints()
    checkpoint = str(run / "checkpoints" / "iter_000002")
    assert steps() == [1, 2]
    config, units, trainer = resumed(checkpoint, max_steps=3, save_every=0, grad_clip_norm=1e9)  # step 3: not clipped
    assert loop(units, config, str(run))["final_step"] == 3
    trainer.finalize_checkpoints()
    assert steps() == [1, 2, 3] and sorted(p.name for p in (run / "checkpoints").iterdir()) == [
        "iter_000002",
        "latest_checkpoint.txt",
    ]
    config, units, trainer = resumed(checkpoint, max_steps=4, save_every=2, grad_clip_norm=1e9)  # the relaunch
    summary = loop(units, config, str(run))
    trainer.finalize_checkpoints()
    assert summary["start"] == 2 and summary["final_step"] == 4
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert [row["train/grad_clipped"] for row in rows] == [1.0, 1.0, 0.0, 0.0]
    assert [row["train/grad_clipped"] for row in rows] == [float(row["train/grad_clip_coef"] < 1.0) for row in rows]
    assert [row["train/grad_clip_fraction"] for row in rows] == [1.0, 1.0, 2 / 3, 2 / 4]
