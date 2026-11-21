# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU refusal checks for the explicit allocated WTS parity qualification."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools.native_qlora_qualification.test_validation_target import setup_target, write_scores

spec = importlib.util.spec_from_file_location("wts_lifecycle", Path(__file__).with_name("lifecycle.py"))
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


def score_pair(tmp_path):
    _, target, _ = setup_target(tmp_path)
    checkpoint = tmp_path / "iter_000004"
    paths = [tmp_path / name for name in ("native", "fresh")]
    for path in paths:
        rows, summary = write_scores(path, checkpoint, target)
        selection = json.loads((path / "selection.json").read_text())
        selection.update(
            ordered_ids=[r["id"] for r in rows],
            ordered_row_identities=[r["row_identity"] for r in rows],
            selected_content_sha256="1" * 64,
        )
        (path / "selection.json").write_text(json.dumps(selection))
        summary.update(
            selection_sha256=lifecycle.digest(selection),
            adapter_tensors_sha256="2" * 64,
            model_identity_sha256="3" * 64,
            processor_contract={"same": True},
            objective_identity={"ce": 1},
        )
        (path / "summary.json").write_text(json.dumps(summary))
    return paths, checkpoint


def test_native_fresh_all_rows_and_loaded_identity_are_exact(tmp_path):
    paths, checkpoint = score_pair(tmp_path)
    result = lifecycle.compare_wts_scores(*paths, checkpoint=checkpoint, expected_rows=6)
    assert result["passed"] and result["per_example_and_selection_exact"]
    assert len(result["source_sha256"]) == 8 and result["quality_qualified"] is False


@pytest.mark.parametrize("damage", ["row", "missing_row", "selection", "adapter", "checkpoint", "incomplete"])
def test_fresh_score_parity_rejects_numeric_selection_or_identity_damage(tmp_path, damage):
    paths, checkpoint = score_pair(tmp_path)
    path = paths[1]
    summary = json.loads((path / "summary.json").read_text())
    if damage in {"row", "missing_row"}:
        rows = [json.loads(line) for line in (path / "scores.jsonl").read_text().splitlines()]
        if damage == "row":
            rows[0]["native_sft_ce"] += 1e-6
        else:
            rows.pop()
        (path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        import hashlib

        summary["scores_sha256"] = hashlib.sha256((path / "scores.jsonl").read_bytes()).hexdigest()
    elif damage == "selection":
        selection = json.loads((path / "selection.json").read_text())
        selection["source_sha256"] = "f" * 64
        (path / "selection.json").write_text(json.dumps(selection))
        summary["selection_sha256"] = lifecycle.digest(selection)
    elif damage == "adapter":
        summary["adapter_tensors_sha256"] = "f" * 64
    elif damage == "checkpoint":
        summary["checkpoint"] = "/foreign/iter_000004"
    else:
        summary["complete"] = False
    (path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises((AssertionError, ValueError)):
        lifecycle.compare_wts_scores(*paths, checkpoint=checkpoint, expected_rows=6)


def state_pair(smoothing=0):
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig, SFTValidationConfig, SFTWTSConfig
    from cosmos_framework.posttrain.utils.identity import canonical_config_value

    expected = canonical_config_value(
        SFTWTSConfig(
            enabled=True,
            stop_on_target=True,
            data=SFTDatasetConfig(path="dev.jsonl", shuffle=False),
            source_sha256="1" * 64,
            manifest_path="manifest.json",
            manifest_sha256="2" * 64,
            expected_rows=4,
            steps=[2, 4],
            cache_bytes=1024,
            min_correct=4,
            max_token_ce=0,
        )
    )
    validation = canonical_config_value(SFTValidationConfig(enabled=True))
    validation["fields"].pop("wts")
    processor = {"model": "same-processor"}
    metadata = {
        "identity": {
            "base": "same",
            "processor_contract": processor,
            "training_contract": {
                "validation": validation,
                "build": "same",
                "objective": {"same": "ce", "training_source": "unchanged", "grad_accum": 8},
            },
        },
        "optimizer": "AdamW",
    }
    enabled = copy.deepcopy(metadata)
    enabled["identity"]["training_contract"]["validation"]["fields"]["wts"] = expected
    journal = {"version": 1, "observations": [{"step": 2, "target_met": False}]}
    state = {
        "rng": torch.tensor([1, 2]),
        "optimizer_step": 2,
        "validation": {
            "version": 2,
            "identity": {"enabled": True, "source": "ordinary"},
            "progress": {"pending_step": None},
            "publication": None,
        },
    }
    candidate = copy.deepcopy(state)
    candidate["validation"]["version"] = 3
    candidate["validation"]["wts"] = journal
    candidate["validation"]["identity"]["wts"] = {
        **{k: v for k, v in expected["fields"].items() if k not in {"enabled", "data", "manifest_path"}},
        "protocol": "cosmos.native-sft-wts.v1",
        "source_fingerprint": "3" * 64,
        "selection_sha256": "4" * 64,
        "processor": processor,
        "objective": {"same": "ce"},
    }
    if smoothing:
        from cosmos_framework.posttrain.algorithm.sft import make_objective
        from cosmos_framework.posttrain.tests.test_sft_development_evaluation import VideoPacker

        for record in (metadata, enabled):
            record["identity"]["training_contract"]["objective"] = {
                **make_objective(VideoPacker(), wts_answer_smoothing=smoothing).identity,
                "training_source": "unchanged",
                "grad_accum": 8,
            }
        candidate["validation"]["identity"]["wts"]["objective"] = make_objective(VideoPacker()).identity
    return expected, [metadata, enabled], journal, [state, candidate]


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "model",
        "optim",
        "rng",
        "cursor",
        "build",
        "journal",
        "processor",
        "objective",
        "ordinary_validation",
        "wts_config",
    ],
)
@pytest.mark.parametrize("smoothing", [0, 0.02])
def test_real_cpu_checkpoint_payloads_allow_only_explicit_wts_bookkeeping(tmp_path, monkeypatch, damage, smoothing):
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.tests import qlora_integration as integration

    expected, metadata, journal, states = state_pair(smoothing)
    paths = [tmp_path / name for name in ("disabled", "enabled")]
    if damage == "build":
        metadata[1]["identity"]["training_contract"]["build"] = "other"
    elif damage == "journal":
        states[1]["validation"]["wts"] = {"version": 1, "observations": []}
    elif damage == "processor":
        states[1]["validation"]["identity"]["wts"]["processor"] = {"other": True}
    elif damage == "objective":
        states[1]["validation"]["identity"]["wts"]["objective"] = {"other": True}
    elif damage == "ordinary_validation":
        states[1]["validation"]["identity"]["source"] = "other"
    elif damage == "wts_config":
        metadata[1]["identity"]["training_contract"]["validation"]["fields"]["wts"] = {
            **expected,
            "fields": {**expected["fields"], "cache_bytes": 0},
        }
    monkeypatch.setattr(
        adapter, "read_adapter_metadata", lambda path, *, require_resume: metadata[paths.index(Path(path))]
    )
    monkeypatch.setattr(integration, "inspect_checkpoint", lambda path, *, expected_step: {"step": expected_step})
    for i, path in enumerate(paths):
        path.mkdir()
        for component in ("model", "optim"):
            tensor = torch.tensor([1.0, 2.0])
            if i and damage == component:
                tensor[0] += 1e-6
            dcp.save({"parameter": tensor}, checkpoint_id=path / component)
        if i and damage == "rng":
            states[i]["rng"][0] = 9
        torch.save(states[i], path / "trainer.pt")
        torch.save({"cursor": 9 if i and damage == "cursor" else 8}, path / "data.pt")
    if damage:
        with pytest.raises((AssertionError, ValueError)):
            lifecycle.compare_wts_checkpoints(*paths, step=2, expected=expected, journal=journal)
    else:
        result = lifecycle.compare_wts_checkpoints(*paths, step=2, expected=expected, journal=journal)
        assert result["passed"] and result["remaining_controls_rng_and_data_exact"]
        assert result["observation_only"] and result["resume_compatible"] is False
        with pytest.raises(AssertionError):
            integration.compare_checkpoints(SimpleNamespace(reference=paths[0], candidate=paths[1], atol=0, rtol=0))


@pytest.mark.parametrize(
    "damage",
    [
        "epsilon_zero",
        "epsilon_bool",
        "epsilon_nan",
        "epsilon_one",
        "version_bool",
        "denominator",
        "scope",
        "extra",
        "reduction",
        "ce_exponent",
        "moe_lbl",
    ],
)
def test_hard_gold_identity_refuses_unrecognized_training_regularizer(damage):
    _, metadata, _, _ = state_pair(0.02)
    objective = metadata[0]["identity"]["training_contract"]["objective"]
    leaf = objective["wts_answer_smoothing"]
    if damage.startswith("epsilon_"):
        leaf["epsilon"] = {"zero": 0, "bool": True, "nan": float("nan"), "one": 1}[damage[8:]]
    elif damage == "version_bool":
        leaf["version"] = True
    elif damage in {"denominator", "scope", "extra"}:
        leaf[damage] = "other"
    elif damage == "reduction":
        objective[damage] = "different"
    else:
        objective[damage] = 0
    with pytest.raises(ValueError, match="training-only WTS smoothing"):
        lifecycle.hard_gold_evaluation_identity(objective)


def test_hard_gold_identity_preserves_all_other_objective_fields_and_source():
    _, metadata, _, states = state_pair(0.02)
    objective = metadata[0]["identity"]["training_contract"]["objective"]
    before = copy.deepcopy(objective)
    assert lifecycle.hard_gold_evaluation_identity(objective) == states[1]["validation"]["identity"]["wts"]["objective"]
    objective["future_loss_change"] = "must-not-be-normalized"
    assert lifecycle.hard_gold_evaluation_identity(objective)["future_loss_change"] == "must-not-be-normalized"
    objective.pop("future_loss_change")
    assert objective == before


@pytest.mark.parametrize("precision", ["qlora", "lora"])
def test_real_recipe_and_cli_wire_existing_recovery_then_wts_parity(tmp_path, monkeypatch, precision):
    """Exercise the real native config and scorer parsers; substitute only allocated stages."""
    from transformers import Qwen3VLConfig

    from cosmos_framework.posttrain.tests import qlora_integration as integration
    from cosmos_framework.posttrain.utils import adapter_source
    from tools.native_qlora_qualification import score

    model = tmp_path / "model"
    Qwen3VLConfig().save_pretrained(model)
    monkeypatch.setenv("SFT_MODEL_PATH", str(model))
    monkeypatch.setenv("COSMOS_TRAINING_CODE_REVISION", "a" * 40)
    monkeypatch.setenv("COSMOS_TRAINING_WHEEL_SHA256", "b" * 64)
    monkeypatch.setattr(adapter_source, "prepare_sft_adapter_source", lambda _: None)
    data = tmp_path / "development.jsonl"
    data.write_text("{}\n" * 4)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    output = tmp_path / "result"
    commands = []

    def inspection(path, *, expected_step):
        return {"path": str(path), "step": expected_step}

    monkeypatch.setattr(integration, "inspect_checkpoint", inspection)
    monkeypatch.setattr(lifecycle, "compare_next_metrics", lambda *a: {"passed": True})
    monkeypatch.setattr(lifecycle, "compare_wts_scores", lambda *a, **kw: {"passed": True, "rows": 4})
    scored = []
    monkeypatch.setattr(score, "run", lambda args: scored.append(args) or {"complete": True})

    class Stages:
        def __init__(self, out, env, timeout):
            self.outcomes = []
            (out / "stages").mkdir()

        def run(self, name, command, *, interrupted=False, interruption_receipt=None):
            commands.append((name, command))
            folder = output / "stages" / name
            folder.mkdir()
            self.outcomes.append({"name": name, "exit_code": 86 if interrupted else 0})
            if interrupted:
                root = output / "native/qualification/recovery" / name / "canary/checkpoints/iter_000002"
                interruption_receipt.write_text(
                    json.dumps(
                        {
                            "exit_code": 86,
                            "mode": name,
                            "checkpoint": inspection(root, expected_step=2),
                        }
                    )
                )
            if name == "reference":
                root = output / "native/qualification/recovery/reference/canary"
                (root / "wts/step_000004/attempt").mkdir(parents=True)
                records = [
                    {
                        "step": step,
                        "rows": 4,
                        "correct": 3,
                        "token_weighted_ce": 0.2,
                        "target_met": False,
                        "metrics": {},
                        "scores_sha256": "a" * 64,
                        "selected_content_sha256": "b" * 64,
                        "processed_inputs_sha256": "c" * 64,
                        "directory": f"wts/step_{step:06d}/attempt",
                    }
                    for step in (2, 4)
                ]
                (root / "wts/target-result.json").write_text(
                    json.dumps(
                        {
                            "protocol": "cosmos.native-sft-wts-target.v1",
                            "complete": True,
                            "final_step": 4,
                            "first_passing_step": None,
                            "horizon_censored": True,
                            "observations": records,
                        }
                    )
                )
            elif name == "fresh_wts_score":
                monkeypatch.setattr(sys, "argv", command[5:])
                score.main()
            elif name.startswith("wts_state_compare_"):
                settings = json.loads(Path(command[-2]).read_text())
                assert settings["expected"]["fields"]["stop_on_target"] is True
                step = int(command[-3])
                assert [r["step"] for r in settings["journal"]["observations"]] == list(range(2, step + 1, 2))
                Path(command[-1]).write_text(json.dumps({"passed": True, "optimizer_step": step}))
            return folder

    monkeypatch.setattr(lifecycle, "Stages", Stages)
    overrides = [
        "rl.validation.wts.enabled=true",
        "rl.validation.wts.stop_on_target=true",
        f"rl.validation.wts.data.path={data}",
        f"rl.validation.wts.manifest_path={manifest}",
        "rl.validation.wts.source_sha256=" + "a" * 64,
        "rl.validation.wts.manifest_sha256=" + "b" * 64,
        "rl.validation.wts.expected_rows=4",
        "rl.validation.wts.steps=[2,4]",
        "rl.validation.wts.cache_bytes=3221225472",
        "rl.validation.wts.min_correct=4",
        "rl.validation.wts.max_token_ce=0",
        "model.model_config.adapter.dropout=0.05",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lifecycle.py",
            "--config",
            f"cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_{precision}.py",
            "--data",
            str(data),
            "--output",
            str(output),
            "--qualify-wts",
            *[p for value in overrides for p in ("--override", value)],
        ],
    )
    lifecycle.main()
    assert [name for name, _ in commands] == [
        "reference",
        "boundary",
        "boundary_source_compare",
        "boundary_resumed",
        "boundary_compare_3",
        "boundary_compare_4",
        "accumulation",
        "accumulation_source_compare",
        "accumulation_resumed",
        "accumulation_compare_3",
        "accumulation_compare_4",
        "publish_portable",
        "probe_dcp",
        "probe_portable",
        "compare_reload",
        "fresh_wts_score",
        "without_wts",
        "wts_state_compare_2",
        "wts_state_compare_4",
    ]
    reference, disabled = dict(commands)["reference"], dict(commands)["without_wts"]
    ref_opts, disabled_opts = reference[reference.index("--") + 1 :], disabled[disabled.index("--") + 1 :]
    assert "model.model_config.adapter.dropout=0.05" in ref_opts
    assert "rl.validation.wts.enabled=true" in ref_opts and "rl.max_steps=4" in ref_opts
    assert [o for o in disabled_opts if o.startswith("rl.validation.wts.")] == ["rl.validation.wts.enabled=false"]
    assert scored[0].all_rows and scored[0].split == "development"
    assert scored[0].data == str(data) and scored[0].split_manifest == str(manifest)
    assert scored[0].preprocessing_cache_bytes == 3221225472
    assert json.loads((output / "completion.json").read_text())["wts_qualification"]["fresh_score_parity"]["passed"]
