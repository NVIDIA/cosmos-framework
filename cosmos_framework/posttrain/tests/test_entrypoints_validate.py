# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU tests for the entrypoints' manifest preflight: the flag parses, the real recipes exit
0, a broken manifest exits 1, and the DPO validate path imports neither torch nor ray."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.posttrain.entrypoints.dpo import _validate_and_exit
from cosmos_framework.posttrain.utils.entry import cli

_CONFIG = "cosmos_framework/posttrain/config/config_dpo.py"


@pytest.mark.L0
@pytest.mark.CPU
def test_cli_parses_validate_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["dpo", "--validate-manifest", "--", "experiment=x"])
    config_path, opts, validate = cli("dpo", default_config=_CONFIG)
    assert validate is True
    assert opts == ["experiment=x"]
    assert config_path == _CONFIG


@pytest.mark.L0
@pytest.mark.CPU
def test_cli_keeps_legacy_validate_flag_as_hidden_compatibility_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["dpo", "--validate"])
    _, _, validate = cli("dpo", default_config=_CONFIG)
    assert validate is True


@pytest.mark.L0
@pytest.mark.CPU
def test_cli_validate_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["dpo", "--", "experiment=x"])
    _, _, validate = cli("dpo", default_config=_CONFIG)
    assert validate is False


@pytest.mark.L0
@pytest.mark.CPU
def test_validate_exits_zero_for_valid_recipe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate_and_exit()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "bind_check: ok" in out
    assert '"schema_version": 1' in out  # the manifest document is persisted (schema-versioned) for the run log


@pytest.mark.L0
@pytest.mark.CPU
def test_validate_exits_one_on_invalid_manifest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # _validate_and_exit imports dpo_manifest at call time, so patching the module attribute takes effect.
    from cosmos_framework.posttrain.control import recipe
    from cosmos_framework.posttrain.control.manifest import Edge, Manifest, RoleDecl

    dangling = Manifest(roles=(RoleDecl("data", "dpo_env"),), edges=(Edge("data", "ghost"),))
    monkeypatch.setattr(recipe, "dpo_manifest", lambda role_specs: dangling)
    with pytest.raises(SystemExit) as exc:
        _validate_and_exit()
    assert exc.value.code == 1
    assert "error" in capsys.readouterr().out


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_main_validate_does_not_import_training_dependencies() -> None:
    """DPO's static manifest remains usable without importing the training stack."""
    repo_root = Path(__file__).resolve().parents[4]
    code = (
        "import sys;"
        "sys.argv = ['dpo', '--validate-manifest'];"
        "from cosmos_framework.posttrain.entrypoints.dpo import main;"
        "\ntry:\n main()\nexcept SystemExit as exc:\n print('EXIT', exc.code)\n"
        "print('TORCH' if 'torch' in sys.modules else 'notorch', 'RAY' if 'ray' in sys.modules else 'noray')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "EXIT 0" in out.stdout
    assert "bind_check: ok" in out.stdout
    assert "notorch noray" in out.stdout


@pytest.mark.L0
@pytest.mark.CPU
def test_runs_log_successful_bind_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import GRPOConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints import dpo, grpo

    dpo_log: list[str] = []
    monkeypatch.setattr(dpo, "info0", dpo_log.append)
    dpo_config = SimpleNamespace(grid=SimpleNamespace(backend="invalid", grouping="single"))
    with pytest.raises(ValueError, match="grid.backend"):
        dpo.run(config=dpo_config, config_path="x", opts=[], out_dir="/tmp/out")
    assert any("bind_check: ok" in line for line in dpo_log)

    grpo_log: list[str] = []
    monkeypatch.setattr(grpo, "info0", grpo_log.append)
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    grpo_config = SimpleNamespace(
        rl=GRPOConfig(),
        grid=SimpleNamespace(backend="invalid"),
        model_backend="hf",
        model_hf=HFModelConfig(),
        rollout=RolloutConfig(),
    )
    with pytest.raises(ValueError, match="grid.backend"):
        grpo.run(
            config=grpo_config,
            config_path="x",
            opts=[],
            out_dir="/tmp/out",
            build_objective_fn=grpo.build_objective,
        )
    assert any("bind_check: ok" in line for line in grpo_log)


@pytest.mark.L0
@pytest.mark.CPU
def test_run_preflights_the_manifest_before_any_cluster_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run()` bind-checks the same manifest and raises before binding, so a wiring mistake costs
    milliseconds instead of a placement group plus an 8B build. This is what makes the check guard the
    real launch path rather than only `--validate`."""
    from cosmos_framework.posttrain.control import placement, recipe
    from cosmos_framework.posttrain.control.manifest import Edge, Manifest, RoleDecl
    from cosmos_framework.posttrain.entrypoints import dpo

    dangling = Manifest(roles=(RoleDecl("data", "dpo_env"),), edges=(Edge("data", "ghost"),))
    monkeypatch.setattr(recipe, "dpo_manifest", lambda role_specs: dangling)
    logs: list[str] = []
    monkeypatch.setattr(dpo, "info0", logs.append)
    bound: list[object] = []
    monkeypatch.setattr(placement, "bind_device_groups", lambda *a, **k: bound.append(a))

    with pytest.raises(ValueError, match="names no role"):
        dpo.run(config=object(), config_path="x", opts=[], out_dir="/tmp/out")
    assert any("bind_check: 1 error(s)" in line for line in logs)
    assert not bound  # aborted before the cluster was touched


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validate_rejects_only_the_incompatible_pairing_before_source_io(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A structural mismatch is rejected before checking whether the configured path exists."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints.grpo import _validate_and_exit as grpo_validate

    with pytest.raises(SystemExit) as exc:
        grpo_validate(
            SimpleNamespace(
                rl=GRPOConfig(data=DatasetConfig(path="p.jsonl", prompt_column="q"), reward="exact_match"),
                model_backend="hf",
                model_hf=HFModelConfig(),
                rollout=RolloutConfig(),
            )
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"schema_version": 1' in out  # the manifest document is persisted for the run log
    assert "consumer needs 'reference'" in out
    assert "source_check" not in out


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validate_checks_jsonl_readability_after_structural_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints.grpo import _validate_and_exit as grpo_validate

    path = tmp_path / "prompts.jsonl"
    path.write_text('{"question": "2 + 2 ="}\n', encoding="utf-8")
    valid = GRPOConfig(data=DatasetConfig(path=str(path), prompt_column="question"), reward="nonempty")
    with pytest.raises(SystemExit) as exc:
        grpo_validate(
            SimpleNamespace(
                rl=valid,
                model_backend="hf",
                model_hf=HFModelConfig(),
                rollout=RolloutConfig(),
            )
        )
    assert exc.value.code == 0
    assert "source_check" not in capsys.readouterr().out

    missing = GRPOConfig(
        data=DatasetConfig(path=str(tmp_path / "missing.jsonl"), prompt_column="question"), reward="nonempty"
    )
    with pytest.raises(SystemExit) as exc:
        grpo_validate(
            SimpleNamespace(
                rl=missing,
                model_backend="hf",
                model_hf=HFModelConfig(),
                rollout=RolloutConfig(),
            )
        )
    assert exc.value.code == 1
    assert "source_check: error" in capsys.readouterr().out


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_run_preflights_before_any_cluster_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The same check guards the real launch, not only `--validate`. No patched manifest: `run()`
    derives it from the config it was handed, the path an operator actually takes."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig, RolloutConfig
    from cosmos_framework.posttrain.control import placement
    from cosmos_framework.posttrain.entrypoints import grpo

    bound: list[object] = []
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr(placement, "bind_device_groups", lambda *a, **k: bound.append(a))
    config = SimpleNamespace(
        rl=GRPOConfig(data=DatasetConfig(path="p.jsonl", prompt_column="q"), reward="exact_match"),
        model_backend="hf",
        model_hf=HFModelConfig(),
        rollout=RolloutConfig(),
    )
    with pytest.raises(ValueError, match="consumer needs 'reference'"):
        grpo.run(
            config=config,
            config_path="x",
            opts=[],
            out_dir="/tmp/out",
            build_objective_fn=grpo.build_objective,
        )
    assert not bound  # aborted before the cluster was touched

    config.rl = GRPOConfig(
        data=DatasetConfig(path=str(tmp_path / "missing.jsonl"), prompt_column="q"), reward="nonempty"
    )
    with pytest.raises(ValueError, match="not readable"):
        grpo.run(
            config=config,
            config_path="x",
            opts=[],
            out_dir="/tmp/out",
            build_objective_fn=grpo.build_objective,
        )
    assert not bound


@pytest.mark.L0
@pytest.mark.CPU
def test_validate_manifest_rejects_temperature_mismatch_before_cluster_work(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import GRPOConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints.grpo import _validate_and_exit as grpo_validate

    with pytest.raises(SystemExit) as exc:
        grpo_validate(
            SimpleNamespace(
                rl=GRPOConfig(temperature=0.9),
                model_backend="hf",
                model_hf=HFModelConfig(),
                rollout=RolloutConfig(temperature=1.0),
            )
        )
    assert exc.value.code == 1
    assert "rollout.temperature == rl.temperature" in capsys.readouterr().out


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_validate_rejects_temperature_mismatch_before_cluster_work(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.rl import OPDConfig, RolloutConfig, TeacherConfig
    from cosmos_framework.posttrain.entrypoints.opd import _validate_and_exit as opd_validate

    with pytest.raises(SystemExit) as exc:
        opd_validate(
            OPDConfig(
                distillation_mode="direct",
                temperature=0.5,
                teacher=TeacherConfig(scoring_temperature=0.5),
            ),
            RolloutConfig(temperature=0.5),
            SimpleNamespace(backend="local", gpu_placement="colocated"),
        )
    assert exc.value.code == 1
    assert "forced-sequence teacher path" in capsys.readouterr().out


@pytest.mark.L0
@pytest.mark.CPU
def test_validate_deps_are_torch_and_ray_free() -> None:
    # Every control module the --validate path ships, pinned BY NAME rather than transitively: a direct
    # torch/ray import added to any one of them must not be able to slip through a refactor of another.
    repo_root = Path(__file__).resolve().parents[4]
    modules = ("manifest", "capability", "capabilities_builtin", "recipe", "bind")
    code = (
        "import sys;"
        + "".join(f"import cosmos_framework.posttrain.control.{m};" for m in modules)
        + "print('RAY' if 'ray' in sys.modules else 'noray', 'TORCH' if 'torch' in sys.modules else 'notorch')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "noray notorch", out.stdout
