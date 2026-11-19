# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Budget, source integrity and submission regressions for the GB200 launcher."""

import json
import shutil
from types import SimpleNamespace

import pytest
from failuregen.yam_training import launch
from failuregen.yam_training.contract import CONTRACT, EEF_CONTRACT, file_sha256
from failuregen.yam_training.launch import build_plan, training_overrides, validate_resolved_config, write_contract


@pytest.fixture
def training_inputs(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    stats = {"contract": CONTRACT, "q01": [-1] * 14, "q99": [1] * 14}
    (prepared / "action_stats.json").write_text(json.dumps(stats))
    manifest = {
        "status": "prepared",
        "contract": CONTRACT,
        "stats_sha256": file_sha256(prepared / "action_stats.json"),
        "window_counts": {"train": 13238, "val": 2176, "test": 2121},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = tmp_path / "init"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/.metadata").touch()
    (checkpoint / "model/config.json").write_text(json.dumps({"model": {"config": {"num_embodiment_domains": 33}}}))
    (checkpoint / "yam_checkpoint_migration.json").write_text(
        json.dumps({"contract": CONTRACT, "new_domain_count": 33, "existing_rows_preserved": True})
    )
    return prepared, checkpoint


def test_four_gpu_budget_and_smoke(training_inputs):
    prepared, checkpoint = training_inputs
    full = build_plan(prepared, checkpoint, samples_per_rank=4, epochs=4, smoke=False)
    assert (full["world_size"], full["global_batch"], full["max_iter"]) == (4, 16, 3310)
    smaller = build_plan(prepared, checkpoint, samples_per_rank=2, epochs=4, smoke=False)
    assert (smaller["global_batch"], smaller["max_iter"]) == (8, 6619)
    smoke = build_plan(prepared, checkpoint, samples_per_rank=4, epochs=4, smoke=True)
    assert smoke["max_iter"] == 10


def test_wrong_topology_and_incomplete_migration_rejected(training_inputs):
    prepared, checkpoint = training_inputs
    with pytest.raises(ValueError, match="four GPUs"):
        build_plan(prepared, checkpoint, samples_per_rank=4, epochs=4, smoke=False, world_size=8)
    (checkpoint / "model/.metadata").unlink()
    with pytest.raises(ValueError, match="complete"):
        build_plan(prepared, checkpoint, samples_per_rank=4, epochs=4, smoke=False)


def test_rejects_wrong_domain_config_and_modified_stats(training_inputs):
    prepared, checkpoint = training_inputs
    (checkpoint / "model/config.json").write_text(json.dumps({"model": {"config": {"num_embodiment_domains": 32}}}))
    with pytest.raises(ValueError, match="33 domains"):
        build_plan(prepared, checkpoint, samples_per_rank=4, epochs=4, smoke=False)
    (prepared / "action_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="normalizer changed"):
        build_plan(prepared, checkpoint, samples_per_rank=4, epochs=4, smoke=False)


def test_resume_contract_is_immutable(training_inputs, tmp_path):
    plan = build_plan(*training_inputs, samples_per_rank=4, epochs=4, smoke=False)
    destination = tmp_path / "run/contract.json"
    write_contract(destination, plan)
    write_contract(destination, plan)
    with pytest.raises(ValueError, match="Run inputs changed"):
        write_contract(destination, dict(plan, samples_per_rank=2))
    assert json.loads(destination.read_text()) == plan


@pytest.mark.parametrize("smoke", [False, True])
def test_sigma_diagnostic_removed_after_hydra_merges_defaults(tmp_path, smoke):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    # Both layers contribute settings for this callback in the Cosmos recipe.
    # Removing it only from the experiment would leave the basic default active.
    (tmp_path / "callbacks").mkdir()
    (tmp_path / "callbacks/basic.yaml").write_text(
        "# @package trainer.callbacks\n"
        "sigma_loss_analysis:\n"
        "  _target_: cosmos_framework.callbacks.sigma_loss_analysis.SigmaLossAnalysis\n"
        "  every_n: 5000\n"
        "  every_n_viz: 5000\n"
        "  save_s3: false\n"
        "compile_tokenizer:\n"
        "  enabled: true\n"
        "grad_clip:\n"
        "  clip_norm: 1.0\n"
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "defaults": [{"callbacks": "basic"}, "_self_"],
                "job": {"name": "default"},
                "checkpoint": {"save_iter": 100},
                "dataloader_val": None,
                "trainer": {
                    "run_validation": False,
                    "run_validation_on_start": False,
                    "callbacks": {
                        "sigma_loss_analysis": {"every_n": 500, "every_n_viz": 500},
                        "compile_tokenizer": {"enabled": False},
                    },
                },
            }
        ),
        tmp_path / "config.yaml",
    )
    overrides = training_overrides("yam_test", smoke=smoke)
    assert overrides[0] == "--"
    assert all("=" in value for value in overrides[1:])  # Cosmos TOML requires this.
    with hydra.initialize_config_dir(config_dir=str(tmp_path), version_base=None):
        baseline = hydra.compose(config_name="config")
        assert baseline.trainer.callbacks.sigma_loss_analysis.every_n == 500
        configured = hydra.compose(config_name="config", overrides=overrides[1:])
    assert "sigma_loss_analysis" not in configured.trainer.callbacks
    assert not configured.trainer.callbacks.compile_tokenizer.enabled
    assert configured.trainer.callbacks.grad_clip.clip_norm == 1.0
    assert configured.job.name == "yam_test"
    assert configured.checkpoint.save_iter == (5 if smoke else 100)
    assert not configured.trainer.run_validation
    assert not configured.trainer.run_validation_on_start
    OmegaConf.save(configured, tmp_path / "resolved.yaml")
    validate_resolved_config(tmp_path / "resolved.yaml")


@pytest.mark.parametrize("name", ["sigma_loss_analysis", "renamed_diagnostic"])
def test_resolved_config_rejects_sigma_diagnostic(tmp_path, name):
    path = tmp_path / "config.yaml"
    path.write_text(
        json.dumps(
            {
                "trainer": {
                    "callbacks": {
                        name: {
                            "_target_": "cosmos_framework.callbacks.sigma_loss_analysis.SigmaLossAnalysis",
                            "every_n": 500,
                            "every_n_viz": 500,
                            "save_s3": False,
                        }
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="must be removed"):
        validate_resolved_config(path)
    path.write_text(
        json.dumps(
            {
                "trainer": {
                    "run_validation": False,
                    "run_validation_on_start": False,
                    "callbacks": {"grad_clip": {"clip_norm": 1.0}},
                }
            }
        )
    )
    validate_resolved_config(path)


@pytest.mark.parametrize("setting", ["run_validation", "run_validation_on_start", "dataloader_val"])
def test_resolved_config_rejects_unimplemented_validation(tmp_path, setting):
    config = {
        "trainer": {"run_validation": False, "run_validation_on_start": False, "callbacks": {}},
        "dataloader_val": None,
    }
    if setting == "dataloader_val":
        config[setting] = {"dataset_name": "yam_val"}
    else:
        config["trainer"][setting] = True
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="held-out"):
        validate_resolved_config(path)


@pytest.mark.parametrize("invalid_setting", ["sigma_diagnostic", "validation"])
def test_train_orchestration_uses_four_workers_and_guards_resume(
    training_inputs, tmp_path, monkeypatch, invalid_setting
):
    prepared, checkpoint = training_inputs
    checkpoint.rename(tmp_path / "initialization")
    source = tmp_path / "original"
    source.mkdir()
    (source / "config.json").write_text("{}")
    vae = tmp_path / "vae.pth"
    vae.write_bytes(b"fixture")
    args = SimpleNamespace(
        work_dir=tmp_path,
        dataset=tmp_path / "dataset",
        splits=tmp_path / "splits",
        checkpoint=source,
        mode="train",
        samples_per_rank=4,
        epochs=4,
        run_name="unit",
        framework=tmp_path / "framework",
        recipe=tmp_path / "recipe",
        resume=False,
        vae=vae,
    )
    monkeypatch.setattr(launch, "validate_source", lambda *a, **kw: {"window_counts": {"train": 13238}})
    monkeypatch.setattr(launch, "source_fingerprint", lambda *a: {"source_sha256": "fixture"})
    calls = []
    run_dir = tmp_path / "training/cosmos3_yam/fd_id/unit"

    def fake_run(command, *, env=None):
        calls.append((list(map(str, command)), env))
        if "--dryrun" in command:
            (run_dir / "config.yaml").write_text(
                json.dumps(
                    {
                        "trainer": {
                            "run_validation": False,
                            "run_validation_on_start": False,
                            "callbacks": {"grad_clip": {"clip_norm": 1.0}},
                        }
                    }
                )
            )
        if "torch.distributed.run" in command:
            saved = run_dir / "checkpoints/iter_000003310/model"
            saved.mkdir(parents=True, exist_ok=True)
            (saved / ".metadata").touch()
            (run_dir / "checkpoints/latest_checkpoint.txt").write_text("iter_000003310")

    monkeypatch.setattr(launch, "run", fake_run)
    launch.execute(args)
    assert len(calls) == 2
    assert "--dryrun" in calls[0][0]
    for command, _ in calls:
        assert command[command.index("--") :] == training_overrides("unit", smoke=False)
    assert "--nproc-per-node=4" in calls[1][0]
    assert "failuregen.yam_training.train" in calls[1][0]
    assert calls[1][1]["YAM_MAX_ITER"] == "3310"
    assert calls[1][1]["YAM_SAMPLES_PER_RANK"] == "4"
    with pytest.raises(ValueError, match="already started"):
        launch.execute(args)
    args.resume = True
    launch.execute(args)
    args.samples_per_rank = 2
    with pytest.raises(ValueError, match="Run inputs changed"):
        launch.execute(args)

    # Invalid diagnostics/validation must prevent torchrun and the started marker.
    calls.clear()
    args.samples_per_rank = 4
    args.run_name = "bad_callbacks"
    args.resume = False
    run_dir = tmp_path / "training/cosmos3_yam/fd_id/bad_callbacks"

    def bad_dryrun(command, *, env=None):
        calls.append(command)
        assert "--dryrun" in command
        (run_dir / "config.yaml").write_text(
            json.dumps(
                {
                    "trainer": {
                        "run_validation": invalid_setting == "validation",
                        "run_validation_on_start": False,
                        "callbacks": {"sigma_loss_analysis": {"every_n": 500}}
                        if invalid_setting == "sigma_diagnostic"
                        else {},
                    }
                }
            )
        )

    monkeypatch.setattr(launch, "run", bad_dryrun)
    with pytest.raises(ValueError, match="must be removed|unimplemented"):
        launch.execute(args)
    assert len(calls) == 1
    assert not (run_dir / "training_started.json").exists()


def test_eef_launch_reuses_domain_16_without_expanding_checkpoint(training_inputs, tmp_path, monkeypatch):
    prepared, template = training_inputs
    (prepared / "action_stats.json").write_text(
        json.dumps({"contract": EEF_CONTRACT, "q01": [-1] * 20, "q99": [1] * 20})
    )
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest.update(contract=EEF_CONTRACT, stats_sha256=file_sha256(prepared / "action_stats.json"))
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    (template / "model/config.json").write_text(json.dumps({"model": {"config": {"num_embodiment_domains": 32}}}))
    (template / "yam_checkpoint_migration.json").write_text(
        json.dumps(
            {
                "contract": EEF_CONTRACT,
                "new_domain_count": 32,
                "existing_rows_preserved": True,
                "initialization": "reuse_existing_domain",
            }
        )
    )
    with pytest.raises(ValueError, match="representation"):
        build_plan(prepared, template, samples_per_rank=4, epochs=4, smoke=False)
    plan = build_plan(prepared, template, samples_per_rank=4, epochs=4, smoke=False, action_space="eef")
    assert plan["contract"]["domain_id"] == 16 and plan["max_iter"] == 3310
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    vae = tmp_path / "vae.pth"
    vae.write_bytes(b"fixture")
    args = SimpleNamespace(
        work_dir=tmp_path,
        dataset=tmp_path / "dataset",
        splits=tmp_path / "splits",
        checkpoint=source,
        mode="dryrun",
        samples_per_rank=4,
        epochs=4,
        run_name="eef",
        framework=tmp_path / "framework",
        recipe=tmp_path / "recipe",
        resume=False,
        vae=vae,
        action_space="eef",
        eef_root=None,
    )
    monkeypatch.setattr(launch, "validate_source", lambda *a, **kw: manifest)
    monkeypatch.setattr(launch, "source_fingerprint", lambda *a: {"source_sha256": "fixture"})
    calls = []

    def fake_run(command, *, env=None):
        calls.append((command, env))
        if "failuregen.yam_training.convert" in command:
            shutil.copytree(template, tmp_path / "initialization")
        if "--dryrun" in command:
            run_dir = tmp_path / "training/cosmos3_yam/fd_id/eef_dryrun"
            (run_dir / "config.yaml").write_text(
                json.dumps(
                    {
                        "trainer": {
                            "run_validation": False,
                            "run_validation_on_start": False,
                            "callbacks": {},
                        }
                    }
                )
            )

    monkeypatch.setattr(launch, "run", fake_run)
    launch.execute(args)
    assert "--reuse-yam-eef-domain" in calls[0][0]
    assert "--append-yam-domain" not in calls[0][0]
    assert calls[1][1]["YAM_ACTION_SPACE"] == "eef"
    assert "--dryrun" in calls[1][0]
    assert "~trainer.callbacks.sigma_loss_analysis=null" in calls[1][0]
