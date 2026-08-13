# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
from pathlib import Path

import pytest

from cosmos_framework.inference.args import DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT_NAME, OmniSetupOverrides
from cosmos_framework.inference.common.args import (
    CheckpointConfig,
    CheckpointOverrides,
    CheckpointType,
    ConfigArgs,
    ConfigFileType,
    download_file,
)
from cosmos_framework.inference.common.config import deserialize_config_dict

CHECKPOINTS: dict[str, CheckpointConfig] = {
    DEFAULT_CHECKPOINT_NAME: DEFAULT_CHECKPOINT,
}

# Enough of a Cosmos3 'model' section for config resolution; not a loadable architecture.
MODEL_SECTION = {"_target": "omni_mot_model", "config": {"_type": "omni_mot_model_config"}}


def write_local_hf_checkpoint(root: Path, *, config: dict, repo_id: str | None = None) -> Path:
    """Write a minimal local Hugging Face checkpoint directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors").touch()
    if repo_id is not None:
        # Mirrors the layout '_write_modular_model_index' emits.
        (root / "modular_model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "Cosmos3OmniModularPipeline",
                    "transformer": [
                        "diffusers",
                        "Cosmos3OmniTransformer",
                        {
                            "pretrained_model_name_or_path": repo_id,
                            "subfolder": "transformer",
                            "type_hint": ["diffusers", "Cosmos3OmniTransformer"],
                            "variant": None,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    return root


def test_num_iterations_requires_benchmark() -> None:
    overrides = OmniSetupOverrides.model_construct(benchmark=False, num_iterations=2)

    with pytest.raises(ValueError, match="num_iterations > 1 requires benchmark=True"):
        overrides._build_setup()


def test_checkpoint_type_from_diffusers_layout(tmp_path: Path) -> None:
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (transformer_path / "config.json").write_text("{}", encoding="utf-8")
    (transformer_path / "diffusion_pytorch_model.safetensors.index.json").write_text("{}", encoding="utf-8")

    assert CheckpointType.from_path(tmp_path) == CheckpointType.HF


def test_download_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Disable the URL cache; this test asserts each download is independent.
    monkeypatch.delenv("COSMOS_DOWNLOAD_CACHE_DIR", raising=False)

    download_url_1 = (
        "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg"
    )
    file_size_1 = 279410

    download_url_2 = "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/bus_terminal.jpg"
    file_size_2 = 1283715

    # Download file
    download_path = Path(
        download_file(
            download_url_1,
            tmp_path,
            "robot_welding",
        )
    )
    assert download_path.stat().st_size == file_size_1
    meta_path = Path(f"{download_path}.meta")
    assert json.loads(meta_path.read_text()) == {
        "url": download_url_1,
    }
    cache_path = download_path.resolve()

    # Same file should be noop
    download_path = Path(download_file(str(download_path), tmp_path, "robot_welding"))
    assert download_path.resolve() == cache_path

    # Copy file
    copy_path = Path(download_file(str(download_path), tmp_path, "robot_welding_copy"))
    assert copy_path.stat().st_size == file_size_1
    assert copy_path.resolve() == cache_path

    # Re-download should be noop
    copy_path = Path(download_file(str(download_path), tmp_path, "robot_welding_copy"))
    assert copy_path.resolve() == cache_path

    # Force re-download
    os.remove(meta_path)
    download_path = Path(download_file(download_url_1, tmp_path, "robot_welding"))
    assert download_path.resolve() != cache_path
    assert download_path.stat().st_size == file_size_1

    # Different file should overwrite
    download_path = Path(
        download_file(
            download_url_2,
            tmp_path,
            "robot_welding",
        )
    )
    assert download_path.stat().st_size == file_size_2
    assert json.loads(Path(f"{download_path}.meta").read_text()) == {
        "url": download_url_2,
    }


def test_parse_checkpoint_path(tmp_path: Path):
    # Named checkpoint
    args = CheckpointOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "hf"
    assert args.experiment == ""

    # Local HF checkpoint: a copy of the registered repo resolves to its config.
    hf_path = write_local_hf_checkpoint(
        tmp_path / "hf", config={"model": MODEL_SECTION}, repo_id=DEFAULT_CHECKPOINT.hf.repository
    )
    args = CheckpointOverrides(
        checkpoint_path=str(hf_path),
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "hf"
    assert args.experiment == ""
    assert args.config_file == DEFAULT_CHECKPOINT.config_file

    # Local DCP checkpoint
    dcp_path = (
        tmp_path
        / "cosmos3_vfm/t2i_mot_0p6b_qwen3_vl_ablations/t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4/checkpoints/iter_000700000/model"
    )
    dcp_path.mkdir(parents=True, exist_ok=True)
    (dcp_path / ".metadata").touch()
    (dcp_path / "__0_0.distcp").touch()
    args = CheckpointOverrides(
        checkpoint_path=str(dcp_path),
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "dcp"
    assert (
        args.experiment
        == "t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4"
    )

    # S3 DCP checkpoint
    args = CheckpointOverrides(
        checkpoint_path=f"s3://bucket/cosmos3_vfm/t2i_mot_0p6b_qwen3_vl_ablations/t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4/checkpoints/iter_000700000/model",
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "dcp"
    assert (
        args.experiment
        == "t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4"
    )
    assert args.config_file == "cosmos_framework/configs/base/config.py"


def test_local_edge_checkpoint_resolves_registry_config(tmp_path: Path):
    """A local copy of nvidia/Cosmos3-Edge gets its architecture from the registry.

    The published Edge repo's top-level config.json is the reasoner HF config and
    carries no 'model' section, so the architecture is recovered from the repo id
    the checkpoint records for itself rather than from the directory.
    """
    checkpoints = OmniSetupOverrides.CHECKPOINTS
    edge = checkpoints["Cosmos3-Edge"]
    # A renamed copy: the directory name must not be what makes this work.
    checkpoint_dir = write_local_hf_checkpoint(
        tmp_path / "Cosmos-Edge",
        config={"architectures": ["Cosmos3EdgeForConditionalGeneration"], "model_type": "cosmos3_edge"},
        repo_id=edge.hf.repository,
    )

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(checkpoints=checkpoints)

    assert args.checkpoint_type == "hf"
    assert args.config_file == edge.config_file
    assert args.model_memory_bytes == edge.model_memory_bytes
    # The weights still come from the local directory, not a hub download.
    assert args.checkpoint_path == str(checkpoint_dir)
    # The resolved config really does describe the architecture.
    assert "ema" in args.load_model_config_dict()["config"]


def test_local_checkpoint_without_architecture_raises(tmp_path: Path):
    """An unidentifiable local HF dir fails fast instead of building a bogus config."""
    checkpoint_dir = write_local_hf_checkpoint(tmp_path / "mystery", config={"model_type": "cosmos3_edge"})

    with pytest.raises(ValueError, match="records no source repository"):
        CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(
            checkpoints=OmniSetupOverrides.CHECKPOINTS
        )


def test_registered_repository_wins_over_the_directory_config(tmp_path: Path):
    """A registered repository's config wins over the copy's own 'config.json'.

    A published repo's 'config.json' can be an older snapshot of the architecture
    than the registered config: nvidia/Cosmos3-Nano omits 'include_visual' and
    names the text-only processor, which made a local copy load a text-only
    Reasoner. The registered name and a local copy must resolve identically.
    """
    checkpoints = OmniSetupOverrides.CHECKPOINTS
    nano = checkpoints["Cosmos3-Nano"]
    checkpoint_dir = write_local_hf_checkpoint(
        tmp_path / "snapshot", config={"model": MODEL_SECTION}, repo_id=nano.hf.repository
    )

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(checkpoints=checkpoints)

    assert args.config_file == nano.config_file
    vlm_config = args.load_model_config_dict()["config"]["vlm_config"]
    assert vlm_config["model_instance"]["config"]["include_visual"] is True
    assert vlm_config["tokenizer"]["_target_"].endswith("build_processor_lazy")


def test_unregistered_local_checkpoint_uses_its_own_config(tmp_path: Path):
    """A checkpoint we do not publish keeps describing itself.

    Its own 'config.json' is the only description available, so it must keep
    working without '--config-file' (e.g. the ModelOpt FP8 Nano checkpoint, which
    ships from a subdirectory of an unregistered repository).
    """
    checkpoint_dir = write_local_hf_checkpoint(tmp_path / "self-described", config={"model": MODEL_SECTION})

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(
        checkpoints=OmniSetupOverrides.CHECKPOINTS
    )

    assert args.config_file == str(checkpoint_dir / "config.json")


def test_explicit_config_file_bypasses_resolution(tmp_path: Path):
    """An explicit '--config-file' is honoured for any local directory."""
    checkpoint_dir = write_local_hf_checkpoint(tmp_path / "self-described", config={"model": MODEL_SECTION})
    edge_config = OmniSetupOverrides.CHECKPOINTS["Cosmos3-Edge"].config_file

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir), config_file=edge_config).build_checkpoint(
        checkpoints=OmniSetupOverrides.CHECKPOINTS
    )

    assert args.config_file == edge_config
    assert args.checkpoint_path == str(checkpoint_dir)


def test_local_checkpoint_with_unregistered_repository_raises(tmp_path: Path):
    """An unknown source repository is named, and an empty registry still reads cleanly.

    'export_model' / 'export_train_config' pass an empty registry, so the message
    must not degrade into an empty list there.
    """
    checkpoint_dir = write_local_hf_checkpoint(
        tmp_path / "unknown", config={"model_type": "cosmos3_edge"}, repo_id="acme/Not-Registered"
    )

    with pytest.raises(ValueError, match=r"acme/Not-Registered.*Registered checkpoints: <none>"):
        CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(checkpoints={})


def test_load_model_config_dict_requires_model_section(tmp_path: Path):
    """'experiment_overrides' must not be able to synthesize a 'model' section.

    The overrides are dotlists rooted at 'model.'. Merging them into a config
    without a 'model' section used to produce a two-key stub that only failed
    much later, deep in model construction, as 'Missing key ema'.
    """
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "cosmos3_edge"}), encoding="utf-8")
    args = ConfigArgs(
        config_file=str(config_path),
        config_file_type=ConfigFileType.JSON,
        experiment="",
        experiment_overrides=["model.config.vlm_config.pretrained_weights.enabled=False"],
    )

    with pytest.raises(KeyError, match="no 'model' section"):
        args.load_model_config_dict()


def test_registry_model_configs_describe_an_architecture():
    """Every registered checkpoint's config file carries a usable 'model' section."""
    for name, checkpoint in OmniSetupOverrides.CHECKPOINTS.items():
        config_dict = deserialize_config_dict(Path(checkpoint.config_file))
        assert "model" in config_dict, f"{name}: {checkpoint.config_file} has no 'model' section"
        assert "ema" in config_dict["model"]["config"], f"{name}: {checkpoint.config_file} has no 'model.config.ema'"
