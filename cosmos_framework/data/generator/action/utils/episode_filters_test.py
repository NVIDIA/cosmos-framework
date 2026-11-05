# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import runpy
import subprocess
from pathlib import Path

import pytest

from cosmos_framework.data.generator.action.utils.episode_filters import normalize_episode_filter_keys

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    "key",
    [
        "episode-1",
        "dataset/episode-1",
    ],
)
def test_filter_ranges_do_not_depend_on_storage_location(key: str) -> None:
    ranges = [[0, 12], [20, 35]]
    assert normalize_episode_filter_keys({key: ranges}) == {key: ranges}


def test_uri_filter_keys_preserve_nested_episode_ids() -> None:
    key = "s3://example-bucket/dataset/day-1/episode-1/recordings/MP4--s3://example-bucket/dataset/day-1/episode-1/trajectory.h5"
    assert normalize_episode_filter_keys({key: [[0, 12]]}, source_prefix="s3://example-bucket/dataset") == {
        "day-1/episode-1": [[0, 12]]
    }


def test_uri_filter_keys_require_an_operator_prefix() -> None:
    with pytest.raises(ValueError, match="source_prefix is required"):
        normalize_episode_filter_keys({"s3://example-bucket/episode-1": [[0, 12]]})


def test_conflicting_episode_ranges_are_refused() -> None:
    with pytest.raises(ValueError, match="Conflicting filter ranges"):
        normalize_episode_filter_keys(
            {"episode-1": [[0, 12]], "s3://example-bucket/dataset/episode-1": [[0, 24]]},
            source_prefix="s3://example-bucket/dataset",
        )


def test_matching_duplicate_episode_ranges_are_preserved_once() -> None:
    assert normalize_episode_filter_keys(
        {"s3://example-bucket/dataset/episode-1": [[0, 12]], "episode-1": [[0, 12]]},
        source_prefix="s3://example-bucket/dataset",
    ) == {"episode-1": [[0, 12]]}


@pytest.mark.parametrize("cookbook", [False, True])
@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize("success_as_root", [False, True])
def test_droid_launchers_forward_filter_configuration(
    tmp_path: Path, cookbook: bool, with_prefix: bool, success_as_root: bool
) -> None:
    """Exercise the real shells with staged inputs and a recording-only torchrun."""
    root = Path(__file__).resolve().parents[5]
    dataset = tmp_path / "droid_plus_lerobot_640x360_20260412"
    schema = runpy.run_path(
        str(root / "cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset_config.py")
    )
    assert dataset.name in schema["LEROBOT_ROOTS"]
    for folder in (dataset / "success" / "meta",):
        folder.mkdir(parents=True)
        (folder / "info.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    vae, filters, capture = (tmp_path / name for name in ("vae.pth", "filters.json", "args.txt"))
    vae.touch()
    filters.write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "torchrun": 'printf "%s\\n" "$IMAGINAIRE_OUTPUT_ROOT" "$@" > "$TEST_CAPTURE"',
        "python": 'printf "%s\\n" "$TEST_REPO_ROOT"',
    }.items():
        program = bin_dir / name
        program.write_text("#!/usr/bin/env bash\nset -eu\n" + body + "\n")
        program.chmod(0o755)
    key = "dataloader_train.dataloader.datasets.droid.dataset.filter_dict_source_prefix"
    env = os.environ.copy()
    for name in ("DROID_FILTER_SOURCE_PREFIX", "IMAGINAIRE_OUTPUT_ROOT", "NNODES", "NODE_RANK", "MASTER_ADDR"):
        env.pop(name, None)
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        DATASET_PATH=str(dataset),
        DROID_ROOT=str(dataset),
        BASE_CHECKPOINT_PATH=str(checkpoint),
        WAN_VAE_PATH=str(vae),
        FILTER_PATH=str(filters),
        OUTPUT_ROOT=str(tmp_path / "outputs"),
        TEST_REPO_ROOT=str(root),
        TEST_CAPTURE=str(capture),
        EXTRA_TAIL_OVERRIDES=f"{key}=s3://example-bucket/override job.wandb_mode=disabled",
    )
    if with_prefix:
        env["DROID_FILTER_SOURCE_PREFIX"] = "s3://example-bucket/data"
    if success_as_root:
        env["DATASET_PATH"] = env["DROID_ROOT"] = str(dataset / "success")
    directory = ".agents/skills/cookbooks/cosmos3/generator/action/finetune" if cookbook else "examples"
    result = subprocess.run(
        ["bash", str(root / directory / "launch_sft_action_policy_droid_nano.sh")],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if success_as_root:
        assert result.returncode != 0
        assert "success/meta/info.json" in result.stdout + result.stderr
        assert not capture.exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    args = capture.read_text().splitlines()
    assert args[0] == str(tmp_path / "outputs")
    forwarded = [arg for arg in args if arg.startswith(key + "=")]
    expected = [f"{key}=s3://example-bucket/data"] if with_prefix else []
    assert forwarded == expected + [f"{key}=s3://example-bucket/override"]
