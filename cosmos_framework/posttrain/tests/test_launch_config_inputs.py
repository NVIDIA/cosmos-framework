# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.posttrain.utils.launch_multinode import resolve_config

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("missing", ["container_image", "container_mounts", "account", "partition"])
def test_launch_requires_operator_selected_execution_settings(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    for key in (
        "POSTTRAIN_CONTAINER_IMAGE",
        "POSTTRAIN_CONTAINER_MOUNTS",
        "POSTTRAIN_SLURM_ACCOUNT",
        "SBATCH_ACCOUNT",
        "POSTTRAIN_SLURM_PARTITION",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = {
        "container_image": "example/runtime:latest",
        "container_mounts": "/data:/data",
        "account": "example-account",
        "partition": "example-partition",
    }
    settings.pop(missing)
    with pytest.raises(ValueError, match="required|no container image"):
        resolve_config(**settings)


def test_launch_accepts_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTTRAIN_CONTAINER_IMAGE", "example/runtime:latest")
    monkeypatch.setenv("POSTTRAIN_CONTAINER_MOUNTS", "/data:/data")
    monkeypatch.setenv("POSTTRAIN_SLURM_ACCOUNT", "example-account")
    monkeypatch.setenv("POSTTRAIN_SLURM_PARTITION", "example-partition")
    config = resolve_config()
    assert (config.account, config.partition, config.container_mounts) == (
        "example-account",
        "example-partition",
        "/data:/data",
    )
