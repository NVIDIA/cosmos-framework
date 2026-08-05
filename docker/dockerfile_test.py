# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path


def test_runtime_tree_is_accessible_to_arbitrary_container_users() -> None:
    """Pyxis commonly starts the container with the submitting host UID."""

    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()
    assert "chmod a+rx /workspace /workspace/docker /workspace/docker/entrypoint.sh" in dockerfile
    assert "chmod -R a+rX /opt/tao /workspace/.venv /workspace/cosmos_framework" in dockerfile
    assert "test -x /workspace/docker/entrypoint.sh" in dockerfile
    assert "test -x /workspace/.venv/bin/python" in dockerfile
