# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""A shared AOT package appears whole or not at all."""

from __future__ import annotations

import os

import pytest

from cosmos_framework.posttrain.utils.aot_cache import _atomic_compile_and_package


@pytest.mark.L0
@pytest.mark.CPU
def test_the_package_only_appears_once_it_is_complete(tmp_path):
    """_compile_variant treats os.path.exists(pkg) as proof the package is loadable while the
    compiler writes to that same path IN PLACE. Per-process that is fine; across 48 cells
    sharing one directory a sibling can load a half-written .pt2 on a cold cache."""
    final = tmp_path / "v.pt2"
    seen = []

    def _compiler(*, package_path):
        with open(package_path, "w") as f:
            f.write("half")
        seen.append(os.path.exists(final))  # must still be absent mid-compile
        with open(package_path, "a") as f:
            f.write("-whole")
        return package_path

    out = _atomic_compile_and_package(_compiler)(package_path=str(final))

    assert seen == [False], "the final path existed while the compiler was still writing"
    assert final.read_text() == "half-whole"
    assert out == str(final), "callers get the published path, not the temp one"


@pytest.mark.L0
@pytest.mark.CPU
def test_the_temp_path_keeps_the_pt2_extension(tmp_path):
    """torch asserts os.fspath(package_path).endswith('.pt2'). A suffix appended AFTER the
    extension makes every compile raise; _compile_variant swallows it and the run dies with
    'AOT compilation produced no loadable functions' -- which is what happened first time."""
    captured = {}

    def _compiler(*, package_path):
        assert package_path.endswith(".pt2"), package_path  # the real torch assert
        captured["tmp"] = package_path
        open(package_path, "w").close()
        return package_path

    final = tmp_path / "v.pt2"
    _atomic_compile_and_package(_compiler)(package_path=str(final))

    assert captured["tmp"] != str(final), "must not compile straight to the published path"
    assert str(os.getpid()) in captured["tmp"], "temp name must be per-process"


@pytest.mark.L0
@pytest.mark.CPU
def test_a_failed_compile_leaves_nothing_behind(tmp_path):
    """Nothing sweeps a persistent shared cache dir, so a partial file would accumulate run
    after run -- and could be mistaken for a package by a path-based existence check."""
    final = tmp_path / "v.pt2"

    def _compiler(*, package_path):
        open(package_path, "w").close()
        raise RuntimeError("inductor blew up")

    with pytest.raises(RuntimeError, match="inductor"):
        _atomic_compile_and_package(_compiler)(package_path=str(final))

    assert not final.exists()
    assert list(tmp_path.iterdir()) == [], f"leftovers: {list(tmp_path.iterdir())}"
