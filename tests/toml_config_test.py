# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the structured-TOML → Hydra-override flow.

Focused on ``[job].upload_reproducible_setup`` (the OSS-friendly default-False
knob): it must land at the top-level ``config.upload_reproducible_setup`` field
on both tasks, and must be emitted as ``false`` even when the TOML omits it (so
it overrides the VLM base config's ``True``).
"""

from __future__ import annotations

import textwrap

import pytest

from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides


@pytest.mark.parametrize("task", ["vfm", "vlm"])
@pytest.mark.parametrize(("value", "expected"), [(True, "true"), (False, "false")])
def test_upload_setup_remapped_to_toplevel(task: str, value: bool, expected: str) -> None:
    """The knob is hoisted out of ``[job]`` to the top-level config field on
    both tasks — never emitted as ``job.upload_reproducible_setup``."""
    overrides = build_hydra_overrides(
        {"job": {"task": task, "experiment": "e", "upload_reproducible_setup": value}}
    )
    assert f"upload_reproducible_setup={expected}" in overrides
    assert not any(o.startswith("job.upload_reproducible_setup") for o in overrides)


def _capture_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path, toml_text: str) -> list[str]:
    """Run the real ``load_experiment_from_toml`` but stub ``load_config`` so we
    exercise only validate → inject → ``build_hydra_overrides`` (no Hydra
    compose, which would need the full model config tree)."""
    import cosmos_framework.utils.config as config_mod
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    captured: dict[str, object] = {}

    def fake_load_config(base_config_path: str, overrides: list[str]):
        captured["base"] = base_config_path
        captured["overrides"] = overrides
        return object()

    monkeypatch.setattr(config_mod, "load_config", fake_load_config)
    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text(textwrap.dedent(toml_text))
    load_experiment_from_toml(toml_path)
    return captured["overrides"]  # type: ignore[return-value]


@pytest.mark.parametrize("task", ["vfm", "vlm"])
def test_omitted_knob_forces_false(monkeypatch: pytest.MonkeyPatch, tmp_path, task: str) -> None:
    """Omitting the knob still emits ``upload_reproducible_setup=false`` — the
    whole point (the VLM base config defaults it ``True``)."""
    overrides = _capture_overrides(
        monkeypatch, tmp_path, f'[job]\ntask = "{task}"\nexperiment = "e"\n'
    )
    assert "upload_reproducible_setup=false" in overrides


@pytest.mark.parametrize("task", ["vfm", "vlm"])
def test_explicit_true_opts_in(monkeypatch: pytest.MonkeyPatch, tmp_path, task: str) -> None:
    """Explicitly setting ``true`` in the TOML opts the run back into S3 upload."""
    overrides = _capture_overrides(
        monkeypatch,
        tmp_path,
        f'[job]\ntask = "{task}"\nexperiment = "e"\nupload_reproducible_setup = true\n',
    )
    assert "upload_reproducible_setup=true" in overrides
    assert "upload_reproducible_setup=false" not in overrides


def test_unknown_key_still_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The strict schema (extra='forbid') still rejects typos — adding the new
    field didn't loosen validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _capture_overrides(
            monkeypatch,
            tmp_path,
            '[job]\ntask = "vfm"\nexperiment = "e"\nupload_reproducable_setup = false\n',
        )
