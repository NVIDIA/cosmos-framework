# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T-N1.7 model config: horizon ordering, identity hashing, and import weight."""

from __future__ import annotations

import subprocess
import sys

import pytest

from cosmos_framework.posttrain.config.model import OXE_DROID_TAG, Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config


def _cfg(**overrides) -> Gr00tN1d7ModelConfig:
    """A minimally valid config; the modality maps must be non-empty."""
    base = {"camera_map": {"cam": "env/cam"}, "state_map": {"s": "env/s"}}
    base.update(overrides)
    return Gr00tN1d7ModelConfig(**base)


@pytest.mark.L0
@pytest.mark.CPU
def test_horizon_ordering_is_enforced() -> None:
    _cfg(execution_horizon=8, training_prefix=16, model_action_horizon=16)
    # Executing past the supervised prefix runs the policy on unconstrained action steps.
    with pytest.raises(ValueError, match="execution_horizon <= training_prefix"):
        _cfg(execution_horizon=17, training_prefix=16, model_action_horizon=16)
    with pytest.raises(ValueError, match="execution_horizon <= training_prefix"):
        _cfg(execution_horizon=8, training_prefix=32, model_action_horizon=16)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("field", ["model_action_horizon", "training_prefix", "execution_horizon", "denoising_steps"])
def test_positive_fields_rejected_at_zero(field: str) -> None:
    with pytest.raises(ValueError, match=f"gr00t.{field} must be >= 1"):
        _cfg(**{field: 0})


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_embodiment_tag_rejected() -> None:
    with pytest.raises(ValueError, match="embodiment_tag"):
        Gr00tN1d7ModelConfig(embodiment_tag="  ", camera_map={"c": "e"}, state_map={"s": "e"})


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_modality_maps_rejected() -> None:
    """An empty map would silently send the model an observation with no cameras or no state."""
    with pytest.raises(ValueError, match="camera_map"):
        Gr00tN1d7ModelConfig(camera_map={}, state_map={"s": "e"})
    with pytest.raises(ValueError, match="state_map"):
        Gr00tN1d7ModelConfig(camera_map={"c": "e"}, state_map={})


@pytest.mark.L0
@pytest.mark.CPU
def test_identity_hash_ignores_mapping_key_order() -> None:
    """Two configs that mean the same thing (a recipe vs a CLI dotlist) must hash the same."""
    a = Gr00tN1d7ModelConfig(camera_map={"x": "env/x", "y": "env/y"}, state_map={"s": "env/s"})
    b = Gr00tN1d7ModelConfig(camera_map={"y": "env/y", "x": "env/x"}, state_map={"s": "env/s"})
    assert a.camera_map == b.camera_map
    assert a.identity_hash() == b.identity_hash()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "override",
    [
        {"execution_horizon": 4},
        {"embodiment_tag": "NEW_EMBODIMENT"},
        {"denoising_steps": 8},
        {"camera_map": {"cam": "env/other"}},
        {"state_map": {"s": "env/other"}},
        {"language_key": "annotation.other"},
        {"revision": "deadbeef"},
    ],
)
def test_identity_hash_is_sensitive_to_every_shared_field(override: dict) -> None:
    """Each field is one the trainer and rollout must agree on; a difference must change the hash."""
    base = _cfg()
    other = _cfg(**override)
    assert base.identity_hash() != other.identity_hash(), override


@pytest.mark.L0
@pytest.mark.CPU
def test_recipe_builds_and_rejects_unknown_fields() -> None:
    config = make_model_config()
    assert config.embodiment_tag == OXE_DROID_TAG
    assert config.execution_horizon <= config.training_prefix <= config.model_action_horizon
    with pytest.raises(AttributeError, match="unknown gr00t model config field"):
        make_model_config(nonexistent_field=1)


@pytest.mark.L0
@pytest.mark.CPU
def test_recipe_overrides_are_revalidated() -> None:
    """An override that breaks the horizon ordering must still fail."""
    with pytest.raises(ValueError, match="execution_horizon <= training_prefix"):
        make_model_config(execution_horizon=999)


@pytest.mark.L0
@pytest.mark.CPU
def test_config_model_module_does_not_import_isaac_gr00t() -> None:
    """`config/model.py` must stay dependency-light at module scope; checked in a SUBPROCESS because
    the parent session may already have `gr00t` imported.

    The module now carries every backend schema, so this also pins that reading a GR00T contract
    costs no model stack for the roles that only wanted the shape."""
    code = (
        "import sys;"
        "import cosmos_framework.posttrain.config.model as m;"
        "assert 'gr00t' not in sys.modules, sorted(k for k in sys.modules if k.startswith('gr00t'));"
        "assert m.Gr00tN1d7ModelConfig is not None"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"


@pytest.mark.L0
@pytest.mark.CPU
def test_resolved_checkpoint_path_pins_a_repo_id_to_the_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """No loader on the serving path takes a revision, so the engine is handed the revision's snapshot."""
    calls: list[tuple[str, str]] = []

    def fake_snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.append((repo_id, revision))
        return f"/hf/snapshots/{revision}"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    config = _cfg(checkpoint_path="nvidia/GR00T-N1.7-DROID", revision="deadbeef")
    assert config.resolved_checkpoint_path() == "/hf/snapshots/deadbeef"
    assert calls == [("nvidia/GR00T-N1.7-DROID", "deadbeef")]


@pytest.mark.L0
@pytest.mark.CPU
def test_resolved_checkpoint_path_is_the_path_itself_when_nothing_is_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Empty `revision` keeps its documented meaning: whatever the path holds, with no resolution at all."""
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_: pytest.fail("must not resolve"))
    assert _cfg(checkpoint_path="nvidia/GR00T-N1.7-DROID").resolved_checkpoint_path() == "nvidia/GR00T-N1.7-DROID"
    assert _cfg(checkpoint_path=str(tmp_path)).resolved_checkpoint_path() == str(tmp_path)


@pytest.mark.L0
@pytest.mark.CPU
def test_resolved_checkpoint_path_refuses_to_pin_a_directory(tmp_path) -> None:
    """A directory has no revision; serving it under a pinned hash would claim a pin it lacks."""
    with pytest.raises(ValueError, match="cannot pin the local directory"):
        _cfg(checkpoint_path=str(tmp_path), revision="deadbeef").resolved_checkpoint_path()
