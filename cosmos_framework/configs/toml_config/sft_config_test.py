# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Loader tests for the free-form ``[custom]`` escape-hatch section of the SFT TOML."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

# Representative payload: scalars, a nested sub-table, and an array-of-tables.
_CUSTOM_PAYLOAD = {
    "scalar_int": 5,
    "scalar_str": "hello",
    "flag": True,
    "ratio": 0.3,
    "sampling": {"bug_ratio": 0.3, "nested": {"deep": 1}},
    "items": [
        {"path": "/data/a", "weight": 1.0},
        {"path": "/data/b", "weight": 2.0},
    ],
}


# --------------------------------------------------------------------------- #
# 1. pydantic schema validation                                               #
# --------------------------------------------------------------------------- #
class TestSchemaValidation:
    def test_custom_section_validates_arbitrary_nested_content(self) -> None:
        """Arbitrary nested [custom] content passes through untouched."""
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "custom": _CUSTOM_PAYLOAD,
        }
        cfg = SFTExperimentConfig.model_validate(raw)
        # The framework stores it verbatim — no coercion, no inner validation.
        assert cfg.custom == _CUSTOM_PAYLOAD

    def test_no_custom_section_defaults_empty(self) -> None:
        cfg = SFTExperimentConfig.model_validate({"job": {"task": "vfm", "experiment": "vision_sft_nano"}})
        assert cfg.custom == {}

    def test_unknown_top_level_key_raises(self) -> None:
        """Any unknown top-level section that is NOT `custom` still raises."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "bogus_section": {"x": 1},
                }
            )

    def test_unknown_key_inside_optimizer_raises(self) -> None:
        """A typo inside a KNOWN section is still a hard error (extra='forbid')."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "optimizer": {"lr": 1.0e-4, "not_a_real_key": 1},
                }
            )

    def test_custom_does_not_loosen_sibling_validation(self) -> None:
        """Presence of [custom] must not relax extra='forbid' elsewhere."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "custom": _CUSTOM_PAYLOAD,
                    "trainer": {"max_iter": 10, "typo_here": True},
                }
            )


# --------------------------------------------------------------------------- #
# 2. build_hydra_overrides must NOT emit [custom] as per-leaf overrides        #
# --------------------------------------------------------------------------- #
class TestBuildHydraOverrides:
    def test_custom_not_emitted_as_overrides(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "optimizer": {"lr": 1.0e-5},
            "custom": _CUSTOM_PAYLOAD,
        }
        overrides = build_hydra_overrides(raw)
        # Nothing under custom (verbatim or remapped) should appear.
        assert all("custom" not in o for o in overrides), overrides

    def test_other_keys_still_emitted(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "optimizer": {"lr": 1.0e-5},
            "custom": {"a": 1},
        }
        overrides = build_hydra_overrides(raw)
        assert "experiment=vision_sft_nano" in overrides
        assert any(o.startswith("optimizer.lr=") for o in overrides), overrides


# --------------------------------------------------------------------------- #
# 3. end-to-end load_experiment_from_toml on the shipped vision_sft_nano recipe #
# --------------------------------------------------------------------------- #
_BASE_TOML = """\
[job]
task         = "vfm"
experiment   = "vision_sft_nano"
project      = "cosmos3"
group        = "sft"
name         = "sft_config_custom_test"
wandb_mode   = "disabled"

[model.tokenizer]
vae_path = "${oc.env:WAN_VAE_PATH}"

[checkpoint]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
"""

_CUSTOM_TOML_BLOCK = """\

[custom]
scalar_int = 5
scalar_str = "hello"
flag       = true
ratio      = 0.3

[custom.sampling]
bug_ratio = 0.3

[custom.sampling.nested]
deep = 1

[[custom.items]]
path   = "/data/a"
weight = 1.0

[[custom.items]]
path   = "/data/b"
weight = 2.0
"""


def _load_or_skip(toml_path: Path):
    """Run the real loader, skipping if the training stack can't be imported."""
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    try:
        return load_experiment_from_toml(str(toml_path))
    except ImportError as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"training stack not importable here: {exc!r}")


@pytest.fixture
def _dummy_recipe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # vision_sft_nano interpolates these env vars into path strings at resolve time.
    monkeypatch.setenv("DATASET_PATH", "/tmp/dummy_dataset")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/dummy_vae.pth")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/dummy_ckpt")


class TestEndToEndLoader:
    def test_load_with_custom_section(self, tmp_path: Path, _dummy_recipe_env: None) -> None:
        toml_path = tmp_path / "with_custom.toml"
        toml_path.write_text(_BASE_TOML + _CUSTOM_TOML_BLOCK)

        config = _load_or_skip(toml_path)

        expected = {
            "scalar_int": 5,
            "scalar_str": "hello",
            "flag": True,
            "ratio": 0.3,
            "sampling": {"bug_ratio": 0.3, "nested": {"deep": 1}},
            "items": [
                {"path": "/data/a", "weight": 1.0},
                {"path": "/data/b", "weight": 2.0},
            ],
        }
        # Injected verbatim as a plain dict after Hydra resolution, so a project
        # can run MyProjectConfig.model_validate(config.custom) directly.
        assert config.custom == expected

    def test_load_without_custom_section_defaults_empty(self, tmp_path: Path, _dummy_recipe_env: None) -> None:
        toml_path = tmp_path / "no_custom.toml"
        toml_path.write_text(_BASE_TOML)

        config = _load_or_skip(toml_path)

        assert config.custom == {}
