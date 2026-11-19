# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Cosmos model ownership and action-image resolution."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

resolve_cosmos_image = importlib.import_module("resolve_cosmos_image")
resolve_cosmos_model = importlib.import_module("resolve_cosmos_model")

COSMOS_SKILL = ROOT / "models" / "cosmos3-reasoner"
COSMOS_SKILL_INFO_PATH = COSMOS_SKILL / "references" / "skill_info.yaml"
COSMOS_SKILL_INFO = resolve_cosmos_model.load_yaml(COSMOS_SKILL_INFO_PATH)
COSMOS_FRAMEWORK_IMAGE = COSMOS_SKILL_INFO["container_image"]


def test_cosmos_nano_default_train_preserves_framework_image_contract():
    resolved = resolve_cosmos_image.resolve_image(ROOT, "nvidia/Cosmos3-Nano", "train")

    assert resolved["image"] == COSMOS_FRAMEWORK_IMAGE
    assert resolved["source"] == "model.container_image"
    assert resolved["resolved_from"] == "absolute"


def test_cosmos_nano_evaluate_uses_same_framework_image_contract():
    resolved = resolve_cosmos_image.resolve_image(ROOT, "nvidia/Cosmos3-Nano", "evaluate")

    assert resolved["image"] == COSMOS_FRAMEWORK_IMAGE


@pytest.mark.parametrize("resolver", [resolve_cosmos_image.resolve_image, resolve_cosmos_model.resolve_model])
def test_unsupported_action_cannot_fall_back_to_model_image(resolver):
    with pytest.raises(ValueError, match="Unsupported model action"):
        resolver(ROOT, "nvidia/Cosmos3-Nano", action="quantize")


def test_action_image_override_precedes_model_image(tmp_path):
    metadata = tmp_path / "models/example/references/skill_info.yaml"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        yaml.safe_dump(
            {
                "container_image": "example/train:1",
                "actions": {"train": {}, "evaluate": {"container_image": "example/eval:2"}},
            }
        )
    )
    assert resolve_cosmos_image.resolve_image(tmp_path, "example", "train")["image"] == "example/train:1"
    assert resolve_cosmos_image.resolve_image(tmp_path, "example", "evaluate")["image"] == "example/eval:2"


def test_cosmos_edge_supported_actions_resolve_image():
    for action in ("train", "evaluate", "inference", "inference_microservice"):
        resolved = resolve_cosmos_image.resolve_image(ROOT, "nvidia/Cosmos3-Edge", action)
        assert resolved["image"] == COSMOS_FRAMEWORK_IMAGE


def test_skill_info_images_are_stamped_from_versions_yaml():
    versions = yaml.safe_load((ROOT / "versions.yaml").read_text(encoding="utf-8"))

    assert COSMOS_FRAMEWORK_IMAGE.endswith(":local")
    assert versions["images"]["containers"]["cosmos_framework"] == COSMOS_FRAMEWORK_IMAGE
    contract = resolve_cosmos_model.load_yaml(COSMOS_SKILL / COSMOS_SKILL_INFO["training_contract"])
    assert "container_image" not in contract

    allowed_image_files = {
        COSMOS_SKILL_INFO_PATH,
        ROOT / "versions.yaml",
    }
    image_offenders: dict[str, list[str]] = {}
    for image in (COSMOS_FRAMEWORK_IMAGE,):
        offenders = []
        for path in ROOT.rglob("*"):
            if path in allowed_image_files or not path.is_file():
                continue
            if path.suffix not in {".json", ".md", ".py", ".toml", ".yaml", ".yml"}:
                continue
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if any(image in line and "versions-key:" not in line and "unpinned:" not in line for line in lines):
                offenders.append(str(path.relative_to(ROOT)))
        image_offenders[image] = offenders
    assert not any(image_offenders.values()), image_offenders


def test_cosmos_consumers_do_not_use_a_versions_image_key():
    legacy_key = "images.containers." + "cosmos_rl"
    offenders = []
    for root in (ROOT / "inference-service",):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".yaml", ".yml"}:
                if legacy_key in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders


def test_resolver_returns_existing_framework_skill_and_relocated_guide():
    resolved = resolve_cosmos_model.resolve_model(ROOT, "cosmos3-reasoner", action="train")
    assert resolved is not None
    assert Path(resolved["skill_path"]).is_file()
    assert Path(resolved["skill_path"]).parent.name == "cosmos3-post-training"
    assert Path(resolved["guide_path"]) == COSMOS_SKILL / "guide.md"
    assert Path(resolved["guide_path"]).is_file()
    assert resolved["container_image"] == COSMOS_FRAMEWORK_IMAGE
