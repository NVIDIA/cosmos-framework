# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest
import yaml
from failuregen.release import materialize_run


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "auxiliary/tokenizer").mkdir(parents=True)
    (root / "auxiliary/vae.pth").write_bytes(b"vae")
    (root / "release.json").write_text(
        json.dumps(dict(format_version=1, vae="auxiliary/vae.pth", tokenizer="auxiliary/tokenizer"))
    )
    config = dict(
        model=dict(
            config=dict(
                tokenizer=dict(vae_path="auxiliary/vae.pth"),
                vlm_config=dict(
                    tokenizer=dict(pretrained_model_name="auxiliary/tokenizer"),
                    pretrained_weights=dict(enabled=True, backbone_path="s3://unused"),
                ),
            )
        )
    )
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "yam_training_contract.json").write_text('{"saved_contract": true}\n')
    return root


def test_release_resolves_assets_without_changing_frozen_inputs(bundle, tmp_path):
    original = (bundle / "config.yaml").read_bytes()
    runtime = materialize_run(bundle, tmp_path / "outputs")
    cfg = yaml.safe_load((runtime / "config.yaml").read_text())["model"]["config"]
    assert cfg["tokenizer"]["vae_path"] == str(bundle / "auxiliary/vae.pth")
    assert cfg["vlm_config"]["tokenizer"]["pretrained_model_name"] == str(bundle / "auxiliary/tokenizer")
    assert cfg["vlm_config"]["pretrained_weights"] == dict(enabled=False, backbone_path="")
    assert (bundle / "config.yaml").read_bytes() == original
    assert (runtime / "yam_training_contract.json").read_bytes() == (bundle / "yam_training_contract.json").read_bytes()
    assert materialize_run(bundle, tmp_path / "outputs") == runtime


def test_release_supports_hf_cache_symlinks(bundle, tmp_path):
    asset = bundle / "auxiliary/vae.pth"
    blob = tmp_path / "cached-blob"
    asset.rename(blob)
    asset.symlink_to(blob)
    assert materialize_run(bundle, tmp_path / "outputs").is_dir()


def test_release_rejects_missing_assets(bundle, tmp_path):
    (bundle / "auxiliary/vae.pth").unlink()
    with pytest.raises(FileNotFoundError):
        materialize_run(bundle, tmp_path / "outputs")


def test_release_rejects_parent_path(bundle, tmp_path):
    (bundle / "release.json").write_text(
        json.dumps(dict(format_version=1, vae="../outside", tokenizer="auxiliary/tokenizer"))
    )
    with pytest.raises(ValueError, match="within the release"):
        materialize_run(bundle, tmp_path / "outputs")
