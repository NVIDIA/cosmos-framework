# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Resolve bundled inference assets without relying on cluster-local paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def materialize_run(bundle: Path, output: Path) -> Path:
    """Write a local runtime config while preserving the original data contract."""
    bundle, output = bundle.resolve(), output.resolve()
    release = json.loads((bundle / "release.json").read_text())
    if release["format_version"] != 1:
        raise ValueError("Unsupported inference release format")
    config = yaml.safe_load((bundle / "config.yaml").read_text())
    model = config["model"]["config"]

    def asset(relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Asset paths must stay within the release")
        path = bundle / relative  # HF snapshot files may be symlinks to cached blobs
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)

    model["tokenizer"]["vae_path"] = asset(release["vae"])
    tokenizer = model["vlm_config"]["tokenizer"]
    tokenizer["pretrained_model_name"] = asset(release["tokenizer"])
    tokenizer["config_variant"] = "hf"
    model["vlm_config"]["pretrained_weights"]["enabled"] = False
    model["vlm_config"]["pretrained_weights"]["backbone_path"] = ""
    if model.get("exclude_reasoner_weights_from_checkpoint"):
        raise ValueError("Release must include the reasoner weights")
    content = yaml.safe_dump(config, sort_keys=False)
    contract = (bundle / "yam_training_contract.json").read_bytes()
    identifier = hashlib.sha256(content.encode() + contract).hexdigest()[:16]
    run_dir = output / ("runtime-" + identifier)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [("config.yaml", content.encode()), ("yam_training_contract.json", contract)]:
        path = run_dir / name
        if path.exists() and path.read_bytes() != value:
            raise ValueError(f"Runtime config changed: {path}")
        if not path.exists():
            path.write_bytes(value)
    return run_dir
