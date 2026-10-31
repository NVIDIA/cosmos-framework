# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native full GR00T FP32 merge with its exact backbone and action processors."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.checkpoint.adapters import (
    adapter_configuration,
    load_adapter_artifact,
    make_adapter_identity,
    prepare_merged_state,
    publish_staged_artifact,
    read_adapter_artifact,
)
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.model.adapters import install_adapters, resolve_targets
from cosmos_framework.posttrain.model.gr00t.sft import Gr00tSFTModelConfig, require_runtime
from cosmos_framework.posttrain.model.gr00t.sft_policy import Gr00tSFTPolicy, construct, external_identity, load_base
from cosmos_framework.posttrain.utils.object_store import file_sha256

FORMAT = "cosmos-gr00t-n1d7-merged-fp32-v1"
MARKER = "merged_model.json"


def export_merged(adapter, base, out, *, processor):
    require_runtime()
    if not processor:
        raise ValueError(
            "GR00T merged export requires --processor with its exact recorded backbone config/tokenizer/processor"
        )
    destination = Path(out).absolute()
    if destination.exists():
        raise FileExistsError(f"merged model destination exists: {destination}")
    identity, _ = read_adapter_artifact(adapter)
    model_config = Gr00tSFTModelConfig(processor, "none")
    net, source = load_base(base, model_config)
    config = adapter_configuration(identity)
    resolved = resolve_targets(net, config, family="Gr00tN1d7")
    expected = make_adapter_identity(
        base,
        processor,
        resolved,
        identity["media"],
        base_dtype="float32",
        architecture="Gr00tN1d7",
        source_identity=source,
    )
    if expected != identity:
        raise ValueError(
            "GR00T export compatibility mismatch: base, backbone/processor, native source, media and targets must match"
        )
    install_adapters(net, config, family="Gr00tN1d7", initialize=False)
    net._posttrain_adapter_identity = expected
    load_adapter_artifact(net, adapter)
    merged = prepare_merged_state(net.state_dict(), identity)
    pointers = set()
    for name, value in merged.items():
        value = value.detach().cpu().contiguous()
        pointer = value.untyped_storage().data_ptr()
        if pointer in pointers:
            value = value.clone()
        pointers.add(pointer)
        merged[name] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gr00t-merge-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        save_file(merged, str(stage / "model.safetensors"))
        for name in {"config.json", *identity["processor"]["files"]}:
            shutil.copyfile(Path(base) / name, stage / name)
        for name in identity["processor"]["backbone"]["files"]:
            target = stage / "backbone" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(processor) / name, target)
        files = {}
        for path in stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                files[str(path.relative_to(stage))] = file_sha256(str(path))
        atomic_write_json(
            str(stage / MARKER), {"format": FORMAT, "dtype": "float32", "files": files, "adapter_identity": identity}
        )
        publish_staged_artifact(str(stage), str(destination), commit_file=MARKER)
    return str(destination)


def load_merged(directory, *, device="cpu"):
    import torch

    require_runtime()
    root = Path(directory)
    if not (root / MARKER).is_file():
        raise ValueError("incomplete GR00T merged artifact: merged_model.json is missing")
    metadata = json.loads((root / MARKER).read_text())
    if metadata.get("format") != FORMAT or metadata.get("dtype") != "float32":
        raise ValueError("unsupported GR00T merged format/precision")
    identity = metadata["adapter_identity"]
    required = {"model.safetensors", "config.json", *identity["processor"]["files"]} | {
        "backbone/" + n for n in identity["processor"]["backbone"]["files"]
    }
    if set(metadata.get("files", {})) != required:
        raise ValueError("incomplete GR00T merged artifact inventory")
    for name, digest in metadata["files"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not (root / name).is_file()
            or file_sha256(str(root / name)) != digest
        ):
            raise ValueError(f"GR00T merged artifact missing or changed file: {name}")
    if identity["model_contract"]["external"] != external_identity():
        raise ValueError("GR00T merged artifact requires its recorded native source")
    net = construct(str(root), Gr00tSFTModelConfig(str(root / "backbone"), "none"))
    net.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    if any(p.is_meta or p.dtype != torch.float32 for p in net.parameters()):
        raise ValueError("GR00T merged reload did not preserve FP32 parameters")
    return Gr00tSFTPolicy(net.to(device).eval(), autocast=False)
