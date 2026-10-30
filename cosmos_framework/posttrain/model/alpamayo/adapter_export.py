# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Self-contained FP32 native R1 inference artifacts, using the shared adapter merge."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

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
from cosmos_framework.posttrain.model.alpamayo.policy import (
    AlpamayoSFTNet,
    AlpamayoSFTPolicy,
    external_identity,
    load_base,
    validate_targets,
)
from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTModelConfig, require_runtime, source_config
from cosmos_framework.posttrain.utils.object_store import file_sha256

FORMAT = "cosmos-alpamayo-r1-merged-fp32-v1"
MARKER = "merged_model.json"


def model_configuration(identity, processor):
    contract = identity["model_contract"]
    return AlpamayoSFTModelConfig(processor, "none", contract["min_pixels"], contract["max_pixels"])


def export_merged(adapter, base, out, *, processor):
    require_runtime()
    if not processor:
        raise ValueError(
            "Alpamayo merged export requires --processor with the exact recorded public Qwen3-VL processor/config"
        )
    destination = Path(out).absolute()
    if destination.exists():
        raise FileExistsError(f"merged model destination exists: {destination}")
    identity, _ = read_adapter_artifact(adapter)
    net, source = load_base(base, model_configuration(identity, processor))
    config = adapter_configuration(identity)
    validate_targets(net, config)
    resolved = resolve_targets(net, config, family="alpamayo_r1")
    expected = make_adapter_identity(
        base,
        processor,
        resolved,
        identity["media"],
        base_dtype="bfloat16",
        architecture="alpamayo_r1",
        source_identity=source,
    )
    if expected != identity:
        raise ValueError(
            "Alpamayo export compatibility mismatch: original base, processor, native source, media and target settings must match"
        )
    install_adapters(net, config, family="alpamayo_r1", initialize=False)
    net._posttrain_adapter_identity = expected
    load_adapter_artifact(net, adapter)
    net.float()
    merged = prepare_merged_state(net.state_dict(), {**identity, "base_storage_dtype": "float32"})
    # Safetensors forbids shared storage. Only alias entries need copies; the
    # architecture owns and restores their tying on a fresh load.
    pointers = set()
    for name, value in merged.items():
        value = value.detach().cpu().contiguous()
        pointer = value.untyped_storage().data_ptr()
        if pointer in pointers:
            value = value.clone()
        pointers.add(pointer)
        merged[name] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alpamayo-merge-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        save_file(merged, str(stage / "model.safetensors"))
        shutil.copyfile(Path(base) / "config.json", stage / "config.json")
        for name in identity["processor"]["files"]:
            target = stage / "processor" / name
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
    require_runtime()
    root = Path(directory)
    if not (root / MARKER).is_file():
        raise ValueError("incomplete Alpamayo merged artifact: merged_model.json is missing")
    metadata = json.loads((root / MARKER).read_text())
    if metadata.get("format") != FORMAT or metadata.get("dtype") != "float32":
        raise ValueError("unsupported Alpamayo merged artifact format/precision")
    identity = metadata["adapter_identity"]
    required = {"model.safetensors", "config.json"} | {"processor/" + n for n in identity["processor"]["files"]}
    if set(metadata.get("files", {})) != required:
        raise ValueError("incomplete Alpamayo merged artifact inventory")
    for name, digest in metadata["files"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not (root / name).is_file()
            or file_sha256(str(root / name)) != digest
        ):
            raise ValueError(f"Alpamayo merged artifact missing or changed file: {name}")
    if identity["model_contract"]["external"] != external_identity():
        raise ValueError("Alpamayo merged artifact requires the exact recorded native source revision")
    config = source_config(str(root), model_configuration(identity, str(root / "processor")))
    config.model_dtype = "float32"
    config.expert_cfg = {**(config.expert_cfg or {}), "dtype": "float32"}
    net, loading = AlpamayoSFTNet.from_pretrained(
        str(root), config=config, dtype=torch.float32, local_files_only=True, output_loading_info=True
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"incomplete Alpamayo merged weights: {loading}")
    if any(p.is_meta or p.dtype != torch.float32 for p in net.parameters()):
        raise ValueError("Alpamayo merged reload did not preserve FP32 parameter storage")
    return AlpamayoSFTPolicy(net.to(device).eval())
