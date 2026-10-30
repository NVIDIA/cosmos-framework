# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""π0.5 FP32 merged inference artifacts; no live model or optimizer mutation.

The native source uses mixed precision and has no HF config/processor. A merged
artifact therefore carries its own complete load contract, rather than pretending
to be a stock HF snapshot or silently applying OpenPI's BF16 roundtrip on reload.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from cosmos_framework.posttrain.checkpoint.adapters import (
    load_adapter_artifact,
    make_adapter_identity,
    prepare_merged_state,
    publish_staged_artifact,
    read_adapter_artifact,
)
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.model.adapters import install_adapters, resolve_targets
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
from cosmos_framework.posttrain.model.pi05.processing import Pi05Processor
from cosmos_framework.posttrain.model.pi05.sft import (
    Pi05SFTPolicy,
    require_sft_runtime,
    source_identity,
    validate_prefix_targets,
)
from cosmos_framework.posttrain.model.pi05.source import Pi05Source, load_model_weights
from cosmos_framework.posttrain.utils.object_store import file_sha256

MERGED_FORMAT = "cosmos-pi05-merged-fp32-v1"
MARKER = "merged_model.json"


def _configuration(identity):
    return AdapterConfig(
        components=[
            AdapterComponentConfig(component=r["component"], targets=[r["name"]], **r["settings"])
            for r in identity["resolved"]["targets"]
        ],
        extra_modules=[
            AdapterExtraModuleConfig(
                component=r["component"],
                targets=[r["name"].rsplit(".", 1)[0]],
                lr_multiplier=r["lr_multiplier"],
                weight_decay=r["weight_decay"],
            )
            for r in identity["resolved"]["extra_parameters"]
        ],
    )


def export_merged(adapter: str, base: str, out: str, *, processor: str | None) -> str:
    require_sft_runtime()
    if not processor:
        raise ValueError(
            "pi05 merged export requires --processor DIR containing the recorded tokenizer.model and norm_stats.json"
        )
    destination = Path(out).absolute()
    if destination.exists():
        raise FileExistsError(f"merged model destination already exists: {destination}")
    identity, _ = read_adapter_artifact(adapter)
    config = Pi05ModelConfig(
        base, str(Path(processor) / "tokenizer.model"), str(Path(processor) / "norm_stats.json"), action_chunk=10
    )
    source = Pi05Source.resolve(config)
    with torch.random.fork_rng(devices=[]):
        net = Pi05Model(config, device="cpu")
        load_model_weights(net, source)
        net.apply_reference_precision()
    source_contract = source_identity(net, source)
    cfg = _configuration(identity)
    validate_prefix_targets(net, cfg)
    resolved = resolve_targets(net, cfg, family="pi05")
    expected = make_adapter_identity(
        base,
        processor,
        resolved,
        identity["media"],
        base_dtype="model_defined",
        architecture="pi05",
        source_identity=source_contract,
    )
    if expected != identity:
        raise ValueError(
            "pi05 adapter compatibility mismatch: original base/processor/precision/targets must match the portable artifact"
        )
    install_adapters(net, cfg, family="pi05", initialize=False)
    net._posttrain_adapter_identity = expected
    load_adapter_artifact(net, adapter)
    # Promote already rounded native weights. This is the same explicit FP32
    # inference reference used for HF adapters, not a second base-model load.
    net.float()
    merged = prepare_merged_state(net.state_dict(), {**identity, "base_storage_dtype": "float32"})
    # Preserve ties by one owner in the file; the native loader verifies aliases.
    del merged["paligemma_with_expert.paligemma.lm_head.weight"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pi05-merge-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        save_file({n: t.detach().cpu().contiguous() for n, t in merged.items()}, str(stage / "model.safetensors"))
        shutil.copyfile(source.tokenizer, stage / "tokenizer.model")
        shutil.copyfile(source.normalization, stage / "norm_stats.json")
        files = {}
        for path in stage.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            files[path.name] = file_sha256(str(path))
        atomic_write_json(
            str(stage / MARKER),
            {"format": MERGED_FORMAT, "dtype": "float32", "files": files, "adapter_identity": identity},
        )
        publish_staged_artifact(str(stage), str(destination), commit_file=MARKER)
    return str(destination)


def load_merged(directory: str, *, device="cpu") -> Pi05SFTPolicy:
    """Load the complete committed FP32 inference profile; never re-round to BF16."""
    require_sft_runtime()
    root = Path(directory)
    if not (root / MARKER).is_file():
        raise ValueError("incomplete pi05 merged artifact: merged_model.json is missing")
    metadata = json.loads((root / MARKER).read_text())
    if metadata.get("format") != MERGED_FORMAT or metadata.get("dtype") != "float32":
        raise ValueError("unsupported pi05 merged artifact format/precision")
    if set(metadata.get("files", {})) != {"model.safetensors", "tokenizer.model", "norm_stats.json"}:
        raise ValueError("incomplete pi05 merged artifact inventory")
    for name, digest in metadata["files"].items():
        if not (root / name).is_file() or file_sha256(str(root / name)) != digest:
            raise ValueError(f"pi05 merged artifact missing or changed file: {name}")
    config = Pi05ModelConfig(str(root), str(root / "tokenizer.model"), str(root / "norm_stats.json"), action_chunk=10)
    with torch.random.fork_rng(devices=[]):
        model = Pi05Model(config, device="cpu").float()
        load_model_weights(
            model, Pi05Source(root / "model.safetensors", root / "tokenizer.model", root / "norm_stats.json")
        )
    model.to(device).eval()
    return Pi05SFTPolicy(model, Pi05Processor(config))
