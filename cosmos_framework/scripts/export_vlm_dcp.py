# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export a Framework Cosmos3 VLM DCP checkpoint to native HF safetensors.

This is intentionally repository-owned: action images and the TAO skill call
this entry point, but do not carry a private copy or patch checkpoint keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_VLM_MODEL_TARGET = "cosmos_framework.model.generator.vlm_model.VLMModel"


def _read_config(config_file: str | Path) -> dict[str, Any]:
    path = Path(config_file)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in Framework config {path}")
    return value


def is_vlm_training_config(config_file: str | Path) -> bool:
    try:
        return _VLM_MODEL_TARGET in Path(config_file).read_text(encoding="utf-8")
    except OSError:
        return False


def infer_vlm_base_model(config_file: str | Path) -> str | None:
    try:
        backbone = _read_config(config_file)["model"]["config"]["policy"]["backbone"]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    for key in ("model_name", "safetensors_path"):
        value = backbone.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def infer_vlm_lora_config(config_file: str | Path) -> dict[str, Any]:
    try:
        policy = _read_config(config_file)["model"]["config"]["policy"]
    except (OSError, ValueError, TypeError, KeyError):
        return {"enabled": False}
    return {
        "enabled": bool(policy.get("lora_enabled", False)),
        "rank": int(policy.get("lora_rank", 16)),
        "alpha": float(policy.get("lora_alpha", 32.0)),
        "dropout": float(policy.get("lora_dropout", 0.0)),
        "target_modules": str(policy.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj")),
        "bias": str(policy.get("lora_bias", "none")),
        "use_rslora": bool(policy.get("lora_use_rslora", False)),
        "modules_to_save": str(policy.get("lora_modules_to_save", "")),
        "precision": policy.get("lora_precision"),
    }


def _checkpoint_model_dir(checkpoint_path: Path) -> Path:
    for candidate in (checkpoint_path / "model", checkpoint_path):
        if (candidate / ".metadata").is_file():
            return candidate
    raise ValueError(f"No Framework model DCP metadata found below {checkpoint_path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_model_files(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser()
    if not root.exists():
        return {"source": str(path), "kind": "uri"}
    root = root.resolve()
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
    )
    files = {
        name: _sha256_file(root / name)
        for name in names
        if (root / name).is_file()
    }
    weights = sorted(root.glob("*.safetensors"))
    files.update({item.name: _sha256_file(item) for item in weights})
    manifest_bytes = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source": str(path),
        "resolved": str(root),
        "kind": "local",
        "files": files,
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def export_vlm_dcp(
    checkpoint_path: str | Path,
    *,
    config_file: str | Path,
    output_dir: str | Path,
    base_model_path_or_uri: str | None = None,
    base_model_revision: str | None = None,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    checkpoint_root = Path(checkpoint_path).expanduser().resolve(strict=True)
    dcp_dir = _checkpoint_model_dir(checkpoint_root)
    config_path = Path(config_file).expanduser().resolve(strict=True)
    export_path = Path(output_dir).expanduser().resolve()
    model_source = base_model_path_or_uri or infer_vlm_base_model(config_path)
    if not model_source:
        raise ValueError("base_model_path_or_uri is required when it cannot be inferred from the saved config")
    local_source = Path(model_source).expanduser()
    if local_source.exists():
        model_source = str(local_source.resolve())

    import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401

    kwargs: dict[str, Any] = {
        "dtype": getattr(torch, dtype),
        "device_map": "cpu",
        "trust_remote_code": True,
    }
    if base_model_revision:
        kwargs["revision"] = base_model_revision
    model = AutoModelForImageTextToText.from_pretrained(model_source, **kwargs)

    lora = infer_vlm_lora_config(config_path)
    if lora["enabled"]:
        from cosmos_framework.utils.generator.lora import inject_lora_pre_fsdp

        inject_lora_pre_fsdp(
            model,
            lora_rank=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            lora_target_modules=lora["target_modules"],
            lora_bias=lora["bias"],
            lora_use_rslora=lora["use_rslora"],
            lora_modules_to_save=lora["modules_to_save"],
            lora_precision=lora["precision"],
        )

    outer = torch.nn.Module()
    inner = torch.nn.Module()
    inner.model = model
    outer.model = inner
    state_dict = get_model_state_dict(outer)
    reader = FileSystemReader(str(dcp_dir))
    checkpoint_keys = set(reader.read_metadata().state_dict_metadata)
    model_keys = set(state_dict)
    if checkpoint_keys != model_keys:
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        raise RuntimeError(
            "Framework VLM DCP exact-key mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"model_keys={len(model_keys)}, checkpoint_keys={len(checkpoint_keys)}"
        )
    dcp.load(state_dict=state_dict, storage_reader=reader)

    merged_adapters = 0
    if lora["enabled"]:
        from cosmos_framework.utils.generator.lora import merge_and_strip_lora_adapters_

        merged_adapters = merge_and_strip_lora_adapters_(model)
        if merged_adapters == 0:
            raise RuntimeError("PEFT checkpoint was restored but no LoRA adapters were merged")

    export_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(export_path, safe_serialization=True, max_shard_size="4GB")
    config_kwargs = {"trust_remote_code": True}
    if base_model_revision:
        config_kwargs["revision"] = base_model_revision
    model_type = AutoConfig.from_pretrained(model_source, **config_kwargs).model_type
    if model_type == "cosmos3_edge":
        from cosmos_framework.data.generator.processors.cosmos3_edge_processing import build_cosmos3_edge_processor

        processor = build_cosmos3_edge_processor(model_source)
    else:
        processor = AutoProcessor.from_pretrained(model_source, **config_kwargs)
    processor.save_pretrained(export_path)

    manifest = {
        "format": "cosmos-framework-vlm-dcp",
        "checkpoint": str(checkpoint_root),
        "checkpoint_metadata_sha256": _sha256_file(dcp_dir / ".metadata"),
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "base_model_path_or_uri": base_model_path_or_uri or model_source,
        "base_model_revision": base_model_revision,
        "base_model_fingerprint": fingerprint_model_files(model_source),
        "tensor_count": len(state_dict),
        "lora": lora,
        "merged_adapters": merged_adapters,
    }
    (export_path / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (export_path / "checkpoint.json").write_text(
        json.dumps({"checkpoint_path": str(checkpoint_root), "checkpoint_type": "vlm_dcp"}, indent=2) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model-path-or-uri")
    parser.add_argument("--base-model-revision")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    args = parser.parse_args()
    export_vlm_dcp(
        args.checkpoint_path,
        config_file=args.config_file,
        output_dir=args.output_dir,
        base_model_path_or_uri=args.base_model_path_or_uri,
        base_model_revision=args.base_model_revision,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
