# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export a committed SFT checkpoint to portable adapters or a merged HF model.

Portable exports contain no optimizer/RNG/data cursor and cannot resume training.
Merged exports pass through the existing identity-bound HF materializer, including
its locking, staged verification, sidecars and atomic publication.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

from cosmos_framework.posttrain.checkpoint.adapters import (
    MANIFEST,
    adapter_configuration,
    checkpoint_adapter_state,
    prepare_merged_state,
    read_adapter_snapshot,
    snapshot_identity,
    state_spec,
    write_adapter_artifact,
)


def merged_export_metadata(adapter: str, identity: dict, *, artifact_record: dict | None = None) -> dict:
    """Retain the validated portable contract in the materializer's commit marker."""
    # Export uses its captured record, so later path replacement cannot label
    # already consumed tensors with another run's weight digest. Validation of
    # an existing export still compares against the currently supplied artifact.
    manifest = artifact_record if artifact_record is not None else json.loads((Path(adapter) / MANIFEST).read_text())
    if manifest.get("kind") != "portable_adapter" or manifest.get("identity") != identity:
        raise ValueError("portable adapter manifest changed during export; use an immutable artifact")
    return {
        "format": "cosmos-merged-adapter-v1",
        "adapter_identity": identity,
        "adapter_weights_sha256": manifest["weights_sha256"],
    }


def validate_merged_export_metadata(record: dict, adapter: str, identity: dict) -> None:
    """Validate the immutable adapter binding and any bounded source exclusions."""
    expected = merged_export_metadata(adapter, identity)
    if not isinstance(record, dict) or {k: v for k, v in record.items() if k != "excluded_base_tensors"} != expected:
        raise ValueError(
            "merged artifact adapter/base/processor/media identity differs; re-export with the current tool"
        )
    excluded = record.get("excluded_base_tensors", {})
    if excluded and (
        identity["architecture"] != "qwen3_5"
        or not isinstance(excluded, dict)
        or any(
            not name.startswith("mtp.")
            or row.get("reason") != "unused_source_mtp_head"
            or not isinstance(row.get("shape"), list)
            for name, row in excluded.items()
        )
    ):
        raise ValueError("merged artifact contains an unsupported source-tensor exclusion")


def validate_export_targets(base: str, identity: dict) -> frozenset[str]:
    """Re-resolve the portable map against the stock export architecture on meta.

    Tensor names and shapes alone cannot prove a module's type, component,
    effective scaling or ownership. Reuse the training resolver before reading
    base weight shards; a hand-edited manifest must not bypass its constraints.
    """
    import transformers
    from accelerate import init_on_device
    from transformers import AutoConfig

    from cosmos_framework.posttrain.model.adapter_families import adapter_family
    from cosmos_framework.posttrain.model.adapters import resolve_targets
    from cosmos_framework.posttrain.model.hf_adapter_buffers import validate_buffer_contract

    config = AutoConfig.from_pretrained(base)
    family = adapter_family(identity["architecture"])
    validate_buffer_contract(identity, config)
    if config.model_type != family.name or family.hf_auto_class is None:
        raise ValueError("merged adapter export requires the artifact's exact HF architecture binding")
    with init_on_device("meta", include_buffers=False):
        net = getattr(transformers, family.hf_auto_class).from_config(
            config, dtype=torch.float32, attn_implementation="sdpa"
        )
    resolved = resolve_targets(net, adapter_configuration(identity), family=family.name)
    if resolved != identity["resolved"]:
        raise ValueError(
            "adapter resolved-target manifest disagrees with the base architecture "
            "(module types/shapes, scaling, ownership or trainable counts); re-export from its training checkpoint"
        )
    return frozenset(net.state_dict())


def export_merged_adapter(adapter: str, base: str, out: str, *, processor: str | None = None) -> str:
    """One adapter into its materialized base, exported in FP32 without live mutation.

    BF16 dense GEMMs do not reproduce BF16 adapter forwards within the real-model
    qualification bounds. FP32 is the supported merged inference precision; its
    reference is the portable adapter promoted to FP32, preserving the already
    rounded training-base values. Portable reload retains the training precision.
    """
    from cosmos_framework.posttrain.checkpoint.tools.materialize_hf import run_request

    record, state = read_adapter_snapshot(adapter)
    identity = record["identity"]
    if identity["architecture"] == "qwen3_vl_moe" and any(
        row["settings"]["method"] == "pissa" for row in identity["resolved"]["targets"]
    ):
        raise ValueError(
            "PiSSA merged export for native Qwen3-VL-MoE is not qualified: real-checkpoint FP32 "
            "rounding changed a near-tied expert route and exceeded the controlled output bound, "
            "including with the native merged forward. Keep the portable adapter for inference "
            "or resume its complete training checkpoint; do not attach raw PiSSA factors to the "
            "original base. See docs/posttrain/adapter-family-progress.md for the measured limitation."
        )
    if identity["architecture"] == "Gr00tN1d7":
        from cosmos_framework.posttrain.model.gr00t.adapter_export import export_merged

        return export_merged(adapter, base, out, processor=processor)
    if identity["architecture"] == "alpamayo_r1":
        from cosmos_framework.posttrain.model.alpamayo.adapter_export import export_merged

        return export_merged(adapter, base, out, processor=processor)
    if identity["architecture"] == "pi05":
        from cosmos_framework.posttrain.model.pi05.adapter_export import export_merged

        return export_merged(adapter, base, out, processor=processor)
    if processor is not None:
        raise ValueError(
            "HF merged adapters require the processor beside the base; --processor is for native model-owned formats"
        )
    metadata = merged_export_metadata(adapter, identity, artifact_record=record)
    if any(
        row["name"] != row["owner"]
        for row in [*identity["resolved"]["targets"], *identity["resolved"]["extra_parameters"]]
    ):
        raise ValueError(
            "stock HF merged export cannot reconstruct custom adapter weight aliases; "
            "use portable reload into the same tied architecture or a model with independent target weights"
        )
    base_dtype = identity.get("base_storage_dtype")
    if base_dtype not in ("float32", "bfloat16"):
        raise ValueError(
            "adapter artifact lacks a supported materialized base dtype; re-export from its training checkpoint"
        )
    if snapshot_identity(base, weights=True) != identity["base"]:
        raise ValueError("wrong base identity for adapter merge; stage the exact original weight snapshot")
    if snapshot_identity(base, weights=False) != identity["processor"]:
        raise ValueError("merged export requires the recorded processor/tokenizer/template beside the base")
    architecture_keys = validate_export_targets(base, identity)
    root = Path(base)
    index = root / "model.safetensors.index.json"
    shards = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    )
    weights = {}
    excluded = {}
    for shard in shards:
        for name, tensor in load_file(str(root / shard), device="cpu").items():
            if name in weights or name in excluded:
                raise ValueError(f"duplicate base tensor {name!r}")
            if name not in architecture_keys:
                # Qwen3.5's public source also carries a multi-token prediction
                # head. Transformers' Qwen3_5ForConditionalGeneration declares
                # ^mtp.* as ignored on load; HFModel never executes or trains it.
                # Bound this exception to that family and exact namespace. Other
                # unexpected weights still fail instead of disappearing silently.
                if identity["architecture"] == "qwen3_5" and name.startswith("mtp."):
                    excluded[name] = {"shape": list(tensor.shape), "reason": "unused_source_mtp_head"}
                    continue
                raise ValueError(f"base tensor {name!r} is absent from the bound export architecture")
            # Native materialization converts all floating parameters before loading.
            # Apply that same conversion before merging, even when source shards are
            # FP32 or mix FP32 normalization weights with BF16 projections.
            weights[name] = tensor.to(getattr(torch, base_dtype)).float() if tensor.is_floating_point() else tensor
    if excluded:
        metadata["excluded_base_tensors"] = excluded
    for row in identity["resolved"]["targets"]:
        name = row["name"] + ".weight"
        if name not in weights or list(weights[name].shape) != row["shape"]:
            raise ValueError(
                f"base cannot resolve native adapter target {name}; packed/renamed HF conversions are unsupported"
            )
    weights.update(state)
    merged = prepare_merged_state(weights, {**identity, "base_storage_dtype": "float32"})
    config = json.loads((root / "config.json").read_text())
    if config.get("tie_word_embeddings", False) and "lm_head.weight" in merged:
        from cosmos_framework.posttrain.model.adapter_families import adapter_family

        embedding = adapter_family(identity["architecture"]).embedding
        if embedding not in merged or not torch.equal(merged[embedding], merged["lm_head.weight"]):
            raise ValueError("base declares tied embeddings but embedding/head tensors disagree")
        # Stock HF serializes one owner and reconstructs this frozen alias.
        del merged["lm_head.weight"]
    with tempfile.TemporaryDirectory(prefix="cosmos-adapter-merge-") as temporary:
        torch.save({"net": merged}, os.path.join(temporary, "model.pt"))
        expected_tensor_count = len(merged)
        # The materializer mmap-loads the saved state and may transpose all MoE
        # experts. Release the original FP32 allocation before that phase.
        del weights, merged, tensor, state
        import gc

        gc.collect()
        result = run_request(
            base=base,
            out=out,
            ckpt_dir=temporary,
            min_tensors=1,
            expect_tensors=expected_tensor_count,
            verify=True,
            output_dtype="float32",
            export_metadata=metadata,
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="committed local consolidated/DCP training checkpoint")
    source.add_argument("--adapter", help="portable adapter directory (merged export only)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--base", help="exact immutable local base snapshot, required for merged export")
    parser.add_argument("--processor", help="native model processor assets, when stored separately from its base")
    args = parser.parse_args(argv)
    if args.merged != bool(args.base) or (args.adapter and not args.merged):
        parser.error("merged export requires --base; --adapter is accepted only with --merged")
    if args.checkpoint:
        identity, state, provenance = checkpoint_adapter_state(args.checkpoint)
        portable = {name: state[name] for name in state_spec(identity)}
        if not args.merged:
            write_adapter_artifact(args.out, identity, portable, provenance=provenance)
            return
        with tempfile.TemporaryDirectory(prefix="cosmos-portable-") as temporary:
            path = os.path.join(temporary, "adapter")
            write_adapter_artifact(path, identity, portable, provenance=provenance)
            export_merged_adapter(path, args.base, args.out, processor=args.processor)
    else:
        export_merged_adapter(args.adapter, args.base, args.out, processor=args.processor)


if __name__ == "__main__":
    main()
