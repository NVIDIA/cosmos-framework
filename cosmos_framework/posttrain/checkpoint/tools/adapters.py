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
    checkpoint_adapter_state,
    prepare_merged_state,
    read_adapter_artifact,
    snapshot_identity,
    state_spec,
    write_adapter_artifact,
)


def merged_export_metadata(adapter: str, identity: dict) -> dict:
    """Retain the validated portable contract in the materializer's commit marker."""
    manifest = json.loads((Path(adapter) / MANIFEST).read_text())
    if manifest.get("kind") != "portable_adapter" or manifest.get("identity") != identity:
        raise ValueError("portable adapter manifest changed during export; use an immutable artifact")
    return {
        "format": "cosmos-merged-adapter-v1",
        "adapter_identity": identity,
        "adapter_weights_sha256": manifest["weights_sha256"],
    }


def export_merged_adapter(adapter: str, base: str, out: str) -> str:
    """One adapter into its materialized base, exported in FP32 without live mutation.

    BF16 dense GEMMs do not reproduce BF16 adapter forwards within the real-model
    qualification bounds. FP32 is the supported merged inference precision; its
    reference is the portable adapter promoted to FP32, preserving the already
    rounded training-base values. Portable reload retains the training precision.
    """
    from cosmos_framework.posttrain.checkpoint.tools.materialize_hf import run_request

    identity, state = read_adapter_artifact(adapter)
    metadata = merged_export_metadata(adapter, identity)
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
    root = Path(base)
    index = root / "model.safetensors.index.json"
    shards = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    )
    weights = {}
    for shard in shards:
        for name, tensor in load_file(str(root / shard), device="cpu").items():
            if name in weights:
                raise ValueError(f"duplicate base tensor {name!r}")
            # Native materialization converts all floating parameters before loading.
            # Apply that same conversion before merging, even when source shards are
            # FP32 or mix FP32 normalization weights with BF16 projections.
            weights[name] = tensor.to(getattr(torch, base_dtype)).float() if tensor.is_floating_point() else tensor
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
        embedding = "model.language_model.embed_tokens.weight"
        if embedding not in merged or not torch.equal(merged[embedding], merged["lm_head.weight"]):
            raise ValueError("base declares tied embeddings but embedding/head tensors disagree")
        # Stock HF serializes one owner and reconstructs this frozen alias.
        del merged["lm_head.weight"]
    with tempfile.TemporaryDirectory(prefix="cosmos-adapter-merge-") as temporary:
        torch.save({"net": merged}, os.path.join(temporary, "model.pt"))
        result = run_request(
            base=base,
            out=out,
            ckpt_dir=temporary,
            min_tensors=1,
            expect_tensors=len(merged),
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
            export_merged_adapter(path, args.base, args.out)
    else:
        export_merged_adapter(args.adapter, args.base, args.out)


if __name__ == "__main__":
    main()
