# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export a committed SFT checkpoint to portable adapters or a merged HF model.

Portable exports contain no optimizer/RNG/data cursor and cannot resume training.
Merged exports pass through the existing identity-bound HF materializer, including
its locking, staged verification, sidecars and atomic publication.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch

from cosmos_framework.posttrain.checkpoint.adapters import (
    adapter_configuration,
    checkpoint_adapter_state,
    read_adapter_snapshot,
    snapshot_identity,
    state_spec,
    write_adapter_artifact,
)
from cosmos_framework.posttrain.checkpoint.adapters import (
    merged_export_metadata as merged_export_metadata,
)
from cosmos_framework.utils.generator.bf16_cache_contract import PROFILE_FAMILIES
from cosmos_framework.utils.generator.smollm_consistent import PROFILE as SMOLLM_PROFILE

DEPLOYMENT_PROFILES = ("qwen3_vl_bf16_residual_fp32_v1", *PROFILE_FAMILIES, SMOLLM_PROFILE)


def validate_merged_export_metadata(
    record: dict,
    adapter: str,
    identity: dict,
    *,
    output_dtype: str = "float32",
    execution_profile: str | None = None,
) -> None:
    """Validate the immutable adapter binding and any bounded source exclusions."""
    expected = merged_export_metadata(adapter, identity, output_dtype=output_dtype, execution_profile=execution_profile)
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


def validate_export_targets(base: str, identity: dict) -> dict[str, tuple[int, ...]]:
    """Re-resolve the portable map against the stock export architecture on meta.

    Tensor names and shapes alone cannot prove a module's type, component,
    effective scaling or ownership. Reuse the training resolver before reading
    base weight shards; a hand-edited manifest must not bypass its constraints.
    """
    import transformers
    from accelerate import init_on_device
    from transformers import AutoConfig

    from cosmos_framework.posttrain.model.adapter_families import adapter_family
    from cosmos_framework.posttrain.model.adapter_moe_execution import validate_model_execution
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
    adapter_config = adapter_configuration(identity)
    validate_model_execution(net, adapter_config, family.name)
    resolved = resolve_targets(net, adapter_config, family=family.name)
    if resolved != identity["resolved"]:
        raise ValueError(
            "adapter resolved-target manifest disagrees with the base architecture "
            "(module types/shapes, scaling, ownership or trainable counts); re-export from its training checkpoint"
        )
    return {name: tuple(tensor.shape) for name, tensor in net.state_dict().items()}


def export_merged_adapter(
    adapter: str,
    base: str,
    out: str,
    *,
    processor: str | None = None,
    output_dtype: str = "float32",
    execution_profile: str | None = None,
) -> str:
    """One adapter into its materialized base, without live mutation.

    BF16 dense GEMMs do not reproduce BF16 adapter forwards within the real-model
    qualification bounds. FP32 retains the merged equivalence contract; its
    reference is the portable adapter promoted to FP32, preserving the already
    rounded training-base values. Explicit BF16 deployment rounds that completed
    FP32 merge once and requires separate approximation/task qualification.
    Portable reload retains the training precision.
    """
    from cosmos_framework.posttrain.checkpoint.tools.materialize_hf import run_request

    if output_dtype not in ("float32", "bfloat16"):
        raise ValueError("adapter merged export output_dtype must be float32|bfloat16")
    if execution_profile is not None:
        expected_dtype = "float32" if execution_profile == SMOLLM_PROFILE else "bfloat16"
        if execution_profile not in DEPLOYMENT_PROFILES or output_dtype != expected_dtype:
            raise ValueError(f"execution_profile requires a supported profile and output_dtype={expected_dtype}")
    record, state = read_adapter_snapshot(adapter)
    identity = record["identity"]
    if execution_profile == SMOLLM_PROFILE:
        from cosmos_framework.utils.generator.smollm_consistent import validate_identity

        # Validate before dispatching to a physical-format exporter: a profile
        # request must never be silently ignored by another family's exporter.
        validate_identity(identity)
    if output_dtype != "float32" and identity["architecture"] in ("Gr00tN1d7", "alpamayo_r1", "pi05"):
        raise ValueError("native physical merged formats currently require float32; BF16 deployment is HF-only")
    rounded = identity["resolved"].get("model_execution") == "merged_weight_bf16_v1"
    if output_dtype == "bfloat16" and not rounded:
        if identity["architecture"] != "qwen3_vl":
            raise ValueError("BF16 adapter deployment is currently qualified only for dense Qwen3-VL; use float32")
        if execution_profile is None and any(
            row["settings"]["method"] == "lora" and not row["settings"]["use_rslora"]
            for row in identity["resolved"]["targets"]
        ):
            raise ValueError(
                "ordinary LoRA BF16 merged deployment is not qualified: the recorded WTS profile exceeds "
                "the preserved FP32-reference confident-token bound. Use float32 merged export, the BF16 portable adapter, "
                "or explicit --execution-profile=qwen3_vl_bf16_residual_fp32_v1 with the Framework inference loader."
            )
    if (
        identity["architecture"] == "qwen3_vl_moe"
        and any(row["settings"]["method"] == "pissa" for row in identity["resolved"]["targets"])
        and identity["resolved"].get("model_execution", "default_v1") != "qwen3_vl_moe_stock_v1"
    ):
        raise ValueError(
            "PiSSA merged export for native Qwen3-VL-MoE is not qualified: real-checkpoint FP32 "
            "rounding changed a near-tied expert route and exceeded the controlled output bound, "
            "including with dense effective weights under default native execution. Keep the portable adapter for inference "
            "or resume its complete training checkpoint; do not attach raw PiSSA factors to the "
            "original base or rewrite execution metadata. New training with model_execution='qwen3_vl_moe_stock_v1' "
            "and dense PiSSA has a separate qualified FP32 merge contract; it does not convert legacy artifacts. "
            "See docs/posttrain/adapter-effective-execution.md for the measured scope."
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
    metadata = merged_export_metadata(
        adapter, identity, artifact_record=record, output_dtype=output_dtype, execution_profile=execution_profile
    )
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
    # Hash the actual held shard inodes under the materializer's output lock.
    # The tensor source retains the captured portable record and original merge
    # rounding; no full-state model.pt or temporary HF model is constructed.
    from cosmos_framework.posttrain.checkpoint.conversion.adapter_stream import AdapterTensorSource

    if snapshot_identity(base, weights=False) != identity["processor"]:
        raise ValueError("merged export requires the recorded processor/tokenizer/template beside the base")
    architecture = validate_export_targets(base, identity)
    source = AdapterTensorSource(base, adapter, record, state, architecture, output_dtype)
    if source.excluded:
        metadata["excluded_base_tensors"] = source.excluded
    result = run_request(
        base=base,
        out=out,
        tensor_source=source,
        min_tensors=1,
        expect_tensors=len(source),
        verify=True,
        output_dtype=output_dtype,
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
    parser.add_argument(
        "--output-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="merged inference precision; BF16 is approximate deployment, not FP32 equivalence",
    )
    parser.add_argument(
        "--execution-profile",
        choices=DEPLOYMENT_PROFILES,
        help="versioned merged inference arithmetic; requires its declared precision and Framework loading",
    )
    args = parser.parse_args(argv)
    if args.merged != bool(args.base) or (args.adapter and not args.merged):
        parser.error("merged export requires --base; --adapter is accepted only with --merged")
    if not args.merged and (args.output_dtype != "float32" or args.execution_profile is not None):
        parser.error(
            "--output-dtype applies only to merged models; portable adapters preserve their training precision"
        )
    if args.checkpoint:
        source_path = os.path.realpath(args.checkpoint)
        output_root = os.path.abspath(args.out.rstrip("/"))
        # The intermediate portable artifact does not replace the original
        # checkpoint's ownership boundary. Protect it before any transaction
        # path or temporary directory can modify its immutable inventory.
        for suffix in ("", ".tmp", ".previous", ".lock"):
            destination = os.path.realpath(output_root + suffix)
            if os.path.commonpath((source_path, destination)) in (source_path, destination):
                raise ValueError(
                    "adapter export output and transaction paths must not overlap the immutable checkpoint"
                )
        if args.merged and os.path.commonpath((source_path, os.path.realpath(tempfile.gettempdir()))) == source_path:
            raise ValueError("merged adapter export TMPDIR must be outside the immutable checkpoint")
        identity, state, provenance = checkpoint_adapter_state(args.checkpoint)
        portable = {name: state[name] for name in state_spec(identity)}
        if not args.merged:
            write_adapter_artifact(args.out, identity, portable, provenance=provenance)
            return
        with tempfile.TemporaryDirectory(prefix="cosmos-portable-") as temporary:
            path = os.path.join(temporary, "adapter")
            write_adapter_artifact(path, identity, portable, provenance=provenance)
            export_merged_adapter(
                path,
                args.base,
                args.out,
                processor=args.processor,
                output_dtype=args.output_dtype,
                execution_profile=args.execution_profile,
            )
    else:
        export_merged_adapter(
            args.adapter,
            args.base,
            args.out,
            processor=args.processor,
            output_dtype=args.output_dtype,
            execution_profile=args.execution_profile,
        )


if __name__ == "__main__":
    main()
