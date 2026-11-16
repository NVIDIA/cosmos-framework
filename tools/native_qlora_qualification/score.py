# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded WTS evaluation through sealed native Cosmos APIs; never trains.

Run separately for each frozen or adapter-loaded arm under torchrun world-size 1.
The default selects at most 256 rows with seed 2026 before constructing the model.
This is teacher-forced next-token scoring, not free-generation task accuracy.

Opt-in allocator events measure scoring separately from model/reload setup.
The scoring interval includes every row, row-file fsync, final integrity checks
and preprocessing-cache close; it excludes summary construction/serialization
and process-group teardown. Allocator peaks include the resident model, not just growth,
and do not establish whole-device or complete host-memory peaks.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import heapq
import importlib.metadata
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    LETTERS as LETTERS,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    OPTION as OPTION,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    OPTION_MARKER as OPTION_MARKER,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    aggregate as aggregate,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    answer_alignment as answer_alignment,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    evaluate_row as evaluate_row,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    json_digest as json_digest,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    parse_options as parse_options,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    score_answer as score_answer,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    score_prediction as score_prediction,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    validate_choices as validate_choices,
)
from cosmos_framework.posttrain.roles.trainer.validation.wts import (
    validate_wts as validate_wts,
)

PROTOCOL = "cosmos.wts-heldout.v1"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def validate_selection_receipt(args: argparse.Namespace, selection: dict[str, Any]) -> None:
    """Bind final/untouched scoring to a selection fixed before its outcomes."""
    if selection.get("split_role") not in {"final", "untouched_diagnostic"}:
        return
    path, expected = getattr(args, "selection_receipt", None), getattr(args, "selection_receipt_sha256", None)
    if not path or not expected or file_digest(Path(path)) != expected:
        raise ValueError("untouched scoring requires the exact precommitted selection receipt and SHA256")
    receipt = json.loads(Path(path).read_text())
    keys = {
        "protocol",
        "split_manifest_sha256",
        "development_manifest_sha256",
        "locked_utc",
        "selected_scoring_config_sha256",
        "development_evidence_sha256",
    }
    if set(receipt) != keys or receipt["protocol"] != "cosmos.qlora-selection.v1":
        raise ValueError("unsupported selection receipt")
    if receipt["split_manifest_sha256"] != selection["split_manifest_sha256"]:
        raise ValueError("selection receipt names a different untouched split")
    stamp = datetime.datetime.fromisoformat(receipt["locked_utc"])
    if stamp.tzinfo is None or stamp > datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("selection lock must have a timezone and precede scoring")
    hashes = [receipt["split_manifest_sha256"], receipt["development_manifest_sha256"]]
    for key in ("selected_scoring_config_sha256", "development_evidence_sha256"):
        if not isinstance(receipt[key], list) or not receipt[key]:
            raise ValueError("selection requires configuration and development-evidence identities")
        hashes.extend(receipt[key])
    if any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None for value in hashes):
        raise ValueError("selection receipt contains an invalid digest")
    if receipt["development_manifest_sha256"] == receipt["split_manifest_sha256"]:
        raise ValueError("development and untouched split must differ")
    configuration = json_digest({"config": args.config, "overrides": args.override})
    if configuration not in receipt["selected_scoring_config_sha256"]:
        raise ValueError("scoring configuration was not selected using development evidence")
    selection["selection_receipt_sha256"] = expected


def select_scoring_rows(args: argparse.Namespace, source: Path):
    if getattr(args, "all_rows", False):
        from tools.native_qlora_qualification.split import select_sealed_rows

        if not all(getattr(args, key, None) for key in ("split_manifest", "split_manifest_sha256", "split")):
            raise ValueError("all-rows scoring requires an explicit split manifest, digest and role")
        rows, selection = select_sealed_rows(source, Path(args.split_manifest), args.split_manifest_sha256)
        if selection["split_role"] != args.split:
            raise ValueError("requested split role differs from sealed manifest")
        for _, row in rows:
            validate_wts(row)
        validate_selection_receipt(args, selection)
        return rows, selection
    if any(
        getattr(args, key, None)
        for key in ("split_manifest", "split_manifest_sha256", "split", "selection_receipt", "selection_receipt_sha256")
    ):
        raise ValueError("sealed split/selection arguments require all-rows scoring")
    return select_rows(source, args.max_rows, args.seed)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def adapter_state_label(identity: dict[str, Any], *, reloaded: bool) -> str:
    """Distinguish frozen zero-B references from intentionally nonzero initial factors."""
    if reloaded:
        return "reloaded_checkpoint"
    method = identity.get("adapter", {}).get("initialization")
    if method == "kaiming_A_zero_B":
        return "initial_zero_B_frozen_base"
    if method == "nf4_residual_svd" and identity.get("format") == "cosmos.adapter-model.v3":
        return "residual_initialized_untrained_adapter"
    raise ValueError("untrained scoring requires an explicit supported adapter initialization identity")


def select_rows(source: Path, max_rows: int, seed: int) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    """Bound retained row payloads by max_rows; validate every source row without dropping any."""
    if max_rows < 1:
        raise ValueError("max-rows must be positive")
    start_stamp = file_stamp(source)
    source_hash = hashlib.sha256()
    seen: set[str] = set()
    selected: list[tuple[int, int, dict[str, Any]]] = []
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            source_hash.update(line)
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}; no silent row drops")
            row = json.loads(line)
            identifier, _, _, _ = validate_wts(row)
            if identifier in seen:
                raise ValueError(f"duplicate WTS id {identifier!r}")
            seen.add(identifier)
            priority = int(json_digest({"seed": seed, "id": identifier, "row": row}), 16)
            candidate = (-priority, -line_number, row)
            if len(selected) < max_rows:
                heapq.heappush(selected, candidate)
            elif candidate[:2] > selected[0][:2]:
                heapq.heapreplace(selected, candidate)
    if not seen:
        raise ValueError("WTS JSONL source is empty")
    if file_stamp(source) != start_stamp:
        raise RuntimeError("data source changed during selection")
    rows = sorted((-negative_line, row) for _, negative_line, row in selected)
    metadata = {
        "protocol": PROTOCOL,
        "source": str(source),
        "source_sha256": source_hash.hexdigest(),
        "source_rows": len(seen),
        "requested_max_rows": max_rows,
        "selected_rows": len(rows),
        "selection_seed": seed,
        "selection_rule": "lowest_sha256_canonical_json_seed_id_raw_row_then_source_line_v1",
        "scoring_order": "source_line_ascending",
    }
    return rows, metadata


def dense_scoring_config(config: Any, manifest: dict[str, Any]) -> Any:
    """Validate scoring semantics without reopening original base or training data.

    The committed dense artifact owns weights and processor files. Its recorded
    source identity remains bound to the requested native recipe.
    """
    from cosmos_framework.posttrain.config.build import select_policy_backend
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_lora import NATIVE_POLICY_BUILDER
    from cosmos_framework.utils.lazy_config import instantiate

    selected = select_policy_backend(config)
    spec = selected.native_spec
    if (
        selected.backend != "native"
        or spec is None
        or spec.builder != NATIVE_POLICY_BUILDER
        or config.grid.backend != "local"
        or config.grid.grouping != "single"
        or getattr(config.grid, "gpu_node_domains", None)
        or spec.parallelism.dp_replicate != 1
        or spec.parallelism.dp_shard != 1
    ):
        raise ValueError("dense scoring requires the single-device native Qwen3-VL recipe")
    requested = spec.model_config.source_identity
    if not requested or any(manifest["base_identity"].get(key) != value for key, value in requested.items()):
        raise ValueError("dense export base differs from the requested recipe identity")
    rl = instantiate(config.rl)
    rl.__post_init__()
    return rl


def verify_dense_parameters(net: Any, manifest: dict[str, Any]) -> None:
    """Bind actual loaded frozen tensors to the export, with one tensor copy at a time."""
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
    from cosmos_framework.posttrain.model.qwen3_vl import canonical_qwen3_vl_weight_name

    parameters = {}
    objects = set()
    for name, value in net.named_parameters(remove_duplicate=False):
        name = canonical_qwen3_vl_weight_name(name)
        if name in parameters or id(value) in objects or value.requires_grad:
            raise ValueError("dense scoring requires distinct frozen parameters and no adapters")
        parameters[name] = value
        objects.add(id(value))
    expected = manifest["tensors"]
    if set(parameters) != set(expected):
        raise ValueError("loaded dense parameter membership differs from the export")
    for name, parameter in parameters.items():
        record = {key: value for key, value in expected[name].items() if key != "file"}
        if tensor_digest(parameter) != record:
            raise ValueError(f"loaded dense parameter differs from exported bytes: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dense_path = getattr(args, "dense_export", None)
    dense_pin = getattr(args, "dense_export_sha256", None)
    if bool(dense_path) != bool(dense_pin) or dense_path and args.checkpoint:
        raise ValueError("dense scoring requires an export plus its manifest SHA256, and no adapter checkpoint")
    if dense_pin and re.fullmatch(r"[0-9a-f]{64}", dense_pin) is None:
        raise ValueError("dense export manifest SHA256 must be a lowercase SHA256")
    cache_bytes = getattr(args, "preprocessing_cache_bytes", 0)
    if type(cache_bytes) is not int or cache_bytes < 0:
        raise ValueError("preprocessing cache bytes must be a nonnegative integer")
    if os.environ.get("WORLD_SIZE") != "1" or os.environ.get("RANK") != "0" or os.environ.get("LOCAL_RANK") != "0":
        raise ValueError("launch with torchrun --standalone --nproc-per-node=1; only rank zero/world one is supported")
    receipt = {
        name: os.environ.get(name, "") for name in ("COSMOS_TRAINING_CODE_REVISION", "COSMOS_TRAINING_WHEEL_SHA256")
    }
    for name, length in (
        ("COSMOS_TRAINING_CODE_REVISION", 40),
        ("COSMOS_TRAINING_WHEEL_SHA256", 64),
    ):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", receipt[name]):
            raise ValueError(f"{name} must identify the installed sealed build")
    source, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source_stamp = file_stamp(source)
    raw_rows, selection = select_scoring_rows(args, source)

    from cosmos_framework.posttrain.data.datasets.sft import normalize_row, row_identity
    from cosmos_framework.posttrain.tests.qlora_integration import (
        _json_digest,
        _restore_probe_execution,
        _write_json,
    )

    media: dict[str, dict[str, Any]] = {}

    def media_digest(path: str) -> str:
        file = Path(path)
        if path not in media:
            before = file_stamp(file)
            digest = file_digest(file)
            if file_stamp(file) != before:
                raise RuntimeError("video changed while hashing")
            media[path] = {"sha256": digest, "stamp": before}
        if file_stamp(file) != media[path]["stamp"]:
            raise RuntimeError("selected video changed after selection")
        return media[path]["sha256"]

    rows, frozen = [], []
    for line, raw in raw_rows:
        row = normalize_row(raw, source.parent, label=f"{source}:{line}")
        identifier, answer, choices, video = validate_wts(row)
        record = {
            "source_line": line,
            "id": identifier,
            "row_identity": row_identity(row, media_digest=media_digest)[0],
            "raw_row_sha256": json_digest(raw),
            "normalized_row_sha256": json_digest(row),
            "gold": answer,
            "choices": choices,
            "media_sha256": media_digest(video),
            "media_size_bytes": media[video]["stamp"][2],
            "media_path": video,
            "row": row,
        }
        rows.append(row)
        frozen.append(record)
    if "sealed_rows" in selection:
        for actual, expected in zip(frozen, selection.pop("sealed_rows"), strict=True):
            if any(actual[key] != expected[key] for key in ("id", "raw_row_sha256", "media_sha256")):
                raise ValueError("actual selected media differs from sealed split")
    selection["ordered_row_identities"] = [record["row_identity"] for record in frozen]
    selection["ordered_ids"] = [record["id"] for record in frozen]
    # Content-only key supports matching the same selected examples across relocated media roots.
    selection["selected_content_sha256"] = _json_digest(
        [{key: record[key] for key in ("id", "row_identity", "gold", "choices", "media_sha256")} for record in frozen]
    )
    with (output / "selected_rows.jsonl").open("x", encoding="utf-8") as stream:
        for record in frozen:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    selection["selected_rows_file_sha256"] = file_digest(output / "selected_rows.jsonl")
    _write_json(output / "selection.json", selection)
    # All rows and all source/media identities above are frozen before any score or model construction.
    import torch

    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.checkpoint.adapter import (
        load_adapter_weights,
        read_adapter_metadata,
    )
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.config.build import build_backbone, build_optimizer
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.entrypoints.sft import (
        sft_policy_source,
        validate_config,
    )
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
    from cosmos_framework.posttrain.utils.adapter_identity import (
        adapter_processor_identity,
    )
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import (
        destroy_process_group_if_initialized,
        init_local_process_group,
    )
    from cosmos_framework.posttrain.utils.identity import canonical_config_value
    from cosmos_framework.posttrain.utils.native_compile import native_compile_diagnostics
    from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.config import load_config

    config = load_config(args.config, ["--", *args.override])
    saved = read_adapter_metadata(args.checkpoint) if args.checkpoint else None
    if saved is not None:
        _restore_probe_execution(config, saved["identity"], args.override)
    dense_manifest = None
    if dense_path:
        from cosmos_framework.posttrain.checkpoint.dense_adapter import read_dense_export

        dense_manifest = read_dense_export(dense_path, expected_manifest_sha256=dense_pin)
        rl = dense_scoring_config(config, dense_manifest)
    else:
        rl = validate_config(config)
    if not rl.deterministic or rl.ce_exponent != 0 or rl.moe_lbl_coeff != 0:
        raise ValueError("heldout comparison requires deterministic=True and ordinary dense token-mean CE")
    enable_determinism(strict=True)
    init_local_process_group()
    preprocessing = None
    scoring_started = scoring_finished = False
    scoring_error = None
    try:
        misc.set_random_seed(rl.seed, by_rank=False)
        model_source, trust = (dense_path, False) if dense_manifest is not None else sft_policy_source(config)
        packer = build_sft_packer(
            model_source,
            max_length=rl.max_length,
            processor_kwargs=rl.processor_kwargs,
            trust_remote_code=trust,
            mask_think_opener=rl.mask_think_opener,
        )
        if dense_manifest is None:
            model = build_backbone(config)
            optimizer = build_optimizer(model.net, rl.optimizer)
            codec = model.net._cosmos_adapter_checkpoint_codec
            codec.bind_processor_identity(
                adapter_processor_identity(packer, source_path=model_source, source_identity=codec.identity["base"])
            )
            codec.assert_live(model.net, optimizer)
            if saved is not None:
                load_adapter_weights(args.checkpoint, codec, manager=CheckpointManager(checkpoint_format="dcp"))
            adapters_before = {name: tensor_digest(parameter) for name, parameter in codec.parameters.items()}
            if saved is None and adapters_before != codec.identity["initial_adapters"]:
                raise RuntimeError("frozen-baseline arm must use exact initial adapters")
            identity = {key: value for key, value in codec.identity.items() if key != "training_contract"}
            processor_contract = codec.identity["processor_contract"]
            runtime_precision = model.net._cosmos_adapter_runtime_precision
        else:
            from transformers import AutoModelForImageTextToText

            from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy
            from cosmos_framework.posttrain.utils.adapter_identity import dense_adapter_processor_identity

            processor_contract = dense_adapter_processor_identity(
                packer, export_path=dense_path, expected_manifest_sha256=dense_pin
            )
            dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dense_manifest["output_dtype"]]
            record_qlora_phase("dense_reload_start", reset_peak=True)
            net = AutoModelForImageTextToText.from_pretrained(
                dense_path,
                dtype=dtype,
                attn_implementation="sdpa",
                device_map={"": torch.cuda.current_device()},
                local_files_only=True,
                trust_remote_code=False,
            )
            net.requires_grad_(False)
            model = Qwen3VLPolicy(net, "qwen3_vl")
            verify_dense_parameters(model.net, dense_manifest)
            record_qlora_phase("dense_reload_complete")
            adapters_before = dense_manifest["adapter_tensors"]
            identity = {
                "model_format": "cosmos_dense_qwen3_vl_export",
                "dense_export_sha256": dense_pin,
                "source_identity": dense_manifest["source_metadata"]["identity"],
                "output_dtype": dense_manifest["output_dtype"],
            }
            runtime_precision = {
                "parameter_dtype": dense_manifest["output_dtype"],
                "compute_dtype": dense_manifest["output_dtype"],
                "attention": "sdpa",
                "adapter_branches": False,
                "quantized_base": False,
            }
        model.net.eval()
        if cache_bytes:
            from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing

            # The processor contract above remains that of the native packer.
            # Omit workers: the original supported native build predates that
            # keyword and is already single-worker; later builds default to one.
            preprocessing = build_sft_preprocessing(
                packer,
                name_or_path=model_source,
                cache_bytes=cache_bytes,
                prefetch=False,
                media_hashes={path: value["sha256"] for path, value in media.items()},
                max_pending=1,
            )
            packer = preprocessing
        objective = make_objective(
            packer,
            ce_exponent=rl.ce_exponent,
            moe_lbl_coeff=rl.moe_lbl_coeff,
            moe_lbl_method=rl.moe_lbl_method,
            serve_contract=rl.serve_contract,
        )
        _write_json(output / "model_identity.json", identity)
        report_rows = []
        # Loading has its own reset boundaries. Keep these two scoring events
        # adjacent so the existing consumer can identify the complete interval.
        record_qlora_phase("scoring_start", reset_peak=True)
        scoring_started = True
        with (output / "scores.jsonl").open("x", encoding="utf-8") as stream:
            for index, (row, selected) in enumerate(zip(rows, frozen, strict=True), 1):
                media_digest(selected["media_path"])
                scores = evaluate_row(model, packer, objective, row)
                media_digest(selected["media_path"])
                record = {key: value for key, value in selected.items() if key != "row"}
                record.update(scores)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                report_rows.append(record)
                if index == 1 or index % 16 == 0 or index == len(rows):
                    print(
                        json.dumps(
                            {
                                "event": "heldout_progress",
                                "completed": index,
                                "total": len(rows),
                            }
                        ),
                        flush=True,
                    )
            os.fsync(stream.fileno())
        if dense_manifest is None:
            codec.validate_frozen_base()
            if adapters_before != {name: tensor_digest(parameter) for name, parameter in codec.parameters.items()}:
                raise RuntimeError("evaluation changed adapter tensors")
        else:
            verify_dense_parameters(model.net, dense_manifest)
            if read_dense_export(dense_path, expected_manifest_sha256=dense_pin) != dense_manifest:
                raise RuntimeError("dense export changed during scoring")
        if file_stamp(source) != source_stamp or file_digest(source) != selection["source_sha256"]:
            raise RuntimeError("data source changed during evaluation")
        for path in media:
            media_digest(path)
        cache_report = None
        if preprocessing is not None:
            preprocessing.close()
            cache_report = {
                "policy": {"cache_bytes": cache_bytes, "prefetch": False, "workers": 1, "max_pending": 1},
                "stats": preprocessing.preprocessing_stats,
                "memory_scope": "bounded retained CPU cache entries, not total host memory",
                "elapsed_scope": "elapsed_seconds includes selection/media hashing, model/reload, all scoring, integrity checks and cache close; excludes final report serialization and process-group teardown",
            }
            preprocessing = None
        record_qlora_phase("scoring_complete")
        scoring_finished = True
        report = {
            "protocol": PROTOCOL,
            "complete": True,
            "arm": args.arm,
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "adapter_state": (
                "merged_effective_base"
                if dense_manifest is not None
                else adapter_state_label(identity, reloaded=saved is not None)
            ),
            "metrics": aggregate(report_rows),
            "selection_sha256": _json_digest(selection),
            "selected_content_sha256": selection["selected_content_sha256"],
            "scores_sha256": file_digest(output / "scores.jsonl"),
            "model_identity_sha256": _json_digest(identity),
            "adapter_tensors_sha256": _json_digest(adapters_before),
            "processor_contract": processor_contract,
            "objective_identity": objective.identity,
            "quantized_matrices": 0 if dense_manifest is not None else len(identity["quantized_tensors"]),
            **(
                {"dense_export": str(Path(dense_path).resolve()), "dense_export_sha256": dense_pin}
                if dense_manifest is not None
                else {}
            ),
            "recipe": args.config,
            "overrides": args.override,
            "model_seed": rl.seed,
            "runtime_precision": runtime_precision,
            "compile_diagnostics": native_compile_diagnostics(model.net.modules()),
            "max_length": rl.max_length,
            "processor_kwargs": canonical_config_value(rl.processor_kwargs),
            "build_receipt": receipt,
            "evaluator_sha256": file_digest(Path(__file__).resolve()),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")},
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "world_size": 1,
            "elapsed_seconds": time.monotonic() - start,
            **({"preprocessing_cache": cache_report} if cache_report is not None else {}),
            "metric_contract": {
                "restricted_choice": "argmax over the actual question's two to four contiguous A/B[/C[/D]] token logits at the first assistant answer position",
                "strict_first_token": "full-vocabulary argmax equals the gold letter token",
                "valid_letter_rate": "full-vocabulary argmax belongs to the actual question choices",
                "answer_nll": "mean full-vocabulary negative log probability of the single gold answer token",
                "token_weighted_ce": "native supervised-token CE; includes assistant end token and preserves existing masks",
                "tie_break": "torch.argmax: first actual choice in canonical A/B/C/D order, lowest full-vocabulary token id",
                "media_clusters": "exact video content SHA256 only; related camera views/events may need broader grouping",
                "limitations": "bounded teacher-forced diagnostic; not free generation or full heldout/release qualification",
            },
        }
        _write_json(output / "summary.json", report)
        return {"output": str(output), "metrics": report["metrics"], "complete": True}
    except BaseException as error:
        scoring_error = error
        raise
    finally:
        cleanup = []
        if preprocessing is not None:
            cleanup.append(("preprocessing cache close", preprocessing.close))
        if scoring_started and not scoring_finished:
            cleanup.append(("failed scoring memory record", lambda: record_qlora_phase("scoring_failed")))
        cleanup.append(("process-group teardown", destroy_process_group_if_initialized))
        primary_error = scoring_error
        for label, operation in cleanup:
            try:
                operation()
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                elif hasattr(primary_error, "add_note"):
                    primary_error.add_note(f"Additional {label} failure: {error!r}")
        if scoring_error is None and primary_error is not None:
            raise primary_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        help="committed DCP or portable adapter; omit for the frozen-base arm",
    )
    parser.add_argument("--dense-export", help="Completed effective-base dense HF export; excludes --checkpoint")
    parser.add_argument("--dense-export-sha256", help="Required SHA256 of the dense export commit manifest")
    parser.add_argument("--data", required=True, help="converted native WTS validation JSONL")
    parser.add_argument(
        "--output",
        required=True,
        help="new output directory; existing directories are refused",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--max-rows", "--maxrows", dest="max_rows", type=int, default=256)
    selection_group.add_argument("--all-rows", action="store_true", help="score every row from a sealed group split")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split-manifest-sha256")
    parser.add_argument("--split", choices=("development", "regression", "untouched_diagnostic", "final"))
    parser.add_argument("--selection-receipt")
    parser.add_argument("--selection-receipt-sha256")
    parser.add_argument(
        "--preprocessing-cache-bytes",
        type=int,
        default=0,
        help="opt-in retained CPU preprocessing cache bound; zero preserves uncached scoring (no prefetch/workers)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="subset seed, independent of model/recipe seed",
    )
    parser.add_argument(
        "--arm",
        default="unspecified",
        help="descriptive label such as frozen_nf4 or trained_nf4",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="repeatable native config dotlist override",
    )
    args = parser.parse_args()
    output_existed = Path(args.output).resolve().exists()
    try:
        print(json.dumps(run(args), sort_keys=True, allow_nan=False), flush=True)
    except Exception as error:
        output = Path(args.output).resolve()
        if not output_existed and output.is_dir() and not (output / "failure.json").exists():
            write_json(
                output / "failure.json",
                {
                    "complete": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    main()
