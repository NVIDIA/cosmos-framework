# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Training-only, single-block BF16 restoration sensitivity of a frozen NF4 model.

This is an exact temporary native forward intervention, not an exclusion policy,
checkpoint format, latency benchmark, or quality qualification. The original
packed matrices remain resident. No optimizer or checkpoint is constructed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

from tools.native_qlora_qualification import score

PROTOCOL = "cosmos.native-layer-sensitivity.v1"


def training_rows(data: Path, manifest_path: Path, expected_sha: str):
    """Require the already sealed 64-row training diagnostic, before model work."""
    raw = manifest_path.read_bytes()
    import hashlib

    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("training calibration manifest digest differs")
    manifest = json.loads(raw)
    provenance = manifest.get("provenance", {})
    if (
        manifest.get("protocol") != "cosmos.wts-group-split.v1"
        or manifest.get("role") != "training_calibration"
        or manifest.get("release_qualified") is not False
        or provenance.get("historical_exposure") != "training_exposed"
        or provenance.get("preexisting_final_data_consumed") is not False
        or provenance.get("rows") != 64
        or len(manifest.get("rows", [])) != 64
    ):
        raise ValueError("requires the sealed 64-row training-only calibration")
    rows, selection = score.select_rows(data, 64, 20261102)
    if selection["source_rows"] != 64 or selection["source_sha256"] != manifest["data_sha256"]:
        raise ValueError("training calibration source differs")
    for (_, row), record in zip(rows, manifest["rows"], strict=True):
        if str(row["id"]) != record["id"] or score.json_digest(row) != record["raw_row_sha256"]:
            raise ValueError("training calibration ordered row differs")
    return rows, manifest, selection


def target_for(row):
    from cosmos_framework.posttrain.schema import SFTTarget

    return SFTTarget(
        messages=row["messages"],
        tools=row.get("tools"),
        supervise=row.get("supervise", "all_assistant"),
        opener=row.get("opener", "self"),
    )


def warm_cache(packer, rows):
    """Use the native byte-bounded cache; refuse a working set that thrashes."""
    for row in rows:
        packer.sft_process_sample(target_for(row))
    first = json.loads(json.dumps(packer.preprocessing_stats))
    for row in rows:
        packer.sft_process_sample(target_for(row))
    second = packer.preprocessing_stats
    if any(first[k]["misses"] != second[k]["misses"] for k in ("samples", "videos")):
        raise ValueError("64 prepared inputs do not fit the declared native CPU cache")
    return first


def assert_cache_hits(packer, warmed):
    if any(packer.preprocessing_stats[k]["misses"] != warmed[k]["misses"] for k in ("samples", "videos")):
        raise RuntimeError("sensitivity scan unexpectedly reprocessed a cached training row")


def locate_targets(net, expected):
    """Resolve canonical adapter objects through AC wrappers, not name rewriting."""
    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit

    parameters = net._cosmos_adapter_parameters
    owners = {}
    for owner in net.modules():
        for attribute, module in owner._modules.items():
            if isinstance(module, NativeLoraLinear4bit):
                key = id(module.lora_A.weight)
                if key in owners:
                    raise ValueError("ambiguous quantized module ownership")
                owners[key] = (owner, attribute, module)
    result = {}
    for name, shape in expected.items():
        a, b = parameters[name + ".lora_A.weight"], parameters[name + ".lora_B.weight"]
        owner, attribute, module = owners.pop(id(a))
        if module.lora_B.weight is not b or tuple(module.logical_shape) != shape:
            raise ValueError("canonical quantized target differs from its adapter mapping")
        module.validate_quantization_state()
        result[name] = (owner, attribute, module)
    if owners or len(parameters) != 2 * len(expected):
        raise ValueError("unexpected quantized target or adapter inventory")
    return result


def dense_replacement(original, source):
    """Native dense operation with the *same* A/B objects and alpha/rank path."""
    import torch

    from cosmos_framework.utils.generator.lora import LoraInjectedLinear

    shape = (original.out_features, original.in_features)
    if (
        source.device.type != "cpu"
        or source.dtype != torch.bfloat16
        or tuple(source.shape) != shape
        or original.bias is not None
        or original._lora_use_rslora
        or original._adapter_compute_dtype != torch.float32
        or original.lora_A.weight.dtype != torch.float32
        or original.lora_B.weight.dtype != torch.float32
        or not torch.isfinite(source).all()
    ):
        raise ValueError("restoration requires finite BF16 source and native FP32 alpha/rank adapters")
    # All constructor allocations are meta; neither a dense GPU scratch matrix
    # nor a random real adapter is allocated. Never .to() the shared adapters.
    base = torch.nn.Linear(shape[1], shape[0], bias=False, device="meta", dtype=torch.bfloat16)
    replacement = LoraInjectedLinear(
        base,
        original._lora_rank,
        original._lora_alpha,
        adapter_dtype=torch.float32,
        adapter_compute_dtype=torch.float32,
    )
    replacement.weight = torch.nn.Parameter(source.to(original.weight.device, copy=True), requires_grad=False)
    replacement.lora_A, replacement.lora_B = original.lora_A, original.lora_B
    replacement.lora_dropout = original.lora_dropout
    replacement.training = original.training
    return replacement


@contextlib.contextmanager
def restore_block(net, targets, read_source):
    """Restore even a partially loaded block on failure; never serialize it."""
    import torch

    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record, tensor_digest

    if torch.is_grad_enabled() or any(module.training for module in net.modules()):
        raise ValueError("temporary restoration requires eval mode with gradients disabled")
    if getattr(net, "_layer_sensitivity_active", False):
        raise ValueError("nested block interventions are forbidden")
    cpu_rng = torch.get_rng_state().clone()
    devices = {module.weight.device for _, _, module in targets.values() if module.weight.is_cuda}
    gpu_rng = {device: torch.cuda.get_rng_state(device).clone() for device in devices}
    originals, records = [], {}
    net._layer_sensitivity_active = True
    try:
        for name, (owner, attribute, original) in targets.items():
            if owner._modules[attribute] is not original:
                raise ValueError("target ownership changed")
            state = quantized_state_record(original)
            adapters = {k: tensor_digest(v) for k, v in original.adapter_parameters().items()}
            source = read_source(name)
            replacement = dense_replacement(original, source)
            records[name] = {
                "source_bf16_sha256": tensor_digest(source),
                "shape": list(source.shape),
                "dense_bytes": source.numel() * source.element_size(),
                "original_quantized_state": state,
                "adapter_sha256": adapters,
            }
            del source
            originals.append((name, owner, attribute, original))
            owner._modules[attribute] = replacement
            del replacement
        yield records
    finally:
        for _, owner, attribute, original in reversed(originals):
            owner._modules[attribute] = original
        net._layer_sensitivity_active = False
        for name, _, _, original in originals:
            if quantized_state_record(original) != records[name]["original_quantized_state"]:
                raise RuntimeError("intervention mutated original packed NF4 state")
            if {k: tensor_digest(v) for k, v in original.adapter_parameters().items()} != records[name][
                "adapter_sha256"
            ]:
                raise RuntimeError("intervention mutated adapter factors")
        if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
            not torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in gpu_rng.items()
        ):
            raise RuntimeError("intervention consumed RNG state")


def paired_change(baseline, intervention):
    if len(baseline) != len(intervention) or not baseline:
        raise ValueError("paired comparison requires the same nonempty rows")
    for base, changed in zip(baseline, intervention, strict=True):
        if any(base[k] != changed[k] for k in ("id", "input_sha256", "media_sha256", "native_supervised_tokens")):
            raise ValueError("intervention changed paired native inputs")
    return {
        "restricted_choice_accuracy_delta": math.fsum(
            int(b["restricted_choice_correct"]) - int(a["restricted_choice_correct"])
            for a, b in zip(baseline, intervention, strict=True)
        )
        / len(baseline),
        "answer_nll_delta": math.fsum(
            b["answer_nll"] - a["answer_nll"] for a, b in zip(baseline, intervention, strict=True)
        )
        / len(baseline),
        "restricted_choice_disagreements": sum(
            a["restricted_choice_prediction"] != b["restricted_choice_prediction"]
            for a, b in zip(baseline, intervention, strict=True)
        ),
    }


def run(args):
    import torch
    from safetensors import safe_open

    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.config.build import build_backbone
    from cosmos_framework.posttrain.data.datasets.sft import normalize_row
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config
    from cosmos_framework.posttrain.model.quantization.loader import (
        _safetensor_inventory,
        canonical_targets,
        tensor_digest,
    )
    from cosmos_framework.posttrain.model.quantization.source import validate_adapter_source
    from cosmos_framework.posttrain.utils.adapter_identity import adapter_processor_identity
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized, init_local_process_group
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.config import load_config

    if os.environ.get("COSMOS_SFT_PROFILE_DIR"):
        raise ValueError("layer sensitivity requires profiler hooks disabled")
    if [os.environ.get(k) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")] != ["1", "0", "0"]:
        raise ValueError("requires owned torchrun world one/rank zero")
    if not 0 < args.cache_bytes <= 8 * 1024**3:
        raise ValueError("explicit parent CPU cache must be at most 8 GiB")
    data, manifest_path, output = Path(args.data), Path(args.manifest), Path(args.output)
    raw_rows, manifest, selection = training_rows(data, manifest_path, args.manifest_sha256)
    rows, selected, media = [], [], {}
    for (line, raw), expected in zip(raw_rows, manifest["rows"], strict=True):
        row = normalize_row(raw, data.parent, label=f"{data}:{line}")
        _, _, _, video = score.validate_wts(row)
        path = Path(video)
        digest = score.file_digest(path)
        if digest != expected["media_sha256"]:
            raise ValueError("training media differs from sealed calibration")
        media[video] = {"sha256": digest, "stamp": score.file_stamp(path)}
        rows.append(row)
        selected.append(dict(expected, source_line=line))
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = load_config(args.config, ["--", *args.override])
    rl = validate_config(config)
    cfg = config.model.model_config
    if (
        cfg.quantized_base is None
        or cfg.adapter.initialization != "kaiming_A_zero_B"
        or cfg.adapter_compute_dtype != "float32"
        or cfg.compile_language_model
        or cfg.sft_logits_mode != "full"
        or not rl.deterministic
        or rl.ce_exponent != 0
        or rl.moe_lbl_coeff != 0
    ):
        raise ValueError("initial sensitivity scope is eager frozen zero-B NF4/FP32 adapters/full logits/ordinary CE")
    build = {k: os.environ.get(k, "") for k in ("COSMOS_TRAINING_CODE_REVISION", "COSMOS_TRAINING_WHEEL_SHA256")}
    if any(
        len(build[k]) != length or any(c not in "0123456789abcdef" for c in build[k])
        for k, length in zip(build, (40, 64), strict=True)
    ):
        raise ValueError("requires sealed native code and wheel identity")
    enable_determinism(strict=True)
    init_local_process_group()
    packer = None
    try:
        misc.set_random_seed(rl.seed, by_rank=False)
        model_source, trust = sft_policy_source(config)
        native_packer = build_sft_packer(
            model_source,
            max_length=rl.max_length,
            processor_kwargs=rl.processor_kwargs,
            trust_remote_code=trust,
            mask_think_opener=rl.mask_think_opener,
        )
        packer = build_sft_preprocessing(
            native_packer,
            name_or_path=model_source,
            cache_bytes=args.cache_bytes,
            prefetch=False,
            media_hashes={k: v["sha256"] for k, v in media.items()},
            max_pending=1,
        )
        cache_warmed = warm_cache(packer, rows)
        model = build_backbone(config)
        codec = model.net._cosmos_adapter_checkpoint_codec
        codec.bind_processor_identity(
            adapter_processor_identity(native_packer, source_path=model_source, source_identity=codec.identity["base"])
        )
        codec.assert_live()
        identity = json.loads(json.dumps(codec.identity))
        adapters = {name: tensor_digest(p) for name, p in codec.parameters.items()}
        if adapters != identity["initial_adapters"] or any(
            torch.count_nonzero(p) for name, p in codec.parameters.items() if name.endswith(".lora_B.weight")
        ):
            raise ValueError("requires exact untrained zero-B adapter state")
        model.net.eval()
        targets = locate_targets(model.net, canonical_targets())
        inventory = _safetensor_inventory(Path(model_source))
        source_stamps = {path: score.file_stamp(path) for path, _, _ in inventory.values()}
        objective = make_objective(
            packer, ce_exponent=0, moe_lbl_coeff=0, moe_lbl_method=rl.moe_lbl_method, serve_contract=rl.serve_contract
        )
        score.write_json(output / "reference-model-identity.json", identity)
        score.write_json(
            output / "selection.json", {**selection, "manifest_sha256": args.manifest_sha256, "rows": selected}
        )

        def read_source(name):
            path, shape, dtype = inventory[name + ".weight"]
            if shape != canonical_targets()[name] or dtype != "BF16" or score.file_stamp(path) != source_stamps[path]:
                raise ValueError("source weight inventory/stamp changed")
            with safe_open(path, framework="pt", device="cpu") as reader:
                return reader.get_tensor(name + ".weight")

        def evaluate(case):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            result = []
            with (output / (case + ".jsonl")).open("x") as stream:
                for row, record in zip(rows, selected, strict=True):
                    value = {**record, **score.evaluate_row(model, packer, objective, row)}
                    assert_cache_hits(packer, cache_warmed)
                    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    result.append(value)
                os.fsync(stream.fileno())
            torch.cuda.synchronize()
            stats = {
                "elapsed_seconds": time.monotonic() - begin,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "metrics": score.aggregate(result),
                "scores_sha256": score.file_digest(output / (case + ".jsonl")),
            }
            print(json.dumps({"event": "layer_sensitivity", "case": case, **stats}), flush=True)
            return result, stats

        cases = {}
        with torch.no_grad():
            baseline, cases["baseline"] = evaluate("baseline")
            for block in range(36):
                name = f"block_{block:02d}"
                block_targets = {
                    key: value
                    for key, value in targets.items()
                    if key.startswith(f"model.language_model.layers.{block}.")
                }
                torch.cuda.synchronize()
                start = time.monotonic()
                with restore_block(model.net, block_targets, read_source) as restored:
                    changed, stats = evaluate(name)
                    stats["paired_change"] = paired_change(baseline, changed)
                    stats["restored_matrices"] = restored
                torch.cuda.synchronize()
                stats["load_score_restore_seconds"] = time.monotonic() - start
                codec.assert_live()
                cases[name] = stats
            control, cases["restore_control"] = evaluate("restore_control")
        if control != baseline:
            raise RuntimeError("restored control differs from the exact original baseline")
        codec.validate_frozen_base()
        if codec.identity != identity or adapters != {name: tensor_digest(p) for name, p in codec.parameters.items()}:
            raise RuntimeError("diagnostic changed model identity or adapter values")
        if validate_adapter_source(cfg) != identity["base"]:
            raise RuntimeError("base source identity changed")
        training_rows(data, manifest_path, args.manifest_sha256)
        for path, record in media.items():
            if score.file_stamp(Path(path)) != record["stamp"] or score.file_digest(Path(path)) != record["sha256"]:
                raise RuntimeError("training media changed")
        report = {
            "protocol": PROTOCOL,
            "complete": True,
            "diagnostic_only": True,
            "training_only": True,
            "reference_identity_sha256": score.json_digest(identity),
            "cases": cases,
            "build": build,
            "config": args.config,
            "overrides": args.override,
            "tool_sha256": score.file_digest(Path(__file__)),
            "scorer_sha256": score.file_digest(Path(score.__file__)),
            "cache": packer.preprocessing_stats,
            "elapsed_seconds": time.monotonic() - started,
            "scope": {
                "intervention": "one language block's seven NF4 projections replaced by original BF16 source with identical adapters",
                "approximation": False,
                "packed_originals_retained": True,
                "dense_extra_bytes_per_block": 385875968,
                "maximum_source_matrix_cpu_bytes": 100663296,
                "native_training_updates": 0,
                "checkpoint_exported": False,
                "quality_qualified": False,
                "exclusion_policy_qualified": False,
                "timing_is_performance_comparison": False,
                "host_total_residency_measured": False,
                "memory": "forward Torch allocator peaks include reference model and temporary dense block; source loading/CPU/page cache and whole-host residency are not these peaks",
            },
        }
        score.write_json(output / "summary.json", report)
        return report
    finally:
        if packer is not None:
            packer.close()
        destroy_process_group_if_initialized()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "manifest", "manifest-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cache-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    output_existed = Path(args.output).exists()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        if (
            not output_existed
            and output.is_dir()
            and not (output / "summary.json").exists()
            and not (output / "failure.json").exists()
        ):
            score.write_json(
                output / "failure.json", {"complete": False, "error": str(error), "error_type": type(error).__name__}
            )
        raise


if __name__ == "__main__":
    main()
