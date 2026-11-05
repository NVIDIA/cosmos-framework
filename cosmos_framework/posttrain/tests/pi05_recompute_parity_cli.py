# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compare frozen compact/current B8 rollout and complete local/RLinf B16 replay.

Run baseline with the frozen project's PYTHONPATH, local with the current project,
and reference with its patched Transformers environment. --records is trusted
compact prepared16 exported from a complete real window, with its provenance manifest.
Only fingerprints and scalar evidence are saved, never model/gradient tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch


def b8_rollout(policy, args):
    import torch

    from cosmos_framework.posttrain.tests.helpers.pi05_parity import NoiseTape, observations, read_request
    from cosmos_framework.posttrain.tests.helpers.pi05_recompute_parity import fingerprint, record_prefix

    request = read_request(args.request)
    if len(request["state"]) != 8:
        raise ValueError("B8 rollout qualification requires exactly eight observations")
    tensors = {}

    def save(name, value):
        tensors[name] = fingerprint(value, name=f"B8 tensor {name}")

    raw = observations(request)
    device = next(policy.model.parameters()).device
    with torch.no_grad():
        inputs = policy.processor.materialize(policy.processor.encode(raw), device=device)
        for index, image in enumerate(inputs.images):
            save(f"inputs.image.{index}", image)
        for name in ("image_masks", "tokens", "token_mask", "state"):
            save(f"inputs.{name}", getattr(inputs, name))
        condition = policy.ppo_condition(raw)
        independently_materialized = policy.model.prepare_condition(inputs)
        for left, right in zip(
            (
                *[leaf for pair in condition.prefix_cache for leaf in pair],
                condition.prefix_mask,
                condition.value_features,
            ),
            (
                *[leaf for pair in independently_materialized.prefix_cache for leaf in pair],
                independently_materialized.prefix_mask,
                independently_materialized.value_features,
            ),
            strict=True,
        ):
            if not torch.equal(left, right):
                raise ValueError("captured preprocessing differs from the public B8 condition path")
        record_prefix(tensors, condition.prefix_mask, condition.prefix_cache, condition.value_features)
        tape = NoiseTape(request["noise"])
        with patch.object(torch, "normal", tape.normal):
            sample = policy.ppo_sample(condition, seed=int(request["seed"]))
        tape.verify()
        for name in ("latent_before", "latent_after", "denoise_index"):
            save(f"trace.{name}", getattr(sample.trace, name))
        save("actions.normalized", sample.normalized_actions)
        save("actions.decoded", policy.processor.decode(sample.normalized_actions))
        save("logprobs", sample.scores.logprobs)
        save("values", sample.scores.values)
    return tensors


def expected_forward_keys(*, rollout):
    keys = {f"inputs.image.{index}" for index in range(3)}
    keys.update(f"inputs.{name}" for name in ("image_masks", "tokens", "token_mask", "state"))
    keys.update(("prefix.mask", "prefix.features", "logprobs", "values"))
    keys.update(f"prefix.{layer}.{part}" for layer in range(18) for part in ("key", "value"))
    if rollout:
        keys.update(f"trace.{name}" for name in ("latent_before", "latent_after", "denoise_index"))
        keys.update(("actions.normalized", "actions.decoded"))
    else:
        keys.add("velocity")
    return keys


def bounded_image_difference(left, right, bounds):
    """Use an independently measured 256-value palette; model scores remain strict."""
    if (
        left["shape"] != right["shape"]
        or left["dtype"] != right["dtype"]
        or left.get("uint8_sha256") != right.get("uint8_sha256")
        or not left.get("uint8_sha256")
        or left.get("used_values") != right.get("used_values")
        or left.get("palette") != bounds["reference_palette"]
        or right.get("palette") != bounds["local_palette"]
    ):
        return None
    delta = max(abs(left["palette"][value] - right["palette"][value]) for value in left["used_values"])
    return delta if delta <= bounds["max_abs"] else None


def pixel_bounds(args):
    import torch
    from openpi.models.model import Observation

    from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
    from cosmos_framework.posttrain.model.pi05.processing import Pi05Processor, Pi05ReplayInputs

    processor = Pi05Processor(Pi05ModelConfig(str(args.checkpoint), str(args.tokenizer), str(args.norm_stats)))
    pixels = torch.arange(256, dtype=torch.uint8).repeat(588).reshape(1, 224, 224, 3)
    reference = Observation.from_dict(
        {
            "image": {"base": pixels.clone()},
            "image_mask": {"base": torch.ones(1, dtype=torch.bool)},
            "state": torch.zeros(1, 32),
        }
    ).images["base"]
    record = Pi05ReplayInputs(
        (pixels, pixels),
        torch.tensor([[True, True, False]]),
        torch.zeros(1, 200, dtype=torch.long),
        torch.zeros(1, 200, dtype=torch.bool),
        torch.zeros(1, 32, dtype=torch.float64),
    )
    local = processor.materialize(record, device="cuda").images[0].cpu()
    left = reference.permute(0, 2, 3, 1).reshape(-1)[:256]
    right = local.permute(0, 2, 3, 1).reshape(-1)[:256]
    maximum = float((left - right).abs().max())
    return {
        "completed": True,
        "kind": "exhaustive uint8 through actual Observation.from_dict CPU and Pi05Processor.materialize GPU",
        "values": 256,
        "max_abs": maximum,
        "different_values": int((left != right).sum()),
        "reference_palette": left.tolist(),
        "local_palette": right.tolist(),
        "torch_version": str(torch.__version__),
    }


def compare(args):
    reports = {name: json.loads(getattr(args, name).read_text()) for name in ("baseline", "local", "reference")}
    baseline, local, reference = (reports[name] for name in ("baseline", "local", "reference"))
    for name, report in reports.items():
        if not report.get("completed"):
            raise ValueError(f"{name} qualification did not complete")
        if name != "reference" and set(report["b8"]) != expected_forward_keys(rollout=True):
            raise ValueError(f"{name} B8 expected tensor inventory differs")
        if set(report["b16"]) != expected_forward_keys(rollout=False):
            raise ValueError(f"{name} B16 expected tensor inventory differs")
        for kind in ("gradients", "parameters"):
            if len(report["update"][kind]) != 216:
                raise ValueError(f"{name} {kind} must include all 216 trained parameters")
        if report["update"]["gradients"].keys() != report["update"]["parameters"].keys():
            raise ValueError(f"{name} gradient/update parameter inventory differs")
    for key in ("checkpoint_sha256", "tokenizer_sha256", "norm_stats_sha256", "request_sha256"):
        if len({report[key] for report in reports.values()}) != 1:
            raise ValueError(f"qualification source mismatch: {key}")
    if len({report["records_sha256"] for report in reports.values()}) != 1:
        raise ValueError("B16 records differ")
    for key in ("optimizer", "actor_lr", "value_lr"):
        if baseline["update"][key] != local["update"][key] or reference["update"][key] != local["update"][key]:
            raise ValueError(f"qualification optimizer mismatch: {key}")
    if any(
        report["update"]["optimizer"]["impl"] != "fused"
        or report["update"]["actor_lr"] != 5e-6
        or report["update"]["value_lr"] != 1e-4
        for report in reports.values()
    ):
        raise ValueError("qualification requires fused actor5e-6/value1e-4 updates")
    if baseline["selection"] != local["selection"] or reference["selection"] != local["selection"]:
        raise ValueError("qualification selected records differ")
    result = {"passed": False, "checks": {}, "bounded_image_inputs": {}}
    bounds = json.loads(args.pixel_bounds.read_text()) if args.pixel_bounds else None
    if bounds is not None and (not bounds.get("completed") or bounds.get("values") != 256):
        raise ValueError("image bounds require a completed exhaustive 256-value probe")
    for name, left, right in (
        ("b8_rollout", baseline["b8"], local["b8"]),
        ("b16_baseline_forward", baseline["b16"], local["b16"]),
        ("b16_baseline_gradients", baseline["update"]["gradients"], local["update"]["gradients"]),
        ("b16_baseline_parameters", baseline["update"]["parameters"], local["update"]["parameters"]),
        ("b16_forward", reference["b16"], local["b16"]),
        ("b16_gradients", reference["update"]["gradients"], local["update"]["gradients"]),
        ("b16_parameters", reference["update"]["parameters"], local["update"]["parameters"]),
    ):
        if left.keys() != right.keys():
            raise ValueError(f"{name} tensor inventory differs")
        failures = [
            key for key in left if any(left[key][field] != right[key][field] for field in ("sha256", "dtype", "shape"))
        ]
        if name == "b16_forward" and bounds is not None:
            for key in ("inputs.image.0", "inputs.image.1"):
                if key in failures:
                    delta = bounded_image_difference(left[key], right[key], bounds)
                    if delta is not None:
                        result["bounded_image_inputs"][key] = {"max_abs": delta, "bound": bounds["max_abs"]}
                        failures.remove(key)
        bounded = len(result["bounded_image_inputs"]) if name == "b16_forward" else 0
        result["checks"][name] = {
            "tensors": len(left),
            "exact": len(left) - len(failures) - bounded,
            "bounded": bounded,
            "nonexact": failures,
        }
    result["b8_input_strides_exact"] = all(
        baseline["b8"][f"inputs.image.{i}"]["stride"] == local["b8"][f"inputs.image.{i}"]["stride"] for i in range(3)
    )
    result["loss_exact"] = baseline["update"]["loss"] == local["update"]["loss"] == reference["update"]["loss"]
    result["gradient_norm_exact"] = (
        baseline["update"]["gradient_norm"] == local["update"]["gradient_norm"] == reference["update"]["gradient_norm"]
    )
    result["unused_reference_head_unchanged"] = reference["update"]["unused_head_unchanged"]
    result["passed"] = all(not row["nonexact"] for row in result["checks"].values()) and all(
        result[key]
        for key in ("b8_input_strides_exact", "loss_exact", "gradient_norm_exact", "unused_reference_head_unchanged")
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "local", "reference", "compare", "pixels"), required=True)
    for name in (
        "output",
        "request",
        "records",
        "checkpoint",
        "tokenizer",
        "norm-stats",
        "baseline",
        "local",
        "reference",
        "pixel-bounds",
    ):
        parser.add_argument(f"--{name}", type=Path, required=name == "output")
    args = parser.parse_args(argv)
    report = {"phase": args.phase, "completed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "compare":
            report = compare(args)
        elif args.phase == "pixels":
            report = pixel_bounds(args)
        else:
            import torch
            import transformers

            from cosmos_framework.posttrain.tests.helpers.pi05_parity import sha256

            if not torch.cuda.is_available():
                raise RuntimeError("full-prefix qualification requires CUDA")
            torch.set_float32_matmul_precision("high")
            report.update(
                torch_version=str(torch.__version__),
                transformers_version=transformers.__version__,
                gpu=torch.cuda.get_device_name(),
            )
            for name, path in (
                ("checkpoint", args.checkpoint / "model.safetensors"),
                ("tokenizer", args.tokenizer),
                ("norm_stats", args.norm_stats),
                ("request", args.request),
            ):
                report[f"{name}_sha256"] = sha256(path)
            if args.phase in ("baseline", "local"):
                from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
                from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy

                policy = Pi05TrainPolicy(
                    Pi05ModelConfig(str(args.checkpoint), str(args.tokenizer), str(args.norm_stats))
                )
                report["b8"] = b8_rollout(policy, args)
            if args.phase in ("baseline", "local", "reference"):
                from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
                from cosmos_framework.posttrain.tests.helpers.pi05_prepared import load_prepared16
                from cosmos_framework.posttrain.tests.helpers.pi05_recompute_parity import (
                    ReferenceReplayPolicy,
                    capture_local,
                    qualify_update,
                )

                config = Pi05ModelConfig(str(args.checkpoint), str(args.tokenizer), str(args.norm_stats))
                samples, manifest = load_prepared16(
                    args.records,
                    config,
                    {key: report[key] for key in ("checkpoint_sha256", "tokenizer_sha256", "norm_stats_sha256")},
                )
                report.update(records_sha256=manifest["records_sha256"], selection=manifest["selection"], b16={})
                if args.phase in ("baseline", "local"):
                    with capture_local(policy, report["b16"]):
                        report["update"] = qualify_update(policy, samples)
                else:
                    from cosmos_framework.posttrain.tests.helpers.pi05_parity import load_reference_model

                    policy = ReferenceReplayPolicy(load_reference_model(args), samples, report["b16"])
                    report["update"] = qualify_update(policy, samples, reference=True)
            report["completed"] = True
    except Exception as error:
        import traceback

        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("b8", "b16", "update")}), flush=True)
    return 0 if report.get("passed", report.get("completed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
