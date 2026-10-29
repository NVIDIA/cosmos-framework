# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded native adapter evaluation and checkpoint comparison, never a training loop.

Run training/resume exclusively through posttrain.entrypoints.sft. On the same
allocated GPU, invoke ``torchrun --standalone --nproc-per-node=1 -m
cosmos_framework.posttrain.tests.qlora_integration probe --config ...
--checkpoint ... --data validation.jsonl --output fresh-directory`` in a fresh
process for each DCP/portable artifact. ``compare-probes`` and
``compare-checkpoints`` run on CPU and require explicit numerical tolerances.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _probe_rows(data: str, video_row: int, image: str | None, output: Path) -> dict[str, dict[str, Any]]:
    from cosmos_framework.posttrain.data.datasets.sft import normalize_row

    if video_row < 0:
        raise ValueError("video-row must be nonnegative")
    source = Path(data).resolve()
    selected = None
    index = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            videos = [
                part
                for message in raw.get("messages", [])
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "video"
            ]
            if not videos:
                continue
            if index == video_row:
                selected = normalize_row(raw, source.parent, label=f"{source}:{line_number}")
                break
            index += 1
    if selected is None:
        raise ValueError(f"probe requires an actual local video row at video index {video_row}")
    video = next(
        part["video"]
        for message in selected["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "video"
    )
    image_path = Path(image).resolve() if image else output / "video_first_frame.png"
    if image is None:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", video, "-frames:v", "1", "-update", "1", str(image_path)],
            check=True,
            timeout=120,
        )
    if not image_path.is_file():
        raise FileNotFoundError("the fixed image probe requires a real image file")
    return {
        "text": {
            "messages": [
                {"role": "user", "content": "What is 2 + 2? Answer with the number only."},
                {"role": "assistant", "content": "4"},
            ]
        },
        "image": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "What kind of input is shown?"},
                    ],
                },
                {"role": "assistant", "content": "A video frame."},
            ]
        },
        "video": selected,
    }


def _evaluate_row(model: Any, packer: Any, objective: Any, row: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    from cosmos_framework.posttrain.data.datasets.sft import row_identity
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
    from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
    from cosmos_framework.posttrain.schema import ObsBundle, SFTTarget, Trajectory, Transition

    target = SFTTarget(
        messages=row["messages"],
        tools=row.get("tools"),
        supervise=row.get("supervise", "all_assistant"),
        opener=row.get("opener", "self"),
    )
    processed = packer.sft_process_sample(target)
    batch = packer.sft_collate_fn([processed], packer.sft_compute_max_len([processed]))
    valid = (
        (batch["labels"][:, 1:] != -100)
        & batch["attention_mask"][:, 1:].bool()
        & batch["attention_mask"][:, :-1].bool()
    )
    positions = torch.nonzero(valid[0]).flatten()[:8]
    if not positions.numel():
        raise RuntimeError("controlled probe has no supervised tokens")
    input_records = {name: tensor_digest(tensor.contiguous()) for name, tensor in sorted(batch.items())}
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _args, kwargs, output):
        for name, expected in input_records.items():
            if name == "labels":
                continue
            if name not in kwargs or tensor_digest(kwargs[name].contiguous()) != expected:
                raise RuntimeError(f"native objective preprocessing changed the controlled {name} input")
        logits = output if isinstance(output, torch.Tensor) else output.logits
        captured["logits"] = logits[0, positions.to(logits.device)].detach().float().cpu()

    handle = model.net.register_forward_hook(capture, with_kwargs=True)
    try:
        metrics = evaluate_sft_batch(
            model, objective, [Trajectory(transitions=[Transition(obs=ObsBundle(), targets=target)])]
        )
    finally:
        handle.remove()
    logits = captured["logits"]
    if not torch.isfinite(logits).all():
        raise RuntimeError("controlled native probe produced nonfinite logits")
    loss = float(metrics["sft_loss"])
    if not math.isfinite(loss):
        raise RuntimeError("controlled native objective produced nonfinite loss")
    result = {
        "row_identity": row_identity(row)[0],
        "loss": loss,
        "supervised_tokens": int(metrics["sft_valid_tokens"]),
        "sequence_length": len(processed.input_ids),
        "logit_positions": positions.tolist(),
        "inputs": input_records,
    }
    pixel_key = next((key for key in ("pixel_values", "pixel_values_videos") if key in batch), None)
    if pixel_key is not None:
        altered = {key: value for key, value in batch.items() if key != "labels"}
        altered[pixel_key] = torch.zeros_like(altered[pixel_key])
        with torch.no_grad():
            output = model.forward_policy_batch(model.prepare_policy_batch(altered)).logits
        altered_logits = output[0, positions.to(output.device)].float().cpu()
        delta = float((logits - altered_logits).abs().max())
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError("changing actual media pixels did not change controlled native logits")
        result["zero_media_max_logit_change"] = delta
    return result, logits


def probe(args: argparse.Namespace) -> dict[str, Any]:
    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.config.build import build_backbone, build_optimizer
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
    from cosmos_framework.posttrain.utils.adapter_identity import adapter_processor_identity
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized, init_local_process_group
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.config import load_config

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config, ["--", *args.override])
    rl = validate_config(config)
    if not rl.deterministic:
        raise ValueError("controlled integration probes require rl.deterministic=True")
    enable_determinism(strict=True)
    # Refuse incomplete/wrong-format artifacts before constructing the 8B base.
    saved = read_adapter_metadata(args.checkpoint) if args.checkpoint else None
    rows = _probe_rows(args.data, args.video_row, args.image, output)
    init_local_process_group()
    try:
        misc.set_random_seed(rl.seed, by_rank=False)
        source, trust = sft_policy_source(config)
        packer = build_sft_packer(
            source,
            max_length=rl.max_length,
            processor_kwargs=rl.processor_kwargs,
            trust_remote_code=trust,
            mask_think_opener=rl.mask_think_opener,
        )
        torch.cuda.reset_peak_memory_stats()
        model = build_backbone(config)
        optimizer = build_optimizer(model.net, rl.optimizer)
        codec = model.net._cosmos_adapter_checkpoint_codec
        codec.bind_processor_identity(
            adapter_processor_identity(packer, source_path=source, source_identity=codec.identity["base"])
        )
        codec.assert_live(model.net, optimizer)
        if saved is not None:
            load_adapter_weights(args.checkpoint, codec, manager=CheckpointManager(checkpoint_format="dcp"))
        model.net.eval()
        objective = make_objective(
            packer,
            ce_exponent=rl.ce_exponent,
            moe_lbl_coeff=rl.moe_lbl_coeff,
            moe_lbl_method=rl.moe_lbl_method,
            serve_contract=rl.serve_contract,
        )
        reports, logits = {}, {}
        for name, row in rows.items():
            reports[name], logits[name] = _evaluate_row(model, packer, objective, row)
        codec.validate_frozen_base()
        adapters = {name: tensor_digest(parameter) for name, parameter in codec.parameters.items()}
        initial = codec.identity["initial_adapters"]
        updates = {
            factor: sum(adapters[name] != initial[name] for name in adapters if name.endswith(f"lora_{factor}.weight"))
            for factor in ("A", "B")
        }
        if args.require_updates and not all(updates.values()):
            raise RuntimeError("saved native training did not change both adapter factors")
        identity = {key: value for key, value in codec.identity.items() if key != "training_contract"}
        report = {
            "protocol": "cosmos.qlora-integration-probe.v1",
            "identity_sha256": _json_digest(identity),
            "processor_contract": codec.identity["processor_contract"],
            "quantized_matrices": len(identity["quantized_tensors"]),
            "trainable_tensors": len(adapters),
            "trainable_elements": identity["trainable_elements"],
            "adapter_updates": updates,
            "adapters": adapters,
            "probes": reports,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "checkpoint": args.checkpoint,
            "optimizer_parameters": sum(len(group["params"]) for group in optimizer.param_groups),
        }
        torch.save(logits, output / "logits.pt")
        _write_json(output / "probe.json", report)
        return {
            "output": str(output),
            "probes": reports,
            "adapter_updates": updates,
            "quantized_matrices": report["quantized_matrices"],
            "trainable_tensors": len(adapters),
        }
    finally:
        destroy_process_group_if_initialized()


def _assert_close_tree(left: Any, right: Any, *, atol: float, rtol: float, path: str) -> None:
    import numpy as np

    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{path}: tensor shape/dtype differs")
        if left.is_floating_point():
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=lambda message: f"{path}: {message}")
        elif not torch.equal(left, right):
            raise AssertionError(f"{path}: exact tensor state differs")
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError(f"{path}: array state differs")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"{path}: mapping keys differ")
        for name in left:
            _assert_close_tree(left[name], right[name], atol=atol, rtol=rtol, path=f"{path}.{name}")
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"{path}: sequence state differs")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_close_tree(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"{path}: scalar/control state differs")


def compare_probes(args: argparse.Namespace) -> dict[str, Any]:
    reference, candidate = Path(args.reference), Path(args.candidate)
    a, b = (json.loads((path / "probe.json").read_text()) for path in (reference, candidate))
    for key in (
        "protocol",
        "identity_sha256",
        "processor_contract",
        "quantized_matrices",
        "trainable_tensors",
        "trainable_elements",
    ):
        if a[key] != b[key]:
            raise AssertionError(f"probe {key} differs")
    if not args.allow_adapter_differences and a["adapters"] != b["adapters"]:
        raise AssertionError("adapter reload changed saved tensor bytes")
    left, right = (
        torch.load(path / "logits.pt", map_location="cpu", weights_only=True) for path in (reference, candidate)
    )
    maxima = {}
    for name in a["probes"]:
        x, y = a["probes"][name], b["probes"][name]
        for key in ("row_identity", "supervised_tokens", "sequence_length", "logit_positions", "inputs"):
            if x[key] != y[key]:
                raise AssertionError(f"{name}: controlled preprocessing/row identity differs at {key}")
        if not math.isclose(x["loss"], y["loss"], abs_tol=args.atol, rel_tol=args.rtol):
            raise AssertionError(f"{name}: controlled loss differs: {x['loss']} versus {y['loss']}")
        _assert_close_tree(left[name], right[name], atol=args.atol, rtol=args.rtol, path=f"{name}.logits")
        maxima[name] = {
            "logits_max_abs": float((left[name] - right[name]).abs().max()),
            "loss_abs": abs(x["loss"] - y["loss"]),
        }
    return {"passed": True, "atol": args.atol, "rtol": args.rtol, "differences": maxima}


def _load_flat_dcp(path: Path) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata()
    state = {
        name: torch.empty(tuple(value.size), dtype=value.properties.dtype) if hasattr(value, "size") else None
        for name, value in metadata.state_dict_metadata.items()
    }
    dcp.load(
        state,
        storage_reader=reader,
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
    )
    return state


def _portable_trainer_controls(state: dict[str, Any]) -> dict[str, Any]:
    """Compare update/RNG/validation semantics apart from storage publication IDs.

    Committed metadata has already validated publication integrity. Destination,
    manifest and generation-derived ID necessarily differ for independent runs.
    A pending evaluation is not a completed qualification boundary.
    """
    state = copy.deepcopy(state)
    validation = state.get("validation")
    if validation is not None:
        if validation["progress"]["pending_step"] is not None:
            raise ValueError("resume comparison requires a completed validation boundary")
        publication = validation.get("publication")
        if publication is not None:
            validation["publication"] = {
                "step": publication["step"],
                "archive_required": publication["archive_required"],
            }
    return state


def compare_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    """Compare adapter-only DCP, optimizer and exact native controls without a base model."""
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata

    reference, candidate = Path(args.reference), Path(args.candidate)
    metadata_a, metadata_b = (read_adapter_metadata(str(path), require_resume=True) for path in (reference, candidate))
    _assert_close_tree(metadata_a, metadata_b, atol=0, rtol=0, path="metadata")
    maxima = {}
    for component in ("model", "optim"):
        left, right = (_load_flat_dcp(path / component) for path in (reference, candidate))
        _assert_close_tree(left, right, atol=args.atol, rtol=args.rtol, path=component)
        maxima[component] = max(
            (
                float((left[key] - right[key]).abs().max())
                for key in left
                if isinstance(left[key], torch.Tensor) and left[key].numel()
            ),
            default=0.0,
        )
        del left, right
    for component in ("trainer", "data"):
        # Native trainer controls include NumPy/Python RNG tuples and are trusted
        # artifacts from this qualification run, not arbitrary uploaded pickles.
        left, right = (
            torch.load(path / f"{component}.pt", map_location="cpu", weights_only=False)
            for path in (reference, candidate)
        )
        if component == "trainer":
            left, right = _portable_trainer_controls(left), _portable_trainer_controls(right)
        _assert_close_tree(left, right, atol=0, rtol=0, path=component)
    return {
        "passed": True,
        "atol": args.atol,
        "rtol": args.rtol,
        "maximum_tensor_difference": maxima,
        "controls_rng_and_data_exact": True,
        "excluded_storage_identity_fields": [
            "validation.publication.destination",
            "validation.publication.id",
            "validation.publication.manifest",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("probe")
    command.add_argument("--config", required=True)
    command.add_argument("--checkpoint")
    command.add_argument("--data", required=True, help="Prepared native SFT JSONL containing actual local video rows")
    command.add_argument("--output", required=True, help="New probe output directory")
    command.add_argument("--video-row", type=int, default=0)
    command.add_argument(
        "--image", help="Existing fixed image; otherwise extract the first selected video frame with ffmpeg"
    )
    command.add_argument("--override", action="append", default=[], help="Native config dotlist override, repeatable")
    command.add_argument("--require-updates", action="store_true")
    command.set_defaults(operation=probe)
    for name, operation in (("compare-probes", compare_probes), ("compare-checkpoints", compare_checkpoints)):
        command = commands.add_parser(name)
        command.add_argument("--reference", required=True)
        command.add_argument("--candidate", required=True)
        command.add_argument("--atol", type=float, required=True, help="Predeclared absolute tolerance")
        command.add_argument("--rtol", type=float, required=True, help="Predeclared relative tolerance")
        if name == "compare-probes":
            command.add_argument(
                "--allow-adapter-differences",
                action="store_true",
                help="Resume comparison only; compare checkpoint tensors separately",
            )
        command.set_defaults(operation=operation)
    args = parser.parse_args()
    if hasattr(args, "atol") and (
        not math.isfinite(args.atol) or not math.isfinite(args.rtol) or min(args.atol, args.rtol) < 0
    ):
        parser.error("tolerances must be finite and nonnegative")
    print(json.dumps(args.operation(args), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
