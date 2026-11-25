# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Transfer-Auto FP8 recipe: static per-tensor E4M3 for the multiview Transfer-Auto checkpoint.

``quantize_fp8_checkpoint`` drives its own single-view denoise loop. Transfer-Auto needs multiview
inputs, WSM control videos and LiDAR, so this recipe calibrates through the native
:meth:`OmniInference.generate` on request JSONL files instead. The recipe has three steps. Each step
runs alone:

1. :func:`calibrate_transfer_auto` records the per-projection activation amax (ModelOpt ``max``
   calibration). Amax files from several runs merge by elementwise max (:func:`load_activation_amax`).
2. :func:`search_weight_scales` (optional) picks a weight scale for each projection by minimum
   projection-output MSE on captured activation rows.
3. :func:`export_transfer_auto_fp8` streams the BF16 source shards and writes the FP8 checkpoint.

The quantized set is the 504 attention and MLP projections of both towers (``projection_map``).
Every other tensor is copied without change.

Command line (``python -m cosmos_framework.quantization.transfer_auto <command> --help``)::

    calibrate   --checkpoint SRC --requests a.jsonl ... --output amax.json
    calibrate   --merge amax_a.json amax_b.json --output amax.json
    search      --source SRC --activation-amax amax.json --captures cap.pt ... --output weight_scales.json
    export      --source SRC --activation-amax amax.json [--weight-scales weight_scales.json] --output OUT
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .export import add_diffusion_step_policy
from .metadata import METADATA_FILENAME, collect_quantization_metadata
from .safetensors_index import SafetensorsIndex, build_root_index

FP8_E4M3_MAX = 448.0
RECIPE_NAME = "transfer_auto_fp8"
PRODUCER = {"name": "cosmos-framework", "recipe": RECIPE_NAME}

# Candidate multipliers of the max-derived weight scale. 1.0 is the control and always wins ties.
DEFAULT_MULTIPLIERS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.125, 1.125, 1.25, 1.5, 2.0)

# Module-name patterns that stay in BF16 (the ModelOpt ``ignore`` list of the exported config).
IGNORE_PATTERNS = (
    "proj_in",
    "proj_out",
    "time_embedder*",
    "lidar_proj_in",
    "lidar_proj_out",
    "visual*",
    "model.visual*",
    "lm_head",
    "projector*",
    "audio_proj*",
    "action_proj*",
)

# Entries of the source checkpoint root that export does not carry over.
_SKIP_ROOT_ENTRIES = {"transformer", "model.safetensors.index.json", ".cache", ".git", "assets", "images"}


def projection_map(keys: Iterable[str], depth: int, scope: str = "both") -> dict[str, str]:
    """Map each quantized diffusers weight key to its native runtime module name.

    Per layer and tower: q/k/v/o and gate/up/down. ``scope`` is ``"gen"``, ``"und"`` or ``"both"``.
    Raises ``ValueError`` when ``keys`` does not hold every projection (504 for 36 layers, ``both``).
    """
    if scope not in {"gen", "und", "both"} or type(depth) is not int or depth < 1:
        raise ValueError("Invalid scope or transformer depth")
    result: dict[str, str] = {}
    for i in range(depth):
        for tower in ("und", "gen") if scope == "both" else (scope,):
            is_gen = tower == "gen"
            dest = f"gen_layers.{i}" if is_gen else f"language_model.layers.{i}"
            for role in ("q", "k", "v", "o"):
                if is_gen:
                    leaf = "to_add_out" if role == "o" else f"add_{role}_proj"
                else:
                    leaf = "to_out" if role == "o" else f"to_{role}"
                module_leaf = "to_out" if role == "o" else f"to_{role}"
                attn = "cross_attention" if is_gen else "self_attn"
                result[f"layers.{i}.self_attn.{leaf}.weight"] = f"{dest}.{attn}.{module_leaf}"
            for role in ("gate", "up", "down"):
                mlp = "mlp_moe_gen" if is_gen else "mlp"
                result[f"layers.{i}.{mlp}.{role}_proj.weight"] = f"{dest}.mlp.{role}_proj"
    missing = set(result) - set(keys)
    if missing:
        raise ValueError(f"Projection inventory missing {sorted(missing)}")
    return result


def _positive_finite(value: float, what: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be finite and positive, got {value}")
    return value


def _source_shards(source: Path) -> list[Path]:
    shards = sorted((source / "transformer").glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards in {source / 'transformer'}")
    return shards


def _source_keys(shards: Sequence[Path]) -> list[str]:
    keys: list[str] = []
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            keys.extend(handle.keys())
    return keys


def _weight_scale_from_max(weight: torch.Tensor) -> torch.Tensor:
    """FP32 ``max|W| / 448`` as a 0-dim tensor (1 for an all-zero weight)."""
    amax = weight.to(torch.float32).abs().amax()
    return torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))


def _input_scale(amax: float) -> float:
    return amax / FP8_E4M3_MAX if amax else 1.0


# --------------------------------------------------------------------------------------------------
# Step 1: calibration
# --------------------------------------------------------------------------------------------------


def load_activation_amax(paths: Sequence[str | Path]) -> dict[str, float]:
    """Read ``activation_amax`` from one or more JSON files and join them by elementwise max.

    Each file is the output of :func:`calibrate_transfer_auto` or a ``quantization_metadata.json``. All
    files must hold the same projection keys. The max reduction is associative, so shards can merge
    in any order.
    """
    if not paths:
        raise ValueError("At least one activation-amax JSON file is required")
    merged: dict[str, float] | None = None
    for path in paths:
        amax = json.loads(Path(path).read_text())["activation_amax"]
        if merged is None:
            merged = {key: float(value) for key, value in amax.items()}
            continue
        if amax.keys() != merged.keys():
            raise ValueError(f"{path} holds different projection keys than the first file")
        merged = {key: max(merged[key], float(amax[key])) for key in merged}
    assert merged is not None
    for key, value in merged.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid activation amax for {key}: {value}")
    return merged


def calibrate_transfer_auto(
    checkpoint_path: str | Path,
    input_files: Sequence[str | Path],
    output_json: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    """ModelOpt ``max`` calibration of the 504 projections through native ``OmniInference.generate``.

    ``input_files`` are request JSONL files for the native inference CLI (7 cameras, all denoising
    steps). The model loads once and every request runs as a normal generation. Quantizers sit on the
    input and weight of each projection only. ``output_json`` gets ``{"activation_amax": {weight_key:
    amax}, ...}``. Run it once for each shard of requests, then join the shards with
    :func:`load_activation_amax`. Returns the ``activation_amax`` map.

    Needs a GPU and the BF16 source checkpoint. Generated videos go to ``output_dir`` (a temporary
    directory by default).
    """
    # init_script must run before the model implementation is imported.
    from cosmos_framework.inference.common.init import init_script

    init_script()
    import tempfile

    import modelopt.torch.quantization as mtq

    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.args import sample_output_dir
    from cosmos_framework.inference.common.init import init_output_dir
    from cosmos_framework.inference.model import _diffusers_to_net_key

    checkpoint_path = Path(checkpoint_path)
    config = json.loads((checkpoint_path / "transformer/config.json").read_text())
    index = json.loads((checkpoint_path / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    mapping = projection_map(index["weight_map"], config["num_hidden_layers"], "both")
    native = {
        key: _diffusers_to_net_key(key, "transformer/weights.safetensors").removesuffix(".weight") for key in mapping
    }

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="transfer-auto-calibration-"))
    setup_args = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_path), output_dir=out_dir, diffusion_cache=True
    ).build_setup()
    init_output_dir(setup_args.output_dir)
    overrides_list = setup_args.get_sample_overrides_cls().from_files(
        [Path(p) for p in input_files], overrides=setup_args.sample_overrides
    )
    for overrides in overrides_list:
        overrides.output_dir = sample_output_dir(setup_args.output_dir, overrides.name)
        overrides.download(overrides.output_dir / "inputs")
    pipe = setup_args.get_inference_cls().create(setup_args)
    samples = [overrides.build_sample(model_config=pipe.model_config) for overrides in overrides_list]

    modules = dict(pipe.model.net.named_modules())
    for key, name in native.items():
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"Missing native projection {key} -> {name}")

    # Same FP8 quantizer specs as the ModelOpt default preset, enabled on the 504 projections only.
    preset = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    specs = {
        kind: next(e["cfg"] for e in preset["quant_cfg"] if e["quantizer_name"] == f"*{kind}_quantizer")
        for kind in ("input", "weight")
    }
    quant_cfg = {
        "algorithm": "max",
        "quant_cfg": [{"quantizer_name": "*", "enable": False}]
        + [
            {"quantizer_name": f"{name}.{kind}_quantizer", "cfg": copy.deepcopy(specs[kind])}
            for name in native.values()
            for kind in ("input", "weight")
        ],
    }

    def forward_loop(_net) -> None:
        results = pipe.generate(samples)
        if not results or any(str(item.status) != "success" for item in results):
            raise RuntimeError(f"Calibration generation failed: {[str(item.status) for item in results]}")

    print(f"[calib] calibrating {len(native)} linears on {len(samples)} native requests", flush=True)
    mtq.quantize(pipe.model.net, quant_cfg, forward_loop)
    modules = dict(pipe.model.net.named_modules())
    amax = {key: float(modules[name].input_quantizer.amax.item()) for key, name in native.items()}

    record = {
        "activation_amax": amax,
        "algorithm": "ModelOpt max on native OmniInference.generate",
        "checkpoint": str(checkpoint_path),
        "requests": [str(p) for p in input_files],
        "samples": [sample.name for sample in samples],
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(record, indent=2))
    print(f"[calib] wrote {output_json}", flush=True)
    return amax


# --------------------------------------------------------------------------------------------------
# Step 2: weight-scale search
# --------------------------------------------------------------------------------------------------


def _qdq(value: torch.Tensor, scale: float) -> torch.Tensor:
    return (value / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale


@torch.inference_mode()
def optimize_weight_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_scale: float,
    weight_scale: float,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
) -> dict:
    """Pick the weight-scale multiplier with the lowest projection-output MSE.

    ``x`` is ``[rows, in_features]``, ``weight`` is ``[out_features, in_features]``. Activations are
    fake-quantized with ``input_scale``; the weight with ``weight_scale * multiplier``. The score is
    ``mean((qdq(x) @ qdq(W).T - x @ W.T) ** 2)`` in FP32. Multiplier 1.0 is always tried first, so a
    tie keeps the max-derived scale. Returns ``multiplier``, ``scale``, ``baseline_mse`` and ``best_mse``.
    """
    xs, ws = _positive_finite(input_scale, "input_scale"), _positive_finite(weight_scale, "weight_scale")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1] or not len(x):
        raise ValueError("Expected compatible nonempty projection matrices")
    if not torch.isfinite(x).all() or not torch.isfinite(weight).all():
        raise ValueError("Projection inputs must be finite")
    multipliers = list(dict.fromkeys([1.0, *(_positive_finite(m, "multiplier") for m in multipliers)]))
    x, weight = x.float(), weight.float()
    reference = F.linear(x, weight)
    x_q = _qdq(x, xs)
    trials = []
    for multiplier in multipliers:
        mse = float((F.linear(x_q, _qdq(weight, ws * multiplier)) - reference).square().mean())
        if not math.isfinite(mse):
            raise ValueError("Nonfinite reconstruction error")
        trials.append((mse, multiplier))
    best_mse, best = min(trials, key=lambda t: t[0])  # min keeps the first (multiplier 1.0) on ties
    return {"multiplier": best, "scale": ws * best, "baseline_mse": trials[0][0], "best_mse": best_mse}


def search_weight_scales(
    source: str | Path,
    activation_amax: dict[str, float] | str | Path | Sequence[str | Path],
    captures: Sequence[str | Path],
    output_json: str | Path,
    *,
    device: str = "cuda",
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
) -> dict[str, float]:
    """Search a weight scale for each of the 504 projections on captured projection inputs.

    ``captures`` are files written by ``torch.save``: a dict ``name -> [rows, in_features]`` tensor,
    where ``name`` is the weight key or ``"<tag>:<weight_key>"``. Rows from all files join for each
    projection. Every projection needs captured rows. ``activation_amax`` is a map, one JSON path,
    or several JSON paths (see :func:`load_activation_amax`).

    Writes and returns ``{weight_key: weight_scale}`` for the projections where a multiplier other
    than 1 wins. Projections that keep the max-derived scale are absent. Pass the file to
    :func:`export_transfer_auto_fp8`.
    """
    source = Path(source)
    if not isinstance(activation_amax, dict):
        paths = [activation_amax] if isinstance(activation_amax, (str, Path)) else list(activation_amax)
        activation_amax = load_activation_amax(paths)
    config = json.loads((source / "transformer/config.json").read_text())
    shards = _source_shards(source)
    selected = projection_map(_source_keys(shards), config["num_hidden_layers"], "both")
    shard_of: dict[str, Path] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            shard_of.update({key: shard for key in handle.keys() if key in selected})

    rows: dict[str, list[torch.Tensor]] = {key: [] for key in selected}
    for path in captures:
        captured = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for name, value in captured.items():
            key = name.split(":", 1)[-1]
            if key not in rows:
                raise ValueError(f"{path}: capture {name!r} is not a quantized projection")
            rows[key].append(value)
    empty = sorted(key for key, values in rows.items() if not values)
    if empty:
        raise ValueError(f"No captured rows for {len(empty)} projections, e.g. {empty[:3]}")

    changed: dict[str, float] = {}
    with torch.inference_mode():
        for n, key in enumerate(sorted(selected), start=1):
            with safe_open(shard_of[key], framework="pt") as handle:
                weight = handle.get_tensor(key).to(device)
            x = torch.cat([value.to(device) for value in rows[key]])
            ws = float(_weight_scale_from_max(weight).cpu())
            xs = float(torch.tensor(_input_scale(activation_amax[key]), dtype=torch.float32))
            best = optimize_weight_scale(x, weight, input_scale=xs, weight_scale=ws, multipliers=multipliers)
            if best["multiplier"] != 1.0:
                changed[key] = best["scale"]
            print(f"[search] {n}/{len(selected)} {key}: W x{best['multiplier']}", flush=True)
            del x, weight
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(changed, indent=2, sort_keys=True))
    print(f"[search] {len(changed)} of {len(selected)} weight scales changed -> {output_json}", flush=True)
    return changed


# --------------------------------------------------------------------------------------------------
# Step 3: export
# --------------------------------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link a file (copy on failure), so the output survives removal of the source."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_text(path: Path, text: str) -> None:
    """Write a file without writing through a hard link into the source checkpoint."""
    path.unlink(missing_ok=True)
    path.write_text(text)


def _copy_auxiliary(source: Path, staging: Path) -> None:
    for entry in source.iterdir():
        if entry.name in _SKIP_ROOT_ENTRIES:
            continue
        dest = staging / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, copy_function=_link_or_copy)
        else:
            _link_or_copy(entry, dest)


def export_transfer_auto_fp8(
    source: str | Path,
    output: str | Path,
    activation_amax: dict[str, float] | str | Path | Sequence[str | Path],
    *,
    weight_scales: dict[str, float] | str | Path | None = None,
    device: str = "cuda",
    mixed_precision_steps: int = 3,
    recipe: dict | None = None,
) -> dict:
    """Stream the BF16 ``source`` shards into a Transfer-Auto FP8 checkpoint at ``output``.

    For each of the 504 projections the export writes the weight as E4M3, a scalar FP32
    ``weight_scale`` and a scalar FP32 ``input_scale = amax / 448``. The weight scale is
    ``max|W| / 448`` unless ``weight_scales`` (a map or a JSON path from :func:`search_weight_scales`)
    holds the weight key. All other tensors and files copy without change. The output also holds
    ``hf_quant_config.json``, ``transformer/config.json`` with ``quantization_config`` and
    ``runtime.diffusion_step_policy`` (A16 in the first and last ``mixed_precision_steps`` steps and
    in the reasoner), both weight indexes and ``quantization_metadata.json``.

    Source must be a pristine BF16 checkpoint. ``output`` must not exist; it is written to
    ``<output>.partial`` and renamed at the end. Returns the metadata dict.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    staging = output.with_name(output.name + ".partial")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must not overlap")
    if not isinstance(activation_amax, dict):
        paths = [activation_amax] if isinstance(activation_amax, (str, Path)) else list(activation_amax)
        activation_amax = load_activation_amax(paths)
    if isinstance(weight_scales, (str, Path)):
        weight_scales = json.loads(Path(weight_scales).read_text())
    weight_scales = {
        key: _positive_finite(value, f"weight scale of {key}") for key, value in (weight_scales or {}).items()
    }

    config = json.loads((source / "transformer/config.json").read_text())
    if config.get("quantization_config") or (source / "hf_quant_config.json").exists():
        raise ValueError("Expected a pristine BF16 source checkpoint")
    shards = _source_shards(source)
    selected = projection_map(_source_keys(shards), config["num_hidden_layers"], "both")
    for key in selected:
        value = activation_amax.get(key)
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid activation amax for {key}")
    unknown = sorted(set(weight_scales) - selected.keys())
    if unknown:
        raise ValueError(f"Weight scales for non-quantized tensors, e.g. {unknown[:3]}")

    staging.mkdir(parents=True)
    _copy_auxiliary(source, staging)
    (staging / "transformer").mkdir()
    for entry in (source / "transformer").iterdir():
        if entry.suffix != ".safetensors" and entry.name != "diffusion_pytorch_model.safetensors.index.json":
            _link_or_copy(entry, staging / "transformer" / entry.name)

    coverage = []
    with torch.no_grad():
        for shard in shards:
            tensors: dict[str, torch.Tensor] = {}
            with safe_open(shard, framework="pt") as handle:
                for key in handle.keys():
                    weight = handle.get_tensor(key)
                    if key not in selected:
                        tensors[key] = weight
                        continue
                    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
                        raise ValueError(f"Expected pristine BF16 matrix at {key}")
                    dense = weight.to(device=device, dtype=torch.float32)
                    if not torch.isfinite(dense).all():
                        raise ValueError(f"Nonfinite source weight at {key}")
                    if key in weight_scales:
                        # Divide by the Python float, not by a tensor: PyTorch multiplies by the
                        # reciprocal of a scalar divisor, and the published checkpoint used that path.
                        divisor = weight_scales[key]
                        scale = torch.tensor(divisor, dtype=torch.float32, device=device)
                    else:
                        scale = divisor = _weight_scale_from_max(dense)
                    prefix = key.removesuffix(".weight")
                    tensors[key] = (
                        (dense / divisor).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).cpu().contiguous()
                    )
                    tensors[prefix + ".weight_scale"] = scale.float().cpu().reshape(())
                    tensors[prefix + ".input_scale"] = torch.tensor(
                        _input_scale(activation_amax[key]), dtype=torch.float32
                    )
                    coverage.append(
                        {"source_weight": key, "runtime_module": selected[key], "shape": list(weight.shape)}
                    )
                    del dense
            save_file(tensors, staging / "transformer" / shard.name, metadata={"format": "pt"})
            print(f"[export] {shard.name}: {len(tensors)} tensors", flush=True)
            del tensors

    quant_config = {
        "quant_method": "modelopt",
        "quant_algo": "FP8",
        "ignore": list(IGNORE_PATTERNS),
        "quantized_layers": {name: {"quant_algo": "FP8"} for name in selected.values()},
        "producer": dict(PRODUCER),
    }
    # The step policy lives in transformer/config.json only; the root file omits it (as upstream).
    config["quantization_config"] = copy.deepcopy(quant_config)
    add_diffusion_step_policy(config, mixed_precision_steps)
    _write_text(staging / "transformer/config.json", json.dumps(config, indent=2))
    _write_text(staging / "hf_quant_config.json", json.dumps(quant_config, indent=2))
    index = SafetensorsIndex().update_dir(staging / "transformer", "")
    _write_text(staging / "transformer/diffusion_pytorch_model.safetensors.index.json", index.to_json())
    _write_text(
        staging / "model.safetensors.index.json", build_root_index(staging, components=("transformer",)).to_json()
    )

    metadata = collect_quantization_metadata(
        source=source,
        input_dir=source,
        recipe={
            "name": RECIPE_NAME,
            "format": "E4M3 static per-tensor symmetric",
            "weight_scale": "max|W|/448, or the searched value from candidate_changes.weight_scales",
            "input_scale": "activation_amax/448",
            "mixed_precision_steps": mixed_precision_steps,
            **(recipe or {}),
        },
        prompts=[],
    )
    metadata.update(
        scope="both",
        quantized_count=len(coverage),
        total_transformer_bytes=index.metadata["total_size"],
        activation_amax=activation_amax,
        tensor_coverage=coverage,
        candidate_changes={"weight_scales": weight_scales},
    )
    _write_text(staging / METADATA_FILENAME, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    staging.rename(output)
    print(f"[export] done: {output} ({len(coverage)} projections, {len(weight_scales)} searched weight scales)")
    return metadata


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="activation amax from native generation, or merge amax files")
    cal.add_argument("--checkpoint", type=Path, help="BF16 source checkpoint")
    cal.add_argument("--requests", type=Path, nargs="+", help="request JSONL files")
    cal.add_argument("--merge", type=Path, nargs="+", help="amax JSON files to join by elementwise max")
    cal.add_argument("--output", type=Path, required=True, help="amax JSON file to write")
    cal.add_argument("--output-dir", type=Path, help="directory for generated videos")

    srch = sub.add_parser("search", help="weight-scale search on captured projection inputs")
    srch.add_argument("--source", type=Path, required=True)
    srch.add_argument("--activation-amax", type=Path, nargs="+", required=True)
    srch.add_argument("--captures", type=Path, nargs="+", required=True)
    srch.add_argument("--output", type=Path, required=True)
    srch.add_argument("--device", default="cuda")

    exp = sub.add_parser("export", help="write the FP8 checkpoint from the BF16 source")
    exp.add_argument("--source", type=Path, required=True)
    exp.add_argument("--activation-amax", type=Path, nargs="+", required=True)
    exp.add_argument("--weight-scales", type=Path)
    exp.add_argument("--output", type=Path, required=True)
    exp.add_argument("--device", default="cuda")
    exp.add_argument("--mixed-precision-steps", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        if args.merge:
            amax = load_activation_amax(args.merge)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {"activation_amax": amax, "algorithm": "elementwise max", "merged": [str(p) for p in args.merge]},
                    indent=2,
                )
            )
        elif args.checkpoint and args.requests:
            calibrate_transfer_auto(args.checkpoint, args.requests, args.output, output_dir=args.output_dir)
        else:
            parser.error("calibrate needs --checkpoint and --requests, or --merge")
    elif args.command == "search":
        search_weight_scales(args.source, args.activation_amax, args.captures, args.output, device=args.device)
    else:
        export_transfer_auto_fp8(
            args.source,
            args.output,
            args.activation_amax,
            weight_scales=args.weight_scales,
            device=args.device,
            mixed_precision_steps=args.mixed_precision_steps,
        )


if __name__ == "__main__":
    main()
