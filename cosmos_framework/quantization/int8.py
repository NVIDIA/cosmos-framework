# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export a Cosmos3-Nano-Transfer-Auto INT8 W8A8 checkpoint for vLLM-Omni.

The format is weight-only on disk: symmetric INT8 weights with one FP32 scale per
output channel. Activations are quantized per token at run time by vLLM-Omni
(``DiffusionInt8Config`` with ``is_checkpoint_int8_serialized=True`` and
``activation_scheme="dynamic"``), so the checkpoint holds no ``input_scale`` and
no calibration is needed. The export is a streaming weight conversion: one source
shard is read, converted and written at a time.

Quantized linears (504 for the 36-layer model; same set as the FP8 recipe):

* understanding tower: ``layers.{i}.self_attn.{to_q,to_k,to_v,to_out}`` and
  ``layers.{i}.mlp.{gate_proj,up_proj,down_proj}``
* generation tower: ``layers.{i}.self_attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}``
  and ``layers.{i}.mlp_moe_gen.{gate_proj,up_proj,down_proj}``

Every other tensor is copied without change.

Variants (ADR 0003). ``dyn`` is the default above. Two options extend it:

* ``activation_amax``: static per-tensor activations. Each linear gets an FP32 scalar
  ``input_scale = amax / 127`` and ``activation_scheme`` is ``"static"``.
* ``smoothquant=(channel_amax, alpha)``: ``s_c = max|X_c|^alpha / max|W[:,c]|^(1-alpha)``. The
  weight ``W * s`` is quantized per output channel and the FP32 ``pre_quant_scale = 1 / s`` ([K])
  is saved. The runtime multiplies the activation by it before quantization.

:func:`collect_channel_amax` records the per-input-channel activation absmax that SmoothQuant needs.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .metadata import collect_quantization_metadata, write_quantization_metadata
from .safetensors_index import SafetensorsIndex, build_root_index

INT8_MAX = 127
SMOOTH_CLAMP = (1e-5, 1e5)
SCOPES = ("both", "und", "gen")

TRANSFORMER_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"

# Checkpoint suffix -> vLLM-Omni module name, for each tower. vLLM-Omni builds the
# understanding tower as ``language_model.layers.{i}`` and the generation tower as
# ``gen_layers.{i}``; ``DiffusionInt8Config`` matches ``ignored_layers`` against these
# module prefixes (see ``Cosmos3MultiviewPipeline._remap_ckpt_key``).
_UND_LINEARS = {
    "self_attn.to_q": "self_attn.to_q",
    "self_attn.to_k": "self_attn.to_k",
    "self_attn.to_v": "self_attn.to_v",
    "self_attn.to_out": "self_attn.to_out",
    "mlp.gate_proj": "mlp.gate_proj",
    "mlp.up_proj": "mlp.up_proj",
    "mlp.down_proj": "mlp.down_proj",
}
_GEN_LINEARS = {
    "self_attn.add_q_proj": "cross_attention.to_q",
    "self_attn.add_k_proj": "cross_attention.to_k",
    "self_attn.add_v_proj": "cross_attention.to_v",
    "self_attn.to_add_out": "cross_attention.to_out",
    "mlp_moe_gen.gate_proj": "mlp.gate_proj",
    "mlp_moe_gen.up_proj": "mlp.up_proj",
    "mlp_moe_gen.down_proj": "mlp.down_proj",
}


def list_transfer_auto_linears(num_layers: int) -> dict[str, dict[str, str]]:
    """Return every vLLM-Omni ``LinearBase`` of the transformer.

    Maps the checkpoint weight prefix (without ``.weight``) to
    ``{"tower": "und" | "gen", "module": <vLLM-Omni module prefix>}``.
    """
    linears: dict[str, dict[str, str]] = {}
    for i in range(num_layers):
        for suffix, module in _UND_LINEARS.items():
            linears[f"layers.{i}.{suffix}"] = {"tower": "und", "module": f"language_model.layers.{i}.{module}"}
        for suffix, module in _GEN_LINEARS.items():
            linears[f"layers.{i}.{suffix}"] = {"tower": "gen", "module": f"gen_layers.{i}.{module}"}
    return linears


def quantize_weight_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 quantization of a ``[N, K]`` weight.

    ``scale = max|W[n, :]| / 127`` in FP32 (1.0 for an all-zero row), saved as FP32.
    ``q = clamp(round(W * (127 / max|W[n, :]|)), -127, 127)``; ``torch.round`` rounds
    half to even. The multiply by ``127 / amax`` (not ``W / scale``) is the arithmetic
    of vLLM's ``scaled_int8_quant`` kernel, which vLLM-Omni online INT8 uses. The two
    forms differ on about 0.03% of the values (exact ties such as ``amax / 2``), so this
    form is needed for the offline checkpoint to equal online INT8 bit for bit.
    Returns ``(int8 [N, K], fp32 scale [N, 1])``.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}.")
    w32 = weight.to(torch.float32)
    amax = w32.abs().amax(dim=1, keepdim=True)
    zero = amax == 0
    scale = torch.where(zero, torch.ones_like(amax), amax / INT8_MAX)
    inv = torch.full_like(amax, float(INT8_MAX)) / torch.where(zero, torch.ones_like(amax), amax)
    q = torch.round(w32 * inv).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def smoothquant_scale(channel_amax: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """SmoothQuant factor ``s_c = max|X_c|^alpha / max|W[:,c]|^(1-alpha)`` ([K], FP32).

    ``channel_amax`` is the per-input-channel activation absmax ([K]), ``weight`` is ``[N, K]``.
    The result is clamped to ``[1e-5, 1e5]``. A channel with zero activation and zero weight gets 1.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    x = channel_amax.to(torch.float32)
    w = weight.to(torch.float32).abs().amax(dim=0)
    if x.shape != w.shape:
        raise ValueError(f"channel_amax has shape {tuple(x.shape)}; the weight has {tuple(w.shape)} input channels.")
    tiny = 1e-12
    s = x.clamp_min(tiny).pow(alpha) / w.clamp_min(tiny).pow(1.0 - alpha)
    s = torch.where((x == 0) & (w == 0), torch.ones_like(s), s)
    return s.clamp_(*SMOOTH_CLAMP)


def smoothed_activation_amax(channel_amax: torch.Tensor, s: torch.Tensor) -> float:
    """Per-tensor amax of the smoothed input ``x / s``: ``max_c(channel_amax_c / s_c)``."""
    return float((channel_amax.to(torch.float32) / s.to(torch.float32)).amax())


def load_activation_amax(path: str | Path) -> dict[str, float]:
    """Read the ``activation_amax`` map ``{weight_key: amax}`` of a ``quantization_metadata.json``."""
    amax = json.loads(Path(path).read_text())["activation_amax"]
    out = {key: float(value) for key, value in amax.items()}
    for key, value in out.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid activation amax for {key}: {value}")
    return out


def load_channel_amax(path: str | Path) -> dict[str, torch.Tensor]:
    """Read the ``channel_amax`` map written by :func:`collect_channel_amax` as FP32 tensors."""
    data = json.loads(Path(path).read_text())["channel_amax"]
    return {key: torch.tensor(values, dtype=torch.float32) for key, values in data.items()}


def build_diffusion_step_policy(mixed_precision_steps: int = 3, reasoner: str = "a16") -> dict | None:
    """The MAPS block ``quantization_config.runtime.diffusion_step_policy``.

    Same JSON as in the FP8 checkpoint. Returns ``None`` (no policy) for 0 steps and a native reasoner.
    """
    if mixed_precision_steps == 0 and reasoner == "native":
        return None
    return {
        "schema_version": 1,
        "type": "first_last_n",
        "index_space": "denoising_loop_iteration",
        "scope": ["transformer"],
        "default_mode": "native",
        "first_steps": {"count": mixed_precision_steps, "mode": "a16"},
        "last_steps": {"count": mixed_precision_steps, "mode": "a16"},
        "overlap": "a16",
        "reasoner": reasoner,
    }


def build_int8_quantization_config(
    ignored_layers: list[str],
    static: bool = False,
    smoothquant_alpha: float | None = None,
    mixed_precision_steps: int = 0,
    reasoner: str = "native",
) -> dict:
    """The ``quantization_config`` block that vLLM-Omni loads as serialized INT8.

    vLLM-Omni passes every key except ``quant_method`` to the ``DiffusionInt8Config``
    constructor. The ``dyn`` block has exactly the keys below. ``static`` changes
    ``activation_scheme``; SmoothQuant adds ``smoothquant``. The MAPS policy (see
    :func:`build_diffusion_step_policy`) goes under ``runtime``; the defaults write none.
    """
    config = {
        "quant_method": "int8",
        "is_checkpoint_int8_serialized": True,
        "activation_scheme": "static" if static else "dynamic",
        "ignored_layers": list(ignored_layers),
    }
    if smoothquant_alpha is not None:
        config["smoothquant"] = {"alpha": float(smoothquant_alpha)}
    policy = build_diffusion_step_policy(mixed_precision_steps, reasoner)
    if policy is not None:
        config["runtime"] = {"diffusion_step_policy": policy}
    return config


def _convert_shard(
    src: Path,
    dst: Path,
    selected: set[str],
    device: str,
    activation_amax: dict[str, float] | None = None,
    smoothquant: tuple[dict[str, torch.Tensor], float] | None = None,
    static: bool = False,
    static_amax_out: dict[str, float] | None = None,
) -> list[str]:
    """Convert one shard. Returns the list of quantized weight prefixes it held."""
    out: dict[str, torch.Tensor] = {}
    done: list[str] = []
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open handle has no __iter__
            tensor = f.get_tensor(key)
            prefix = key[: -len(".weight")] if key.endswith(".weight") else None
            if prefix in selected:
                if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f"{key} has dtype {tensor.dtype}; the source must be a BF16 checkpoint.")
                w = tensor.to(device)
                amax_value = activation_amax[key] if activation_amax is not None else None
                if smoothquant is not None:
                    channel_amax, alpha = smoothquant
                    if key not in channel_amax:
                        raise ValueError(f"No channel amax for {key}.")
                    ch = channel_amax[key].to(device)
                    s = smoothquant_scale(ch, w, alpha)
                    # Scale in FP32 (the quantizer upcasts anyway), so BF16 rounding does not enter.
                    w = w.to(torch.float32) * s
                    out[f"{prefix}.pre_quant_scale"] = (1.0 / s).cpu()
                    if static:
                        amax_value = smoothed_activation_amax(ch, s)
                q, scale = quantize_weight_int8(w)
                out[key] = q.cpu()
                out[f"{prefix}.weight_scale"] = scale.cpu()
                if static:
                    if amax_value is None:
                        raise ValueError(f"No activation amax for {key}.")
                    out[f"{prefix}.input_scale"] = torch.tensor(amax_value / INT8_MAX, dtype=torch.float32)
                    if static_amax_out is not None:
                        static_amax_out[key] = float(amax_value)
                done.append(prefix)
            else:
                if key.endswith((".weight_scale", ".input_scale", ".pre_quant_scale")):
                    raise ValueError(f"{key} found in the source; the source is already quantized.")
                out[key] = tensor
    save_file(out, str(dst), metadata={"format": "pt"})
    return done


def _link_source_entries(source: Path, output: Path, skip: set[str]) -> None:
    """Link every top-level entry of ``source`` (vae, tokenizers, ...) into ``output``."""
    source_abs = source.absolute()
    for entry in sorted(source.iterdir()):
        if entry.name in skip or entry.name.startswith("."):
            continue
        dst = output / entry.name
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(os.path.relpath(source_abs / entry.name, start=output.resolve()))


def collect_channel_amax(
    checkpoint_path: str | Path,
    input_files: Sequence[str | Path],
    output_json: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, list[float]]:
    """Record the per-input-channel activation absmax of the 504 projections (SmoothQuant input).

    Runs native ``OmniInference.generate`` on the request JSONL files in ``input_files`` (7 cameras,
    all denoising steps, eager and uncached) with a forward pre-hook on each projection. The hook
    keeps ``max |x|`` over all tokens and all calls, for each of the K input channels. Writes
    ``{"channel_amax": {weight_key: [K floats]}, ...metadata}`` to ``output_json``. Needs a GPU.
    """
    # init_script must run before the model implementation is imported.
    from cosmos_framework.inference.common.init import init_script

    init_script()
    import tempfile

    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.args import sample_output_dir
    from cosmos_framework.inference.common.init import init_output_dir
    from cosmos_framework.inference.model import _diffusers_to_net_key

    checkpoint_path = Path(checkpoint_path)
    config = json.loads((checkpoint_path / "transformer/config.json").read_text())
    linears = list_transfer_auto_linears(int(config["num_hidden_layers"]))
    native = {
        f"{prefix}.weight": _diffusers_to_net_key(f"{prefix}.weight", "transformer/weights.safetensors").removesuffix(
            ".weight"
        )
        for prefix in linears
    }

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="transfer-auto-channel-amax-"))
    # Eager, uncached runtime: the diffusion cache skips denoiser calls, and compilation or CUDA graphs
    # can bypass the hooks. Every denoising step must reach them.
    setup_args = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_path),
        output_dir=out_dir,
        diffusion_cache=False,
        use_torch_compile=False,
        use_cuda_graphs=False,
        guardrails=False,
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

    amax: dict[str, torch.Tensor] = {}
    calls: dict[str, int] = dict.fromkeys(native, 0)

    def make_hook(key: str):
        def hook(_module, args):
            x = args[0].detach()
            cur = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).to(torch.float32)
            prev = amax.get(key)
            amax[key] = cur if prev is None else torch.maximum(prev, cur)
            calls[key] += 1

        return hook

    handles = [modules[name].register_forward_pre_hook(make_hook(key)) for key, name in native.items()]
    print(f"[amax] hooks on {len(handles)} linears, {len(samples)} native requests", flush=True)
    try:
        results = pipe.generate(samples)
        if not results or any(str(item.status) != "success" for item in results):
            raise RuntimeError(f"Generation failed: {[str(item.status) for item in results]}")
    finally:
        for handle in handles:
            handle.remove()

    missing = [key for key, n in calls.items() if n == 0]
    if missing:
        raise RuntimeError(f"{len(missing)} projections were never called, e.g. {missing[:3]}")
    values = {key: amax[key].cpu().tolist() for key in native}
    for key, row in values.items():
        if not all(math.isfinite(v) for v in row):
            raise ValueError(f"Nonfinite channel amax for {key}")
    record = {
        "channel_amax": values,
        "algorithm": "per-input-channel absmax over all tokens and calls, forward pre-hook on native OmniInference.generate",
        "checkpoint": str(checkpoint_path),
        "requests": [str(p) for p in input_files],
        "samples": [sample.name for sample in samples],
        "calls_per_projection": {"min": min(calls.values()), "max": max(calls.values())},
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(record))
    print(f"[amax] wrote {output_json}", flush=True)
    return values


def export_transfer_auto_int8(
    source: str | Path,
    output: str | Path,
    scope: str = "both",
    device: str = "cpu",
    *,
    activation_amax: dict[str, float] | None = None,
    smoothquant: tuple[dict[str, torch.Tensor], float] | None = None,
    static_activations: bool = False,
    mixed_precision_steps: int = 3,
    reasoner: str = "a16",
) -> dict:
    """Write an INT8 W8A8 checkpoint of Cosmos3-Nano-Transfer-Auto.

    Args:
        source: Local BF16 diffusers-layout checkpoint directory (has ``transformer/``).
        output: Output directory. ``transformer/`` is new; other entries link to ``source``.
        scope: ``"both"`` quantizes both towers (the released recipe). ``"und"`` or
            ``"gen"`` quantizes one tower; the linears of the other stay BF16 and
            are listed in ``ignored_layers``.
        device: Device for the quantization math (``"cpu"`` or ``"cuda"``). CPU and CUDA
            can differ by 1 ulp in ``weight_scale``. Use ``"cuda"`` to match vLLM-Omni
            online INT8 quantization bit for bit.
        activation_amax: ``{weight_key: amax}``. When given, each quantized linear gets
            ``input_scale = amax / 127`` (FP32 scalar) and ``activation_scheme`` is ``"static"``.
        smoothquant: ``(channel_amax, alpha)`` with ``channel_amax = {weight_key: [K] tensor}``.
            Quantizes ``W * s`` and saves ``pre_quant_scale = 1 / s``.
        static_activations: static scheme without ``activation_amax``. Only valid with
            ``smoothquant``; then the amax is ``max_c(channel_amax_c / s_c)``. With both
            ``smoothquant`` and ``activation_amax``, the smoothed amax is used and
            ``activation_amax`` is ignored.
        mixed_precision_steps: MAPS policy. The first and the last N denoising steps run with
            16-bit activations. The policy goes into ``quantization_config.runtime`` (same
            schema as the FP8 checkpoint), so the server needs no ``--additional-config``.
        reasoner: ``"a16"`` or ``"native"``: the mode of the understanding tower in the MAPS
            policy. ``mixed_precision_steps=0`` with ``reasoner="native"`` writes no policy.

    Returns:
        The ``recipe`` dict that is recorded in ``quantization_metadata.json``.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}.")
    if mixed_precision_steps < 0:
        raise ValueError(f"mixed_precision_steps must be >= 0, got {mixed_precision_steps}.")
    if reasoner not in ("native", "a16"):
        raise ValueError(f"reasoner must be 'native' or 'a16', got {reasoner!r}.")
    static = activation_amax is not None or static_activations
    if static and activation_amax is None and smoothquant is None:
        raise ValueError("static_activations needs smoothquant or activation_amax.")
    source, output = Path(source), Path(output)
    src_transformer = source / "transformer"
    config_path = src_transformer / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}; the source must be a diffusers-layout checkpoint.")
    shards = sorted(src_transformer.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards in {src_transformer}.")
    config = json.loads(config_path.read_text())
    if "quantization_config" in config:
        raise ValueError(f"{config_path} already has a quantization_config; the source must be BF16.")

    linears = list_transfer_auto_linears(int(config["num_hidden_layers"]))
    selected = {p for p, info in linears.items() if scope == "both" or info["tower"] == scope}
    ignored_layers = [info["module"] for p, info in linears.items() if p not in selected]

    out_transformer = output / "transformer"
    if out_transformer.exists():
        shutil.rmtree(out_transformer)
    out_transformer.mkdir(parents=True)

    if activation_amax is not None and smoothquant is None:
        absent = sorted(p for p in selected if f"{p}.weight" not in activation_amax)
        if absent:
            raise ValueError(f"activation_amax has no entry for {len(absent)} linears, e.g. {absent[:3]}.")
    quantized: list[str] = []
    static_amax: dict[str, float] = {}
    for n, shard in enumerate(shards, start=1):
        print(f"[int8] shard {n}/{len(shards)}: {shard.name}")
        quantized += _convert_shard(
            shard, out_transformer / shard.name, selected, device, activation_amax, smoothquant, static, static_amax
        )

    missing = selected - set(quantized)
    if missing or len(quantized) != len(set(quantized)):
        raise ValueError(f"Expected each of {len(selected)} linears once; missing {sorted(missing)[:3]}.")
    print(f"[int8] quantized {len(quantized)} linears")

    SafetensorsIndex().update_dir(out_transformer, "").write(out_transformer / TRANSFORMER_INDEX_NAME)
    config["quantization_config"] = build_int8_quantization_config(
        ignored_layers,
        static=static,
        smoothquant_alpha=smoothquant[1] if smoothquant else None,
        mixed_precision_steps=mixed_precision_steps,
        reasoner=reasoner,
    )
    (out_transformer / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    output.mkdir(parents=True, exist_ok=True)
    skip = {"transformer", "model.safetensors.index.json", "hf_quant_config.json", "quantization_metadata.json"}
    _link_source_entries(source, output, skip)
    root_index = output / "model.safetensors.index.json"
    if root_index.is_symlink():
        root_index.unlink()
    build_root_index(output).write(root_index)

    recipe = {
        "format": "int8_w8a8",
        "weights": "symmetric per-output-channel INT8, scale = max|W[n,:]|/127 (FP32), q = round(W * (127/max|W[n,:]|)) half to even, clamp +-127",
        "variant": ("sq-" if smoothquant else "") + ("static" if static else "dyn"),
        "activations": (
            "static per-tensor INT8, input_scale = amax/127 (FP32 scalar)"
            if static
            else "dynamic per-token INT8 at run time (no input_scale in the checkpoint)"
        ),
        "scope": scope,
        "quantized_linears": len(quantized),
        "ignored_layers": ignored_layers,
        "quantization_config": config["quantization_config"],
    }
    if smoothquant:
        recipe["smoothquant"] = {
            "alpha": smoothquant[1],
            "formula": "s_c = max|X_c|^alpha / max|W[:,c]|^(1-alpha), clamp [1e-5, 1e5]; weight = W*s; pre_quant_scale = 1/s",
        }
    if static:
        recipe["activation_amax"] = static_amax
    metadata = collect_quantization_metadata(source=source, input_dir=source, recipe=recipe, prompts=[])
    write_quantization_metadata(output, metadata)
    print(f"[int8] checkpoint ready at {output}")
    return recipe


def main() -> None:
    """``python -m cosmos_framework.quantization.int8 [channel-amax] ...``. Without a command: export."""
    import argparse
    import sys

    if sys.argv[1:2] == ["channel-amax"]:
        parser = argparse.ArgumentParser(
            prog="int8 channel-amax", description="Collect per-input-channel absmax (GPU)."
        )
        parser.add_argument("--checkpoint", required=True, help="BF16 checkpoint directory")
        parser.add_argument("--requests", nargs="+", required=True, help="request JSONL files")
        parser.add_argument("--output", required=True, help="output JSON")
        parser.add_argument("--output-dir", default=None, help="directory for the generated videos")
        args = parser.parse_args(sys.argv[2:])
        collect_channel_amax(args.checkpoint, args.requests, args.output, output_dir=args.output_dir)
        return

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="BF16 Cosmos3-Nano-Transfer-Auto checkpoint directory")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--scope", choices=SCOPES, default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--activation-amax", default=None, help="quantization_metadata.json with activation_amax: static"
    )
    parser.add_argument("--channel-amax", default=None, help="channel_amax.json: SmoothQuant")
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--static", action="store_true", help="static activations from the smoothed channel amax")
    parser.add_argument(
        "--mixed-precision-steps", type=int, default=3, help="MAPS: first and last N steps in 16-bit activations"
    )
    parser.add_argument("--reasoner", choices=("native", "a16"), default="a16", help="MAPS: reasoner mode")
    args = parser.parse_args()
    export_transfer_auto_int8(
        args.source,
        args.output,
        scope=args.scope,
        device=args.device,
        activation_amax=load_activation_amax(args.activation_amax) if args.activation_amax else None,
        smoothquant=(load_channel_amax(args.channel_amax), args.smoothquant_alpha) if args.channel_amax else None,
        static_activations=args.static,
        mixed_precision_steps=args.mixed_precision_steps,
        reasoner=args.reasoner,
    )


if __name__ == "__main__":
    main()
