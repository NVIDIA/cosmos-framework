# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export a framework model as a vllm-omni-loadable FP8 Diffusers checkpoint.

The calibrated :class:`~cosmos_framework.model.generator.omni_mot_model.OmniMoTModel`
owns its quantized denoiser under ``model.net``. Its FP8 weights and scales are mapped
back to the source Diffusers keys, then overlaid onto the full bf16 transformer. This
preserves non-quantized projections, modality towers, and the language-model head.
The resulting drop-in checkpoint has a new ``transformer/`` and links the remaining
components back to the quantization source directory.

FP8 only — the NIM pipeline's NVFP4 exporter is out of scope for the cookbook.
The exported transformer config includes the vLLM-Omni inference-time A16 policy.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import modelopt
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict
from modelopt.torch.export.unified_export_hf import _process_quantized_modules

from cosmos_framework.inference.model import _diffusers_to_net_key

from .checkpoint_io import load_sharded_safetensors, save_sharded_safetensors
from .safetensors_index import build_root_index


def collapse_input_amax_to_scalar(mdl) -> int:
    """Collapse per-channel ``input_quantizer.amax`` to scalar (max over channels).

    ModelOpt's ``_fuse_qkv_linears_diffusion`` asserts input quantizer amax is scalar.
    Per-tensor ``max`` calibration already yields scalars, so this is normally a no-op;
    the conservative max-collapse keeps the widest dynamic range if a per-channel
    quantizer ever slips through.
    """
    try:
        from modelopt.torch.quantization.nn import TensorQuantizer
    except Exception:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer  # type: ignore

    cnt = 0
    for name, module in mdl.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if "input_quantizer" not in name:
            continue
        if getattr(module, "amax", None) is None:
            continue
        if module.amax.numel() <= 1:
            continue
        with torch.no_grad():
            module.amax = module.amax.abs().max().reshape(()).clone()
        cnt += 1
    return cnt


def _build_net_to_diffusers_key_map(bf16_state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    """Invert the framework's Diffusers-load map for this source transformer.

    The source checkpoint is the authority for its Diffusers key set. Inverting the
    framework map against that set keeps FP8 export aligned with framework loading,
    including future changes to the MoT network's internal module names.
    """
    net_to_diffusers: dict[str, str] = {}
    for diffusers_key in bf16_state_dict:
        net_key = _diffusers_to_net_key(diffusers_key, "transformer/weights.safetensors")
        if net_key is None:
            continue
        previous = net_to_diffusers.setdefault(net_key, diffusers_key)
        if previous != diffusers_key:
            raise KeyError(
                "Multiple Diffusers keys map to one framework network key: "
                f"{previous!r} and {diffusers_key!r} -> {net_key!r}."
            )
    return net_to_diffusers


def remap_framework_state_dict(
    state_dict: dict[str, torch.Tensor], bf16_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Map ModelOpt-exported ``OmniMoTModel.net`` tensors to Diffusers keys."""
    net_to_diffusers = _build_net_to_diffusers_key_map(bf16_state_dict)
    remapped: dict[str, torch.Tensor] = {}

    for net_key, value in state_dict.items():
        diffusers_key = net_to_diffusers.get(net_key)
        if diffusers_key is None:
            for suffix in _SCALE_SUFFIXES:
                if not net_key.endswith(suffix):
                    continue
                weight_key = net_key[: -len(suffix)] + ".weight"
                weight_diffusers_key = net_to_diffusers.get(weight_key)
                if weight_diffusers_key is not None:
                    diffusers_key = weight_diffusers_key[: -len(".weight")] + suffix
                break
        if diffusers_key is None:
            continue
        if diffusers_key in remapped:
            raise KeyError(f"Multiple framework tensors map to Diffusers key {diffusers_key!r}.")
        remapped[diffusers_key] = value
    return remapped


# Top-level projections produced by reverse_remap use the cosmos3_vfm names; the
# bf16 diffusers base provides the equivalently-named proj_in/proj_out, so we drop
# ours and keep the base copies.
_DROP_FROM_QUANTIZED = ("vae2llm.", "llm2vae.")
_FP8_DTYPES = {torch.float8_e4m3fn, getattr(torch, "float8_e5m2", torch.float8_e4m3fn)}
FP8_MAX = 448.0
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale", ".input_scale_2", ".alpha", ".pre_quant_scale")

# Linears the diffusers checkpoint keeps in bf16 (never quantized), emitted in the
# ModelOpt ``ignore`` list so a reasoning-path loader (vllm Cosmos3ForConditionalGeneration)
# does not FP8-quantize the vision tower / projections it can't supply scales for.
_DIFFUSERS_IGNORE_MODULES = [
    "proj_in",
    "proj_out",
    "time_embedder*",
    "audio_proj_in",
    "audio_proj_out",
    "action_proj_in",
    "action_proj_out",
    "lm_head",
    "model.visual*",
    "visual*",
]


def add_diffusion_step_policy(config: dict, mixed_precision_steps: int) -> None:
    """Add vLLM-Omni's symmetric first/last-step A16 policy in place.

    The policy belongs to the Diffusers transformer's ``quantization_config``;
    root-level Transformers metadata intentionally does not carry it.
    """
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("Transformer config must contain a 'quantization_config' object before adding the policy.")

    quantization_config["runtime"] = {
        "diffusion_step_policy": {
            "schema_version": 1,
            "type": "first_last_n",
            "index_space": "denoising_loop_iteration",
            "scope": ["transformer"],
            "default_mode": "native",
            "first_steps": {"count": mixed_precision_steps, "mode": "a16"},
            "last_steps": {"count": mixed_precision_steps, "mode": "a16"},
            "overlap": "a16",
            "reasoner": "a16",
        }
    }


class Fp8DiffusersExporter:
    """Wrap a modelopt-FP8-quantized cosmos3 DiT as a vllm-omni diffusers checkpoint.

    Materializes the quantized tensors (fuse QKV + process + reverse-remap), overlays
    them onto the full bf16 diffusers transformer (which supplies proj_in/proj_out,
    the audio/action towers, lm_head and the LM final norm), and stamps a ModelOpt
    FP8 ``quantization_config``. FP8 weights keep the bf16 shape (dtype-only change).
    """

    quant_format = "fp8"

    def build_quant_config_json(self) -> dict:
        modelopt_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        modelopt_config["algorithm"] = "max"
        modelopt_config["quant_cfg"][1]["cfg"]["fake_quant"] = False
        for name in _DIFFUSERS_IGNORE_MODULES:
            modelopt_config["quant_cfg"].append({"quantizer_name": f"*{name}*", "enable": False})
        return {
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "targets": ["Linear"],
                }
            },
            "ignore": list(_DIFFUSERS_IGNORE_MODULES),
            "quant_algo": "FP8",
            "producer": {"name": "modelopt", "version": modelopt.__version__},
            "quant_method": "modelopt",
            "quant_type": "FP8_FP8",
            "weight_only": False,
            "modules_to_not_convert": list(_DIFFUSERS_IGNORE_MODULES),
            "modelopt_config": modelopt_config,
        }

    def write_diffusers_modelopt_state(self, config: dict, state_dict: dict, export_dir: Path) -> None:
        """Write the graph state Diffusers needs before loading pre-quantized weights."""
        import diffusers

        model_config = dict(config)
        model_config.pop("quantization_config", None)
        with torch.device("meta"):
            transformer = diffusers.Cosmos3OmniTransformer.from_config(model_config)

        quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        quant_cfg["algorithm"] = None
        mtq.quantize(transformer, quant_cfg)

        active = {key.removesuffix(".weight_scale") for key in state_dict if key.endswith(".weight_scale")}
        mtq.disable_quantizer(
            transformer,
            lambda name: name.rsplit(".", 1)[0] not in active,
        )
        for name in active:
            module = transformer.get_submodule(name)
            module.weight_quantizer.amax = state_dict[f"{name}.weight_scale"].float().cpu() * FP8_MAX
            input_scale = state_dict.get(f"{name}.input_scale")
            if input_scale is not None:
                module.input_quantizer.amax = input_scale.float().cpu() * FP8_MAX

        torch.save(mto.modelopt_state(transformer), export_dir / "modelopt_state.pth")
        print(f"[export] wrote Diffusers ModelOpt graph state for {len(active)} FP8 linears")

    def write_transformers_modelopt_state(self, state: dict, export_dir: Path) -> None:
        """Write ModelOpt graph state using the Transformers model's module names."""
        modes = {name: mode_state for name, mode_state in state["modelopt_state_dict"]}
        quantizer_state = modes["quantize"]["metadata"]["quantizer_state"]
        remapped = {}
        for name, value in quantizer_state.items():
            if name.startswith("net.language_model.model.layers.") and "_moe_gen." not in name:
                new_name = name.replace(
                    "net.language_model.model.layers.",
                    "model.language_model.layers.",
                    1,
                )
            elif name.startswith("net.language_model.visual."):
                new_name = name.replace("net.language_model.visual.", "model.visual.", 1)
            elif name.startswith("net.language_model.lm_head."):
                new_name = name.replace("net.language_model.lm_head.", "lm_head.", 1)
            else:
                continue
            remapped[new_name] = value
        disabled_state = next(
            copy.deepcopy(value) for name, value in remapped.items() if name.endswith("output_quantizer")
        )
        attention_names = {
            name.rsplit(".", 2)[0]
            for name in remapped
            if name.endswith(("self_attn.q_proj.input_quantizer", "attn.qkv.input_quantizer"))
        }
        for attention_name in attention_names:
            for quantizer_name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer", "softmax_quantizer"):
                remapped[f"{attention_name}.{quantizer_name}"] = copy.deepcopy(disabled_state)
        modes["quantize"]["config"] = self.build_quant_config_json()["modelopt_config"]
        for name, value in remapped.items():
            if name.endswith("weight_quantizer"):
                value["_fake_quant"] = False
            if name.startswith(("model.visual.", "lm_head.")):
                value["_disabled"] = True
                value["_pytorch_state_metadata"]["buffers"].clear()
            if name.endswith("input_quantizer") and not value["_disabled"]:
                value["_pytorch_state_metadata"]["buffers"]["_amax"] = {
                    "shape": torch.Size([]),
                    "dtype": torch.float32,
                }
        modes["quantize"]["metadata"]["quantizer_state"] = remapped

        real_quantize = modes["real_quantize"]
        for key in ("real_quantizer_state", "q_tensor_state"):
            real_quantize["metadata"][key] = {
                (
                    name.replace("net.language_model.model.layers.", "model.language_model.layers.", 1)
                    if name.startswith("net.language_model.model.layers.")
                    else "lm_head"
                ): value
                for name, value in real_quantize["metadata"][key].items()
                if (name.startswith("net.language_model.model.layers.") and "_moe_gen" not in name)
                or name == "net.language_model.lm_head"
            }
        for value in real_quantize["metadata"]["real_quantizer_state"].values():
            for buffer in value["_pytorch_state_metadata"]["buffers"].values():
                buffer["dtype"] = torch.float32
        state["modelopt_state_dict"] = [
            ("quantize", modes["quantize"]),
            ("real_quantize", real_quantize),
        ]
        torch.save(state, export_dir / "transformers_modelopt_state.pth")
        print(f"[export] wrote Transformers ModelOpt graph state for {len(remapped)} quantizers")

    def overlay(self, bf16: dict, quantized: dict) -> dict:
        merged = dict(bf16)
        n_weight = n_scale = n_dropped = 0
        quantized_modules: set[str] = set()
        for k, v in quantized.items():
            if k.startswith(_DROP_FROM_QUANTIZED):
                n_dropped += 1
                continue
            if k.endswith(_SCALE_SUFFIXES):
                merged[k] = v
                n_scale += 1
                continue
            if k.endswith(".weight") and v.dtype in _FP8_DTYPES:
                if k not in bf16:
                    raise KeyError(
                        f"quantized weight {k!r} has no bf16 counterpart — layer-key "
                        "schema mismatch; the overlay assumption is broken."
                    )
                if bf16[k].shape != v.shape:
                    raise ValueError(f"shape mismatch for {k}: bf16 {tuple(bf16[k].shape)} vs fp8 {tuple(v.shape)}")
                merged[k] = v
                quantized_modules.add(k[: -len(".weight")])
                n_weight += 1
                continue
            # Remaining bf16 tensors (norms, embed_tokens, renamed projections) are
            # identical to / superseded by the base — keep base.

        if n_weight == 0:
            raise ValueError("No FP8 framework network weights matched the source Diffusers transformer.")
        missing = [f"{m}.weight_scale" for m in quantized_modules if f"{m}.weight_scale" not in merged]
        if missing:
            raise ValueError(f"{len(missing)} quantized modules missing weight_scale, e.g. {missing[:3]}")
        for module in quantized_modules:
            weight_scale = merged[f"{module}.weight_scale"].float()
            input_scale = merged[f"{module}.input_scale"].float()
            merged[f"{module}.weight_quantizer._amax"] = weight_scale * FP8_MAX
            merged[f"{module}.weight_quantizer._scale"] = weight_scale.clone()
            merged[f"{module}.input_quantizer._amax"] = input_scale * FP8_MAX
        print(
            f"[export] overlaid {n_weight} fp8 weights + {n_scale} scales onto bf16 base "
            f"(dropped {n_dropped} vae2llm/llm2vae); {len(merged)} tensors total"
        )
        return merged

    def export(
        self,
        mdl,
        transformer_export_dir: Path,
        src_transformer_dir: Path,
        dtype=torch.bfloat16,
        mixed_precision_steps: int = 3,
    ) -> None:
        net = mdl.net
        transformer_export_dir.mkdir(parents=True, exist_ok=True)

        n_collapsed = collapse_input_amax_to_scalar(net)
        if n_collapsed > 0:
            print(f"[export] collapsed per-channel input amax to scalar on {n_collapsed} quantizers")

        # Record ModelOpt's real-quantized graph metadata before the unified
        # exporter unpacks those weights into plain FP8 checkpoint tensors.
        mtq.compress(mdl)
        transformers_modelopt_state = copy.deepcopy(mto.modelopt_state(mdl))

        # TODO: removed _fuse_qkv_linear_diffusion and dummy_forward due to
        # no-op for fp8. Add back when nvfp4 is supported.
        _process_quantized_modules(net, dtype)

        print(f"[export] loading bf16 base from {src_transformer_dir}")
        bf16 = load_sharded_safetensors(src_transformer_dir)

        with hide_quantizers_from_state_dict(net):
            net_state_dict = {k: v.detach().contiguous().cpu() for k, v in net.state_dict().items()}
        quantized = remap_framework_state_dict(net_state_dict, bf16)
        merged = self.overlay(bf16, quantized)
        n_shards = save_sharded_safetensors(merged, transformer_export_dir)
        print(f"[export] wrote {len(merged)} tensors across {n_shards} shard(s)")

        src_config = src_transformer_dir / "config.json"
        if not src_config.is_file():
            raise FileNotFoundError(f"Missing source config.json at {src_config}; cannot stamp quantization_config.")
        with open(src_config, encoding="utf-8") as f:
            config = json.load(f)
        config["quantization_config"] = self.build_quant_config_json()
        add_diffusion_step_policy(config, mixed_precision_steps)
        with open(transformer_export_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        self.write_diffusers_modelopt_state(config, merged, transformer_export_dir)
        self.write_transformers_modelopt_state(transformers_modelopt_state, transformer_export_dir)
        print(f"[export] wrote {len(merged)} tensors (fp8) to {transformer_export_dir}")


def export_quantized_transformer(
    mdl,
    transformer_export_dir: Path,
    src_transformer_dir: Path,
    dtype=torch.bfloat16,
    mixed_precision_steps: int = 3,
):
    """Export the FP8-quantized DiT as a vllm-omni-loadable diffusers checkpoint dir."""
    Fp8DiffusersExporter().export(
        mdl,
        Path(transformer_export_dir),
        Path(src_transformer_dir),
        dtype=dtype,
        mixed_precision_steps=mixed_precision_steps,
    )


def _write_root_hf_quant_config(root_dir: Path, transformer_dir: Path) -> None:
    """Mirror the transformer's ``quantization_config`` to a root ``hf_quant_config.json``.

    vllm-omni's diffusers pipeline reads the quant config from ``transformer/config.json``.
    Upstream vLLM's ``Cosmos3ForConditionalGeneration`` looks only at the model root, so
    we write a standalone copy there (the transformer config stays the source of truth).
    """
    with open(transformer_dir / "config.json", encoding="utf-8") as f:
        quant_config = json.load(f).get("quantization_config")
    if quant_config is None:
        raise ValueError(f"No 'quantization_config' in {transformer_dir}/config.json.")
    quant_config = copy.deepcopy(quant_config)
    quant_config.pop("runtime", None)
    with open(root_dir / "hf_quant_config.json", "w", encoding="utf-8") as f:
        json.dump(quant_config, f, indent=4)
    print(f"[assemble] wrote {root_dir / 'hf_quant_config.json'} for upstream-vLLM discovery")


def _write_root_weight_index(root_dir: Path) -> None:
    """(Re)generate the root ``model.safetensors.index.json`` for whole-model loading."""
    index = build_root_index(root_dir)
    dst = root_dir / "model.safetensors.index.json"
    index.write(dst)
    print(f"[assemble] wrote {dst.name} ({len(index.weight_map)} tensors, {index.metadata['total_size']} bytes)")


def assemble_output_dir(input_dir: Path, output_dir: Path, quantized_transformer_dir: Path):
    """Materialize a drop-in checkpoint dir mirroring ``input_dir``.

    ``output_dir`` mirrors ``input_dir`` (vae, scheduler, model_index.json, tokenizers,
    vision_encoder symlinked back); only ``transformer/`` is physically new (the
    quantized export). A fresh root weight index + a root ``hf_quant_config.json`` are
    written so both the diffusers and transformers loaders resolve the new weights.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_transformer = output_dir / "transformer"
    if target_transformer.exists():
        shutil.rmtree(target_transformer)
    print(f"[assemble] moving quantized -> {target_transformer}")
    shutil.move(str(quantized_transformer_dir), str(target_transformer))

    # Wire everything except transformer/ + the stale root index as relative symlinks
    # back to input_dir (relative links survive bind-mount path differences).
    _SKIP_LINK = {"transformer", "model.safetensors.index.json", "modelopt_state.pth"}
    input_dir_abs = input_dir.absolute()
    for entry in input_dir.iterdir():
        if entry.name in _SKIP_LINK:
            continue
        dst = output_dir / entry.name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        src = input_dir_abs / entry.name
        rel_src = os.path.relpath(src, start=output_dir.resolve())
        print(f"[assemble] linking {entry.name} -> {rel_src}")
        dst.symlink_to(rel_src)
    _write_root_weight_index(output_dir)
    shutil.move(str(target_transformer / "transformers_modelopt_state.pth"), str(output_dir / "modelopt_state.pth"))
    _write_root_hf_quant_config(output_dir, target_transformer)
    print(f"[assemble] drop-in dir ready at {output_dir}")
