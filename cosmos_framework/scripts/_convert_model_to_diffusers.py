# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Implementation for `convert_model_to_diffusers.py`."""

import contextlib
import inspect
import json
import pathlib
import re
import shutil
import struct
from typing import Any, Literal

import pydantic
import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    Cosmos3OmniTransformer,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)

try:
    from diffusers import Cosmos3OmniPipeline
except ImportError:
    # Some intermediate main-branch revisions used this renamed export.
    from diffusers import Cosmos3OmniDiffusersPipeline as Cosmos3OmniPipeline
from transformers import AutoConfig, AutoTokenizer

from cosmos_framework.inference.model import Cosmos3OmniModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface
from cosmos_framework.utils import log

DEFAULT_SOUND_TOKENIZER_CONFIG = {
    "model_type": "autoencoder_v2",
    "sampling_rate": 48000,
    "stereo": True,
    "use_wav_as_input": True,
    "normalize_volume": True,
    "hop_size": 1920,
    "input_channels": 1,
    "enc_type": "spec_convnext",
    "enc_dim": 192,
    "enc_intermediate_dim": 768,
    "enc_num_layers": 12,
    "enc_num_blocks": 2,
    "enc_n_fft": 64,
    "enc_hop_length": 16,
    "enc_latent_dim": 128,
    "enc_c_mults": [1, 2, 4],
    "enc_strides": [4, 5, 6],
    "enc_identity_init": False,
    "enc_use_snake": True,
    "dec_type": "oobleck",
    "dec_dim": 320,
    "dec_c_mults": [1, 2, 4, 8, 16],
    "dec_strides": [2, 4, 5, 6, 8],
    "dec_use_snake": True,
    "dec_final_tanh": False,
    "dec_out_channels": 2,
    "dec_anti_aliasing": False,
    "dec_use_nearest_upsample": False,
    "dec_use_tanh_at_final": False,
    "bottleneck_type": "vae",
    "bottleneck": {"type": "vae"},
    "activation": "snakebeta",
    "snake_logscale": True,
    "anti_aliasing": False,
    "use_cuda_kernel": False,
    "causal": False,
    "padding_mode": "zeros",
    "vocoder_input_dim": 64,
    "latent_mean": None,
    "latent_std": None,
}

# Wrapper prefixes that may appear on every key of a legacy AVAE state dict
# (DDP wrappers, full-model saves, training exports). Stripped iteratively
# until each key reaches a recognised target prefix.
_SOUND_TOKENIZER_PER_KEY_PREFIXES = ("module.", "generator.", "model.", "state_dict.")
_SOUND_TOKENIZER_TARGET_PREFIXES = ("decoder.", "encoder.", "bottleneck.")

# Inside a residual unit the legacy `nn.Sequential` layout was
# [snake1, conv1, snake2, conv2]; map sub-index → named attribute.
_SOUND_TOKENIZER_RES_UNIT_INNER_NAMES = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}

# The source language_model nests its transformer stack under a `model.`
# attribute (HF Qwen-style); the diffusers `Cosmos3OmniTransformer` holds
# those layers flat, so the leading `model.` prefix is stripped. The
# PackedAttentionMoT projections are renamed from the source Qwen-style
# names (`q_proj`/… plus cosmos-specific `*_moe_gen`) to the diffusers
# AttentionModuleMixin canonical names. Order matters: the `*_moe_gen`
# substrings must be substituted before the plain ones.
_ATTN_KEY_REMAP = (
    (".q_proj_moe_gen.", ".add_q_proj."),
    (".k_proj_moe_gen.", ".add_k_proj."),
    (".v_proj_moe_gen.", ".add_v_proj."),
    (".o_proj_moe_gen.", ".to_add_out."),
    (".q_norm_moe_gen.", ".norm_added_q."),
    (".k_norm_moe_gen.", ".norm_added_k."),
    (".q_proj.", ".to_q."),
    (".k_proj.", ".to_k."),
    (".v_proj.", ".to_v."),
    (".o_proj.", ".to_out."),
    (".q_norm.", ".norm_q."),
    (".k_norm.", ".norm_k."),
)

_LANGUAGE_MODEL_VISION_PREFIXES = ("model.visual.", "visual.")

# Legacy TimestepEmbedder stored its MLP as `nn.Sequential([Linear, SiLU, Linear])`,
# so state-dict keys were `mlp.0.*` / `mlp.2.*`. The diffusers `TimestepEmbedding`
# stores them as named attributes `linear_1` / `linear_2`. Index 1 (SiLU) has no
# parameters and therefore does not appear in either state dict.
_TIME_EMBEDDER_KEY_REMAP = {
    "mlp.0.weight": "linear_1.weight",
    "mlp.0.bias": "linear_1.bias",
    "mlp.2.weight": "linear_2.weight",
    "mlp.2.bias": "linear_2.bias",
}

DEFAULT_VISION_ENCODER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VISION_ENCODER_CHECKPOINT_PREFIX = "model.visual."
VISION_ENCODER_CHECKPOINT_SUBFOLDER = "vision_encoder"

COSMOS3_EDGE_REASONER = "nvidia/Cosmos3-Edge"
COSMOS3_EDGE_REASONER_REVISION = "be935d6931e4e176d7353abad41ca529d7b33b12"
COSMOS3_EDGE_VAE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Keep the Edge metadata and vision sidecar, but do not copy the source
# Transformers shards into the Diffusers output. The source reasoner is only an
# intermediate Stage 1 export; Stage 2 owns the Diffusers transformer shards.
COSMOS3_EDGE_REASONER_METADATA_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
)
COSMOS3_EDGE_REASONER_INDEX_FILE = "model.safetensors.index.json"


def _get_config_value(*configs, name, default=None):
    for config in configs:
        if config is None:
            continue
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
        if isinstance(config, dict) and config.get(name) is not None:
            return config[name]
    return default


def _resolve_fixed_step_sampler_config(model_cfg) -> dict | None:
    """Extract the distilled fixed-step sampler settings from a checkpoint config.

    Distilled (few-step) Cosmos3 checkpoints carry a `fixed_step_sampler_config`
    with a fixed flow-sigma schedule (`t_list`) and a `sample_type` (`ode`/`sde`).
    Returns `None` for non-distilled checkpoints.
    """
    fixed_step_cfg = _get_config_value(model_cfg, name="fixed_step_sampler_config", default=None)
    if fixed_step_cfg is None:
        return None

    t_list = _get_config_value(fixed_step_cfg, name="t_list", default=None)
    if t_list is None:
        raise ValueError("`fixed_step_sampler_config` is set, but `t_list` is missing.")
    t_list = [float(t) for t in t_list]
    if len(t_list) == 0:
        raise ValueError("`fixed_step_sampler_config.t_list` must contain at least one value.")
    # Training convention excludes terminal 0.0; normalize defensively if it is present.
    if t_list[-1] == 0.0:
        t_list = t_list[:-1]
    if len(t_list) == 0:
        raise ValueError("`fixed_step_sampler_config.t_list` cannot contain only terminal 0.0.")

    # Default "sde" matches i4's FixedStepSamplerConfig; "ode" is still a valid explicit value.
    sample_type = str(_get_config_value(fixed_step_cfg, name="sample_type", default="sde")).lower()
    if sample_type not in {"ode", "sde"}:
        raise ValueError(
            f"Unsupported `fixed_step_sampler_config.sample_type={sample_type!r}` (expected 'ode' or 'sde')."
        )

    rf_inference_cfg = _get_config_value(model_cfg, name="rectified_flow_inference_config", default=None)
    num_train_timesteps = int(_get_config_value(rf_inference_cfg, name="num_train_timesteps", default=1000))

    return {
        "t_list": t_list,
        "sample_type": sample_type,
        "num_train_timesteps": num_train_timesteps,
    }


def _remap_language_model_key(key: str) -> str:
    """Rename a source language-model state-dict key to the diffusers
    `Cosmos3OmniTransformer` layout: strip the leading `model.` prefix and
    apply the attention-projection renames from `_ATTN_KEY_REMAP`.
    """
    key = key.removeprefix("model.")
    for old, new in _ATTN_KEY_REMAP:
        if old in key:
            key = key.replace(old, new)
            break
    return key


def _remap_time_embedder_key(key: str) -> str:
    try:
        return _TIME_EMBEDDER_KEY_REMAP[key]
    except KeyError as exc:
        supported_keys = sorted(_TIME_EMBEDDER_KEY_REMAP)
        raise ValueError(
            f"Unsupported time_embedder state-dict key {key!r}; expected one of {supported_keys}."
        ) from exc


def _remap_language_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap language-model keys and reject ambiguous source layouts."""
    remapped_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(_LANGUAGE_MODEL_VISION_PREFIXES):
            continue
        remapped_key = _remap_language_model_key(key)
        if remapped_key in remapped_state_dict:
            raise RuntimeError(
                "Language-model key remap collision while applying diffusers key remap: "
                f"{key!r} maps to existing key {remapped_key!r}."
            )
        remapped_state_dict[remapped_key] = value
    return remapped_state_dict


def _load_sound_tokenizer_state_dict(checkpoint_path: pathlib.Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading AVAE .safetensors checkpoints requires safetensors.") from exc
        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"AVAE checkpoint must be a dict, got {type(checkpoint)!r}.")

    for key in ("generator", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict = {
        key: value.detach().cpu().contiguous() for key, value in checkpoint.items() if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise RuntimeError(f"No tensor state dict found in AVAE checkpoint keys: {list(checkpoint.keys())[:16]}")
    return state_dict


def _sound_tokenizer_strip_per_key_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip wrapper prefixes (`module.`, `generator.`, `model.`, `state_dict.`)
    from every key until the key reaches a recognised target prefix
    (`decoder.`, `encoder.`, `bottleneck.`) or no further prefix matches.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        while not new_key.startswith(_SOUND_TOKENIZER_TARGET_PREFIXES):
            for prefix in _SOUND_TOKENIZER_PER_KEY_PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break
            else:
                break
        out[new_key] = value
    return out


def _sound_tokenizer_filter_supported_modules(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only encoder/decoder keys — `Cosmos3AVAEAudioTokenizer` supports
    the parameter-free `vae` bottleneck only, so bottleneck keys are dropped.
    """
    return {k: v for k, v in state_dict.items() if k.startswith(("encoder.", "decoder."))}


def _sound_tokenizer_infer_num_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    """Count the decoder blocks present in a flat-`Sequential` legacy
    decoder state dict by spotting `decoder.layers.{N}.layers.{M}.*` keys.
    """
    block_indices: set[int] = set()
    for key in state_dict:
        m = re.fullmatch(r"decoder\.layers\.(\d+)\.layers\.\d+\..+", key)
        if m:
            block_indices.add(int(m.group(1)))
    return len(block_indices)


def _sound_tokenizer_remap_flat_layout(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite a legacy `decoder.layers.{N}.*` flat-`Sequential` layout to the
    diffusers Cosmos3AudioDecoder named-attribute layout.

    The legacy decoder is
        Sequential([conv1, block_0, ..., block_{N-1}, snake1, conv2])
    and each block is itself
        Sequential([snake1, conv_t1, res_unit1, res_unit2, res_unit3])
    where each residual unit is
        Sequential([snake1, conv1, snake2, conv2]).

    For the default `dec_strides=[2, 4, 5, 6, 8]` there are 5 blocks, so
    `decoder.layers.0` is conv1, `decoder.layers.{1..5}` are the blocks,
    `decoder.layers.6` is snake1, and `decoder.layers.7` is conv2.
    """
    if not any(re.match(r"decoder\.layers\.\d+\.", key) for key in state_dict):
        return state_dict

    num_blocks = _sound_tokenizer_infer_num_blocks(state_dict)
    if num_blocks == 0:
        raise RuntimeError(
            "Detected flat `decoder.layers.*` layout but no blocks "
            "(`decoder.layers.N.layers.M.*`) were found — cannot remap."
        )
    snake1_idx = num_blocks + 1
    conv2_idx = num_blocks + 2

    def _remap(key: str) -> str:
        m = re.fullmatch(r"decoder\.layers\.(\d+)\.layers\.(\d+)\.layers\.(\d+)\.(.+)", key)
        if m:
            block_n, res_n, inner_n, rest = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            if res_n not in (2, 3, 4):
                raise RuntimeError(f"Unexpected residual position res_n={res_n} in {key!r}.")
            inner_name = _SOUND_TOKENIZER_RES_UNIT_INNER_NAMES.get(inner_n)
            if inner_name is None:
                raise RuntimeError(f"Unexpected residual inner index inner_n={inner_n} in {key!r}.")
            return f"decoder.block.{block_n - 1}.res_unit{res_n - 1}.{inner_name}.{rest}"

        m = re.fullmatch(r"decoder\.layers\.(\d+)\.layers\.(\d+)\.(.+)", key)
        if m:
            block_n, sub_n, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            block_idx = block_n - 1
            if sub_n == 0:
                return f"decoder.block.{block_idx}.snake1.{rest}"
            if sub_n == 1:
                return f"decoder.block.{block_idx}.conv_t1.{rest}"
            raise RuntimeError(f"Unexpected block sub-index sub_n={sub_n} in {key!r}.")

        m = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", key)
        if m:
            layer_n, rest = int(m.group(1)), m.group(2)
            if layer_n == 0:
                return f"decoder.conv1.{rest}"
            if layer_n == snake1_idx:
                return f"decoder.snake1.{rest}"
            if layer_n == conv2_idx:
                return f"decoder.conv2.{rest}"
            raise RuntimeError(
                f"Unexpected leaf layer index {layer_n} (expected 0, {snake1_idx}, or {conv2_idx}) in {key!r}."
            )

        return key

    return {_remap(key): value for key, value in state_dict.items()}


def _sound_tokenizer_reshape_snake_params(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Snake1d alpha/beta are stored with shape `[C]` in the legacy AVAE but
    `[1, C, 1]` in diffusers' Snake1d — unsqueeze when they arrive as 1-D.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if (
            key.startswith(("encoder.", "decoder."))
            and (key.endswith(".alpha") or key.endswith(".beta"))
            and value.ndim == 1
        ):
            value = value.unsqueeze(0).unsqueeze(-1).contiguous()
        out[key] = value
    return out


def _sound_tokenizer_reapply_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """If a Conv layer has the folded `.weight` tensor but neither `.weight_g`
    nor `.weight_v`, reconstruct the pair so the resulting checkpoint can be
    loaded into the weight-norm-wrapped diffusers Cosmos3AudioDecoder.

    `weight_norm` with `dim=0` parameterises `weight = g * v / ||v||`. Setting
    `v = weight` and `g = ||weight||` along all non-zero axes is an exact
    inverse: `g * v / ||v|| = ||weight|| * weight / ||weight|| = weight`.
    """
    out = dict(state_dict)
    candidate_keys = [
        key
        for key in state_dict
        if key.endswith(".weight")
        and (
            any(f".{layer}." in key for layer in ("conv1", "conv2", "conv_t1"))
            or re.fullmatch(r"encoder\.layers\.\d+\.weight", key)
        )
    ]
    for key in candidate_keys:
        stem = key[: -len(".weight")]
        weight_g_key = f"{stem}.weight_g"
        weight_v_key = f"{stem}.weight_v"
        if weight_g_key in state_dict or weight_v_key in state_dict:
            continue
        weight = state_dict[key]
        norm_dims = tuple(range(1, weight.ndim))
        weight_g = weight.norm(p=2, dim=norm_dims, keepdim=True).contiguous()
        out.pop(key)
        out[weight_g_key] = weight_g
        out[weight_v_key] = weight
    return out


def _remap_sound_tokenizer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the full legacy → diffusers conversion pipeline to a sound
    tokenizer state dict: strip prefixes, drop unsupported (bottleneck) keys,
    remap the flat `nn.Sequential` layout to named attributes, reshape Snake1d
    params, and reconstruct `weight_g` / `weight_v` for any folded conv weights.
    """
    state_dict = _sound_tokenizer_strip_per_key_prefixes(state_dict)
    state_dict = _sound_tokenizer_filter_supported_modules(state_dict)
    if not state_dict:
        raise RuntimeError("Sound tokenizer state dict has no `encoder.*`/`decoder.*` keys after prefix stripping.")
    if not any(key.startswith("decoder.") for key in state_dict):
        raise RuntimeError("Sound tokenizer state dict has no `decoder.*` keys after prefix stripping.")
    state_dict = _sound_tokenizer_remap_flat_layout(state_dict)
    state_dict = _sound_tokenizer_reshape_snake_params(state_dict)
    state_dict = _sound_tokenizer_reapply_weight_norm(state_dict)
    if any(re.match(r"decoder\.layers\.\d+", key) for key in state_dict):
        raise RuntimeError("Flat `decoder.layers.*` keys remain after remap; conversion is incomplete.")
    return state_dict


def _load_sound_tokenizer_config(config_path: pathlib.Path | None) -> dict:
    if config_path is None:
        return dict(DEFAULT_SOUND_TOKENIZER_CONFIG)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _build_sound_tokenizer(
    checkpoint_path: pathlib.Path,
    config_path: pathlib.Path | None,
) -> Any:
    try:
        from diffusers import Cosmos3AVAEAudioTokenizer
    except ImportError as error:
        raise RuntimeError(
            "This Diffusers build does not provide Cosmos3AVAEAudioTokenizer, which is required for sound exports."
        ) from error

    config = _load_sound_tokenizer_config(config_path)

    log.info(f"Loading AVAE sound tokenizer weights from {checkpoint_path} …")
    raw_state_dict = _load_sound_tokenizer_state_dict(checkpoint_path)
    state_dict = _remap_sound_tokenizer_state_dict(raw_state_dict)
    has_encoder = any(key.startswith("encoder.") for key in state_dict)
    log.info(
        f"Remapped {len(raw_state_dict)} → {len(state_dict)} tokenizer keys "
        f"({'encoder+decoder' if has_encoder else 'decoder-only'})."
    )

    # `Cosmos3AVAEAudioTokenizer` accepts exactly the keys of the default
    # config; unknown keys in a source config JSON are ignored.
    tokenizer_kwargs = {name: config.get(name, default) for name, default in DEFAULT_SOUND_TOKENIZER_CONFIG.items()}
    sound_tokenizer = Cosmos3AVAEAudioTokenizer(**tokenizer_kwargs, encoder_enabled=has_encoder)
    load_result = sound_tokenizer.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Cosmos3 AVAE sound tokenizer load did not match strictly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}."
        )
    return sound_tokenizer


def _checkpoint_weight_map(checkpoint_path: pathlib.Path) -> dict[str, str]:
    index_path = checkpoint_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return index.get("weight_map", {})


def _is_edge_exported_checkpoint(checkpoint_path: pathlib.Path) -> bool:
    """Return whether a Stage 1 HF export uses the Nemotron Edge backbone."""
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        return False
    config = _load_json(config_path)
    model_config = config.get("model", {}).get("config", {})
    vlm_config = model_config.get("vlm_config", {})
    pretrained_weights = vlm_config.get("pretrained_weights", {})
    model_instance = vlm_config.get("model_instance", {})
    # Internal config uses `_target_` with the class name (Nemotron3DenseVL…); this
    # repo's public export config uses `_target` with the aliased name
    # (nemotron_3_dense_vl_…). Accept both spellings/formats.
    target = str(model_instance.get("_target_") or model_instance.get("_target") or "")
    return bool(
        pretrained_weights.get("checkpoint_format") == "nemotron_3_dense_vl"
        or "Nemotron3" in target
        or "nemotron_3_dense_vl" in target.lower()
    )


def _checkpoint_has_weight_prefix(checkpoint_path: pathlib.Path, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in _checkpoint_weight_map(checkpoint_path))


def _checkpoint_vision_subfolder_files(checkpoint_path: pathlib.Path) -> dict[str, list[str]]:
    """Group root-index keys stored under vision_encoder/ by shard file.

    Diffusers-layout source checkpoints keep bare Qwen3VLVisionModel keys
    (`blocks.*`, `patch_embed.*`, …) in the root weight map, mapped to files
    under the vision_encoder/ subfolder.
    """
    files_to_keys: dict[str, list[str]] = {}
    for key, filename in _checkpoint_weight_map(checkpoint_path).items():
        if filename.replace("\\", "/").startswith(f"{VISION_ENCODER_CHECKPOINT_SUBFOLDER}/"):
            files_to_keys.setdefault(filename, []).append(key)
    return files_to_keys


def _load_prefixed_safetensors_state_dict(checkpoint_path: pathlib.Path, prefix: str) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("Loading sharded safetensors vision weights requires safetensors.") from exc

    weight_map = _checkpoint_weight_map(checkpoint_path)
    if not weight_map:
        raise FileNotFoundError(
            f"Could not find model.safetensors.index.json under {checkpoint_path}; cannot stream {prefix!r} weights."
        )

    files_to_keys: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if key.startswith(prefix):
            files_to_keys.setdefault(filename, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for filename, keys in sorted(files_to_keys.items()):
        shard_path = checkpoint_path / filename
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in sorted(keys):
                state_dict[key[len(prefix) :]] = shard.get_tensor(key).detach().cpu().contiguous()

    if not state_dict:
        raise RuntimeError(f"No checkpoint tensors found with prefix {prefix!r}.")
    return state_dict


def _get_source_vision_state_dict(model) -> dict[str, torch.Tensor] | None:
    for candidate in (
        getattr(model, "visual", None),
        getattr(getattr(model, "net", None), "visual", None),
        getattr(getattr(getattr(model, "net", None), "language_model", None), "visual", None),
    ):
        if candidate is None:
            continue
        state_dict = {
            key.removeprefix("visual.").removeprefix("model.visual."): value.detach().cpu().contiguous()
            for key, value in candidate.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        if state_dict:
            return state_dict
    return None


def _build_vision_encoder(
    state_dict: dict[str, torch.Tensor],
    model_name_or_path: str,
    dtype: torch.dtype,
):
    try:
        from transformers import Qwen3VLVisionModel
    except ImportError as exc:
        raise ImportError(
            "Saving the Cosmos3 Qwen3-VL vision encoder requires a transformers version "
            "that provides Qwen3VLVisionModel."
        ) from exc

    qwen_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    vision_config = getattr(qwen_config, "vision_config", None)
    if vision_config is None:
        raise ValueError(f"{model_name_or_path!r} does not provide a Qwen3-VL vision_config.")

    with init_empty_weights():
        vision_encoder = Qwen3VLVisionModel(vision_config)
    load_result = vision_encoder.load_state_dict(state_dict, strict=True, assign=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Qwen3-VL vision encoder load did not match strictly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}."
        )
    return vision_encoder.to(dtype=dtype)


def _load_vision_subfolder_state_dict(checkpoint_path: pathlib.Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("Loading sharded safetensors vision weights requires safetensors.") from exc

    files_to_keys = _checkpoint_vision_subfolder_files(checkpoint_path)
    state_dict: dict[str, torch.Tensor] = {}
    for filename, keys in sorted(files_to_keys.items()):
        shard_path = checkpoint_path / filename
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in sorted(keys):
                state_dict[key] = shard.get_tensor(key).detach().cpu().contiguous()

    if not state_dict:
        raise RuntimeError(
            f"No vision encoder tensors found under {VISION_ENCODER_CHECKPOINT_SUBFOLDER}/. "
            "If the source checkpoint has no Qwen3-VL visual weights (e.g. a vision-generation-only "
            "post-training checkpoint), pass --skip-vision-encoder to export without vision_encoder/."
        )
    return state_dict


def _load_vision_encoder(
    checkpoint_path: pathlib.Path,
    source_model,
    model_name_or_path: str,
    dtype: torch.dtype,
):
    state_dict = _get_source_vision_state_dict(source_model)
    if state_dict is not None:
        log.info("Extracting Qwen3-VL vision encoder weights from loaded source model …")
    elif _checkpoint_has_weight_prefix(checkpoint_path, VISION_ENCODER_CHECKPOINT_PREFIX):
        log.info(f"Loading Qwen3-VL vision encoder weights from {checkpoint_path} …")
        state_dict = _load_prefixed_safetensors_state_dict(checkpoint_path, VISION_ENCODER_CHECKPOINT_PREFIX)
    else:
        log.info(f"Loading Qwen3-VL vision encoder weights from {checkpoint_path}/vision_encoder …")
        state_dict = _load_vision_subfolder_state_dict(checkpoint_path)
    log.info(f"Building Qwen3-VL vision encoder from {model_name_or_path} …")
    return _build_vision_encoder(state_dict, model_name_or_path, dtype)


class _MetadataVideoTokenizer(VideoTokenizerInterface):
    """Expose video-tokenizer metadata without loading the source VAE."""

    def __init__(
        self,
        latent_ch: int,
        spatial_compression_factor: int,
        temporal_compression_factor: int,
        chunk_duration: int,
        causal: bool,
    ) -> None:
        self._latent_ch = latent_ch
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor
        self._chunk_duration = chunk_duration
        self._causal = causal

    def reset_dtype(self) -> None:
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The metadata-only converter tokenizer cannot encode media.")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The metadata-only converter tokenizer cannot decode media.")

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // self._temporal_compression_factor

    def get_pixel_num_frames(self, num_latent_frames: int, **kwargs: Any) -> int:
        del kwargs
        return (num_latent_frames - 1) * self._temporal_compression_factor + 1

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._temporal_compression_factor

    @property
    def spatial_resolution(self) -> int:
        return 512

    @property
    def pixel_chunk_duration(self) -> int:
        return self._chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self._chunk_duration)

    @property
    def latent_ch(self) -> int:
        return self._latent_ch


@contextlib.contextmanager
def _skip_source_tokenizer_load():
    original_set_up_tokenizers = OmniMoTModel.set_up_tokenizers

    def set_up_tokenizers_without_source_checkpoints(self):
        source_tokenizer = self.config.tokenizer
        source_tokenizer_target = str(source_tokenizer.get("_target_", ""))
        skip_video_tokenizer = "Wan2pt2VAEInterface" in source_tokenizer_target
        skip_sound_tokenizer = bool(getattr(self.config, "sound_gen", False))
        original_video_tokenizer = source_tokenizer
        original_sound_gen = self.config.sound_gen
        if skip_video_tokenizer:
            self.config.tokenizer = {
                "_target_": f"{__name__}._MetadataVideoTokenizer",
                "latent_ch": int(self.config.state_ch),
                "spatial_compression_factor": int(source_tokenizer.get("spatial_compression_factor", 16)),
                "temporal_compression_factor": int(source_tokenizer.get("temporal_compression_factor", 4)),
                "chunk_duration": int(source_tokenizer.get("chunk_duration", 93)),
                "causal": bool(source_tokenizer.get("causal", True)),
            }
        if skip_sound_tokenizer:
            self.config.sound_gen = False
        try:
            return original_set_up_tokenizers(self)
        finally:
            self.config.tokenizer = original_video_tokenizer
            self.config.sound_gen = original_sound_gen

    OmniMoTModel.set_up_tokenizers = set_up_tokenizers_without_source_checkpoints
    try:
        yield
    finally:
        OmniMoTModel.set_up_tokenizers = original_set_up_tokenizers


def _validate_edge_action_pipeline_support() -> None:
    pipeline_params = inspect.signature(Cosmos3OmniPipeline.__call__).parameters
    required_params = {"action", "action_latents"}
    missing_params = sorted(required_params - set(pipeline_params))
    transformer_params = inspect.signature(Cosmos3OmniTransformer.__init__).parameters
    required_transformer_params = {"action_gen", "action_dim", "num_embodiment_domains"}
    missing_transformer_params = sorted(required_transformer_params - set(transformer_params))
    if missing_params or missing_transformer_params:
        details = []
        if missing_params:
            details.append(f"pipeline call parameters {missing_params}")
        if missing_transformer_params:
            details.append(f"transformer constructor parameters {missing_transformer_params}")
        raise RuntimeError(
            "The checkpoint has action generation weights, but the checked-out Diffusers main does not provide "
            "the action-enabled Cosmos3 pipeline/transformer API (missing "
            f"{'; '.join(details)}). Use the merged action-enabled Diffusers main before converting an Edge "
            "policy checkpoint; this converter does not drop action weights."
        )


def _validate_edge_transformer_support() -> None:
    """Fail fast when the installed Diffusers cannot build the Edge transformer.

    The Cosmos3 Edge (Nemotron dense) backbone is structurally different from the
    Qwen-based one: it has no QK-norm (`qk_norm_for_text=False`) and a non-gated
    ReLU² MLP (`hidden_act`). Those are only honored when the Diffusers
    `Cosmos3OmniTransformer` exposes them as constructor arguments. On a build
    that predates the Edge API they are silently dropped, the transformer is
    built Qwen-style, and the Edge weights fail to load with an opaque
    missing-key error (`norm_q` / `norm_k` / `mlp.gate_proj`). Detect that here so
    the failure is actionable.
    """
    params = inspect.signature(Cosmos3OmniTransformer.__init__).parameters
    accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = sorted({"hidden_act", "qk_norm_for_text"} - set(params))
    if missing and not accepts_var_kwargs:
        raise RuntimeError(
            "This Diffusers build cannot convert a Cosmos3 Edge checkpoint: its Cosmos3OmniTransformer does not "
            f"accept the structural Edge constructor argument(s) {missing} (it would build the Qwen-style "
            "transformer, and the Edge weights would fail to load with missing norm_q/norm_k/mlp.gate_proj keys). "
            "Upgrade to an Edge-capable diffusers-cosmos3 build before converting an Edge checkpoint."
        )


def _resolve_edge_reasoner_path(args, exported_checkpoint_path: pathlib.Path | None = None) -> pathlib.Path:
    if args.reasoner_path is not None:
        reasoner_path = pathlib.Path(args.reasoner_path).expanduser().absolute()
    elif exported_checkpoint_path is not None and (exported_checkpoint_path / "edge_reasoner").is_dir():
        reasoner_path = exported_checkpoint_path / "edge_reasoner"
    else:
        from huggingface_hub import snapshot_download

        print(
            "Downloading the pinned Cosmos3 Edge reasoner snapshot "
            f"({args.reasoner_repo_id}@{args.reasoner_revision}) …"
        )
        reasoner_path = pathlib.Path(
            snapshot_download(
                repo_id=args.reasoner_repo_id,
                revision=args.reasoner_revision,
                allow_patterns=[
                    *COSMOS3_EDGE_REASONER_METADATA_FILES,
                    COSMOS3_EDGE_REASONER_INDEX_FILE,
                    "*.safetensors",
                    "vision_encoder/*.safetensors",
                ],
            )
        )

    if not reasoner_path.is_dir():
        raise FileNotFoundError(f"Cosmos3 Edge reasoner directory not found: {reasoner_path}")

    missing_files = [
        filename
        for filename in COSMOS3_EDGE_REASONER_METADATA_FILES + (COSMOS3_EDGE_REASONER_INDEX_FILE,)
        if filename != "processor_config.json" and not (reasoner_path / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"Cosmos3 Edge reasoner at {reasoner_path} is missing required files: {missing_files}")
    source_index = _load_json(reasoner_path / COSMOS3_EDGE_REASONER_INDEX_FILE)
    source_weight_map = source_index.get("weight_map", {})
    referenced_shards = {str(shard) for shard in source_weight_map.values()}
    missing_shards = [filename for filename in referenced_shards if not (reasoner_path / filename).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"Cosmos3 Edge reasoner at {reasoner_path} is missing weight shards: {sorted(missing_shards)}"
        )
    vision_keys = {key for key in source_weight_map if key.startswith(("model.visual.", "model.projector."))}
    vision_shards = {source_weight_map[key] for key in vision_keys}
    missing_vision_shards = [filename for filename in vision_shards if not (reasoner_path / filename).is_file()]
    if not vision_keys or missing_vision_shards:
        raise FileNotFoundError(
            f"Cosmos3 Edge reasoner at {reasoner_path} has invalid vision assets: "
            f"vision_keys={len(vision_keys)}, missing_shards={missing_vision_shards}"
        )
    return reasoner_path


def _load_json(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(payload: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _native_edge_text_config(source_config: dict) -> dict:
    """Express the source text layout as standard Edge decoder blocks."""
    source_rope_parameters = source_config.get("rope_parameters") or {}
    source_layer_count = int(source_config.get("num_hidden_layers", 28))
    if source_config.get("model_type") == "nemotron_h":
        if source_layer_count % 2:
            raise ValueError(f"Legacy Nemotron Edge text layer count must be even, got {source_layer_count}.")
        source_layer_count //= 2
    native_config = {
        "model_type": "cosmos3_edge_text",
        "num_hidden_layers": source_layer_count,
        "hidden_act": source_config.get("hidden_act", source_config.get("mlp_hidden_act", "relu2")),
        "rms_norm_eps": source_config.get("rms_norm_eps", source_config.get("layer_norm_epsilon", 1e-5)),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": source_rope_parameters.get("rope_theta", source_config.get("rope_theta", 100_000_000.0)),
            "mrope_section": source_rope_parameters.get(
                "mrope_section", source_config.get("mrope_section", [24, 20, 20])
            ),
        },
    }
    for key in (
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "head_dim",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "mlp_bias",
        "num_attention_heads",
        "num_key_value_heads",
        "pad_token_id",
        "use_cache",
        "vocab_size",
    ):
        if key in source_config:
            native_config[key] = source_config[key]
    return native_config


def _native_edge_projector_config(source_config: dict) -> dict:
    merger_intermediate_size = source_config.get("merger_intermediate_size")
    if merger_intermediate_size is None:
        merger_intermediate_size = source_config.get("merger_intermedia")
    if merger_intermediate_size is None:
        raise ValueError(
            "Cosmos3 Edge projector config must define `merger_intermediate_size` "
            "(or the legacy `merger_intermedia` alias)."
        )
    return {
        "model_type": "cosmos3_edge_projector",
        "input_hidden_size": source_config["input_hidden_size"],
        "merger_intermediate_size": merger_intermediate_size,
        "out_hidden_size": source_config["out_hidden_size"],
        "spatial_merge_size": source_config["spatial_merge_size"],
        "use_postshuffle_norm": source_config["use_postshuffle_norm"],
    }


def _native_edge_vision_config(source_config: dict) -> dict:
    native_config = {"model_type": "cosmos3_edge_vision"}
    for key in (
        "attention_dropout",
        "hidden_act",
        "hidden_size",
        "intermediate_size",
        "layer_norm_eps",
        "num_attention_heads",
        "num_channels",
        "num_hidden_layers",
        "num_patches",
        "patch_size",
        "spatial_merge_size",
    ):
        if key in source_config:
            native_config[key] = source_config[key]
    return native_config


def _native_edge_config(source_config: dict) -> dict:
    native_config = {
        key: value
        for key, value in source_config.items()
        if key not in {"architectures", "auto_map", "model_type", "projector_config", "text_config", "vision_config"}
    }
    native_config["architectures"] = ["Cosmos3EdgeForConditionalGeneration"]
    native_config["model_type"] = "cosmos3_edge"
    native_config["allow_patterns_overrides"] = ["*/*.safetensors"]
    native_config["text_config"] = _native_edge_text_config(source_config["text_config"])
    native_config["vision_config"] = _native_edge_vision_config(source_config["vision_config"])

    projector_config = _native_edge_projector_config(source_config["projector_config"])
    native_config["projector_config"] = projector_config
    native_config["vision_config"].setdefault("spatial_merge_size", projector_config["spatial_merge_size"])
    if projector_config["input_hidden_size"] != native_config["vision_config"]["hidden_size"]:
        raise ValueError("Edge projector input size must match the vision tower hidden size.")
    if projector_config["out_hidden_size"] != native_config["text_config"]["hidden_size"]:
        raise ValueError("Edge projector output size must match the text tower hidden size.")
    if projector_config["use_postshuffle_norm"]:
        raise ValueError("The native Cosmos3 Edge architecture only supports pre-shuffle projector normalization.")
    native_config["projector_hidden_size"] = projector_config["merger_intermediate_size"]
    return native_config


def _native_edge_tokenizer_config(source_config: dict) -> dict:
    native_config = dict(source_config)
    native_config["return_mm_token_type_ids"] = True
    return native_config


def _build_edge_processor_config(reasoner_path: pathlib.Path) -> dict:
    """Build current Edge processor metadata from an older reasoner export."""
    image_config = _normalize_edge_preprocessor_config(
        _load_json(reasoner_path / "preprocessor_config.json"),
        processor_type_key="image_processor_type",
        processor_type="Cosmos3EdgeImageProcessor",
    )
    video_config = _normalize_edge_preprocessor_config(
        _load_json(reasoner_path / "video_preprocessor_config.json"),
        processor_type_key="video_processor_type",
        processor_type="Cosmos3EdgeVideoProcessor",
    )
    video_config["return_metadata"] = True
    return {"processor_class": "Cosmos3EdgeProcessor", "image_processor": image_config, "video_processor": video_config}


def _normalize_edge_preprocessor_config(
    config: dict[str, Any],
    processor_type_key: str,
    processor_type: str,
) -> dict[str, Any]:
    """Map legacy reasoner preprocessing metadata to the native Edge processor."""
    normalized_config = json.loads(json.dumps(config))
    normalized_config.pop("auto_map", None)
    normalized_config["processor_class"] = "Cosmos3EdgeProcessor"
    normalized_config[processor_type_key] = processor_type
    return normalized_config


def _write_edge_vision_encoder(
    reasoner_path: pathlib.Path,
    output_dir: pathlib.Path,
    source_weight_map: dict[str, str],
) -> None:
    vision_keys = {key for key in source_weight_map if key.startswith(("model.visual.", "model.projector."))}
    vision_shards = {source_weight_map[key] for key in vision_keys}
    if len(vision_shards) != 1:
        raise RuntimeError(f"Edge visual tensors unexpectedly span shards: {sorted(vision_shards)}")

    vision_dir = output_dir / "vision_encoder"
    vision_dir.mkdir(parents=True, exist_ok=True)
    vision_path = vision_dir / "model.safetensors"
    print(f"Copying {len(vision_keys)} Cosmos3 Edge vision/projector tensors to {vision_path} …")
    source_vision_path = reasoner_path / next(iter(vision_shards))
    from safetensors import safe_open
    from safetensors.torch import save_file

    vision_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(source_vision_path), framework="pt", device="cpu") as source_file:
        for key in source_file.keys():
            if key in vision_keys:
                vision_tensors[key] = source_file.get_tensor(key)
    if set(vision_tensors) != vision_keys:
        missing_keys = sorted(vision_keys - set(vision_tensors))
        raise RuntimeError(f"Missing Edge vision/projector tensors while writing {vision_path}: {missing_keys}")
    save_file(vision_tensors, str(vision_path), metadata={"format": "pt"})
    # This is an auxiliary shard consumed through the root safetensors index;
    # it is not a standalone Diffusers component. Do not leave a stale legacy
    # config beside it when converting into an existing output directory.
    (vision_dir / "config.json").unlink(missing_ok=True)


def _write_diffusers_safetensors_index(output_dir: pathlib.Path) -> None:
    """Write the existing Edge root index over shared component shards."""
    metadata = {"total_size": 0}
    weight_map: dict[str, str] = {}
    for subdir in ("transformer", "vision_encoder"):
        component_dir = output_dir / subdir
        if not component_dir.is_dir():
            continue
        for safetensors_path in sorted(component_dir.glob("*.safetensors")):
            with safetensors_path.open("rb") as file:
                header_size = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(header_size).decode("utf-8"))
            relative_path = f"{subdir}/{safetensors_path.name}"
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if not _is_edge_shared_diffusers_key(name):
                    continue
                if name in weight_map:
                    raise ValueError(f"Key {name} already exists in the Diffusers weight map.")
                metadata["total_size"] += info["data_offsets"][1] - info["data_offsets"][0]
                weight_map[name] = relative_path

    if not weight_map:
        raise FileNotFoundError(f"No Diffusers safetensors were found under {output_dir}.")
    _save_json(
        {"metadata": metadata, "weight_map": weight_map},
        output_dir / COSMOS3_EDGE_REASONER_INDEX_FILE,
    )


def _is_edge_shared_diffusers_key(name: str) -> bool:
    """Return whether a Diffusers key belongs in the existing Edge root index."""
    if name.startswith("model."):
        return True
    if name in {"embed_tokens.weight", "lm_head.weight", "norm.weight"}:
        return True
    if not name.startswith("layers."):
        return False
    generator_markers = (
        ".input_layernorm_moe_gen.",
        ".post_attention_layernorm_moe_gen.",
        ".mlp_moe_gen.",
        ".self_attn.add_",
        ".self_attn.norm_added_",
        ".self_attn.to_add_out.",
    )
    return not any(marker in name for marker in generator_markers)


def _copy_edge_reasoner_metadata(reasoner_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Copy Edge metadata and its vision sidecar into a Diffusers repository."""
    print(f"Writing the shared Cosmos3 Edge metadata into {output_dir} …")
    source_index = _load_json(reasoner_path / COSMOS3_EDGE_REASONER_INDEX_FILE)
    for filename in COSMOS3_EDGE_REASONER_METADATA_FILES:
        if filename == "config.json":
            continue
        if filename == "preprocessor_config.json":
            _save_json(
                _normalize_edge_preprocessor_config(
                    _load_json(reasoner_path / filename),
                    processor_type_key="image_processor_type",
                    processor_type="Cosmos3EdgeImageProcessor",
                ),
                output_dir / filename,
            )
            continue
        if filename == "video_preprocessor_config.json":
            _save_json(
                _normalize_edge_preprocessor_config(
                    _load_json(reasoner_path / filename),
                    processor_type_key="video_processor_type",
                    processor_type="Cosmos3EdgeVideoProcessor",
                )
                | {"return_metadata": True},
                output_dir / filename,
            )
            continue
        if filename == "processor_config.json" and not (reasoner_path / filename).is_file():
            _save_json(_build_edge_processor_config(reasoner_path), output_dir / filename)
            continue
        if filename == "tokenizer_config.json":
            _save_json(_native_edge_tokenizer_config(_load_json(reasoner_path / filename)), output_dir / filename)
            continue
        shutil.copy2(reasoner_path / filename, output_dir / filename)

    for filename in (
        "configuration_nemotron_siglip2_h.py",
        "modeling_cosmos3_edge_omni.py",
        "modeling_nemotron_siglip2_h.py",
        "processing.py",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    _save_json(_native_edge_config(_load_json(reasoner_path / "config.json")), output_dir / "config.json")
    _write_edge_vision_encoder(reasoner_path, output_dir, source_index["weight_map"])
    for filename in ("00000.safetensors", "00001.safetensors", "model.safetensors"):
        (output_dir / filename).unlink(missing_ok=True)
    _write_diffusers_safetensors_index(output_dir)


def _write_edge_transformer_config(output_dir: pathlib.Path, model_cfg: Any) -> None:
    """Persist Edge fields and the existing multi-axis RoPE format."""
    transformer_config_path = output_dir / "transformer" / "config.json"
    transformer_config = _load_json(transformer_config_path)
    tokenizer_config = _get_config_value(model_cfg, name="tokenizer", default=None)
    temporal_compression_factor = _get_config_value(tokenizer_config, name="temporal_compression_factor", default=None)
    if temporal_compression_factor is None:
        raise ValueError("Cosmos3 Edge export config is missing tokenizer.temporal_compression_factor.")
    transformer_config.update(
        {
            "backbone_type": "cosmos3_edge_nemotron_dense",
            "temporal_compression_factor": int(temporal_compression_factor),
        }
    )
    rope_axes_dim = transformer_config.get("rope_axes_dim")
    if rope_axes_dim is not None:
        transformer_config["rope_scaling"] = {"mrope_section": rope_axes_dim}
    _save_json(transformer_config, transformer_config_path)


def _normalize_edge_model_index(output_dir: pathlib.Path) -> None:
    """Keep Edge pipeline metadata consistent with the existing Hub export."""
    model_index_path = output_dir / "model_index.json"
    model_index = _load_json(model_index_path)
    model_index.update(
        {
            "default_use_system_prompt": False,
            "text_tokenizer": ["transformers", "PreTrainedTokenizerFast"],
            "use_native_flow_schedule": True,
        }
    )
    _save_json(model_index, model_index_path)


def _add_edge_reasoner_to_pipeline(args) -> None:
    output_dir = pathlib.Path(args.output).expanduser().absolute()
    expected_paths = ("model_index.json", "scheduler", "text_tokenizer", "transformer", "vae")
    missing_paths = [str(output_dir / path) for path in expected_paths if not (output_dir / path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Expected an existing Cosmos3 Edge Diffusers pipeline before adding its reasoner; "
            f"missing paths: {missing_paths}"
        )

    reasoner_path = _resolve_edge_reasoner_path(args)
    _copy_edge_reasoner_metadata(reasoner_path, output_dir)
    _normalize_edge_model_index(output_dir)
    print("Done.")


class Args(pydantic.BaseModel):
    checkpoint_path: pathlib.Path
    """Path to a Stage-1 HF export or a named non-Edge HF checkpoint."""
    output: str
    """Directory to save the converted diffusers model."""
    save_pipeline: bool = False
    """Save the full pipeline (transformer + VAE + tokenizer + scheduler)."""
    dtype: str = "bf16"
    """Dtype to save the transformer in."""
    sound_tokenizer_path: str | None = None
    """Optional AVAE sound tokenizer checkpoint to save under sound_tokenizer/."""
    sound_tokenizer_config_path: str | None = None
    """Optional AVAE config JSON describing the sound tokenizer architecture."""
    include_sound_tokenizer: bool = False
    """Require saving sound_tokenizer/ even if the source transformer is video-only."""
    vision_encoder_model: str = DEFAULT_VISION_ENCODER_MODEL
    """Qwen3-VL model/config to instantiate model.visual.* weights."""
    skip_vision_encoder: bool = False
    """Do not save vision_encoder/ when saving a full pipeline."""
    distilled_scheduler: Literal["auto", "on", "off"] = "auto"
    """How to export the scheduler for distilled (few-step) checkpoints.

    * auto: export the distilled scheduler when `fixed_step_sampler_config` is present.
    * on: require and always export the distilled scheduler.
    * off: always export the UniPC scheduler.
    """

    include_reasoner: bool = True
    """Include the pinned Edge metadata and vision sidecar in an exported Edge checkpoint."""

    reasoner_repo_id: str = COSMOS3_EDGE_REASONER
    """Hugging Face repository containing the Cosmos3 Edge reasoner checkpoint."""

    reasoner_revision: str = COSMOS3_EDGE_REASONER_REVISION
    """Pinned revision of the Cosmos3 Edge reasoner checkpoint."""

    reasoner_path: pathlib.Path | None = None
    """Optional local Cosmos3 Edge reasoner snapshot, used instead of downloading it."""

    copy_edge_reasoner: bool = False
    """Add the shared-weight Edge reasoner to an existing Diffusers pipeline at `output`."""


def convert_model_to_diffusers(args: Args) -> None:
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if args.copy_edge_reasoner:
        _add_edge_reasoner_to_pipeline(args)
        return

    # A raw DCP is loaded directly by ``Cosmos3OmniModel.from_pretrained_dcp`` below
    # (this repo's existing DCP -> Diffusers path). Only the Edge reasoner flow
    # requires a config-driven HF export first, which the wrapper enforces
    # separately (its ``is_edge_model and .metadata`` guard).

    sound_tokenizer_path = (
        pathlib.Path(args.sound_tokenizer_path).expanduser().absolute() if args.sound_tokenizer_path else None
    )
    sound_tokenizer_config_path = (
        pathlib.Path(args.sound_tokenizer_config_path).expanduser().absolute()
        if args.sound_tokenizer_config_path
        else None
    )
    if args.include_sound_tokenizer and sound_tokenizer_path is None:
        raise ValueError("Sound tokenizer output was requested, but --sound-tokenizer-path was not provided.")
    if sound_tokenizer_path is not None and not sound_tokenizer_path.exists():
        raise FileNotFoundError(f"Sound tokenizer checkpoint not found: {sound_tokenizer_path}")
    if sound_tokenizer_config_path is not None and not sound_tokenizer_config_path.exists():
        raise FileNotFoundError(f"Sound tokenizer config not found: {sound_tokenizer_config_path}")

    checkpoint_path = args.checkpoint_path
    is_edge_model = _is_edge_exported_checkpoint(checkpoint_path)
    if is_edge_model:
        # Fail fast (before the reasoner download) if this Diffusers build lacks
        # the Edge transformer API, instead of dying later on opaque missing keys.
        _validate_edge_transformer_support()
    edge_reasoner_path = None
    if is_edge_model and args.include_reasoner:
        if not args.save_pipeline:
            raise ValueError(
                "Including the Cosmos3 Edge reasoner requires --save-pipeline so the native Transformers "
                "repository can be written alongside the Diffusers components."
            )
        edge_reasoner_path = _resolve_edge_reasoner_path(args, exported_checkpoint_path=checkpoint_path)

    log.info("Instantiating model and loading weights from exported HF checkpoint …")
    log.info("Skipping source AVAE tokenizer instantiation during converter-only model load …")
    with _skip_source_tokenizer_load():
        _tmp = Cosmos3OmniModel.from_pretrained_dcp(checkpoint_path).model

    # Extract network components and architecture config from the Stage 1 HF export.
    language_model = _tmp.net.language_model
    vae2llm = _tmp.net.vae2llm
    llm2vae = _tmp.net.llm2vae
    time_embedder = _tmp.net.time_embedder
    # The language model may carry a nested VL config (e.g. Qwen3VLConfig);
    # the text-model fields read below live on its text config.
    lm_cfg = _tmp.net.language_model.config.get_text_config()
    net_cfg = _tmp.net.config
    model_cfg = _tmp.config
    patch_latent_dim = _tmp.net.patch_latent_dim
    hidden_size = _tmp.net.hidden_size
    num_attention_heads = _tmp.net.num_heads
    num_key_value_heads = _tmp.net.num_kv_heads
    head_dim = _tmp.net.head_dim
    num_hidden_layers = _tmp.net.num_hidden_layers
    latent_patch_size = _tmp.net.latent_patch_size
    latent_channel = _tmp.net.latent_channel
    timestep_scale = _tmp.net.timestep_scale
    base_fps = int(net_cfg.base_fps)
    enable_fps_modulation = net_cfg.enable_fps_modulation
    max_action_dim = _tmp.config.max_action_dim
    unified_3d_mrope_reset_spatial_ids = _tmp.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids
    unified_3d_mrope_temporal_modality_margin = (
        _tmp.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin
    )
    action_proj_in = getattr(_tmp.net, "action2llm", None)
    if action_proj_in is None:
        action_proj_in = getattr(_tmp.net, "action_proj_in", None)
    action_proj_out = getattr(_tmp.net, "llm2action", None)
    if action_proj_out is None:
        action_proj_out = getattr(_tmp.net, "action_proj_out", None)
    action_modality_embed = getattr(_tmp.net, "action_modality_embed", None)
    has_action_projection_weights = any(
        module is not None for module in (action_proj_in, action_proj_out, action_modality_embed)
    )
    action_gen = bool(
        _get_config_value(net_cfg, model_cfg, name="action_gen", default=False) or has_action_projection_weights
    )
    if action_gen and args.save_pipeline:
        _validate_edge_action_pipeline_support()
    action_dim = _get_config_value(net_cfg, model_cfg, name="action_dim", default=None)
    if action_dim is None and action_proj_in is not None:
        action_dim = getattr(action_proj_in, "input_size", None)
    if action_dim is None:
        action_dim = max_action_dim
    num_embodiment_domains = int(_get_config_value(net_cfg, model_cfg, name="num_embodiment_domains", default=32))
    sound2llm = getattr(_tmp.net, "sound2llm", None)
    llm2sound = getattr(_tmp.net, "llm2sound", None)
    sound_modality_embed = getattr(_tmp.net, "sound_modality_embed", None)
    has_sound_projection_weights = any(module is not None for module in (sound2llm, llm2sound, sound_modality_embed))
    sound_gen = bool(
        _get_config_value(net_cfg, model_cfg, name="sound_gen", default=False) or has_sound_projection_weights
    )
    sound_dim = _get_config_value(net_cfg, model_cfg, name="sound_dim", default=None)
    if sound_dim is None and sound2llm is not None:
        sound_dim = sound2llm.in_features
    sound_latent_fps = _get_config_value(net_cfg, model_cfg, name="sound_latent_fps", default=25.0)
    if sound_gen:
        missing_sound_modules = [
            name
            for name, module in (
                ("sound2llm", sound2llm),
                ("llm2sound", llm2sound),
                ("sound_modality_embed", sound_modality_embed),
            )
            if module is None
        ]
        if missing_sound_modules:
            raise RuntimeError(
                "Source checkpoint is configured for sound generation but is missing "
                f"sound projection weights: {missing_sound_modules}."
            )
        if sound_dim is None:
            raise RuntimeError("Source checkpoint is configured for sound generation but sound_dim is missing.")
    if action_gen:
        missing_action_modules = [
            name
            for name, module in (
                ("action_proj_in/action2llm", action_proj_in),
                ("action_proj_out/llm2action", action_proj_out),
                ("action_modality_embed", action_modality_embed),
            )
            if module is None
        ]
        if missing_action_modules:
            raise RuntimeError(
                "Source checkpoint is configured for action generation but is missing "
                f"action projection weights: {missing_action_modules}."
            )

    # A ViT sidecar can be built from any of three sources: the loaded reasoner
    # model's own `.visual` tower (only present when include_visual is truthy),
    # root `model.visual.*` weights, or a vision_encoder/ subfolder.
    has_source_visual = any(
        module is not None and next(module.parameters(), None) is not None
        for module in (
            getattr(_tmp, "visual", None),
            getattr(getattr(_tmp, "net", None), "visual", None),
            getattr(getattr(getattr(_tmp, "net", None), "language_model", None), "visual", None),
        )
    )
    has_vision_encoder_weights = (
        has_source_visual
        or _checkpoint_has_weight_prefix(checkpoint_path, VISION_ENCODER_CHECKPOINT_PREFIX)
        or bool(_checkpoint_vision_subfolder_files(checkpoint_path))
    )
    vision_gen = bool(
        _get_config_value(net_cfg, model_cfg, name="vision_gen", default=False) or has_vision_encoder_weights
    )
    want_vision_encoder = bool(args.save_pipeline and vision_gen and not args.skip_vision_encoder and not is_edge_model)
    # Only build the vision_encoder/ sidecar when the checkpoint actually ships
    # extractable ViT weights. A generation-only checkpoint (e.g. include_visual
    # unset) reports vision_gen=True but has no reasoner ViT to export — auto-skip
    # it with a warning instead of failing, so --skip-vision-encoder is not required.
    include_vision_encoder = want_vision_encoder and has_vision_encoder_weights
    vision_encoder = None
    if include_vision_encoder:
        vision_encoder = _load_vision_encoder(checkpoint_path, _tmp, args.vision_encoder_model, dtype)
    elif want_vision_encoder:
        log.warning(
            f"No extractable vision encoder weights found (no loaded reasoner ViT, no '{VISION_ENCODER_CHECKPOINT_PREFIX}*' "
            "root weights, no vision_encoder/ subfolder); skipping vision_encoder/ save. Convert a checkpoint exported "
            "with include_visual=True (or one bundling a vision_encoder/) to include the reasoner vision tower."
        )
    elif args.save_pipeline and vision_gen and args.skip_vision_encoder:
        log.info("Skipping vision_encoder/ save because --skip-vision-encoder was set.")
    del _tmp

    source_vlm_config = _get_config_value(model_cfg, name="vlm_config", default=None)
    source_model_instance = _get_config_value(source_vlm_config, name="model_instance", default=None)
    source_vlm_instance_config = _get_config_value(source_model_instance, name="config", default=None)
    hidden_act = _get_config_value(lm_cfg, name="hidden_act", default=None)
    if hidden_act is None:
        hidden_act = _get_config_value(lm_cfg, name="mlp_hidden_act", default="silu")
    qk_norm_for_text = bool(_get_config_value(source_vlm_instance_config, name="qk_norm_for_text", default=True))
    use_und_k_norm_for_gen = bool(
        _get_config_value(source_vlm_instance_config, name="use_und_k_norm_for_gen", default=False)
    )
    log.info(
        "Building Diffusers transformer from exported config: "
        f"hidden_act={hidden_act!r}, qk_norm_for_text={qk_norm_for_text}, "
        f"use_und_k_norm_for_gen={use_und_k_norm_for_gen}, action_gen={action_gen}"
    )

    # Build Diffusers Cosmos3OmniTransformer from the exported HF architecture config.
    transformer_kwargs: dict[str, Any] = {
        "attention_bias": lm_cfg.attention_bias,
        "attention_dropout": lm_cfg.attention_dropout,
        "base_fps": base_fps,
        "enable_fps_modulation": enable_fps_modulation,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "intermediate_size": lm_cfg.intermediate_size,
        "latent_channel": latent_channel,
        "latent_patch_size": latent_patch_size,
        "action_dim": action_dim,
        "action_gen": action_gen,
        "num_embodiment_domains": num_embodiment_domains,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_key_value_heads,
        "patch_latent_dim": patch_latent_dim,
        "rms_norm_eps": lm_cfg.rms_norm_eps,
        "rope_scaling": lm_cfg.rope_scaling,
        "rope_theta": lm_cfg.rope_theta,
        "sound_dim": sound_dim,
        "sound_gen": sound_gen,
        "sound_latent_fps": sound_latent_fps,
        "timestep_scale": timestep_scale,
        "unified_3d_mrope_reset_spatial_ids": unified_3d_mrope_reset_spatial_ids,
        "unified_3d_mrope_temporal_modality_margin": unified_3d_mrope_temporal_modality_margin,
        "vocab_size": lm_cfg.vocab_size,
    }
    # hidden_act / qk_norm_for_text / rope_axes_dim are only explicit constructor
    # arguments on newer Diffusers Cosmos3OmniTransformer builds. Older builds
    # (e.g. the upstreamed diffusers 0.39.0) fix hidden_act/qk_norm_for_text and
    # derive rope_axes_dim internally from rope_scaling, and reject these as
    # unexpected keyword arguments. Pass each only when the installed constructor
    # accepts it so conversion works on both old and new builds; warn (not silent)
    # when omitting so a non-default exported value is not quietly dropped.
    optional_transformer_kwargs = {
        "hidden_act": hidden_act,
        "qk_norm_for_text": qk_norm_for_text,
        "use_und_k_norm_for_gen": use_und_k_norm_for_gen,
        "rope_axes_dim": _get_config_value(lm_cfg.rope_scaling, name="mrope_section", default=None),
    }
    transformer_params = inspect.signature(Cosmos3OmniTransformer.__init__).parameters
    accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in transformer_params.values())
    for name, value in optional_transformer_kwargs.items():
        if accepts_var_kwargs or name in transformer_params:
            transformer_kwargs[name] = value
        else:
            log.warning(
                f"Installed Diffusers Cosmos3OmniTransformer does not accept {name!r}; omitting it "
                "(the checked-out Diffusers derives/fixes this field internally). Upgrade Diffusers to a "
                f"build that exposes it to honor the exported value ({name}={value!r})."
            )
    with init_empty_weights():
        transformer = Cosmos3OmniTransformer(**transformer_kwargs)
    state_dict = _remap_language_model_state_dict(language_model.state_dict())
    for k, v in vae2llm.state_dict().items():
        state_dict[f"proj_in.{k}"] = v
    for k, v in llm2vae.state_dict().items():
        state_dict[f"proj_out.{k}"] = v
    for k, v in time_embedder.state_dict().items():
        state_dict[f"time_embedder.{_remap_time_embedder_key(k)}"] = v
    if action_gen:
        for k, v in action_proj_in.state_dict().items():
            state_dict[f"action_proj_in.{k}"] = v
        for k, v in action_proj_out.state_dict().items():
            state_dict[f"action_proj_out.{k}"] = v
        state_dict["action_modality_embed"] = action_modality_embed
    if sound_gen:
        for k, v in sound2llm.state_dict().items():
            state_dict[f"audio_proj_in.{k}"] = v
        for k, v in llm2sound.state_dict().items():
            state_dict[f"audio_proj_out.{k}"] = v
        state_dict["audio_modality_embed"] = sound_modality_embed
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    del (
        language_model,
        vae2llm,
        llm2vae,
        time_embedder,
        action_proj_in,
        action_proj_out,
        action_modality_embed,
        sound2llm,
        llm2sound,
        sound_modality_embed,
        state_dict,
    )

    transformer = transformer.to(dtype=dtype)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_sound_tokenizer = (
        args.include_sound_tokenizer or sound_tokenizer_path is not None or (sound_gen and args.save_pipeline)
    )
    if include_sound_tokenizer and sound_tokenizer_path is None:
        raise ValueError(
            "The source checkpoint is configured for sound generation, so --sound-tokenizer-path "
            "is required when saving a full pipeline."
        )

    if is_edge_model and args.save_pipeline and args.skip_vision_encoder:
        log.info("Skipping standalone vision_encoder/ construction; the embedded Edge reasoner owns those weights.")

    fixed_step_sampler_cfg = _resolve_fixed_step_sampler_config(model_cfg)
    if args.distilled_scheduler == "on":
        if fixed_step_sampler_cfg is None:
            raise ValueError(
                "distilled_scheduler='on' was requested, but the checkpoint does not define "
                "`fixed_step_sampler_config`."
            )
        use_distilled_scheduler = True
    elif args.distilled_scheduler == "off":
        if fixed_step_sampler_cfg is not None:
            raise ValueError(
                "distilled_scheduler='off' was requested, but the checkpoint defines "
                "`fixed_step_sampler_config`. Exporting a UniPC scheduler for a distilled "
                "checkpoint produces a broken pipeline (it is trained for the fixed few-step "
                "schedule). Use 'auto' or 'on' instead."
            )
        use_distilled_scheduler = False
    else:
        use_distilled_scheduler = fixed_step_sampler_cfg is not None

    if use_distilled_scheduler and fixed_step_sampler_cfg is not None:
        log.info(
            "Detected distilled checkpoint scheduler config: "
            f"sample_type={fixed_step_sampler_cfg['sample_type']}, t_list={fixed_step_sampler_cfg['t_list']}"
        )

    if args.save_pipeline:
        tokenizer_source = str(edge_reasoner_path) if is_edge_model else "Qwen/Qwen3-VL-8B-Instruct"
        text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if is_edge_model:
            for token in ("<|vision_start|>", "<|vision_end|>"):
                token_id = text_tokenizer.convert_tokens_to_ids(token)
                if token_id is None or token_id < 0 or token_id >= transformer.config.vocab_size:
                    raise ValueError(
                        f"Cosmos3 Edge tokenizer token {token!r} has invalid ID {token_id!r} for "
                        f"vocab_size={transformer.config.vocab_size}."
                    )

        diffusers_vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
        )

        sound_tokenizer = None
        if include_sound_tokenizer:
            sound_tokenizer = _build_sound_tokenizer(sound_tokenizer_path, sound_tokenizer_config_path)

        if use_distilled_scheduler:
            assert fixed_step_sampler_cfg is not None
            # Distilled checkpoints are trained against a fixed flow-sigma schedule
            # (`fixed_step_sampler_config.t_list`). We export FlowMatchEulerDiscreteScheduler because:
            #   1) it is a flow-prediction scheduler (same parameterization expected
            #      by distilled Cosmos3 checkpoints),
            #   2) it supports explicit sigma injection through `set_timesteps(..., sigmas=...)`,
            #   3) `stochastic_sampling` maps directly to the distilled sample_type:
            #      `sde` -> stochastic, `ode` -> deterministic.
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=fixed_step_sampler_cfg["num_train_timesteps"],
                shift=1.0,
                use_dynamic_shifting=False,
                use_karras_sigmas=False,
                use_exponential_sigmas=False,
                use_beta_sigmas=False,
                invert_sigmas=False,
                stochastic_sampling=(fixed_step_sampler_cfg["sample_type"] == "sde"),
            )
            # Persist checkpoint-defined fixed-step settings so distilled inference can
            # call scheduler.set_timesteps(..., sigmas=t_list) at runtime.
            scheduler.register_to_config(
                fixed_step_sampler_config={
                    "t_list": fixed_step_sampler_cfg["t_list"],
                    "sample_type": fixed_step_sampler_cfg["sample_type"],
                },
                fixed_step_requires_explicit_sigmas=True,
            )
        else:
            # Karras schedule approximating FlowUniPCMultistepScheduler with shift=5, 35 steps.
            # Measured from that schedule: first flow-sigma=0.9998, last flow-sigma=0.1281.
            # EDM sigma = flow_sigma / (1 - flow_sigma), so:
            #   sigma_max = 0.9998 / 0.0002 = 4999  (but capped at 200 to avoid duplicate
            #               integer timesteps from Karras clustering near the top)
            #   sigma_min = 0.1281 / (1 - 0.1281)  = 0.1281 / 0.8719 ≈ 0.147
            scheduler = UniPCMultistepScheduler(
                use_karras_sigmas=True,
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
                sigma_max=200.0,
                sigma_min=0.147,
            )

        # enable_safety_checker=False: constructing the checker imports
        # cosmos_guardrail, which the converter environment does not ship.
        # Consumers can opt back in at load time via
        # Use the pipeline class exported by the checked-out Diffusers main.
        # Keep optional components gated by the constructor signature because
        # the main branch does not expose the legacy sound-tokenizer and safety
        # checker arguments used by older Cosmos3 pipeline implementations.
        pipeline_kwargs: dict[str, Any] = {
            "transformer": transformer,
            "text_tokenizer": text_tokenizer,
            "vae": diffusers_vae,
            "scheduler": scheduler,
        }
        pipeline_parameters = inspect.signature(Cosmos3OmniPipeline.__init__).parameters
        if "sound_tokenizer" in pipeline_parameters:
            pipeline_kwargs["sound_tokenizer"] = sound_tokenizer
        if "enable_safety_checker" in pipeline_parameters:
            pipeline_kwargs["enable_safety_checker"] = False
        pipeline = Cosmos3OmniPipeline(**pipeline_kwargs)
        log.info(f"Saving full pipeline to {output_dir} …")
        pipeline.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
        if vision_encoder is not None:
            # Not a Cosmos3OmniPipeline component — saved as a sidecar folder for
            # the transformers/vLLM consumers of the exported repository.
            log.info(f"Saving Qwen3-VL vision encoder to {output_dir / 'vision_encoder'} …")
            vision_encoder.save_pretrained(str(output_dir / "vision_encoder"), safe_serialization=True)
        if is_edge_model:
            assert edge_reasoner_path is not None
            _copy_edge_reasoner_metadata(edge_reasoner_path, output_dir)
            _normalize_edge_model_index(output_dir)
            _write_edge_transformer_config(output_dir, model_cfg)
    else:
        log.info(f"Saving transformer to {output_dir} …")
        transformer.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
        if include_sound_tokenizer:
            log.info("Skipping sound_tokenizer/ save because --save-pipeline was not set.")
        if vision_gen and not args.skip_vision_encoder:
            log.info("Skipping vision_encoder/ save because --save-pipeline was not set.")

    log.info("Done.")
