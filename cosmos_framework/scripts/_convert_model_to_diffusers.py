# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Implementation for `convert_model_to_diffusers.py`."""

import contextlib
import json
import pathlib
import re

import pydantic
import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers_cosmos3 import Cosmos3OmniDiffusersPipeline, Cosmos3OmniTransformer
from transformers import AutoConfig, AutoTokenizer

from cosmos_framework.inference.model import Cosmos3OmniModel
from cosmos_framework.utils import log
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel

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

SOUND_TOKENIZER_MODEL_INDEX_ENTRY = [
    "diffusers",
    "Cosmos3AVAEAudioTokenizer",
]

LEGACY_SOUND_TOKENIZER_CHECKPOINT_NAME = "model.safetensors"
DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME = "diffusion_pytorch_model.safetensors"

# Wrapper prefixes that may appear on every key of a legacy AVAE state dict
# (DDP wrappers, full-model saves, training exports). Stripped iteratively
# until each key reaches a recognised target prefix.
_SOUND_TOKENIZER_PER_KEY_PREFIXES = ("module.", "generator.", "model.", "state_dict.")
_SOUND_TOKENIZER_TARGET_PREFIXES = ("decoder.", "encoder.", "bottleneck.")

# Inside a residual unit the legacy `nn.Sequential` layout was
# [snake1, conv1, snake2, conv2]; map sub-index → named attribute.
_SOUND_TOKENIZER_RES_UNIT_INNER_NAMES = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}

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


def _sound_tokenizer_filter_decoder(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop encoder and bottleneck keys — only the decoder is used at inference."""
    return {k: v for k, v in state_dict.items() if k.startswith("decoder.")}


def _sound_tokenizer_infer_num_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    """Count the OobleckDecoderBlocks present in a flat-`Sequential` legacy
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
    diffusers OobleckDecoder named-attribute layout.

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
        if (key.endswith(".alpha") or key.endswith(".beta")) and value.ndim == 1:
            value = value.unsqueeze(0).unsqueeze(-1).contiguous()
        out[key] = value
    return out


def _sound_tokenizer_reapply_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """If a Conv layer has the folded `.weight` tensor but neither `.weight_g`
    nor `.weight_v`, reconstruct the pair so the resulting checkpoint can be
    loaded into the weight-norm-wrapped diffusers OobleckDecoder.

    `weight_norm` with `dim=0` parameterises `weight = g * v / ||v||`. Setting
    `v = weight` and `g = ||weight||` along all non-zero axes is an exact
    inverse: `g * v / ||v|| = ||weight|| * weight / ||weight|| = weight`.
    """
    out = dict(state_dict)
    candidate_keys = [
        key
        for key in state_dict
        if key.endswith(".weight") and any(f".{layer}." in key for layer in ("conv1", "conv2", "conv_t1"))
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


def _remap_time_embedder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap a legacy `TimestepEmbedder` state dict (with `nn.Sequential`-style
    `mlp.0` / `mlp.2` keys) to the diffusers `TimestepEmbedding` layout
    (`linear_1` / `linear_2`). Keys not in `_TIME_EMBEDDER_KEY_REMAP` pass
    through unchanged.
    """
    return {_TIME_EMBEDDER_KEY_REMAP.get(key, key): value for key, value in state_dict.items()}


def _remap_sound_tokenizer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the full legacy → diffusers conversion pipeline to a sound
    tokenizer state dict: strip prefixes, drop non-decoder keys, remap the
    flat `nn.Sequential` layout to named attributes, reshape Snake1d params,
    and reconstruct `weight_g` / `weight_v` for any folded conv weights.
    """
    state_dict = _sound_tokenizer_strip_per_key_prefixes(state_dict)
    state_dict = _sound_tokenizer_filter_decoder(state_dict)
    if not state_dict:
        raise RuntimeError("Sound tokenizer state dict has no `decoder.*` keys after prefix stripping.")
    state_dict = _sound_tokenizer_remap_flat_layout(state_dict)
    state_dict = _sound_tokenizer_reshape_snake_params(state_dict)
    state_dict = _sound_tokenizer_reapply_weight_norm(state_dict)
    if any(re.match(r"decoder\.layers\.\d+", key) for key in state_dict):
        raise RuntimeError("Flat `decoder.layers.*` keys remain after remap; conversion is incomplete.")
    return state_dict


def _load_sound_tokenizer_config(config_path: pathlib.Path | None, fallback_config_path: pathlib.Path) -> dict:
    selected_config_path = config_path
    if selected_config_path is None and fallback_config_path.exists():
        selected_config_path = fallback_config_path
    if selected_config_path is None:
        return dict(DEFAULT_SOUND_TOKENIZER_CONFIG)
    with open(selected_config_path, encoding="utf-8") as f:
        return json.load(f)


def _save_sound_tokenizer(
    output_dir: pathlib.Path,
    checkpoint_path: pathlib.Path,
    config_path: pathlib.Path | None,
    remap_keys: bool = False,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("Saving AVAE tokenizer weights requires safetensors.") from exc

    sound_tokenizer_dir = output_dir / "sound_tokenizer"
    sound_tokenizer_dir.mkdir(parents=True, exist_ok=True)

    config = _load_sound_tokenizer_config(config_path, sound_tokenizer_dir / "config.json")
    with open(sound_tokenizer_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    log.info(f"Loading AVAE sound tokenizer weights from {checkpoint_path} …")
    state_dict = _load_sound_tokenizer_state_dict(checkpoint_path)
    if remap_keys:
        log.info("Remapping AVAE sound tokenizer state dict to diffusers OobleckDecoder layout …")
        state_dict = _remap_sound_tokenizer_state_dict(state_dict)
    output_filename = (
        DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME if remap_keys else LEGACY_SOUND_TOKENIZER_CHECKPOINT_NAME
    )
    log.info(f"Saving AVAE sound tokenizer to {sound_tokenizer_dir / output_filename} …")
    save_file(state_dict, str(sound_tokenizer_dir / output_filename), metadata={"format": "pt"})


def _add_sound_tokenizer_to_model_index(output_dir: pathlib.Path) -> None:
    model_index_path = output_dir / "model_index.json"
    if not model_index_path.exists():
        return
    with open(model_index_path, encoding="utf-8") as f:
        model_index = json.load(f)
    model_index["sound_tokenizer"] = SOUND_TOKENIZER_MODEL_INDEX_ENTRY
    with open(model_index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")


def _checkpoint_weight_map(checkpoint_path: pathlib.Path) -> dict[str, str]:
    index_path = checkpoint_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return index.get("weight_map", {})


def _checkpoint_has_weight_prefix(checkpoint_path: pathlib.Path, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in _checkpoint_weight_map(checkpoint_path))


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


def _load_vision_encoder(
    checkpoint_path: pathlib.Path,
    source_model,
    model_name_or_path: str,
    dtype: torch.dtype,
):
    state_dict = _get_source_vision_state_dict(source_model)
    if state_dict is None:
        log.info(f"Loading Qwen3-VL vision encoder weights from {checkpoint_path} …")
        state_dict = _load_prefixed_safetensors_state_dict(checkpoint_path, VISION_ENCODER_CHECKPOINT_PREFIX)
    else:
        log.info("Extracting Qwen3-VL vision encoder weights from loaded source model …")
    log.info(f"Building Qwen3-VL vision encoder from {model_name_or_path} …")
    return _build_vision_encoder(state_dict, model_name_or_path, dtype)


@contextlib.contextmanager
def _skip_source_sound_tokenizer_load():
    original_set_up_tokenizers = OmniMoTModel.set_up_tokenizers

    def set_up_tokenizers_without_sound(self):
        if not getattr(self.config, "sound_gen", False):
            return original_set_up_tokenizers(self)

        sound_gen = self.config.sound_gen
        self.config.sound_gen = False
        try:
            return original_set_up_tokenizers(self)
        finally:
            self.config.sound_gen = sound_gen

    OmniMoTModel.set_up_tokenizers = set_up_tokenizers_without_sound
    try:
        yield
    finally:
        OmniMoTModel.set_up_tokenizers = original_set_up_tokenizers


class Args(pydantic.BaseModel):
    checkpoint_path: pathlib.Path
    """Named checkpoint (e.g. 'Cosmos3-Nano') or path to DCP checkpoint dir."""
    output: str
    """Directory to save the converted diffusers model."""
    save_pipeline: bool = False
    """Save the full pipeline (transformer + VAE + tokenizer + scheduler)."""
    dtype: str = "bf16"
    """Dtype to save the transformer in."""
    sound_tokenizer_path: str | None = None
    """Optional AVAE sound tokenizer checkpoint to save under sound_tokenizer/."""
    sound_tokenizer_config_path: str | None = None
    """Optional AVAE config JSON to save under sound_tokenizer/config.json."""
    include_sound_tokenizer: bool = False
    """Require saving sound_tokenizer/ even if the source transformer is video-only."""
    remap_sound_tokenizer_keys: bool = False
    """Convert the legacy AVAE state dict into the diffusers OobleckDecoder
    layout (strips wrapper prefixes, drops encoder/bottleneck keys, remaps the
    flat `nn.Sequential` decoder to named attributes, reshapes Snake1d
    alpha/beta, and reconstructs `weight_g` / `weight_v` for folded conv
    weights). When enabled, the saved file is `diffusion_pytorch_model.safetensors`
    instead of `model.safetensors`."""
    remap_time_embedder_keys: bool = False
    """Remap the transformer's `time_embedder` state dict from the legacy
    `nn.Sequential` layout (`mlp.0.*` / `mlp.2.*`) to the diffusers
    `TimestepEmbedding` layout (`linear_1.*` / `linear_2.*`). Off by default —
    keys are forwarded verbatim."""
    vision_encoder_model: str = DEFAULT_VISION_ENCODER_MODEL
    """Qwen3-VL model/config to instantiate model.visual.* weights."""
    skip_vision_encoder: bool = False
    """Do not save vision_encoder/ when saving a full pipeline."""


def convert_model_to_diffusers(args: Args) -> None:
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
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

    log.info("Instantiating model and loading weights from DCP checkpoint …")
    log.info("Skipping source AVAE tokenizer instantiation during converter-only model load …")
    with _skip_source_sound_tokenizer_load():
        _tmp = Cosmos3OmniModel.from_pretrained_dcp(checkpoint_path).model

    # Extract network components and architecture config from DCP model
    language_model = _tmp.net.language_model
    vae2llm = _tmp.net.vae2llm
    llm2vae = _tmp.net.llm2vae
    time_embedder = _tmp.net.time_embedder
    lm_cfg = _tmp.net.language_model.config
    net_cfg = _tmp.net.config
    model_cfg = _tmp.config
    vlm_cfg = _tmp.net.config.vlm_config
    patch_latent_dim = _tmp.net.patch_latent_dim
    hidden_size = _tmp.net.hidden_size
    num_attention_heads = _tmp.net.num_heads
    num_key_value_heads = _tmp.net.num_kv_heads
    head_dim = _tmp.net.head_dim
    num_hidden_layers = _tmp.net.num_hidden_layers
    latent_patch_size = _tmp.net.latent_patch_size
    latent_channel = _tmp.net.latent_channel
    timestep_scale = _tmp.net.timestep_scale
    use_moe = _tmp.net.use_moe
    joint_attn_implementation = net_cfg.joint_attn_implementation
    base_fps = int(net_cfg.base_fps)
    enable_fps_modulation = net_cfg.enable_fps_modulation
    max_action_dim = _tmp.config.max_action_dim
    position_embedding_type = net_cfg.position_embedding_type
    unified_3d_mrope_reset_spatial_ids = _tmp.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids
    unified_3d_mrope_temporal_modality_margin = (
        _tmp.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin
    )
    video_temporal_causal = net_cfg.video_temporal_causal
    action2llm = getattr(_tmp.net, "action2llm", None)
    llm2action = getattr(_tmp.net, "llm2action", None)
    action_modality_embed = getattr(_tmp.net, "action_modality_embed", None)
    has_action_projection_weights = any(
        module is not None for module in (action2llm, llm2action, action_modality_embed)
    )
    action_gen = bool(
        _get_config_value(net_cfg, model_cfg, name="action_gen", default=False) or has_action_projection_weights
    )
    action_dim = _get_config_value(net_cfg, model_cfg, name="action_dim", default=None)
    if action_dim is None and action2llm is not None:
        action_dim = getattr(action2llm, "input_size", None)
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
    temporal_compression_factor_sound = _get_config_value(
        net_cfg, model_cfg, name="temporal_compression_factor_sound", default=1
    )
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
                ("action2llm", action2llm),
                ("llm2action", llm2action),
                ("action_modality_embed", action_modality_embed),
            )
            if module is None
        ]
        if missing_action_modules:
            raise RuntimeError(
                "Source checkpoint is configured for action generation but is missing "
                f"action projection weights: {missing_action_modules}."
            )

    has_vision_encoder_weights = _checkpoint_has_weight_prefix(checkpoint_path, VISION_ENCODER_CHECKPOINT_PREFIX)
    vision_gen = bool(
        _get_config_value(net_cfg, model_cfg, name="vision_gen", default=False) or has_vision_encoder_weights
    )
    include_vision_encoder = bool(args.save_pipeline and vision_gen and not args.skip_vision_encoder)
    vision_encoder = None
    if include_vision_encoder:
        vision_encoder = _load_vision_encoder(checkpoint_path, _tmp, args.vision_encoder_model, dtype)
    elif args.save_pipeline and vision_gen and args.skip_vision_encoder:
        log.info("Skipping vision_encoder/ save because --skip-vision-encoder was set.")
    del _tmp

    # Init diffusers Cosmos3OmniTransformer with full architecture config from DCP
    with init_empty_weights():
        transformer = Cosmos3OmniTransformer(
            attention_bias=lm_cfg.attention_bias,
            attention_dropout=lm_cfg.attention_dropout,
            base_fps=base_fps,
            enable_fps_modulation=enable_fps_modulation,
            freeze_und=vlm_cfg.freeze_und,
            head_dim=head_dim,
            hidden_act=lm_cfg.hidden_act,
            hidden_size=hidden_size,
            initializer_range=lm_cfg.initializer_range,
            intermediate_size=lm_cfg.intermediate_size,
            joint_attn_implementation=joint_attn_implementation,
            latent_channel=latent_channel,
            latent_patch_size=latent_patch_size,
            action_dim=action_dim,
            action_gen=action_gen,
            max_action_dim=max_action_dim,
            max_position_embeddings=lm_cfg.max_position_embeddings,
            model_type=lm_cfg.model_type,
            num_embodiment_domains=num_embodiment_domains,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            patch_latent_dim=patch_latent_dim,
            position_embedding_type=position_embedding_type,
            qk_norm_for_diffusion=True,
            qk_norm_for_text=vlm_cfg.qk_norm_for_text,
            rms_norm_eps=lm_cfg.rms_norm_eps,
            rope_scaling=lm_cfg.rope_scaling,
            rope_theta=lm_cfg.rope_theta,
            sound_dim=sound_dim,
            sound_gen=sound_gen,
            sound_latent_fps=sound_latent_fps,
            temporal_compression_factor_sound=temporal_compression_factor_sound,
            timestep_scale=timestep_scale,
            unified_3d_mrope_reset_spatial_ids=unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=unified_3d_mrope_temporal_modality_margin,
            use_cache=lm_cfg.use_cache,
            use_moe=use_moe,
            video_temporal_causal=video_temporal_causal,
            vocab_size=lm_cfg.vocab_size,
        )
    state_dict = language_model.state_dict()
    for k, v in vae2llm.state_dict().items():
        state_dict[f"vae2llm.{k}"] = v
    for k, v in llm2vae.state_dict().items():
        state_dict[f"llm2vae.{k}"] = v
    time_embedder_state = time_embedder.state_dict()
    if args.remap_time_embedder_keys:
        log.info("Remapping transformer time_embedder state dict to diffusers TimestepEmbedding layout …")
        time_embedder_state = _remap_time_embedder_state_dict(time_embedder_state)
    for k, v in time_embedder_state.items():
        state_dict[f"time_embedder.{k}"] = v
    if action_gen:
        for k, v in action2llm.state_dict().items():
            state_dict[f"action2llm.{k}"] = v
        for k, v in llm2action.state_dict().items():
            state_dict[f"llm2action.{k}"] = v
        state_dict["action_modality_embed"] = action_modality_embed
    if sound_gen:
        for k, v in sound2llm.state_dict().items():
            state_dict[f"sound2llm.{k}"] = v
        for k, v in llm2sound.state_dict().items():
            state_dict[f"llm2sound.{k}"] = v
        state_dict["sound_modality_embed"] = sound_modality_embed
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    del (
        language_model,
        vae2llm,
        llm2vae,
        time_embedder,
        action2llm,
        llm2action,
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

    if args.save_pipeline:
        text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        diffusers_vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
        )

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

        pipeline = Cosmos3OmniDiffusersPipeline(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=diffusers_vae,
            scheduler=scheduler,
            vision_encoder=vision_encoder,
        )
        log.info(f"Saving full pipeline to {output_dir} …")
        pipeline.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
        if include_sound_tokenizer:
            _save_sound_tokenizer(
                output_dir,
                sound_tokenizer_path,
                sound_tokenizer_config_path,
                remap_keys=args.remap_sound_tokenizer_keys,
            )
            _add_sound_tokenizer_to_model_index(output_dir)
    else:
        log.info(f"Saving transformer to {output_dir} …")
        transformer.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
        if include_sound_tokenizer:
            log.info("Skipping sound_tokenizer/ save because --save-pipeline was not set.")
        if vision_gen and not args.skip_vision_encoder:
            log.info("Skipping vision_encoder/ save because --save-pipeline was not set.")

    log.info("Done.")
