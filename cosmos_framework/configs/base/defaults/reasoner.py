# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Configs for VLM / LLM models

import math
import os
from numbers import Integral, Real
from typing import Any

import attrs
import torch.distributed as dist

from cosmos_framework.utils.flags import INTERNAL
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_framework.utils import log
from cosmos_framework.utils.config_helper import ConfigStore
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.model.generator.mot.unified_mot import (
    Nemotron3DenseVLMoTConfig,
    Nemotron3DenseVLTextForCausalLM,
    Qwen3MoTConfig,
    Qwen3VLMoeMoTConfig,
    Qwen3VLMoeTextForCausalLM,
    Qwen3VLMoTConfig,
    Qwen3VLTextForCausalLM,
)
from cosmos_framework.data.generator.processors import LLMTokenizerProcessor, build_processor_lazy
from cosmos_framework.model.generator.tokenizers.tokenization_qwen2 import Qwen2Tokenizer

_AUDIO_LAYOUTS = (
    "separate_no_timestamps",
    "separate_with_timestamps",
    "interleaved_av",
)


def create_vlm_config(base_config: LazyDict, **overrides):
    vlm_config = lazy_instantiate(base_config)
    for key, value in overrides.items():
        setattr(vlm_config, key, value)
    return vlm_config


def get_rank_safe() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0  # default to rank 0 when not in distributed mode


################################################################################
# Download tokenizer files from s3
# Download to ~/.cache/imaginaire4/tokenizer_files/{model_name} and then load from there.
def download_tokenizer_files(model_name: str, config_variant: str) -> str:
    if config_variant == "hf":
        return model_name

    if config_variant == "s3":
        ckpt_bucket = "bucket4"
        credentials = "credentials/s3_checkpoint.secret"
    elif config_variant == "s3_east2":
        ckpt_bucket = "nv-cosmos-checkpoint-us-east-2"
        credentials = "credentials/s3_east2_checkpoint.secret"
    elif config_variant == "gcp":
        ckpt_bucket = "bucket0"
        credentials = "credentials/gcp_checkpoint.secret"
    else:
        raise ValueError(f"Invalid config variant: {config_variant}")

    model_path = f"s3://{ckpt_bucket}/cosmos3/pretrained/huggingface/{model_name}"
    if not INTERNAL:
        from cosmos_framework.utils.checkpoint_db import download_checkpoint_v2

        model_path = download_checkpoint_v2(model_path)
        if "://" not in model_path:
            return model_path

    imaginaire_cache_dir = os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire4"))
    destination_dir = os.path.join(imaginaire_cache_dir, f"tokenizer_files/{model_name}/rank_{get_rank_safe()}")
    s3_backend_args = {
        "backend": "s3",
        "s3_credential_path": credentials,
    }

    extensions = ["json", "txt", "jinja"]
    for extension in extensions:
        for file_path in easy_io.list_dir_or_file(
            model_path,
            list_dir=False,
            list_file=True,
            suffix=extension,
            recursive=False,
            backend_args=s3_backend_args,
        ):
            full_path = easy_io.join_path(model_path, file_path, backend_args=s3_backend_args)
            local_path = f"{destination_dir}/{file_path}"
            if os.path.exists(local_path):
                log.debug(f"Skipping already downloaded tokenizer file: {local_path}")
                continue
            log.info(f"Downloading tokenizer file: {full_path} to {local_path}, cwd: {os.getcwd()}")
            # Download the file
            file_data = easy_io.get(full_path, backend_args=s3_backend_args)
            easy_io.put(file_data, local_path)
    return destination_dir


def create_qwen2_tokenizer_with_download(pretrained_model_name: str, config_variant: str):
    destination_dir = download_tokenizer_files(pretrained_model_name, config_variant)
    return LLMTokenizerProcessor(Qwen2Tokenizer.from_pretrained(destination_dir))


@attrs.define(slots=False)
class PretrainedWeightsConfig:
    # Master switch. When False, the trainer skips the pretrained-weights load
    # path entirely.
    enabled: bool = True

    # Path to the pretrained-weights snapshot for the backbone. Accepts s3://,
    # gs://, hf://, or a local filesystem path. Empty means no overlay; the
    # trainer falls back to whatever the AutoModel constructor produces.
    backbone_path: str = ""

    # Path to the credentials .secret used to fetch backbone_path from object
    # storage. Empty means anonymous (works for hf:// and public buckets).
    credentials_path: str = ""

    # Apply the boto3 GCS-compatibility patch when loading DCP shards from a
    # gs:// URI. Required for DCP loads from GCS; harmless otherwise.
    enable_gcs_patch_in_boto3: bool = False

    # Force a specific safetensors weight remapping (e.g. "qwen3" vs
    # "nemotron_3_dense_vl" / "nemotron_3_llm"). None lets the loader auto-detect.
    checkpoint_format: str | None = None


@attrs.define(slots=False)
class VLMConfig:
    """VLM backbone identity shared by OmniMoTModelConfig.vlm_config and VLMModelConfig.policy.backbone.

    model_instance and tokenizer are typed Any | None instead of LazyDict | None
    because OmegaConf 2.3 rejects LazyDict as a structured-config annotation; the
    runtime value is still a LazyDict.
    """

    # HuggingFace model identifier or local path. Drives AutoConfig + AutoModel selection.
    model_name: str = ""

    # Safetensor path for model for load a safetensor from different folder
    safetensors_path: str = ""

    # Optional pretrained-weights overlay (separate from the AutoModel structural
    # init driven by model_name).
    pretrained_weights: PretrainedWeightsConfig = PretrainedWeightsConfig()

    # Optional LazyCall override for the language-model class to instantiate.
    # When set, the trainer routes construction through lazy_instantiate(model_instance)
    # instead of the AutoModelForCausalLM / AutoModelForVision2Seq from_pretrained path.
    model_instance: Any | None = None

    # Optional LazyCall override for the tokenizer/processor (a BaseVLMProcessor
    # subclass). When None, callers may auto-derive via AutoTokenizer.from_pretrained.
    tokenizer: Any | None = None

    # Override class name for the decoder layer (e.g. "Qwen2MoTDecoderLayer"); the
    # substring "Mo" gates MoE detection in cosmos3_vfm_network. None means no swap.
    layer_module: str | None = None

    # Apply QK normalization in the language-model decoder.
    qk_norm: bool = False

    # Whether input and output word-embedding matrices are tied. Affects the
    # safetensors loader (lm_head load is skipped when tied) and the FSDP wrapper.
    tie_word_embeddings: bool = False

    # Prepend a system prompt during text tokenization. Checkpoints trained with
    # system prompt enabled require this set true at inference time.
    use_system_prompt: bool = False


@attrs.define(slots=False)
class SoundUnderstandingConfig:
    """Optional audio-understanding pathway for standalone Reasoner/VLM models.

    The encoder artifact initializes the frozen modality encoder independently
    of the Reasoner checkpoint. The audio projector is trainable by default and
    may be frozen explicitly after alignment.
    """

    # These strings are registered in the target tokenizer only when the
    # standalone Reasoner ``sound_und`` gate is enabled. The model embedding
    # table is not resized; the resolved IDs must fit its reserved capacity.
    # Qwen-based Nano/Super have spare preallocated rows for the defaults below.
    # Edge's 131072-row vocabulary is full, so model setup reuses its existing
    # <SPECIAL_23>/<SPECIAL_24>/<SPECIAL_25> reserved slots.
    audio_start_token: str = "<audio_start>"
    audio_pad_token: str = "<audio_pad>"
    audio_end_token: str = "<audio_end>"
    projection_hidden_size: int = 4096
    # Virtual video FPS used only to synthesize timestamp anchors for audio-only
    # samples. This is neither the waveform sample rate nor the audio-token rate.
    # Paired audio/video samples ignore this value and reuse the video timestamps.
    audio_timestamp_fps: float = 4.0
    # Keep the separate timestamped sequence as the compatibility default.
    # ``separate_no_timestamps`` omits only audio-side timestamps, while
    # ``interleaved_av`` inserts each audio segment after its native video chunk.
    audio_layout: str = "separate_with_timestamps"
    freeze_projector: bool = False
    # Keep the full input embedding table as an optimizer parameter while
    # allowing gradients only through the audio start/end token positions.
    train_boundary_token_embeddings_only: bool = False

    encoder_checkpoint_enabled: bool = False
    encoder_checkpoint_path: str = ""
    encoder_checkpoint_credentials_path: str = ""
    exclude_frozen_encoder_from_training_checkpoint: bool = False


def validate_sound_understanding_config(config: Any, *, sound_und: bool) -> None:
    """Validate the optional standalone Reasoner audio configuration."""
    if not isinstance(sound_und, bool):
        raise TypeError(f"sound_und must be a bool, got {type(sound_und).__name__}")
    audio_tokens = (config.audio_start_token, config.audio_pad_token, config.audio_end_token)
    if sound_und and any(not isinstance(token, str) or not token for token in audio_tokens):
        raise ValueError("Audio start, pad, and end tokens must be non-empty strings")
    if sound_und and len(set(audio_tokens)) != 3:
        raise ValueError("Audio start, pad, and end tokens must be distinct")
    if (
        isinstance(config.projection_hidden_size, bool)
        or not isinstance(config.projection_hidden_size, Integral)
        or config.projection_hidden_size <= 0
    ):
        raise ValueError("sound_und_config.projection_hidden_size must be positive")
    if (
        isinstance(config.audio_timestamp_fps, bool)
        or not isinstance(config.audio_timestamp_fps, Real)
        or not math.isfinite(float(config.audio_timestamp_fps))
        or config.audio_timestamp_fps <= 0
    ):
        raise ValueError("sound_und_config.audio_timestamp_fps must be positive and finite")
    if config.audio_layout not in _AUDIO_LAYOUTS:
        raise ValueError(f"sound_und_config.audio_layout must be one of {_AUDIO_LAYOUTS}, got {config.audio_layout!r}")
    for name, value in (
        ("freeze_projector", config.freeze_projector),
        ("train_boundary_token_embeddings_only", config.train_boundary_token_embeddings_only),
        ("encoder_checkpoint_enabled", config.encoder_checkpoint_enabled),
        ("exclude_frozen_encoder_from_training_checkpoint", config.exclude_frozen_encoder_from_training_checkpoint),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"sound_und_config.{name} must be a bool, got {type(value).__name__}")
    if sound_und and not config.encoder_checkpoint_enabled:
        raise ValueError("Audio understanding requires an enabled standalone encoder checkpoint")
    if config.encoder_checkpoint_enabled and not config.encoder_checkpoint_path.strip():
        raise ValueError("An enabled audio encoder checkpoint requires a non-empty path")
    if config.exclude_frozen_encoder_from_training_checkpoint and not config.encoder_checkpoint_enabled:
        raise ValueError("Excluding the frozen audio encoder from training checkpoints requires checkpoint loading")


# Configs for LLM models
Qwen3MoT_LLM_0p6b_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-0.6B",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3MoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/llm/qwen3/configs/Qwen3-0.6B.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-0.6B",
        config_variant="hf",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket4/cosmos3/pretrained/huggingface/Qwen/Qwen3-0.6B/",
        credentials_path="credentials/s3_training.secret",
    ),
)

Qwen3MoT_LLM_0p6b_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-0.6B",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3MoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/llm/qwen3/configs/Qwen3-0.6B.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-0.6B",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Qwen/Qwen3-0.6B/",
        credentials_path="credentials/gcp_checkpoint.secret",
    ),
)

Nemotron3_LLM_2b_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/NVIDIA-Nemotron-3-2B-BF16",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Nemotron/NVIDIA-Nemotron-3-2B-BF16",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Nemotron/NVIDIA-Nemotron-3-2B-BF16/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_llm",
    ),
)

# Configs for VL instruct models

# Config for Qwen3VL 30B A3B Instruct model
# Qwen3VLMoE uses Qwen2Tokenizer
Qwen3VLMoT_VLM_30b_a3b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    model_instance=L(Qwen3VLMoeTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoeMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl_moe/configs/Qwen3-VL-30B-A3B-Instruct.json"
            ),
            layer_module="Qwen3VLMoeTextMoTDecoderLayer",
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-30B-A3B-Instruct",
        config_variant="s3",
    ),
    layer_module="Qwen3VLMoeTextMoTDecoderLayer",
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket4/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct/",
        credentials_path="credentials/s3_training.secret",
    ),
)


Qwen3VLMoT_VLM_30b_a3b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    model_instance=L(Qwen3VLMoeTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoeMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl_moe/configs/Qwen3-VL-30B-A3B-Instruct.json"
            ),
            layer_module="Qwen3VLMoeTextMoTDecoderLayer",
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-30B-A3B-Instruct",
        config_variant="gcp",
    ),
    layer_module="Qwen3VLMoeTextMoTDecoderLayer",
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

# Config for Qwen3VL 235B A22B Instruct model
# Qwen3VLMoE uses Qwen2Tokenizer
Qwen3VLMoT_VLM_235b_a22b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    model_instance=L(Qwen3VLMoeTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoeMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl_moe/configs/Qwen3-VL-235B-A22B-Instruct.json"
            ),
            layer_module="Qwen3VLMoeTextMoTDecoderLayer",
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-235B-A22B-Instruct",
        config_variant="gcp",
    ),
    layer_module="Qwen3VLMoeTextMoTDecoderLayer",
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-235B-A22B-Instruct/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)


# Config for Qwen3VL 2B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_2b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="s3",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket4/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-2B-Instruct/",
        credentials_path="credentials/s3_training.secret",
    ),
)

Qwen3VLMoT_VLM_2b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-2B-Instruct/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Qwen3VLMoT_VLM_2b_Instruct_HF_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="hf",
    ),
)

Nemotron3DenseVL_VLM_2b_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Nemotron-3-Dense-VL-2B-BF16-Alignment",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Nemotron/NVIDIA-Nemotron-3-Dense-VL-2B-BF16-Alignment",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Nemotron/NVIDIA-Nemotron-3-Dense-VL-2B-BF16-Alignment/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_dense_vl",
    ),
)

Cosmos3Reasoner_Nemotron_VLM_2b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Reasoner-2B-Private",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="nvidia/Cosmos3-Reasoner-2B-Private",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Reasoner-2B-Private/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_dense_vl",
    ),
)

CosmosReason2_VLM_2b_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos-Reason2-2B",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-2B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-2B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos-Reason2-2B/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

# Config for Qwen3VL 4B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_4b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-4B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-4B-Instruct",
        config_variant="s3",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket4/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-4B-Instruct/",
        credentials_path="credentials/s3_training.secret",
    ),
)

Qwen3VLMoT_VLM_4b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-4B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-4B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-4B-Instruct/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

# Config for Qwen3VL 8B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_8b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="s3",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket4/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-8B-Instruct/",
        credentials_path="credentials/s3_training.secret",
    ),
)

Qwen3VLMoT_VLM_8b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-8B-Instruct/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Cosmos3Reasoner_VLM_8b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Reasoner-8B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-8B-Private/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Cosmos3NanoReasoner_VLM_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Nano-Reasoner",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Nano-Reasoner/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Cosmos3NanoReasoner_VLM_GCP_Config_0517: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Nano-Reasoner",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Nano-Reasoner-bb9c6f5/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Cosmos3NanoReasoner_VLM_S3_EAST2_Config_0517: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Nano-Reasoner",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
        config_variant="s3_east2",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://nv-cosmos-checkpoint-us-east-2/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Nano-Reasoner-bb9c6f5/",
        credentials_path="credentials/s3_east2_checkpoint.secret",
    ),
)

# Config for Qwen3VL 32B Instruct model
# Qwen3VL uses Qwen2Tokenizer
Qwen3VLMoT_VLM_32b_Instruct_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="s3",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket4/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-32B-Instruct/",
        credentials_path="credentials/s3_training.secret",
    ),
)

Qwen3VLMoT_VLM_32b_Instruct_GCP_Config: VLMConfig = VLMConfig(
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-32B-Instruct/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Cosmos3Reasoner_VLM_32b_Private_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Reasoner-32B-Private",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-32B-Private/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Cosmos3SuperReasoner_VLM_GCP_Config: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Super-Reasoner",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Super-Reasoner/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

Cosmos3SuperReasoner_VLM_GCP_Config_0517: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Super-Reasoner",
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
            ),
            qk_norm_for_text=True,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Super-Reasoner-b6df0d1/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
    ),
)

# Cosmos3-Edge-Reasoner at commit 4acb717.
# Edge reasoner architecture (model_type "cosmos3_edge" in the public
# nvidia/Cosmos3-Edge; historically remote-code "nemotron_siglip2"):
# Nemotron text backbone (56-block hybrid layout, 2048 hidden)
# + SigLIP2 vision encoder.  The text transformer is identical in shape to
# Nemotron-3-Dense-VL-2B (hidden_size=2048, 56 alternating attn/MLP blocks → 28
# effective MoT layers after _transform_text_dict).  Uses the same
# nemotron_3_dense_vl weight remapping and config JSON.
Cosmos3EdgeReasoner_VLM_GCP_Config_4acb717: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Edge-Reasoner",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="nvidia/Cosmos3-Edge-Reasoner",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Edge-Reasoner-4acb717/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_dense_vl",
    ),
)

# Cosmos3-Edge-Reasoner at commit 9b4c028 (2026-05-29).
# Same Edge reasoner (cosmos3_edge) architecture as 4acb717; new weights uploaded 2026-05-29.
Cosmos3EdgeReasoner_VLM_GCP_Config_9b4c028: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Edge-Reasoner",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="nvidia/Cosmos3-Edge-Reasoner",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Edge-Reasoner-9b4c028/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_dense_vl",
    ),
)

# Cosmos3-Edge-Reasoner at commit 590c1c0 (2026-06-28).
# Updated weights uploaded 2026-06-28; bit-identical to the reasoner tower
# shipped inside the public nvidia/Cosmos3-Edge release.
Cosmos3EdgeReasoner_VLM_GCP_Config_590c1c0: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Edge-Reasoner",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="nvidia/Cosmos3-Edge-Reasoner",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Edge-Reasoner-590c1c0/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_dense_vl",
    ),
)

# Same as 590c1c0 but with use_und_k_norm_for_gen=True: normalises K_und before
# it is used as a key in the gen→und cross-attention path (the qk-norm fix for the
# generator). Adds a freshly-initialised k_norm_und_for_gen RMSNorm.
Cosmos3EdgeReasoner_VLM_GCP_Config_590c1c0_UndKNorm: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Edge-Reasoner",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
            use_und_k_norm_for_gen=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="nvidia/Cosmos3-Edge-Reasoner",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Edge-Reasoner-590c1c0/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_dense_vl",
    ),
)

# Same as 9b4c028 but with use_und_k_norm_for_gen=True: normalises K_und before
# it is used as a key in the gen→und cross-attention path.
Cosmos3EdgeReasoner_VLM_GCP_Config_9b4c028_UndKNorm: VLMConfig = VLMConfig(
    model_name="nvidia/Cosmos3-Edge-Reasoner",
    model_instance=L(Nemotron3DenseVLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                json_file="cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
            ),
            qk_norm_for_text=False,
            use_und_k_norm_for_gen=True,
        ),
    ),
    tokenizer=L(build_processor_lazy)(
        tokenizer_type="nvidia/Cosmos3-Edge-Reasoner",
        config_variant="gcp",
    ),
    pretrained_weights=PretrainedWeightsConfig(
        backbone_path="s3://bucket0/cosmos3/pretrained/huggingface/nvidia/Cosmos3-Edge-Reasoner-9b4c028/",
        credentials_path="credentials/gcp_checkpoint.secret",
        enable_gcs_patch_in_boto3=True,
        checkpoint_format="nemotron_3_dense_vl",
    ),
)


def register_vlm():
    cs = ConfigStore.instance()
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_mot_0p6b",
        node=Qwen3MoT_LLM_0p6b_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_mot_0p6b_gcp",
        node=Qwen3MoT_LLM_0p6b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="nemotron_3_llm_2b_gcp",
        node=Nemotron3_LLM_2b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_30b_a3b_instruct",
        node=Qwen3VLMoT_VLM_30b_a3b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_30b_a3b_instruct_gcp",
        node=Qwen3VLMoT_VLM_30b_a3b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_235b_a22b_instruct_gcp",
        node=Qwen3VLMoT_VLM_235b_a22b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_2b_instruct",
        node=Qwen3VLMoT_VLM_2b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_2b_instruct_gcp",
        node=Qwen3VLMoT_VLM_2b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_2b_instruct_hf",
        node=Qwen3VLMoT_VLM_2b_Instruct_HF_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="nemotron_3_dense_vl_2b_gcp",
        node=Nemotron3DenseVL_VLM_2b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_reasoner_nemotron_vlm_2b_private_gcp",
        node=Cosmos3Reasoner_Nemotron_VLM_2b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos_reason2_vlm_2b_gcp",
        node=CosmosReason2_VLM_2b_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_reasoner_vlm_8b_private_gcp",
        node=Cosmos3Reasoner_VLM_8b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_nano_reasoner_vlm_gcp",
        node=Cosmos3NanoReasoner_VLM_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_nano_reasoner_vlm_gcp_0517",
        node=Cosmos3NanoReasoner_VLM_GCP_Config_0517,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_nano_reasoner_vlm_s3_east2_0517",
        node=Cosmos3NanoReasoner_VLM_S3_EAST2_Config_0517,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_reasoner_vlm_32b_private_gcp",
        node=Cosmos3Reasoner_VLM_32b_Private_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_super_reasoner_vlm_gcp",
        node=Cosmos3SuperReasoner_VLM_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_super_reasoner_vlm_gcp_0517",
        node=Cosmos3SuperReasoner_VLM_GCP_Config_0517,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_4b_instruct",
        node=Qwen3VLMoT_VLM_4b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_4b_instruct_gcp",
        node=Qwen3VLMoT_VLM_4b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_8b_instruct",
        node=Qwen3VLMoT_VLM_8b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_8b_instruct_gcp",
        node=Qwen3VLMoT_VLM_8b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_32b_instruct",
        node=Qwen3VLMoT_VLM_32b_Instruct_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="qwen3_vl_mot_vlm_32b_instruct_gcp",
        node=Qwen3VLMoT_VLM_32b_Instruct_GCP_Config,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_edge_reasoner_vlm_gcp_4acb717",
        node=Cosmos3EdgeReasoner_VLM_GCP_Config_4acb717,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_edge_reasoner_vlm_gcp_9b4c028",
        node=Cosmos3EdgeReasoner_VLM_GCP_Config_9b4c028,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_edge_reasoner_vlm_gcp_9b4c028_und_k_norm",
        node=Cosmos3EdgeReasoner_VLM_GCP_Config_9b4c028_UndKNorm,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_edge_reasoner_vlm_gcp_590c1c0",
        node=Cosmos3EdgeReasoner_VLM_GCP_Config_590c1c0,
    )
    cs.store(
        group="vlm_config",
        package="model.config.vlm_config",
        name="cosmos3_edge_reasoner_vlm_gcp_590c1c0_und_k_norm",
        node=Cosmos3EdgeReasoner_VLM_GCP_Config_590c1c0_UndKNorm,
    )


def register_sound_understanding() -> None:
    """Register opt-in sound-understanding presets for standalone Reasoner models."""
    ConfigStore.instance().store(
        group="sound_und_config",
        package="model.config.sound_und_config",
        name="nemotron_3_nano_omni_parakeet_encoder_gcp",
        node=SoundUnderstandingConfig(
            encoder_checkpoint_enabled=True,
            encoder_checkpoint_path=(
                "s3://bucket0/cosmos3/pretrained/huggingface/"
                "Nemotron/Nemotron-3-Nano-Omni-Parakeet-Encoder-BF16/"
            ),
            encoder_checkpoint_credentials_path="credentials/gcp_checkpoint.secret",
        ),
    )
