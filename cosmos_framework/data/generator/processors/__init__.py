# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import sys
from types import SimpleNamespace
from typing import Optional

from transformers import PreTrainedTokenizerFast

from cosmos_framework.data.generator.processors.base import BaseVLMProcessor
from cosmos_framework.data.generator.processors.nemotron3densevl_processor import Nemotron3DenseVLProcessor
from cosmos_framework.data.generator.processors.nemotronvl_processor import NemotronVLProcessor
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos_framework.model.generator.tokenizers.tokenization_qwen2 import Qwen2Tokenizer
from cosmos_framework.utils.generator.reasoner.pretrained_models_downloader import maybe_download_hf_model_from_s3

_VARIANT_TO_CREDENTIALS = {
    "s3": ("credentials/s3_training.secret", "bucket4"),
    "gcp": ("credentials/gcp_checkpoint.secret", "bucket0"),
    # "hf" => no S3 backing store: pass empty credentials/bucket so the downloader
    # falls back to a direct HuggingFace Hub download (matches the legacy
    # ``download_tokenizer_files(model_name, "hf")`` behavior on origin/main, which
    # simply returned the model name and let from_pretrained pull from HF).
    "hf": ("", ""),
}

# S3 prefix under which HuggingFace model files are stored in the checkpoint buckets.
_LLM_S3_PREFIX = "cosmos3/pretrained/huggingface"


class LLMTokenizerProcessor(BaseVLMProcessor):
    """Wrapper that adapts a bare LLM tokenizer to the ``BaseVLMProcessor`` API.

    Used by LLM-only (no-vision) tokenizer configs so that all augmentors and
    model code can treat LLM-only and full VLM configs uniformly through the
    same ``proc.tokenizer`` / ``proc.tokenize_text`` surface. The base class
    handles ``tokenize_text`` / ``encode`` / ``decode``; we only need to wire
    up ``self.processor`` so ``.tokenizer`` resolves.
    """

    def __init__(self, tokenizer):
        self.processor = SimpleNamespace(tokenizer=tokenizer)


def _patch_nemotron_llm_tokenizer_vision_tokens(destination_dir: str) -> None:
    """Remap reserved placeholder tokens to vision special tokens in-place.

    The Nemotron LLM tokenizer reserves ``<SPECIAL_20>`` / ``<SPECIAL_21>``
    at IDs 20/21 -- the same slots the VLM tokenizer uses for
    ``<|vision_start|>`` / ``<|vision_end|>``.  Renaming them here keeps
    every vision-token ID inside the original vocab_size (131072) so no
    embedding-layer resize is needed during FSDP training.  The function is
    idempotent: re-applying it after the tokens are already renamed is a no-op.
    """
    remap = {"<SPECIAL_20>": "<|vision_start|>", "<SPECIAL_21>": "<|vision_end|>"}

    tokenizer_json_path = os.path.join(destination_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path) as f:
            data = json.load(f)
        for entry in data.get("added_tokens", []):
            if entry["content"] in remap:
                entry["content"] = remap[entry["content"]]
        vocab = data.get("model", {}).get("vocab", {})
        for old_name, new_name in remap.items():
            if old_name in vocab:
                vocab[new_name] = vocab.pop(old_name)
        with open(tokenizer_json_path, "w") as f:
            json.dump(data, f)

    tokenizer_config_path = os.path.join(destination_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tc_data = json.load(f)
        for entry in tc_data.get("added_tokens_decoder", {}).values():
            if entry.get("content") in remap:
                entry["content"] = remap[entry["content"]]
        with open(tokenizer_config_path, "w") as f:
            json.dump(tc_data, f)


def _download_llm_tokenizer(
    tokenizer_type: str,
    credentials: str,
    bucket: str,
    cache_dir: Optional[str] = None,
) -> str:
    return maybe_download_hf_model_from_s3(
        tokenizer_type,
        credentials=credentials,
        bucket=bucket,
        include_model_weights=False,
        cache_dir=cache_dir,
        s3_prefix=_LLM_S3_PREFIX,
    )


def build_processor(
    tokenizer_type: str,
    config_variant: Optional[str] = None,
    credentials: Optional[str] = None,
    bucket: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    # Local artifact path: source the processor from a bundled directory
    # (e.g. the top level of nvidia/Cosmos3-Nano, which ships its own
    # preprocessor_config.json, tokenizer.json, etc). Avoids the redundant
    # upstream Qwen/Qwen3-VL-*-Instruct fetch. Cosmos3-Nano/Super both ship
    # a Qwen3VL-compatible processor, so dispatch to Qwen3VLProcessor.
    if os.path.isdir(tokenizer_type):
        return Qwen3VLProcessor(tokenizer_type, cache_dir=cache_dir)
    if credentials is None or bucket is None:
        if config_variant is None:
            config_variant = "s3"
        if config_variant not in _VARIANT_TO_CREDENTIALS:
            raise ValueError(f"config_variant must be one of {list(_VARIANT_TO_CREDENTIALS)}, got {config_variant!r}")
        variant_credentials, variant_bucket = _VARIANT_TO_CREDENTIALS[config_variant]
        credentials = credentials if credentials is not None else variant_credentials
        bucket = bucket if bucket is not None else variant_bucket
    elif config_variant is not None:
        raise ValueError("Provide either config_variant or (credentials, bucket), not both")
    if "Qwen/Qwen3-VL" in tokenizer_type or "Siglip2-Qwen3-1.7B" in tokenizer_type:
        return Qwen3VLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16" in tokenizer_type:
        return NemotronVLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif (
        "NVIDIA-Nemotron-3-Dense-VL" in tokenizer_type
        or "Qwen3-2B-ViT" in tokenizer_type
        or "nvidia/Cosmos3-Reasoner-2B-Private" in tokenizer_type
        or "nvidia/Cosmos3-Edge-Reasoner" in tokenizer_type
    ):
        return Nemotron3DenseVLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif "Qwen/Qwen3-0.6B" in tokenizer_type:
        local_path = _download_llm_tokenizer(tokenizer_type, credentials, bucket, cache_dir)
        return LLMTokenizerProcessor(Qwen2Tokenizer.from_pretrained(local_path))
    elif "Nemotron/NVIDIA-Nemotron-3-2B-BF16" in tokenizer_type:
        local_path = _download_llm_tokenizer(tokenizer_type, credentials, bucket, cache_dir)
        _patch_nemotron_llm_tokenizer_vision_tokens(local_path)
        return LLMTokenizerProcessor(PreTrainedTokenizerFast.from_pretrained(local_path, trust_remote_code=True))
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported")


def build_processor_lazy(
    *args,
    repository: Optional[str] = None,
    revision: Optional[str] = None,
    subdir: str = "",
    **kwargs,
):
    """LazyCall wrapper that resolves ``build_processor`` on this module at call time.

    Two modes:
      1. Upstream tokenizer (legacy): pass ``tokenizer_type="<HF repo>"``
         (and optional ``config_variant`` / ``credentials`` / ``bucket``).
         The processor is sourced from the upstream HF repo (e.g.
         ``Qwen/Qwen3-VL-8B-Instruct``).
      2. Local artifact: pass ``repository`` + ``revision`` (and optional
         ``subdir``). The processor is sourced from the HF cache of the
         named artifact (e.g. ``nvidia/Cosmos3-Nano``), reusing the same
         revision the OmniModel checkpoint download uses. Avoids a
         redundant upstream Qwen3-VL-*-Instruct fetch.

    LazyCall captures its target at config-construction time, so a direct
    ``L(build_processor)`` would freeze the original function reference and
    bypass any later ``monkeypatch.setattr`` on this module's
    ``build_processor`` attribute. This wrapper performs a fresh module-level
    lookup on every call, so test fixtures patching ``build_processor`` are
    honored when the config is instantiated.
    """
    if repository is not None:
        from cosmos_framework.utils.checkpoint_db import CheckpointDirHf

        if revision is None:
            raise ValueError("'revision' is required when 'repository' is set")
        local_path = CheckpointDirHf(repository=repository, revision=revision).download()
        if subdir:
            local_path = os.path.join(local_path, subdir)
        return sys.modules[__name__].build_processor(local_path, **kwargs)
    return sys.modules[__name__].build_processor(*args, **kwargs)
