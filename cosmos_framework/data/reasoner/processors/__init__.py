# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Optional

from cosmos_framework.data.reasoner.processors.nemotron3densevl_processor import Nemotron3DenseVLProcessor
from cosmos_framework.data.reasoner.processors.nemotronvl_processor import NemotronVLProcessor
from cosmos_framework.data.reasoner.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos_framework.utils.reasoner.pretrained_models_downloader import resolve_hf_model_store


def build_processor(
    tokenizer_type: str,
    cache_dir: Optional[str] = None,
    credentials: str = "./credentials/s3_training.secret",
    bucket: str = "bucket4",
    processor_family: str = "auto",
):
    if processor_family not in ("auto", "dense_vl"):
        raise ValueError("processor_family must be auto or dense_vl")
    credentials, bucket = resolve_hf_model_store(credentials, bucket)
    if processor_family == "dense_vl" or "nvidia/Cosmos3-Edge" in tokenizer_type:
        return Nemotron3DenseVLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif "Qwen3-VL" in tokenizer_type or "Siglip2-Qwen3-1.7B" in tokenizer_type:
        return Qwen3VLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    elif "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16" in tokenizer_type:
        return NemotronVLProcessor(tokenizer_type, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported")
