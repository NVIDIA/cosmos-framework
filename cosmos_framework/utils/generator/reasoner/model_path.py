# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Resolve staged VLM training checkpoints without changing converter downloads."""

from pathlib import Path

from cosmos_framework.utils.generator.reasoner.pretrained_models_downloader import maybe_download_hf_model_from_s3


def resolve_hf_model_path(
    model_name_or_path: str,
    credentials: str,
    bucket: str,
    *,
    include_model_weights: bool,
    require_s3_exists: bool = False,
) -> str:
    """Use a staged local directory or retain the normal remote download contract."""
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    return maybe_download_hf_model_from_s3(
        model_name_or_path,
        credentials,
        bucket,
        include_model_weights=include_model_weights,
        require_s3_exists=require_s3_exists,
    )
