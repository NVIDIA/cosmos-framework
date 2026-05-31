# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import attrs

# We are hardcoding the unittest assets in this file.

# add codeowner for cosmos_framework/model/vfm/tokenizers


@attrs.define(slots=False)
class SwfitStackPDXrConfig:
    """
    Config for the cluster specific information.
    Everything cluster specific should be here.
    """

    object_store_bucket_data: str
    object_store_credential_data: str


UNITTEST_CONFIG = SwfitStackPDXrConfig(
    object_store_bucket_data="unittest",
    object_store_credential_data="credentials/pdx_dir.secret",
)

TOKENIZER_RECONSTRUCTION_VIDEO_PATH = "tokenizer/video/panda70m_test_0000039_00000.mp4"
AVAE_RECONSTRUCTION_AUDIO_PATH = "tokenizer/audio/test_audio.wav"
