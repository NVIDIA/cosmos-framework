# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.utils.flags import TRAINING
from cosmos_framework.utils.easy_io.backends.base_backend import BaseStorageBackend
from cosmos_framework.utils.easy_io.backends.http_backend import HTTPBackend
from cosmos_framework.utils.easy_io.backends.local_backend import LocalBackend
from cosmos_framework.utils.easy_io.backends.registry_utils import backends, prefix_to_backends, register_backend

__all__ = [
    "BaseStorageBackend",
    "LocalBackend",
    "HTTPBackend",
    "register_backend",
    "backends",
    "prefix_to_backends",
]

if TRAINING:
    try:
        from cosmos_framework.utils.easy_io.backends.boto3_backend import Boto3Backend
    except ModuleNotFoundError as error:
        # Only a missing optional S3 client is tolerated; any other import failure surfaces here.
        if (error.name or "").split(".")[0] not in {"boto3", "botocore", "multistorageclient"}:
            raise
    else:
        __all__.append("Boto3Backend")
    try:
        from cosmos_framework.utils.easy_io.backends.msc_backend import MSCBackend
    except ModuleNotFoundError as error:
        # Only a missing optional S3 client is tolerated; any other import failure surfaces here.
        if (error.name or "").split(".")[0] not in {"boto3", "botocore", "multistorageclient"}:
            raise
    else:
        __all__.append("MSCBackend")
