# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Storage selection shared by internal distillation and strict checkpoint loading."""

from torch.distributed.checkpoint.filesystem import FileSystemReader

from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader


def get_storage_reader(checkpoint_path: str, credential_path: str | None = None) -> FileSystemReader:
    if "s3://" in checkpoint_path and credential_path:
        storage_reader = S3StorageReader(credential_path=credential_path, path=checkpoint_path)
    else:
        storage_reader = FileSystemReader(checkpoint_path)
    return storage_reader
