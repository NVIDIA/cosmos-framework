# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import pytest

from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.helper_test import RunIf


def setup_s3():
    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "path_mapping": None,
            "s3_credential_path": "credentials/pbss_getty.secret",
        }
    )


@pytest.mark.L1("Requires data loading from S3.")
@RunIf(requires_file="credentials/pbss_getty.secret")
def test_s3_backend():
    setup_s3()
    for ith, _ in enumerate(easy_io.list_dir_or_file("s3://bucket6/")):
        if ith > 5:
            break

    easy_io.copyfile_from_local("pyproject.toml", "s3://bucket6/pyproject.toml")
    easy_io.remove("s3://bucket6/pyproject.toml")


@pytest.mark.L1("Requires data uploading to S3.")
@RunIf(requires_file="credentials/pbss_getty.secret")
def test_s3_dump():
    if easy_io.exists("s3://debug/test_00.mp4"):
        easy_io.remove("s3://debug/test_00.mp4")
    np_frames, metadata = easy_io.load("s3://debug/00.mp4")
    easy_io.dump(
        np_frames,
        "s3://debug/test_00.mp4",
        format="mp4",
        fps=metadata.get("fps", 30),
        codec=metadata.get("codec", "h264"),
    )
    if easy_io.exists("s3://debug/dummy_dict.pkl"):
        easy_io.remove("s3://debug/dummy_dict.pkl")

    dummy_dict = {"a": 1, "b": 2}
    easy_io.dump(dummy_dict, "s3://debug/dummy_dict.pkl")
    dummy_np = np.array([1, 2, 3])
    easy_io.dump(dummy_np, "s3://debug/dummy_np.npy")


@pytest.mark.L0
def test_local_backend():
    num_files = len(list(easy_io.list_dir_or_file(".")))
    assert num_files > 0

    if easy_io.exists("dummy_dict.pkl"):
        easy_io.remove("dummy_dict.pkl")

    dummy_dict = {"a": 1, "b": 2}
    easy_io.dump(dummy_dict, "dummy_dict.pkl")
    load_dict = easy_io.load("dummy_dict.pkl")
    for key in dummy_dict:
        assert dummy_dict[key] == load_dict[key]

    if easy_io.exists("dummy_dict.pkl"):
        easy_io.remove("dummy_dict.pkl")
