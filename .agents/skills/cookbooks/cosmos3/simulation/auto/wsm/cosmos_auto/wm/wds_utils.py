# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import io
from pathlib import Path

import numpy as np
from termcolor import cprint
from webdataset import TarWriter, WebDataset, non_empty


def get_sample(url):
    """Get a sample from a URL with basic auto-decoding."""
    if isinstance(url, Path):
        url = url.as_posix()

    dataset = WebDataset(url, nodesplitter=non_empty, workersplitter=None, shardshuffle=False).decode()
    return next(iter(dataset))


def write_to_tar(sample, output_file):
    if type(output_file) == str:
        output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # write tar file
    sink = TarWriter(str(output_file))
    sink.write(sample)
    sink.close()
    cprint(f"Saved {output_file}", "green")


def encode_dict_to_npz_bytes(data_dict):
    buffer = io.BytesIO()
    np.savez(buffer, **data_dict)
    buffer.seek(0)

    return buffer.getvalue()
