# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoPhy-2 local-disk data source.

After running ``prepare_videophy2_from_hf.py``, the dataset lives at::

    ${VIDEOPHYSICS_ROOT}/videophy2_train/  (and /videophy2_val/)
        meta.json
        media/video_XXXX.mp4
        text/conversation_XXXX.json

``VIDEOPHYSICS_ROOT`` defaults to ``examples/data/videophysics`` (matching the
launcher default) and can be overridden via env var.
"""

import os

from cosmos_framework.data.reasoner.local_dataset_utils import LocalDataSource

_DEFAULT_ROOT = "examples/data/videophysics"
VIDEOPHYSICS_ROOT = os.environ.get("VIDEOPHYSICS_ROOT", _DEFAULT_ROOT)

_TRAIN_DIR = os.path.join(VIDEOPHYSICS_ROOT, "videophy2_train")
_VAL_DIR = os.path.join(VIDEOPHYSICS_ROOT, "videophy2_val")

DATAINFO: dict[str, LocalDataSource] = {
    "videophy2_train": LocalDataSource(
        manifest_path={"train": os.path.join(_TRAIN_DIR, "meta.json")},
        data_root=_TRAIN_DIR,
        media_field_name="video_0",
        # total_key_count is informational; the loader reads len(manifest) at runtime.
        total_key_count=6000,
    ),
    "videophy2_val": LocalDataSource(
        manifest_path={"val": os.path.join(_VAL_DIR, "meta.json")},
        data_root=_VAL_DIR,
        media_field_name="video_0",
        total_key_count=3400,
    ),
}


def url_to_category(url) -> str | None:
    """Map a sample's ``__url__`` (or path string) to its category key.

    For single-manifest datasets the partitioning is one-to-one; we key on the
    parent directory name so the same function handles both train and val.
    """
    if hasattr(url, "root"):
        root = str(url.root)
    else:
        root = str(url)
    base = os.path.basename(os.path.normpath(root))
    if base in DATAINFO:
        return base
    return None
