# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attr


@attr.define(slots=False)
class DataSource:
    """DataSource configuration for Eagle datasets."""

    wdinfo_path: dict[str, str]
    text_keys: list[str]
    media_keys: list[str] | None = None
    bucket_name: str | None = None
    text_only: bool = False
    total_key_count: int = 1
    num_urls: int | None = None


@attr.define(slots=False)
class LocalDataSource:
    """DataSource configuration for manifest-indexed local datasets.

    Sibling of `DataSource` for datasets stored as per-sample files on local
    disk (media files + conversation JSONs) indexed by a JSON manifest, rather
    than as WebDataset tar shards indexed by wdinfo.json.

    Manifest schema:
        [{"id": "<sample_id>", "media": "<rel path to mp4>",
          "conversation": "<rel path to conversation JSON>"}, ...]

    Each conversation JSON references its media via `media_field_name` (e.g.
    `{"type": "video", "video": "video_0"}`), so the loader emits the media
    bytes under that same key into `data_dict["media"]`.
    """

    manifest_path: dict[str, str]  # {"train": "/abs/path/meta.json", "val": "..."}
    data_root: str  # directory the manifest's relative media/conversation paths resolve against
    media_field_name: str = "video_0"
    text_keys: list[str] = attr.Factory(lambda: ["texts"])
    media_keys: list[str] | None = attr.Factory(lambda: ["media"])
    text_only: bool = False
    total_key_count: int = 0
