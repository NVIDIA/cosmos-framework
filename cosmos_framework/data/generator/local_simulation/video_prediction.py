# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Target-only video prediction from an existing encoded Transfer bundle.

Controls are deliberately not read. The singleton view identity -1 is the model's
non-rig sentinel, not a learned driving-camera identity. This data adapter does
not assert that a user's bundle reproduces the reference HQ-video distribution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.augmentor_provider import get_video_augmentor_v3_json_caption
from cosmos_framework.data.generator.augmentors.merge_datadict import NonRigVideoMetadata
from cosmos_framework.data.generator.indexed_transfer_webdataset_loader import _IndexedTarSampleURL
from cosmos_framework.data.generator.lance.augment_timing import iter_augmented_with_timing
from cosmos_framework.data.generator.lance.sample_rng import isolated_sample_rng
from cosmos_framework.data.generator.local_simulation.core import local_path
from cosmos_framework.data.generator.local_simulation.dataset import (
    LocalSimulationDataset,
    SampleRejectedError,
    tokenizer_config,
)
from cosmos_framework.utils.lazy_config import instantiate


class LocalVideoPredictionDataset(LocalSimulationDataset):
    """Apply the reference single-video transformations to local encoded media."""

    def __init__(
        self, bundle_path: str | Path, *, resolution_hw: tuple[int, int], num_video_frames: int = 400, seed: int = 42
    ) -> None:
        super().__init__(bundle_path, model="transfer", seed=seed)
        self.resolution_hw = tuple(resolution_hw)
        if self.resolution_hw not in ((480, 832), (720, 1280)):
            raise ValueError("Prediction resolution must select the reference 480 or 720 tier")
        if num_video_frames < 1:
            raise ValueError("num_video_frames must be positive")
        self.resolution = "480" if self.resolution_hw == (480, 832) else "720"
        self.num_video_frames = num_video_frames
        # Included in the existing local-loader resume digest. A changed data
        # policy must not silently resume the old reader cursor.
        self.manifest = {
            **self.manifest,
            "prediction_policy": {
                "resolution_hw": self.resolution_hw,
                "num_video_frames": num_video_frames,
                "view_id": -1,
                "augmentor": "video_basic_augmentor_v3_json_caption",
                "preprocessing_version": 2,
                "use_system_prompt": False,
            },
        }

    def _pipeline(self, stratum: str) -> list[Any]:
        del stratum
        if self._pid != os.getpid():
            self._pipelines = {}
            self._pid = os.getpid()
        if "prediction" not in self._pipelines:
            # Use the same provider and arguments as the reference HQ-video
            # stream, including chunk parsing, aspect-preserving resize and
            # reflection padding. Only the encoded-byte source is local.
            augmentors = get_video_augmentor_v3_json_caption(
                resolution=self.resolution,
                dataset_resolution_type=f"gt{self.resolution}p",
                caption_key="caption",
                caption_config={"json_field_dropout_rate": 0.0},
                tokenizer_config=tokenizer_config(self.root),
                cfg_dropout_rate=0.1,
                use_system_prompt=False,
                conditioning_config={0: 0.5, 1: 0.5},
                num_video_frames=self.num_video_frames,
                max_num_frames=1 + 4 * ((self.num_video_frames - 1) // 4),
                min_stride=1,
                max_stride=1,
                use_dynamic_fps=False,
                resize_on_read=True,
                extract_audio=False,
                append_duration_fps_timestamps=True,
                append_resolution_info=True,
            )
            self._pipelines["prediction"] = [instantiate(item) for item in augmentors.values()] + [
                NonRigVideoMetadata(input_keys=[], output_keys=["view_indices_selection"])
            ]
        return self._pipelines["prediction"]

    def materialize(self, record: dict[str, Any], draw_seed: int) -> dict[str, Any]:
        entry = record["entry"]
        raw = {
            "video": local_path(self.root, record["assets"]["rgb"]).read_bytes(),
            "metas": {
                "width": entry["source_width"],
                "height": entry["source_height"],
                "framerate": entry["source_fps"],
                "nb_frames": entry["source_frames"],
                "caption": json.dumps(
                    {
                        "window": {
                            "start_frame": entry["caption_start"],
                            "end_frame": entry["caption_end"],
                            "caption": entry["caption_json"],
                        }
                    }
                ),
            },
            "__key__": record["identity"],
            "__url__": _IndexedTarSampleURL("local-video", record["identity"], entry["aspect_ratio"]),
            "source_dataset_name": "general_video_prediction",
        }
        with isolated_sample_rng(draw_seed):
            sample = next(iter_augmented_with_timing([raw], self._pipeline(record["stratum"])), None)
        if sample is None:
            raise SampleRejectedError(f"Video prediction transforms rejected {record['identity']}")
        sample["num_frames"] = int(sample["video"].shape[1])
        sample["dataset_name"] = "general_video_prediction"
        if sample["view_indices_selection"].tolist() != [-1]:
            raise ValueError("General video must not acquire a physical driving-camera identity")
        sample.update(local_sample_id=record["identity"], local_epoch=self.epoch)
        return sample
