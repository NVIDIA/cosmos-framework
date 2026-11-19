# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepared YAM windows with per-example FD/ID selection and rank/worker sharding."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from failuregen.yam_training.contract import (
    CAMERAS,
    CHUNK,
    FPS,
    compose_views,
    decode_frames,
    episode_actions,
    file_sha256,
    load_prepared,
    normalize,
    training_contract,
)


class YamReelDataset(IterableDataset):
    """Use the prepared joint or EEF contract for training, validation and replay.

    Window sharding (not episode sharding) prevents empty workers on the seven
    validation episodes. The common epoch permutation is disjoint across workers.
    FD/ID selection has a separate worker-specific seeded RNG.
    """

    def __init__(
        self,
        prepared_root: str,
        dataset_root: str,
        split: str = "train",
        tokenizer_config=None,
        max_action_dim: int = 64,
        seed: int = 42,
        ffmpeg: str = "ffmpeg",
        action_space: str = "joint",
    ):
        super().__init__()
        self.root, self.dataset_root = Path(prepared_root), Path(dataset_root)
        manifest, self.stats = load_prepared(self.root)
        self.contract = manifest["contract"]
        if self.contract != training_contract(action_space):
            raise ValueError("Prepared action representation differs from the selected training recipe")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split}")
        self.split, self.seed, self.ffmpeg = split, seed, ffmpeg
        self.episodes = [ep for ep in manifest["episodes"] if ep["split"] == split]
        self.actions, self.frame_indices = {}, {}
        self.windows = []
        for ep in self.episodes:
            path = self.root / ep["states"]
            if file_sha256(path) != ep["states_sha256"]:
                raise ValueError(f"Prepared states changed: {path}")
            with np.load(path) as arrays:
                self.actions[ep["episode"]] = episode_actions(arrays, self.contract)
                self.frame_indices[ep["episode"]] = arrays["source_indices"]
            for key in CAMERAS:
                video = self.dataset_root / ep["videos"][key]
                if video.stat().st_size != ep["video_sizes"][key]:
                    raise ValueError(f"Source video changed: {video}")
            self.windows.extend((ep, start) for start in ep["valid_starts"])
        if not self.windows:
            raise ValueError(f"No eligible {split} windows")
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=0.1 if split == "train" else 0,
            max_action_dim=max_action_dim,
            format_prompt_as_json=True,
            append_idle_frames=False,
        )
        self.shard_world_size, self.shard_rank = 1, 0

    def __len__(self):
        return len(self.windows)

    def sample(self, index: int, mode: str):
        if mode not in {"forward_dynamics", "inverse_dynamics"}:
            raise ValueError(mode)
        ep, start = self.windows[index]
        first = int(self.frame_indices[ep["episode"]][start])
        frames = [
            decode_frames(self.dataset_root / ep["videos"][key], first, CHUNK + 1, ffmpeg=self.ffmpeg)
            for key in CAMERAS
        ]
        mosaic = compose_views(*frames)
        actions = self.actions[ep["episode"]][start : start + CHUNK]
        sample = {
            "video": torch.from_numpy(mosaic).permute(3, 0, 1, 2).contiguous(),
            "action": torch.from_numpy(normalize(actions, self.stats)),
            "ai_caption": ep["caption"],
            "conditioning_fps": torch.tensor(FPS, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self.contract["domain_id"], dtype=torch.long),
            "viewpoint": "concat_view",
        }
        return self.transform(sample, "720")

    def __iter__(self):
        worker = get_worker_info()
        workers, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        shards = int(self.shard_world_size) * workers
        shard = int(self.shard_rank) * workers + worker_id
        if shards > len(self.windows):
            raise ValueError("More loader workers than YAM windows; reduce num_workers")
        task_rng = np.random.default_rng(self.seed + 1_000_003 * (shard + 1))
        epoch = 0
        while True:
            permutation = np.random.default_rng(self.seed + epoch).permutation(len(self.windows))
            for ordinal, index in enumerate(permutation[shard::shards]):
                choose_id = bool(task_rng.integers(2)) if self.split == "train" else bool((ordinal + shard) % 2)
                yield self.sample(int(index), "inverse_dynamics" if choose_id else "forward_dynamics")
            if self.split == "train":
                epoch += 1
