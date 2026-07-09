# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style action SFT dataset: ``DROIDLeRobotDataset`` → ``ActionTransformPipeline``.

The base ``DROIDLeRobotDataset.__getitem__`` returns the raw sample
(``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``idle_frames``). The model expects each sample to be passed through
``ActionTransformPipeline`` (spatial resize/pad, text tokenization, action
padding to ``max_action_dim``, and ``sequence_plan`` construction). This thin
wrapper composes the two so the experiment can hand a single map-style dataset
to ``RankPartitionedDataLoader`` (mirroring how the vision recipe uses
``get_sft_dataset``).
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import (
    DROIDLeRobotDataset,
)
from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample."""

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transform(self._dataset[idx], self._resolution)

    def get_shuffle_blocks(self):
        """Delegate to the inner DROIDLeRobotDataset (per-episode/segment flat-index blocks)."""
        return self._dataset.get_shuffle_blocks()


class ActionIterableShuffleDataset(IterableDataset):
    """Streaming view of a map-style ``ActionSFTDataset``.

    Each ``(rank, worker)`` is assigned a DISJOINT subset of episodes (sharded over
    ``shard_world_size * num_workers``), shuffles its episode ORDER, and streams the
    windows WITHIN each episode sequentially -> within-rank batch diversity (the N
    workers of a rank stream N different episodes) AND cross-rank diversity, while
    keeping reads sequential (I/O locality + COW; no RandomSampler random-access OOM).
    Re-shuffles each epoch and streams indefinitely (the trainer stops at ``max_iter``).

    ``shard_world_size`` / ``shard_rank`` are set by ``RankPartitionedDataLoader``.
    """

    def __init__(self, dataset: "ActionSFTDataset", seed: int = 42):
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    def __iter__(self):
        import torch

        blocks = self._dataset.get_shuffle_blocks()
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * nw + wid
        total_shards = max(1, int(self.shard_world_size) * nw)
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch)  # same permutation across all (rank,worker) -> disjoint shard
            order = torch.randperm(len(blocks), generator=g).tolist()
            for b in order[global_shard::total_shards]:
                start, length = blocks[b]
                for idx in range(start, start + length):
                    yield self._dataset[idx]
            epoch += 1


def get_action_droid_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    action_space: str = "joint_pos",
    mode: str = "policy",
    use_state: bool = True,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    use_success_only: bool = True,
) -> Dataset:
    """Build the DROID action SFT dataset: ``action_space='joint_pos'`` (8D) +
    ``use_state`` (raw/un-normalized), concat_view, chunk_length 32.

    Reads ``root`` (a merged/versioned DROID LeRobot root) as a single flat
    dataset; ``use_success_only=True`` filters to the ``success/`` split."""
    shard_kwargs = dict(
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,  # i4: bundles random-crop+resize+ColorJitter
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
        use_success_only=use_success_only,
    )
    dataset: Dataset = DROIDLeRobotDataset(root=root, **shard_kwargs)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_libero_sft_dataset(
    *,
    root: str,
    fps: float = 20.0,
    chunk_length: int = 16,
    image_size: int = 256,
    mode: str = "policy",
    camera_mode: str = "concat_view",
    action_space: str = "frame_wise_relative",
    rotation_space: str = "6d",
    pose_coordinate_frame: str = "native",
    action_normalization: str | None = "quantile_rot",
    action_stats_path: str | None = None,
    split: str = "train",
    val_ratio: float = 0.01,
    seed: int = 0,
    resolution: str | int | None = None,
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the LIBERO action-policy SFT dataset (GA reproduction defaults).

    Feeds ``LIBEROLeRobotDataset`` (frame-wise-relative rot6d actions,
    ``quantile_rot``-normalized, concat_view third-person + wrist at 256x256 each
    → 256x512) through ``ActionTransformPipeline``. ``root`` is a LOCAL LeRobot dir
    (read parquet + video directly); pre-sync the HF dataset once, e.g.
    ``hf download lerobot/libero_10 --repo-type dataset --local-dir <root>``. Point
    ``root`` at libero_10 alone. The
    dataset is FPS-agnostic (decodes at real frame timestamps); ``fps`` is metadata
    for ``conditioning_fps`` / prompt duration.
    """
    dataset = LIBEROLeRobotDataset(
        root=root,
        image_size=image_size,
        chunk_length=chunk_length,
        fps=fps,
        mode=mode,
        split=split,
        val_ratio=val_ratio,
        seed=seed,
        camera_mode=camera_mode,
        action_space=action_space,
        rotation_space=rotation_space,
        pose_coordinate_frame=pose_coordinate_frame,
        action_normalization=action_normalization,
        action_stats_path=action_stats_path,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
