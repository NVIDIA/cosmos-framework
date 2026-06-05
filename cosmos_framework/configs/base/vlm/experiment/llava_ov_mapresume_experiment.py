# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style, resumable variant of the LLaVA-OV VLM recipe for resume validation.

Materialises the first ``n`` items of a single LLaVA-OneVision-Data subset into
an in-memory ``datasets.Dataset`` (map-style) so that ``MapDistributor`` can
checkpoint exact ``(epoch, index)`` positions per worker.  The
``dataloader_state`` callback is wired with ``distributor_type="data_packer"``
so the ``DP_STATE_WORKER_<id>_{EPOCH,INDEX}`` env vars are set on resume, and
the ``MapDistributor`` fast-forwards each worker to the saved position.

Usage (smoke / dryrun)::

    python -m cosmos_framework.scripts.train \\
        --sft-toml=examples/toml/sft_config/llava_ov_mapresume.toml --dryrun -- \\
        data_setting.max_tokens=16000

Resume run::

    torchrun --nproc_per_node=4 --master_port=12344 \\
        -m cosmos_framework.scripts.train \\
        --sft-toml=examples/toml/sft_config/llava_ov_mapresume.toml -- \\
        data_setting.max_tokens=16000 \\
        checkpoint.load_path=/tmp/imaginaire4-output/<project>/<group>/<name>/checkpoints/iter_000000100 \\
        checkpoint.load_training_state=true
"""

from __future__ import annotations

import itertools
from typing import Any

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.data.vfm.dataflow import CosmosDataLoader, MapDistributor, PoolPackingBatcher
from cosmos_framework.data.vfm.processors import build_processor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX
from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMProcessor, VLMCollator
from cosmos_framework.callbacks.dataloader_state import DataLoaderStateCallback

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# Map-style data source factory
#
# Streams the first ``n`` filtered items and materialises them into a
# ``datasets.Dataset`` so MapDistributor can resume at exact sample positions.
# ---------------------------------------------------------------------------


def get_llava_ov_map(
    subset: str = "ai2d(gpt4v)",
    split: str = "train",
    n: int = 800,
) -> Any:
    """Materialize the first ``n`` filtered LLaVA-OV items into a map-style Dataset
    (so the dataloader is resumable via MapDistributor). Streams to avoid a full download.

    Args:
        subset: Dataset config/subset name (e.g. ``"ai2d(gpt4v)"``).
        split: Dataset split (default ``"train"``).
        n: Number of items to materialise.

    Returns:
        A ``datasets.Dataset`` (map-style) with columns from LLaVA-OV.
    """
    from datasets import load_dataset, Dataset

    stream = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=subset, split=split, streaming=True)
    stream = stream.filter(lambda x: x.get("image") is not None and len(x.get("conversations") or []) >= 2)
    items = list(itertools.islice(stream, n))
    return Dataset.from_list(items)


# ---------------------------------------------------------------------------
# Experiment registration
# ---------------------------------------------------------------------------


pre_exp012_llava_ov_mapresume = LazyDict(
    dict(
        # Same Hydra defaults as pre_exp012_llava_ov — pins VLM model, checkpoint
        # backend, and the basic_vlm + basic_log callback groups (the latter
        # includes dataloader_state).
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "basic_log"]},
            "_self_",
        ],
        job=dict(
            name="pre_exp012_llava_ov_mapresume_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            group="vlm_llava_ov_demo",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=10,
            logging_iter=1,
            run_validation=False,
            # Override dataloader_state.distributor_type to "data_packer" so the
            # DataLoaderStateCallback activates resume env vars (DP_STATE_WORKER_*)
            # on load_state_dict.  We cannot set data_setting.distributor_type
            # because its attrs validator only accepts "with_replace"/"no_replace".
            callbacks=dict(
                dataloader_state=L(DataLoaderStateCallback)(
                    distributor_type="data_packer",
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-5,
            fused=True,
        ),
        model=dict(
            config=dict(
                freeze=dict(
                    trainable_params=[".*"],
                ),
                parallelism=dict(
                    data_parallel_shard_degree=4,
                    data_parallel_replicate_degree=-1,
                ),
            ),
        ),
        # Local-only mode: disable object-store IO, keep load_path=??? sentinel.
        checkpoint=dict(
            save_iter=100,
            load_from_object_store=dict(enabled=False, credentials="", bucket=""),
            save_to_object_store=dict(enabled=False, credentials="", bucket=""),
        ),
        # Map-style CosmosDataLoader: MapDistributor wraps the materialised
        # Dataset so each worker gets a deterministic shard with a resumable
        # (epoch, index) position.  num_workers=0 keeps the worker bookkeeping
        # simple for resume tests (multi-worker resume is covered by unit tests).
        dataloader_train=L(CosmosDataLoader)(
            distributor=L(MapDistributor)(
                dataset=L(get_llava_ov_map)(subset="ai2d(gpt4v)", split="train", n=800),
                shuffle=True,
                seed=42,
                name="",
            ),
            processor=L(VLMProcessor)(
                processor=L(build_processor)(
                    tokenizer_type="${model.config.policy.backbone.model_name}",
                    config_variant="hf",
                ),
                ignore_index=IGNORE_INDEX,
            ),
            batcher=L(PoolPackingBatcher)(
                max_tokens=16000, pool_size=16, max_batch_size=1, long_threshold=6400,
            ),
            collator=L(VLMCollator)(),
            num_workers=0,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="pre_exp012_llava_ov_mapresume",
    node=pre_exp012_llava_ov_mapresume,
)
