# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoPhy-2 SFT recipe on the Cosmos3-Edge reasoner.

Mirrors ``videophy2_sft_nano`` but targets the Cosmos3-Edge reasoner
(Nemotron-2B-Dense-VL LM + SigLIP2 vision tower, ``model_type="cosmos3_edge"``)
instead of Qwen3-VL-8B. ``model_name`` is the public omni release
``nvidia/Cosmos3-Edge`` and supplies arch/config/tokenizer AND the reasoner
weights: the training loader follows the snapshot's root safetensors index, so
weights load directly from the download — no converter step.
``VLM_SAFETENSORS_PATH`` is an OPTIONAL launcher override for loading from a
local safetensors directory instead (same mechanism as nano/super).

Deltas vs ``videophy2_sft_nano`` (everything else is identical):
    * ``override /vlm_policy``: ``qwen3_vl_8b_instruct`` -> ``cosmos3_edge_reasoner``.
    * ``optimizer.lr_multipliers``: nano sets ``{"model.visual": 1.0}`` to lift its
      projector (which sits UNDER ``model.visual`` in Qwen) off the inherited default
      ``0.1x``; edge OMITS the override. Edge's projector is a top-level
      ``model.projector`` (the Edge reasoner's PatchMerger), so the default
      ``{"model.visual": 0.1}`` only matches the frozen SigLIP2 tower and the
      projector already trains at the uniform ``1.0x`` -- same net effect as fixed
      nano (projector + LM + lm_head all at ``1e-6``), no key needed.
    * ``model.config.freeze``: ``freeze_vision_encoder=True`` only supports the
      known Qwen/Intern towers (vlm_model.py ``_get_vision_encoder_modules``), not
      the Edge (``cosmos3_edge``) tower, so freeze the SigLIP2 tower via the regex
      ``frozen_params=[r"model\\.visual\\."]`` instead — same intent (vision
      frozen, projector + LM trainable).

The dataflow helpers below are duplicated from ``videophy2_sft_nano`` on purpose
(NOT imported): the reasoner config loader reloads experiment modules with
``reload=True`` in alphabetical order, so importing nano's ``build_..`` function
here would capture a pre-reload object and fail the dataloader-worker pickle
identity check. Keeping the recipe self-contained makes every LazyCall reference
this module's own (co-reloaded) callables.

Launch via examples/launch_sft_videophy2_edge.sh after running
prepare_videophy2_from_hf to populate $VIDEOPHYSICS_ROOT.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.data.generator.dataflow import CosmosDataLoader, IterableDistributor, PoolPackingBatcher
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.data.reasoner.local_sft_dataset import LocalSFTDataset
from cosmos_framework.data.reasoner.data_sources_videophy2.videophy2 import DATAINFO
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX
from cosmos_framework.configs.base.reasoner.experiment.dataflow_roles import VLMCollator
from cosmos_framework.configs.base.reasoner.experiment.videophy2_dataflow_roles import VideoPhy2Processor

cs = ConfigStore.instance()


class _UnshardedLocalSFTDataset(LocalSFTDataset):
    """Yield the full shuffled manifest per iteration.

    Why: ``CosmosDataLoader``'s IterableDistributor already shards by
    ``dp_rank * num_workers + worker_id``; stock ``LocalSFTDataset`` shards
    again inside ``__iter__``, double-sharding to ``1 / (world*workers)^2``.
    """

    def _per_partition_indices(self, epoch: int) -> list[int]:
        import random

        manifest = self._load_manifest()
        indices = list(range(len(manifest)))
        if self.shuffle:
            rng = random.Random(self.distributor_seed + epoch)
            rng.shuffle(indices)
        return indices


def build_videophy2_local_dataset(
    dataset_key: str,
    split: str,
) -> _UnshardedLocalSFTDataset:
    # augmentor_config=None: the Processor decodes+tokenizes inline; the
    # BytesToMedia/TokenizeData augmentors aren't shipped in OSS.
    source = DATAINFO[dataset_key]
    if split not in source.manifest_path:
        raise KeyError(
            f"split={split!r} not present in DATAINFO[{dataset_key!r}].manifest_path "
            f"(have: {list(source.manifest_path)})"
        )
    return _UnshardedLocalSFTDataset(
        manifest_path=source.manifest_path[split],
        data_root=source.data_root,
        media_field_name=source.media_field_name,
        augmentor_config=None,
        text_only=source.text_only,
        shuffle=True,
        distributor_seed=1993,
        is_infinite_loader=True,
        split=split,
        dataset_name=dataset_key,
    )


def _dl(dataset_key, split, num_workers, persistent_workers=False, pin_memory=False, prefetch_factor=None):
    return L(CosmosDataLoader)(
        distributor=L(IterableDistributor)(
            iterable=L(build_videophy2_local_dataset)(dataset_key=dataset_key, split=split),
        ),
        processor=L(VideoPhy2Processor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            ignore_index=IGNORE_INDEX,
        ),
        batcher=L(PoolPackingBatcher)(
            max_tokens="${data_setting.max_tokens}",
            pool_size=16,
            max_batch_size=1,
            long_threshold=6400,
        ),
        collator=L(VLMCollator)(),
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


videophy2_sft_edge = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "local"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "cosmos3_edge_reasoner"},  # Edge delta (nano=qwen3_vl_8b_instruct)
            {"override /callbacks": ["basic_vlm", "basic_log", "hf_export"]},
            "_self_",
        ],
        job=dict(
            project="cosmos3_reasoner",
            group="sft",
            wandb_mode="disabled",
        ),
        trainer=dict(
            callbacks=dict(
                log_tensor_shape=dict(num_log=2),
            ),
            max_iter=50,
            logging_iter=1,
            run_validation=True,
            validation_iter=10,
            max_val_iter=50,
            grad_accum_iter=8,
        ),
        optimizer=dict(
            lr=1e-6,
            fused=True,
            weight_decay=0.05,
            betas=[0.9, 0.999],
            # Uniform 1.0x LR across projector + LM + lm_head, matching the fixed
            # videophy2_sft_nano. Nano must set {"model.visual": 1.0} because its
            # projector lives UNDER model.visual (Qwen model.visual.merger) and would
            # otherwise inherit the reasoner default {"model.visual": 0.1}. Edge needs
            # NO override: its projector is a top-level model.projector
            # (the Edge reasoner's PatchMerger), not under model.visual, so the inherited
            # default {"model.visual": 0.1} only matches the frozen (optimizer-
            # excluded) SigLIP2 tower and never reaches the projector -- which thus
            # already trains at the default 1.0x = 1e-6. So lr_multipliers is omitted.
        ),
        scheduler=dict(
            warm_up_steps=[5],
            cycle_lengths=[50],
            f_min=[0.1],
        ),
        data_setting=dict(
            # The "qwen_" names are historical knobs, not a Qwen-backbone requirement;
            # kept for parity with nano.
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
            distributor_seed=1993,
        ),
        model=dict(
            config=dict(
                # Edge delta: freeze the SigLIP2 tower via regex —
                # freeze_vision_encoder=True only supports Qwen/Intern towers.
                # Projector + LM stay trainable, matching nano's intent.
                freeze=dict(
                    frozen_params=[r"model\.visual\."],
                ),
                # FSDP full shard, auto from WORLD_SIZE — supports the documented
                # 4-GPU (NPROC_PER_NODE=4) and 8-GPU launches without a clamp warning.
                parallelism=dict(
                    data_parallel_shard_degree=-1,
                    data_parallel_replicate_degree=1,
                ),
                policy=dict(
                    monkey_patch_for_text_only_data=True,
                ),
            ),
        ),
        # hf_export so eval_videophy2 can read each save as HF safetensors.
        checkpoint=dict(
            save_iter=100,
            hf_export=dict(enabled=True),
        ),
        upload_reproducible_setup=False,
        dataloader_train=_dl("videophy2_train", "train", 2, persistent_workers=True, pin_memory=True, prefetch_factor=2),
        dataloader_val=_dl("videophy2_val", "val", 0, persistent_workers=False, pin_memory=True, prefetch_factor=None),
    ),
    flags={"allow_objects": True},
)


for _item in [videophy2_sft_edge]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
