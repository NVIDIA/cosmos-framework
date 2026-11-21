# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``transfer_multicontrol_lora_nano``: one Cosmos3-Nano LoRA for image-to-video, seg->video and depth->video.

Recipe: ``docs/transfer_multicontrol_lora_nano.md``. Derived from ``vision_sft_nano``:

* Three datasets in one run, one per GPU group (ratios 3:3:2 on 8 GPUs), so every optimizer step sees
  all three tasks:

  ======  ===========================  ==========================  ===================================
  stream  dataset                      frames                      conditioning
  ======  ===========================  ==========================  ===================================
  video   ``get_sft_dataset``          150 (the whole clip)        70% image->video, 20% text->video,
                                                                   10% video->video
  seg     ``get_transfer_sft_dataset`` 61, random window           50% with the first frame given
  depth   ``get_transfer_sft_dataset`` 61, random window           50% with the first frame given
  ======  ===========================  ==========================  ===================================

  First-frame conditioning on the transfer streams is what makes "seed image + control" work at inference.
* Resolution 720 set explicitly: the ``vision_sft_nano`` dataset defaults to 256.
* ``action_gen`` kept True, so the base checkpoint's action tower is carried through unchanged.
  Vision/transfer batches carry no actions, so the action loss path is skipped.
* Validation on ``val_*.jsonl`` with the same three streams, no caption dropout, fresh loader workers each
  time, and a fixed ``trainer.validation_seed``, so every validation scores the same 12 batches (samples, frame
  windows, conditioning modes, noise and timesteps) and the curve is comparable across checkpoints.

LoRA rank/targets, learning rate, iterations and checkpoint cadence come from the TOML.

``DATASET_PATH`` must contain ``vision.jsonl``, ``seg.jsonl``, ``depth.jsonl`` and
``val_vision.jsonl``, ``val_seg.jsonl``, ``val_depth.jsonl``; relative clip paths resolve against that folder.
Transfer samples are large: launch with ``COSMOS_TORCH_SHARING_STRATEGY=file_system`` (the launcher sets it).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.generator.local_datasets.transfer_sft_dataset import get_transfer_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()

RES = "720"
TOKENIZER = "${model.config.vlm_config.tokenizer}"


def _transfer(control_type: str):
    return L(get_transfer_sft_dataset)(
        jsonl_paths=[f"${{oc.env:DATASET_PATH}}/{control_type}.jsonl"],
        resolution=RES,
        num_video_frames=61,
        control_type=control_type,
        temporal_interval_mode="force_one",
        frame_selection_mode="random",
        tokenizer_config=TOKENIZER,
        cfg_dropout_rate=0.1,
        conditioning_config={0: 0.5, 1: 0.5},
    )


transfer_multicontrol_lora_nano = copy.deepcopy(vision_sft_nano)
transfer_multicontrol_lora_nano.job.group = "transfer_multicontrol_lora_nano"
transfer_multicontrol_lora_nano.job.name = "transfer_multicontrol_lora_nano"
transfer_multicontrol_lora_nano.model.config.action_gen = True
transfer_multicontrol_lora_nano.model.config.resolution = RES

_dl = transfer_multicontrol_lora_nano.dataloader_train
_dl.max_sequence_length = 45056
_video = copy.deepcopy(_dl.dataloader.datasets["video"]["dataset"])
_video.resolution = RES
_video.jsonl_paths = ["${oc.env:DATASET_PATH}/vision.jsonl"]
_video.conditioning_config = {1: 0.7, 0: 0.2, 2: 0.1}
# Keep the vision stream's key "video": the TOML maps [dataloader_train].max_caption_tokens onto
# datasets.video.dataset.max_caption_tokens.
# A 61-frame 720p transfer sample is two float32 videos (~1.35 GB) passed from loader workers through
# /dev/shm; the default 4 workers x prefetch 4 per rank exhausts it. Keep transfer loaders at 2 x 2.
_TRANSFER_LOADER = dict(num_workers=2, prefetch_factor=2)
_dl.dataloader.prefetch_factor = 2
_dl.dataloader.datasets = dict(
    video=dict(ratio=3, dataset=_video),
    seg=dict(ratio=3, dataset=_transfer("seg"), dataloader_kwargs=dict(_TRANSFER_LOADER)),
    depth=dict(ratio=2, dataset=_transfer("depth"), dataloader_kwargs=dict(_TRANSFER_LOADER)),
)

_val = copy.deepcopy(_dl)
for _name, _d in _val.dataloader.datasets.items():
    _jsonl = {"video": "vision"}.get(_name, _name)
    _d["dataset"].jsonl_paths = [f"${{oc.env:DATASET_PATH}}/val_{_jsonl}.jsonl"]
    _d["dataset"].cfg_dropout_rate = 0.0
# Fresh, seeded workers each validation -> identical samples every time.
_val.dataloader.persistent_workers = False
for _d in _val.dataloader.datasets.values():
    _d.setdefault("dataloader_kwargs", {})
    _d["dataloader_kwargs"]["persistent_workers"] = False
transfer_multicontrol_lora_nano.dataloader_val = _val
transfer_multicontrol_lora_nano.trainer.run_validation = True
transfer_multicontrol_lora_nano.trainer.run_validation_on_start = True
transfer_multicontrol_lora_nano.trainer.validation_iter = 50
transfer_multicontrol_lora_nano.trainer.max_val_iter = 12
transfer_multicontrol_lora_nano.trainer.validation_seed = 2026

cs.store(
    group="experiment",
    package="_global_",
    name="transfer_multicontrol_lora_nano",
    node=transfer_multicontrol_lora_nano,
)
