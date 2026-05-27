# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# PushT experiment — Cosmos3 2B pretrained and 8B GA midtrain bases
#
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \
#     --master_port=12341 cosmos_framework/scripts/train.py \
#     --config=cosmos_framework/configs/base/config.py \
#     -- experiment=pusht_exp

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.configs.base.experiment.action._experiment_helpers import register_modes
from cosmos_framework.configs.base.experiment.action.midtrain_ga_config.cosmos3_8B import make_8b_experiment
from cosmos_framework.configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos_framework.data.vfm.action.pusht_dataset import PushTDataset
from cosmos_framework.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

TRAINING_ITERATIONS = 4_000
DATALOADER_SEED = 0
MAX_SAMPLES_PER_BATCH = 256

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
PUSHT_TRAIN_DATASET = [
    L(dataset_entry)(
        name="pusht",
        dataset=L(PushTDataset)(
            repo_id="lerobot/pusht_image",
            split="train",
            split_seed=DATALOADER_SEED,
            split_val_ratio=0.05,
            mode="forward_dynamics",
        ),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, 4k iters
# ---------------------------------------------------------------------------
pusht_exp = make_2b_experiment(
    exp_name="pusht_exp",
    datasets=PUSHT_TRAIN_DATASET,
    training_iterations=TRAINING_ITERATIONS,
)

# Checkpoint save interval
pusht_exp["checkpoint"]["save_iter"] = 1000
pusht_exp["dataloader_train"]["max_sequence_length"] = None
pusht_exp["dataloader_train"]["max_samples_per_batch"] = MAX_SAMPLES_PER_BATCH

cs.store(
    group="experiment",
    package="_global_",
    name="pusht_exp",
    node=pusht_exp,
)
register_modes(cs, "pusht_exp", pusht_exp, dataloader_key="action_data")

# ---------------------------------------------------------------------------
# Base experiment — 8B GA midtrain, 4k iters
# ---------------------------------------------------------------------------
pusht_8b_exp_fd = make_8b_experiment(
    exp_name="pusht_8b_exp",
    datasets=PUSHT_TRAIN_DATASET,
    training_iterations=TRAINING_ITERATIONS,
    batch_size=32,
    num_workers=16,
    max_samples_per_batch=MAX_SAMPLES_PER_BATCH,
)

# Checkpoint save interval
pusht_8b_exp_fd["checkpoint"]["save_iter"] = 1000

cs.store(
    group="experiment",
    package="_global_",
    name="pusht_8b_exp_fd",
    node=pusht_8b_exp_fd,
)
register_modes(cs, "pusht_8b_exp_fd", pusht_8b_exp_fd, dataloader_key="action_data")
