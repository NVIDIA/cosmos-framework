# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Dummy dataset experiment — Cosmos3 2B pretrained base (for debugging)
#
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \
#     --master_port=12341 cosmos_framework/scripts/train.py \
#     --config=cosmos_framework/configs/base/config.py \
#     -- experiment=action_dummy_dataset_exp

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from cosmos_framework.data.vfm.action.dummy_dataset import DummyDataset
from cosmos_framework.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
DUMMY_TRAIN_DATASET = [
    L(dataset_entry)(
        name="default",
        dataset=L(DummyDataset)(length=1e6),
        ratio=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — 2B, long run for debugging
# ---------------------------------------------------------------------------
action_dummy_dataset_exp = make_2b_experiment(
    exp_name="dummy_dataset_exp",
    datasets=DUMMY_TRAIN_DATASET,
    batch_size=1,
    num_workers=2,
    training_iterations=1_000_000,
)

# --- Experiment-specific overrides ---
action_dummy_dataset_exp["job"]["group"] = "debugging"
action_dummy_dataset_exp["checkpoint"]["save_iter"] = 100_000_000

cs.store(group="experiment", package="_global_", name="action_dummy_dataset_exp", node=action_dummy_dataset_exp)
