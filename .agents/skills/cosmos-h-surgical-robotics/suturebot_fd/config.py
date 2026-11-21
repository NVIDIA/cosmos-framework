# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Forward-dynamics post-training of Cosmos3-H-Embodied-Nano on SutureBot.

Config entry point for cosmos-framework (``--config-file suturebot_fd/config.py``, run from the
recipe directory with it on ``PYTHONPATH``). It follows the "fine-tune on your own data" path of
``cosmos3_h_embodied``: SutureBot is added as a dataset family (``FAMILIES`` in
``cosmos3_h_embodied/configs/openh_datasets.py``) and the experiments are built with
``register_experiment`` (``cosmos3_h_embodied/configs/experiment_h16.py``), i.e. like the released
recipe ``openh_h16_task_prompt`` except for the data, the mode and the run length:

* ``suturebot_fd``: forward dynamics only (first frame + 24 actions -> 24 frames) on the training
  tissues, continuing from the released checkpoint.
* ``suturebot_fd_smoke``: the same on one GPU with few windows; combine with ``COSMOS_SMOKE=1``
  (2-layer model) to check the data path and the training loop.
* ``suturebot_fd_test``: family ``suturebot_test``, the held-out test tissues, for evaluation only
  (``--dataset.dataset suturebot_test --dataset.dataset-experiment suturebot_fd_test
  --dataset.dataset-split full``).

Data: the converted roots are ``$OPENH_DATA_ROOT/suturebot/tissue_<N>`` (resolved when the data is
loaded). Read when this file is imported:

* ``SUTUREBOT_TRAIN_TISSUES``: comma-separated tissue IDs to train on (default: all but tissue 3).
* ``SUTUREBOT_TEST_TISSUES``: comma-separated held-out tissue IDs (default: 3).
* ``SUTUREBOT_NORMALIZER``: action statistics (default: the SutureBot statistics shipped with
  ``cosmos3_h_embodied``, which the checkpoint was trained with). An absolute path is used as is.
"""

from __future__ import annotations

import os

from hydra.core.config_store import ConfigStore

from cosmos3_h_embodied.configs import openh_datasets
from cosmos3_h_embodied.configs.config import make_config as _make_h_embodied_config
from cosmos3_h_embodied.configs.openh_datasets import Family, Root, _dvrk
from cosmos3_h_embodied.data.history_dataset import GeneralizedDvrkHistoryDataset

# Same robot as the SutureBot data in the checkpoint's training mixture: keep its embodiment
# (domain id), caption context and action statistics.
DEFAULT_NORMALIZER = "suturebot_backward_anchored_worldtrans_rot6d.json"


def suturebot_family(name: str, tissues: str) -> Family:
    """A ``GeneralizedDvrkHistoryDataset`` family over ``$OPENH_DATA_ROOT/suturebot/tissue_<N>``."""
    roots = tuple(Root(f"suturebot/tissue_{t.strip()}", "suturebot") for t in tissues.split(",") if t.strip())
    kwargs = _dvrk(
        "suturebot",
        os.environ.get("SUTUREBOT_NORMALIZER") or DEFAULT_NORMALIZER,
        "observation.images.endoscope.left",
        duplicate_frame_max_fraction=0.5,
        # The Open-H exclusion list is keyed by Open-H paths; it does not apply here.
        exclude_episodes_overlay=None,
    )
    return Family(name, 1.0, GeneralizedDvrkHistoryDataset, roots, kwargs)


FAMILIES = (
    suturebot_family("suturebot", os.environ.get("SUTUREBOT_TRAIN_TISSUES", "1,2,4,5,6,7,8,9,10")),
    suturebot_family("suturebot_test", os.environ.get("SUTUREBOT_TEST_TISSUES", "3")),
)
# Add the families to the package's FAMILIES on import, as appending them to FAMILIES in
# openh_datasets.py would; tools/compute_action_normalizer.py then finds them too.
_known = {f.name for f in openh_datasets.FAMILIES}
openh_datasets.FAMILIES = (*openh_datasets.FAMILIES, *(f for f in FAMILIES if f.name not in _known))


def _register(name: str, family: str, *, max_iter: int, save_iter: int, **kwargs) -> None:
    from cosmos3_h_embodied.configs.experiment_h16 import register_experiment

    exp = register_experiment(
        name,
        families=[family],
        # One family: every worker samples it (no rank-per-family allocation).
        shard_across_workers=False,
        # Forward dynamics only: first frame + action chunk (and the 16-step action history) -> video.
        mode="forward_dynamics",
        joint_mode_weights=None,
        **kwargs,
    )
    exp["trainer"]["max_iter"] = max_iter
    exp["trainer"]["grad_accum_iter"] = 1
    exp["checkpoint"]["save_iter"] = save_iter
    # World size is set on the command line (model.config.parallelism.data_parallel_shard_degree).
    exp["model"]["config"]["parallelism"]["data_parallel_shard_degree"] = 8
    ConfigStore.instance().store(group="experiment", package="_global_", name=name, node=exp)


def make_config():
    config = _make_h_embodied_config()
    _register("suturebot_fd", "suturebot", max_iter=3000, save_iter=500, num_workers=8, prefetch_factor=4)
    # With COSMOS_SMOKE=1 (the launcher's SMOKE=1), cosmos-framework forces trainer.max_iter=2, so the smoke
    # test runs 2 iterations and writes iter_000000002; max_iter/save_iter here apply only without it.
    _register(
        "suturebot_fd_smoke", "suturebot", max_iter=20, save_iter=10, num_workers=2, prefetch_factor=2, max_windows=64
    )
    _register("suturebot_fd_test", "suturebot_test", max_iter=0, save_iter=0, num_workers=2, prefetch_factor=2)
    return config
