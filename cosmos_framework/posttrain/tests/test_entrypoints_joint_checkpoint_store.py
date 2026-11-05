# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Resume real Joint PPO windows through the entrypoint's selected checkpoint store."""

import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest
from cosmos_framework.posttrain.checkpoint import POSTTRAIN_MANIFEST, CheckpointManager
from cosmos_framework.posttrain.config.build import REPLICA_INDEX_ENV
from cosmos_framework.posttrain.config.config_joint_ppo import make_config
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle, save_policy_checkpoint
from cosmos_framework.posttrain.control.loop.decision_window import collect_decision_window
from cosmos_framework.posttrain.entrypoints.joint_ppo import CollectorCheckpointState, setup_experience
from cosmos_framework.posttrain.roles.reward.worker import RewardWorker
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.joint_ppo import (
    GaussianEnvironment,
    GaussianGateway,
    GaussianPolicy,
    reward_provider,
)
from cosmos_framework.posttrain.utils.entry import resolve_all
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class FileObjects:
    """Translate the network adapter's URI namespace to durable test-owned files."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, uri: str) -> Path:
        parsed = urlsplit(uri)
        assert parsed.scheme == "s3"
        return self.root / parsed.netloc / parsed.path.lstrip("/")

    def get(self, uri: str) -> bytes:
        return self.path(uri).read_bytes()

    def put(self, payload: bytes, uri: str) -> None:
        path = self.path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def exists(self, uri: str) -> bool:
        return self.path(uri).is_file()


@pytest.fixture
def file_objects(tmp_path, monkeypatch):
    from cosmos_framework.checkpoint import s3_filesystem
    from cosmos_framework.utils.easy_io import easy_io

    objects = FileObjects(tmp_path / "objects")

    def backend(*, backend_args):
        assert backend_args["backend"] == "s3"
        return objects

    def reader(*, path, credential_path, enable_gcs_patch_in_boto3):
        return FileSystemReader(objects.path(path))

    def writer(*, path, credential_path, enable_gcs_patch_in_boto3):
        return FileSystemWriter(objects.path(path))

    monkeypatch.setattr(easy_io, "get_file_backend", backend)
    monkeypatch.setattr(s3_filesystem, "S3StorageReader", reader)
    monkeypatch.setattr(s3_filesystem, "S3StorageWriter", writer)
    return objects


@pytest.mark.parametrize("backend", ["local", "object_store"])
def test_setup_experience_restores_selected_store_cursor_and_continues_training(
    tmp_path, monkeypatch, file_objects, backend
):
    config = make_config()
    config.job.name = "joint-checkpoint-test"
    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    recipe.validation.enabled = False
    recipe.checkpoint_store.backend = backend
    recipe.checkpoint_store.bucket = "test-checkpoints"
    recipe.checkpoint_store.prefix = "joint-ppo/checkpoints"
    recipe.checkpoint_store.credentials = "test-owned-offline-transport"
    config.rl = recipe
    config.environment = L(GaussianEnvironment)()
    monkeypatch.setenv(REPLICA_INDEX_ENV, "0")
    output = str(tmp_path / "run")
    lifecycle = CheckpointLifecycle.from_config(config, recipe, run_out_dir=output)
    model = GaussianPolicy()
    learner = JointPPOTrainer(model, recipe, checkpoint_manager=lifecycle.manager)
    collector = setup_experience(config, output, None, recipe, 1, members=("collector",))["collector"]
    source = WorkerGroup("collector", "env", body=collector)
    rollout = WorkerGroup("rollout", "rollout", body=model)
    gateway = GaussianGateway()
    simulator = WorkerGroup("sim", "env", body=gateway)
    resolve_all(source.bind_rollout(rollout))
    reward = RewardWorker([reward_provider()], training_group_size=1)

    def train_window(active_learner, active_collector, active_simulator, version):
        samples = collect_decision_window(
            active_collector, active_simulator, version=version, decisions=recipe.window_decisions
        ).result()
        receipt = active_learner.prepare_ppo_targets(reward.score(samples))
        active_learner.train_batch(receipt["batch_id"])
        active_learner.clear_batch(receipt["batch_id"])
        active_learner.set_publication_version(version + 1)

    resumed_collector = None
    resumed_gateway = None
    try:
        train_window(learner, source, simulator, 0)
        assert collector.state_dict()["next_window"] == 1
        hint = str(tmp_path / "run" / "checkpoints" / "iter_000002")
        save_policy_checkpoint(learner, hint, data=CollectorCheckpointState(source))
        learner.finalize_checkpoints()
        location = lifecycle.manager.store.iteration_location(hint)
        before = canonical_digest(learner.evaluation_state())
        if backend == "object_store":
            assert location.startswith("s3://") and not Path(hint).exists()
            manifest = json.loads(file_objects.get(f"{location}/{POSTTRAIN_MANIFEST}"))
            assert set(manifest["bulk_components"]) == {"model", "optim"}
            assert "data" in manifest["control_components"]
            assert CheckpointManager(checkpoint_format="dcp").load_component(location, "data") is None
        else:
            assert location == hint
            assert (Path(hint) / "trainer.pt").is_file()
            assert (Path(hint) / "data.pt").is_file()

        recovery = CheckpointLifecycle.from_config(config, recipe, run_out_dir=output, resume_checkpoint=location)
        restored = JointPPOTrainer(GaussianPolicy(), recipe, checkpoint_manager=recovery.manager)
        recovery.load_trainer(restored)
        resumed_collector = setup_experience(config, output, location, recipe, 1, members=("collector",))["collector"]
        assert canonical_digest(restored.evaluation_state()) == before
        assert resumed_collector.state_dict() == collector.state_dict()
        assert restored.optimizer_step == 2 and restored.publication_version() == 1

        resumed = WorkerGroup("collector", "env", body=resumed_collector)
        resolve_all(resumed.bind_rollout(WorkerGroup("rollout", "rollout", body=restored.model)))
        resumed_gateway = GaussianGateway()
        train_window(restored, resumed, WorkerGroup("sim", "env", body=resumed_gateway), 1)
        assert restored.optimizer_step == 4 and restored.publication_version() == 2
        assert resumed_collector.state_dict()["next_window"] == 2
    finally:
        collector.close()
        gateway.close()
        if resumed_collector is not None:
            resumed_collector.close()
        if resumed_gateway is not None:
            resumed_gateway.close()
