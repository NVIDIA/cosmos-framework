# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from cosmos_framework.configs.tokenizer.lidar_v1p2 import LidarV1P2Recipe
from cosmos_framework.data.tokenizer.lidar import LidarTarDataset, read_manifest
from cosmos_framework.model.generator.tokenizers.lidar.checkpoint import (
    load_model_checkpoint,
    parse_lidar_checkpoint_stats,
)
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import infer_lidar_compression
from cosmos_framework.model.tokenizer.lidar.loss import LidarVAELoss
from cosmos_framework.model.tokenizer.lidar.model import LidarTokenizerModel, LidarTokenizerModelConfig
from cosmos_framework.trainer.tokenizer.lidar import export_checkpoint, restore_training_state
from cosmos_framework.trainer.tokenizer.lidar_evaluation import evaluate
from cosmos_framework.utils.config import PowerEMAConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def tiny_model() -> LidarTokenizerModel:
    config = LidarTokenizerModelConfig(
        network={
            "resolution": [128, 1808],
            "in_channels": 3,
            "out_channels": 3,
            "z_dim": 128,
            "base_channels": 4,
            "patch_size": [16, 16],
            "window_size": [1, 1],
            "depths": [0],
            "num_heads": [1],
            "dilation": [1],
            "temporal_downsample": [],
            "temporal_upsample": [],
            "mapping_depth": 0,
            "bottleneck_3d": False,
            "formulation": "VAE",
        },
        loss=LidarVAELoss(
            recon_weight=2, point_weight=1, gradient_weight=1, validity_weight=1, recon_reduction="masked_mean_all_rays"
        ),
        precision="float32",
        ema=PowerEMAConfig(enabled=True, s=0.1),
    )
    config.range_projection = LidarV1P2Recipe().model_config().range_projection
    return LidarTokenizerModel(config)


def test_recipe_contract_and_linear_schedule() -> None:
    recipe = LidarV1P2Recipe()
    config = recipe.model_config()
    assert infer_lidar_compression(config.network) == ((16, 16), 1)
    assert config.range_projection.semantic_width == 1800
    assert config.range_projection.model_width == 1808
    assert config.precision == "float32" and recipe.batch_size == 2
    model = tiny_model()
    optimizer, scheduler = model.init_optimizer_scheduler(recipe.optimizer_config(), recipe.scheduler_config())
    multiplier = scheduler.lr_lambdas[0]
    assert multiplier(0) == pytest.approx(0.001)
    assert multiplier(1000) == pytest.approx(1)
    assert multiplier(15500) == pytest.approx(0.55)
    assert multiplier(30000) == pytest.approx(0.1)


@pytest.fixture
def cpu_process_group(tmp_path: Path) -> Iterator[None]:
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_backward_resume_and_serving_export(tmp_path: Path, cpu_process_group: None) -> None:
    torch.manual_seed(7)
    model = tiny_model()
    recipe = LidarV1P2Recipe()
    optimizer, scheduler = model.init_optimizer_scheduler(recipe.optimizer_config(), recipe.scheduler_config())
    # Exercise normalization, geometric loss and the actual VAE through forward.
    video = torch.ones(1, 3, 2, 128, 1808)  # [B,C,T,H,W]
    video[:, 0] = 25.0  # [B,T,H,W]
    video[:, 1] = 0.25  # [B,T,H,W]
    video[:, 0, :, :, 4:904] = 0.0  # [B,T,H,900]
    video[:, 2, :, :, 4:904] = 0.0  # [B,T,H,900]
    target = video[:, :2, :, :, 4:-4]  # [B,2,T,H,1800]
    valid_mask = video[:, 2:3, :, :, 4:-4]  # [B,1,T,H,1800]
    ddp = DistributedDataParallel(model, broadcast_buffers=True, static_graph=True)
    output, loss = ddp({"video": video, "target": target, "valid_mask": valid_mask})  # loss: []
    assert torch.isfinite(loss)
    # Compare metric units directly on returned rays, not the all-ray loss denominator.
    prediction_m = (output["reconstructions"][:, 0] + 1.0) * 47.5 + 5.0  # [B,T,H,1800]
    returned = valid_mask[:, 0].bool()  # [B,T,H,1800]
    expected_mae = (prediction_m - target[:, 0]).abs()[returned].mean()  # []
    torch.testing.assert_close(output["loss_dict"]["train-video_rec/range_mae_meters"], expected_mae)
    assert returned.float().mean() == 0.5
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    scheduler.step()
    model.ema.update_average(model, iteration=0)
    manifest = recipe.manifest()
    manifest["network"] = model.config.network
    identity = {"recipe": manifest, "samples": [{"path": "/private-training-source/sentinel.tar"}]}
    payload = {
        "model": model.state_dict(),
        "parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": 1,
        "identity": identity,
    }
    checkpoint = tmp_path / "resume.pt"
    torch.save(payload, checkpoint)
    restored = tiny_model()
    opt2, sched2 = restored.init_optimizer_scheduler(recipe.optimizer_config(), recipe.scheduler_config())
    assert restore_training_state(torch.load(checkpoint, weights_only=False), restored, opt2, sched2, identity) == 1
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor)
    assert opt2.state_dict()["state"] and sched2.last_epoch == scheduler.last_epoch
    # Make EMA observably different and verify the requested variant actually lands.
    with torch.no_grad():
        next(model.network.parameters()).add_(0.5)  # parameter shape unchanged
    payload["model"] = model.state_dict()
    torch.save(payload, checkpoint)
    for weights in ("live", "ema"):
        destination = tmp_path / weights
        export_checkpoint(checkpoint, destination, weights)
        loaded = load_model_checkpoint(restored.network, str(destination / "tokenizer.pt"))
        mean, std, minimum, maximum = parse_lidar_checkpoint_stats(loaded)  # mean/std: [1,C,1,1,1]
        assert minimum == 5 and maximum == 100
        torch.testing.assert_close(mean, model.latent_mean)
        assert torch.isfinite(std).all()
        export_config = (destination / "config.json").read_text()
        assert json.loads(export_config)["weights"] == weights
        assert "private-training-source" not in export_config
        assert set(torch.load(destination / "tokenizer.pt", weights_only=True)) == {"model"}
        assert set(torch.load(destination / "latent_stats.pt", weights_only=True)) == {
            "mean",
            "std",
            "min_range",
            "max_range",
        }
        expected = (
            next(model.network.parameters()) if weights == "live" else next(model.ema.buffers())
        )  # parameter shape
        torch.testing.assert_close(next(restored.network.parameters()), expected)
    manifest_path = tmp_path / "eval.txt"
    archive = tmp_path / "measured.tar"
    write_sample_tar(archive)
    manifest_path.write_text(str(archive) + "\n")
    evaluate(tmp_path / "live", manifest_path, tmp_path / "eval.json", device="cpu")
    report = json.loads((tmp_path / "eval.json").read_text())
    assert report["range_mae_m"] >= 0 and 0 <= report["validity_iou"] <= 1
    with pytest.raises(ValueError, match="identical"):
        restore_training_state(payload, restored, opt2, sched2, {"different": True})


def test_manifest_rejects_alias_duplicates(tmp_path: Path) -> None:
    (tmp_path / "clip.tar").touch()
    manifest = tmp_path / "train.txt"
    manifest.write_text("clip.tar\n./clip.tar\n")
    with pytest.raises(ValueError, match="unique"):
        read_manifest(manifest)


def write_sample_tar(path: Path) -> None:
    with tarfile.open(path, "w") as archive:
        for frame in (0, 3):
            for kind, value in (("row", 64), ("col", 900), ("range", 25.0), ("intensity", 0.25)):
                stream = io.BytesIO()
                np.savez(stream, np.array([value]))
                contents = stream.getvalue()
                member = tarfile.TarInfo(f"scene.{frame:06d}.lidar_{kind}.npz")
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))


def test_local_tar_matches_projection_and_padding(tmp_path: Path) -> None:
    archive = tmp_path / "measured.tar"
    write_sample_tar(archive)
    sample = LidarTarDataset([archive])[0]
    assert sample["video"].shape == (3, 9, 128, 1808)
    assert sample["target"].shape == (2, 9, 128, 1800)
    assert sample["valid_mask"].shape == (1, 9, 128, 1800)
    torch.testing.assert_close(sample["video"][:, 0], sample["video"][:, 1])
    torch.testing.assert_close(sample["video"][:, -1], sample["video"][:, -2])
    measured_return = sample["target"][:, 0, 64, 450]  # [2]
    expected_return = torch.tensor([25.0, 0.25])  # [2]
    torch.testing.assert_close(measured_return, expected_return)


def test_initializer_requires_exact_network_and_seeds_ema(tmp_path: Path) -> None:
    model = tiny_model()
    checkpoint = tmp_path / "initializer.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    config = model.config
    config.init_from_checkpoint = str(checkpoint)
    initialized = LidarTokenizerModel(config)
    for name, parameter in model.network.named_parameters():
        torch.testing.assert_close(dict(initialized.network.named_parameters())[name], parameter)
    assert initialized.ema is not None
    for name, parameter in initialized.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(dict(initialized.ema.named_buffers())[name.replace(".", "-")], parameter)
    broken = {"model": dict(model.state_dict())}
    # Remove a network tensor so a permissive warm start would succeed incorrectly.
    del broken["model"]["network.quant_conv.weight"]
    torch.save(broken, checkpoint)
    with pytest.raises(RuntimeError, match="missing="):
        LidarTokenizerModel(config)
