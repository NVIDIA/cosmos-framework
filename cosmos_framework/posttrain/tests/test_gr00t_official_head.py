# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Official head control contracts; CPU tests are not LIBERO task qualification."""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.config.build import build_optimizer, build_sft_lr_scheduler
from cosmos_framework.posttrain.config.gr00t_official import OFFICIAL_HEAD_PROFILE, Gr00tOfficialHeadSchedule
from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_optimizer_config
from cosmos_framework.posttrain.model.gr00t.official_head import (
    attach_official_codec,
    materialize_official_parameters,
    processor_mode,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class HeadFixture(nn.Module):
    def __init__(self, *, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(404)
            self.backbone = nn.Linear(2, 4, bias=False, device=device)
            self.action_head = nn.Sequential(
                nn.Linear(4, 4, device=device), nn.LayerNorm(4, device=device), nn.Linear(4, 1, device=device)
            )
        self.backbone.requires_grad_(False)

    def forward(self, x):
        return self.action_head(self.backbone(x))


def build_fixture():
    net = HeadFixture()
    codec = attach_official_codec(net, {"base": "immutable-fixture", "task": "fixture-official-head"})
    optimizer = build_optimizer(net, make_optimizer_config())
    return net, codec, optimizer


def test_official_profile_matches_existing_native_optimizer_and_exact_head_membership():
    from transformers.trainer_pt_utils import get_parameter_names

    net, codec, optimizer = build_fixture()
    assert type(optimizer) is torch.optim.AdamW
    assert len(optimizer.param_groups) == 2
    assert optimizer.defaults["fused"] is None and optimizer.defaults["foreach"] is None
    patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay = set(get_parameter_names(net, [nn.LayerNorm], patterns))
    names = {id(p): name for name, p in net.named_parameters()}
    assert [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups] == [
        [name for name, p in net.named_parameters() if p.requires_grad and name in decay],
        [name for name, p in net.named_parameters() if p.requires_grad and name not in decay],
    ]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-5, 0.0]
    codec.assert_live(optimizer=optimizer)
    net.backbone.requires_grad_(True)
    with pytest.raises(ValueError, match="membership"):
        build_optimizer(net, make_optimizer_config())


@pytest.mark.parametrize("mutation", ["profile", "format", "auxiliary", "fused"])
def test_official_optimizer_refuses_incompatible_declarations(mutation):
    net, _, _ = build_fixture()
    config = make_optimizer_config()
    if mutation == "profile":
        net._cosmos_adapter_optimizer_profile = "unknown"
    if mutation == "format":
        net._cosmos_adapter_metadata["format"] = "cosmos.family-backbone.v1"
    if mutation == "auxiliary":
        net._cosmos_adapter_metadata["auxiliary_parameters"] = []
    if mutation == "fused":
        config.impl = "fused"
    with pytest.raises(ValueError, match="official"):
        build_optimizer(net, config)


def test_native_lora_optimizer_guard_is_unchanged():
    net, _, _ = build_fixture()
    del net._cosmos_adapter_optimizer_profile
    with pytest.raises(ValueError, match="native LoRA requires"):
        build_optimizer(net, make_optimizer_config())


def test_actual_native_recipe_keeps_official_profile_separate():
    from cosmos_framework.posttrain.entrypoints.action_sft import training_identity, validate_config
    from cosmos_framework.utils.config import load_config

    config = load_config("cosmos_framework/posttrain/configs/gr00t_n1d7_libero_official_head.py", ["--"])
    rl = validate_config(config)
    assert isinstance(rl.lr_schedule, Gr00tOfficialHeadSchedule)
    assert rl.optimizer_profile == OFFICIAL_HEAD_PROFILE
    assert config.model.backbone_adaptation is None
    assert (rl.micro_batch_size, rl.max_steps, rl.optimizer.warmup_steps) == (1, 300, 15)
    matched = load_config("cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py", ["--"])
    previous = validate_config(matched)
    assert previous.optimizer.impl == "fused" and previous.optimizer.decay_groups == ""
    identity = training_identity(
        matched,
        previous,
        SimpleNamespace(identity={"objective": "fixture"}),
        SimpleNamespace(identity={"source": "fixture"}),
    )
    assert "optimizer_profile" not in identity


def test_official_cosine_exact_hf_update_sequence_and_scheduler_resume():
    from transformers import get_cosine_schedule_with_warmup

    _, _, optimizer = build_fixture()
    reference_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=1e-4)
    config = make_optimizer_config()
    config.warmup_steps = 3
    schedule = build_sft_lr_scheduler(optimizer, config, Gr00tOfficialHeadSchedule(), 12)
    reference = get_cosine_schedule_with_warmup(reference_optimizer, num_warmup_steps=3, num_training_steps=12)
    rates = []
    for step in range(13):
        rates.append(schedule.get_last_lr()[0])
        assert schedule.get_last_lr()[0] == reference.get_last_lr()[0]
        if step == 7:
            saved_opt, saved_schedule = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(schedule.state_dict())
        optimizer.step()
        schedule.step()
        reference_optimizer.step()
        reference.step()
    assert rates[0] == rates[-1] == 0 and rates[3] == 1e-4
    _, _, restored_opt = build_fixture()
    restored = build_sft_lr_scheduler(restored_opt, config, Gr00tOfficialHeadSchedule(), 12)
    restored_opt.load_state_dict(saved_opt)
    restored.load_state_dict(saved_schedule)
    restored_opt.step()
    restored.step()
    assert restored.get_last_lr()[0] == rates[8]


def test_streamed_fp32_materialization_preserves_aliases_buffer_values_and_membership():
    with torch.device("meta"):
        net = HeadFixture(device="meta")
        net.backbone_alias = net.backbone
    net.register_buffer("rotary", torch.tensor([0.25, 0.5]))
    values = {
        name: torch.full(tuple(tensor.shape), 0.125, dtype=torch.bfloat16) for name, tensor in net.state_dict().items()
    }
    calls = []
    materialize_official_parameters(net, lambda name: calls.append(name) or values[name], torch.device("cpu"))
    assert net.backbone.weight is net.backbone_alias.weight
    assert all(p.dtype == torch.float32 and not p.is_meta for p in net.parameters())
    assert all(name.startswith("action_head.") for name, p in net.named_parameters() if p.requires_grad)
    assert len(calls) == len(list(net.parameters()))
    torch.testing.assert_close(net.rotary, torch.tensor([0.25, 0.5]), rtol=0, atol=0)


def test_processor_mode_restores_even_on_failure():
    class Processor:
        training = False

        def train(self):
            self.training = True

        def eval(self):
            self.training = False

    processor = Processor()
    with pytest.raises(RuntimeError):
        with processor_mode(processor, True):
            assert processor.training
            raise RuntimeError("fixture")
    assert not processor.training


def test_pinned_real_augmentations_have_restorable_global_rng_and_nonzero_effect():
    pytest.importorskip("gr00t")
    import albumentations
    from gr00t.model.gr00t_n1d7.image_augmentations import build_image_transformations_albumentations

    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state

    assert albumentations.__version__ == "1.4.18"
    train, evaluation = build_image_transformations_albumentations(
        [256, 256],
        [230, 230],
        0,
        {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08},
        256,
        0.95,
    )
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    rng = collect_rng_state()
    try:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        initial = collect_rng_state()
        a, b = train(image=image)["image"], train(image=image)["image"]
        assert not np.array_equal(a, b)
        restore_rng_state(initial)
        np.testing.assert_array_equal(train(image=image)["image"], a)
        between = collect_rng_state()
        evaluation(image=image)
        restore_rng_state(between)
        np.testing.assert_array_equal(train(image=image)["image"], b)
    finally:
        restore_rng_state(rng)


def test_official_trained_dcp_portable_and_next_update_restore_complete_state(tmp_path):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, publish_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests.test_action_sft import assert_state_equal
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state, state_for_current_rank

    def build():
        net, codec, optimizer = build_fixture()
        config = make_optimizer_config()
        config.warmup_steps = 2
        scheduler = build_sft_lr_scheduler(optimizer, config, Gr00tOfficialHeadSchedule(), 10)
        codec.bind_training_identity({"profile": OFFICIAL_HEAD_PROFILE, "source": "fixed-rows"})

        def objective(policy, batch):
            inputs = torch.tensor(batch, dtype=torch.float32)
            target = torch.rand(len(batch), 1)
            return (policy.net(inputs) - target).square().mean(), {}

        model = SimpleNamespace(net=net)
        trainer = LLMTrainer(
            model,
            Objective(objective, reduction=Reduction.MEAN, identity={"objective": "official-fixture"}),
            optimizer,
            lr_scheduler=scheduler,
            checkpoint_manager=CheckpointManager(checkpoint_format="dcp", keep_optim_last=0),
            require_gradients=True,
            nonfinite_update="fail_stop",
        )
        return net, codec, trainer

    saved_rng = collect_rng_state()
    batch = [[1.0, -1.0], [0.5, 1.0]]
    try:
        torch.manual_seed(17)
        net, codec, trainer = build()
        frozen = net.backbone.weight.detach().clone()
        for _ in range(3):
            assert trainer.train(batch)["optim/update_outcome"] == "applied"
        path = str(tmp_path / "checkpoints/iter_000003")
        trainer.save(path, data_state=collect_rank_state({"cursor": 6, "evaluation": [{"step": 3}]}))
        trainer.finalize_checkpoints()
        saved_head = copy.deepcopy(net.action_head.state_dict())
        portable = publish_adapter_artifact(
            path, str(tmp_path / "portable/checkpoints/iter_000003"), manager=trainer.checkpoint_manager
        )
        trainer.train(batch)
        expected = {
            "model": copy.deepcopy(net.state_dict()),
            "optimizer": copy.deepcopy(trainer.opt.state_dict()),
            "trainer": copy.deepcopy(trainer._trainer_state()),
            "scheduler": copy.deepcopy(trainer.lr_scheduler.state_dict()),
            "rng": collect_rng_state(),
        }
        restored, restored_codec, resumed = build()
        assert resumed.load(path) == 3
        assert state_for_current_rank(resumed.checkpoint_manager.load_component(path, "data")) == {
            "cursor": 6,
            "evaluation": [{"step": 3}],
        }
        resumed.train(batch)
        restored_codec.assert_live(optimizer=resumed.opt)
        restored_codec.validate_frozen_base()
        assert_state_equal(
            {
                "model": restored.state_dict(),
                "optimizer": resumed.opt.state_dict(),
                "trainer": resumed._trainer_state(),
                "scheduler": resumed.lr_scheduler.state_dict(),
                "rng": collect_rng_state(),
            },
            expected,
        )
        fresh, fresh_codec, _ = build()
        load_adapter_weights(portable, fresh_codec, manager=CheckpointManager(checkpoint_format="dcp"))
        assert_state_equal(fresh.action_head.state_dict(), saved_head)
        torch.testing.assert_close(fresh.backbone.weight, frozen, rtol=0, atol=0)
        assert all(name.startswith("action_head.") for name in fresh_codec.parameters)
    finally:
        restore_rng_state(saved_rng)
