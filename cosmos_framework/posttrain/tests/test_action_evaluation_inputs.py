# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Row-local preparation ownership and native evaluation lifecycle, without GPU claims."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from cosmos_framework.posttrain.entrypoints.action_sft import _Source, evaluate
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.test_action_sft import assert_state_equal
from cosmos_framework.posttrain.utils import action_evaluation_inputs as reuse

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@dataclass
class Settings:
    observation_profile: str = "physicalai_four_camera_v1"
    max_pixels: int = 64


class Tokenizer:
    padding_side = "left"
    chat_template = "unchanged"
    special_tokens_map = {"eos_token": "eos"}

    def __init__(self):
        self.init_kwargs = {}

    def __len__(self):
        return 32

    def get_added_vocab(self):
        return {"eos": 31}


class MediaProcessor:
    def __init__(self):
        self.size = 64

    def to_dict(self):
        return {"size": self.size}


class Processor(SimpleNamespace):
    def _merge_kwargs(self, *args, **kwargs):
        from transformers.processing_utils import ProcessorMixin

        return ProcessorMixin._merge_kwargs(self, *args, **kwargs)


class Packer:
    def __init__(self):
        self.config = Settings()
        self.processor = Processor(chat_template="template", tokenizer=Tokenizer(), image_processor=MediaProcessor())
        self.model_config = SimpleNamespace(action_space_cfg={"n_waypoints": 64})
        self.action_space = nn.Module()
        self.action_space.register_buffer("scale", torch.tensor(1.0))
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        value = batch[0].transitions[0].obs.representations["flow_sample"]["actions"][0]
        return {"inputs": torch.tensor([[value, -value]]), "mask": torch.ones(1), "metadata": [1, "fixed", None]}

    def validation_semantics_identity(self):
        return {"original": "unchanged"}


class Data:
    identity = {"format": "content-bound-fixture", "sha256": "a" * 64}

    def __init__(self, tmp_path, count=3):
        self.items = []
        path = tmp_path / "image.png"
        path.write_bytes(b"immutable-media-fixture")
        self.path = path
        self.media_hashes = {str(path): "a" * 64}
        for index in range(count):
            sample = {
                "media": {"front": [str(path)]},
                "actions": [float(index + 1)],
                "observation": {"history": [0, 1]},
            }
            step = SimpleNamespace(obs=SimpleNamespace(representations={"flow_sample": sample}), algo_extra={})
            self.items.append(SimpleNamespace(id=f"row-{index}", transitions=[step]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def cfg(enabled=True):
    return SimpleNamespace(
        policy_factory=reuse._FACTORY + "policy.build_policy",
        data_factory=reuse._FACTORY + "training.build_data",
        objective_factory=reuse._FACTORY + "training.build_objective",
        evaluation_callback=reuse._FACTORY + "training.evaluate_trajectories",
        rl=SimpleNamespace(
            reuse_evaluation_inputs=enabled, micro_batch_size=1, evaluation_seed=123, evaluation_max_samples=0
        ),
    )


def test_consume_once_clone_ownership_and_training_passthrough(tmp_path):
    original = Packer()
    wrapped = reuse._EvaluationPacker(original)
    batch = [Data(tmp_path)[0]]
    before = collect_rng_state()
    with wrapped.batch_scope(batch):
        first = wrapped(batch)
        first["inputs"].fill_(99)
        first["metadata"].append("changed")
        second = wrapped(batch)
        assert second["inputs"].tolist() == [[1, -1]]
        assert second["metadata"] == [1, "fixed", None]
        assert wrapped._prepared is None
        with pytest.raises(RuntimeError, match="exactly two"):
            wrapped(batch)
    assert not wrapped.pending
    assert original.calls == 1
    assert_state_equal(collect_rng_state(), before)
    assert wrapped.validation_semantics_identity() == original.validation_semantics_identity()
    wrapped(batch)
    wrapped(batch)
    assert original.calls == 3
    assert wrapped.stats["hits"] == wrapped.stats["misses"] == 1
    assert wrapped.stats["retained_tensor_bytes_peak"] == 12


@pytest.mark.parametrize(
    "change", ["row", "mask", "processor", "template", "tokenizer", "config", "action", "media", "replace_media"]
)
def test_mutation_is_rejected_and_state_cleared(tmp_path, change):
    data = Data(tmp_path)
    batch = [data[0]]
    packer = Packer()
    wrapped = reuse._EvaluationPacker(packer)
    with pytest.raises(RuntimeError, match="changed before reuse"):
        with wrapped.batch_scope(batch):
            wrapped(batch)
            if change == "row":
                batch[0].transitions[0].obs.representations["flow_sample"]["actions"][0] = 5
            elif change == "mask":
                batch[0].transitions[0].algo_extra["validation_weight"] = 0
            elif change == "processor":
                packer.processor.image_processor.size = 32
            elif change == "template":
                packer.processor.chat_template = "different"
            elif change == "tokenizer":
                packer.processor.tokenizer.padding_side = "right"
            elif change == "config":
                packer.config.max_pixels = 32
            elif change == "action":
                packer.action_space.scale.fill_(2)
            elif change == "media":
                data.path.write_bytes(b"modified")
            else:
                alternate = tmp_path / "alternate"
                alternate.write_bytes(data.path.read_bytes())
                alternate.replace(data.path)
            wrapped(batch)
    assert not wrapped.pending and wrapped._identity is None
    with wrapped.batch_scope(batch):
        wrapped(batch)
        wrapped(batch)
    assert wrapped.stats["hits"] == 1


def test_mutation_during_preparation_and_missing_second_use_fail(tmp_path):
    data = Data(tmp_path)
    batch = [data[0]]

    class Changing(Packer):
        def __call__(self, batch):
            result = super().__call__(batch)
            data.path.write_bytes(b"changed while decoding")
            return result

    wrapped = reuse._EvaluationPacker(Changing())
    with pytest.raises(RuntimeError, match="during preparation"), wrapped.batch_scope(batch):
        wrapped(batch)
    assert not wrapped.pending
    wrapped = reuse._EvaluationPacker(Packer())
    with pytest.raises(RuntimeError, match="exactly two preparations"), wrapped.batch_scope(batch):
        wrapped(batch)
    assert not wrapped.pending


@pytest.mark.parametrize("bad", ["grad", "view", "stride", "object", "sparse", "cap"])
def test_unsupported_or_unbounded_preparation_fails_before_cache(tmp_path, monkeypatch, bad):
    batch = [Data(tmp_path)[0]]
    tensors = {
        "grad": torch.ones(1, requires_grad=True),
        "view": torch.ones(8)[:1],
        "stride": torch.ones(2, 3).t(),
        "object": object(),
        "sparse": torch.eye(2).to_sparse(),
        "cap": torch.ones(8),
    }

    class Bad(Packer):
        def __call__(self, batch):
            return {"bad": tensors[bad]}

    if bad == "cap":
        monkeypatch.setattr(reuse, "MAX_RETAINED_BYTES", 1)
    wrapped = reuse._EvaluationPacker(Bad())
    with pytest.raises(ValueError), wrapped.batch_scope(batch):
        wrapped(batch)
    assert not wrapped.pending
    assert wrapped.stats["misses"] == wrapped.stats["hits"] == 0


@pytest.mark.parametrize("path", ["relative.png", "flowpack:///archive#row", "https://host/image"])
def test_unsupported_sources_are_rejected_before_loading(path):
    config = cfg()
    with pytest.raises(ValueError, match="absolute local"):
        reuse.validate_evaluation_input_reuse(
            config, config.rl, SimpleNamespace(dataset=SimpleNamespace(media_hashes={path: "a" * 64}))
        )


def test_opt_in_family_guard_and_typed_identity():
    from cosmos_framework.posttrain.config.action_sft import ActionSFTConfig

    with pytest.raises(ValueError, match="boolean"):
        ActionSFTConfig(reuse_evaluation_inputs=1)
    config = cfg()
    config.policy_factory = "cosmos_framework.posttrain.model.gr00t.policy.build_policy"
    with pytest.raises(ValueError, match="native deterministic Alpamayo"):
        reuse.validate_evaluation_input_reuse(config, config.rl)
    config.rl.reuse_evaluation_inputs = False
    reuse.validate_evaluation_input_reuse(config, config.rl)
    assert reuse.reuse_identity(config.rl)["enabled"] is False
    config.rl.reuse_evaluation_inputs = True
    assert reuse.reuse_identity(config.rl)["enabled"] is True


@pytest.mark.parametrize("failure", [None, "objective", "callback", "metric"])
def test_native_evaluation_parity_rng_modes_and_exception_cleanup(tmp_path, failure):
    data = _Source(Data(tmp_path))
    initial_rng = collect_rng_state()
    results = []
    try:
        for enabled in (False, True):
            torch.manual_seed(77)
            net = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.5), nn.Linear(3, 1))
            net[0].eval()
            packer = Packer()
            model = SimpleNamespace(net=net, packer=reuse._EvaluationPacker(packer) if enabled else packer)
            captured = model.packer  # Same closure capture as native flow_bc.make_objective.

            def objective(model, batch):
                prepared = captured(batch)
                prediction = model.net(prepared["inputs"] + torch.rand(1, 2))
                loss = prediction.square().mean() + random.random() + float(np.random.random())
                if failure == "objective":
                    raise RuntimeError("injected objective")
                return loss, {}

            def callback(model, batch, config):
                prepared = model.packer(batch)
                value = float(model.net(prepared["inputs"]).sum())
                if failure == "callback":
                    raise RuntimeError("injected callback")
                return {"action": {"sum": value, "count": 0 if failure == "metric" else 1}}

            state = copy.deepcopy(net.state_dict())
            modes = [module.training for module in net.modules()]
            rng = collect_rng_state()
            if failure:
                with pytest.raises((ValueError, RuntimeError)):
                    evaluate(model, objective, data, cfg(), callback)
            else:
                results.append(evaluate(model, objective, data, cfg(), callback))
            assert_state_equal(collect_rng_state(), rng)
            assert_state_equal(net.state_dict(), state)
            assert [module.training for module in net.modules()] == modes
            if enabled:
                assert not model.packer.pending
                assert reuse.preparation_observations(model)["hits"] == (
                    3 if not failure else int(failure != "objective")
                )
            if not failure:
                assert packer.calls == (3 if enabled else 6)
        if not failure:
            assert results[0] == results[1]
    finally:
        restore_rng_state(initial_rng)


def test_native_trainer_updates_and_resume_state_equal_with_cached_evaluation(tmp_path):
    data = _Source(Data(tmp_path))
    initial_rng = collect_rng_state()

    def run(enabled):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        net = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.2), nn.Linear(3, 1))
        model = SimpleNamespace(net=net, packer=Packer())
        if enabled:
            model.packer = reuse._EvaluationPacker(model.packer)
        captured = model.packer

        def loss(model, batch):
            prepared = captured(batch)
            return (model.net(prepared["inputs"]) - torch.rand(1, 1)).square().mean(), {}

        objective = Objective(loss, reduction=Reduction.MEAN, identity={"fixture": "native-reuse"})
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        trainer = LLMTrainer(
            model, objective, optimizer, lr_scheduler=scheduler, require_gradients=True, nonfinite_update="fail_stop"
        )
        cursor = IndexedCursor(data, shuffle=True, seed=19, cursor_resume="strict")

        def callback(model, batch, config):
            return {"action": {"sum": float(model.net(model.packer(batch)["inputs"]).sum()), "count": 1}}

        scores = []
        for step in range(4):
            trainer.train([data[cursor.next_index()]])
            if step in (1, 3):
                scores.append(evaluate(model, objective, data, cfg(), callback))
        return copy.deepcopy(
            {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trainer": trainer._trainer_state(),
                "cursor": cursor.state_dict(),
                "rng": collect_rng_state(),
                "scores": scores,
            }
        )

    try:
        assert_state_equal(run(False), run(True))
    finally:
        restore_rng_state(initial_rng)


@pytest.mark.parametrize("entrypoint", ["train", "eval"])
def test_entrypoints_install_wrapper_before_objective_captures_packer(tmp_path, monkeypatch, entrypoint):
    """Execute each entrypoint through model/data admission, stopping at objective construction."""
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.entrypoints import action_eval, action_sft
    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTPacker
    from cosmos_framework.posttrain.utils import determinism, entry
    from cosmos_framework.utils import misc

    config = cfg()
    config.rl.seed = 17
    config.model = SimpleNamespace(backbone_adaptation=SimpleNamespace(mode="bf16_frozen"))
    original = AlpamayoSFTPacker.__new__(AlpamayoSFTPacker)
    original.__dict__.update(vars(Packer()))
    model = SimpleNamespace(net=nn.Linear(2, 1), packer=original)
    model.net._cosmos_adapter_checkpoint_codec = SimpleNamespace(identity={"fixture": "unchanged"})
    source = _Source(Data(tmp_path))

    class ReachedObjective(Exception):
        pass

    def objective(policy, config):
        assert isinstance(policy.packer, reuse._EvaluationPacker)
        assert policy.packer.original is original
        raise ReachedObjective

    factories = {
        config.policy_factory: lambda *args, **kwargs: model,
        config.data_factory: lambda *args, **kwargs: source.dataset,
        config.objective_factory: objective,
    }
    monkeypatch.setattr(action_sft, "_factory", factories.__getitem__)
    monkeypatch.setattr(action_sft, "validate_config", lambda _: config.rl)
    monkeypatch.setattr(action_sft, "prepare_sources", lambda _: (source, source))
    monkeypatch.setattr(
        entry, "pin_run_attempt", lambda *_: SimpleNamespace(out_dir=str(tmp_path / "native"), resume_checkpoint=None)
    )
    monkeypatch.setattr(entry, "out_dir", lambda _: str(tmp_path / "native"))
    monkeypatch.setattr(entry, "dump_resolved", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(CheckpointLifecycle, "from_config", lambda *_args, **_kwargs: SimpleNamespace(manager=None))
    monkeypatch.setattr(determinism, "enable_determinism", lambda **_: None)
    monkeypatch.setattr(misc, "set_random_seed", lambda **_: None)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    with pytest.raises(ReachedObjective):
        if entrypoint == "train":
            action_sft.run(config, config_path="fixture", opts=[])
        else:
            action_eval.run(config, checkpoint=None, output=str(tmp_path / "evaluated"), base=True)
    assert model.net._cosmos_adapter_checkpoint_codec.identity == {"fixture": "unchanged"}


def _actual_native_packer(tmp_path):
    pytest.importorskip("alpamayo_r1")
    import transformers

    if transformers.__version__ != "4.57.6":
        pytest.skip("actual pinned family processor requires transformers4.57.6")
    from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace
    from PIL import Image

    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTModelConfig, AlpamayoSFTPacker
    from cosmos_framework.posttrain.tests.test_alpamayo_qlora_contract import sample
    from cosmos_framework.posttrain.tests.test_alpamayo_qlora_flow import source

    native = source(tmp_path)
    packer = AlpamayoSFTPacker.__new__(AlpamayoSFTPacker)
    packer.config = AlpamayoSFTModelConfig(
        str(tmp_path / "base"), str(tmp_path / "processor"), min_pixels=16, max_pixels=64
    )
    packer.model_config = native.config
    packer.processor = native.config._build_processor()
    packer.processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]}
    )
    packer.processor.chat_template = "{% for message in messages %}{% for content in message['content'] %}{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endfor %}"
    packer.action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=64)
    row = sample()
    for camera in row["media"]:
        frames = []
        for index in range(4):
            media = tmp_path / f"{camera}-{index}.png"
            Image.new("RGB", (4, 4), (10 + index, 20, 30)).save(media)
            frames.append(str(media))
        row["media"][camera] = frames
    row["observation"].update(
        ego_history_xyz=[[i * 0.1, 0, 0] for i in range(-15, 1)],
        ego_history_rot=[torch.eye(3).tolist()] * 16,
        ego_future_rot=[torch.eye(3).tolist()] * 64,
    )
    row["actions"] = [[i * 0.1, 0, 0] for i in range(1, 65)]
    step = SimpleNamespace(obs=SimpleNamespace(representations={"flow_sample": row}), algo_extra={})
    batch = [SimpleNamespace(id="physical-fixture", transitions=[step])]
    return packer, batch


def test_actual_native_packer_cpu_image_action_tree_and_processor_identity(tmp_path):
    """Actual pinned processor, image decoding and native trajectory normalization; no model-quality claim."""
    packer, batch = _actual_native_packer(tmp_path)
    before = collect_rng_state()
    reference = packer(batch)
    wrapped = reuse._EvaluationPacker(packer)
    with wrapped.batch_scope(batch):
        first, second = wrapped(batch), wrapped(batch)
    assert_state_equal(reference, first)
    assert_state_equal(reference, second)
    assert first["actions"].shape == (1, 64, 2)
    assert first["history"]["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert wrapped.stats["retained_tensor_bytes_peak"] > 0
    assert_state_equal(collect_rng_state(), before)


@pytest.mark.parametrize("change", ["image_token", "image_token_id", "init_padding", "init_rescale"])
def test_actual_processor_token_and_resolved_defaults_cannot_reuse_stale_inputs(tmp_path, change):
    packer, batch = _actual_native_packer(tmp_path)
    wrapped = reuse._EvaluationPacker(packer)
    rng = collect_rng_state()
    initial = reuse._processor_identity(packer)
    with pytest.raises(RuntimeError, match="changed before reuse"), wrapped.batch_scope(batch):
        first = wrapped(batch)
        if change == "image_token":
            packer.processor.image_token = "<|changed_image|>"
        elif change == "image_token_id":
            packer.processor.image_token_id = -1
        elif change == "init_padding":
            packer.processor.tokenizer.init_kwargs.update(padding="max_length", max_length=512)
        else:
            packer.processor.tokenizer.init_kwargs["do_rescale"] = False
        assert reuse._processor_identity(packer) != initial
        # Reproduce the actual uncached consequence. The cache must refuse before
        # returning previously prepared tensors even if the new processor fails.
        if change.startswith("image_token"):
            with pytest.raises(ValueError, match="token count"):
                packer(batch)
        else:
            changed = packer(batch)
            key = "input_ids" if change == "init_padding" else "pixel_values"
            assert not torch.equal(first["tokenized"][key], changed["tokenized"][key])
        wrapped(batch)
    assert wrapped._prepared is None and wrapped._identity is None and not wrapped.pending
    assert wrapped.stats["hits"] == 0
    assert_state_equal(collect_rng_state(), rng)


@pytest.mark.parametrize(
    "name",
    [
        name
        for token in ("image_token", "video_token", "vision_start_token", "vision_end_token")
        for name in (token, token + "_id")
    ],
)
def test_every_pinned_multimodal_token_field_is_identity_bound(name):
    packer = Packer()
    initial = reuse._processor_identity(packer)
    setattr(packer.processor, name, -1 if name.endswith("_id") else "changed")
    assert reuse._processor_identity(packer) != initial


def test_resolved_processor_kwargs_follow_live_precedence_without_mutating_defaults():
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs

    packer = Packer()
    tokenizer = packer.processor.tokenizer
    tokenizer.init_kwargs.update(padding="max_length", max_length=512, do_rescale=False)
    tokenizer.max_length = 1024  # The pinned merge uses a live attribute when present.
    original_defaults = copy.deepcopy(Qwen3VLProcessorKwargs._defaults)
    original_init = copy.deepcopy(tokenizer.init_kwargs)
    snapshot = reuse._processor_identity(packer)["resolved_kwargs"]
    assert snapshot["text_kwargs"]["padding"] == "max_length"
    assert snapshot["text_kwargs"]["max_length"] == 1024
    assert snapshot["images_kwargs"]["do_rescale"] is False
    assert snapshot["text_kwargs"]["return_tensors"] == "pt"
    assert Qwen3VLProcessorKwargs._defaults == original_defaults
    assert tokenizer.init_kwargs == original_init
    before = reuse._processor_identity(packer)
    tokenizer.init_kwargs["name_or_path"] = "unused-local-path"
    assert reuse._processor_identity(packer) == before  # No blanket init dictionary/vocabulary serialization.


def test_cached_evaluation_checkpoint_recovery_and_portable_identity(tmp_path):
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
    from cosmos_framework.posttrain.utils.action_training_identity import action_training_source_identity
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state, state_for_current_rank

    source = _Source(Data(tmp_path))
    contract = {
        "implementation": action_training_source_identity(),
        "evaluation_input_reuse": reuse.reuse_identity(cfg().rl),
    }
    original_rng = collect_rng_state()

    def build(enabled=True):
        torch.manual_seed(93)
        net = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.25), nn.Linear(4, 1))
        net._cosmos_adapter_parameters = dict(net.named_parameters())
        codec = AdapterCheckpointCodec(net, identity={"family": "portable-unchanged-fixture"})
        identity = copy.deepcopy(contract)
        identity["evaluation_input_reuse"]["enabled"] = enabled
        codec.bind_training_identity(identity)
        net._cosmos_adapter_checkpoint_codec = codec
        model = SimpleNamespace(net=net, packer=reuse._EvaluationPacker(Packer()))
        captured = model.packer

        def loss(model, batch):
            value = model.net(captured(batch)["inputs"])
            return (value - torch.rand_like(value)).square().mean(), {}

        objective = Objective(loss, reduction=Reduction.MEAN, identity={"fixture": "cached-recovery"})
        optimizer = torch.optim.AdamW(codec.parameters.values(), lr=0.01)
        trainer = LLMTrainer(
            model,
            objective,
            optimizer,
            lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9),
            checkpoint_manager=CheckpointManager(checkpoint_format="dcp", keep_optim_last=0),
            require_gradients=True,
            nonfinite_update="fail_stop",
        )
        cursor = IndexedCursor(source, shuffle=True, seed=31, cursor_resume="strict")
        return model, trainer, cursor, codec

    def callback(model, batch, config):
        return {"action": {"sum": float(model.net(model.packer(batch)["inputs"]).sum()), "count": 1}}

    def snapshot(model, trainer, cursor):
        return copy.deepcopy(
            {
                "model": model.net.state_dict(),
                "optimizer": trainer.opt.state_dict(),
                "scheduler": trainer.lr_scheduler.state_dict(),
                "trainer": trainer._trainer_state(),
                "cursor": cursor.state_dict(),
                "rng": collect_rng_state(),
            }
        )

    def update(trainer, cursor):
        assert trainer.train([source[cursor.next_index()]])["optim/update_outcome"] == "applied"

    try:
        model, trainer, cursor, codec = build()
        update(trainer, cursor)
        update(trainer, cursor)
        score = evaluate(model, trainer.objective, source, cfg(), callback)
        assert not model.packer.pending
        path = str(tmp_path / "iter_000002")
        trainer.save(path, data_state=collect_rank_state({"cursor": cursor.state_dict(), "evaluation": score}))
        trainer.finalize_checkpoints()
        metadata = read_adapter_metadata(path, require_resume=True)
        update(trainer, cursor)
        expected = snapshot(model, trainer, cursor)
        restored, resumed, restored_cursor, _ = build()
        assert resumed.load(path) == 2
        data = state_for_current_rank(resumed.checkpoint_manager.load_component(path, "data"))
        restored_cursor.load_state_dict(data["cursor"])
        assert data["evaluation"] == score
        evaluate(restored, resumed.objective, source, cfg(), callback)
        update(resumed, restored_cursor)
        assert_state_equal(snapshot(restored, resumed, restored_cursor), expected)
        assert not restored.packer.pending
        _, wrong, _, wrong_codec = build(False)
        with pytest.raises(ValueError, match="identity differs"):
            wrong.load(path)
        wrong_codec.validate_metadata(metadata, purpose="reload")
        assert codec.identity["family"] == "portable-unchanged-fixture"
    finally:
        restore_rng_state(original_rng)
