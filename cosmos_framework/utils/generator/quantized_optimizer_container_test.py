# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared eight-bit AdamW through the generator/VFM ``OptimizersContainer``.

CPU checks cover the legacy route and fail-before-construction rules; CUDA checks
exercise the real bitsandbytes backend, grouping identity and DCP exact resume.
The two-rank FSDP2 gate lives in ``quantized_optimizer_container_fsdp_test.py``.
"""

from copy import deepcopy

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.utils.generator.optimizer import build_lr_scheduler, build_optimizer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
KWARGS = dict(
    lr=5e-5,
    betas=[0.9, 0.99],
    eps=1e-8,
    weight_decay=0.05,
    keys_to_select=["gen", "llm2action"],
    lr_multipliers={"llm2action": 5.0},
)
# The legacy optimizers require fused kernels; the eight-bit backend refuses a fused request.
LEGACY = dict(KWARGS, fused=True)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = torch.nn.Linear(256, 512)  # 131k weight elements: quantized
        self.llm2action = torch.nn.Linear(512, 16)  # 8k: quantized, 5x LR
        self.frozen = torch.nn.Linear(16, 16)

    def forward(self, x):
        return self.frozen(self.llm2action(self.gen(x)))


class _Model(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        torch.manual_seed(0)
        self.net = _Net().to(device)


def _groups(container):
    return [
        (
            group["lr"],
            group["weight_decay"],
            group.get("betas"),
            group.get("eps"),
            [id(p) for p in group["params"]],
        )
        for optimizer in container
        for group in optimizer.param_groups
    ]


def test_legacy_route_and_defaults_are_unchanged():
    model = _Model()
    container = build_optimizer(model, "adamw", **deepcopy(LEGACY))
    assert not container.rank_local_checkpoint
    assert [type(o) for o in container] == [torch.optim.AdamW]
    assert not model.net.frozen.weight.requires_grad
    assert {g["lr"] for g in container.optimizers[0].param_groups} == {5e-5, 2.5e-4}


def test_legacy_state_report_tolerates_fused_adam_master_placeholders():
    """FusedAdam keeps ``param_groups_master`` None (pre-step) or None-filled (no master needed)."""
    from types import SimpleNamespace

    container = build_optimizer(_Model(), "adamw", **deepcopy(LEGACY))
    master = torch.zeros(3)
    for groups in (None, [{"params": [None, master]}]):
        fake = SimpleNamespace(state={}, param_groups=[], param_groups_master=groups)
        container.optimizers = [fake]
        expected = master.numel() * master.element_size() if groups else 0
        assert container.state_report() == {"moment_bytes": 0, "metadata_bytes": 0, "master_bytes": expected}


@pytest.mark.parametrize("option", [{"amsgrad": True}, {"master_weights": True}, {"capturable": True}])
def test_unsupported_options_fail_before_backend_construction(option, monkeypatch):
    import cosmos_framework.utils.quantized_optimizer as backend

    monkeypatch.setattr(backend, "require_backend", lambda **_: pytest.fail("backend constructed"))
    with pytest.raises(ValueError, match="does not support optimizer options"):
        build_optimizer(_Model(), "adamw_bnb_8bit", **deepcopy(KWARGS), **option)


def test_cpu_selection_fails_explicitly_without_fallback():
    with pytest.raises((ValueError, RuntimeError)):
        build_optimizer(_Model(), "adamw_bnb_8bit", **deepcopy(KWARGS))


def _step(model, container, scheduler, seed):
    torch.manual_seed(seed)
    x = torch.randn(4, 256, device="cuda")
    model.net(x).square().mean().backward()
    container.step()
    scheduler.step()
    container.zero_grad()


def _scheduler(container):
    return build_lr_scheduler(
        container,
        lr_scheduler_type="LambdaLinear",
        cycle_lengths=[100],
        f_max=[1.0],
        f_min=[0.0],
        f_start=[1e-6],
        warm_up_steps=[5],
        verbosity_interval=0,
    )


@CUDA
@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_eight_bit_reaches_backend_with_identical_groups(name):
    legacy_model, model = _Model("cuda"), _Model("cuda")
    legacy = build_optimizer(legacy_model, "adamw", **deepcopy(LEGACY))
    container = build_optimizer(model, name, **deepcopy(KWARGS))
    assert container.rank_local_checkpoint and len(container) == 1
    optimizer = container.optimizers[0]
    assert type(optimizer.backend).__name__ == "CosmosAdamW8bit"
    assert optimizer.backend.is_paged == (name == "paged_adamw_8bit")
    names = {id(p): n for n, p in model.net.named_parameters()}
    legacy_names = {id(p): n for n, p in legacy_model.net.named_parameters()}
    as_names = lambda groups, table: [(*g[:4], [table[i] for i in g[4]]) for g in groups]  # noqa: E731
    assert as_names(_groups(container), names) == as_names(_groups(legacy), legacy_names)
    # FP32 FSDP/unsharded storage is its own master: aliased, never copied.
    assert all(m.data_ptr() == p.data_ptr() for m, p in zip(optimizer.masters, optimizer.live))
    frozen = model.net.frozen.weight.detach().clone()
    before = model.net.gen.weight.detach().clone()
    scheduler = _scheduler(container)
    _step(model, container, scheduler, 1)
    assert not torch.equal(before, model.net.gen.weight)
    assert torch.equal(frozen, model.net.frozen.weight)
    report = container.state_report()
    assert report["quantized_elements"] == sum(p.numel() for p in optimizer.live if p.numel() >= 4096)
    assert report["master_bytes"] == 0 and report["moment_bytes"] > 0
    assert optimizer.backend.state[optimizer.masters[0]]["state1"].dtype == torch.uint8
    # The scheduler-owned LR in effect at the update reaches the backend on every step.
    used = [g["lr"] for g in optimizer.param_groups]
    assert used[0] != 5e-5 * 1e-6  # warmup has advanced
    _step(model, container, scheduler, 2)
    assert [g["lr"] for g in optimizer.backend.param_groups] == used


@CUDA
def test_dcp_exact_resume_and_transactional_rejection(tmp_path, monkeypatch):
    # Memory reporting runs on the first step and must not alter resume identity.
    monkeypatch.setenv("COSMOS_OPTIMIZER_STATE_REPORT", "1")
    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    scheduler = _scheduler(container)
    for seed in range(3):
        _step(model, container, scheduler, seed)
    dcp.save({"model": model.state_dict()}, checkpoint_id=tmp_path / "model")
    dcp.save(container.state_dict(), checkpoint_id=tmp_path / "optim")
    dcp.save(scheduler.state_dict(), checkpoint_id=tmp_path / "scheduler")
    for seed in range(3, 6):
        _step(model, container, scheduler, seed)
    expected = {n: p.detach().clone() for n, p in model.net.named_parameters()}

    resumed = _Model("cuda")
    fresh = build_optimizer(resumed, "adamw_bnb_8bit", **deepcopy(KWARGS))
    fresh_scheduler = _scheduler(fresh)
    model_state = {"model": resumed.state_dict()}
    dcp.load(model_state, checkpoint_id=tmp_path / "model")
    resumed.load_state_dict(model_state["model"])
    sched_state = fresh_scheduler.state_dict()
    dcp.load(sched_state, checkpoint_id=tmp_path / "scheduler")
    fresh_scheduler.load_state_dict(sched_state)

    # Rejected restores leave optimizer state untouched on every path.
    template = fresh.checkpoint_load_template(dcp.FileSystemReader(tmp_path / "optim").read_metadata())
    dcp.load(template, checkpoint_id=tmp_path / "optim", planner=dcp.DefaultLoadPlanner(allow_partial_load=False))
    (format_key,) = template
    payload = template[format_key]

    def pristine():
        state = fresh.optimizers[0].state_dict()
        return [m.clone() for m in state["masters"]], state["backend"]["state"]

    before = pristine()
    for broken in (
        {format_key: dict(payload, world_size=torch.tensor([2]))},
        {format_key: {"world_size": payload["world_size"]}},
        {"cosmos_rank_master_v0": payload},
    ):
        with pytest.raises(ValueError):
            fresh.load_state_dict(broken)
        after = pristine()
        assert all(torch.equal(a, b) for a, b in zip(before[0], after[0])) and after[1] == before[1] == {}
    restored_bias = resumed.net.gen.bias.detach().clone()
    with torch.no_grad():
        resumed.net.gen.bias.add_(1)
    with pytest.raises(ValueError, match="disagree"):
        fresh.load_state_dict(template)
    assert not fresh.optimizers[0].backend.state
    with torch.no_grad():
        resumed.net.gen.bias.copy_(restored_bias)

    fresh.load_state_dict(template)
    for seed in range(3, 6):
        _step(resumed, fresh, fresh_scheduler, seed)
    for name, parameter in resumed.net.named_parameters():
        assert torch.equal(parameter, expected[name]), name

    # A full-precision (legacy) optimizer checkpoint has no rank-local payload to load.
    legacy = build_optimizer(_Model("cuda"), "adamw", **deepcopy(LEGACY))
    dcp.save(legacy.state_dict(), checkpoint_id=tmp_path / "legacy")
    assert fresh.checkpoint_load_template(dcp.FileSystemReader(tmp_path / "legacy").read_metadata()) is None


@CUDA
def test_nonfinite_gradient_skips_the_step_on_every_rank_like_legacy_optimizers():
    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    scheduler = _scheduler(container)
    _step(model, container, scheduler, 0)
    optimizer = container.optimizers[0]
    assert optimizer.nonfinite_policy == "skip"
    before = optimizer.state_dict()
    weights = {n: p.detach().clone() for n, p in model.net.named_parameters()}
    model.net(torch.full((4, 256), float("nan"), device="cuda")).square().mean().backward()
    container.step()  # no exception: the generator trainer keeps running, as with legacy optimizers
    scheduler.step()
    container.zero_grad()
    after = optimizer.state_dict()
    assert optimizer.skipped_nonfinite_steps == 1
    assert all(torch.equal(a, b) for a, b in zip(before["masters"], after["masters"]))
    assert [s["step"] for s in before["backend"]["state"].values()] == [
        s["step"] for s in after["backend"]["state"].values()
    ]
    assert all(torch.equal(p, weights[n]) for n, p in model.net.named_parameters())
    _step(model, container, scheduler, 1)
    assert not torch.equal(weights["gen.weight"], model.net.gen.weight)


@CUDA
@pytest.mark.parametrize("strict_resume", [True, False])
def test_checkpointer_resumes_from_a_legacy_optimizer_checkpoint_only_when_not_strict(
    tmp_path, monkeypatch, strict_resume
):
    import torch.distributed as dist

    from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
    from cosmos_framework.utils.config import CheckpointConfig, JobConfig

    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    dist.init_process_group(
        "nccl", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1, device_id=torch.device("cuda", 0)
    )
    try:

        def checkpointer(strict):
            config = CheckpointConfig(strict_resume=strict, dcp_async_mode_enabled=False)
            return DistributedCheckpointer(config, JobConfig(project="q", group="g", name="legacy-to-8bit"))

        scaler = torch.amp.GradScaler(enabled=False)
        model = _Model("cuda")
        legacy = build_optimizer(model, "adamw", **deepcopy(LEGACY))
        legacy_scheduler = _scheduler(legacy)
        _step(model, legacy, legacy_scheduler, 0)
        writer = checkpointer(True)
        writer.save(model, legacy, legacy_scheduler, scaler, iteration=1)
        writer.finalize()

        resumed = _Model("cuda")
        container = build_optimizer(resumed, "adamw_bnb_8bit", **deepcopy(KWARGS))
        scheduler = _scheduler(container)
        reader = checkpointer(strict_resume)
        try:
            if strict_resume:
                with pytest.raises(ValueError, match="no rank-local quantized optimizer state"):
                    reader.load(resumed, container, scheduler, scaler)
                return
            assert reader.load(resumed, container, scheduler, scaler) == 1
            assert torch.equal(resumed.net.gen.weight, model.net.gen.weight)  # the model did load
            assert not container.optimizers[0].backend.state  # fresh moments
            _step(resumed, container, scheduler, 1)  # masters re-seeded from the loaded model
        finally:
            reader.finalize()
    finally:
        dist.destroy_process_group()


def test_async_pinned_checkpointer_refuses_rank_local_optimizer_before_training():
    """dcp_async_mode_enabled defaults to true in generator experiments; fail at the start-of-run load."""
    from types import SimpleNamespace

    from cosmos_framework.checkpoint.dcp import AsyncMode, DistributedCheckpointer

    checkpointer = object.__new__(DistributedCheckpointer)  # no background process / process group
    checkpointer.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
    checkpointer.callbacks = None
    rank_local = SimpleNamespace(rank_local_checkpoint=True)
    with pytest.raises(ValueError, match="dcp_async_mode_enabled=false"):
        checkpointer.load(model=None, optimizer=rank_local)
    with pytest.raises(ValueError, match="synchronous DCP saves"):
        checkpointer.save(None, rank_local, None, None, iteration=1)
    # Ordinary optimizers and synchronous checkpointers are unaffected by the check.
    checkpointer._require_sync_for_rank_local_optimizer(SimpleNamespace())
    checkpointer.async_mode = AsyncMode.DISABLED
    checkpointer._require_sync_for_rank_local_optimizer(rank_local)


@CUDA
def test_warm_start_after_optimizer_construction_keeps_loaded_bf16_weights():
    """The generator trainer builds the optimizer before checkpointer.load; a model-only
    (warm-start/EMA) load must not be overwritten by construction-time BF16 masters."""
    model = _Model("cuda")
    model.net.to(torch.bfloat16)
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    optimizer = container.optimizers[0]
    torch.manual_seed(123)
    loaded = {name: torch.randn_like(value) for name, value in model.net.state_dict().items()}
    model.net.load_state_dict(loaded)  # what a model-only checkpoint load does after construction
    scheduler = _scheduler(container)
    torch.manual_seed(1)
    model.net(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)).float().square().mean().backward()
    container.step()
    scheduler.step()
    for name, parameter in model.net.named_parameters():
        # One warmup-scaled update moves weights by far less than the gap to the stale copy.
        assert (parameter.float() - loaded[name].float()).abs().max() < 1e-2, name
    for master, live in zip(optimizer.masters, optimizer.live):
        assert torch.equal(master.to(live.dtype), live.detach())


@CUDA
@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_documented_selection_without_weight_decay_uses_adamw_default(name):
    """The guide selects these by optimizer_type alone; decay falls back to AdamW's 0.01."""
    model = _Model("cuda")
    kwargs = {key: value for key, value in deepcopy(KWARGS).items() if key != "weight_decay"}
    container = build_optimizer(model, name, **kwargs)
    legacy = build_optimizer(_Model("cuda"), "adamw", **deepcopy(kwargs), fused=True)
    assert {g["weight_decay"] for g in container.optimizers[0].param_groups} == {
        g["weight_decay"] for g in legacy.optimizers[0].param_groups
    }
    assert 0.01 in {g["weight_decay"] for g in container.optimizers[0].param_groups}


@pytest.mark.parametrize(
    "invalid,match",
    [
        ({"fused": True}, "does not support fused=True"),
        ({"bnb_min_8bit_size": 0}, "positive integer"),
        ({"lr": float("nan")}, "finite and non-negative"),
        ({"eps": -1.0}, "finite and non-negative"),
        ({"weight_decay": -0.1}, "finite and non-negative"),
        ({"betas": [0.9, 1.0]}, r"\[0, 1\)"),
        ({"bnb_fp32_state_names": ["gen.weight", "gen.weight"]}, "duplicate"),
        ({"bnb_fp32_state_names": [""]}, "nonempty"),
        # Each group's effective values too: a multiplier can make a valid base invalid.
        ({"lr_multipliers": {"llm2action": -1.0}}, "parameter group lr=-5e-05"),
        (
            {"weight_decay": 1e300, "weight_decay_multipliers_for_1d_params": {r"gen\.bias": 1e10}},
            "parameter group weight_decay=inf",
        ),
    ],
)
def test_eight_bit_options_get_the_shared_validation_before_backend_construction(invalid, match, monkeypatch):
    import cosmos_framework.utils.quantized_optimizer as backend

    monkeypatch.setattr(backend, "require_backend", lambda **_: pytest.fail("backend constructed"))
    with pytest.raises(ValueError, match=match):
        build_optimizer(_Model(), "adamw_bnb_8bit", **{**deepcopy(KWARGS), **invalid})


def test_skip_patterns_match_the_model_state_dict_namespace():
    from cosmos_framework.utils.generator.optimizer import rank_local_skip_matches

    names = ["latent_pos_embed.seq", "action2llm.weight", "gen.weight"]
    # The CustomLoadPlanner docstring's own examples: model keys carry the "net." prefix.
    assert rank_local_skip_matches(["net.latent_pos_embed.seq", "net_ema.latent_pos_embed.seq"], names) == [
        "net.latent_pos_embed.seq"
    ]
    assert rank_local_skip_matches(["action2llm", "llm2action"], names) == ["action2llm"]
    assert rank_local_skip_matches(["net.frozen"], names) == []


def _assert_refuses_rank_local(cls):
    from types import SimpleNamespace

    checkpointer = object.__new__(cls)  # no object store, background process or process group
    for optimizer in (SimpleNamespace(rank_local_checkpoint=True), SimpleNamespace(cosmos_local_master_optimizer=True)):
        with pytest.raises(ValueError, match="cannot checkpoint rank-local"):
            checkpointer.load(None, optimizer)
        with pytest.raises(ValueError, match="cannot checkpoint rank-local"):
            checkpointer.save(None, optimizer, None, None, iteration=1)


def test_single_file_checkpointers_refuse_rank_local_state_before_training():
    from cosmos_framework.utils.checkpointer import Checkpointer, MultiRankCheckpointer

    _assert_refuses_rank_local(Checkpointer)
    _assert_refuses_rank_local(MultiRankCheckpointer)


def test_reasoner_checkpointer_refuses_rank_local_state_before_training():
    pytest.importorskip("transformer_engine", reason="the reasoner checkpointer module imports transformer_engine")
    from cosmos_framework.utils.reasoner.dcp_checkpointer import DistributedCheckpointer

    _assert_refuses_rank_local(DistributedCheckpointer)


def _write_checkpoint(tmp_path, model, container, scheduler, name):
    """One synchronous DCP checkpoint at iteration 1; returns its directory (for a warm start)."""
    import glob

    from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
    from cosmos_framework.utils.config import CheckpointConfig, JobConfig

    writer = DistributedCheckpointer(
        CheckpointConfig(dcp_async_mode_enabled=False), JobConfig(project="q", group="g", name=name)
    )
    writer.save(model, container, scheduler, torch.amp.GradScaler(enabled=False), iteration=1)
    writer.finalize()
    (path,) = glob.glob(str(tmp_path / "**" / name / "**" / "iter_000000001"), recursive=True)
    return path


def _warm_start(path, model, container, scheduler, **config):
    from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
    from cosmos_framework.utils.config import CheckpointConfig, JobConfig

    config = CheckpointConfig(dcp_async_mode_enabled=False, load_path=path, load_training_state=True, **config)
    reader = DistributedCheckpointer(config, JobConfig(project="q", group="g", name="reader"))
    try:
        return reader.load(model, container, scheduler, torch.amp.GradScaler(enabled=False))
    finally:
        reader.finalize()


@pytest.fixture
def _single_rank_group(tmp_path, monkeypatch):
    import torch.distributed as dist

    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    dist.init_process_group(
        "nccl", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1, device_id=torch.device("cuda", 0)
    )
    yield
    dist.destroy_process_group()


@CUDA
@pytest.mark.parametrize("mode", ["strict", "not_strict", "skip_pattern"])
def test_topology_mismatch_starts_fresh_unless_strict_resume(tmp_path, _single_rank_group, mode):
    """The strict_resume / keys_to_skip_loading decision comes before (and overrides) a topology refusal."""
    import os
    import shutil

    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    scheduler = _scheduler(container)
    _step(model, container, scheduler, 0)
    path = _write_checkpoint(tmp_path, model, container, scheduler, "writer")
    # Rewrite the optimizer component as if two ranks had written it.
    payload = container.state_dict()
    (format_key,) = payload
    payload[format_key]["rank_1"] = payload[format_key]["rank_0"].clone()
    shutil.rmtree(os.path.join(path, "optim"))
    dcp.save(payload, checkpoint_id=os.path.join(path, "optim"))

    resumed = _Model("cuda")
    fresh = build_optimizer(resumed, "adamw_bnb_8bit", **deepcopy(KWARGS))
    fresh_scheduler = _scheduler(fresh)
    config = {"strict_resume": mode != "not_strict"}
    if mode == "skip_pattern":
        config["keys_to_skip_loading"] = ["llm2action"]  # a trained parameter: exact resume is impossible anyway
    if mode == "strict":
        with pytest.raises(ValueError, match="topology mismatch"):
            _warm_start(path, resumed, fresh, fresh_scheduler, **config)
        assert not fresh.optimizers[0].backend.state
        return
    assert _warm_start(path, resumed, fresh, fresh_scheduler, **config) == 1
    assert torch.equal(resumed.net.gen.weight, model.net.gen.weight)  # the model did load
    assert not fresh.optimizers[0].backend.state  # fresh moments
    _step(resumed, fresh, fresh_scheduler, 1)


def test_ema_to_reg_warm_start_never_loads_optimizer_state_so_the_exact_master_check_holds():
    """Saved masters are compared with the model loaded from the same checkpoint. The only load
    that replaces those weights from another source (EMA into regular) cannot load optimizer state."""
    from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
    from cosmos_framework.utils.config import CheckpointConfig, JobConfig

    config = CheckpointConfig(load_ema_to_reg=True, load_training_state=True, dcp_async_mode_enabled=False)
    with pytest.raises(ValueError, match="load_ema_to_reg=True requires load_training_state=False"):
        DistributedCheckpointer(config, JobConfig(project="q", group="g", name="ema"))


@CUDA
def test_resume_with_changed_decay_and_eps_reapplies_the_configured_values_like_legacy(tmp_path):
    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    scheduler = _scheduler(container)
    for seed in range(2):
        _step(model, container, scheduler, seed)
    dcp.save(container.state_dict(), checkpoint_id=tmp_path / "optim")
    saved_lrs = [group["lr"] for group in container.optimizers[0].param_groups]

    def resume(**changes):
        resumed = _Model("cuda")
        resumed.load_state_dict(model.state_dict())
        target = build_optimizer(resumed, "adamw_bnb_8bit", **dict(deepcopy(KWARGS), **changes))
        template = target.checkpoint_load_template(dcp.FileSystemReader(tmp_path / "optim").read_metadata())
        dcp.load(template, checkpoint_id=tmp_path / "optim")
        target.load_state_dict(template)
        return resumed, target

    changed = dict(weight_decay=0.2, eps=1e-6)
    resumed, target = resume(**changed)
    optimizer = target.optimizers[0]
    reference = build_optimizer(_Model("cuda"), "adamw_bnb_8bit", **dict(deepcopy(KWARGS), **changed))
    configured = [(g["weight_decay"], g["eps"]) for g in reference.optimizers[0].param_groups]
    assert configured != [(g["weight_decay"], g["eps"]) for g in container.optimizers[0].param_groups]
    assert [(g["weight_decay"], g["eps"]) for g in optimizer.param_groups] == configured
    assert [(g["weight_decay"], g["eps"]) for g in optimizer.backend.param_groups] == configured
    assert [group["lr"] for group in optimizer.param_groups] == saved_lrs  # lr comes from the checkpoint
    assert all(state["step"] == 2 for state in optimizer.backend.state.values())  # moments were loaded
    _step(resumed, target, _scheduler(target), 2)
    # The structure stays strict: another parameter selection is refused without mutation.
    with pytest.raises(ValueError):
        resume(keys_to_select=["gen"])


@CUDA
@pytest.mark.parametrize("partial", [False, True])
def test_lenient_resume_starts_fresh_when_a_partial_model_load_kept_initial_values(
    tmp_path, _single_rank_group, partial
):
    import os
    import shutil

    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    scheduler = _scheduler(container)
    for seed in range(2):
        _step(model, container, scheduler, seed)
    path = _write_checkpoint(tmp_path, model, container, scheduler, "writer")
    if partial:
        # A model component without one trained parameter: strict_resume=False loads the rest and
        # keeps the initial value of `gen.bias`, which the saved FP32 master cannot match.
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        state = {k: v for k, v in get_model_state_dict(model).items() if k != "net.gen.bias"}
        shutil.rmtree(os.path.join(path, "model"))
        dcp.save(state, checkpoint_id=os.path.join(path, "model"))

    resumed = _Model("cuda")
    torch.manual_seed(5)
    with torch.no_grad():
        resumed.net.gen.bias.normal_()
    fresh = build_optimizer(resumed, "adamw_bnb_8bit", **deepcopy(KWARGS))
    assert _warm_start(path, resumed, fresh, _scheduler(fresh), strict_resume=False) == 1
    optimizer = fresh.optimizers[0]
    if partial:
        assert not optimizer.backend.state  # fresh moments instead of an abort
        for master, live in zip(optimizer.masters, optimizer.live):
            assert torch.equal(master.to(live.dtype), live.detach())
    else:
        assert all(state["step"] == 2 for state in optimizer.backend.state.values())  # loaded as usual
    _step(resumed, fresh, _scheduler(fresh), 3)


@CUDA
def test_state_report_is_counters_and_scalar_identity_without_layouts():
    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS), bnb_fp32_state_names=["gen.bias"])
    _step(model, container, _scheduler(container), 0)
    report = container.state_report()
    assert "master_identity" not in report and not {"groups", "topology"} & set(report["backend"])
    assert report["backend"]["name"] == "adamw_bnb_8bit" and report["quantized_elements"] > 0
    assert report["exceptions"] == {"gen.bias": "explicit", "llm2action.bias": "small_tensor"}
    # Nothing in the report aliases the live identity or report state.
    report["backend"]["fp32_state_names"].append("x")
    report["exceptions"].clear()
    backend = container.optimizers[0].backend
    assert backend.identity["fp32_state_names"] == ["gen.bias"] and container.state_report()["exceptions"]


@CUDA
@pytest.mark.parametrize("strict_resume", [True, False])
@pytest.mark.parametrize(
    "change", [{"lr_multipliers": {}}, {"bnb_min_8bit_size": 1}], ids=["changed_grouping", "changed_min_8bit_size"]
)
def test_structural_change_starts_fresh_unless_strict_resume(tmp_path, _single_rank_group, strict_resume, change):
    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    scheduler = _scheduler(container)
    _step(model, container, scheduler, 0)
    path = _write_checkpoint(tmp_path, model, container, scheduler, "writer")

    resumed = _Model("cuda")
    fresh = build_optimizer(resumed, "adamw_bnb_8bit", **dict(deepcopy(KWARGS), **change))
    if strict_resume:
        with pytest.raises(ValueError, match="identity"):
            _warm_start(path, resumed, fresh, _scheduler(fresh), strict_resume=True)
        assert not fresh.optimizers[0].backend.state
        return
    # Before the fix the identity refusal propagated although strict_resume=False promises fresh moments.
    assert _warm_start(path, resumed, fresh, _scheduler(fresh), strict_resume=False) == 1
    assert torch.equal(resumed.net.gen.weight, model.net.gen.weight)  # the model did load
    optimizer = fresh.optimizers[0]
    assert not optimizer.backend.state
    assert all(torch.equal(m.to(p.dtype), p.detach()) for m, p in zip(optimizer.masters, optimizer.live))
    _step(resumed, fresh, _scheduler(fresh), 1)


@CUDA
def test_checkpoint_snapshot_refusal_is_converged_before_dcp_collectives(monkeypatch):
    from cosmos_framework.posttrain.utils import dist as posttrain_dist

    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    _step(model, container, _scheduler(container), 0)

    def refuse():  # e.g. masters that disagree with a model changed under moments, on this rank only
        raise ValueError("model weights changed under an initialized FP32-master optimizer")

    monkeypatch.setattr(container.optimizers[0], "state_dict", refuse)
    contexts = []
    real = posttrain_dist.run_rank_converged
    monkeypatch.setattr(
        posttrain_dist,
        "run_rank_converged",
        lambda action, context, **kwargs: contexts.append(context) or real(action, context, **kwargs),
    )
    with pytest.raises(ValueError, match="changed under an initialized"):
        container.state_dict()
    assert contexts == ["quantized optimizer checkpoint snapshot"]  # peers raise with it instead of hanging


@CUDA
@pytest.mark.parametrize("strict", [True, False])
def test_backend_payload_schema_mismatch_is_decided_while_staging(strict):
    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint

    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    _step(model, container, _scheduler(container), 0)
    optimizer = container.optimizers[0]
    saved = optimizer.state_dict()
    entry = next(state for state in saved["backend"]["state"].values() if state["state1"].dtype == torch.uint8)
    entry["state1"] = entry["state1"].float()  # a moment schema only the backend's own load checked
    loaded = {checkpoint._FORMAT: {"world_size": torch.tensor([1]), "rank_0": checkpoint._encode_rank_state(saved)}}
    before = optimizer.state_dict()
    if strict:
        with pytest.raises(ValueError, match="state1 shape/dtype"):
            checkpoint.restore_into_live_model(optimizer, loaded, "test")
    else:
        # Before the fix this raised from the backend's commit although the caller was lenient.
        assert checkpoint.restore_into_live_model(optimizer, loaded, "test", strict=False) is False
    after = optimizer.state_dict()
    assert all(torch.equal(a, b) for a, b in zip(before["masters"], after["masters"]))
    assert [s["step"] for s in before["backend"]["state"].values()] == [
        s["step"] for s in after["backend"]["state"].values()
    ]


@CUDA
def test_state_report_switch_is_read_once_at_construction(monkeypatch):
    from cosmos_framework.utils.generator import optimizer as generator_optimizer

    reports = []
    monkeypatch.setattr(
        generator_optimizer.log,
        "info",
        lambda message, **kwargs: reports.append(message) if message.startswith("optimizer_state_report") else None,
    )
    for enabled in (True, False):
        reports.clear()
        if enabled:
            monkeypatch.setenv("COSMOS_OPTIMIZER_STATE_REPORT", "1")
        else:
            monkeypatch.delenv("COSMOS_OPTIMIZER_STATE_REPORT", raising=False)
        model = _Model("cuda")
        container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
        scheduler = _scheduler(container)
        # Flipping the variable after construction changes nothing: steps no longer read it.
        if enabled:
            monkeypatch.delenv("COSMOS_OPTIMIZER_STATE_REPORT")
        else:
            monkeypatch.setenv("COSMOS_OPTIMIZER_STATE_REPORT", "1")
        for seed in range(2):
            _step(model, container, scheduler, seed)
        assert len(reports) == (1 if enabled else 0)
