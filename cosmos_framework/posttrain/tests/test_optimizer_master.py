# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU numerical/ownership checks; CUDA/FSDP gates run separately."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


def model():
    net = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        net.weight.fill_(1)
    net.register_parameter("tied", net.weight)
    net.register_parameter("unused", torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16)))
    net.register_parameter("frozen", torch.nn.Parameter(torch.ones(3), requires_grad=False))
    return net


def make(net, warmup=0):
    return build_optimizer(
        net, OptimizerConfig(lr=1e-4, weight_decay=0, warmup_steps=warmup, update_policy="fp32_master")
    )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif isinstance(a, np.ndarray):  # NumPy RNG state needs array equality.
        assert np.array_equal(a, b)
    else:
        assert a == b


def test_master_retains_sub_bf16_updates_and_exact_membership():
    net = model()
    opt = make(net)
    reference = torch.nn.Parameter(net.weight.detach().float().clone())
    control = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0)
    assert opt.param_groups[0]["params"][0] is net.weight
    assert opt.names == ("weight", "unused")
    for _ in range(40):
        net.weight.grad = torch.ones_like(net.weight)
        reference.grad = torch.ones_like(reference)
        opt.step()
        control.step()
        assert torch.equal(opt.masters[0], reference)
        assert torch.equal(net.weight, reference.bfloat16())
    assert (net.weight < 1).all()  # Direct BF16 updates of this size would vanish.
    assert not opt.state.get(net.unused)
    assert torch.equal(net.frozen, torch.ones(3))


def test_master_resume_next_updates_and_rejected_state_is_transactional():
    net = model()
    opt = make(net)
    for _ in range(7):
        net.weight.grad = torch.full_like(net.weight, 0.5)
        opt.step()
    saved = opt.state_dict()
    other = model()
    other.load_state_dict(net.state_dict())
    restored = make(other)
    restored.load_state_dict(saved)
    for _ in range(4):
        net.weight.grad = torch.full_like(net.weight, -0.3)
        other.weight.grad = net.weight.grad.clone()
        opt.step()
        restored.step()
        assert_equal(opt.state_dict(), restored.state_dict())
        assert torch.equal(net.weight, other.weight)
    before = restored.state_dict()
    for corruption in ("dtype", "topology", "missing"):
        bad = deepcopy(saved)
        if corruption == "dtype":
            bad["masters"] = (bad["masters"][0].bfloat16(), *bad["masters"][1:])
        elif corruption == "topology":
            bad["identity"]["world_size"] = 2
        else:
            del bad["masters"]
        with pytest.raises(ValueError):
            restored.load_state_dict(bad)
        assert_equal(before, restored.state_dict())


def test_nonfinite_does_not_advance_master_moments_or_model():
    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    before, weights = opt.state_dict(), deepcopy(net.state_dict())
    net.unused.grad = torch.full_like(net.unused, float("nan"))
    with pytest.raises(Exception, match="nonfinite"):
        opt.step()
    assert_equal(before, opt.state_dict())
    assert_equal(weights, net.state_dict())


def test_fp32_storage_is_already_the_master_and_rejects_rebinding():
    net = model().float()
    opt = make(net)
    assert opt.masters[0].data_ptr() == net.weight.data_ptr()
    assert opt.state_report()["master_bytes"] == 0
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    saved = opt.state_dict()
    net.weight.data = net.weight.detach().clone()
    with pytest.raises(Exception, match="storage changed"):
        opt.step()
    # Reload rebinds the final live storage, with no second FP32 master allocation.
    opt.load_state_dict(saved)
    assert opt.masters[0].data_ptr() == net.weight.data_ptr()
    assert opt.state_report()["master_bytes"] == 0


def test_dcp_rank_payload_round_trip_and_rejects_damaged_or_unknown_payloads(tmp_path):
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint
    from cosmos_framework.posttrain.checkpoint.manifest import dcp_tensor_inventory

    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    payload = checkpoint.optimizer_payload(opt)
    dcp.save(payload, checkpoint_id=tmp_path)
    metadata = dcp.FileSystemReader(tmp_path).read_metadata()
    # Tensor leaves only: DCP writes them natively and never pickles (or unpickles) the payload,
    # and it never interprets internal scales or masters as model shards.
    assert set(metadata.state_dict_metadata) == {"cosmos_rank_master_v4.world_size", "cosmos_rank_master_v4.rank_0"}
    assert all(hasattr(entry, "size") for entry in metadata.state_dict_metadata.values())
    shapes = dcp_tensor_inventory(metadata, str(tmp_path))
    assert checkpoint.tensor_shapes(metadata) == shapes
    target = checkpoint.load_template(opt, shapes)
    dcp.load(target, checkpoint_id=tmp_path)
    saved = checkpoint.stage_payload(opt, target)
    assert_equal(saved, opt.state_dict())
    tensor = target["cosmos_rank_master_v4"]["rank_0"]
    changed = tensor.clone()
    changed[-1] ^= 1
    for broken, match in (
        (tensor[:-1], "checksum"),
        (changed, "checksum"),
        (tensor.float(), "schema/length"),
        (tensor.reshape(1, -1), "schema/length"),
    ):
        with pytest.raises(ValueError, match=match):
            checkpoint._decode_rank_state(broken)
    # Only the format this module writes is accepted; an older or foreign layout has no template.
    assert checkpoint.load_template(opt, {"format": None, "world_size": None, "ranks.0": None}) is None
    with pytest.raises(ValueError, match="format mismatch"):
        checkpoint.stage_payload(opt, {"cosmos_rank_master_v3": target["cosmos_rank_master_v4"]})
    wrong_world = {"cosmos_rank_master_v4": dict(target["cosmos_rank_master_v4"], world_size=torch.tensor([2]))}
    with pytest.raises(ValueError, match="topology"):
        checkpoint.stage_payload(opt, wrong_world)
    # The retained serialized byte tensor owns a stable independent snapshot.
    net.weight.grad = torch.full_like(net.weight, -0.3)
    opt.step()
    assert_equal(checkpoint._decode_rank_state(tensor), saved)


@pytest.mark.parametrize("format", ["dcp", "consolidated"])
def test_native_trainer_save_restore_master_transaction(tmp_path, format, monkeypatch):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    def trainer():
        net = model()
        objective = Objective(
            lambda model, batch: (model.net(batch[0]).float().square().mean(), {}), reduction=Reduction.MEAN
        )
        from types import SimpleNamespace

        policy = SimpleNamespace(net=net)
        from cosmos_framework.posttrain.config.build import build_lr_scheduler

        optimizer = make(net, warmup=4)
        return LLMTrainer(
            policy,
            objective,
            optimizer,
            lr_scheduler=build_lr_scheduler(optimizer, optimizer.config),
            checkpoint_manager=CheckpointManager(checkpoint_format=format, save_async=False),
        )

    original = trainer()
    # Use the native trainer counter/finite/clipping path, not manual optimizer updates.
    batch = [torch.ones(2, 4, dtype=torch.bfloat16)]
    original.train(batch)
    path = str(tmp_path / "iter_000001")
    original.save(path, data_state={"cursor": 2})
    restored = trainer()
    restored.load(path)
    assert_equal(original.opt.state_dict(), restored.opt.state_dict())
    for _ in range(3):
        original.train(batch)
        restored.train(batch)
        assert_equal(original.opt.state_dict(), restored.opt.state_dict())

    before_opt = restored.opt.state_dict()
    before_net = deepcopy(restored.model.net.state_dict())
    before_control = restored._trainer_state()
    # A rank-local metadata rejection must precede every model/control mutation.
    from cosmos_framework.posttrain.checkpoint.master_optimizer import _decode_rank_state, _encode_rank_state

    reader = restored.checkpoint_manager.load_dcp_component

    def corrupted(path, name, state, **kwargs):
        found = reader(path, name, state, **kwargs)
        if name == "optim":
            payload = state["cosmos_rank_master_v4"]
            saved = _decode_rank_state(payload["rank_0"])
            del saved["masters"]
            payload["rank_0"] = _encode_rank_state(saved)
        return found

    with monkeypatch.context() as context:
        context.setattr(restored.checkpoint_manager, "load_dcp_component", corrupted)
        with pytest.raises(Exception, match="identity"):
            restored.load(path)
    assert_equal(before_opt, restored.opt.state_dict())
    assert_equal(before_net, restored.model.net.state_dict())
    # Application failure after both model and optimizer have changed must restore
    # the old values, scheduler and clocks, rather than leaving a mixed checkpoint.
    commit = restored.opt.commit_state_dict
    calls = 0
    import gc
    import weakref

    staged_backends = []
    create_backend = restored.opt._backend

    def track_backend(masters):
        backend = create_backend(masters)
        staged_backends.append(weakref.ref(backend))
        return backend

    def fail_once(staged):
        nonlocal calls
        commit(staged)
        calls += 1
        if calls == 1:
            raise RuntimeError("injected post-commit failure")

    with monkeypatch.context() as context:
        context.setattr(restored.opt, "commit_state_dict", fail_once)
        context.setattr(restored.opt, "_backend", track_backend)
        with pytest.raises(Exception, match="injected") as failure:
            restored.load(path)
    assert failure.value is not None
    gc.collect()
    # Staging validates on the host and commits into the live tensors: no second backend
    # (moments) or master set is ever built next to the live ones.
    assert not staged_backends
    assert_equal(before_opt, restored.opt.state_dict())
    assert_equal(before_net, restored.model.net.state_dict())
    assert_equal(before_control, restored._trainer_state())

    # A valid-but-replaced generation during reads must be rejected before the
    # staged model, optimizer and clocks become live, including direct load().
    control_file = tmp_path / "iter_000001" / "trainer.pt"
    original_control_bytes = control_file.read_bytes()

    def mutated_generation(path, name, state, **kwargs):
        found = reader(path, name, state, **kwargs)
        if name == "optim":
            control_file.write_bytes(original_control_bytes + b"changed generation")
        return found

    with monkeypatch.context() as context:
        context.setattr(restored.checkpoint_manager, "load_dcp_component", mutated_generation)
        with pytest.raises(Exception, match="changed while it was being loaded"):
            restored.checkpoint_manager.load_checkpoint(path, lambda: restored.load(path))
    assert_equal(before_opt, restored.opt.state_dict())
    assert_equal(before_net, restored.model.net.state_dict())
    assert_equal(before_control, restored._trainer_state())


def test_fused_control_memory_counts_initialized_step_tensors():
    net = model()
    optimizer = make(net)
    assert optimizer.state_report()["metadata_bytes"] == 0
    net.weight.grad = torch.ones_like(net.weight)
    optimizer.step()
    counters = [state["step"] for state in optimizer.backend.state.values()]
    assert len(counters) == 1  # The unused parameter must not invent a counter.
    expected = sum(value.numel() * value.element_size() for value in counters)
    assert expected > 0
    assert optimizer.state_report()["metadata_bytes"] == expected


def test_warm_start_load_after_construction_reseeds_masters_before_the_first_step():
    """Trainers may build the optimizer before a model-only (warm-start/EMA) load."""
    net = model()
    opt = make(net)
    loaded = torch.full_like(net.weight, 0.25)
    net.load_state_dict({**net.state_dict(), "weight": loaded})  # model-only load, optimizer untouched
    reference = torch.nn.Parameter(loaded.float().clone())
    control = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0)
    for _ in range(3):
        net.weight.grad = torch.ones_like(net.weight)
        reference.grad = torch.ones_like(reference)
        opt.step()
        control.step()
        # The update starts from the loaded weights, never from the stale construction-time copy.
        assert torch.equal(opt.masters[0], reference)
        assert torch.equal(net.weight, reference.to(net.weight.dtype))


def test_model_only_load_under_initialized_moments_is_refused_without_mutation():
    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    before = deepcopy(opt.state_dict())
    with torch.no_grad():
        net.weight.fill_(3)
    opt._verify_masters = True  # as after any re-binding; a later load must not be silently overwritten
    net.weight.grad = torch.ones_like(net.weight)
    with pytest.raises(Exception, match="model weights changed under an initialized"):
        opt.step()
    # A save would write masters that disagree with the saved model: refused the same way.
    with pytest.raises(ValueError, match="model weights changed under an initialized"):
        opt.state_dict()
    assert_equal(before["masters"], [master.detach() for master in opt.masters])
    assert_equal(before["backend"], opt.backend.state_dict())
    assert torch.equal(net.weight, torch.full_like(net.weight, 3))


def test_rank_payload_decoding_refuses_arbitrary_objects_but_keeps_numpy_group_scalars():
    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint

    class Exploit:
        def __reduce__(self):
            return (print, ("arbitrary code ran on resume",))

    numpy_groups = {"param_groups": [{"lr": np.float64(1e-4), "weight_decay": np.float32(0.05)}]}
    restored = checkpoint._decode_rank_state(checkpoint._encode_rank_state(numpy_groups))
    assert restored["param_groups"][0]["lr"] == np.float64(1e-4)
    with pytest.raises(ValueError, match="refusing to load"):
        checkpoint._decode_rank_state(checkpoint._encode_rank_state({"value": Exploit()}))


def test_activation_checkpointed_model_resumes_under_canonical_state_dict_names(tmp_path):
    from types import SimpleNamespace

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    def trainer():
        torch.manual_seed(0)
        net = torch.nn.Sequential(checkpoint_wrapper(torch.nn.Linear(4, 4, dtype=torch.bfloat16)))
        objective = Objective(
            lambda model, batch: (model.net(batch[0]).float().square().mean(), {}), reduction=Reduction.MEAN
        )
        return LLMTrainer(
            SimpleNamespace(net=net),
            objective,
            make(net),
            checkpoint_manager=CheckpointManager(checkpoint_format="dcp", save_async=False),
        )

    original = trainer()
    assert original.opt.names == ("0.weight", "0.bias")  # no "_checkpoint_wrapped_module." segment
    batch = [torch.ones(2, 4, dtype=torch.bfloat16)]
    original.train(batch)
    path = str(tmp_path / "iter_000001")
    original.save(path)
    restored = trainer()
    restored.load(path)
    assert_equal(original.opt.state_dict(), restored.opt.state_dict())
    original.train(batch)
    restored.train(batch)
    assert_equal(original.model.net.state_dict(), restored.model.net.state_dict())


def _llm_trainer(net, optimizer=None, **kwargs):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    objective = Objective(lambda model, batch: (model.net(batch[0]).float().sum(), {}), reduction=Reduction.MEAN)
    return LLMTrainer(SimpleNamespace(net=net), objective, optimizer or make(net), **kwargs)


def test_offload_refuses_rank_local_optimizer_state_before_moving_anything():
    trainer = _llm_trainer(model())
    moved = []
    trainer._update_offload.offload = lambda *args, **kwargs: moved.append(args)
    # Even before the first step: offload also moves the policy, whose FP32 storage masters alias.
    with pytest.raises(ValueError, match="offload is unsupported"):
        trainer.offload()
    assert not moved and not trainer._offload.offloaded


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))],
)
def test_prepare_weight_sync_refuses_only_when_it_would_move_optimizer_state(device):
    net = model().to(device)
    trainer = _llm_trainer(net)
    # No optimizer state yet: nothing of the optimizer would move, so the hand-off proceeds.
    assert trainer.prepare_weight_sync() == 0.0
    trainer.onload()
    trainer.train([torch.ones(2, 4, dtype=torch.bfloat16, device=device)])
    moved = []
    trainer._update_offload.offload = lambda *args, **kwargs: moved.append(args) or 0.0
    if device == "cpu":  # host-resident moments: nothing to move, nothing refused
        assert trainer.prepare_weight_sync() == 0.0
        return
    with pytest.raises(ValueError, match="offload is unsupported"):  # CUDA moments would move
        trainer.prepare_weight_sync()
    assert not moved


def test_payload_presence_and_topology_are_decided_identically_on_every_rank_before_any_collective():
    from types import SimpleNamespace

    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint

    def rank(index, world):
        return SimpleNamespace(identity={"rank": index, "world_size": world})

    def shapes(world):
        return {
            "cosmos_rank_master_v4.world_size": (1,),
            **{f"cosmos_rank_master_v4.rank_{r}": (64,) for r in range(world)},
        }

    # Every current rank, including ones with a key (rank 0) and without one (rank 1), refuses alike.
    for current in (rank(0, 2), rank(1, 2)):
        with pytest.raises(ValueError, match="written by 1 ranks, resuming on 2"):
            checkpoint.load_template(current, shapes(1))
    with pytest.raises(ValueError, match="written by 2 ranks, resuming on 1"):
        checkpoint.load_template(rank(0, 1), shapes(2))
    assert checkpoint.load_template(rank(0, 1), {"state.weight": (4,)}) is None
    assert checkpoint.load_template(rank(1, 2), shapes(2))["cosmos_rank_master_v4"]["rank_1"].shape == (64,)

    class Manager:
        def dcp_tensor_inventory(self, path, name):
            return shapes(1)

        def load_dcp_component(self, *args, **kwargs):
            raise AssertionError("a collective load must not start after a local refusal")

    with pytest.raises(ValueError, match="topology mismatch"):
        checkpoint.staged_optimizer(Manager(), "unused", rank(1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
def test_pristine_reset_frees_paged_eight_bit_buffers():
    import gc
    from types import SimpleNamespace

    from cosmos_framework.posttrain.roles.trainer.base import BaseTrainer

    net = torch.nn.Linear(1024, 1024).cuda()
    opt = build_optimizer(net, OptimizerConfig(name="paged_adamw_8bit", bnb_min_8bit_size=1))
    net(torch.randn(2, 1024, device="cuda")).sum().backward()
    opt.step()
    owners = [buffer._cosmos_managed_owner for buffer in opt.page_mng.paged_tensors]
    assert owners
    trainer = SimpleNamespace(opt=opt)
    trainer._clear_optimizer_state = lambda: BaseTrainer._clear_optimizer_state(trainer)
    BaseTrainer.reset_optimizer_to_pristine(trainer)
    gc.collect()
    assert not opt.state and all(owner() is None for owner in owners)


def test_fp32_master_requires_the_trainers_finite_gradient_check():
    with pytest.raises(ValueError, match="fp32_master requires force_finite_grads=True"):
        OptimizerConfig(update_policy="fp32_master", force_finite_grads=False)
    OptimizerConfig(force_finite_grads=False)  # the legacy opt-out of the parameter policy is unchanged


def test_bf16_master_is_one_fp32_copy_and_fp32_storage_aliases():
    net = model()
    opt = make(net)
    assert opt.masters[0].dtype == torch.float32 and opt.masters[0].data_ptr() != net.weight.data_ptr()
    assert torch.equal(opt.masters[0], net.weight.float())
    fp32 = torch.nn.Linear(4, 3)
    assert all(m.data_ptr() == p.data_ptr() for m, p in zip(make(fp32).masters, fp32.parameters()))


def test_model_facing_state_is_rebuilt_only_when_backend_state_appears():
    net = model()
    opt = make(net)

    def step(*parameters):
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        opt.step()

    step(net.weight)
    mapping = opt.state
    assert mapping[net.weight] is opt.backend.state[opt.masters[0]]
    step(net.weight)
    assert opt.state is mapping  # no new backend state: the mapping is reused, not rebuilt
    step(net.weight, net.unused)
    assert opt.state is not mapping and opt.state[net.unused] is opt.backend.state[opt.masters[1]]


def test_pristine_reset_drops_fp32_master_adamw_backend_moments():
    """The master owner's ``state`` is a view: clearing it would leave the backend moments alive."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.roles.trainer.base import BaseTrainer

    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    assert opt.backend.state[opt.masters[0]]["step"] == 1
    trainer = SimpleNamespace(opt=opt)
    trainer._clear_optimizer_state = lambda: BaseTrainer._clear_optimizer_state(trainer)
    BaseTrainer.reset_optimizer_to_pristine(trainer)
    assert not opt.backend.state and not opt.state
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    assert opt.backend.state[opt.masters[0]]["step"] == 1  # a first step again, not the second


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
@pytest.mark.parametrize("force_finite_grads", [True, False])
def test_trainer_finite_check_spares_the_bare_eight_bit_backend_its_own_scan(force_finite_grads, monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(512, 64, device="cuda")
            self.head = torch.nn.Linear(64, 512, device="cuda")

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    packer = MemoryPacker()
    batch = [packer.add(list(range(64)), [-100] * 8 + list(range(8, 64)))]
    policy = Policy(Net())
    opt = build_optimizer(policy, OptimizerConfig(name="adamw_bnb_8bit", lr=1e-3))
    trainer = LLMTrainer(policy, make_objective(packer), opt, force_finite_grads=force_finite_grads)
    scans = []
    real = torch._foreach_norm
    monkeypatch.setattr(torch, "_foreach_norm", lambda *a, **k: scans.append(1) or real(*a, **k))
    for _ in range(2):
        trainer.train(batch)
    # The trainer's collective check already decided; only an unchecked window scans in the backend.
    assert len(scans) == (0 if force_finite_grads else 2)


def test_every_rank_validates_every_rank_payload_entry_before_any_collective():
    from types import SimpleNamespace

    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint

    shapes = {
        "cosmos_rank_master_v4.world_size": (1,),
        "cosmos_rank_master_v4.rank_0": (64,),
        "cosmos_rank_master_v4.rank_1": (2, 32),  # only rank 1's own entry is malformed
    }
    for rank in (0, 1):  # before the fix rank 0 accepted it and entered the load alone
        with pytest.raises(ValueError, match="byte tensor schema"):
            checkpoint.load_template(SimpleNamespace(identity={"rank": rank, "world_size": 2}), shapes)


def test_rollback_snapshots_converge_a_local_refusal(monkeypatch):
    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint

    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    saved = opt.state_dict()
    opt._verify_masters = True
    with torch.no_grad():
        net.weight.add_(1)  # this rank's state_dict() now refuses (masters disagree under moments)
    contexts = []
    real = checkpoint.run_rank_converged

    def converged(action, context, **kwargs):
        contexts.append(context)
        return real(action, context, **kwargs)

    monkeypatch.setattr(checkpoint, "run_rank_converged", converged)
    with pytest.raises(ValueError, match="changed under an initialized"):
        checkpoint._commit_with_rollback(opt, saved, "restore")
    # The refusal was raised inside a converged action, so peer ranks raise with it.
    assert contexts == ["restore rollback snapshot"]


@pytest.mark.parametrize("policy", ["fp32_master", "default"])
def test_save_releases_completed_gradients_only_for_opt_in_optimizers(tmp_path, policy, monkeypatch):
    from cosmos_framework.posttrain.checkpoint import CheckpointManager

    net = model()
    optimizer = make(net) if policy == "fp32_master" else torch.optim.AdamW([net.weight, net.unused], lr=1e-4)
    trainer = _llm_trainer(
        net, optimizer, checkpoint_manager=CheckpointManager(checkpoint_format="dcp", save_async=False)
    )
    trainer.train([torch.ones(2, 4, dtype=torch.bfloat16)])
    assert net.weight.grad is not None
    released = []
    monkeypatch.setattr(
        trainer,
        "_release_update_memory_for_checkpoint",
        lambda real=trainer._release_update_memory_for_checkpoint: released.append(1) or real(),
    )
    trainer.save(str(tmp_path / "iter_000001"))
    # SFT and AlpaGym get the same release as Joint PPO; default optimizers keep their save path.
    assert (net.weight.grad is None) is (policy == "fp32_master")
    assert released == ([1] if policy == "fp32_master" else [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
@pytest.mark.parametrize("policy", ["fp32_master", "parameter"])
def test_save_returns_the_device_cache_only_for_fp32_master(policy, monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    dtype = torch.bfloat16 if policy == "fp32_master" else torch.float32
    net = torch.nn.Linear(64, 64, dtype=dtype, device="cuda")
    opt = build_optimizer(net, OptimizerConfig(name="adamw_bnb_8bit", update_policy=policy))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    LLMTrainer._release_update_memory_for_checkpoint(SimpleNamespace(opt=opt))
    assert all(parameter.grad is None for parameter in net.parameters())  # every opt-in optimizer
    # The single-device parameter policy has no DCP/NCCL pressure: no device sync or cache release.
    assert emptied == ([1] if policy == "fp32_master" else [])
