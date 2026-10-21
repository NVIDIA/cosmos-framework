# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for LLMTrainer (DESIGN §6, model/algorithm decoupled). Tiny random GPT2 on CPU.

Verifies the engine trainer is objective-driven: it drives the GRPO objective
(`algorithm.grpo.make_objective`) — `train(batch)` returns metrics + the optimizer steps.
The algorithm is NOT baked into the trainer (that's the point of the refactor)."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective, make_objective
from cosmos_framework.posttrain.algorithm.primitives.objective import MetricAdapter, Objective, Reduction
from cosmos_framework.posttrain.data.packer import init_data_packer
from cosmos_framework.posttrain.roles.trainer import LLMTrainer
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


def _tiny():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return model, tok


def _traj(tok, group_id, reward, prompt="1 + 1 =", completion=" 2"):
    pids = tok(prompt, return_tensors="pt")["input_ids"][0]
    cids = tok(completion, return_tensors="pt")["input_ids"][0]
    tr = Transition(
        obs=ObsBundle(representations={"prompt": prompt, "reference": 2}),
        action=cids,
        policy_info={"prompt_ids": pids, "completion_text": completion},
        reward={"final": reward},
        done=True,
    )
    return Trajectory(transitions=[tr], id="t", group_id=group_id, status=Status.COMPLETED)


def _with_advantages(trajs, unbiased: bool = False):
    """Apply the reward-stage estimator, as the real pipeline does before dispatch.

    The objective no longer computes the group-relative advantage; it reads `transition.advantage`.
    A test that hands it raw trajectories is skipping the reward stage, so it must supply what that
    stage would have written.
    """
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    group_relative(unbiased=unbiased)(list(trajs))
    return trajs


@pytest.mark.L0
@pytest.mark.CPU
def test_concat_batches_flattens_trajectory_lists_in_order():
    # The ("trainer","train","batch") reduce adaptation (DESIGN §2.3): n producer units, each a
    # list of Trajectory, concatenate into ONE batch the body consumes — order-preserving so the
    # first producer unit's trajectories lead.
    from cosmos_framework.posttrain.roles.trainer.llm import concat_batches

    _, tok = _tiny()
    b0 = [_traj(tok, "g0", 1.0), _traj(tok, "g0", 0.0)]
    b1 = [_traj(tok, "g1", 1.0)]
    merged = concat_batches([b0, b1])
    assert [t.group_id for t in merged] == ["g0", "g0", "g1"]  # flat, order preserved
    assert merged == b0 + b1


@pytest.mark.L0
@pytest.mark.CPU
def test_concat_batches_single_unit_is_identity_contents():
    # n == 1 (P == R_c, the op=="one" path) never routes through reduce, but the fn must still be a
    # faithful identity on contents so a defensive/1-unit call is a no-op.
    from cosmos_framework.posttrain.roles.trainer.llm import concat_batches

    _, tok = _tiny()
    b0 = [_traj(tok, "g0", 1.0)]
    assert concat_batches([b0]) == b0


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_driven_train_steps_params():
    model, tok = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)  # large lr so the step is visible
    trainer = LLMTrainer(model, make_objective(init_data_packer("gpt2")), opt)
    batch = _with_advantages([_traj(tok, "g0", 1.0, completion=" 2"), _traj(tok, "g0", 0.0, completion=" 9")])
    before = next(model.parameters()).detach().clone()
    metrics = trainer.train(batch)
    after = next(model.parameters()).detach()
    assert isinstance(metrics, dict)
    assert metrics["train/loss_avg"] == metrics["train/loss_avg"]  # not NaN
    assert metrics["train/valid_sequence_count"] == 2
    assert metrics["train/learning_rate"] == 1.0  # engine-truth lr exported (i4 WandBCallback parity)
    assert not torch.allclose(before, after)  # optimizer actually stepped


@pytest.mark.L0
@pytest.mark.CPU
def test_save_load_component_layout_unified_with_diffusion(tmp_path):
    """The HF-model trainer writes the SAME component layout as DiffusionTrainer
    (checkpoint package) — one checkpoint shape across engines/backends — and resume
    round-trips weights/optimizer/iteration."""
    model, tok = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = LLMTrainer(model, make_objective(init_data_packer("gpt2")), opt)
    trainer.train(_with_advantages([_traj(tok, "g0", 1.0, completion=" 2"), _traj(tok, "g0", 0.0, completion=" 9")]))
    ckpt = tmp_path / "checkpoints" / "iter_000001"
    trainer.save(str(ckpt))
    for name in ("model", "optim", "trainer"):
        assert (ckpt / f"{name}.pt").is_file()
    assert (tmp_path / "checkpoints" / "latest_checkpoint.txt").read_text().strip() == "iter_000001"

    model2, _ = _tiny()
    trainer2 = LLMTrainer(
        model2, make_objective(init_data_packer("gpt2")), torch.optim.SGD(model2.parameters(), lr=1.0)
    )
    assert not torch.allclose(next(model2.parameters()), next(model.parameters()))
    assert trainer2.load(str(ckpt)) == 1
    assert trainer2.iteration() == 1
    assert torch.allclose(next(model2.parameters()), next(model.parameters()))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
@pytest.mark.parametrize("selected", ["model", "optimizer"])
def test_checkpoint_plan_extracts_only_selected_llm_state(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_format: str,
    selected: str,
) -> None:
    """A partial plan never materializes an excluded LLM state family."""
    import torch.distributed.checkpoint.state_dict as state_dict_module
    import torch.nn as nn

    def _forbid_extraction(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("excluded checkpoint state was extracted")

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        optimizer,
        checkpoint_format=checkpoint_format,
    )
    monkeypatch.setattr(trainer, "_trainer_state", _forbid_extraction)
    if selected == "model":
        monkeypatch.setattr(optimizer, "state_dict", _forbid_extraction)
        monkeypatch.setattr(state_dict_module, "get_optimizer_state_dict", _forbid_extraction)
    else:
        monkeypatch.setattr(model, "state_dict", _forbid_extraction)
        monkeypatch.setattr(state_dict_module, "get_model_state_dict", _forbid_extraction)

    save = Mock(return_value="unused")
    monkeypatch.setattr(trainer.checkpoint_manager, f"save_{checkpoint_format}", save)
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        trainer.save(
            str(tmp_path / "checkpoints" / "iter_000001"),
            update_latest=False,
            components=(selected,),
        )
    captured = save.call_args.args[1]
    physical = "model" if selected == "model" else "optim"
    assert {name for name, state in captured.items() if state is not None} == {physical}
    if checkpoint_format == "dcp":
        assert save.call_args.args[2] == {}


# ---------------------------------------------------------------------------
# `.net` contract + grad-clip knob (Task 7). The trainer's `model` may be a raw
# nn.Module (no `.net` — today's GPT2 path) or an HFModel body (owns `.net`); the
# `_net()` helper unifies both and everything parameter-touching goes through it.
# ---------------------------------------------------------------------------


class _WithNet:
    """Stand-in for an HFModel body: a plain object that owns its module under `.net`
    (NOT an nn.Module itself), so `_net()` must resolve through the attribute."""

    def __init__(self, net):
        self.net = net


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_trainer_net_helper_raw_model():
    """A raw module has no `.net` ⇒ `_net()` returns the module itself, so the existing
    raw-GPT2 path (train/save/load/objective) is byte-for-byte preserved."""
    import torch.nn as nn

    m = nn.Linear(4, 4)
    t = LLMTrainer(m, make_objective(init_data_packer("gpt2")), torch.optim.SGD(m.parameters(), lr=0.1))
    assert t._net() is m  # raw model has no .net -> returns itself


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_trainer_net_helper_body_with_net():
    """A body exposing `.net` (HFModel shape) ⇒ `_net()` returns the owned module."""
    import torch.nn as nn

    net = nn.Linear(4, 4)
    body = _WithNet(net)
    t = LLMTrainer(body, make_objective(init_data_packer("gpt2")), torch.optim.SGD(net.parameters(), lr=0.1))
    assert t._net() is net  # body.net is the trainable module


def _big_grad_objective(model, batch):
    """Tiny objective with a deterministic, LARGE-gradient loss so grad-clipping has a
    visible effect (decoupled from GRPO — just drives train()'s clip→step path)."""
    x = torch.full((1, 4), 10.0)
    loss = model(x).sum() * 100.0
    return loss, {"loss": float(loss.detach())}


def _grad_clip_step_delta(max_grad_norm: float) -> float:
    """One train step on a fixed-seed Linear (lr=1.0); return the param-update L2 norm.
    With SGD lr=1.0 the update norm equals the (possibly clipped) gradient norm."""
    import torch.nn as nn

    torch.manual_seed(0)
    m = nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    t = LLMTrainer(m, Objective(_big_grad_objective, reduction=Reduction.MEAN), opt, max_grad_norm=max_grad_norm)
    before = torch.cat([p.detach().flatten() for p in m.parameters()]).clone()
    t.train([None])  # batch is unused by _big_grad_objective
    after = torch.cat([p.detach().flatten() for p in m.parameters()])
    return float((after - before).norm())


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_trainer_grad_clip_noop_at_zero_and_clips_when_set():
    """max_grad_norm=0.0 is a no-op (default, today's behavior); a positive cap bounds the
    update so the clipped step is strictly smaller and its norm respects the cap."""
    unclipped = _grad_clip_step_delta(0.0)
    clipped = _grad_clip_step_delta(1.0)
    assert unclipped > 1.0  # large-gradient objective ⇒ big unclipped step (sanity)
    assert clipped < unclipped  # clipping actually shrinks the step
    assert clipped <= 1.0 + 1e-4  # lr=1.0 ⇒ update norm == clipped grad norm <= max_grad_norm


def _zero_grad_objective(model: Any, _batch: list[Any]) -> tuple[torch.Tensor, dict[str, float]]:
    """Materialize exactly-zero gradients on every parameter used by the scalar loss."""
    loss = sum(parameter.sum() for parameter in model.parameters()) * 0.0  # []
    return loss, {"loss": float(loss.detach())}


def _make_zero_grad_trainer(*, skip_zero_grad: bool) -> tuple[LLMTrainer, torch.nn.Module, list[torch.Tensor]]:
    """Build an AdamW trainer whose zero-gradient step makes weight decay observable."""
    import torch.nn as nn

    torch.manual_seed(41)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        optimizer,
        max_grad_norm=1.0,
        skip_zero_grad=skip_zero_grad,
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]  # each [*parameter]
    return trainer, model, before


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_sync_preparation_parks_only_update_state_and_full_offload_moves_policy_last() -> None:
    """Two-stage residency keeps policy shards live for IPC and preserves one full round-trip."""
    import torch.nn as nn

    events: list[tuple[str, str]] = []

    class _Mover:
        def __init__(self, name: str, moved_mib: float) -> None:
            self.name = name
            self.moved_mib = moved_mib
            self.offloaded = False
            self.modules: list[torch.nn.Module] = []
            self.optimizer: torch.optim.Optimizer | None = None

        def offload(
            self,
            modules: Any = (),
            *,
            optimizer: torch.optim.Optimizer | None = None,
            grads_of: Any = (),
        ) -> float:
            del grads_of
            if self.offloaded:
                return 0.0
            self.modules = list(modules)
            self.optimizer = optimizer
            self.offloaded = True
            events.append((self.name, "offload"))
            return self.moved_mib

        def onload(self) -> float:
            if not self.offloaded:
                return 0.0
            self.offloaded = False
            events.append((self.name, "onload"))
            return self.moved_mib

    policy = nn.Linear(4, 4)
    reference = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    policy(torch.ones(1, 4)).sum().backward()  # [1,4] -> []
    optimizer.step()
    trainer = LLMTrainer(
        policy,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        optimizer,
        auxiliary_models=[reference],
    )
    policy_mover = _Mover("policy", 10.0)
    update_mover = _Mover("update", 30.0)
    trainer._offload = policy_mover
    trainer._update_offload = update_mover

    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert trainer.prepare_weight_sync() == 30.0
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert update_mover.modules == [reference]
    assert update_mover.optimizer is optimizer
    assert policy not in update_mover.modules
    assert policy_mover.modules == []
    assert trainer.prepare_weight_sync() == 0.0

    assert trainer.offload() == 40.0
    assert policy_mover.modules == [policy]
    assert policy_mover.optimizer is None
    assert trainer.offload() == 0.0
    assert trainer.onload() == 40.0
    assert events == [
        ("update", "offload"),
        ("policy", "offload"),
        ("policy", "onload"),
        ("update", "onload"),
    ]
    assert trainer.prepare_weight_sync() == 30.0  # a restored trainer can begin the next cycle


@pytest.mark.L0
@pytest.mark.CPU
def test_full_offload_preserves_an_open_accumulation_window_for_nft() -> None:
    """NFT can yield the shared card mid-window without invoking boundary-only selective parking."""
    import torch.nn as nn

    events: list[str] = []

    class _Mover:
        def __init__(self, moved_mib: float) -> None:
            self.moved_mib = moved_mib
            self.offloaded = False
            self.modules: list[torch.nn.Module] = []
            self.optimizer: torch.optim.Optimizer | None = None

        def offload(
            self,
            modules: Any = (),
            *,
            optimizer: torch.optim.Optimizer | None = None,
            grads_of: Any = (),
        ) -> float:
            del grads_of
            self.modules = list(modules)
            self.optimizer = optimizer
            self.offloaded = True
            events.append("offload")
            return self.moved_mib

        def onload(self) -> float:
            if not self.offloaded:
                return 0.0
            self.offloaded = False
            events.append("onload")
            return self.moved_mib

    policy = nn.Linear(4, 4)
    reference = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    trainer = LLMTrainer(
        policy,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        optimizer,
        auxiliary_models=[reference],
    )
    combined_mover = _Mover(40.0)
    update_mover = _Mover(30.0)
    trainer._offload = combined_mover
    trainer._update_offload = update_mover
    trainer._accum = 1

    assert trainer.offload() == 40.0
    assert combined_mover.modules == [policy, reference]
    assert combined_mover.optimizer is optimizer
    assert not update_mover.offloaded
    assert trainer.onload() == 40.0
    assert events == ["offload", "onload"]


@pytest.mark.L1
@pytest.mark.GPU
def test_two_stage_residency_round_trip_keeps_policy_live_and_restores_training_state() -> None:
    """Real CUDA tensors follow update-only -> policy -> policy -> update ordering without drift."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    import os

    import torch.nn as nn

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID", "0")))
    policy = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 16)).cuda()
    reference = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 16)).cuda()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    inputs = torch.ones(8, 32, device="cuda")  # [B,32]
    policy(inputs).sum().backward()  # [B,16] -> []
    optimizer.step()
    trainer = LLMTrainer(
        policy,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        optimizer,
        auxiliary_models=[reference],
    )
    policy_before = [parameter.detach().clone() for parameter in policy.parameters()]  # each [*parameter]
    reference_before = [parameter.detach().clone() for parameter in reference.parameters()]  # each [*parameter]
    optimizer_before = {
        parameter: {
            name: (value.detach().clone(), value.device) for name, value in state.items() if torch.is_tensor(value)
        }
        for parameter, state in optimizer.state.items()
    }
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()

    update_mib = trainer.prepare_weight_sync()

    assert update_mib > 0.0
    assert all(parameter.device.type == "cuda" for parameter in policy.parameters())
    assert all(parameter.device.type == "cpu" for parameter in reference.parameters())
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
    )
    assert torch.cuda.memory_allocated() < allocated_before

    total_mib = trainer.offload()
    assert total_mib > update_mib
    assert all(parameter.device.type == "cpu" for parameter in policy.parameters())
    assert trainer.onload() == pytest.approx(total_mib)
    assert all(parameter.device.type == "cuda" for parameter in policy.parameters())
    assert all(parameter.device.type == "cuda" for parameter in reference.parameters())
    for parameter, saved in zip(policy.parameters(), policy_before, strict=True):
        torch.testing.assert_close(parameter, saved)
    for parameter, saved in zip(reference.parameters(), reference_before, strict=True):
        torch.testing.assert_close(parameter, saved)
    for parameter, saved_state in optimizer_before.items():
        for name, (saved, original_device) in saved_state.items():
            assert optimizer.state[parameter][name].device == original_device
            torch.testing.assert_close(optimizer.state[parameter][name], saved)

    optimizer.zero_grad(set_to_none=True)
    policy(inputs).sum().backward()  # [B,16] -> []
    optimizer.step()


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_policy_is_owned_by_the_shared_manager() -> None:
    """LLM trainers accept shared archive/retention without growing a second lifecycle."""
    import torch.nn as nn

    model = nn.Linear(2, 2)
    upload = {"bucket": "archive", "prefix": "posttrain/checkpoints"}
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        torch.optim.SGD(model.parameters(), lr=0.1),
        save_async=True,
        keep_optim_last=2,
        keep_model_last=3,
        upload=upload,
    )

    assert trainer.save_async is True
    assert trainer.keep_optim_last == 2
    assert trainer.keep_model_last == 3
    assert trainer.upload == upload
    assert trainer.checkpoint_manager.save_async is True
    assert trainer.checkpoint_manager.keep_optim_last == 2
    assert trainer.checkpoint_manager.keep_model_last == 3
    assert trainer.checkpoint_manager.upload == upload


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_primary_store_is_owned_by_the_shared_manager_and_keeps_dcp_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn as nn

    from cosmos_framework.posttrain.callbacks.base import Callback
    from cosmos_framework.posttrain.checkpoint import CheckpointManager

    published: list[str] = []

    class _PublishSpy(Callback):
        def on_checkpoint(self, _trainer: Any, path: str) -> None:
            published.append(path)

    model = nn.Linear(2, 2)
    manager = CheckpointManager(
        checkpoint_format="dcp",
        checkpoint_store={
            "backend": "object_store",
            "bucket": "primary",
            "prefix": "run/checkpoints",
            "credentials": "unused-remote-credentials.json",
        },
    )
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[_PublishSpy()],
        checkpoint_manager=manager,
    )

    assert trainer.checkpoint_manager is manager
    expected = "s3://primary/run/checkpoints/iter_000000"
    monkeypatch.setattr(trainer.checkpoint_manager, "save_dcp", lambda *_args, **_kwargs: expected)
    trainer.save("/local/run/checkpoints/iter_000000")
    assert published == [expected]
    with pytest.raises(ValueError, match="requires checkpoint_format='dcp'"):
        trainer.checkpoint_format = "consolidated"


@pytest.mark.L0
@pytest.mark.CPU
def test_unsharded_consolidated_save_forces_sync_and_forwards_archive_retention_policy(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain state_dict aliases live parameters, so save_async must not enqueue its write."""
    import torch.nn as nn

    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager_module

    captured: dict[str, Any] = {}

    def _capture_save(
        iter_dir: str,
        components: dict[str, dict | None],
        async_write: bool = False,
        keep_optim_last: int = 0,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        update_latest: bool = True,
    ) -> None:
        captured.update(
            iter_dir=iter_dir,
            components=components,
            async_write=async_write,
            keep_optim_last=keep_optim_last,
            keep_model_last=keep_model_last,
            upload=upload,
            update_latest=update_latest,
        )

    monkeypatch.setattr(checkpoint_manager_module, "save_components", _capture_save)
    model = nn.Linear(2, 2)
    upload = {"bucket": "archive", "prefix": "posttrain/checkpoints"}
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        torch.optim.SGD(model.parameters(), lr=0.1),
        save_async=True,
        keep_optim_last=2,
        keep_model_last=3,
        upload=upload,
    )
    checkpoint = tmp_path / "checkpoints" / "iter_000000"

    trainer.save(str(checkpoint))

    assert captured["iter_dir"] == str(checkpoint)
    assert captured["async_write"] is False
    assert captured["keep_optim_last"] == 2
    assert captured["keep_model_last"] == 3
    assert captured["upload"] == upload
    assert captured["update_latest"] is True
    saved_weight = captured["components"]["model"]["net"]["weight"]  # [O,I]
    assert saved_weight.data_ptr() == model.weight.data_ptr()


@pytest.mark.L0
@pytest.mark.CPU
def test_sharded_cpu_snapshot_can_write_async_and_finalize(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-offloaded full state is immutable before the background writer owns it."""
    import torch.distributed.checkpoint.state_dict as state_dict_module
    import torch.nn as nn

    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager_module
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    captured: dict[str, Any] = {}

    def _get_model_state_dict(
        net: torch.nn.Module,
        *,
        options: Any,
    ) -> dict[str, torch.Tensor]:
        assert options.full_state_dict and options.cpu_offload
        return {
            "weight": net.weight.detach().cpu().clone(),  # [O,I]
            "bias": net.bias.detach().cpu().clone(),  # [O]
        }

    def _get_optimizer_state_dict(
        _net: torch.nn.Module,
        _optimizer: torch.optim.Optimizer,
        *,
        options: Any = None,
    ) -> dict[str, Any]:
        # The sharded optimizer is written as DCP shards from the per-rank extraction, never gathered.
        assert options is None, "the consolidated save must not gather the optimizer on rank 0"
        return {
            "state": {"weight": {"moment": torch.ones(2, 2)}},  # moment: [O,I]
            "param_groups": [],
        }

    def _blocked_write(
        iter_dir: str,
        components: dict[str, dict | None],
        keep_optim_last: int = 0,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        update_latest: bool = True,
    ) -> None:
        captured.update(
            iter_dir=iter_dir,
            components=components,
            keep_optim_last=keep_optim_last,
            keep_model_last=keep_model_last,
            upload=upload,
            update_latest=update_latest,
        )
        entered.set()
        assert release.wait(timeout=5.0), "test did not release the checkpoint writer"
        finished.set()

    monkeypatch.setattr(llm_module, "is_dtensor_sharded", lambda _net: True)
    monkeypatch.setattr(state_dict_module, "get_model_state_dict", _get_model_state_dict)
    monkeypatch.setattr(state_dict_module, "get_optimizer_state_dict", _get_optimizer_state_dict)
    monkeypatch.setattr(checkpoint_manager_module, "_write_files", _blocked_write)

    model = nn.Linear(2, 2)
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        torch.optim.SGD(model.parameters(), lr=0.1),
        save_async=True,
    )
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    expected_weight = model.weight.detach().clone()  # [O,I]

    try:
        trainer.save(str(checkpoint))
        assert entered.wait(timeout=2.0), "save did not launch the background writer"
        assert not finished.is_set(), "save unexpectedly waited for the background writer"

        with torch.no_grad():
            model.weight.add_(10.0)  # [O,I]
        saved_weight = captured["components"]["model"]["net"]["weight"]  # [O,I]
        assert saved_weight.device.type == "cpu"
        assert torch.equal(saved_weight, expected_weight)

        release.set()
        trainer.finalize_checkpoints()
        assert finished.is_set()
    finally:
        release.set()
        # Never leak the process-global writer slot into the next checkpoint test.
        trainer.finalize_checkpoints()


@pytest.mark.L0
@pytest.mark.CPU
def test_sharded_cpu_snapshot_that_aliases_live_state_forces_sync(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU offload is not sufficient when a state-dict implementation returns a live CPU tensor."""
    import torch.distributed.checkpoint.state_dict as state_dict_module
    import torch.nn as nn

    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager_module
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    captured: dict[str, Any] = {}

    def _get_model_state_dict(net: torch.nn.Module, *, options: Any) -> dict[str, torch.Tensor]:
        assert options.full_state_dict and options.cpu_offload
        return {
            "weight": net.weight,  # [O,I] — intentionally a live alias
            "bias": net.bias,  # [O] — intentionally a live alias
        }

    def _get_optimizer_state_dict(
        _net: torch.nn.Module,
        _optimizer: torch.optim.Optimizer,
        *,
        options: Any = None,
    ) -> dict[str, Any]:
        assert options is None, "the consolidated save must not gather the optimizer on rank 0"
        return {"state": {}, "param_groups": []}

    def _capture_save(
        iter_dir: str,
        components: dict[str, dict | None],
        async_write: bool = False,
        keep_optim_last: int = 0,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        update_latest: bool = True,
    ) -> None:
        captured.update(
            iter_dir=iter_dir,
            components=components,
            async_write=async_write,
            keep_optim_last=keep_optim_last,
            keep_model_last=keep_model_last,
            upload=upload,
            update_latest=update_latest,
        )

    monkeypatch.setattr(llm_module, "is_dtensor_sharded", lambda _net: True)
    monkeypatch.setattr(state_dict_module, "get_model_state_dict", _get_model_state_dict)
    monkeypatch.setattr(state_dict_module, "get_optimizer_state_dict", _get_optimizer_state_dict)
    monkeypatch.setattr(checkpoint_manager_module, "save_components", _capture_save)

    model = nn.Linear(2, 2)
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        torch.optim.SGD(model.parameters(), lr=0.1),
        save_async=True,
    )
    trainer.save(str(tmp_path / "checkpoints" / "iter_000000"))

    assert captured["async_write"] is False
    assert captured["components"]["model"]["net"]["weight"].data_ptr() == model.weight.data_ptr()


def _sharded_layout_trainer(seed: int, **kwargs: Any) -> tuple[LLMTrainer, torch.nn.Module]:
    """An AdamW trainer on a small untied two-layer net; the caller patches `is_dtensor_sharded`."""
    import torch.nn as nn

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    return LLMTrainer(model, Objective(_big_grad_objective, reduction=Reduction.MEAN), optimizer, **kwargs), model


def _spy_optimizer_extraction(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the `options` of every `get_optimizer_state_dict` call while delegating to the real one."""
    import torch.distributed.checkpoint.state_dict as state_dict_module

    real = state_dict_module.get_optimizer_state_dict
    seen: list[Any] = []

    def _spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("options"))
        return real(*args, **kwargs)

    monkeypatch.setattr(state_dict_module, "get_optimizer_state_dict", _spy)
    return seen


def _assert_same_weights_and_moments(source: LLMTrainer, restored: LLMTrainer) -> None:
    """Weights, AdamW moments/steps and group options agree; neither trainer is advanced."""
    source_params = list(source._net().parameters())
    restored_params = list(restored._net().parameters())
    assert all(torch.equal(a, b) for a, b in zip(source_params, restored_params, strict=True))
    for a, b in zip(source_params, restored_params, strict=True):
        assert set(source.opt.state[a]) == set(restored.opt.state[b]) == {"step", "exp_avg", "exp_avg_sq"}
        assert all(torch.equal(source.opt.state[a][key], restored.opt.state[b][key]) for key in source.opt.state[a])
    assert restored.opt.param_groups[0]["lr"] == source.opt.param_groups[0]["lr"]


def _assert_same_training_state(source: LLMTrainer, restored: LLMTrainer) -> None:
    """Weights, AdamW moments/steps and group options agree, and one more step stays in lockstep.

    Advances BOTH trainers; to compare one source against several restores use
    `_assert_same_weights_and_moments`.
    """
    _assert_same_weights_and_moments(source, restored)
    source_params = list(source._net().parameters())
    restored_params = list(restored._net().parameters())
    source.train([None])
    restored.train([None])
    assert all(torch.equal(a, b) for a, b in zip(source_params, restored_params, strict=True))


@pytest.mark.L0
@pytest.mark.CPU
def test_sharded_consolidated_save_writes_the_optimizer_as_dcp_shards_and_resumes(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sharded trainer's consolidated full save never gathers the optimizer onto one host.

    The gathered fp32 moments of a 30 B-parameter policy are about 244 GB next to the gathered
    model; on a one-node colocated run that gather reached 857 GB of a 900 GB node and Ray killed
    the trainer, so no terminated run had a resumable checkpoint. The optimizer is written as
    per-rank DCP shards next to the consolidated model and control files, the directory is a
    complete exact-resume checkpoint for every reader, and it loads back into a sharded trainer.
    """
    from cosmos_framework.posttrain.checkpoint.io import component_kind, is_complete_iter_dir
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    monkeypatch.setattr(llm_module, "is_dtensor_sharded", lambda _net: True)
    extractions = _spy_optimizer_extraction(monkeypatch)
    trainer, _ = _sharded_layout_trainer(seed=1)
    trainer.train([None])

    root = tmp_path / "checkpoints"
    checkpoint = root / "iter_000001"
    trainer.save(str(checkpoint))
    trainer.finalize_checkpoints()

    assert extractions, "the optimizer component was not extracted"
    assert all(options is None or not options.full_state_dict for options in extractions), extractions
    assert {path.name for path in checkpoint.iterdir()} == {"model.pt", "optim", "trainer.pt"}
    assert (checkpoint / "optim" / ".metadata").is_file()
    assert component_kind(str(checkpoint), "model") == "pt"
    assert component_kind(str(checkpoint), "optim") == "dcp"
    assert is_complete_iter_dir(str(checkpoint))
    assert (root / "latest_checkpoint.txt").read_text().strip() == "iter_000001"
    manager = trainer.checkpoint_manager
    assert manager.resolve_resume("auto", str(root)) == str(checkpoint)
    assert manager.checkpoint_provenance(str(checkpoint))["source"] == str(checkpoint.resolve())
    assert not manager.has_dcp_component(str(checkpoint), "model")
    assert manager.has_dcp_component(str(checkpoint), "optim")

    restored, _ = _sharded_layout_trainer(seed=2)
    assert not torch.equal(next(restored._net().parameters()), next(trainer._net().parameters()))
    assert restored.load(str(checkpoint)) == 1
    assert restored.iteration() == 1
    _assert_same_training_state(trainer, restored)


@pytest.mark.L0
@pytest.mark.CPU
def test_sharded_trainer_still_resumes_the_all_consolidated_and_all_dcp_layouts(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoints written before the sharded-optimizer layout, and DCP checkpoints, load unchanged."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    from cosmos_framework.posttrain.checkpoint.io import component_kind, is_complete_iter_dir
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    monkeypatch.setattr(llm_module, "is_dtensor_sharded", lambda _net: True)
    trainer, model = _sharded_layout_trainer(seed=3)
    trainer.train([None])

    # The pre-change layout: `optim.pt` holds what `get_optimizer_state_dict(..., full_state_dict=True)`
    # gathered on rank 0. Rebuild it from the new save by swapping the shards for that file.
    legacy = tmp_path / "legacy" / "iter_000001"
    trainer.save(str(legacy))
    trainer.finalize_checkpoints()
    import shutil

    shutil.rmtree(legacy / "optim")
    gathered = get_optimizer_state_dict(
        model, trainer.opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    torch.save(gathered, legacy / "optim.pt")
    assert component_kind(str(legacy), "optim") == "pt" and is_complete_iter_dir(str(legacy))
    restored, _ = _sharded_layout_trainer(seed=4)
    assert restored.load(str(legacy)) == 1
    _assert_same_training_state(trainer, restored)

    # The all-DCP layout of `checkpoint_format="dcp"` shares the sharded optimizer restore.
    dcp_trainer, _ = _sharded_layout_trainer(seed=5, checkpoint_format="dcp")
    dcp_trainer.train([None])
    dcp = tmp_path / "dcp" / "iter_000001"
    dcp_trainer.save(str(dcp))
    dcp_trainer.finalize_checkpoints()
    assert component_kind(str(dcp), "model") == "dcp" and component_kind(str(dcp), "optim") == "dcp"
    dcp_restored, _ = _sharded_layout_trainer(seed=6, checkpoint_format="dcp")
    assert dcp_restored.load(str(dcp)) == 1
    _assert_same_training_state(dcp_trainer, dcp_restored)


@pytest.mark.L0
@pytest.mark.CPU
def test_ensure_resumable_upgrades_a_model_only_recipe_without_gathering_the_optimizer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RLVR recipe's only full save is the coordinator's emergency `ensure_resumable`.

    `rl.checkpoint_components=[model]` makes every cadence save a model-only artifact; on a
    termination trigger the coordinator requests the full plan. That save must produce a checkpoint
    `rl.resume=auto` adopts, through the same manager the trainer was built with, without a full
    optimizer gather.
    """
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.checkpoint.io import component_kind
    from cosmos_framework.posttrain.control.checkpointing import CheckpointCoordinator
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    monkeypatch.setattr(llm_module, "is_dtensor_sharded", lambda _net: True)
    extractions = _spy_optimizer_extraction(monkeypatch)
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"])
    trainer, _ = _sharded_layout_trainer(seed=7, checkpoint_manager=manager)
    trainer.train([None])
    coordinator = CheckpointCoordinator(out_dir=str(tmp_path), policy=trainer, manager=manager)

    assert coordinator.save_configured_if_needed(1)
    artifact = tmp_path / "checkpoint_artifacts" / "iter_000001"
    assert component_kind(str(artifact), "model") == "pt" and component_kind(str(artifact), "optim") is None
    assert extractions == []
    assert manager.resolve_resume("auto", str(tmp_path / "checkpoints")) is None

    assert coordinator.ensure_resumable(1)
    assert not coordinator.ensure_resumable(1)
    trainer.finalize_checkpoints()
    checkpoint = tmp_path / "checkpoints" / "iter_000001"
    assert extractions and all(options is None or not options.full_state_dict for options in extractions)
    assert component_kind(str(checkpoint), "model") == "pt" and component_kind(str(checkpoint), "optim") == "dcp"
    assert manager.resolve_resume("auto", str(tmp_path / "checkpoints")) == str(checkpoint)

    restored, _ = _sharded_layout_trainer(seed=8, checkpoint_manager=manager)
    assert restored.load(str(checkpoint)) == 1
    _assert_same_training_state(trainer, restored)


def _write_legacy_consolidated_layout(trainer: LLMTrainer, model: torch.nn.Module, checkpoint: Any) -> None:
    """Turn a sharded trainer's save into the legacy layout: `optim/` shards -> gathered `optim.pt`."""
    import shutil

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    trainer.save(str(checkpoint))
    trainer.finalize_checkpoints()
    shutil.rmtree(checkpoint / "optim")
    gathered = get_optimizer_state_dict(
        model, trainer.opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    torch.save(gathered, checkpoint / "optim.pt")


def _tensor_leaves(state: Any) -> list[torch.Tensor]:
    if torch.is_tensor(state):
        return [state]
    if isinstance(state, dict):
        return [leaf for value in state.values() for leaf in _tensor_leaves(value)]
    return []


def _live_optimizer_bytes(trainer: LLMTrainer) -> int:
    return sum(
        value.numel() * value.element_size()
        for per_parameter in trainer.opt.state.values()
        for value in per_parameter.values()
        if torch.is_tensor(value)
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_sharded_trainer_resumes_a_legacy_consolidated_checkpoint_by_copying_only_its_own_shards(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resuming a sharded trainer from an all-.pt checkpoint costs each rank its own shards, not the file.

    Three one-node Qwen3-VL-30B arms adopted their preemption emergency checkpoint (58 GB model.pt +
    223 GB optim.pt) and died in setup on every retry: Ray killed the trainers at 855 GB of a 900 GB
    job cgroup with four `run_setup` workers at ~203 GB each, because the legacy route handed the
    whole mmap-backed dicts to the `full_state_dict` setters, which move every consolidated tensor
    whole to the device and scatter it from rank 0 on EVERY rank. Now: the files are read with
    mmap, the very objects `torch.load` returned reach the chunk copy untouched, the bytes a rank
    materializes equal its live model and optimizer state, no `full_state_dict` setter runs, and the
    restored trainer is in lockstep with the source.
    """
    import torch.distributed.checkpoint.state_dict as state_dict_module

    from cosmos_framework.posttrain.checkpoint.io import component_kind, is_complete_iter_dir
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    monkeypatch.setattr(llm_module, "is_dtensor_sharded", lambda _net: True)
    trainer, model = _sharded_layout_trainer(seed=11)
    trainer.train([None])
    legacy = tmp_path / "legacy" / "iter_000001"
    _write_legacy_consolidated_layout(trainer, model, legacy)
    assert component_kind(str(legacy), "model") == "pt" and component_kind(str(legacy), "optim") == "pt"
    assert is_complete_iter_dir(str(legacy))

    # (1) every .pt of the checkpoint is read through mmap; keep what torch.load handed back.
    loaded_by_file: dict[str, Any] = {}
    real_torch_load = torch.load

    def _load_spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_torch_load(file, *args, **kwargs)
        if isinstance(file, str) and file.startswith(str(legacy)) and file.endswith(".pt"):
            assert kwargs.get("mmap") is True, (file, kwargs)
            loaded_by_file[file.rsplit("/", 1)[-1]] = result
        return result

    monkeypatch.setattr(torch, "load", _load_spy)

    # (2) the full_state_dict setters are the path that put world-size copies on one host.
    full_state_calls: list[str] = []
    for name in ("set_model_state_dict", "set_optimizer_state_dict"):
        real_setter = getattr(state_dict_module, name)

        def _setter_spy(*args: Any, _real: Any = real_setter, _name: str = name, **kwargs: Any) -> Any:
            options = kwargs.get("options")
            if options is not None and options.full_state_dict:
                full_state_calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(state_dict_module, name, _setter_spy)

    # (3) the chunk copy receives torch.load's objects themselves and reports what it materialized.
    fills: dict[str, int] = {}
    real_fill = getattr(llm_module, "fill_sharded_state_from_consolidated", None)

    def _fill_spy(target: Any, consolidated: Any, **kwargs: Any) -> int:
        component = "model" if consolidated is loaded_by_file.get("model.pt", {}).get("net") else None
        component = "optimizer" if consolidated is loaded_by_file.get("optim.pt") else component
        assert component is not None, "the chunk copy did not receive the mmap-loaded dict itself"
        assert _tensor_leaves(consolidated), component
        fills[component] = real_fill(target, consolidated, **kwargs)
        return fills[component]

    monkeypatch.setattr(llm_module, "fill_sharded_state_from_consolidated", _fill_spy, raising=False)

    restored, restored_model = _sharded_layout_trainer(seed=12)
    assert not torch.equal(next(restored_model.parameters()), next(model.parameters()))
    assert restored.load(str(legacy)) == 1

    assert full_state_calls == [], full_state_calls
    assert set(loaded_by_file) == {"model.pt", "optim.pt", "trainer.pt"}, sorted(loaded_by_file)
    live_model_bytes = sum(value.numel() * value.element_size() for value in restored_model.state_dict().values())
    assert fills == {"model": live_model_bytes, "optimizer": _live_optimizer_bytes(restored)}, fills
    _assert_same_training_state(trainer, restored)


@pytest.mark.L0
@pytest.mark.CPU
def test_load_routes_each_checkpoint_layout_to_its_restore_path(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each checkpoint layout a sharded trainer can meet takes exactly one restore route.

    all-.pt (legacy emergency saves): chunk copy for the model AND the optimizer;
    model.pt + optim/ (what `save` writes now): chunk copy for the model, DCP restore for the optimizer;
    all-DCP (`checkpoint_format="dcp"`): `_load_dcp`;
    pristine all-.pt (never-stepped optimizer): chunk copy for the model, group options only, no fill.
    """
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    monkeypatch.setattr(llm_module, "is_dtensor_sharded", lambda _net: True)
    routes: list[str] = []
    for name in (
        "_load_dcp",
        "_restore_sharded_optimizer",
        "_restore_sharded_model_from_consolidated",
        "_restore_sharded_optimizer_from_consolidated",
    ):
        real_method = getattr(LLMTrainer, name)

        def _route_spy(self: Any, *args: Any, _real: Any = real_method, _name: str = name, **kwargs: Any) -> Any:
            routes.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(LLMTrainer, name, _route_spy)

    trainer, model = _sharded_layout_trainer(seed=21)
    trainer.train([None])
    mixed = tmp_path / "mixed" / "iter_000001"
    trainer.save(str(mixed))
    trainer.finalize_checkpoints()
    legacy = tmp_path / "legacy" / "iter_000001"
    _write_legacy_consolidated_layout(trainer, model, legacy)

    dcp_trainer, _ = _sharded_layout_trainer(seed=22, checkpoint_format="dcp")
    dcp_trainer.train([None])
    dcp = tmp_path / "dcp" / "iter_000001"
    dcp_trainer.save(str(dcp))
    dcp_trainer.finalize_checkpoints()

    pristine_trainer, _ = _sharded_layout_trainer(seed=23)
    pristine_trainer.opt.param_groups[0]["lr"] = 0.0125
    pristine = tmp_path / "pristine" / "iter_000000"
    pristine_trainer.save(str(pristine))
    pristine_trainer.finalize_checkpoints()
    import shutil

    shutil.rmtree(pristine / "optim")
    torch.save({"state": {}, "param_groups": pristine_trainer.opt.state_dict()["param_groups"]}, pristine / "optim.pt")

    def _resume(checkpoint: Any, seed: int, **kwargs: Any) -> tuple[list[str], LLMTrainer]:
        routes.clear()
        restored, _ = _sharded_layout_trainer(seed=seed, **kwargs)
        restored.load(str(checkpoint))
        return list(routes), restored

    taken, restored = _resume(legacy, seed=31)
    assert taken == ["_restore_sharded_model_from_consolidated", "_restore_sharded_optimizer_from_consolidated"], taken
    _assert_same_weights_and_moments(trainer, restored)

    taken, restored = _resume(mixed, seed=32)
    assert taken == ["_restore_sharded_model_from_consolidated", "_restore_sharded_optimizer"], taken
    _assert_same_weights_and_moments(trainer, restored)

    taken, restored = _resume(dcp, seed=33, checkpoint_format="dcp")
    assert taken == ["_load_dcp", "_restore_sharded_optimizer"], taken
    _assert_same_weights_and_moments(dcp_trainer, restored)

    taken, restored = _resume(pristine, seed=34)
    assert taken == ["_restore_sharded_model_from_consolidated"], taken
    assert restored.iteration() == 0
    assert restored.opt.state == {}
    assert restored.opt.param_groups[0]["lr"] == 0.0125
    assert all(
        torch.equal(a, b)
        for a, b in zip(pristine_trainer._net().parameters(), restored._net().parameters(), strict=True)
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_zero_gradient_skip_is_disabled_by_default() -> None:
    """The legacy path still calls AdamW.step and keeps its metric/checkpoint schema unchanged."""
    trainer, model, before = _make_zero_grad_trainer(skip_zero_grad=False)

    metrics = trainer.train([None])

    assert metrics["optim/update_outcome"] == "applied"
    assert trainer.iteration() == 1
    assert any(not torch.equal(saved, current) for saved, current in zip(before, model.parameters()))
    assert "optim/zero_grad_skips_total" not in metrics
    state = trainer._trainer_state()
    assert "skip_zero_grad" not in state
    assert "zero_grad_skips_total" not in state


@pytest.mark.L0
@pytest.mark.CPU
def test_zero_gradient_skip_is_opt_in_and_checkpointed(tmp_path: Any) -> None:
    """The AlpaGym opt-in skips AdamW mutation and its attempt clock survives save/load."""
    trainer, model, before = _make_zero_grad_trainer(skip_zero_grad=True)

    metrics = trainer.train([None])

    assert metrics["optim/update_outcome"] == "skipped"
    assert metrics["optim/zero_grad_skips_total"] == 1
    assert trainer.iteration() == 0
    assert all(torch.equal(saved, current) for saved, current in zip(before, model.parameters()))

    checkpoint = tmp_path / "iter_000000"
    trainer.save(str(checkpoint))
    restored, _, _ = _make_zero_grad_trainer(skip_zero_grad=True)
    assert restored.load(str(checkpoint)) == 0
    restored_state = restored._trainer_state()
    assert restored_state["skip_zero_grad"] == 1
    assert restored_state["zero_grad_skips_total"] == 1
    assert restored_state["attempt_id"] == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_zero_gradient_skip_completes_scheduler_batch_across_save_load(tmp_path: Any) -> None:
    """A zero-gradient mini-batch consumes the second slot and resume starts a fresh batch."""
    import torch.nn as nn

    def objective(model: torch.nn.Module, batch: list[float]) -> tuple[torch.Tensor, dict[str, float]]:
        loss = sum(parameter.sum() for parameter in model.parameters()) * batch[0]  # []
        return loss, {"loss": float(loss.detach())}

    def make_trainer() -> LLMTrainer:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return LLMTrainer(
            model,
            Objective(objective, reduction=Reduction.MEAN),
            optimizer,
            n_mini=2,
            max_grad_norm=1.0,
            skip_zero_grad=True,
            lr_scheduler=scheduler,
        )

    trainer = make_trainer()
    assert trainer.train([1.0])["optim/update_outcome"] == "applied"
    assert trainer.train([0.0])["optim/update_outcome"] == "skipped"
    assert trainer._sched_accum == 0
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(0.05)

    checkpoint = tmp_path / "iter_000001"
    trainer.save(str(checkpoint))
    restored = make_trainer()
    assert restored.load(str(checkpoint)) == 1
    assert restored._sched_accum == 0

    assert restored.train([1.0])["optim/update_outcome"] == "applied"
    assert restored._sched_accum == 1
    assert restored.opt.param_groups[0]["lr"] == pytest.approx(0.05)


@pytest.mark.L0
@pytest.mark.CPU
def test_nonfinite_skip_does_not_advance_scheduler_during_restore() -> None:
    """Resume excludes rejected non-finite attempts from both scheduler clocks."""
    import torch.nn as nn

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = LLMTrainer(
        model,
        Objective(_zero_grad_objective, reduction=Reduction.MEAN),
        optimizer,
        n_mini=2,
        lr_scheduler=scheduler,
    )
    trainer._restore_trainer_state_local(
        {
            "iter": 1,
            "optimizer_step": 1,
            "accum": 0,
            "n_mini": 2,
            "attempt_id": 2,
            "nonfinite_skips_total": 1,
            "nonfinite_skips_consecutive": 1,
        }
    )

    assert trainer._sched_accum == 1
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(0.1)


def _nan_grad_step_state(force_finite_grads: bool) -> dict[str, Any]:
    """One train step with a FINITE-loss / NaN-backward-grad objective (a custom autograd.Function) —
    the loop's post-step loss check can't see this (the loss is finite), only pre-step grad hygiene can.
    AdamW makes optimizer-moment mutation visible."""
    import torch.nn as nn

    class _NanGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.sum()  # finite forward loss

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return torch.full_like(x, float("nan"))  # NaN gradient — poisons the step if it reaches it

    torch.manual_seed(0)
    m = nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1.0)
    before = [parameter.detach().clone() for parameter in m.parameters()]

    def _fn(model, batch):
        loss = _NanGrad.apply(next(model.parameters()))
        return loss, {"loss": float(loss.detach())}

    obj = Objective(_fn, reduction=Reduction.MEAN)  # the ctor validates .reduction against grad_accum
    trainer = LLMTrainer(m, obj, opt, force_finite_grads=force_finite_grads)
    metrics = trainer.train([None])
    params_finite = all(bool(torch.isfinite(p).all()) for p in m.parameters())
    moments = [s[k] for s in opt.state_dict()["state"].values() for k in ("exp_avg", "exp_avg_sq") if k in s]
    moments_finite = all(bool(torch.isfinite(v).all()) for v in moments)
    return {
        "metrics": metrics,
        "params_finite": params_finite,
        "params_unchanged": all(torch.equal(saved, current) for saved, current in zip(before, m.parameters())),
        "moments_finite": moments_finite,
        "optimizer_state_created": bool(opt.state_dict()["state"]),
        "iteration": trainer.iteration(),
    }


# ---------------------------------------------------------------------------
# Objective-owned token_sum reduction (algorithm/primitives/objective.py): the objective normalizes by its own
# DP-global token count; the trainer backprops it as-is (no /grad_accum) and refuses grad_accum>1.
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_trainer_token_sum_steps():
    """A token_sum objective normalizes internally; the trainer backprops it as-is and the optimizer
    steps."""
    model, tok = _tiny()
    trainer = LLMTrainer(model, make_objective(), torch.optim.SGD(model.parameters(), lr=1.0))
    batch = _with_advantages([_traj(tok, "g0", 1.0, completion=" 2"), _traj(tok, "g0", 0.0, completion=" 3 4 5")])
    before = next(model.parameters()).detach().clone()
    metrics = trainer.train(batch)
    assert metrics["train/valid_token_count"] > 0
    assert not torch.allclose(before, next(model.parameters()))  # stepped


@pytest.mark.L0
@pytest.mark.CPU
def test_lr_schedule_advances_on_the_collected_batch_clock():
    """cosmos-rl steps its LR schedule once per COLLECTED BATCH, so with n_mini optimizer steps per
    batch the whole batch trains at one LR. Stepping every optimizer step instead advances warmup
    n_mini x too fast -- the divergence this port closes. n_mini=2, grad_accum=1: four committed
    optimizer steps must tick the schedule EXACTLY twice."""
    from cosmos_framework.posttrain.config.build import build_lr_scheduler
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    model, tok = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = build_lr_scheduler(opt, OptimizerConfig(warmup_steps=10))
    ticks = {"n": 0}
    inner_step = sched.step  # patch AFTER construction so the scheduler's own init step isn't counted

    def counting_step(*args, **kwargs):
        ticks["n"] += 1
        return inner_step(*args, **kwargs)

    sched.step = counting_step
    trainer = LLMTrainer(model, make_objective(), opt, n_mini=2, lr_scheduler=sched)
    batch = _with_advantages([_traj(tok, "g0", 1.0, completion=" 2"), _traj(tok, "g0", 0.0, completion=" 3 4 5")])
    for _ in range(4):  # grad_accum=1 -> each train() commits one optimizer step
        trainer.train(batch)
    assert ticks["n"] == 2, (
        f"schedule ticked {ticks['n']}x over 4 optimizer steps at n_mini=2; expected 2 (collected-batch clock)"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_trainer_token_sum_grad_accum_gt_1_steps_with_deferred_denom():
    """token_sum + grad_accum>1 now completes end to end: `train()` forwards the objective's own
    divisor (`metrics[OBJECTIVE_WEIGHT_SUM]`) into `backward`, which recovers each chunk's unnormalised
    SUM, and the trainer defers the token-mean division to the optimizer step (accumulating Σ
    denom across the window in `_denom_accum`, spec §3.5) instead of raising. Two micro-batches
    of one trajectory each accumulate into ONE optimizer step."""
    model, tok = _tiny()
    trainer = LLMTrainer(model, make_objective(), torch.optim.SGD(model.parameters(), lr=1.0), grad_accum=2)
    batch = _with_advantages([_traj(tok, "g0", 1.0, completion=" 2"), _traj(tok, "g0", 0.0, completion=" 3 4 5")])
    before = next(model.parameters()).detach().clone()
    accumulating = trainer.train(batch[:1])
    assert accumulating["optim/update_outcome"] == "accumulating"
    applied = trainer.train(batch[1:])
    assert applied["optim/update_outcome"] == "applied"
    assert not torch.allclose(before, next(model.parameters()))  # optimizer actually stepped


@pytest.mark.L0
@pytest.mark.CPU
def test_reduce_backward_token_sum_backstop_still_raises():
    """`backward` keeps its own raise, so a caller bypassing __init__ still can't normalize per
    micro-batch without the objective's divisor."""
    from cosmos_framework.posttrain.algorithm.primitives.objective import Reduction
    from cosmos_framework.posttrain.algorithm.primitives.objective import backward as reduce_backward

    loss = torch.zeros((), requires_grad=True)
    with pytest.raises(NotImplementedError, match="objective weight sum"):
        reduce_backward(loss, Reduction.TOKEN_SUM, grad_accum=2)


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_trainer_force_finite_grads_guards_optimizer():
    """The shared flag has the same meaning as DiffusionTrainer: reject without stepping.

    The explicit opt-out disables only the gradient check, so the deliberately bad
    gradient reaches AdamW and demonstrates that the setting has real effect.
    """
    checked = _nan_grad_step_state(force_finite_grads=True)
    assert checked["metrics"]["optim/update_outcome"] == "skipped"
    assert checked["metrics"]["optim/skip_reason"] == "nonfinite_gradient"
    assert checked["params_finite"] and checked["params_unchanged"]
    assert not checked["optimizer_state_created"]
    assert checked["iteration"] == 0

    unchecked = _nan_grad_step_state(force_finite_grads=False)
    assert unchecked["metrics"]["optim/update_outcome"] == "applied"
    assert unchecked["metrics"]["optim/finite_gradient_check_enabled"] == 0.0
    assert unchecked["optimizer_state_created"]
    assert not (unchecked["params_finite"] and unchecked["moments_finite"])
    assert unchecked["iteration"] == 1


@pytest.mark.L1
@pytest.mark.GPU
def test_llm_trainer_dcp_round_trip_fsdp(tmp_path):
    """FSDP2 DCP save→load round-trip: shard a TinyDecoder-backed trainer, take a step,
    save "dcp", then restore into a fresh sharded trainer and assert iter + full weights
    match. Runs on the cluster (>=2 GPUs); DCP reshards natively via the live state dicts."""
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan
    from cosmos_framework.posttrain.tests.helpers.tiny_decoder import Block, TinyDecoder

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    torch.cuda.set_device(dist.get_rank())

    def _sharded_trainer():
        m = TinyDecoder().cuda()
        pd = ParallelDims(world_size=world, dp_shard=world)
        pd.build_device_mesh("cuda")
        parallelize(m, ShardPlan((Block,), torch.bfloat16, torch.float32), pd)
        opt = torch.optim.SGD(m.parameters(), lr=1.0)

        def _obj(model, batch):
            x = torch.ones(2, 16, device="cuda")
            loss = model(x).sum()
            return loss, {"loss": float(loss.detach())}

        return LLMTrainer(m, Objective(_obj, reduction=Reduction.MEAN), opt, checkpoint_format="dcp"), m

    t1, m1 = _sharded_trainer()
    t1.train([None])
    # torchrun gives each rank its own tmp_path; broadcast rank 0's so the COLLECTIVE DCP
    # save/load uses ONE shared dir (single node -> shared /tmp). Otherwise rank 1 looks for
    # rank 0's .metadata in its own tmp_path and fails with FileNotFoundError.
    _obj = [str(tmp_path / "iter_000001")] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(_obj, src=0)
    ckpt = _obj[0]
    t1.save(ckpt)
    t1.finalize_checkpoints()
    assert t1.iteration() == 1

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    ref = {k: v.clone() for k, v in get_model_state_dict(m1, options=opts).items()}

    t2, m2 = _sharded_trainer()
    assert t2.load(ckpt) == 1
    assert t2.iteration() == 1
    got = get_model_state_dict(m2, options=opts)
    for k in ref:
        assert torch.allclose(got[k], ref[k]), f"param {k} mismatch after DCP round-trip"


# ---------------------------------------------------------------------------
# `_reduce_metrics` cross-rank reduction. It preserves public counts in an int64
# SUM and packs the remaining numerics into one float SUM (NCCL-only), so this must
# be a real multi-rank NCCL group. Verifies: two independent weighted metrics,
# ordinary scalar/tensor means, count sums, pass-through values, and sorted pack order.
# ---------------------------------------------------------------------------


def _reduce_metrics_worker(rank: int, world: int, port: int) -> None:
    import os

    import torch.distributed as dist
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    try:
        m = nn.Linear(2, 2).cuda()
        # A bare lambda here raised `AttributeError: 'function' object has no attribute 'reduction'`
        # inside LLMTrainer.__init__ -- this test has never reached its own assertions. The trainer
        # takes an `Objective` (callable + declared Reduction); the reduction is irrelevant to what
        # is under test (metric packing), so declare the simple one.
        trainer = LLMTrainer(
            m,
            Objective(
                fn=lambda model, batch: (torch.zeros((), device="cuda"), {}),
                reduction=Reduction.MEAN,
                metric_adapter=MetricAdapter(
                    internal_prefix="example_",
                    names={
                        "example_loss": "loss_avg",
                        "example_weight": "weight_sum",
                        "example_diag": "diag_avg",
                        "example_n": "valid_sequence_count",
                        "example_tokens": "valid_token_count",
                    },
                    phases=frozenset({"train"}),
                    weighted_by={
                        "example_loss": "example_weight",
                        "example_diag": "example_tokens",
                    },
                ),
            ),
            torch.optim.SGD(m.parameters(), lr=0.1),
        )
        assert trainer.training_metric_weight_map() == {
            "train/loss_avg": "train/weight_sum",
            "train/diag_avg": "train/valid_token_count",
        }
        # Rank 0 is an EMPTY contribution to both weighted metrics; rank 1 is non-empty. Keys are
        # deliberately NOT pre-sorted so the sorted() pack order keeps the two ranks aligned.
        large_exact_count = 2**24 + 1  # float32 cannot represent this integer; int64 SUM must.
        metrics = {
            "z_scalar": float(rank),  # mean(0, 1) == 0.5
            "a_int": rank * 10,  # int -> float; mean(0, 10) == 5.0
            "m_vec": torch.tensor([float(rank), float(rank) + 1.0]),  # mean -> [0.5, 1.5]
            "example_loss": 0.0 if rank == 0 else 2.0,
            "example_weight": 0.0 if rank == 0 else 4.0,
            "example_diag": 0.0 if rank == 0 else 3.0,
            "example_n": 0 if rank == 0 else 2,
            "example_tokens": 0 if rank == 0 else large_exact_count,
            "label": "keep",  # non-numeric -> pass-through
            "flag": True,  # bool is NOT numeric here -> pass-through, unreduced
        }
        out = trainer._reduce_metrics(metrics)
        assert isinstance(out["z_scalar"], float) and abs(out["z_scalar"] - 0.5) < 1e-5, out["z_scalar"]
        assert isinstance(out["a_int"], float) and abs(out["a_int"] - 5.0) < 1e-5, out["a_int"]
        assert torch.allclose(out["m_vec"].cpu(), torch.tensor([0.5, 1.5])), out["m_vec"]
        assert out["train/loss_avg"] == pytest.approx(2.0)
        assert out["train/weight_sum"] == pytest.approx(2.0)  # public divisor remains the DP mean
        assert out["train/diag_avg"] == pytest.approx(3.0)
        assert out["train/valid_sequence_count"] == 2
        assert out["train/valid_token_count"] == large_exact_count
        assert out["label"] == "keep"
        assert out["flag"] is True  # bool passed through, not averaged to 0.5

        invalid = dict(metrics)
        if rank == 0:
            invalid["example_loss"] = 1.0  # forbidden: this rank's example_weight is zero
        with pytest.raises(RuntimeError, match="must be zero when 'train/weight_sum' is zero"):
            trainer._reduce_metrics(invalid)
    finally:
        dist.destroy_process_group()


@pytest.mark.L1
@pytest.mark.GPU
def test_reduce_metrics_single_collective_cross_rank_mean():
    """Two real NCCL ranks: ordinary means and two independently weighted metrics (including one
    empty rank) share one float pack, while public counts retain an exact int64 SUM. An assert
    failing in any worker propagates as a ProcessRaisedException and fails the test."""
    import socket

    import torch.multiprocessing as mp

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    world = 2
    with socket.socket() as s:  # grab a free port for the rendezvous
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(_reduce_metrics_worker, args=(world, port), nprocs=world, join=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_get_bucket_plan_lazy_cached_and_requires_backend():
    # Weight-channel plan negotiation (peer-pull design §1.3): the policy computes its plan lazily
    # from its OWN manifest, caches it on the backend (same object returned thereafter), and refuses
    # without a weight backend (a plan is meaningless without a weight edge).
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync

    model = nn.Linear(8, 8)
    objective = Objective(lambda *a, **k: None, reduction=Reduction.MEAN)  # never called; ctor reads .reduction
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=0.1))
    with pytest.raises(ValueError):
        trainer.get_bucket_plan()  # no backend bound

    trainer._weight_backend = WeightSync(CopyMover())
    plan = trainer.get_bucket_plan(bucket_bytes=64)  # weight + bias split into >=2 buckets
    assert len(plan) >= 2
    assert trainer.get_bucket_plan() is plan  # cached: second call returns the same object
    # push_bucket needs no pre-set plan either (lazy path) — and stamps the TRAINING ITERATION
    # (the version is _iter; push reads it, never advances it)
    trainer._iter = 2
    payload = trainer.push_bucket(0)
    assert payload is not None and payload.version == 2
    assert trainer._iter == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_lr_scheduler_state_survives_save_load(tmp_path):
    """The warmup scheduler is STATEFUL, so it must ride in the checkpoint. Without this a resumed
    run replays the ramp from zero and drops back to the early low LR — silently re-annealing
    weights already past warmup. Matters here specifically: the entrypoint resumes by default and
    this workload gets preempted."""
    import torch

    from cosmos_framework.posttrain.config.build import build_lr_scheduler
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    model, _ = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = build_lr_scheduler(opt, OptimizerConfig(warmup_steps=10))
    trainer = LLMTrainer(model, make_objective(init_data_packer("gpt2")), opt, lr_scheduler=sched)
    for _ in range(4):  # advance partway INTO the ramp
        sched.step()
    lr_before = opt.param_groups[0]["lr"]
    ckpt = str(tmp_path / "iter_000001")
    trainer.save(ckpt)

    model2, _ = _tiny()
    opt2 = torch.optim.SGD(model2.parameters(), lr=1.0)
    sched2 = build_lr_scheduler(opt2, OptimizerConfig(warmup_steps=10))
    t2 = LLMTrainer(model2, make_objective(init_data_packer("gpt2")), opt2, lr_scheduler=sched2)
    t2.load(ckpt)
    assert opt2.param_groups[0]["lr"] == pytest.approx(lr_before), "warmup restarted on resume"


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_identity_is_durable_and_rejects_changed_loss_semantics() -> None:
    model, _ = _tiny()
    objective = make_objective()
    objective.identity = {
        "schema_version": 1,
        "algorithm": "grpo",
        "surrogate": "legacy_pg",
        "temperature": 0.9,
    }
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=1.0))
    state = trainer._trainer_state()
    assert state["objective_identity"] == objective.identity

    changed_model, _ = _tiny()
    changed_objective = make_objective()
    changed_objective.identity = {**objective.identity, "temperature": 1.0}
    changed = LLMTrainer(
        changed_model,
        changed_objective,
        torch.optim.SGD(changed_model.parameters(), lr=1.0),
    )
    with pytest.raises(RuntimeError, match="objective exact-resume identity differs"):
        changed._restore_trainer_state_local(state)

    legacy = dict(state)
    legacy.pop("objective_identity")
    with pytest.raises(RuntimeError, match="predates the objective exact-resume identity"):
        trainer._restore_trainer_state_local(legacy)


@pytest.mark.L0
@pytest.mark.CPU
def test_resume_refuses_a_changed_mini_batch_shape(tmp_path):
    """The driver resumes by dividing the durable OPTIMIZER-step count by `n_mini` to recover its own
    COLLECTED-BATCH step. Under a different `n_mini` that division lands on a silently wrong batch —
    a smaller value restarts too far along and skips data, a larger one re-runs it — and the stored
    count alone cannot reveal either. So the checkpoint carries the shape and restore compares it."""
    model, tok = _tiny()
    trainer = LLMTrainer(
        model, make_objective(init_data_packer("gpt2")), torch.optim.SGD(model.parameters(), lr=1.0), n_mini=2
    )
    batch = _with_advantages([_traj(tok, "g0", 1.0, completion=" 2"), _traj(tok, "g0", 0.0, completion=" 9")])
    trainer.train(batch)
    with pytest.raises(RuntimeError, match="collected-batch boundary"):
        trainer.save(str(tmp_path / "checkpoints" / "iter_000001"))
    trainer.train(batch)
    ckpt = str(tmp_path / "checkpoints" / "iter_000002")
    trainer.save(ckpt)

    same, _ = _tiny()
    ok = LLMTrainer(
        same, make_objective(init_data_packer("gpt2")), torch.optim.SGD(same.parameters(), lr=1.0), n_mini=2
    )
    assert ok.load(ckpt) == 2  # same shape resumes normally

    other, _ = _tiny()
    changed = LLMTrainer(
        other, make_objective(init_data_packer("gpt2")), torch.optim.SGD(other.parameters(), lr=1.0), n_mini=4
    )
    with pytest.raises(RuntimeError, match="n_mini"):
        changed.load(ckpt)


# --- π_old ahead pass (record/replay window; algorithm/grpo.py::OldLogprobCache) -----------------


def _deterministic_tiny():
    """Same tiny GPT2, but with every dropout OFF.

    The ahead pass refuses to run on a stochastic forward (a dropout mask resampled between the
    π_old pass and training would make the ratio noise rather than a policy delta), so an
    ahead-pass test must use a model that actually satisfies that precondition.
    """
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=50257,
        n_positions=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_first_dropout=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return model, tok


def _clipped(packer):
    """The clipped objective — the one that consults a frozen π_old and therefore carries a cache.

    Wide epsilons and kl_beta=0 keep this test about the record/replay plumbing rather than about
    where the clip happens to bind.
    """
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    cfg = GRPOConfig(epsilon_low=10.0, epsilon_high=10.0, kl_beta=0.0, temperature=1.0, group_size=4)
    return make_clipped_objective(packer, cfg)


def _four_chunks(tok):
    """One group of four trajectories, split one-per-chunk — the n_mini=2 x n_micro=2 walk."""
    trajs = _with_advantages(
        [
            _traj(tok, "g0", 1.0, completion=" 2"),
            _traj(tok, "g0", 0.0, completion=" 3 4 5"),
            _traj(tok, "g0", 1.0, completion=" 2 2"),
            _traj(tok, "g0", 0.0, completion=" 7"),
        ]
    )
    return [trajs[i : i + 1] for i in range(4)]


@pytest.mark.L0
@pytest.mark.CPU
def test_ahead_pass_records_once_then_every_train_call_replays():
    """n_mini=2 x n_micro=2: record the whole window ONCE, then run all four train calls.

    Regression test for the transition living in `train()`. When `train` performed the
    record -> replay flip, the first chunk succeeded and the second saw mode "replay", read it as
    a missed ahead pass, and aborted the batch — so no n_mini > 1 run could get past its second
    chunk. The transition now happens exactly once, in `begin_replay_window()`, and `train` only
    consumes. All four calls must complete and the two optimizer steps must land.
    """
    model, tok = _deterministic_tiny()
    objective = _clipped(init_data_packer("gpt2"))
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=1.0), grad_accum=2, n_mini=2)
    chunks = _four_chunks(tok)
    cache = objective.old_logp_cache
    assert cache is not None

    for chunk in chunks:
        trainer.compute_old_logprobs(chunk)
    assert cache.mode == "record"
    assert cache.recorded == 4  # n_mini * n_micro, one record per forward chunk

    assert trainer.begin_replay_window()["old_logp/window"] == 4
    assert cache.mode == "replay"

    before = next(model.parameters()).detach().clone()
    outcomes = [trainer.train(chunk)["optim/update_outcome"] for chunk in chunks]
    assert outcomes == ["accumulating", "applied", "accumulating", "applied"]
    assert cache.mode == "replay"  # stays replayed for the whole batch; never re-validated per call
    assert not torch.allclose(before, next(model.parameters()))


@pytest.mark.L0
@pytest.mark.CPU
def test_sequence_level_replay_reports_a_unit_ratio_until_the_policy_moves():
    """Sequence-level (GSPO) record/replay at n_mini=2 x n_micro=2: the response ratio is 1 on the
    first chunk (weights unmoved since the ahead pass, so nothing clips; 1e-6 is far below the
    ln(1+4e-4) band) and leaves 1 only after the first optimizer step moves the policy against the
    frozen pi_old."""
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    cfg = GRPOConfig(
        importance_sampling_level="sequence",
        lower_bound_ratio=0.0,  # sequence level refuses the dual clip
        epsilon_low=3e-4,
        epsilon_high=4e-4,
        kl_beta=0.0,
        temperature=1.0,
        group_size=4,
        batch_per_replica=4,
        mini_batch_per_replica=2,  # n_mini=2; the attestation holds for `_deterministic_tiny`
        certified_deterministic_forward=True,
    )
    model, tok = _deterministic_tiny()
    objective = make_clipped_objective(init_data_packer("gpt2"), cfg)
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=1.0), grad_accum=2, n_mini=2)
    chunks = _four_chunks(tok)
    for chunk in chunks:
        trainer.compute_old_logprobs(chunk)
    assert trainer.begin_replay_window()["old_logp/window"] == 4

    first = trainer.train(chunks[0])
    assert first["optim/update_outcome"] == "accumulating"
    assert first["train/seq_clip_frac"] == 0.0
    assert first["train/seq_log_ratio_abs"] <= 1e-6
    assert first["train/ratio_absdev"] <= 1e-6

    assert trainer.train(chunks[1])["optim/update_outcome"] == "applied"  # the first optimizer step lands
    third = trainer.train(chunks[2])
    assert third["optim/update_outcome"] == "accumulating"
    assert third["train/seq_log_ratio_abs"] > 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_begin_replay_window_refuses_a_cell_that_missed_the_ahead_pass():
    """A policy cell that never recorded would train against `cur.detach()` while its peers score
    against a frozen reference. That is silent surrogate corruption, so it must be fatal."""
    model, tok = _deterministic_tiny()
    objective = _clipped(init_data_packer("gpt2"))
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=1.0), grad_accum=2, n_mini=2)
    with pytest.raises(RuntimeError, match="missed the ahead pass"):
        trainer.begin_replay_window()


@pytest.mark.L0
@pytest.mark.CPU
def test_begin_replay_window_accepts_a_short_window_from_empty_shards():
    """FEWER records than chunks is legal and must NOT raise.

    Under DP, `select_chunk` may hand a rank an empty shard for some chunk. The objective returns
    from its empty-shard path before reaching π_old, so that chunk records nothing — and skips
    identically in the train pass, keeping order-keyed replay aligned. Demanding exactly
    `n_mini * n_micro` records made every such distributed run abort before its first update.
    """
    model, tok = _deterministic_tiny()
    objective = _clipped(init_data_packer("gpt2"))
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=1.0), grad_accum=2, n_mini=2)
    chunks = _four_chunks(tok)
    recorded = [chunks[0], chunks[2]]  # chunks 1 and 3 stand in for empty shards: nothing recorded
    for chunk in recorded:
        trainer.compute_old_logprobs(chunk)
    assert trainer.begin_replay_window()["old_logp/window"] == 2
    assert objective.old_logp_cache.mode == "replay"
    # and the short window replays cleanly for exactly the chunks that produced it
    for chunk in recorded:
        trainer.train(chunk)


@pytest.mark.L0
@pytest.mark.CPU
def test_begin_replay_window_refuses_a_window_longer_than_the_train_walk():
    """MORE records than chunks means the ahead loop ran more times than training will — a driver
    bug, and the one direction that cannot be explained by an empty shard."""
    model, tok = _deterministic_tiny()
    objective = _clipped(init_data_packer("gpt2"))
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=1.0), grad_accum=1, n_mini=2)
    for chunk in _four_chunks(tok):  # 4 records against a 2-chunk walk
        trainer.compute_old_logprobs(chunk)
    with pytest.raises(RuntimeError, match="holds 4 records but the driver walks only 2"):
        trainer.begin_replay_window()


@pytest.mark.L0
@pytest.mark.CPU
def test_begin_replay_window_is_inert_at_n_mini_1():
    """At n_mini == 1 there is no ahead pass and no window; the call must be a no-op, not a raise."""
    model, _ = _deterministic_tiny()
    trainer = LLMTrainer(model, _clipped(init_data_packer("gpt2")), torch.optim.SGD(model.parameters(), lr=1.0))
    assert trainer.begin_replay_window() == {"old_logp/window": 0}


@pytest.mark.L0
@pytest.mark.CPU
def test_replayed_pi_old_stays_frozen_while_the_policy_moves():
    """The recorded π_old must NOT track the weights as the optimizer-step window advances.

    This is the property the whole ahead pass exists for, and no other test asserts it: the existing
    ones check cache mechanics (record/replay/exhaustion) rather than that the replayed VALUES stay
    put while the model changes underneath them. If π_old drifted with the policy, the ratio would
    sit near 1 for the whole window, the ε-clip would never engage, and the run would take N
    unconstrained updates — the exact failure the ahead pass was built to remove, while every
    mechanical assertion still passed.
    """
    model, tok = _deterministic_tiny()
    objective = _clipped(init_data_packer("gpt2"))
    # lr large enough that one step visibly moves the logprobs
    trainer = LLMTrainer(model, objective, torch.optim.SGD(model.parameters(), lr=10.0), grad_accum=1, n_mini=2)
    chunks = _four_chunks(tok)[:2]
    cache = objective.old_logp_cache

    for chunk in chunks:
        trainer.compute_old_logprobs(chunk)
    recorded = [t.detach().clone() for t in cache._store]
    trainer.begin_replay_window()

    before = next(model.parameters()).detach().clone()
    trainer.train(chunks[0])  # weights move here
    # Without this the assertion below would pass vacuously: if the optimizer never moved the
    # policy, a π_old that DID track the weights would still look frozen.
    assert not torch.allclose(before, next(model.parameters())), (
        "the optimizer step did not move the policy, so this test cannot prove anything"
    )

    # ...and the SECOND chunk still replays the ORIGINAL recording, not a value recomputed under the
    # updated weights.
    replayed = cache.old_for(torch.zeros_like(recorded[1]))
    assert torch.equal(replayed, recorded[1]), "π_old drifted with the policy instead of staying frozen"


# ---------------------------------------------------------------------------
# `_reduce_metrics` MAX path: metrics the adapter declares `max_reduced` ride their own
# all_reduce(MAX) instead of the SUM pack, so a per-rank maximum comes back as the maximum over
# ranks -- not a mean (86.75) or a sum (173.5). Two real Gloo ranks on CPU.
# ---------------------------------------------------------------------------


def _reduce_metrics_max_worker(rank: int, world_size: int, rendezvous_path: str, output_queue: Any) -> None:
    import traceback

    import torch.distributed as dist
    import torch.nn as nn

    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group

    try:
        init_gloo_process_group(rank, world_size, rendezvous_path, timeout_seconds=20.0)
        m = nn.Linear(2, 2)
        trainer = LLMTrainer(
            m,
            Objective(
                fn=lambda model, batch: (torch.zeros(()), {}),
                reduction=Reduction.MEAN,
                metric_adapter=MetricAdapter(
                    internal_prefix="example_",
                    names={
                        "example_loss": "loss_avg",
                        "example_weight": "weight_sum",
                        "example_peak": "kl_max",
                        "example_tokens": "valid_token_count",
                    },
                    phases=frozenset({"train"}),
                    weighted_by={"example_loss": "example_weight"},
                    max_reduced=frozenset({"example_peak"}),
                ),
            ),
            torch.optim.SGD(m.parameters(), lr=0.1),
        )
        metrics = {
            "example_loss": 1.0 + rank,  # weighted mean over equal weights -> 1.5
            "example_weight": 2.0,
            "example_peak": 0.5 if rank == 0 else 173.0,  # MAX -> 173.0
            "example_tokens": 3 + rank,  # int64 SUM -> 7
            "plain": float(rank),  # plain mean -> 0.5
            "peak_vec": torch.tensor([float(rank), 10.0 - rank]),  # plain mean -> [0.5, 9.5]
        }
        out = trainer._reduce_metrics(metrics)
        output_queue.put(
            (
                rank,
                {
                    "max_keys": sorted(trainer.training_metric_max_keys()),
                    "kl_max": out["train/kl_max"],
                    "loss": out["train/loss_avg"],
                    "plain": out["plain"],
                    "peak_vec": out["peak_vec"].tolist(),
                    "tokens": out["train/valid_token_count"],
                },
            )
        )
    except BaseException:  # noqa: BLE001 -- the parent must receive a traceback instead of hanging
        output_queue.put((rank, {"worker_error": traceback.format_exc()}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
def test_reduce_metrics_max_reduced_keys_take_the_cross_rank_max_on_gloo(tmp_path) -> None:
    """Two real Gloo ranks: the declared maximum is the cross-rank MAX while the weighted mean, the
    plain means and the int64 count keep their reductions and their pack order."""
    from cosmos_framework.posttrain.tests.helpers.distributed import run_spawned_ranks

    results = run_spawned_ranks(
        _reduce_metrics_max_worker,
        world_size=2,
        rendezvous_path=tmp_path / "reduce-metrics-max.rendezvous",
        timeout_seconds=90.0,
        context="max-reduced metric reduction",
    )
    for rank, result in results.items():
        assert result["max_keys"] == ["train/kl_max"], (rank, result)
        assert result["kl_max"] == 173.0, (rank, result)
        assert result["loss"] == pytest.approx(1.5), (rank, result)
        assert result["plain"] == pytest.approx(0.5), (rank, result)
        assert result["peak_vec"] == pytest.approx([0.5, 9.5]), (rank, result)
        assert result["tokens"] == 7, (rank, result)


@pytest.mark.L0
@pytest.mark.CPU
def test_trainer_host_memory_lines_come_from_the_shared_reader_and_formatter(monkeypatch) -> None:
    """The pre-gather line of a consolidated save and the consolidated-resume line carry the ONE host-memory
    reading (`utils.host_memory`: RSS, cgroup usage against its limit, node meminfo) in the one format the
    checkpoint manager logs at every save and finalize. Before, both lines were built from
    `residency.host_memory_gib()`, a second `/proc` reader that knew nothing of the cgroup and formatted its
    own `host rss=... GiB, available=... GiB of ... GiB` tail, so a fixed reading planted in the shared
    reader never reached them."""
    import logging

    from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan
    from cosmos_framework.posttrain.utils import host_memory as host_memory_module
    from cosmos_framework.posttrain.utils.host_memory import HostMemory
    from cosmos_framework.posttrain.utils.log import logger as posttrain_logger

    fixed = HostMemory(2.0, 880.6, 900.0, "v2", 920.0, 40.0)
    monkeypatch.setattr(host_memory_module, "host_memory", lambda: fixed)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    posttrain_logger.addHandler(handler)
    try:
        LLMTrainer._log_host_memory(
            "gathering the consolidated model on rank 0", CheckpointComponentPlan.from_value(None)
        )
        LLMTrainer._log_consolidated_restore("model", 3 * 2**30, "/runs/x/checkpoints/iter_000010")
    finally:
        posttrain_logger.removeHandler(handler)

    messages = [(record.levelno, record.getMessage()) for record in records]
    assert messages == [
        (
            logging.INFO,
            "host memory [checkpoint save: gathering the consolidated model on rank 0 "
            f"(components=['model', 'optimizer', 'training_state'])] {fixed.format()}",
        ),
        (
            logging.INFO,
            "host memory [checkpoint resume: copied 3.00 GiB of this rank's model shards out of the consolidated "
            f"files of /runs/x/checkpoints/iter_000010] {fixed.format()}",
        ),
    ], messages
    assert fixed.format() == "rss=2.0GiB cgroup(v2)=880.6/900.0GiB (97.8%) node_available=40.0/920.0GiB"
