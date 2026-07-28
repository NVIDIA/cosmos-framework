# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the distillation checkpointer ModelWrapper fake-score handling."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

import cosmos_framework.checkpoint.dcp_distill as dcp_distill_module
from cosmos_framework.checkpoint.dcp import AsyncMode
from cosmos_framework.checkpoint.dcp_distill import (
    DistributedCheckpointer,
    ModelWrapper,
    OptimizerWrapper,
)


class _ListWrappedFakeScoreModel(nn.Module):
    """Mimics SF-DMD: a registered student ``net`` and an unregistered fake-score net."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 2)  # registered student
        self._fake_score_holder = [nn.Linear(2, 2)]  # held outside the registry

    @property
    def net_fake_score(self) -> nn.Module:
        return self._fake_score_holder[0]


class _RegisteredFakeScoreModel(nn.Module):
    """Legacy compatibility case: ``net_fake_score`` is a registered submodule."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 2)
        self.net_fake_score = nn.Linear(2, 2)


class _StrictRejectingModel(nn.Module):
    """Mimics OmniMoTModel: ``load_state_dict`` rejects strict=True.

    The fake-score net is held outside the registry and loaded explicitly, and the
    model-level load (reached via ``set_model_state_dict`` during resume) must call
    ``super().load_state_dict`` with ``strict=False``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 2)
        self._fake_score_holder = [nn.Linear(2, 2)]

    @property
    def net_fake_score(self) -> nn.Module:
        return self._fake_score_holder[0]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        if strict:
            raise ValueError("Strict mode is not supported")
        return super().load_state_dict(state_dict, strict=False, assign=assign)


class _SaveContractModel(_ListWrappedFakeScoreModel):
    """Minimal multi-network model exposing the checkpointer's phase map."""

    def __init__(self) -> None:
        super().__init__()
        self.net_teacher = nn.Linear(2, 2)

    def model_dict(self) -> dict[str, nn.Module]:
        return {"net": self.net, "fake_score": self.net_fake_score}


@pytest.mark.L0
@pytest.mark.CPU
def test_public_checkpointer_exports_are_explicit() -> None:
    assert dcp_distill_module.__all__ == (
        "DistributedCheckpointer",
        "ModelWrapper",
        "OptimizerWrapper",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_optimizer_wrapper_accepts_structurally_compatible_foreign_container() -> None:
    state_dict = {"foreign": True}
    foreign_container = SimpleNamespace(
        model=nn.Linear(2, 2),
        optimizers=[MagicMock()],
        step=MagicMock(),
        zero_grad=MagicMock(),
        state_dict=MagicMock(return_value=state_dict),
        load_state_dict=MagicMock(),
    )

    wrapper = OptimizerWrapper(foreign_container.model, foreign_container)

    assert wrapper.state_dict() == state_dict
    wrapper.load_state_dict(state_dict)
    foreign_container.load_state_dict.assert_called_once_with(state_dict)


@pytest.mark.L0
@pytest.mark.CPU
def test_model_wrapper_persists_unregistered_fake_score() -> None:
    model = _ListWrappedFakeScoreModel()

    # net_fake_score is hidden from the module registry (inference loads stay student-only)
    assert "net_fake_score" not in dict(model.named_children())

    state_dict = ModelWrapper(model).state_dict()
    assert any(k.startswith("net.") for k in state_dict)
    assert any(k.startswith("net_fake_score.") for k in state_dict)


@pytest.mark.L0
@pytest.mark.CPU
def test_model_wrapper_round_trips_unregistered_fake_score() -> None:
    src = _ListWrappedFakeScoreModel()
    with torch.no_grad():
        for param in src.net_fake_score.parameters():
            param.add_(1.0)
    saved = {k: v.clone() for k, v in ModelWrapper(src).state_dict().items()}

    dst = _ListWrappedFakeScoreModel()
    ModelWrapper(dst, strict_resume=True).load_state_dict(saved)

    for src_param, dst_param in zip(src.net_fake_score.parameters(), dst.net_fake_score.parameters()):
        torch.testing.assert_close(src_param, dst_param)


@pytest.mark.L0
@pytest.mark.CPU
def test_model_wrapper_load_uses_non_strict_for_strict_rejecting_model() -> None:
    # Resume path: ModelWrapper.load_state_dict -> set_model_state_dict(model) ->
    # model.load_state_dict, which must not pass strict=True to a model that
    # rejects it (OmniMoTModel). This previously raised on resume.
    src = _StrictRejectingModel()
    with torch.no_grad():
        for param in src.net_fake_score.parameters():
            param.add_(0.5)
    saved = {k: v.clone() for k, v in ModelWrapper(src).state_dict().items()}

    dst = _StrictRejectingModel()
    ModelWrapper(dst, strict_resume=True).load_state_dict(saved)

    for src_param, dst_param in zip(src.net_fake_score.parameters(), dst.net_fake_score.parameters()):
        torch.testing.assert_close(src_param, dst_param)


@pytest.mark.L0
@pytest.mark.CPU
def test_model_wrapper_does_not_double_handle_registered_fake_score() -> None:
    model = _RegisteredFakeScoreModel()

    # Registered fake-score already appears via get_model_state_dict; the explicit
    # path must be skipped for compatibility with older dense DMD2 layouts.
    assert "net_fake_score" in dict(model.named_children())
    state_dict = ModelWrapper(model).state_dict()
    fake_score_keys = [k for k in state_dict if k.startswith("net_fake_score.")]
    # Exactly the registry keys, with no duplicated explicit entries.
    assert fake_score_keys
    assert len(fake_score_keys) == len(set(fake_score_keys))


@pytest.mark.L0
@pytest.mark.CPU
def test_model_wrapper_excludes_registered_teacher_weights() -> None:
    model = _SaveContractModel()

    state_dict = ModelWrapper(model, exclude_teacher_weights=True).state_dict()

    assert any(key.startswith("net.") for key in state_dict)
    assert any(key.startswith("net_fake_score.") for key in state_dict)
    assert not any(key.startswith("net_teacher.") for key in state_dict)


@pytest.mark.L0
@pytest.mark.CPU
def test_strict_resume_rejects_missing_fake_score_tensor() -> None:
    source = _ListWrappedFakeScoreModel()
    state_dict = {key: value.clone() for key, value in ModelWrapper(source).state_dict().items()}
    missing_key = next(key for key in state_dict if key.startswith("net_fake_score."))
    del state_dict[missing_key]

    destination = _ListWrappedFakeScoreModel()
    with pytest.raises(ValueError, match="Strict resume failed"):
        ModelWrapper(destination, strict_resume=True).load_state_dict(state_dict)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("async_mode", [AsyncMode.DISABLED, AsyncMode.ASYNC_WITH_PINNED_MEM])
def test_save_emits_exact_distillation_components(async_mode: AsyncMode) -> None:
    model = _SaveContractModel()
    optimizers = {
        "net": torch.optim.AdamW(model.net.parameters(), lr=1e-3),
        "fake_score": torch.optim.AdamW(model.net_fake_score.parameters(), lr=1e-3),
    }
    schedulers = {"net": MagicMock(), "fake_score": MagicMock()}
    grad_scaler = MagicMock(spec=torch.amp.GradScaler)
    grad_scaler.state_dict.return_value = {"scale": 1.0}

    checkpointer = object.__new__(DistributedCheckpointer)
    checkpointer.async_mode = async_mode
    checkpointer.callbacks = None
    checkpointer.save_to_object_store = False
    checkpointer._local_dirname = "/tmp/distillation-checkpoints"
    checkpointer.cpu_offload_state_dict = object()
    checkpointer._wait_for_previous_async_checkpoint = MagicMock()
    checkpointer._checkpoint_async_with_pinned_memory = MagicMock()
    checkpointer.save_state_dict_worker = MagicMock()

    dataloader_wrapper = MagicMock()
    dataloader_wrapper.has_state.return_value = True
    dataloader_wrapper.state_dict.return_value = {"cursor": 7}
    with (
        patch.object(dcp_distill_module, "_DataloaderWrapper", return_value=dataloader_wrapper),
        patch.object(dcp_distill_module.dist, "get_rank", return_value=0),
        patch.object(dcp_distill_module, "get_rand_state_dict", return_value={"torch_rng_state": torch.zeros(1)}),
    ):
        checkpointer.save(model, optimizers, schedulers, grad_scaler, iteration=7)

    if async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
        checkpointer._wait_for_previous_async_checkpoint.assert_called_once_with()
        checkpointer._checkpoint_async_with_pinned_memory.assert_called_once()
        to_save_dict = checkpointer._checkpoint_async_with_pinned_memory.call_args.args[1]
        assert checkpointer.cpu_offload_state_dict is None
    else:
        checkpointer.save_state_dict_worker.assert_called_once()
        to_save_dict = checkpointer.save_state_dict_worker.call_args.args[0]

    assert set(to_save_dict) == {
        "model",
        "trainer",
        "optim_net",
        "optim_fake_score",
        "scheduler_net",
        "scheduler_fake_score",
        "dataloader",
    }
    model_state = to_save_dict["model"][0]
    assert any(key.startswith("net.") for key in model_state)
    assert any(key.startswith("net_fake_score.") for key in model_state)
    assert not any(key.startswith("net_teacher.") for key in model_state)
