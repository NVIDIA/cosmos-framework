# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.model._base import ImaginaireModel, close_model
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.trainer import ImaginaireTrainer

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


class _RecordingModel(ImaginaireModel):
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        super().__init__()
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _RecordingTrainer(ImaginaireTrainer):
    def __init__(self, *, train_error: BaseException | None = None) -> None:
        self.train_error = train_error
        self.train_calls = 0

    def _train(
        self,
        model: ImaginaireModel,
        dataloader_train: Any,
        dataloader_val: Any,
    ) -> None:
        del model, dataloader_train, dataloader_val
        self.train_calls += 1
        if self.train_error is not None:
            raise self.train_error


class _RecordingProvider:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_close_model_preserves_primary_error_when_cleanup_also_fails() -> None:
    primary_error = RuntimeError("primary failure")
    close_error = ValueError("cleanup failure")
    model = _RecordingModel(close_error=close_error)

    close_model(model, primary_error=primary_error)

    assert model.close_calls == 1
    if hasattr(primary_error, "add_note"):
        assert any("ValueError: cleanup failure" in note for note in getattr(primary_error, "__notes__", ()))

    with pytest.raises(ValueError, match="cleanup failure") as raised:
        close_model(_RecordingModel(close_error=close_error))
    assert raised.value is close_error


def test_trainer_closes_model_after_successful_training() -> None:
    trainer = _RecordingTrainer()
    model = _RecordingModel()
    dataloader: Any = object()

    trainer.train(model, dataloader, dataloader)

    assert trainer.train_calls == 1
    assert model.close_calls == 1


def test_trainer_closes_model_and_preserves_training_failure() -> None:
    train_error = RuntimeError("training failure")
    trainer = _RecordingTrainer(train_error=train_error)
    model = _RecordingModel()
    dataloader: Any = object()

    with pytest.raises(RuntimeError, match="training failure") as raised:
        trainer.train(model, dataloader, dataloader)

    assert raised.value is train_error
    assert trainer.train_calls == 1
    assert model.close_calls == 1


def test_omni_setup_failure_closes_provider_once_and_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RecordingProvider()
    created_models: list[OmniMoTModel] = []

    def create_provider(model: OmniMoTModel) -> _RecordingProvider:
        created_models.append(model)
        return provider

    def fail_setup(_model: OmniMoTModel) -> None:
        raise RuntimeError("setup failure")

    monkeypatch.setattr(OmniMoTModel, "_create_reasoner_feature_provider", create_provider)
    monkeypatch.setattr(OmniMoTModel, "set_precision", fail_setup)
    config = SimpleNamespace(reasoner_conditioning={"backend": "joint"})

    with pytest.raises(RuntimeError, match="setup failure"):
        OmniMoTModel(config)  # type: ignore[arg-type]

    assert len(created_models) == 1
    model = created_models[0]
    assert provider.close_calls == 1
    assert model.reasoner_feature_provider is None

    model.close()
    model.close()
    assert provider.close_calls == 1
