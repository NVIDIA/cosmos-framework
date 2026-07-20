# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from cosmos_framework.inference.common.args import GuardrailArgs
from cosmos_framework.inference.common.inference import GuardrailRunners, _download_on_rank0


def test_download_on_rank0_broadcasts_shared_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    download = Mock(return_value="/shared/cache/model")
    broadcast = Mock()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    assert _download_on_rank0(download) == Path("/shared/cache/model")
    download.assert_called_once_with()
    broadcast.assert_called_once_with(["/shared/cache/model", None], src=0)


def test_download_on_nonzero_rank_reuses_broadcast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    download = Mock()

    def broadcast(payload: list[str | None], *, src: int) -> None:
        assert src == 0
        payload[:] = ["/shared/cache/model", None]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    assert _download_on_rank0(download) == Path("/shared/cache/model")
    download.assert_not_called()


def test_guardrail_runners() -> None:
    from cosmos_framework.auxiliary.guardrail.common import presets

    guardrail_args = GuardrailArgs(guardrails=True, offload_guardrail_models=False)
    runners = GuardrailRunners.create(guardrail_args)
    assert runners.text is not None
    assert runners.video is not None

    assert presets.run_text_guardrail("test", runners.text)
    assert not presets.run_text_guardrail("Tesla Cybertruck", runners.text)

    frames_thwc = np.random.randint(0, 255, (1, 16, 16, 3), dtype=np.uint8)
    clean_frames_thwc = presets.run_video_guardrail(frames_thwc, runners.video)
    assert clean_frames_thwc is not None
    np.testing.assert_allclose(frames_thwc, clean_frames_thwc)
