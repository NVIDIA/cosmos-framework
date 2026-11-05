# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Importable custom-data fixtures used by posttrain extension-point tests."""

from __future__ import annotations

from typing import Any

from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.schema import ObsBundle


def build_prompt_dataset(count: int = 2, prefix: str = "question") -> list[dict[str, str]]:
    """Return raw prompt rows whose columns are mapped by the framework."""
    return [{"question": f"{prefix}-{index}", "answer": f"answer-{index}"} for index in range(count)]


def build_preference_dataset(count: int = 2) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return canonical DPO preference triples."""
    return [
        (
            f"prompt-{index}",
            {"sample": f"winner-{index}"},
            {"sample": f"loser-{index}"},
        )
        for index in range(count)
    ]


class IdentifiedPreferenceDataset:
    """Canonical custom triples with metadata-only leakage identities."""

    def __init__(self, count: int = 2) -> None:
        self._items: list[tuple[str, dict[str, Any], dict[str, Any]]] = build_preference_dataset(count)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
        return self._items[index]

    def membership_ids(self) -> tuple[str, ...]:
        return tuple(f"custom-pair-{index}" for index in range(len(self._items)))


def build_identified_preference_dataset(count: int = 2) -> IdentifiedPreferenceDataset:
    """Return a custom source with explicit metadata-only membership identities."""
    return IdentifiedPreferenceDataset(count)


class StatefulPreferenceDataset:
    """Canonical triples with checkpointable internal preprocessing state."""

    def __init__(self, count: int = 2) -> None:
        self._items: list[tuple[str, dict[str, Any], dict[str, Any]]] = build_preference_dataset(count)
        self._offset: int = 0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
        self._offset += 1
        return self._items[index]

    def cursor_state_dict(self) -> dict[str, Any]:
        return {"offset": self._offset}

    def load_cursor_state_dict(self, state: dict[str, Any]) -> None:
        self._offset = int(state["offset"])


def build_stateful_preference_dataset(count: int = 2) -> StatefulPreferenceDataset:
    """Return a custom source whose internal state must survive framework wrapping."""
    return StatefulPreferenceDataset(count)


def build_empty_dataset() -> list[dict[str, str]]:
    """Return an invalid empty custom source."""
    return []


def build_iterable_dataset() -> Any:
    """Return an invalid iterable-only custom source."""
    return iter([{"question": "question", "answer": "answer"}])


def format_prompt(row: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a raw prompt value while preserving its schema."""
    return {**row, "question": f"formatted: {row['question']}"}


class NamedDataPacker(DataPacker):
    """Minimal setup-observable DataPacker for factory resolution tests."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.setup_name_or_path: str = ""
        self.setup_trust_remote_code: bool = False

    def setup(self, name_or_path: str, trust_remote_code: bool = False) -> None:
        self.setup_name_or_path = name_or_path
        self.setup_trust_remote_code = trust_remote_code

    def get_rollout_input(self, item: ObsBundle) -> Any:
        return item.representations

    def get_policy_input(
        self,
        sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> Any:
        return sample, rollout_output, n_ignore_prefix_tokens, prompt_ids

    def policy_compute_max_len(self, processed_samples: list) -> int:
        return len(processed_samples)

    def policy_collate_fn(self, processed_samples: list, computed_max_len: int) -> dict:
        return {"samples": processed_samples, "max_len": computed_max_len}


def build_default_packer() -> DataPacker:
    return NamedDataPacker("default")


def build_teacher_model_packer() -> DataPacker:
    return NamedDataPacker("teacher-model")


def build_non_packer() -> object:
    return object()
