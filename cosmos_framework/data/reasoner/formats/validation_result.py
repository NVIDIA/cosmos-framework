# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared result type for Cosmos dataset validators."""

from typing import List


class ValidationResult:
    """Result of a validation operation."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.files_checked: int = 0
        self.files_passed: int = 0

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        return "\n".join(
            [
                f"Files checked: {self.files_checked}",
                f"Files passed: {self.files_passed}",
                f"Errors: {len(self.errors)}",
                f"Warnings: {len(self.warnings)}",
            ]
        )
