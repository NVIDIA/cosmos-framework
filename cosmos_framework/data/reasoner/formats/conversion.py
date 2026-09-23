# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-pair contract for Cosmos dataset converters.

Each (source_format, target_format) pair is a concrete subclass of
``BaseConverter``. The CLI groups pairs by ``source_format`` to build
nested subparsers and routes ``cosmos-dataset convert <source> <target>`` to
the matching class.

Conversion is uni-directional: data-centric → training-centric.
"""

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Optional


@dataclass
class ConversionResult:
    """Result of a format conversion."""

    samples_written: int = 0
    samples_skipped: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_success(self) -> bool:
        """Return True iff every requested input sample produced an output sample.

        Both errors and skipped samples count as failure. The converter
        contract is "one output per input item"; if any input was dropped
        (missing source media, malformed answer, etc.) the run did not
        deliver what was asked, and CI gates / pipelines should treat the
        exit code as non-zero. The output that was written is still valid
        and loadable by the target validator — it's just smaller than the
        caller asked for.
        """
        return len(self.errors) == 0 and self.samples_skipped == 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Samples written : {self.samples_written}",
            f"Samples skipped : {self.samples_skipped}",
        ]
        if self.warnings:
            lines.append(f"Warnings        : {len(self.warnings)}")
        if self.errors:
            lines.append(f"Errors          : {len(self.errors)}")
        return "\n".join(lines)


class BaseConverter(ABC):
    """Cross-pair converter contract."""

    source_format: ClassVar[str] = ""
    target_format: ClassVar[str] = ""

    # Auto-populated registry of concrete converter pairs. Each subclass
    # appends itself in ``__init_subclass__``; the CLI iterates this list
    # to wire up subparsers.
    converters: ClassVar[List[type["BaseConverter"]]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        assert cls.source_format, f"{cls.__name__} must set the `source_format` class attribute"
        assert cls.target_format, f"{cls.__name__} must set the `target_format` class attribute"
        BaseConverter.converters.append(cls)

    def __init__(self, args: Optional[argparse.Namespace] = None) -> None:
        """Construct a converter.

        ``args`` is the parsed CLI namespace. It is required for ``run()`` but
        optional for programmatic use (calling ``convert_scene`` directly).
        """
        self.args = args

    @classmethod
    def pair_name(cls) -> str:
        """Stable identifier for diagnostics and tests (e.g. ``metropolis-v3.0_to_cosmos-reason-v1.0``).

        The CLI dispatches via ``(args.source, args.target)`` rather than this
        joined name, so the format is internal and not user-typed.
        """
        return f"{cls.source_format}_to_{cls.target_format}"

    @staticmethod
    def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
        """Add args every conversion needs: ``--path`` and ``--output``."""
        parser.add_argument(
            "--path",
            type=Path,
            required=True,
            help="Path to source scene or dataset root",
        )
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Output directory for the converted dataset",
        )

    @classmethod
    def validate_registry(cls) -> None:
        """Sanity-check that every registered pair targets known formats.

        Called once from ``cli.main()`` after both validators and converters
        have been imported. Catches typos in ``source_format`` / ``target_format``
        ClassVars that would otherwise fail lazily at dispatch time.
        """
        # Local import: deferred to avoid a converters↔validators import cycle.
        from cosmos_framework.data.reasoner.formats.validation import BaseValidator

        known = {v.format for v in BaseValidator.formats}
        for converter_cls in cls.converters:
            assert converter_cls.source_format in known, (
                f"{converter_cls.__name__}: source_format "
                f"'{converter_cls.source_format}' is not a registered validator format"
            )
            assert converter_cls.target_format in known, (
                f"{converter_cls.__name__}: target_format "
                f"'{converter_cls.target_format}' is not a registered validator format"
            )

    @classmethod
    @abstractmethod
    def register_subparser(cls, target_subparsers: "argparse._SubParsersAction") -> None:
        """Register this pair's argparse subparser under its ``target_format``.

        ``target_subparsers`` is the subparser group attached to a *source*
        subparser; the implementation should call ``add_parser(cls.target_format, ...)``.
        """
        ...

    @abstractmethod
    def run(self) -> int:
        """Drive the conversion using ``self.args``.

        Returns a process exit code (0 on success, non-zero on failure).
        """
        ...
