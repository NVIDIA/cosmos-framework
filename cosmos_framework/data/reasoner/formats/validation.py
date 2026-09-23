# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-format contract for Cosmos dataset validators.

Each annotation format (``metropolis-v3.0``, ``cosmos-reason-v1.0``, ...) is a
concrete subclass of ``BaseValidator``. The CLI iterates the registry of
formats and routes ``cosmos-dataset validate <format>`` to the matching class.
"""

import argparse
import json
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from jsonschema import Draft7Validator


class BaseValidator(ABC):
    """Cross-format validator contract."""

    format: ClassVar[str] = ""

    # Auto-populated registry of concrete format validators. Each subclass
    # appends itself in ``__init_subclass__``; the CLI iterates this list
    # to wire up subparsers.
    formats: ClassVar[List[type["BaseValidator"]]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        assert cls.format, f"{cls.__name__} must set the `format` class attribute"
        BaseValidator.formats.append(cls)

    def __init__(self, args: Optional[argparse.Namespace] = None) -> None:
        """Construct a validator.

        ``args`` is the parsed CLI namespace. It is required for ``run()`` but
        optional for programmatic use (calling ``validate_scene`` directly).
        """
        self.args = args
        self._schema_validators: Dict[str, Draft7Validator] = {}

    @cached_property
    def schemas(self) -> dict:
        """Load the bundled format schemas without an external package."""
        return json.loads(Path(__file__).with_name("schemas.json").read_text())

    def _validator_for(self, schema_name: str) -> Draft7Validator:
        """Load a schema by name and return a cached ``Draft7Validator``."""
        cached = self._schema_validators.get(schema_name)
        if cached is not None:
            return cached
        schema = self.schemas[f"{self.format}/{schema_name}"]
        validator = Draft7Validator(schema)
        self._schema_validators[schema_name] = validator
        return validator

    def _validate_data(self, data: Dict[str, Any], schema_name: str) -> List[str]:
        """Validate a dict against a named schema, returning enriched error messages.

        For common jsonschema failure modes whose default message lacks the
        information a user needs to fix the dataset, the message is enriched:
        ``additionalProperties`` gains an ``allowed: ...`` list, ``pattern``
        gains the schema's ``description`` when present.
        """
        try:
            validator = self._validator_for(schema_name)
        except Exception as e:
            return [f"Error loading schema '{schema_name}': {e}"]

        errors = []
        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.absolute_path)
            msg = self._format_schema_error(error)
            errors.append(f"[{path}] {msg}" if path else msg)
        return errors

    @staticmethod
    def _format_schema_error(error: Any) -> str:
        """Add an actionable hint to a jsonschema error message when we can."""
        message: str = str(error.message)
        if error.validator == "additionalProperties":
            allowed = sorted((error.schema.get("properties") or {}).keys())
            if allowed:
                return f"{message} (allowed: {', '.join(allowed)})"
        elif error.validator == "pattern":
            description = error.schema.get("description")
            if description:
                return f"{message} — {description}"
        return message

    @classmethod
    @abstractmethod
    def register_subparser(cls, subparsers: "argparse._SubParsersAction") -> None:
        """Register this format's argparse subparser."""
        ...

    @abstractmethod
    def run(self) -> int:
        """Drive target discovery and the per-target validation loop.

        Reads CLI configuration from ``self.args`` (set at construction).
        Returns a process exit code (0 on success, non-zero on failure).
        """
        ...
