# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Canonical finite-JSON identity, with no dependencies beyond the standard library.

**Why this is a separate module from ``utils/identity.py`` rather than a function in it.**
``utils/identity.py`` is the repo's canonical-identity primitive and it already implements
this rule -- but it imports ``torch``, ``attrs`` and ``omegaconf`` at module scope, because
it also has to identify tensors, dtypes and DictConfigs. The simulator surface cannot pay
that: ``embodied/`` is imported inside the BEHAVIOR-1K container, whose venv deliberately
carries neither omegaconf nor attrs, and one module-scope ``import torch`` there is the
difference between a role that loads and one that does not (the same mechanism
``roles/env/__init__.py``'s lazy barrel exists for).

So the FINITE-JSON SUBSET lives here, alone, and the heavier module is free to build on it:
one encoding rule, reachable from both sides, rather than a second definition that happens to
agree today.

**THE ENCODING IS FROZEN BECAUSE IDENTITIES ARE PERSISTED.** ``WorkUnit.unit_id`` is the
cross-pod claim key: it names a lease file on NFS, and a campaign that resumes after a
restart matches units by that string. Changing separators, key order, hash or truncation
length would silently re-identify every unit in flight -- every claimed unit would look
unclaimed and every finished one unfinished. The settings below are therefore the ones
``WorkUnit.make`` has always used, reproduced exactly:

* ``sort_keys=True`` -- dict ordering must not reach the digest.
* **default separators** (``", "`` / ``": "``), NOT the compact ones. Compact separators are
  the more usual choice for a canonical form and would be a better default in a green field;
  they are not used here because the existing identities were taken with these.
* ``sha1``, truncated to 16 hex characters.

``tests/test_embodied_unit_identity.py`` pins a known unit id against these constants, so a
well-intentioned "canonicalise the separators" edit fails a test instead of a campaign.

**What this module adds over a bare ``json.dumps``:** the value is validated first, so the
two ways a digest can silently go wrong are refused rather than encoded.

1. A non-finite float. ``json.dumps`` emits bare ``NaN``/``Infinity`` by default, which is
   not JSON, so two readers of the same blob can disagree -- and ``NaN != NaN`` makes the
   value that produced the id unrecoverable.
2. A non-string mapping key. ``json.dumps`` coerces ``1`` and ``"1"`` to the same key, so two
   different scenarios can collide on one identity.

Neither can change an id that is valid today: both cases either crashed or were already
broken. See ``utils.identity._canonical_mapping_key`` for the richer tagged-key encoding used
where non-string keys must be SUPPORTED rather than refused; this module refuses them because
the work-unit surface has never produced one.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

#: The digest settings, named so a reader sees them together and a test can pin them.
HASH = "sha1"
ID_LENGTH = 16


def finite_json(value: Any, *, context: str = "$") -> Any:
    """Return ``value`` as validated finite-JSON data, or raise saying exactly where.

    Structure is preserved, not rewritten: mappings become ``dict`` and sequences become
    ``list``, which is what ``json.dumps`` would produce anyway, so the encoded bytes are
    unchanged for every value that was already encodable.

    Args:
        value: the object to validate.
        context: a JSON-path-ish breadcrumb used in error messages; callers rarely set it.

    Returns:
        The same data, as plain ``dict`` / ``list`` / scalars.

    Raises:
        TypeError: on an unsupported type or a non-string mapping key.
        ValueError: on a non-finite float.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"identity input {context} is a non-finite float {value!r}")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"identity input {context} has a non-string mapping key {key!r} "
                    f"({type(key).__name__}); JSON would coerce it and two different inputs "
                    f"could then share one id"
                )
            out[key] = finite_json(item, context=f"{context}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [finite_json(item, context=f"{context}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"identity input {context} is an unsupported type {type(value).__name__}: {value!r}")


def canonical_blob(value: Any) -> str:
    """The exact text an identity is taken over. Stable across processes and versions."""
    return json.dumps(finite_json(value), sort_keys=True, allow_nan=False)


def digest(value: Any, *, length: int = ID_LENGTH) -> str:
    """Content digest of ``value``: the same input always yields the same id.

    Args:
        value: any finite-JSON value.
        length: hex characters to keep. Defaults to :data:`ID_LENGTH`; callers that persist
            the result should not pass anything else.

    Returns:
        ``length`` hex characters of the sha1 of :func:`canonical_blob`.
    """
    return hashlib.new(HASH, canonical_blob(value).encode()).hexdigest()[:length]
