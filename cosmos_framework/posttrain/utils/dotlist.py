# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""`a.b.c=value` config overrides, coerced to the field's existing type.

The CLI override surface for the posttrain entrypoints: everything after `--` on the command line
lands here. Extracted from a recipe-specific runner that had embedded ~95 lines of general-purpose
config machinery -- the shape a second entrypoint copies rather than imports.

WHY NOT OmegaConf. The obvious answer is `OmegaConf.structured(cfg)` plus `from_dotlist`. It does
not work here: the config schema uses PEP-604 `int | None` unions, and under
`from __future__ import annotations` those are strings at runtime that OmegaConf's structured path
cannot re-evaluate on older interpreters. So this is a small typed setter instead, deliberately
dependency-light.

WHY IT COERCES AT ALL. A dotlist value arrives as a string. Assigning it raw means `rl.seed=123`
sets the STRING "123", which then reaches vLLM or a numeric comparison as text and fails somewhere
unrelated -- or worse, silently compares wrong. The type comes from the field's current value, and
when that value is `None` (which carries no type) from the dataclass annotation.

WHY IT RAISES ON UNKNOWN KEYS. Dotlist keys are strings, so a config rename cannot break them at
import time. They fail at runtime, and a bare
`AttributeError: 'PosttrainGRPOConfig' object has no attribute 'loop'` does not say which override
produced it. This reports the offending key AND the fields available at the point it broke. That is
not hypothetical: the MR1 -> flat-config rename ported every `cfg.loop.x` attribute access and
missed the `"loop.x=..."` STRINGS.
"""

from __future__ import annotations


def coerce_by_annotation(raw: str, owner, leaf: str):
    """Coerce using the DECLARED field type when the current value is `None`.

    `coerce` infers from the current value, which carries no type information when that value is
    `None` — so `rollout.seed=123` on an `int | None` field would set the STRING "123" and reach
    vLLM (or a numeric comparison) as text. The dataclass annotation still knows, so read it.

    Args:
        raw: the string from the dotlist.
        owner: the object holding the field (its class carries `__dataclass_fields__`).
        leaf: the field name.

    Returns:
        The coerced value, or `raw` unchanged when the annotation is absent or not one we handle
        (a genuinely `str | None` field must keep its string).
    """
    ann = ""
    fields = getattr(type(owner), "__dataclass_fields__", None)
    if fields and leaf in fields:
        ann = str(fields[leaf].type)
    if "bool" in ann:
        return raw.lower() in ("1", "true", "yes", "on")
    if "int" in ann:
        return int(raw)
    if "float" in ann:
        return float(raw)
    return raw


def coerce(raw: str, current):
    """Coerce a dotlist string value to the type of the field it overrides. Bare, dependency-light
    (the config schema uses PEP-604 `int | None` unions that OmegaConf's structured path can't
    re-evaluate under `from __future__ import annotations` on older interpreters) — so this typed
    setter, not `OmegaConf.structured(cfg)`, is the example's override surface."""
    import ast

    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):  # bool handled above
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        parsed = ast.literal_eval(raw) if raw.strip().startswith("[") else raw.split(",")
        elem = type(current[0]) if current else str  # keep element type (e.g. betas -> float)
        return [elem(x) for x in parsed]
    return raw  # str; a None-valued field is routed to coerce_by_annotation by the caller


def apply_dotlist(cfg, dotlist: list[str]) -> None:
    """Apply `a.b.c=value` overrides IN PLACE, coercing each value to the existing field's type.

    Args:
        cfg: the config root to mutate (`PosttrainGRPOConfig`).
        dotlist: `key=value` strings, keys resolved against the LANDED layout — `rl.*` for loop and
            clipped-objective knobs, `rl.optimizer.*`, `rl.data.*`, top-level `model_hf.*` /
            `rollout.*`.

    Raises:
        AttributeError: a key names a field that does not exist, reported WITH the offending key and
            the available fields at the point it broke. Dotlist keys are strings, so a config rename
            cannot break them at import time — they fail here, at runtime, and a bare
            `AttributeError: 'PosttrainGRPOConfig' object has no attribute 'loop'` does not say which
            override produced it. (That is exactly how the MR1->flat-config rename slipped through:
            `cfg.loop.x` attribute accesses were ported, the `"loop.x=..."` STRINGS were not.)
    """
    for item in dotlist:
        key, _, raw = item.partition("=")
        obj = cfg
        parts = key.split(".")
        for i, p in enumerate(parts[:-1]):
            try:
                obj = getattr(obj, p)
            except AttributeError:
                avail = ", ".join(sorted(k for k in vars(obj) if not k.startswith("_"))) or "<none>"
                raise AttributeError(
                    f"dotlist override {item!r}: {type(obj).__name__} has no field {p!r} "
                    f"(at '{'.'.join(parts[: i + 1])}'). Available: {avail}"
                ) from None
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            avail = ", ".join(sorted(k for k in vars(obj) if not k.startswith("_"))) or "<none>"
            raise AttributeError(
                f"dotlist override {item!r}: {type(obj).__name__} has no field {leaf!r}. Available: {avail}"
            )
        cur = getattr(obj, leaf)
        if raw.lower() in ("none", "null"):
            value = None  # explicit unset, whatever the field currently holds
        elif cur is None:
            value = coerce_by_annotation(raw, obj, leaf)  # None carries no type to infer from
        else:
            value = coerce(raw, cur)
        setattr(obj, leaf, value)
