# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attrs


@attrs.define(slots=False)
class QuantizationConfig:
    """Configuration for low-precision quantization of model parameters.

    Controls which quantization method is applied (mxfp8, nvfp4), and which
    parameters are selected for quantization via include/exclude key filters.
    When ``method`` is None, quantization is disabled and all other fields are
    inert.
    """

    # Quantization method for the model.
    method: str | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.in_({"mxfp8", "nvfp4"})),
    )

    # How to select parameters to select for the quantization. Each key is a
    # regular expression matched against a module's fully-qualified name with
    # `re.search` (a plain substring is still a valid pattern, so substring-style
    # keys keep working, while anchors like `^`/`$`, alternation `a|b`, and
    # character classes are also supported). A module is selected only if its FQN
    # matches at least one pattern in `include_regex` and matches none in
    # `exclude_regex`. If `include_regex` is empty, all parameters are
    # considered as included. If `exclude_regex` is empty, no parameters are
    # considered as excluded.
    include_regex: list[str] = attrs.field(factory=list)
    exclude_regex: list[str] = attrs.field(factory=list)
