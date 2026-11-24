# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read a notebook that was migrated to a Markdown reference as notebook-shaped cells.

The Simulation cookbook notebooks were converted cell by cell into the skills' ``references/``:
every code cell became one fenced ``python`` block and every Markdown cell was copied verbatim.
None of the source notebooks had a cell magic or a ``python`` fence inside a Markdown cell, so each
``python`` fence is exactly one code cell and the text between two of them is the Markdown that
stood there (adjacent Markdown cells come back as one cell).
"""

import re
from pathlib import Path

_OPEN = re.compile(r"^(`{3,})python\s*$")


def reference_notebook(path):
    """Return ``{"cells": [{"cell_type": ..., "source": [...]}]}`` for a converted reference."""
    cells = []
    markdown = []
    code = None
    fence = ""

    def flush_markdown():
        text = "".join(markdown).strip("\n")
        if text:
            cells.append({"cell_type": "markdown", "source": [text]})
        markdown.clear()

    for line in Path(path).read_text(encoding="utf-8").splitlines(keepends=True):
        if code is None:
            match = _OPEN.match(line.rstrip("\n"))
            if match:
                flush_markdown()
                code = []
                fence = match.group(1)
            else:
                markdown.append(line)
        elif line.rstrip("\n") == fence:
            cells.append({"cell_type": "code", "source": ["".join(code).rstrip("\n")]})
            code = None
        else:
            code.append(line)
    if code is not None:
        raise ValueError(f"Unclosed python fence in {path}")
    flush_markdown()
    return {"cells": cells}
