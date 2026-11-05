# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""MkDocs hooks for the doc site. tools/docs/stage.py adds this file to build/mkdocs.yml.

Each page's staged Markdown is published next to its HTML, at the page's site path
(for example get_started/quickstart.md), so llms.txt and agents can read pages as
Markdown. Links in those files are relative, so they resolve to the Markdown of the
linked pages. Each HTML page also points to its Markdown with a
<link rel="alternate" type="text/markdown">.
"""

from __future__ import annotations

import posixpath
import shutil
from html import escape
from pathlib import Path


def on_post_page(output: str, page, config, **kwargs) -> str:
    page_dir = page.url if page.url.endswith("/") or not page.url else posixpath.dirname(page.url)
    href = posixpath.relpath(page.file.src_uri, page_dir or ".")
    link = f'<link rel="alternate" type="text/markdown" href="{escape(href)}">'
    return output.replace("</head>", f"{link}\n</head>", 1)


def on_post_build(config, **kwargs) -> None:
    docs, site = Path(config["docs_dir"]), Path(config["site_dir"])
    for src in docs.rglob("*.md"):
        dst = site / src.relative_to(docs)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
