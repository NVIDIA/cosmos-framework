# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Every relative documentation link in tracked posttrain content must resolve to a file on disk.

The restructure filed every topic document under an area subdirectory, and `docs/index.md` is now
the map users navigate from, so a stale link silently removes a document from the manual. This
keeps the check in CI rather than in a reviewer's head.

Every tracked text file is scanned, not just markdown and Python. Shell launchers, Dockerfiles and
even a `.gitignore` cite documentation paths, and a restructure breaks those exactly as readily —
a review of this branch found four such files still pointing at pre-restructure paths that the
narrower scan could not see.

Two link shapes are checked, because the sources write them differently:

* Markdown files use ordinary relative links (``[text](../reference/config.md)``), resolved
  against the linking file's own directory. Every relative target is checked, not only ``.md``
  ones: moving a document one directory deeper breaks its ``../pyproject.toml`` links exactly as
  readily as its ``../README.md`` links, and a target may legitimately be a directory.
* Everything else — Python docstrings, shell launchers, Dockerfiles, ``.gitignore`` — points at
  documents by repository-relative path (``docs/design/DESIGN.md``), resolved against the posttrain
  package root.

Only files tracked by git are scanned. Untracked local scratch notes may legitimately record
historical paths, and planning artifacts under ``docs/superpowers/`` reference paths that are
planned but not yet created.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

POSTTRAIN = Path(__file__).resolve().parents[1]
_MD_LINK = re.compile(r"\]\(\s*([^)\s#]+)(?:#[^)\s]*)?\s*\)")
_NOT_A_PATH = ("http://", "https://", "mailto:", "#")
_PY_DOC_PATH = re.compile(r"(?<![\w/])((?:\.\./)*docs/[\w./-]+\.md)")
_SKIP_SUBTREE = Path("docs") / "superpowers"
# Any tracked text file may cite a document; a restructure breaks a shell script's reference exactly
# as readily as a markdown link.
_TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml", ".cfg", ".txt", ".json"}
_TEXT_NAMES = {".gitignore"}
# `Dockerfile.behavior1k` has suffix ".behavior1k", so neither the suffix set nor an exact-name set
# admits it. Match the stem instead, or the image definitions are silently excluded — which is
# exactly how four stale references survived the first expansion of this scan.
_TEXT_STEMS = ("Dockerfile",)
_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".sqsh", ".lock", ".ipynb"}


def _tracked_files() -> list[Path]:
    """Return the git-tracked ``.md`` and ``.py`` files under the posttrain package."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=POSTTRAIN,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = [POSTTRAIN / name for name in listing.split("\0") if name]
    return [p for p in paths if _SKIP_SUBTREE not in p.relative_to(POSTTRAIN).parents and _is_scannable(p)]


def _is_scannable(path: Path) -> bool:
    """Text files that plausibly cite a documentation path. Binary and data files are skipped."""
    if path.suffix in _BINARY_SUFFIXES:
        return False
    if path.name.startswith(_TEXT_STEMS):
        return True
    return path.suffix in _TEXT_SUFFIXES or path.name in _TEXT_NAMES


def collect_doc_links(files: list[Path] | None = None) -> list[tuple[Path, int, str, Path]]:
    """Find every relative documentation link and the path it should resolve to.

    Args:
        files: Files to scan. Defaults to every tracked posttrain markdown and Python file.

    Returns:
        One ``(source_file, line_number, raw_target, resolved_path)`` tuple per link found.
    """
    found: list[tuple[Path, int, str, Path]] = []
    for path in files if files is not None else _tracked_files():
        if path.is_symlink():
            continue  # AGENTS.md is scanned directly; .claude/CLAUDE.md symlinks to it
        is_markdown = path.suffix == ".md"
        if path.name.startswith("Dockerfile"):
            is_markdown = False
        pattern = _MD_LINK if is_markdown else _PY_DOC_PATH
        base = path.parent if is_markdown else POSTTRAIN
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for target in pattern.findall(line):
                if target.startswith(_NOT_A_PATH):
                    continue
                found.append((path, lineno, target, (base / target).resolve()))
    return found


@pytest.mark.L0
@pytest.mark.CPU
def test_every_docs_link_resolves() -> None:
    links = collect_doc_links()
    assert links, "scanner found no documentation links at all - the patterns or the root are wrong"
    broken = [
        f"{path.relative_to(POSTTRAIN)}:{lineno} -> {target}"
        for path, lineno, target, resolved in links
        if not resolved.exists()
    ]
    assert not broken, "dead documentation links:\n" + "\n".join(broken)


@pytest.mark.L0
@pytest.mark.CPU
def test_index_reaches_every_user_document() -> None:
    """`docs/index.md` is the documentation home: no user-facing document may be unreachable."""
    index = POSTTRAIN / "docs" / "index.md"
    linked = {resolved for _, _, _, resolved in collect_doc_links([index])}
    user_areas = ("guides", "howto", "reference", "eval")
    expected = {path.resolve() for area in user_areas for path in (POSTTRAIN / "docs" / area).glob("*.md")}
    expected |= {
        (POSTTRAIN / "docs" / name).resolve() for name in ("getting-started.md", "glossary.md", "troubleshooting.md")
    }
    missing = sorted(str(p.relative_to(POSTTRAIN)) for p in expected - linked if p.is_file())
    assert not missing, "documents not reachable from docs/index.md:\n" + "\n".join(missing)
