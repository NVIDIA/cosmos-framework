#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage repository Markdown for the doc site and write the MkDocs config.

Sources stay GitHub-flavored Markdown. This script copies the pages listed in
tools/docs/nav.yml into <out>/docs-site/ under their site paths, adapts them
for MkDocs (strips <!--TOC--> blocks, renders <details> content and GitHub
alert labels, rewrites relative links and copies embedded images), and writes
<out>/mkdocs.yml. It also builds the agent skills table from every
.agents/skills/**/SKILL.md and writes llms.txt, which links every page's
Markdown (published by tools/docs/hooks.py) and every skill.

    uv run --project tools/docs python tools/docs/stage.py
    uv run --project tools/docs mkdocs build --strict -f build/mkdocs.yml

Exits non-zero on any ERROR. See docs/README.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import posixpath
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from urllib.parse import unquote

import yaml

GITHUB = "https://github.com/NVIDIA/cosmos-framework"
RAW_GITHUB = "https://raw.githubusercontent.com/NVIDIA/cosmos-framework/main"
SYNCED_PREFIX = "docs/posttrain/"  # synced from upstream: never edit; problems only warn
SITE_ONLY_PREFIX = "tools/docs/site/"  # pages that exist only on the site; links are site paths
DOCS_DIR = "docs-site"
SITE_DIR = "site"
MARKER = ".staged-by-stage-py"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
GENERATED_KEYS = ("docs_dir", "site_dir", "nav", "not_in_nav", "hooks")
THEME_DIRS = ("stylesheets", "images")  # tools/docs/<dir> copied to the site root, referenced by mkdocs.yml
HOOKS = "tools/docs/hooks.py"  # publishes each page's Markdown next to its HTML
SKILLS_ROOT = ".agents/skills"
SKILLS_MARKER = "<!-- agent-skills-table -->"  # replaced with the table of agent skills
LLMS_TXT = "llms.txt"  # https://llmstxt.org/; site-only pages may link to it

# ---------------------------------------------------------------- Markdown patterns

FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
TOC_RE = re.compile(r"<!--TOC-->.*?<!--TOC-->", re.S)
DETAILS_RE = re.compile(r"<details(\s[^>]*)?>", re.I)
CODE_SPAN_RE = re.compile(r"(`+)(?:(?!\n[ \t]*\n).)+?(?<!`)\1(?!`)", re.S)
_TEXT = r"(?:[^\[\]\\]|\\.|\[(?:[^\[\]\\]|\\.)*\])*"
_DEST = r"<[^<>\n]*>|(?:[^\s()<>\\]|\\.|\((?:[^\s()\\]|\\.)*\))+"
_TITLE = r"""(?:\s+(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\)))?"""
IMAGE_RE = re.compile(rf"!\[({_TEXT})\]\(\s*({_DEST})({_TITLE})\s*\)")
LINK_RE = re.compile(rf"(?<![!\\])\[({_TEXT})\]\(\s*({_DEST})({_TITLE})\s*\)")
REFDEF_RE = re.compile(r"^( {0,3}\[(?!\^)[^\]\n]+\]:[ \t]*)(<[^<>\n]*>|\S+)", re.M)
HTML_ATTR_RE = re.compile(r"(<(a|img|video|source)\b[^>]*?\s(href|src)\s*=\s*)([\"'])(.*?)\4", re.I | re.S)
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
ATX_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
ATTR_ID_RE = re.compile(r"[ \t]*\{[^}]*#([\w-]+)[^}]*\}[ \t]*$")
HTML_ID_RE = re.compile(r"<[a-zA-Z][^>]*?\s(?:id|name)\s*=\s*[\"']([^\"']+)[\"']")
DUP_ANCHOR_RE = re.compile(r"^(.+)-(\d+)$")
ALERT_RE = re.compile(r"^([ \t]*>[ \t]*)\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*$", re.M | re.I)
FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*$", re.S | re.M)
H1_RE = re.compile(r"^#[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", re.M)
TRAILING_ATTR_LIST_RE = re.compile(r"[ \t]*\{[^}]*\}[ \t]*$")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
SKILLS_MARKER_RE = re.compile(rf"^[ \t]*{re.escape(SKILLS_MARKER)}[ \t]*$", re.M)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->[ \t]*\n?", re.S)


class BaseConfigLoader(yaml.SafeLoader):
    """Reads tools/docs/mkdocs.yml, ignoring Python tags such as the slugify function and tags such as !ENV."""


BaseConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)
BaseConfigLoader.add_multi_constructor("!", lambda loader, suffix, node: None)


def strip_toc(text: str) -> str:
    """Remove <!--TOC--> ... <!--TOC--> blocks, keeping line numbers stable."""
    return TOC_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def add_details_markdown(text: str) -> str:
    """Add markdown="1" to <details> tags so md_in_html renders their content."""

    def repl(m: re.Match) -> str:
        attrs = m.group(1) or ""
        if "markdown=" in attrs:
            return m.group(0)
        return f'<details{attrs} markdown="1">'

    return DETAILS_RE.sub(repl, text)


def label_alerts(text: str) -> str:
    """Turn GitHub alert markers (> [!NOTE]) into a bold label; MkDocs renders a plain blockquote."""
    return ALERT_RE.sub(lambda m: f"{m.group(1)}**{m.group(2).capitalize()}**", text)


def split_fences(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_code, chunk) runs; fenced code blocks are left untouched."""
    out: list[tuple[bool, str]] = []
    buf: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        m = FENCE_RE.match(line)
        if fence is None:
            if m:
                if buf:
                    out.append((False, "".join(buf)))
                buf = [line]
                fence = m.group(1)
            else:
                buf.append(line)
        else:
            buf.append(line)
            if (
                m
                and m.group(1)[0] == fence[0]
                and len(m.group(1)) >= len(fence)
                and not line.strip()[len(m.group(1)) :].strip()
            ):
                out.append((True, "".join(buf)))
                buf, fence = [], None
    if buf:
        out.append((fence is not None, "".join(buf)))
    return out


def slugify(text: str) -> str:
    """Approximate pymdownx.slugs.slugify(case="lower") on a Markdown heading."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images contribute no text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> text
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"(?<!\w)(\*\*|__)(.+?)\1(?!\w)", r"\2", text)
    text = re.sub(r"(?<!\w)(\*|_)(.+?)\1(?!\w)", r"\2", text)
    text = text.replace("`", "")
    text = re.sub(r"\\(.)", r"\1", text)
    text = unescape(text)
    text = re.sub(r"</?[^>]*>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def heading_ids(text: str) -> tuple[list[str], set[str]]:
    """Return (base slugs in order, final MkDocs anchor ids) for a Markdown page."""
    bases: list[str] = []
    ids: set[str] = set()

    def unique(slug: str) -> str:
        # python-markdown toc: second "x" becomes "x_1", third "x_2", ...
        while slug in ids or not slug:
            m = re.match(r"^(.*)_([0-9]+)$", slug)
            slug = f"{m.group(1)}_{int(m.group(2)) + 1}" if m else f"{slug}_1"
        return slug

    for is_code, chunk in split_fences(strip_toc(text)):
        if is_code:
            continue
        lines = chunk.split("\n")
        for i, line in enumerate(lines):
            title = None
            m = ATX_RE.match(line)
            if m:
                title = m.group(2)
            elif (
                i > 0
                and SETEXT_RE.match(line)
                and lines[i - 1].strip()
                and not re.match(r"^\s*([-*+>|#<]|\d+[.)])", lines[i - 1])
            ):
                title = lines[i - 1].strip()
            if title is None:
                continue
            explicit = ATTR_ID_RE.search(title)
            if explicit:
                ids.add(explicit.group(1))
                continue
            base = slugify(title)
            bases.append(base)
            ids.add(unique(base))
        ids.update(HTML_ID_RE.findall(chunk))
    return bases, ids


def github_url(rel: str, is_dir: bool, query: str = "", anchor: str = "") -> str:
    url = f"{GITHUB}/{'tree' if is_dir else 'blob'}/main/{rel}"
    if query:
        url += "?" + query
    if anchor:
        url += "#" + anchor
    return url


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(len(LFS_POINTER)) == LFS_POINTER
    except OSError:
        return False


def matches(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, p) for p in patterns)


def first_heading(text: str) -> str | None:
    """The first H1 as plain text: links reduced to their text, a trailing { #id } dropped."""
    for is_code, chunk in split_fences(text):
        if not is_code and (m := H1_RE.search(chunk)):
            return LINK_RE.sub(lambda link: link.group(1), TRAILING_ATTR_LIST_RE.sub("", m.group(1)))
    return None


def link_text(text: str) -> str:
    """Escape text for the [...] part of a Markdown link."""
    return re.sub(r"([\\\[\]])", r"\\\1", text)


def table_cell(text: str) -> str:
    """Escape '|' and '<' outside code spans, so text stays in one Markdown table cell."""
    out, pos = [], 0
    for m in [*CODE_SPAN_RE.finditer(text), None]:
        end = m.start() if m else len(text)
        out.append(text[pos:end].replace("<", "&lt;").replace("|", "\\|"))
        if m:
            out.append(m.group(0))
            pos = m.end()
    return "".join(out)


@dataclass
class Skill:
    name: str
    description: str  # front matter description, whitespace collapsed
    summary: str
    path: str  # repository path of the SKILL.md


def summarize(description: str, limit: int = 160) -> str:
    """First sentence of a skill description, cut at a word boundary to at most limit characters."""
    text = SENTENCE_END_RE.split(description, maxsplit=1)[0]
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:")
        if text.count("`") % 2:  # never cut inside a code span
            text = text[: text.rindex("`")].rstrip()
        text += "…"
    return text


# ---------------------------------------------------------------- policy


@dataclass
class Policy:
    nav: list
    sources: dict[str, str] = field(default_factory=dict)
    extra_pages: list[str] = field(default_factory=list)
    hold: list[str] = field(default_factory=list)
    not_on_site: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Policy:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {"nav", "sources", "extra_pages", "hold", "not_on_site"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        nav = data.get("nav") or []
        nav_pages(nav)  # raises on a malformed entry
        return cls(
            nav=nav,
            sources=data.get("sources") or {},
            extra_pages=data.get("extra_pages") or [],
            hold=data.get("hold") or [],
            not_on_site=data.get("not_on_site") or [],
        )

    def site_to_source(self, site: str) -> str:
        if site in self.sources:
            return self.sources[site]
        if site.startswith("skills/"):
            return ".agents/skills/" + site[len("skills/") :]
        return "docs/" + site

    def source_to_site(self, src: str) -> str | None:
        for site, s in self.sources.items():
            if s == src:
                return site
        if src.startswith(".agents/skills/"):
            return "skills/" + src[len(".agents/skills/") :]
        if src.startswith("docs/"):
            return src[len("docs/") :]
        return None


def nav_pages(nav: list) -> list[str]:
    """Return the site path of every page entry of a MkDocs nav."""
    out: list[str] = []

    def walk(items: list) -> None:
        for item in items:
            if isinstance(item, str):
                out.append(item)
                continue
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"bad nav entry: {item!r}")
            ((_, value),) = item.items()
            if isinstance(value, list):
                walk(value)
            elif not SCHEME_RE.match(str(value)):
                out.append(str(value))

    walk(nav)
    return out


def prune_nav(nav: list, keep: dict[str, str]) -> list:
    """Drop page entries whose site path is not in keep, and sections left empty."""
    out = []
    for item in nav:
        if isinstance(item, str):
            if item in keep:
                out.append(item)
            continue
        ((title, value),) = item.items()
        if isinstance(value, list):
            if value := prune_nav(value, keep):
                out.append({title: value})
        elif SCHEME_RE.match(str(value)) or value in keep:
            out.append(item)
    return out


# ---------------------------------------------------------------- staging


class Stager:
    def __init__(
        self,
        repo: Path,
        policy: Policy,
        base_config: str,
        out: Path,
        list_lfs: bool = False,
    ) -> None:
        self.repo = repo
        self.policy = policy
        self.base_config = base_config
        self.out = out
        self.docs_out = out / DOCS_DIR
        self.list_lfs = list_lfs
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: Counter[str] = Counter()
        self.site_to_src: dict[str, str] = {}
        self.src_to_site: dict[str, str] = {}
        self.lfs_pointers: set[str] = set()
        self.copied_assets: set[str] = set()
        self.skills: list[Skill] = []
        self._ids_cache: dict[str, tuple[list[str], set[str]]] = {}

    # ---- logging

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    # ---- page map and coverage

    def build_map(self) -> None:
        seen: Counter[str] = Counter(nav_pages(self.policy.nav))
        for site, n in seen.items():
            if n > 1:
                self.error(f"nav.yml: '{site}' appears {n} times in nav")
        entries = [(site, self.policy.site_to_source(site), True) for site in seen]
        for src in self.policy.extra_pages:
            site = self.policy.source_to_site(src)
            if site is None:
                self.error(f"nav.yml: extra page '{src}' has no site path (add it to sources)")
                continue
            entries.append((site, src, False))
        for site, src, in_nav in entries:
            if not site.endswith(".md"):
                self.error(f"nav.yml: '{site}' is not a Markdown page")
                continue
            if site in self.site_to_src:
                self.error(f"nav.yml: site path '{site}' is listed twice")
                continue
            if not (self.repo / src).is_file():
                where = "nav entry" if in_nav else "extra page"
                if src.startswith(SYNCED_PREFIX):  # removed upstream: leave it off the site
                    self.warn(f"nav.yml: {where} '{site}' -> '{src}' does not exist; left off the site")
                else:
                    self.error(f"nav.yml: {where} '{site}' -> '{src}' does not exist")
                continue
            if src in self.src_to_site:
                self.error(f"nav.yml: '{src}' is staged twice, as '{self.src_to_site[src]}' and '{site}'")
                continue
            if matches(src, self.policy.hold) or matches(src, self.policy.not_on_site):
                self.error(f"nav.yml: '{src}' is on the site and also in hold/not_on_site")
            self.site_to_src[site] = src
            self.src_to_site[src] = site

    def check_coverage(self) -> None:
        docs = self.repo / "docs"
        if docs.is_dir():
            for path in sorted(docs.rglob("*.md")):
                rel = path.relative_to(self.repo).as_posix()
                if any(part.startswith(".") for part in rel.split("/")):
                    continue
                if rel in self.src_to_site or matches(rel, self.policy.hold) or matches(rel, self.policy.not_on_site):
                    continue
                if rel.startswith(SYNCED_PREFIX):
                    self.warn(f"{rel}: not listed in tools/docs/nav.yml; treated as not_on_site")
                else:
                    self.error(
                        f"{rel}: not listed in tools/docs/nav.yml; add it to nav, extra_pages, hold or not_on_site"
                        " (see docs/README.md)"
                    )
        for key in ("hold", "not_on_site"):
            for pattern in getattr(self.policy, key):
                if not any(self.repo.glob(pattern)):
                    self.warn(f"nav.yml: {key} entry '{pattern}' matches no file")

    # ---- link resolution

    def ids_for(self, src: str) -> tuple[list[str], set[str]]:
        if src not in self._ids_cache:
            path = self.repo / src
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            self._ids_cache[src] = heading_ids(text)
        return self._ids_cache[src]

    def fix_anchor(self, target_src: str, anchor: str, where: str, warn_only: bool = False) -> str | None:
        """Return the anchor to use on the site, or None to drop it."""
        if not anchor:
            return None
        bases, ids = self.ids_for(target_src)
        if unquote(anchor) in ids:
            return anchor
        m = DUP_ANCHOR_RE.match(unquote(anchor))
        if m:
            base, n = m.group(1), int(m.group(2))
            if bases.count(base) > n and f"{base}_{n}" in ids:
                self.stats["anchor (duplicate heading)"] += 1
                return f"{base}_{n}"
        if warn_only or target_src.startswith(SYNCED_PREFIX):
            self.warn(f"{where}: anchor '#{anchor}' not found in {target_src}; linking to the page")
            self.stats["anchor dropped"] += 1
            return None
        return anchor  # hand-owned target: MkDocs reports a broken anchor

    def site_link(self, page_site: str, target_site: str, anchor: str | None) -> str:
        if target_site == page_site and anchor:
            return "#" + anchor
        rel = posixpath.relpath(target_site, posixpath.dirname(page_site) or ".")
        return rel + ("#" + anchor if anchor else "")

    def asset_site_path(self, rel: str) -> str:
        site = self.policy.source_to_site(rel)
        return site if site is not None else f"_assets/{rel}"

    def copy_asset(self, rel: str, where: str) -> str | None:
        site = self.asset_site_path(rel)
        if site in self.site_to_src:
            self.error(f"{where}: asset '{rel}' collides with page '{site}'")
            return None
        src = self.repo / rel
        if is_lfs_pointer(src):
            if rel not in self.lfs_pointers and not self.list_lfs:
                self.error(f"{where}: '{rel}' is a Git LFS pointer; run: git lfs pull --include='{rel}'")
            self.lfs_pointers.add(rel)
            return site
        if site not in self.copied_assets:
            dst = self.docs_out / site
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            self.copied_assets.add(site)
            self.stats["assets copied"] += 1
        return site

    def resolve(self, page_src: str, page_site: str, target: str, kind: str, line: int) -> str | None:
        """Rewrite one link target. Returns the new target, or None to render the link as plain text.

        kind is "link" or "image" (Markdown), "refdef" (reference definition),
        "html_link" (<a href>) or "html_embed" (<img|video|source src>).
        """
        where = f"{page_src}:{line}"
        synced = page_src.startswith(SYNCED_PREFIX)
        problem = self.warn if synced else self.error  # synced pages cannot be fixed here
        bracketed = target.startswith("<") and target.endswith(">")
        raw = target[1:-1] if bracketed else target
        if SCHEME_RE.match(raw) or raw.startswith("//"):
            return target
        path_q, _, anchor = raw.partition("#")
        path, _, query = path_q.partition("?")

        def wrap(t: str) -> str:
            return f"<{t}>" if bracketed or " " in t else t

        def unlink(reason: str, stat: str, report=None) -> str | None:
            """Render a link as its text (an image as its alt text); report says how loudly."""
            if kind not in ("link", "image"):
                problem(f"{where}: {reason}; only Markdown link or image syntax can be unlinked, fix the source")
                return target
            if report:
                report(f"{where}: {reason}; rendered as plain text")
            self.stats[stat] += 1
            return None

        if not path:
            if not anchor:
                return target
            fixed = self.fix_anchor(page_src, anchor, where, warn_only=synced)
            if fixed or kind not in ("link", "image"):
                return "#" + fixed if fixed else target
            return None  # anchor dropped (already reported): keep the link text

        decoded = unquote(path)
        if page_src.startswith(SITE_ONLY_PREFIX):
            # Site-only pages are written against the site layout: links are site paths.
            site = posixpath.normpath(posixpath.join(posixpath.dirname(page_site), decoded))
            if site == LLMS_TXT and kind in ("link", "refdef"):
                return target  # written by write_llms_txt
            if site in self.site_to_src and kind in ("link", "refdef"):
                fixed = self.fix_anchor(self.site_to_src[site], anchor, where)
                self.stats["site page"] += 1
                return wrap(self.site_link(page_site, site, fixed))
            self.error(f"{where}: '{target}' must be a Markdown link to a staged site page or an absolute URL")
            return target

        if decoded.startswith("/"):
            rel = posixpath.normpath(decoded.lstrip("/"))
        else:
            rel = posixpath.normpath(posixpath.join(posixpath.dirname(page_src), decoded))
        if rel == ".." or rel.startswith("../"):
            return unlink(f"'{target}' points outside the repository", "plain text (missing)", problem)
        abs_path = self.repo / rel
        if abs_path.is_dir():
            readme = "README.md" if rel == "." else f"{rel}/README.md"
            if readme not in self.src_to_site and not matches(readme, self.policy.hold):
                self.stats["GitHub"] += 1
                return github_url("" if rel == "." else rel, True, query, anchor)
            rel, abs_path = readme, self.repo / readme
        if rel in self.src_to_site:
            if kind.startswith("html"):
                problem(f"{where}: MkDocs does not rewrite raw HTML '{target}'; use Markdown link or image syntax")
                return target
            fixed = self.fix_anchor(rel, anchor, where, warn_only=synced)
            self.stats["site page"] += 1
            return wrap(self.site_link(page_site, self.src_to_site[rel], fixed))
        if matches(rel, self.policy.hold):
            return unlink(f"'{target}' points to a held page", "plain text (hold)")
        if abs_path.is_file():
            if kind == "html_embed":
                problem(f"{where}: MkDocs does not rewrite raw HTML '{target}'; use Markdown image syntax")
                return target
            if kind == "image":
                asset = self.copy_asset(rel, where)
                return wrap(posixpath.relpath(asset, posixpath.dirname(page_site) or ".")) if asset else target
            self.stats["GitHub"] += 1
            return github_url(rel, False, query, anchor)
        # A synced page removed upstream only warns, wherever the link comes from.
        report = self.warn if rel.startswith(SYNCED_PREFIX) else problem
        return unlink(f"'{target}' -> '{rel}' does not exist", "plain text (missing)", report)

    # ---- page transform

    def transform(self, src: str, site: str, text: str) -> str:
        text = strip_toc(text)
        out: list[str] = []
        line = 1
        for is_code, chunk in split_fences(text):
            if not is_code:
                chunk = self.rewrite_chunk(src, site, chunk, line)
            out.append(chunk)
            line += chunk.count("\n")
        return "".join(out)

    def rewrite_chunk(self, src: str, site: str, chunk: str, first_line: int) -> str:
        spans: list[str] = []

        def protect(m: re.Match) -> str:
            spans.append(m.group(0))
            return f"\x00{len(spans) - 1}\x00"

        protected = label_alerts(add_details_markdown(CODE_SPAN_RE.sub(protect, chunk)))

        def line_of(m: re.Match, s: str) -> int:
            return first_line + s.count("\n", 0, m.start())

        def image(m: re.Match) -> str:
            new = self.resolve(src, site, m.group(2), "image", line_of(m, protected))
            return m.group(1) if new is None else f"![{m.group(1)}]({new}{m.group(3)})"

        protected = IMAGE_RE.sub(image, protected)

        def link(m: re.Match) -> str:
            new = self.resolve(src, site, m.group(2), "link", line_of(m, protected))
            return m.group(1) if new is None else f"[{m.group(1)}]({new}{m.group(3)})"

        protected = LINK_RE.sub(link, protected)

        def refdef(m: re.Match) -> str:
            new = self.resolve(src, site, m.group(2), "refdef", line_of(m, protected))
            return m.group(1) + (new or m.group(2))

        protected = REFDEF_RE.sub(refdef, protected)

        def html(m: re.Match) -> str:
            kind = "html_link" if m.group(2).lower() == "a" else "html_embed"
            new = self.resolve(src, site, m.group(5), kind, line_of(m, protected))
            return m.group(1) + m.group(4) + (new or m.group(5)) + m.group(4)

        protected = HTML_ATTR_RE.sub(html, protected)
        return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], protected)

    # ---- agent skills and llms.txt

    def git_files(self, path: str) -> list[str] | None:
        """Files under path that Git does not ignore, or None when the repo is not the root of a Git work tree."""

        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, check=True).stdout.decode()

        try:
            if git("rev-parse", "--show-prefix").strip():
                return None  # inside another work tree, whose ignore rules do not describe this one
            listed = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", path)
        except (OSError, subprocess.CalledProcessError):
            return None
        return [f for f in listed.split("\0") if f]

    def find_skills(self) -> list[Skill]:
        """Every SKILL.md under .agents/skills/ that Git does not ignore, sorted by path."""
        files = self.git_files(SKILLS_ROOT)
        if files is None:
            root = self.repo / SKILLS_ROOT
            files = [p.relative_to(self.repo).as_posix() for p in root.rglob("SKILL.md")] if root.is_dir() else []
        skills = []
        for rel in sorted({f for f in files if f.endswith("/SKILL.md")}):
            path = self.repo / rel
            if not path.is_file():
                continue
            m = FRONTMATTER_RE.match(path.read_text(encoding="utf-8-sig"))
            try:
                meta = yaml.safe_load(m.group(1)) if m else None
            except yaml.YAMLError:
                meta = None
            meta = meta if isinstance(meta, dict) else {}
            name, description = meta.get("name"), meta.get("description")
            if not (isinstance(name, str) and isinstance(description, str)):
                self.warn(f"{rel}: no 'name' and 'description' front matter; left out of the agent skills list")
                continue
            description = " ".join(description.split())
            skills.append(Skill(name, description, summarize(description), rel))
        self.stats["agent skills"] = len(skills)
        return skills

    def skills_table(self) -> str:
        rows = ["| Skill | What it does | Folder |", "| --- | --- | --- |"]
        for s in self.skills:
            link = f"[{link_text(s.name)}]({github_url(s.path, False)})"
            rows.append(f"| {link} | {table_cell(s.summary)} | `{posixpath.dirname(s.path)}/` |")
        return "\n".join(rows)

    def finish_site_only(self, src: str, text: str) -> str:
        """Fill in the skills table and drop HTML comments, which are notes for maintainers."""
        out = []
        for is_code, chunk in split_fences(text):
            if not is_code:
                if SKILLS_MARKER_RE.search(chunk):
                    if not self.skills:
                        self.error(f"{src}: found no agent skills under {SKILLS_ROOT}/ for the skills table")
                    chunk = SKILLS_MARKER_RE.sub(lambda m: self.skills_table(), chunk)
                chunk = HTML_COMMENT_RE.sub("", chunk)
            out.append(chunk)
        return "".join(out)

    def write_llms_txt(self) -> None:
        """Write llms.txt: each nav tab as a section of links to page Markdown, then the agent skills."""
        try:
            config = yaml.load(self.base_config, Loader=BaseConfigLoader) or {}
        except yaml.YAMLError as exc:
            self.warn(f"tools/docs/mkdocs.yml: cannot read it for llms.txt ({exc}); using defaults")
            config = {}
        base = str(config.get("site_url") or "").rstrip("/")

        def url(target: str) -> str:
            if SCHEME_RE.match(target):  # a GitHub file is linked as raw text, which agents read directly
                blob = f"{GITHUB}/blob/main/"
                return RAW_GITHUB + "/" + target[len(blob) :] if target.startswith(blob) else target
            return f"{base}/{target}" if base else target

        def entries(items: list, trail: list[str]) -> list[str]:
            out = []
            for item in items:
                title, value = next(iter(item.items())) if isinstance(item, dict) else (item, item)
                title = str(title).removesuffix(" ↗")
                if isinstance(value, list):
                    out += entries(value, trail + [title])
                else:
                    out.append(f"- [{link_text(' › '.join(trail + [title]))}]({url(str(value))})")
            return out

        lines = [f"# {config.get('site_name', 'Documentation')}", ""]
        if config.get("site_description"):
            lines += [f"> {config['site_description']}", ""]
        lines += [
            "Links to this site point to each page's Markdown; skill links point to the SKILL.md on GitHub.",
            "Agent skills are step-by-step instructions for coding agents; they refer to files in the repository,",
            f"so use them from a clone of {GITHUB}.",
            "",
        ]
        nav = prune_nav(self.policy.nav, self.site_to_src)
        for item in nav:
            title, value = next(iter(item.items())) if isinstance(item, dict) else (item, item)
            lines += [f"## {str(title).removesuffix(' ↗')}", ""]
            lines += [*entries(value if isinstance(value, list) else [item], []), ""]
        if self.skills:
            lines += ["## Agent skills", ""]
            lines += [f"- [{link_text(s.name)}]({RAW_GITHUB}/{s.path}): {s.description}" for s in self.skills]
            lines.append("")
        nav_sites = set(nav_pages(nav))
        extra = sorted(s for s in self.site_to_src if s not in nav_sites)
        if extra:
            lines += ["## Optional", ""]
            for site in extra:
                title = first_heading((self.docs_out / site).read_text(encoding="utf-8")) or site
                lines.append(f"- [{link_text(title)}]({url(site)})")
            lines.append("")
        (self.docs_out / LLMS_TXT).write_text("\n".join(lines), encoding="utf-8")

    # ---- output

    def prepare_out(self) -> None:
        if self.docs_out.exists():
            if not (self.docs_out / MARKER).is_file():
                raise SystemExit(f"ERROR: {self.docs_out} exists and was not written by stage.py; move it away")
            shutil.rmtree(self.docs_out)
        self.docs_out.mkdir(parents=True)
        (self.docs_out / MARKER).write_text("Generated by tools/docs/stage.py; safe to delete.\n")

    def stage_pages(self) -> None:
        for site, src in sorted(self.site_to_src.items()):
            dst = self.docs_out / site
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = self.transform(src, site, (self.repo / src).read_text(encoding="utf-8"))
            if src.startswith(SITE_ONLY_PREFIX):
                text = self.finish_site_only(src, text)
            dst.write_text(text, encoding="utf-8")
            self.stats["pages staged"] += 1
        for name in THEME_DIRS:
            src_dir = self.repo / "tools/docs" / name
            if not src_dir.is_dir():
                continue
            if (self.docs_out / name).exists():
                self.error(f"tools/docs/{name}/ collides with a staged page or asset at {name}/")
                continue
            for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
                rel = src.relative_to(self.repo).as_posix()
                if is_lfs_pointer(src):  # *.png is in Git LFS, so the logo is too
                    if not self.list_lfs:
                        self.error(f"'{rel}' is a Git LFS pointer; run: git lfs pull --include='{rel}'")
                    self.lfs_pointers.add(rel)
                    continue
                dst = self.docs_out / src.relative_to(self.repo / "tools/docs")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def write_config(self) -> None:
        for key in GENERATED_KEYS:
            if re.search(rf"^{key}\s*:", self.base_config, re.M):
                self.error(f"tools/docs/mkdocs.yml must not set '{key}' (stage.py writes it)")
        nav = prune_nav(self.policy.nav, self.site_to_src)
        nav_sites = set(nav_pages(nav))
        extra = sorted(s for s in self.site_to_src if s not in nav_sites)
        parts = [
            self.base_config.rstrip("\n"),
            "",
            "# ---- generated by tools/docs/stage.py from tools/docs/nav.yml ----",
            f"docs_dir: {DOCS_DIR}",
            f"site_dir: {SITE_DIR}",
            yaml.safe_dump({"nav": nav}, sort_keys=False, allow_unicode=True, width=1000).rstrip("\n"),
        ]
        if extra:
            parts.append("not_in_nav: |")
            parts += [f"  /{s}" for s in extra]
        if (self.repo / HOOKS).is_file():
            try:  # MkDocs resolves hook paths against the config file's directory
                hooks = Path(os.path.relpath(self.repo / HOOKS, self.out)).as_posix()
            except ValueError:  # another drive on Windows
                hooks = (self.repo / HOOKS).as_posix()
            parts.append(yaml.safe_dump({"hooks": [hooks]}, sort_keys=False).rstrip("\n"))
        (self.out / "mkdocs.yml").write_text("\n".join(parts) + "\n", encoding="utf-8")

    def run(self) -> int:
        self.build_map()
        self.check_coverage()
        self.skills = self.find_skills()
        self.prepare_out()
        self.stage_pages()
        self.write_llms_txt()
        self.write_config()
        return 1 if self.errors else 0

    def report(self, stream) -> None:
        # On GitHub Actions, also surface each message as a workflow annotation.
        gha = os.environ.get("GITHUB_ACTIONS") == "true"
        for w in self.warnings:
            print(f"{'::warning::' if gha else ''}WARNING: {w}", file=sys.stderr)
        for e in self.errors:
            print(f"{'::error::' if gha else ''}ERROR: {e}", file=sys.stderr)
        print(f"Staged {self.stats['pages staged']} pages into {self.docs_out}", file=stream)
        for key in (
            "site page",
            "GitHub",
            "plain text (hold)",
            "plain text (missing)",
            "anchor (duplicate heading)",
            "anchor dropped",
            "assets copied",
            "agent skills",
        ):
            print(f"  {key}: {self.stats[key]}", file=stream)
        print(f"  warnings: {len(self.warnings)}, errors: {len(self.errors)}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="build", help="output directory, relative to the repo root (default: build)")
    parser.add_argument(
        "--list-lfs",
        action="store_true",
        help="print the comma-separated Git LFS pointer files that staged pages embed or the theme uses, then exit",
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    tools = repo / "tools/docs"
    out = (repo / args.out).resolve()
    try:
        policy = Policy.load(tools / "nav.yml")
    except (ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    base_config = (tools / "mkdocs.yml").read_text(encoding="utf-8")
    stager = Stager(repo, policy, base_config, out, list_lfs=args.list_lfs)
    code = stager.run()
    if args.list_lfs:
        stager.report(sys.stderr)
        print(",".join(sorted(stager.lfs_pointers)))
        return code
    stager.report(sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
