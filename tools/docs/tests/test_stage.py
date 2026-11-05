# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for tools/docs/stage.py. See tools/docs/README.md for the command."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stage  # noqa: E402

GH = "https://github.com/NVIDIA/cosmos-framework"
BASE_CONFIG = "site_name: Test\n"


def write(root: Path, rel: str, text: str | bytes = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    write(root, "docs/guide.md", "# Guide\n\n## Setup\n\nOne.\n\n## Setup\n\nTwo.\n\n## Setup\n\nThree.\n")
    write(root, "docs/other.md", "# Other\n")
    write(root, "docs/held.md", "# Held\n")
    write(root, "docs/repo_only.md", "# Repo only\n")
    write(root, "docs/img/diagram.png", b"\x89PNG fake")
    write(root, "docs/posttrain/index.md", "# RL\n\n## Quick start\n")
    write(root, "scripts/run.py", "print()\n")
    write(root, ".agents/skills/cookbook/README.md", "# Cookbook\n")
    write(root, ".agents/skills/cookbook/extra.md", "# Extra\n")
    return root


def make_stager(root: Path, page: str | None = None, src: str = "docs/page.md", **kwargs) -> stage.Stager:
    if page is not None:
        write(root, src, page)
    policy = stage.Policy(
        nav=[
            {"Guide": "guide.md"},
            {"Other": "other.md"},
            {"RL": "posttrain/index.md"},
            {"Cookbook": "skills/cookbook/README.md"},
        ],
        sources={"index.md": "tools/docs/site/index.md"},
        extra_pages=[".agents/skills/cookbook/extra.md"],
        hold=["docs/held.md"],
        not_on_site=["docs/repo_only.md"],
    )
    site = policy.source_to_site(src)
    if site not in stage.nav_pages(policy.nav) and src not in policy.extra_pages:
        policy.nav.insert(0, {"Page": site})
    return stage.Stager(root, policy, BASE_CONFIG, root / "build", **kwargs)


def run(root: Path, page: str, src: str = "docs/page.md", **kwargs) -> tuple[stage.Stager, str]:
    stager = make_stager(root, page, src, **kwargs)
    assert stager.run() == (1 if stager.errors else 0)  # warnings alone never fail the build
    site = stager.src_to_site[src]
    return stager, (root / "build/docs-site" / site).read_text(encoding="utf-8")


# ---------------------------------------------------------------- simple transforms


def test_strip_toc_removes_block_and_keeps_line_numbers() -> None:
    text = "# T\n<!--TOC-->\n- [A](#a)\n<!--TOC-->\nbody\n"
    out = stage.strip_toc(text)
    assert "TOC" not in out and "[A]" not in out
    assert out.count("\n") == text.count("\n")
    assert out.endswith("body\n")


def test_details_get_markdown_attribute() -> None:
    text = '<details><summary>a</summary>\n\n<details open>\n<details markdown="1">\n'
    out = stage.add_details_markdown(text)
    assert '<details markdown="1"><summary>' in out
    assert '<details open markdown="1">' in out
    assert out.count("markdown=") == 3


def test_details_inside_code_fence_untouched(repo: Path) -> None:
    _, out = run(repo, "# P\n\n```html\n<details>\n```\n\n<details>\n")
    assert "```html\n<details>\n```" in out
    assert out.rstrip().endswith('<details markdown="1">')


def test_details_and_alerts_inside_code_span_untouched(repo: Path) -> None:
    _, out = run(repo, "# P\n\nUse `<details>` and `> [!NOTE]`.\n")
    assert "`<details>`" in out and "`> [!NOTE]`" in out


def test_github_alert_marker_becomes_label() -> None:
    out = stage.label_alerts("> [!NOTE]\n> Text.\n\n> [!important]  \n> More.\n\n> [!NOTE] inline\n")
    assert out == "> **Note**\n> Text.\n\n> **Important**\n> More.\n\n> [!NOTE] inline\n"


# ---------------------------------------------------------------- link rewriting


def test_link_to_staged_page_becomes_site_link(repo: Path) -> None:
    _, out = run(
        repo, "# P\n\n[g](guide.md#setup) [c](../.agents/skills/cookbook/README.md) [d](../.agents/skills/cookbook/)\n"
    )
    assert "[g](guide.md#setup)" in out
    assert "[c](skills/cookbook/README.md)" in out
    assert "[d](skills/cookbook/README.md)" in out


def test_link_from_nested_page_is_relative(repo: Path) -> None:
    _, out = run(
        repo,
        "# E\n\n[g](../../../docs/guide.md) [rl](../../../docs/posttrain/index.md)\n",
        src=".agents/skills/cookbook/extra.md",
    )
    assert "[g](../../guide.md)" in out
    assert "[rl](../../posttrain/index.md)" in out


def test_link_to_held_page_becomes_plain_text(repo: Path) -> None:
    stager, out = run(repo, "# P\n\nSee [the held page](held.md#x) now.\n")
    assert "See the held page now." in out
    assert stager.stats["plain text (hold)"] == 1
    assert not stager.errors and not stager.warnings


def test_link_to_dir_with_held_readme_becomes_plain_text(repo: Path) -> None:
    write(repo, "docs/sub/README.md", "# Sub\n")
    stager = make_stager(repo, "# P\n\nSee [sub](sub/).\n")
    stager.policy.hold.append("docs/sub/README.md")
    stager.run()
    out = (repo / "build/docs-site/page.md").read_text(encoding="utf-8")
    assert "See sub." in out
    assert not stager.errors


def test_reference_definition_to_held_page_is_error(repo: Path) -> None:
    stager, out = run(repo, "# P\n\n[x][h]\n\n[h]: held.md\n")
    assert "[h]: held.md" in out
    assert any("held page" in e for e in stager.errors)


def test_link_to_repo_file_goes_to_github(repo: Path) -> None:
    _, out = run(repo, "# P\n\n[r](repo_only.md#top) [s](../scripts/run.py) [dir](../scripts) [root](../)\n")
    assert f"[r]({GH}/blob/main/docs/repo_only.md#top)" in out
    assert f"[s]({GH}/blob/main/scripts/run.py)" in out
    assert f"[dir]({GH}/tree/main/scripts)" in out
    assert f"[root]({GH}/tree/main/)" in out


def test_external_and_code_links_untouched(repo: Path) -> None:
    page = "# P\n\n[x](https://example.com/a.md) `[y](missing.md)`\n\n```\n[z](missing.md)\n```\n"
    stager, out = run(repo, page)
    assert out == page
    assert not stager.errors


def test_missing_target_in_hand_owned_page_is_error(repo: Path) -> None:
    stager, _ = run(repo, "# P\n\n[m](missing.md)\n")
    assert any("missing.md" in e for e in stager.errors)


def test_missing_target_in_synced_page_warns_and_drops_link(repo: Path) -> None:
    stager, out = run(repo, "# RL\n\n## Quick start\n\nSee [gone](howto/gone.md).\n", src="docs/posttrain/index.md")
    assert "See gone." in out
    assert not stager.errors
    assert any("gone.md" in w for w in stager.warnings)


def test_synced_page_problems_only_warn(repo: Path) -> None:
    page = (
        "# RL\n\n## Quick start\n\n"
        '<a href="../guide.md">g</a> ![m](missing.png) [s](#renamed) [o](../guide.md#nope) [i](../held.md)\n'
    )
    stager, out = run(repo, page, src="docs/posttrain/index.md")
    assert not stager.errors
    assert '<a href="../guide.md">g</a>' in out  # left for the owner of the synced page
    assert " m s [o](../guide.md) i\n" in out
    assert len(stager.warnings) == 4  # raw HTML, image, two anchors; held links are expected


def test_synced_page_removed_upstream_only_warns(repo: Path) -> None:
    stager = make_stager(repo, "# P\n\nSee [old](posttrain/guide/old.md).\n")
    stager.policy.nav.append({"RL extra": [{"Old": "posttrain/guide/old.md"}]})
    assert stager.run() == 0
    assert any("posttrain/guide/old.md" in w and "left off the site" in w for w in stager.warnings)
    out = (repo / "build/docs-site/page.md").read_text(encoding="utf-8")
    assert "See old." in out
    config = yaml.safe_load((repo / "build/mkdocs.yml").read_text(encoding="utf-8"))
    assert all("RL extra" not in item for item in config["nav"])


def test_image_is_copied_next_to_page(repo: Path) -> None:
    stager, out = run(repo, "# P\n\n![d](img/diagram.png)\n")
    assert "![d](img/diagram.png)" in out
    assert (repo / "build/docs-site/img/diagram.png").read_bytes() == b"\x89PNG fake"
    assert stager.stats["assets copied"] == 1


def test_theme_dirs_are_copied_to_site_root(repo: Path) -> None:
    write(repo, "tools/docs/images/logo.png", b"\x89PNG logo")
    write(repo, "tools/docs/stylesheets/extra.css", "a {}\n")
    stager, _ = run(repo, "# P\n")
    assert not stager.errors
    assert (repo / "build/docs-site/images/logo.png").read_bytes() == b"\x89PNG logo"
    assert (repo / "build/docs-site/stylesheets/extra.css").is_file()


def test_lfs_pointer_in_theme_dir_is_error_unless_listing(repo: Path) -> None:
    write(repo, "tools/docs/images/logo.png", stage.LFS_POINTER + b"\noid sha256:abc\nsize 1\n")
    stager, _ = run(repo, "# P\n")
    assert any("git lfs pull --include='tools/docs/images/logo.png'" in e for e in stager.errors)
    assert not (repo / "build/docs-site/images/logo.png").exists()
    lister, _ = run(repo, "# P\n", list_lfs=True)
    assert lister.lfs_pointers == {"tools/docs/images/logo.png"}
    assert not lister.errors


def test_theme_dir_colliding_with_page_asset_is_error(repo: Path) -> None:
    write(repo, "tools/docs/images/logo.png", b"\x89PNG logo")
    write(repo, "docs/images/shot.png", b"\x89PNG shot")
    stager, _ = run(repo, "# P\n\n![s](images/shot.png)\n")
    assert any("tools/docs/images/ collides" in e for e in stager.errors)


def test_lfs_pointer_image_is_error_unless_listing(repo: Path) -> None:
    write(repo, "docs/img/lfs.png", stage.LFS_POINTER + b"\noid sha256:abc\nsize 1\n")
    stager, _ = run(repo, "# P\n\n![l](img/lfs.png)\n")
    assert any("git lfs pull --include='docs/img/lfs.png'" in e for e in stager.errors)
    lister, _ = run(repo, "# P\n\n![l](img/lfs.png)\n", list_lfs=True)
    assert lister.lfs_pointers == {"docs/img/lfs.png"}
    assert not lister.errors


def test_lfs_pointer_reported_once_and_not_copied(repo: Path) -> None:
    write(repo, "docs/img/lfs.png", stage.LFS_POINTER + b"\noid sha256:abc\nsize 1\n")
    stager, _ = run(repo, "# P\n\n![a](img/lfs.png) ![b](img/lfs.png)\n")
    assert len(stager.errors) == 1
    assert not (repo / "build/docs-site/img/lfs.png").exists()


def test_raw_html_link_to_site_page_is_error(repo: Path) -> None:
    stager, _ = run(repo, '# P\n\n<a href="guide.md">g</a>\n')
    assert any("raw HTML" in e for e in stager.errors)


def test_site_only_page_uses_site_paths(repo: Path) -> None:
    stager, out = run(
        repo, "# Home\n\n[g](guide.md#setup-1) [c](skills/cookbook/README.md)\n", src="tools/docs/site/index.md"
    )
    assert "[g](guide.md#setup_1)" in out
    assert "[c](skills/cookbook/README.md)" in out
    assert not stager.errors


# ---------------------------------------------------------------- anchors


def test_github_duplicate_heading_anchor_rewritten(repo: Path) -> None:
    _, out = run(repo, "# P\n\n[a](guide.md#setup-1) [b](guide.md#setup-2) [c](guide.md#setup-3)\n")
    assert "[a](guide.md#setup_1)" in out
    assert "[b](guide.md#setup_2)" in out
    assert "[c](guide.md#setup-3)" in out  # only three headings: left for MkDocs to report


def test_same_page_duplicate_anchor_rewritten(repo: Path) -> None:
    _, out = run(repo, "# P\n\n[n](#notes-1)\n\n## Notes\n\n## Notes\n")
    assert "[n](#notes_1)" in out


def test_missing_anchor_into_synced_page_degrades_to_page_link(repo: Path) -> None:
    stager, out = run(repo, "# P\n\n[q](posttrain/index.md#quick-start) [x](posttrain/index.md#renamed)\n")
    assert "[q](posttrain/index.md#quick-start)" in out
    assert "[x](posttrain/index.md)" in out
    assert any("#renamed" in w for w in stager.warnings)
    assert not stager.errors


def test_missing_anchor_into_hand_owned_page_is_kept(repo: Path) -> None:
    stager, out = run(repo, "# P\n\n[x](guide.md#nope)\n")
    assert "[x](guide.md#nope)" in out
    assert not stager.warnings


def test_heading_ids_match_python_markdown() -> None:
    bases, ids = stage.heading_ids("# A `b_c` [d](e.md)\n\n## Q&A: **x**\n\n## Q&A: **x**\n\nT\n---\n")
    assert bases == ["a-b_c-d", "qa-x", "qa-x", "t"]
    assert ids == {"a-b_c-d", "qa-x", "qa-x_1", "t"}


# ---------------------------------------------------------------- coverage and config


def test_unlisted_hand_owned_doc_is_error(repo: Path) -> None:
    write(repo, "docs/new_page.md", "# New\n")
    stager, _ = run(repo, "# P\n")
    assert any("docs/new_page.md" in e for e in stager.errors)


def test_unlisted_synced_doc_only_warns(repo: Path) -> None:
    write(repo, "docs/posttrain/howto/new.md", "# New\n")
    stager, _ = run(repo, "# P\n")
    assert not stager.errors
    assert any("docs/posttrain/howto/new.md" in w for w in stager.warnings)


def test_listed_page_in_hold_is_error(repo: Path) -> None:
    stager = make_stager(repo, "# P\n")
    stager.policy.hold.append("docs/guide.md")
    assert stager.run() == 1
    assert any("also in hold/not_on_site" in e for e in stager.errors)


def test_stale_hold_pattern_warns(repo: Path) -> None:
    stager = make_stager(repo, "# P\n")
    stager.policy.not_on_site.append("docs/gone-*.md")
    assert stager.run() == 0
    assert any("docs/gone-*.md" in w for w in stager.warnings)


def test_missing_nav_source_is_error(repo: Path) -> None:
    stager = make_stager(repo)  # docs/page.md does not exist
    assert stager.run() == 1
    assert any("docs/page.md" in e for e in stager.errors)


def test_page_listed_twice_is_error(repo: Path) -> None:
    stager = make_stager(repo, "# P\n")
    stager.policy.nav.append({"Again": "guide.md"})
    stager.policy.sources["alias.md"] = "docs/other.md"
    stager.policy.nav.append({"Alias": "alias.md"})
    assert stager.run() == 1
    assert any("'guide.md' appears 2 times" in e for e in stager.errors)
    assert any("'docs/other.md' is staged twice" in e for e in stager.errors)


def test_malformed_nav_or_unknown_key_is_rejected(tmp_path: Path) -> None:
    for text in ("nav: [{A: a.md, B: b.md}]\n", "nav: []\nhidden: []\n"):
        (tmp_path / "nav.yml").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            stage.Policy.load(tmp_path / "nav.yml")


def test_prune_nav_drops_missing_pages_and_empty_sections() -> None:
    nav = [{"A": "a.md"}, {"S": [{"B": "b.md"}, {"T": [{"C": "c.md"}]}]}, {"Ext": "https://x.org"}, "d.md"]
    assert stage.prune_nav(nav, {"a.md": "", "b.md": ""}) == [
        {"A": "a.md"},
        {"S": [{"B": "b.md"}]},
        {"Ext": "https://x.org"},
    ]


def test_config_has_nav_and_not_in_nav(repo: Path) -> None:
    stager, _ = run(repo, "# P\n")
    assert not stager.errors
    config_text = (repo / "build/mkdocs.yml").read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    assert config["site_name"] == "Test"
    assert config["docs_dir"] == "docs-site"
    assert config["nav"][1] == {"Guide": "guide.md"}
    assert config["nav"][0] == {"Page": "page.md"}
    assert config["not_in_nav"].split() == ["/skills/cookbook/extra.md"]


def test_config_lists_hooks_when_present(repo: Path) -> None:
    write(repo, "tools/docs/hooks.py", "")
    run(repo, "# P\n")
    config = yaml.safe_load((repo / "build/mkdocs.yml").read_text(encoding="utf-8"))
    assert config["hooks"] == ["../tools/docs/hooks.py"]


def test_hooks_publish_page_markdown(tmp_path: Path) -> None:
    import hooks

    write(tmp_path, "docs/a/page.md", "# Page\n")
    write(tmp_path, "docs/a/img.png", b"\x89PNG")
    hooks.on_post_build({"docs_dir": str(tmp_path / "docs"), "site_dir": str(tmp_path / "site")})
    assert (tmp_path / "site/a/page.md").read_text(encoding="utf-8") == "# Page\n"
    assert not (tmp_path / "site/a/img.png").exists()


def test_hooks_link_markdown_from_head() -> None:
    import hooks

    def page(src_uri: str, url: str) -> SimpleNamespace:
        return SimpleNamespace(file=SimpleNamespace(src_uri=src_uri), url=url)

    html = "<head></head>"
    out = hooks.on_post_page(html, page("get_started/quickstart.md", "get_started/quickstart/"), {})
    assert out == '<head><link rel="alternate" type="text/markdown" href="../quickstart.md">\n</head>'
    assert 'href="index.md"' in hooks.on_post_page(html, page("index.md", ""), {})
    assert 'href="README.md"' in hooks.on_post_page(html, page("models/README.md", "models/"), {})


# ---------------------------------------------------------------- agent skills and llms.txt

SKILL = "---\nname: {name}\ndescription: >\n  {description}\n---\n\n# Skill\n"


def test_skills_table_replaces_marker_and_comments_are_dropped(repo: Path) -> None:
    write(repo, ".agents/skills/alpha/SKILL.md", SKILL.format(name="alpha", description="Use A | `x|y` <dir>. More."))
    write(repo, ".agents/skills/group/beta/SKILL.md", SKILL.format(name="beta", description="Do B."))
    page = "<!-- maintainer note -->\n# Skills\n\n<!-- agent-skills-table -->\n\n```text\n<!-- agent-skills-table -->\n```\n"
    stager, out = run(repo, page, src="tools/docs/site/index.md")
    assert not stager.errors
    row = f"| [alpha]({GH}/blob/main/.agents/skills/alpha/SKILL.md) | Use A \\| `x|y` &lt;dir>. | `.agents/skills/alpha/` |"
    assert row in out
    assert "| [beta](" in out and "`.agents/skills/group/beta/` |" in out
    assert "maintainer note" not in out
    assert "```text\n<!-- agent-skills-table -->\n```" in out  # code is left alone
    assert stager.stats["agent skills"] == 2


def test_skills_marker_without_skills_is_error(repo: Path) -> None:
    stager, _ = run(repo, "# Skills\n\n<!-- agent-skills-table -->\n", src="tools/docs/site/index.md")
    assert any("found no agent skills" in e for e in stager.errors)


def test_skill_with_bom_is_read(repo: Path) -> None:
    write(repo, ".agents/skills/alpha/SKILL.md", "﻿" + SKILL.format(name="alpha", description="Do A."))
    stager, _ = run(repo, "# P\n")
    assert [s.name for s in stager.skills] == ["alpha"]


def test_skill_without_front_matter_warns(repo: Path) -> None:
    write(repo, ".agents/skills/broken/SKILL.md", "# No front matter\n")
    stager, _ = run(repo, "# P\n")
    assert not stager.errors
    assert any(".agents/skills/broken/SKILL.md" in w for w in stager.warnings)
    assert stager.skills == []


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_skills_ignored_by_git_are_skipped(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    write(repo, ".gitignore", "packages/\n")
    write(repo, ".agents/skills/alpha/SKILL.md", SKILL.format(name="alpha", description="Do A."))
    write(repo, ".agents/skills/packages/vendor/SKILL.md", SKILL.format(name="vendor", description="Vendored."))
    stager, _ = run(repo, "# P\n")
    assert [s.name for s in stager.skills] == ["alpha"]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_skills_found_when_repo_is_ignored_by_an_outer_repo(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    write(outer, ".gitignore", "repo/\n")
    write(outer, "repo/docs/guide.md", "# Guide\n")
    write(outer, "repo/.agents/skills/alpha/SKILL.md", SKILL.format(name="alpha", description="Do A."))
    stager = stage.Stager(outer / "repo", stage.Policy(nav=[{"Guide": "guide.md"}]), BASE_CONFIG, outer / "repo/build")
    assert stager.run() == 0
    assert [s.name for s in stager.skills] == ["alpha"]


def test_summarize_keeps_first_sentence_within_limit() -> None:
    assert stage.summarize("Install it, e.g. with uv. Use when asked.") == "Install it, e.g. with uv."
    long = stage.summarize("word " * 60 + "end.", limit=40)
    assert len(long) <= 40 and long.endswith("word…")
    assert stage.summarize("word " * 6 + "run `uv run pytest -q` first.", limit=40) == "word word word word word word run…"


def test_llms_txt_lists_tabs_skills_and_optional_pages(repo: Path) -> None:
    write(repo, ".agents/skills/alpha/SKILL.md", SKILL.format(name="alpha", description="Do A. More."))
    stager = make_stager(repo, "# P\n")
    stager.base_config = "site_name: Test\nsite_description: A test site.\nsite_url: https://x.org/docs/\n"
    stager.policy.nav.append(
        {"Section [x]": [{"Sub": [{"Deep": "other.md"}]}, {"Skill ↗": f"{GH}/blob/main/.agents/skills/a/SKILL.md"}]}
    )
    stager.policy.nav.remove({"Other": "other.md"})
    write(repo, ".agents/skills/cookbook/extra.md", "# Extra [page](README.md) { #extra }\n")
    assert stager.run() == 0
    text = (repo / "build/docs-site/llms.txt").read_text(encoding="utf-8")
    assert text.startswith("# Test\n\n> A test site.\n")
    assert "## Guide\n\n- [Guide](https://x.org/docs/guide.md)\n" in text
    assert "- [Sub › Deep](https://x.org/docs/other.md)\n" in text
    assert f"- [Skill]({stage.RAW_GITHUB}/.agents/skills/a/SKILL.md)\n" in text
    assert f"- [alpha]({stage.RAW_GITHUB}/.agents/skills/alpha/SKILL.md): Do A. More.\n" in text
    assert text.index("## Agent skills") < text.index("## Optional")
    assert "- [Extra page](https://x.org/docs/skills/cookbook/extra.md)" in text


def test_llms_txt_tolerates_mkdocs_tags_and_escapes_link_text(repo: Path) -> None:
    stager = make_stager(repo, "# P\n")
    stager.base_config = "site_name: Test\nsite_url: !ENV [SITE_URL, 'https://x.org/']\n"
    stager.policy.nav.append({"Odd ] title ↗": [{"x]": "other.md"}]})
    stager.policy.nav.remove({"Other": "other.md"})
    assert stager.run() == 0
    assert not stager.warnings
    text = (repo / "build/docs-site/llms.txt").read_text(encoding="utf-8")
    assert "## Odd ] title\n\n- [x\\]](other.md)\n" in text


def test_site_only_page_may_link_llms_txt(repo: Path) -> None:
    stager, out = run(repo, "# Home\n\n[l](llms.txt)\n", src="tools/docs/site/index.md")
    assert not stager.errors
    assert "[l](llms.txt)" in out


def test_base_config_must_not_set_generated_keys(repo: Path) -> None:
    write(repo, "docs/page.md", "# P\n")
    stager = make_stager(repo)
    stager.base_config = "site_name: Test\nnav: []\n"
    assert stager.run() == 1
    assert any("'nav'" in e for e in stager.errors)
