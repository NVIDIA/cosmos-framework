---
name: cosmos-framework-docs
description: >
  Edit the Cosmos-Framework documentation and its MkDocs site. Use when asked to fix,
  update, add, move or remove a docs page, find the source file behind a docs-site page
  or URL, change the site navigation, landing page, logo or styling, or preview the site
  and fix a failing docs build.
---

# Cosmos-Framework Docs

The documentation site is built from Markdown in this repository by `tools/docs/`. Sources stay plain GitHub-flavored Markdown and must read well on GitHub too. `tools/docs/stage.py` copies them to `build/docs-site/`, rewrites their links for the site and writes `build/mkdocs.yml`. Human-facing guide: [docs/README.md](../../../docs/README.md).

All paths below are relative to the repository root.

## When to use this skill

- Fix or update a docs page, including one reported from the site.
- Add, move, rename or remove a page, or change the site navigation.
- Change the landing page, logo, colors or MkDocs configuration.
- Preview the site, or fix a failing `Docs` workflow.

## Find the source of a page

Never edit `build/`: it is regenerated on every build.

1. Find the page's site path in the `nav` of [`tools/docs/nav.yml`](../../../tools/docs/nav.yml), by its title in the site navigation or by its URL: `<site>/get_started/quickstart/` is `get_started/quickstart.md`, and `<site>/models/` is `models/README.md`. A page's Markdown URL, as linked from `llms.txt`, is the site path itself: `<site>/get_started/quickstart.md`.
2. Map the site path to a repository file; the first match wins:
   - `sources` in `nav.yml`. For example, the landing page `index.md` is `tools/docs/site/index.md`, and `develop/contributing.md` is `CONTRIBUTING.md`.
   - `skills/<path>` is `.agents/skills/<path>`.
   - Anything else, `<path>`, is `docs/<path>`.

`nav` holds site paths. `extra_pages` (pages reached only through links, such as the NIM guides), `hold` and `not_on_site` hold repository paths; the last two accept globs.

## Rules

- **Never edit `docs/posttrain/`.** It is synced from an upstream repository (`.posttrain_file_mapping.json` maps each file), so local edits are overwritten. Leave the file unchanged and, in your answer, give the file, the lines and the replacement text, checked against `cosmos_framework/posttrain/`. Problems in these pages only warn in the build.
- Link to other repository files with relative Markdown links, such as `../setup.md#cuda-variants`. Do not use site URLs, which are not checked and break on GitHub, or relative raw HTML `<a href>`/`<img src>`, which `stage.py` rejects. The build turns each link into a site link when the target is on the site, a GitHub link to `main` when it is not, and plain text when the target is held.
- For notes, use GitHub alerts (`> [!NOTE]`) and `<details>`; both render on the site. Material-only syntax, such as admonitions, buttons and card grids, belongs only in the pages under `tools/docs/site/`, such as the landing page `index.md`. Those pages exist only on the site, so their links are site paths.
- Every `docs/**/*.md` must be covered by `nav.yml`, by a `nav` entry or by its repository path in `extra_pages`, `hold` or `not_on_site`, or `stage.py` fails. In `docs/posttrain/`, an unlisted file only warns and stays off the site; publishing it is a `nav.yml` edit. Files outside `docs/` are rendered only when `nav` or `extra_pages` lists them.
- Headings get GitHub-compatible anchors, so `#anchor` links work on GitHub and on the site. Renaming a heading breaks links to it: find them with `git grep -n --untracked '#<old-anchor>'`. `stage.py` fails on a link to a missing file, and the strict build fails on a missing anchor.
- Do not edit `<!--TOC-->` blocks by hand. Pre-commit regenerates them, and the site drops them.
- The content is published in the public NVIDIA/cosmos-framework repository and its site. Do not add internal hostnames, cluster paths such as `/lustre/...`, people's names, internal tickets or merge request numbers.
- Every command, flag, default and number must match the code or a source you can cite; do not copy it from another page unchecked. CLI flags come from pydantic fields through tyro: the field `checkpoint_path` is the flag `--checkpoint-path`. Read the entry point's `main()`, such as `cosmos_framework/scripts/inference.py`, to find the model it parses; inference fields are in `cosmos_framework/inference/args.py` and `cosmos_framework/inference/common/args.py`, and checkpoint names and sizes in `_CHECKPOINTS` in `cosmos_framework/inference/args.py`.
- Keep pages short: the conclusion first, then steps, and tables for choices. Model a new page on `docs/get_started/quickstart.md`; Markdown pages need no SPDX header. Keep a page's H1 unless asked to change it.

## Common changes

| Change                        | Files to edit                                                                                                                                                                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text on a page                | The page's source                                                                                                                                                                                                                                                                 |
| Add a page                    | A new file in `docs/get_started/`, `docs/models/`, `docs/deploy/` or `docs/post_training/` when it belongs there, otherwise in `docs/`; its `nav` entry; a link from a related page                                                                                               |
| Rename or move a page         | The file (`git mv`); its own relative links if its directory changed; its `nav.yml` entries; every inbound link (`git grep -n --untracked '<old file name>'`). The old URL is not redirected                                                                                      |
| Take a page off the site      | Delete its `nav` entry. For a page in `docs/`, also add its repository path to `not_on_site` (links go to GitHub) or `hold` (links become plain text)                                                                                                                             |
| Nav label or order            | `tools/docs/nav.yml` only                                                                                                                                                                                                                                                         |
| Landing page                  | `tools/docs/site/index.md`                                                                                                                                                                                                                                                        |
| Agent skills page, `llms.txt` | Page text: `tools/docs/site/agent_skills.md`. `llms.txt` text: `write_llms_txt` in `tools/docs/stage.py`; its page sections follow `nav`. The skills table and the skills in `llms.txt` come from each `SKILL.md`'s `name` and `description`, so a new skill needs no docs change |
| Logo, colors, MkDocs settings | `tools/docs/images/`, `tools/docs/stylesheets/extra.css`, `tools/docs/mkdocs.yml` (it must not set `docs_dir`, `site_dir`, `nav`, `not_in_nav` or `hooks`)                                                                                                                        |
| Build behavior                | `tools/docs/stage.py`, with a test in `tools/docs/tests/`                                                                                                                                                                                                                         |

## Build and check

Run from the repository root, in this order, because pre-commit may rewrite files:

```bash
uvx pre-commit@4.5.1 run --files <changed files>
uvx pre-commit@4.5.1 run -c ci/.pre-commit-config-base.yaml --files <changed files>
uv run --project tools/docs python tools/docs/stage.py
uv run --project tools/docs mkdocs build --strict -f build/mkdocs.yml
```

- Pre-commit formats Markdown and regenerates TOCs. Keep its changes, fix by hand what it reports but cannot fix, and run it again until it passes. It skips `.agents/skills/cookbooks/` and `.agents/skills/evaluation/`; for those pages only `stage.py` and the strict build check your edit.
- `stage.py` prints each WARNING and ERROR with the file, often with the fix, ends with a `warnings: N, errors: N` line, and exits non-zero on any error. Fix every error and every warning about a file you changed.
- Judge the strict build by its exit code. Its red banner about MkDocs 2.0 is not a warning; `NO_MKDOCS_2_WARNING=true` hides it.
- Embedded images are copied next to their page. `*.png` and `*.jpg` are stored in Git LFS: if `stage.py` reports an LFS pointer, run the `git lfs pull` command it prints.
- If you changed `tools/docs/stage.py`, also run `uv run --project tools/docs pytest -c tools/docs/pyproject.toml --confcutdir=tools/docs tools/docs/tests`.
- To preview, run `uv run --project tools/docs mkdocs serve -f build/mkdocs.yml -a 127.0.0.1:8000`. Pages are under the `site_url` path, so `get_started/quickstart.md` is <http://127.0.0.1:8000/cosmos-framework/get_started/quickstart/>. If the port is taken, another preview may be running: leave it alone, use another port, and stop your own server when you finish. The server reads the staged copy, so re-run `stage.py` after each source edit. Without a browser, read `build/site/<site path without .md>/index.html` after the strict build.
- The commands write git-ignored output: `build/`, `.rumdl_cache/` and `tools/docs/.venv/`. Do not commit them.

The change is done when pre-commit passes, `stage.py` reports no errors and no warnings about your files, the strict build passes, and the built page shows your change with working links.
