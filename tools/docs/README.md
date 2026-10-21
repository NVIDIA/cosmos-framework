# Doc site tooling

How to write, build, preview and deploy the docs: [`docs/README.md`](../../docs/README.md). This directory holds the tooling that builds the site.

| File                     | Purpose                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `nav.yml`                | Navigation and publishing policy: which pages are on the site, held, or repo only              |
| `stage.py`               | Copies pages into `build/docs-site/`, rewrites links, writes `llms.txt` and `build/mkdocs.yml` |
| `mkdocs.yml`             | Base MkDocs config, without `nav` or `docs_dir`                                                |
| `site/index.md`          | Landing page; it exists only on the site, so its links are site paths                          |
| `site/agent_skills.md`   | Agent skills page; `stage.py` fills in its table from the `SKILL.md` files                     |
| `hooks.py`               | MkDocs hook that publishes each page's Markdown next to its HTML                               |
| `stylesheets/extra.css`  | Palette, logo size and table overflow                                                          |
| `images/cosmos-logo.png` | Logo and favicon; `stage.py` copies `stylesheets/` and `images/` to the site root              |
| `tests/`                 | Tests for `stage.py` and `hooks.py`                                                            |
| `pyproject.toml`         | Standalone uv project with pinned versions; not part of the root project                       |

Run the tests from the repository root:

```bash
uv run --project tools/docs pytest -c tools/docs/pyproject.toml --confcutdir=tools/docs tools/docs/tests
```

`*.png` files are stored in Git LFS, including the logo. CI checks out without LFS and pulls only the files that `stage.py --list-lfs` reports: the images that pages embed and the files in `images/`.
