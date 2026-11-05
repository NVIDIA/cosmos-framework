# Documentation

This directory holds the Cosmos-Framework docs. The doc site at <https://nvidia.github.io/cosmos-framework/> is built from them, plus the notebook, deployment and benchmark pages under `.agents/skills/`, with MkDocs and Material for MkDocs. The site tooling lives in [`tools/docs/`](../tools/docs/README.md).

Coding agents: the [`cosmos-framework-docs`](../.agents/skills/cosmos-framework-docs/SKILL.md) skill covers editing pages and checking the build.

## Change a page

1. Edit the Markdown source. Pages stay plain GitHub-flavored Markdown; the build adapts a copy of them for the site.
2. For a new page, add it to [`tools/docs/nav.yml`](../tools/docs/nav.yml), which decides where each page appears in the site navigation and which pages stay off the site.
3. Build and preview locally (below), then open a pull request. CI builds the site and uploads a preview.

| To change                                        | Edit                                                                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| A page                                           | Its source: `docs/<path>`, or `.agents/skills/<path>` for `skills/<path>` on the site; `nav.yml` `sources` lists the exceptions |
| The navigation, or whether a page is on the site | `tools/docs/nav.yml`                                                                                                            |
| The landing page                                 | `tools/docs/site/index.md`                                                                                                      |
| The agent skills page                            | `tools/docs/site/agent_skills.md`; its table is generated                                                                       |
| Theme, colors, logo                              | `tools/docs/mkdocs.yml`, `tools/docs/stylesheets/`, `tools/docs/images/`                                                        |

## Rules

- Write sources as GitHub-flavored Markdown that also reads well on GitHub. Use relative links to files in the repository; the build turns them into site links, GitHub links or plain text.
- Never edit `docs/posttrain/`. It is synced from another repository, so problems there only warn: a page added upstream stays off the site until it is listed, a page removed upstream drops out of the nav, and links to it become plain text.
- Every page under `docs/` needs a line in `nav.yml`: in `nav`, `extra_pages`, `hold` or `not_on_site`. Otherwise the build fails.
- Keep pages concise: conclusion first, then steps.

## Build and preview

Run from the repository root:

```bash
uv run --project tools/docs python tools/docs/stage.py
uv run --project tools/docs mkdocs build --strict -f build/mkdocs.yml
uv run --project tools/docs mkdocs serve -f build/mkdocs.yml
```

- `stage.py` copies the pages into `build/docs-site/`, rewrites their links and writes `build/mkdocs.yml`. It exits non-zero on any error, and the strict build fails on any MkDocs warning. CI runs both.
- `mkdocs serve` serves pages under the `site_url` path: open <http://127.0.0.1:8000/cosmos-framework/>. It watches the staged copy, not the sources, so re-run `stage.py` after editing a page.
- Set `NO_MKDOCS_2_WARNING=true` to hide the Material for MkDocs banner about MkDocs 2.0; it is not a build warning.
- Images that a page embeds are copied next to it. `*.png` and `*.jpg` are stored in Git LFS: if a file is still an LFS pointer, run the `git lfs pull` command that `stage.py` prints.

## Deploy

The `Docs` workflow ([`.github/workflows/docs.yml`](../.github/workflows/docs.yml)) builds the site for every pull request, because a code change can break links from the docs, and for every push to `main` that touches documentation or `tools/docs/`. It uploads the site as the `docs-preview` artifact for 14 days. To preview it, unzip the artifact into a folder named `site`, run `python3 -m http.server -d site` and open <http://localhost:8000/>; opening the files directly does not work.

The site is deployed to GitHub Pages from `main` of `NVIDIA/cosmos-framework` only, on such a push or a manual run of the workflow, and only when the repository variable `DOCS_DEPLOY` is `true`. Pushes made by a workflow with `GITHUB_TOKEN` do not start workflows; after one, run the workflow manually.

One-time setup, by a repository admin (an organization owner, if the organization restricts who can publish Pages sites):

1. Confirm the NVIDIA web compliance requirements and the hosting location (see TODO).
2. In **Settings > Pages**, set **Source** to **GitHub Actions**. Do not add a suggested workflow; `docs.yml` is the Pages workflow.
3. In **Settings > Environments > github-pages**, under **Deployment branches and tags**, allow only `main`.
4. In **Settings > Secrets and variables > Actions > Variables**, add `DOCS_DEPLOY` with the value `true`.
5. In **Actions > Docs**, click **Run workflow** on `main` for the first deployment. The site appears at <https://nvidia.github.io/cosmos-framework/>.

To stop deploying, set `DOCS_DEPLOY` to `false`; the site stays as last deployed. To take it down, unpublish it in **Settings > Pages**.

## For agents

- The build writes [`llms.txt`](https://llmstxt.org/) at the site root: one section per nav tab linking each page's Markdown, an `Agent skills` section linking every `SKILL.md` on GitHub, and an `Optional` section with the pages that are not in the navigation. Its intro text is in `write_llms_txt` in `tools/docs/stage.py`.
- Each page's Markdown is published at its site path, for example `get_started/quickstart.md`, and linked from the page's `<head>`. Pages under `tools/docs/site/` are published without their HTML comments.
- The table on the Agent skills page and the skills in `llms.txt` come from the `name` and `description` front matter of every `.agents/skills/**/SKILL.md` that Git does not ignore. A new skill appears without any docs change. A `SKILL.md` without that front matter only warns.

## TODO

- Before the first public deploy: confirm NVIDIA web compliance for the site (cookie consent and analytics, as on the Cosmos Cookbook site) and where the docs should be hosted.
- Restructure `docs/` so directories mirror the site hierarchy. This needs the upstream sync destination for `docs/posttrain` changed to `docs/post_training/rl/`, about 140 inbound references updated, and redirects.
- Unify heading levels and writing style across existing pages, and write a style guide.
- Link each page to its matching agent skill, for example `training_transfer_auto.md` to `cosmos3-nano-transfer-auto-post-training`.
- AI Q&A assistant (researched separately).
- Troubleshooting index, glossary and release notes.
- Weekly external link check that does not block merges.
- Static API reference (mkdocstrings, without importing the package).
- Render the agent skill bundles.
- Switch to Zensical before Material for MkDocs maintenance ends on 2027-05-05.
