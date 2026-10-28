<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Public repository hygiene review

`check_public_hygiene.py` is an **advisory**, offline text scanner. It highlights
new references worth a release review; it does not determine whether a resource
is private, whether a public workflow reaches it, or whether a backport is safe.
It never edits files, imports Cosmos, contacts model/dependency services, or
executes notebook cells. No agent, GPU, model download, or third-party Python
dependency is required.

## Running

From the repository root, with Python 3.10 or newer:

```bash
# Compare tracked staged/unstaged changes and commits with the branch merge base.
git fetch origin main
python3 ci/check_public_hygiene.py --base origin/main

# Reproduce an exact committed PR comparison.
python3 ci/check_public_hygiene.py --base <base-sha> --head <head-sha>

# Review all tracked files, or an explicitly prepared public release directory.
python3 ci/check_public_hygiene.py --root .
python3 ci/check_public_hygiene.py --root /path/to/public-release

# Scanner-only checks, independent of the training/test environment.
python3 -m unittest discover -s ci -p 'test_public_hygiene.py' -v
```

PR CI runs on `ubuntu-latest`, with a read-only token and no repository secrets,
using `pull_request` rather than a privileged target event. It scans added lines
against the merge base with the target branch. A manual workflow run scans all
tracked files at the selected ref. This job neither gates nor replaces GPU tests,
existing lint, or Gitleaks. **Findings always exit successfully**; invalid
configuration or incomplete execution exits with status 2, and scanner-test
failures remain visible as tooling failures. No branch protection is changed.

Local diff scans omit untracked files: stage a new file to include it. Repository
tree scans also omit untracked/private worktree files; an exported directory scan
reads all its files. No pre-commit exclusions are inherited: Markdown, `.agents`,
fixtures, configuration, scripts, and notebook JSON text are included. Binary,
non-UTF-8, symlink, submodule, and files larger than 2 MiB are skipped and counted.
Git LFS payloads are not downloaded; the workflow reads their pointer files.
The scanner is not a substitute for reviewing excluded payloads at release time.

## Initial rules and their limits

| Rule                  | Review scope                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `internal-link`       | A small set of known internal development-link patterns                                                                                          |
| `internal-resource`   | Known internal service, storage, and cluster-path patterns                                                                                       |
| `dependency-source`   | GitHub Git dependency URLs outside a reviewed organization list; this does **not** classify the account as personal or the repository as private |
| `model-access-review` | A small set of known experimental/private-model naming patterns; no live access checks                                                           |
| `internal-provenance` | Explicit internal review/bug/codename provenance wording; not arbitrary usernames or identifiers                                                 |
| `release-wording`     | Contextual release-stage phrases in documentation/text, notebooks, or line comments; not a blanket ban on EA/GA identifiers                      |

Rules live in `public_hygiene_rules.json`. Public GitHub citations without a Git
installation URL are not dependency findings. Ordinary public forks, gated model
licenses, supported caller overrides, and internal compatibility branches can be
legitimate. An internal path used only as a lookup key is not proof of an active
internal dependency. The filename/line-prefix context label is only a heuristic;
it does not parse execution paths or distinguish all docstrings and shell usage
messages. Unknown private resources, arbitrary internal codenames, separate
revision fields, dynamically assembled values, and split-line URLs can be missed.
Release wording in license/attribution files is not checked; copyright and
third-party notices must be preserved.

The initial rollout emits notices, not blocking violations. Evaluate usefulness
and false positives before proposing any narrowly scoped blocking rules in a
separate review. Do not change I4 production defaults, model IDs, dependency pins,
or machine-consumed comments merely to eliminate a hint. Prefer scoped public
documentation or release-output changes when shared internal behavior must stay
unchanged. No mapping manifests are modified by this check.

## Review output and exceptions

Each hint includes its file, line, rule, heuristic context, and the SHA-256 of the
stripped source line. Logs and summaries deliberately omit matched values and
source excerpts: do not upload the internal audit report or a raw-match baseline
to the public repository. Paths themselves remain visible. GitHub output is
escaped, limited to 20 notices and 100 summary rows, and includes the total count.
Local output lists every finding as JSON; it does not print source excerpts.

For an intentional reference, add an exact exception to the rules file:

```json
{
  "rule": "internal-resource",
  "path": "relative/path/to/file.py",
  "line_sha256": "<64-character SHA-256 from the finding>",
  "owner": "<reviewing team or owner>",
  "reason": "Intentional compatibility lookup; the supported public path resolves to a public source.",
  "expires": "2027-01-31"
}
```

The containing `exceptions` array is initially empty. Owner and reason are
mandatory; expiration is optional and uses the UTC date. Expired exceptions stop
suppressing hints. Rule, exact path, and line-content fingerprint must all match,
so moving a line does not invalidate an exception but changing its content does.
Exceptions and organization-list changes need human review in the PR; there is
no directory-wide ignore or automatic baseline growth. These hygiene exceptions
do not affect Gitleaks or grant access to private resources.
