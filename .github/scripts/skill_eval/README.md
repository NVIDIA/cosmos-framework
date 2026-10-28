# Skill-eval CI orchestrator

`run_matrix.py` runs the cosmos skill-eval matrix (eval cases × agent backends) for one PR head on the
self-hosted GPU runners and keeps **one** PR comment up to date while it runs. In CI every cell is its own
GitHub Actions job, so GitHub queues the cells over all idle runners (three H100 nodes = three cells in
flight); a `plan` job posts the comment first and a `report` job finalises it. Python 3.10+ standard
library only (the runners have 3.10.12); settings live in [`.github/skill-eval.yml`](../../skill-eval.yml).

## What triggers it

Never automatically. `.github/workflows/skill-eval.yml` starts a run when someone

- applies a trigger label to a PR (works before the workflow file is on `main`; the labels are removed again
  at the end so they can be re-applied): `run-skill-eval` (default run: the touched skills, no canary),
  `run-skill-eval-smoke` (canary only, = token `smoke-only`), `run-skill-eval-failed` (re-run the cells that did
  not PASS, see [Re-running](#re-running-cache-semantics)) or `run-skill-eval-force` (re-run every cell),
- comments `/run-skill-eval [arguments]` on a PR (write, maintain or admin role on the repository, see below; needs the
  file on `main`) — the
  comment's first line, anything after `/run-skill-eval` is parsed as [trigger arguments](#trigger-arguments), or
- dispatches the workflow manually with a PR number, a scope (`full`, or `smoke` = token `smoke-only`) and optional
  `args` (the same trigger arguments as in a comment, e.g. `with-smoke`).

**Who may start it.** Every trigger passes a GitHub-hosted `gate` job first (no checkout, so nothing from the PR runs
before it): it reads the role of the person who commented, applied the label or dispatched
(`GET /repos/<owner>/<repo>/collaborators/<login>/permission`) and accepts `write`, `maintain` or `admin` only.
Organisation membership, a read or triage role, or a PR from a fork is refused — a refused comment gets a 😕 reaction
and the run ends before a self-hosted runner is touched; `plan` refuses fork heads a second time. (Applying a label
needs triage access on GitHub anyway; the gate still requires write.)

After the gate the workflow has four stages, all on self-hosted runners (label `cosmos-skill-eval`; cells also need `h100`, and
cursor cells `cursor-serial` — see [Runners](#runners)):

1. **`plan`** checks out the PR head (full history — every branch, so the base commit and the merge base that the
   git fallback of the skill detection needs are already in the clone) and runs
   `run_matrix.py plan --trigger-args "$TRIGGER_ARGS"`: it detects the skills,
   builds the cells, works out each skill's version and dashboard group, asks the dashboard which cells already
   have a reusable result, posts the **one** comment (everything queued / reused / not selected) and emits two
   job matrices as outputs (`canary`, `skills`: the cells that still have to run, each entry with the job's
   `runs_on` labels) plus the plan itself (`plan`, JSON). The label's meaning, the comment text after
   `/run-skill-eval` or the dispatch inputs are turned into `TRIGGER_ARGS` in that step (shell/python, not in the
   job's `if`).
2. **`canary`** — one job per canary cell, `strategy.matrix.include` from the plan; skipped entirely unless the
   canary was asked for (see [Which skills run](#which-skills-run)).
3. **`cells`** — one job per skill cell; `needs: canary`, so canary cells run first (ordering only, not gating).
4. **`report`** (`if: always()`) downloads the per-cell result artifacts and runs `run_matrix.py report
   --results-dir …`, which writes the final comment from them (then the dashboard, the comment ledger and the
   job list for anything a cell job did not record). Before declaring a finished-but-unrecorded cell an error it
   waits up to 4 minutes for the dashboard to list it (the publisher refreshes the dashboard right after each
   upload; this covers the case where that call failed and the catalog's ~2-minute re-list has to catch up).
5. **`labels`** (`if: always()`, GitHub-hosted) removes the four trigger labels — also when `plan` failed or the
   run was cancelled, so a label can always be re-applied.

Concurrency is per PR at the `plan` job (later triggers queue behind a running plan; an unrelated label or
comment event never reaches that job and therefore cannot cancel a pending run). `cells` does not start when
the run was cancelled during the canary phase. Fork PRs are refused on every trigger path. The job-level
timeout (200 min) is above the orchestrator's 3 h cell timeout, so a hung cell is stopped by the launcher
(SIGTERM into the container) rather than by GitHub killing the job and orphaning the container.

Each cell job checks out the PR head and runs
`run_matrix.py cell --plan "$PLAN_JSON" --skill … --backend … --case …`. The job is named
`cell: <skill>/<backend>/<case>` — that name is how the comment maps GitHub's job states back to cells.
Environment set by the workflow: `GH_TOKEN` (the job's `GITHUB_TOKEN`: `pull-requests: write`,
`actions: read`), `GITHUB_REPOSITORY`, `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT`, and for `plan` also `PR_NUMBER`,
`HEAD_SHA` (full), `BASE_SHA` (the PR's `base.sha`, full), `HEAD_REF`. The checkout **must** be at `HEAD_SHA`: the
publisher records `git rev-parse HEAD` of the mounted checkout as the run's commit, one of the two keys the
cache matches on (the other is the skill's content hash, see below).
`run_matrix.py run` (the default mode) still runs the whole matrix sequentially in one process — handy locally —
with the same trigger arguments, cache rule and groups.

### Secrets

| secret                             | purpose                                                                                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COSMOS_SKILL_EVAL_NVIDIA_API_KEY` | The NVIDIA inference-gateway key that drives the claude / codex / opencode agents and the judge inside the evaluator container (fallback: `NVIDIA_API_KEY`, see below). |

Which secret is read: the "Run the cell" step env is `secrets.COSMOS_SKILL_EVAL_NVIDIA_API_KEY || secrets.NVIDIA_API_KEY`
— the namespaced secret when it is set, otherwise the repository's older `NVIDIA_API_KEY` secret (set 2026-10-12,
referenced by no other workflow). Since image `v20261031` (2026-10-26) the evaluator image does **not** bake the key,
so only CI and deliberate local runs spend its budget. Set or rotate the namespaced one with
`gh secret set COSMOS_SKILL_EVAL_NVIDIA_API_KEY --repo <owner>/<repo>` (paste the key on
stdin; or Settings → Secrets and variables → Actions → *New repository secret*, repo admin) — it overrides the old
one without deleting it. When both are empty the cell job fails before launching with
`::error::neither repository secret COSMOS_SKILL_EVAL_NVIDIA_API_KEY nor NVIDIA_API_KEY is set (the evaluator image
no longer bakes the inference key)`.

How it reaches the container: the "Run the cell" step of the `canary` and `cells` jobs exports it as
`NVIDIA_API_KEY` (step `env:`), the orchestrator passes its whole environment to the launcher
(`Deps.environ` → `build_launcher_env` → `run_launcher(env=…)`; only `GH_TOKEN` is removed — a test pins this), and
the launcher forwards a host `NVIDIA_API_KEY` with a bare `-e NVIDIA_API_KEY` (docker reads the value from the
launcher's environment, so it never appears on the `docker run` command line). The docker CLI applies `--env-file`
entries first and `-e` entries after them, whatever their order on the command line, so the secret takes precedence
over the runner's `~/cosmos-skill-eval.env` (which keeps the key for manual runs). The launcher logs one provenance
line, `[cosmos-skill-eval] NVIDIA_API_KEY: from host environment` in CI, never the value (GitHub masks the secret in
the job log as well). That line lands in the cell log on the runner (`<workdir>/<sha7>/<cell>.log`, see Execution),
not in the Actions log, so the orchestrator repeats it there once the launcher has returned —
`[run-matrix …] <skill>/<backend>/<case>: [cosmos-skill-eval] NVIDIA_API_KEY: from host environment` — the
GitHub-visible proof that the secret, not the env file, fed the container. The secret is available on all three
triggers. Until the PR merges, the label (`pull_request: labeled`, which runs the PR's copy of the workflow) and
`gh workflow run skill-eval.yml --ref <PR branch> -f pr_number=<N> -f scope=smoke` are the only triggers that start
the workflow at all — the file is not on `main` yet, so a `/run-skill-eval` comment does nothing and a default-ref
dispatch is refused. After the merge all three triggers run a copy with the step env and the guard (`issue_comment`
and default dispatches use `main`'s copy, the label the PR's). Proof of the secret path is the provenance echo
`… NVIDIA_API_KEY: from host environment`; a `from --env-file` line would mean the workflow copy in use lacks the
step env. When the secret is missing the cell job fails **before launching**: the step's first line prints
`::error::neither repository secret COSMOS_SKILL_EVAL_NVIDIA_API_KEY nor NVIDIA_API_KEY is set …` and exits 1, no cell result file is
written, and the PR comment renders the cell as `💥 error (n)` in the table with the entry
``n. `<backend> · <case>` — launcher error (failure) — see [job log](…)`` in the **Errors** list (`reconcile_cells`
turns a failed / timed-out job without a result into `launcher error (<conclusion>)`; `cancelled` and `skipped` keep
their own wording, `cancelled — see [job log](…)` and `not run — job skipped — see [job log](…)`); the `::error::`
annotation in the job log is the actual cause — the launcher never started. A container started without a key (an older
workflow, a manual run without the env file) exits 2 with `[cosmos-eval] ERROR: NVIDIA_API_KEY is not set …` before
any task container, which the orchestrator reports as `💥 error` with that line as the reason. The `plan` and
`report` jobs never receive the secret. The other credentials the image carries (S3, dashboard refresh token,
HF / NGC / Cursor) are unchanged and can move to the same path later.

## Trigger arguments

Whitespace-separated tokens after `/run-skill-eval` (or in the dispatch `args` input; the labels map to one token
each). Any order; unknown tokens are **reported in the comment** (`⚠️ Trigger: unknown argument …`) and ignored,
and the run proceeds with the defaults, so a typo never silently turns a selective re-run into something else.

| token               | effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `smoke-only`        | canary only, skill detection skipped; alias `smoke` (= label `run-skill-eval-smoke`, dispatch scope `smoke`, CLI `--scope smoke`). `smoke: false` in the config does not stop it                                                                                                                                                                                                                                                                                                                                                                 |
| `with-smoke`        | the canary first, then the skills of a full run. Without it (or `smoke: true` in the config) a run has no canary, whatever the PR touches; `smoke: false` in the config wins over this token (the log says so)                                                                                                                                                                                                                                                                                                                                   |
| `rerun-failed`      | re-run every cell whose newest result for this skill version is missing or not `PASS`; `PASS` cells are reused (= label `run-skill-eval-failed`)                                                                                                                                                                                                                                                                                                                                                                                                 |
| `force`             | re-run the selected cells even when a reusable result exists; nothing is reused (= label `run-skill-eval-force`)                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `cells=<p1>,<p2>,…` | restrict the run to the cells matching the patterns — `<backend>/<case>` or `<skill>/<backend>/<case>`, shell-style globs (`codex/*`, `*/trtllm-nano-t2i`, `cosmos3-generate-media/cursor/distilled-*`), case-sensitive. The other cells keep the skill version's newest published result — shown and counted like a fresh one (`reused`), picked up at every update, also when a concurrent run publishes it — or read `— not selected` (outside every count) while the version has none for them. Aliases: `rerun=<p1,p2>` and `rerun <p1,p2>` |
| `group=<id>`        | file every skill's runs under this dashboard group instead of the anchored / new one (letters, digits, `.`, `_`, `-`; CLI `--group <id>`)                                                                                                                                                                                                                                                                                                                                                                                                        |

Examples: `/run-skill-eval rerun-failed` — the six cells that ended without a report run again, 46 are reused;
`/run-skill-eval force cells=codex/trtllm-nano-t2i,cursor/distilled-trtllm-t2i` — exactly those two cells run, whatever
their previous result; `/run-skill-eval cells=opencode/*` — opencode's cells that have no reusable result run, the
rest of the matrix keeps its newest published result of this skill version, shown and counted like a fresh one (`reused`), or reads `(not selected)` when the version has no result for the cell yet. Note that `cells=` alone still reuses a reusable (`PASS`/`FAIL`) result
of a selected cell; add `force` to replace it.

The comment header shows the mode when it is not the default: `· canary smoke-only`, `· canary with-smoke`,
`· mode rerun-failed (6 selected, 46 reused)`, `· mode force`, `· cells=\`codex/*\``.

## Which skills run

There is no skill list in the config. Every run evaluates, in this order:

1. **The canary `cosmos3-ci-smoke`** — two trivial CPU-only cases that live in [`cosmos3-ci-smoke/`](cosmos3-ci-smoke/)
   next to this file (`SKILL.md` + `eval/eval.config`). It is a CI-only special case: it is *not* under
   `.agents/skills` and is never treated as a shipped skill. It runs **only on request**, and then **first**:
   trigger argument `smoke-only` (the canary alone — a pipeline check that costs minutes, no GPU-hour),
   `with-smoke` (the canary, then the skills) or `smoke: true` in the config (every full run). `smoke: false`
   disables it even for `with-smoke` (the log says the config wins); unset (`auto`) means off unless asked for.
   Whether the PR touches `.github/scripts/skill_eval/cosmos3-ci-smoke/` is logged but decides nothing.
2. **The skills the PR touches** — every `<name>` with a changed file under `.agents/skills/<name>/**` between
   the PR base and head, kept only when `.agents/skills/<name>/eval/eval.config` exists at the checked-out head,
   in alphabetical order. Skills deleted or renamed away by the PR and touched skills without an eval config are
   not run; the comment lists them under **Not evaluated** with the reason.

The changed-file list comes from the GitHub compare API (`GET /repos/<repo>/compare/<BASE_SHA>...<HEAD_SHA>`,
requested with `per_page=100&page=N`; GitHub returns the whole `files` list — at most 300 entries — on the first
page and no `files` on later pages, so another page is requested only while a page comes back exactly full; a
renamed file counts under both of its names).
When a request fails, or the comparison reaches the API's 300-file cap (the list may be truncated), the
orchestrator falls back to `git -C <checkout> diff --name-only --no-renames -z <BASE_SHA>...HEAD`, which needs the
base commit *and a merge base* in the checkout. That is the case in a full local clone and in the `plan` job
(full-history checkout: every branch is fetched, so both are present). If the fallback fails too, the run exits `2`
with a message saying what to do.

**Explicit override.** Adding `skills: [a, b]` back to `.github/skill-eval.yml` runs exactly those shipped skills
(config order) and skips the detection — the log says so. The canary is controlled by the trigger (`with-smoke`,
`smoke-only`) and by `smoke:`, never by `skills:`; listing it there is ignored with a warning. **Scope `smoke`**
(`run-skill-eval-smoke` label, `/run-skill-eval smoke-only`, dispatch scope `smoke`) runs only the canary and skips
the skill detection entirely — a pipeline check that costs minutes.

## What it does

- **Cells.** For every selected skill it reads its `eval/eval.config` (`evals[].id`, file order) and builds
  cases × `backends:` (config order), canary first.
- **One comment per PR head commit.** The first line of the comment is the marker
  `<!-- cosmos-skill-eval:pr<N>:<head sha> -->` — every run of the **same** commit (a label re-trigger, a partial
  `cells=` / `rerun-failed` run, the automatic re-run of report-less cells) rewrites that commit's comment, while a
  run of a **different** commit posts a new comment and leaves the earlier commits' comments as they are; the second
  line, `<!-- cosmos-skill-eval-run:<run_id>|<sha>|<attempt> -->`, names the Actions run whose view the body shows,
  and `<!-- cosmos-skill-eval-seq:<attempt>.<n> -->` counts the finished cells. When a newer run of the same commit
  takes the comment over, the previous run's outcome (its final line, or where it stood) moves into a
  **Previous runs on this PR** list at the bottom (hidden `<!-- cosmos-skill-eval-history:<run_id>|<sha>|<summary> -->`
  lines, the last 10 runs); a new commit's comment starts with the newest earlier comment's list plus that comment's
  own run, so the PR's history stays in the latest comment; an older run whose jobs finish late never overwrites the
  newer run's view (its results still reach the dashboard and its own job logs). A comment with this commit's marker
  is reused, otherwise one is created (comments with the pre-October-2026 one-per-PR marker
  `<!-- cosmos-skill-eval:pr<N> -->` are never edited again; their run becomes the first history entry). In the
  distributed workflow the state is never
  held by one process: every update rebuilds it from the plan, the dashboard (the results **this run** published,
  matched by `source.github.run_id`, or any run of the skill version created since the plan started — then the
  newest-result rule of the next bullet for every cell that is not about to run) and the run's
  own job list (queued / running / finished), so **any** cell job can PATCH the comment — at its start and at its
  end. Right before each PATCH the poster re-reads the live comment, merges the finished cells recorded in its
  hidden **ledger** lines (`<!-- cosmos-skill-eval-cell:<skill>/<backend>/<case>|<status>|<url>|<run_id> -->` —
  only the lines written by **this** Actions run count, so a re-trigger of the same commit never inherits the
  previous run's results; union, never a downgrade) and drops its own body if the live comment already shows more
  finished cells (or a later attempt).
  GitHub has no conditional update for comments, so two jobs can still interleave between that read and the
  PATCH: the loss is one live update, cosmetic, repaired by the next update or by the report. The sequential
  `run` mode PATCHes at most once per 10 s. Comment failures are logged, never fatal.
- **Newest result per cell, always.** For every cell the comment shows the newest published result of the skill
  version across all runs — what the dashboard group shows — so a partial run (`cells=`, `rerun-failed`) replaces
  only the cells it ran. After the per-cell reconciliation, every live update and the final report run
  `refresh_from_catalog` over the skills' catalogs: a `(not selected)` cell whose result was published since the plan
  (by the full run this partial run overlaps with, say) becomes a reused cell; a cell whose own result — identified
  by the dashboard run id in its link — is older than the version's newest for that cell shows the newer result
  instead (log: `superseded …`), rendered like a fresh one and counted by the status it shows — but not as `reused`:
  this run did execute the cell, and the header's `(n selected, m reused)`, the progress line's `reused` and the
  plan print count the plan's decision (`job` in the plan JSON's `cells[]`), not what a cell shows. Newest = the
  result whose run started last (the manifest `created_at`, stamped at the run's start, as the dashboard orders a
  group); equal start seconds keep the own result. Queued and running cells are left alone (they are about to
  publish), a launcher error (💥) has no published result to compare, and an own result the catalog does not list
  yet keeps its place until the catalog dates it — never a downgrade on a guess. A cell job's exit code is still its
  own outcome. Catalog GETs: one per skill at plan time (the same catalog decides the cache), one per skill at a cell
  job's start and at its end, one per skill per report poll (the final pass reuses the last poll's catalogs). The
  comment is one view: it is final when the run that owns it ends; results published after that are in the dashboard
  group.
- **Per-cell results are persisted independently of the comment.** Every cell job writes
  `<cell>.json` (state, status, report URL, the result's dashboard run id — and its `created_at` when the publisher
  provides one; today's `published.json` carries none — reason, duration, sha, attempt) and uploads it as the run
  artifact `cell-<skill>-<backend>-<case>` (`overwrite: true`, so a re-run attempt replaces it). The `report` job
  downloads all `cell-*` artifacts and takes them as the first source of truth — before the dashboard, the comment
  ledger and the job list — so the final verdict of a cell never depends on dashboard indexing or on a live comment
  update having survived. A cell whose job died before writing its file is still reported from its job state.
  The body: the header `## Skill eval — [<sha7>](commit) · run [#<id>](Actions run)` (plus the mode note, see
  above), a global progress line `n / N cells done · p passed · f failed · r no report · e errors · c reused`,
  a `⚠️ Trigger:` note when arguments were not understood, one section per evaluated skill headed
  `` ### `<skill>` · [dashboard overview](<dashboard>/skills/<skill>/groups/<group id>/) `` (the skill's own group, see
  [Skill versions and dashboard groups](#skill-versions-and-dashboard-groups); the canary's heading reads
  `` ### `cosmos3-ci-smoke` (canary) · … ``; once the dashboard lists a run of the skill version, the heading ends
  with `· static <n>/100`, the version's static-assessment score as the catalog reports it — `static_score` of the
  group's newest scored run (a group filled through `group=` can hold several skill versions: until this run's
  first result is listed, it may show the previous version's score), looked up by the report job and by a cell's
  live update when it fetched the catalog anyway, absent when the dashboard is unreachable) with its table
  (`⏳ queued`, `🟢 running`, the five statuses below,
  `(reused)`, `(not selected)`, `💥 error` with the reason in an error list), a **Not evaluated** note when a touched
  skill has nothing to run, a legend and a final `done — p/N passed (f failed, r no report, e errors)` line. Bodies
  stay under 60 000 characters.
- **Execution.** One cell per job (CI) or strictly one at a time (`run` mode), through the host launcher
  (`launcher:` in the config), with
  `--image`, `--skill-src <checkout>`, `--output-dir <workdir>/<sha7>/<skill>-<backend>-<case>`,
  `-e COSMOS_EVAL_SRC_HOST=<checkout>`, `-e COSMOS_EVAL_CODEX_MODEL=…` (codex only), `--github-env`, then
  `eval /workspace/skill-source/<skill dir> --backend <b> --case <c>`, where `<skill dir>` is
  `.agents/skills/<skill>` or `.github/scripts/skill_eval/cosmos3-ci-smoke` for the canary (the whole checkout is
  mounted at `/workspace/skill-source`). `--env-file` is passed only when `env_file:` exists on the runner; the
  job's `NVIDIA_API_KEY` (from the secret, see [Secrets](#secrets)) reaches the launcher through its environment and
  is forwarded into the container with `-e`, taking precedence over the file (docker applies `-e` entries after every
  `--env-file` entry, whatever the flag order).
  Launcher output goes to `<output-dir>.log` and stays there; the Actions log carries the orchestrator's own lines
  (`start` / `end`, result file, outcome, docker-gate and retry warnings) plus its echo of the launcher's
  `[cosmos-skill-eval] NVIDIA_API_KEY: …` provenance line (repeated on a retry, once per attempt); 3 h timeout per cell (the launcher gets SIGTERM, which `docker run`
  forwards into the container). When a cell runs again for the same sha, the previous attempt's log and
  `published.json` are moved aside as `*.prev` first.
- **Runner health gate.** Before every cell the orchestrator runs `docker_check:` (`docker info` in CI). If it
  fails, `docker_repair:` (`sudo -n chgrp docker /run/docker.sock`) runs once and the check is repeated; a
  successful repair is counted in a `🔧 …` note under the progress line. A cell whose launcher died on a
  docker-socket error (`permission denied … docker.sock`, `Cannot connect to the Docker daemon`) is retried
  once after that check/repair (its first log is kept as `*.log.prev`; a second failure reads `… (after 1
  retry)`). In `run` mode a check that still fails **stops** the matrix: the remaining cells are marked
  `not run`, the comment ends with `🛑 aborted — runner cannot reach docker — …` and the job exits `1`. In CI a
  cell job instead **holds** its cell (re-checking every minute for `docker_wait_minutes:`, default 30) before
  failing it, so a node with a dead docker takes one cell per half hour instead of eating the queue one cell per
  minute while the healthy nodes keep draining it. Both settings are optional; without them there is no gate
  (the retry still happens, blindly, once); `--fake-launcher` skips the gate.
- **Provenance.** Every launcher call gets `COSMOS_EVAL_GROUP_ID=<the skill's group id>` and
  `COSMOS_EVAL_SKILL_CONTENT_SHA=<the skill's content hash>` plus the `GITHUB_*` variables and a synthetic
  `GITHUB_EVENT_PATH` (`<workdir>/event-<sha7>.json`, built from the PR fetched via the API), so the published
  manifests show the PR, author, commit, Actions run, group and skill version (the publisher compares the content
  hash with its own and notes a mismatch in the manifest). The job's own `GITHUB_TOKEN` is **not** exported to the
  launcher: eval configs forward a `GITHUB_TOKEN` credential into agent containers, and that must stay the runner
  env file's read-only token, never the write-scoped job token. The checkout the cell bind-mounts into the task
  container is mounted without credentials as well: every `actions/checkout` step of the workflow sets
  `persist-credentials: false`, so the job token is not left in `<checkout>/.git/config` (the plan, cell and report
  steps call the API with the `GH_TOKEN` env var and never `git push`).
- **Outcome.** A published cell gets its status from the `status:` line in the log (one of the five below),
  linked to the URL in `<output-dir>/<skill>/published.json`. No `published.json` (or `status != ok`) means
  `💥 error` with the last log line as the reason. The **Errors** list under the tables groups identical reasons:
  one numbered entry per distinct reason, listing every affected cell (`1. \`codex · a\`, \`cursor · a\` — reason`),
  and the table cells carry that number.
- **Exit code.** `run` / `plan` / `report`: `0` when the orchestration completed — failing cells are reported
  in the comment, not as a job failure; `run` exits `1` when the matrix was aborted because the runner lost
  docker (see the health gate above); `report` exits `1` when it could not list the run's jobs or could not
  write the final comment (re-run the report job). `plan` exits `2` when the comment cannot be created (no
  cell starts without the comment) or a matrix would exceed GitHub's 256-jobs cap. A `cell` job exits `0` for
  every published status (PASS / FAIL / PARTIAL / NO-REPORT / ERROR — the dashboard has the report) and `1` when
  its cell ended in `💥 error` (no published result), so the Actions UI shows exactly the cells whose logs need
  reading. `2` for orchestration errors: config missing or invalid, PR/env missing (including `BASE_SHA` when
  skills are detected), changed-file list unavailable (compare API *and* git fallback failed), launcher missing,
  a malformed `--group`. `130` when the job was cancelled (the comment is finalised as "interrupted" first).
- **Log.** One stdout line per event: changed-file source and detected skills, each skill's version and group,
  reused cells and why a cell is not reused, cell start/end with duration and link, comment id, warnings.

## Statuses

Status taxonomy v2 of the evaluator (image `v20261028` and later). Per cell:

| status      | icon | meaning                                                                                                                                                                                                  | counted as |
| ----------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `PASS`      | ✅   | `eval_report.md` present with `**Verdict: PASS**`                                                                                                                                                        | passed     |
| `FAIL`      | ❌   | `eval_report.md` present without a PASS verdict — a real verdict on the skill                                                                                                                            | failed     |
| `PARTIAL`   | ⚠️   | aggregate of a case with mixed backends (not expected for a single cell)                                                                                                                                 | failed     |
| `NO-REPORT` | ❔   | the agent finished on its own (exit 0, no harness or API error) but wrote no `eval_report.md`: output truncated at the model's cap, stopped early, …                                                     | no report  |
| `ERROR`     | ⛔   | the arm did not finish normally — model gateway (`Selected model is at capacity`, rate limit), agent killed (incl. `pkill` self-kill), timeout, GPU fabric / docker on the node, other harness exception | errors     |
| 💥 error    | 💥   | the **launcher** failed before anything was published (no `published.json`) — the orchestrator's own failure class, with the reason in the **Errors** list                                               | errors     |

The report linked from a `NO-REPORT` / `ERROR` cell carries `status_reason` and `failure_class`, so nothing has
to be dug out of a job log. `NO-REPORT`, `ERROR` and `PARTIAL` are never reused by the cache: they are not a
verdict on the skill. The progress and final lines count `passed`, `failed` (FAIL + PARTIAL), `no report` and
`errors` (ERROR + launcher errors); the four buckets partition the finished cells. The comment's ledger lines, the
cell result files and the `status:` line parser accept all five values.

## Runners

Cells run wherever a runner with the labels `self-hosted`, `cosmos-skill-eval`, `h100` (`runner_labels:` in the
config) is idle; one runner process per node means one cell per GPU node at a time. **Cursor cells are serialised**:
their jobs also need the label `cursor-serial` (`serial_backends: [cursor]`, `serial_runner_label: cursor-serial`),
which today only one runner (the 4× H100 NVL node) carries, so they queue on that one runner and never run concurrently. The working
hypothesis: a per-key concurrency cap on Cursor's side (2026-10-26: 2 of 7 cursor cells died with
`[resource_exhausted]`, both while another cursor cell was running); the serialised runs confirm or refute it. A
second runner can take the label later if cursor becomes the bottleneck, at the price of that concurrency.
Registering a node (after the lease is populated with
the GPU-host setup and lease onboarding scripts of the evaluator repo):

```bash
GH_REG_TOKEN=$(gh api -X POST repos/<repo>/actions/runners/registration-token --jq .token) \
  <evaluator repo>/onboard-runner.sh <host> h100         # installs ~/actions-runner as a systemd service
gh api repos/<repo>/actions/runners --jq '.runners[] | "\(.name) \(.status) busy=\(.busy)"'
```

A runner that is offline simply does not pick up jobs; its cells wait in the queue (GitHub gives up after 24 h).

## Skill versions and dashboard groups

What is evaluated is the **skill directory's content**, not the commit that happens to carry it. For every
evaluated skill the plan computes a `SkillVersion`:

- `content_sha256` — SHA-256 over the skill directory (regular files, `__pycache__`, `.DS_Store` and `*.pyc`
  skipped, sorted by POSIX relative path, `relpath + "\0" + sha256(bytes) + "\n"` per file). The publisher
  computes the identical value (`cosmos_eval.metadata.skill_content_sha256`) into the manifest's
  `skill.content_sha256`, so a catalog run can be matched to the checkout without any commit in common.
- `tree_sha` — `git rev-parse <HEAD_SHA>:<skill dir>`, git's tree id of the directory (None outside a repository).
- `equivalent_commits` — `HEAD_SHA` plus every distinct `skill.commit_sha` the dashboard knows for the skill whose
  tree id of the skill directory equals HEAD's (`git rev-parse <sha>:<skill dir>`; a commit the clone cannot
  resolve is simply not equivalent). A commit that only touched CI or harness files therefore stays the same
  version; pre-v2 manifests, which have no content hash, are matched this way.

A catalog run **matches the version** when `skill.content_sha256` equals the content hash or `skill.commit_sha` is
an equivalent commit. The plan JSON carries the versions under `skills: {<name>: {content_sha256, tree_sha,
equivalent_commits, group_id}}`, so `cell` and `report` jobs (depth-1 checkouts) never need git or a second look at
the catalog for it.

**Group id, per skill.** The dashboard collects a PR's runs of a skill in a group (`<dashboard>/skills/<skill>/
groups/<group id>/`, newest result per cell). The plan picks it in this order:

1. `--group <id>` / trigger argument `group=<id>` → that id for every skill of the run;
2. otherwise the **PR's newest catalog run that matches the version** (`group_id` starting with `ci-pr<N>-`) → its
   group id, so a re-run after a CI or harness commit — or after the dashboard reported a problem — lands in the
   same group and replaces the cell there (the group view shows the newest result per cell, the old runs stay);
3. otherwise a new `ci-pr<N>-<sha7>`.

A fix that changes the skill's content opens a new group by construction, which is what the dashboard's compare
view needs. The canary gets its own version and group like any skill. Each skill heading in the comment links its
own group.

## Re-running (cache semantics)

Before running, the orchestrator asks the dashboard for the skill's runs
(`GET <dashboard>/api/catalog?skill=<skill>`, 10 s timeout, 3 attempts). For a cell `(case, backend)` it takes the
**newest version-matching run** (by `created_at`) that contains the case with that backend; its status is the
backend entry's status (fallback: the case status) and the link is `<dashboard>/runs/<run_id>/`. The result is
**reused** — the cell renders as `✅ PASS` / `❌ FAIL` (indistinguishable from a fresh result; the progress line counts them as *reused*), gets no job and is not re-run — only if

- its status is in the mode's **cacheable set**: default `{PASS, FAIL}`; `rerun-failed` → `{PASS}`; `force` → `{}`
  (nothing is reused), and
- its `verdict` is neither `NO-REPORT` nor `ERROR` (pre-v2 manifests say `FAIL` for an agent that never wrote a
  report; those cells run again under any mode).

So `PARTIAL`, `NO-REPORT` and `ERROR` results are never reused, and a newer `NO-REPORT` hides an older `PASS` (only
the newest result counts).

**Three strikes.** A cell whose three newest results for this skill version all ended without a report
(`NO-REPORT` / `ERROR`, or a pre-v2 `FAIL` with verdict `NO-REPORT`) is treated as a harness-side failure and is
**not re-run again automatically**: it renders as `❔ NO-REPORT (not re-run: 3 report-less attempts) · report`, keeps
its latest status in the counts and gets no job. `force` or an explicit `cells=` selection runs it anyway. This is
what keeps a plain `run-skill-eval` label (or `rerun-failed`) from burning GPU time on a cell that a model keeps
abandoning: after three tries, leave it, look at the reports, and fix the harness or the skill.

Consequences:

- A commit that changes a skill's files changes its version: that skill runs again, untouched skills are reused.
  A commit that only touches CI, docs or harness files keeps every version: everything is reused.
- `rerun-failed` (label `run-skill-eval-failed`) is the normal way to retry the cells that ended without a verdict
  or with a FAIL after a harness or gateway problem; `force` (label `run-skill-eval-force`) replaces everything;
  `force cells=…` replaces exactly the named cells. Published runs are immutable — a re-run adds a newer run to the
  same group, and the group view shows the newest result per cell.
- If the dashboard is unreachable the orchestrator logs a warning and treats every cell as a miss (and every
  skill as a new group).

While a run is in progress, a cell that is still to be resolved (queued or running) is looked up among **this run's**
results only: the run whose `source.github.run_id` is this Actions run, or — for a manifest without it — the newest
version-matching run created since the plan started. An older result of the version never stands in for a cell that
is about to run (in `force` mode it is exactly the result being replaced). Every cell that is **not** about to run
follows the newest-result rule instead ([What it does](#what-it-does), "Newest result per cell"): the comment shows
the version's newest published result for it, whichever run produced it.

## Config

`backends:`, `image:`, `dashboard:`, `launcher:`, `env_file:`, `codex_model:`, `cursor_model:` and `workdir:` are plain key/value
settings in `.github/skill-eval.yml`, plus the optional `smoke:`, `skills:`, `docker_check:` and `docker_repair:`
keys described above (the two commands are split like a shell would), `docker_wait_minutes:`, and the runner labels
(`codex_model:` / `cursor_model:` reach the evaluator container as `COSMOS_EVAL_CODEX_MODEL` / `COSMOS_EVAL_CURSOR_MODEL`
for the cells of that backend — the model the agent runs; a cursor value is a Cursor model slug prefixed `cursor/`,
e.g. `cursor/claude-sonnet-5-5-medium`, served by Cursor's own backend),
of the cell jobs: `runner_labels:` (default `[self-hosted, cosmos-skill-eval, h100]`, every cell job's `runs-on`),
`serial_backends:` (default `[cursor]`; `[]` switches the pinning off) and `serial_runner_label:` (default
`cursor-serial`) — a cell of a serial backend gets that extra label, which exactly one runner carries, see
[Runners](#runners). The plan writes the labels into every matrix entry (`runs_on`, a JSON list) and the workflow's
cell jobs use `runs-on: ${{ fromJSON(matrix.runs_on) }}`. The file is
parsed by a tiny built-in reader, so keep it flat: `key: value` and `key: [a, b]` only, `#` comments allowed,
no nesting. `image:` and `dashboard:` are placeholders in the committed file (this tree is mirrored to a public
repository): the workflow exports the repository variables `COSMOS_SKILL_EVAL_IMAGE` and `COSMOS_SKILL_EVAL_DASHBOARD`,
which `load_config` applies over the file's values (set them with `gh variable set …`; a local run exports them the same
way). `image:` must be an evaluator image with status taxonomy v2 (`v20261028` or later): the statuses, the
cache rule and the `COSMOS_EVAL_SKILL_CONTENT_SHA` hand-over assume it. From `v20261031` the image expects the
inference key from its environment (see [Secrets](#secrets)) instead of carrying it.

## Runner health (why the docker gate exists)

Harbor 0.24.0's docker environment ends every task with `chown -R <uid>:<gid>` of each writable mount target
of the task container, to hand files back to the host user. Inside the evaluator's controller container that
user is uid 0, and for eval cases with `"docker": true` the host's `/var/run/docker.sock` is one of those
mounts — so on 2026-10-19 the host socket turned `root:root` 30–90 s before every vllm-omni / sglang / NIM cell
ended and the next 31 launches died with `permission denied … docker.sock` (run 39215832922). Evaluator images
from `v20261027` mount the socket read-only (chown/chmod on it fail with EROFS; the docker CLI only needs
`connect()`), which removes the cause; the gate above is the defence in depth for this and for anything else
that takes docker away from the runner mid-matrix.

## Resuming

Re-run the Actions job (or comment `/run-skill-eval` again) for the same PR head. The rerun finds the
comment by its marker and the finished cells through the dashboard (reusable results of this skill version), so
only the missing cells run — plus every cell whose result was `NO-REPORT` / `ERROR`, which is never reused. A cell
that ended in `💥 error` (including `not run` after an abort) was never published and therefore runs again. Use
`/run-skill-eval rerun-failed` to also retry the `FAIL`s, `force` to replace everything. The dashboard re-lists the
bucket every ~2 minutes, so a rerun started within that window after a publish may not see the run yet and will
run (and publish) that cell once more.

## Local checks

Without a token (`--no-github`) the changed files come from `git diff`, so `BASE_SHA` must be reachable in the
local clone (for example the merge base with `origin/main`). `--trigger-args` takes the same text as a comment.
The unit tests also run in GitHub Actions: `.github/workflows/skill-eval-tests.yml` (`skill-eval tests`, ubuntu-latest,
the repo's python from `.python-version` — 3.13 today — and `pytest` alone) on every pull request and push to `main`
that touches `.github/scripts/skill_eval/**`.

```bash
# plan only: cells, versions, groups and reusable results for the checkout's HEAD; nothing is run or posted
HEAD_SHA=$(git rev-parse HEAD) BASE_SHA=$(git merge-base HEAD origin/main) \
  python3 .github/scripts/skill_eval/run_matrix.py --config .github/skill-eval.yml --checkout . --dry-run --no-github
# the same with the canary in front of the skills (a default plan has no canary; `--scope smoke` = the canary alone)
HEAD_SHA=$(git rev-parse HEAD) BASE_SHA=$(git merge-base HEAD origin/main) python3 .github/scripts/skill_eval/run_matrix.py \
  --config .github/skill-eval.yml --checkout . --dry-run --no-github --trigger-args with-smoke
# the same for a selective re-run
HEAD_SHA=$(git rev-parse HEAD) BASE_SHA=$(git merge-base HEAD origin/main) python3 .github/scripts/skill_eval/run_matrix.py \
  --config .github/skill-eval.yml --checkout . --dry-run --no-github --trigger-args "rerun-failed cells=*/trtllm-*"
# the distributed stages without GitHub: plan (prints the comment body, writes the plan), one cell, the report
HEAD_SHA=$(git rev-parse HEAD) BASE_SHA=$(git merge-base HEAD origin/main) python3 .github/scripts/skill_eval/run_matrix.py \
  plan --config .github/skill-eval.yml --checkout . --scope smoke --no-github --plan-out /tmp/plan.json
python3 .github/scripts/skill_eval/run_matrix.py cell --config .github/skill-eval.yml --checkout . --plan /tmp/plan.json \
  --skill cosmos3-ci-smoke --backend claude --case smoke-hello --fake-launcher --no-github
# whole flow with a stub launcher; comment bodies are printed instead of posted
BASE_SHA=$(git merge-base HEAD origin/main) python3 .github/scripts/skill_eval/run_matrix.py \
  --config .github/skill-eval.yml --checkout . --fake-launcher --no-github
# unit tests (fake GitHub API, fake launcher, fake dashboard, a throwaway git repo — no network, no docker)
uvx --with pytest python -m pytest .github/scripts/skill_eval/tests -q
uvx ruff check .github/scripts/skill_eval && uvx ruff format --check .github/scripts/skill_eval
```
