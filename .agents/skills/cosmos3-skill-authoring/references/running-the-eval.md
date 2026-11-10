# Running the skill eval: GitHub CI, local runs, and reading the dashboard

Everything here was exercised on the `cosmos3-generate-media` skill (PRs #324 and #327, October 2026); the
eval-derived fixes and the 44/44 result cited in § 4 come from the stacked PR #327, which merges into the #324
branch before that branch reaches `main`. The evaluator is `cosmos-skill-eval` (Harbor-based); its image and the dashboard URL are the `image:` and
`dashboard:` values of `.github/skill-eval.yml` (CI reads them from the repository variables `COSMOS_SKILL_EVAL_IMAGE` and
`COSMOS_SKILL_EVAL_DASHBOARD` when set).

## 1. The GitHub CI runner

### What runs, and where

- **Which skills**: every `.agents/skills/<name>/` touched by the PR (base…head diff) that has an
  `eval/eval.config` at the PR head. Nothing is configured per skill in CI; the PR diff decides.
- **Cells**: `cases × backends` = every `evals[].id` of the eval.config × `claude, codex, cursor, opencode`
  (`.github/skill-eval.yml` `backends:`). One cell is one GitHub Actions job; GitHub queues the cells over all
  idle runners. 11 cases × 4 backends ≈ 2 h on three nodes.
- **Hardware supply** (self-hosted runners labelled `cosmos-skill-eval, h100`; one cell at a time per node):

  | runner            | GPU                | host RAM | notes                                                                   |
  | ----------------- | ------------------ | -------- | ----------------------------------------------------------------------- |
  | runner A          | 1× H100 PCIe 80 GB | 125 GiB  | the smallest host-RAM class: offload routes that pin > 120 GiB die here |
  | runner B          | 1× H100 SXM 80 GB  | 251 GiB  | SXM in a VM (NVLink disabled by the driver config)                      |
  | runner C          | 4× H100 NVL 94 GB  | 1 TB     | only GPU 0 is used per cell; 95,830 MiB total                           |

  A cell lands on any of them, so a skill must work on all three classes (80 GB and 94 GB GPUs, 125 GiB to
  1 TB of host RAM). Hardware-dependent choices belong inside the documented command (SKILL.md § "Writing
  commands that pass on the first attempt"), not in prose. Cursor cells are no exception any more: they were pinned to
  runner C (label `cursor-serial`, one at a time) while the cursor backend ran Composer 2.5 and the Cursor service
  rejected sessions with `RetriableError: [resource_exhausted]`; with Claude Sonnet 5.5 as the cursor model the 11
  cursor cells ran four at a time on 2026-10-29 without a rejection, so `serial_backends` is empty in
  `.github/skill-eval.yml`. Since evaluator 0.3.3 (image `v20261101`) such an exit is a transient `model_api` failure
  that the cell retries by itself after `COSMOS_EVAL_RETRY_BACKOFF_S` (120 s).
- **Task container**: the evaluator image itself (Ubuntu 22.04, python 3.10 + a 3.12 venv first on PATH, docker
  CLI, no CUDA toolkit, no ffmpeg); the agent runs as root; `docker: true` cases get the host docker socket
  (read-only mount) so they can run container-served backends. Everything else (framework wheels, venvs,
  checkpoints) is installed by the agent inside the cell, which is exactly what the skill must document.

### Triggers

The workflow never runs automatically. The trigger is a PR comment whose first line is `/run-skill-eval [args]`
— it fires only once the workflow file is on `main` (the comment runs `main`'s copy) — or a manual dispatch of
the workflow with the same args, which is the way until then:

| what you want                                                                                                                                                                                                                                       | `/run-skill-eval` args                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| default run: the touched skills, no canary; cells with a reusable result are reused                                                                                                                                                                 | (none)                                         |
| force: every selected cell runs, nothing is reused (after a skill change you want fully re-measured)                                                                                                                                                | `force`                                        |
| rerun-failed: only cells whose newest result is not PASS run; PASS cells are reused                                                                                                                                                                 | `rerun-failed`                                 |
| smoke-only: the CPU-only canary `cosmos3-ci-smoke` alone (pipeline check, minutes) — combine with `force`, otherwise cached canary cells are reused and nothing runs                                                                                | `smoke-only force` (alias `smoke`)             |
| with-smoke: the canary first, then the skills (the canary never runs unless asked for)                                                                                                                                                              | `with-smoke`                                   |
| cells: only the matching cells, shell globs over `<backend>/<case>` or `<skill>/<backend>/<case>` (case-sensitive); the rest keep their newest published result (shown and counted as reused), or read "not selected" when the version has none yet | `cells=codex/*,cursor/<case>` (alias `rerun=`) |
| group: file the runs under this dashboard group                                                                                                                                                                                                     | `group=<id>`                                   |

Args combine (`/run-skill-eval force cells=opencode/*`); an unknown token is reported in the PR comment and
ignored. A comment, label or dispatch counts only when its author holds the write, maintain or admin role on the
repository (the workflow's `gate` job asks the collaborators API; organisation membership or a read / triage role is
refused), and a comment only once the workflow file is on `main` (the comment runs `main`'s copy; on a PR that adds or
changes the workflow, a comment does nothing until the merge). `workflow_dispatch` takes the same args — Actions →
skill-eval → Run workflow (PR number, scope `full` | `smoke`, args), or from a shell against the copy on any branch:
`gh workflow run skill-eval.yml --repo <owner>/<repo> --ref <branch> -f pr_number=<N> -f scope=full|smoke -f args='<args>'`
— the way to run the eval before the workflow is on `main`, and also how one cell is re-measured
(`force cells=<backend>/<case>`). PRs from forks never run, whatever
the trigger: the `plan` job refuses a head that is not in this repository (fork code is not executed on the
self-hosted runners). One `plan` at a time per PR (`cancel-in-progress: false`: later triggers queue). Parser:
`.github/scripts/skill_eval/run_matrix.py` `parse_trigger_args`.

### What you get back

- **One PR comment per head commit**, updated live: header (head sha, run link), one table per skill with a cell per
  case × backend — ⏳ queued · 🟢 running · ✅ PASS · ⚠️ PARTIAL · ❌ FAIL · ❔ NO-REPORT · ⛔ ERROR · 💥 launcher error
  (failed before publishing) — each finished cell linking its report. Re-runs of the same commit (`cells=`,
  `rerun-failed`, a re-trigger) rewrite that commit's comment; a run of a new commit posts a new comment, and the
  earlier commits' comments stay as they were. Every comment lists the PR's earlier runs under "Previous runs on
  this PR".
- **Newest result per cell**: the comment always shows, for every cell, the newest result of the skill version
  across runs — a partial run (`cells=`, `rerun-failed`) replaces only the cells it ran; every other cell keeps the
  newest known result, picked up at each update even when a concurrent run published it; a cell with no result yet
  reads `— not selected`; a cell's own older result is superseded by a newer run's result for that cell.
- **A dashboard group per skill version**: `ci-pr<N>-<sha7>` of the first head that produced that skill
  content (the skill directory's content hash). Harness-only commits land in the same group and replace cells;
  a skill change opens a new group.
- **Cache rule**: a cell is reused when the newest result for the same skill content is PASS or FAIL
  (`rerun-failed`: PASS only; `force`: never). NO-REPORT and ERROR cells always run again, at most three
  times for report-less failures ("three strikes"): after the third such result `rerun-failed` leaves the cell alone;
  a `force` run or an explicit `cells=<backend>/<case>` selection runs it again (`force cells=…` re-measures just that cell).
- **Statuses**: as defined in SKILL.md § "How the evaluator judges your skill". `⛔ ERROR` with
  `failure_class: model_api` and "Budget has been exceeded" means the inference-hub key is out of budget —
  nothing about the skill.

### Cost and credentials

Claude, Codex, OpenCode and the judge run through the NVIDIA inference hub; the key is passed into the cell as
the repository secret `COSMOS_SKILL_EVAL_NVIDIA_API_KEY` (never baked into the image). A full 11 × 4 matrix
costs roughly $100–130 including judging; Cursor is billed to the Cursor account. Check the key's remaining
budget before a campaign (the inference gateway's key-info endpoint, with the key; see the evaluator runbook).

## 2. Running locally

Any host with docker, an NVIDIA GPU and the image can run one cell exactly as CI does. On the lease hosts the
pieces are already in place (`~/cosmos-skill-eval-src`, `~/cosmos-skill-eval.env`, a checkout of this repository).

```bash
# one cell: <skill-checkout> is a checkout of this repository (the skill directory is mounted from it)
~/cosmos-skill-eval-src/scripts/cosmos-skill-eval \
  --env-file ~/cosmos-skill-eval.env \
  --image "$(sed -n 's/^image: //p' .github/skill-eval.yml)" \
  --skill-src ~/cosmos-framework \
  --output-dir ~/my-run-claude-<case> \
  eval /workspace/skill-source/.agents/skills/<skill-name> --backend claude --case <case>
```

- `--env-file`: `NVIDIA_API_KEY` (inference hub), `HF_TOKEN`, `NGC_API_KEY`, `CURSOR_API_KEY` as the cases
  need them (`credentials:` in eval.config lists what is forwarded to the agent). Without `NVIDIA_API_KEY`
  the container exits 2 before any task container starts (a cursor-only `--no-judge` run is the one exception);
  the launcher prints one provenance line — `from host environment`, `from --env-file <path>`, `from -e option`
  or `NOT PROVIDED` — never the value.
- `--backend` takes one or several (`claude,codex,cursor,opencode`); `--case <id>` one case; `--no-publish`
  keeps the run off the dashboard; `--arms execution` skips the opt-in lift arms.
- `COSMOS_EVAL_GROUP_ID=<group>` in the host environment files the run under a dashboard group (use the PR's
  `ci-pr<N>-<sha7>` group to put a manual re-run next to the CI cells; CI reuses it when the skill content
  matches).
- The container **publishes by itself**: it uploads `report.json`, `report.html`, every artifact and media file
  to S3, refreshes the dashboard and prints the links last. Image `v20261030` and older print two lines:
  `[cosmos-eval] published: <dashboard>/runs/<run_id>/` and
  `[cosmos-eval] s3: s3://…/runs/<run_id>/`. Image `v20261031` and newer (evaluator 0.3.2) print four:
  `[cosmos-eval] group: <site>/skills/<skill>/groups/<group>/`, then
  `[cosmos-eval] compare: <site>/skills/<skill>/compare?b=group%3A<group>  (pick A: a baseline group:<id>, branch:<name> or run:<id>)`,
  then `published:` and `s3:` as before; `<group>` is `COSMOS_EVAL_GROUP_ID` when set, else the run id (a
  one-run group). Open `group:` for this run's matrix; open `compare:`, which has B preselected, and pick A
  (`group:<id>`, `branch:<name>` or `run:<run_id>`) — or write the URL yourself (§ 3). The same values are in
  `published.json` beside `report.json` (`url`, `group_url`, `compare_url`, `s3_prefix`) for scripts.
- Lease shortcut (generate-media only): `IMAGE=… OUT_PREFIX=loop ~/run-case.sh <backend> <case>` runs one
  `cosmos3-generate-media` cell detached with the docker-socket guard — the script names that skill directory;
  copy and edit it for another skill. Outputs under `~/<prefix>-<backend>-<case>/`, log
  `~/<prefix>-<backend>-<case>.log`. Run directories contain root-owned files (written from inside containers):
  remove them with `sudo rm -rf`.
- Never run two cells on the same GPU at once: every task container gets "GPU 0".

## 3. Reading the results

- **Run page** `/runs/<run_id>/`: one tab per case × backend with the agent's `eval_report.md` (verdict,
  "First attempt: SUCCEEDED | FAILED", self-corrections), the transcript, the judge's assertion audit and
  correction ownership, the LV1/LV2 fix suggestions (file:line edits proposed into the skill), media and the
  verifier output; a Static box closes the kpi row of each cell tab (the run's static score — one score per run,
  not per cell). The fix suggestions and the corrections are the to-do list for the next skill revision.
- **Skill page** `/skills/<skill>/` (branch selector, `?branch=<name>` or `all`): the newest result per cell
  for a branch, the groups, the pass rate of the last runs and, right after "Pass rate", a "Static" tile with
  the version's static score (green ≥ 90, amber 70–89, red below 70; it links to the run report's section 4).
- **Group page** `/skills/<skill>/groups/<group>/`: the case × backend matrix of one skill version with
  `status`, `first_attempt` (1st ✓ / ✗), overall score, and the same "Static" tile for that skill version;
  `?asof=` views a group as it was after an earlier CI run.
- **Compare view** `/skills/<skill>/compare?a=<ref>&b=<ref>`: each ref is `group:<id>[@<ISO asof>]`,
  `branch:<name>` or `run:<run_id>` (colons may be URL-encoded as `%3A`, slashes as `%2F`). The API requires
  both A and B; the page with only B (what the launcher prints) or with neither shows pickers ("Choose both A
  and B to compare"). It shows per-cell deltas (improved / regressed / changed / same), first-attempt deltas,
  "Static · A" and "Static · B" tiles next to the Passed tiles (the delta on B, e.g. "−5 vs A")
  and "Copy as Markdown" (= `/api/skills/<skill>/compare?a=…&b=…&format=md`) for a PR description. Examples:
  `compare?a=group:ci-pr324-b50b766&b=group:ci-pr327-b7ad087` (the baseline group vs the final run of the fix
  PR) and `compare?a=branch:skills/generate-media-migration-and-eval-public&b=branch:skills/generate-media-fixes-public`
  (the newest result per cell on each branch).
- **API** for scripts: `/api/skills/<skill>/groups/<group>` (`cells[case][backend]`: `status`, `first_attempt`,
  `status_reason`, `run_id`), `/runs/<run_id>/files/report.json`,
  `/runs/<run_id>/files/cases/<case>/<backend>/{transcript.md,eval_report.md}`.

## 4. The loop that gets every cell to first-attempt PASS

The loop runs as a **separate PR stacked on the skill's PR** (base = the skill PR's branch), so the measured skill
and the eval-derived fixes are reviewed apart and the fixes PR merges into the skill PR's branch first. Every
iteration is proposed to the skill owner — which cells, which fix, which rerun, whether a full run is worth its
budget — and runs after their confirmation; nothing is pushed to the skill PR until the loop is done.

1. **Baseline**: run the full matrix on the skill PR (`/run-skill-eval`, or the dispatch) and open every cell that
   is not PASS with `first_attempt` PASS. The baseline group is the "before" of every later compare.
2. **Diagnose**: per cell read `eval_report.md` ("First attempt: FAILED — <root cause>", "Self-correction applied"),
   the judge's corrections with their ownership (only `skill`, and `framework` when the skill documents that
   command, need a skill change) and the LV1 suggestions (`file:line` edits into the skill). Group them by root
   cause. An `ERROR` whose reason is the model gateway, a budget cap, a runner or an agent service (for example
   Cursor's `[resource_exhausted]`) is not a skill finding: the cell retries such transient failures once by itself
   (0.3.3); if it still shows, rerun that cell — `rerun-failed`, or `force cells=<backend>/<case>` once it has three
   report-less results in a row.
3. **Fix** the skill in the fixes PR, never `eval/eval.config`: one commit per root cause, every addition marked as a
   skill note (or a declared in-block deviation when it lands inside notebook-verbatim text) and recorded in the
   skill's migration record.
4. **Re-measure only what changed**: locally with the launcher on the runner class that exposed the failure (host
   RAM, GPU size, cold cache; `COSMOS_EVAL_GROUP_ID=<group>` files it next to the CI cells, § 2), or in CI with a
   dispatch that carries `force cells=<backend>/<case>` (§ 1). Repeat 2–4 until every affected cell is
   first-attempt PASS.
5. **Proof**: a forced full run on the fixes PR head (`force`), about two hours and the full matrix's inference
   budget — a decision for the owner; the alternative they may choose is the compare view's newest result per cell
   (`compare?a=branch:<skill PR branch>&b=branch:<fixes PR branch>`, § 3).
6. **Describe**: the fix table (ID, problem, fix, where as `file:lines` permalinks at the head, evidence = the
   affected cells' reports before the fix, result = the cells in the proof run) and the compare link in the fixes PR
   body, plus the sentence that `eval/eval.config` is unchanged.

Worked example (PRs #324 → #327, October 2026): baseline group `ci-pr324-b50b766` 42/44 PASS, 14/44 first-attempt;
the fixes were developed in five iterations of per-cell reruns (a failing cell is re-run alone after each fix, the
full matrix only once the last cell passes), then proven with a full run on the fixes PR's head (group
`ci-pr327-b7ad087`): 44/44 PASS, 44/44 first-attempt; the branch compare view shows the newest result per cell.
