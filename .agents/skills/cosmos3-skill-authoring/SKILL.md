---
name: cosmos3-skill-authoring
description: >
  Author new Cosmos3 agent skills and their eval configs, and get them through
  the skill eval. Use when adding a skill under .agents/skills/, migrating a
  cookbook notebook or README into a skill, writing or updating an eval.config,
  choosing eval cases and assertions, fixing a skill after a failed or
  self-corrected eval cell, reading an eval report on the dashboard, or
  preparing a skill PR. Covers the required structure, the writing rules that
  decide first-attempt pass/fail, the eval.config schema (multi-case evals),
  how the evaluator judges, and how to run the eval in GitHub CI or locally.
  Do not use it to run the Cosmos3 workloads themselves (the per-workload
  skills do that) or to change the evaluator.
---

# Cosmos3 Skill Authoring

How to write a skill for this repository and make it automatically evaluable.
`.agents/skills/` is the source of truth for agent skills; keep the existing
layout (one directory per skill) and do not restructure it.

The bar: a skill passes its eval when every documented command works unchanged
on the first attempt, for each of the four agent backends, on every runner class
(see [`references/running-the-eval.md`](references/running-the-eval.md)).
Self-corrections count against the skill. `../cosmos3-generate-media/SKILL.md`
is the worked example behind every rule in this file.

## Skill directory structure

```
.agents/skills/<skill-name>/
├── SKILL.md            # required — the runnable path (read first by agents)
├── eval/
│   └── eval.config     # required for CI evaluation — see below
├── assets/             # vendored sample data the eval cases consume
├── references/         # optional: sources migrated 1:1 (notebooks, READMEs) with marked skill notes
└── scripts/            # optional: deterministic helpers SKILL.md links to (shebang, executable)
```

`eval/` is the home for everything evaluation-related (config today; fixtures,
goldens or overrides later). The evaluator also accepts a legacy top-level
`eval.config`.

Rules the automated static checks enforce (keep them green):

- **Frontmatter**: `name:` must equal the directory name; `description:` must
  exist, be ~60–1200 chars, and contain an explicit trigger clause
  ("Use when ...") — the description is the ONLY text agents see when
  deciding to load your skill, so pack it with the user phrasings you want
  to match.
- **Body size**: SKILL.md stays within 500 lines. The evaluator's static tier
  enforces the budget (a `body size` warning that costs 5 points of the static
  score, visible as the dashboard's "Static" tile) — but the budget is never a
  reason to drop a pin, a prerequisite, a success signal or a recovery step.
  Keep the essential workflow, the worked example and every decision rule in
  SKILL.md; put conditional detail, long notes and 1:1 migrated sources in
  `references/` and link to them at the step where they matter — the worked
  example is held to the same budget, with its long notes under
  `references/`; put deterministic, repetitive operations in `scripts/`. One
  canonical source per shared fact (checkpoint names, prompt formats, request
  bodies) — link to it, do not copy it into files that drift.
- **References resolve**: every relative path you mention (backticked or
  linked) must exist, resolved from the skill dir or the repo root. If you
  rely on repo-root-relative paths, state the path convention explicitly
  (see `cosmos3-setup`'s "Path convention" section).
- **Scripts**: any `*.sh` needs a shebang.

Section order that works (the `cosmos3-generate-media` layout): a short
intro; **Read this first** (numbered rules the agent must apply before the
first command: which reference to read, what to check on the host, which tool
versions each variant uses, how to verify outputs); any decision tables
(model sizes vs GPU with *measured* figures, per-backend switches); **one
section per variant / use case** with a `Reference:` line, the runnable block
and, for server-mode variants, the request body and the readiness rule;
**Verify the output**; **Reference files**; **Things not obvious from the
docs**; **Related skills**. Keep a "When to use this skill" list near the top.

Before writing a line, define: the one user job and its outcome; the target
user and what they must already have (hardware, access, artifacts); one
representative request and one accepted result; the shortest executable happy
path; which content belongs in SKILL.md, in `references/`, in `scripts/`.
Write two or three positive trigger phrasings, two negative ones (requests the
skill must NOT answer — e.g. Reasoner inference for a Generator skill) and one
forward workflow test; run them against the finished skill with a fresh agent.

## Writing commands that pass on the first attempt

- **Every runnable block is the documented first attempt.** Agents run blocks
  literally and in order, including one the prose says not to run on their
  hardware. Put hardware or environment choices inside one command that
  measures and branches (one launch script), and quote a source command you do
  not want run as text, not as a fenced block.
- **A block must not depend on shell state from an earlier block.** Agents run
  blocks in separate shell calls; `export`, `source …/activate` and `cd` do not
  carry over. Give every command an explicit target (`--python <venv>`,
  `<venv>/bin/python`, absolute paths), say when a block must run as one call,
  and make the failure mode a stop, not a silent install somewhere else.
- **Pin what the agent cannot guess**: exact versions and the pairings between
  them, package indexes and resolver flags, system packages, environment
  variables, and the tool version each variant uses (a repo-level tool pin
  leaks into standalone environments — say how to bypass it). Say why the
  neighbouring versions fail, so the agent does not upgrade its way out of a pin.
- **Make install blocks self-contained.** A fix placed in an "if you see this
  error" note is applied after the failure; fold it into the install command.
  Add a preflight (`python -c "import …"`, `<tool> --help | grep …`) before any
  large download, so a wrong install fails in seconds, not after the download.
- **Server-mode variants need the headless recipe**: detached, named start from
  a script (`setsid … &`), the readiness route with its warm-up behaviour and a
  deadline, logs kept (no `--rm`), teardown by process group with escalation,
  and never a second server on a busy GPU.
- **Credentials and containers**: forward tokens into containers explicitly
  (anonymous downloads are rate-limited on a cold cache), name every switch in
  the launch command itself, not only in a table, and give the daemon-side
  preflight when a container runs as a non-root uid against a bind mount.
- **Verify the output inside the skill**: the exact commands and the facts a
  correct run produces (format, dimensions, frame count, duration), and where
  each tool is and is not available, so agents do not improvise verification.
- **Scope and label**: every note names its variant and host class ("Variant 2
  only"; "on 80 GB GPUs"); every number says where it was measured and when
  ("one run" is not a benchmark); untested paths are "unverified — not a route".
- **Keep migrated sources verbatim and mark what you add.** A `references/` file
  migrated from a notebook or README keeps the source wording and headings;
  every eval-derived addition sits next to the source text as a marked note
  ("Skill note (not in the source): …"), and every deviation from the source
  is recorded in the skill's migration record.

Before/after evidence for each rule: the worked example
`../cosmos3-generate-media/SKILL.md` and its migration record,
`docs/cookbook_to_skill_migration.md` § "Eval-derived fixes". Those fixes and
the 44/44 result cited below come from the stacked PR #250, which merges into
the #128 branch before that branch reaches `main`.

## Writing steps a human can follow and an agent can execute

- Open with a brief overview: what the skill does, the user task, one concrete
  input → output example; introduce only the concepts the steps need.
- Numbered lists for sequential actions and stage handoffs; bullets for
  prerequisites and choices without an order; nesting one level deep.
- Each step starts with an action (prepare, inspect, configure, run, verify),
  says why when it is not obvious, names the working directory, environment
  and placeholders before the command, keeps the command in its own fenced
  block, and states what to inspect and which result permits continuation.
- Long commands: the executable on the first line, one option and its value
  per indented line, trailing `\` on every continued line, launcher options
  before the entry point (`torchrun … -m module …`), quoting and order preserved.
- Choices and recovery are labelled branches (**If the host has under 125 GiB of RAM …**)
  with an observable condition and a concrete next action; "adjust as needed"
  is not an instruction.
- Short paragraphs for context only; do not hide prerequisites or recovery in
  prose.
- Carry one worked example from inputs to outputs: the same asset, config
  name, paths and output directory in every step, with the artifact each step
  hands to the next.
- Before the first heavy command, a prerequisites block: supported runtime
  (repo revision, container or venv, GPU/driver/CUDA/memory as established by
  the sources, validated vs recommended kept apart), the install or activation
  commands, access and writable locations, and lightweight readiness checks
  (import, GPU visible, artifact reachable, output writable) with the expected
  result and the next action when a check fails. Let a reader skip installation
  when the checks pass.
- Three labels and nothing in between: *verified* (a command or value that
  ran, with where and when), *recommended* (a reasoned starting point, said so),
  *unverified* (never list it as a route). Never invent a supported range, a
  threshold or a quality claim; write "unknown" and what evidence would settle it.
- Gotchas and troubleshooting entries follow one shape: symptom → bounded
  likely causes → diagnostic command → corrective action → how to verify the
  recovery. One-off correlations stay hypotheses.

## How the evaluator judges your skill

One cell = one eval case × one agent backend (claude, codex, cursor,
opencode). Per cell the report carries:

- **status** — `PASS` / `FAIL` (the tester's own verdict in `eval_report.md`),
  `NO-REPORT` (the agent finished without writing the report), `ERROR` (model
  gateway, agent or node failure: `status_reason` says which; a gateway budget
  error is not a skill result).
- **first_attempt** — `PASS` only when the documented workflow needed no
  correction: from `grading.json` (`self_correction.was_needed`, with
  `first_attempt_errors` items tagged `owner: skill | tester | environment` —
  tester-only slips such as quoting errors or a missing editing tool do not
  count) and the report's "First attempt: SUCCEEDED | FAILED" line.
- **corrections** with an ownership (`skill`, `framework`, `docs`, `evaluator`,
  `environment`, `expected-agent-variance`) and **fix suggestions** (`lv1_serious`
  with `file:line` edits into the skill, `lv2_improvements`). Only `skill` (and
  `framework` when the skill documents that framework command) need a skill
  change; take the diagnosis, check the proposed wording against the sources.
- A cell's PASS / FAIL is the tester's own verdict. A first-attempt failure
  makes the first-attempt assertion fail (`first_attempt` = FAIL) even if a
  retry later succeeded — it does not change the cell status. "The skill's
  documented workflow executed without errors on the first attempt" should
  therefore be an assertion of every case.

The eval.config is the measuring stick: once a case exists, fix the skill, never
the prompt, the assertions or the checks — a before/after comparison is only
meaningful on an identical config.

## Writing eval/eval.config

`eval.config` is JSON. The evaluator (cosmos-skill-eval, Harbor-based) runs
every case with real agent harnesses — Claude Code, Codex, Cursor CLI and
OpenCode (`--backend claude,codex,cursor,opencode`) — and produces one report
with four criteria: Discovery (pass/fail — did the agent find and apply the
skill?), Execution (graded Correctness / Effectiveness / Discoverability /
Efficiency), Skill lift (opt-in), Static.

```jsonc
{
  "evals": [                          // one entry PER USE-CASE (multi-case supported)
    {
      "id": "t2v-smoke",              // short, stable, kebab-case
      "prompt": "User-voice TASK ONLY: what to produce, which bundled asset to use (skill-relative path), and the required output filename under outputs/. Environment setup, model loading details and flags belong in the SKILL, not here; artifact/summary/headless rules are injected by the evaluator automatically.",
      "expected_outcome": "One sentence a grader can verify from artifacts.",
      "assertions": [                 // authored ONCE, reused every run — do not leave empty
        "Concrete, evidence-checkable statements about first-attempt behavior and outputs"
      ],
      "checks": [                     // deterministic file checks (no LLM)
        {"path": "result_summary.md", "min_bytes": 100},
        {"path": "outputs/video_0.mp4", "min_bytes": 1000}
      ],
      "stages": ["prepare", "inference", "verify"]   // optional: pins the report's stage table rows
    }
  ],
  "credentials": ["HF_TOKEN"],        // env vars forwarded to the agent (HF_TOKEN, NGC_API_KEY, GITHUB_TOKEN; the inference key is the evaluator's own)
  "gpus": 1,                          // GPUs for the task container (runners: H100 80 GB PCIe/SXM and 94 GB NVL — every case must fit all of them)
  "docker": false,                    // per-case or global: give the agent the host docker socket (read-only mount) for container-served backends (vLLM-Omni, SGLang, NIM)
  "budget": {"minutes": 45, "tokens": 3000000},  // drives timeouts + Efficiency scoring; cells that install a framework in-run take 15–45 min
  "setup": {                          // for skills that need the repo checkout
    "repo": "https://github.com/NVIDIA/cosmos-framework.git",
    "ref": "<pin a commit sha>",
    "dest": "/app/repo",
    "src_mount": "/opt/cosmos-src"    // runners without GitHub egress mount a snapshot here
  },
  "lift": {"enabled": true}           // OPT-IN with-skill vs no-skill comparison arms
}
```

**Skill lift is OFF by default.** It triples agent runs, so enable it only
where the comparison is informative — utility skills whose value is guidance
(setup, troubleshooting, authoring). For workload recipes (inference/training
runs), leave it off: the execution criterion already measures what matters.

Authoring rules:

- **One eval case per use-case.** When a skill consolidates several cookbook
  notebooks, each original notebook becomes its own case (`id` named after
  the variant). The report shows a case matrix plus a full section per case.
- **Author assertions; don't let the grader invent them.** 3–6 per case:
  cover the expected outputs, first-attempt success of the documented
  workflow, and anything the notebook treats as the success signal. A cell's
  PASS / FAIL is the tester's own verdict; a first-attempt failure makes the
  first-attempt assertion fail (`first_attempt` = FAIL) even if a retry later
  succeeded — it does not change the cell status.
- **Smoke data must be tiny — and vendored.** Use the ORIGINAL workload's
  data, cut to 1–2 images/videos (or the smallest input the pipeline
  accepts), and COPY those files into the skill's `assets/` directory. Never
  reference assets in another repository — source material may be retired
  once a skill ships. If the original data is gated or unavailable, pick a
  small open Hugging Face asset of the same modality and say so in the
  prompt. Never require a multi-GB download for a smoke case.
- **Reference bundled data naturally and skill-relatively.** Prompts should
  read like a user task: "using the skill's bundled text2image prompt
  (assets/prompts/robot_draping.json)". The evaluator delivers the whole
  skill directory (assets included) to the agent, so skill-relative paths
  always resolve. For the rare lift-enabled skill, phrase the task so the
  no-skill baseline can still proceed from the prompt alone (describe the
  input, don't depend on skill files).
- **No checkpoints unless the case needs them**; when it does, state the
  exact checkpoint id in the prompt and account for it in `budget.minutes`.
- **Budget honestly**: top-level `budget` drives arm timeouts and is what the
  Efficiency dimension scores against — set it to what a competent run
  actually needs, not a wish.
- **Don't write boilerplate in prompts.** The evaluator injects the artifact
  rules (save outputs under outputs/, write result_summary.md and
  transcript.md, run headless, keep venvs/caches out of the artifacts dir)
  into every arm; repeating them per-case just drifts.
- **Checks name the real path.** The verifier checks exact files (no globs), so
  a check must name the file the workflow actually writes (look at the output
  tree of one real run — frameworks often nest outputs under per-backend and
  per-run directories instead of `outputs/<name>.png`) and the prompt should
  say where to put it.
- **Blocker-tolerant wording is for acceptable limitations**, e.g. a 122 GiB
  checkpoint on an 80 GB GPU: "only after a real attempt, document the blocker
  with measured numbers". It never excuses a documented command that fails.
- **Freeze it.** Once a skill is being evaluated across revisions the
  eval.config is the baseline: change the skill, not the config, and compare
  runs only on identical configs.

## Running the eval

Full detail: [`references/running-the-eval.md`](references/running-the-eval.md).

**GitHub CI (self-hosted H100 runners).** Nothing runs automatically. Two
triggers: an owner, member or collaborator comments `/run-skill-eval [args]`
on the PR — this fires only once the workflow file is on `main` (the comment
runs `main`'s copy) — or someone dispatches the workflow with the same args
(Actions → skill-eval → Run workflow, or
`gh workflow run skill-eval.yml --ref <branch> -f pr_number=<N> -f scope=full|smoke -f args='<args>'`,
which runs the copy on the branch you name): the dispatch is the way until the
workflow is on `main`, and also the way to re-measure one cell
(`force cells=<backend>/<case>`). No args: every touched skill with an
`eval/eval.config`, all four backends; cells with a reusable result for the
same skill content are reused. Args: `force` (run
everything, reuse nothing — after a skill change you want fully re-measured),
`rerun-failed` (only the cells that are not PASS), `smoke-only` (alias `smoke`;
the CPU-only canary alone, a pipeline check — add `force`, otherwise cached
canary cells are reused and nothing runs), `with-smoke` (the canary first, then
the skills — the canary never runs unless asked for), `cells=<backend>/<case>,…`
(only the matching cells; shell globs over `<backend>/<case>` or
`<skill>/<backend>/<case>`), `group=<id>` (file the runs under that dashboard
group).
The run posts one live comment per head commit (case × backend table with report links; same-commit re-runs update it)
and files the cells under the dashboard group `ci-pr<N>-<sha7>` of the skill
version. Three runners, one cell each at a time: H100 PCIe 80 GB with 125 GiB
host RAM, H100 SXM 80 GB with 251 GiB, H100 NVL 94 GB with 1 TB — a cell lands
on any of them, so the skill must work on all three. An 11 × 4 matrix takes
about two hours and roughly $100–130 of inference-hub budget (the
`COSMOS_SKILL_EVAL_NVIDIA_API_KEY` secret); `ERROR` cells with `model_api`
reasons are gateway or budget problems, not the skill.

**Locally** (any docker + GPU host with the image; the CI runner hosts already
have the launcher, env file and image):

```bash
~/cosmos-skill-eval-src/scripts/cosmos-skill-eval --env-file ~/cosmos-skill-eval.env \
  --image "$(sed -n 's/^image: //p' .github/skill-eval.yml)" --skill-src <checkout> \
  --output-dir ~/run-claude-my-case \
  eval /workspace/skill-source/.agents/skills/<skill-name> --backend claude --case <id>
```

The env file supplies `NVIDIA_API_KEY` (required, never baked into the image;
the launcher prints where the key came from, never its value) and the case
credentials; `COSMOS_EVAL_GROUP_ID=<group>` files the run under a dashboard
group; `--no-publish` keeps it private. The container publishes by itself (S3
upload + dashboard refresh) and ends with the links (`<dashboard>` below is the
dashboard host named in the reference). Image `v20261030` and older print two
lines: `[cosmos-eval] published: <dashboard>/runs/<run_id>/` and
`[cosmos-eval] s3: s3://…/runs/<run_id>/`. Image `v20261031` and newer
(evaluator 0.3.2) print four: `[cosmos-eval] group:
<dashboard>/skills/<skill>/groups/<group>/`, `[cosmos-eval] compare:
<dashboard>/skills/<skill>/compare?b=group%3A<group>` (B is preselected, pick
A), then `published:` and `s3:`; `<group>` is `COSMOS_EVAL_GROUP_ID` when set,
else the run id (a one-run group); `published.json` beside `report.json` holds
the same `url`, `group_url`, `compare_url` and `s3_prefix`. Read the run page
(one tab per case × backend: report, transcript, corrections, fix suggestions,
media, a Static box with the run's static score), the group page (matrix with
first-attempt marks; the "Static" tile next to "Pass rate" on the skill and
group pages is the version's static score) and the compare view
`/skills/<skill>/compare?a=<ref>&b=<ref>` (Static · A and Static · B tiles
with the delta on B), where a ref is
`group:<id>[@<ISO asof>]`, `branch:<name>` or `run:<run_id>`; both refs are
required (with only `b` the page shows pickers). Examples:
`compare?a=group:ci-pr128-0ba391d&b=group:ci-pr250-a292f5e` and
`compare?a=branch:skills/generate-media-migration-and-eval&b=branch:skills/generate-media-fixes`.

**Static checks** (<1 s, no agent): the evaluator's static tier enforces the
structure rules above and scores them 0–100 (critical −15, warning −5) — the
dashboard's "Static" tile; CI's pre-commit (`rumdl`) formats the Markdown.

## The fix loop: a stacked PR, one cell at a time, gated by the owner

A skill is done when every cell of its matrix is PASS and first-attempt PASS.
The way there is a loop, run as a separate PR stacked on the skill's PR so the
measured skill and its fixes stay reviewable apart (full detail:
[`references/running-the-eval.md`](references/running-the-eval.md) § 4):

1. **Baseline.** Run the full matrix on the skill PR; its dashboard group is
   the "before".
2. **Fixes PR.** Open a PR whose base is the skill PR's branch. For every cell
   that is not PASS with first-attempt PASS, read its report, the corrections
   with their ownership and the LV1 suggestions; fix the skill, never
   `eval/eval.config`; one commit per root cause, every addition marked as a
   skill note and recorded in the migration record.
3. **Re-measure only what changed**: locally on the runner class that exposed
   the failure, or in CI with a dispatch carrying `force cells=<backend>/<case>`.
   Repeat 2–3 until every affected cell is first-attempt PASS. An `ERROR` from
   the gateway, a budget cap, a runner or an agent service is a rerun, not a fix.
4. **Proof.** A forced full run on the fixes PR head, or, when the owner decides
   against spending it, the compare view's newest result per cell.
5. **Describe.** The fix table (ID, problem, fix, where, evidence, result) and
   the compare link in the fixes PR; the sentence that `eval/eval.config` is
   unchanged.

Owner gates: each iteration — which cells to treat, which fix, which rerun, and
any full run — is proposed to the skill owner and runs only after they confirm.
Commits land on the fixes PR branch; the skill PR is not touched until the loop
is done. The fixes PR merges into the skill PR's branch first; the skill PR then
carries both to `main`.

## Submitting

- Keep changes inside `.agents/skills/<skill-name>/` (plus docs the skill
  references). Don't touch other skills' layouts.
- Sign off commits (`git commit -s`).
- PR description: what the skill does, which notebooks/use-cases each eval
  case covers, where the smoke data comes from, and the GPU footprint. For a
  fix PR: one table — ID, problem, fix applied, where (`file:lines` permalinks),
  evidence (the report of each affected cell before the fix, fatal vs
  self-correction), result in the final run — plus the compare-view link and the
  sentence that `eval/eval.config` is unchanged.
- Trigger the eval (above); read every cell that is not PASS with first-attempt
  PASS before asking for review.
- Never publish: secrets or tokens, private endpoints, internal paths or
  buckets, customer or individual identities, third-party material without
  redistribution rights, internal log excerpts, or claims such as
  "production-ready" without approved evidence. Keep internal detail in the
  migration record or the PR, not in the skill.
- When writing a skill as an agent: do not choose between conflicting sources,
  do not invent supported ranges or thresholds, do not declare a path verified
  without a run that shows it — ask, or write "unknown".

## Related skills

| Skill                                | When to use                                                                                                                                                                                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `../cosmos3-setup/SKILL.md`          | Environment setup the eval cases assume                                                                                                                                                                                                                            |
| `../cosmos3-inference/SKILL.md`      | Inference workflows your cases exercise                                                                                                                                                                                                                            |
| `../cosmos3-codebase-nav/SKILL.md`   | Finding configs/params to reference                                                                                                                                                                                                                                |
| `../cosmos3-generate-media/SKILL.md` | Worked example: 7 variants, 11 eval cases; 44/44 PASS, 43/44 first-attempt in the final forced run, the 44th fixed and re-verified by a manual rerun and a CI rerun of that one cell (newest result per cell: 44/44 first-attempt; fixes from the stacked PR #250) |
