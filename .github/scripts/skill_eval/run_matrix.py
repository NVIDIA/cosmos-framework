#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run the skill-eval matrix (cases x backends) for one PR head and keep ONE PR comment updated.

Which skills run is detected from the PR: every `.agents/skills/<name>/**` changed between the PR base
and head whose `eval/eval.config` exists at the head, alphabetically. The `cosmos3-ci-smoke` canary
(.github/scripts/skill_eval/cosmos3-ci-smoke) runs first, but only on request: trigger argument `with-smoke`
(canary + skills) or `smoke-only` (canary alone), or `smoke: true` in the config (`smoke: false` = never).

Stdlib only: the self-hosted runner has no extra Python packages. See README.md next to
this file for the full contract (trigger, skill detection, cache semantics, resume).

    run_matrix.py [run] --config .github/skill-eval.yml --checkout <repo root>      # whole matrix, one process
                  [--fake-launcher] [--no-github] [--dry-run] [--trigger-args "<text>"] [--group <id>]
    run_matrix.py plan   --config … --checkout … [--scope full|smoke] [--trigger-args "<text>"] [--plan-out FILE]
    run_matrix.py cell   --config … --checkout … --plan FILE|JSON --skill S --backend B --case C   # one job per cell
    run_matrix.py report --config … --checkout … --plan FILE|JSON                       # final comment

Environment (set by the workflow): GH_TOKEN, GITHUB_REPOSITORY, PR_NUMBER, HEAD_SHA, BASE_SHA, HEAD_REF,
GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_SERVER_URL (default https://github.com).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Optional

USER_AGENT = "cosmos-skill-eval-ci"
MARKER_PREFIX = "<!-- cosmos-skill-eval:"  # first comment line: `<!-- cosmos-skill-eval:pr<N>:<head sha> -->` — one comment per PR head commit
OWNER_PREFIX = (
    "<!-- cosmos-skill-eval-run:"  # 2nd line: <run_id>|<sha>|<attempt> — the Actions run that owns the live body
)
HISTORY_PREFIX = "<!-- cosmos-skill-eval-history:"  # hidden line per earlier run of the PR: <run_id>|<sha>|<summary>
MAX_HISTORY = 10  # earlier runs kept in the comment
MAX_REPORTLESS_ATTEMPTS = 3  # consecutive NO-REPORT/ERROR results of a cell after which it is not re-run automatically
MAX_BODY_CHARS = 60_000
CELL_TIMEOUT_S = 3 * 60 * 60
LAUNCHER_STOP_GRACE_S = 60.0  # SIGTERM -> SIGKILL grace when a cell times out
CATALOG_TIMEOUT_S = 10.0
COMMENT_MIN_INTERVAL_S = 10.0
GITHUB_RETRIES = 3
COMPARE_PAGE_SIZE = 100  # GET /compare is asked for 100 per page (GitHub serves the whole `files` list on page 1) ...
COMPARE_MAX_FILES = 300  # ... and lists at most 300 of them: at the cap the list may be truncated
GIT_TIMEOUT_S = 60.0
DOCKER_CHECK_TIMEOUT_S = 30.0
DOCKER_REPAIR_TIMEOUT_S = 60.0
# A launcher that dies with one of these never reached the evaluator: the runner's docker socket was unusable.
DOCKER_FAILURE_RE = re.compile(
    r"permission denied while trying to connect to the [Dd]ocker (?:API|daemon)"
    r"|Cannot connect to the Docker daemon"
    r"|dial unix /var/run/docker\.sock: connect: permission denied"
)

SKILLS_DIR = PurePosixPath(".agents/skills")  # shipped skills, relative to the checkout root
CANARY_SKILL = "cosmos3-ci-smoke"
CANARY_DIR = PurePosixPath(".github/scripts/skill_eval") / CANARY_SKILL  # CI-only smoke skill, never shipped
# Where the cell jobs run (.github/skill-eval.yml; the plan writes the labels into every matrix entry's `runs_on`):
# `runner_labels:` for every cell job; the cells of a backend in `serial_backends:` also carry `serial_runner_label:`,
# a label exactly ONE runner has, so they run one at a time. Working hypothesis behind the cursor default: a per-key
# concurrency cap on Cursor's side (2026-10-26: 2 of 7 cursor cells died with `[resource_exhausted]`, both while
# another cursor cell was running); the serialised runs confirm or refute it.
DEFAULT_RUNNER_LABELS = ("self-hosted", "cosmos-skill-eval", "h100")
DEFAULT_SERIAL_BACKENDS = ("cursor",)
DEFAULT_SERIAL_RUNNER_LABEL = "cursor-serial"

PASS, FAIL, PARTIAL = "PASS", "FAIL", "PARTIAL"
# Status taxonomy v2 (evaluator images >= v20261028): a published cell carries one of the five statuses below.
# NO-REPORT = the agent finished on its own but wrote no eval_report.md; ERROR = the arm did not finish normally
# (model gateway, agent killed, node). CellState.ERROR — the launcher died before publishing anything — is separate.
NO_REPORT = "NO-REPORT"
ERROR_STATUS = "ERROR"
KNOWN_STATUSES = (PASS, FAIL, PARTIAL, NO_REPORT, ERROR_STATUS)
STATUS_ICON = {PASS: "✅", FAIL: "❌", PARTIAL: "⚠️", NO_REPORT: "❔", ERROR_STATUS: "⛔"}
# Which published statuses a plan reuses instead of re-running the cell (TriggerOptions.cacheable picks the set).
# PARTIAL, NO-REPORT and ERROR are never reused: they are not a verdict on the skill.
CACHEABLE_DEFAULT = frozenset({PASS, FAIL})
CACHEABLE_RERUN_FAILED = frozenset({PASS})
PR_GROUP_PREFIX = "ci-pr"  # dashboard group ids of CI runs: ci-pr<N>-<sha7> (unless `group=<id>` names another)
GROUP_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")  # a group id ends up in URLs and bucket keys
NOT_RUN_PREFIX = "not run — "  # reason prefix of cells skipped because the runner lost docker
PUBLISHING = "publishing"  # RUNNING-cell reason: its job finished, the dashboard has not listed the result yet
PLAN_SCHEMA = "cosmos-skill-eval-plan/v1"
SEQ_PREFIX = "<!-- cosmos-skill-eval-seq:"  # 2nd comment line: number of finished cells the body shows
RESULT_SCHEMA = "cosmos-skill-eval-cell-result/v1"  # per-cell outcome file, uploaded as a run artifact
LEDGER_PREFIX = "<!-- cosmos-skill-eval-cell:"  # hidden line per finished cell: <label>|<status>|<url>; the
# comment thereby carries its own results, so no renderer depends on the dashboard having listed a run yet
JOB_NAME_PREFIX = "cell: "  # workflow job name of a cell = "cell: <skill>/<backend>/<case>" (= Cell.label)
MATRIX_MAX_JOBS = 256  # GitHub Actions: a job matrix may generate at most 256 jobs
CATALOG_SETTLE_S = 240.0  # report: how long to wait for the dashboard to list the results of finished cell jobs
CATALOG_SETTLE_POLL_S = 20.0
CATALOG_RETRIES = 3  # dashboard catalog GETs: attempts before a lookup counts as "no cache"
DOCKER_WAIT_POLL_S = 60.0  # cell: re-check interval while the runner cannot reach docker
STATUS_ALTERNATION = "|".join(re.escape(status) for status in KNOWN_STATUSES)
STATUS_LINE_RE = re.compile(rf"status:\s*({STATUS_ALTERNATION})\b")
# <label>|<status>|<url>[|<actions run id>]: the run id (added 2026-10-22) scopes a ledger line to the run that
# wrote it — a re-trigger of the same sha must not inherit the previous run's results through the shared comment
LEDGER_RE = re.compile(re.escape(LEDGER_PREFIX) + rf"([^|\s]+)\|({STATUS_ALTERNATION})\|([^\s|]*)(?:\|([^\s|]*))? -->")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class OrchestrationError(Exception):
    """A problem with the orchestration itself (config, PR, launcher) -> non-zero exit."""


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    """One line per event on stdout (the Actions log)."""
    stamp = utc_now().strftime("%H:%M:%S")
    print(f"[run-matrix {stamp}] {message}", flush=True)


def warn(message: str) -> None:
    log(f"WARN: {message}")


# ---------------------------------------------------------------------------
# config (flat YAML subset, parsed without PyYAML)
# ---------------------------------------------------------------------------


@dataclass
class Config:
    backends: list[str]
    image: str
    dashboard: str
    launcher: Path
    workdir: Path
    env_file: Path | None = None
    codex_model: str | None = None
    skills: list[str] | None = None  # explicit override; None = detect the skills the PR touches
    smoke: bool | None = None  # None = only on request (`with-smoke`/`smoke-only`); True = always; False = never
    docker_check: list[str] | None = None  # must succeed before a cell starts (e.g. `docker info`); None = no gate
    docker_repair: list[str] | None = (
        None  # run once when the check fails (e.g. `sudo -n chgrp docker /run/docker.sock`)
    )
    docker_wait_minutes: int = 30  # cell: hold the cell this long while docker is unavailable instead of failing fast
    # runs-on of the cell jobs: every cell gets `runner_labels`; a cell of a backend in `serial_backends` also gets
    # `serial_runner_label`, which exactly one runner carries, so those cells queue there and run one at a time (cursor:
    # most likely a per-key concurrency cap on Cursor's side, see DEFAULT_SERIAL_BACKENDS). Defaults = today's CI.
    runner_labels: list[str] = field(default_factory=lambda: list(DEFAULT_RUNNER_LABELS))
    serial_backends: list[str] = field(default_factory=lambda: list(DEFAULT_SERIAL_BACKENDS))
    serial_runner_label: str | None = DEFAULT_SERIAL_RUNNER_LABEL


def _strip_comment(line: str) -> str:
    """Drop a `# ...` comment that starts the line or follows whitespace."""
    if line.lstrip().startswith("#"):
        return ""
    for index, char in enumerate(line):
        if char == "#" and index > 0 and line[index - 1] in " \t":
            return line[:index]
    return line


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_simple_yaml(text: str) -> dict[str, str | list[str]]:
    """Parse `key: value` and `key: [a, b]` lines. Anything nested or indented is an error."""
    data: dict[str, str | list[str]] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        if line[0] in " \t":
            raise OrchestrationError(f"config line {lineno}: indented/nested YAML is not supported")
        key, separator, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise OrchestrationError(f"config line {lineno}: expected `key: value`, got {raw!r}")
        if value.startswith("[") and value.endswith("]"):
            items = [_unquote(item) for item in value[1:-1].split(",")]
            data[key] = [item for item in items if item]
        else:
            data[key] = _unquote(value)
    return data


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def load_config(path: Path, environ: Mapping[str, str] | None = None) -> Config:
    if not path.is_file():
        raise OrchestrationError(f"config not found: {path}")
    data = parse_simple_yaml(path.read_text(encoding="utf-8"))

    def required(key: str) -> str | list[str]:
        if key not in data:
            raise OrchestrationError(f"config {path}: missing key `{key}`")
        return data[key]

    def required_list(key: str) -> list[str]:
        value = required(key)
        if not isinstance(value, list) or not value:
            raise OrchestrationError(f"config {path}: `{key}` must be a non-empty list like [a, b]")
        return value

    def required_scalar(key: str) -> str:
        value = required(key)
        if not isinstance(value, str):
            raise OrchestrationError(f"config {path}: `{key}` must be a scalar")
        return value

    def optional_tristate(key: str) -> bool | None:
        value = data.get(key)
        if value is None or (isinstance(value, str) and value.lower() == "auto"):
            return None
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise OrchestrationError(f"config {path}: `{key}` must be auto, true or false")

    def optional_command(key: str) -> list[str] | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not shlex.split(value):
            raise OrchestrationError(f"config {path}: `{key}` must be a command line like `docker info`")
        return shlex.split(value)

    def optional_int(key: str, default: int) -> int:
        value = data.get(key)
        if value is None:
            return default
        if not isinstance(value, str) or not value.isdigit():
            raise OrchestrationError(f"config {path}: `{key}` must be a whole number")
        return int(value)

    def optional_list(key: str, default: tuple[str, ...], *, allow_empty: bool = False) -> list[str]:
        value = data.get(key)
        if value is None:
            return list(default)
        if not isinstance(value, list) or (not value and not allow_empty):
            raise OrchestrationError(f"config {path}: `{key}` must be a list like [a, b]")
        return value

    backends = required_list("backends")
    serial_backends = optional_list("serial_backends", DEFAULT_SERIAL_BACKENDS, allow_empty=True)
    # an explicit `serial_backends` list is checked against `backends`; the default is inert for a config without cursor
    if "serial_backends" in data:
        unknown = [backend for backend in serial_backends if backend not in backends]
        if unknown:
            raise OrchestrationError(
                f"config {path}: `serial_backends` names backends not in `backends`: {', '.join(unknown)}"
            )
    serial_runner_label = data.get("serial_runner_label", DEFAULT_SERIAL_RUNNER_LABEL)
    if not isinstance(serial_runner_label, str) or (serial_backends and not serial_runner_label):
        raise OrchestrationError(
            f"config {path}: `serial_runner_label` must be one runner label (the cells of `serial_backends` need it)"
        )
    skills = data.get("skills")
    if skills is not None and (not isinstance(skills, list) or not skills):
        raise OrchestrationError(
            f"config {path}: `skills` must be a non-empty list like [a, b] (remove the key to detect skills from the PR)"
        )
    env_file = data.get("env_file")
    codex_model = data.get("codex_model")
    # The evaluator image and the dashboard URL are deployment values: the repository variables
    # COSMOS_SKILL_EVAL_IMAGE / COSMOS_SKILL_EVAL_DASHBOARD (exported by the workflow) override the config file, whose
    # own values are placeholders in the public mirror of this tree.
    env = os.environ if environ is None else environ
    image = (env.get("COSMOS_SKILL_EVAL_IMAGE") or "").strip() or required_scalar("image")
    dashboard = ((env.get("COSMOS_SKILL_EVAL_DASHBOARD") or "").strip() or required_scalar("dashboard")).rstrip("/")
    return Config(
        backends=backends,
        image=image,
        dashboard=dashboard,
        launcher=_expand_path(required_scalar("launcher")),
        workdir=_expand_path(required_scalar("workdir")),
        env_file=_expand_path(env_file) if isinstance(env_file, str) else None,
        codex_model=codex_model if isinstance(codex_model, str) else None,
        skills=skills,
        smoke=optional_tristate("smoke"),
        docker_check=optional_command("docker_check"),
        docker_repair=optional_command("docker_repair"),
        docker_wait_minutes=optional_int("docker_wait_minutes", 30),
        runner_labels=optional_list("runner_labels", DEFAULT_RUNNER_LABELS),
        serial_backends=serial_backends,
        serial_runner_label=serial_runner_label or None,
    )


# ---------------------------------------------------------------------------
# run identity (PR, sha, run) and the cells
# ---------------------------------------------------------------------------


@dataclass
class SkillVersion:
    """What exactly is evaluated for one skill, independent of the commit that happens to carry it.

    `content_sha256` hashes the files of the skill directory (the publisher writes the same value into the
    manifest's `skill.content_sha256`); `tree_sha` is git's tree id of that directory at HEAD; `equivalent_commits`
    is HEAD plus every commit the dashboard knows for the skill whose tree id is the same — so a run published
    from a commit that only touched CI or harness files still counts as a run of this version. `group_id` is the
    dashboard group the PR's runs of this version are collected in (see anchored_group).
    """

    content_sha256: str
    tree_sha: str | None = None
    equivalent_commits: list[str] = field(default_factory=list)
    group_id: str = ""

    @classmethod
    def of_commit(cls, sha: str) -> SkillVersion:
        """A version known only by its commit (a plan without a block for the skill): matches that sha alone."""
        return cls(content_sha256="", equivalent_commits=[sha])

    def matches(self, run: Mapping) -> bool:
        """Is this catalog run an evaluation of this skill version? (content sha, else an equivalent commit)"""
        skill_block = run.get("skill") if isinstance(run.get("skill"), dict) else {}
        if self.content_sha256 and skill_block.get("content_sha256") == self.content_sha256:
            return True
        return skill_block.get("commit_sha") in self.equivalent_commits

    def to_dict(self) -> dict:
        return {
            "content_sha256": self.content_sha256,
            "tree_sha": self.tree_sha,
            "equivalent_commits": list(self.equivalent_commits),
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> SkillVersion:
        return cls(
            content_sha256=str(data.get("content_sha256") or ""),
            tree_sha=str(data["tree_sha"]) if data.get("tree_sha") else None,
            equivalent_commits=[str(sha) for sha in data.get("equivalent_commits") or []],
            group_id=str(data.get("group_id") or ""),
        )


@dataclass(frozen=True)
class RunInfo:
    repo: str
    server_url: str
    pr_number: int
    head_sha: str
    head_ref: str
    run_id: str
    run_attempt: str
    dashboard: str
    base_sha: str = ""  # PR base sha; needed to detect the skills the PR touches
    # per skill (filled by the plan, carried by the plan JSON): content/tree sha, equivalent commits, group id
    versions: dict[str, SkillVersion] = field(default_factory=dict)
    group_override: str | None = None  # `--group` / `group=<id>`: one dashboard group for every skill of this run

    @property
    def sha7(self) -> str:
        return self.head_sha[:7]

    @property
    def default_group(self) -> str:
        """The group a skill's runs open when the PR has none for this version yet: ci-pr<N>-<sha7>."""
        return f"{PR_GROUP_PREFIX}{self.pr_number}-{self.sha7}"

    def version_for(self, skill: str) -> SkillVersion:
        """The skill's version block, or — without one — the exact head commit (the pre-v2 cache key)."""
        return self.versions.get(skill) or SkillVersion.of_commit(self.head_sha)

    def group_for(self, skill: str) -> str:
        """Dashboard group of this run's results for the skill (the plan anchored or opened it)."""
        version = self.versions.get(skill)
        if version is not None and version.group_id:
            return version.group_id
        return self.group_override or self.default_group

    def groups_summary(self) -> str:
        """`skill=group, …` for the log lines."""
        if not self.versions:
            return self.group_override or self.default_group
        return ", ".join(f"{skill}={self.group_for(skill)}" for skill in self.versions)

    @property
    def marker(self) -> str:
        """First line of the skill-eval comment for THIS head commit. Every run of the same commit — label
        re-triggers, partial re-runs, the automatic re-runs of report-less cells — rewrites that comment; a run of a
        different commit posts a new comment and lists the earlier commits' runs under "Previous runs"."""
        return f"{MARKER_PREFIX}pr{self.pr_number}:{self.head_sha} -->"

    @property
    def pr_marker_prefix(self) -> str:
        """Common prefix of every skill-eval comment of this PR (any head sha, and the pre-October-2026 one-per-PR
        form `<!-- cosmos-skill-eval:pr<N> -->`)."""
        return f"{MARKER_PREFIX}pr{self.pr_number}"

    def is_pr_marker(self, body: object) -> bool:
        """Does `body` start with a skill-eval marker of this PR (this or another head sha, or the legacy form)?"""
        first = str(body or "").split("\n", 1)[0]
        return first.startswith(self.pr_marker_prefix) and first[len(self.pr_marker_prefix):].startswith((":", " -->"))

    @property
    def owner_line(self) -> str:
        return f"{OWNER_PREFIX}{self.run_id}|{self.head_sha}|{self.run_attempt or '1'} -->"

    @property
    def commit_url(self) -> str:
        return f"{self.server_url}/{self.repo}/commit/{self.head_sha}"

    @property
    def run_url(self) -> str:
        url = f"{self.server_url}/{self.repo}/actions/runs/{self.run_id}"
        if self.run_attempt and self.run_attempt != "1":
            url += f"/attempts/{self.run_attempt}"
        return url

    def group_url(self, skill: str) -> str:
        return f"{self.dashboard}/skills/{skill}/groups/{self.group_for(skill)}/"


def git_head(checkout: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def read_run_info(environ: Mapping[str, str], config: Config, checkout: Path, *, no_github: bool) -> RunInfo:
    """Build the run identity from the workflow env; lenient defaults only in --no-github mode."""

    def required(name: str, fallback: str | None = None) -> str:
        value = (environ.get(name) or "").strip()
        if value:
            return value
        if no_github and fallback is not None:
            return fallback
        raise OrchestrationError(f"environment variable {name} is required")

    pr_raw = required("PR_NUMBER", fallback="0")
    try:
        pr_number = int(pr_raw)
    except ValueError as exc:
        raise OrchestrationError(f"PR_NUMBER must be an integer, got {pr_raw!r}") from exc
    if not no_github and not (environ.get("GH_TOKEN") or "").strip():
        raise OrchestrationError("environment variable GH_TOKEN is required (use --no-github to skip GitHub)")
    return RunInfo(
        repo=required("GITHUB_REPOSITORY", fallback="local/checkout"),
        server_url=(environ.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/"),
        pr_number=pr_number,
        head_sha=required("HEAD_SHA", fallback=git_head(checkout) or "0" * 40),
        head_ref=(environ.get("HEAD_REF") or "").strip(),
        run_id=required("GITHUB_RUN_ID", fallback="local"),
        run_attempt=(environ.get("GITHUB_RUN_ATTEMPT") or "1").strip(),
        dashboard=config.dashboard,
        base_sha=(environ.get("BASE_SHA") or "").strip(),
    )


class CellState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CACHED = "cached"  # a published result of this skill version is reused (counted in the progress line)
    SKIPPED = "skipped"  # outside the trigger's `cells=` selection AND no result of this skill version yet: never a job, in no count
    EXHAUSTED = "exhausted"  # MAX_REPORTLESS_ATTEMPTS consecutive NO-REPORT/ERROR results: counted, not re-run
    ERROR = "error"  # the launcher failed before publishing (💥); distinct from the published status ERROR


@dataclass
class Cell:
    skill: str
    case: str
    backend: str
    state: CellState = CellState.QUEUED
    status: str | None = None  # one of KNOWN_STATUSES once known (PASS | FAIL | PARTIAL | NO-REPORT | ERROR)
    url: str | None = None  # dashboard report link
    reason: str | None = None  # short explanation for ERROR cells; "<n> report-less attempts" for EXHAUSTED ones
    duration_s: float | None = None
    canary: bool = False  # the skill lives in CANARY_DIR, not under .agents/skills
    # the published result shown: its dashboard run id (`<dashboard>/runs/<run_id>/` — NOT the Actions run) and that
    # run's `created_at` as the catalog lists it; how refresh_from_catalog tells the cell's own result from a newer one
    run_id: str | None = None
    created_at: str | None = None
    # the plan's decision, kept apart from the state shown: True = this run executes the cell (a job), False = reused /
    # not selected / not re-run, None = not decided (a hand-built cell, a plan without the field): read the state. A
    # later refresh_from_catalog may show another run's newer result in an own cell (CACHED); the header's
    # `(n selected, m reused)`, the progress line's `reused` and the plan print count the decision, so they do not drift
    job: bool | None = None

    @property
    def is_job(self) -> bool:
        """This run executes the cell — the plan's decision, derived from the state while the plan has not recorded it."""
        if self.job is not None:
            return self.job
        return self.state not in (CellState.CACHED, CellState.SKIPPED, CellState.EXHAUSTED)

    def adopt(self, result: CachedResult, state: CellState) -> None:
        """Show this published result (status, link, run id, time) in the given state; the reason is cleared."""
        self.state, self.status, self.url, self.reason = state, result.status, result.url, None
        self.run_id, self.created_at = result.run_id, result.created_at or None

    @property
    def finished(self) -> bool:
        return self.state in (CellState.DONE, CellState.CACHED, CellState.ERROR, CellState.EXHAUSTED)

    @property
    def selected(self) -> bool:
        """Part of this run (a `(not selected)` cell is only shown for context)."""
        return self.state is not CellState.SKIPPED

    @property
    def passed(self) -> bool:
        return self.finished and self.status == PASS

    @property
    def name(self) -> str:
        return f"{self.skill}-{self.backend}-{self.case}"

    @property
    def label(self) -> str:
        return f"{self.skill}/{self.backend}/{self.case}"


@dataclass
class SkillPlan:
    name: str
    cases: list[str] = field(default_factory=list)
    skipped_reason: str | None = None  # set -> listed under "Not evaluated" instead of getting a table
    canary: bool = False


def skill_dir(skill: str, *, canary: bool = False) -> PurePosixPath:
    """Skill directory relative to the checkout root (= its path under /workspace/skill-source in the container)."""
    return CANARY_DIR if canary else SKILLS_DIR / skill


def eval_config_path(checkout: Path, skill: str, *, canary: bool = False) -> Path:
    return checkout / skill_dir(skill, canary=canary) / "eval" / "eval.config"


def load_skill_plan(checkout: Path, skill: str, *, canary: bool = False) -> SkillPlan:
    path = eval_config_path(checkout, skill, canary=canary)
    relative = path.relative_to(checkout)
    if not path.parent.parent.is_dir():
        reason = f"`{relative.parent.parent}` does not exist at this head (deleted or renamed)"
        return SkillPlan(skill, canary=canary, skipped_reason=reason)
    if not path.is_file():
        return SkillPlan(skill, canary=canary, skipped_reason=f"eval.config not found at `{relative}`")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return SkillPlan(skill, canary=canary, skipped_reason=f"`{relative}` is not valid JSON: {exc}")
    evals = data.get("evals") if isinstance(data, dict) else None
    if not isinstance(evals, list):
        return SkillPlan(skill, canary=canary, skipped_reason=f"`{relative}` has no `evals` list")
    cases = [str(entry["id"]) for entry in evals if isinstance(entry, dict) and entry.get("id")]
    return SkillPlan(skill, cases=cases, canary=canary)


def load_skill_plans(checkout: Path, skills: list[str], *, smoke: bool = True) -> list[SkillPlan]:
    """Plans for the canary (first, when it runs) followed by the given shipped skills, in order."""
    plans = [load_skill_plan(checkout, CANARY_SKILL, canary=True)] if smoke else []
    plans.extend(load_skill_plan(checkout, skill) for skill in skills)
    for plan in plans:
        label = f"canary {plan.name}" if plan.canary else f"skill {plan.name}"
        if plan.skipped_reason:
            warn(f"{label} not evaluated: {plan.skipped_reason}")
        else:
            log(f"{label}: {len(plan.cases)} case(s)")
    return plans


def build_cells(plans: list[SkillPlan], backends: list[str]) -> list[Cell]:
    """Cells = cases (file order) x backends (config order), skill by skill (canary first)."""
    cells = []
    for plan in plans:
        for case in plan.cases:
            for backend in backends:
                cells.append(Cell(skill=plan.name, case=case, backend=backend, canary=plan.canary))
    return cells


# ---------------------------------------------------------------------------
# dashboard cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedResult:
    """One cell's result as the dashboard lists it (from one catalog run that contains the cell)."""

    status: str
    url: str
    run_id: str
    created_at: str
    verdict: str | None = None  # the evaluator's verdict; a pre-v2 manifest may say FAIL with verdict NO-REPORT
    group_id: str | None = None
    source_run_id: str | None = None  # source.github.run_id: the Actions run that published it


def fetch_catalog(
    dashboard: str,
    skill: str,
    urlopen: Callable,
    timeout_s: float = CATALOG_TIMEOUT_S,
    *,
    retries: int = CATALOG_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """GET <dashboard>/api/catalog?skill=<skill>, retried with backoff; a final failure means 'no cache' (warning)."""
    url = f"{dashboard}/api/catalog?{urllib.parse.urlencode({'skill': skill})}"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    attempts = max(retries, 1)
    payload: object = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            break
        except Exception as exc:  # any failure at all: network, HTTP status, bad JSON
            if attempt >= attempts:
                warn(f"dashboard catalog unavailable for {skill} ({type(exc).__name__}: {exc}) — treating as no cache")
                return []
            delay = 2.0 ** (attempt - 1)
            warn(f"dashboard catalog GET for {skill} failed ({type(exc).__name__}: {exc}); retry in {delay:.0f}s")
            sleep(delay)
    if not isinstance(payload, list):
        warn(f"dashboard catalog for {skill} is not a list — treating as no cache")
        return []
    return [run for run in payload if isinstance(run, dict)]


UNDATED = datetime.min.replace(tzinfo=timezone.utc)  # sort key of a manifest without a usable `created_at`: oldest


def parse_created_at(value: object) -> datetime | None:
    """A manifest's ISO `created_at` as an aware UTC datetime; None when missing or unparsable."""
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass
    return None


def _created_at_key(value: object) -> datetime:
    return parse_created_at(value) or UNDATED


def _backend_result(run: dict, case: str, backend: str) -> tuple[str, str | None] | None:
    """(status, verdict) of (case, backend) inside one run, or None when the run does not contain the cell."""
    for case_entry in run.get("cases") or []:
        if not isinstance(case_entry, dict) or case_entry.get("case_id") != case:
            continue
        for backend_entry in case_entry.get("backends") or []:
            if isinstance(backend_entry, dict) and backend_entry.get("backend") == backend:
                status = backend_entry.get("status") or case_entry.get("status")
                if status not in KNOWN_STATUSES:
                    return None
                verdict = backend_entry.get("verdict")
                return status, (verdict if isinstance(verdict, str) else None)
    return None


def _run_result(run: dict, case: str, backend: str, dashboard: str) -> CachedResult | None:
    """The cell's result inside one catalog run, or None when the run does not contain the cell."""
    if not run.get("run_id"):
        return None
    found = _backend_result(run, case, backend)
    if found is None:
        return None
    source = run.get("source") if isinstance(run.get("source"), dict) else {}
    github = source.get("github") if isinstance(source.get("github"), dict) else {}
    status, verdict = found
    return CachedResult(
        status=status,
        url=f"{dashboard}/runs/{run['run_id']}/",
        run_id=str(run["run_id"]),
        created_at=str(run.get("created_at") or ""),
        verdict=verdict,
        group_id=str(run["group_id"]) if run.get("group_id") else None,
        source_run_id=str(github["run_id"]) if github.get("run_id") else None,
    )


def _newer(candidate: CachedResult, best: CachedResult | None) -> bool:
    return best is None or _created_at_key(candidate.created_at) > _created_at_key(best.created_at)


def newest_result(
    runs: list[dict], version: SkillVersion, case: str, backend: str, dashboard: str
) -> CachedResult | None:
    """Newest run of this skill version (content sha or an equivalent commit) that contains the cell, any status."""
    best: CachedResult | None = None
    for run in runs:
        if not version.matches(run):
            continue
        candidate = _run_result(run, case, backend, dashboard)
        if candidate is not None and _newer(candidate, best):
            best = candidate
    return best


def reportless_streak(runs: list[dict], version: SkillVersion, case: str, backend: str, dashboard: str) -> int:
    """How many of the cell's newest version-matching results in a row ended without a report (NO-REPORT / ERROR,
    or a pre-v2 FAIL whose verdict says NO-REPORT). Stops at the first PASS / FAIL / PARTIAL."""
    results = [r for r in (_run_result(run, case, backend, dashboard) for run in runs if version.matches(run)) if r]
    results.sort(key=lambda r: _created_at_key(r.created_at), reverse=True)
    streak = 0
    for result in results:
        if result.status in (NO_REPORT, ERROR_STATUS) or result.verdict in (NO_REPORT, ERROR_STATUS):
            streak += 1
        else:
            break
    return streak


def static_score_for(catalog_runs: list[dict], group_id: str | None) -> int | None:
    """The skill version's static-assessment score as the dashboard knows it: the `static_score` (int 0–100) of the
    NEWEST run of the group that carries one. The evaluator assesses once per run and a group shares the skill
    content, so the group's values agree; an older manifest may lack the field. None without a group, a match or
    a usable (int) value."""
    if not group_id:
        return None
    best: dict | None = None
    for run in catalog_runs:
        if not isinstance(run, dict) or str(run.get("group_id") or "") != group_id:
            continue
        score = run.get("static_score")
        if isinstance(score, bool) or not isinstance(score, int):  # null, a string, a bool: not a score
            continue
        if best is None or _created_at_key(run.get("created_at")) > _created_at_key(best.get("created_at")):
            best = run
    return None if best is None else int(best["static_score"])


def static_scores_for(plans: list[SkillPlan], info: RunInfo, catalogs: Mapping[str, list[dict]]) -> dict[str, int]:
    """skill -> static score of its group, for the skills whose catalog is at hand and lists one (heading suffix)."""
    scores: dict[str, int] = {}
    for plan in plans:
        score = static_score_for(catalogs.get(plan.name) or [], info.group_for(plan.name))
        if score is not None:
            scores[plan.name] = score
    return scores


def reusable(result: CachedResult | None, cacheable: frozenset[str]) -> bool:
    """The cache rule: the status is in the active cacheable set AND the verdict is neither NO-REPORT nor ERROR.

    The verdict guard is for pre-v2 manifests, which say FAIL for an agent that never wrote a report.
    """
    return result is not None and result.status in cacheable and result.verdict not in (NO_REPORT, ERROR_STATUS)


def find_cached_result(
    runs: list[dict],
    version: SkillVersion,
    case: str,
    backend: str,
    dashboard: str,
    cacheable: frozenset[str] = CACHEABLE_DEFAULT,
) -> CachedResult | None:
    """The newest version-matching result of the cell when it may be reused, else None."""
    result = newest_result(runs, version, case, backend, dashboard)
    return result if reusable(result, cacheable) else None


def find_run_result(
    runs: list[dict], info: RunInfo, case: str, backend: str, *, started_at: datetime | None
) -> CachedResult | None:
    """The cell's result as published by THIS Actions run (`source.github.run_id`), whatever its status.

    Fallback: the newest version-matching run created since the plan started (a manifest without the run id).
    Older runs of the version are never taken: the cache already decided about them — and in `force` mode they
    are exactly the results being replaced. `started_at=None` disables the fallback.
    """
    best: CachedResult | None = None
    for run in runs:
        candidate = _run_result(run, case, backend, info.dashboard)
        if candidate is not None and candidate.source_run_id == info.run_id and _newer(candidate, best):
            best = candidate
    if best is not None or started_at is None:
        return best
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    for run in runs:
        skill_block = run.get("skill") if isinstance(run.get("skill"), dict) else {}
        if not info.version_for(str(skill_block.get("name") or "")).matches(run):
            continue
        candidate = _run_result(run, case, backend, info.dashboard)
        if candidate is None or _created_at_key(candidate.created_at) < started_at:
            continue
        if _newer(candidate, best):
            best = candidate
    return best


def apply_cache(
    cells: list[Cell],
    info: RunInfo,
    urlopen: Callable,
    *,
    sleep: Callable[[float], None] = time.sleep,
    catalogs: Mapping[str, list[dict]] | None = None,
    options: TriggerOptions | None = None,
) -> int:
    """Decide every cell from the dashboard. `(reused)` when its newest version-matching result passes the cache
    rule, otherwise it runs. A cell the trigger's `cells=` leaves out is `(not selected)` — and then takes the
    version's newest result as a reused cell whatever that result's status (refresh_from_catalog, below), so a
    partial re-run's comment shows the whole matrix as the dashboard group does, with only the selected cells
    replaced. Catalogs the caller already fetched are reused. Returns the number of reused cells.
    """
    options = options or TriggerOptions()
    catalogs = dict(catalogs or {})
    for cell in cells:
        if cell.skill not in catalogs:
            catalogs[cell.skill] = fetch_catalog(info.dashboard, cell.skill, urlopen, sleep=sleep)
        if not options.selects(cell):
            cell.state = CellState.SKIPPED  # no job; the refresh below carries the version's newest result, if any
            continue
        version = info.version_for(cell.skill)
        newest = newest_result(catalogs[cell.skill], version, cell.case, cell.backend, info.dashboard)
        if reusable(newest, options.cacheable):
            cell.adopt(newest, CellState.CACHED)
            log(f"reused {cell.label}: {newest.status} {newest.url}")
        elif newest is not None:
            streak = reportless_streak(catalogs[cell.skill], version, cell.case, cell.backend, info.dashboard)
            if streak >= MAX_REPORTLESS_ATTEMPTS and not options.force and not options.cells:
                # Three report-less results in a row for the same skill content: another automatic re-run would most
                # likely produce a fourth. The cell keeps its latest status in the matrix and the counts; an explicit
                # `force` or `cells=` trigger still runs it.
                cell.adopt(newest, CellState.EXHAUSTED)
                cell.reason = f"{streak} report-less attempts"
                log(
                    f"not re-run {cell.label}: {streak} consecutive report-less results ({newest.status}, {newest.run_id})"
                )
                continue
            log(
                f"run {cell.label}: the newest result {newest.status} (verdict {newest.verdict or '-'},"
                f" {newest.run_id}) is not reused in this mode"
            )
    # the not-selected cells take the version's newest result (same catalogs: no further GET) — the first comment of a
    # partial run is complete for everything published so far, including a concurrent run's cells
    refresh_from_catalog(cells, info, catalogs, at_plan=True)
    for cell in cells:  # the decision, recorded: a later refresh may show another run's result in a cell that ran
        cell.job = cell.state is CellState.QUEUED
    return sum(1 for cell in cells if cell.state is CellState.CACHED)


def run_id_from_url(url: str | None) -> str | None:
    """`<dashboard>/runs/<run_id>/` -> `<run_id>`, the dashboard run that holds the result; None for any other link."""
    match = re.search(r"/runs/([^/?#]+)/?(?:[?#].*)?$", url or "")
    return match.group(1) if match else None


def _catalog_created_at(runs: list[dict], run_id: str) -> str | None:
    """`created_at` of the catalog run with this id; None while the dashboard does not list it (or lists no date)."""
    for run in runs:
        if str(run.get("run_id") or "") == run_id:
            return str(run["created_at"]) if run.get("created_at") else None
    return None


def refresh_from_catalog(
    cells: list[Cell], info: RunInfo, catalogs: Mapping[str, list[dict]], *, at_plan: bool = False
) -> int:
    """Newest result per cell, like the dashboard group: every cell this run is not about to run shows the skill
    version's newest published result for it, whichever run produced it — at every update, not only at plan time.

    - `(not selected)` (SKIPPED) with a result listed now -> CACHED with it (at plan time: the catalog already had it;
      later: published since the plan, e.g. by the full run this partial run overlaps with);
    - DONE / CACHED / EXHAUSTED whose own result — identified by the dashboard run id of its link — is older than the
      version's newest for the cell -> CACHED with the newer one, no reason: rendered like a fresh result, counted by
      the status shown ("superseded" in the log), not as reused (Cell.is_job stays). Newest = the result whose run
      STARTED last: the manifest `created_at` the publisher stamps at the run's start (second resolution), the order
      the dashboard shows a group in; equal seconds keep the own result. A cell started earlier never supersedes one
      started later, however late it finishes.
    Never a downgrade on a guess: the own result stays when it is the newest, when its run is not listed yet (its
    `created_at` is unknown — the next refresh compares once the catalog dates it) or when the skill's catalog was not
    fetched. QUEUED / RUNNING cells are about to publish their own result and a launcher error (💥) has none to
    compare: both are left alone. Returns the number of cells changed.
    """
    changed = 0
    for cell in cells:
        if cell.state not in (CellState.DONE, CellState.CACHED, CellState.SKIPPED, CellState.EXHAUSTED):
            continue
        runs = catalogs.get(cell.skill)
        if runs is None:
            continue
        newest = newest_result(runs, info.version_for(cell.skill), cell.case, cell.backend, info.dashboard)
        if newest is None:
            continue
        if cell.state is CellState.SKIPPED:
            cell.adopt(newest, CellState.CACHED)
            changed += 1
            since = "" if at_plan else ", published since the plan"
            log(f"reused {cell.label} (outside cells={since}): {newest.status} {newest.url}")
            continue
        own_run_id = cell.run_id or run_id_from_url(cell.url)
        if own_run_id is None or newest.run_id == own_run_id:
            continue  # the cell's own result is the newest (or cannot be told apart from it)
        own_created = parse_created_at(cell.created_at or _catalog_created_at(runs, own_run_id))
        newest_created = parse_created_at(newest.created_at)
        if own_created is None or newest_created is None or newest_created <= own_created:
            continue
        previous = f"{cell.status} ({own_run_id})"
        cell.adopt(newest, CellState.CACHED)
        changed += 1
        log(f"superseded {cell.label}: {previous} -> {newest.status} — newer result from run {newest.run_id}")
    return changed


# ---------------------------------------------------------------------------
# trigger arguments: `/run-skill-eval [smoke-only|with-smoke] [force] [rerun-failed] [cells=<p1,p2>] [group=<id>]`
# ---------------------------------------------------------------------------

TRIGGER_KEYWORDS = ("smoke-only", "smoke", "with-smoke", "force", "rerun-failed", "rerun")


@dataclass
class TriggerOptions:
    """What the trigger asked for (comment text, label or dispatch inputs), parsed by parse_trigger_args."""

    scope: str = "full"  # full | smoke (`smoke-only`: canary only, skill detection skipped)
    with_smoke: bool = False  # `with-smoke`: the canary first, then the skills (scope full; `smoke: false` beats it)
    mode: str = "default"  # default | rerun-failed
    force: bool = False  # re-run the selected cells even when a reusable result exists
    cells: list[str] = field(default_factory=list)  # fnmatch patterns `<backend>/<case>` | `<skill>/<backend>/<case>`
    group: str | None = None  # dashboard group id for every skill of this run
    errors: list[str] = field(default_factory=list)  # tokens that were not understood (reported, defaults used)

    @property
    def cacheable(self) -> frozenset[str]:
        """The statuses a published result may have to be reused instead of re-run."""
        if self.force:
            return frozenset()
        if self.mode == "rerun-failed":
            return CACHEABLE_RERUN_FAILED
        return CACHEABLE_DEFAULT

    def selects(self, cell: Cell) -> bool:
        """Does `cells=` (when given) match this cell? Patterns are matched case-sensitively, `*` crosses `/`."""
        if not self.cells:
            return True
        short = f"{cell.backend}/{cell.case}"
        return any(fnmatch.fnmatchcase(short, p) or fnmatch.fnmatchcase(cell.label, p) for p in self.cells)

    @property
    def canary_request(self) -> str | None:
        """`smoke-only` / `with-smoke` when the trigger asked for the canary, else None (off by default)."""
        if self.scope == "smoke":
            return "smoke-only"
        return "with-smoke" if self.with_smoke else None

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "with_smoke": self.with_smoke,
            "mode": self.mode,
            "force": self.force,
            "cells": list(self.cells),
            "group": self.group,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> TriggerOptions:
        return cls(
            scope=str(data.get("scope") or "full"),
            with_smoke=bool(data.get("with_smoke")),
            mode=str(data.get("mode") or "default"),
            force=bool(data.get("force")),
            cells=[str(pattern) for pattern in data.get("cells") or []],
            group=str(data["group"]) if data.get("group") else None,
            errors=[str(error) for error in data.get("errors") or []],
        )


def parse_trigger_args(text: str | None) -> TriggerOptions:
    """Parse the text after `/run-skill-eval` (or the label's / dispatch's equivalent) into TriggerOptions.

    Whitespace-separated tokens: `smoke-only` (alias `smoke`), `with-smoke`, `force`, `rerun-failed`,
    `cells=<p1,p2>` (alias `rerun=<p1,p2>`, or `rerun <p1,p2>` with the list as the next token), `group=<id>`.
    Patterns are fnmatch globs over `<backend>/<case>` or `<skill>/<backend>/<case>`. Anything else lands in
    `errors` and is ignored, so a typo
    never turns a selective re-run into something else silently: the comment shows what was not understood.
    """
    options = TriggerOptions()
    tokens = (text or "").split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        key, separator, value = token.partition("=")
        lowered = key.lower()
        if lowered in ("smoke-only", "smoke") and not separator:
            options.scope = "smoke"
        elif lowered == "with-smoke" and not separator:
            options.with_smoke = True
        elif lowered == "force" and not separator:
            options.force = True
        elif lowered == "rerun-failed" and not separator:
            options.mode = "rerun-failed"
        elif (lowered == "cells" and separator) or lowered == "rerun":
            if not separator:  # `rerun <patterns>`: the list is the next token, unless that is a keyword
                following = tokens[index] if index < len(tokens) else ""
                if following and "=" not in following and following.lower() not in TRIGGER_KEYWORDS:
                    value = following
                    index += 1
                else:
                    options.errors.append(f"`{token}`: expected a cell list, e.g. `rerun codex/my-case,cursor/*`")
                    continue
            patterns = [pattern for pattern in value.split(",") if pattern]
            if not patterns or any("/" not in pattern for pattern in patterns):
                options.errors.append(
                    f"`{token}`: cells are `<backend>/<case>` or `<skill>/<backend>/<case>` patterns, comma-separated"
                )
            options.cells.extend(pattern for pattern in patterns if "/" in pattern)
        elif lowered == "group" and separator:
            if GROUP_ID_RE.fullmatch(value):
                options.group = value
            else:
                options.errors.append(f"`{token}`: a group id is letters, digits, `.`, `_` or `-` (1–100 chars)")
        else:
            options.errors.append(f"unknown argument `{token}`")
    return options


def describe_trigger(options: TriggerOptions) -> str:
    parts = [f"scope {options.scope}", f"mode {options.mode}"]
    if options.canary_request:
        parts.append(f"canary {options.canary_request}")
    if options.force:
        parts.append("force")
    if options.cells:
        parts.append("cells=" + ",".join(options.cells))
    if options.group:
        parts.append(f"group {options.group}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# GitHub REST client (urllib, retries)
# ---------------------------------------------------------------------------


class TransportError(Exception):
    """Connection-level failure (no HTTP status)."""


class GitHubApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"GitHub API {status}: {detail}")
        self.status = status


# (method, url, headers, body, timeout) -> (status, headers, body). This alias is evaluated at import time, so
# no `X | None` here: the README's local commands run on whatever `python3` the box has (macOS ships 3.9).
Transport = Callable[[str, str, dict[str, str], Optional[bytes], float], tuple[int, dict[str, str], bytes]]


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        # HTTPException (IncompleteRead, BadStatusLine, ...) is not an OSError: without it here a torn
        # response would escape both the retry loop and the comment updater's error handling.
        raise TransportError(f"{type(exc).__name__}: {exc}") from exc


class GitHubApi:
    def __init__(
        self,
        server_url: str,
        repo: str,
        token: str,
        *,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        retries: int = GITHUB_RETRIES,
        timeout: float = 30.0,
    ):
        self.repo = repo
        self.token = token
        self.transport = transport
        self.sleep = sleep
        self.retries = retries
        self.timeout = timeout
        server_url = server_url.rstrip("/")
        self.api_base = "https://api.github.com" if server_url == "https://github.com" else f"{server_url}/api/v3"

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, payload: dict | None = None, *, retries: int | None = None) -> object:
        """JSON request with 3 attempts and exponential backoff on 5xx / connection errors."""
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        retries = self.retries if retries is None else retries
        last_error = "unknown error"
        for attempt in range(1, retries + 1):
            try:
                status, _headers, data = self.transport(method, url, self._headers(), body, self.timeout)
            except TransportError as exc:
                last_error = str(exc)
                status, data = None, b""
            if status is not None and status < 500:
                text = data.decode("utf-8", errors="replace")
                if status >= 400:
                    raise GitHubApiError(status, text[:300])
                return json.loads(text) if text.strip() else None
            if status is not None:
                last_error = f"HTTP {status}"
            if attempt < retries:
                delay = 2.0 ** (attempt - 1)
                warn(f"GitHub {method} {path} failed ({last_error}); retry {attempt}/{retries - 1} in {delay:.0f}s")
                self.sleep(delay)
        raise GitHubApiError(0, f"{method} {path} failed after {retries} attempts: {last_error}")

    def get_pull(self, number: int) -> dict:
        result = self.request("GET", f"/repos/{self.repo}/pulls/{number}")
        if not isinstance(result, dict):
            raise GitHubApiError(0, f"unexpected payload for pull #{number}")
        return result

    def list_issue_comments(self, number: int) -> list[dict]:
        comments: list[dict] = []
        page = 1
        while True:
            result = self.request("GET", f"/repos/{self.repo}/issues/{number}/comments?per_page=100&page={page}")
            batch = result if isinstance(result, list) else []
            comments.extend(item for item in batch if isinstance(item, dict))
            if len(batch) < 100:
                return comments
            page += 1

    def create_issue_comment(self, number: int, body: str, *, retries: int | None = None) -> dict:
        result = self.request("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body}, retries=retries)
        return result if isinstance(result, dict) else {}

    def update_issue_comment(self, comment_id: int, body: str) -> dict:
        result = self.request("PATCH", f"/repos/{self.repo}/issues/comments/{comment_id}", {"body": body})
        return result if isinstance(result, dict) else {}

    def get_issue_comment(self, comment_id: int) -> dict:
        result = self.request("GET", f"/repos/{self.repo}/issues/comments/{comment_id}")
        return result if isinstance(result, dict) else {}

    def list_run_jobs(self, run_id: str) -> list[dict]:
        """All jobs of the latest attempt of a workflow run (paginated, 100 per page)."""
        jobs: list[dict] = []
        page = 1
        while True:
            result = self.request("GET", f"/repos/{self.repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}")
            batch = result.get("jobs") if isinstance(result, dict) else None
            batch = [item for item in batch if isinstance(item, dict)] if isinstance(batch, list) else []
            jobs.extend(batch)
            total = result.get("total_count") if isinstance(result, dict) else None
            if len(batch) < 100 or (isinstance(total, int) and len(jobs) >= total):
                return jobs
            page += 1


# ---------------------------------------------------------------------------
# which skills does the PR touch? (GitHub compare API, git fallback)
# ---------------------------------------------------------------------------


def skills_in_paths(paths: Iterable[str]) -> list[str]:
    """Sorted skill names owning the given repo-relative paths (`.agents/skills/<name>/...`).

    The canary's name is reserved: a leftover copy under .agents/skills is never a shipped skill.
    """
    depth = len(SKILLS_DIR.parts)
    names: set[str] = set()
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) > depth + 1 and parts[:depth] == SKILLS_DIR.parts:  # a file *inside* the skill directory
            names.add(parts[depth])
    if CANARY_SKILL in names:
        names.discard(CANARY_SKILL)
        warn(f"ignoring {SKILLS_DIR / CANARY_SKILL}: the canary lives in {CANARY_DIR} and is never a shipped skill")
    return sorted(names)


def compare_changed_paths(api: GitHubApi, base_sha: str, head_sha: str) -> list[str] | None:
    """Paths changed between base and head per `GET /repos/{repo}/compare/{base}...{head}`.

    GitHub puts the whole `files` list — at most 300 entries — on the FIRST page and leaves the key out of the
    later pages (`per_page`/`page` paginate the commits, not the files). The loop asks for another page only while
    a page comes back exactly full: a longer page means GitHub sent everything at once, a shorter one (or one
    without `files`) ends the list — so a paginated `files` list would be read completely as well.
    Renamed entries contribute both names. Returns None when the API cannot be trusted — a request failed, the
    first page has no `files` list, or the listing reached the 300-file cap (it may be truncated) — so that the
    caller falls back to git.
    """
    paths: list[str] = []
    seen = 0
    page = 1
    while True:
        path = f"/repos/{api.repo}/compare/{base_sha}...{head_sha}?per_page={COMPARE_PAGE_SIZE}&page={page}"
        try:
            result = api.request("GET", path)
        except (GitHubApiError, ValueError) as exc:  # ValueError: a 2xx with a non-JSON body
            warn(f"GitHub compare unavailable ({exc}) — falling back to git")
            return None
        files = result.get("files") if isinstance(result, dict) else None
        if not isinstance(files, list):
            if page > 1:  # a later page without `files`: the first page already listed them all
                return paths
            warn("GitHub compare returned no `files` list — falling back to git")
            return None
        for entry in files:
            if isinstance(entry, dict):
                paths.extend(str(entry[key]) for key in ("filename", "previous_filename") if entry.get(key))
        seen += len(files)
        if seen >= COMPARE_MAX_FILES:
            warn(
                f"GitHub compare lists {seen} files (API cap {COMPARE_MAX_FILES}, possibly truncated) — falling back to git"
            )
            return None
        if len(files) != COMPARE_PAGE_SIZE:  # shorter: last page of a paginated list; longer: the whole list at once
            return paths
        page += 1


def git_changed_paths(checkout: Path, base_sha: str) -> list[str]:
    """`git diff --name-only --no-renames -z <base>...HEAD` in the checkout (needs the base commit and a merge base).

    `-z` yields NUL-separated raw paths: without it git quotes a path with non-ASCII bytes
    (`".agents/skills/x/assets/caf\\303\\251.json"`) and that skill would go undetected.
    """
    command = ["git", "-C", str(checkout), "diff", "--name-only", "--no-renames", "-z", f"{base_sha}...HEAD"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
    else:
        if result.returncode == 0:
            return [path for path in result.stdout.decode("utf-8", errors="replace").split("\0") if path]
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        detail = (stderr or f"exit {result.returncode}").splitlines()[-1]
    raise OrchestrationError(
        f"cannot list the files changed by the PR: `git diff --name-only {base_sha[:7]}...HEAD` failed in {checkout}"
        f" ({detail}). Fetch the base commit first (git fetch --depth=1 origin {base_sha}), or set `skills:` in the"
        " config to bypass detection."
    )


def canary_touched(paths: Iterable[str]) -> bool:
    """True when a file *inside* the canary's directory changed."""
    depth = len(CANARY_DIR.parts)
    return any(
        len(PurePosixPath(p).parts) > depth and PurePosixPath(p).parts[:depth] == CANARY_DIR.parts for p in paths
    )


def changed_paths(info: RunInfo, checkout: Path, api: GitHubApi | None) -> list[str]:
    """Repo-relative paths the PR changes (GitHub compare API, git fallback)."""
    if not info.base_sha:
        raise OrchestrationError(
            "environment variable BASE_SHA (the PR base sha) is required to detect the skills touched by the PR"
            " — or set `skills:` in the config to bypass detection"
        )
    paths = compare_changed_paths(api, info.base_sha, info.head_sha) if api is not None else None
    source = "GitHub compare API"
    if paths is None:
        paths = git_changed_paths(checkout, info.base_sha)
        source = "git diff"
    log(f"{len(paths)} changed path(s) via {source} ({info.base_sha[:7]}...{info.sha7})")
    return paths


def detect_changed_skills(info: RunInfo, checkout: Path, api: GitHubApi | None) -> list[str]:
    """Shipped skills touched by the PR: `.agents/skills/<name>/**` in the base...head diff, alphabetical."""
    skills = skills_in_paths(changed_paths(info, checkout, api))
    log(f"skills touched: {', '.join(skills) or 'none'}")
    return skills


def select_plan(
    config: Config, info: RunInfo, checkout: Path, api: GitHubApi | None, *, with_smoke: bool = False
) -> tuple[list[str], bool]:
    """(shipped skills to evaluate, run the canary?).

    Skills: the config's explicit `skills:` list, otherwise detection from the PR diff. Canary: off unless the
    trigger said `with-smoke`; `smoke: true|false` in the config beats both (always / never). Whether the PR
    touches the canary's own directory is logged but decides nothing. (Scope `smoke` never gets here.)
    """
    if config.skills is None:
        paths = changed_paths(info, checkout, api)
        skills = skills_in_paths(paths)
        touched = "yes" if canary_touched(paths) else "no"
        log(f"skills touched: {', '.join(skills) or 'none'}; canary touched: {touched} (informational)")
    else:
        skills = list(dict.fromkeys(config.skills))
        if CANARY_SKILL in skills:
            skills.remove(CANARY_SKILL)
            warn(
                f"`skills:` lists {CANARY_SKILL}: the canary is controlled by the trigger (`with-smoke` / `smoke-only`)"
                " and `smoke:`, not by `skills:` — ignored there"
            )
        log(
            f"skills from the config's explicit `skills:` list (PR diff detection skipped): {', '.join(skills) or 'none'}"
        )
    if config.smoke is None:
        run_canary = with_smoke
        why = "trigger argument `with-smoke`" if with_smoke else "not requested: no `with-smoke`, `smoke:` unset"
    else:
        run_canary = config.smoke
        why = f"`smoke: {'true' if config.smoke else 'false'}` in the config"
        if with_smoke and not config.smoke:
            why += " wins over the trigger argument `with-smoke`"
    log(f"canary {CANARY_SKILL}: {'on' if run_canary else 'off'} ({why})")
    return skills, run_canary


def select_skills(config: Config, info: RunInfo, checkout: Path, api: GitHubApi | None) -> list[str]:
    """The shipped skills to evaluate (see select_plan)."""
    return select_plan(config, info, checkout, api)[0]


# ---------------------------------------------------------------------------
# skill version: content sha, git tree sha, equivalent commits, dashboard group
# ---------------------------------------------------------------------------


def skill_content_sha256(skill_dir: Path) -> str:
    """SHA-256 over the content of a skill directory, independent of git.

    Identical algorithm to the publisher's `cosmos_eval.metadata.skill_content_sha256` (the manifest's
    `skill.content_sha256`), so a catalog run can be matched to the checkout without any commit in common: walk the
    regular files under the directory (`os.walk`, symlinked directories not followed), skip directories named
    `__pycache__` and files named `.DS_Store` or `*.pyc`, sort by POSIX relative path and feed
    `relpath + "\0" + sha256(file bytes).hexdigest() + "\n"` per file into one SHA-256.
    """
    if not skill_dir.is_dir():  # never hash "nothing" quietly: every missing directory would share one sha
        raise FileNotFoundError(f"skill directory not found: {skill_dir}")
    entries: list[tuple[str, Path]] = []
    for root, dirnames, filenames in os.walk(skill_dir):
        dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
        for name in filenames:
            if name == ".DS_Store" or name.endswith(".pyc"):
                continue
            path = Path(root) / name
            if not path.is_file():  # broken symlinks, sockets, ...
                continue
            entries.append((path.relative_to(skill_dir).as_posix(), path))
    digest = hashlib.sha256()
    for relpath, path in sorted(entries):
        digest.update(f"{relpath}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode("utf-8"))
    return digest.hexdigest()


def skill_tree_sha(checkout: Path, sha: str, skill_dir_rel: str | PurePosixPath) -> str | None:
    """`git -C <checkout> rev-parse <sha>:<path>`: the tree id of the skill directory at that commit; None on error.

    None also when the commit is not in the clone (a force-pushed-away PR head, a depth-1 checkout): such a commit
    is then simply not equivalent to HEAD.
    """
    command = ["git", "-C", str(checkout), "rev-parse", "--verify", "--quiet", f"{sha}:{skill_dir_rel}"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def anchored_group(runs: list[dict], version: SkillVersion, info: RunInfo) -> str | None:
    """Group id of this PR's newest catalog run of the skill version, so a re-run joins it instead of opening a new one.

    Only groups of this PR (`ci-pr<N>-…`) qualify: another PR's or main's runs of the same version are cache
    material, not this PR's history.
    """
    prefix = f"{PR_GROUP_PREFIX}{info.pr_number}-"
    best: dict | None = None
    for run in runs:
        group_id = run.get("group_id")
        if not isinstance(group_id, str) or not group_id.startswith(prefix) or not version.matches(run):
            continue
        if best is None or _created_at_key(run.get("created_at")) > _created_at_key(best.get("created_at")):
            best = run
    return str(best["group_id"]) if best is not None else None


def resolve_skill_version(checkout: Path, info: RunInfo, plan: SkillPlan, runs: list[dict]) -> SkillVersion:
    """Content sha, tree sha and equivalent commits of the plan's skill at HEAD, plus its dashboard group.

    Group: `--group`/`group=` when given; else the PR's newest run of this version (anchoring); else a new
    `ci-pr<N>-<sha7>`. Equivalence is best effort: a commit git cannot resolve is just not equivalent.
    """
    path = skill_dir(plan.name, canary=plan.canary)
    content = skill_content_sha256(checkout / path)
    tree = skill_tree_sha(checkout, info.head_sha, path)
    equivalent = [info.head_sha]
    if tree is not None:
        seen = {info.head_sha}
        for run in runs:
            skill_block = run.get("skill") if isinstance(run.get("skill"), dict) else {}
            sha = skill_block.get("commit_sha")
            if not isinstance(sha, str) or not sha or sha in seen:
                continue
            seen.add(sha)
            if skill_tree_sha(checkout, sha, path) == tree:
                equivalent.append(sha)
    version = SkillVersion(content, tree, equivalent)
    if info.group_override:
        version.group_id, origin = info.group_override, "set by group="
    else:
        anchored = anchored_group(runs, version, info)
        version.group_id = anchored or info.default_group
        origin = "the PR's existing group for this skill version" if anchored else "new"
    log(
        f"{'canary' if plan.canary else 'skill'} {plan.name}: content {content[:7]} · tree {(tree or '-')[:7]}"
        f" · {len(equivalent)} equivalent commit(s) · group {version.group_id} ({origin})"
    )
    return version


def resolve_skill_versions(
    checkout: Path, info: RunInfo, plans: list[SkillPlan], catalogs: Mapping[str, list[dict]]
) -> dict[str, SkillVersion]:
    """SkillVersion per evaluated plan (skipped plans have no directory to hash); failures are warnings."""
    versions: dict[str, SkillVersion] = {}
    for plan in plans:
        if plan.skipped_reason:
            continue
        try:
            versions[plan.name] = resolve_skill_version(checkout, info, plan, catalogs.get(plan.name) or [])
        except OSError as exc:  # unreadable file in the skill dir: the cell still runs, under the default group
            warn(f"{plan.name}: cannot compute the skill version ({exc}); using the exact commit and the default group")
    return versions


# ---------------------------------------------------------------------------
# the ONE comment: find-or-create, rate-limited PATCHes, never raises
# ---------------------------------------------------------------------------


def start_daemon_timer(delay_s: float, callback: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(delay_s, callback)
    timer.daemon = True
    timer.start()
    return timer


class PrintingCommentUpdater:
    """--no-github: print every would-be comment body to stdout."""

    def __init__(self) -> None:
        self.updates = 0
        self.last_body: str | None = None

    def publish(self, body: str, *, final: bool = False) -> None:
        self.updates += 1
        self.last_body = body
        kind = "final" if final else f"update {self.updates}"
        print(f"----- comment body ({kind}) -----\n{body}\n----- end comment body -----", flush=True)

    def flush(self) -> None:
        return None


class GitHubCommentUpdater:
    """Keeps one PR comment (identified by its marker line) in sync with the matrix.

    PATCHes are rate-limited to one per `min_interval_s`: an update that arrives early is
    parked and flushed by a timer once the window elapses (later updates replace it, so only
    the newest body is sent). `final=True` bypasses the limit. API failures are logged, never raised.
    """

    def __init__(
        self,
        api: GitHubApi,
        pr_number: int,
        marker: str,
        *,
        min_interval_s: float = COMMENT_MIN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        start_timer: Callable[[float, Callable[[], None]], object] = start_daemon_timer,
    ):
        self.api = api
        self.pr_number = pr_number
        self.marker = marker
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.start_timer = start_timer
        self.comment_id: int | None = None
        self.failures = 0
        self._pending: str | None = None
        self._timer: object | None = None
        self._last_sent_at: float | None = None
        self._lock = threading.Lock()

    def publish(self, body: str, *, final: bool = False) -> None:
        with self._lock:
            if final or self._due():
                self._cancel_timer()
                self._pending = None
                self._send(body)
                return
            self._pending = body
            if self._timer is None:
                elapsed = self.clock() - (self._last_sent_at or 0.0)
                self._timer = self.start_timer(max(self.min_interval_s - elapsed, 0.0), self.flush)

    def flush(self) -> None:
        """Send the parked body, if any (called by the timer and at shutdown)."""
        with self._lock:
            self._timer = None
            if self._pending is not None:
                body, self._pending = self._pending, None
                self._send(body)

    def _due(self) -> bool:
        return self._last_sent_at is None or self.clock() - self._last_sent_at >= self.min_interval_s

    def _cancel_timer(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None and hasattr(timer, "cancel"):
            timer.cancel()

    def _send(self, body: str) -> None:
        try:
            if self.comment_id is None:
                self.comment_id = self._find_or_create(body)
            else:
                self.api.update_issue_comment(self.comment_id, body)
            self._last_sent_at = self.clock()
        except Exception as exc:  # anything at all: a comment update must never take the matrix down
            self.failures += 1
            warn(f"comment update failed (ignored): {type(exc).__name__}: {exc}")

    def find_existing(self) -> int | None:
        """Id of the first comment whose body starts with the PR's marker line, if any."""
        for comment in self.api.list_issue_comments(self.pr_number):
            if str(comment.get("body") or "").startswith(self.marker) and comment.get("id") is not None:
                return int(comment["id"])
        return None

    def _find_or_create(self, body: str) -> int:
        """Reuse the marker comment or create it exactly once.

        The POST is never retried blindly: a lost response may hide a comment that was in fact created,
        so every retry lists the comments again first (that is what keeps the ONE-comment guarantee).
        """
        for attempt in range(1, self.api.retries + 1):
            existing = self.find_existing()
            if existing is not None:
                log(f"reusing comment {existing}")
                self.api.update_issue_comment(existing, body)
                return existing
            try:
                created = self.api.create_issue_comment(self.pr_number, body, retries=1)
            except GitHubApiError as exc:
                if 400 <= exc.status < 500 or attempt == self.api.retries:
                    raise
                delay = 2.0 ** (attempt - 1)
                warn(f"comment creation failed ({exc}); re-listing comments before retry {attempt} in {delay:.0f}s")
                self.api.sleep(delay)
                continue
            if created.get("id") is None:
                raise GitHubApiError(0, "comment creation returned no id")
            comment_id = int(created["id"])
            log(f"created comment {comment_id}")
            return comment_id
        raise GitHubApiError(0, "comment creation gave up")  # only reachable with retries == 0


# ---------------------------------------------------------------------------
# comment rendering
# ---------------------------------------------------------------------------


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


ErrorGroups = list[tuple[str, list[str]]]  # (reason, cell labels): identical reasons share one numbered entry


def render_cell_text(cell: Cell, errors: ErrorGroups, reason_limit: int | None) -> str:
    if cell.state is CellState.QUEUED:
        return "⏳ queued"
    if cell.state is CellState.RUNNING:
        return "🟢 publishing" if cell.reason == PUBLISHING else "🟢 running"
    if cell.state is CellState.SKIPPED:  # outside `cells=`, no result of the version yet (one listed makes it CACHED)
        return "— not selected"
    if cell.state is CellState.EXHAUSTED:
        link = f" · [report]({cell.url})" if cell.url else ""
        return f"{STATUS_ICON.get(cell.status, '⚠️')} {cell.status} (not re-run: {cell.reason}){link}"
    if cell.state is CellState.ERROR:
        reason = cell.reason or "launcher failed without publishing"
        if reason_limit is not None:
            reason = reason[:reason_limit]
        label = f"`{cell.backend} · {cell.case}`"
        for index, (seen, labels) in enumerate(errors, start=1):
            if seen == reason:
                labels.append(label)
                return f"💥 error ({index})"
        errors.append((reason, [label]))
        return f"💥 error ({len(errors)})"
    status = cell.status or PARTIAL
    icon = STATUS_ICON.get(status, "⚠️")
    # a reused result reads exactly like a fresh one: the matrix shows the state of the skill version, and the
    # progress line's "n reused" says how many cells this run did not have to execute
    link = f" · [report]({cell.url})" if cell.url else ""
    return f"{icon} {status}{link}"


def render_skill_heading(plan: SkillPlan, info: RunInfo, static_score: int | None = None) -> str:
    """`### \\`<skill>\\` · [dashboard overview](<group url>)`, plus ` · static <n>/100` when the dashboard knows the
    skill version's static-assessment score; the canary is labelled as such."""
    label = f"`{plan.name}` (canary)" if plan.canary else f"`{plan.name}`"
    heading = f"### {label} · [dashboard overview]({info.group_url(plan.name)})"
    if isinstance(static_score, int) and not isinstance(static_score, bool):
        heading += f" · static {static_score}/100"
    return heading


def render_skill_section(
    plan: SkillPlan,
    cells: list[Cell],
    backends: list[str],
    errors: ErrorGroups,
    reason_limit: int | None,
    info: RunInfo,
    static_scores: Mapping[str, int | None] | None = None,
) -> str:
    lines = [render_skill_heading(plan, info, (static_scores or {}).get(plan.name))]
    if not plan.cases:
        lines.append("_no eval cases_")
        return "\n".join(lines)
    by_key = {(cell.case, cell.backend): cell for cell in cells if cell.skill == plan.name}
    lines.append("| Case | " + " | ".join(backends) + " |")
    lines.append("|" + "---|" * (len(backends) + 1))
    for case in plan.cases:
        row = [f"`{_escape_table_cell(case)}`"]
        for backend in backends:
            cell = by_key.get((case, backend))
            row.append(render_cell_text(cell, errors, reason_limit) if cell else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_not_evaluated(plans: list[SkillPlan]) -> str | None:
    """Short note listing the skills that were touched (or listed) but have nothing to run, with the reason."""
    skipped = [plan for plan in plans if plan.skipped_reason]
    if not skipped:
        return None
    lines = ["**Not evaluated**"]
    lines.extend(f"- `{plan.name}`{' (canary)' if plan.canary else ''} — {plan.skipped_reason}" for plan in skipped)
    return "\n".join(lines)


def render_mode_note(options: TriggerOptions | None, cells: Iterable[Cell]) -> str:
    """`canary smoke-only|with-smoke` / `mode rerun-failed (5 selected, 39 reused)` / `mode force` / `cells=…`;
    empty for a default run."""
    if options is None:
        return ""
    parts = []
    if options.canary_request:
        parts.append(f"canary {options.canary_request}")
    # the plan's decision (Cell.is_job), not the state shown: an own cell that a refresh shows with another run's newer
    # result (CACHED) stays selected, so the counts hold for the whole run
    cells = list(cells)
    selected = sum(1 for cell in cells if cell.is_job)
    reused = sum(1 for cell in cells if cell.state is CellState.CACHED and not cell.is_job)
    if options.mode == "rerun-failed":
        parts.append(f"mode rerun-failed ({selected} selected, {reused} reused)")
    if options.force:
        parts.append("mode force")
    if options.cells:
        parts.append(f"cells=`{','.join(options.cells)}` ({selected} selected, {reused} reused)")
    return " · ".join(parts)


def render_header(info: RunInfo, options: TriggerOptions | None = None, cells: Iterable[Cell] = ()) -> str:
    header = f"## Skill eval — [{info.sha7}]({info.commit_url}) · run [#{info.run_id}]({info.run_url})"
    note = render_mode_note(options, cells)
    return f"{header} · {note}" if note else header


@dataclass(frozen=True)
class Tally:
    """Cell counts for the progress and final lines. `(not selected)` cells are outside every count; the four
    outcome buckets partition the finished cells: passed + failed + no_report + errors == done."""

    total: int
    done: int
    passed: int
    failed: int  # published FAIL or PARTIAL
    no_report: int  # published NO-REPORT
    errors: int  # published ERROR plus launcher errors (💥)
    reused: int  # CACHED cells the plan did not run (Cell.is_job False)


def tally(cells: Iterable[Cell]) -> Tally:
    active = [cell for cell in cells if cell.selected]
    finished = [cell for cell in active if cell.finished]
    published = [cell for cell in finished if cell.state is not CellState.ERROR]
    return Tally(
        total=len(active),
        done=len(finished),
        passed=sum(1 for cell in finished if cell.passed),
        failed=sum(1 for cell in published if cell.status in (FAIL, PARTIAL)),
        no_report=sum(1 for cell in published if cell.status == NO_REPORT),
        errors=sum(1 for cell in finished if cell.state is CellState.ERROR or cell.status == ERROR_STATUS),
        # cells this run did not execute (the plan's decision): an own cell shown with another run's newer result
        # (refresh_from_catalog) was executed by this run and is not "reused"
        reused=sum(1 for cell in active if cell.state is CellState.CACHED and not cell.is_job),
    )


def render_progress(cells: list[Cell], started_at: datetime, now: datetime) -> str:
    counts = tally(cells)
    return (
        f"{counts.done} / {counts.total} cells done · {counts.passed} passed · {counts.failed} failed"
        f" · {counts.no_report} no report · {counts.errors} errors · {counts.reused} reused"
        f" · started {format_utc(started_at)} · updated {format_utc(now)}"
    )


LEGEND = (
    "Legend: ⏳ queued · 🟢 running · ✅ PASS · ❌ FAIL · ⚠️ PARTIAL"
    " · ❔ NO-REPORT (the agent finished without writing a report) · ⛔ ERROR (model gateway / agent / node error,"
    " see the report) · 💥 error (launcher failed before publishing)"
    " · reused (progress line) = cells this run did not execute: another run's newest result of this skill version,"
    " shown like a fresh one"
    " · (not selected) = outside the trigger's `cells=` with no result of this skill version yet"
    f" · (not re-run: n report-less attempts) = {MAX_REPORTLESS_ATTEMPTS} consecutive NO-REPORT/ERROR results for this"
    " skill version, treated as a harness-side failure (`force` or `cells=` re-runs it)"
    " · not run = the runner lost docker (job stopped)"
)


def render_final_line(cells: list[Cell], interrupted: bool, aborted: str | None = None) -> str | None:
    counts = tally(cells)
    if aborted:
        not_run = sum(1 for cell in cells if (cell.reason or "").startswith(NOT_RUN_PREFIX))
        return f"🛑 aborted — {aborted} · {counts.passed}/{counts.total} passed · {not_run} not run"
    if interrupted:
        not_run = sum(1 for cell in cells if cell.selected and not cell.finished)
        return f"⚠️ interrupted — {counts.passed}/{counts.total} passed · {not_run} not run"
    if not cells:
        return "✅ done — nothing to evaluate"
    if not counts.total:
        return "✅ done — no cell selected"
    if counts.done < counts.total:
        return None
    icon = "✅" if counts.passed == counts.total else "❌"
    return (
        f"{icon} done — {counts.passed}/{counts.total} passed"
        f" ({counts.failed} failed, {counts.no_report} no report, {counts.errors} errors)"
    )


def _render_body(
    info: RunInfo,
    plans: list[SkillPlan],
    cells: list[Cell],
    backends: list[str],
    started_at: datetime,
    now: datetime,
    interrupted: bool,
    reason_limit: int | None,
    include_errors: bool,
    notes: tuple[str, ...] = (),
    aborted: str | None = None,
    seq: str | None = None,
    options: TriggerOptions | None = None,
    history: tuple[str, ...] = (),
    static_scores: Mapping[str, int | None] | None = None,
) -> str:
    errors: ErrorGroups = []
    parts = [info.marker, info.owner_line]
    if seq is not None:
        parts.append(f"{SEQ_PREFIX}{seq} -->")
        parts.extend(ledger_line(cell, info.run_id) for cell in cells if cell.state is CellState.DONE and cell.status)
    parts.extend(f"{HISTORY_PREFIX}{entry} -->" for entry in history)
    parts += [render_header(info, options, cells), "", render_progress(cells, started_at, now)]
    if options is not None and options.errors:
        parts.append(
            "⚠️ Trigger: " + "; ".join(options.errors) + " — ignored, the run uses the defaults (syntax: README)"
        )
    if notes:
        parts.append("🔧 " + " · ".join(notes))
    parts.append("")
    evaluated = [plan for plan in plans if not plan.skipped_reason]
    for plan in evaluated:
        parts.append(render_skill_section(plan, cells, backends, errors, reason_limit, info, static_scores))
        parts.append("")
    if not evaluated:
        parts.extend(["_Nothing to evaluate: this PR touches no skill with an `eval/eval.config`._", ""])
    note = render_not_evaluated(plans)
    if note:
        parts.extend([note, ""])
    if errors:
        if include_errors:
            parts.append("**Errors**")
            parts.extend(
                f"{index}. {', '.join(labels)} — {reason}" for index, (reason, labels) in enumerate(errors, start=1)
            )
        else:
            parts.append("_Error details omitted (comment size limit) — see the Actions log._")
        parts.append("")
    if history:
        parts.extend([render_history(info, history), ""])
    parts.append(LEGEND)
    final_line = render_final_line(cells, interrupted, aborted)
    if final_line:
        parts.extend(["", final_line])
    return "\n".join(parts) + "\n"


def render_comment(
    info: RunInfo,
    plans: list[SkillPlan],
    cells: list[Cell],
    backends: list[str],
    *,
    started_at: datetime,
    now: datetime,
    interrupted: bool = False,
    notes: Iterable[str] = (),
    aborted: str | None = None,
    seq: str | None = None,
    options: TriggerOptions | None = None,
    history: Iterable[str] = (),
    static_scores: Mapping[str, int | None] | None = None,
) -> str:
    """Full comment body, kept under MAX_BODY_CHARS (footnotes are truncated first). `static_scores` (skill -> the
    dashboard's static score of its version) adds ` · static <n>/100` to the headings of the skills it lists."""
    notes = tuple(notes)
    # (max reason length, include the error list): full detail, then clipped reasons, then no list at all.
    attempts = ((None, True), (80, True), (None, False))
    body = ""
    for reason_limit, include_errors in attempts:
        body = _render_body(
            info,
            plans,
            cells,
            backends,
            started_at,
            now,
            interrupted,
            reason_limit,
            include_errors,
            notes,
            aborted,
            seq,
            options,
            tuple(history),
            static_scores,
        )
        if len(body) <= MAX_BODY_CHARS:
            return body
    suffix = "\n\n_…truncated (comment size limit)_\n"
    return body[: MAX_BODY_CHARS - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# launcher execution
# ---------------------------------------------------------------------------


def build_launcher_command(
    config: Config, checkout: Path, output_dir: Path, cell: Cell, content_sha: str | None = None
) -> list[str]:
    command = [str(config.launcher)]
    if config.env_file is not None and config.env_file.exists():
        command += ["--env-file", str(config.env_file)]
    command += ["--image", config.image]
    command += ["--skill-src", str(checkout)]
    command += ["--output-dir", str(output_dir)]
    command += ["-e", f"COSMOS_EVAL_SRC_HOST={checkout}"]
    if content_sha:
        # explicit `-e`: reaches the container even with a launcher that predates the pass-through of this variable
        command += ["-e", f"COSMOS_EVAL_SKILL_CONTENT_SHA={content_sha}"]
    if cell.backend == "codex" and config.codex_model:
        command += ["-e", f"COSMOS_EVAL_CODEX_MODEL={config.codex_model}"]
    command += ["--github-env"]
    command += ["eval", f"/workspace/skill-source/{skill_dir(cell.skill, canary=cell.canary)}"]
    command += ["--backend", cell.backend, "--case", cell.case]
    return command


def synthetic_pull_request(info: RunInfo, environ: Mapping[str, str]) -> dict:
    """Stand-in PR object for --no-github runs."""
    actor = environ.get("GITHUB_ACTOR") or "unknown"
    return {
        "number": info.pr_number,
        "title": f"PR #{info.pr_number} (no-github run)",
        "html_url": f"{info.server_url}/{info.repo}/pull/{info.pr_number}",
        "user": {"login": actor, "avatar_url": None, "html_url": f"{info.server_url}/{actor}"},
        "head": {"sha": info.head_sha, "ref": info.head_ref},
        "base": {"ref": "main"},
        "draft": False,
        "state": "open",
    }


def write_event_file(workdir: Path, info: RunInfo, pull_request: dict) -> Path:
    event = {
        "action": "synchronize",
        "number": info.pr_number,
        "pull_request": pull_request,
        "repository": {"full_name": info.repo, "html_url": f"{info.server_url}/{info.repo}"},
    }
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"event-{info.sha7}.json"
    path.write_text(json.dumps(event, indent=2), encoding="utf-8")
    return path


def build_launcher_env(base: Mapping[str, str], info: RunInfo, pull_request: dict, event_path: Path) -> dict[str, str]:
    """Env for every launcher call: the publisher inside the container reads these for provenance.

    The job's GITHUB_TOKEN (pull-requests: write) is deliberately NOT exported: eval.config credentials such as
    GITHUB_TOKEN are forwarded into agent containers, and a prompt-injected agent must not be able to edit the
    PR. Agents get the read-only token from the runner's env file instead. The checkout that is bind-mounted
    into the task container carries no credentials either: every checkout step of the workflow sets
    `persist-credentials: false`, so the job token is not left in <checkout>/.git/config.
    """
    user = pull_request.get("user") if isinstance(pull_request.get("user"), dict) else {}
    head = pull_request.get("head") if isinstance(pull_request.get("head"), dict) else {}
    base_block = pull_request.get("base") if isinstance(pull_request.get("base"), dict) else {}
    env = dict(base)
    env.update(
        {
            "GITHUB_ACTIONS": "true",
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_SHA": info.head_sha,
            "GITHUB_HEAD_REF": info.head_ref or str(head.get("ref") or ""),
            "GITHUB_BASE_REF": str(base_block.get("ref") or ""),
            "GITHUB_REF": f"refs/pull/{info.pr_number}/merge",
            "GITHUB_REF_NAME": f"{info.pr_number}/merge",
            "GITHUB_REPOSITORY": info.repo,
            "GITHUB_RUN_ID": info.run_id,
            "GITHUB_RUN_ATTEMPT": info.run_attempt,
            "GITHUB_SERVER_URL": info.server_url,
            "GITHUB_WORKFLOW": "skill-eval",
            "GITHUB_JOB": "run",
            "GITHUB_ACTOR": str(user.get("login") or base.get("GITHUB_ACTOR") or "unknown"),
            "COSMOS_EVAL_GROUP_ID": info.group_override or info.default_group,  # per skill: launcher_env_for_cell
        }
    )
    env.pop("GH_TOKEN", None)
    return env


def launcher_env_for_cell(env: Mapping[str, str], info: RunInfo, cell: Cell) -> dict[str, str]:
    """The cell's launcher env: the skill's dashboard group and content sha on top of the common provenance env.

    The publisher files the run under COSMOS_EVAL_GROUP_ID and compares COSMOS_EVAL_SKILL_CONTENT_SHA with its own
    hash of the mounted skill directory (a mismatch becomes a note in the manifest, never an error).
    """
    out = dict(env)
    out["COSMOS_EVAL_GROUP_ID"] = info.group_for(cell.skill)
    content_sha = info.version_for(cell.skill).content_sha256
    if content_sha:
        out["COSMOS_EVAL_SKILL_CONTENT_SHA"] = content_sha
    else:
        out.pop("COSMOS_EVAL_SKILL_CONTENT_SHA", None)
    return out


def read_log_tail(path: Path, max_bytes: int = 64_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_log_head(path: Path, max_bytes: int = 16_000) -> str:
    try:
        with path.open("rb") as handle:
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


KEY_PROVENANCE_PREFIX = "[cosmos-skill-eval] NVIDIA_API_KEY:"


def key_provenance(log_text: str) -> str | None:
    """The launcher's one provenance line for the inference key (`[cosmos-skill-eval] NVIDIA_API_KEY: from host
    environment` / `from --env-file …` / `from -e option` / `NOT PROVIDED …`), or None when the log has none.

    The launcher writes it to the cell log on the runner before `docker run`; `_attempt_cell` repeats it in the
    Actions log, so a CI run shows whether the repository secret or the runner's env file fed the container
    (README, "Secrets"). Never the value: the launcher prints only the source.
    """
    for raw in log_text.splitlines():
        line = ANSI_RE.sub("", raw).strip()
        if line.startswith(KEY_PROVENANCE_PREFIX):
            return line
    return None


def parse_status(log_text: str) -> str | None:
    """Last `status: <one of KNOWN_STATUSES>` in the log (the eval CLI prints `[cosmos-eval] status: X`)."""
    matches = STATUS_LINE_RE.findall(log_text)
    return matches[-1] if matches else None


def last_meaningful_line(log_text: str, limit: int = 200) -> str:
    """Last non-empty launcher line, skipping the notes run_launcher itself appends to the log."""
    for raw in reversed(log_text.splitlines()):
        line = ANSI_RE.sub("", raw).replace("\r", "").strip()
        if line and not line.startswith("[run-matrix]"):
            return line[:limit]
    return "(empty log)"


def read_published(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def evaluate_outcome(cell: Cell, output_dir: Path, log_path: Path, returncode: int | None, dashboard: str) -> None:
    """Fill in cell.state/status/url/reason from the launcher log and published.json."""
    log_text = read_log_tail(log_path)
    last_line = last_meaningful_line(log_text)
    published = read_published(output_dir / cell.skill / "published.json")
    if returncode is None:
        cell.state, cell.reason = CellState.ERROR, f"timed out after {CELL_TIMEOUT_S // 3600} h — {last_line}"
        return
    if published is None:
        cell.state, cell.reason = CellState.ERROR, f"no published.json (launcher exit {returncode}) — {last_line}"
        return
    if published.get("status") != "ok":
        detail = published.get("error") or published.get("reason") or published.get("status") or "unknown"
        cell.state, cell.reason = CellState.ERROR, f"publish {detail} — {last_line}"
        return
    status = parse_status(log_text)
    if status is None:
        cell.state, cell.reason = CellState.ERROR, f"no `status:` line in the launcher log — {last_line}"
        return
    url = published.get("url") or (f"{dashboard}/runs/{published['run_id']}/" if published.get("run_id") else None)
    cell.state, cell.status, cell.url = CellState.DONE, status, url
    # the result's dashboard run id, to recognise it in the catalog later; its date is the catalog's to tell (today's
    # published.json carries no `created_at`, so the field stays None until the publisher provides one)
    cell.run_id = str(published["run_id"]) if published.get("run_id") else run_id_from_url(url)
    cell.created_at = published["created_at"] if isinstance(published.get("created_at"), str) else None


def clear_stale_outputs(output_dir: Path, skill: str, log_path: Path) -> None:
    """Move a previous attempt's published.json and log aside so the outcome reflects this attempt only.

    A cell is rerun only when the dashboard did not list it (launcher error, or the dashboard was
    unreachable) or when the same cell runs again for the same head on the same host (`force`,
    `rerun-failed`, a second dispatch); without this a launcher that dies early would be judged by the
    stale PASS line and link.

    The evaluator container writes `<output_dir>/<skill>/` as root (publish.py runs inside it), so the
    runner user cannot rename a file in that directory (EACCES, seen 2026-10-26 on a second smoke run of
    the same head). Renaming the directory itself only needs the user-owned `output_dir`, so on a
    PermissionError the whole previous `<skill>/` output is set aside instead (`<skill>.prev`, `.prev2`, …).
    """
    published = output_dir / skill / "published.json"
    if published.exists():
        try:
            published.replace(published.with_name(published.name + ".prev"))
        except PermissionError:
            stale = output_dir / skill
            n = 1
            while (target := stale.with_name(f"{skill}.prev{'' if n == 1 else n}")).exists():
                n += 1
            stale.replace(target)
    if log_path.exists():
        log_path.replace(log_path.with_name(log_path.name + ".prev"))


def _stop(process: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM first: `docker run` forwards it into the container, which exits and is removed (--rm).

    A plain kill() would only take down the docker CLI and leave the evaluator container running on
    the GPU while the next cell starts.
    """
    process.terminate()
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_launcher(command: list[str], env: Mapping[str, str], log_path: Path, timeout_s: float) -> int | None:
    """Run the launcher with stdout+stderr in log_path. Returns the exit code, or None on timeout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        try:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=dict(env))
        except OSError as exc:
            # The launcher vanished or lost its exec bit mid-run: report it as a cell error, keep going.
            handle.write(f"\n[run-matrix] cannot execute launcher {command[0]}: {exc}\n".encode())
            return 127
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            handle.write(f"\n[run-matrix] launcher timed out after {timeout_s:.0f}s — stopping it\n".encode())
            _stop(process, LAUNCHER_STOP_GRACE_S)
            return None
        except KeyboardInterrupt:
            # Job cancelled: give the launcher a moment to stop its container, then post the final comment.
            _stop(process, 3.0)
            raise


def fake_launch(cell: Cell, output_dir: Path, log_path: Path, dashboard: str, index: int, sleep: Callable) -> int:
    """--fake-launcher stub: plausible log + published.json, PASS/FAIL alternating by cell index."""
    sleep(0.1)
    status = PASS if index % 2 == 0 else FAIL
    run_id = f"fake-{cell.skill}-{cell.backend}-{cell.case}"
    url = f"{dashboard}/runs/{run_id}/"
    skill_out = output_dir / cell.skill
    skill_out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"[cosmos-eval] (fake launcher) skill={cell.skill} backend={cell.backend} case={cell.case}",
        f"[cosmos-eval] case {cell.case}: {status}",
        "",
        f"[cosmos-eval] status: {status}",
        f"[cosmos-eval] report: {skill_out / 'report.html'}",
        f"[cosmos-eval] published: {url}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    published = {"status": "ok", "url": url, "run_id": run_id, "group_id": f"fake-{cell.skill}"}
    (skill_out / "published.json").write_text(json.dumps(published, indent=2), encoding="utf-8")
    return 0 if status == PASS else 1


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


@dataclass
class Deps:
    """Injection points so the whole flow can be tested without network, docker or waiting."""

    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    github_transport: Transport = urllib_transport
    urlopen: Callable = urllib.request.urlopen
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    start_timer: Callable[[float, Callable[[], None]], object] = start_daemon_timer
    now: Callable[[], datetime] = utc_now
    comment_interval_s: float = COMMENT_MIN_INTERVAL_S
    cell_timeout_s: float = CELL_TIMEOUT_S


class RunnerUnavailable(Exception):
    """The runner cannot launch containers: the docker check failed even after the repair."""


@dataclass
class Preflight:
    """Runner health gate, run before every cell and before a retry.

    `check` (e.g. `docker info`) must succeed; when it fails, `repair` (e.g. `sudo -n chgrp docker
    /run/docker.sock`) is run once and the check repeated. Both are optional config settings.
    """

    check: list[str] | None = None
    repair: list[str] | None = None
    repairs: int = 0  # check failed, repair ran, check passed
    last_error: str = ""

    @staticmethod
    def _run(command: list[str], timeout_s: float) -> tuple[bool, str]:
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout_s, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{command[0]}: {exc}"
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        return result.returncode == 0, last_meaningful_line(output) or f"exit {result.returncode}"

    def ensure(self) -> bool:
        """True when the runner can launch containers; False (with `last_error` set) when it cannot."""
        if not self.check:
            return True
        ok, detail = self._run(self.check, DOCKER_CHECK_TIMEOUT_S)
        if ok:
            return True
        warn(f"docker check failed ({' '.join(self.check)}): {detail}")
        if self.repair:
            repaired, repair_detail = self._run(self.repair, DOCKER_REPAIR_TIMEOUT_S)
            log(f"docker repair ({' '.join(self.repair)}): {'ok' if repaired else repair_detail}")
            ok, detail = self._run(self.check, DOCKER_CHECK_TIMEOUT_S)
            if ok:
                self.repairs += 1
                log("docker check passes after the repair")
                return True
        self.last_error = detail
        return False


def docker_failure(text: str) -> bool:
    """Does a launcher log / reason show the runner's docker socket was unusable?"""
    return bool(DOCKER_FAILURE_RE.search(ANSI_RE.sub("", text)))


@dataclass
class Orchestrator:
    config: Config
    checkout: Path
    info: RunInfo
    plans: list[SkillPlan]
    cells: list[Cell]
    deps: Deps
    updater: PrintingCommentUpdater | GitHubCommentUpdater
    launcher_env: dict[str, str]
    fake_launcher: bool
    started_at: datetime = field(default_factory=utc_now)
    preflight: Preflight = field(default_factory=Preflight)
    retries: int = 0  # cells launched a second time after a docker failure
    aborted: str | None = None
    options: TriggerOptions = field(default_factory=TriggerOptions)

    @property
    def sha_workdir(self) -> Path:
        return self.config.workdir / self.info.sha7

    def notes(self) -> list[str]:
        notes = []
        if self.preflight.repairs:
            notes.append(f"docker repaired {self.preflight.repairs}× on the runner")
        if self.retries:
            notes.append(f"{self.retries} cell{'s' if self.retries != 1 else ''} retried after a docker error")
        return notes

    def publish(self, *, final: bool = False, interrupted: bool = False) -> None:
        body = render_comment(
            self.info,
            self.plans,
            self.cells,
            self.config.backends,
            started_at=self.started_at,
            now=self.deps.now(),
            interrupted=interrupted,
            notes=self.notes(),
            aborted=self.aborted,
            options=self.options,
        )
        self.updater.publish(body, final=final)

    def run(self) -> int:
        self.publish()
        try:
            for index, cell in enumerate(self.cells):
                if cell.state in (CellState.CACHED, CellState.SKIPPED, CellState.EXHAUSTED):
                    continue
                try:
                    if not self.preflight.ensure():
                        raise RunnerUnavailable(self.preflight.last_error)
                    self.run_cell(cell, index)
                except RunnerUnavailable as exc:
                    self.abort_remaining(f"runner cannot reach docker — {exc}")
                    break
                except Exception as exc:  # one broken cell (disk full, unreadable log, ...) must not end the matrix
                    cell.state, cell.reason = CellState.ERROR, f"orchestrator error: {type(exc).__name__}: {exc}"[:200]
                    warn(f"{cell.label}: {cell.reason}")
                    self.publish()
        except KeyboardInterrupt:
            for cell in self.cells:
                if cell.state is CellState.RUNNING:
                    cell.state, cell.reason = CellState.ERROR, "interrupted (job cancelled)"
            warn("interrupted — posting final comment")
            self.publish(final=True, interrupted=True)
            return 130
        self.publish(final=True)
        passed = sum(1 for cell in self.cells if cell.passed)
        errors = sum(1 for cell in self.cells if cell.state is CellState.ERROR)
        extra = f", {self.preflight.repairs} docker repair(s), {self.retries} retried" if self.notes() else ""
        log(f"done: {passed}/{tally(self.cells).total} passed, {errors} launcher error(s){extra}")
        log(f"dashboard group(s): {self.info.groups_summary()}")
        if isinstance(self.updater, GitHubCommentUpdater):
            log(f"comment id: {self.updater.comment_id} ({self.updater.failures} failed update(s))")
        if self.aborted:
            log(f"ABORTED: {self.aborted}")
            return 1
        return 0

    def abort_remaining(self, reason: str) -> None:
        """Stop launching: cells not finished yet are marked `not run` and the job will exit 1."""
        self.aborted = reason
        for cell in self.cells:
            if cell.selected and not cell.finished:
                cell.state, cell.reason = CellState.ERROR, f"{NOT_RUN_PREFIX}{reason}"
        warn(f"aborting the matrix: {reason}")

    def run_cell(self, cell: Cell, index: int) -> None:
        self.retries += execute_cell(
            cell,
            index,
            config=self.config,
            checkout=self.checkout,
            info=self.info,
            deps=self.deps,
            launcher_env=self.launcher_env,
            fake_launcher=self.fake_launcher,
            preflight=self.preflight,
            on_update=self.publish,
        )


def execute_cell(
    cell: Cell,
    index: int,
    *,
    config: Config,
    checkout: Path,
    info: RunInfo,
    deps: Deps,
    launcher_env: Mapping[str, str],
    fake_launcher: bool,
    preflight: Preflight,
    on_update: Callable[[], None],
) -> int:
    """Run one cell: one attempt, plus a single retry when the launcher died on a docker-socket error.

    Returns the number of retries (0 or 1). Raises RunnerUnavailable when the docker check fails before the retry.
    """
    workdir = config.workdir / info.sha7
    _attempt_cell(cell, index, workdir, config, checkout, info, deps, launcher_env, fake_launcher, on_update)
    if cell.state is not CellState.ERROR:
        return 0
    log_path = workdir / f"{cell.name}.log"
    if not docker_failure(read_log_tail(log_path) + "\n" + (cell.reason or "")):
        return 0
    warn(f"{cell.label}: docker failure on the first attempt — checking the runner and retrying once")
    if not preflight.ensure():
        raise RunnerUnavailable(preflight.last_error)
    _attempt_cell(cell, index, workdir, config, checkout, info, deps, launcher_env, fake_launcher, on_update)
    if cell.state is CellState.ERROR:
        cell.reason = f"{cell.reason} (after 1 retry)"
    return 1


def _attempt_cell(
    cell: Cell,
    index: int,
    workdir: Path,
    config: Config,
    checkout: Path,
    info: RunInfo,
    deps: Deps,
    launcher_env: Mapping[str, str],
    fake_launcher: bool,
    on_update: Callable[[], None],
) -> None:
    output_dir = workdir / cell.name
    log_path = workdir / f"{cell.name}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs(output_dir, cell.skill, log_path)
    cell.state, cell.status, cell.url, cell.reason = CellState.RUNNING, None, None, None
    cell.run_id = cell.created_at = None
    log(f"start {cell.label} -> {output_dir}")
    on_update()
    started = deps.clock()
    if fake_launcher:
        returncode: int | None = fake_launch(cell, output_dir, log_path, info.dashboard, index, deps.sleep)
    else:
        command = build_launcher_command(
            config, checkout, output_dir, cell, info.version_for(cell.skill).content_sha256 or None
        )
        returncode = run_launcher(
            command, launcher_env_for_cell(launcher_env, info, cell), log_path, deps.cell_timeout_s
        )
    cell.duration_s = deps.clock() - started
    evaluate_outcome(cell, output_dir, log_path, returncode, info.dashboard)
    provenance = key_provenance(read_log_head(log_path))
    if provenance:
        log(f"{cell.label}: {provenance}")
    outcome = cell.status if cell.state is CellState.DONE else f"error: {cell.reason}"
    log(f"end {cell.label}: {outcome} in {cell.duration_s:.0f}s {cell.url or ''}".rstrip())
    on_update()


# ---------------------------------------------------------------------------
# distributed mode: `plan` -> one GitHub Actions job per cell (`cell`) -> `report`
#
# The state of the matrix is never held by one process: every render rebuilds it from durable sources —
# the plan (cells, cache hits), the dashboard (published results for this sha) and the workflow's own job
# list (queued / running / finished). Any cell job can therefore update the single PR comment, and a
# runner that dies takes only its own cell with it.
# ---------------------------------------------------------------------------


def plan_to_dict(
    info: RunInfo,
    plans: list[SkillPlan],
    cells: list[Cell],
    backends: list[str],
    started_at: datetime,
    options: TriggerOptions,
) -> dict:
    """The plan JSON (schema v1, additive): `skills` carries each skill's version and group so the depth-1
    `cell`/`report` checkouts never need git, `trigger` what the run was asked to do."""
    return {
        "schema": PLAN_SCHEMA,
        "scope": options.scope,
        "trigger": options.to_dict(),
        "started_at": started_at.isoformat(),
        "info": {
            "repo": info.repo,
            "server_url": info.server_url,
            "pr_number": info.pr_number,
            "head_sha": info.head_sha,
            "head_ref": info.head_ref,
            "run_id": info.run_id,
            "run_attempt": info.run_attempt,
            "dashboard": info.dashboard,
            "base_sha": info.base_sha,
            "group_override": info.group_override,
        },
        "skills": {name: version.to_dict() for name, version in info.versions.items()},
        "backends": list(backends),
        "plans": [
            {"name": plan.name, "cases": list(plan.cases), "skipped_reason": plan.skipped_reason, "canary": plan.canary}
            for plan in plans
        ],
        "cells": [
            {
                "skill": cell.skill,
                "case": cell.case,
                "backend": cell.backend,
                "canary": cell.canary,
                "state": cell.state.value,
                "status": cell.status,
                "url": cell.url,
                "reason": cell.reason,
                "dashboard_run_id": cell.run_id,  # of the result shown — not the Actions run
                "created_at": cell.created_at,
                "job": cell.job,  # the plan's decision: True = this run executes the cell
            }
            for cell in cells
        ],
    }


def plan_from_dict(
    data: dict,
) -> tuple[RunInfo, list[SkillPlan], list[Cell], list[str], datetime, TriggerOptions]:
    if not isinstance(data, dict) or data.get("schema") != PLAN_SCHEMA:
        raise OrchestrationError(f"plan: expected schema {PLAN_SCHEMA}")
    raw = data.get("info") or {}
    raw_skills = data.get("skills") if isinstance(data.get("skills"), dict) else {}
    try:
        info = RunInfo(
            repo=str(raw["repo"]),
            server_url=str(raw["server_url"]),
            pr_number=int(raw["pr_number"]),
            head_sha=str(raw["head_sha"]),
            head_ref=str(raw.get("head_ref") or ""),
            run_id=str(raw["run_id"]),
            run_attempt=str(raw.get("run_attempt") or "1"),
            dashboard=str(raw["dashboard"]).rstrip("/"),
            base_sha=str(raw.get("base_sha") or ""),
            versions={
                str(name): SkillVersion.from_dict(block)
                for name, block in raw_skills.items()
                if isinstance(block, dict)
            },
            group_override=str(raw["group_override"]) if raw.get("group_override") else None,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise OrchestrationError(f"plan: bad `info`/`skills` block ({exc})") from exc
    raw_trigger = data.get("trigger")
    if isinstance(raw_trigger, dict):
        options = TriggerOptions.from_dict(raw_trigger)
    else:  # a plan written before trigger arguments existed
        options = TriggerOptions(scope=str(data.get("scope") or "full"))
    try:
        plans = [
            SkillPlan(
                str(p["name"]), [str(c) for c in p.get("cases") or []], p.get("skipped_reason"), bool(p.get("canary"))
            )
            for p in data.get("plans") or []
        ]
        cells: list[Cell] = []
        for raw_cell in data.get("cells") or []:
            cell = Cell(
                str(raw_cell["skill"]),
                str(raw_cell["case"]),
                str(raw_cell["backend"]),
                canary=bool(raw_cell.get("canary")),
            )
            if raw_cell.get("state") in (CellState.CACHED.value, CellState.SKIPPED.value, CellState.EXHAUSTED.value):
                cell.state = CellState(raw_cell["state"])
                cell.status, cell.url = raw_cell.get("status"), raw_cell.get("url")
                cell.reason = raw_cell.get("reason") or None
                cell.run_id = (
                    str(raw_cell["dashboard_run_id"]) if raw_cell.get("dashboard_run_id") else run_id_from_url(cell.url)
                )
                cell.created_at = raw_cell["created_at"] if isinstance(raw_cell.get("created_at"), str) else None
            # the plan's decision, for every cell (a queued cell's True matters once a refresh shows another run's result
            # in it); None = a plan without the field: `is_job` then reads the state
            cell.job = raw_cell["job"] if isinstance(raw_cell.get("job"), bool) else None
            cells.append(cell)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise OrchestrationError(f"plan: bad `plans`/`cells` entry ({type(exc).__name__}: {exc})") from exc
    started_at = _created_at_key(data.get("started_at"))
    if started_at == datetime.min.replace(tzinfo=timezone.utc):
        raise OrchestrationError(f"plan: bad `started_at` {data.get('started_at')!r}")
    return info, plans, cells, [str(b) for b in data.get("backends") or []], started_at, options


def load_plan(spec: str) -> dict:
    """`--plan` is either a path to the plan file or the JSON text itself (as passed through job outputs)."""
    text = spec.strip()
    if not text.startswith("{"):
        path = Path(spec).expanduser()
        if not path.is_file():
            raise OrchestrationError(f"plan not found: {path}")
        text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except ValueError as exc:
        raise OrchestrationError(f"plan is not valid JSON: {exc}") from exc


def runs_on(backend: str, config: Config | None = None) -> list[str]:
    """`runs-on` labels of the job for a cell of this backend: `runner_labels`, plus `serial_runner_label` for a backend
    in `serial_backends` — exactly one runner carries that label, so those cells queue there and run one at a time
    (cursor: most likely a per-key concurrency cap on Cursor's side, see DEFAULT_SERIAL_BACKENDS). None = the defaults.
    """
    labels = list(config.runner_labels if config else DEFAULT_RUNNER_LABELS)
    serial = config.serial_backends if config else DEFAULT_SERIAL_BACKENDS
    serial_label = config.serial_runner_label if config else DEFAULT_SERIAL_RUNNER_LABEL
    if backend in serial and serial_label and serial_label not in labels:
        labels.append(serial_label)
    return labels


def matrix_entries(cells: list[Cell], *, canary: bool, config: Config | None = None) -> list[dict]:
    """Matrix `include` entries for the cells that still have to run (reused, not-selected and exhausted cells are not
    jobs). `runs_on` is the job's runner-label list as a JSON string: the workflow reads it with
    `runs-on: ${{ fromJSON(matrix.runs_on) }}`."""
    return [
        {
            "skill": cell.skill,
            "backend": cell.backend,
            "case": cell.case,
            "runs_on": json.dumps(runs_on(cell.backend, config)),
        }
        for cell in cells
        if bool(cell.canary) == canary and cell.state not in (CellState.CACHED, CellState.SKIPPED, CellState.EXHAUSTED)
    ]


def append_outputs(path: Path, outputs: Mapping[str, str]) -> None:
    """Append `key=value` pairs to a GITHUB_OUTPUT file (heredoc form, safe for JSON and newlines)."""
    with path.open("a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            delimiter = f"EOF_{key}_{abs(hash(value)) % 10_000_000}"
            handle.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


def cell_job_name(cell: Cell) -> str:
    return f"{JOB_NAME_PREFIX}{cell.label}"


def fetch_run_jobs(api: GitHubApi, info: RunInfo) -> dict[str, dict] | None:
    """Job name -> {status, conclusion, html_url} for this run; None (with a warning) when the API fails."""
    try:
        jobs = api.list_run_jobs(info.run_id)
    except (GitHubApiError, ValueError) as exc:
        warn(f"cannot list the jobs of run {info.run_id} ({exc}); cells without a published result show as queued")
        return None
    return {
        str(job.get("name")): {
            "status": job.get("status"),
            "conclusion": job.get("conclusion"),
            "html_url": job.get("html_url"),
        }
        for job in jobs
        if job.get("name")
    }


def reconcile_cells(
    cells: list[Cell],
    info: RunInfo,
    urlopen: Callable,
    jobs: Mapping[str, dict],
    *,
    final: bool,
    started_at: datetime | None,
    local: Mapping[str, Cell] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    ledger: Mapping[str, tuple[str, str]] | None = None,
    catalogs: dict[str, list[dict]] | None = None,
) -> None:
    """Rebuild every runnable cell's state: own outcome, then the dashboard, then the comment's ledger, then its job;
    finally the newest-result rule (refresh_from_catalog) for every cell that is not about to run.

    For a cell still to be resolved the dashboard lookup takes what THIS run published (find_run_result), never an
    older result of the version — the cache decided about those at plan time. The refresh then needs the catalog of
    every skill that has cells: one GET per skill per refresh for a skill the lookups above did not touch (every cell
    of it reused / not selected / in a result file or the ledger). An (empty) `catalogs` dict passed in is filled
    with the catalogs fetched here, so the caller can read the skills' static scores from them without a second GET.
    """
    if catalogs is None:
        catalogs = {}
    for cell in cells:
        if cell.state in (CellState.CACHED, CellState.SKIPPED, CellState.EXHAUSTED):
            continue
        own = (local or {}).get(cell.name)
        if own is not None and own.state is not CellState.QUEUED:
            if own is not cell:
                cell.state, cell.status, cell.url, cell.reason = own.state, own.status, own.url, own.reason
                cell.duration_s, cell.run_id, cell.created_at = own.duration_s, own.run_id, own.created_at
            continue
        if cell.skill not in catalogs:
            catalogs[cell.skill] = fetch_catalog(info.dashboard, cell.skill, urlopen, sleep=sleep)
        hit = find_run_result(catalogs[cell.skill], info, cell.case, cell.backend, started_at=started_at)
        if hit is not None:
            cell.adopt(hit, CellState.DONE)
            continue
        recorded = (ledger or {}).get(cell.label)
        if recorded is not None:  # a cell job already posted this result; the dashboard may simply not list it yet
            cell.state, cell.status, cell.url, cell.reason = CellState.DONE, recorded[0], recorded[1] or None, None
            cell.run_id, cell.created_at = run_id_from_url(cell.url), None
            continue
        job = jobs.get(cell_job_name(cell))
        cell.status, cell.url, cell.run_id, cell.created_at = None, None, None, None
        status = (job or {}).get("status")
        if job is None or status in ("queued", "waiting", "requested", "pending"):
            cell.state, cell.reason = CellState.QUEUED, None
        elif status == "in_progress":
            cell.state, cell.reason = CellState.RUNNING, None
        else:
            link = f" — see [job log]({job['html_url']})" if job.get("html_url") else ""
            conclusion = job.get("conclusion")
            if conclusion == "success":
                if final:
                    cell.state, cell.reason = CellState.ERROR, f"finished without a published result{link}"
                else:
                    cell.state, cell.reason = CellState.RUNNING, PUBLISHING
            elif conclusion == "cancelled":
                cell.state, cell.reason = CellState.ERROR, f"cancelled{link}"
            elif conclusion == "skipped":
                cell.state, cell.reason = CellState.ERROR, f"{NOT_RUN_PREFIX}job skipped{link}"
            else:
                cell.state, cell.reason = CellState.ERROR, f"launcher error ({conclusion or 'failed'}){link}"
    # the refresh below compares every finished cell with its skill's catalog: fetch the ones no lookup above needed
    for skill in dict.fromkeys(cell.skill for cell in cells):
        if skill not in catalogs:
            catalogs[skill] = fetch_catalog(info.dashboard, skill, urlopen, sleep=sleep)
    refresh_from_catalog(cells, info, catalogs)


def comment_seq(cells: list[Cell]) -> int:
    return sum(1 for cell in cells if cell.finished)


def ledger_line(cell: Cell, run_id: str | None = None) -> str:
    """Hidden comment line carrying one finished cell; `run_id` (the Actions run) scopes it to this run."""
    suffix = f"|{run_id}" if run_id else ""
    return f"{LEDGER_PREFIX}{cell.label}|{cell.status}|{cell.url or ''}{suffix} -->"


def parse_ledger(body: object, run_id: str | None = None) -> dict[str, tuple[str, str]]:
    """Cell label -> (status, url) for every finished cell the posted comment already carries.

    With `run_id`, only lines written by that Actions run count: the comment is per sha, so a re-trigger
    of the same commit (force, rerun-failed, a non-reusable result) finds the previous run's final comment
    and must not resurrect its cells as finished. Lines without a run id (pre-2026-10-22 comments) are
    then ignored too — they belong to an earlier run by definition.
    """
    ledger: dict[str, tuple[str, str]] = {}
    for match in LEDGER_RE.finditer(str(body or "")):
        if run_id is not None and (match.group(4) or "") != run_id:
            continue
        ledger[match.group(1)] = (match.group(2), match.group(3))
    return ledger


def parse_owner(body: object) -> tuple[str, str, str] | None:
    """(run_id, sha, attempt) of the Actions run whose view the posted comment shows; None for a legacy body."""
    match = re.search(re.escape(OWNER_PREFIX) + r"([^|\s]+)\|([0-9a-f]+)\|([^\s|]*) -->", str(body or ""))
    return (match.group(1), match.group(2), match.group(3) or "1") if match else None


def parse_history(body: object) -> list[str]:
    """`<run_id>|<sha>|<summary>` entries of earlier runs, newest first, as the posted comment carries them."""
    return [match.group(1) for match in re.finditer(re.escape(HISTORY_PREFIX) + r"([^\n]*?) -->", str(body or ""))]


def run_id_newer(candidate: str, than: str) -> bool:
    """Actions run ids grow monotonically; non-numeric ids (`local`) never outrank a numeric one."""
    try:
        return int(candidate) > int(than)
    except ValueError:
        return False


def history_entry(body: object) -> str | None:
    """One history line for the run that owns `body`: its final line when it finished, else where it stood."""
    owner = parse_owner(body)
    if owner is None:
        return None
    summary = None
    for line in str(body or "").splitlines():
        if re.match(r"^[✅❌🛑⚠️]+ (done|aborted|interrupted) ", line):
            summary = line.strip()
    if summary is None:
        for line in str(body or "").splitlines():
            if line.startswith("<!--"):  # hidden ledger/history lines can carry an earlier "cells done" summary
                continue
            if " cells done · " in line:
                summary = "unfinished — " + line.split(" · started ")[0].strip()
                break
    summary = (summary or "no result recorded").replace("|", "/").replace("\n", " ")[:300]
    return f"{owner[0]}|{owner[1]}|{summary}"


def render_history(info: RunInfo, history: Iterable[str]) -> str:
    lines = ["**Previous runs on this PR**"]
    for entry in history:
        run_id, _sep, rest = entry.partition("|")
        sha, _sep, summary = rest.partition("|")
        sha7 = sha[:7] or "?"
        commit = f"[{sha7}]({info.server_url}/{info.repo}/commit/{sha})" if sha else sha7
        lines.append(f"- {commit} · run [#{run_id}]({info.server_url}/{info.repo}/actions/runs/{run_id}) · {summary}")
    return "\n".join(lines)


def read_comment(api: GitHubApi, info: RunInfo) -> tuple[int | None, str]:
    """(id, body) of this head commit's marker comment, or (None, "") when there is none yet (or GitHub is unreachable)."""
    try:
        existing = GitHubCommentUpdater(api, info.pr_number, info.marker).find_existing()
        if existing is None:
            return None, ""
        return existing, str(api.get_issue_comment(existing).get("body") or "")
    except (GitHubApiError, ValueError) as exc:
        warn(f"cannot read the existing comment ({exc})")
        return None, ""


def read_previous_comment(api: GitHubApi, info: RunInfo) -> tuple[int | None, str]:
    """(id, body) of the newest skill-eval comment this PR has for ANOTHER head commit (or the legacy one-per-PR
    comment), or (None, "") — the source of the "Previous runs" history when this commit's comment is created."""
    try:
        newest: tuple[int, str] | None = None
        for comment in api.list_issue_comments(info.pr_number):
            body = str(comment.get("body") or "")
            if comment.get("id") is None or body.startswith(info.marker) or not info.is_pr_marker(body):
                continue
            if newest is None or int(comment["id"]) > newest[0]:
                newest = (int(comment["id"]), body)
        return newest if newest is not None else (None, "")
    except (GitHubApiError, ValueError) as exc:
        warn(f"cannot read the PR's earlier comments ({exc})")
        return None, ""


def write_cell_result(results_dir: Path, info: RunInfo, cell: Cell) -> Path:
    """Persist this cell's outcome as `<results_dir>/<cell name>.json`; the workflow uploads it as an artifact."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{cell.name}.json"
    payload = {
        "schema": RESULT_SCHEMA,
        "label": cell.label,
        "skill": cell.skill,
        "backend": cell.backend,
        "case": cell.case,
        "state": cell.state.value,
        "status": cell.status,
        "url": cell.url,
        "reason": cell.reason,
        "duration_s": cell.duration_s,
        "dashboard_run_id": cell.run_id,  # the result's run on the dashboard (`run_id` below is the Actions run)
        "created_at": cell.created_at,
        "head_sha": info.head_sha,
        "run_id": info.run_id,
        "run_attempt": info.run_attempt,
        "written_at": utc_now().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_cell_results(results_dir: Path | None, cells: list[Cell], info: RunInfo) -> dict[str, Cell]:
    """Finished outcomes the cell jobs wrote (downloaded artifacts), keyed by cell name; unknown or foreign files are skipped."""
    found: dict[str, Cell] = {}
    if results_dir is None or not results_dir.is_dir():
        return found
    by_label = {cell.label: cell for cell in cells}
    for path in sorted(results_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warn(f"cell result {path.name} unreadable ({exc}); ignored")
            continue
        if not isinstance(data, dict) or data.get("schema") != RESULT_SCHEMA:
            continue
        if data.get("head_sha") != info.head_sha:
            warn(f"cell result {path.name} is for sha {str(data.get('head_sha'))[:7]}, not {info.sha7}; ignored")
            continue
        cell = by_label.get(str(data.get("label")))
        if cell is None or data.get("state") not in (CellState.DONE.value, CellState.ERROR.value):
            continue
        out = Cell(cell.skill, cell.case, cell.backend, canary=cell.canary)
        out.state = CellState(data["state"])
        out.status, out.url, out.reason = data.get("status"), data.get("url"), data.get("reason")
        out.duration_s = data.get("duration_s") if isinstance(data.get("duration_s"), (int, float)) else None
        out.run_id = str(data["dashboard_run_id"]) if data.get("dashboard_run_id") else run_id_from_url(out.url)
        out.created_at = data["created_at"] if isinstance(data.get("created_at"), str) else None
        found[cell.name] = out
    return found


def merge_ledger(cells: list[Cell], ledger: Mapping[str, tuple[str, str]]) -> int:
    """Union: cells the posted comment already shows as finished stay finished. Never downgrades. Returns the merges."""
    merged = 0
    for cell in cells:
        recorded = ledger.get(cell.label)
        if recorded is not None and cell.state in (CellState.QUEUED, CellState.RUNNING):
            cell.state, cell.status, cell.url, cell.reason = CellState.DONE, recorded[0], recorded[1] or None, None
            cell.run_id, cell.created_at = run_id_from_url(cell.url), None
            merged += 1
    return merged


def seq_label(info: RunInfo, cells: list[Cell]) -> str:
    """`<attempt>.<finished cells>`: a re-run attempt always outranks the previous attempt's final comment."""
    return f"{info.run_attempt or '1'}.{comment_seq(cells)}"


def parse_seq(body: object) -> tuple[int, int] | None:
    """(attempt, finished cells) from the seq line; the pre-attempt format `seq:<n>` reads as attempt 1."""
    match = re.search(re.escape(SEQ_PREFIX) + r"(?:(\d+)\.)?(\d+) -->", str(body or ""))
    if not match:
        return None
    return int(match.group(1) or 1), int(match.group(2))


def publish_distributed(
    api: GitHubApi,
    info: RunInfo,
    plans: list[SkillPlan],
    cells: list[Cell],
    backends: list[str],
    started_at: datetime,
    now: datetime,
    *,
    final: bool,
    options: TriggerOptions | None = None,
    static_scores: Mapping[str, int | None] | None = None,
) -> bool:
    """Render the whole matrix and update this head commit's marker comment (creating it, with the earlier
    commits' runs carried over into "Previous runs", when the commit has none yet).

    Right before the PATCH the posted comment is read again: finished cells recorded in its ledger are merged
    into this view (union, never a downgrade) and a body that would show fewer finished cells than the live
    comment is dropped (the seq line). GitHub has no conditional update for comments, so two jobs can still
    interleave between that read and the PATCH; the loss is then cosmetic (one live update) and is repaired by
    the next update or by the report job, which aggregates the per-cell result artifacts instead of trusting
    the comment. The report job always writes. Returns False when the update was skipped or failed (never raises).
    """
    comment_id, live_body = read_comment(api, info)
    owner = parse_owner(live_body)
    history = parse_history(live_body)
    if comment_id is None:
        # First comment for this head commit: the earlier commits' comments stay as they are; their runs are
        # carried over into this comment's "Previous runs" list (the newest earlier comment's history plus itself).
        previous_id, previous_body = read_previous_comment(api, info)
        if previous_id is not None:
            history = parse_history(previous_body)
            entry = history_entry(previous_body)
            if entry:
                history = [h for h in history if not h.startswith(entry.split("|", 1)[0] + "|")]
                history.insert(0, entry)
            history = history[:MAX_HISTORY]
            log(f"new comment for {info.sha7}; {len(history)} earlier run(s) carried over from comment {previous_id}")
    if comment_id is not None and owner is not None and owner[0] != info.run_id:
        # The commit's one comment shows one run at a time. A newer run has taken it over: this (older) run's cells
        # keep publishing to the dashboard, but they do not write here any more — not even their final report.
        if run_id_newer(owner[0], info.run_id):
            log(f"comment {comment_id} belongs to the newer run {owner[0]}; update skipped")
            return False
        # This run takes the comment over from an older run: that run's outcome moves into the history list.
        entry = history_entry(live_body)
        history = [h for h in history if not h.startswith(f"{owner[0]}|")]
        if entry:
            history.insert(0, entry)
        history = history[:MAX_HISTORY]
        log(f"taking over comment {comment_id} from run {owner[0]} ({len(history)} earlier run(s) kept)")
    else:
        merged = merge_ledger(cells, parse_ledger(live_body, info.run_id))
        if merged:
            log(f"merged {merged} finished cell(s) from the posted comment's ledger")
    seq = seq_label(info, cells)
    mine = parse_seq(f"{SEQ_PREFIX}{seq} -->")
    if comment_id is not None and not final and (owner is None or owner[0] == info.run_id):
        current = parse_seq(live_body)
        if current is not None and mine is not None and current > mine:
            log(f"comment {comment_id} already shows seq {current[0]}.{current[1]} (> {seq}); update skipped")
            return False
    body = render_comment(
        info,
        plans,
        cells,
        backends,
        started_at=started_at,
        now=now,
        seq=seq,
        options=options,
        history=history,
        static_scores=static_scores,
    )
    updater = GitHubCommentUpdater(api, info.pr_number, info.marker)
    updater.comment_id = comment_id
    updater.publish(body, final=True)
    return updater.failures == 0


def _open_github(args: argparse.Namespace, deps: Deps, info: RunInfo) -> GitHubApi | None:
    if args.no_github:
        return None
    token = (deps.environ.get("GH_TOKEN") or "").strip()
    return GitHubApi(info.server_url, info.repo, token, transport=deps.github_transport, sleep=deps.sleep)


def trigger_options_from_args(args: argparse.Namespace) -> TriggerOptions:
    """`--trigger-args` text, with the CLI shortcuts on top: `--scope smoke`, `--group <id>`."""
    options = parse_trigger_args(getattr(args, "trigger_args", None))
    if getattr(args, "scope", "full") == "smoke":
        options.scope = "smoke"
    group = getattr(args, "group", None)
    if group:
        if not GROUP_ID_RE.fullmatch(group):
            raise OrchestrationError(f"--group {group!r}: a group id is letters, digits, `.`, `_` or `-` (1–100 chars)")
        options.group = group
    for error in options.errors:
        warn(f"trigger: {error}")
    return options


def prepare_matrix(
    config: Config, info: RunInfo, checkout: Path, api: GitHubApi | None, deps: Deps, options: TriggerOptions
) -> tuple[RunInfo, list[SkillPlan], list[Cell], int]:
    """Skills -> plans -> cells, skill versions and groups, cache and `cells=` selection: shared by `run` and `plan`.

    Returns (info with the versions filled in, plans, cells, number of reused cells).
    """
    if options.scope == "smoke":
        log(f"scope smoke (smoke-only): only the canary {CANARY_SKILL} runs, skill detection skipped")
        skills, run_canary = [], True
    else:
        skills, run_canary = select_plan(config, info, checkout, api, with_smoke=options.with_smoke)
    plans = load_skill_plans(checkout, skills, smoke=run_canary)
    catalogs = {
        plan.name: fetch_catalog(info.dashboard, plan.name, deps.urlopen, sleep=deps.sleep)
        for plan in plans
        if not plan.skipped_reason
    }
    info = replace(info, versions=resolve_skill_versions(checkout, info, plans, catalogs))
    cells = build_cells(plans, config.backends)
    hits = apply_cache(cells, info, deps.urlopen, sleep=deps.sleep, catalogs=catalogs, options=options)
    return info, plans, cells, hits


def mode_plan(args: argparse.Namespace, deps: Deps) -> int:
    """Detect skills, build the cells, apply the cache, post the initial comment and emit the job matrices."""
    config = load_config(args.config, deps.environ)
    checkout = args.checkout.expanduser().resolve()
    if not checkout.is_dir():
        raise OrchestrationError(f"checkout not found: {checkout}")
    options = trigger_options_from_args(args)
    info = replace(
        read_run_info(deps.environ, config, checkout, no_github=args.no_github), group_override=options.group
    )
    api = _open_github(args, deps, info)
    log(f"plan: PR #{info.pr_number} · sha {info.sha7} · run {info.run_id} · {describe_trigger(options)}")
    info, plans, cells, hits = prepare_matrix(config, info, checkout, api, deps, options)
    started_at = deps.now()
    canary_jobs = matrix_entries(cells, canary=True, config=config)
    skill_jobs = matrix_entries(cells, canary=False, config=config)
    not_selected = sum(1 for cell in cells if not cell.selected)
    exhausted = sum(1 for cell in cells if cell.state is CellState.EXHAUSTED)
    log(
        f"{len(cells)} cell(s): {hits} reused, {not_selected} not selected, {exhausted} not re-run,"
        f" {len(canary_jobs)} canary job(s), {len(skill_jobs)} skill job(s)"
    )
    serial = sum(1 for job in canary_jobs + skill_jobs if job["backend"] in config.serial_backends)
    if serial:
        log(
            f"{serial} {'/'.join(config.serial_backends)} job(s) pinned to the runner labelled"
            f" `{config.serial_runner_label}`: they run one at a time"
        )
    for name, jobs in (("canary", canary_jobs), ("skill", skill_jobs)):
        if len(jobs) > MATRIX_MAX_JOBS:
            raise OrchestrationError(
                f"{len(jobs)} {name} cells exceed GitHub's limit of {MATRIX_MAX_JOBS} jobs per matrix; "
                "split the PR or set `skills:` in .github/skill-eval.yml"
            )
    plan = plan_to_dict(info, plans, cells, config.backends, started_at, options)
    plan_json = json.dumps(plan, separators=(",", ":"))
    if args.plan_out:
        args.plan_out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.expanduser().write_text(plan_json + "\n", encoding="utf-8")
        log(f"plan written to {args.plan_out}")
    outputs = {
        "plan": plan_json,
        "canary": json.dumps(canary_jobs, separators=(",", ":")),
        "skills": json.dumps(skill_jobs, separators=(",", ":")),
        "has_canary": "true" if canary_jobs else "false",
        "has_skills": "true" if skill_jobs else "false",
        "cells": str(len(cells)),
        "cached": str(hits),
        "selected": str(len(canary_jobs) + len(skill_jobs)),
    }
    output_path = (deps.environ.get("GITHUB_OUTPUT") or "").strip()
    if output_path:
        append_outputs(Path(output_path), outputs)
    print_plan(plans, cells, info)
    if api is None:
        body = render_comment(
            info,
            plans,
            cells,
            config.backends,
            started_at=started_at,
            now=deps.now(),
            seq=seq_label(info, cells),
            options=options,
        )
        print(body)
        return 0
    if not publish_distributed(
        api, info, plans, cells, config.backends, started_at, deps.now(), final=True, options=options
    ):
        # Cells would otherwise each try to create the missing comment (duplicates) — better no run at all.
        raise OrchestrationError("the PR comment could not be created or updated; not starting any cell")
    return 0


def _load_plan_for(
    args: argparse.Namespace,
) -> tuple[RunInfo, list[SkillPlan], list[Cell], list[str], datetime, TriggerOptions]:
    if not args.plan:
        raise OrchestrationError(f"`{args.mode}` needs --plan (path or JSON from the plan job)")
    return plan_from_dict(load_plan(args.plan))


def mode_cell(args: argparse.Namespace, deps: Deps) -> int:
    """Run exactly one cell of the plan on this runner and refresh the comment from the shared state."""
    config = load_config(args.config, deps.environ)
    checkout = args.checkout.expanduser().resolve()
    if not checkout.is_dir():
        raise OrchestrationError(f"checkout not found: {checkout}")
    info, plans, cells, backends, started_at, options = _load_plan_for(args)
    if not (args.skill and args.backend and args.case):
        raise OrchestrationError("`cell` needs --skill, --backend and --case")
    cell = next((c for c in cells if (c.skill, c.backend, c.case) == (args.skill, args.backend, args.case)), None)
    if cell is None:
        raise OrchestrationError(f"cell {args.skill}/{args.backend}/{args.case} is not in the plan")
    if cell.state is CellState.CACHED:
        log(f"{cell.label}: reused ({cell.status}), nothing to run")
        return 0
    if cell.state is CellState.SKIPPED:
        log(f"{cell.label}: not selected by the trigger, nothing to run")
        return 0
    if cell.state is CellState.EXHAUSTED:
        log(f"{cell.label}: not re-run ({cell.reason}), nothing to run")
        return 0
    if not args.fake_launcher and not config.launcher.exists():
        raise OrchestrationError(f"launcher not found: {config.launcher}")
    check_checkout_head(checkout, info.head_sha)
    api = _open_github(args, deps, info)
    if api is None:
        pull_request = synthetic_pull_request(info, deps.environ)
    else:
        try:
            pull_request = api.get_pull(info.pr_number)
        except (GitHubApiError, ValueError) as exc:
            raise OrchestrationError(f"cannot fetch PR #{info.pr_number} of {info.repo}: {exc}") from exc
    event_path = write_event_file(config.workdir, info, pull_request)
    launcher_env = build_launcher_env(deps.environ, info, pull_request, event_path)
    # the stub launcher needs no docker: local `--fake-launcher` runs must not fail on a laptop without it
    preflight = Preflight(None, None) if args.fake_launcher else Preflight(config.docker_check, config.docker_repair)

    def refresh() -> None:
        try:
            if api is None:
                body = render_comment(
                    info,
                    plans,
                    cells,
                    backends,
                    started_at=started_at,
                    now=deps.now(),
                    seq=seq_label(info, cells),
                    options=options,
                )
                print(body)
                return
            jobs = fetch_run_jobs(api, info) or {}
            catalogs: dict[str, list[dict]] = {}  # the heading's static score reads the catalogs fetched here anyway
            reconcile_cells(
                cells,
                info,
                deps.urlopen,
                jobs,
                final=False,
                started_at=started_at,
                local={cell.name: cell},
                sleep=deps.sleep,
                catalogs=catalogs,
            )
            publish_distributed(
                api,
                info,
                plans,
                cells,
                backends,
                started_at,
                deps.now(),
                final=False,
                options=options,
                static_scores=static_scores_for(plans, info, catalogs),
            )
        except Exception as exc:  # a comment refresh must never change the outcome of the cell
            warn(f"comment refresh failed (ignored): {type(exc).__name__}: {exc}")

    cell.state = CellState.RUNNING
    refresh()
    if not wait_for_docker(preflight, deps, config.docker_wait_minutes * 60.0):
        # Failing fast would let a runner with a dead docker eat the queue one cell per minute; the cell was held
        # for docker_wait_minutes instead, so the other runners kept draining the queue meanwhile.
        cell.state, cell.reason = CellState.ERROR, f"runner cannot reach docker — {preflight.last_error}"
        warn(f"{cell.label}: {cell.reason}")
        if args.results_dir is not None:
            write_cell_result(args.results_dir, info, cell)
        refresh()
        return 1
    try:
        retries = execute_cell(
            cell,
            cells.index(cell),
            config=config,
            checkout=checkout,
            info=info,
            deps=deps,
            launcher_env=launcher_env,
            fake_launcher=args.fake_launcher,
            preflight=preflight,
            on_update=lambda: None,
        )
    except RunnerUnavailable as exc:
        cell.state, cell.reason = CellState.ERROR, f"runner cannot reach docker — {exc}"
        warn(f"{cell.label}: {cell.reason}")
        retries = 0
    except Exception as exc:  # disk full, unreadable log, ...: report it in the comment instead of a bare traceback
        cell.state, cell.reason = CellState.ERROR, f"orchestrator error: {type(exc).__name__}: {exc}"[:200]
        warn(f"{cell.label}: {cell.reason}")
        retries = 0
    if args.results_dir is not None:
        try:
            log(f"cell result written to {write_cell_result(args.results_dir, info, cell)}")
        except OSError as exc:
            warn(
                f"cannot write the cell result file ({exc}); the report will fall back to the dashboard and the job state"
            )
    # the job's verdict is the cell's own outcome, taken before the refresh: the comment may already show a newer
    # result of another run for this cell (refresh_from_catalog); the job log and exit code say what THIS cell did
    outcome = cell.status if cell.state is CellState.DONE else f"error: {cell.reason}"
    exit_code = 0 if cell.state is CellState.DONE else 1
    refresh()
    log(f"cell {cell.label}: {outcome}" + (" (retried once)" if retries else ""))
    return exit_code


def wait_for_docker(preflight: Preflight, deps: Deps, max_wait_s: float) -> bool:
    """Preflight with patience: re-check every DOCKER_WAIT_POLL_S for up to max_wait_s before giving up."""
    polls = max(0, int(max_wait_s // DOCKER_WAIT_POLL_S))
    deadline = deps.clock() + max_wait_s
    for attempt in range(polls + 1):
        if preflight.ensure():
            return True
        if attempt >= polls or deps.clock() >= deadline:
            return False
        warn(
            f"docker unavailable on this runner ({preflight.last_error}); holding the cell and re-checking in "
            f"{DOCKER_WAIT_POLL_S:.0f}s ({attempt + 1}/{polls})"
        )
        deps.sleep(DOCKER_WAIT_POLL_S)
    return False


def mode_report(args: argparse.Namespace, deps: Deps) -> int:
    """Final render once every cell job has finished: unpublished finished jobs become launcher errors."""
    info, plans, cells, backends, started_at, options = _load_plan_for(args)
    api = _open_github(args, deps, info)
    if api is None:
        body = render_comment(
            info,
            plans,
            cells,
            backends,
            started_at=started_at,
            now=deps.now(),
            seq=seq_label(info, cells),
            options=options,
        )
        print(body)
        return 0
    jobs = fetch_run_jobs(api, info)
    results = load_cell_results(args.results_dir, cells, info)  # the cell jobs' own outcomes (run artifacts)
    log(f"report: {len(results)} cell result file(s) found" + (f" in {args.results_dir}" if args.results_dir else ""))
    posted = read_comment(api, info)
    ledger = parse_ledger(posted[1], info.run_id)  # results the cell jobs of THIS run already wrote into the comment
    # A cell whose job finished but whose result is neither in the ledger (its final refresh failed) nor listed
    # by the dashboard yet (the publisher's refresh call failed; the catalog re-lists every ~2 minutes) gets a
    # settle window before it is called an error.
    polls = int(CATALOG_SETTLE_S // CATALOG_SETTLE_POLL_S)
    deadline = deps.clock() + CATALOG_SETTLE_S
    catalogs: dict[str, list[dict]] = {}
    for attempt in range(polls + 1):
        catalogs = {}  # one catalog GET per skill per poll (reconcile_cells fills it): each poll sees the dashboard afresh
        reconcile_cells(
            cells,
            info,
            deps.urlopen,
            jobs or {},
            final=False,
            started_at=started_at,
            local=results,
            sleep=deps.sleep,
            ledger=ledger,
            catalogs=catalogs,
        )
        unlisted = [cell.label for cell in cells if cell.state is CellState.RUNNING and cell.reason == PUBLISHING]
        if not unlisted or attempt >= polls or deps.clock() >= deadline:
            break
        log(
            f"report: {len(unlisted)} finished cell(s) not listed by the dashboard yet ({', '.join(unlisted[:4])})"
            f" — waiting {CATALOG_SETTLE_POLL_S:.0f}s"
        )
        deps.sleep(CATALOG_SETTLE_POLL_S)
        if jobs is None:
            jobs = fetch_run_jobs(api, info)
    # the final pass reads the last poll's catalogs again (no second GET for the same state): a finished job without a
    # result anywhere becomes a launcher error, every other cell shows the version's newest result
    reconcile_cells(
        cells,
        info,
        deps.urlopen,
        jobs or {},
        final=True,
        started_at=started_at,
        local=results,
        sleep=deps.sleep,
        ledger=ledger,
        catalogs=catalogs,
    )
    # The headings' `static <n>/100` comes from the dashboard's catalog (one assessment per published run). A skill
    # without cells (no eval cases) was never looked up above, so it gets one GET here; an unavailable dashboard only
    # costs the suffix (fetch_catalog warns and returns []), never the report.
    for plan in plans:
        if not plan.skipped_reason and plan.name not in catalogs:
            catalogs[plan.name] = fetch_catalog(info.dashboard, plan.name, deps.urlopen, sleep=deps.sleep)
    written = publish_distributed(
        api,
        info,
        plans,
        cells,
        backends,
        started_at,
        deps.now(),
        final=True,
        options=options,
        static_scores=static_scores_for(plans, info, catalogs),
    )
    counts = tally(cells)
    errors = sum(1 for cell in cells if cell.state is CellState.ERROR)
    log(
        f"report: {counts.passed}/{counts.total} passed, {errors} launcher error(s)"
        f" · dashboard group(s) {info.groups_summary()}"
    )
    if jobs is None:
        warn("the run's job list was unavailable: unpublished cells are shown as queued — re-run the report job")
        return 1
    if not written:
        warn("the final comment could not be written — re-run the report job")
        return 1
    return 0


def print_plan(plans: list[SkillPlan], cells: list[Cell], info: RunInfo) -> None:
    reused = sum(1 for cell in cells if cell.state is CellState.CACHED and not cell.is_job)
    to_run = sum(1 for cell in cells if cell.is_job)
    not_selected = sum(1 for cell in cells if not cell.selected)
    print(
        f"plan: sha {info.sha7} · {len(cells)} cell(s): {reused} reused, {to_run} to run, {not_selected} not selected"
    )
    for plan in plans:
        kind = "canary" if plan.canary else "skill"
        if plan.skipped_reason:
            print(f"  {kind} {plan.name}: not evaluated — {plan.skipped_reason}")
            continue
        version = info.versions.get(plan.name)
        if version is None:
            print(f"  group {plan.name}: {info.group_for(plan.name)}")
            continue
        print(
            f"  group {plan.name}: {info.group_for(plan.name)} · content {version.content_sha256[:7]}"
            f" · tree {(version.tree_sha or '-')[:7]} · {len(version.equivalent_commits)} equivalent commit(s)"
        )
    for cell in cells:
        if cell.state is CellState.CACHED:
            print(f"  {cell.label}: reused {cell.status} {cell.url}")
        elif cell.state is CellState.SKIPPED:
            print(f"  {cell.label}: not selected")
        elif cell.state is CellState.EXHAUSTED:
            print(f"  {cell.label}: not re-run ({cell.reason}, latest {cell.status})")
        else:
            print(f"  {cell.label}: run")


def check_checkout_head(checkout: Path, head_sha: str) -> None:
    head = git_head(checkout)
    if head and head != head_sha:
        warn(f"checkout HEAD {head[:7]} != HEAD_SHA {head_sha[:7]}: published runs will not be cache hits for this sha")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "plan", "cell", "report"],
        help="run = whole matrix in this process (default); plan / cell / report = the distributed workflow stages",
    )
    parser.add_argument(
        "--scope",
        default="full",
        choices=["full", "smoke"],
        help="plan/run: smoke = canary only (= trigger arg `smoke-only`)",
    )
    parser.add_argument(
        "--trigger-args",
        default="",
        help=(
            "plan/run: the text after `/run-skill-eval` — smoke-only, with-smoke, force, rerun-failed,"
            " cells=<b>/<c>,…, group=<id>"
        ),
    )
    parser.add_argument("--group", help="plan/run: dashboard group id for every skill of this run (no anchoring)")
    parser.add_argument("--plan", help="cell/report: plan file path or the plan JSON text")
    parser.add_argument("--plan-out", type=Path, help="plan: also write the plan JSON to this file")
    parser.add_argument("--skill", help="cell: skill of the cell to run")
    parser.add_argument("--backend", help="cell: backend of the cell to run")
    parser.add_argument("--case", help="cell: case of the cell to run")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="cell: write <cell>.json here (uploaded as an artifact); report: read the downloaded artifacts from here",
    )
    parser.add_argument("--config", required=True, type=Path, help="path to .github/skill-eval.yml")
    parser.add_argument("--checkout", required=True, type=Path, help="repo root of the PR head checkout")
    parser.add_argument("--fake-launcher", action="store_true", help="use an internal stub instead of the launcher")
    parser.add_argument("--no-github", action="store_true", help="print comment bodies instead of calling GitHub")
    parser.add_argument("--dry-run", action="store_true", help="print the plan (cells, cache hits) and exit")
    return parser.parse_args(argv)


def orchestrate(args: argparse.Namespace, deps: Deps) -> int:
    config = load_config(args.config, deps.environ)
    checkout = args.checkout.expanduser().resolve()
    if not checkout.is_dir():
        raise OrchestrationError(f"checkout not found: {checkout}")
    options = trigger_options_from_args(args)
    info = replace(
        read_run_info(deps.environ, config, checkout, no_github=args.no_github), group_override=options.group
    )
    log(
        f"PR #{info.pr_number} · sha {info.sha7} · base {info.base_sha[:7] or '-'}"
        f" · run {info.run_id}/{info.run_attempt} · {describe_trigger(options)}"
    )

    gh_token = (deps.environ.get("GH_TOKEN") or "").strip()
    api: GitHubApi | None = None  # None in --no-github mode: skill detection then goes straight to git
    if not args.no_github:
        api = GitHubApi(info.server_url, info.repo, gh_token, transport=deps.github_transport, sleep=deps.sleep)

    info, plans, cells, hits = prepare_matrix(config, info, checkout, api, deps, options)
    not_selected = sum(1 for cell in cells if not cell.selected)
    to_run = sum(1 for cell in cells if cell.is_job)
    log(f"{len(cells)} cell(s): {hits} reused, {not_selected} not selected, {to_run} to run")
    if args.dry_run:
        print_plan(plans, cells, info)
        return 0

    if not args.fake_launcher and not config.launcher.exists():
        raise OrchestrationError(f"launcher not found: {config.launcher}")
    check_checkout_head(checkout, info.head_sha)

    updater: PrintingCommentUpdater | GitHubCommentUpdater
    if api is None:
        pull_request = synthetic_pull_request(info, deps.environ)
        updater = PrintingCommentUpdater()
    else:
        try:
            pull_request = api.get_pull(info.pr_number)
        except (GitHubApiError, ValueError) as exc:  # ValueError: a 2xx with a non-JSON body
            raise OrchestrationError(f"cannot fetch PR #{info.pr_number} of {info.repo}: {exc}") from exc
        log(f"PR title: {pull_request.get('title')!r} by {(pull_request.get('user') or {}).get('login')}")
        updater = GitHubCommentUpdater(
            api,
            info.pr_number,
            info.marker,
            min_interval_s=deps.comment_interval_s,
            clock=deps.clock,
            start_timer=deps.start_timer,
        )

    event_path = write_event_file(config.workdir, info, pull_request)
    log(f"event file: {event_path}")
    launcher_env = build_launcher_env(deps.environ, info, pull_request, event_path)
    orchestrator = Orchestrator(
        config=config,
        checkout=checkout,
        info=info,
        plans=plans,
        cells=cells,
        deps=deps,
        updater=updater,
        launcher_env=launcher_env,
        fake_launcher=args.fake_launcher,
        started_at=deps.now(),
        preflight=Preflight(config.docker_check, config.docker_repair),
        options=options,
    )
    return orchestrator.run()


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parse_args(argv)
    modes = {"run": orchestrate, "plan": mode_plan, "cell": mode_cell, "report": mode_report}
    try:
        return modes[args.mode](args, deps or Deps())
    except OrchestrationError as exc:
        log(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
