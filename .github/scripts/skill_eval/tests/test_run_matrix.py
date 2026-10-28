# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for run_matrix.py — no network, no docker.

Fakes: an in-memory GitHub REST API, a bash launcher script that writes the log line and
published.json like the real evaluator, and a dashboard catalog served from a dict.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_matrix as rm

REPO = "NVIDIA/cosmos-framework"
SHA = "a" * 40
SHA7 = SHA[:7]
OTHER_SHA = "b" * 40
BASE_SHA = "c" * 40
DASH = "http://dash.test"
CANARY = rm.CANARY_SKILL  # cosmos3-ci-smoke: lives in .github/scripts/skill_eval, never under .agents/skills
SKILL = "cosmos3-generate-media"  # a shipped skill under .agents/skills
BACKENDS = ["claude", "codex", "cursor", "opencode"]
FIXED_NOW = datetime(2026, 10, 18, 7, 0, 0, tzinfo=timezone.utc)

FAKE_LAUNCHER = """#!/usr/bin/env bash
# Fake host launcher: records argv + env, then behaves like the evaluator CLI for the given case.
set -eu
out=""; backend=""; case_id=""; skill=""
args=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) out="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --case) case_id="$2"; shift 2 ;;
    eval) skill="$(basename "$2")"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$out/$skill"
printf '%s\\n' "${args[@]}" > "$out.argv"
env | sort > "$out.env"
echo "[cosmos-eval] fake launcher for $skill/$backend/$case_id"
case "$case_id" in
  explode)
    echo "Traceback: boom goes the launcher"
    exit 1 ;;
  unpublished)
    echo "[cosmos-eval] status: PASS"
    echo '{"status": "failed", "error": "S3 down"}' > "$out/$skill/published.json"
    exit 0 ;;
  sockflip)
    if [ ! -e "$out.attempt1" ]; then
      touch "$out.attempt1"
      echo "docker: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock"
      exit 1
    fi ;;
  noreport|gateway)
    # status taxonomy v2: the evaluator publishes NO-REPORT (exit 3) / ERROR (exit 4) like any other result
    if [ "$case_id" = noreport ]; then status=NO-REPORT; code=3; else status=ERROR; code=4; fi
    echo "[cosmos-eval] case $case_id: $status — stopped_early / model_api"
    echo "[cosmos-eval] status: $status"
    run_id="r-$backend-$case_id"
    printf '{"status": "ok", "url": "%s/runs/%s/", "run_id": "%s"}\n' "__DASH__" "$run_id" "$run_id" > "$out/$skill/published.json"
    exit $code ;;
esac
status=PASS
if [ "$backend" = codex ]; then status=FAIL; fi
echo "[cosmos-eval] status: $status"
run_id="r-$backend-$case_id"
printf '{"status": "ok", "url": "%s/runs/%s/", "run_id": "%s"}\\n' "__DASH__" "$run_id" "$run_id" > "$out/$skill/published.json"
"""


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeGitHub:
    """In-memory GitHub REST API used as the `transport` of GitHubApi."""

    def __init__(self, pr_number: int = 128, comments: list[dict] | None = None):
        self.pr_number = pr_number
        self.comments: list[dict] = list(comments or [])
        self.next_id = 500
        self.calls: list[tuple[str, str, dict | None]] = []
        self.scripted: list[object] = []  # HTTP status codes or exceptions injected before normal handling
        self.lose_post_response = False  # apply the next POST server-side but fail the response (torn connection)
        # GET /compare/{base}...{head}: like GitHub, the whole `files` list (at most 300 entries) is on the first page
        # and later pages have no `files` key at all (`per_page`/`page` paginate the commits). None -> 404 (forces the
        # git fallback). `compare_paginates_files` models a GitHub that sliced `files` by `per_page` instead.
        self.compare_files: list[dict] | None = [{"filename": f".agents/skills/{SKILL}/SKILL.md", "status": "modified"}]
        self.compare_paginates_files = False
        self.jobs: list[dict] = []  # GET /actions/runs/{id}/jobs: {"name","status","conclusion","html_url"}
        self.fail_jobs_api = False  # every jobs-API GET answers 503
        self.pr = {
            "number": pr_number,
            "title": "Smoke test",
            "html_url": f"https://github.com/{REPO}/pull/{pr_number}",
            "user": {"login": "octocat", "avatar_url": "https://avatars/1", "html_url": "https://github.com/c"},
            "head": {"sha": SHA, "ref": "skill-eval-config"},
            "base": {"ref": "main"},
            "draft": True,
            "state": "open",
        }

    def __call__(self, method, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((method, url, payload))
        if self.scripted:
            item = self.scripted.pop(0)
            if isinstance(item, Exception):
                raise item
            return item, {}, b'{"message": "injected"}'
        parsed = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        path = parsed.path
        if method == "GET" and path.startswith(f"/repos/{REPO}/compare/"):
            if self.compare_files is None:
                return 404, {}, b'{"message": "Not Found"}'
            per_page, page = int(query.get("per_page", 30)), int(query.get("page", 1))
            compare: dict = {"status": "ahead", "total_commits": 1, "commits": []}
            if self.compare_paginates_files:
                compare["files"] = self.compare_files[(page - 1) * per_page : page * per_page]
            elif page == 1:
                compare["files"] = self.compare_files[: rm.COMPARE_MAX_FILES]
            return 200, {}, json.dumps(compare).encode()
        if method == "GET" and path == f"/repos/{REPO}/pulls/{self.pr_number}":
            return 200, {}, json.dumps(self.pr).encode()
        if method == "GET" and path.startswith(f"/repos/{REPO}/actions/runs/") and path.endswith("/jobs"):
            if self.fail_jobs_api:
                return 503, {}, b'{"message": "unavailable"}'
            per_page, page = int(query.get("per_page", 30)), int(query.get("page", 1))
            start = (page - 1) * per_page
            payload = {"total_count": len(self.jobs), "jobs": self.jobs[start : start + per_page]}
            return 200, {}, json.dumps(payload).encode()
        if method == "GET" and path.startswith(f"/repos/{REPO}/issues/comments/"):
            comment_id = int(path.rsplit("/", 1)[1])
            for comment in self.comments:
                if comment["id"] == comment_id:
                    return 200, {}, json.dumps(comment).encode()
            return 404, {}, b'{"message": "Not Found"}'
        if method == "GET" and path == f"/repos/{REPO}/issues/{self.pr_number}/comments":
            per_page, page = int(query.get("per_page", 30)), int(query.get("page", 1))
            start = (page - 1) * per_page
            return 200, {}, json.dumps(self.comments[start : start + per_page]).encode()
        if method == "POST" and path == f"/repos/{REPO}/issues/{self.pr_number}/comments":
            comment = {"id": self.next_id, "body": payload["body"]}
            self.next_id += 1
            self.comments.append(comment)
            if self.lose_post_response:
                self.lose_post_response = False
                raise rm.TransportError("connection reset after the server applied the POST")
            return 201, {}, json.dumps(comment).encode()
        if method == "PATCH" and path.startswith(f"/repos/{REPO}/issues/comments/"):
            comment_id = int(path.rsplit("/", 1)[1])
            for comment in self.comments:
                if comment["id"] == comment_id:
                    comment["body"] = payload["body"]
                    return 200, {}, json.dumps(comment).encode()
        return 404, {}, b'{"message": "Not Found"}'

    def bodies(self, method: str) -> list[str]:
        return [payload["body"] for called, _url, payload in self.calls if called == method and payload]

    def count(self, method: str) -> int:
        return sum(1 for called, _url, _payload in self.calls if called == method)

    def urls(self, method: str, containing: str) -> list[str]:
        return [url for called, url, _payload in self.calls if called == method and containing in url]


class FakeResponse:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.data


class FakeDashboard:
    """Stands in for urllib.request.urlopen for GET <dashboard>/api/catalog?skill=..."""

    def __init__(self, runs_by_skill: dict[str, list[dict]] | None = None, error: Exception | None = None, raw=None):
        self.runs_by_skill = runs_by_skill or {}
        self.error = error
        self.raw = raw
        self.urls: list[str] = []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        if self.error is not None:
            raise self.error
        if self.raw is not None:
            return FakeResponse(self.raw)
        skill = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(request.full_url).query)).get("skill")
        return FakeResponse(json.dumps(self.runs_by_skill.get(skill, [])).encode())


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeTimer:
    def __init__(self, delay: float, callback):
        self.delay = delay
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeTimers:
    def __init__(self):
        self.timers: list[FakeTimer] = []

    def __call__(self, delay: float, callback) -> FakeTimer:
        timer = FakeTimer(delay, callback)
        self.timers.append(timer)
        return timer

    def fire(self) -> None:
        pending = [timer for timer in self.timers if not timer.cancelled]
        self.timers.clear()
        for timer in pending:
            timer.callback()


def catalog_run(run_id, sha, case, backend, status, created_at, *, case_status=None, backend_status=True):
    backend_entry = {"backend": backend}
    if backend_status:
        backend_entry["status"] = status
    return {
        "run_id": run_id,
        "created_at": created_at,
        "skill": {"name": SKILL, "commit_sha": sha},
        "cases": [{"case_id": case, "status": case_status or status, "backends": [backend_entry]}],
    }


def write_eval_config(root: Path, skill: str, case_ids: list[str], *, canary: bool = False) -> None:
    path = rm.eval_config_path(root, skill, canary=canary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"evals": [{"id": case_id, "prompt": "x"} for case_id in case_ids]}))


def write_config(
    tmp_path: Path,
    launcher: Path,
    *,
    skills=None,
    smoke: bool | None = None,
    env_file=None,
    docker_check=None,
    docker_repair=None,
) -> Path:
    """Config like .github/skill-eval.yml: no `skills:` unless given (detection), `smoke:` only when given."""
    lines = [
        "# SPDX-FileCopyrightText: header comment",
        "",
        f"backends: [{', '.join(BACKENDS)}]",
        "image: registry.example.com/cosmos-skill-eval:v1",
        f"dashboard: {DASH}/",
        f"launcher: {launcher}",
        f"env_file: {env_file or tmp_path / 'missing.env'}",
        "codex_model: openai/openai/gpt-6-astra",
        f"workdir: {tmp_path / 'work'}",
    ]
    if skills is not None:
        lines.insert(2, f"skills: [{', '.join(skills)}]            # explicit override, trailing comment")
    if smoke is not None:
        lines.append(f"smoke: {'true' if smoke else 'false'}")
    if docker_check is not None:
        lines.append(f"docker_check: {docker_check}")
    if docker_repair is not None:
        lines.append(f"docker_repair: {docker_repair}")
    path = tmp_path / "skill-eval.yml"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


SOCKET_DENIED = "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock"


def ci_env(**overrides) -> dict[str, str]:
    env = {
        "GH_TOKEN": "ghs_test",
        "GITHUB_REPOSITORY": REPO,
        "PR_NUMBER": "128",
        "HEAD_SHA": SHA,
        "BASE_SHA": BASE_SHA,
        "HEAD_REF": "skill-eval-config",
        "GITHUB_RUN_ID": "987",
        "GITHUB_RUN_ATTEMPT": "2",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    env.update(overrides)
    return env


def make_deps(env, github, dashboard, *, timers=None, clock=None, interval=0.0) -> rm.Deps:
    return rm.Deps(
        environ=env,
        github_transport=github,
        urlopen=dashboard,
        clock=clock or FakeClock(),
        sleep=lambda seconds: None,
        start_timer=timers or FakeTimers(),
        now=lambda: FIXED_NOW,
        comment_interval_s=interval,
        cell_timeout_s=60,
    )


def run_info(**overrides) -> rm.RunInfo:
    values = dict(
        repo=REPO,
        server_url="https://github.com",
        pr_number=128,
        head_sha=SHA,
        head_ref="skill-eval-config",
        run_id="987",
        run_attempt="2",
        dashboard=DASH,
        base_sha=BASE_SHA,
    )
    values.update(overrides)
    return rm.RunInfo(**values)


def read_argv(workdir: Path, cell_name: str) -> list[str]:
    return (workdir / SHA7 / f"{cell_name}.argv").read_text().splitlines()


def read_env(workdir: Path, cell_name: str) -> dict[str, str]:
    pairs = (line.partition("=") for line in (workdir / SHA7 / f"{cell_name}.env").read_text().splitlines())
    return {key: value for key, _sep, value in pairs}


@pytest.fixture
def launcher(tmp_path: Path) -> Path:
    script = tmp_path / "fake-launcher"
    script.write_text(FAKE_LAUNCHER.replace("__DASH__", DASH))
    script.chmod(0o755)
    return script


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Plain directory (no git): the canary with one case, SKILL with two, and a touched skill without eval.config."""
    root = tmp_path / "checkout"
    write_eval_config(root, CANARY, ["smoke-hello"], canary=True)
    write_eval_config(root, SKILL, ["hello", "world"])
    inference = root / ".agents" / "skills" / "cosmos3-inference" / "SKILL.md"
    inference.parent.mkdir(parents=True)
    inference.write_text("no eval config\n")
    return root


def git(root: Path, *args: str) -> str:
    options = ["-c", "user.name=skill-eval tests", "-c", "user.email=skill-eval@test", "-c", "commit.gpgsign=false"]
    options += ["-c", "core.hooksPath=/dev/null"]
    result = subprocess.run(["git", *options, "-C", str(root), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def git_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    """A real repository: a base commit with shipped skills and a head commit shaped like a PR.

    The head modifies SKILL, deletes cosmos3-gone, adds cosmos3-inference without an eval.config, adds the canary
    under .github/scripts/skill_eval, leaves a stray copy of the canary under .agents/skills, adds one asset with
    a non-ASCII name to cosmos3-accent and does not touch cosmos3-untouched. Returns (root, base sha, head sha).
    """
    root = tmp_path / "git-checkout"
    root.mkdir()
    git(root, "init", "-q")
    write_eval_config(root, SKILL, ["hello", "world"])
    (root / ".agents" / "skills" / SKILL / "SKILL.md").write_text("v1\n")
    write_eval_config(root, "cosmos3-gone", ["g"])
    write_eval_config(root, "cosmos3-untouched", ["u"])
    write_eval_config(root, "cosmos3-accent", ["a"])
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    base_sha = git(root, "rev-parse", "HEAD")

    (root / ".agents" / "skills" / SKILL / "SKILL.md").write_text("v2\n")
    shutil.rmtree(root / ".agents" / "skills" / "cosmos3-gone")
    inference = root / ".agents" / "skills" / "cosmos3-inference" / "SKILL.md"
    inference.parent.mkdir(parents=True)
    inference.write_text("no eval config\n")
    write_eval_config(root, CANARY, ["smoke-hello"], canary=True)
    stray = root / ".agents" / "skills" / CANARY / "SKILL.md"
    stray.parent.mkdir(parents=True)
    stray.write_text("stray copy of the canary\n")
    accent = root / ".agents" / "skills" / "cosmos3-accent" / "assets" / "café.json"  # `git diff` quotes this path
    accent.parent.mkdir(parents=True)
    accent.write_text("{}\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "head")
    return root, base_sha, git(root, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_load_config_takes_image_and_dashboard_from_the_environment(tmp_path: Path):
    """The config file's `image:` / `dashboard:` are placeholders in the mirrored tree: the repository variables
    COSMOS_SKILL_EVAL_IMAGE / COSMOS_SKILL_EVAL_DASHBOARD (exported by the workflow) override them when set."""
    path = write_config(tmp_path, tmp_path / "launch")
    plain = rm.load_config(path, {})
    assert plain.image == "registry.example.com/cosmos-skill-eval:v1" and plain.dashboard == DASH
    env = {"COSMOS_SKILL_EVAL_IMAGE": " nvcr.example/org/cosmos-skill-eval:v9 ", "COSMOS_SKILL_EVAL_DASHBOARD": "https://dash.example/"}
    overridden = rm.load_config(path, env)
    assert overridden.image == "nvcr.example/org/cosmos-skill-eval:v9" and overridden.dashboard == "https://dash.example"
    assert rm.load_config(path, {"COSMOS_SKILL_EVAL_IMAGE": ""}).image == plain.image  # empty = unset


def test_parse_simple_yaml_scalars_lists_and_comments():
    text = (
        "# header comment\n"
        "\n"
        "skills: [a, b]  # trailing\n"
        "backends: ['claude', \"codex\"]\n"
        "image: registry.example.com/cosmos-skill-eval:v1\n"
        "dashboard: https://dashboard.example.com\n"
        "launcher: ~/cosmos-skill-eval-src/scripts/cosmos-skill-eval\n"
    )
    data = rm.parse_simple_yaml(text)
    assert data == {
        "skills": ["a", "b"],
        "backends": ["claude", "codex"],
        "image": "registry.example.com/cosmos-skill-eval:v1",
        "dashboard": "https://dashboard.example.com",
        "launcher": "~/cosmos-skill-eval-src/scripts/cosmos-skill-eval",
    }


@pytest.mark.parametrize("text", ["parent:\n  child: 1\n", "novalue:\n", "just text\n"])
def test_parse_simple_yaml_rejects_unsupported_shapes(text):
    with pytest.raises(rm.OrchestrationError):
        rm.parse_simple_yaml(text)


def test_load_config_expands_paths_and_validates(tmp_path: Path):
    path = tmp_path / "c.yml"
    path.write_text(
        "backends: [claude]\nimage: img\ndashboard: http://d/\n"
        "launcher: ~/bin/launch\nworkdir: ~/work\nenv_file: ~/e.env\ncodex_model: m\n"
    )
    config = rm.load_config(path)
    assert config.launcher == Path.home() / "bin" / "launch"
    assert config.workdir == Path.home() / "work"
    assert config.env_file == Path.home() / "e.env"
    assert config.dashboard == "http://d"
    assert config.codex_model == "m"
    assert config.skills is None and config.smoke is None  # no `skills:` -> detect from the PR; canary off by default

    path.write_text("backends: [claude]\ndashboard: http://d\nlauncher: l\nworkdir: w\n")
    with pytest.raises(rm.OrchestrationError, match="image"):
        rm.load_config(path)
    with pytest.raises(rm.OrchestrationError, match="config not found"):
        rm.load_config(tmp_path / "nope.yml")


def test_load_config_optional_skills_override_and_smoke_flag(tmp_path: Path):
    path = tmp_path / "c.yml"
    base = "backends: [claude]\nimage: i\ndashboard: http://d\nlauncher: l\nworkdir: w\n"
    path.write_text(base + "skills: [s2, s1]   # explicit override\nsmoke: false\n")
    config = rm.load_config(path)
    assert config.skills == ["s2", "s1"] and config.smoke is False

    path.write_text(base + "smoke: TRUE\n")
    assert rm.load_config(path).smoke is True
    path.write_text(base + "smoke: auto\n")
    assert rm.load_config(path).smoke is None
    for bad in ("skills: []\n", "skills: s\n"):
        path.write_text(base + bad)
        with pytest.raises(rm.OrchestrationError, match="skills"):
            rm.load_config(path)
    path.write_text(base + "smoke: maybe\n")
    with pytest.raises(rm.OrchestrationError, match="smoke"):
        rm.load_config(path)


# ---------------------------------------------------------------------------
# cells and cache
# ---------------------------------------------------------------------------


def test_build_cells_follows_file_then_config_order_and_explains_missing_skills(checkout: Path):
    plans = rm.load_skill_plans(checkout, [SKILL, "cosmos3-inference", "cosmos3-gone"], smoke=False)
    assert [plan.name for plan in plans] == [SKILL, "cosmos3-inference", "cosmos3-gone"]
    assert plans[0].cases == ["hello", "world"] and not plans[0].canary
    assert plans[1].skipped_reason == "eval.config not found at `.agents/skills/cosmos3-inference/eval/eval.config`"
    assert plans[2].skipped_reason == "`.agents/skills/cosmos3-gone` does not exist at this head (deleted or renamed)"
    cells = rm.build_cells(plans, BACKENDS)
    assert [(cell.case, cell.backend) for cell in cells[:5]] == [
        ("hello", "claude"),
        ("hello", "codex"),
        ("hello", "cursor"),
        ("hello", "opencode"),
        ("world", "claude"),
    ]
    assert len(cells) == 8 and all(cell.state is rm.CellState.QUEUED and not cell.canary for cell in cells)


def test_load_skill_plans_puts_the_canary_first_unless_disabled(checkout: Path):
    plans = rm.load_skill_plans(checkout, [SKILL])
    assert [(plan.name, plan.canary) for plan in plans] == [(CANARY, True), (SKILL, False)]
    assert plans[0].cases == ["smoke-hello"]
    cells = rm.build_cells(plans, ["claude"])
    assert [(cell.skill, cell.case, cell.canary) for cell in cells] == [
        (CANARY, "smoke-hello", True),
        (SKILL, "hello", False),
        (SKILL, "world", False),
    ]
    assert rm.skill_dir(CANARY, canary=True) == PurePosixPath(".github/scripts/skill_eval/cosmos3-ci-smoke")
    assert rm.skill_dir(SKILL) == PurePosixPath(f".agents/skills/{SKILL}")
    assert [plan.name for plan in rm.load_skill_plans(checkout, [SKILL], smoke=False)] == [SKILL]

    rm.eval_config_path(checkout, CANARY, canary=True).unlink()  # canary without its eval.config: reported, not fatal
    plan = rm.load_skill_plans(checkout, [])[0]
    assert plan.canary and plan.skipped_reason == (
        "eval.config not found at `.github/scripts/skill_eval/cosmos3-ci-smoke/eval/eval.config`"
    )


def test_load_skill_plan_tolerates_bad_json(checkout: Path):
    rm.eval_config_path(checkout, SKILL).write_text("{not json")
    plan = rm.load_skill_plan(checkout, SKILL)
    assert plan.skipped_reason is not None and "not valid JSON" in plan.skipped_reason


def test_find_cached_result_takes_newest_run_of_this_version():
    runs = [
        catalog_run("run-old", SHA, "hello", "claude", "PASS", "2026-10-17T01:00:00Z"),
        catalog_run("run-new", SHA, "hello", "claude", "FAIL", "2026-10-17T02:00:00Z"),
        catalog_run("run-other-sha", OTHER_SHA, "hello", "claude", "PASS", "2026-10-17T03:00:00Z"),
        catalog_run("run-fallback", SHA, "world", "codex", "PASS", "2026-10-17T01:00:00Z", backend_status=False),
    ]
    version = rm.SkillVersion.of_commit(SHA)  # a version known by its commit only: the pre-v2 cache key
    hit = rm.find_cached_result(runs, version, "hello", "claude", DASH)
    assert hit is not None
    assert (hit.status, hit.url, hit.run_id) == ("FAIL", f"{DASH}/runs/run-new/", "run-new")
    assert hit.verdict is None and hit.group_id is None and hit.source_run_id is None
    fallback = rm.find_cached_result(runs, version, "world", "codex", DASH)
    assert fallback is not None and fallback.status == "PASS"  # backend entry without status -> case status
    assert rm.find_cached_result(runs, version, "hello", "cursor", DASH) is None
    assert rm.find_cached_result(runs, rm.SkillVersion.of_commit(OTHER_SHA), "world", "codex", DASH) is None
    # a version is matched by the manifest's content sha or by any equivalent commit, not only by the head commit
    runs[2]["skill"]["content_sha256"] = "c" * 64
    by_content = rm.SkillVersion("c" * 64, None, ["f" * 40])
    assert rm.find_cached_result(runs, by_content, "hello", "claude", DASH).run_id == "run-other-sha"
    equivalent = rm.SkillVersion("", None, [SHA, OTHER_SHA])
    assert rm.find_cached_result(runs, equivalent, "hello", "claude", DASH).run_id == "run-other-sha"  # the newest


def test_fetch_catalog_failures_mean_no_cache():
    broken = FakeDashboard(error=urllib.error.URLError("connection refused"))
    assert rm.fetch_catalog(DASH, SKILL, broken, retries=1) == []
    assert broken.urls == [f"{DASH}/api/catalog?skill={SKILL}"]
    assert rm.fetch_catalog(DASH, SKILL, FakeDashboard(raw=b'{"not": "a list"}'), retries=1) == []
    assert rm.fetch_catalog(DASH, SKILL, FakeDashboard(raw=b"garbage"), retries=1) == []
    # transient failures are retried with backoff (1 s, 2 s) before the lookup counts as "no cache"
    sleeps: list[float] = []
    flaky = FakeDashboard(error=urllib.error.URLError("blip"))
    assert rm.fetch_catalog(DASH, SKILL, flaky, sleep=sleeps.append) == []
    assert sleeps == [1.0, 2.0] and len(flaky.urls) == 3


def test_apply_cache_marks_cells(checkout: Path):
    plans = rm.load_skill_plans(checkout, [SKILL], smoke=False)
    cells = rm.build_cells(plans, BACKENDS)
    dashboard = FakeDashboard({SKILL: [catalog_run("r1", SHA, "hello", "cursor", "PASS", "2026-10-17T01:00:00Z")]})
    assert rm.apply_cache(cells, run_info(), dashboard) == 1
    cached = [cell for cell in cells if cell.state is rm.CellState.CACHED]
    assert [(cell.case, cell.backend, cell.status, cell.url) for cell in cached] == [
        ("hello", "cursor", "PASS", f"{DASH}/runs/r1/")
    ]


# ---------------------------------------------------------------------------
# skill detection: which skills does the PR touch?
# ---------------------------------------------------------------------------


def compare_entry(path: str, status: str = "modified", previous: str | None = None) -> dict:
    entry = {"filename": path, "status": status}
    if previous:
        entry["previous_filename"] = previous
    return entry


def test_skills_in_paths_filters_sorts_and_reserves_the_canary_name(capsys):
    paths = [
        ".agents/skills/cosmos3-zeta/SKILL.md",
        ".agents/skills/cosmos3-alpha/eval/eval.config",
        ".agents/skills/cosmos3-alpha/assets/x.json",
        ".agents/skills/README.md",  # a file directly under skills/ is not a skill
        f".agents/skills/{CANARY}/SKILL.md",  # leftover copy of the canary: ignored with a warning
        ".github/scripts/skill_eval/cosmos3-ci-smoke/eval/eval.config",
        "docs/setup.md",
        "agents/skills/cosmos3-not-really/SKILL.md",
    ]
    assert rm.skills_in_paths(paths) == ["cosmos3-alpha", "cosmos3-zeta"]
    assert "never a shipped skill" in capsys.readouterr().out
    assert rm.skills_in_paths([]) == []


def test_compare_changed_paths_reads_the_whole_files_list_and_includes_renamed_from_names():
    github = FakeGitHub()
    github.compare_files = [compare_entry(f"docs/page-{index}.md") for index in range(148)] + [
        compare_entry(".agents/skills/cosmos3-new/SKILL.md", "renamed", previous=".agents/skills/cosmos3-old/SKILL.md"),
        compare_entry(".agents/skills/cosmos3-gone/eval/eval.config", "removed"),
    ]
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    paths = rm.compare_changed_paths(api, BASE_SHA, SHA)
    assert paths is not None and len(paths) == 151  # 150 files; the rename counts under both names
    assert rm.skills_in_paths(paths) == ["cosmos3-gone", "cosmos3-new", "cosmos3-old"]
    # GitHub serves all 150 files on page 1: more than a page, so nothing else is requested
    assert github.urls("GET", "/compare/") == [
        f"https://api.github.com/repos/{REPO}/compare/{BASE_SHA}...{SHA}?per_page=100&page=1"
    ]
    github.compare_paginates_files, github.calls = True, []  # a GitHub that paginated `files` would be read fully too
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) == paths
    assert github.urls("GET", "/compare/") == [
        f"https://api.github.com/repos/{REPO}/compare/{BASE_SHA}...{SHA}?per_page=100&page={page}" for page in (1, 2)
    ]


@pytest.mark.parametrize(("count", "requests"), [(100, 2), (111, 1), (299, 1)])
def test_compare_changed_paths_keeps_every_file_of_a_100_to_299_file_pr(count, requests):
    """Regression: GitHub's later pages carry no `files` key — that must end the listing, not discard it.

    Exactly 100 files is ambiguous (a full page of a paginated list?), so page 2 is asked for; more than 100
    on one page proves the list came whole and nothing else is requested.
    """
    github = FakeGitHub()
    github.compare_files = [compare_entry(f".agents/skills/cosmos3-s{index:03d}/SKILL.md") for index in range(count)]
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    paths = rm.compare_changed_paths(api, BASE_SHA, SHA)
    assert paths is not None and len(paths) == count and len(rm.skills_in_paths(paths)) == count
    assert len(github.urls("GET", "/compare/")) == requests


def test_compare_changed_paths_gives_up_on_the_file_cap_and_on_api_errors():
    github = FakeGitHub()
    github.compare_files = [compare_entry(f"f{index}") for index in range(313)]
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github, sleep=lambda seconds: None)
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) is None  # GitHub lists 300 of the 313 files: truncated
    assert len(github.urls("GET", "/compare/")) == 1
    github.compare_files = github.compare_files[:300]
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) is None  # exactly the cap is ambiguous too
    github.compare_paginates_files, github.calls = True, []
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) is None  # 3 x 100 reaches the cap as well
    assert len(github.urls("GET", "/compare/")) == 3
    github.compare_paginates_files = False

    github.compare_files = [compare_entry(f"f{index}") for index in range(250)]
    paths = rm.compare_changed_paths(api, BASE_SHA, SHA)
    assert paths is not None and len(paths) == 250

    github.compare_files = None  # 404, e.g. the base sha vanished after a force-push
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) is None
    github.compare_files = []
    github.scripted = [502, 502, 502]  # 5xx after all retries
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) is None
    github.scripted = [200]  # 2xx without a `files` list on the FIRST page
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) is None
    assert rm.compare_changed_paths(api, BASE_SHA, SHA) == []  # an empty diff (`files: []`) is a valid answer


def test_detect_changed_skills_prefers_the_api_and_falls_back_to_git(git_checkout):
    root, base_sha, head_sha = git_checkout
    info = run_info(head_sha=head_sha, base_sha=base_sha)
    github = FakeGitHub()
    github.compare_files = [compare_entry(".agents/skills/cosmos3-from-api/SKILL.md")]
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github, sleep=lambda seconds: None)
    assert rm.detect_changed_skills(info, root, api) == ["cosmos3-from-api"]

    github.compare_files = None  # API path fails -> `git diff --name-only <base>...HEAD` in the checkout
    # alphabetical; the deleted skill is a candidate too; cosmos3-accent only changed a non-ASCII-named asset
    expected = ["cosmos3-accent", SKILL, "cosmos3-gone", "cosmos3-inference"]
    assert rm.detect_changed_skills(info, root, api) == expected
    assert rm.detect_changed_skills(info, root, None) == expected  # --no-github: git only


def test_detect_changed_skills_errors_are_clear(tmp_path: Path):
    with pytest.raises(rm.OrchestrationError, match="BASE_SHA"):
        rm.detect_changed_skills(run_info(base_sha=""), tmp_path, None)
    with pytest.raises(rm.OrchestrationError, match="cannot list the files changed by the PR") as exc_info:
        rm.detect_changed_skills(run_info(), tmp_path, None)  # not a git repository
    message = str(exc_info.value)
    assert f"git diff --name-only {BASE_SHA[:7]}...HEAD" in message
    assert f"git fetch --depth=1 origin {BASE_SHA}" in message and "`skills:`" in message


def test_select_skills_explicit_override_skips_detection_and_drops_the_canary_name(tmp_path: Path, capsys):
    config = rm.Config(backends=BACKENDS, image="i", dashboard=DASH, launcher=Path("l"), workdir=tmp_path)
    config.skills = ["s2", CANARY, "s1", "s2"]
    github = FakeGitHub()
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    assert rm.select_skills(config, run_info(base_sha=""), tmp_path, api) == ["s2", "s1"]  # config order, deduplicated
    assert github.calls == [] and "the canary is controlled by the trigger" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_render_comment_states_header_progress_and_legend():
    info = run_info()
    plans = [rm.SkillPlan(SKILL, ["c1", "c2"]), rm.SkillPlan("gone", skipped_reason="eval.config not found at `x`")]
    cells = rm.build_cells(plans, BACKENDS)
    cells[0].state, cells[0].status, cells[0].url = rm.CellState.DONE, "PASS", f"{DASH}/runs/a/"
    cells[1].state, cells[1].status, cells[1].url = rm.CellState.CACHED, "FAIL", f"{DASH}/runs/b/"
    cells[2].state, cells[2].status, cells[2].url = rm.CellState.DONE, "PARTIAL", f"{DASH}/runs/c/"
    cells[3].state, cells[3].reason = rm.CellState.ERROR, "no published.json (launcher exit 1) — Traceback: boom"
    cells[4].state = rm.CellState.RUNNING
    body = rm.render_comment(info, plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)

    lines = body.splitlines()
    assert lines[0] == f"<!-- cosmos-skill-eval:pr128:{SHA} -->"  # one comment per PR head commit
    assert lines[1] == f"{rm.OWNER_PREFIX}987|{SHA}|2 -->"  # the run that owns the live body
    assert lines[2] == (  # no dashboard group link in the header any more: each skill heading carries its own
        f"## Skill eval — [{SHA7}](https://github.com/{REPO}/commit/{SHA})"
        f" · run [#987](https://github.com/{REPO}/actions/runs/987/attempts/2)"
    )
    assert "dashboard group" not in body
    assert (  # the FAIL and the PARTIAL are "failed", the launcher error is an "error", the cached FAIL is "reused"
        "4 / 8 cells done · 1 passed · 2 failed · 0 no report · 1 errors · 1 reused"
        " · started 2026-10-18 07:00:00 UTC · updated"
    ) in body
    assert (
        f"### `{SKILL}` · [dashboard overview]({DASH}/skills/{SKILL}/groups/ci-pr128-{SHA7}/)\n"
        "| Case | claude | codex | cursor | opencode |"
    ) in body
    assert f"| `c1` | ✅ PASS · [report]({DASH}/runs/a/) | ❌ FAIL · [report]({DASH}/runs/b/) |" in body
    assert f"⚠️ PARTIAL · [report]({DASH}/runs/c/) | 💥 error (1) |" in body
    assert "| `c2` | 🟢 running | ⏳ queued | ⏳ queued | ⏳ queued |" in body
    assert "1. `opencode · c1` — no published.json (launcher exit 1) — Traceback: boom" in body
    assert "**Not evaluated**\n- `gone` — eval.config not found at `x`" in body
    assert "### `gone`" not in body  # a skill with nothing to run gets no section of its own
    assert "Legend: ⏳ queued · 🟢 running · ✅ PASS · ❌ FAIL · ⚠️ PARTIAL · ❔ NO-REPORT" in body
    assert "⛔ ERROR (" in body and "💥 error (launcher failed" in body and "reused (progress line) = " in body
    assert "· mode " not in body and "Trigger:" not in body  # a default run carries no mode note
    assert "done —" not in body  # not finished yet


def test_render_comment_final_lines_and_per_skill_dashboard_links():
    info = run_info(run_attempt="1")
    plans = [rm.SkillPlan(CANARY, ["c"], canary=True), rm.SkillPlan("s2", ["c"])]
    cells = rm.build_cells(plans, ["claude"])
    for cell in cells:
        cell.state, cell.status, cell.url = rm.CellState.DONE, "PASS", "http://x/"
    body = rm.render_comment(info, plans, cells, ["claude"], started_at=FIXED_NOW, now=FIXED_NOW)
    assert body.rstrip().endswith("✅ done — 2/2 passed (0 failed, 0 no report, 0 errors)")
    assert f"run [#987](https://github.com/{REPO}/actions/runs/987)" in body  # attempt 1: no /attempts suffix
    canary_heading = f"### `{CANARY}` (canary) · [dashboard overview]({DASH}/skills/{CANARY}/groups/ci-pr128-{SHA7}/)"
    other_heading = f"### `s2` · [dashboard overview]({DASH}/skills/s2/groups/ci-pr128-{SHA7}/)"
    assert canary_heading in body and other_heading in body
    assert body.index(canary_heading) < body.index(other_heading)  # canary section first
    assert "Not evaluated" not in body and "Nothing to evaluate" not in body

    cells[1].status = "FAIL"
    body = rm.render_comment(info, plans, cells, ["claude"], started_at=FIXED_NOW, now=FIXED_NOW)
    assert body.rstrip().endswith("❌ done — 1/2 passed (1 failed, 0 no report, 0 errors)")

    cells[1].state, cells[1].status = rm.CellState.QUEUED, None
    body = rm.render_comment(info, plans, cells, ["claude"], started_at=FIXED_NOW, now=FIXED_NOW, interrupted=True)
    assert body.rstrip().endswith("⚠️ interrupted — 1/2 passed · 1 not run")


def test_render_comment_with_nothing_to_evaluate():
    reason = "`.agents/skills/gone` does not exist at this head (deleted or renamed)"
    plans = [rm.SkillPlan("gone", skipped_reason=reason)]
    body = rm.render_comment(run_info(), plans, [], BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert "0 / 0 cells done" in body
    assert "_Nothing to evaluate: this PR touches no skill with an `eval/eval.config`._" in body
    assert f"**Not evaluated**\n- `gone` — {reason}" in body
    assert body.rstrip().endswith("✅ done — nothing to evaluate")


def test_render_comment_stays_under_size_limit():
    info = run_info()
    plans = [rm.SkillPlan(SKILL, [f"case-{index}" for index in range(120)])]
    cells = rm.build_cells(plans, BACKENDS)
    for index, cell in enumerate(cells):
        cell.state, cell.reason = rm.CellState.ERROR, f"{index}-" + "x" * 1000  # distinct even when clipped to 80 chars
    body = rm.render_comment(info, plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert len(body) <= rm.MAX_BODY_CHARS
    assert body.startswith(info.marker)
    assert "💥 error (480)" in body  # the table survives; only the error details are cut


def test_static_score_for_takes_the_newest_scored_run_of_the_group():
    group = f"ci-pr128-{SHA7}"

    def run(run_id: str, created_at: str, group_id: str | None = group, **extra) -> dict:
        return {"run_id": run_id, "created_at": created_at, "group_id": group_id, **extra}

    runs = [
        run("old", "2026-10-17T01:00:00Z", static_score=80),
        run("new", "2026-10-17T03:00:00Z", static_score=95),
        run("other-group", "2026-10-17T04:00:00Z", group_id="ci-pr128-0000000", static_score=100),
        run("no-group", "2026-10-17T04:30:00Z", group_id=None, static_score=100),
        run("newest-unscored", "2026-10-17T05:00:00Z"),  # an older manifest: no static_score at all
        run("newest-null", "2026-10-17T06:00:00Z", static_score=None),
        run("newest-odd", "2026-10-17T07:00:00Z", static_score="95"),
        run("newest-bool", "2026-10-17T08:00:00Z", static_score=True),
        run("undated", "", static_score=70),  # an unparsable created_at sorts oldest
        "not a run",
    ]
    assert rm.static_score_for(runs, group) == 95  # the newest run of the group that carries a usable value
    assert rm.static_score_for(runs, "ci-pr128-0000000") == 100
    assert rm.static_score_for(runs, "ci-pr999-1234567") is None
    assert rm.static_score_for(runs, None) is None and rm.static_score_for(runs, "") is None
    assert rm.static_score_for([], group) is None
    assert rm.static_score_for([run("x", "2026-10-17T01:00:00Z", static_score=None)], group) is None
    assert rm.static_score_for([run("x", "2026-10-17T01:00:00Z", static_score=0)], group) == 0  # 0 is a score

    info = run_info()  # every skill of this run is in the default group ci-pr128-<sha7>
    plans = [rm.SkillPlan(SKILL, ["c"]), rm.SkillPlan("s2", ["c"]), rm.SkillPlan("gone", skipped_reason="x")]
    catalogs = {SKILL: runs, "s2": [run("s2", "2026-10-17T01:00:00Z", static_score=None)]}
    assert rm.static_scores_for(plans, info, catalogs) == {SKILL: 95}  # s2 is unscored, gone has no catalog


def test_render_skill_heading_shows_the_static_score_only_when_known():
    info = run_info()
    plan = rm.SkillPlan(SKILL, ["c"])
    base = f"### `{SKILL}` · [dashboard overview]({DASH}/skills/{SKILL}/groups/ci-pr128-{SHA7}/)"
    assert rm.render_skill_heading(plan, info) == base
    assert rm.render_skill_heading(plan, info, static_score=None) == base
    assert rm.render_skill_heading(plan, info, static_score=95) == f"{base} · static 95/100"
    canary = rm.render_skill_heading(rm.SkillPlan(CANARY, ["c"], canary=True), info, static_score=100)
    assert canary == (
        f"### `{CANARY}` (canary) · [dashboard overview]({DASH}/skills/{CANARY}/groups/ci-pr128-{SHA7}/)"
        " · static 100/100"
    )


def test_render_comment_puts_the_static_score_in_the_skill_heading():
    info = run_info()
    plans = [rm.SkillPlan(CANARY, ["c"], canary=True), rm.SkillPlan(SKILL, ["c"])]
    cells = rm.build_cells(plans, ["claude"])
    scored = rm.render_comment(
        info, plans, cells, ["claude"], started_at=FIXED_NOW, now=FIXED_NOW, static_scores={SKILL: 95, CANARY: None}
    )
    assert (
        f"### `{SKILL}` · [dashboard overview]({DASH}/skills/{SKILL}/groups/ci-pr128-{SHA7}/) · static 95/100\n"
        in scored
    )
    assert f"### `{CANARY}` (canary) · [dashboard overview]({DASH}/skills/{CANARY}/groups/ci-pr128-{SHA7}/)\n" in scored
    plain = rm.render_comment(info, plans, cells, ["claude"], started_at=FIXED_NOW, now=FIXED_NOW)
    assert "· static " not in plain and scored.replace(" · static 95/100", "") == plain


# ---------------------------------------------------------------------------
# GitHub client and the comment updater
# ---------------------------------------------------------------------------


def test_github_api_retries_on_5xx_and_connection_errors():
    github = FakeGitHub()
    github.scripted = [503, rm.TransportError("connection reset")]
    sleeps: list[float] = []
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github, sleep=sleeps.append)
    assert api.get_pull(128)["title"] == "Smoke test"
    assert len(github.calls) == 3 and sleeps == [1.0, 2.0]
    assert api.api_base == "https://api.github.com"

    github.scripted = [500, 502, 504]
    with pytest.raises(rm.GitHubApiError):
        api.get_pull(128)

    before = len(github.calls)
    with pytest.raises(rm.GitHubApiError) as exc_info:
        api.get_pull(999)  # 404: no retry
    assert exc_info.value.status == 404 and len(github.calls) == before + 1
    assert rm.GitHubApi("https://ghe.example.com/", REPO, "t").api_base == "https://ghe.example.com/api/v3"


def test_comment_updater_reuses_marker_comment_across_pages():
    marker = run_info().marker
    filler = [{"id": index, "body": f"unrelated {index}"} for index in range(100)]
    github = FakeGitHub(comments=filler + [{"id": 777, "body": marker + "\nold body"}])
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    updater = rm.GitHubCommentUpdater(api, 128, marker, min_interval_s=0, clock=FakeClock(), start_timer=FakeTimers())
    updater.publish(marker + "\nnew body")
    assert updater.comment_id == 777 and github.count("POST") == 0
    assert github.comments[-1]["body"] == marker + "\nnew body"
    assert github.count("GET") == 2  # two pages of comments


def test_comment_updater_creates_comment_when_marker_absent_and_then_patches():
    marker = run_info().marker
    github = FakeGitHub(comments=[{"id": 1, "body": "unrelated"}])
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    updater = rm.GitHubCommentUpdater(api, 128, marker, min_interval_s=0, clock=FakeClock(), start_timer=FakeTimers())
    updater.publish("v1")
    updater.publish("v2", final=True)
    assert updater.comment_id == 500
    assert github.count("POST") == 1 and github.bodies("PATCH") == ["v2"]
    assert github.comments[-1] == {"id": 500, "body": "v2"}


def test_comment_updater_coalesces_updates_within_the_window():
    marker = run_info().marker
    github = FakeGitHub()
    clock, timers = FakeClock(), FakeTimers()
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    updater = rm.GitHubCommentUpdater(api, 128, marker, min_interval_s=10, clock=clock, start_timer=timers)
    updater.publish("A")  # first send: creates the comment
    clock.now = 3.0
    updater.publish("B")  # too early: parked, timer armed for the rest of the window
    clock.now = 5.0
    updater.publish("C")  # replaces B
    assert github.bodies("PATCH") == [] and len(timers.timers) == 1 and timers.timers[0].delay == 7.0
    clock.now = 10.0
    timers.fire()
    assert github.bodies("PATCH") == ["C"]
    clock.now = 12.0
    updater.publish("D")  # parked again
    updater.publish("E", final=True)  # final bypasses the limit and drops D
    timers.fire()
    assert github.bodies("PATCH") == ["C", "E"]
    clock.now = 30.0
    updater.publish("F")  # window elapsed: sent immediately
    assert github.bodies("PATCH") == ["C", "E", "F"]


def test_comment_updater_never_raises_on_api_failure():
    marker = run_info().marker
    github = FakeGitHub()
    github.scripted = [500, 500, 500]
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github, sleep=lambda seconds: None)
    updater = rm.GitHubCommentUpdater(api, 128, marker, min_interval_s=0, clock=FakeClock(), start_timer=FakeTimers())
    updater.publish("v1")
    assert updater.comment_id is None and updater.failures == 1
    updater.publish("v2")  # recovers on the next event
    assert updater.comment_id == 500 and github.comments[-1]["body"] == "v2"


# ---------------------------------------------------------------------------
# launcher command, outcome parsing
# ---------------------------------------------------------------------------


def test_build_launcher_command_matches_contract(tmp_path: Path):
    env_file = tmp_path / "ci.env"
    config = rm.Config(
        backends=BACKENDS,
        image="img:v1",
        dashboard=DASH,
        launcher=Path("/opt/launch"),
        workdir=tmp_path,
        env_file=env_file,
        codex_model="openai/openai/gpt-6-astra",
    )
    checkout = tmp_path / "co"
    out = tmp_path / "out"
    canary = rm.build_launcher_command(config, checkout, out, rm.Cell(CANARY, "smoke-hello", "claude", canary=True))
    assert canary[canary.index("eval") + 1] == "/workspace/skill-source/.github/scripts/skill_eval/cosmos3-ci-smoke"
    claude = rm.build_launcher_command(config, checkout, out, rm.Cell(SKILL, "hello", "claude"))
    assert claude == [
        "/opt/launch",
        "--image",
        "img:v1",
        "--skill-src",
        str(checkout),
        "--output-dir",
        str(out),
        "-e",
        f"COSMOS_EVAL_SRC_HOST={checkout}",
        "--github-env",
        "eval",
        f"/workspace/skill-source/.agents/skills/{SKILL}",
        "--backend",
        "claude",
        "--case",
        "hello",
    ]
    env_file.write_text("X=1\n")
    codex = rm.build_launcher_command(config, checkout, out, rm.Cell(SKILL, "hello", "codex"))
    assert codex[1:3] == ["--env-file", str(env_file)]
    assert "COSMOS_EVAL_CODEX_MODEL=openai/openai/gpt-6-astra" in codex
    assert "COSMOS_EVAL_CODEX_MODEL=openai/openai/gpt-6-astra" not in claude


def test_build_launcher_env_passes_the_job_environment_through_minus_the_github_token(tmp_path: Path):
    """The Secrets contract (README): NVIDIA_API_KEY — and any other job variable — reaches the launcher; GH_TOKEN never."""
    info = run_info()
    base = ci_env(NVIDIA_API_KEY="k", ARBITRARY_JOB_VAR="kept")
    env = rm.build_launcher_env(base, info, {"user": {"login": "octocat"}}, tmp_path / "event.json")
    assert env["NVIDIA_API_KEY"] == "k" and env["ARBITRARY_JOB_VAR"] == "kept"
    assert "GH_TOKEN" not in env
    assert env["GITHUB_ACTOR"] == "octocat" and env["GITHUB_EVENT_PATH"] == str(tmp_path / "event.json")
    cell_env = rm.launcher_env_for_cell(env, info, rm.Cell(SKILL, "hello", "claude"))
    assert cell_env["NVIDIA_API_KEY"] == "k" and "GH_TOKEN" not in cell_env
    assert cell_env["COSMOS_EVAL_GROUP_ID"] == info.group_for(SKILL)


def test_key_provenance_reads_the_launcher_line_and_nothing_else():
    text = "\x1b[33m[cosmos-skill-eval] NVIDIA_API_KEY: from host environment\x1b[0m\n[cosmos-eval] status: PASS\n"
    assert rm.key_provenance(text) == "[cosmos-skill-eval] NVIDIA_API_KEY: from host environment"
    assert rm.key_provenance("[cosmos-eval] status: PASS\n") is None
    assert rm.key_provenance("") is None


def test_parse_status_and_last_meaningful_line():
    text = "[cosmos-eval] case a: FAIL\n\x1b[32m[cosmos-eval] status: PASS\x1b[0m\r\n\n   \n"
    assert rm.parse_status("status: FAIL\nlater status: PASS") == "PASS"
    assert rm.parse_status('"status": "PASS"') is None
    assert rm.last_meaningful_line(text) == "[cosmos-eval] status: PASS"
    assert rm.last_meaningful_line("\n\n") == "(empty log)"


def test_evaluate_outcome_timeout_and_missing_publish(tmp_path: Path):
    cell = rm.Cell(SKILL, "hello", "claude")
    log_path = tmp_path / "cell.log"
    log_path.write_text("still running\n")
    rm.evaluate_outcome(cell, tmp_path / "out", log_path, None, DASH)
    assert cell.state is rm.CellState.ERROR and cell.reason is not None
    assert cell.reason.startswith("timed out after 3 h — still running")

    cell = rm.Cell(SKILL, "hello", "claude")
    rm.evaluate_outcome(cell, tmp_path / "out", log_path, 1, DASH)
    assert cell.reason == "no published.json (launcher exit 1) — still running"

    published = tmp_path / "out" / SKILL / "published.json"
    published.parent.mkdir(parents=True)
    published.write_text(json.dumps({"status": "ok", "run_id": "r1"}))
    log_path.write_text("[cosmos-eval] status: PARTIAL\n")
    cell = rm.Cell(SKILL, "hello", "claude")
    rm.evaluate_outcome(cell, tmp_path / "out", log_path, 2, DASH)
    assert (cell.state, cell.status, cell.url) == (rm.CellState.DONE, "PARTIAL", f"{DASH}/runs/r1/")


# ---------------------------------------------------------------------------
# end to end through main()
# ---------------------------------------------------------------------------


def test_main_runs_matrix_and_keeps_one_comment_updated(tmp_path: Path, checkout: Path, launcher: Path):
    env_file = tmp_path / "ci.env"
    env_file.write_text("NVIDIA_API_KEY=x\n")
    config = write_config(tmp_path, launcher, env_file=env_file)  # no `skills:` -> detected from the PR diff
    github = FakeGitHub()
    github.compare_files = [  # unsorted; one deleted skill, one touched skill without an eval.config, noise
        {"filename": f".agents/skills/{SKILL}/SKILL.md", "status": "modified"},
        {"filename": ".agents/skills/cosmos3-inference/SKILL.md", "status": "modified"},
        {"filename": ".agents/skills/cosmos3-gone/eval/eval.config", "status": "removed"},
        {"filename": "docs/setup.md", "status": "modified"},
        {
            "filename": ".github/scripts/skill_eval/cosmos3-ci-smoke/eval/eval.config",
            "status": "added",
        },  # canary touched: logged only — the canary runs because the trigger says `with-smoke`
    ]
    dashboard = FakeDashboard(
        {
            SKILL: [
                catalog_run("cached-run", SHA, "hello", "cursor", "PASS", "2026-10-17T01:00:00Z"),
                catalog_run("stale-run", OTHER_SHA, "hello", "claude", "FAIL", "2026-10-17T02:00:00Z"),
            ]
        }
    )
    deps = make_deps(ci_env(), github, dashboard)

    args = ["--config", str(config), "--checkout", str(checkout), "--trigger-args", "with-smoke"]
    assert rm.main(args, deps) == 0

    # the changed files came from ONE compare page (fewer than 100 files)
    assert github.urls("GET", "/compare/") == [
        f"https://api.github.com/repos/{REPO}/compare/{BASE_SHA}...{SHA}?per_page=100&page=1"
    ]
    # one comment, created once, then patched on every event (interval 0 in tests)
    assert github.count("POST") == 1
    first, final = github.bodies("POST")[0], github.comments[-1]["body"]
    assert first.startswith(f"<!-- cosmos-skill-eval:pr128:{SHA} -->\n")
    first_tables = first.split("\nLegend:")[0]  # the legend line repeats the state tokens
    assert first_tables.count("⏳ queued") == 11 and f"✅ PASS · [report]({DASH}/runs/cached-run/)" in first
    assert github.count("PATCH") == 2 * 11 + 1  # start + end per executed cell, plus the final update
    # the canary section comes first and ran first; every heading carries its own dashboard group link
    canary_heading = f"### `{CANARY}` (canary) · [dashboard overview]({DASH}/skills/{CANARY}/groups/ci-pr128-{SHA7}/)"
    skill_heading = f"### `{SKILL}` · [dashboard overview]({DASH}/skills/{SKILL}/groups/ci-pr128-{SHA7}/)"
    assert final.index(canary_heading) < final.index(skill_heading)
    assert "| `smoke-hello` | 🟢 running | ⏳ queued | ⏳ queued | ⏳ queued |" in github.bodies("PATCH")[0]
    assert final.rstrip().endswith("❌ done — 9/12 passed (3 failed, 0 no report, 0 errors)")
    assert "12 / 12 cells done · 9 passed · 3 failed · 0 no report · 0 errors · 1 reused" in final
    assert (
        f"| `hello` | ✅ PASS · [report]({DASH}/runs/r-claude-hello/) | ❌ FAIL · [report]({DASH}/runs/r-codex-hello/) |"
        in final
    )
    assert f"✅ PASS · [report]({DASH}/runs/cached-run/)" in final
    assert (
        "**Not evaluated**\n"
        "- `cosmos3-gone` — `.agents/skills/cosmos3-gone` does not exist at this head (deleted or renamed)\n"
        "- `cosmos3-inference` — eval.config not found at `.agents/skills/cosmos3-inference/eval/eval.config`"
    ) in final
    assert f"run [#987](https://github.com/{REPO}/actions/runs/987/attempts/2)" in final

    # the cached cell never ran; the others got the exact launcher command line (the canary from its CI-only path)
    workdir = tmp_path / "work"
    assert not (workdir / SHA7 / f"{SKILL}-cursor-hello.argv").exists()
    canary_argv = read_argv(workdir, f"{CANARY}-claude-smoke-hello")
    assert (
        canary_argv[canary_argv.index("eval") + 1]
        == "/workspace/skill-source/.github/scripts/skill_eval/cosmos3-ci-smoke"
    )
    claude_argv = read_argv(workdir, f"{SKILL}-claude-hello")
    assert claude_argv == [
        "--env-file",
        str(env_file),
        "--image",
        "registry.example.com/cosmos-skill-eval:v1",
        "--skill-src",
        str(checkout.resolve()),
        "--output-dir",
        str(workdir / SHA7 / f"{SKILL}-claude-hello"),
        "-e",
        f"COSMOS_EVAL_SRC_HOST={checkout.resolve()}",
        "-e",
        f"COSMOS_EVAL_SKILL_CONTENT_SHA={rm.skill_content_sha256(checkout / '.agents' / 'skills' / SKILL)}",
        "--github-env",
        "eval",
        f"/workspace/skill-source/.agents/skills/{SKILL}",
        "--backend",
        "claude",
        "--case",
        "hello",
    ]
    codex_argv = read_argv(workdir, f"{SKILL}-codex-world")
    assert "COSMOS_EVAL_CODEX_MODEL=openai/openai/gpt-6-astra" in codex_argv
    assert "COSMOS_EVAL_CODEX_MODEL=openai/openai/gpt-6-astra" not in claude_argv

    # provenance env + synthetic event file for the publisher inside the container
    env = read_env(workdir, f"{SKILL}-opencode-world")
    event_path = Path(env["GITHUB_EVENT_PATH"])
    assert event_path == workdir / f"event-{SHA7}.json"
    event = json.loads(event_path.read_text())
    assert event["action"] == "synchronize" and event["number"] == 128
    assert event["pull_request"]["title"] == "Smoke test" and event["repository"]["full_name"] == REPO
    assert env["COSMOS_EVAL_GROUP_ID"] == f"ci-pr128-{SHA7}"  # no earlier run of this PR: a new group
    assert env["COSMOS_EVAL_SKILL_CONTENT_SHA"] == rm.skill_content_sha256(checkout / ".agents" / "skills" / SKILL)
    canary_env = read_env(workdir, f"{CANARY}-claude-smoke-hello")
    assert canary_env["COSMOS_EVAL_SKILL_CONTENT_SHA"] == rm.skill_content_sha256(checkout / rm.CANARY_DIR)
    assert env["GITHUB_SHA"] == SHA and env["GITHUB_REF"] == "refs/pull/128/merge"
    assert env["GITHUB_HEAD_REF"] == "skill-eval-config" and env["GITHUB_BASE_REF"] == "main"
    assert env["GITHUB_ACTOR"] == "octocat"
    assert "GITHUB_TOKEN" not in env and "GH_TOKEN" not in env  # the write-scoped job token never reaches containers
    assert env["GITHUB_RUN_ID"] == "987" and env["GITHUB_RUN_ATTEMPT"] == "2"
    assert env["GITHUB_ACTIONS"] == "true" and env["GITHUB_EVENT_NAME"] == "pull_request"
    assert env["GITHUB_WORKFLOW"] == "skill-eval" and env["GITHUB_JOB"] == "run"
    assert "status: PASS" in (workdir / SHA7 / f"{SKILL}-claude-hello.log").read_text()


def test_main_resumes_by_reusing_comment_and_cache(tmp_path: Path, checkout: Path, launcher: Path):
    config = write_config(tmp_path, launcher, smoke=False)
    marker = run_info().marker
    github = FakeGitHub(comments=[{"id": 42, "body": marker + "\nprevious attempt"}])
    runs = [
        catalog_run(f"r-{backend}-{case}", SHA, case, backend, "PASS", "2026-10-17T01:00:00Z")
        for case in ("hello", "world")
        for backend in BACKENDS
        if (case, backend) != ("world", "opencode")
    ]
    deps = make_deps(ci_env(), github, FakeDashboard({SKILL: runs}))

    assert rm.main(["--config", str(config), "--checkout", str(checkout)], deps) == 0
    assert github.count("POST") == 0 and all(
        url.endswith("/comments/42") for method, url, _ in github.calls if method == "PATCH"
    )
    executed = sorted(path.name for path in (tmp_path / "work" / SHA7).glob("*.argv"))
    assert executed == [f"{SKILL}-opencode-world.argv"]
    assert github.comments[0]["body"].rstrip().endswith("✅ done — 8/8 passed (0 failed, 0 no report, 0 errors)")
    assert "8 / 8 cells done · 8 passed · 0 failed · 0 no report · 0 errors · 7 reused" in github.comments[0]["body"]
    assert "(canary)" not in github.comments[0]["body"]  # smoke: false keeps the canary out


def test_main_reports_launcher_errors_without_failing_the_job(tmp_path: Path, launcher: Path):
    checkout = tmp_path / "checkout"
    write_eval_config(checkout, SKILL, ["explode", "unpublished"])
    config = write_config(tmp_path, launcher, smoke=False)
    github = FakeGitHub()
    deps = make_deps(ci_env(), github, FakeDashboard())

    assert rm.main(["--config", str(config), "--checkout", str(checkout)], deps) == 0
    final = github.comments[-1]["body"]
    assert final.split("\nLegend:")[0].count("💥 error") == 8 and "**Errors**" in final
    assert (  # identical reasons share one numbered entry listing every affected cell
        "1. `claude · explode`, `codex · explode`, `cursor · explode`, `opencode · explode`"
        " — no published.json (launcher exit 1) — Traceback: boom goes the launcher"
    ) in final
    assert (
        "2. `claude · unpublished`, `codex · unpublished`, `cursor · unpublished`, `opencode · unpublished`"
        " — publish S3 down — [cosmos-eval] status: PASS"
    ) in final
    assert final.rstrip().endswith("❌ done — 0/8 passed (0 failed, 0 no report, 8 errors)")


def test_main_dry_run_prints_plan_without_touching_github(tmp_path: Path, checkout: Path, launcher: Path, capsys):
    config = write_config(tmp_path, launcher)
    github = FakeGitHub()
    github.compare_files.append({"filename": ".agents/skills/cosmos3-inference/SKILL.md", "status": "modified"})
    github.compare_files.append(
        {"filename": ".github/scripts/skill_eval/cosmos3-ci-smoke/eval/eval.config", "status": "added"}
    )  # touches the canary — which no longer puts it into a default run
    dashboard = FakeDashboard({SKILL: [catalog_run("r1", SHA, "hello", "claude", "PASS", "2026-10-17T01:00:00Z")]})
    deps = make_deps(ci_env(), github, dashboard)

    args = ["--config", str(config), "--checkout", str(checkout), "--dry-run"]
    assert rm.main(args, deps) == 0  # default run: the skills only; the touched canary is logged, not planned
    out = capsys.readouterr().out
    assert "canary touched: yes (informational)" in out
    assert f"canary {CANARY}: off (not requested: no `with-smoke`, `smoke:` unset)" in out
    assert f"plan: sha {SHA7} · 8 cell(s): 1 reused, 7 to run, 0 not selected" in out
    assert f"  group {CANARY}:" not in out and f"{CANARY}/claude/smoke-hello" not in out

    assert rm.main(args + ["--trigger-args", "with-smoke"], deps) == 0  # with-smoke: the canary first, then the skills
    out = capsys.readouterr().out
    assert f"canary {CANARY}: on (trigger argument `with-smoke`)" in out
    assert f"plan: sha {SHA7} · 12 cell(s): 1 reused, 11 to run, 0 not selected" in out
    assert (
        f"  group {SKILL}: ci-pr128-{SHA7} · content " in out and f"  group {CANARY}: ci-pr128-{SHA7} · content " in out
    )
    assert "  skill cosmos3-inference: not evaluated — eval.config not found at" in out
    assert out.index(f"{CANARY}/claude/smoke-hello: run") < out.index(
        f"{SKILL}/claude/hello: reused PASS {DASH}/runs/r1/"
    )
    assert f"{SKILL}/codex/hello: run" in out
    assert [method for method, _url, _payload in github.calls] == ["GET", "GET"]  # compare lookups only, nothing posted
    assert not (tmp_path / "work").exists()


def test_main_fake_launcher_without_github_detects_skills_with_git_and_prints_bodies(
    tmp_path: Path, git_checkout, capsys
):
    root, base_sha, head_sha = git_checkout
    config = write_config(tmp_path, tmp_path / "no-such-launcher")
    # no GH_TOKEN, no PR_NUMBER, no HEAD_SHA (git HEAD is used); BASE_SHA drives the git fallback of the detection
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp"), "BASE_SHA": base_sha}
    deps = make_deps(env, FakeGitHub(), FakeDashboard(error=OSError("dashboard down")))

    args = ["--config", str(config), "--checkout", str(root), "--fake-launcher", "--no-github"]
    args += ["--trigger-args", "with-smoke"]  # the head adds the canary, which by itself would not make it run
    assert rm.main(args, deps) == 0
    out = capsys.readouterr().out
    assert "----- comment body (final) -----" in out and "WARN: dashboard catalog unavailable" in out
    assert f"canary {CANARY}: on (trigger argument `with-smoke`)" in out
    assert "changed path(s) via git diff" in out and "never a shipped skill" in out  # stray canary copy ignored
    final = out.rsplit("----- comment body (final) -----\n", 1)[1]
    assert f"<!-- cosmos-skill-eval:pr0:{head_sha} -->" in final and f"|{head_sha}|" in final and "ci-pr0-" in final
    assert final.index(f"### `{CANARY}` (canary) · [dashboard overview]") < final.index(
        "### `cosmos3-accent` · [dashboard overview]"
    )
    assert final.index("### `cosmos3-accent` · [dashboard overview]") < final.index(
        f"### `{SKILL}` · [dashboard overview]"
    )
    assert f"| `smoke-hello` | ✅ PASS · [report]({DASH}/runs/fake-{CANARY}-claude-smoke-hello/) |" in final
    assert f"| `hello` | ✅ PASS · [report]({DASH}/runs/fake-{SKILL}-claude-hello/) |" in final
    assert f"❌ FAIL · [report]({DASH}/runs/fake-{SKILL}-codex-hello/)" in final
    assert (
        "**Not evaluated**\n"
        "- `cosmos3-gone` — `.agents/skills/cosmos3-gone` does not exist at this head (deleted or renamed)\n"
        "- `cosmos3-inference` — eval.config not found at `.agents/skills/cosmos3-inference/eval/eval.config`"
    ) in final
    assert "cosmos3-untouched" not in final
    assert "❌ done — 8/16 passed (8 failed, 0 no report, 0 errors)" in final  # canary 1 + accent 1 + SKILL 2, x4
    assert len(list((tmp_path / "work").glob("event-*.json"))) == 1


def test_main_explicit_skills_override_skips_detection(tmp_path: Path, checkout: Path, launcher: Path, capsys):
    config = write_config(tmp_path, launcher, skills=(SKILL, CANARY, "cosmos3-inference"), smoke=False)
    github = FakeGitHub()
    github.compare_files = None  # the compare API would fail and the checkout is not a git repo: never consulted
    deps = make_deps(ci_env(BASE_SHA=""), github, FakeDashboard())  # and BASE_SHA is not needed either

    args = ["--config", str(config), "--checkout", str(checkout), "--trigger-args", "with-smoke"]
    assert rm.main(args, deps) == 0
    out = capsys.readouterr().out
    assert "PR diff detection skipped" in out and "the canary is controlled by the trigger" in out
    assert f"canary {CANARY}: off (`smoke: false` in the config wins over the trigger argument `with-smoke`)" in out
    assert github.urls("GET", "/compare/") == []
    final = github.comments[-1]["body"]
    assert "(canary)" not in final  # `skills:` cannot smuggle the canary in, and `smoke: false` beats `with-smoke`
    assert (
        f"### `{SKILL}` · [dashboard overview]" in final and "- `cosmos3-inference` — eval.config not found at" in final
    )
    assert final.rstrip().endswith("❌ done — 6/8 passed (2 failed, 0 no report, 0 errors)")


def test_main_exit_codes_for_orchestration_errors(tmp_path: Path, checkout: Path, launcher: Path, capsys):
    github = FakeGitHub()
    dashboard = FakeDashboard()
    missing_config = ["--config", str(tmp_path / "nope.yml"), "--checkout", str(checkout)]
    assert rm.main(missing_config, make_deps(ci_env(), github, dashboard)) == 2

    config = write_config(tmp_path, launcher)
    args = ["--config", str(config), "--checkout", str(checkout)]
    assert rm.main(args, make_deps(ci_env(PR_NUMBER=""), github, dashboard)) == 2
    assert rm.main(args, make_deps(ci_env(GH_TOKEN=""), github, dashboard)) == 2
    assert rm.main(args, make_deps(ci_env(PR_NUMBER="999"), github, dashboard)) == 2  # PR fetch 404
    assert (
        rm.main(
            ["--config", str(config), "--checkout", str(tmp_path / "missing")], make_deps(ci_env(), github, dashboard)
        )
        == 2
    )

    # skill detection needs BASE_SHA (unless `skills:` overrides it) ...
    assert rm.main(args, make_deps(ci_env(BASE_SHA=""), github, dashboard)) == 2
    assert "ERROR: environment variable BASE_SHA" in capsys.readouterr().out
    # ... and a way to list the PR's files: here the compare API 404s and the checkout is not a git repository
    github.compare_files = None
    assert rm.main(args, make_deps(ci_env(), github, dashboard)) == 2
    assert "ERROR: cannot list the files changed by the PR" in capsys.readouterr().out
    github.compare_files = FakeGitHub().compare_files

    broken = write_config(tmp_path, tmp_path / "no-such-launcher")
    assert rm.main(["--config", str(broken), "--checkout", str(checkout)], make_deps(ci_env(), github, dashboard)) == 2
    assert github.count("POST") == 0


# ---------------------------------------------------------------------------
# robustness: torn responses, at-most-once comment creation, stale outputs, timeouts
# ---------------------------------------------------------------------------


def test_urllib_transport_maps_http_client_errors_to_transport_error(monkeypatch):
    def torn(request, timeout):
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(rm.urllib.request, "urlopen", torn)
    with pytest.raises(rm.TransportError, match="IncompleteRead"):
        rm.urllib_transport("GET", "https://api.github.com/x", {}, None, 1.0)


def test_comment_updater_swallows_unexpected_transport_exceptions():
    def exploding(method, url, headers, body, timeout):
        raise RuntimeError("torn response")

    api = rm.GitHubApi("https://github.com", REPO, "t", transport=exploding, sleep=lambda seconds: None)
    marker = run_info().marker
    updater = rm.GitHubCommentUpdater(api, 128, marker, min_interval_s=0, clock=FakeClock(), start_timer=FakeTimers())
    updater.publish("v1")  # must not raise
    assert updater.failures == 1 and updater.comment_id is None


def test_comment_updater_does_not_duplicate_comment_when_post_response_is_lost():
    marker = run_info().marker
    github = FakeGitHub()
    github.lose_post_response = True
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github, sleep=lambda seconds: None)
    updater = rm.GitHubCommentUpdater(api, 128, marker, min_interval_s=0, clock=FakeClock(), start_timer=FakeTimers())
    updater.publish(marker + "\nv1")
    assert updater.comment_id == 500 and updater.failures == 0
    assert github.count("POST") == 1 and len(github.comments) == 1  # found by marker on the re-list, not re-created
    assert github.comments[0]["body"] == marker + "\nv1"


def test_last_meaningful_line_skips_orchestrator_notes():
    text = "[cosmos-eval] pulling image\nError: no space left on device\n\n[run-matrix] launcher timed out after 60s\n"
    assert rm.last_meaningful_line(text) == "Error: no space left on device"


def test_clear_stale_outputs_moves_previous_attempt_aside(tmp_path: Path):
    output_dir, log_path = tmp_path / "cell", tmp_path / "cell.log"
    published = output_dir / SKILL / "published.json"
    published.parent.mkdir(parents=True)
    published.write_text('{"status": "ok"}')
    log_path.write_text("[cosmos-eval] status: PASS\n")
    rm.clear_stale_outputs(output_dir, SKILL, log_path)
    assert not published.exists() and not log_path.exists()
    assert published.with_name("published.json.prev").exists() and log_path.with_name("cell.log.prev").exists()
    rm.clear_stale_outputs(output_dir, SKILL, log_path)  # nothing left to move: no error


def test_clear_stale_outputs_sets_a_root_owned_skill_dir_aside(tmp_path: Path):
    """The container writes <cell>/<skill>/ as root: renaming published.json inside it is denied, so the
    whole previous <skill>/ output is moved aside at the user-owned cell level (2026-10-26, smoke run #2)."""
    import os
    if os.geteuid() == 0:
        pytest.skip("root can rename anything; the fallback is not reachable")
    output_dir, log_path = tmp_path / "cell", tmp_path / "cell.log"
    skill_dir = output_dir / SKILL
    (skill_dir / "smoke-hello").mkdir(parents=True)
    (skill_dir / "published.json").write_text('{"status": "ok"}')
    log_path.write_text("[cosmos-eval] status: PASS\n")
    skill_dir.chmod(0o555)  # read-only directory stands in for root ownership
    try:
        rm.clear_stale_outputs(output_dir, SKILL, log_path)
        assert not skill_dir.exists() and (output_dir / f"{SKILL}.prev" / "published.json").exists()
        assert log_path.with_name("cell.log.prev").exists() and not log_path.exists()
        # a third attempt finds `.prev` taken and uses `.prev2`
        (skill_dir / "smoke-hello").mkdir(parents=True)
        (skill_dir / "published.json").write_text('{"status": "ok"}')
        skill_dir.chmod(0o555)
        rm.clear_stale_outputs(output_dir, SKILL, log_path)
        assert (output_dir / f"{SKILL}.prev2" / "published.json").exists()
    finally:
        for d in output_dir.glob(f"{SKILL}*"):
            d.chmod(0o755)


def test_main_rerun_does_not_report_stale_results_for_a_cell_that_dies_early(tmp_path: Path, launcher: Path):
    checkout = tmp_path / "checkout"
    write_eval_config(checkout, SKILL, ["explode"])
    config = write_config(tmp_path, launcher, smoke=False)
    cell_dir = tmp_path / "work" / SHA7 / f"{SKILL}-claude-explode"
    (cell_dir / SKILL).mkdir(parents=True)
    stale = {"status": "ok", "url": f"{DASH}/runs/old/", "run_id": "old"}
    (cell_dir / SKILL / "published.json").write_text(json.dumps(stale))
    (tmp_path / "work" / SHA7 / f"{SKILL}-claude-explode.log").write_text("[cosmos-eval] status: PASS\n")
    github = FakeGitHub()

    args = ["--config", str(config), "--checkout", str(checkout)]
    assert rm.main(args, make_deps(ci_env(), github, FakeDashboard())) == 0
    final = github.comments[-1]["body"]
    assert "runs/old/" not in final
    assert (
        "1. `claude · explode`, `codex · explode`, `cursor · explode`, `opencode · explode`"
        " — no published.json (launcher exit 1) — Traceback: boom goes the launcher"
    ) in final


def test_run_launcher_timeout_terminates_the_child(tmp_path: Path):
    script = tmp_path / "slow"
    script.write_text("#!/usr/bin/env bash\necho started\nexec sleep 30\n")
    script.chmod(0o755)
    log_path = tmp_path / "slow.log"
    started = time.monotonic()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    assert rm.run_launcher([str(script)], env, log_path, 0.5) is None
    assert time.monotonic() - started < 10  # SIGTERM stopped `sleep 30`; no 30 s wait, no SIGKILL escalation
    assert "launcher timed out" in log_path.read_text()


def test_main_survives_an_exception_inside_a_cell(tmp_path: Path, checkout: Path, launcher: Path, monkeypatch):
    config = write_config(tmp_path, launcher, smoke=False)
    real_evaluate = rm.evaluate_outcome

    def flaky(cell, *args, **kwargs):
        if (cell.case, cell.backend) == ("hello", "codex"):
            raise RuntimeError("disk full")
        return real_evaluate(cell, *args, **kwargs)

    monkeypatch.setattr(rm, "evaluate_outcome", flaky)
    github = FakeGitHub()
    args = ["--config", str(config), "--checkout", str(checkout)]
    assert rm.main(args, make_deps(ci_env(), github, FakeDashboard())) == 0
    final = github.comments[-1]["body"]
    assert "`codex · hello` — orchestrator error: RuntimeError: disk full" in final
    # codex cells FAIL in the fake launcher, so one orchestrator error on hello/codex leaves 6 passes
    assert "8 / 8 cells done" in final
    assert final.rstrip().endswith("❌ done — 6/8 passed (1 failed, 0 no report, 1 errors)")


def test_canary_touched_only_for_files_inside_its_directory():
    assert rm.canary_touched([".github/scripts/skill_eval/cosmos3-ci-smoke/eval/eval.config"])
    assert rm.canary_touched([".github/scripts/skill_eval/cosmos3-ci-smoke/SKILL.md", "README.md"])
    assert not rm.canary_touched([".github/scripts/skill_eval/run_matrix.py", f".agents/skills/{SKILL}/SKILL.md"])
    assert not rm.canary_touched([".github/scripts/skill_eval/cosmos3-ci-smoke"])  # the directory entry itself
    assert not rm.canary_touched([])


def test_select_plan_runs_the_canary_only_on_request_or_by_config(
    tmp_path: Path, launcher: Path, checkout: Path, capsys
):
    info = run_info()
    github = FakeGitHub()  # default PR diff: only SKILL/SKILL.md modified
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    cfg_auto = rm.load_config(write_config(tmp_path, launcher))  # `smoke:` unset
    assert rm.select_plan(cfg_auto, info, checkout, api) == ([SKILL], False)  # default: off
    github.compare_files.append(
        {"filename": ".github/scripts/skill_eval/cosmos3-ci-smoke/SKILL.md", "status": "modified"}
    )
    assert rm.select_plan(cfg_auto, info, checkout, api) == ([SKILL], False)  # touching the canary changes nothing…
    out = capsys.readouterr().out
    assert "canary touched: yes (informational)" in out  # …but it is logged
    assert f"canary {CANARY}: off (not requested: no `with-smoke`, `smoke:` unset)" in out
    assert rm.select_plan(cfg_auto, info, checkout, api, with_smoke=True) == ([SKILL], True)  # `with-smoke`
    assert f"canary {CANARY}: on (trigger argument `with-smoke`)" in capsys.readouterr().out
    cfg_off = rm.load_config(write_config(tmp_path, launcher, smoke=False))
    assert rm.select_plan(cfg_off, info, checkout, api, with_smoke=True)[1] is False  # the config beats the trigger
    assert f"canary {CANARY}: off (`smoke: false` in the config wins over the trigger argument `with-smoke`)" in (
        capsys.readouterr().out
    )
    github.compare_files.pop()
    cfg_on = rm.load_config(write_config(tmp_path, launcher, smoke=True))
    assert rm.select_plan(cfg_on, info, checkout, api)[1] is True  # `smoke: true`: on without any request
    assert f"canary {CANARY}: on (`smoke: true` in the config)" in capsys.readouterr().out
    cfg_override = rm.load_config(write_config(tmp_path, launcher, skills=(SKILL,)))
    assert rm.select_plan(cfg_override, info, checkout, api) == ([SKILL], False)  # explicit list: still off…
    assert rm.select_plan(cfg_override, info, checkout, api, with_smoke=True) == ([SKILL], True)  # …unless asked


# ---------------------------------------------------------------------------
# runner health gate: docker check / repair / retry / abort, grouped errors
# ---------------------------------------------------------------------------


def test_load_config_docker_check_and_repair_are_optional_commands(tmp_path: Path, launcher: Path):
    config = rm.load_config(write_config(tmp_path, launcher))
    assert config.docker_check is None and config.docker_repair is None
    path = write_config(
        tmp_path, launcher, docker_check="docker info", docker_repair="sudo -n chgrp docker /run/docker.sock"
    )
    config = rm.load_config(path)
    assert config.docker_check == ["docker", "info"]
    assert config.docker_repair == ["sudo", "-n", "chgrp", "docker", "/run/docker.sock"]
    path.write_text(path.read_text() + 'docker_check: ""\n')
    with pytest.raises(rm.OrchestrationError, match="docker_check"):
        rm.load_config(path)


def test_preflight_repairs_once_and_reports_failures(tmp_path: Path, capsys):
    marker = tmp_path / "socket-ok"
    check = write_script(tmp_path / "check", f'[ -e "{marker}" ] || {{ echo "{SOCKET_DENIED}" >&2; exit 1; }}\n')
    repair = write_script(tmp_path / "repair", f'touch "{marker}"\n')
    pre = rm.Preflight([str(check)], [str(repair)])
    assert pre.ensure() and pre.repairs == 1  # check failed, the repair fixed it, the re-check passed
    assert pre.ensure() and pre.repairs == 1  # healthy: nothing to repair
    marker.unlink()
    pre = rm.Preflight([str(check)], None)
    assert not pre.ensure() and "permission denied" in pre.last_error
    assert rm.Preflight(None, None).ensure()  # no gate configured
    assert rm.Preflight(["/nonexistent/docker"], None).ensure() is False
    captured = capsys.readouterr()
    assert "docker check failed" in captured.out + captured.err
    assert "docker check passes after the repair" in captured.out + captured.err


def test_docker_failure_signatures():
    assert rm.docker_failure(f"docker: {SOCKET_DENIED}\n")
    assert rm.docker_failure("Cannot connect to the Docker daemon at unix:///var/run/docker.sock")
    assert rm.docker_failure("\x1b[31mGot permission denied while trying to connect to the Docker daemon socket\x1b[0m")
    assert not rm.docker_failure("Traceback: boom goes the launcher")


def test_main_aborts_remaining_cells_when_the_runner_loses_docker(tmp_path: Path, checkout: Path, launcher: Path):
    check = write_script(tmp_path / "check", f'echo "{SOCKET_DENIED}" >&2; exit 1\n')
    config = write_config(tmp_path, launcher, smoke=False, docker_check=str(check))
    github = FakeGitHub()

    assert (
        rm.main(["--config", str(config), "--checkout", str(checkout)], make_deps(ci_env(), github, FakeDashboard()))
        == 1
    )
    final = github.comments[-1]["body"]
    assert final.split("\nLegend:")[0].count("💥 error (1)") == 8  # one grouped reason for all eight cells
    assert (
        "**Errors**\n1. `claude · hello`, `codex · hello`, `cursor · hello`, `opencode · hello`, `claude · world`"
        in final
    )
    assert f"— not run — runner cannot reach docker — {SOCKET_DENIED}" in final
    assert final.rstrip().endswith(
        f"🛑 aborted — runner cannot reach docker — {SOCKET_DENIED} · 0/8 passed · 8 not run"
    )
    assert not (tmp_path / "work" / SHA7 / f"{SKILL}-claude-hello.argv").exists()  # the launcher never ran
    assert "8 / 8 cells done · 0 passed · 0 failed · 0 no report · 8 errors · 0 reused" in final


def test_main_retries_a_cell_once_after_a_docker_socket_error(tmp_path: Path, launcher: Path, capsys):
    checkout = tmp_path / "checkout"
    write_eval_config(checkout, SKILL, ["sockflip"])
    marker = tmp_path / "socket-ok"
    marker.touch()
    check = write_script(tmp_path / "check", f'[ -e "{marker}" ] || {{ echo "{SOCKET_DENIED}" >&2; exit 1; }}\n')
    repairs = tmp_path / "repairs"
    repair = write_script(tmp_path / "repair", f'echo repaired >> "{repairs}"; touch "{marker}"\n')
    config = write_config(tmp_path, launcher, smoke=False, docker_check=str(check), docker_repair=str(repair))
    github = FakeGitHub()

    assert (
        rm.main(["--config", str(config), "--checkout", str(checkout)], make_deps(ci_env(), github, FakeDashboard()))
        == 0
    )
    final = github.comments[-1]["body"]
    table = final.split("\nLegend:")[0]
    assert table.count("✅ PASS") == 3 and table.count("❌ FAIL") == 1 and "💥" not in table  # codex FAILs by design
    assert "🔧 4 cells retried after a docker error" in final and "docker repaired" not in final
    assert final.rstrip().endswith("❌ done — 3/4 passed (1 failed, 0 no report, 0 errors)")
    assert not repairs.exists()  # the check passed every time, so the repair never ran
    work = tmp_path / "work" / SHA7
    assert (work / f"{SKILL}-claude-sockflip.log.prev").exists()  # first attempt's log kept aside
    assert "status: PASS" in (work / f"{SKILL}-claude-sockflip.log").read_text()
    assert "docker failure on the first attempt" in capsys.readouterr().out

    # the retry path also repairs: flip the marker away so the check fails once before the second attempt
    marker.unlink()
    write_eval_config(checkout, SKILL, ["sockflip"])
    shutil.rmtree(work)
    github = FakeGitHub()
    assert (
        rm.main(["--config", str(config), "--checkout", str(checkout)], make_deps(ci_env(), github, FakeDashboard()))
        == 0
    )
    final = github.comments[-1]["body"]
    # pre-cell check fails -> repair recreates the marker -> check passes -> cells run, each retried once as before
    assert "🔧 docker repaired 1× on the runner · 4 cells retried after a docker error" in final
    assert repairs.read_text().count("repaired") == 1  # repaired once, then healthy for the rest of the matrix
    assert final.rstrip().endswith("❌ done — 3/4 passed (1 failed, 0 no report, 0 errors)")


def test_render_comment_groups_identical_error_reasons():
    plans = [rm.SkillPlan(SKILL, ["c1"])]
    cells = rm.build_cells(plans, BACKENDS)
    for cell in cells[:3]:
        cell.state, cell.reason = rm.CellState.ERROR, "no published.json (launcher exit 1) — permission denied"
    cells[3].state, cells[3].reason = rm.CellState.ERROR, "timed out after 3 h — still running"
    body = rm.render_comment(run_info(), plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert "| `c1` | 💥 error (1) | 💥 error (1) | 💥 error (1) | 💥 error (2) |" in body
    assert (
        "1. `claude · c1`, `codex · c1`, `cursor · c1` — no published.json (launcher exit 1) — permission denied"
        in body
    )
    assert "2. `opencode · c1` — timed out after 3 h — still running" in body
    body = rm.render_comment(
        run_info(),
        plans,
        cells,
        BACKENDS,
        started_at=FIXED_NOW,
        now=FIXED_NOW,
        notes=["docker repaired 2× on the runner"],
    )
    assert "\n🔧 docker repaired 2× on the runner\n\n### `" in body


# ---------------------------------------------------------------------------
# distributed mode: plan -> one job per cell -> report
# ---------------------------------------------------------------------------


def job(cell_label: str, status: str, conclusion: str | None = None, run: str = "987") -> dict:
    return {
        "name": f"{rm.JOB_NAME_PREFIX}{cell_label}",
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run}/job/{abs(hash(cell_label)) % 100000}",
    }


def plan_env(tmp_path: Path, **overrides) -> dict[str, str]:
    env = ci_env(**overrides)
    env["GITHUB_OUTPUT"] = str(tmp_path / "github_output")
    return env


def read_outputs(path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        key, _sep, delimiter = lines[index].partition("<<")
        end = lines.index(delimiter, index + 1)
        outputs[key] = "\n".join(lines[index + 1 : end])
        index = end + 1
    return outputs


def matrix_entry(skill: str, backend: str, case: str) -> dict:
    """A matrix `include` entry as the plan emits it: `runs_on` = the default labels (+ the serial label for cursor)."""
    return {"skill": skill, "backend": backend, "case": case, "runs_on": json.dumps(rm.runs_on(backend))}


def test_plan_mode_posts_the_comment_and_emits_matrices_without_cached_cells(
    tmp_path: Path, checkout: Path, launcher: Path
):
    config = write_config(tmp_path, launcher)
    github = FakeGitHub()
    github.compare_files = [
        {"filename": f".agents/skills/{SKILL}/SKILL.md", "status": "modified"},
        {"filename": ".github/scripts/skill_eval/cosmos3-ci-smoke/eval/eval.config", "status": "added"},  # logged only
    ]
    dashboard = FakeDashboard(
        {SKILL: [catalog_run("cached-run", SHA, "hello", "cursor", "PASS", "2026-10-17T01:00:00Z")]}
    )
    env = plan_env(tmp_path)
    args = ["plan", "--config", str(config), "--checkout", str(checkout), "--plan-out", str(tmp_path / "plan.json")]
    args += ["--trigger-args", "with-smoke"]  # the canary first, then the skills
    assert rm.main(args, make_deps(env, github, dashboard)) == 0

    outputs = read_outputs(tmp_path / "github_output")
    assert outputs["has_canary"] == "true" and outputs["has_skills"] == "true"
    assert (outputs["cells"], outputs["cached"]) == ("12", "1")
    canary, skills = json.loads(outputs["canary"]), json.loads(outputs["skills"])
    assert canary == [matrix_entry(CANARY, b, "smoke-hello") for b in BACKENDS]
    assert len(skills) == 7 and matrix_entry(SKILL, "cursor", "hello") not in skills  # cached: no job
    # every entry carries the job's runner labels; a cursor cell is pinned to the serial runner (README, Runners)
    assert json.loads(canary[0]["runs_on"]) == ["self-hosted", "cosmos-skill-eval", "h100"]
    assert json.loads(canary[2]["runs_on"]) == ["self-hosted", "cosmos-skill-eval", "h100", "cursor-serial"]
    plan = json.loads(outputs["plan"])
    assert plan == json.loads((tmp_path / "plan.json").read_text())
    assert plan["schema"] == rm.PLAN_SCHEMA and plan["scope"] == "full" and plan["info"]["head_sha"] == SHA
    cached = [c for c in plan["cells"] if c["state"] == "cached"]
    assert cached == [
        {
            "skill": SKILL,
            "case": "hello",
            "backend": "cursor",
            "canary": False,
            "state": "cached",
            "status": "PASS",
            "url": f"{DASH}/runs/cached-run/",
            "reason": None,
            "dashboard_run_id": "cached-run",  # the shown result's identity: refresh_from_catalog compares against it
            "created_at": "2026-10-17T01:00:00Z",
            "job": False,  # the plan's decision: reused, not this run's job (True for every queued cell)
        }
    ]
    assert all(c["job"] is True for c in plan["cells"] if c["state"] == "queued")
    # ONE comment, created by the plan job, every runnable cell queued, the seq line counts the cached cell
    assert github.count("POST") == 1 and len(github.comments) == 1
    body = github.comments[0]["body"]
    assert " · canary with-smoke\n" in body  # the header names the canary request, like the other modes
    assert body.splitlines()[:3] == [
        f"<!-- cosmos-skill-eval:pr128:{SHA} -->",
        f"{rm.OWNER_PREFIX}987|{SHA}|2 -->",
        f"{rm.SEQ_PREFIX}2.1 -->",
    ]
    assert "1 / 12 cells done · 1 passed · 0 failed · 0 no report · 0 errors · 1 reused" in body
    assert body.split("\nLegend:")[0].count("⏳ queued") == 11 and "✅ PASS" in body
    # the plan carries every skill's version and group, so cell/report jobs never need git or the catalog for it
    assert plan["trigger"] == {
        "scope": "full",
        "with_smoke": True,
        "mode": "default",
        "force": False,
        "cells": [],
        "group": None,
        "errors": [],
    }
    assert plan["skills"][SKILL]["content_sha256"] == rm.skill_content_sha256(checkout / ".agents" / "skills" / SKILL)
    assert plan["skills"][SKILL]["group_id"] == f"ci-pr128-{SHA7}" and plan["skills"][SKILL]["tree_sha"] is None
    assert plan["skills"][SKILL]["equivalent_commits"] == [SHA] and plan["info"]["group_override"] is None
    assert set(plan["skills"]) == {CANARY, SKILL}


@pytest.mark.parametrize(
    "cli",
    [["--scope", "smoke"], ["--trigger-args", "smoke-only"], ["--trigger-args", "smoke force"]],
    ids=["cli-scope", "smoke-only", "smoke-alias"],
)
def test_plan_mode_smoke_scope_runs_only_the_canary(tmp_path: Path, checkout: Path, launcher: Path, cli: list[str]):
    config = write_config(tmp_path, launcher, smoke=False)  # `smoke: false` never stops a smoke-only run: its point
    github = FakeGitHub()  # the PR touches SKILL, which the smoke scope must ignore
    assert (
        rm.main(
            ["plan", "--config", str(config), "--checkout", str(checkout), *cli],
            make_deps(plan_env(tmp_path), github, FakeDashboard()),
        )
        == 0
    )
    outputs = read_outputs(tmp_path / "github_output")
    assert outputs["has_canary"] == "true" and outputs["has_skills"] == "false" and outputs["skills"] == "[]"
    assert outputs["cells"] == "4" and json.loads(outputs["plan"])["trigger"]["scope"] == "smoke"
    body = github.comments[0]["body"]
    assert f"### `{CANARY}` (canary)" in body and f"### `{SKILL}`" not in body
    assert " · canary smoke-only" in body  # the header names the canary request, like the other modes


def test_plan_mode_config_smoke_true_runs_the_canary_on_a_default_run(
    tmp_path: Path, checkout: Path, launcher: Path, capsys
):
    config = write_config(tmp_path, launcher, smoke=True)  # the standing equivalent of `with-smoke`
    github = FakeGitHub()  # default PR diff: SKILL touched, the canary not
    args = ["plan", "--config", str(config), "--checkout", str(checkout)]
    assert rm.main(args, make_deps(plan_env(tmp_path), github, FakeDashboard())) == 0
    assert f"canary {CANARY}: on (`smoke: true` in the config)" in capsys.readouterr().out
    outputs = read_outputs(tmp_path / "github_output")
    assert outputs["has_canary"] == "true" and outputs["has_skills"] == "true" and outputs["cells"] == "12"
    assert json.loads(outputs["canary"]) == [matrix_entry(CANARY, b, "smoke-hello") for b in BACKENDS]
    body = github.comments[0]["body"]
    assert f"### `{CANARY}` (canary)" in body and f"### `{SKILL}`" in body
    assert "canary with-smoke" not in body  # the config asked, not the trigger: a default-mode header


def _planned(tmp_path: Path, checkout: Path, launcher: Path, github: FakeGitHub, dashboard: FakeDashboard):
    """Run the plan stage and return (config path, plan JSON text)."""
    config = write_config(tmp_path, launcher)
    github.compare_files = [{"filename": f".agents/skills/{SKILL}/SKILL.md", "status": "modified"}]
    assert (
        rm.main(
            ["plan", "--config", str(config), "--checkout", str(checkout), "--plan-out", str(tmp_path / "plan.json")],
            make_deps(plan_env(tmp_path), github, dashboard),
        )
        == 0
    )
    return config, (tmp_path / "plan.json").read_text()


def test_cell_mode_runs_one_cell_and_rebuilds_the_comment_from_jobs_and_dashboard(
    tmp_path: Path, checkout: Path, launcher: Path
):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    # another runner already published codex/hello; claude/world is running elsewhere; the rest is queued
    dashboard.runs_by_skill[SKILL] = [
        catalog_run("r-codex-hello", SHA, "hello", "codex", "FAIL", "2026-10-18T07:10:00Z")
    ]
    dashboard.runs_by_skill[SKILL][0].update(group_id=f"ci-pr128-{SHA7}", static_score=95)  # the version's score
    github.jobs = [
        job(f"{SKILL}/codex/hello", "completed", "success"),
        job(f"{SKILL}/claude/world", "in_progress"),
        job(f"{SKILL}/claude/hello", "in_progress"),
    ]
    posts_before = github.count("PATCH")

    args = [
        "cell",
        "--config",
        str(config),
        "--checkout",
        str(checkout),
        "--plan",
        plan,
        "--skill",
        SKILL,
        "--backend",
        "claude",
        "--case",
        "hello",
    ]
    assert rm.main(args, make_deps(ci_env(), github, dashboard)) == 0
    assert github.count("PATCH") - posts_before == 2 and len(github.comments) == 1  # start + end, same comment
    body = github.comments[0]["body"]
    assert (
        f"| `hello` | ✅ PASS · [report]({DASH}/runs/r-claude-hello/) | ❌ FAIL · [report]({DASH}/runs/r-codex-hello/) | ⏳ queued | ⏳ queued |"
        in body
    )
    assert "| `world` | 🟢 running | ⏳ queued | ⏳ queued | ⏳ queued |" in body
    assert "2 / 8 cells done · 1 passed · 1 failed · 0 no report · 0 errors · 0 reused" in body
    assert f"{rm.SEQ_PREFIX}2.2 -->" in body
    assert f"groups/ci-pr128-{SHA7}/) · static 95/100\n| Case |" in body  # from the catalog the refresh fetched anyway
    assert (tmp_path / "work" / SHA7 / f"{SKILL}-claude-hello" / SKILL / "published.json").exists()
    env = read_env(tmp_path / "work", f"{SKILL}-claude-hello")
    assert env["COSMOS_EVAL_GROUP_ID"] == f"ci-pr128-{SHA7}" and env["GITHUB_RUN_ID"] == "987"

    # a cell whose launcher dies: the job exits 1 so the Actions UI is red; the comment shows the error
    args[-1] = "explode"
    write_eval_config(checkout, SKILL, ["hello", "world", "explode"])
    plan_dict = json.loads(plan)
    plan_dict["cells"].append(
        {"skill": SKILL, "case": "explode", "backend": "claude", "canary": False, "state": "queued"}
    )
    plan_dict["plans"][-1]["cases"].append("explode")
    args[args.index("--plan") + 1] = json.dumps(plan_dict)
    assert rm.main(args, make_deps(ci_env(), github, dashboard)) == 1
    assert (
        "`claude · explode` — no published.json (launcher exit 1) — Traceback: boom goes the launcher"
        in github.comments[0]["body"]
    )


def test_cell_mode_skips_cached_cells_and_rejects_unknown_ones(tmp_path: Path, checkout: Path, launcher: Path, capsys):
    github = FakeGitHub()
    dashboard = FakeDashboard(
        {SKILL: [catalog_run("cached-run", SHA, "hello", "cursor", "PASS", "2026-10-17T01:00:00Z")]}
    )
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    base = ["cell", "--config", str(config), "--checkout", str(checkout), "--plan", plan, "--skill", SKILL]
    assert rm.main(base + ["--backend", "cursor", "--case", "hello"], make_deps(ci_env(), github, dashboard)) == 0
    assert not (tmp_path / "work" / SHA7 / f"{SKILL}-cursor-hello.argv").exists()  # cached: never launched
    assert rm.main(base + ["--backend", "cursor", "--case", "nope"], make_deps(ci_env(), github, dashboard)) == 2
    assert "is not in the plan" in capsys.readouterr().out


def test_reconcile_cells_prefers_dashboard_then_own_outcome_then_job_state():
    info = run_info()
    plans = [rm.SkillPlan(SKILL, ["c"])]
    cells = rm.build_cells(plans, BACKENDS)  # claude, codex, cursor, opencode
    cells[3].state, cells[3].status, cells[3].url = rm.CellState.CACHED, "PASS", "u"
    dashboard = FakeDashboard({SKILL: [catalog_run("r1", SHA, "c", "claude", "PASS", "2026-10-18T07:00:00Z")]})
    jobs = {
        rm.cell_job_name(cells[0]): {"status": "in_progress", "conclusion": None, "html_url": "j0"},  # dashboard wins
        rm.cell_job_name(cells[1]): {"status": "completed", "conclusion": "success", "html_url": "j1"},
        rm.cell_job_name(cells[2]): {"status": "completed", "conclusion": "failure", "html_url": "j2"},
    }
    rm.reconcile_cells(cells, info, dashboard, jobs, final=False, started_at=FIXED_NOW)
    assert (cells[0].state, cells[0].status, cells[0].url) == (rm.CellState.DONE, "PASS", f"{DASH}/runs/r1/")
    assert (cells[1].state, cells[1].reason) == (rm.CellState.RUNNING, rm.PUBLISHING)  # finished, not listed yet
    assert (cells[2].state, cells[2].reason) == (rm.CellState.ERROR, "launcher error (failure) — see [job log](j2)")
    assert cells[3].state is rm.CellState.CACHED
    body = rm.render_comment(info, plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert "🟢 publishing" in body
    # final pass: a finished job without a published result is an error; own outcome beats everything
    own = rm.Cell(SKILL, "c", "codex", state=rm.CellState.DONE, status="PARTIAL", url="mine")
    rm.reconcile_cells(cells, info, dashboard, jobs, final=True, started_at=FIXED_NOW, local={own.name: own})
    assert (cells[1].state, cells[1].status, cells[1].url) == (rm.CellState.DONE, "PARTIAL", "mine")
    rm.reconcile_cells(
        cells,
        info,
        dashboard,
        {rm.cell_job_name(cells[1]): jobs[rm.cell_job_name(cells[1])]},
        final=True,
        started_at=FIXED_NOW,
    )
    assert cells[1].reason == "finished without a published result — see [job log](j1)"
    rm.reconcile_cells(cells, info, dashboard, {}, final=False, started_at=FIXED_NOW)
    assert cells[1].state is rm.CellState.QUEUED and cells[2].state is rm.CellState.QUEUED  # no job yet -> queued
    jobs[rm.cell_job_name(cells[2])]["conclusion"] = "cancelled"
    rm.reconcile_cells(cells, info, dashboard, jobs, final=True, started_at=FIXED_NOW)
    assert cells[2].reason == "cancelled — see [job log](j2)"
    # a result of the version published BEFORE the plan started is not this run's: the cache decided about it
    stale = FakeDashboard({SKILL: [catalog_run("r0", SHA, "c", "cursor", "PASS", "2026-10-18T06:59:00Z")]})
    rm.reconcile_cells(cells, info, stale, {}, final=False, started_at=FIXED_NOW)
    assert cells[2].state is rm.CellState.QUEUED
    mine = catalog_run("r0", SHA, "c", "cursor", "NO-REPORT", "2026-10-18T06:59:00Z")
    mine["source"] = {"kind": "github-ci", "github": {"run_id": "987"}}  # ... unless THIS run published it
    rm.reconcile_cells(cells, info, FakeDashboard({SKILL: [mine]}), {}, final=False, started_at=FIXED_NOW)
    assert (cells[2].state, cells[2].status) == (rm.CellState.DONE, "NO-REPORT")


def test_publish_distributed_merges_the_live_ledger_and_never_regresses(tmp_path: Path):
    info = run_info()
    plans = [rm.SkillPlan(SKILL, ["c"])]
    github = FakeGitHub()
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    view_a = rm.build_cells(plans, BACKENDS)  # job A's view: only A (claude) finished
    view_a[0].state, view_a[0].status, view_a[0].url = rm.CellState.DONE, "PASS", "http://x/a/"
    assert rm.publish_distributed(api, info, plans, view_a, BACKENDS, FIXED_NOW, FIXED_NOW, final=False)
    body = github.comments[0]["body"]
    assert f"{rm.SEQ_PREFIX}2.1 -->" in body and rm.parse_ledger(body) == {f"{SKILL}/claude/c": ("PASS", "http://x/a/")}

    # job B read the comment before A's PATCH landed: its view has B finished and A still queued (equal seq).
    # The re-read right before the PATCH merges A from the live ledger, so neither result is lost.
    view_b = rm.build_cells(plans, BACKENDS)
    view_b[1].state, view_b[1].status, view_b[1].url = rm.CellState.DONE, "FAIL", "http://x/b/"
    assert rm.publish_distributed(api, info, plans, view_b, BACKENDS, FIXED_NOW, FIXED_NOW, final=False)
    body = github.comments[0]["body"]
    assert f"{rm.SEQ_PREFIX}2.2 -->" in body
    assert rm.parse_ledger(body) == {
        f"{SKILL}/claude/c": ("PASS", "http://x/a/"),
        f"{SKILL}/codex/c": ("FAIL", "http://x/b/"),
    }
    assert view_b[0].state is rm.CellState.DONE and view_b[0].status == "PASS"  # merged into the caller's view too

    # the reviewer's reproduction: a delayed job with an older snapshot (only A) publishes after 2.2 was posted
    stale = rm.build_cells(plans, BACKENDS)
    stale[0].state, stale[0].status, stale[0].url = rm.CellState.DONE, "PASS", "http://x/a/"
    assert rm.publish_distributed(api, info, plans, stale, BACKENDS, FIXED_NOW, FIXED_NOW, final=False)
    body = github.comments[0]["body"]
    assert f"{rm.SEQ_PREFIX}2.2 -->" in body and "codex/c|FAIL|http://x/b/" in body  # B's entry survives

    # a view that would show fewer finished cells than the live comment even after the merge is dropped:
    # the live comment counts an ERROR cell (errors carry no ledger line, so the merge cannot restore it)
    with_error = rm.build_cells(plans, BACKENDS)
    with_error[2].state, with_error[2].reason = rm.CellState.ERROR, "launcher exit 1"
    assert rm.publish_distributed(api, info, plans, with_error, BACKENDS, FIXED_NOW, FIXED_NOW, final=False)
    assert f"{rm.SEQ_PREFIX}2.3 -->" in github.comments[0]["body"]
    older = rm.build_cells(plans, BACKENDS)
    assert rm.publish_distributed(api, info, plans, older, BACKENDS, FIXED_NOW, FIXED_NOW, final=False) is False
    assert f"{rm.SEQ_PREFIX}2.3 -->" in github.comments[0]["body"]
    # ... unless it is the final report, which always writes
    assert rm.publish_distributed(api, info, plans, older, BACKENDS, FIXED_NOW, FIXED_NOW, final=True)
    assert f"{rm.SEQ_PREFIX}2.2 -->" in github.comments[0]["body"]  # the two ledgered cells were merged back in
    assert len(github.comments) == 1  # still one comment
    # a later attempt ("Re-run failed jobs") outranks any seq of the previous attempt, even its final one
    rerun = run_info(run_attempt="3")
    assert rm.publish_distributed(
        api, rerun, plans, rm.build_cells(plans, BACKENDS), BACKENDS, FIXED_NOW, FIXED_NOW, final=False
    )
    assert f"{rm.SEQ_PREFIX}3.2 -->" in github.comments[0]["body"]
    assert rm.parse_seq("<!-- cosmos-skill-eval-seq:7 -->") == (1, 7)  # pre-attempt format
    assert rm.parse_seq("<!-- cosmos-skill-eval-seq:3.12 -->") == (3, 12) and rm.parse_seq("nothing") is None


def test_report_mode_finalises_the_comment(tmp_path: Path, checkout: Path, launcher: Path):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    dashboard.runs_by_skill[SKILL] = [
        catalog_run(f"r-{b}-{c}", SHA, c, b, "PASS", "2026-10-18T07:10:00Z")
        for c in ("hello", "world")
        for b in ("claude", "codex", "cursor")
    ]
    github.jobs = [
        job(f"{SKILL}/opencode/hello", "completed", "failure"),
        job(f"{SKILL}/opencode/world", "completed", "success"),
    ]
    assert (
        rm.main(
            ["report", "--config", str(config), "--checkout", str(checkout), "--plan", plan],
            make_deps(ci_env(), github, dashboard),
        )
        == 0
    )
    body = github.comments[0]["body"]
    assert "8 / 8 cells done · 6 passed · 0 failed · 0 no report · 2 errors · 0 reused" in body
    assert "`opencode · hello` — launcher error (failure) — see [job log](" in body
    assert "`opencode · world` — finished without a published result — see [job log](" in body
    assert (
        body.rstrip().endswith("❌ done — 6/8 passed (0 failed, 0 no report, 2 errors)") and len(github.comments) == 1
    )


def test_list_run_jobs_paginates():
    github = FakeGitHub()
    github.jobs = [job(f"{SKILL}/claude/case-{i}", "queued") for i in range(130)]
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    assert len(api.list_run_jobs("987")) == 130
    assert len(github.urls("GET", "/actions/runs/987/jobs")) == 2
    assert len(rm.fetch_run_jobs(api, run_info())) == 130


def test_plan_roundtrip_and_load_plan_errors(tmp_path: Path):
    info = run_info()
    plans = [
        rm.SkillPlan(CANARY, ["s"], canary=True),
        rm.SkillPlan("gone", skipped_reason="no eval"),
        rm.SkillPlan(SKILL, ["c"]),
    ]
    cells = rm.build_cells(plans, BACKENDS)
    cells[1].state, cells[1].status, cells[1].url = rm.CellState.CACHED, "FAIL", "u"
    cells[1].run_id, cells[1].created_at, cells[1].job = "u-run", "2026-10-18T06:00:00Z", False  # the plan decided
    cells[2].state = cells[3].state = rm.CellState.SKIPPED  # outside `cells=`, no result (one listed makes it CACHED)
    version = rm.SkillVersion("c" * 64, "t" * 40, [SHA, OTHER_SHA], f"ci-pr128-{OTHER_SHA[:7]}")
    info = run_info(versions={SKILL: version})
    options = rm.TriggerOptions(
        with_smoke=True, mode="rerun-failed", cells=["codex/*"], errors=["unknown argument `bogus`"]
    )
    data = rm.plan_to_dict(info, plans, cells, BACKENDS, FIXED_NOW, options)
    info2, plans2, cells2, backends2, started2, options2 = rm.plan_from_dict(json.loads(json.dumps(data)))
    assert (info2, backends2, started2, options2) == (info, BACKENDS, FIXED_NOW, options)
    assert data["scope"] == "full" and data["skills"] == {SKILL: version.to_dict()}
    assert data["trigger"]["with_smoke"] is True and options2.with_smoke  # the canary request travels with the plan
    assert info2.group_for(SKILL) == f"ci-pr128-{OTHER_SHA[:7]}" and info2.group_for(CANARY) == f"ci-pr128-{SHA7}"
    assert info2.version_for(CANARY) == rm.SkillVersion.of_commit(SHA)  # no block: the exact commit, as before
    # a plan written before trigger arguments and skill versions existed still loads (scope only)
    legacy = {key: value for key, value in data.items() if key not in ("trigger", "skills")}
    legacy["scope"] = "smoke"
    legacy["cells"] = [{key: value for key, value in raw.items() if key != "job"} for raw in data["cells"]]
    info3, _plans3, cells3, _backends3, _started3, options3 = rm.plan_from_dict(json.loads(json.dumps(legacy)))
    assert options3 == rm.TriggerOptions(scope="smoke") and info3.versions == {}
    assert all(c.job is None for c in cells3) and [c.is_job for c in cells3[:4]] == [True, False, False, False]
    assert [(p.name, p.cases, p.skipped_reason, p.canary) for p in plans2] == [
        (p.name, p.cases, p.skipped_reason, p.canary) for p in plans
    ]
    assert [(c.skill, c.case, c.backend, c.canary, c.state, c.status, c.url) for c in cells2] == [
        (c.skill, c.case, c.backend, c.canary, c.state, c.status, c.url) for c in cells
    ]
    assert (cells2[1].run_id, cells2[1].created_at) == ("u-run", "2026-10-18T06:00:00Z")  # the result's identity too
    # the plan's decision travels as recorded (None = not decided: `is_job` then reads the state)
    assert [c.job for c in cells2] == [None, False, None, None, None, None, None, None]
    assert [c.is_job for c in cells2] == [True, False, False, False, True, True, True, True]
    # the reused (codex) and the two not-selected canary cells are not jobs; the four skill cells all are
    assert rm.matrix_entries(cells2, canary=True) == [matrix_entry(CANARY, "claude", "s")]
    assert len(rm.matrix_entries(cells2, canary=False)) == 4
    with pytest.raises(rm.OrchestrationError, match="schema"):
        rm.plan_from_dict({"schema": "other"})
    with pytest.raises(rm.OrchestrationError, match="not found"):
        rm.load_plan(str(tmp_path / "missing.json"))
    with pytest.raises(rm.OrchestrationError, match="not valid JSON"):
        rm.load_plan("{not json")


class LateDashboard(FakeDashboard):
    """Lists `late_runs` for `skill` only from the Nth catalog GET on: models the ~2-minute re-list lag."""

    def __init__(self, skill: str, late_runs: list[dict], after_calls: int):
        super().__init__({skill: []})
        self.skill, self.late_runs, self.after_calls = skill, late_runs, after_calls

    def __call__(self, request, timeout=None):
        if len(self.urls) + 1 >= self.after_calls:
            self.runs_by_skill[self.skill] = self.late_runs
        return super().__call__(request, timeout)


def test_report_mode_waits_for_the_dashboard_to_list_finished_cells(tmp_path: Path, checkout: Path, launcher: Path):
    github = FakeGitHub()
    config, plan = _planned(tmp_path, checkout, launcher, github, FakeDashboard({SKILL: []}))
    results = [
        catalog_run(f"r-{b}-{c}", SHA, c, b, "PASS", "2026-10-18T07:10:00Z")
        for c in ("hello", "world")
        for b in BACKENDS
    ]
    github.jobs = [job(f"{SKILL}/{b}/{c}", "completed", "success") for c in ("hello", "world") for b in BACKENDS]
    late = LateDashboard(SKILL, results, after_calls=4)  # listed only from the 4th catalog GET
    sleeps: list[float] = []
    deps = make_deps(ci_env(), github, late)
    deps.sleep = sleeps.append
    assert rm.main(["report", "--config", str(config), "--checkout", str(checkout), "--plan", plan], deps) == 0
    body = github.comments[0]["body"]
    assert "8 / 8 cells done · 8 passed · 0 failed" in body
    assert body.rstrip().endswith("✅ done — 8/8 passed (0 failed, 0 no report, 0 errors)")
    assert "finished without a published result" not in body
    assert sleeps and set(sleeps) == {rm.CATALOG_SETTLE_POLL_S}  # waited between catalog polls instead of guessing


def test_report_mode_exits_1_when_the_jobs_api_is_down(tmp_path: Path, checkout: Path, launcher: Path):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    dashboard.runs_by_skill[SKILL] = [catalog_run("r1", SHA, "hello", "claude", "PASS", "2026-10-18T07:10:00Z")]
    github.fail_jobs_api = True
    assert (
        rm.main(
            ["report", "--config", str(config), "--checkout", str(checkout), "--plan", plan],
            make_deps(ci_env(), github, dashboard),
        )
        == 1
    )
    body = github.comments[0]["body"]  # the comment is still written with what is known
    assert "1 / 8 cells done" in body and body.split("\nLegend:")[0].count("⏳ queued") == 7


def test_plan_mode_fails_when_the_comment_cannot_be_posted(tmp_path: Path, checkout: Path, launcher: Path, capsys):
    config = write_config(tmp_path, launcher)
    github = FakeGitHub()
    github.scripted = [503] * 40  # GitHub is down for every call the plan makes (smoke scope: no detection calls)
    assert (
        rm.main(
            ["plan", "--config", str(config), "--checkout", str(checkout), "--scope", "smoke"],
            make_deps(plan_env(tmp_path), github, FakeDashboard()),
        )
        == 2
    )
    assert "not starting any cell" in capsys.readouterr().out and github.comments == []


def test_plan_mode_rejects_matrices_over_the_github_cap(
    tmp_path: Path, checkout: Path, launcher: Path, monkeypatch, capsys
):
    monkeypatch.setattr(rm, "MATRIX_MAX_JOBS", 3)
    config = write_config(tmp_path, launcher)
    assert (
        rm.main(
            ["plan", "--config", str(config), "--checkout", str(checkout)],
            make_deps(plan_env(tmp_path), FakeGitHub(), FakeDashboard()),
        )
        == 2
    )
    assert "exceed GitHub's limit of 3 jobs per matrix" in capsys.readouterr().out


def test_cell_mode_holds_the_cell_while_docker_is_down_then_fails(
    tmp_path: Path, checkout: Path, launcher: Path, capsys
):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    counter = tmp_path / "checks"
    check = write_script(tmp_path / "check", f'echo x >> "{counter}"; echo "{SOCKET_DENIED}" >&2; exit 1\n')
    config = write_config(tmp_path, launcher, docker_check=str(check))  # same path the plan used
    config.write_text(config.read_text() + "docker_wait_minutes: 3\n")
    args = [
        "cell",
        "--config",
        str(config),
        "--checkout",
        str(checkout),
        "--plan",
        plan,
        "--skill",
        SKILL,
        "--backend",
        "claude",
        "--case",
        "hello",
    ]
    sleeps: list[float] = []
    deps = make_deps(ci_env(), github, dashboard)
    deps.sleep = sleeps.append
    assert rm.main(args, deps) == 1
    assert counter.read_text().count("x") == 4 and sleeps == [rm.DOCKER_WAIT_POLL_S] * 3  # 3 min / 60 s = 3 holds
    assert "holding the cell and re-checking" in capsys.readouterr().out
    assert f"`claude · hello` — runner cannot reach docker — {SOCKET_DENIED}" in github.comments[0]["body"]
    assert not (tmp_path / "work" / SHA7 / f"{SKILL}-claude-hello.argv").exists()  # never launched

    # the stub launcher needs no docker: the same config passes with --fake-launcher
    assert rm.main(args + ["--fake-launcher"], make_deps(ci_env(), github, dashboard)) == 0
    assert counter.read_text().count("x") == 4  # the check was not even run


def test_load_config_docker_wait_minutes(tmp_path: Path, launcher: Path):
    path = write_config(tmp_path, launcher)
    assert rm.load_config(path).docker_wait_minutes == 30
    path.write_text(path.read_text() + "docker_wait_minutes: 5\n")
    assert rm.load_config(path).docker_wait_minutes == 5
    path.write_text(path.read_text() + "docker_wait_minutes: soon\n")
    with pytest.raises(rm.OrchestrationError, match="docker_wait_minutes"):
        rm.load_config(path)


def test_ledger_keeps_peer_results_when_the_dashboard_has_not_listed_them(
    tmp_path: Path, checkout: Path, launcher: Path
):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})  # the dashboard never lists anything in this test
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    # a peer job finished codex/hello and wrote its result into the comment's ledger; its job is done
    peer = rm.Cell(SKILL, "hello", "codex", state=rm.CellState.DONE, status="FAIL", url=f"{DASH}/runs/r-codex-hello/")
    body = github.comments[0]["body"].replace(
        f"{rm.SEQ_PREFIX}2.0 -->", f"{rm.SEQ_PREFIX}2.1 -->\n{rm.ledger_line(peer, '987')}"
    )
    github.comments[0]["body"] = body
    github.jobs = [job(f"{SKILL}/codex/hello", "completed", "success"), job(f"{SKILL}/claude/hello", "in_progress")]
    assert rm.parse_ledger(body) == {f"{SKILL}/codex/hello": ("FAIL", f"{DASH}/runs/r-codex-hello/")}

    args = ["cell", "--config", str(config), "--checkout", str(checkout), "--plan", plan]
    args += ["--skill", SKILL, "--backend", "claude", "--case", "hello"]
    assert rm.main(args, make_deps(ci_env(), github, dashboard)) == 0
    final = github.comments[0]["body"]
    # claude's own PASS and codex's ledgered FAIL both survive, nothing shows as "publishing"
    assert (
        f"| `hello` | ✅ PASS · [report]({DASH}/runs/r-claude-hello/) | ❌ FAIL · [report]({DASH}/runs/r-codex-hello/) |"
        in final
    )
    assert "publishing" not in final.split("\nLegend:")[0]
    assert rm.parse_ledger(final) == {
        f"{SKILL}/claude/hello": ("PASS", f"{DASH}/runs/r-claude-hello/"),
        f"{SKILL}/codex/hello": ("FAIL", f"{DASH}/runs/r-codex-hello/"),
    }
    # the report trusts the ledger too: no false "finished without a published result", no waiting
    github.jobs = [job(f"{SKILL}/{b}/{c}", "completed", "success") for c in ("hello", "world") for b in BACKENDS]
    for c in ("hello", "world"):
        for b in BACKENDS:
            if (c, b) in (("hello", "claude"), ("hello", "codex")):
                continue
            done = rm.Cell(SKILL, c, b, state=rm.CellState.DONE, status="PASS", url="u")
            github.comments[0]["body"] += "\n" + rm.ledger_line(done, "987")
    sleeps: list[float] = []
    deps = make_deps(ci_env(), github, dashboard)
    deps.sleep = sleeps.append
    assert rm.main(["report", "--config", str(config), "--checkout", str(checkout), "--plan", plan], deps) == 0
    assert github.comments[0]["body"].rstrip().endswith("❌ done — 7/8 passed (1 failed, 0 no report, 0 errors)")
    assert sleeps == []


def test_cell_mode_writes_its_result_file(tmp_path: Path, checkout: Path, launcher: Path):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    results = tmp_path / "cell-results"
    args = ["cell", "--config", str(config), "--checkout", str(checkout), "--plan", plan, "--results-dir", str(results)]
    assert (
        rm.main(
            args + ["--skill", SKILL, "--backend", "codex", "--case", "hello"], make_deps(ci_env(), github, dashboard)
        )
        == 0
    )
    data = json.loads((results / f"{SKILL}-codex-hello.json").read_text())
    assert data["schema"] == rm.RESULT_SCHEMA and data["label"] == f"{SKILL}/codex/hello"
    assert (data["state"], data["status"], data["url"]) == ("done", "FAIL", f"{DASH}/runs/r-codex-hello/")
    assert (data["head_sha"], data["run_id"], data["run_attempt"]) == (SHA, "987", "2")
    # a launcher error leaves a file too, with the reason, so the report can show it without the dashboard
    write_eval_config(checkout, SKILL, ["hello", "world", "explode"])
    plan_dict = json.loads(plan)
    plan_dict["cells"].append(
        {"skill": SKILL, "case": "explode", "backend": "claude", "canary": False, "state": "queued"}
    )
    args[args.index("--plan") + 1] = json.dumps(plan_dict)
    assert (
        rm.main(
            args + ["--skill", SKILL, "--backend", "claude", "--case", "explode"],
            make_deps(ci_env(), github, dashboard),
        )
        == 1
    )
    data = json.loads((results / f"{SKILL}-claude-explode.json").read_text())
    assert data["state"] == "error" and "Traceback: boom goes the launcher" in data["reason"]


def test_cell_mode_forwards_the_secret_to_the_launcher_and_echoes_its_provenance_line(
    tmp_path: Path, checkout: Path, capsys
):
    # a launcher that, like the real one, prints the key's provenance line to stderr before anything else
    provenance = (
        'if [ -n "${NVIDIA_API_KEY:-}" ]; then echo "[cosmos-skill-eval] NVIDIA_API_KEY: from host environment" >&2; '
        'else echo "[cosmos-skill-eval] NVIDIA_API_KEY: NOT PROVIDED (agents and the judge need it)" >&2; fi\n'
    )
    launcher = write_script(tmp_path / "provenance-launcher", provenance + FAKE_LAUNCHER.split("\n", 1)[1])
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    args = ["cell", "--config", str(config), "--checkout", str(checkout), "--plan", plan]
    args += ["--skill", SKILL, "--backend", "claude", "--case", "hello"]
    assert rm.main(args, make_deps(ci_env(NVIDIA_API_KEY="sekret"), github, dashboard)) == 0
    env = read_env(tmp_path / "work", f"{SKILL}-claude-hello")
    assert env["NVIDIA_API_KEY"] == "sekret" and "GH_TOKEN" not in env
    out = capsys.readouterr().out
    assert f"{SKILL}/claude/hello: [cosmos-skill-eval] NVIDIA_API_KEY: from host environment" in out
    assert "sekret" not in out
    # without the variable the echo still tells the truth (the runner's env file would be the only source)
    capsys.readouterr()
    args[args.index("--case") + 1] = "world"
    assert rm.main(args, make_deps(ci_env(), github, dashboard)) == 0
    assert f"{SKILL}/claude/world: [cosmos-skill-eval] NVIDIA_API_KEY: NOT PROVIDED" in capsys.readouterr().out


def test_report_mode_takes_the_cell_result_files_as_the_first_source(tmp_path: Path, checkout: Path, launcher: Path):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})  # the dashboard lists nothing
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    info = run_info()
    results = tmp_path / "cell-results"
    cells = rm.build_cells([rm.SkillPlan(SKILL, ["hello", "world"])], BACKENDS)
    for cell in cells:
        if cell.case == "hello":
            cell.state, cell.status, cell.url = rm.CellState.DONE, "PASS", f"{DASH}/runs/r-{cell.backend}-hello/"
        elif cell.backend == "claude":
            cell.state, cell.reason = rm.CellState.ERROR, "no published.json (launcher exit 1) — disk full"
        else:
            continue  # no file for the other three `world` cells
        rm.write_cell_result(results, info, cell)
    # a file from another sha must be ignored
    foreign = rm.Cell(SKILL, "world", "codex", state=rm.CellState.DONE, status="PASS", url="u")
    rm.write_cell_result(results, run_info(head_sha=OTHER_SHA), foreign)
    loaded = rm.load_cell_results(results, cells, info)
    assert sorted(loaded) == sorted(c.name for c in cells if c.case == "hello" or c.backend == "claude")
    github.jobs = [
        job(f"{SKILL}/{b}/{c}", "completed", "success" if c == "hello" or b != "claude" else "failure")
        for c in ("hello", "world")
        for b in BACKENDS
    ]
    sleeps: list[float] = []
    deps = make_deps(ci_env(), github, dashboard)
    deps.sleep = sleeps.append
    args = [
        "report",
        "--config",
        str(config),
        "--checkout",
        str(checkout),
        "--plan",
        plan,
        "--results-dir",
        str(results),
    ]
    assert rm.main(args, deps) == 0
    body = github.comments[0]["body"]
    assert (
        f"| `hello` | ✅ PASS · [report]({DASH}/runs/r-claude-hello/) | ✅ PASS · [report]({DASH}/runs/r-codex-hello/) |"
        in body
    )
    assert (
        "`claude · world` — no published.json (launcher exit 1) — disk full" in body
    )  # the file's reason, not a job link
    # the three `world` cells without a file: job success but nothing published anywhere -> error with the job link
    assert body.count("finished without a published result — see [job log](") == 3
    assert "8 / 8 cells done · 4 passed · 0 failed · 0 no report · 4 errors" in body
    assert body.rstrip().endswith("❌ done — 4/8 passed (0 failed, 0 no report, 4 errors)")
    assert sleeps and set(sleeps) == {
        rm.CATALOG_SETTLE_POLL_S
    }  # it still waited for the dashboard for the unrecorded ones


def test_report_mode_adds_the_static_score_from_the_dashboard(tmp_path: Path, checkout: Path, launcher: Path):
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    info = run_info()
    results = tmp_path / "cell-results"
    cells = rm.build_cells([rm.SkillPlan(SKILL, ["hello", "world"])], BACKENDS)
    for cell in cells:  # every cell job brought its result file: the reconcile never needs the catalog
        cell.state, cell.status, cell.url = rm.CellState.DONE, "PASS", f"{DASH}/runs/r-{cell.backend}-{cell.case}/"
        rm.write_cell_result(results, info, cell)
    github.jobs = [job(f"{SKILL}/{b}/{c}", "completed", "success") for c in ("hello", "world") for b in BACKENDS]
    newest = catalog_run("r-claude-hello", SHA, "hello", "claude", "PASS", "2026-10-18T07:10:00Z")
    newest.update(group_id=f"ci-pr128-{SHA7}", static_score=95)
    older = catalog_run("r-old", SHA, "hello", "claude", "PASS", "2026-10-18T06:00:00Z")
    older.update(group_id=f"ci-pr128-{SHA7}", static_score=100)  # an earlier run of the group: the newest wins
    dashboard.runs_by_skill[SKILL] = [older, newest]
    dashboard.urls.clear()
    args = ["report", "--config", str(config), "--checkout", str(checkout), "--plan", plan, "--results-dir"]
    args.append(str(results))
    assert rm.main(args, make_deps(ci_env(), github, dashboard)) == 0
    body = github.comments[0]["body"]
    heading = f"### `{SKILL}` · [dashboard overview]({DASH}/skills/{SKILL}/groups/ci-pr128-{SHA7}/) · static 95/100"
    assert heading in body and body.rstrip().endswith("✅ done — 8/8 passed (0 failed, 0 no report, 0 errors)")
    assert dashboard.urls == [f"{DASH}/api/catalog?skill={SKILL}"]  # one GET for the suffix, nothing else

    # a dashboard that is down costs the suffix, not the report
    github = FakeGitHub()
    github.jobs = [job(f"{SKILL}/{b}/{c}", "completed", "success") for c in ("hello", "world") for b in BACKENDS]
    broken = FakeDashboard(error=urllib.error.URLError("down"))
    assert rm.main(args, make_deps(ci_env(), github, broken)) == 0
    body = github.comments[0]["body"]
    assert "· static " not in body and body.rstrip().endswith("✅ done — 8/8 passed (0 failed, 0 no report, 0 errors)")
    assert len(broken.urls) == rm.CATALOG_RETRIES  # the one lookup, retried like any catalog GET


# ---------------------------------------------------------------------------
# status taxonomy v2, skill versions and groups, trigger arguments, re-run modes
# ---------------------------------------------------------------------------


def test_new_statuses_in_status_line_ledger_and_comment():
    assert rm.KNOWN_STATUSES == ("PASS", "FAIL", "PARTIAL", "NO-REPORT", "ERROR")
    assert rm.parse_status("[cosmos-eval] status: NO-REPORT\n") == "NO-REPORT"
    assert rm.parse_status("[cosmos-eval] status: FAIL\n[cosmos-eval] status: ERROR\n") == "ERROR"
    assert rm.parse_status("[cosmos-eval] status: ERRORS\n") is None  # word boundary
    no_report = rm.Cell(SKILL, "c", "codex", state=rm.CellState.DONE, status="NO-REPORT", url="u1")
    error = rm.Cell(SKILL, "c", "cursor", state=rm.CellState.DONE, status="ERROR", url="u2")
    ledger = rm.parse_ledger(rm.ledger_line(no_report) + "\n" + rm.ledger_line(error))
    assert ledger == {f"{SKILL}/codex/c": ("NO-REPORT", "u1"), f"{SKILL}/cursor/c": ("ERROR", "u2")}
    assert rm.parse_ledger(f"{rm.LEDGER_PREFIX}x/y/z|BOGUS|u -->") == {}

    plans = [rm.SkillPlan(SKILL, ["c1", "c2"])]
    cells = rm.build_cells(plans, BACKENDS)  # c1: claude, codex, cursor, opencode; c2: the same
    cells[0].state, cells[0].status, cells[0].url = rm.CellState.DONE, "PASS", "u0"
    cells[1].state, cells[1].status, cells[1].url = rm.CellState.DONE, "NO-REPORT", "u1"
    cells[2].state, cells[2].status, cells[2].url = rm.CellState.DONE, "ERROR", "u2"
    cells[3].state, cells[3].reason = rm.CellState.ERROR, "launcher exit 1"
    cells[4].state, cells[4].status, cells[4].url = rm.CellState.CACHED, "FAIL", "u4"
    cells[5].state, cells[5].status, cells[5].url = rm.CellState.DONE, "PARTIAL", "u5"
    cells[7].state = rm.CellState.RUNNING
    body = rm.render_comment(run_info(), plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert (
        "| `c1` | ✅ PASS · [report](u0) | ❔ NO-REPORT · [report](u1) | ⛔ ERROR · [report](u2) | 💥 error (1) |"
        in body
    )
    assert "| `c2` | ❌ FAIL · [report](u4) | ⚠️ PARTIAL · [report](u5) | ⏳ queued | 🟢 running |" in body
    # the four outcome buckets partition the finished cells: FAIL + PARTIAL = failed, ERROR + launcher error = errors
    assert "6 / 8 cells done · 1 passed · 2 failed · 1 no report · 2 errors · 1 reused" in body
    assert "done —" not in body
    assert not cells[1].passed and not cells[2].passed and cells[0].passed
    for cell in (cells[6], cells[7]):
        cell.state, cell.status, cell.url = rm.CellState.DONE, "PASS", "u"
    body = rm.render_comment(run_info(), plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert body.rstrip().endswith("❌ done — 3/8 passed (2 failed, 1 no report, 2 errors)")


def test_main_run_mode_publishes_new_statuses_and_counts_them(tmp_path: Path, launcher: Path):
    checkout = tmp_path / "checkout"
    write_eval_config(checkout, SKILL, ["noreport", "gateway", "hello"])
    config = write_config(tmp_path, launcher, smoke=False)
    github = FakeGitHub()
    deps = make_deps(ci_env(), github, FakeDashboard())
    assert rm.main(["--config", str(config), "--checkout", str(checkout)], deps) == 0
    final = github.comments[-1]["body"]
    assert f"| `noreport` | ❔ NO-REPORT · [report]({DASH}/runs/r-claude-noreport/) |" in final
    assert f"| `gateway` | ⛔ ERROR · [report]({DASH}/runs/r-claude-gateway/) |" in final
    assert "💥" not in final.split("\nLegend:")[0]  # a published NO-REPORT / ERROR is not a launcher error
    assert "12 / 12 cells done · 3 passed · 1 failed · 4 no report · 4 errors · 0 reused" in final
    assert final.rstrip().endswith("❌ done — 3/12 passed (1 failed, 4 no report, 4 errors)")


def test_parse_trigger_args_tokens_aliases_and_errors():
    assert rm.parse_trigger_args("") == rm.TriggerOptions() == rm.parse_trigger_args(None)
    assert rm.TriggerOptions().cacheable == rm.CACHEABLE_DEFAULT == frozenset({"PASS", "FAIL"})
    options = rm.parse_trigger_args("  smoke FORCE rerun-failed cells=codex/a,cursor/b group=ci-pr128-0ba391d ")
    assert options == rm.TriggerOptions(
        scope="smoke", mode="rerun-failed", force=True, cells=["codex/a", "cursor/b"], group="ci-pr128-0ba391d"
    )
    assert options.cacheable == frozenset()  # force: nothing is reused
    assert rm.parse_trigger_args("rerun-failed").cacheable == frozenset({"PASS"})
    # aliases: `rerun=<list>`, and `rerun <list>` with the list as the next token (a keyword there is not a list)
    assert rm.parse_trigger_args("rerun=codex/*").cells == ["codex/*"]
    options = rm.parse_trigger_args("rerun codex/x,*/y force")
    assert options.cells == ["codex/x", "*/y"] and options.force and options.errors == []
    options = rm.parse_trigger_args("rerun force")
    assert options.force and options.cells == []
    assert options.errors == ["`rerun`: expected a cell list, e.g. `rerun codex/my-case,cursor/*`"]
    # the canary is asked for explicitly: `smoke-only` (alias `smoke`, the label/dispatch token) or `with-smoke`
    assert rm.TriggerOptions().canary_request is None  # a default run has no canary
    assert rm.parse_trigger_args("smoke-only") == rm.TriggerOptions(scope="smoke") == rm.parse_trigger_args("Smoke")
    assert rm.parse_trigger_args("smoke-only").canary_request == "smoke-only"
    assert rm.parse_trigger_args("with-smoke") == rm.TriggerOptions(with_smoke=True)
    assert rm.parse_trigger_args("with-smoke").canary_request == "with-smoke"
    assert rm.parse_trigger_args("with-smoke smoke-only").canary_request == "smoke-only"  # canary only wins
    options = rm.parse_trigger_args("rerun with-smoke")  # a keyword is not a cell list
    assert options.with_smoke and options.cells == [] and len(options.errors) == 1
    assert rm.parse_trigger_args("with-smoke=1 smoke-only=x").errors == [
        "unknown argument `with-smoke=1`",
        "unknown argument `smoke-only=x`",
    ]
    # unknown tokens and malformed values are reported; what was understood still applies
    options = rm.parse_trigger_args("rerun-failed bogus cells= cells=nodash,codex/ok group=bad/id smoke=1")
    assert (options.mode, options.cells, options.group, options.scope) == ("rerun-failed", ["codex/ok"], None, "full")
    assert options.errors == [
        "unknown argument `bogus`",
        "`cells=`: cells are `<backend>/<case>` or `<skill>/<backend>/<case>` patterns, comma-separated",
        "`cells=nodash,codex/ok`: cells are `<backend>/<case>` or `<skill>/<backend>/<case>` patterns, comma-separated",
        "`group=bad/id`: a group id is letters, digits, `.`, `_` or `-` (1–100 chars)",
        "unknown argument `smoke=1`",
    ]
    assert rm.TriggerOptions.from_dict(json.loads(json.dumps(options.to_dict()))) == options
    assert rm.describe_trigger(options) == "scope full · mode rerun-failed · cells=codex/ok"
    # the canary request round-trips through the plan JSON; a plan written before the field existed = not requested
    requested = rm.parse_trigger_args("with-smoke")
    assert rm.TriggerOptions.from_dict(json.loads(json.dumps(requested.to_dict()))) == requested
    assert requested.to_dict()["with_smoke"] is True
    assert rm.TriggerOptions.from_dict({"scope": "full", "mode": "default"}) == rm.TriggerOptions()
    # describe_trigger (the log line) and the comment header name the canary request like the other modes
    assert rm.describe_trigger(requested) == "scope full · mode default · canary with-smoke"
    assert (
        rm.describe_trigger(rm.parse_trigger_args("with-smoke force"))
        == "scope full · mode default · canary with-smoke · force"
    )
    assert rm.describe_trigger(rm.parse_trigger_args("smoke-only")) == "scope smoke · mode default · canary smoke-only"
    assert rm.render_header(run_info(), rm.parse_trigger_args("with-smoke force")).endswith(
        " · canary with-smoke · mode force"
    )
    assert rm.render_header(run_info(), rm.parse_trigger_args("smoke")).endswith(" · canary smoke-only")
    assert rm.render_header(run_info(), rm.TriggerOptions()) == rm.render_header(run_info())  # default: no note


def test_trigger_options_select_cells_by_pattern():
    cell = rm.Cell(SKILL, "trtllm-nano-t2i", "codex")
    assert rm.TriggerOptions().selects(cell)
    assert rm.TriggerOptions(cells=["codex/trtllm-nano-t2i"]).selects(cell)
    assert rm.TriggerOptions(cells=[f"{SKILL}/codex/trtllm-*"]).selects(cell)
    assert rm.TriggerOptions(cells=["*/trtllm-nano-t2i"]).selects(cell)
    assert rm.TriggerOptions(cells=["cursor/*", "codex/*"]).selects(cell)
    assert not rm.TriggerOptions(cells=["cursor/*"]).selects(cell)
    assert not rm.TriggerOptions(cells=["CODEX/*"]).selects(cell)  # case-sensitive
    assert not rm.TriggerOptions(cells=["other-skill/codex/*"]).selects(cell)


def test_skill_content_sha256_matches_hand_computed_value_and_ignores_junk(tmp_path: Path):
    skill = tmp_path / "skill"
    (skill / "eval").mkdir(parents=True)
    (skill / "SKILL.md").write_bytes(b"# skill\n")
    (skill / "eval" / "eval.config").write_bytes(b'{"evals": []}\n')
    expected = hashlib.sha256()
    for relpath, content in (("SKILL.md", b"# skill\n"), ("eval/eval.config", b'{"evals": []}\n')):  # POSIX path order
        expected.update(f"{relpath}\0{hashlib.sha256(content).hexdigest()}\n".encode())
    clean = rm.skill_content_sha256(skill)
    assert clean == expected.hexdigest()
    (skill / "__pycache__").mkdir()
    (skill / "__pycache__" / "x.cpython-310.pyc").write_bytes(b"\x00")
    (skill / "eval" / ".DS_Store").write_bytes(b"junk")
    (skill / "helper.pyc").write_bytes(b"\x00")
    assert rm.skill_content_sha256(skill) == clean  # none of that is skill content
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run.sh").write_text("echo hi\n")
    assert rm.skill_content_sha256(skill) != clean
    with pytest.raises(FileNotFoundError):
        rm.skill_content_sha256(tmp_path / "missing")  # never "the sha of nothing"


def test_skill_tree_sha_and_group_anchoring(git_checkout, tmp_path: Path, capsys):
    root, base_sha, head_sha = git_checkout
    path = rm.skill_dir(SKILL)
    head_tree = rm.skill_tree_sha(root, head_sha, path)
    assert head_tree and len(head_tree) == 40
    assert rm.skill_tree_sha(root, base_sha, path) != head_tree  # the head changed SKILL.md
    untouched = rm.skill_dir("cosmos3-untouched")
    assert rm.skill_tree_sha(root, base_sha, untouched) == rm.skill_tree_sha(root, head_sha, untouched)
    assert rm.skill_tree_sha(root, "0" * 40, path) is None  # unknown commit
    assert rm.skill_tree_sha(root, head_sha, "nope/nothing") is None
    assert rm.skill_tree_sha(tmp_path, head_sha, path) is None  # not a repository

    # a later commit that only touches CI files leaves the skill tree unchanged: same version, equivalent commit
    (root / ".github").mkdir(exist_ok=True)
    (root / ".github" / "ci.txt").write_text("harness change\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "ci only")
    ci_sha = git(root, "rev-parse", "HEAD")
    info = run_info(head_sha=ci_sha, base_sha=base_sha)
    plan = rm.SkillPlan(SKILL, ["hello", "world"])
    old_group = f"ci-pr128-{head_sha[:7]}"
    runs = [
        {**catalog_run("r-head", head_sha, "hello", "claude", "PASS", "2026-10-20T01:00:00Z"), "group_id": old_group},
        {
            **catalog_run("r-base", base_sha, "hello", "claude", "PASS", "2026-10-21T01:00:00Z"),
            "group_id": f"ci-pr128-{base_sha[:7]}",
        },
        {
            **catalog_run("r-other-pr", head_sha, "hello", "claude", "PASS", "2026-10-22T01:00:00Z"),
            "group_id": "ci-pr99-abcdef0",
        },
    ]
    version = rm.resolve_skill_version(root, info, plan, runs)
    assert version.content_sha256 == rm.skill_content_sha256(root / path) and version.tree_sha == head_tree
    assert version.equivalent_commits == [ci_sha, head_sha]  # the base changed the skill: not equivalent
    assert version.group_id == old_group  # anchored to THIS PR's newest run of the version, not the other PR's
    assert version.matches(runs[0]) and version.matches(runs[2]) and not version.matches(runs[1])
    assert (
        f"skill {SKILL}: content {version.content_sha256[:7]} · tree {head_tree[:7]} · 2 equivalent commit(s)"
        f" · group {old_group} (the PR's existing group for this skill version)"
    ) in capsys.readouterr().out
    # no run of this PR for the version: a new group named after this head
    fresh = rm.resolve_skill_version(root, info, plan, [runs[1], runs[2]])
    assert fresh.group_id == f"ci-pr128-{ci_sha[:7]}" and fresh.equivalent_commits == [ci_sha, head_sha]
    # `group=` / --group wins for every skill
    forced = rm.resolve_skill_version(root, run_info(head_sha=ci_sha, group_override="ci-pr128-fixed"), plan, runs)
    assert forced.group_id == "ci-pr128-fixed"
    # skipped plans have nothing to hash; a plan whose directory is missing is a warning, not a failure
    plans = [plan, rm.SkillPlan("cosmos3-gone", skipped_reason="gone"), rm.SkillPlan("ghost", ["c"])]
    versions = rm.resolve_skill_versions(root, info, plans, {SKILL: runs})
    assert sorted(versions) == [SKILL] and versions[SKILL].group_id == old_group
    assert "WARN: ghost: cannot compute the skill version" in capsys.readouterr().out
    # the group lookup on RunInfo: per skill, override, default
    info = run_info(head_sha=ci_sha, versions=versions)
    assert info.group_for(SKILL) == old_group and info.group_for("ghost") == f"ci-pr128-{ci_sha[:7]}"
    assert info.group_url(SKILL) == f"{DASH}/skills/{SKILL}/groups/{old_group}/"
    assert run_info(group_override="g1").group_for(SKILL) == "g1" and info.groups_summary() == f"{SKILL}={old_group}"


def test_cache_rule_never_reuses_no_report_or_error_and_follows_the_mode():
    version = rm.SkillVersion.of_commit(SHA)

    def run(run_id, status, created, *, verdict=None):
        entry = catalog_run(run_id, SHA, "c", "codex", status, created)
        if verdict is not None:
            entry["cases"][0]["backends"][0]["verdict"] = verdict
        return entry

    for status in ("NO-REPORT", "ERROR", "PARTIAL"):
        assert rm.find_cached_result([run("r", status, "2026-10-20T00:00:00Z")], version, "c", "codex", DASH) is None
    old_style = run("r-old", "FAIL", "2026-10-20T00:00:00Z", verdict="NO-REPORT")  # pre-v2 manifest
    assert rm.find_cached_result([old_style], version, "c", "codex", DASH) is None
    real_fail = run("r-fail", "FAIL", "2026-10-20T00:00:00Z", verdict="FAIL")
    assert rm.find_cached_result([real_fail], version, "c", "codex", DASH).run_id == "r-fail"
    assert rm.find_cached_result([real_fail], version, "c", "codex", DASH, rm.CACHEABLE_RERUN_FAILED) is None
    passed = run("r-pass", "PASS", "2026-10-19T00:00:00Z", verdict="PASS")
    assert rm.find_cached_result([passed], version, "c", "codex", DASH, rm.CACHEABLE_RERUN_FAILED).run_id == "r-pass"
    assert rm.find_cached_result([passed], version, "c", "codex", DASH, frozenset()) is None  # force
    # only the NEWEST version-matching result counts: a newer NO-REPORT hides an older PASS
    both = [passed, run("r-nr", "NO-REPORT", "2026-10-21T00:00:00Z")]
    assert rm.find_cached_result(both, version, "c", "codex", DASH) is None
    newest = rm.newest_result(both, version, "c", "codex", DASH)
    assert (newest.run_id, newest.status) == ("r-nr", "NO-REPORT")
    assert rm.reusable(None, rm.CACHEABLE_DEFAULT) is False


def test_apply_cache_modes_and_cells_selection_in_the_comment(checkout: Path):
    plans = rm.load_skill_plans(checkout, [SKILL], smoke=False)
    runs = [
        catalog_run("r-pass", SHA, "hello", "claude", "PASS", "2026-10-20T00:00:00Z"),
        catalog_run("r-fail", SHA, "hello", "codex", "FAIL", "2026-10-20T00:00:00Z"),
        catalog_run("r-nr", SHA, "hello", "cursor", "NO-REPORT", "2026-10-20T00:00:00Z"),
        catalog_run("r-err", SHA, "world", "claude", "ERROR", "2026-10-20T00:00:00Z"),
    ]
    dashboard = FakeDashboard({SKILL: runs})

    def decide(options):
        cells = rm.build_cells(plans, BACKENDS)
        hits = rm.apply_cache(cells, run_info(), dashboard, options=options)
        return cells, hits, {(c.case, c.backend): c.state for c in cells if c.state is not rm.CellState.QUEUED}

    cached = rm.CellState.CACHED
    assert decide(rm.TriggerOptions())[1:] == (2, {("hello", "claude"): cached, ("hello", "codex"): cached})
    assert decide(rm.parse_trigger_args("rerun-failed"))[1:] == (1, {("hello", "claude"): cached})
    assert decide(rm.parse_trigger_args("force"))[1:] == (0, {})
    cells, hits, states = decide(rm.parse_trigger_args("cells=codex/*,cursor/hello"))
    assert hits == 3 and states[("hello", "codex")] is cached  # selected and reusable: reused, not re-run
    assert ("hello", "cursor") not in states  # selected; NO-REPORT is never reused: runs
    # outside `cells=`: the version's newest result stands in whatever its status (the matrix stays whole);
    # a cell without any result of this version reads (not selected)
    assert states[("hello", "claude")] is cached and states[("world", "claude")] is cached
    assert states[("world", "opencode")] is rm.CellState.SKIPPED
    # `selected` = part of this run's matrix and counts: the three trigger-selected cells plus the two carried over
    assert [c.label for c in cells if c.selected] == [
        f"{SKILL}/{b}/{c}"
        for c, b in (("hello", "claude"), ("hello", "codex"), ("hello", "cursor"), ("world", "claude"), ("world", "codex"))
    ]
    assert rm.matrix_entries(cells, canary=False) == [  # only the trigger-selected cells without a reusable result run
        matrix_entry(SKILL, b, c) for c, b in (("hello", "cursor"), ("world", "codex"))
    ]
    cells, hits, states = decide(rm.parse_trigger_args("force cells=codex/*,cursor/hello"))
    assert hits == 2 and ("hello", "codex") not in states  # forced: the selected FAIL runs again; outside cells keep theirs
    assert rm.matrix_entries(cells, canary=False) == [
        matrix_entry(SKILL, b, c) for c, b in (("hello", "codex"), ("hello", "cursor"), ("world", "codex"))
    ]

    # the comment: mode note in the header with the selection counts, trigger errors as a note, cells outside the
    # selection carry their last result like fresh ones, `(not selected)` only without one
    options = rm.parse_trigger_args("force cells=codex/*,cursor/hello bogus")
    cells = rm.build_cells(plans, BACKENDS)
    rm.apply_cache(cells, run_info(), dashboard, options=options)
    body = rm.render_comment(run_info(), plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW, options=options)
    assert (
        f"run [#987](https://github.com/{REPO}/actions/runs/987/attempts/2) · mode force · cells=`codex/*,cursor/hello`"
        " (3 selected, 2 reused)" in body
    )
    assert "\n⚠️ Trigger: unknown argument `bogus` — ignored, the run uses the defaults" in body
    assert (
        f"| `hello` | ✅ PASS · [report]({DASH}/runs/r-pass/) | ⏳ queued | ⏳ queued | — not selected |"
        in body
    )
    assert (
        f"| `world` | ⛔ ERROR · [report]({DASH}/runs/r-err/) | ⏳ queued | — not selected | — not selected |"
        in body
    )
    assert "2 / 5 cells done · 1 passed · 0 failed · 0 no report · 1 errors · 2 reused" in body  # 3 selected + 2 carried over
    assert "done —" not in body
    for cell in cells:
        if cell.state is rm.CellState.QUEUED:  # the three trigger-selected cells finish; the carried-over two stay as they are
            cell.state, cell.status, cell.url = rm.CellState.DONE, "PASS", "u"
    body = rm.render_comment(run_info(), plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW, options=options)
    assert body.rstrip().endswith("❌ done — 4/5 passed (0 failed, 0 no report, 1 errors)")  # the carried-over ERROR counts
    # rerun-failed: the header counts what runs and what is reused
    options = rm.parse_trigger_args("rerun-failed")
    cells = rm.build_cells(plans, BACKENDS)
    rm.apply_cache(cells, run_info(), dashboard, options=options)
    body = rm.render_comment(run_info(), plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW, options=options)
    assert " · mode rerun-failed (7 selected, 1 reused)\n" in body and "cells=" not in body.split("\nLegend:")[0]
    # a selection that matches nothing: nothing runs; the four cells the version has results for are carried over
    # (the matrix reads like the dashboard group), the other four read (not selected)
    options = rm.parse_trigger_args("cells=nothing/*")
    cells = rm.build_cells(plans, BACKENDS)
    rm.apply_cache(cells, run_info(), dashboard, options=options)
    body = rm.render_comment(run_info(), plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW, options=options)
    assert " · cells=`nothing/*` (0 selected, 4 reused)\n" in body
    assert "4 / 4 cells done · 1 passed · 1 failed · 1 no report · 1 errors · 4 reused" in body
    assert body.rstrip().endswith("❌ done — 1/4 passed (1 failed, 1 no report, 1 errors)")
    assert body.count("— not selected") == 4
    assert rm.matrix_entries(cells, canary=False) == []


def test_find_run_result_prefers_this_runs_publish_then_newer_version_matches():
    info = run_info()  # run id 987, plan started at FIXED_NOW (07:00)
    mine = catalog_run("r-mine", SHA, "c", "codex", "NO-REPORT", "2026-10-18T07:30:00Z")
    mine["source"] = {"kind": "github-ci", "github": {"run_id": "987"}}
    foreign = catalog_run("r-foreign", SHA, "c", "codex", "PASS", "2026-10-18T07:40:00Z")
    foreign["source"] = {"kind": "github-ci", "github": {"run_id": "111"}}
    older = catalog_run("r-older", SHA, "c", "codex", "PASS", "2026-10-18T06:00:00Z")  # before the plan started
    other_version = catalog_run("r-other", OTHER_SHA, "c", "codex", "PASS", "2026-10-18T08:00:00Z")
    hit = rm.find_run_result([older, foreign, mine, other_version], info, "c", "codex", started_at=FIXED_NOW)
    assert (hit.run_id, hit.status, hit.source_run_id) == ("r-mine", "NO-REPORT", "987")  # any status, by run id
    # without a publish of this run: the newest version-matching run created since the plan started
    hit = rm.find_run_result([older, foreign, other_version], info, "c", "codex", started_at=FIXED_NOW)
    assert hit.run_id == "r-foreign"
    assert rm.find_run_result([older, other_version], info, "c", "codex", started_at=FIXED_NOW) is None
    assert rm.find_run_result([older, foreign], info, "c", "codex", started_at=None) is None  # fallback disabled
    assert rm.find_run_result([older, foreign, mine], info, "c", "cursor", started_at=FIXED_NOW) is None  # other cell
    naive = FIXED_NOW.replace(tzinfo=None)  # a naive started_at is read as UTC
    assert rm.find_run_result([older, foreign], info, "c", "codex", started_at=naive).run_id == "r-foreign"


def test_plan_mode_trigger_args_rerun_failed_anchors_the_group_and_exports_it_to_cells(
    tmp_path: Path, checkout: Path, launcher: Path, capsys
):
    config = write_config(tmp_path, launcher)
    github = FakeGitHub()
    github.compare_files = [{"filename": f".agents/skills/{SKILL}/SKILL.md", "status": "modified"}]
    # an earlier run of this PR evaluated the same skill version (same commit here) in group ci-pr128-0000000;
    # codex/hello ended without a report back then
    runs = [
        catalog_run(f"r-{b}-{c}", SHA, c, b, "PASS", "2026-10-17T01:00:00Z")
        for c in ("hello", "world")
        for b in BACKENDS
    ]
    runs[1]["cases"][0]["backends"][0]["status"] = "NO-REPORT"
    for run in runs:
        run["group_id"] = "ci-pr128-0000000"
    dashboard = FakeDashboard({SKILL: runs})
    plan_path = tmp_path / "plan.json"
    args = ["plan", "--config", str(config), "--checkout", str(checkout), "--plan-out", str(plan_path)]
    args += ["--trigger-args", "rerun-failed"]
    assert rm.main(args, make_deps(plan_env(tmp_path), github, dashboard)) == 0
    outputs = read_outputs(tmp_path / "github_output")
    assert json.loads(outputs["skills"]) == [matrix_entry(SKILL, "codex", "hello")]
    assert (outputs["cached"], outputs["selected"]) == ("7", "1")
    plan = json.loads(plan_path.read_text())
    assert plan["trigger"] == {
        "scope": "full",
        "with_smoke": False,
        "mode": "rerun-failed",
        "force": False,
        "cells": [],
        "group": None,
        "errors": [],
    }
    assert plan["skills"][SKILL]["group_id"] == "ci-pr128-0000000" and plan["skills"][SKILL]["equivalent_commits"] == [
        SHA
    ]
    body = github.comments[0]["body"]
    assert " · mode rerun-failed (1 selected, 7 reused)\n" in body
    assert f"### `{SKILL}` · [dashboard overview]({DASH}/skills/{SKILL}/groups/ci-pr128-0000000/)" in body
    assert "7 / 8 cells done · 7 passed · 0 failed · 0 no report · 0 errors · 7 reused" in body
    assert f"| `hello` | ✅ PASS · [report]({DASH}/runs/r-claude-hello/) | ⏳ queued |" in body

    # the cell job files its run under the anchored group and hands the publisher the content sha
    cell_args = ["cell", "--config", str(config), "--checkout", str(checkout), "--plan", str(plan_path)]
    cell_args += ["--skill", SKILL, "--backend", "codex", "--case", "hello"]
    assert rm.main(cell_args, make_deps(ci_env(), github, dashboard)) == 0
    cell_env = read_env(tmp_path / "work", f"{SKILL}-codex-hello")
    assert cell_env["COSMOS_EVAL_GROUP_ID"] == "ci-pr128-0000000"
    assert cell_env["COSMOS_EVAL_SKILL_CONTENT_SHA"] == rm.skill_content_sha256(checkout / ".agents" / "skills" / SKILL)
    assert " · mode rerun-failed (1 selected, 7 reused)\n" in github.comments[0]["body"]  # cells keep the note
    # a reused cell is not run by its job either
    assert rm.main(cell_args[:-1] + ["world"], make_deps(ci_env(), github, dashboard)) == 0
    assert not (tmp_path / "work" / SHA7 / f"{SKILL}-codex-world.argv").exists()

    # `force` + `--group`: nothing reused, one group for every skill, an unknown token is reported, not fatal
    github = FakeGitHub()
    github.compare_files = [{"filename": f".agents/skills/{SKILL}/SKILL.md", "status": "modified"}]
    args[args.index("--trigger-args") + 1] = "force bogus"
    assert rm.main(args + ["--group", "ci-pr128-fixed"], make_deps(plan_env(tmp_path), github, dashboard)) == 0
    plan = json.loads(plan_path.read_text())
    assert plan["skills"][SKILL]["group_id"] == "ci-pr128-fixed" and plan["info"]["group_override"] == "ci-pr128-fixed"
    assert plan["trigger"]["force"] is True and plan["trigger"]["errors"] == ["unknown argument `bogus`"]
    assert len(json.loads(read_outputs(tmp_path / "github_output")["skills"])) == 8
    body = github.comments[0]["body"]
    assert " · mode force\n" in body and "⚠️ Trigger: unknown argument `bogus` — ignored" in body
    assert f"[dashboard overview]({DASH}/skills/{SKILL}/groups/ci-pr128-fixed/)" in body
    assert "(reused)" not in body.split("\nLegend:")[0]
    # `cells=` alone reuses a reusable result of a selected cell (nothing to run); with `force` it runs; a cell
    # job for a cell outside the selection does nothing
    args[args.index("--trigger-args") + 1] = "cells=cursor/world"
    assert rm.main(args, make_deps(plan_env(tmp_path), FakeGitHub(), dashboard)) == 0
    assert read_outputs(tmp_path / "github_output")["skills"] == "[]"
    args[args.index("--trigger-args") + 1] = "force cells=cursor/world"
    assert rm.main(args, make_deps(plan_env(tmp_path), FakeGitHub(), dashboard)) == 0
    assert json.loads(read_outputs(tmp_path / "github_output")["skills"]) == [matrix_entry(SKILL, "cursor", "world")]
    capsys.readouterr()
    assert rm.main(cell_args, make_deps(ci_env(), FakeGitHub(), dashboard)) == 0  # codex/hello: outside `cells=`
    # outside the selection the cell carries the version's newest result (here NO-REPORT) and is nothing to run
    assert f"{SKILL}/codex/hello: reused (NO-REPORT), nothing to run" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # argparse: --scope only knows full | smoke
        rm.main(["plan", "--config", str(config), "--checkout", str(checkout), "--scope", "nope"])
    assert rm.main(args + ["--group", "bad/id"], make_deps(plan_env(tmp_path), FakeGitHub(), dashboard)) == 2


def test_ledger_lines_are_scoped_to_the_actions_run_that_wrote_them():
    """A re-trigger of the same sha finds the previous run's comment; its ledger must not revive old results."""
    old = rm.Cell(SKILL, "c", "codex", state=rm.CellState.DONE, status="FAIL", url="http://x/old/")
    new = rm.Cell(SKILL, "c", "cursor", state=rm.CellState.DONE, status="PASS", url="http://x/new/")
    legacy = f"{rm.LEDGER_PREFIX}{SKILL}/c/opencode|PASS|http://x/legacy/ -->"  # pre-2026-10-22 format: no run id
    body = "\n".join([rm.ledger_line(old, "111"), rm.ledger_line(new, "222"), legacy])
    assert rm.parse_ledger(body) == {  # unscoped: every line, old format included
        f"{SKILL}/codex/c": ("FAIL", "http://x/old/"),
        f"{SKILL}/cursor/c": ("PASS", "http://x/new/"),
        f"{SKILL}/c/opencode": ("PASS", "http://x/legacy/"),
    }
    assert rm.parse_ledger(body, "222") == {f"{SKILL}/cursor/c": ("PASS", "http://x/new/")}
    assert rm.parse_ledger(body, "333") == {}
    # the rendered comment carries the run id on every ledger line
    info = run_info()
    cells = [old, new]
    rendered = rm.render_comment(
        info, [rm.SkillPlan(SKILL, ["c"])], cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW, seq="1.2"
    )
    assert rm.parse_ledger(rendered, info.run_id) == rm.parse_ledger(rendered) and len(rm.parse_ledger(rendered)) == 2
    assert f"|{info.run_id} -->" in rendered


def test_publish_distributed_ignores_the_previous_runs_ledger(tmp_path: Path):
    info = run_info()  # run 987
    plans = [rm.SkillPlan(SKILL, ["c"])]
    github = FakeGitHub()
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    previous = rm.Cell(SKILL, "c", "claude", state=rm.CellState.DONE, status="FAIL", url="http://x/prev/")
    github.comments.append(
        {"id": 1, "body": f"{info.marker}\n{rm.SEQ_PREFIX}1.4 -->\n{rm.ledger_line(previous, '555')}\nold"}
    )
    cells = rm.build_cells(plans, BACKENDS)  # everything queued in the new run
    assert rm.publish_distributed(api, info, plans, cells, BACKENDS, FIXED_NOW, FIXED_NOW, final=True)
    assert all(cell.state is rm.CellState.QUEUED for cell in cells)  # nothing resurrected
    assert "http://x/prev/" not in github.comments[0]["body"]


def test_build_launcher_command_forwards_the_skill_content_sha(tmp_path: Path):
    config = rm.Config(
        backends=BACKENDS, image="img:v1", dashboard=DASH, launcher=Path("/opt/launch"), workdir=tmp_path
    )
    cell = rm.Cell(SKILL, "hello", "claude")
    command = rm.build_launcher_command(config, tmp_path / "co", tmp_path / "out", cell, "ab" * 32)
    assert f"COSMOS_EVAL_SKILL_CONTENT_SHA={'ab' * 32}" in command
    assert command.index("-e") < command.index("--github-env")
    assert "COSMOS_EVAL_SKILL_CONTENT_SHA" not in " ".join(
        rm.build_launcher_command(config, tmp_path / "co", tmp_path / "out", cell)
    )


def test_coordinated_comment_is_taken_over_by_a_newer_run_and_keeps_history():
    """One comment per head commit: a re-trigger of the SAME commit rewrites that commit's comment (the earlier run
    moves under 'Previous runs', a late update from it is ignored); a run of a NEW commit posts a new comment that
    inherits the history, and leaves the earlier commit's comment untouched."""
    plans = [rm.SkillPlan(SKILL, ["c"])]
    github = FakeGitHub()
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    old = run_info()  # run 987 on SHA
    cells_old = rm.build_cells(plans, BACKENDS)
    for cell, status in zip(cells_old, ["PASS", "FAIL", "PASS", "NO-REPORT"]):
        cell.state, cell.status, cell.url = rm.CellState.DONE, status, f"http://x/{cell.backend}/"
    assert rm.publish_distributed(api, old, plans, cells_old, BACKENDS, FIXED_NOW, FIXED_NOW, final=True)
    assert (
        len(github.comments) == 1
        and "❌ done — 2/4 passed (1 failed, 1 no report, 0 errors)" in github.comments[0]["body"]
    )

    # a re-trigger of the same commit: the same comment is rewritten and the earlier run goes into the history
    rerun = run_info(run_id="988", head_sha=SHA, run_attempt="1")
    cells_rerun = rm.build_cells(plans, BACKENDS)
    cells_rerun[0].state, cells_rerun[0].status, cells_rerun[0].url = rm.CellState.CACHED, "PASS", "http://x/claude/"
    assert rm.publish_distributed(api, rerun, plans, cells_rerun, BACKENDS, FIXED_NOW, FIXED_NOW, final=True)
    assert len(github.comments) == 1  # rewritten, not a second comment
    body = github.comments[0]["body"]
    assert body.splitlines()[0] == f"<!-- cosmos-skill-eval:pr128:{SHA} -->"
    assert rm.parse_owner(body) == ("988", SHA, "1")
    assert rm.parse_history(body) == [f"987|{SHA}|❌ done — 2/4 passed (1 failed, 1 no report, 0 errors)"]
    assert "**Previous runs on this PR**" in body and f"run [#987](https://github.com/{REPO}/actions/runs/987)" in body
    assert "http://x/codex/" not in body  # the old run's cells were not resurrected into the new run's view
    assert "1 / 4 cells done" in body  # the new run's own view: one reused cell, three queued

    # the old run's report job finishes late: it must not clobber the newer run's view of the same commit
    assert rm.publish_distributed(api, old, plans, cells_old, BACKENDS, FIXED_NOW, FIXED_NOW, final=True) is False
    assert rm.parse_owner(github.comments[0]["body"]) == ("988", SHA, "1")

    # a NEW commit: a second comment, the first one left as it is, both earlier runs listed newest first
    new = run_info(run_id="989", head_sha=OTHER_SHA, run_attempt="1")
    cells_new = rm.build_cells(plans, BACKENDS)
    assert rm.publish_distributed(api, new, plans, cells_new, BACKENDS, FIXED_NOW, FIXED_NOW, final=True)
    assert len(github.comments) == 2
    first, second = github.comments[0]["body"], github.comments[1]["body"]
    assert rm.parse_owner(first) == ("988", SHA, "1")  # untouched
    assert second.splitlines()[0] == f"<!-- cosmos-skill-eval:pr128:{OTHER_SHA} -->"
    assert rm.parse_owner(second) == ("989", OTHER_SHA, "1")
    assert f"## Skill eval — [{OTHER_SHA[:7]}]" in second
    history = rm.parse_history(second)
    assert [h.split("|")[0] for h in history] == ["988", "987"]
    assert history[0].endswith(
        "unfinished — 1 / 4 cells done · 1 passed · 0 failed · 0 no report · 0 errors · 1 reused"
    )
    assert f"[{SHA7}](https://github.com/{REPO}/commit/{SHA})" in second  # the earlier commit is named in the list
    # a third commit keeps every predecessor, newest first, capped at MAX_HISTORY
    third = run_info(run_id="990", head_sha="c" * 40, run_attempt="1")
    assert rm.publish_distributed(
        api, third, plans, rm.build_cells(plans, BACKENDS), BACKENDS, FIXED_NOW, FIXED_NOW, final=True
    )
    assert len(github.comments) == 3
    assert [h.split("|")[0] for h in rm.parse_history(github.comments[2]["body"])] == ["989", "988", "987"]


def test_new_commit_inherits_the_history_of_a_legacy_one_per_pr_comment():
    """A PR whose comment predates the per-commit markers (`<!-- cosmos-skill-eval:pr<N> -->`) gets a new comment
    for the new commit; the legacy comment is left alone and its run is the first history entry."""
    plans = [rm.SkillPlan(SKILL, ["c"])]
    legacy = (
        "<!-- cosmos-skill-eval:pr128 -->\n"
        f"{rm.OWNER_PREFIX}900|{SHA}|1 -->\n"
        f"{rm.HISTORY_PREFIX}899|{SHA}|❌ done — 1/4 passed (3 failed, 0 no report, 0 errors) -->\n"
        "## Skill eval — old\n\n✅ done — 4/4 passed (0 failed, 0 no report, 0 errors)\n"
    )
    github = FakeGitHub(comments=[{"id": 77, "body": legacy}])
    api = rm.GitHubApi("https://github.com", REPO, "t", transport=github)
    info = run_info(run_id="901", head_sha=OTHER_SHA, run_attempt="1")
    assert rm.publish_distributed(api, info, plans, rm.build_cells(plans, BACKENDS), BACKENDS, FIXED_NOW, FIXED_NOW, final=True)
    assert len(github.comments) == 2 and github.comments[0]["body"] == legacy
    body = github.comments[1]["body"]
    assert body.splitlines()[0] == f"<!-- cosmos-skill-eval:pr128:{OTHER_SHA} -->"
    assert rm.parse_history(body) == [
        f"900|{SHA}|✅ done — 4/4 passed (0 failed, 0 no report, 0 errors)",
        f"899|{SHA}|❌ done — 1/4 passed (3 failed, 0 no report, 0 errors)",
    ]
    # the same commit again: the per-commit comment is reused, the legacy one still untouched
    again = run_info(run_id="902", head_sha=OTHER_SHA, run_attempt="1")
    assert rm.publish_distributed(api, again, plans, rm.build_cells(plans, BACKENDS), BACKENDS, FIXED_NOW, FIXED_NOW, final=True)
    assert len(github.comments) == 2 and rm.parse_owner(github.comments[1]["body"]) == ("902", OTHER_SHA, "1")
    assert "pr128 -->" not in github.comments[1]["body"].splitlines()[0]


def test_cells_are_not_re_run_after_three_reportless_attempts(checkout: Path):
    """A cell whose three newest results (same skill version) ended without a report is a harness-side failure:
    it keeps its latest status in the matrix and the counts but gets no job — unless the trigger forces it."""
    plans = rm.load_skill_plans(checkout, [SKILL], smoke=False)
    runs = [
        catalog_run("r3", SHA, "hello", "claude", "NO-REPORT", "2026-10-23T03:00:00Z"),
        catalog_run("r2", SHA, "hello", "claude", "ERROR", "2026-10-23T02:00:00Z"),
        catalog_run("r1", SHA, "hello", "claude", "FAIL", "2026-10-23T01:00:00Z", case_status="FAIL"),  # pre-v2 FAIL …
        catalog_run("r0", SHA, "hello", "claude", "PASS", "2026-10-22T01:00:00Z"),
        catalog_run("w2", SHA, "world", "codex", "NO-REPORT", "2026-10-23T02:00:00Z"),
        catalog_run("w1", SHA, "world", "codex", "NO-REPORT", "2026-10-23T01:00:00Z"),
    ]
    runs[2]["cases"][0]["backends"][0]["verdict"] = "NO-REPORT"  # … whose verdict says the agent wrote no report
    version = rm.SkillVersion.of_commit(SHA)
    assert rm.reportless_streak(runs, version, "hello", "claude", DASH) == 3
    assert rm.reportless_streak(runs, version, "world", "codex", DASH) == 2
    assert rm.reportless_streak(runs, version, "hello", "cursor", DASH) == 0

    def plan_with(options):
        cells = rm.build_cells(plans, BACKENDS)
        rm.apply_cache(cells, run_info(), FakeDashboard({SKILL: runs}), options=options)
        return {(c.case, c.backend): c for c in cells}

    cells = plan_with(rm.TriggerOptions())
    exhausted = cells[("hello", "claude")]
    assert exhausted.state is rm.CellState.EXHAUSTED and exhausted.status == "NO-REPORT"
    assert exhausted.reason == "3 report-less attempts" and exhausted.url == f"{DASH}/runs/r3/"
    assert cells[("world", "codex")].state is rm.CellState.QUEUED  # two attempts: still re-run
    assert exhausted.finished and exhausted.selected and not exhausted.passed
    jobs = rm.matrix_entries(list(cells.values()), canary=False)
    assert matrix_entry(SKILL, "claude", "hello") not in jobs
    assert matrix_entry(SKILL, "codex", "world") in jobs
    counts = rm.tally(cells.values())
    assert (counts.total, counts.done, counts.no_report) == (8, 1, 1)  # counted, like any published result
    body = rm.render_comment(run_info(), plans, list(cells.values()), BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert f"❔ NO-REPORT (not re-run: 3 report-less attempts) · [report]({DASH}/runs/r3/)" in body
    # `force` and an explicit `cells=` selection still run it; rerun-failed respects the cap
    assert plan_with(rm.TriggerOptions(force=True))[("hello", "claude")].state is rm.CellState.QUEUED
    assert plan_with(rm.TriggerOptions(cells=["claude/hello"]))[("hello", "claude")].state is rm.CellState.QUEUED
    assert plan_with(rm.TriggerOptions(mode="rerun-failed"))[("hello", "claude")].state is rm.CellState.EXHAUSTED
    # the state survives the plan JSON round trip, so the cell job knows there is nothing to run
    info = run_info()
    plan = rm.plan_to_dict(info, plans, list(cells.values()), BACKENDS, FIXED_NOW, rm.TriggerOptions())
    _i, _p, back, _b, _s, _o = rm.plan_from_dict(json.loads(json.dumps(plan)))
    again = next(c for c in back if (c.case, c.backend) == ("hello", "claude"))
    assert (
        again.state is rm.CellState.EXHAUSTED
        and again.reason == "3 report-less attempts"
        and again.url == f"{DASH}/runs/r3/"
    )


# ---------------------------------------------------------------------------
# newest result per cell across runs (refresh_from_catalog); cursor cells on one runner
# ---------------------------------------------------------------------------


def test_refresh_from_catalog_shows_the_newest_result_per_cell_and_never_downgrades():
    """A partial run replaces only the cells it ran; every other cell shows the skill version's newest result, as
    the dashboard group does — and a cell's own result gives way to a newer run's result for the same cell."""
    info = run_info()
    plans = [rm.SkillPlan(SKILL, ["c"])]
    cells = rm.build_cells(plans, BACKENDS)  # claude, codex, cursor, opencode
    claude, codex, cursor, opencode = cells
    # claude: this run's own ERROR (07:10); another run re-measured the cell later (PASS, 07:40) -> superseded
    own = catalog_run("r-mine", SHA, "c", "claude", "ERROR", "2026-10-18T07:10:00Z")
    fix = catalog_run("r-fix", SHA, "c", "claude", "PASS", "2026-10-18T07:40:00Z")
    claude.adopt(rm._run_result(own, "c", "claude", DASH), rm.CellState.DONE)
    # codex: this run's own PASS is the newest; an older FAIL of the version must not replace it
    older = catalog_run("r-older", SHA, "c", "codex", "FAIL", "2026-10-18T06:00:00Z")
    newest_own = catalog_run("r-codex", SHA, "c", "codex", "PASS", "2026-10-18T07:30:00Z")
    codex.adopt(rm._run_result(newest_own, "c", "codex", DASH), rm.CellState.DONE)
    # cursor: outside `cells=` with no result at plan time; a result was published since
    cursor.state = rm.CellState.SKIPPED
    late = catalog_run("r-late", SHA, "c", "cursor", "FAIL", "2026-10-18T07:20:00Z")
    # opencode: queued — about to publish its own result; a result of the version is not shown for it
    foreign = catalog_run("r-foreign", SHA, "c", "opencode", "PASS", "2026-10-18T07:35:00Z")
    runs = [own, fix, older, newest_own, late, foreign]
    assert rm.refresh_from_catalog(cells, info, {SKILL: runs}) == 2
    assert (claude.state, claude.status, claude.url) == (rm.CellState.CACHED, "PASS", f"{DASH}/runs/r-fix/")
    assert (claude.run_id, claude.created_at, claude.reason) == ("r-fix", "2026-10-18T07:40:00Z", None)
    assert (codex.state, codex.status, codex.run_id) == (rm.CellState.DONE, "PASS", "r-codex")  # own newest: kept
    assert (cursor.state, cursor.status, cursor.url) == (rm.CellState.CACHED, "FAIL", f"{DASH}/runs/r-late/")
    assert (opencode.state, opencode.status) == (rm.CellState.QUEUED, None)
    counts = rm.tally(cells)  # the counts follow the displayed statuses: the superseded ERROR is a PASS now
    assert (counts.total, counts.done, counts.passed, counts.failed, counts.errors, counts.reused) == (4, 3, 2, 1, 0, 2)
    body = rm.render_comment(info, plans, cells, BACKENDS, started_at=FIXED_NOW, now=FIXED_NOW)
    assert (
        f"| `c` | ✅ PASS · [report]({DASH}/runs/r-fix/) | ✅ PASS · [report]({DASH}/runs/r-codex/)"
        f" | ❌ FAIL · [report]({DASH}/runs/r-late/) | ⏳ queued |"
    ) in body
    assert "3 / 4 cells done · 2 passed · 1 failed · 0 no report · 0 errors · 2 reused" in body
    assert rm.refresh_from_catalog(cells, info, {SKILL: runs}) == 0  # idempotent
    # an own result the catalog does not list yet has no date: never downgraded on a guess; a launcher error (no
    # published result) is never touched; a running cell neither
    unlisted = rm.Cell(SKILL, "c", "claude", state=rm.CellState.DONE, status="ERROR", url=f"{DASH}/runs/r-unlisted/")
    crashed = rm.Cell(SKILL, "c", "codex", state=rm.CellState.ERROR, reason="launcher exit 1")
    running = rm.Cell(SKILL, "c", "cursor", state=rm.CellState.RUNNING)
    assert rm.refresh_from_catalog([unlisted, crashed, running], info, {SKILL: runs}) == 0
    assert (unlisted.status, crashed.state, running.state) == ("ERROR", rm.CellState.ERROR, rm.CellState.RUNNING)
    # … until the catalog lists it, older than the fix: then it is superseded like any dated result
    listed = catalog_run("r-unlisted", SHA, "c", "claude", "ERROR", "2026-10-18T07:00:00Z")
    assert rm.refresh_from_catalog([unlisted], info, {SKILL: runs + [listed]}) == 1
    assert (unlisted.state, unlisted.status, unlisted.run_id) == (rm.CellState.CACHED, "PASS", "r-fix")
    # no catalog for the skill, or no result of the version for the cell: untouched
    assert rm.refresh_from_catalog([rm.Cell(SKILL, "c", "claude", state=rm.CellState.SKIPPED)], info, {}) == 0
    other = rm.Cell(SKILL, "c", "claude", state=rm.CellState.SKIPPED)
    other_version = catalog_run("x", OTHER_SHA, "c", "claude", "PASS", "2026-10-18T09:00:00Z")
    assert rm.refresh_from_catalog([other], info, {SKILL: [other_version]}) == 0
    assert other.state is rm.CellState.SKIPPED
    # a cell not re-run after three report-less attempts takes a newer result too, and loses the "not re-run" note
    tired = rm.Cell(SKILL, "c", "claude", state=rm.CellState.EXHAUSTED, status="NO-REPORT", url=f"{DASH}/runs/r-mine/")
    tired.reason = "3 report-less attempts"
    assert rm.refresh_from_catalog([tired], info, {SKILL: runs}) == 1
    assert (tired.state, tired.status, tired.reason) == (rm.CellState.CACHED, "PASS", None)
    assert rm.run_id_from_url(f"{DASH}/runs/abc-1/") == "abc-1" and rm.run_id_from_url("u") is None
    assert rm.run_id_from_url("https://other.host/runs/x") == "x" and rm.run_id_from_url(None) is None
    assert rm.parse_created_at("2026-10-18T07:00:00Z") == FIXED_NOW and rm.parse_created_at("") is None
    assert rm.parse_created_at(None) is None and rm._created_at_key("nope") == rm.UNDATED


def test_distributed_comment_follows_the_newest_result_per_cell_across_concurrent_runs(
    tmp_path: Path, checkout: Path, launcher: Path, capsys
):
    """The observed race: a single-cell rerun planned while the full matrix run was still publishing. Its first
    comment carries everything published so far, each refresh picks up what was published since, and the final
    report replaces an own result that yet another run superseded meanwhile — the job's verdict stays its own."""
    github = FakeGitHub()
    github.compare_files = [{"filename": f".agents/skills/{SKILL}/SKILL.md", "status": "modified"}]
    config = write_config(tmp_path, launcher)

    def published_by_matrix_run(case, backend, status, created_at):
        run = catalog_run(f"m-{backend}-{case}", SHA, case, backend, status, created_at)
        run["source"] = {"kind": "github-ci", "github": {"run_id": "111"}}  # the full matrix run, still going
        return run

    # at plan time the matrix run has published 3 of the 8 cells; its codex/hello ended in ERROR -> this rerun's cell
    dashboard = FakeDashboard(
        {
            SKILL: [
                published_by_matrix_run("hello", "claude", "PASS", "2026-10-18T06:30:00Z"),
                published_by_matrix_run("hello", "codex", "ERROR", "2026-10-18T06:40:00Z"),
                published_by_matrix_run("world", "claude", "PASS", "2026-10-18T06:50:00Z"),
            ]
        }
    )
    plan_path = tmp_path / "plan.json"
    args = ["plan", "--config", str(config), "--checkout", str(checkout), "--plan-out", str(plan_path)]
    args += ["--trigger-args", "force cells=codex/hello"]
    assert rm.main(args, make_deps(plan_env(tmp_path), github, dashboard)) == 0
    assert json.loads(read_outputs(tmp_path / "github_output")["skills"]) == [matrix_entry(SKILL, "codex", "hello")]
    body = github.comments[0]["body"]
    # the first comment: the published cells are carried (reused), the selected cell queued, the rest not selected yet
    assert (
        f"| `hello` | ✅ PASS · [report]({DASH}/runs/m-claude-hello/) | ⏳ queued | — not selected | — not selected |"
    ) in body
    assert (
        f"| `world` | ✅ PASS · [report]({DASH}/runs/m-claude-world/)"
        " | — not selected | — not selected | — not selected |"
    ) in body
    assert "2 / 3 cells done · 2 passed · 0 failed · 0 no report · 0 errors · 2 reused" in body
    assert " · mode force · cells=`codex/hello` (1 selected, 2 reused)\n" in body
    assert f"reused {SKILL}/claude/world (outside cells=): PASS" in capsys.readouterr().out

    # the matrix run publishes two more cells before this rerun's cell job starts
    dashboard.runs_by_skill[SKILL] += [
        published_by_matrix_run("hello", "cursor", "FAIL", "2026-10-18T07:05:00Z"),
        published_by_matrix_run("world", "codex", "PASS", "2026-10-18T07:06:00Z"),
    ]
    github.jobs = [job(f"{SKILL}/codex/hello", "in_progress")]
    results = tmp_path / "cell-results"
    cell_args = ["cell", "--config", str(config), "--checkout", str(checkout), "--plan", str(plan_path)]
    cell_args += ["--skill", SKILL, "--backend", "codex", "--case", "hello", "--results-dir", str(results)]
    assert rm.main(cell_args, make_deps(ci_env(), github, dashboard)) == 0
    out = capsys.readouterr().out
    assert f"reused {SKILL}/cursor/hello (outside cells=, published since the plan): FAIL" in out
    body = github.comments[0]["body"]
    # this run's own FAIL (the fake launcher fails codex) replaced the matrix run's ERROR; the two cells the matrix
    # run published since the plan are carried
    assert (
        f"| `hello` | ✅ PASS · [report]({DASH}/runs/m-claude-hello/)"
        f" | ❌ FAIL · [report]({DASH}/runs/r-codex-hello/)"
        f" | ❌ FAIL · [report]({DASH}/runs/m-cursor-hello/) | — not selected |"
    ) in body
    assert (
        f"| `world` | ✅ PASS · [report]({DASH}/runs/m-claude-world/)"
        f" | ✅ PASS · [report]({DASH}/runs/m-codex-world/) | — not selected | — not selected |"
    ) in body
    assert "5 / 5 cells done · 3 passed · 2 failed · 0 no report · 0 errors · 4 reused" in body
    assert " (1 selected, 4 reused)\n" in body
    own_file = json.loads((results / f"{SKILL}-codex-hello.json").read_text())
    assert (own_file["status"], own_file["dashboard_run_id"], own_file["created_at"]) == ("FAIL", "r-codex-hello", None)

    # before the report, the dashboard lists this run's FAIL (07:30) and a later rerun's PASS for the same cell (07:50,
    # Actions run 999): the final comment shows the newest result, the job's own verdict stays in its result file
    mine = catalog_run("r-codex-hello", SHA, "hello", "codex", "FAIL", "2026-10-18T07:30:00Z")
    mine["source"] = {"kind": "github-ci", "github": {"run_id": "987"}}
    later = catalog_run("l-codex-hello", SHA, "hello", "codex", "PASS", "2026-10-18T07:50:00Z")
    later["source"] = {"kind": "github-ci", "github": {"run_id": "999"}}
    dashboard.runs_by_skill[SKILL] += [mine, later]
    github.jobs = [job(f"{SKILL}/codex/hello", "completed", "success")]
    report_args = ["report", "--config", str(config), "--checkout", str(checkout), "--plan", str(plan_path)]
    report_args += ["--results-dir", str(results)]
    assert rm.main(report_args, make_deps(ci_env(), github, dashboard)) == 0
    out = capsys.readouterr().out
    assert f"superseded {SKILL}/codex/hello: FAIL (r-codex-hello) -> PASS — newer result from run l-codex-hello" in out
    body = github.comments[0]["body"]
    assert f"| ✅ PASS · [report]({DASH}/runs/l-codex-hello/) |" in body
    assert f"{DASH}/runs/r-codex-hello/" not in body  # neither in the table nor in a ledger line
    # the counts follow the plan's decision, not the display: the superseded cell was selected and did run
    assert " · mode force · cells=`codex/hello` (1 selected, 4 reused)\n" in body
    assert "5 / 5 cells done · 4 passed · 1 failed · 0 no report · 0 errors · 4 reused" in body
    assert body.rstrip().endswith("❌ done — 4/5 passed (1 failed, 0 no report, 0 errors)")
    assert json.loads((results / f"{SKILL}-codex-hello.json").read_text())["status"] == "FAIL"


def test_cell_job_verdict_is_its_own_outcome_even_when_the_comment_shows_a_newer_result(
    tmp_path: Path, checkout: Path, launcher: Path, capsys
):
    """codex/hello fails in this job (exit 1 would be wrong: a published FAIL is exit 0) while another run's newer PASS
    for the cell is already listed: the comment shows the PASS, the job reports and persists its own FAIL."""
    github = FakeGitHub()
    dashboard = FakeDashboard({SKILL: []})
    config, plan = _planned(tmp_path, checkout, launcher, github, dashboard)
    mine = catalog_run("r-codex-hello", SHA, "hello", "codex", "FAIL", "2026-10-18T07:30:00Z")  # listed at once
    mine["source"] = {"kind": "github-ci", "github": {"run_id": "987"}}
    newer = catalog_run("l-codex-hello", SHA, "hello", "codex", "PASS", "2026-10-18T07:50:00Z")
    newer["source"] = {"kind": "github-ci", "github": {"run_id": "999"}}
    dashboard.runs_by_skill[SKILL] = [mine, newer]
    github.jobs = [job(f"{SKILL}/codex/hello", "in_progress")]
    args = ["cell", "--config", str(config), "--checkout", str(checkout), "--plan", plan, "--results-dir"]
    args += [str(tmp_path / "results"), "--skill", SKILL, "--backend", "codex", "--case", "hello"]
    assert rm.main(args, make_deps(ci_env(), github, dashboard)) == 0
    assert f"cell {SKILL}/codex/hello: FAIL" in capsys.readouterr().out
    assert json.loads((tmp_path / "results" / f"{SKILL}-codex-hello.json").read_text())["status"] == "FAIL"
    body = github.comments[0]["body"]
    assert f"| `hello` | ⏳ queued | ✅ PASS · [report]({DASH}/runs/l-codex-hello/) | ⏳ queued | ⏳ queued |" in body
    # counted by the status shown, not as reused: this run executed the cell (the plan selected it)
    assert "1 / 8 cells done · 1 passed · 0 failed · 0 no report · 0 errors · 0 reused" in body


def test_matrix_entries_pin_serial_backends_to_their_runner_label(tmp_path: Path):
    cells = rm.build_cells([rm.SkillPlan(SKILL, ["c"])], BACKENDS)
    entries = rm.matrix_entries(cells, canary=False)  # the defaults are today's CI config
    assert [json.loads(entry["runs_on"]) for entry in entries] == [
        ["self-hosted", "cosmos-skill-eval", "h100"],
        ["self-hosted", "cosmos-skill-eval", "h100"],
        ["self-hosted", "cosmos-skill-eval", "h100", "cursor-serial"],
        ["self-hosted", "cosmos-skill-eval", "h100"],
    ]
    assert entries[2] == matrix_entry(SKILL, "cursor", "c")
    assert set(entries[0]) == {"skill", "backend", "case", "runs_on"}
    # the config keys default when absent; explicit values are used; `serial_backends: []` switches the pinning off
    base = "backends: [claude, codex, cursor, opencode]\nimage: i\ndashboard: http://d\nlauncher: l\nworkdir: w\n"
    path = tmp_path / "c.yml"
    path.write_text(base)
    config = rm.load_config(path)
    assert config.runner_labels == ["self-hosted", "cosmos-skill-eval", "h100"]
    assert (config.serial_backends, config.serial_runner_label) == (["cursor"], "cursor-serial")
    custom = "runner_labels: [self-hosted, gpu]\nserial_backends: [cursor, codex]\nserial_runner_label: one\n"
    path.write_text(base + custom)
    config = rm.load_config(path)
    assert rm.runs_on("codex", config) == ["self-hosted", "gpu", "one"]
    assert rm.runs_on("claude", config) == ["self-hosted", "gpu"]
    assert [json.loads(e["runs_on"])[-1] for e in rm.matrix_entries(cells, canary=False, config=config)] == [
        "gpu",
        "one",
        "one",
        "gpu",
    ]
    path.write_text(base + "serial_backends: []\n")
    config = rm.load_config(path)
    assert config.serial_backends == [] and rm.runs_on("cursor", config) == ["self-hosted", "cosmos-skill-eval", "h100"]
    for bad, match in (
        ("runner_labels: []\n", "runner_labels"),
        ("runner_labels: h100\n", "runner_labels"),
        ("serial_backends: [gemini]\n", "serial_backends"),
        ("serial_backends: cursor\n", "serial_backends"),
        ("serial_runner_label: [a]\n", "serial_runner_label"),
        ('serial_runner_label: ""\n', "serial_runner_label"),
    ):
        path.write_text(base + bad)
        with pytest.raises(rm.OrchestrationError, match=match):
            rm.load_config(path)
    # a config without cursor among its backends keeps working: the default pinning is inert for it
    path.write_text("backends: [claude]\nimage: i\ndashboard: http://d\nlauncher: l\nworkdir: w\n")
    assert rm.load_config(path).serial_backends == ["cursor"]
    # the repository's config and workflow agree with this: the keys are set, the cell jobs read `runs_on`
    github_dir = Path(__file__).resolve().parents[3]
    live = rm.load_config(github_dir / "skill-eval.yml")
    assert (live.runner_labels, live.serial_backends, live.serial_runner_label) == (
        ["self-hosted", "cosmos-skill-eval", "h100"],
        ["cursor"],
        "cursor-serial",
    )
    workflow = (github_dir / "workflows" / "skill-eval.yml").read_text(encoding="utf-8")
    assert workflow.count("runs-on: ${{ fromJSON(matrix.runs_on) }}") == 2  # the canary and the cells jobs
