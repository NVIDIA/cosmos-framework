# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cosmos_job_record — the ONLY job-record writer.

Invariants under test: open binds results_dir before any launch handle exists
and the record validates against the M0 schema; transitions are append-only
with immutable terminal states; every write is redacted; concurrent marks
(agent + poller) lose nothing; a job id can never traverse outside .cosmos/jobs.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cosmos_job_record as jr  # noqa: E402

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "execution/artifacts/references/job_record.schema.json").read_text()
)

SECRET = "nvapi-SUPERSECRET123"


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COSMOS_STATE_DIR", str(tmp_path / ".cosmos"))
    return tmp_path / ".cosmos"


def open_job(capsys, **overrides):
    argv = [
        "open",
        "--platform",
        "slurm",
        "--image",
        "cosmos-framework:local",  # unpinned: test fixture
        "--network-arch",
        "cosmos3",
        "--action",
        "train",
        "--storage-tier",
        "A",
        "--results-root",
        "/lustre/fsw/users/me/results",
        "--upload-exclude",
        "inputs/",
    ]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag, v]
    assert jr.main(argv) == 0
    return capsys.readouterr().out.strip()


def read_record(state_dir, job_id):
    return json.loads((state_dir / "jobs" / f"{job_id}.json").read_text())


def test_atomic_write_cleans_temporary_file_after_io_failure(tmp_path, monkeypatch):
    path = tmp_path / "record.json"

    def fail_dump(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jr.json, "dump", fail_dump)
    with pytest.raises(OSError, match="No space left on device"):
        jr._atomic_write(path, {"id": "example"})

    assert not list(tmp_path.glob(".record.json.*.tmp"))


# --------------------------------------------------------------------------- #
# open
# --------------------------------------------------------------------------- #


def test_open_writes_schema_valid_pending_record(state_dir, capsys):
    job_id = open_job(capsys)
    rec = read_record(state_dir, job_id)
    jsonschema.validate(rec, SCHEMA)  # M0 schema conformance
    assert rec["transitions"][0]["state"] == "PENDING"
    assert rec["results_dir"] == f"/lustre/fsw/users/me/results/{job_id}"
    assert rec["backend_ref"] is None and rec["terminal_state"] is None
    assert rec["redacted"] is True


def test_open_id_is_the_printed_handle_and_matches_pattern(capsys):
    job_id = open_job(capsys)
    assert re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", job_id)
    assert job_id.startswith("cosmos3-train-")


def test_open_explicit_results_dir_used_verbatim(state_dir, capsys):
    argv = [
        "open",
        "--platform",
        "docker",
        "--image",
        "nvcr.io/x/y:1",  # unpinned: test fixture
        "--network-arch",
        "cosmos3",
        "--action",
        "evaluate",
        "--storage-tier",
        "C",
        "--results-dir",
        "/data/out/run1",
    ]
    assert jr.main(argv) == 0
    job_id = capsys.readouterr().out.strip()
    assert read_record(state_dir, job_id)["results_dir"] == "/data/out/run1"


def test_open_requires_exactly_one_results_arg(capsys):
    base = [
        "open",
        "--platform",
        "docker",
        "--image",
        "nvcr.io/x/y:1",  # unpinned: test fixture
        "--network-arch",
        "d",
        "--action",
        "a",
        "--storage-tier",
        "C",
    ]
    with pytest.raises(SystemExit):
        jr.main(base)  # neither
    with pytest.raises(SystemExit):
        jr.main(base + ["--results-dir", "/a", "--results-root", "/b"])  # both


def test_open_never_overwrites_existing_record(state_dir, capsys, monkeypatch):
    class FixedUUID:
        hex = "aaaaaa" * 6

    monkeypatch.setattr(jr.uuid, "uuid4", lambda: FixedUUID)
    open_job(capsys)
    with pytest.raises(SystemExit, match="already exists"):
        open_job(capsys)


def test_open_sanitizes_weird_arch_names(state_dir, capsys):
    argv = [
        "open",
        "--platform",
        "docker",
        "--image",
        "nvcr.io/x/y:1",  # unpinned: test fixture
        "--network-arch",
        "visual changenet!",
        "--action",
        "train",
        "--storage-tier",
        "C",
        "--results-dir",
        "/data/out",
    ]
    assert jr.main(argv) == 0
    job_id = capsys.readouterr().out.strip()
    assert re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", job_id)


def test_open_with_parent_and_retry_links(state_dir, capsys):
    job_id = open_job(capsys, parent_job="automl-exp-7", retry_of="cosmos3-train-000000")
    rec = read_record(state_dir, job_id)
    jsonschema.validate(rec, SCHEMA)
    assert rec["parent_job"] == "automl-exp-7"
    assert rec["retry_of"] == "cosmos3-train-000000"


# --------------------------------------------------------------------------- #
# mark
# --------------------------------------------------------------------------- #


def test_mark_appends_and_sets_backend_ref(state_dir, capsys):
    job_id = open_job(capsys)
    assert (
        jr.main(
            ["mark", job_id, "--state", "RUNNING", "--message", "sbatch 4211337 dispatched", "--backend-ref", "4211337"]
        )
        == 0
    )
    rec = read_record(state_dir, job_id)
    jsonschema.validate(rec, SCHEMA)
    assert [t["state"] for t in rec["transitions"]] == ["PENDING", "RUNNING"]
    assert rec["backend_ref"] == "4211337"
    assert rec["terminal_state"] is None


def test_terminal_mark_stamps_and_is_immutable(state_dir, capsys):
    job_id = open_job(capsys)
    assert (
        jr.main(
            [
                "mark",
                job_id,
                "--state",
                "ERROR",
                "--message",
                "NODE_FAIL",
                "--source",
                "poller",
                "--err-class",
                "ERR_INFRA",
            ]
        )
        == 0
    )
    rec = read_record(state_dir, job_id)
    jsonschema.validate(rec, SCHEMA)
    assert rec["terminal_state"] == "ERROR"
    assert rec["terminal_write_by"] == "poller"
    assert rec["err_class"] == "ERR_INFRA"

    # idempotent re-mark of the SAME terminal state
    n_before = len(rec["transitions"])
    assert jr.main(["mark", job_id, "--state", "ERROR"]) == 0
    assert len(read_record(state_dir, job_id)["transitions"]) == n_before

    # any DIFFERENT state after terminal is refused
    with pytest.raises(SystemExit, match="terminal"):
        jr.main(["mark", job_id, "--state", "RUNNING"])


def test_mark_missing_record_fails(capsys):
    with pytest.raises(SystemExit, match="no job-record"):
        jr.main(["mark", "cosmos3-train-zzzzzz", "--state", "RUNNING"])


def test_traversal_id_rejected(capsys):
    with pytest.raises(SystemExit, match="invalid job id"):
        jr.main(["show", "../../etc/passwd"])
    with pytest.raises(SystemExit, match="invalid job id"):
        jr.main(["mark", ".hidden", "--state", "RUNNING"])


def test_invalid_state_rejected_by_cli(capsys):
    with pytest.raises(SystemExit):
        jr.main(["mark", "x-y-z", "--state", "QUEUED"])  # not in the vocabulary


# --------------------------------------------------------------------------- #
# redaction on write
# --------------------------------------------------------------------------- #


def test_secret_in_message_never_persisted(state_dir, capsys):
    job_id = open_job(capsys)
    jr.main(["mark", job_id, "--state", "ERROR", "--message", f"launch failed: docker run -e NGC_KEY={SECRET} img"])
    raw = (state_dir / "jobs" / f"{job_id}.json").read_text()
    assert SECRET not in raw
    assert "NGC_KEY" in raw  # the variable NAME survives for diagnosis


def test_show_output_is_redacted_and_list_works(state_dir, capsys):
    job_id = open_job(capsys)
    jr.main(["mark", job_id, "--state", "RUNNING", "--message", f"HF_TOKEN={SECRET} leaked"])
    jr.main(["show", job_id])
    out = capsys.readouterr().out
    assert SECRET not in out

    jr.main(["list"])
    out = capsys.readouterr().out
    assert job_id in out and "RUNNING" in out and "slurm" in out


# --------------------------------------------------------------------------- #
# concurrency: agent + detached poller marking the same record
# --------------------------------------------------------------------------- #


def test_concurrent_marks_lose_nothing(state_dir, capsys):
    job_id = open_job(capsys)
    N = 25

    def mark(i):
        return jr.main(["mark", job_id, "--state", "RUNNING", "--message", f"tick-{i}", "--source", "poller"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        assert all(r == 0 for r in ex.map(mark, range(N)))

    rec = read_record(state_dir, job_id)
    ticks = {t["message"] for t in rec["transitions"] if t["message"].startswith("tick-")}
    assert ticks == {f"tick-{i}" for i in range(N)}  # flock: no lost updates
    jsonschema.validate(rec, SCHEMA)


# --------------------------------------------------------------------------- #
# red-team regressions (M1 writer)
# --------------------------------------------------------------------------- #


def test_rt_tail_leak_in_message_redacted_whole(state_dir, capsys):
    job_id = open_job(capsys)
    # value contains & ? , — the shell redactor stopped at the first; field mode must not
    jr.main(["mark", job_id, "--state", "ERROR", "--message", "AWS_SECRET_ACCESS_KEY=headPART&tailSECRETpart,more?q=z"])
    raw = (state_dir / "jobs" / f"{job_id}.json").read_text()
    assert "tailSECRETpart" not in raw and "headPART" not in raw


def test_rt_out_of_pattern_secrets_redacted(state_dir, capsys):
    job_id = open_job(capsys)
    jr.main(
        ["mark", job_id, "--state", "ERROR", "--message", 'nvapi-BARESECRETa1b2c3d4e5 and {"password": "hunter2pass"}']
    )
    raw = (state_dir / "jobs" / f"{job_id}.json").read_text()
    assert "nvapi-BARESECRETa1b2c3d4e5" not in raw
    assert "hunter2pass" not in raw


def test_rt_url_userinfo_redacted_in_results_dir(state_dir, capsys):
    argv = [
        "open",
        "--platform",
        "docker",
        "--image",
        "nvcr.io/x/y:1",  # unpinned: test fixture
        "--network-arch",
        "cosmos3",
        "--action",
        "train",
        "--storage-tier",
        "C",
        "--results-dir",
        "https://user:leakSECRETpw@host/bucket/out",
    ]
    assert jr.main(argv) == 0
    job_id = capsys.readouterr().out.strip()
    raw = (state_dir / "jobs" / f"{job_id}.json").read_text()
    assert "leakSECRETpw" not in raw and "host/bucket/out" in raw  # host kept, pw gone


def test_rt_legit_results_dir_with_credword_segment_untouched(state_dir, capsys):
    argv = [
        "open",
        "--platform",
        "slurm",
        "--image",
        "nvcr.io/x/y:1",  # unpinned: test fixture
        "--network-arch",
        "cosmos3",
        "--action",
        "train",
        "--storage-tier",
        "A",
        "--results-dir",
        "/lustre/exp/token=v1/keyframes/run",
    ]
    assert jr.main(argv) == 0
    job_id = capsys.readouterr().out.strip()
    rec = read_record(state_dir, job_id)
    assert rec["results_dir"] == "/lustre/exp/token=v1/keyframes/run"  # NOT corrupted


def test_rt_empty_required_fields_rejected_before_id(capsys):
    for bad_arg in (["--image", ""], ["--network-arch", "  "], ["--action", ""]):
        argv = [
            "open",
            "--platform",
            "docker",
            "--image",
            "img:1",
            "--network-arch",
            "cosmos3",
            "--action",
            "train",
            "--storage-tier",
            "C",
            "--results-dir",
            "/tmp/o",
        ]
        # overwrite the targeted field
        i = argv.index(bad_arg[0])
        argv[i + 1] = bad_arg[1]
        with pytest.raises(SystemExit, match="must be non-empty"):
            jr.main(argv)


def test_rt_state_dir_set_but_empty_refused(monkeypatch, capsys):
    monkeypatch.setenv("COSMOS_STATE_DIR", "")
    with pytest.raises(SystemExit, match="set but empty"):
        jr.main(["list"])


def test_rt_state_dir_relative_refused(monkeypatch, capsys):
    monkeypatch.setenv("COSMOS_STATE_DIR", "relative/state")
    with pytest.raises(SystemExit, match="absolute"):
        jr.main(["list"])


def test_default_state_root_is_outside_the_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("COSMOS_STATE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user-home")
    assert jr.state_root() == tmp_path / "user-home" / ".cosmos"


def test_configured_state_root_is_used_without_moving_existing_records(state_dir):
    assert jr.state_root() == state_dir


def test_rt_leading_underscore_arch_makes_valid_id(state_dir, capsys):
    argv = [
        "open",
        "--platform",
        "docker",
        "--image",
        "img:1",
        "--network-arch",
        "_weird",
        "--action",
        "train",
        "--storage-tier",
        "C",
        "--results-dir",
        "/tmp/o",
    ]
    assert jr.main(argv) == 0
    job_id = capsys.readouterr().out.strip()
    assert re.match(r"^[A-Za-z0-9]", job_id)  # id starts alphanumeric


def test_rt_terminal_err_class_enrichment(state_dir, capsys):
    job_id = open_job(capsys)
    jr.main(["mark", job_id, "--state", "ERROR", "--source", "poller", "--message", "OOM"])
    # the agent classifies err_class AFTER the poller marked ERROR
    assert jr.main(["mark", job_id, "--state", "ERROR", "--err-class", "ERR_INFRA"]) == 0
    rec = read_record(state_dir, job_id)
    assert rec["err_class"] == "ERR_INFRA"
    jsonschema.validate(rec, SCHEMA)


# ---------------------------------------------------------------------------
# The platform field is an OPEN slug, not a registry
# ---------------------------------------------------------------------------


def test_external_platform_slug_is_accepted_and_schema_valid(capsys, state_dir):
    """An externally installed platform skill (e.g. custom-platform) must be
    able to open records under its short name with NO bank edit."""
    job_id = open_job(capsys, platform="kratos")
    record = read_record(state_dir, job_id)
    assert record["platform"] == "kratos"
    jsonschema.validate(record, SCHEMA)


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("Docker ", "docker"),  # case + whitespace variance
        ("SLURM", "slurm"),
        ("My_Platform", "my-platform"),  # separator style
        ("-kratos-", "kratos"),  # stray hyphens
        ("my  platform", "my-platform"),
    ],
)
def test_platform_input_is_normalized_not_rejected(capsys, state_dir, raw, canonical):
    # --platform=<raw> form: a raw value may start with '-' (argparse-safe).
    assert (
        jr.main(
            [
                "open",
                f"--platform={raw}",
                "--image",
                "cosmos-framework:local",  # unpinned: test fixture
                "--network-arch",
                "cosmos3",
                "--action",
                "train",
                "--storage-tier",
                "A",
                "--results-root",
                "/tmp/results",
            ]
        )
        == 0
    )
    job_id = capsys.readouterr().out.strip()
    record = read_record(state_dir, job_id)
    assert record["platform"] == canonical
    jsonschema.validate(record, SCHEMA)


@pytest.mark.parametrize("garbage", ["", "   ", "k8s!", "a" * 40, "$(id)"])
def test_irreducible_platform_garbage_is_rejected(garbage):
    """Only values that cannot be normalized into a name are refused —
    silently mangling garbage into a plausible name would be worse."""
    with pytest.raises(SystemExit):
        jr.main(
            [
                "open",
                "--platform",
                garbage,
                "--image",
                "x",
                "--network-arch",
                "d",
                "--action",
                "t",
                "--storage-tier",
                "A",
                "--results-root",
                "/tmp/x",
            ]
        )
