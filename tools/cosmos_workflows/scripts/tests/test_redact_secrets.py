# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for redact_secrets.

The invariant under test everywhere: literal credential material is detected,
and it NEVER appears in lint findings or redacted output. Sanctioned patterns
(scoped $VAR references, --password-stdin, valueFrom.secretKeyRef, docker run
port mappings) must pass untouched.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import redact_secrets as rs  # noqa: E402

SECRET = "nvapi-SUPERSECRET123"


def kinds(text):
    return [(f.kind, f.name) for f in rs.scan(text)]


# --------------------------------------------------------------------------- #
# env assignments
# --------------------------------------------------------------------------- #


def test_literal_env_assignment_flagged():
    text = f"NGC_KEY={SECRET} docker run nvcr.io/img"
    fs = rs.scan(text)
    assert len(fs) == 1 and fs[0].name == "NGC_KEY"


def test_scoped_env_reference_passes():
    # the sanctioned data staging pattern
    text = 'AWS_ACCESS_KEY_ID="$ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$SECRET_KEY" aws s3 ls'
    assert rs.scan(text) == []


def test_docker_dash_e_literal_flagged_but_passthrough_ok():
    assert len(rs.scan(f"docker run -e HF_TOKEN=hf_{SECRET} img")) == 1
    assert rs.scan("docker run -e HF_TOKEN -e NGC_KEY img") == []


def test_export_literal_flagged_reference_ok():
    assert len(rs.scan(f"export SECRET_KEY='{SECRET}'")) == 1
    assert rs.scan('export SECRET_KEY="$SECRET_KEY"') == []


def test_non_secret_env_names_pass():
    assert rs.scan("RESULTS_DIR=/results NUM_GPUS=4 EPOCHS=10 train.sh") == []


def test_empty_value_passes():
    assert rs.scan("NGC_KEY= docker run img") == []


# --------------------------------------------------------------------------- #
# CLI flags
# --------------------------------------------------------------------------- #


def test_docker_login_dash_p_flagged():
    text = f"docker login nvcr.io -u '$oauthtoken' -p {SECRET}"  # lint-ok: secret-on-argv
    fs = rs.scan(text)
    assert [(f.kind, f.name) for f in fs] == [("credential passed as CLI argument", "-p")]


def test_docker_run_port_mapping_passes():
    assert rs.scan("docker run -d -p 8080:8080 -p 9090:90 img serve") == []


def test_password_stdin_passes_but_literal_flagged():
    assert rs.scan("docker login nvcr.io -u '$oauthtoken' --password-stdin") == []
    assert len(rs.scan(f"docker login --password {SECRET}")) >= 1  # lint-ok: secret-on-argv
    assert len(rs.scan(f"hf auth login --token=hf_{SECRET}")) == 1


def test_flag_with_var_reference_passes():
    assert rs.scan('hf auth login --token "$HF_TOKEN"') == []


# --------------------------------------------------------------------------- #
# k8s manifests
# --------------------------------------------------------------------------- #

K8S_BAD = f"""
      env:
        - name: MODEL_PATH
          value: /models/cosmos3.pth
        - name: AWS_SECRET_ACCESS_KEY
          value: {SECRET}
"""

K8S_GOOD = """
      env:
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: cosmos-creds-j123
              key: secret-key
"""


def test_k8s_inline_secret_value_flagged_nonsecret_passes():
    fs = rs.scan(K8S_BAD)
    assert [(f.kind.split(" ")[0], f.name) for f in fs] == [("inline", "AWS_SECRET_ACCESS_KEY")]


def test_k8s_secretkeyref_passes():
    assert rs.scan(K8S_GOOD) == []


# --------------------------------------------------------------------------- #
# presigned URLs
# --------------------------------------------------------------------------- #


def test_presigned_url_flagged_and_redacted():
    url = f"https://s3.io/bkt/x?X-Amz-Credential=AKIA123&X-Amz-Signature={SECRET}"
    fs = rs.scan(url)
    assert {f.name for f in fs} == {"X-Amz-Credential", "X-Amz-Signature"}
    red = rs.redact(url)
    assert SECRET not in red and "AKIA123" not in red
    assert "https://s3.io/bkt/x" in red  # rest of URL intact


# --------------------------------------------------------------------------- #
# redact mode
# --------------------------------------------------------------------------- #


def test_redact_env_to_placeholder():
    red = rs.redact(f"NGC_KEY={SECRET} docker run img")
    assert red == 'NGC_KEY="${NGC_KEY}" docker run img'


def test_redact_flag_and_login_p():
    red = rs.redact(f"docker login nvcr.io -u '$oauthtoken' -p {SECRET}")  # lint-ok: secret-on-argv
    assert SECRET not in red and "-p <redacted>" in red


def test_redact_k8s_value():
    red = rs.redact(K8S_BAD)
    assert SECRET not in red and "secretKeyRef" in red
    assert "value: /models/cosmos3.pth" in red  # non-secret value untouched


def test_redact_idempotent():
    once = rs.redact(f"export HF_TOKEN={SECRET}\ndocker login -p {SECRET}")  # lint-ok: secret-on-argv
    assert rs.redact(once) == once
    assert rs.scan(once) == []  # redacted output lints clean


def test_redact_clean_text_unchanged():
    text = 'AWS_ACCESS_KEY_ID="$ACCESS_KEY" aws s3 sync /r s3://b/results/j1/ --exclude ".cosmos/*"\n'
    assert rs.redact(text) == text


# --------------------------------------------------------------------------- #
# the global invariant: secrets never leak through either mode
# --------------------------------------------------------------------------- #

FIXTURES = [
    f"NGC_KEY={SECRET} docker run img",
    f"docker run -e AWS_SECRET_ACCESS_KEY='{SECRET}' img",
    f"docker login -u x -p {SECRET}",  # lint-ok: secret-on-argv
    f"hf download --token hf_{SECRET}",
    K8S_BAD,
    f'export SECRET_KEY="{SECRET}"',
    f"curl 'https://s3.io/x?X-Amz-Signature={SECRET}'",
]


@pytest.mark.parametrize("text", FIXTURES)
def test_secret_never_in_findings_or_redaction(text):
    findings = rs.scan(text)
    assert findings, f"fixture should be flagged: {text[:40]}"
    rendered = "\n".join(f.render("src") for f in findings)
    assert SECRET not in rendered
    assert SECRET not in rs.redact(text)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_lint_exit_codes(tmp_path, capsys):
    bad = tmp_path / "cmd.sh"
    bad.write_text(f"NGC_KEY={SECRET} run\n")
    assert rs.main(["lint", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "NGC_KEY" in err and SECRET not in err

    good = tmp_path / "ok.sh"
    good.write_text('NGC_KEY="$NGC_KEY" run\n')
    assert rs.main(["lint", str(good)]) == 0


def test_cli_redact_stdout(tmp_path, capsys):
    f = tmp_path / "cmd.sh"
    f.write_text(f"export HF_TOKEN={SECRET}\n")
    assert rs.main(["redact", str(f)]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and 'HF_TOKEN="${HF_TOKEN}"' in out


# --------------------------------------------------------------------------- #
# red-team regressions (confirmed bypasses / leaks / false positives)
# --------------------------------------------------------------------------- #


def test_rt_sbatch_export_comma_splice():
    text = f"sbatch --export=ALL,NGC_API_KEY=nvapi-{SECRET} train.sh"
    fs = rs.scan(text)
    assert any(f.name == "NGC_API_KEY" for f in fs)
    assert SECRET not in rs.redact(text)


def test_rt_single_quoted_dollar_literal():
    text = f"docker run -e MYSQL_PASSWORD='${SECRET}' img"
    assert len(rs.scan(text)) == 1
    assert SECRET not in rs.redact(text)


def test_rt_login_dash_p_concatenated():
    text = f"docker login nvcr.io -u nvidia -p{SECRET}"
    assert len(rs.scan(text)) == 1
    assert SECRET not in rs.redact(text)


@pytest.mark.parametrize(
    "payload",
    [
        f'{{"env": [{{"name": "NGC_KEY", "value": "{SECRET}"}}]}}',
        f"- {{name: NGC_KEY, value: {SECRET}}}",
        f"- {{value: {SECRET}, name: NGC_KEY}}",  # reordered
    ],
)
def test_rt_k8s_json_flow_reordered(payload):
    assert rs.scan(payload), f"should flag: {payload}"
    assert SECRET not in rs.redact(payload)


def test_rt_args_array_flag():
    text = f'        args: ["--token", "{SECRET}"]'
    assert len(rs.scan(text)) == 1
    assert SECRET not in rs.redact(text)


def test_rt_param_default_expansion():
    text = f"export NGC_API_KEY=${{NGC_API_KEY:-{SECRET}}}"
    assert len(rs.scan(text)) == 1
    assert SECRET not in rs.redact(text)


def test_rt_partial_quote_tail_no_leak():
    text = f'docker run -e NGC_API_KEY=nvapi-"{SECRET}" img'
    assert rs.scan(text)
    assert SECRET not in rs.redact(text)  # the quoted tail must not survive


def test_rt_k8s_block_scalar_no_leak():
    text = (
        "        env:\n"
        "        - name: NGC_KEY\n"
        "          value: |\n"
        f"            {SECRET}\n"
        "        - name: NEXT\n"
        "          value: keep\n"
    )
    assert rs.scan(text)
    red = rs.redact(text)
    assert SECRET not in red
    assert "value: keep" in red  # dedented sibling entry preserved


def test_rt_backslash_newline_contiguous_no_leak():
    text = f"NGC_API_KEY=nvapi-\\\n{SECRET} img"  # contiguous splice (no space)
    assert rs.scan(text)
    assert SECRET not in rs.redact(text)


def test_rt_k8s_value_dependent_env_reference_passes():
    text = "        - name: NGC_KEY\n          value: $(NGC_API_TOKEN)\n"
    assert rs.scan(text) == []  # $(VAR) is a reference, not a literal


@pytest.mark.parametrize(
    "ident",
    [
        "TOKENIZERS_PARALLELISM=false",
        "max_new_tokens=256",
        "skip_special_tokens=True",
        "num_tokens=10 tokenizer=fast",
    ],
)
def test_rt_substring_false_positive_fixed(ident):
    assert rs.scan(ident) == [], f"non-credential identifier flagged: {ident}"


def test_rt_multiline_command_continuations_preserved():
    # a normal continued docker command must round-trip unchanged
    text = "docker run --gpus all \\\n  -e NGC_KEY \\\n  -v /data:/data img\n"
    assert rs.scan(text) == []
    assert rs.redact(text) == text
