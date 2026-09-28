# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render + security tests for the single-node Kubernetes Job template.

The template exists so the security-critical shape is faithful: creds come from
a per-job Secret via envFrom.secretRef (NEVER inline plaintext env values — the
SDK handler's V1EnvVar mistake), GPU is a proper resource limit, /dev/shm is
sized (64Mi default silently hangs NCCL), the Job self-cleans via TTL, and a
rendered instance parses as valid k8s YAML with no unsubstituted markers.
"""

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "templates/k8s/single-pod-job.yaml.tmpl"
sys.path.insert(0, str(REPO / "scripts"))
import redact_secrets  # noqa: E402

BASE = {
    "JOB_NAME": "dino-train-a1b2c3",
    "TTL_SECONDS": "3600",
    "IMAGE_PULL_SECRET": "ngc-pull-secret",
    "IMAGE": "cosmos-framework:local",  # unpinned: test fixture
    "COMMAND": "dino train -e /data/specs/spec.yaml",
    "NUM_GPUS": "1",
    "CRED_SECRET": "cosmos-creds-dino-train-a1b2c3",
    "RESULTS_DIR": "/data/results/dino-train-a1b2c3",
    "MOUNT_PATH": "/data",
    "SHM_SIZE": "16Gi",
    "PVC_CLAIM": "edgeai-datasets",
}


def render(overrides=None):
    text = TEMPLATE.read_text()
    for k, v in {**BASE, **(overrides or {})}.items():
        text = text.replace(f"@@{k}@@", v)
    return text


def load(overrides=None):
    return yaml.safe_load(render(overrides))


def test_all_markers_substituted():
    assert re.findall(r"@@[A-Z_]+@@", render()) == []


def test_renders_valid_k8s_job():
    doc = load()
    assert doc["kind"] == "Job"
    assert doc["metadata"]["name"] == "dino-train-a1b2c3"
    assert doc["spec"]["backoffLimit"] == 0
    assert doc["spec"]["ttlSecondsAfterFinished"] == 3600
    assert doc["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_gpu_resource_limit():
    c = load()["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_dev_shm_sized_memory():
    vols = {v["name"]: v for v in load()["spec"]["template"]["spec"]["volumes"]}
    assert vols["dshm"]["emptyDir"]["medium"] == "Memory"
    assert vols["dshm"]["emptyDir"]["sizeLimit"] == "16Gi"  # not the 64Mi default


def test_creds_via_secretref_never_inline():
    c = load()["spec"]["template"]["spec"]["containers"][0]
    # creds arrive via envFrom.secretRef — only the secret NAME, no values
    assert c["envFrom"][0]["secretRef"]["name"] == "cosmos-creds-dino-train-a1b2c3"
    # the only inline env is the non-secret results root
    inline = {e["name"] for e in c.get("env", [])}
    assert inline == {"COSMOS_RESULTS_ROOT"}
    assert not any(k in inline for k in ("AWS_SECRET_ACCESS_KEY", "NGC_KEY", "HF_TOKEN"))


def test_rendered_manifest_lints_clean():
    # even a literal secret smuggled into the command must be caught by the gate,
    # and a clean render must pass
    assert redact_secrets.scan(render()) == []
    bad = render({"COMMAND": "train -e s.yaml && export NGC_KEY=nvapi-LEAKED1234567890"})
    assert redact_secrets.scan(bad)


def test_ttl_present_so_outputs_must_be_bound_before_deletion():
    # TTL deletes the Job — the results_dir must be a persistent mount path, and
    # the template wires COSMOS_RESULTS_ROOT to the mounted-volume path
    c = load()["spec"]["template"]["spec"]["containers"][0]
    rr = next(e["value"] for e in c["env"] if e["name"] == "COSMOS_RESULTS_ROOT")
    mount = next(m["mountPath"] for m in c["volumeMounts"] if m["name"] == "data")
    assert rr.startswith(mount)  # results land on the mounted (surviving) volume


def test_no_cross_node_rendezvous_singlenode():
    # single-pod must not carry the multi-node rendezvous env / Service (that is M7)
    text = render()
    assert "MASTER_ADDR" not in text and "Indexed" not in text and "subdomain" not in text


def test_container_command_uses_posix_sh():
    """The container command must run under a shell every image actually has.

    The template hardcoded `/bin/bash -lc`, so any image without bash could not
    start — busybox and distroless ship only /bin/sh. That is precisely the
    small image a cheap lifecycle smoke or eval reaches for, so the bug bit the
    case the template most needs to support. Verified live: a busybox Job from
    the rendered template reaches Completed only with /bin/sh.

    Pinned here because the same defect shipped in both k8s templates, so it is
    a repeat pattern rather than a one-off.
    """
    job = load()
    cmd = job["spec"]["template"]["spec"]["containers"][0]["command"]
    assert cmd[0] == "/bin/sh", (
        f"container command interpreter is {cmd[0]!r}; use /bin/sh so images "
        f"without bash (busybox, distroless) can start"
    )
    assert "-lc" not in cmd, (
        "drop -l: busybox sh treats login mode inconsistently, and container "
        "images set PATH via ENV rather than profile scripts"
    )
