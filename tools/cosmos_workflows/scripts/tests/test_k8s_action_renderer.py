# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for producer action requests consumed by Kubernetes."""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
import yaml

REPO = Path(__file__).resolve().parents[2]
PAS_DS_IMAGE = "cosmos-framework:local"  # versions-key: images.containers.deft_pas_data_services
SCRIPT = REPO / "execution/platforms/kubernetes/scripts/render_action_job.py"
SPEC = importlib.util.spec_from_file_location("render_action_job", SCRIPT)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


RESULTS = "/localhome/user/workspace/results/run_pas_k8s"
CONFIG = RESULTS + "/config"
PATCHES = "/opt/cosmos-workflow/runtime-support"
CACHE = "/localhome/user/workspace/cache"


def pas_pool_embed_request():
    """The five-mount shape emitted for a real PAS pool_embed action."""
    output = RESULTS + "/embeddings/source/embeddings.parquet"
    return {
        "schema_version": "1",
        "platform": "kubernetes",
        "workload_image": PAS_DS_IMAGE,
        "spec_bundle": {
            "network_arch": "deft-pas",
            "action": "deft-pas-pool_embed-abc123",
            "image": PAS_DS_IMAGE,
            "mode": "args",
            "command": "embedding",
            "args": [
                "text_embeddings",
                "-e",
                "/specs/text_embed_spec.yaml",
                "input_parquet=/results/embeddings/source/source_pool.parquet",
                "output_parquet=/results/embeddings/source/embeddings.parquet",
            ],
            "compute_shape": {"gpus": 1, "nodes": 1},
        },
        "mounts": [
            {"source": RESULTS, "target": "/results", "read_only": False},
            {"source": RESULTS, "target": RESULTS, "read_only": False},
            {"source": CONFIG, "target": "/specs", "read_only": True},
            {"source": PATCHES, "target": "/patches", "read_only": True},
            {"source": CACHE, "target": "/cache", "read_only": False},
        ],
        "environment": {
            "HOME": "/tmp",
            "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface",
            "XDG_CACHE_HOME": "/cache",
        },
        "forward_env": [],
        "fresh_outputs": [output],
    }


def pas_staging_map():
    return {
        "schema_version": "1",
        "sources": [
            {"source": RESULTS, "sub_path": "jobs/abc123/results"},
            {"source": CONFIG, "sub_path": "jobs/abc123/results/config"},
            {"source": PATCHES, "sub_path": "jobs/abc123/patches"},
            {"source": CACHE, "sub_path": "jobs/abc123/cache"},
        ],
    }


def config_request(config_format="yaml", command="visual_changenet train -e {config_path}"):
    """A producer-owned mode=config request, as emitted by DEFT stages."""
    request = pas_pool_embed_request()
    bundle = request["spec_bundle"]
    bundle.pop("args")
    bundle.update(
        {
            "mode": "config",
            "command": command,
            "config_format": config_format,
            "spec": {
                "dataset": {"batch_size": 4, "class_names": ["OK", "NG"]},
                "train": {"num_epochs": 2, "use_amp": True},
            },
        }
    )
    return request


def materialize_and_stage(tmp_path, request):
    source = renderer.materialize_config(request, tmp_path / "rendered-configs")
    staging = pas_staging_map()
    staging["sources"].append(
        {
            "source": str(source),
            "sub_path": f"jobs/abc123/config/{source.name}",
        }
    )
    return source, staging


def render(request=None, staging=None, **kwargs):
    return renderer.render_action_job(
        request or pas_pool_embed_request(),
        staging or pas_staging_map(),
        job_id="cosmos-job-abc123",
        namespace="cosmos-jobs",
        pvc_claim="cosmos-workspace",
        **kwargs,
    )


def container(manifest):
    job = yaml.safe_load(manifest)
    return job, job["spec"]["template"]["spec"]["containers"][0]


def test_pas_five_mount_request_preserves_aliases_and_access_modes():
    job, action = container(render())
    mounts = {item["mountPath"]: item for item in action["volumeMounts"]}

    assert job["metadata"]["namespace"] == "cosmos-jobs"
    assert job["metadata"]["annotations"]["cosmos.nvidia.com/job-record-id"] == "cosmos-job-abc123"
    assert set(mounts) == {
        "/dev/shm",
        "/results",
        RESULTS,
        "/specs",
        "/patches",
        "/cache",
    }
    assert mounts["/results"]["subPath"] == "jobs/abc123/results"
    assert mounts[RESULTS]["subPath"] == "jobs/abc123/results"
    assert mounts["/results"]["readOnly"] is False
    assert mounts[RESULTS]["readOnly"] is False
    assert mounts["/specs"]["readOnly"] is True
    assert mounts["/patches"]["readOnly"] is True
    assert mounts["/cache"]["readOnly"] is False
    assert action["command"] == ["embedding"]
    assert action["args"] == pas_pool_embed_request()["spec_bundle"]["args"]
    assert action["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_no_credentials_or_registry_secret_means_no_dangling_secret_refs():
    job, action = container(render())
    pod = job["spec"]["template"]["spec"]
    assert pod["imagePullSecrets"] == []
    assert "envFrom" not in action


def test_forwarded_credentials_are_secret_references_not_inline_values():
    request = pas_pool_embed_request()
    request["forward_env"] = ["HF_TOKEN"]
    manifest = render(
        request=request,
        credential_secret="cosmos-creds-abc123",
        image_pull_secret="ngc-pull-secret",
    )
    job, action = container(manifest)
    assert job["spec"]["template"]["spec"]["imagePullSecrets"] == [{"name": "ngc-pull-secret"}]
    assert "envFrom" not in action
    environment = {item["name"]: item for item in action["env"]}
    assert environment["HF_TOKEN"] == {
        "name": "HF_TOKEN",
        "valueFrom": {
            "secretKeyRef": {
                "name": "cosmos-creds-abc123",
                "key": "HF_TOKEN",
            }
        },
    }


def test_secret_projects_only_explicitly_forwarded_names():
    request = pas_pool_embed_request()
    request["forward_env"] = ["HF_TOKEN", "AWS_ACCESS_KEY_ID"]
    _, action = container(render(request=request, credential_secret="shared-credential-secret"))

    assert "envFrom" not in action
    forwarded = {item["name"]: item["valueFrom"]["secretKeyRef"] for item in action["env"] if "valueFrom" in item}
    assert forwarded == {
        "HF_TOKEN": {"name": "shared-credential-secret", "key": "HF_TOKEN"},
        "AWS_ACCESS_KEY_ID": {
            "name": "shared-credential-secret",
            "key": "AWS_ACCESS_KEY_ID",
        },
    }


def test_forwarded_credentials_require_a_secret():
    request = pas_pool_embed_request()
    request["forward_env"] = ["HF_TOKEN"]
    with pytest.raises(renderer.RenderError, match="credential-secret"):
        render(request=request)


def test_credential_secret_is_rejected_when_no_names_are_approved():
    with pytest.raises(renderer.RenderError, match="must be omitted"):
        render(credential_secret="unexpected-shared-secret")


def test_inline_and_forwarded_name_collision_is_rejected():
    request = pas_pool_embed_request()
    request["environment"]["HF_TOKEN"] = "must-not-be-inline"
    request["forward_env"] = ["HF_TOKEN"]
    with pytest.raises(renderer.RenderError, match="must not be present"):
        render(request=request, credential_secret="cosmos-creds-abc123")


@pytest.mark.parametrize("schema_version", [None, "0", "2", 1])
def test_unsupported_request_schema_version_is_rejected(schema_version):
    request = pas_pool_embed_request()
    if schema_version is None:
        request.pop("schema_version")
    else:
        request["schema_version"] = schema_version
    with pytest.raises(renderer.RenderError, match="request.schema_version"):
        render(request=request)


def test_command_tokens_are_not_interpolated_into_a_shell_string():
    request = pas_pool_embed_request()
    request["spec_bundle"]["args"].append("literal=$(do-not-run); 'quoted'")
    _, action = container(render(request=request))
    assert action["command"] == ["embedding"]
    assert action["args"][-1] == "literal=$(do-not-run); 'quoted'"


def test_explicit_args_mode_shell_is_rendered_as_native_argv():
    request = pas_pool_embed_request()
    request["spec_bundle"]["command"] = "bash -lc"
    request["spec_bundle"]["args"] = ["printf '%s\\n' \"$HOME\""]
    _, action = container(render(request=request))
    assert action["command"] == ["bash", "-lc"]
    assert action["args"] == ["printf '%s\\n' \"$HOME\""]


def test_config_mode_materializes_stages_and_mounts_the_exact_spec(tmp_path):
    request = config_request()
    source, staging = materialize_and_stage(tmp_path, request)

    _, action = container(
        render(
            request=request,
            staging=staging,
            config_source=source,
        )
    )

    expected = "/cosmos-action-config/spec-" + source.stem.removeprefix("cosmos-action-config-") + ".yaml"
    assert action["command"] == ["visual_changenet", "train", "-e", expected]
    assert action["args"] == []
    assert "{config_path}" not in json.dumps(action)
    config_mount = next(mount for mount in action["volumeMounts"] if mount["mountPath"] == expected)
    assert config_mount == {
        "name": "workspace",
        "mountPath": expected,
        "subPath": f"jobs/abc123/config/{source.name}",
        "readOnly": True,
    }
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == request["spec_bundle"]["spec"]


@pytest.mark.parametrize("config_format", ["json", "yaml"])
def test_json_compatible_config_materialization_is_canonical_and_idempotent(tmp_path, config_format):
    request = config_request(config_format)
    request["spec_bundle"]["spec"]["optional"] = None
    first = renderer.materialize_config(request, tmp_path / "configs")
    second = renderer.materialize_config(request, tmp_path / "configs")

    assert first == second
    assert len(first.stem.removeprefix("cosmos-action-config-")) == 64
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert json.loads(first.read_text(encoding="utf-8")) == request["spec_bundle"]["spec"]


def test_toml_config_materialization_preserves_nested_values(tmp_path):
    request = config_request("toml", "cosmos-rl --config {config_path} train.py")
    request["spec_bundle"]["spec"]["train"]["schedulers"] = [
        {"name": "warmup", "steps": 5},
        {"name": "cosine", "steps": 20},
    ]
    source = renderer.materialize_config(request, tmp_path / "configs")

    with source.open("rb") as handle:
        assert tomllib.load(handle) == request["spec_bundle"]["spec"]


def test_toml_rejects_null_instead_of_silently_changing_it(tmp_path):
    request = config_request("toml")
    request["spec_bundle"]["spec"]["train"]["optional"] = None
    with pytest.raises(renderer.RenderError, match="NoneType"):
        renderer.materialize_config(request, tmp_path / "configs")


def test_config_mode_requires_the_materialize_stage_render_sequence():
    with pytest.raises(renderer.RenderError, match="materialize-config"):
        render(request=config_request())


def test_config_source_must_be_declared_by_the_staging_receipt(tmp_path):
    request = config_request()
    source = renderer.materialize_config(request, tmp_path / "configs")
    with pytest.raises(renderer.RenderError, match="no staged PVC subPath"):
        render(request=request, config_source=source)


def test_stale_or_altered_config_source_is_rejected(tmp_path):
    request = config_request()
    source, staging = materialize_and_stage(tmp_path, request)
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(renderer.RenderError, match="does not match"):
        render(request=request, staging=staging, config_source=source)


def test_config_source_symlink_is_rejected(tmp_path):
    request = config_request()
    source, staging = materialize_and_stage(tmp_path, request)
    link = tmp_path / "config-link.yaml"
    link.symlink_to(source)
    staging["sources"][-1]["source"] = str(link)
    with pytest.raises(renderer.RenderError, match="cannot open|non-symlink"):
        render(request=request, staging=staging, config_source=link)


def test_config_command_must_contain_the_contract_placeholder(tmp_path):
    request = config_request(command="visual_changenet train")
    source, staging = materialize_and_stage(tmp_path, request)
    with pytest.raises(renderer.RenderError, match=r"requires \{config_path\}"):
        render(request=request, staging=staging, config_source=source)


def test_args_mode_rejects_a_config_source(tmp_path):
    source = tmp_path / "unexpected.yaml"
    source.write_text("{}\n", encoding="utf-8")
    staging = pas_staging_map()
    staging["sources"].append({"source": str(source), "sub_path": "jobs/abc123/unexpected.yaml"})
    with pytest.raises(renderer.RenderError, match="only for spec_bundle.mode=config"):
        render(staging=staging, config_source=source)


def test_config_shell_script_is_preserved_for_cosmos_rl(tmp_path):
    command = (
        "hook=$(python -c 'import cosmos_rl; print(cosmos_rl.__file__)')\n"
        'test -n "$hook"\n'
        'exec cosmos-rl --config {config_path} "$hook"'
    )
    request = config_request("toml", command)
    source, staging = materialize_and_stage(tmp_path, request)
    _, action = container(render(request=request, staging=staging, config_source=source))

    assert action["command"] == ["/bin/sh", "-c"]
    assert action["args"][1] == "cosmos-action"
    assert "hook=$(python" in action["args"][0]
    assert "{config_path}" not in action["args"][0]
    assert "/cosmos-action-config/spec-" in action["args"][0]


def test_empty_argument_and_environment_value_are_preserved():
    request = pas_pool_embed_request()
    request["spec_bundle"]["args"].append("")
    request["environment"]["OPTIONAL_SETTING"] = ""
    _, action = container(render(request=request))
    assert action["args"][-1] == ""
    environment = {item["name"]: item["value"] for item in action["env"]}
    assert environment["OPTIONAL_SETTING"] == ""


def test_duplicate_source_aliases_need_only_one_staging_entry():
    manifest = render()
    _, action = container(manifest)
    workspace_mounts = [item for item in action["volumeMounts"] if item["name"] == "workspace"]
    assert len(workspace_mounts) == 5
    assert sum(item["subPath"] == "jobs/abc123/results" for item in workspace_mounts) == 2


def test_missing_staging_mapping_is_rejected():
    staging = pas_staging_map()
    staging["sources"] = [row for row in staging["sources"] if row["source"] != PATCHES]
    with pytest.raises(renderer.RenderError, match="no staged PVC subPath"):
        render(staging=staging)


def test_undeclared_staging_mapping_is_rejected():
    staging = pas_staging_map()
    staging["sources"].append({"source": "/undeclared/input", "sub_path": "jobs/abc123/extra"})
    with pytest.raises(renderer.RenderError, match="undeclared mount sources"):
        render(staging=staging)


@pytest.mark.parametrize("sub_path", ["/absolute", "../escape", "a/../escape", ".", "a\\b"])
def test_unsafe_pvc_subpaths_are_rejected(sub_path):
    staging = pas_staging_map()
    staging["sources"][0]["sub_path"] = sub_path
    with pytest.raises(renderer.RenderError, match="sub_path|subPath|relative|traversal"):
        render(staging=staging)


def test_duplicate_mount_target_is_rejected():
    request = pas_pool_embed_request()
    request["mounts"][1]["target"] = "/results"
    with pytest.raises(renderer.RenderError, match="repeats Kubernetes mount target"):
        render(request=request)


def test_fresh_outputs_require_a_writable_covering_mount():
    request = pas_pool_embed_request()
    request["mounts"][0]["read_only"] = True
    request["mounts"][1]["read_only"] = True
    with pytest.raises(renderer.RenderError, match="not covered by a writable"):
        render(request=request)


def test_staging_map_rejects_duplicate_sources():
    staging = pas_staging_map()
    staging["sources"].append(dict(staging["sources"][0]))
    with pytest.raises(renderer.RenderError, match="repeats source"):
        render(staging=staging)


def test_distinct_sources_cannot_share_one_exact_pvc_subpath():
    staging = pas_staging_map()
    staging["sources"][1]["sub_path"] = staging["sources"][0]["sub_path"]
    with pytest.raises(renderer.RenderError, match="one PVC subPath"):
        render(staging=staging)


def test_real_pas_job_record_id_is_normalized_without_losing_identity():
    job_id = "data-services-deft-pas-pool_embed-0123456789abcdef-1234567890abcdef-abcdef"
    manifest = renderer.render_action_job(
        pas_pool_embed_request(),
        pas_staging_map(),
        job_id=job_id,
        namespace="cosmos-jobs",
        pvc_claim="cosmos-workspace",
    )
    job = yaml.safe_load(manifest)
    name = job["metadata"]["name"]
    assert len(name) <= renderer.MAX_JOB_NAME
    assert renderer.DNS_LABEL_RE.fullmatch(name)
    assert "_" not in name
    assert job["metadata"]["annotations"]["cosmos.nvidia.com/job-record-id"] == job_id
    assert renderer.kubernetes_job_name(job_id) == name


def test_normalized_names_cannot_collide_after_character_replacement():
    underscored = renderer.kubernetes_job_name("action_with_underscore")
    hyphenated = renderer.kubernetes_job_name("action-with-underscore")
    assert underscored != hyphenated


def test_cli_renders_the_same_contract(tmp_path):
    request_path = tmp_path / "action.json"
    staging_path = tmp_path / "staging.json"
    request_path.write_text(json.dumps(pas_pool_embed_request()), encoding="utf-8")
    staging_path.write_text(json.dumps(pas_staging_map()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "render",
            "--request",
            str(request_path),
            "--staging-map",
            str(staging_path),
            "--job-id",
            "cosmos-job-abc123",
            "--namespace",
            "cosmos-jobs",
            "--pvc-claim",
            "cosmos-workspace",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    job = yaml.safe_load(completed.stdout)
    assert job["kind"] == "Job"
    assert job["metadata"]["name"] == "cosmos-job-abc123"


def test_cli_materializes_then_renders_a_config_mode_request(tmp_path):
    request = config_request()
    request_path = tmp_path / "action.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    materialized = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "materialize-config",
            "--request",
            str(request_path),
            "--output-dir",
            str(tmp_path / "configs"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert materialized.returncode == 0, materialized.stderr
    source = Path(materialized.stdout.strip())
    staging = pas_staging_map()
    staging["sources"].append(
        {
            "source": str(source),
            "sub_path": f"jobs/abc123/config/{source.name}",
        }
    )
    staging_path = tmp_path / "staging.json"
    staging_path.write_text(json.dumps(staging), encoding="utf-8")

    rendered = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "render",
            "--request",
            str(request_path),
            "--staging-map",
            str(staging_path),
            "--config-source",
            str(source),
            "--job-id",
            "cosmos-job-abc123",
            "--namespace",
            "cosmos-jobs",
            "--pvc-claim",
            "cosmos-workspace",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert rendered.returncode == 0, rendered.stderr
    _, action = container(rendered.stdout)
    assert action["command"][:3] == ["visual_changenet", "train", "-e"]
    assert action["volumeMounts"][-1]["readOnly"] is True


def test_name_cli_returns_backend_object_name():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "name", "--job-id", "PAS.pool_embed_01"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == renderer.kubernetes_job_name("PAS.pool_embed_01")
