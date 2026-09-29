# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema tests for the shared Cosmos execution artifacts.

The spec-bundle cases mirror real skill_info.yaml shapes: the config-mode
bundle is native SFT (nested TOML); the args-mode bundle is video inference. The load-bearing rejections: dotted spec keys
at ANY depth (the #1 authoring mistake), mode cross-contamination, unresolved
image keys, and job-records that skip the results_dir-at-submit invariant.
"""

import copy
import json
from pathlib import Path

import jsonschema
import pytest

REF = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((REF / name).read_text())


@pytest.fixture(scope="module")
def spec_schema():
    return load("spec_bundle.schema.json")


@pytest.fixture(scope="module")
def record_schema():
    return load("job_record.schema.json")


def ok(instance, schema):
    jsonschema.validate(instance, schema)


def bad(instance, schema):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance, schema)


# --------------------------------------------------------------------------- #
# spec-bundle fixtures
# --------------------------------------------------------------------------- #

TRAIN_BUNDLE = {
    "network_arch": "cosmos3",
    "action": "train",
    "image": "cosmos-framework:local",  # unpinned: test fixture
    "mode": "config",
    "command": "python -m cosmos_framework.scripts.train --sft-toml {config_path}",
    "config_format": "toml",
    "spec": {
        "job": {"task": "vlm", "experiment": "cosmos_video_conversation"},
        "model": {"backbone": {"safetensors_path": "/models/base"}},
        "trainer": {"num_epochs": 12},
        "optimizer": {"lr": 0.0002},
    },
    "declared_inputs": [
        {"spec_key": "model.backbone.safetensors_path", "type": "folder", "uri": "/models/base"},
    ],
    "declared_outputs": [{"spec_key": "results_dir", "type": "folder"}],
    "upload_excludes": ["inputs/"],
    "compute_shape": {"gpus": 8, "nodes": 1},
}

ARGS_BUNDLE = {
    "network_arch": "cosmos3",
    "action": "inference",
    "image": "cosmos-framework:local",  # unpinned: test fixture
    "mode": "args",
    "command": "python -m cosmos_framework.scripts.inference",
    "args": ["-i", "/data/input.json", "-o", "/results", "--checkpoint-path", "/models/base", "--seed", "42"],
    "declared_inputs": [{"spec_key": "input", "type": "file", "uri": "s3://bkt/input.json"}],
    "declared_outputs": [{"spec_key": "results_dir", "type": "folder"}],
    "compute_shape": {"gpus": 0, "nodes": 1},
}


# --------------------------------------------------------------------------- #
# spec-bundle: accepts
# --------------------------------------------------------------------------- #


def test_cosmos3_config_bundle_valid(spec_schema):
    ok(TRAIN_BUNDLE, spec_schema)


def test_args_bundle_valid(spec_schema):
    ok(ARGS_BUNDLE, spec_schema)


def test_bundle_accepts_model_owned_action_lifecycle(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["execution"] = {
        "environment": {
            "PYTHONUNBUFFERED": "1",
            "COSMOS_JOB_ID": "{job_id}",
        },
        "pre_commands": ["python -m package.runtime_preflight"],
        "post_commands": ["python -m package.verify_results --config {config_path}"],
        "post_scope": "leader",
        "distributed": {
            "launcher": "torchrun",
            "processes_per_node": 8,
            "tasks_per_node": 1,
        },
        "supporting_files": [
            {
                "source": "scripts/checkpoint_action.py",
                "destination": "checkpoint_action.py",
                "sha256": "a" * 64,
            }
        ],
        "completion": {
            "child_exit_code_path": "{results_dir}/child_exit_code",
            "structured_status_path": "{results_dir}/status.json",
            "success_states": ["SUCCESS"],
        },
    }
    ok(b, spec_schema)


def test_bundle_rejects_secret_like_or_unhashed_lifecycle_inputs(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["execution"] = {"environment": {"bad-name": "value"}}
    bad(b, spec_schema)

    b = copy.deepcopy(TRAIN_BUNDLE)
    b["execution"] = {"supporting_files": [{"source": "scripts/helper.py", "destination": "helper.py"}]}
    bad(b, spec_schema)

    b = copy.deepcopy(TRAIN_BUNDLE)
    b["execution"] = {
        "supporting_files": [
            {
                "source": "../outside.py",
                "destination": "helper.py",
                "sha256": "a" * 64,
            }
        ]
    }
    bad(b, spec_schema)


def test_dotted_pointer_allowed_in_declared_inputs(spec_schema):
    # spec_key is a pointer; dots + [0] indices are correct THERE
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["declared_inputs"][0]["spec_key"] = "custom.sources[0].json_file"
    ok(b, spec_schema)


# --------------------------------------------------------------------------- #
# spec-bundle: the nested-not-dotted rule at every depth
# --------------------------------------------------------------------------- #


def test_reject_top_level_dotted_spec_key(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["spec"]["train.num_epochs"] = 12  # the #1 mistake
    bad(b, spec_schema)


def test_reject_nested_dotted_spec_key(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["spec"]["optimizer"]["optim.lr"] = 0.001
    bad(b, spec_schema)


def test_reject_dotted_key_inside_array_of_objects(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["spec"]["custom"] = {"sources": [{"media.root": "/x"}]}
    bad(b, spec_schema)


# --------------------------------------------------------------------------- #
# spec-bundle: mode discrimination
# --------------------------------------------------------------------------- #


def test_reject_config_mode_with_args(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["args"] = ["--foo"]
    bad(b, spec_schema)


def test_reject_config_mode_missing_config_format(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    del b["config_format"]
    bad(b, spec_schema)


def test_reject_config_command_without_config_path(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["command"] = "python -m cosmos_framework.scripts.train"
    bad(b, spec_schema)


def test_reject_args_mode_with_spec(spec_schema):
    b = copy.deepcopy(ARGS_BUNDLE)
    b["spec"] = {"train": {"num_epochs": 1}}
    bad(b, spec_schema)


def test_reject_missing_mode(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    del b["mode"]
    bad(b, spec_schema)


# --------------------------------------------------------------------------- #
# spec-bundle: other seam invariants
# --------------------------------------------------------------------------- #


def test_reject_unresolved_image_key(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["image"] = "containers.pyt"  # a versions.yaml key, not a resolved URI
    bad(b, spec_schema)


def test_reject_declared_input_missing_uri(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    del b["declared_inputs"][0]["uri"]
    bad(b, spec_schema)


def test_reject_empty_declared_outputs(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["declared_outputs"] = []
    bad(b, spec_schema)


def test_reject_zero_nodes(spec_schema):
    b = copy.deepcopy(TRAIN_BUNDLE)
    b["compute_shape"]["nodes"] = 0
    bad(b, spec_schema)


# --------------------------------------------------------------------------- #
# job-record
# --------------------------------------------------------------------------- #

RECORD = {
    "schema_version": 1,
    "id": "cosmos3-train-a1b2c3",
    "platform": "slurm",
    "backend_ref": None,
    "image": "cosmos-framework:local",  # unpinned: test fixture
    "network_arch": "cosmos3",
    "action": "train",
    "results_dir": "/lustre/fsw/portfolios/users/me/results/cosmos3-train-a1b2c3",
    "storage_tier": "A",
    "upload_excludes": ["inputs/"],
    "submitted_at": "2026-07-09T18:00:00+00:00",
    "transitions": [{"ts": "2026-07-09T18:00:00+00:00", "state": "PENDING", "message": "opened", "source": "agent"}],
    "terminal_state": None,
    "redacted": True,
}


def test_job_record_valid_at_open(record_schema):
    ok(RECORD, record_schema)


def test_job_record_valid_terminal_with_retry_chain(record_schema):
    r = copy.deepcopy(RECORD)
    r["backend_ref"] = "4211337"
    r["transitions"].append(
        {"ts": "2026-07-09T19:00:00+00:00", "state": "ERROR", "message": "NODE_FAIL", "source": "poller"}
    )
    r["terminal_state"] = "ERROR"
    r["terminal_write_by"] = "poller"
    r["err_class"] = "ERR_INFRA"
    r["retry_of"] = None
    r["parent_job"] = "automl-exp-7"
    ok(r, record_schema)


def test_job_record_rejects_unknown_state(record_schema):
    r = copy.deepcopy(RECORD)
    r["transitions"][0]["state"] = "QUEUED"  # not in the fixed vocabulary
    bad(r, record_schema)


def test_job_record_rejects_empty_transitions(record_schema):
    r = copy.deepcopy(RECORD)
    r["transitions"] = []
    bad(r, record_schema)


def test_job_record_rejects_missing_results_dir(record_schema):
    r = copy.deepcopy(RECORD)
    del r["results_dir"]
    bad(r, record_schema)
    r["results_dir"] = ""  # empty is as bad as missing
    bad(r, record_schema)


def test_job_record_rejects_unredacted(record_schema):
    r = copy.deepcopy(RECORD)
    r["redacted"] = False
    bad(r, record_schema)


def test_job_record_rejects_bad_tier_and_malformed_platform(record_schema):
    r = copy.deepcopy(RECORD)
    r["storage_tier"] = "D"
    bad(r, record_schema)
    # The schema sees STORED values (the writer normalizes input into canonical
    # kebab-case first) — non-canonical forms fail the pattern, not a registry.
    for malformed in ("Docker", "my platform", "-kratos", "a" * 40):
        r = copy.deepcopy(RECORD)
        r["platform"] = malformed
        bad(r, record_schema)


def test_job_record_platform_is_an_open_set(record_schema):
    """External platform skills (kratos, lepton, ...) record under their own
    slug with no schema edit — install equals register."""
    for external in ("kratos", "lepton", "osmo"):
        r = copy.deepcopy(RECORD)
        r["platform"] = external
        ok(r, record_schema)
