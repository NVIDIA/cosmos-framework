# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small artifact rejection checks; real prepared16 success is qualified on the captured GPU tape."""

import json

import pytest
import torch

from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.tests import pi05_batch_parity_cli
from cosmos_framework.posttrain.tests.helpers.pi05_prepared import load_prepared16, parity_recipe
from cosmos_framework.posttrain.utils.object_store import file_sha256

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("corruption", ["bytes", "assets", "capture_hash", "record_count"])
def test_prepared16_rejects_unbound_or_incomplete_artifacts(tmp_path, corruption):
    # A valid pickle with no records cannot qualify. Earlier provenance failures
    # must be distinguished from this final inventory rejection.
    path = tmp_path / "prepared16.pt"
    torch.save((), path)
    capture = tmp_path / "capture.json"
    capture.write_text('{"fixture": true}')
    assets = {"checkpoint_sha256": "cpu-fixture-checkpoint"}
    manifest = {
        "assets": assets,
        "records_sha256": file_sha256(path),
        "source_capture": {"path": str(capture), "sha256": file_sha256(capture)},
        "selection": [],
    }
    path.with_suffix(".json").write_text(json.dumps(manifest))
    if corruption == "bytes":
        with path.open("ab") as stream:
            stream.write(b"changed")
        match = "bytes changed"
    elif corruption == "assets":
        assets = {"checkpoint_sha256": "foreign"}
        match = "model assets differ"
    elif corruption == "capture_hash":
        capture.write_text('{"fixture": false}')
        match = "capture manifest changed"
    else:
        match = "sixteen unique real decisions"
    with pytest.raises(ValueError, match=match):
        load_prepared16(path, Pi05ModelConfig("checkpoint", "tokenizer", "stats"), assets)


def test_batch_probe_requires_explicit_records_and_formal_optimizer(tmp_path):
    with pytest.raises(SystemExit) as error:
        pi05_batch_parity_cli.main(["--output", str(tmp_path / "report.json")])
    assert error.value.code == 2
    recipe = parity_recipe()
    assert (recipe.actor_lr, recipe.value_lr, recipe.optimizer.impl) == (5e-6, 1e-4, "fused")
