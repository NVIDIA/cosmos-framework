# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cosmos_framework.utils.workflow_status import monitor_status

pytestmark = pytest.mark.L0


def test_monitor_status_reads_results_dir_from_a_utf8_toml_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("COSMOS_STATUS_FILE", raising=False)
    monkeypatch.delenv("COSMOS_RESULTS_DIR", raising=False)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    config_path = tmp_path / "evaluate.toml"
    # The non-ASCII comment checks that the config is read as UTF-8 whatever the locale.
    config_path.write_text(
        f'# résultats de l\'évaluation\nresults_dir = "{results_dir.as_posix()}"\n',
        encoding="utf-8",
    )

    @monitor_status(name="Cosmos", mode="evaluate")
    def run(args):
        return "done"

    assert run(SimpleNamespace(results_dir=None, config=str(config_path))) == "done"

    records = [json.loads(line) for line in (results_dir / "status.json").read_text(encoding="utf-8").splitlines()]
    assert [record["status"] for record in records] == ["STARTED", "SUCCESS"]
